"""Fit a neural dynamics model of one of HDMI's tasks offline (``ndm_gym.worldmodels.hoi``) on transitions from
``wmbench/collect_hoi.py`` and score its open-loop k-step prediction on held-out datasets. No simulator runs.

Scores after k open-loop steps (continuous windows without an episode end, drawn once with a fixed seed): the 16
keypoints' position error (m, our forward kinematics), the joint angle error (rad), a free object's position (m) and
rotation (rad) errors or a hinge's angle error (rad), and the contact forces' mean absolute errors per component (N;
the end effectors' in the contact body's frame, the feet's in the world frame) (``window_scores``;
``wmbench/score_hoi.py`` scores a fitted model the same way through other inference paths).

Run: ``python -m ndm_gym.wmbench.fit_hoi --task move_suitcase --kind accel --train data_wm/hoi/a.pt --test held=data_wm/hoi/b.pt
       --steps 20000 --out results/wmbench_hoi/<name>.csv``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ndm_gym.state import unflatten_object, unflatten_state

KEYS = ("s", "s1", "o", "o1", "u", "ctx", "f_eef", "f_feet", "done")
SCORE_COLUMNS = "test,k,keypoint_mm,keypoint_p90_mm,joint_mrad,obj_mm,obj_mrad,eef_force_n,feet_force_n,n"


def parse_args():
    from ndm_gym.worldmodels.hoi import KINDS

    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--kind", type=str, required=True, choices=KINDS)
    p.add_argument("--train", nargs="+", required=True, help="training datasets (same number of steps each)")
    p.add_argument("--test", nargs="+", required=True, help="name=path held-out datasets")
    p.add_argument("--steps", type=int, default=20000, help="world-model updates")
    p.add_argument("--batch", type=int, default=512, help="windows per update")
    p.add_argument("--window", type=int, default=8, help="open-loop steps per training window")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "rigidformer"])
    p.add_argument("--rf_dim", type=int, default=128, help="rigidformer: token width")
    p.add_argument("--rf_layers", type=int, default=3)
    p.add_argument("--rf_heads", type=int, default=4)
    p.add_argument("--rf_registers", type=int, default=2)
    p.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048], help="mlp: widths")
    p.add_argument("--act", type=str, default="silu")
    p.add_argument("--no_ln", action="store_true", help="no LayerNorm in the MLP")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
    p.add_argument("--eager", action="store_true",
                   help="the training loss without torch.compile (as before it was compiled; the same windows, losses within float rounding)")  # fmt: skip
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_windows", type=int, default=4096)
    p.add_argument("--eval_k", type=int, nargs="+", default=[8, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="csv of the test errors over training")
    return p.parse_args()


def load_buffer(paths: list[str], dev: str):
    """The datasets' fields that the fit reads (KEYS) in one ReplayBuffer, environments concatenated: each field copied
    from the memory-mapped file into its slice of the buffer, without a concatenated copy on the host or the GPU."""
    from ndm_gym.buffer import ReplayBuffer

    parts = [torch.load(p, map_location="cpu", mmap=True) for p in paths]
    metas = [p["meta"] for p in parts]
    for m in metas[1:]:
        if (m["task"], m["context_layout"]) != (metas[0]["task"], metas[0]["context_layout"]):
            raise ValueError(f"datasets of different tasks or context layouts: {[(m['task'], m['context_layout']) for m in metas]}")
    T = parts[0]["data"]["s"].shape[0]
    if any(p["data"]["s"].shape[0] != T for p in parts):
        raise ValueError("datasets must have the same number of steps")
    fields = {k: (tuple(parts[0]["data"][k].shape[2:]), parts[0]["data"][k].dtype) for k in KEYS}
    buf = ReplayBuffer(T, sum(p["data"]["s"].shape[1] for p in parts), dev, fields=fields)
    buf.window_keys = KEYS
    i = 0
    for p in parts:
        n = p["data"]["s"].shape[1]
        for k in KEYS:
            buf.data[k][:, i : i + n].copy_(p["data"][k])
        i += n
    buf.ptr, buf.size = 0, T
    return buf, metas


def fixed_windows(buf, n: int, k: int, seed: int) -> dict:
    """n k-step windows without an episode end before their last step, drawn once with a fixed seed."""
    g = torch.Generator(device=buf.device).manual_seed(seed)
    out = []
    while sum(len(w["s"]) for w in out) < n:
        t0 = torch.randint(0, buf.size - k + 1, (2 * n,), device=buf.device, generator=g)
        env = torch.randint(0, buf.N, (2 * n,), device=buf.device, generator=g)
        rows = buf._row(t0.unsqueeze(-1) + torch.arange(k, device=buf.device))
        e = env.unsqueeze(-1).expand_as(rows)
        w = {key: buf.data[key][rows, e] for key in KEYS}
        ok = ~w["done"][:, :-1].any(-1)
        out.append({key: v[ok] for key, v in w.items()})
    return {key: torch.cat([w[key] for w in out])[:n] for key in out[0]}


def object_state(o: torch.Tensor, hinged: bool) -> tuple:
    """A dataset's object field as the world model's So: a hinge's (angle, rate), or a free object's (p, R, v, w)."""
    return (o[..., :1], o[..., 1:2]) if hinged else unflatten_object(o)


def true_step(w: dict, j: int, hinged: bool) -> tuple:
    """The recorded outcome (S', So', F') of step j of windows w."""
    return unflatten_state(w["s1"][:, j]), object_state(w["o1"][:, j], hinged), (w["f_eef"][:, j], w["f_feet"][:, j])


def world_model(a: dict, layout, recorded_constants: dict | None = None):
    """The world model that fit_hoi fits, as its arguments ``a`` (a dict, as saved with the model) describe it, for
    datasets of context layout ``layout``; a RigidFormer takes the task's ``ndm_constants``, which must equal the
    datasets' ``recorded_constants`` if given. Initialized from the global random state, on the CPU."""
    from ndm_gym.sim.rigid.hdmi_sim import ndm_constants
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.worldmodels.hoi import HOIWorldModel, rigidformer_args

    cfg = TASKS[a["task"]]
    rf = None
    if a["backbone"] == "rigidformer":
        consts = ndm_constants(cfg)
        if recorded_constants is not None and recorded_constants != consts:
            raise ValueError("the datasets' task constants differ from ndm_constants(cfg)")
        rf = rigidformer_args(consts, dim=a["rf_dim"], layers=a["rf_layers"], heads=a["rf_heads"], registers=a["rf_registers"])
    return HOIWorldModel(a["kind"], cfg.object_joint is not None, cfg.num_eefs, layout, hidden=a["hidden"], act=a["act"],
                         layer_norm=not a["no_ln"], rigidformer=rf)  # fmt: skip


@torch.no_grad()
def window_scores(step, w: dict, kin, hinged: bool) -> dict:
    """The scores of k open-loop steps on windows w (``fixed_windows``): ``step`` (S, So, u, ctx) -> (S', So', F') fed
    back on itself from the windows' first states with their actions and contexts, against the recording after the
    last step. The position error averaged over the keypoints (``kin``: G1's kinematics tracking KEYPOINT_NAMES), its
    mean and 90th percentile over windows (mm); the joint angle error (mrad); a free object's position (mm) and
    rotation (mrad) errors, or a hinge's angle error (obj_mrad; obj_mm NaN); the contact forces' mean absolute errors
    (N); the number of windows. Python numbers, keyed by SCORE_COLUMNS' names after test and k.

    The object's rotation error is the angle of R_true^T R_pred, the product written out elementwise and the angle
    taken by atan2 from its skew part and trace (``tasks.hdmi``'s relative_rot and rot_angle, as in the HOI
    algorithms): exact to fp32 rounding at any angle. Before, it was the arccos of the trace of a batched matmul, which
    ran in TF32 under fit_hoi's allow_tf32 (on an A100 a true 1 mrad read as 7.5 mrad on average and up to 32.5 mrad),
    and arccos near 1 loses small angles even in fp32 (1 mrad read as 0.87 mrad on average, as 0 at worst)."""
    from ndm_gym.tasks.hdmi import relative_rot, rot_angle

    k = w["u"].shape[1]
    S, So = unflatten_state(w["s"][:, 0]), object_state(w["o"][:, 0], hinged)
    for j in range(k):
        S, So, F = step(S, So, w["u"][:, j], w["ctx"][:, j])
    St, Sot, Ft = true_step(w, k - 1, hinged)
    kp = (kin.forward(*S)[0] - kin.forward(*St)[0]).norm(dim=-1).mean(-1)
    jq = (S[4] - St[4]).abs().mean(-1)
    if hinged:
        op, orot = torch.full_like(kp, float("nan")), (So[0] - Sot[0]).abs().squeeze(-1)
    else:
        op, orot = (So[0] - Sot[0]).norm(dim=-1), rot_angle(relative_rot(Sot[1], So[1]))
    fe, ff = (F[0] - Ft[0]).abs().mean(), (F[1] - Ft[1]).abs().mean()
    return {"keypoint_mm": float(1e3 * kp.mean()), "keypoint_p90_mm": float(1e3 * kp.quantile(0.9)), "joint_mrad": float(1e3 * jq.mean()),
            "obj_mm": float(1e3 * op.mean()), "obj_mrad": float(1e3 * orot.mean()), "eef_force_n": float(fe), "feet_force_n": float(ff),
            "n": len(kp)}  # fmt: skip


def score_fields(s: dict) -> str:
    """window_scores' numbers as the csv's fields from keypoint_mm to n."""
    return (f"{s['keypoint_mm']:.3f},{s['keypoint_p90_mm']:.3f},{s['joint_mrad']:.3f},{s['obj_mm']:.3f},{s['obj_mrad']:.3f},"
            f"{s['eef_force_n']:.4f},{s['feet_force_n']:.4f},{s['n']}")  # fmt: skip


def main():
    args = parse_args()

    import ndm_gym.core.nets as nets
    from ndm_gym.robots import g1
    from ndm_gym.tasks.hdmi import KEYPOINT_NAMES
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.worldmodels.hoi import ContextLayout

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    cfg = TASKS[args.task]
    buf, train_meta = load_buffer(args.train, dev)
    if train_meta[0]["task"] != args.task:
        raise ValueError(f"datasets of task {train_meta[0]['task']}, not {args.task}")
    hinged = cfg.object_joint is not None
    layout = ContextLayout(*train_meta[0]["context_layout"])
    wm = world_model(vars(args), layout, train_meta[0].get("ndm_constants")).to(dev)
    kin = g1.kinematics(dev, tracked=list(KEYPOINT_NAMES))

    # normalizers from all training transitions, as online collection fits them
    for t in range(buf.size):
        d = {k: buf.data[k][t] for k in KEYS}
        S, So, S1, So1 = unflatten_state(d["s"]), object_state(d["o"], hinged), unflatten_state(d["s1"]), object_state(d["o1"], hinged)
        F1 = (d["f_eef"], d["f_feet"])
        wm.in_norm.update(wm.inputs(S, So, d["u"], d["ctx"]))
        wm.out_norm.update(wm.targets(S, So, S1, So1, F1))
        wm.err_norm.update(wm.increments(S, So, S1, So1, F1))
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    tests = {}
    for spec in args.test:
        name, path = spec.split("=", 1)
        tb, tm = load_buffer([path], dev)
        if (tm[0]["task"], tm[0]["context_layout"]) != (args.task, train_meta[0]["context_layout"]):
            raise ValueError(f"test set {path} is of another task or context layout")
        tests[name] = {k: fixed_windows(tb, args.eval_windows, k, seed=k) for k in args.eval_k}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    f = open(args.out, "w")
    f.write(f"step,wall_s,train_loss,{SCORE_COLUMNS}\n")
    t0 = time.time()

    @torch.no_grad()
    def score(step: int, tl: float):
        summary = []
        for name, by_k in tests.items():
            for k, w in by_k.items():
                s = window_scores(wm.step, w, kin, hinged)
                f.write(f"{step},{time.time() - t0:.1f},{tl:.6g},{name},{k},{score_fields(s)}\n")
                summary.append(f"{name} k{k} keypoints {s['keypoint_mm']:.1f} mm, object {s['obj_mrad' if hinged else 'obj_mm']:.1f}")
        f.flush()
        print(f"[wmbench_hoi] step {step}: train loss {tl:.4f} | " + " | ".join(summary), flush=True)

    def window_loss(w: dict) -> torch.Tensor:
        S, So = unflatten_state(w["s"][:, 0]), object_state(w["o"][:, 0], hinged)
        preds = wm.rollout(S, So, w["u"], w["ctx"])
        return wm.window_loss(preds, [true_step(w, j, hinged) for j in range(args.window)], w["mask"])[0]

    # compiled as HOI training's model update (algos.hoi_algos._wm_window): eager, the window's few thousand small
    # kernels per update make the fit launch-bound
    loss_fn = window_loss if args.eager else torch.compile(window_loss)
    loss_acc = []
    for step in range(1, args.steps + 1):
        loss = loss_fn(buf.sample_windows(args.batch, args.window))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        loss_acc.append(loss.detach() / args.window)
        if step % args.eval_every == 0 or step == args.steps:
            score(step, float(torch.stack(loss_acc).mean()))
            loss_acc = []
    json.dump({"args": vars(args), "train": train_meta}, open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=1)
    torch.save({"wm": wm.state_dict(), "args": vars(args), "layout": tuple(layout)}, os.path.splitext(args.out)[0] + "_wm.pt")


if __name__ == "__main__":
    main()
