"""Policy-gradient statistics of a training run's checkpoint at learning rate 0: how often the gradient of the BPTT
through the world model spikes, and how well each update points along the mean gradient, with and without the tricks
against spikes (``--bptt_clip_k``, ``--rf_rope_detach``, ``--wm_bptt_fp32``).

Loads the run's full state (``last.pt``) and a replay dataset recorded with the run's own policy
(``python -m ndm_gym.wmbench.collect``), sets every learning rate to 0 and runs ``--updates`` policy updates, update i
on its own batch drawn with seed ``--seed0 + i``, so that every setting sees the same batches and noise. The policy,
world model and critic stay fixed: their normalizers, which an update may refit, are restored after every update, and
the parameters are checked to be unchanged at the end. Per update: the policy-gradient norm before clipping (the
training log's ``pi_gradnorm``), the policy loss, and the cosine between the gradient and a reference direction. As
training clips nearly every update to norm 1, directions are averaged as unit vectors: the reference is the mean unit
gradient of the run's own setting over ``--ref_updates`` other batches (seeds from 10^6; ``--save_ref`` writes it,
``--ref`` reads it), and the length of a setting's own mean unit gradient (1: every update points the same way) measures
how consistent its updates are. ``--set key=value`` overrides the run's training arguments.

Usage: python tools/grad_stats.py --ckpt logs/<run>/last.pt --data <dataset.pt> --out <dir>/<setting>.json
       (--save_ref <dir>/ref.pt | --ref <dir>/ref.pt [--set bptt_clip_k=3 | rf_rope_detach=1 | wm_bptt_fp32=1 ...])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
REF_SEED0 = 10**6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="the run's last.pt (the full training state)")
    p.add_argument("--data", nargs="+", required=True, help="replay dataset(s) recorded with the run's policy (ndm_gym.wmbench.collect)")
    p.add_argument("--out", required=True, help="JSON with the per-update values and their summary")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--ref_updates", type=int, default=500)
    p.add_argument("--seed0", type=int, default=0)
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--save_ref", help="compute the reference direction with the run's own setting (no --set) and write it here")
    ref.add_argument("--ref", help="reference direction written by --save_ref for the same checkpoint")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="training arguments to override")
    p.add_argument("--motion_dir", default=os.path.join(REPO, "data", "lafan1_g1"))
    return p.parse_args()


def run_arguments(ck_args: dict, overrides: list[str], launched: set[str]) -> dict:
    """The run's training arguments over the current defaults (flags added after the run take their defaults, which
    keep its behavior), then ``key=value`` overrides parsed with the argument's type. ``launched``: keys the simulator's
    launcher added to the arguments after train.py wrote ``args.json``."""
    from isaaclab.app import add_launcher_args

    from ndm_gym.train import add_train_args

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_launcher_args(parser)
    defaults = vars(parser.parse_args(["--algo", ck_args["algo"]]))
    unknown = sorted(set(ck_args) - set(defaults) - launched)
    if unknown:
        raise ValueError(f"the checkpoint has training arguments this code does not know: {unknown}")
    a = {**defaults, **ck_args}
    for kv in overrides:
        k, sep, v = kv.partition("=")
        if not sep or k not in defaults or defaults[k] is None:
            raise ValueError(f"--set {kv!r}: expected key=value for a training argument with a typed default")
        d = defaults[k]
        if isinstance(d, bool):
            if v.lower() not in ("0", "1", "false", "true"):
                raise ValueError(f"--set {kv!r}: a flag takes 0/1/false/true")
            a[k] = v.lower() in ("1", "true")
        else:
            a[k] = type(d)(v)
    return a


def main():
    args = parse_args()
    if args.save_ref and args.set:
        raise SystemExit("--save_ref computes the reference with the run's own setting: no --set")

    import numpy as np
    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1, read_clip_list, select
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP
    from ndm_gym.train import training_reward_cfg
    from ndm_gym.wmbench.fit import load_buffer

    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(args.ckpt, map_location=dev)
    for key in ("pi", "wm", "opt_pi", "iteration", "args"):
        if key not in ck:
            raise SystemExit(f"{args.ckpt} has no {key!r}: pass the run's last.pt (the full training state), not a ckpt_*.pt")
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    with open(os.path.join(run_dir, "args.json")) as f:
        a = run_arguments(ck["args"], args.set, launched=set(ck["args"]) - set(json.load(f)))
    nets.AMP_DTYPE = torch.bfloat16 if a["amp"] == "bf16" else None
    kin = g1.kinematics(dev)
    clips = lafan1.load_dir(args.motion_dir)
    if a["clips"]:  # the clips the run trained on, as train.py recorded them
        clips = select(clips, read_clip_list(os.path.join(run_dir, "clips.txt")))
    motion = MotionLib(clips, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), training_reward_cfg(a["reward_std_scale"]))
    algo = ALGOS[a["algo"]](argparse.Namespace(**a), kin, motion, mdp, dev)
    algo.load_state_dict(ck)
    buf, data_meta = load_buffer(args.data, dev)
    for m in data_meta:
        if m["algo"] != a["algo"]:
            raise SystemExit(f"dataset recorded with a {m['algo']} policy, the checkpoint is {a['algo']}")
    for o in algo.optimizers:
        for group in o.param_groups:
            group["lr"] = 0.0
    mods = [m for m in (algo.pi, algo.wm, getattr(algo, "critic", None)) if m is not None]
    params0 = [p.detach().clone() for m in mods for p in m.parameters()]
    state0 = [{k: v.clone() for k, v in m.state_dict().items() if k not in dict(m.named_parameters())} for m in mods]

    # the raw policy gradient of every update, before policy_update clips it
    params = list(algo.pi.parameters())
    sizes = [p.numel() for p in params]
    offs = np.cumsum([0] + sizes).tolist()
    flat = torch.zeros(offs[-1], device=dev)

    def capture(i):
        def hook(p):
            flat[offs[i] : offs[i + 1]].copy_(p.grad.reshape(-1))

        return hook

    for i, p in enumerate(params):
        p.register_post_accumulate_grad_hook(capture(i))
    unit = lambda v: v / max(float(v.norm()), 1e-30)  # noqa: E731

    def update(seed: int):
        torch.manual_seed(seed)
        flat.zero_()
        t0 = time.time()
        out = algo.policy_update(buf)
        torch.cuda.synchronize()
        dt = time.time() - t0
        for m, s in zip(mods, state0):
            m.load_state_dict(s, strict=False)
        gn = float(out[3])
        if np.isfinite(gn) and abs(float(flat.norm()) - gn) > 1e-3 * max(gn, 1e-6):
            raise RuntimeError(f"captured gradient norm {float(flat.norm())} differs from the update's {gn}")
        return gn, float(out[0]), dt

    run = os.path.basename(run_dir)
    if args.save_ref:
        ref = torch.zeros_like(flat)
        for j in range(args.ref_updates):
            if np.isfinite(update(REF_SEED0 + j)[0]):
                ref += unit(flat)
        ref = unit(ref)
        torch.save({"ref": ref.cpu(), "run": run, "iteration": ck["iteration"], "updates": args.ref_updates}, args.save_ref)
    else:
        saved = torch.load(args.ref, map_location=dev)
        if (saved["run"], saved["iteration"]) != (run, ck["iteration"]):
            raise SystemExit(f"{args.ref} is for {saved['run']} at {saved['iteration']}, not {run} at {ck['iteration']}")
        ref = saved["ref"]

    gns, losses, cos, secs = [], [], [], []
    mean = torch.zeros_like(flat)
    for i in range(args.updates):
        gn, loss, dt = update(args.seed0 + i)
        gns.append(gn), losses.append(loss), secs.append(dt)
        cos.append(float(unit(flat) @ ref))
        if np.isfinite(gn):
            mean += unit(flat)
    for p0, p in zip(params0, (p for m in mods for p in m.parameters())):
        if not torch.equal(p0, p):
            raise RuntimeError("a parameter changed at learning rate 0")

    g = np.array(gns)
    fin = np.isfinite(g)
    med = float(np.median(g[fin]))
    c = np.array(cos)[fin]
    summary = {
        "updates": args.updates,
        "nonfinite": int((~fin).sum()),
        "gradnorm_median": med,
        "gradnorm_p90": float(np.percentile(g[fin], 90)),
        "gradnorm_p99": float(np.percentile(g[fin], 99)),
        "gradnorm_max": float(g[fin].max()),
        "above_3x_median": int((g[fin] > 3 * med).sum()),
        "above_10x_median": int((g[fin] > 10 * med).sum()),
        "cos_to_ref_mean": float(c.mean()),
        "cos_to_ref_median": float(np.median(c)),
        "cos_to_ref_spikes_mean": float(c[g[fin] > 3 * med].mean()) if (g[fin] > 3 * med).any() else None,
        "mean_unit_gradient_length": float(mean.norm()) / int(fin.sum()),
        "mean_unit_gradient_cos_to_ref": float(unit(mean) @ ref),
        "sec_per_update_median": float(np.median(secs)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"ckpt": os.path.abspath(args.ckpt), "run": run, "iteration": ck["iteration"], "algo": a["algo"], "backbone": a["wm_backbone"],
            "set": args.set, "data": [os.path.abspath(d) for d in args.data], "data_meta": data_meta, "ref": args.save_ref or args.ref,
            "seed0": args.seed0, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}  # fmt: skip
    json.dump({"meta": meta, "summary": summary, "gradnorm": gns, "loss": losses, "cos_to_ref": cos, "sec": secs}, open(args.out, "w"))
    print(json.dumps({"run": run, "set": args.set, **summary}))


if __name__ == "__main__":
    main()
