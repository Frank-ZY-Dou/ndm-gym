"""Fit one world-model variant offline on collected transitions (``ndm_gym.wmbench.collect``) and score its k-step
open-loop prediction error on held-out datasets.

The world model, its loss and the window sampling are exactly those of online training: the algorithm selected by
``--algo`` (with all training flags) is built and its ``wm_update`` is called on a replay buffer filled from the
training datasets. No simulator runs.

Run: ``python -m ndm_gym.wmbench.fit --algo supertrack --train data_wm/st.pt --test st=data_wm/st_test.pt
       sm=data_wm/sm.pt --steps 20000 --out results/wmbench/<name>.csv``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ndm_gym.train import add_train_args


def parse_args():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    parser.add_argument("--train", nargs="+", required=True, help="training datasets (same number of steps each)")
    parser.add_argument("--test", nargs="+", required=True, help="name=path held-out datasets")
    parser.add_argument("--steps", type=int, default=20000, help="world-model updates")
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_windows", type=int, default=8192)
    parser.add_argument("--eval_k", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--out", type=str, required=True, help="csv of the test errors over training")
    parser.add_argument("--init_wm", type=str, default=None, help="start from a fitted world model (its _wm.pt); with --steps 0 only re-score it")
    args = parser.parse_args()
    if args.clips:  # the recorded transitions index the frames of every clip
        parser.error("--clips is not supported: the datasets were collected on every clip in --motion_dir")
    return args


def load_buffer(paths: list[str], dev: str):
    """The datasets side by side (env axis) in one replay buffer on ``dev``. The files are memory-mapped and each one is
    unpacked on ``dev`` straight into its columns (``collect.unpack``: packed and earlier full-field files alike)."""
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.wmbench.collect import unpack

    parts = [torch.load(p, map_location="cpu", mmap=True) for p in paths]
    T = parts[0]["data"]["s"].shape[0]
    assert all(p["data"]["s"].shape[0] == T for p in parts), "datasets must have the same number of steps"
    N = sum(p["data"]["s"].shape[1] for p in parts)
    buf = ReplayBuffer(T, N, dev)
    n0 = 0
    for p in parts:
        data = unpack(p["data"], dev)
        n = data["s"].shape[1]
        for k in buf.data:
            buf.data[k][:, n0:n0 + n].copy_(data[k])
        n0 += n
    buf.ptr, buf.size = 0, T
    return buf, [p["meta"] for p in parts]


@torch.no_grad()
def fixed_windows(buf, n: int, k: int, seed: int):
    """n continuous k-step windows, sampled once with a fixed seed: no episode end inside, and every step starts
    where the previous one ended (excludes the instants of external pushes, which no world model can predict)."""
    g = torch.Generator(device=buf.device).manual_seed(seed)
    out = []
    while sum(len(w["s"]) for w in out) < n:
        t0 = torch.randint(0, buf.size - k + 1, (2 * n,), device=buf.device, generator=g)
        env = torch.randint(0, buf.N, (2 * n,), device=buf.device, generator=g)
        rows = buf._row(t0.unsqueeze(-1) + torch.arange(k, device=buf.device))
        e = env.unsqueeze(-1).expand_as(rows)
        w = {key: buf.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        ok = ~w["done"][:, :-1].any(-1) & ((w["s1"][:, :-1] - w["s"][:, 1:]).abs().amax((-1, -2)) == 0)
        out.append({key: v[ok] for key, v in w.items()})
    return {key: torch.cat([w[key] for w in out])[:n] for key in out[0]}


@torch.no_grad()
def pre_failure_windows(buf, n: int, k: int, window: int, failed, seed: int):
    """Up to n distinct continuous k-step windows that end in the last ``window`` steps up to a termination (the
    ``--wm_fail_windows`` windows), drawn once with a fixed seed; None if the dataset holds no termination that fits."""
    g = torch.Generator(device=buf.device).manual_seed(seed)
    w = buf.pre_failure_windows(2 * n, k, window, failed, generator=g)
    if w is None:
        return None
    key = w["start"][0] * buf.N + w["start"][1]
    ok = (w["s1"][:, :-1] - w["s"][:, 1:]).abs().amax((-1, -2)) == 0
    first = torch.zeros_like(ok)
    order = torch.argsort(key, stable=True)
    first[order] = torch.cat((torch.ones(1, dtype=torch.bool, device=buf.device), key[order][1:] != key[order][:-1]))
    idx = (ok & first).nonzero().squeeze(-1)[:n]
    return {k_: v[idx] for k_, v in w.items() if k_ != "start"}


@torch.no_grad()
def window_errors(algo, w):
    """Tracked-body position (m) and joint angle (rad) errors after the window's k open-loop steps."""
    from ndm_gym.state import normalize_xi, unflatten_state

    Sh = unflatten_state(w["s"][:, 0])
    for j in range(w["a"].shape[1]):
        Sh = algo.wm.step(Sh, w["a"][:, j], normalize_xi(w["xi"][:, j]))
    St = unflatten_state(w["s1"][:, -1])
    eb = (algo.kin.forward(*Sh)[0] - algo.kin.forward(*St)[0]).norm(dim=-1).mean(-1)
    eq = (Sh[4] - St[4]).abs().mean(-1)
    return eb, eq


def main():
    args = parse_args()

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1
    from ndm_gym.robots import g1
    from ndm_gym.state import normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    kin = g1.kinematics(dev)
    motion = MotionLib(lafan1.load_dir(args.motion_dir), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    algo = ALGOS[args.algo](args, kin, motion, mdp, dev)
    buf, train_meta = load_buffer(args.train, dev)
    # normalizers from all training transitions, as online collection fits them
    for t in range(buf.size):
        S0, S1 = unflatten_state(buf.data["s"][t]), unflatten_state(buf.data["s1"][t])
        algo.wm.in_norm.update(algo.wm.inputs(S0, buf.data["a"][t], normalize_xi(buf.data["xi"][t])))
        algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    if args.init_wm:
        algo.wm.load_state_dict(torch.load(args.init_wm, map_location=dev)["wm"])
    failed = lambda s1, f: mdp.termination(unflatten_state(s1), f + 1)  # noqa: E731  (as TrackingSim.step)
    tests = {}
    for spec in args.test:
        name, path = spec.split("=", 1)
        tb, _ = load_buffer([path], dev)
        tests[name] = {k: fixed_windows(tb, args.eval_windows, k, seed=k) for k in args.eval_k}
        # the same test data restricted to windows that end in the last second before a termination
        pf = {k: pre_failure_windows(tb, args.eval_windows, k, args.wm_fail_window, failed, seed=k) for k in args.eval_k}
        if all(w is not None for w in pf.values()):
            tests[name + "_pf"] = pf
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    f = open(args.out, "w")
    f.write("step,wall_s,train_loss,test,k,body_mm,body_p90_mm,q_mrad,n\n")
    t0 = time.time()

    def score(step: int, tl: float):
        summary = []
        for name, by_k in tests.items():
            for k, w in by_k.items():
                eb, eq = window_errors(algo, w)
                f.write(f"{step},{time.time() - t0:.1f},{tl:.6g},{name},{k},{1e3 * eb.mean():.3f},{1e3 * eb.quantile(0.9):.3f},{1e3 * eq.mean():.3f},{len(eb)}\n")
                summary.append(f"{name} k{k} {1e3 * eb.mean():.1f} mm")
        f.flush()
        print(f"[wmbench] step {step}: train loss {tl:.4f} | " + " | ".join(summary), flush=True)

    if args.steps == 0:
        score(0, float("nan"))
    loss_acc = []
    for step in range(1, args.steps + 1):
        loss_acc.append(algo.wm_update(buf)[0])
        if step % args.eval_every == 0 or step == args.steps:
            tl = float(torch.stack(loss_acc).mean())
            loss_acc = []
            score(step, tl)
    json.dump({"args": vars(args), "train": train_meta}, open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=1)
    torch.save({"wm": algo.wm.state_dict(), "args": vars(args)}, os.path.splitext(args.out)[0] + "_wm.pt")


if __name__ == "__main__":
    main()
