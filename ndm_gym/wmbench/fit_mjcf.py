"""Fit the world model of a MuJoCo model (``ndm_gym.worldmodels.mjcf``) offline on recorded transitions and score its
open-loop k-step prediction on held-out recordings. No simulator runs.

A recording is an .npz with ``qpos`` (T + 1, N, nq), ``qvel`` (T + 1, N, nv), ``ctrl`` (T, N, nu) and ``reset`` (T, N) bool,
True where the state after step t does not follow from it (an episode ended), and ``dt``, the step's length, which must be the
model's timestep. Windows never cross a reset. ``--cut K`` also cuts each rollout at its first step whose tangent-state increment
in some degree of freedom exceeds K times that degree of freedom's median absolute increment over the training recordings plus
the error floor (a simulator blow-up: MuJoCo Warp's coarse rollouts of the pen reach 1,100 m and 10^6 rad/s): from that step to
the rollout's end the steps count as resets, so no window and no normalizer reads them. The medians are the training
recordings', pooled, for every recording of the fit, so that a held-out recording of one long trajectory is judged by the
same scale. ``--cut_scale_floor`` takes as a degree of freedom's scale the larger of its median and its error floor, for
recordings in which a body lies still (median 0) and the floor alone would cut its first motion.

Scores after k open-loop steps (windows drawn once with a fixed seed): ``norm``, the mean absolute error of the tangent state
(positions and velocities) in units of one true step's spread, as the fit's loss weighs it (``MjcfWorldModel.error``;
``benchmarks/robokit_ndm/select_arms.py`` selects the fits by it); the anchor's and the objects' position (mm) and rotation
(mrad) errors, the hinges' (mrad) and slides' (mm) mean absolute errors and the ball joints' rotation errors (mrad), from the
tangent difference of the predicted and the recorded state (``worldmodels.mjcf.differentiate_pos``).

Run: ``python -m ndm_gym.wmbench.fit_mjcf --model scene.xml --train a.npz --test held=b.npz --steps 20000 --out fit.csv``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

SCORE_COLUMNS = "test,k,norm,anchor_mm,anchor_mrad,obj_mm,obj_mrad,hinge_mrad,slide_mm,ball_mrad,n"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="the MuJoCo model: an MJCF (.xml) or a compiled .mjb")
    p.add_argument("--anchor", type=str, default=None, help="the anchor's free joint (default: MjcfLayout's choice)")
    p.add_argument("--dt", type=float, default=None, help="the step learned (default: the model's timestep; a planner's coarse stage: its own)")
    p.add_argument("--train", nargs="+", required=True, help="training recordings")
    p.add_argument("--test", nargs="+", required=True, help="name=path held-out recordings")
    p.add_argument("--steps", type=int, default=20000, help="world-model updates")
    p.add_argument("--batch", type=int, default=512, help="windows per update")
    p.add_argument("--window", type=int, default=8, help="open-loop steps per training window")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--wm_param", default="accel", choices=["accel", "increment"], help="SuperTrack's accelerations or SuperMimic's increments")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--act", type=str, default="silu")
    p.add_argument("--no_ln", action="store_true", help="no LayerNorm in the MLP")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
    p.add_argument("--compile", action="store_true", help="the window loss compiled with torch.compile, its forward and backward")
    p.add_argument("--cut", type=float, default=None, help="cut rollouts at a step whose increment exceeds K median increments")
    p.add_argument("--cut_scale_floor", action="store_true", help="--cut: a degree of freedom's scale is at least its error floor")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_windows", type=int, default=4096)
    p.add_argument("--eval_k", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="csv of the test errors over training")
    p.add_argument("--save", type=str, default=None, help="the fitted model's state and layout (.pt)")
    return p.parse_args(argv)


def load_model(path: str):
    import mujoco

    return mujoco.MjModel.from_binary_path(path) if path.endswith(".mjb") else mujoco.MjModel.from_xml_path(path)


def load(path: str, lay, device) -> dict:
    """A recording as tensors, checked against the layout and its timestep."""
    z = np.load(path)
    rec = {k: torch.as_tensor(z[k], device=device) for k in ("qpos", "qvel", "ctrl", "reset")}
    T, N = rec["ctrl"].shape[:2]
    want = {"qpos": (T + 1, N, lay.nq), "qvel": (T + 1, N, lay.nv), "ctrl": (T, N, lay.nu), "reset": (T, N)}
    for k, shape in want.items():
        if tuple(rec[k].shape) != shape:
            raise ValueError(f"{path}: {k} has shape {tuple(rec[k].shape)}, the model needs {shape}")
    if abs(float(z["dt"]) - lay.dt) > 1e-12:
        raise ValueError(f"{path}: steps of {float(z['dt'])} s, the world model's are {lay.dt} s (--dt)")
    for k in ("qpos", "qvel", "ctrl"):
        rec[k] = rec[k].float()
    rec["reset"] = rec["reset"].bool()
    return rec


def increments(rec: dict, lay) -> torch.Tensor:
    """(T, N, 2 nv) the absolute tangent-state increment of every step of a recording."""
    from ndm_gym.worldmodels.mjcf import differentiate_pos

    q, v = rec["qpos"], rec["qvel"]
    T = rec["reset"].shape[0]
    return torch.cat([torch.cat((differentiate_pos(lay, q[t : min(t + 4096, T)], q[t + 1 : min(t + 4096, T) + 1]),
                                 v[t + 1 : min(t + 4096, T) + 1] - v[t : min(t + 4096, T)]), -1).abs() for t in range(0, T, 4096)])  # fmt: skip


def median_increments(recs: list, lay, sample: int = 1_000_000) -> torch.Tensor:
    """(2 nv) each degree of freedom's median absolute increment over the kept steps of ``recs`` (at most ``sample`` steps of
    each, drawn with a fixed seed)."""
    g, kept = torch.Generator(device="cpu").manual_seed(0), []
    for rec in recs:
        k = increments(rec, lay)[~rec["reset"]]
        kept.append(k[torch.randperm(len(k), generator=g)[:sample].to(k.device)])
    return torch.cat(kept).median(0).values


def cut_blowups(rec: dict, lay, K: float, med: torch.Tensor | None = None, scale_floor: bool = False) -> float:
    """``--cut``: the steps from each rollout's first outlier step to its next reset become resets (in place); the share of the
    kept steps cut. ``med``: the medians to judge by (the training recordings'); by default the recording's own. ``scale_floor``:
    each degree of freedom's scale is the larger of its median and its error floor (``--cut_scale_floor``)."""
    r = rec["reset"]
    inc = increments(rec, lay)
    med = median_increments([rec], lay) if med is None else med
    floor = torch.as_tensor(lay.floors(), dtype=inc.dtype, device=inc.device)
    scale = torch.maximum(med, floor) if scale_floor else med
    bad = (inc > K * scale + floor).any(-1) & ~r
    # within each rollout (the steps between resets), every step from the first bad one on
    start = torch.ones_like(r)
    start[1:] = r[:-1]
    c = bad.long().cumsum(0)
    before = torch.where(start, c - bad.long(), torch.zeros_like(c)).cummax(0).values  # the count at the rollout's start
    cut = (c - before) > 0
    share = (cut & ~r).float().sum().item() / max((~r).sum().item(), 1)
    rec["reset"] = r | cut
    return share


def starts(rec: dict, k: int) -> torch.Tensor:
    """(M, 2) the (t, n) whose k steps t .. t + k - 1 all follow on from their states: no reset among them."""
    r = rec["reset"].float()
    T, N = r.shape
    if T < k:
        return torch.zeros(0, 2, dtype=torch.long, device=r.device)
    csum = torch.cat((torch.zeros(1, N, device=r.device), r.cumsum(0)), 0)
    t = torch.arange(T - k + 1, device=r.device)
    return ((csum[t + k] - csum[t]) == 0).nonzero()


def gather(rec: dict, idx: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The windows at idx (M, 2): qpos, qvel (k + 1, M, .), ctrl (k, M, .)."""
    t, n = idx[:, 0], idx[:, 1]
    span = torch.arange(k + 1, device=idx.device)[:, None] + t[None]
    return rec["qpos"][span, n], rec["qvel"][span, n], rec["ctrl"][span[:-1], n]


@torch.no_grad()
def scores(wm, rec: dict, idx: torch.Tensor, k: int) -> dict:
    """The errors after k open-loop steps from the windows' first states (see the module's docstring)."""
    from ndm_gym.worldmodels.mjcf import differentiate_pos

    lay = wm.lay
    qpos, qvel, ctrl = gather(rec, idx, k)
    q, v = qpos[0], qvel[0]
    for i in range(k):
        q, v = wm.step(q, v, ctrl[i])
    e = differentiate_pos(lay, qpos[k], q)
    out = {"n": len(idx), "norm": wm.error(q, v, qpos[k], qvel[k]).abs().mean().item()}

    def norms(addr, off):
        return torch.stack([e[:, d + off : d + off + 3].norm(dim=-1) for d in addr], -1).mean().item() if addr else float("nan")

    out["anchor_mm"] = 1e3 * norms([lay.anchor[1]], 0) if lay.anchor else float("nan")
    out["anchor_mrad"] = 1e3 * norms([lay.anchor[1]], 3) if lay.anchor else float("nan")
    out["obj_mm"], out["obj_mrad"] = 1e3 * norms([d for _, d in lay.objects], 0), 1e3 * norms([d for _, d in lay.objects], 3)
    hinge = [d for _, d, _ in lay.scalars if d not in lay.slides]
    out["hinge_mrad"] = 1e3 * e[:, hinge].abs().mean().item() if hinge else float("nan")
    out["slide_mm"] = 1e3 * e[:, list(lay.slides)].abs().mean().item() if lay.slides else float("nan")
    out["ball_mrad"] = 1e3 * norms([d for _, d in lay.balls], 0)
    return out


def main(argv=None):
    args = parse_args(argv)
    import ndm_gym.core.nets as nets
    from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel

    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lay = MjcfLayout.from_model(load_model(args.model), anchor=args.anchor)
    if args.dt is not None:  # a coarse stage of the planner steps the same model with a longer timestep
        lay = dataclasses.replace(lay, dt=args.dt)
    train = [load(p, lay, dev) for p in args.train]
    tests = {}
    for spec in args.test:
        name, path = spec.split("=", 1)
        tests[name] = load(path, lay, dev)
    if args.cut is not None:
        med = median_increments(train, lay)  # the training recordings' scale, for every recording of the fit
        for path, rec in list(zip(args.train, train)) + [(spec.split("=", 1)[1], tests[spec.split("=", 1)[0]]) for spec in args.test]:
            share = cut_blowups(rec, lay, args.cut, med, args.cut_scale_floor)
            print(f"[fit_mjcf] --cut {args.cut:g}{' (scale floor)' if args.cut_scale_floor else ''}: {100 * share:.3f}% of the steps of {path} cut",
                  flush=True)  # fmt: skip
    wm = MjcfWorldModel(lay, hidden=tuple(args.hidden), act=args.act, layer_norm=not args.no_ln, kind=args.wm_param).to(dev)
    for rec in train:  # the normalizers from every training transition, before the fit
        ok = ~rec["reset"]
        wm.update_norms(rec["qpos"][:-1][ok], rec["qvel"][:-1][ok], rec["ctrl"][ok], rec["qpos"][1:][ok], rec["qvel"][1:][ok])
    pools = [starts(rec, args.window) for rec in train]
    if sum(len(x) for x in pools) == 0:
        raise ValueError(f"no training window of {args.window} steps without a reset")
    g = torch.Generator(device="cpu").manual_seed(1234)  # the test windows, fixed for every evaluation
    fixed = {}
    for name, rec in tests.items():
        for k in args.eval_k:
            pool = starts(rec, k)
            if len(pool) == 0:
                raise ValueError(f"test {name}: no window of {k} steps without a reset")
            fixed[name, k] = pool[torch.randperm(len(pool), generator=g)[: args.eval_windows].to(pool.device)]
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("step,seconds,loss," + SCORE_COLUMNS + "\n")
    window_loss = torch.compile(wm.window_loss, dynamic=False) if args.compile else wm.window_loss
    weights = torch.tensor([len(x) for x in pools], dtype=torch.float)
    t0, loss = time.time(), float("nan")
    for step in range(1, args.steps + 1):
        r = int(torch.multinomial(weights, 1))
        idx = pools[r][torch.randint(len(pools[r]), (args.batch,), device=pools[r].device)]
        qpos, qvel, ctrl = gather(train[r], idx, args.window)
        x = window_loss(qpos, qvel, ctrl)
        opt.zero_grad(set_to_none=True)
        x.backward()
        opt.step()
        loss = x.item() if step % 100 == 0 or step == args.steps else loss
        if step % args.eval_every == 0 or step == args.steps:
            with open(args.out, "a") as f:
                for (name, k), idx_k in fixed.items():
                    s = scores(wm, tests[name], idx_k, k)
                    f.write(f"{step},{time.time() - t0:.1f},{loss:.5f},{name},{k}," + ",".join(f"{s[c]:.4f}" if c != "n" else str(s[c])
                                                                                            for c in SCORE_COLUMNS.split(",")[2:]) + "\n")  # fmt: skip
            print(f"[fit_mjcf] step {step} loss {loss:.4f} ({time.time() - t0:.0f} s)", flush=True)
    if args.save:
        torch.save({"state": wm.state_dict(), "layout": json.dumps(lay.__dict__), "args": vars(args)}, args.save)


if __name__ == "__main__":
    main()
