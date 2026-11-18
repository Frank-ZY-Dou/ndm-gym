"""Fit the world model of a MuJoCo model (``ndm_gym.worldmodels.mjcf``) offline on recorded transitions and score its
open-loop k-step prediction on held-out recordings. No simulator runs.

A recording is an .npz with ``qpos`` (T + 1, N, nq), ``qvel`` (T + 1, N, nv), ``ctrl`` (T, N, nu) and ``reset`` (T, N) bool,
True where the state after step t does not follow from it (an episode ended), and ``dt``, the step's length, which must be the
world model's: the model's timestep, ``--dt``, or ``--substeps`` timesteps (a dm_control control step). Windows never cross a
reset. ``--cut K`` also cuts each rollout at its first step whose tangent-state increment
in some degree of freedom exceeds K times that degree of freedom's median absolute increment over the training recordings plus
the error floor (a simulator blow-up, as in some of MuJoCo Warp's coarse rollouts): from that step to
the rollout's end the steps count as resets, so no window and no normalizer reads them. The medians are the training
recordings', pooled, for every recording of the fit, so that a held-out recording of one long trajectory is judged by the
same scale. ``--cut_scale_floor`` takes as a degree of freedom's scale the larger of its median and its error floor, for
recordings in which a body lies still (median 0) and the floor alone would cut its first motion.

Scores after k open-loop steps (windows drawn once with a fixed seed): ``norm``, the mean absolute error of the tangent state
(positions and velocities) in units of one true step's spread, as the fit's loss weighs it (``MjcfWorldModel.error``;
``benchmarks/robokit_ndm/select_arms.py`` selects the fits by it); the anchor's and the objects' position (mm) and rotation
(mrad) errors, the hinges' (mrad) and slides' (mm) mean absolute errors and the ball joints' rotation errors (mrad), from the
tangent difference of the predicted and the recorded state (``worldmodels.mjcf.differentiate_pos``).

The network is the MLP (``--hidden``, ``--act``, ``--no_ln``) or, with ``--backbone rigidformer``, the body-token Transformer
(``worldmodels.mjcf.MjcfRigidFormerNet``: ``--rf_dim``, ``--rf_layers``, ``--rf_heads``, ``--rf_registers``, ``--rf_rope_max_freq``,
``--rf_point``); each backbone's options must stay at their defaults with the other (``worldmodels.mjcf.backbone_kwargs``).

The budget is ``--steps`` updates, or with ``--until_flat MARGIN EVALS`` as many as the held-out error needs: the fit stops
when the first test's ``norm`` at the largest ``--eval_k`` has not fallen MARGIN (relative) below its best for EVALS
evaluations in a row, ``--steps`` is then the cap, and the state saved is the best evaluation's. ``--init`` starts from an
earlier fit's state, its network and its normalizers, which then stay as they were: a refit on more recordings. A fit whose
contact inputs read static geoms (``--contact_static``) is saved with the revision of their gaps (``contact_gaps``, beside the
layout and the arguments); ``--init`` and speed_mjcf.load_ndm refuse such a fit without it (worldmodels.mjcf.check_gaps_revision).
With ``--contact_scale`` each contact column is divided by its spread over the training recordings where that exceeds one; the
scales are the input normalizer's, saved with the fit. With ``--contact_pair_margin`` the margin the contact inputs read their
gaps beyond is the simulator's contacts' (as the installed MuJoCo puts two geoms' margins together), saved with the fit too.
With ``--accel_mean_velocity`` the acceleration model advances the position with the mean of the velocities the control step
passes through under a constant acceleration, by the model's integrator and its number of physics steps
(``worldmodels.mjcf.mean_velocity_lead``), in the place of the step's last velocity; the weight is saved with the fit's state.

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
    from ndm_gym.worldmodels.mjcf import RF_DEFAULTS

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="the MuJoCo model: an MJCF (.xml) or a compiled .mjb")
    p.add_argument("--anchor", type=str, default=None, help="the anchor's free joint (default: MjcfLayout's choice)")
    p.add_argument("--planar_root", action="store_true", help="the root's horizontal slide positions out of the inputs (a planar model's rootx)")
    p.add_argument("--dt", type=float, default=None, help="the step learned (default: the model's timestep; a planner's coarse stage: its own)")
    p.add_argument("--substeps", type=int, default=None,
                   help="the step learned is this many of the model's timesteps (a dm_control control step: its n_sub_steps); saved as --dt")
    p.add_argument("--train", nargs="+", required=True, help="training recordings")
    p.add_argument("--test", nargs="+", required=True, help="name=path held-out recordings")
    p.add_argument("--steps", type=int, default=20000, help="world-model updates")
    p.add_argument("--batch", type=int, default=512, help="windows per update")
    p.add_argument("--window", type=int, default=8, help="open-loop steps per training window")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--wm_param", default="accel", choices=["accel", "increment"], help="SuperTrack's accelerations or SuperMimic's increments")
    p.add_argument("--accel_mean_velocity", action="store_true",
                   help="accel: the position advances with the mean velocity of the control step's physics steps under a constant "
                        "acceleration (Euler, k steps: v + (k + 1) / (2 k) (v' - v); RK4: (v + v') / 2); "
                        "default: with the next velocity")  # fmt: skip
    p.add_argument("--backbone", default="mlp", choices=["mlp", "rigidformer"], help="the network: the MLP or the body-token Transformer")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024], help="the MLP's widths")
    p.add_argument("--act", type=str, default="silu", help="the MLP's activation")
    p.add_argument("--no_ln", action="store_true", help="no LayerNorm in the MLP")
    p.add_argument("--rf_dim", type=int, default=RF_DEFAULTS["dim"], help="the RigidFormer's token width")
    p.add_argument("--rf_layers", type=int, default=RF_DEFAULTS["layers"], help="the RigidFormer's blocks")
    p.add_argument("--rf_heads", type=int, default=RF_DEFAULTS["heads"], help="the RigidFormer's attention heads")
    p.add_argument("--rf_registers", type=int, default=RF_DEFAULTS["registers"], help="the RigidFormer's register tokens")
    p.add_argument("--rf_rope_max_freq", type=float, default=RF_DEFAULTS["rope_max_freq"],
                   help="the RigidFormer's highest rotary frequency (rad/m; the lowest is 1)")  # fmt: skip
    p.add_argument("--rf_point", default=RF_DEFAULTS["point"], choices=["com", "origin"],
                   help="the RigidFormer's token position: the body's centre of mass or its frame's origin")  # fmt: skip
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
    p.add_argument("--compile", action="store_true", help="the window loss compiled with torch.compile, its forward and backward")
    p.add_argument("--cut", type=float, default=None, help="cut rollouts at a step whose increment exceeds K median increments")
    p.add_argument("--cut_scale_floor", action="store_true", help="--cut: a degree of freedom's scale is at least its error floor")
    p.add_argument("--cut_medians_from", nargs="+", default=None,
                   help="--cut: the recordings whose pooled medians judge every recording (default: the training recordings; recordings of runs that "
                        "lost their object would raise the medians until nothing is cut)")  # fmt: skip
    p.add_argument("--cut_nonfinite", action="store_true",
                   help="cut each rollout from a step whose state or control is not finite to its next reset (default: such a recording is refused)")
    p.add_argument("--cut_sample", type=int, default=1_000_000,
                   help="--cut: the steps of each training recording its medians are taken over (fewer for many recordings on a small card)")
    p.add_argument("--geometry", action="store_true",
                   help="the contact geometry among the inputs: the robot's collision geoms in each object's frame, with their signed distances")  # fmt: skip
    p.add_argument("--contacts", action="store_true",
                   help="contact inputs: the robot's collision geoms as spheres, each geom's gap to the objects, its nearest point and relative velocity")  # fmt: skip
    p.add_argument("--contact_spacing", type=float, default=0.0025, help="--contacts: the spheres' spacing over a geom's surface (m)")
    p.add_argument("--contact_cap", type=int, default=384, help="--contacts: the most spheres over one geom's faces (a wider spacing beyond)")
    p.add_argument("--contact_basis", action="store_true",
                   help="--contacts: also, summed over the geoms in contact, the drag and push terms a rigid contact passes on and their moments")  # fmt: skip
    p.add_argument("--contact_scale", action="store_true",
                   help="--contacts: each contact column divided by its spread over the training recordings where that exceeds one "
                        "(a foot on a floor); default: the columns keep their units")  # fmt: skip
    p.add_argument("--contact_pair_margin", action="store_true",
                   help="--contacts: the gaps are read beyond the margin of a contact between a robot geom and an object's or a static geom, "
                        "as the installed MuJoCo sets it from the two; default: beyond the largest of the geoms' margins")  # fmt: skip
    p.add_argument("--contact_reach", type=float, default=None,
                   help="--contacts: only the geoms that come within this gap (m, beyond the contact margin) of an object in a sample of the training states (default: every geom)")  # fmt: skip
    p.add_argument("--contact_objects", nargs="+", default=None, metavar="BODY",
                   help="--contacts: bodies that are objects too although no free joint carries them (one that rides on slides or hinges); "
                        "default: the free objects alone")  # fmt: skip
    p.add_argument("--contact_static", nargs="*", default=None, metavar="GEOM",
                   help="--contacts: also each robot geom's gap to these static geoms of the world body (a floor plane, boxes, ...; no name: "
                        "every collision geom of the world body); default: none")  # fmt: skip
    p.add_argument("--obj_w", type=float, default=1.0,
                   help="the training loss's weight of the objects' error components (free joints other than the anchor; 1: all equal)")  # fmt: skip
    p.add_argument("--loss", default="l1", choices=["l1", "l2", "huber"],
                   help="the training loss of an error component: absolute (the default), squared, or squared within one step's spread and absolute beyond")  # fmt: skip
    p.add_argument("--data_device", default="model", choices=["model", "cpu"],
                   help="where the recordings are held: with the model (the default) or on the host, each update's windows moved to the model (more recordings than the card holds)")  # fmt: skip
    p.add_argument("--init", default=None,
                   help="start from this fit's state (its network and its normalizers, which then stay as they were) instead of a new network: a refit on more recordings")  # fmt: skip
    p.add_argument("--until_flat", type=float, nargs=2, default=None, metavar=("MARGIN", "EVALS"),
                   help="stop when the first test's norm at the largest --eval_k has not fallen MARGIN (relative) below its best for EVALS evaluations in a "
                        "row; --steps is then the cap and the state saved is the best evaluation's")  # fmt: skip
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


def nonfinite_steps(rec: dict) -> torch.Tensor:
    """(T, N) the kept steps of a recording whose state, next state or control is not finite."""
    ok = torch.isfinite(rec["qpos"]).all(-1) & torch.isfinite(rec["qvel"]).all(-1)
    return ~(ok[:-1] & ok[1:] & torch.isfinite(rec["ctrl"]).all(-1)) & ~rec["reset"]


def _to_the_reset(r: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """(T, N) within each rollout (the steps between the resets ``r``), every step from its first ``bad`` one on."""
    start = torch.ones_like(r)
    start[1:] = r[:-1]
    c = bad.long().cumsum(0)
    before = torch.where(start, c - bad.long(), torch.zeros_like(c)).cummax(0).values  # the count at the rollout's start
    return (c - before) > 0


def cut_nonfinite(rec: dict) -> tuple[int, int]:
    """``--cut_nonfinite``: the steps from each rollout's first step that is not finite (``nonfinite_steps``) to its next reset
    become resets (in place). Returns the number of steps that are not finite and the number of kept steps cut."""
    r = rec["reset"]
    bad = nonfinite_steps(rec)
    cut = _to_the_reset(r, bad) & ~r
    rec["reset"] = r | cut
    return int(bad.sum()), int(cut.sum())


def median_increments(recs: list, lay, sample: int = 1_000_000) -> torch.Tensor:
    """(2 nv) each degree of freedom's median absolute increment over the kept steps of ``recs`` (at most ``sample`` steps of
    each, drawn with a fixed seed). A kept step that is not finite is an error (``--cut_nonfinite`` cuts such steps first)."""
    g, kept = torch.Generator(device="cpu").manual_seed(0), []
    for rec in recs:
        k = increments(rec, lay)[~rec["reset"]]
        if not torch.isfinite(k).all():
            lost = int((~torch.isfinite(k).all(-1)).sum())
            raise ValueError(f"median_increments: {lost} kept steps are not finite (fit_mjcf --cut_nonfinite cuts them)")
        kept.append(k[torch.randperm(len(k), generator=g)[:sample].to(k.device)])
    return torch.cat(kept).median(0).values


def cut_blowups(rec: dict, lay, K: float, med: torch.Tensor | None = None, scale_floor: bool = False) -> float:
    """``--cut``: the steps from each rollout's first outlier step to its next reset become resets (in place); the share of the
    kept steps cut. ``med``: the medians to judge by (the training recordings'); by default the recording's own. ``scale_floor``:
    each degree of freedom's scale is the larger of its median and its error floor (``--cut_scale_floor``). A kept step that is
    not finite is an error, not a blow-up to drop (``--cut_nonfinite`` cuts such steps first)."""
    r = rec["reset"]
    inc = increments(rec, lay)
    lost = ~torch.isfinite(inc).all(-1) & ~r
    if lost.any():
        raise ValueError(f"cut_blowups: {int(lost.sum())} kept steps are not finite (fit_mjcf --cut_nonfinite cuts them)")
    med = median_increments([rec], lay) if med is None else med
    floor = torch.as_tensor(lay.floors(), dtype=inc.dtype, device=inc.device)
    scale = torch.maximum(med, floor) if scale_floor else med
    bad = (inc > K * scale + floor).any(-1) & ~r
    cut = _to_the_reset(r, bad)  # within each rollout (the steps between resets), every step from the first bad one on
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


def gather(rec: dict, idx: torch.Tensor, k: int, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The windows at idx (M, 2): qpos, qvel (k + 1, M, .), ctrl (k, M, .); on ``device`` if given (a recording held on the host)."""
    t, n = idx[:, 0], idx[:, 1]
    span = torch.arange(k + 1, device=idx.device)[:, None] + t[None]
    out = rec["qpos"][span, n], rec["qvel"][span, n], rec["ctrl"][span[:-1], n]
    return out if device is None else tuple(x.to(device, non_blocking=True) for x in out)


@torch.no_grad()
def scores(wm, rec: dict, idx: torch.Tensor, k: int) -> dict:
    """The errors after k open-loop steps from the windows' first states (see the module's docstring)."""
    from ndm_gym.worldmodels.mjcf import differentiate_pos

    lay = wm.lay
    qpos, qvel, ctrl = gather(rec, idx, k, wm.err_floor.device)
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
    from ndm_gym.worldmodels.mjcf import (GAPS_REVISION, MjcfLayout, MjcfWorldModel, backbone_kwargs, check_gaps_revision, contact_spheres,
                                          mean_velocity_lead)  # fmt: skip

    backbone = backbone_kwargs(vars(args))  # a refused combination of options fails before the recordings load
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model)
    if args.substeps is not None:  # the recordings' dt is then checked against the model's timestep times the substeps
        if args.dt is not None or args.substeps < 1:
            raise ValueError("fit_mjcf: --substeps (at least 1) or --dt, not both")
        args.dt = float(model.opt.timestep) * args.substeps
    lay = MjcfLayout.from_model(model, anchor=args.anchor, planar_root=args.planar_root)
    if args.dt is not None:  # a coarse stage of the planner steps the same model with a longer timestep
        lay = dataclasses.replace(lay, dt=args.dt)
    lead = {}  # MjcfWorldModel's pos_lead: the option's, or an --init's own
    if args.accel_mean_velocity:
        if args.wm_param != "accel" or args.init is not None:
            raise ValueError("--accel_mean_velocity is an option of --wm_param accel, and not of --init: the start's weight stands")
        if args.substeps is None and args.dt is not None:  # a longer timestep of the same model, or several of its steps?
            raise ValueError("--accel_mean_velocity with --dt: name the step as --substeps of the model's timesteps")
        k = 1 if args.substeps is None else args.substeps
        lead = {"pos_lead": mean_velocity_lead(model, k)}
        print(f"[fit_mjcf] --accel_mean_velocity: the position advances with v + {lead['pos_lead']:g} (v' - v) "
              f"(integrator {int(model.opt.integrator)}, {k} physics steps a control step)", flush=True)  # fmt: skip
        lead = {} if lead["pos_lead"] == 1.0 else lead  # one semi-implicit Euler step: the next velocity, as without the option
    elif args.init is not None:
        start = torch.load(args.init, map_location="cpu", weights_only=False)["state"]
        lead = {"pos_lead": float(start["pos_lead"])} if "pos_lead" in start else {}
        del start
    data_dev = dev if args.data_device == "model" else torch.device("cpu")
    train = [load(p, lay, data_dev) for p in args.train]
    tests = {}
    for spec in args.test:
        name, path = spec.split("=", 1)
        tests[name] = load(path, lay, data_dev)
    named = list(zip(args.train, train)) + [(spec.split("=", 1)[1], tests[spec.split("=", 1)[0]]) for spec in args.test]
    refs = [] if args.cut is None or args.cut_medians_from is None else [(p, load(p, lay, data_dev)) for p in args.cut_medians_from]
    lost = [(path, int(nonfinite_steps(rec).sum())) for path, rec in named + refs]
    if any(n for _, n in lost) and not args.cut_nonfinite:  # never dropped without being asked
        raise ValueError("fit_mjcf: steps whose state or control is not finite: " + ", ".join(f"{n} in {path}" for path, n in lost if n)
                         + "; --cut_nonfinite cuts each rollout from such a step to its next reset")  # fmt: skip
    for path, rec in named + refs if args.cut_nonfinite else []:
        n, cut = cut_nonfinite(rec)
        print(f"[fit_mjcf] --cut_nonfinite: {n} steps of {path} are not finite, {cut} steps cut", flush=True)
    if args.cut is not None:
        ref = train if args.cut_medians_from is None else [rec for _, rec in refs]
        med = median_increments(ref, lay, args.cut_sample)  # the training recordings' scale (or the named ones'), for every recording of the fit
        del ref, refs
        for path, rec in named:
            share = cut_blowups(rec, lay, args.cut, med, args.cut_scale_floor)
            print(f"[fit_mjcf] --cut {args.cut:g}{' (scale floor)' if args.cut_scale_floor else ''}: {100 * share:.3f}% of the steps of {path} cut",
                  flush=True)  # fmt: skip
    if not args.contacts and (args.contact_objects is not None or args.contact_static is not None):
        raise ValueError("--contact_objects and --contact_static are options of --contacts")
    if args.contact_scale and not args.contacts:
        raise ValueError("--contact_scale is an option of --contacts")
    if args.contact_pair_margin and not args.contacts:
        raise ValueError("--contact_pair_margin is an option of --contacts")
    if args.contact_pair_margin and args.init is not None:
        raise ValueError("--contact_pair_margin with --init: the start's margin stands, the one it was fitted with")
    if args.contact_scale and args.init is not None:
        raise ValueError("--contact_scale with --init: the start's normalizers stand, with the scales it was fitted with")
    contacts = None
    if args.contacts:
        more = {} if args.contact_objects is None else {"objects": tuple(args.contact_objects)}  # the defaults' call as it was
        more |= {} if args.contact_static is None else {"static": tuple(args.contact_static)}
        more |= {"pair_margin": True} if args.contact_pair_margin else {}
        contacts = contact_spheres(model, lay, args.contact_spacing, cap=args.contact_cap, **more)
    if contacts is not None and args.contact_basis:
        contacts["basis"] = True
    if contacts is not None and args.contact_reach is not None:  # the geoms that come near an object in the training recordings
        probe = MjcfWorldModel(lay, contacts=contacts).to(dev)
        near, g = torch.zeros(len(contacts["count"]), dtype=torch.bool, device=dev), torch.Generator(device="cpu").manual_seed(0)
        for rec in train:
            q = rec["qpos"][:-1][~rec["reset"]]
            q = q[torch.randperm(len(q), generator=g)[:20000].to(q.device)].to(dev)
            for i in range(0, len(q), 2048):
                near |= (probe.contact_gaps(q[i : i + 2048])[0].amin(-2) < args.contact_reach + contacts["margin"]).flatten(0, -2).any(0)
        keep = near.nonzero().flatten().cpu().numpy()
        if len(keep) == 0:
            raise ValueError(f"--contact_reach {args.contact_reach}: no robot collision geom comes that near an object in the training recordings")
        contacts = {k: (v[keep] if k in ("center", "radius", "body", "count", "geom") else v) for k, v in contacts.items()}
        contacts["center"], contacts["radius"] = (contacts[k][:, : int(contacts["count"].max())] for k in ("center", "radius"))
        del probe
    if contacts is not None:
        print(f"[fit_mjcf] --contacts: {int(contacts['count'].sum())} spheres on {len(contacts['count'])} geoms (at most {contacts['center'].shape[1]} "
              f"on one), margin {1e3 * contacts['margin']:g} mm", flush=True)  # fmt: skip
        if "tb" in contacts or "sg" in contacts:
            print(f"[fit_mjcf] --contacts: {len(contacts.get('tb', ()))} named object bodies, {len(contacts.get('sg', ()))} static geoms", flush=True)
    wm = MjcfWorldModel(lay, hidden=tuple(args.hidden), act=args.act, layer_norm=not args.no_ln, kind=args.wm_param, geometry=args.geometry,
                        contacts=contacts, **backbone, **lead).to(dev)  # fmt: skip
    scale = {"contact_scale": True} if args.contact_scale else {}  # the defaults' call as it was
    for rec in train if args.init is None else []:  # the normalizers from every training transition, before the fit (--init keeps its own)
        ok = ~rec["reset"]
        wm.update_norms(*(x.to(dev) for x in (rec["qpos"][:-1][ok], rec["qvel"][:-1][ok], rec["ctrl"][ok], rec["qpos"][1:][ok], rec["qvel"][1:][ok])),
                        **scale)  # fmt: skip
    if args.contact_scale:
        cols = wm.tok_contact_cols if wm.backbone == "rigidformer" else slice(wm.n_in - wm.n_contact, wm.n_in)
        s = wm.in_norm.var[cols].sqrt()
        print(f"[fit_mjcf] --contact_scale: {int((s > 1).sum())} of {s.numel()} contact columns divided by their spread, the largest "
              f"{float(s.max()):.3g}", flush=True)  # fmt: skip
    if args.init is not None:  # after the normalizers' pass, so that the loaded ones stand
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        check_gaps_revision(init, f"fit_mjcf: --init {args.init}")  # a fit on the static geoms' gaps of before is not a start
        saved = json.loads(init["layout"])
        for key in ("nq", "nv", "nu", "anchor", "objects"):  # as speed_mjcf.load_ndm: the frame and the state the network was fitted in
            if json.loads(json.dumps(getattr(lay, key))) != saved[key]:
                raise ValueError(f"fit_mjcf: --init {args.init} was fitted on another layout ({key} {saved[key]}, this model's is {getattr(lay, key)})")
        state = torch.load(args.init, map_location=dev, weights_only=False)["state"]
        if "cs_basis" in wm.state_dict() and "cs_basis" not in state:  # a contact fit from before --contact_basis
            state["cs_basis"] = torch.tensor(False, device=dev)
        wm.load_state_dict(state)
        print(f"[fit_mjcf] --init: the state of {args.init}", flush=True)
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
    weights = None  # the loss's weights of the error components: all equal unless --obj_w
    if args.obj_w != 1.0:
        if not lay.objects:
            raise ValueError("--obj_w: the model has no object (a free joint other than the anchor)")
        weights = torch.ones(2 * lay.nv, device=dev)
        for _, d in lay.objects:
            weights[d : d + 6] = weights[lay.nv + d : lay.nv + d + 6] = args.obj_w
    shares = torch.tensor([len(x) for x in pools], dtype=torch.float)
    flat_on = (args.test[0].split("=", 1)[0], max(args.eval_k))  # --until_flat reads this test's norm
    best, best_step, best_state, waited, stopped = float("inf"), 0, None, 0, None
    t0, loss = time.time(), float("nan")
    for step in range(1, args.steps + 1):
        r = int(torch.multinomial(shares, 1))
        idx = pools[r][torch.randint(len(pools[r]), (args.batch,), device=pools[r].device)]
        qpos, qvel, ctrl = gather(train[r], idx, args.window, None if data_dev == dev else dev)
        kw = ({} if weights is None else {"weights": weights}) | ({} if args.loss == "l1" else {"loss": args.loss})  # the defaults' call as it was
        x = window_loss(qpos, qvel, ctrl, **kw)
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
                    if args.until_flat is not None and (name, k) == flat_on:
                        if s["norm"] < best * (1.0 - args.until_flat[0]):
                            best, best_step, waited = s["norm"], step, 0
                            best_state = {key: x.detach().clone() for key, x in wm.state_dict().items()}
                        else:
                            waited += 1
            print(f"[fit_mjcf] step {step} loss {loss:.4f} ({time.time() - t0:.0f} s)", flush=True)
            if args.until_flat is not None and waited >= args.until_flat[1]:
                stopped = step
                break
    extra = {}
    if args.until_flat is not None:
        if best_state is None:
            raise RuntimeError(f"fit_mjcf: --until_flat saw no evaluation of {flat_on[0]} at {flat_on[1]} steps within {args.steps} updates")
        wm.load_state_dict(best_state)
        extra["until_flat"] = {"test": flat_on[0], "k": flat_on[1], "best": best, "best_step": best_step, "stopped": stopped, "cap": args.steps}
        print(f"[fit_mjcf] --until_flat: {'stopped at step ' + str(stopped) if stopped else 'the cap of ' + str(args.steps) + ' updates reached'}; "
              f"the state kept is step {best_step}'s ({flat_on[0]} norm at {flat_on[1]} steps {best:.4f})", flush=True)  # fmt: skip
    if contacts is not None and "sg" in contacts:  # a fit with static geoms carries the revision of the gaps it was fitted on
        extra["contact_gaps"] = GAPS_REVISION
    if args.save:
        torch.save({"state": wm.state_dict(), "layout": json.dumps(lay.__dict__), "args": vars(args), **extra}, args.save)


if __name__ == "__main__":
    main()
