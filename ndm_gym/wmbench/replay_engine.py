"""Windows of a G1 recording for an engine that runs outside this environment (benchmarks/steel: Steel runs in its own),
and the open-loop errors of engines and world models on them, measured one way for all (benchmarks/steel/results.md).

export  the windows of a recording (wmbench/collect.py), drawn as wmbench/speed_pair.py --replay draws them (fit.py's
        fixed_windows on the CPU), with the G1's constants an engine needs (joint names, default pose, action scales, drive
        gains, armature, effort limits; ndm_gym/robots/g1.py), as one .npz: first states (n, 71), setpoints (n, len, 29),
        contexts (n, 33), the recorded states after every step (n, len, 71) and their frames.
score   against the recording: an engine's rollouts (``name=file.npz`` with ``states`` (n, len, 71) in our layout, from the
        windows' first states with their setpoints) and the world models of training checkpoints (``name=ckpt.pt``, rolled
        over the same windows with the recorded setpoints and contexts as wmbench/fit.py rolls them, eager, the run's
        autocast). After 1, 8 and 32 steps (those within --len): the tracked bodies' position error (mm, mean over the
        bodies), the joint angle error (rad, mean), the root position error (mm) and rotation error (rad); over every step the
        foot spheres' penetration of the ground (the MJCF's collision spheres on the ankles; mean over contacting spheres and
        peak, mm) and the task's terminations (tasks/tracking.py's test on the recorded frames). The recording itself is
        scored as "recording" (its penetration and terminations). Means come with their standard error over windows. One
        JSON line per source to --out.

Run: ``python -m ndm_gym.wmbench.replay_engine export --recording rec.pt --windows 8192 --out win.npz`` and
``python -m ndm_gym.wmbench.replay_engine score --windows win.npz --rollout steel=r.npz --ckpt mlp=last.pt --out s.jsonl``
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HORIZONS = (1, 8, 32)


def g1_constants() -> dict:
    """The G1's per-joint constants in G1_JOINT_NAMES order, as TrackingSim's actuators and targets use them."""
    from ndm_gym.robots import g1

    names = g1.G1_JOINT_NAMES
    motors = [g1.motor_of(n) for n in names]
    return {"joint_names": np.array(names), "default_q": np.array(g1.default_joint_pos_list(), np.float32),
            "alpha": np.array([g1.action_scale(n) for n in names], np.float32),
            "kp": np.array([g1.kp(m[2]) for m in motors], np.float32), "kd": np.array([g1.kd(m[2]) for m in motors], np.float32),
            "armature": np.array([m[2] for m in motors], np.float32), "effort": np.array([m[0] for m in motors], np.float32),
            "velocity": np.array([m[1] for m in motors], np.float32)}  # fmt: skip


def foot_spheres(mjcf: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The MJCF's colliding spheres: their bodies' names, positions in those bodies' frames (k, 3) and radii (k,)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    k = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and (m.geom_contype[g] or m.geom_conaffinity[g])]
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in k]
    return bodies, m.geom_pos[k].copy(), m.geom_size[k, 0].copy()


class Scorer:
    """The measures of the module's docstring for states (n, len, 71) against the recorded ones."""

    def __init__(self, win: dict, dev: str, motion_dir: str):
        from ndm_gym.motion.library import MotionLib
        from ndm_gym.motion.sources import lafan1
        from ndm_gym.robots import g1
        from ndm_gym.tasks.tracking import MDP

        self.dev = dev
        self.kin = g1.kinematics(dev)
        self.mdp = MDP(self.kin, MotionLib(lafan1.load_dir(motion_dir), self.kin, dev), torch.tensor(g1.default_joint_pos_list(), device=dev))
        bodies, pos, rad = foot_spheres(g1.MJCF_PATH)
        self.sph_body = torch.tensor([self.kin.body_names.index(b) for b in bodies], device=dev)
        self.sph_pos = torch.tensor(pos, dtype=torch.float32, device=dev)
        self.sph_rad = torch.tensor(rad, dtype=torch.float32, device=dev)
        self.rec = torch.as_tensor(win["s1"], device=dev)  # (n, L, 71)
        self.f1 = torch.as_tensor(win["f"], device=dev) + 1  # the frame of each recorded state after the step

    def _fk(self, x: torch.Tensor):
        from ndm_gym.state import unflatten_state

        S = unflatten_state(x)
        return S, self.kin.forward(*S, all_bodies=True)

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, chunk: int = 8192) -> dict:
        n, L = states.shape[:2]
        sums = {}
        pen_sum = pen_n = 0.0
        pen_peak = 0.0
        term_any = torch.zeros(n, dtype=torch.bool, device=self.dev)
        finite = torch.isfinite(states).all(-1).all(-1).to(self.dev)
        per_h = {h: {"body_mm": [], "joint_rad": [], "root_mm": [], "root_rad": []} for h in HORIZONS if h <= L}
        for c0 in range(0, n, chunk):
            x = states[c0:c0 + chunk].to(self.dev)
            r = self.rec[c0:c0 + chunk]
            for j in range(L):
                S, (bp, bR, _, _) = self._fk(x[:, j])
                if j + 1 in per_h:
                    S0, (rp, rR, _, _) = self._fk(r[:, j])
                    tr = self.kin.tracked_idx
                    d = per_h[j + 1]
                    d["body_mm"].append(1e3 * (bp[:, tr] - rp[:, tr]).norm(dim=-1).mean(-1))
                    d["joint_rad"].append((S[4] - S0[4]).abs().mean(-1))
                    d["root_mm"].append(1e3 * (S[0] - S0[0]).norm(dim=-1))
                    rel = S0[1].transpose(-1, -2) @ S[1]
                    d["root_rad"].append(torch.arccos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
                z = (bp[:, self.sph_body] + (bR[:, self.sph_body] @ self.sph_pos.unsqueeze(-1)).squeeze(-1))[..., 2]
                pen = (self.sph_rad - z).clamp(min=0)  # (b, k): depth below the ground plane z = 0
                hit = pen > 0
                pen_sum += float(pen[hit].sum())
                pen_n += int(hit.sum())
                pen_peak = max(pen_peak, float(pen.max()))
                term_any[c0:c0 + chunk] |= self.mdp.termination(S, self.f1[c0:c0 + chunk, j])
        for h, d in per_h.items():
            for k, v in d.items():
                v = torch.cat(v)
                sums[f"{k}@{h}"] = [float(v.mean()), float(v.std() / v.numel() ** 0.5)]
        rate = term_any.float()
        return {"windows": n, "steps": L, "finite_windows": int(finite.sum()), **sums,
                "penetration_mean_mm": 1e3 * pen_sum / max(pen_n, 1), "penetration_peak_mm": 1e3 * pen_peak, "contact_sphere_steps": pen_n,
                "terminated_windows": [float(rate.mean()), float(rate.std() / rate.numel() ** 0.5)]}  # fmt: skip


@torch.no_grad()
def world_model_states(ckpt: str, win: dict, dev: str, chunk: int = 8192) -> torch.Tensor:
    """The world model of a training checkpoint rolled over the windows (eager, under the run's autocast): (n, len, 71)."""
    from types import SimpleNamespace

    import ndm_gym.core.nets as nets
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.wmbench.speed_pair import build_world_model

    wm, ta, _ = build_world_model(SimpleNamespace(wm_ckpt=ckpt, wm_args=None), dev)
    wm.eval()
    amp = nets.AMP_DTYPE
    nets.AMP_DTYPE = torch.bfloat16 if getattr(ta, "amp", "none") == "bf16" else None
    try:
        s0, a, xi = (torch.as_tensor(win[k]) for k in ("s0", "a", "xi"))
        out = torch.empty(a.shape[0], a.shape[1], s0.shape[-1])
        for c0 in range(0, len(s0), chunk):
            S = unflatten_state(s0[c0:c0 + chunk].to(dev))
            xn = normalize_xi(xi[c0:c0 + chunk].to(dev))
            for j in range(a.shape[1]):
                S = wm.step(S, a[c0:c0 + chunk, j].to(dev), xn)
                out[c0:c0 + chunk, j] = flatten_state(S).float().cpu()
    finally:
        nets.AMP_DTYPE = amp
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--recording", required=True)
    e.add_argument("--windows", type=int, default=8192)
    e.add_argument("--len", type=int, default=32)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--windows", required=True, help="export's .npz")
    s.add_argument("--rollout", nargs="*", default=[], help="name=file.npz with states (n, len, 71)")
    s.add_argument("--ckpt", nargs="*", default=[], help="name=checkpoint.pt whose world model is rolled")
    s.add_argument("--motion_dir", default=os.path.join(REPO, "data", "lafan1_g1"))
    s.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    s.add_argument("--out", required=True, help="JSON lines, one per source")
    args = p.parse_args()

    if args.cmd == "export":
        from ndm_gym.wmbench.speed_pair import replay_windows

        w = replay_windows(args.recording, args.windows, args.len, args.seed, "cpu", full=True)
        meta = torch.load(args.recording, map_location="cpu", mmap=True)["meta"]
        np.savez(args.out, **{k: v.numpy() for k, v in w.items()}, **g1_constants(), recording=os.path.abspath(args.recording),
                 seed=args.seed, recording_meta=json.dumps(meta, default=str))  # fmt: skip
        print(f"[replay_engine] {args.windows} windows of {args.len} steps of {args.recording} -> {args.out}", flush=True)
        return

    win = dict(np.load(args.windows, allow_pickle=False))
    scorer = Scorer(win, args.device, args.motion_dir)
    sources = [("recording", torch.as_tensor(win["s1"]))]
    for item in args.rollout:
        name, path = item.split("=", 1)
        sources.append((name, torch.as_tensor(np.load(path)["states"])))
    for item in args.ckpt:
        name, path = item.split("=", 1)
        sources.append((name, world_model_states(path, win, args.device)))
    with open(args.out, "a") as f:
        for name, states in sources:
            if states.shape[:2] != win["a"].shape[:2]:
                raise ValueError(f"{name}: states {tuple(states.shape)} for {win['a'].shape[0]} windows of {win['a'].shape[1]} steps")
            row = {"source": name, "windows_file": os.path.abspath(args.windows), **scorer(states)}
            f.write(json.dumps(row) + "\n")
            print(f"[replay_engine] {name}: " + ", ".join(f"{k} {v}" for k, v in row.items() if "@" in k or "penetration" in k
                                                       or k == "terminated_windows"), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
