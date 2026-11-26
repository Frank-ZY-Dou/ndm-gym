"""Evaluate an animal tracking checkpoint (ndm_gym.train_animal) in the simulator: each clip played from its first frame (and
``--repeats`` more times from perturbed starts) to its last, the policy without exploration noise.

    python -m ndm_gym.evaluate_animal --ckpt logs/<run>/last.pt [--clip_ids 10-19] [--repeats 19] [--seed 0] [--out <dir>]

The episodes start as the run's did, at its reference's positions and velocities (train_animal's ``--ref_qvel``);
``metrics.json``'s ``ended_by_clip`` counts the ended episodes per clip.

Reports first the share of the episodes ended by a termination before their clip's end (``ended_share``), then the errors
of the episodes that reached it (``completed``): per episode the mean over its steps, then the mean over these episodes, of
the tracked bodies' and the end effectors' position errors and the root's (mm: the fly's centimetres and the others' metres
converted) and the joints' RMS error (deg); none when no episode completed. ``lived`` holds the same errors over the steps
that every episode lived, the ended ones' too (not comparable between runs that end different shares of their episodes).
``size_mm`` is the body's size (the task's ``length``: the median over the reference of the tracked bodies' largest distance
from the root), the unit of the benchmark's target.

``--clip_ids`` defaults to the run's clips; held-out clips evaluate generalization. ``--out`` writes ``metrics.json`` and,
with ``--record``, ``rollout.npz`` (each world's qpos per step, the reference's, the frames, the steps each episode lived and
whether it ended) for tools/render_animal.py; a world steps on after its episode ended. ``--ckpts`` evaluates several
checkpoints of one run in one process (one build), each from the same starts, into ``<out>/<iteration>/metrics.json``.
``Evaluator`` is the same evaluation for a process that evaluates checkpoints as they appear or one after another
(benchmarks/mimic_mjx_animal_tracking/watch.py, final.py).
"""

from __future__ import annotations

import argparse
import json
import os

MM = {"rat": 1000.0, "stick": 1000.0, "fly": 10.0}  # the body's length unit in mm
ERRORS = ("body_mm", "end_eff_mm", "root_pos_mm", "joint_deg")
# what the checkpoints of one --ckpts evaluation share, with each option's value in a run of before it
SAME_RUN = (("animal", None), ("clip_ids", None), ("ref_qvel", "stored"), ("action_scale", None), ("goal_stride", None),
            ("reward_std_scale", None), ("fused_fk", None))  # fmt: skip


def same_run(first: dict, other: dict) -> str | None:
    """Why the checkpoint whose run's args are ``other`` cannot be evaluated with the evaluator built for ``first``'s run,
    or None: their body, clips, the task's reference options and the task's settings the evaluator builds it with (the
    action scale, the goals' stride, the reward widths, the fused kinematics) must be the same."""
    for k, before in SAME_RUN:
        if first.get(k, before) != other.get(k, before):
            return f"--ckpts evaluates the checkpoints of one run: their {k} differ ({first.get(k, before)!r}, {other.get(k, before)!r})"
    return None


def summarize(sums: dict, steps, ended, scale: dict) -> dict:
    """A batch of episodes' summary: the episodes, those ``ended`` by a termination and their share, those completed; per
    episode the mean of each error over the steps it lived (``sums[k] / steps``, times ``scale[k]``), and the means of these
    over the completed episodes (None when none completed) and, under ``lived``, over every episode."""
    per_episode = {k: sums[k] / steps.clamp_min(1) * scale[k] for k in ERRORS}
    ok = ~ended
    n, n_ok = int(ended.numel()), int(ok.sum())
    return {"episodes": n, "ended": n - n_ok, "ended_share": float(ended.float().mean()), "completed": n_ok,
            **{k: (float(per_episode[k][ok].mean()) if n_ok else None) for k in ERRORS},
            "lived": {k: float(per_episode[k].mean()) for k in ERRORS}}  # fmt: skip


class Evaluator:
    """The evaluation of one run's checkpoints (``run``: a checkpoint's ``args``) on ``clip_ids`` (None: the run's clips): the
    task, its simulator and the policy built once; every checkpoint from the same starts (the draws of ``seed``)."""

    def __init__(self, run: dict, clip_ids: list[int] | None = None, repeats: int = 0, joint_noise: float = 0.02, seed: int = 1,
                 data_dir: str | None = None, device: str = "cuda:0"):  # fmt: skip
        import torch

        from ndm_gym.policies.hand_policy import HandPolicy
        from ndm_gym.robots.animals import ANIMALS
        from ndm_gym.sim.rigid.animal_sim import AnimalSim
        from ndm_gym.train_animal import build, parse_clips

        if repeats < 0 or joint_noise < 0:
            raise ValueError(f"--repeats >= 0 and --joint_noise >= 0, got {repeats}, {joint_noise}")
        r = self.run = argparse.Namespace(**run)
        self.clip_ids = parse_clips(r.clip_ids) if clip_ids is None else list(clip_ids)
        self.repeats, self.joint_noise, self.seed, self.dev = repeats, joint_noise, seed, device
        spec = ANIMALS[r.animal]
        self.ref_qvel = getattr(r, "ref_qvel", "stored")  # the run's reference velocities (a run of before the option: the stored ones)
        model, _, _, self.task = build(r.animal, self.clip_ids, data_dir or r.data_dir, device, r.reward_std_scale, r.action_scale,
                                       r.goal_stride, fused_fk=r.fused_fk, ref_qvel=self.ref_qvel)  # fmt: skip
        self.starts = torch.unique(self.task.ref.start)
        self.f0 = self.starts.repeat(1 + repeats)
        self.clip_of = torch.arange(self.starts.shape[0], device=device).repeat(1 + repeats)  # each episode's clip, in clip_ids' order
        self.N = self.f0.shape[0]
        self.sim = AnimalSim(model, self.task, self.N, device, spec.ctrl_dt, episode=10**9, joint_noise=0.0, nconmax=spec.nconmax,
                             njmax=spec.njmax)  # fmt: skip
        self.pi = HandPolicy(self.task.obs_dim, self.task.goal_frame_dim * r.goal_frames, self.task.nu).to(device)
        self.mm = MM[r.animal]

    def __call__(self, ck: dict, record: bool = False) -> tuple[dict, dict | None]:
        """The metrics of the checkpoint ``ck`` (its ``pi``, ``iteration`` and ``wall_time``), and with ``record`` the rollout's
        arrays (rollout.npz's)."""
        import numpy as np
        import torch

        task, sim, dev, N = self.task, self.sim, self.dev, self.N
        torch.manual_seed(self.seed)
        sim.reset(torch.ones(N, dtype=torch.bool, device=dev), frame=self.f0)
        if self.repeats:  # the perturbed starts: noise on the joints of the repeated worlds
            noisy = torch.arange(N, device=dev) >= self.starts.shape[0]
            sim.qpos[noisy.nonzero().squeeze(-1)[:, None], task.jq] += self.joint_noise * torch.randn(int(noisy.sum()), task.nj, device=dev)
        self.pi.load_state_dict(ck["pi"])
        self.pi.eval()
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        ended = torch.zeros(N, dtype=torch.bool, device=dev)
        a_prev = torch.zeros(N, task.nu, device=dev)
        sums = {k: torch.zeros(N, device=dev) for k in ERRORS}
        steps = torch.zeros(N, device=dev)
        frames, qposes = [], []
        with torch.no_grad():
            while bool(alive.any()):
                S, f = task.unflatten(sim.state()), sim.f.clone()
                a = self.pi(task.obs(S, task.kin(S[0]), a_prev), task.goal(S, f, self.run.goal_frames))
                s1, term, done = sim.step(task.setpoint(a, f), reset=False)
                S1, f1 = task.unflatten(s1), sim.f
                e = task.errors(S1, task.kin(S1[0]), f1)
                # the ended worlds step on unreset and may diverge: their errors are left out by selection, not by a zero weight
                zero = torch.zeros_like(steps)
                for k, x in (("body_mm", e["body"].mean(-1)), ("end_eff_mm", e["end_eff"].mean(-1)), ("root_pos_mm", e["root_pos"]),
                             ("joint_deg", e["joint"])):  # fmt: skip
                    sums[k] += torch.where(alive, x, zero)
                steps += alive.float()
                ended |= alive & term
                if record:
                    frames.append(f1.cpu().numpy()), qposes.append(S1[0].cpu().numpy())
                alive &= ~done
                a_prev = a
        scale = {"body_mm": self.mm, "end_eff_mm": self.mm, "root_pos_mm": self.mm, "joint_deg": 180.0 / np.pi}
        metrics = {
            "checkpoint": ck.get("path"), "iteration": int(ck["iteration"]), "minutes": float(ck["wall_time"]) / 60, "animal": self.run.animal,
            "clips": len(self.starts), "clip_ids": self.clip_ids, "repeats": self.repeats, "joint_noise": self.joint_noise, "seed": self.seed,
            "ref_qvel": self.ref_qvel,
            **summarize(sums, steps, ended, scale),
            "ended_by_clip": [int(ended[self.clip_of == c].sum()) for c in range(len(self.starts))],  # in clip_ids' order
            "lived_share": float((steps / (self.task.ref.end[self.f0] - self.f0).float()).mean()),
            "size_mm": self.task.length * self.mm,
        }  # fmt: skip
        rollout = None
        if record:
            rollout = {"qpos": np.stack(qposes, 1), "frame": np.stack(frames, 1), "ref_qpos": self.task.ref.qpos.cpu().numpy(),
                       "start": self.f0.cpu().numpy(), "animal": self.run.animal, "lived": steps.long().cpu().numpy(),
                       "ended": ended.cpu().numpy()}  # fmt: skip
        return metrics, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpts", type=str, nargs="+", default=None, help="several checkpoints of one run, each into <out>/<iteration>")
    parser.add_argument("--clip_ids", type=str, nargs="+", default=None, help="indices or ranges (default: the run's clips)")
    parser.add_argument("--repeats", type=int, default=0, help="also play each clip this many times from perturbed starts")
    parser.add_argument("--joint_noise", type=float, default=0.02, help="the perturbed starts' joint noise (rad)")
    parser.add_argument("--data_dir", type=str, default=None, help="default: the run's")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if (args.ckpt is None) == (args.ckpts is None):
        parser.error("give --ckpt or --ckpts")
    if args.record and args.out is None:
        parser.error("--record writes <out>/rollout.npz: give --out")
    if args.repeats < 0:
        parser.error("--repeats >= 0")
    if not args.repeats and args.joint_noise != parser.get_default("joint_noise"):
        parser.error("--joint_noise perturbs the repeated starts: give --repeats")
    paths = args.ckpts or [args.ckpt]

    import numpy as np
    import torch

    from ndm_gym.train_animal import parse_clips

    dev = "cuda:0"
    ev = None
    for path in paths:
        ck = torch.load(path, map_location=dev, weights_only=False)
        for k in ("pi", "iteration", "wall_time", "args"):
            if k not in ck:
                raise ValueError(f"{path}: the checkpoint has no {k!r} (train_animal writes it)")
        if ev is None:
            ev = Evaluator(ck["args"], parse_clips(args.clip_ids) if args.clip_ids else None, args.repeats, args.joint_noise, args.seed,
                           args.data_dir, dev)  # fmt: skip
        elif (why := same_run(vars(ev.run), ck["args"])) is not None:
            raise ValueError(f"{path}: {why}")
        metrics, rollout = ev({**ck, "path": os.path.abspath(path)}, record=args.record)
        print(json.dumps(metrics, indent=2))
        if args.out:
            out = os.path.join(args.out, f"{ck['iteration']:06d}") if args.ckpts else args.out
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "metrics.json"), "w") as fh:
                json.dump(metrics, fh, indent=2)
            if rollout is not None:
                np.savez_compressed(os.path.join(out, "rollout.npz"), **rollout)


if __name__ == "__main__":
    main()
