"""Evaluate an animal tracking checkpoint (ndm_gym.train_animal) in the simulator: each clip played from its first frame (and
``--repeats`` more times from perturbed starts) to its last, the policy without exploration noise. Reports, over the steps
the episodes lived, the tracked bodies' and the end effectors' position errors and the root's (mm: the fly's centimetres and
the others' metres converted), the joints' RMS error (deg), and the share of episodes ended before their clip's end.

    python -m ndm_gym.evaluate_animal --ckpt logs/<run>/last.pt [--clip_ids 0-9] [--out <dir>]

``--clip_ids`` defaults to the run's clips; held-out clips evaluate generalization. ``--out`` writes ``metrics.json`` and,
with ``--record``, ``rollout.npz`` (each world's qpos per step, the reference's, the frames) for tools/render_animal.py.
"""

from __future__ import annotations

import argparse
import json
import os

MM = {"rat": 1000.0, "stick": 1000.0, "fly": 10.0}  # the body's length unit in mm


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clip_ids", type=str, nargs="+", default=None, help="indices or ranges (default: the run's clips)")
    parser.add_argument("--repeats", type=int, default=0, help="also play each clip this many times from perturbed starts")
    parser.add_argument("--joint_noise", type=float, default=0.02, help="the perturbed starts' joint noise (rad)")
    parser.add_argument("--data_dir", type=str, default=None, help="default: the run's")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch

    from ndm_gym.policies.hand_policy import HandPolicy
    from ndm_gym.robots.animals import ANIMALS
    from ndm_gym.sim.rigid.animal_sim import AnimalSim
    from ndm_gym.train_animal import build, parse_clips

    dev = "cuda:0"
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    run = argparse.Namespace(**ck["args"])
    clip_ids = parse_clips(args.clip_ids or run.clip_ids)
    data_dir = args.data_dir or run.data_dir
    spec = ANIMALS[run.animal]
    model, _, _, task = build(run.animal, clip_ids, data_dir, dev, run.reward_std_scale, run.action_scale, run.goal_stride,
                             fused_fk=getattr(run, "fused_fk", False))
    ref = task.ref
    starts = torch.unique(ref.start)
    reps = 1 + args.repeats
    f0 = starts.repeat(reps)
    N = f0.shape[0]
    sim = AnimalSim(model, task, N, dev, spec.ctrl_dt, episode=10**9, joint_noise=0.0, nconmax=spec.nconmax, njmax=spec.njmax)
    sim.reset(torch.ones(N, dtype=torch.bool, device=dev), frame=f0)
    if args.repeats:  # the perturbed starts: noise on the joints of the repeated worlds
        noisy = torch.arange(N, device=dev) >= starts.shape[0]
        sim.qpos[noisy.nonzero().squeeze(-1)[:, None], task.jq] += args.joint_noise * torch.randn(int(noisy.sum()), task.nj, device=dev)
    goal_dim = task.goal_frame_dim * run.goal_frames
    pi = HandPolicy(task.obs_dim, goal_dim, task.nu).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()

    alive = torch.ones(N, dtype=torch.bool, device=dev)
    ended = torch.zeros(N, dtype=torch.bool, device=dev)
    a_prev = torch.zeros(N, task.nu, device=dev)
    sums = {k: torch.zeros(N, device=dev) for k in ("body", "end_eff", "root_pos", "joint")}
    steps = torch.zeros(N, device=dev)
    frames, qposes = [], []
    with torch.no_grad():
        while bool(alive.any()):
            S, f = task.unflatten(sim.state()), sim.f.clone()
            a = pi(task.obs(S, task.kin(S[0]), a_prev), task.goal(S, f, run.goal_frames))
            s1, term, done = sim.step(task.setpoint(a, f), reset=False)
            S1, f1 = task.unflatten(s1), sim.f
            e = task.errors(S1, task.kin(S1[0]), f1)
            # the ended worlds step on unreset and may diverge: their errors are left out by selection, not by a zero weight
            zero = torch.zeros_like(steps)
            sums["body"] += torch.where(alive, e["body"].mean(-1), zero)
            sums["end_eff"] += torch.where(alive, e["end_eff"].mean(-1), zero)
            sums["root_pos"] += torch.where(alive, e["root_pos"], zero)
            sums["joint"] += torch.where(alive, e["joint"], zero)
            steps += alive.float()
            ended |= alive & term
            if args.record:
                frames.append(f1.cpu().numpy()), qposes.append(S1[0].cpu().numpy())
            alive &= ~done
            a_prev = a
    mm = MM[run.animal]
    n = steps.clamp_min(1)
    metrics = {
        "checkpoint": os.path.abspath(args.ckpt), "iteration": ck.get("iteration"), "animal": run.animal, "clips": len(starts),
        "episodes": N, "ended_share": float(ended.float().mean()),
        "body_mm": float((sums["body"] / n).mean() * mm), "end_eff_mm": float((sums["end_eff"] / n).mean() * mm),
        "root_pos_mm": float((sums["root_pos"] / n).mean() * mm), "joint_deg": float(np.degrees(float((sums["joint"] / n).mean()))),
        "lived_share": float((steps / (ref.end[f0] - f0).float()).mean()),
    }  # fmt: skip
    print(json.dumps(metrics, indent=2))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        if args.record:
            np.savez_compressed(os.path.join(args.out, "rollout.npz"), qpos=np.stack(qposes, 1), frame=np.stack(frames, 1),
                                ref_qpos=ref.qpos.cpu().numpy(), start=f0.cpu().numpy(), animal=run.animal)  # fmt: skip


if __name__ == "__main__":
    main()
