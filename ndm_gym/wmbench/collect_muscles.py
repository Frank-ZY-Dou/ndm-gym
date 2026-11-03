"""Collect transitions of MuscleVAE's muscle-actuated character (ndm_gym.sim.rigid.muscle_sim.MuscleSim) for the neural
dynamics models, in replay-buffer layout (MuscleSim.transition_fields: the bodies' states before and after the control
step, the action, the muscles' fatigue states before and after, the reference frame, the episode step, the flags).

Policy: their released tracking policy (ndm_gym.policies.musclevae_pretrained) as their data collection runs it (TC:53-61):
the posterior's latent sampled (std 0.3), Gaussian action noise (std --action_noise, theirs 0.05), and with probability
--prior_prob (theirs 0.4) a step from the prior's sampled latent instead. Environment: their training environment's
(MuscleSim with ball_joints "euler" and joint_damping "ode", their simulator; random reference switches every 96 steps
and at a clip's end; resets of the environments that end, to frames drawn uniformly: theirs reweights them by returns).

The transitions stay on the device in contiguous (rows, N, ...) buffers that the steps write in place (MuscleSim's
before_step and transition with ``out``), --chunk rows at a time; each chunk is packed (wmbench.collect.pack: the next
state and the next fatigue state only where they differ from the next row's, the episode step and reference frame only
where they do not count on) and saved, the only host copy. Size in the full layout: 10,358 bytes per transition, 16.97
GB per 400 x 4096 transitions (two thirds of it the 284 muscles' fatigue states before and after); packed about half.
GPU memory: the buffers, 10,358 bytes x --num_envs x --chunk (1.06 GB at 4096 x 25, 4.24 GB at 4096 x 100), the
packing's copies of most of them, the simulator and the policy. The process's peak at 4096 environments (an A100): 4.3
GB at the default --chunk 25, 5.6 GB at 40, 10.7 GB at 100.

Run: ``python -m ndm_gym.wmbench.collect_muscles --num_envs 4096 --steps 400 --chunk 25 --out data_wm/muscles/<name>``
(files <name>_000.pt, ... with {"meta", "data"} as wmbench.collect's). TC = MuscleVAECore/Model/trajectory_collection.py of
wangchek/MuscleVAE_ @ 54c001c.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

MUSCLE_RULES = {"s1": ("next", "s"), "fatigue1": ("next", "fatigue"), "ep_step": ("count", None), "t": ("count", None)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="path prefix of the chunk files")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=400, help="control steps per environment")
    p.add_argument("--chunk", type=int, default=25,
                   help="rows per file; the device buffers hold one chunk, 10,358 bytes x num_envs x chunk, and packing copies most of it: the "
                        "process's peak GPU memory at 4096 environments 4.3 GB at chunk 25, 5.6 GB at 40, 10.7 GB at 100")  # fmt: skip
    p.add_argument("--latent", type=str, default="sample", choices=["sample", "mean"])
    p.add_argument("--action_noise", type=float, default=0.05)
    p.add_argument("--prior_prob", type=float, default=0.4)
    p.add_argument("--ball_joints", type=str, default="euler", choices=["euler", "ball"])
    p.add_argument("--joint_damping", type=str, default="ode", choices=["ode", "implicit"])
    p.add_argument("--muscle_model", type=str, default="released", choices=["released", "paper"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    from ndm_gym.policies import musclevae_pretrained as mp
    from ndm_gym.sim.rigid.muscle_sim import RANDOM_COUNT, MuscleSim
    from ndm_gym.wmbench.collect import pack

    if args.steps % args.chunk:
        raise ValueError(f"--steps {args.steps} must be a multiple of --chunk {args.chunk}")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    dev = "cuda:0"
    sim = MuscleSim(args.num_envs, dev, muscle_model=args.muscle_model, random_count=RANDOM_COUNT, seed=args.seed, ball_joints=args.ball_joints,
                    joint_damping=args.joint_damping, capture_step=False)  # fmt: skip
    runner = mp.PolicyRunner(sim, mp.MuscleVAEPolicy.load(device=dev), latent=args.latent, action_noise=args.action_noise, prior_prob=args.prior_prob)
    buf = sim.transition_buffers(args.chunk)
    rows = [{k: v[t] for k, v in buf.items()} for t in range(args.chunk)]
    per_transition = sum(v[0, 0].numel() * v.element_size() for v in buf.values())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_term = n_done = torch.zeros((), dtype=torch.long, device=dev)  # counted on the device: no synchronization per step
    t0 = time.time()
    saved = []
    for c in range(args.steps // args.chunk):
        for row in rows:
            sim.before_step(out=row)
            terminated, truncated = runner.step()
            sim.transition(row, sim.action, terminated, truncated, out=row)
            done = terminated | truncated
            n_term, n_done = n_term + terminated.sum(), n_done + done.sum()
            runner.reset(done)
        data = pack(buf, MUSCLE_RULES)
        meta = {"num_envs": args.num_envs, "steps": args.chunk, "chunk": c, "chunks": args.steps // args.chunk, "seed": args.seed,
                "policy": f"MuscleVAE pretrained tracking policy ({mp.CHECKPOINT_SHA256[:12]}), latent {args.latent}, action noise {args.action_noise}, "
                          f"prior steps {args.prior_prob}",
                "simulator": {"ball_joints": args.ball_joints, "joint_damping": args.joint_damping, "muscle_model": args.muscle_model,
                              "random_count": RANDOM_COUNT},
                "terminations_so_far": int(n_term), "episodes_ended_so_far": int(n_done), "bytes_per_transition_full": per_transition}  # fmt: skip
        path = f"{args.out}_{c:03d}.pt"
        torch.save({"meta": meta, "data": {k: ({kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.cpu())
                                           for k, v in data.items()}}, path)  # fmt: skip
        saved.append(path)
    torch.cuda.synchronize()
    dt = time.time() - t0
    total = args.num_envs * args.steps
    print(f"saved {len(saved)} files {args.out}_*.pt: {args.steps} x {args.num_envs} transitions in {dt:.0f} s ({total / dt:,.0f} per s with packing and "
          f"saving), {int(n_term)} terminations, {int(n_done)} episodes ended; {per_transition} bytes per transition in the full layout, "
          f"{per_transition * total / 1e9:.2f} GB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
