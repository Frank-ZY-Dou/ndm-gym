"""Collect transitions of one of HDMI's human-object interaction tasks (``ndm_gym.sim.rigid.hdmi_sim``) for the neural
dynamics models, in replay-buffer layout (``HDMISim.transition_fields``), under HDMI's training conditions: its reset
noise and startup randomization, delayed and filtered actions, terminations and resets.

Actions: the reference's tracking joints as targets, HDMI's residual base ``(q_ref - q0) / scale`` at the current
frame, plus Gaussian noise of std ``--action_noise``. Without feedback this tracks open loop until the robot falls; a
trained policy replaces it once one exists. A collection with a non-finite value anywhere is not saved.

Run: ``python -m ndm_gym.wmbench.collect_hoi --task move_suitcase --num_envs 4096 --steps 400 --out data_wm/hoi/<name>.pt``
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from isaaclab.app import add_launcher_args, launch_simulation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="one of ndm_gym.tasks.hdmi_tasks.TASKS")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=400, help="control steps per environment")
    parser.add_argument("--action_noise", type=float, default=0.2, help="std of the Gaussian noise on the actions")
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    return parser.parse_args()


def merge_states(mask, a, b):
    """HDMIState with ``b``'s fields where ``mask`` (N,) is true, ``a``'s elsewhere."""
    def pick(x, y):
        if x is None:
            return None
        m = mask.view(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, y, x)

    return type(a)(*(pick(x, y) for x, y in zip(a, b)))


def main():
    args = parse_args()

    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim, make_hdmi_sim
    from ndm_gym.tasks.hdmi import HDMITask
    from ndm_gym.tasks.hdmi_tasks import TASKS

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    cfg = TASKS[args.task]
    sim_cfg = make_hdmi_sim(dev)
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = HDMISim(sim_cfg, cfg, args.num_envs)
        task = HDMITask(cfg, args.num_envs, env.joint_limits, env.robot_mass, noise=True, device=dev)
        buf = ReplayBuffer(args.steps, args.num_envs, dev, fields=env.transition_fields())
        q0 = env.q0[env.j_act]  # HDMI order, the 23 action joints
        prev = env.read()
        task.reset(torch.arange(args.num_envs, device=dev), prev)
        n_term = n_done = torch.zeros((), dtype=torch.long, device=dev)  # counted on the device: no synchronization per step
        t0 = time.time()
        for _ in range(args.steps):
            before = env.before_step()
            a = (env.clip.joint_pos[env.t] - q0) / env.scale
            a = a + args.action_noise * torch.randn_like(a)
            x, truncated = env.step(a, prev)
            out = task.step(x)
            buf.add(env.transition(before, a, x, out.terminated, truncated))
            done = out.terminated | truncated
            n_term, n_done = n_term + out.terminated.sum(), n_done + done.sum()
            prev = x.now
            if done.any():
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                env.reset(ids)
                env.refresh()
                prev = merge_states(done, prev, env.read())
                task.reset(ids, prev)
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_term, n_done = int(n_term), int(n_done)
        # a non-finite simulator state (docs/hoi_design.md §1) would reach the models' fits: nothing is saved then
        nonfinite = {k: int((~torch.isfinite(v)).sum()) for k, v in buf.data.items() if v.is_floating_point()}
        if any(nonfinite.values()):
            raise RuntimeError(f"non-finite values in the collected transitions, {args.out} not written: "
                               + ", ".join(f"{k} {n}" for k, n in nonfinite.items() if n))  # fmt: skip
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        meta = {"task": args.task, "num_envs": args.num_envs, "steps": args.steps, "action_noise": args.action_noise, "seed": args.seed,
                "terminations": n_term, "episodes_ended": n_done, "context_dim": buf.data["ctx"].shape[-1], "context_layout": env.context_layout,
                "ndm_constants": env.ndm_constants(),  # the simulator's check of hdmi_sim.ndm_constants(cfg)
                "hinged": env.hinged,
                "seconds": dt, "policy": "reference + noise"}  # fmt: skip
        torch.save({"meta": meta, "data": {k: v.cpu() for k, v in buf.data.items()}}, args.out)
        print(f"saved {args.out}: {args.steps} x {args.num_envs} transitions in {dt:.0f} s, {n_term} terminations, {n_done} episodes ended",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
