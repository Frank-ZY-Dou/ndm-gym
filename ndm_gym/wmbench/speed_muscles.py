"""Throughput of MuscleVAE's muscle-actuated character in the physics simulator on one GPU: 50 ms control steps per
second times the number of environments.

Modes:
  physics  MuscleSim.step (ndm_gym.sim.rigid.muscle_sim): per control step six substeps, each MuscleVAE's muscle model
           (torch, compiled) and a step of Newton's MuJoCo-Warp solver, then MuscleVAE's done flags and reference
           frames, all one CUDA graph. Actions (--actions) from the data policy (the reference's muscle lengths plus noise
           of std --action_noise), or MuscleVAE's pretrained tracking policy (ndm_gym.policies.musclevae_pretrained,
           posterior mean), whose observation, networks and action are in the step's graph; the environments that
           terminate or time out reset (one more graph) unless --no_resets.
  collect  physics, and every transition written into the rows of contiguous device buffers (MuscleSim.transition_buffers,
           --record_rows rows in turn): the simulator producing NDM data.
  solver   the six solver steps alone, without muscle forces: the rigid-body share of a physics step (a diagnostic).
  muscles  the six muscle substeps alone, on the poses of the moment: the muscle model's share (a diagnostic).
Each mode times --steps control steps after --warmup, synchronizing the GPU around the timed loop; --repeats runs, the
median reported. --ball_joints and --joint_damping choose MuscleSim's joints (their simulator: euler and ode);
--solver_iterations caps the solver's iterations (MuJoCo's 100 by default; fewer is faster and changes some solves).
Development numbers only on a shared GPU.

Run: ``python -m ndm_gym.wmbench.speed_muscles --mode physics --num_envs 4096 --out results/speed_muscles.csv``
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["physics", "collect", "solver", "muscles"])
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3, help="timed runs; the median is reported")
    p.add_argument("--action_noise", type=float, default=0.05, help="std of the data policy's Gaussian noise")
    p.add_argument("--actions", type=str, default="data", choices=["data", "pretrained"], help="physics and collect: the data policy or theirs")
    p.add_argument("--ball_joints", type=str, default="ball", choices=["ball", "euler"])
    p.add_argument("--joint_damping", type=str, default="implicit", choices=["implicit", "ode"])
    p.add_argument("--solver_iterations", type=int, default=None, help="the solver's iteration limit (MuJoCo's default, 100, if not given)")
    p.add_argument("--no_resets", action="store_true", help="physics: keep stepping terminated environments")
    p.add_argument("--record_rows", type=int, default=16, help="collect: rows of the transition buffers, written in turn")
    p.add_argument("--muscle_model", type=str, default="released", choices=["released", "paper"])
    p.add_argument("--no_compile", action="store_true", help="the muscle substep and the done flags without torch.compile")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="csv to append one row to")
    return p.parse_args()


def timed(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    args = parse_args()

    import warp as wp

    from ndm_gym.robots import musclevae_char as mc
    from ndm_gym.sim.rigid.muscle_sim import SIM_DT, MuscleSim

    dev, N = "cuda:0", args.num_envs
    torch.manual_seed(args.seed)
    sim = MuscleSim(N, dev, muscle_model=args.muscle_model, compiled=not args.no_compile, seed=args.seed, ball_joints=args.ball_joints,
                    joint_damping=args.joint_damping, solver_iterations=args.solver_iterations, capture_step=args.actions == "data")  # fmt: skip
    runner = None
    if args.actions == "pretrained":
        if args.mode not in ("physics", "collect"):
            raise ValueError("--actions pretrained times the physics and collect modes")
        from ndm_gym.policies.musclevae_pretrained import MuscleVAEPolicy, PolicyRunner

        runner = PolicyRunner(sim, MuscleVAEPolicy.load(device=dev), compiled=not args.no_compile)
    if args.mode in ("physics", "collect"):
        resets = not args.no_resets
        buf = sim.transition_buffers(args.record_rows) if args.mode == "collect" else None
        rows = [{k: v[t] for k, v in buf.items()} for t in range(args.record_rows)] if buf is not None else None
        tick = [0]

        def fn():
            row = None
            if rows is not None:
                row = rows[tick[0] % len(rows)]
                tick[0] += 1
                sim.before_step(out=row)
            if runner is None:
                a = sim.data_policy(args.action_noise)
                terminated, truncated = sim.step(a)
            else:
                terminated, truncated = runner.step()
                a = sim.action
            if row is not None:
                sim.transition(row, a, terminated, truncated, out=row)
            if resets:
                (sim if runner is None else runner).reset(terminated | truncated)

        extra = (f"data policy noise {args.action_noise}" if runner is None else "pretrained policy (posterior mean)") + f"; {'resets' if resets else 'no resets'}"
        if buf is not None:
            per_row = sum(v[0].numel() * v.element_size() for v in buf.values())
            extra += f"; transitions written into {args.record_rows} rows of device buffers ({per_row / N / 1024:.1f} KB per transition)"
    elif args.mode == "solver":
        for v in sim.views:
            v.body_f.zero_()

        def solver_steps():
            for i in range(mc.SUBSTEPS):
                sim.solver.step(sim.states[i % 2], sim.states[1 - i % 2], None, None, SIM_DT)

        fn = sim.capture(solver_steps).replay
        extra = "six solver steps without muscle forces"
    else:
        sim.length_target.copy_(sim.muscles.target_length(sim.reference_action()))

        def muscle_substeps():
            for _ in range(mc.SUBSTEPS):
                sim.muscle_substep(sim.views[0].body_q, sim.views[0].body_f)

        fn = sim.capture(muscle_substeps).replay
        extra = "six muscle substeps"
    extra += f"; muscle model {args.muscle_model}{'' if args.no_compile else ' compiled'}; {args.ball_joints} ball joints, {args.joint_damping} damping"
    if args.solver_iterations is not None:
        extra += f"; solver iterations {args.solver_iterations}"
    torch.cuda.reset_peak_memory_stats()
    runs = sorted(timed(fn, args.warmup if r == 0 else 0, args.steps) for r in range(args.repeats))
    sec = runs[len(runs) // 2]
    checks = sim.check_solver()  # raises if a contact buffer overflowed
    extra += f"; worlds at the solver's iteration limit {checks['iterations']} | runs " + " ".join(f"{r:.4f}" for r in runs)
    rate = N * args.steps / sec
    mem_torch, mem_warp = torch.cuda.max_memory_allocated() / 2**30, wp.get_mempool_used_mem_high(dev) / 2**30
    row = (f"musclevae,{args.mode},{N},{args.steps},{sec:.4f},{rate:.1f},{1e3 * sec / args.steps:.3f},none,{torch.cuda.get_device_name()},"
           f"{mem_torch + mem_warp:.2f},{extra} (memory: torch {mem_torch:.2f} GB + warp {mem_warp:.2f} GB)")  # fmt: skip
    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        if new:
            f.write("task,mode,num_envs,steps,seconds,env_steps_per_s,ms_per_step,amp,gpu,peak_mem_gb,detail\n")
        f.write(row + "\n")
    print(f"[speed_muscles] {args.mode} N={N}: {rate:,.0f} env-steps/s, {1e3 * sec / args.steps:.2f} ms per step ({extra})", flush=True)


if __name__ == "__main__":
    main()
