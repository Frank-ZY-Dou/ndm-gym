"""Throughput of the two-G1 boxing scene (two robots per env, robot-robot contacts) in the physics simulator and in the
tracking world models on one GPU, as wmbench/speed_hoi.py measures HDMI's scenes: 20 ms control steps per second times
the number of robots (and per pair: times the number of envs).

Modes:
  physics  the two-robot scene (PairSceneCfg) in Isaac Lab 3 + Newton/MuJoCo-Warp, both robots placed at frame 0 of the
           two-person clip (their reference poses, every velocity zero); each step sets both robots' PD targets, writes
           them and runs one call of the scene's four 5 ms substeps (collision, solver), then updates the state buffers:
           the simulator's best case, without task terms;
  env      PairedTrackingSim.step (rewards, terminations, paired resets): the environment the policies are trained in,
           driven by the reference setpoints plus noise;
  ndm      the tracking world model's step (AccelWorldModel or IncrementWorldModel, built from the training arguments
           --wm_args, e.g. the final SuperTrack configurations' MLP or RigidFormer, or from a training checkpoint
           --wm_ckpt with its weights) fed back on itself for both robots of every env (an open-loop rollout, as a
           simulator), torch.compile (--compile) under bf16 autocast (--amp bf16), as in training; its speed does not
           depend on the weights. It starts from both robots' frame-0 reference states, tiled to N envs. As speed_hoi.py:
           --inference_copy runs the model's inference copy instead (worldmodels/g1_infer.py: bf16 end to end or fp32
           parameters under autocast, the RigidFormer's attention by --rf_attention, --fused step kernels), --rf_kernels
           its blocks or whole network as fused Triton kernels (kernels/rf_block.py's fused_copy) in the model as trained
           or its fp32 inference copy (--inference_copy amp); every such path is compared with the model on the GPU
           first and refused on a mismatch. ``inference_model`` builds the path. --check_steps K also rolls the path and
           the model as trained (eager, bf16 autocast; and in fp32) K steps open loop from the clip's reference states at
           frames spread over it, with the reference's setpoints, and records the largest differences.
--single runs one G1 per env (the one-robot tracking scene, N robots) instead of the pair (2 N robots).
physics drives the PD targets with the default pose plus noise, as speed_hoi.py. Each mode times --steps control steps
after --warmup, synchronizing the GPU around the timed loop, --repeats times; the median is reported. Development numbers
only on a shared GPU.

Run: ``python -m ndm_gym.wmbench.speed_pair --pair <clip>.npz --mode physics --num_envs 1024 --out results/speed_pair.csv``
(``--num_envs``: envs, i.e. pairs; twice as many robots).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time

import torch
from isaaclab.app import add_launcher_args, launch_simulation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=str, required=True, help="a two-person clip (.npz, tools/convert_interhuman.py)")
    p.add_argument("--mode", type=str, required=True, choices=["physics", "env", "ndm"])
    p.add_argument("--num_envs", type=int, default=1024, help="envs (pairs); the robots are twice as many")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3, help="timed runs; the median is reported")
    p.add_argument("--wm_args", type=str, default=None,
                   help="ndm: the training arguments that build the world model (default --algo supertrack); not with --wm_ckpt")
    p.add_argument("--wm_ckpt", type=str, default=None, help="ndm: a training checkpoint whose arguments build the world model and whose weights it loads")
    p.add_argument("--compile", type=str, default="reduce-overhead", choices=["none", "default", "reduce-overhead"], help="ndm: torch.compile mode")
    p.add_argument("--inference_copy", type=str, default="none", choices=["none", "amp", "bf16"],
                   help="ndm: the world model's inference copy (g1_infer.inference_copy), the network bf16 end to end (bf16) or fp32 under autocast (amp)")
    p.add_argument("--rf_attention", type=str, default="token", choices=["token", "sdpa"], help="ndm, inference copy of the RigidFormer: attention")
    p.add_argument("--fused", action="store_true", help="ndm, inference copy: the step outside the network as two Triton kernels")
    p.add_argument("--rf_kernels", type=str, default="none", choices=["none", "blocks", "whole"],
                   help="ndm, RigidFormer: its blocks or whole network as fused Triton kernels, in the model as trained or --inference_copy amp")
    p.add_argument("--check_steps", type=int, default=0, help="ndm: open-loop steps of the accuracy check against the model as trained (0: none)")
    p.add_argument("--single", action="store_true", help="one G1 per env (the one-robot scene) instead of the pair; not with --mode env")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
    p.add_argument("--action_noise", type=float, default=0.2, help="env: noise on the reference setpoints (action units)")
    p.add_argument("--out", type=str, required=True, help="csv to append one row to")
    add_launcher_args(p)
    args = p.parse_args()
    if args.single and args.mode == "env":
        p.error("--single: the env mode is the paired training environment")
    if args.wm_ckpt and args.wm_args is not None:
        p.error("--wm_ckpt builds the world model from the checkpoint's arguments: leave --wm_args out")
    if args.mode != "ndm" and (args.inference_copy != "none" or args.fused or args.rf_kernels != "none" or args.check_steps or args.wm_ckpt):
        p.error("--inference_copy, --fused, --rf_kernels, --check_steps and --wm_ckpt configure --mode ndm")
    return args


def inference_model(wm, args):
    """The model that --mode ndm times and its step, from the G1 world model ``wm`` as trained: ``wm`` itself or its
    inference copy (--inference_copy, --rf_attention, --fused: g1_infer.inference_copy, compared with ``wm`` on the GPU
    first), with the RigidFormer's blocks or whole network as fused kernels (--rf_kernels: ``fused_copy``, each piece
    compared with its reference first; it takes fp32 parameters, so a bf16 copy is refused), the step compiled by
    --compile. Returns the model, its step and the path's description."""
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    rf = isinstance(wm.net, RigidFormerNet)
    if args.rf_kernels != "none" and not rf:
        raise ValueError(f"--rf_kernels {args.rf_kernels} fuses the RigidFormer's blocks or whole network: the MLP backbone has neither")
    if args.inference_copy != "none":
        from ndm_gym.worldmodels.g1_infer import inference_copy

        dtype = torch.bfloat16 if args.inference_copy == "bf16" else None
        wm = inference_copy(wm, dtype, args.rf_attention if rf else None, fused=args.fused)
    elif args.fused:
        raise ValueError("--fused needs --inference_copy")
    if args.rf_kernels != "none":
        from ndm_gym.kernels.rf_block import fused_copy

        wm = fused_copy(wm, whole_network=args.rf_kernels == "whole")
    step = wm.step if args.compile == "none" else torch.compile(wm.step, mode=None if args.compile == "default" else args.compile)
    path = f"compile {args.compile}"
    if args.inference_copy != "none":
        path += f" inference copy {args.inference_copy}" + (f" attention {args.rf_attention}" if rf else "")
        path += " fused" if args.fused else ""
    path += f" rf_kernels {args.rf_kernels}" if args.rf_kernels != "none" else ""
    return wm, step, path


def state_differences(S, S_ref) -> dict:
    """Largest differences of two state batches by component: positions (m), rotations (rad, the angle of R_ref^T R),
    velocities, angular velocities, joint angles, joint rates."""
    from ndm_gym.core.rotations import so3_log

    p, R, v, w, q, qd = S
    p0, R0, v0, w0, q0, qd0 = S_ref
    ang = so3_log(R0.transpose(-1, -2) @ R).norm(dim=-1)
    return {"p": float((p - p0).abs().max()), "R": float(ang.max()), "v": float((v - v0).abs().max()), "w": float((w - w0).abs().max()),
            "q": float((q - q0).abs().max()), "qd": float((qd - qd0).abs().max())}  # fmt: skip


def check_rollouts(base, step, S, actions, xi_n, graphs: bool) -> str:
    """The path's ``step`` and the model as trained (``base``: eager under bf16 autocast, and in fp32 with TF32 off) rolled
    open loop from the states S with the setpoints ``actions`` (K, n, J): per component, the path's largest difference
    from the model as trained over the K steps, the model's own bf16-vs-fp32 difference, and their largest ratio."""
    import ndm_gym.core.nets as nets

    def roll(fn, marks: bool):
        out, s = [], S
        for a in actions:
            if marks:
                torch.compiler.cudagraph_mark_step_begin()
            s = tuple(t.clone() for t in fn(s, a, xi_n))
            out.append(s)
        return out

    with torch.no_grad():
        path = roll(step, graphs)
        ref16 = roll(base.step, False)
        amp, tf32 = nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32
        nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32 = None, False
        try:
            ref32 = roll(base.step, False)
        finally:
            nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32 = amp, tf32
    keys = ("p", "R", "v", "w", "q", "qd")
    dp, d16 = {k: 0.0 for k in keys}, {k: 0.0 for k in keys}
    for a, b, c in zip(path, ref16, ref32):
        da, db = state_differences(a, b), state_differences(b, c)
        dp, d16 = {k: max(dp[k], da[k]) for k in keys}, {k: max(d16[k], db[k]) for k in keys}
    ratio = max(dp[k] / max(d16[k], 1e-9) for k in keys)
    fmt = lambda d: " ".join(f"{k} {d[k]:.2e}" for k in keys)  # noqa: E731
    return f"check {len(actions)} steps vs as trained: {fmt(dp)}; as trained bf16 vs fp32: {fmt(d16)}; largest ratio {ratio:.2f}"


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

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.core.rotations import mat_to_quat
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, PairSceneCfg, TrackingSceneCfg, make_sim, motion_pairs
    from ndm_gym.state import normalize_xi
    from ndm_gym.tasks.tracking import MDP

    torch.backends.cuda.matmul.allow_tf32 = True
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev, N = "cuda:0", args.num_envs
    per_env = 1 if args.single else 2  # robots per env
    torch.manual_seed(0)
    kin = g1.kinematics(dev)
    motion = MotionLib(interhuman.load(args.pair), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    pc = motion_pairs(motion)[:, 0]  # the file's first pair
    f0 = motion.clip_start[pc.to(dev)]  # both sides' first frames
    sim_cfg = make_sim(dev, args, paired=args.mode != "physics" or not args.single)
    with launch_simulation(sim_cfg, args):
        if args.mode == "physics":
            from isaaclab.sim import SimulationContext
            from isaaclab.utils import instantiate

            sim = SimulationContext(sim_cfg)
            scene_cfg = TrackingSceneCfg if args.single else PairSceneCfg
            scene = instantiate(scene_cfg(num_envs=N, env_spacing=0.0, replicate_physics=True))
            sim.reset()
            robots = (scene["robot"],) if args.single else (scene["robot"], scene["robot_b"])
            ids = torch.arange(N, device=dev)
            for r, f in zip(robots, f0.tolist()):
                p, R, _, _, q, _ = motion.ref_state(torch.full((N,), f, device=dev))
                quat = mat_to_quat(R)
                r.write_root_link_pose_to_sim_index(root_pose=torch.cat((p + scene.env_origins, quat[:, 1:4], quat[:, :1]), -1), env_ids=ids)
                r.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev), env_ids=ids)
                r.write_joint_position_to_sim_index(position=q.clone(), env_ids=ids)
                r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
            scene.reset(ids)
            target0 = [r.data.default_joint_pos.torch.clone() for r in robots]

            def fn():
                for r, t in zip(robots, target0):
                    r.set_joint_position_target_index(target=t + 0.05 * torch.randn_like(t))
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim_cfg.dt)

            extra = "clip frame 0, default pose targets + noise"
        elif args.mode == "env":
            env = PairedTrackingSim(sim_cfg, 2 * N, motion, mdp)
            ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)

            def fn():
                a = ref_u[env.f + 1]
                env.step(a + args.action_noise * torch.randn_like(a), env.state())

            extra = "reference setpoints + noise, rewards, terminations, paired resets"
        else:
            from ndm_gym.train import add_speed_args, add_train_args

            if args.wm_ckpt:  # the run's arguments and its world model's weights
                ck = torch.load(args.wm_ckpt, map_location=dev, weights_only=False)
                ta = argparse.Namespace(**ck["args"])
                wm = ALGOS[ta.algo].world_model_cls(ck["args"]).to(dev)
                wm.load_state_dict(ck["wm"])
                source = f"ckpt {args.wm_ckpt} (iteration {ck.get('iteration')})"
            else:
                tp = argparse.ArgumentParser()
                add_train_args(tp)
                add_speed_args(tp)
                wm_args = args.wm_args if args.wm_args is not None else "--algo supertrack"
                ta = tp.parse_args(shlex.split(wm_args))
                wm = ALGOS[ta.algo].world_model_cls(vars(ta)).to(dev)
                source = f"wm_args {wm_args}"
            wm.requires_grad_(False)
            n_par = sum(t.numel() for t in wm.net.parameters())
            env = PairedTrackingSim(sim_cfg, 2 * min(N, 512), motion, mdp, eval_mode=True)  # the reference states and contexts
            env.reset(torch.arange(env.N, device=dev), f0.repeat_interleave(env.P))
            env.refresh()
            idx = torch.arange(per_env * N, device=dev) % env.N
            S = tuple(t[idx].contiguous() for t in env.state())
            xi_n = normalize_xi(env.xi)[idx].contiguous()
            ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
            a = ref_u[env.f[idx] + 1].contiguous()
            model, step, path = inference_model(wm, args)
            graphs = args.compile == "reduce-overhead"
            state = {"S": S}

            @torch.no_grad()
            def fn():
                if graphs:  # CUDA graphs reuse their output buffers: mark the step and keep copies of the state
                    torch.compiler.cudagraph_mark_step_begin()
                S1 = step(state["S"], a, xi_n)
                state["S"] = tuple(t.clone() for t in S1) if graphs else S1

            extra = f"{ta.wm_backbone} ({n_par / 1e6:.2f} M parameters) {path} | {source}"
            if args.check_steps:  # the clip's reference states at frames spread over it (both sides at the same offset)
                K, P = args.check_steps, env.P
                L = int(motion.clip_len[pc[0]])
                if L <= K + 2:
                    raise ValueError(f"--check_steps {K}: the clip has {L} frames")
                off = torch.linspace(0, L - K - 2, P, device=dev).long()
                frames = torch.cat((f0[0] + off, f0[1] + off))
                env.reset(torch.arange(env.N, device=dev), frames)
                env.refresh()
                S_chk = tuple(t.contiguous() for t in env.state())
                acts = [ref_u[frames + 1 + k].contiguous() for k in range(K)]
                extra += " | " + check_rollouts(wm, step, S_chk, acts, normalize_xi(env.xi).contiguous(), graphs)
        torch.cuda.reset_peak_memory_stats()
        runs = sorted(timed(fn, args.warmup if r == 0 else 0, args.steps) for r in range(args.repeats))
        sec = runs[len(runs) // 2]
        extra += " | runs " + " ".join(f"{r:.4f}" for r in runs)
        robots_rate = per_env * N * args.steps / sec
        mem = torch.cuda.max_memory_allocated() / 2**30
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", uuid], capture_output=True, text=True,
                                check=True).stdout.strip()  # fmt: skip
        # --single: mode "<mode>_single", and the per-env column holds env-steps, one robot each
        mode = args.mode + ("_single" if args.single else "")
        row = (f"{os.path.basename(args.pair)},{mode},{N},{per_env * N},{args.steps},{sec:.4f},{robots_rate:.1f},{robots_rate / per_env:.1f},"
               f"{1e3 * sec / args.steps:.3f},{args.amp},{torch.cuda.get_device_name()},{driver},{mem:.2f},{extra}")  # fmt: skip
        new = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            if new:
                f.write("pair,mode,envs,robots,steps,seconds,robot_steps_per_s,pair_steps_per_s,ms_per_step,amp,gpu,driver,peak_mem_gb,detail\n")
            f.write(row + "\n")
        print(f"[speed_pair] {mode} {N} envs: {robots_rate:,.0f} robot-steps/s ({robots_rate / per_env:,.0f} env-steps/s), "
              f"{1e3 * sec / args.steps:.2f} ms per step ({extra})", flush=True)


if __name__ == "__main__":
    main()
