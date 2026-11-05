"""Throughput of one of HDMI's scenes in the physics simulator and in a neural dynamics model on one GPU: 20 ms control
steps per second times the number of environments.

Modes:
  physics  the scene in Isaac Lab 3 + Newton/MuJoCo-Warp, placed at frame 0 of the task's clip (the robot, the object
           and the static props at their reference poses, without reset noise, every velocity zero; hdmi_sim.place_scene);
           each step sets the PD targets, writes them and runs one call of HDMI's four 5 ms substeps (collision, solver),
           then updates the robot's and the object's state buffers: the simulator's best case, without HDMI's per-substep
           action delay and filter, contact sensors or task terms;
  env      HDMISim.step (the four substeps stepped one by one with HDMI's delayed and filtered targets, contact sensors),
           with HDMITask.step under --with_task: the environment the policies are trained in;
  ndm      HOIWorldModel.step fed back on itself (an open-loop rollout, as a simulator), compiled with --compile and
           under bf16 autocast with --amp bf16, as in training; its speed does not depend on the weights. It starts from
           the reset states and contexts of at most 1024 simulated environments, tiled to N. --inference_copy runs the
           model's inference copy instead (HOIWorldModel.inference_copy: bf16 end to end or fp32 parameters under
           autocast, the RigidFormer's attention by --rf_attention, --fused step kernels). --rf_kernels runs the
           RigidFormer's blocks or its whole network as fused Triton kernels (kernels/rf_block.py's fused_copy, each
           compared with its reference on the GPU first) in the model as trained or in its fp32 inference copy
           (--inference_copy amp); a bf16 copy and the MLP are refused, and so is the whole network on compute
           capability 9.x (an H100) unless NDM_RF_WHOLE_ON_SM90=1. ``inference_model`` builds this path;
           wmbench/score_hoi.py scores a fitted model through the same paths.
physics drives the PD targets with the default pose plus noise from the clip's frame 0; env with the reference's joints
plus noise (the data policy of collect_hoi.py) from HDMI's reset states, without resets. Each mode times --steps
control steps after --warmup, synchronizing the GPU around the timed loop. Development numbers only on a shared GPU.

Run: ``python -m ndm_gym.wmbench.speed_hoi --task move_suitcase --mode physics --num_envs 4096 --out results/speed_hoi.csv``
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from ndm_gym.sim.launcher import add_launcher_args, launch_simulation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["physics", "env", "ndm"])
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3, help="timed runs; the median is reported")
    p.add_argument("--with_task", action="store_true", help="env: also HDMITask.step (observations, rewards, terminations)")
    p.add_argument("--kind", type=str, default="accel", choices=["accel", "increment"], help="ndm: parameterization")
    p.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "rigidformer"], help="ndm: network")
    p.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048], help="ndm: MLP widths")
    p.add_argument("--rf_dim", type=int, default=128, help="ndm: RigidFormer token width")
    p.add_argument("--rf_layers", type=int, default=3)
    p.add_argument("--rf_heads", type=int, default=4)
    p.add_argument("--rf_registers", type=int, default=2)
    p.add_argument("--compile", type=str, default="default", choices=["none", "default", "reduce-overhead"], help="ndm: torch.compile mode")
    p.add_argument("--inference_copy", type=str, default="none", choices=["none", "amp", "bf16"],
                   help="ndm: HOIWorldModel.inference_copy with the network in bf16 end to end (bf16) or in fp32 under autocast (amp)")
    p.add_argument("--rf_attention", type=str, default="token", choices=["token", "sdpa"], help="ndm, inference copy of the RigidFormer: attention")
    p.add_argument("--fused", action="store_true", help="ndm, inference copy: the step outside the network as two Triton kernels")
    p.add_argument("--rf_kernels", type=str, default="none", choices=["none", "blocks", "whole"],
                   help="ndm, RigidFormer: its blocks or whole network as fused Triton kernels, in the model as trained or --inference_copy amp")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
    p.add_argument("--action_noise", type=float, default=0.2)
    p.add_argument("--out", type=str, required=True, help="csv to append one row to")
    add_launcher_args(p)
    return p.parse_args()


def inference_model(wm, args):
    """The model that --mode ndm times and its step, from the HOIWorldModel ``wm``: the model itself or its inference
    copy (--inference_copy, --rf_attention, --fused), with the RigidFormer's blocks or whole network as fused kernels
    (--rf_kernels: ``fused_copy``, which takes a model with fp32 parameters and refuses a bf16 copy), the step compiled
    by --compile. Returns the model, its step and the path's description (the detail column after the model's size)."""
    from ndm_gym.worldmodels.hoi import HOIRigidFormerNet

    rf = isinstance(wm.net, HOIRigidFormerNet)
    if args.rf_kernels != "none" and not rf:
        raise ValueError(f"--rf_kernels {args.rf_kernels} fuses the RigidFormer's blocks or whole network: the MLP backbone has neither")
    if args.inference_copy != "none":
        dtype = torch.bfloat16 if args.inference_copy == "bf16" else None
        wm = wm.inference_copy(dtype, args.rf_attention if rf else None, fused=args.fused)
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
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim, check_prop_contacts, exclude_prop_contacts, load_clip, make_hdmi_sim, place_scene, scene_cfg
    from ndm_gym.tasks.hdmi import NUM_ACTIONS, HDMITask
    from ndm_gym.tasks.hdmi_tasks import TASKS

    torch.backends.cuda.matmul.allow_tf32 = True
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev, N, cfg = "cuda:0", args.num_envs, TASKS[args.task]
    torch.manual_seed(0)
    extra = ""
    with launch_simulation(make_hdmi_sim(dev, substeps=4 if args.mode == "physics" else 1), args):
        # env builds HDMISim with N environments; ndm with at most 1024, whose reset states and contexts are tiled to N
        n_sim = N if args.mode == "env" else min(N, 1024)
        env = HDMISim(make_hdmi_sim(dev), cfg, n_sim) if args.mode != "physics" else None
        if args.mode == "env":
            task = HDMITask(cfg, N, env.joint_limits, env.robot_mass, noise=True, device=dev) if args.with_task else None
            q0 = env.q0[env.j_act]
            state = {"prev": env.read()}
            if task is not None:
                task.reset(torch.arange(N, device=dev), state["prev"])

            def fn():
                a = (env.clip.joint_pos[env.t.clamp(max=env.L - 1)] - q0) / env.scale
                x, _ = env.step(a + args.action_noise * torch.randn_like(a), state["prev"])
                if task is not None:
                    task.step(x)
                state["prev"] = x.now

            extra = "with_task" if args.with_task else "no_task"
        elif args.mode == "physics":
            from isaaclab.sim import SimulationContext
            from isaaclab.utils import instantiate

            sim_cfg = make_hdmi_sim(dev, substeps=4)
            sim = SimulationContext(sim_cfg)
            _, side = load_clip(cfg.name)
            scene = instantiate(scene_cfg(cfg, side, N))
            exclude_prop_contacts(sim, scene, cfg)
            sim.reset()
            check_prop_contacts(sim, scene, cfg)
            place_scene(scene, cfg)
            robot = scene["robot"]
            target0 = robot.data.default_joint_pos.torch.clone()

            def fn():
                robot.set_joint_position_target_index(target=target0 + 0.05 * torch.randn_like(target0))
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim_cfg.dt)

            extra = "clip frame 0, default pose targets + noise"
        else:
            from ndm_gym.worldmodels.hoi import ContextLayout, HOIWorldModel, rigidformer_args

            rf = None if args.backbone == "mlp" else rigidformer_args(env.ndm_constants(), args.rf_dim, args.rf_layers, args.rf_heads, args.rf_registers)
            wm = HOIWorldModel(args.kind, env.hinged, cfg.num_eefs, ContextLayout(*env.context_layout), hidden=args.hidden, rigidformer=rf).to(dev)
            n_par = sum(t.numel() for t in wm.net.parameters())
            wm, step, path = inference_model(wm, args)
            idx = torch.arange(N, device=dev) % n_sim
            S, So, ctx = tuple(t[idx] for t in env.robot_state()), tuple(t[idx] for t in env.object_state()), env.context()[idx]
            u = torch.zeros(N, 4, NUM_ACTIONS, device=dev)
            state = {"S": S, "So": So}

            graphs = args.compile == "reduce-overhead"

            @torch.no_grad()
            def fn():
                if graphs:  # CUDA graphs reuse their output buffers: mark the step and keep copies of the state
                    torch.compiler.cudagraph_mark_step_begin()
                S1, So1, _ = step(state["S"], state["So"], u, ctx)
                state["S"], state["So"] = (tuple(t.clone() for t in S1), tuple(t.clone() for t in So1)) if graphs else (S1, So1)

            size = f"mlp {'x'.join(map(str, args.hidden))}" if rf is None else f"rigidformer d{args.rf_dim} l{args.rf_layers} h{args.rf_heads}"
            extra = f"{args.kind} {size} ({n_par / 1e6:.2f} M parameters) {path}"
        torch.cuda.reset_peak_memory_stats()
        runs = sorted(timed(fn, args.warmup if r == 0 else 0, args.steps) for r in range(args.repeats))
        sec = runs[len(runs) // 2]
        extra += " | runs " + " ".join(f"{r:.4f}" for r in runs)
        rate = N * args.steps / sec
        mem = torch.cuda.max_memory_allocated() / 2**30
        row = (f"{args.task},{args.mode},{N},{args.steps},{sec:.4f},{rate:.1f},{1e3 * sec / args.steps:.3f},{args.amp},"
               f"{torch.cuda.get_device_name()},{mem:.2f},{extra}")  # fmt: skip
        new = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            if new:
                f.write("task,mode,num_envs,steps,seconds,env_steps_per_s,ms_per_step,amp,gpu,peak_mem_gb,detail\n")
            f.write(row + "\n")
        print(f"[speed_hoi] {args.task} {args.mode} N={N}: {rate:,.0f} env-steps/s, {1e3 * sec / args.steps:.2f} ms per step ({extra})", flush=True)


if __name__ == "__main__":
    main()
