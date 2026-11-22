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
--mode ndm --grad_horizon H times, instead of the step, what a first-order learner pays through the G1 world models: one
call is an H-step rollout of the model as trained from the start states with the reference's setpoints, its loss
1/N sum_robots sum_t [0.5 |q(t) - q_default|^2 + 0.5 * 0.01 (|v(t)|^2 + |w(t)|^2 + |qd(t)|^2)] and the reverse-mode
gradient of that loss with respect to the H setpoint vectors of every robot (``rollout_gradient``; the measure and the
loss of the differentiable simulators' benchmark, benchmarks/ndm_speed/diffsim/common.py). --compile none runs it
eagerly, default with the step compiled, reduce-overhead (the default) with the step compiled and the whole call replayed
as one CUDA graph, as training's --update_graphs does. The row's steps are the calls' control steps (--steps calls of H)
and its ms per step the forward and backward time per control step.
physics drives the PD targets with the default pose plus noise, as speed_hoi.py. Each mode times --steps control steps
after --warmup, synchronizing the GPU around the timed loop, --repeats times; the median is reported. Development numbers
only on a shared GPU.
--replay (--single, physics or ndm; the workload of benchmarks/steel/results.md) replaces the start states and setpoints by
open-loop windows of a recording (wmbench/collect.py): --replay_windows windows of --replay_len steps drawn once with
--replay_seed as wmbench/fit.py draws its scoring windows (``fixed_windows``). The timed steps run as segments of
--replay_len steps (the last one shorter); before each segment every robot restarts from a window's first recorded state
(physics: written into the simulator as a reset writes it; ndm: the world model's state) and then takes that window's
recorded setpoints, the PD targets default pose + recorded joint offsets + action scale x setpoint. Only the segments are
timed, never the restarts between them.

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

from ndm_gym.sim.launcher import add_launcher_args, launch_simulation
from ndm_gym.wmbench.warm import keep_busy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=str, required=True, help="a two-person clip (.npz, tools/convert_interhuman.py)")
    p.add_argument("--mode", type=str, required=True, choices=["physics", "env", "ndm"])
    p.add_argument("--num_envs", type=int, default=1024, help="envs (pairs); the robots are twice as many")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--warm_seconds", type=float, default=0.0,
                   help="after --warmup, keep calling this many seconds before the first timed run (wmbench/warm.py: an idle GPU's "
                   "clock)")
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
    p.add_argument("--grad_horizon", type=int, default=0,
                   help="ndm, the G1 world models as trained: time rollouts of this many steps with the reverse-mode gradient of their loss "
                   "with respect to every step's setpoints instead of the step (0: the step); --steps and --warmup then count rollouts")
    p.add_argument("--single", action="store_true", help="one G1 per env (the one-robot scene) instead of the pair; not with --mode env")
    p.add_argument("--replay", type=str, default=None, help="--single, physics or ndm: a recording (wmbench/collect.py) whose windows are the "
                   "workload (the module's docstring)")
    p.add_argument("--replay_len", type=int, default=32, help="--replay: steps per window and timed segment")
    p.add_argument("--replay_windows", type=int, default=8192,
                   help="--replay: windows drawn; robot i takes window (i + k N) mod this in segment k")
    p.add_argument("--replay_seed", type=int, default=0, help="--replay: the windows' draw")
    p.add_argument("--barrier", type=str, default=None, help="--replay: a folder where --barrier_n processes meet after their warm-up, "
                   "to start their first timed run together (the multi-GPU rows of benchmarks/steel)")
    p.add_argument("--barrier_n", type=int, default=0, help="--barrier: the processes that meet")
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
    if args.grad_horizon < 0:
        p.error("--grad_horizon: a number of steps (0: the step is timed)")
    if args.grad_horizon and (args.mode != "ndm" or args.inference_copy != "none" or args.fused or args.rf_kernels != "none" or args.check_steps):
        p.error("--grad_horizon differentiates the G1 world models as trained (--mode ndm): not with --inference_copy, --fused, --rf_kernels or "
                "--check_steps")
    if args.replay is not None:
        if not args.single or args.mode not in ("physics", "ndm"):
            p.error("--replay: one G1 per env (--single) in --mode physics or ndm")
        if args.grad_horizon or args.check_steps:
            p.error("--replay times the step: not with --grad_horizon or --check_steps")
        if args.replay_len < 1 or args.replay_windows < 1:
            p.error("--replay_len and --replay_windows: positive")
    if (args.barrier is None) != (args.barrier_n == 0) or (args.barrier is not None and (args.replay is None or args.barrier_n < 1)):
        p.error("--barrier and --barrier_n go together, with --replay")
    return args


def build_world_model(args, dev: str):
    """The world model that --mode ndm times, on ``dev``: from a training checkpoint's arguments and weights (--wm_ckpt),
    or untrained from training arguments (--wm_args). Returns the model, its arguments and the source's description."""
    from ndm_gym.algos import ALGOS
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
    if getattr(ta, "elementwise_3x3", False):  # the run's compiled functions as its training compiled them
        from ndm_gym.train import elementwise_3x3_products

        elementwise_3x3_products()
        source += " | elementwise_3x3"
    return wm, ta, source


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


W_V = 0.01  # the velocity term's weight in the rollout loss, as the differentiable simulators' benchmark has it


def rollout_gradient(step, S, actions, xi_n, q_ref):
    """An open-loop rollout of ``step`` from the states S with the setpoints ``actions`` (H, n, J; a leaf that requires
    gradients) and the reverse-mode gradient of its loss with respect to them. The loss is the differentiable simulators'
    (benchmarks/ndm_speed/diffsim/common.py): 1/n sum_robots sum_t [0.5 |q(t) - q_ref|^2 + 0.5 W_V |velocities(t)|^2]
    over the H states after the start, the velocities being the root's linear and angular velocity and the joint rates.
    Returns the loss and the gradient (H, n, J)."""
    loss = actions.new_zeros(())
    for a in actions:
        S = step(S, a, xi_n)
        _, _, v, w, q, qd = S
        speed2 = (v * v).sum(-1) + (w * w).sum(-1) + (qd * qd).sum(-1)
        loss = loss + (0.5 * ((q - q_ref) ** 2).sum(-1) + 0.5 * W_V * speed2).mean()
    (grad,) = torch.autograd.grad(loss, actions)
    return loss.detach(), grad


def segment_lengths(steps: int, length: int) -> list[int]:
    """--replay: the timed steps as segments of ``length`` steps, the last one shorter."""
    return [length] * (steps // length) + ([steps % length] if steps % length else [])


def replay_schedule(n_windows: int, robots: int, segments: int, device=None) -> torch.Tensor:
    """--replay: (segments, robots) window indices, robot i taking window (i + k robots) mod n_windows in segment k."""
    i = torch.arange(robots, device=device)
    return torch.stack([(i + k * robots) % n_windows for k in range(segments)])


def timed_segments(fn, restart, lengths: list[int], warmup: int = 0, warm_seconds: float = 0.0, sync=None, wait=None) -> float:
    """--replay's timing: ``restart(k)`` sets segment k's start (untimed); ``fn(j)`` is the segment's step j. The warm-up
    runs from segment 0's start, then each segment restarts and its steps are timed between two synchronizations; returns
    the segments' time summed. ``wait``: called after the warm-up, before the first segment (--barrier)."""
    sync = torch.cuda.synchronize if sync is None else sync
    restart(0)
    for j in range(warmup):
        fn(j % lengths[0])
    sync()
    keep_busy(lambda: fn(0), sync, warm_seconds)
    if wait is not None:
        wait()
    total = 0.0
    for k, n in enumerate(lengths):
        restart(k)
        sync()
        t0 = time.perf_counter()
        for j in range(n):
            fn(j)
        sync()
        total += time.perf_counter() - t0
    return total


def replay_windows(path: str, n: int, length: int, seed: int, dev: str, full: bool = False) -> dict:
    """--replay: ``n`` continuous windows of ``length`` steps of the recording at ``path`` (wmbench/fit.py's fixed_windows,
    seed ``seed``, drawn on the CPU so that every tool and device gets the same windows; a ``.npz`` is replay_engine export's
    file of windows drawn so, read instead of drawn), on ``dev``: the first states (n,
    STATE_DIM), the setpoints (n, length, NUM_J) and the context (n, XI_DIM); ``full``: also every recorded state after each
    step (n, length, STATE_DIM), the frames (n, length) and the window's first row and env."""
    if path.endswith(".npz"):  # replay_engine export's windows, drawn by this function with their seed
        import numpy as np

        e = np.load(path, allow_pickle=False)
        if int(e["seed"]) != seed or len(e["s0"]) < n or e["a"].shape[1] < length:
            raise ValueError(f"{path}: {len(e['s0'])} windows of {e['a'].shape[1]} steps with seed {int(e['seed'])}, not {n} of {length} "
                             f"with seed {seed}")
        out = {"s0": torch.as_tensor(e["s0"][:n]), "a": torch.as_tensor(e["a"][:n, :length]), "xi": torch.as_tensor(e["xi"][:n])}
        if full:
            out.update(s1=torch.as_tensor(e["s1"][:n, :length]), f=torch.as_tensor(e["f"][:n, :length]), t0=torch.as_tensor(e["t0"][:n]),
                       env=torch.as_tensor(e["env"][:n]))  # fmt: skip
        return {k: v.contiguous().to(dev) for k, v in out.items()}
    from ndm_gym.wmbench.fit import fixed_windows, load_buffer

    buf, _ = load_buffer([path], "cpu")
    w = fixed_windows(buf, n, length, seed, index=full)
    out = {"s0": w["s"][:, 0], "a": w["a"], "xi": w["xi"][:, 0]}
    if full:
        rows = w["t0"].unsqueeze(-1) + torch.arange(length)  # load_buffer keeps the file's rows in time order
        out.update(s1=w["s1"], f=buf.data["f"][rows, w["env"].unsqueeze(-1).expand_as(rows)], t0=w["t0"], env=w["env"])
    return {k: v.contiguous().to(dev) for k, v in out.items()}


def barrier(path: str, n: int, lead: float = 2.0) -> float:
    """--barrier: this process is ready (``path``/ready.<pid>.<thread>); wait for ``n`` of them, then until the common start, written once
    by the first caller to see ``n`` (``path``/go, ``lead`` s later), so that the processes' timed runs overlap. Returns that start."""
    import threading

    os.makedirs(path, exist_ok=True)
    me = f"{os.getpid()}.{threading.get_ident()}"
    open(os.path.join(path, f"ready.{me}"), "w").close()  # one per caller
    go = os.path.join(path, "go")
    while not os.path.exists(go):
        if len([f for f in os.listdir(path) if f.startswith("ready.")]) >= n and not os.path.exists(go):
            tmp = os.path.join(path, f"go.{me}")
            with open(tmp, "w") as f:
                f.write(f"{time.time() + lead:.3f}")
            try:  # one start for all: a link fails if another caller that saw n made go first (a rename would replace its start)
                os.link(tmp, go)
            except FileExistsError:
                pass
            os.remove(tmp)
            break
        time.sleep(0.01)
    t_go = float(open(go).read())  # whole: go appears by the link, after its contents were written
    while time.time() < t_go:
        time.sleep(0.001)
    return t_go


def gpu_driver(uuid: str) -> str:
    """The driver version for the row: $NDM_GPU_DRIVER when set (a block that runs no nvidia-smi while timing), else nvidia-smi's."""
    if os.environ.get("NDM_GPU_DRIVER"):
        return os.environ["NDM_GPU_DRIVER"]
    return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", uuid], capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def timed(fn, warmup: int, steps: int, warm_seconds: float = 0.0) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    keep_busy(fn, torch.cuda.synchronize, warm_seconds)
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    args = parse_args()

    import ndm_gym.core.nets as nets
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
    win = lengths = sched = restart = fn_seg = None  # --replay: the windows, the timed segments, each segment's windows
    if args.replay is not None:
        from ndm_gym.state import unflatten_state

        win = replay_windows(args.replay, args.replay_windows, args.replay_len, args.replay_seed, dev)
        lengths = segment_lengths(args.steps, args.replay_len)
        sched = replay_schedule(args.replay_windows, N, len(lengths), dev)
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
            if win is not None:  # restarts written as TrackingSim.reset writes them, targets as TrackingSim.step sets them
                robot = robots[0]
                assert list(robot.joint_names) == g1.G1_JOINT_NAMES, robot.joint_names
                alpha = torch.tensor([g1.action_scale(n) for n in g1.G1_JOINT_NAMES], device=dev)
                default_q = robot.data.default_joint_pos.torch[0].clone()
                cur = {}

                def restart(k):
                    sel = sched[k]
                    p, R, v, w, q, qd = unflatten_state(win["s0"][sel])
                    quat = mat_to_quat(R)
                    robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p + scene.env_origins, quat[:, 1:4], quat[:, :1]), -1), env_ids=ids)
                    robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=ids)
                    robot.write_joint_position_to_sim_index(position=q.contiguous(), env_ids=ids)
                    robot.write_joint_velocity_to_sim_index(velocity=qd.contiguous(), env_ids=ids)
                    scene.reset(ids)
                    cur["target"] = (default_q + win["xi"][sel, 1:30]).unsqueeze(0) + alpha * win["a"][sel].transpose(0, 1)

                def fn_seg(j):
                    robot.set_joint_position_target_index(target=cur["target"][j])
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim_cfg.dt)

                extra = "replayed windows: their recorded start states and setpoints"
        elif args.mode == "env":
            env = PairedTrackingSim(sim_cfg, 2 * N, motion, mdp)
            ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)

            def fn():
                a = ref_u[env.f + 1]
                env.step(a + args.action_noise * torch.randn_like(a), env.state())

            extra = "reference setpoints + noise, rewards, terminations, paired resets"
        else:
            wm, ta, source = build_world_model(args, dev)
            wm.requires_grad_(False)
            n_par = sum(t.numel() for t in wm.net.parameters())
            if win is None:
                env = PairedTrackingSim(sim_cfg, 2 * min(N, 512), motion, mdp, eval_mode=True)  # the reference states and contexts
                env.reset(torch.arange(env.N, device=dev), f0.repeat_interleave(env.P))
                env.refresh()
                idx = torch.arange(per_env * N, device=dev) % env.N
                S = tuple(t[idx].contiguous() for t in env.state())
                xi_n = normalize_xi(env.xi)[idx].contiguous()
                ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
                a = ref_u[env.f[idx] + 1].contiguous()
            else:  # the windows hold the states, setpoints and contexts (the reference's environment is not built)
                S = tuple(t.contiguous() for t in unflatten_state(win["s0"][sched[0]]))
                xi_n = normalize_xi(win["xi"][sched[0]]).contiguous()
                a = win["a"][sched[0], 0].contiguous()
            graphs = args.compile == "reduce-overhead"
            if args.grad_horizon:  # the model as trained: its step compiled, the whole call one CUDA graph (--compile)
                step = wm.step if args.compile == "none" else torch.compile(wm.step)
                path = f"compile {args.compile}" + (" (the step compiled, the call one CUDA graph)" if graphs else "")
            else:
                model, step, path = inference_model(wm, args)
            state = {"S": S}

            @torch.no_grad()
            def fn():
                if graphs:  # CUDA graphs reuse their output buffers: mark the step and keep copies of the state
                    torch.compiler.cudagraph_mark_step_begin()
                S1 = step(state["S"], a, xi_n)
                state["S"] = tuple(t.clone() for t in S1) if graphs else S1

            extra = f"{ta.wm_backbone} ({n_par / 1e6:.2f} M parameters) {path} | {source}"
            if win is not None:  # the world model's state from each window's first recorded state, its setpoints and context
                xi_win = normalize_xi(win["xi"])
                cur = {}

                def restart(k):
                    sel = sched[k]
                    state["S"] = tuple(t.contiguous() for t in unflatten_state(win["s0"][sel]))
                    cur["a"] = win["a"][sel].transpose(0, 1).contiguous()
                    cur["xi"] = xi_win[sel].contiguous()

                @torch.no_grad()
                def fn_seg(j):
                    if graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    S1 = step(state["S"], cur["a"][j], cur["xi"])
                    state["S"] = tuple(t.clone() for t in S1) if graphs else S1
            if args.grad_horizon:  # rollouts with their gradient instead of steps: the setpoints of the clip's next frames
                from ndm_gym.algos.blinkmimic import Graphed

                H = args.grad_horizon
                if int(motion.clip_len[pc].min()) <= H + 1:
                    raise ValueError(f"--grad_horizon {H}: the clip has {int(motion.clip_len[pc].min())} frames")
                if graphs and args.warmup < 4:  # the graph's eager calls and its capture are not the replayed call
                    raise ValueError(f"--warmup {args.warmup}: the CUDA graph is captured in the first four calls, which must not be timed")
                acts = torch.stack([ref_u[env.f[idx] + 1 + k] for k in range(H)]).contiguous().requires_grad_(True)
                q_ref = mdp.default_q.to(dev)
                call = lambda: rollout_gradient(step, S, acts, xi_n, q_ref)  # noqa: E731
                fn = Graphed(call, lambda: True, f"the {H}-step rollout and its gradient") if graphs else call
                loss, grad = fn()  # also the first of the graph's eager calls
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError(f"--grad_horizon {H}: the gradient is not finite")
                extra += (f" | gradient of a {H}-step rollout's loss with respect to its setpoints, {args.steps} rollouts per run, loss "
                          f"{float(loss):.4f}, mean |gradient| {float(grad.abs().mean()):.3e}")  # fmt: skip
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
        first = lambda r: (args.warmup, args.warm_seconds) if r == 0 else (0, 0.0)  # noqa: E731
        if win is None:
            runs = sorted(timed(fn, first(r)[0], args.steps, first(r)[1]) for r in range(args.repeats))
        else:
            wait = (lambda: barrier(args.barrier, args.barrier_n)) if args.barrier else None
            runs = sorted(timed_segments(fn_seg, restart, lengths, *first(r), wait=wait if r == 0 else None) for r in range(args.repeats))
            extra += (f" | replay {os.path.basename(args.replay)}: {args.replay_windows} windows of {args.replay_len} steps (seed "
                      f"{args.replay_seed}), segments {lengths}, restarts untimed")  # fmt: skip
        sec = runs[len(runs) // 2]
        extra += " | runs " + " ".join(f"{r:.4f}" for r in runs)
        extra += f" | warm {args.warm_seconds:g} s" if args.warm_seconds > 0 else ""
        steps = args.steps * max(args.grad_horizon, 1)  # control steps of a timed run
        robots_rate = per_env * N * steps / sec
        mem = torch.cuda.max_memory_allocated() / 2**30
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        driver = gpu_driver(uuid)
        # --single: mode "<mode>_single", and the per-env column holds env-steps, one robot each
        mode = args.mode + ("_single" if args.single else "") + (f"_grad{args.grad_horizon}" if args.grad_horizon else "")
        mode += "_replay" if args.replay is not None else ""
        row = (f"{os.path.basename(args.pair)},{mode},{N},{per_env * N},{steps},{sec:.4f},{robots_rate:.1f},{robots_rate / per_env:.1f},"
               f"{1e3 * sec / steps:.3f},{args.amp},{torch.cuda.get_device_name()},{driver},{mem:.2f},{extra}")  # fmt: skip
        new = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            if new:
                f.write("pair,mode,envs,robots,steps,seconds,robot_steps_per_s,pair_steps_per_s,ms_per_step,amp,gpu,driver,peak_mem_gb,detail\n")
            f.write(row + "\n")
        print(f"[speed_pair] {mode} {N} envs: {robots_rate:,.0f} robot-steps/s ({robots_rate / per_env:,.0f} env-steps/s), "
              f"{1e3 * sec / steps:.2f} ms per step ({extra})", flush=True)


if __name__ == "__main__":
    main()
