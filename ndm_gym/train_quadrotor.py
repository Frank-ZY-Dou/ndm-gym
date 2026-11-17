"""Training loop for quadrotor trajectory tracking on flights of Race Against the Machine (docs/tasks/quadrotor.md): collect
in QuadSim -> fit the world model on replay windows -> policy update through imagined rollouts (ndm_gym.algos.quadrotor_algos).
The loop, the schedule and the systems options are hand-object tracking's (ndm_gym.train_hand, ndm_gym.algos.hand_fast) with
the defaults of QUAD_DEFAULTS; ndm_gym.train's arguments that this training does not implement are refused unless left at
their defaults (UNSUPPORTED).

An iteration is one control step of every world, ``--wm_batches`` world-model steps and, after ``--wm_only_iters``, one policy
(and critic) update. Writes ``train_log.csv``, ``args.json`` and ``last.pt`` to ``--log_dir`` (``--ckpt_keep``: also
``ckpt_<iteration>.pt`` at every ``--ckpt_every``). ``--resume`` continues from a run's ``last.pt``; the replay is not saved.
``QUAD_PROFILE=1`` prints each part's time with every log row.

Run: ``python -m ndm_gym.train_quadrotor --flight flight-01a-ellipse --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ndm_gym.train import add_train_args

LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "pos_mm",
               "ep_len", "term_frac", "it_per_s", "lr")  # fmt: skip
# ndm_gym.train's defaults that this training replaces (hand-object tracking's, with a longer imagined window)
QUAD_DEFAULTS = {"algo": "supermimic", "num_envs": 2048, "iterations": 20_000, "lr_decay_iters": 20_000, "buffer_steps": 256, "warmup_steps": 32,
                 "wm_only_iters": 300, "horizon": 16, "goal_frames": 5, "wm_hidden": [1024, 1024, 1024], "wm_batches": 4, "amp": "none"}  # fmt: skip
# ndm_gym.train's arguments that this training does not implement; each is refused unless at its default
# (tests/test_quadrotor_tracking.py checks that every argument is either implemented or here)
UNSUPPORTED = ("st_no_root_terms", "st_no_task_regs", "st_substeps", "st_balance_w", "imag_start_perturb", "bptt_clip_k",
               "pi_rollout_clip", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window", "sm_residual", "st_com", "st_quad_w",
               "st_late_w", "st_fail_margin", "utd", "hist", "wm_input_noise", "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs",
               "wm_backbone", "rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_rope_detach",
               "rf_fused_rope", "rf_anchors", "rf_anchor_mode", "rf_rope_center", "rf_merge", "wm_sample", "wm_prio_alpha",
               "wm_fail_windows", "wm_fail_window", "st_wm_loss", "wm_lr", "no_critic", "pi_updates", "lr_decay_hours", "ckpt_hours",
               "motion_dir", "motions", "paired", "clips")  # fmt: skip
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the options a resumed run must share with its checkpoint: the algorithm and SuperTrack's weights, the optimizers' layouts
# and states, the policy average
RESUME_MATCH = ("algo", "st_task_weights", "fused_adam", "update_graphs", "pi_opt", "wm_opt", "pi_ema")


def parse_args(argv: list[str] | None = None):
    from ndm_gym.algos.hand_fast import add_hand_fast_args, check_hand_fast_args

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_hand_fast_args(parser)  # the systems options
    parser.add_argument("--flight", type=str, nargs="+", required=True,
                        help="the flights to track: names in --data_dir (flight-01a-ellipse) or paths of converted flights (.npz)")  # fmt: skip
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO, "data", "drone_racing"), help="tools/download_drone_racing.py's output")
    parser.add_argument("--goal_stride", type=int, default=5, help="the goal's frames are 1, 1 + stride, 1 + 2 stride, ... control steps ahead")
    parser.add_argument("--episode", type=int, default=500, help="an episode's longest length in control steps (100 Hz)")
    parser.add_argument("--st_task_weights", action="store_true",
                        help="supertrack: no calibration; each of the task's four tracked quantities weighs, per unit of error, the task's reward "
                        "weight over its kernel's width")  # fmt: skip
    parser.add_argument("--ckpt_keep", action="store_true", help="also keep ckpt_<iteration>.pt at every --ckpt_every (the policy and its arguments)")
    for action in parser._actions:
        if action.dest == "algo":
            action.required = False
    parser.set_defaults(**QUAD_DEFAULTS)
    from ndm_gym import configs

    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first
        argv = configs.expand(sys.argv[1:] if argv is None else list(argv), "train_quadrotor")
    except ValueError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    if args.algo not in ("supertrack", "supermimic"):
        parser.error("quadrotor training implements --algo supertrack and --algo supermimic")
    st_only = {"st_task_weights": False, **{k: parser.get_default(k) for k in ("st_reg", "st_imag_noise", "st_fail_w")}}
    sm_only = {k: parser.get_default(k) for k in ("bptt_decay", "gamma", "lam", "critic_epochs", "critic_minibatches")}  # its critic and return
    if args.algo == "supermimic":
        for name, default in st_only.items():
            if getattr(args, name) != default:
                parser.error(f"--{name} is SuperTrack's option")
    else:
        for name, default in sm_only.items():
            if getattr(args, name) != default:
                parser.error(f"--{name} is SuperMimic's option here")
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that quadrotor training does not implement")
    if args.compile_eager:
        parser.error("--compile_eager compiles the hand's forward kinematics; the quadrotor's collection step has none")
    if args.goal_stride < 1 or args.episode < 2:
        parser.error("--goal_stride >= 1, --episode >= 2")
    try:
        check_hand_fast_args(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def flight_paths(flights: list[str], data_dir: str) -> list[str]:
    return [f if f.endswith(".npz") else os.path.join(data_dir, f + ".npz") for f in flights]


def build(flights: list[str], data_dir: str, device: str, reward_std_scale: float = 1.0, goal_stride: int = 5, model_spec=None):
    """The drone's model and specification and the task of the flights. ``model_spec``: the specification of the simulated
    model where it is not the task's (evaluate_quadrotor's scaled inertia, drag and mass)."""
    from ndm_gym.motion.sources.drone_racing import load_flight
    from ndm_gym.robots.quadrotor import CONTROL_DT, QuadrotorSpec, load_model
    from ndm_gym.tasks.quadrotor_tracking import QuadTask, RewardCfg, build_reference

    spec = QuadrotorSpec()
    ref = build_reference([load_flight(p, CONTROL_DT) for p in flight_paths(flights, data_dir)], spec, device)
    return load_model(model_spec or spec), spec, QuadTask(ref, spec, RewardCfg().scaled(reward_std_scale), goal_stride)


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import hand_fast
    from ndm_gym.algos.blinkmimic import PolicyLR
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.quadrotor_algos import ACC_KEYS, QUAD_ALGOS, WINDOW_KEYS, QuadCollector, collect_step, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.robots.quadrotor import NUM_U
    from ndm_gym.sim.rigid.quadrotor_sim import QuadSim

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    RunningNorm.no_sync = args.no_sync  # --no_sync: the normalizers' freeze test on the device, their statistics written in place
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path, header = os.path.join(args.log_dir, "train_log.csv"), ",".join(LOG_COLUMNS)
    resume_ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
    if resume_ck is not None:
        for k in RESUME_MATCH:
            if resume_ck["args"][k] != getattr(args, k):
                raise ValueError(f"--resume: --{k} must match the checkpoint's run ({resume_ck['args'][k]!r})")
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, _, task = build(args.flight, args.data_dir, dev, args.reward_std_scale, args.goal_stride)
    sim = QuadSim(model, task, args.num_envs, dev, episode=args.episode, stagger=True)
    algo = QUAD_ALGOS[args.algo](args, task, dev, substeps=sim.substeps)
    average = hand_fast.PolicyAverage(algo.pi, args.pi_ema) if args.pi_ema > 0 else None  # --pi_ema
    pi_lr = PolicyLR(args)  # --pi_lr_mult
    start_it, wall_offset = 0, 0.0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck if average is None else average.resume(resume_ck))
        pi_lr.load_state_dict(resume_ck)
        start_it, wall_offset = resume_ck["iteration"], resume_ck["wall_time"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    if args.collect_graph or args.update_graphs:  # the graphs read the replay's write head and fill level on the device
        buffer.use_device_ptr(device_size=True)
    N = args.num_envs
    if args.no_sync:  # the step without a host read; --collect_graph: as one CUDA graph
        collect = QuadCollector(sim, task, algo, buffer, args)
        acc = collect.acc
    else:
        a_prev, ep_len = torch.zeros(N, NUM_U, device=dev), torch.zeros(N, device=dev)
        acc = {k: torch.zeros((), device=dev) for k in ACC_KEYS}

        def host_step(u):
            out = sim.step(u, reset=not args.reset_masked)
            if args.reset_masked:  # the ended worlds reset by masked writes, the draws made for all worlds
                sim.reset_masked(out[2])
            return out

        @torch.no_grad()
        def collect():
            collect_step(sim, task, algo, buffer, args, a_prev, ep_len, acc, host_step, buffer.add)

    # the iteration's updates; --update_graphs: each as a CUDA graph; --pi_ema: the average's step with the policy's
    wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, average)
    timers = None
    if os.environ.get("QUAD_PROFILE", "0") == "1":  # each part's time per call, the device synchronized around it, with every log row
        timers = collect, wm_update, policy_update = [hand_fast.Timed(f) for f in (collect, wm_update, policy_update)]

    def save(iteration: int, keep: bool = False):
        ck = {**algo.state_dict(), **pi_lr.state_dict(), "iteration": iteration, "wall_time": time.time() - t_start, "sim_steps": iteration * N,
              "args": vars(args)}  # fmt: skip
        if average is not None:  # --pi_ema: 'pi' is the average (the policy evaluated), 'pi_live' the trained policy
            ck = average.checkpoint(ck)
        if args.no_sync:  # the checkpoint's tensors to the host in one transfer per dtype, not one each
            ck = hand_fast.to_host(ck)
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        if keep:  # what evaluate_quadrotor reads of a checkpoint
            torch.save({k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "args")}, os.path.join(args.log_dir, f"ckpt_{iteration:06d}.pt"))

    for _ in range(args.warmup_steps):
        collect()
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:
        log_f.write(header + "\n")
    t_log = time.time()
    t_start = t_log - wall_offset
    agg = {k: [] for k in ("wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival")}
    it = start_it
    for it in range(start_it + 1, args.iterations + 1):
        lr = lr_at(it, args.lr, args.lr_decay_iters)
        for opt in algo.optimizers:
            set_lr(opt, lr)
        if pi_lr.on:  # --pi_lr_mult
            set_lr(algo.opt_pi, lr * pi_lr.at(it - 1))
        collect()
        agg["wm_loss"].append(wm_update())
        if it > args.wm_only_iters:
            for k, x in zip(("pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival"), policy_update(), strict=True):
                agg[k].append(x)
        if it % args.log_every == 0:
            nan = torch.full((), float("nan"), device=dev)
            vals = torch.stack([*(torch.stack(agg[k]).mean() if agg[k] else nan for k in agg), acc["done"], acc["r"] / acc["steps"],
                                acc["pos_err"] / acc["steps"], acc["ep_len"], acc["term"]]).tolist()  # fmt: skip
            # (the row's values above in one read of the device)
            means, (n_done, sim_r, pos_err, ep_sum, n_term) = vals[: len(agg)], vals[len(agg) :]
            done, now = max(n_done, 1.0), time.time()
            row = [it, (now - t_start) / 3600, it * N, *means, sim_r, 1000 * pos_err, ep_sum / done, n_term / done,
                   args.log_every / (now - t_log), lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            log_f.flush()
            print(" ".join(f"{k}={x:.4g}" for k, x in zip(LOG_COLUMNS, row, strict=True)), flush=True)
            if timers is not None:
                print("[prof] collect {:.1f} ms | model {:.1f} ms | policy {:.1f} ms".format(*(t.take() for t in timers)), flush=True)
            msg = pi_lr.observe(it, row[LOG_COLUMNS.index("sim_r")])  # --lr_decay_ret's trigger
            if msg:
                print(msg, flush=True)
            t_log = now
            for xs in agg.values():
                xs.clear()
            for x in acc.values():
                x.zero_()
        if it % args.ckpt_every == 0:
            save(it, keep=args.ckpt_keep)
        if args.max_hours is not None and time.time() - t_start > args.max_hours * 3600:
            break
    save(it)


if __name__ == "__main__":
    main()
