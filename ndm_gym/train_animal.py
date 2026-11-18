"""Training loop for animal motion tracking (docs/tasks/animals.md): an animal body (robots.animals: the rat, the fruit fly or the
stick insect of MIMIC-MJX) tracks clips of its animal's recorded kinematics (motion.sources.mimic_mjx). Collect in AnimalSim ->
fit the world model on replay windows -> policy update through imagined rollouts (ndm_gym.algos.animal_algos).

The loop, the schedule and the systems options are hand-object tracking's (ndm_gym.train_hand, ndm_gym.algos.hand_fast) with
the defaults of ANIMAL_DEFAULTS; ndm_gym.train's arguments that this training does not implement are refused unless left at
their defaults (UNSUPPORTED).

An iteration is one control step of every world, ``--wm_batches`` world-model steps and, after ``--wm_only_iters``, one policy
(and critic) update. Writes ``train_log.csv``, ``args.json`` and ``last.pt`` to ``--log_dir`` (``--ckpt_keep``: also
``ckpt_<iteration>.pt`` at every ``--ckpt_every``). ``--resume`` continues from a run's ``last.pt``; the replay is not saved.
``ANIMAL_PROFILE=1`` prints each part's time with every log row.

Run: ``python -m ndm_gym.train_animal --animal rat --clips 0-99 --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ndm_gym.train import add_train_args

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "body_err",
               "ep_len", "term_frac", "it_per_s", "lr")  # fmt: skip
# ndm_gym.train's defaults that this training replaces (hand-object tracking's)
ANIMAL_DEFAULTS = {"algo": "supermimic", "num_envs": 2048, "iterations": 6000, "lr_decay_iters": 6000, "buffer_steps": 256, "warmup_steps": 32,
                   "wm_only_iters": 300, "horizon": 16, "goal_frames": 5, "wm_hidden": [1024, 1024, 1024], "wm_batches": 4, "amp": "none"}  # fmt: skip
# ndm_gym.train's arguments that this training does not implement; each is refused unless at its default
UNSUPPORTED = ("st_no_root_terms", "st_no_task_regs", "st_substeps", "st_balance_w", "imag_start_perturb", "bptt_clip_k",
               "pi_rollout_clip", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window", "sm_residual", "st_com", "st_quad_w",
               "st_late_w", "st_fail_margin", "utd", "hist", "wm_input_noise", "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs",
               "wm_backbone", "rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_rope_detach",
               "rf_fused_rope", "rf_anchors", "rf_anchor_mode", "rf_rope_center", "rf_merge", "wm_sample", "wm_prio_alpha",
               "wm_fail_windows", "wm_fail_window", "st_wm_loss", "no_critic", "pi_updates", "lr_decay_hours", "ckpt_hours",
               "motion_dir", "motions", "paired", "clips")  # fmt: skip
# the options a resumed run must share with its checkpoint
RESUME_MATCH = ("algo", "animal", "st_task_weights", "fused_adam", "update_graphs", "pi_opt", "wm_opt", "pi_ema")


def parse_clips(tokens: list[str]) -> list[int]:
    """Clip indices from tokens such as ``3`` or ``0-99`` (inclusive)."""
    out = []
    for t in tokens:
        a, _, b = t.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_args(argv: list[str] | None = None):
    from ndm_gym.algos.hand_fast import add_hand_fast_args, check_hand_fast_args

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_hand_fast_args(parser)  # the systems options
    parser.add_argument("--animal", type=str, required=True, choices=["rat", "fly", "stick"])
    parser.add_argument("--clip_ids", type=str, nargs="+", required=True, help="the clips to track: indices or ranges (0-99)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO, "data", "mimic_mjx"), help="tools/download_mimic_mjx.py's output")
    parser.add_argument("--goal_stride", type=int, default=1, help="the goal's frames are 1, 1 + stride, 1 + 2 stride, ... control steps ahead")
    parser.add_argument("--episode", type=int, default=1000, help="an episode's longest length in control steps")
    parser.add_argument("--action_scale", type=float, default=0.5, help="a position servo's offset from the reference's joint at action 1 (rad)")
    parser.add_argument("--fused_fk", action="store_true",
                        help="the bodies' forward kinematics body by body with elementwise 3 x 3 products and no scatter (MjcfWorldModel.fused's "
                        "path): far fewer kernels once compiled (changes results by rounding)")  # fmt: skip
    parser.add_argument("--st_task_weights", action="store_true",
                        help="supertrack: no calibration; each tracked quantity weighs, per unit of error, the task's reward weight over its "
                        "kernel's width")  # fmt: skip
    parser.add_argument("--ckpt_keep", action="store_true", help="also keep ckpt_<iteration>.pt at every --ckpt_every (the policy and its arguments)")
    for action in parser._actions:
        if action.dest == "algo":
            action.required = False
    parser.set_defaults(**ANIMAL_DEFAULTS)
    from ndm_gym import configs

    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first
        argv = configs.expand(sys.argv[1:] if argv is None else list(argv), "train_animal")
    except ValueError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    if args.algo not in ("supertrack", "supermimic"):
        parser.error("animal training implements --algo supertrack and --algo supermimic")
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
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that animal training does not implement")
    if args.goal_stride < 1 or args.episode < 2 or args.action_scale <= 0:
        parser.error("--goal_stride >= 1, --episode >= 2, --action_scale > 0")
    try:
        check_hand_fast_args(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def build(animal: str, clip_ids: list[int], data_dir: str, device: str, reward_std_scale: float = 1.0, action_scale: float = 0.5,
          goal_stride: int = 1, fused_fk: bool = False):  # fmt: skip
    """The body's MjModel, the world model's layout and its position lead, and the task of the clips."""
    import dataclasses

    import torch

    from ndm_gym.motion.sources.mimic_mjx import load_clips, resample
    from ndm_gym.robots.animals import ANIMALS, activation_filter, build_model
    from ndm_gym.tasks.animal_tracking import AnimalKinematics, AnimalTask, RewardCfg, build_reference
    from ndm_gym.worldmodels.mjcf import MjcfLayout, mean_velocity_lead

    spec = ANIMALS[animal]
    model = build_model(animal, data_dir)
    substeps = round(spec.ctrl_dt / model.opt.timestep)
    lay = MjcfLayout.from_model(build_model(animal, data_dir, activation=False))
    wm_lay = dataclasses.replace(lay, dt=spec.ctrl_dt)  # the world model's step: one control step
    names = [model.body(i).name for i in range(model.nbody)]
    root = names.index(spec.root_body + spec.suffix)
    bodies = [i for i in range(1, model.nbody) if model.body_rootid[i] == root]  # the animal's tree
    ee = [bodies.index(names.index(n + spec.suffix)) for n in spec.end_effectors]
    kin = AnimalKinematics(lay, bodies).to(device)
    kin.fused = fused_fk  # --fused_fk
    data = load_clips(data_dir, animal, clip_ids)
    quat = [lay.anchor[0] + 3]
    clips = [resample(torch.from_numpy(q), torch.from_numpy(v), spec.mocap_hz, spec.ctrl_dt, quat) for q, v in zip(data["qpos"], data["qvel"], strict=True)]
    ref = build_reference(clips, kin, device)
    r, rbar = activation_filter(model, substeps)
    task = AnimalTask(animal, model, kin, ref, ee, spec.ctrl_dt, r, rbar, RewardCfg().scaled(reward_std_scale), action_scale, goal_stride)
    return model, wm_lay, mean_velocity_lead(model, substeps), task


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import hand_fast
    from ndm_gym.algos.animal_algos import ACC_KEYS, ANIMAL_ALGOS, WINDOW_KEYS, AnimalCollector, collect_step, replay_fields
    from ndm_gym.algos.blinkmimic import PolicyLR
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.robots.animals import ANIMALS
    from ndm_gym.sim.rigid.animal_sim import AnimalSim

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

    spec = ANIMALS[args.animal]
    model, wm_lay, pos_lead, task = build(args.animal, parse_clips(args.clip_ids), args.data_dir, dev, args.reward_std_scale, args.action_scale,
                                          args.goal_stride, args.fused_fk)  # fmt: skip
    print(f"{args.animal}: {task.ref.frames} reference frames, {task.nq} qpos, {task.nu} actuators, {task.na} activations, size {task.length:.4g}, "
          f"obs {task.obs_dim}, goal {task.goal_frame_dim} x {args.goal_frames}", flush=True)  # fmt: skip
    sim = AnimalSim(model, task, args.num_envs, dev, spec.ctrl_dt, episode=args.episode, stagger=True, nconmax=spec.nconmax, njmax=spec.njmax)
    algo = ANIMAL_ALGOS[args.algo](args, task, wm_lay, dev, pos_lead)
    average = hand_fast.PolicyAverage(algo.pi, args.pi_ema) if args.pi_ema > 0 else None  # --pi_ema
    pi_lr = PolicyLR(args)  # --pi_lr_mult
    start_it, wall_offset = 0, 0.0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck if average is None else average.resume(resume_ck))
        pi_lr.load_state_dict(resume_ck)
        start_it, wall_offset = resume_ck["iteration"], resume_ck["wall_time"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    if args.collect_graph or args.update_graphs:  # the graphs read the replay's write head and fill level on the device
        buffer.use_device_ptr(device_size=True)
    N = args.num_envs
    if args.no_sync:  # the step without a host read; --collect_graph: as one CUDA graph
        collect = AnimalCollector(sim, task, algo, buffer, args)
        acc = collect.acc
    else:
        a_prev, ep_len = torch.zeros(N, task.nu, device=dev), torch.zeros(N, device=dev)
        acc = {k: torch.zeros((), device=dev) for k in ACC_KEYS}

        def host_step(u):
            out = sim.step(u, reset=not args.reset_masked)
            if args.reset_masked:  # the ended worlds reset by masked writes, the draws made for all worlds
                sim.reset_masked(out[2])
            return out

        @torch.no_grad()
        def collect():
            collect_step(sim, task, algo, args, a_prev, ep_len, acc, host_step, buffer.add, task.kin)

    # the iteration's updates; --update_graphs: each as a CUDA graph; --pi_ema: the average's step with the policy's
    wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, average)
    timers = None
    if os.environ.get("ANIMAL_PROFILE", "0") == "1":  # each part's time per call, the device synchronized around it, with every log row
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
        if keep:  # what evaluate_animal reads of a checkpoint
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
        if args.wm_lr is not None:  # --wm_lr: the world model's own rate on the same schedule (LAFAN1 training's)
            set_lr(algo.opt_wm, lr * args.wm_lr / args.lr)
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
                                acc["body_err"] / acc["steps"], acc["ep_len"], acc["term"]]).tolist()  # fmt: skip
            # (the row's values above in one read of the device)
            means, (n_done, sim_r, body_err, ep_sum, n_term) = vals[: len(agg)], vals[len(agg) :]
            done, now = max(n_done, 1.0), time.time()
            row = [it, (now - t_start) / 3600, it * N, *means, sim_r, body_err / task.length, ep_sum / done, n_term / done,
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
