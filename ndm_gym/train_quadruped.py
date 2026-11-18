"""Training loop for quadruped motion tracking: the Go2 follows retargeted dog motion capture (motion.sources.lifelike),
one clip or many with one policy: collect in QuadrupedSim -> fit the world model on replay windows -> policy update
through imagined rollouts (ndm_gym.algos.quadruped_algos). The schedule and the algorithm's settings are LAFAN1's
(ndm_gym.train's arguments) with the defaults of QUADRUPED_DEFAULTS; those that quadruped training does not implement
are refused unless left at their defaults (UNSUPPORTED).

An iteration is one control step of every world, ``--wm_batches`` world-model steps and, after ``--wm_only_iters``,
one policy (and critic) update. Writes ``train_log.csv``, ``args.json`` and ``last.pt`` to ``--log_dir``. ``last.pt``
holds what ``--resume`` continues from: the networks with their normalizers and the optimizers' states. The replay is not
saved; a resumed run refills it with ``--warmup_steps`` of collection, as a new run starts, and continues its log.
``--keep_ckpts`` also keeps the policy of every ``--ckpt_every`` iterations (``ckpt_<iteration>.pt``, with the run's wall
time), which ``evaluate_quadruped --ckpts`` evaluates for the training time to a given success.

Systems options (the hand-object task's, ndm_gym.algos.hand_fast, off by default): ``--fused_adam``, ``--reset_masked``,
``--no_sync`` (an iteration without host-device synchronization), ``--collect_graph`` and ``--update_graphs`` (the
collection step and the updates as CUDA graphs), ``--compile_eager`` (the collection's forward kinematics compiled),
``--pi_opt muon``, ``--wm_opt muon``, ``--pi_ema`` and ``--pi_lr_mult``. ``QUADRUPED_PROFILE=1`` prints each part's time
with every log row.

Run: ``python -m ndm_gym.train_quadruped --clips dog_quad_walk_002_1 --log_dir logs/<run>`` (``--clips``: clip names or
globs in data/lifelike_go2, comma-separated, or a directory; default every clip)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ndm_gym.train import add_train_args

LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "trunk_mm",
               "feet_mm", "ep_len", "term_frac", "it_per_s", "lr")  # fmt: skip
# ndm_gym.train's defaults that quadruped training replaces
QUADRUPED_DEFAULTS = {"algo": "supermimic", "num_envs": 4096, "iterations": 5000, "lr_decay_iters": 5000, "buffer_steps": 256, "warmup_steps": 32,
                      "wm_only_iters": 300, "horizon": 16, "goal_frames": 5, "wm_hidden": [1024, 1024, 1024], "wm_batches": 4, "amp": "none"}  # fmt: skip
# ndm_gym.train's arguments that quadruped training does not implement (SuperTrack's, LAFAN1's tricks, data, samplers and
# the RigidFormer's options); each is refused unless at its default. tests/test_quadruped_tracking.py checks that every
# argument is either implemented or here.
UNSUPPORTED = ("st_no_root_terms", "st_no_task_regs", "st_substeps", "st_balance_w", "imag_start_perturb", "bptt_clip_k",
               "pi_rollout_clip", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window", "sm_residual", "st_com", "st_quad_w",
               "st_late_w", "st_fail_margin", "utd", "hist", "wm_input_noise", "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs",
               "wm_backbone", "rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_rope_detach",
               "rf_fused_rope", "rf_anchors", "rf_anchor_mode", "rf_rope_center", "rf_merge", "wm_sample", "wm_prio_alpha",
               "wm_fail_windows", "wm_fail_window", "st_wm_loss", "wm_lr", "no_critic", "pi_updates", "lr_decay_hours", "ckpt_hours",
               "motion_dir", "motions", "paired")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    from ndm_gym.algos.hand_fast import add_hand_fast_args, check_hand_fast_args
    from ndm_gym.robots.go2 import JOINT_DAMPING, KD, KP

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_hand_fast_args(parser)  # the systems options
    parser.add_argument("--keep_ckpts", action="store_true",
                        help="also keep the policy at every --ckpt_every iterations as ckpt_<iteration>.pt (evaluate_quadruped --ckpts)")  # fmt: skip
    for action in parser._actions:
        if action.dest == "algo":
            action.required = False
        if action.dest == "clips":
            action.help = "the clips: names or globs in data/lifelike_go2 (comma-separated), paths, or a directory (default: every clip)"
    parser.add_argument("--kp", type=float, default=KP, help="the joints' position servos' stiffness (N m / rad)")
    parser.add_argument("--joint_damping", type=float, default=JOINT_DAMPING, help="the joints' damping (N m s / rad); negative: the model's own (Menagerie: 2)")
    parser.add_argument("--kd", type=float, default=KD, help="the servos' damping (N m s / rad), on top of the model's joint damping")
    parser.add_argument("--joint_noise", type=float, default=0.05, help="the joints' perturbation at an episode's start (rad, std)")
    parser.add_argument("--st_task_weights", action="store_true",
                        help="supertrack: no calibration; each of the task's six tracked quantities weighs, per unit of error, the task's "
                        "reward weight over its kernel's width")  # fmt: skip
    parser.set_defaults(**QUADRUPED_DEFAULTS)
    from ndm_gym import configs

    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first
        argv = configs.expand(sys.argv[1:] if argv is None else list(argv), "train_quadruped")
    except ValueError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    if args.algo not in ("supertrack", "supermimic"):
        parser.error("quadruped training implements --algo supertrack and --algo supermimic")
    st_only = {"st_task_weights": False, **{k: parser.get_default(k) for k in ("st_reg", "st_imag_noise", "st_fail_w")}}
    sm_only = {k: parser.get_default(k) for k in ("bptt_decay", "gamma", "lam", "critic_epochs", "critic_minibatches")}  # its critic and return
    for name, default in (st_only if args.algo == "supermimic" else sm_only).items():
        if getattr(args, name) != default:
            parser.error(f"--{name} is {'SuperTrack' if args.algo == 'supermimic' else 'SuperMimic'}'s option")
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that quadruped training does not implement")
    try:
        check_hand_fast_args(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def build(clips: str | list[str] | None, device: str, reward_std_scale: float = 1.0, kp: float | None = None, kd: float | None = None,
          joint_damping: float | None = None):
    """The scene, kinematics, task and clip names of the clips (motion.sources.lifelike.clip_paths)."""
    from ndm_gym.motion.sources.lifelike import clip_paths, load_clip
    from ndm_gym.robots.go2 import KD, KP, Go2Kinematics, load_scene
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.quadruped_tracking import QuadrupedTask, RewardCfg, build_reference

    model = load_scene(kp=KP if kp is None else kp, kd=KD if kd is None else kd, joint_damping=joint_damping)
    kin = Go2Kinematics(model, device)
    loaded = [load_clip(p, CONTROL_DT) for p in clip_paths(clips)]
    ref = build_reference(loaded, kin, device)
    return model, kin, QuadrupedTask(kin, ref, RewardCfg().scaled(reward_std_scale)), [c["name"] for c in loaded]


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import hand_fast
    from ndm_gym.algos.blinkmimic import PolicyLR
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.quadruped_algos import QUADRUPED_ALGOS, WINDOW_KEYS, QuadrupedCollector, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.robots.go2 import NUM_J
    from ndm_gym.sim.rigid.quadruped_sim import QuadrupedSim
    from ndm_gym.tasks.quadruped_tracking import unflatten_state

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
        hand_fast.check_resume(resume_ck["args"], args)
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")

    model, kin, task, names = build(args.clips, dev, args.reward_std_scale, args.kp, args.kd, args.joint_damping if args.joint_damping >= 0 else None)
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "clip_names": names}, f, indent=2)
    print(f"{len(names)} clips, {task.ref.frames} frames ({task.ref.frames * 0.02:.1f} s)", flush=True)
    sim = QuadrupedSim(model, task, args.num_envs, dev, joint_noise=args.joint_noise)
    algo = QUADRUPED_ALGOS[args.algo](args, model, kin, task, dev)
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
    a_prev = torch.zeros(N, NUM_J, device=dev)
    ep_len = torch.zeros(N, device=dev)
    acc = {k: torch.zeros((), device=dev) for k in ("r", "steps", "done", "term", "ep_len", "trunk_err", "feet_err")}

    @torch.no_grad()
    def collect():
        nonlocal a_prev
        s, f = sim.state(), sim.f.clone()
        S = unflatten_state(s)
        a = algo.act(S, a_prev, f, noise=args.action_noise)
        u = task.setpoint(a, f)
        s1, term, done = sim.step(u, reset=not args.reset_masked)
        if args.reset_masked:  # the ended worlds reset by masked writes, the draws made for all worlds
            sim.reset_masked(done)
        S1, f1 = unflatten_state(s1), task.ref.next_frame(f)
        fk1 = kin.forward(S1[0])
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done})
        ep_len.add_(1)
        e = task.errors(S1, fk1, f1)
        acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
        acc["trunk_err"] += e["trunk_pos"].sum()
        acc["feet_err"] += e["feet"].mean(-1).sum()
        acc["steps"] += N
        acc["done"] += done.sum()
        acc["term"] += term.sum()
        acc["ep_len"] += ep_len[done].sum()
        ep_len[done] = 0
        a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)

    if args.no_sync:  # the same step without a host read; --collect_graph: as one CUDA graph (hand_fast.Collector)
        collect = QuadrupedCollector(sim, task, algo, buffer, args)
        acc = collect.acc
    # the iteration's updates; --update_graphs: each as a CUDA graph; --pi_ema: the average's step with the policy's
    wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, average)
    timers = None
    if os.environ.get("QUADRUPED_PROFILE", "0") == "1":  # each part's time per call, the device synchronized around it
        timers = collect, wm_update, policy_update = [hand_fast.Timed(f) for f in (collect, wm_update, policy_update)]

    def save_last(iteration: int, keep: bool = False):
        ck = {**algo.state_dict(), **pi_lr.state_dict(), "iteration": iteration, "wall_time": time.time() - t_start, "sim_steps": iteration * N,
              "args": vars(args)}  # fmt: skip
        if average is not None:  # --pi_ema: 'pi' is the average (the policy evaluated), 'pi_live' the trained policy
            ck = average.checkpoint(ck)
        if args.no_sync:  # the checkpoint's tensors to the host in one transfer per dtype, not one each
            ck = hand_fast.to_host(ck)
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        if keep:  # --keep_ckpts: the policy evaluated, with the run's wall time
            torch.save({k: ck[k] for k in ("pi", "iteration", "wall_time", "sim_steps", "args")}, os.path.join(args.log_dir, f"ckpt_{iteration:06d}.pt"))

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
                                acc["trunk_err"] / acc["steps"], acc["feet_err"] / acc["steps"], acc["ep_len"], acc["term"]]).tolist()  # fmt: skip
            means, (n_done, sim_r, trunk_err, feet_err, ep_sum, n_term) = vals[: len(agg)], vals[len(agg) :]
            done, now = max(n_done, 1.0), time.time()
            row = [it, (now - t_start) / 3600, it * N, *means, sim_r, 1000 * trunk_err, 1000 * feet_err, ep_sum / done, n_term / done,
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
            save_last(it, keep=args.keep_ckpts)
        if args.max_hours is not None and time.time() - t_start > args.max_hours * 3600:
            break
    save_last(it)


if __name__ == "__main__":
    main()
