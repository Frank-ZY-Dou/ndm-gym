"""Training loop for hand-object tracking on one SPIDER clip (docs/tasks/hand.md): collect in HandSim -> fit the world
model on replay windows -> policy update through imagined rollouts (ndm_gym.algos.hand_algos). The schedule and the
algorithm's settings are LAFAN1's (ndm_gym.train's arguments) with the defaults of HAND_DEFAULTS; those that hand training
does not implement are refused unless left at their defaults (UNSUPPORTED).

An iteration is one control step of every world, ``--wm_batches`` world-model steps and, after ``--wm_only_iters``,
one policy and critic update. Writes ``train_log.csv``, ``args.json`` and ``last.pt`` to ``--log_dir``. ``last.pt``
holds what ``--resume`` continues from: the networks with their normalizers and the optimizers' states. The replay is not
saved; a resumed run refills it with ``--warmup_steps`` of collection, as a new run starts, and continues its log.

Run: ``python -m ndm_gym.train_hand --clip data/spider/processed/<dataset>/xhand/right/<task>/0 --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import time

from ndm_gym.train import add_train_args

LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "object_mm",
               "ep_len", "term_frac", "it_per_s", "lr")  # fmt: skip
# ndm_gym.train's defaults that hand training replaces: one clip of a few seconds in 2048 worlds
HAND_DEFAULTS = {"algo": "supermimic", "num_envs": 2048, "iterations": 20_000, "lr_decay_iters": 20_000, "buffer_steps": 256, "warmup_steps": 32,
                 "wm_only_iters": 300, "horizon": 16, "goal_frames": 5, "wm_hidden": [1024, 1024, 1024], "wm_batches": 4, "amp": "none"}  # fmt: skip
# ndm_gym.train's arguments that hand training does not implement (SuperTrack's, LAFAN1's tricks, data, samplers and the
# RigidFormer's options); each is refused unless at its default. tests/test_hand_tracking.py checks that every argument
# is either implemented or here.
UNSUPPORTED = ("st_reg", "st_imag_noise", "st_no_root_terms", "st_no_task_regs", "st_substeps", "st_balance_w", "imag_start_perturb", "bptt_clip_k",
               "bptt_decay", "pi_rollout_clip", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window", "sm_residual", "st_com", "st_quad_w",
               "st_late_w", "st_fail_w", "st_fail_margin", "utd", "hist", "wm_input_noise", "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs",
               "wm_backbone", "rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_rope_detach",
               "rf_fused_rope", "rf_anchors", "rf_anchor_mode", "rf_rope_center", "rf_merge", "wm_sample", "wm_prio_alpha", "wm_fail_windows",
               "wm_fail_window", "st_wm_loss", "wm_lr", "no_critic", "pi_updates", "lr_decay_hours", "ckpt_hours", "motion_dir", "motions",
               "paired", "clips")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    parser.add_argument("--clip", type=str, required=True, help="a SPIDER trial directory (.../<task>/<trial>)")
    for action in parser._actions:
        if action.dest == "algo":
            action.required = False
    parser.set_defaults(**HAND_DEFAULTS)
    args = parser.parse_args(argv)
    if args.algo != "supermimic":
        parser.error("hand training implements --algo supermimic")
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that hand training does not implement")
    return args


def build(clip: str, device: str, reward_std_scale: float = 1.0):
    """The scene, kinematics and task of a clip."""
    from ndm_gym.motion.sources.spider import load_reference, load_scene
    from ndm_gym.robots.xhand import HandKinematics
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.hand_tracking import HandTask, RewardCfg, build_reference

    model = load_scene(clip)
    kin = HandKinematics(model, device)
    ref = build_reference(load_reference(clip, model, CONTROL_DT), kin, device)
    return model, kin, HandTask(kin, ref, RewardCfg().scaled(reward_std_scale))


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.hand_algos import HAND_ALGOS, WINDOW_KEYS, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.sim.rigid.hand_sim import HandSim
    from ndm_gym.tasks.hand_tracking import unflatten_state

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path, header = os.path.join(args.log_dir, "train_log.csv"), ",".join(LOG_COLUMNS)
    resume_ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, kin, task = build(args.clip, dev, args.reward_std_scale)
    sim = HandSim(model, task, args.num_envs, dev, stagger=True)
    algo = HAND_ALGOS[args.algo](args, kin, task, dev)
    start_it, wall_offset = 0, 0.0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck)
        start_it, wall_offset = resume_ck["iteration"], resume_ck["wall_time"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    N = args.num_envs
    a_prev = torch.zeros(N, kin.q_lo.shape[0], device=dev)
    ep_len = torch.zeros(N, device=dev)
    acc = {k: torch.zeros((), device=dev) for k in ("r", "steps", "done", "term", "ep_len", "object_err")}

    @torch.no_grad()
    def collect():
        nonlocal a_prev
        s, f = sim.state(), sim.f.clone()
        S = unflatten_state(s)
        a = algo.act(S, a_prev, f, noise=args.action_noise)
        u = task.setpoint(a, f)
        s1, term, done = sim.step(u)
        S1, f1 = unflatten_state(s1), task.ref.next_frame(f)
        fk1 = kin.forward(S1[0])
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done})
        ep_len.add_(1)
        acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
        acc["object_err"] += task.errors(S1, fk1, f1)["object_pos"].sum()
        acc["steps"] += N
        acc["done"] += done.sum()
        acc["term"] += term.sum()
        acc["ep_len"] += ep_len[done].sum()
        ep_len[done] = 0
        a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)

    def save_last(iteration: int):
        ck = {**algo.state_dict(), "iteration": iteration, "wall_time": time.time() - t_start, "sim_steps": iteration * N, "args": vars(args)}
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)

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
        collect()
        agg["wm_loss"].append(algo.wm_update(buffer))
        if it > args.wm_only_iters:
            for k, x in zip(("pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival"), algo.policy_update(buffer), strict=True):
                agg[k].append(x)
        if it % args.log_every == 0:
            mean = lambda xs: float(torch.stack(xs).mean()) if xs else float("nan")  # noqa: E731
            done, now = max(float(acc["done"]), 1.0), time.time()
            row = [it, (now - t_start) / 3600, it * N, *(mean(agg[k]) for k in agg), float(acc["r"] / acc["steps"]),
                   1000 * float(acc["object_err"] / acc["steps"]), float(acc["ep_len"]) / done, float(acc["term"]) / done,
                   args.log_every / (now - t_log), lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            log_f.flush()
            print(" ".join(f"{k}={x:.4g}" for k, x in zip(LOG_COLUMNS, row, strict=True)), flush=True)
            t_log = now
            for xs in agg.values():
                xs.clear()
            for x in acc.values():
                x.zero_()
        if it % args.ckpt_every == 0:
            save_last(it)
        if args.max_hours is not None and time.time() - t_start > args.max_hours * 3600:
            break
    save_last(it)


if __name__ == "__main__":
    main()
