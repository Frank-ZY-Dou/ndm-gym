"""Training loop for MuscleVAE's muscle-actuated character (docs/musclevae_design.md §4): collect in MuscleSim with
MuscleVAE's environment (its resets, reference switches and termination; ball joints as Euler hinges and ODE's damped
step by default, the simulator their released policy tracks in) -> fit the world model on replay windows -> policy update
through imagined rollouts (ndm_gym.algos.muscle_algos). The schedule and the algorithms' settings are LAFAN1's
(ndm_gym.train's arguments): those in action units mapped by tasks.muscle_tracking.ACTION_UNIT, those only LAFAN1 training
implements refused unless left at their defaults (UNSUPPORTED).

Collection: the simulator's control step is a CUDA graph; the policy, the transition's writes into the replay, the
normalizers' updates, the statistics and the masked reset run on the device without a host synchronization per step.
Every step is guarded (algos.muscle_guard): an env whose state after the step is non-finite or beyond the bounds is
terminated and reset, and its transition enters the replay with its next state replaced by its state (``bad``), kept out
of the world model's windows, the imagination's starts and the normalizers. Each log row counts the flagged envs and the
skipped updates; a run stops at a log row, before writing a checkpoint, when they or the parameters are out of bounds
(muscle_guard.check). ``last.pt`` holds what ``--resume`` continues from: the networks with their normalizers, the
optimizers' states and SuperTrack's loss weights. The replay is not saved; a resumed run refills it with
``--warmup_steps`` of collection, as a new run starts, and continues its log. MUSCLE_PROFILE=1 times the phases.

Of ndm_gym.train's speed options and BlinkMimic's, off by default, muscle training implements --fused_adam, --elementwise_3x3,
--combo_kernels, --pi_ema, --pi_opt/--wm_opt muon (--wm_opt also with the RigidFormer, Muon on its blocks' weight matrices as
in LAFAN1 training),
--pi_lr_mult ramped at --lr_decay_by, --no_sync, --update_graphs, --graph_early, --ckpt_light (with --last_every_min,
the full last.pt too, every that many minutes of training), and --collect_graph with --reset_masked: a collection step
with its masked resets as one CUDA graph (MuscleSim.capture: the policy, the control step's body, the replay write, the
normalizers, the statistics, the reset's body); the others are refused (SPEED_UNSUPPORTED).

Run: ``python -m ndm_gym.train_muscles --algo supertrack --st_fail_w 5 --st_balance_w 4 --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ndm_gym.train import add_train_args

CLIPS = ("lafan_jump", "lafan_run", "lafan_walk")
LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "mpjpe_root_mm",
               "ep_len", "term_frac", "action_rate", "it_per_s", "lr", "peak_mem_gb", "nonfinite_envs", "wm_skipped", "pi_skipped")  # fmt: skip
# ndm_gym.train's arguments that muscle training does not implement (LAFAN1's tricks, data, losses, samplers and the G1
# RigidFormer's options); each is refused unless at its default. tests/test_muscle_training.py checks that every argument
# is either implemented or here.
UNSUPPORTED = ("pi_updates", "st_no_root_terms", "st_no_task_regs", "st_substeps", "bptt_clip_k", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window",
               "sm_residual", "st_com", "st_quad_w", "st_fail_margin", "goal_frames", "hist", "rf_rope_detach", "rf_fused_rope", "wm_sample",
               "wm_prio_alpha", "wm_fail_windows", "wm_fail_window", "st_wm_loss", "no_critic", "motion_dir", "clips", "rf_anchors", "rf_anchor_mode",
               "rf_rope_center", "bptt_decay", "motions", "paired", "rf_merge")  # fmt: skip
# --wm_backbone rigidformer: the muscle RigidFormer (worldmodels.muscle.MuscleRigidFormerNet), sized by these; each
# backbone refuses the other's options unless at their defaults
RF_ARGS = ("rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_muscle_tokens", "rf_muscle_dim", "rf_fused_train",
           "rf_fused_features")  # fmt: skip
MLP_ARGS = ("wm_hidden", "wm_act", "wm_no_ln")
# ndm_gym.train's speed options and BlinkMimic's options that muscle training does not implement, refused unless at their
# defaults: LAFAN1's CUDA graphs of compiled parts (--update_graphs captures whole updates), compiled kinematics and goal options, the
# --algo blinkmimic preset, the mirror augmentation (the character's muscles are not mirrored), LAFAN1 SuperMimic's
# semi-implicit world model and foot-contact reward, --lr_decay_ret (LAFAN1's normalized return), and the failure-focused
# and evaluation-guided resets (MuscleSim draws its resets inside its reset graph from MuscleVAE's reset frames of three
# clips: not ported).
SPEED_UNSUPPORTED = ("wm_cudagraphs", "pi_cudagraphs", "critic_cudagraphs", "compile_eager", "goal_table", "pi_encode_once", "preset", "wm_mirror",
                     "wm_structured", "contact_reward", "ref_contact_fix", "lr_decay_ret", "fail_reset", "bin_fix", "eprobe_share", "eprobe_tile",
                     "eprobe_lead_min", "eprobe_lead_max", "eprobe_every", "eprobe_start", "eprobe_win", "eprobe_decay", "eprobe_graph",
                     "eprobe_async")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    from ndm_gym.algos.blinkmimic import add_blinkmimic_args
    from ndm_gym.train import add_speed_args

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_speed_args(parser, rigidformer=False)
    add_blinkmimic_args(parser)
    parser.add_argument("--mv_clips", type=str, nargs="+", default=list(CLIPS), help="MuscleVAE's reference clips (data/musclevae/clips)")
    parser.add_argument("--ball_joints", type=str, default="euler", choices=["euler", "ball"], help="MuscleSim's ball joints (theirs: euler)")
    parser.add_argument("--joint_damping", type=str, default="ode", choices=["ode", "implicit"], help="MuscleSim's joint damping (theirs: ode)")
    parser.add_argument("--muscle_model", type=str, default="released", choices=["released", "paper"])
    parser.add_argument("--rf_muscle_tokens", type=str, default="all", choices=["all", "body"],
                        help="rigidformer: a token per body and per muscle, or per body with the muscles' wrench and pooled encodings")  # fmt: skip
    parser.add_argument("--rf_muscle_dim", type=int, default=32, help="rigidformer, --rf_muscle_tokens body: width of the muscle encodings")
    parser.add_argument("--rf_fused_train", action="store_true",
                        help="rigidformer, --rf_muscle_tokens body, --rf_registers 9: the blocks' bf16 passes as fused Triton kernels")  # fmt: skip
    parser.add_argument("--rf_fused_features", action="store_true",
                        help="rigidformer, --rf_muscle_tokens body: the muscles' lengths, rates and wrench as fused Triton kernels")  # fmt: skip
    parser.add_argument("--st_ref_replay", type=float, default=0.0,
                        help="supertrack: DiffMimic's demonstration replay in the imagined window: a rollout whose bodies are on average "
                        "more than this many metres from the reference (re-anchored to its root) continues from its state (0: off)")  # fmt: skip
    from ndm_gym import configs

    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first
        argv = configs.expand(sys.argv[1:] if argv is None else list(argv), "train_muscles")
    except ValueError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    for name in UNSUPPORTED + SPEED_UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that muscle training does not implement")
    # the speed and BlinkMimic options implemented; --wm_opt muon also with the RigidFormer (Muon on its blocks' matrices,
    # as LAFAN1 training's: algos.common.wm_optimizer)
    if not 0 <= args.pi_ema < 1:
        parser.error(f"--pi_ema must be in [0, 1), got {args.pi_ema}")
    if args.pi_lr_mult <= 0 or args.lr_decay_len < 1 or args.lr_decay_by < 0:
        parser.error("--pi_lr_mult > 0, --lr_decay_len >= 1, --lr_decay_by >= 0")
    if args.lr_decay_by and args.pi_lr_mult == 1:
        parser.error("--lr_decay_by ramps --pi_lr_mult, which is 1")
    if args.no_sync and not args.fused_adam:
        parser.error("--no_sync skips a non-finite update on the device, with the fused Adam: needs --fused_adam")
    if args.no_sync and args.pi_rollout_clip > 0:
        parser.error("--no_sync: not with --pi_rollout_clip (it reads the device on the host)")
    if args.update_graphs and not args.no_sync:
        parser.error("--update_graphs captures the updates written without host synchronization: needs --no_sync")
    if args.graph_early and not args.update_graphs:
        parser.error("--graph_early needs --update_graphs")
    if args.collect_graph != args.reset_masked or (args.collect_graph and not args.graph_early):
        parser.error("--collect_graph and --reset_masked go together (the collection graph holds the resets) and need --graph_early")
    if args.last_every_min is not None and not args.ckpt_light:
        parser.error("--last_every_min needs --ckpt_light (without it, every periodic checkpoint writes the full last.pt)")
    if args.last_every_min is not None and not args.last_every_min > 0:
        parser.error(f"--last_every_min must be > 0, got {args.last_every_min}")
    for name in ("st_late_w", "st_ref_replay"):
        if getattr(args, name) != parser.get_default(name) and args.algo != "supertrack":
            parser.error(f"--{name} is an option of --algo supertrack")
    if args.st_ref_replay < 0:
        parser.error("--st_ref_replay must be >= 0")
    other = RF_ARGS if args.wm_backbone == "mlp" else MLP_ARGS
    for name in other:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} configures the other world-model backbone than --wm_backbone {args.wm_backbone}")
    if args.rf_muscle_tokens == "all" and args.rf_muscle_dim != parser.get_default("rf_muscle_dim"):
        parser.error("--rf_muscle_dim is the width of the muscle encodings of --rf_muscle_tokens body")
    if args.rf_fused_train and (args.rf_muscle_tokens != "body" or args.rf_registers != 9 or args.amp != "bf16"):
        parser.error("--rf_fused_train runs the body tokens' blocks: it needs --rf_muscle_tokens body, --rf_registers 9 (32 tokens) and --amp bf16")
    if args.rf_fused_features and args.rf_muscle_tokens != "body":
        parser.error("--rf_fused_features computes the body tokens' muscle features: it needs --rf_muscle_tokens body")
    return args


def step_stats_fn(ref):
    """The collection's per-step statistics of the reference ``ref`` (main compiles them): per env the reward and the
    root-frame MPJPE of the state after the step (0 where flagged), the action rate (tasks.muscle_tracking.action_rate, the
    G1's units; 0 at an episode's first step) and whether it counts."""
    import torch

    from ndm_gym.tasks import muscle_tracking as mt

    task_cfg = mt.MuscleRewardCfg()  # the logged reward keeps the task's widths

    def step_stats(s1, f, a, a_prev, fresh, bad):
        S1 = mt.unflatten_bodies(s1)
        ok = (~bad).float()
        rate = mt.action_rate(a, torch.where(fresh.unsqueeze(-1), a, a_prev))
        return mt.reward(S1, ref, f, task_cfg, rate) * ok, mt.root_frame_mpjpe(S1, ref, f) * ok, rate, (~fresh).float()

    return step_stats


def collection_step(sim, algo, buffer, stats, acc: dict, a_prev, captured: bool = False):
    """One collection step: the policy's normalizers, its action and noise, MuscleSim's control step, the guard, the
    transition into the replay and the model's normalizers, the statistics added into ``acc``, the previous action
    ``a_prev`` written in place, the ended episodes reset by mask. ``captured`` (--collect_graph, the collection graph's
    body): the control step's and the reset's bodies in place of their own graphs (MuscleSim.step, MuscleSim.reset; the
    same computations) and the replay written at its device-side head."""
    import torch

    from ndm_gym.algos.muscle_guard import flagged
    from ndm_gym.policies.muscle_policy import policy_observation
    from ndm_gym.tasks import muscle_tracking as mt

    v = sim.views[0]
    fresh = sim.just_reset.clone()
    s = mt.body_states(v.body_q, v.body_qd, sim.before, fresh)
    fat = sim.part_fatigue()
    obs, f = policy_observation(mt.unflatten_bodies(s), fat), sim.ref_frame.clone()
    algo.update_pi_norms(obs, f)
    a, _ = algo.act(obs, f)
    a = a + algo.action_noise * torch.randn_like(a)
    ep_step, off = sim.ep_step.int(), sim.off_count.int()
    if captured:
        sim.action.copy_(a)
        sim.step_body()
        terminated, truncated = sim.terminated.clone(), sim.truncated.clone()
    else:
        terminated, truncated = sim.step(a)
    s1, fat1 = mt.body_states(v.body_q, v.body_qd, sim.before, sim.just_reset), sim.part_fatigue()
    bad = flagged(s1, fat1)
    terminated = terminated | bad
    done = terminated | truncated
    tr = {"s": s, "s1": torch.where(bad[:, None, None], s, s1), "fat": fat, "fat1": torch.where(bad[:, None], fat, fat1), "a": a, "t": f,
          "ep_step": ep_step, "off": off, "done": done, "term": terminated, "bad": bad}  # fmt: skip
    buffer.add_device(tr) if captured else buffer.add(tr)
    algo.update_wm_norms(tr)
    r, e, rate, n_rate = stats(tr["s1"], f, a, a_prev, fresh, bad)
    ok = (~bad).float()
    for k, x in (("r", r.sum()), ("mpjpe", e.sum()), ("n_ok", ok.sum()), ("rate", rate.sum()), ("n_rate", n_rate.sum()),
                 ("ep_len", (sim.ep_step * done).sum()), ("done", done.sum()), ("term", terminated.sum()), ("bad", bad.sum())):  # fmt: skip
        acc[k].add_(x)
    a_prev.copy_(a)
    if captured:
        sim.reset_mask.copy_(done)
        sim._reset_body(sim.sample_frames(sim.N))
    else:
        sim.reset(done)


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.blinkmimic import Graphed, PolicyLR, check_resume
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.muscle_algos import ACTION_UNIT, MUSCLE_ALGOS, WINDOW_KEYS, replay_fields
    from ndm_gym.algos.muscle_guard import check
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.robots.musclevae_char import NUM_MUSCLES
    from ndm_gym.sim.rigid import muscle_sim as ms
    from ndm_gym.tasks import muscle_tracking as mt
    from ndm_gym.train import LightCheckpoints, combo_kernels, elementwise_3x3_products, last_due

    if (ms.ERR_THRESHOLD, ms.ERR_LENGTH, ms.MIN_LENGTH, ms.OBS_LIMIT) != (mt.ERR_THRESHOLD, mt.ERR_LENGTH, mt.MIN_LENGTH, mt.OBS_LIMIT):
        raise ValueError("tasks.muscle_tracking's termination constants differ from ndm_gym.sim.rigid.muscle_sim's")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    RunningNorm.no_sync = args.no_sync  # --no_sync: the normalizers' freeze test on the device (the critic's; the others are masked_update's)
    if args.elementwise_3x3:
        elementwise_3x3_products()
    if args.combo_kernels:
        combo_kernels()
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    columns = LOG_COLUMNS + (("replay_frac",) if args.st_ref_replay > 0 else ())  # --st_ref_replay: the replaced fraction of imagined steps
    log_path, header = os.path.join(args.log_dir, "train_log.csv"), ",".join(columns)
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=dev, weights_only=False)
        if "opt_wm" not in resume_ck:
            raise ValueError(f"--resume {args.resume}: no optimizer states; a run's last.pt holds them")
        if resume_ck["args"]["algo"] != args.algo:
            raise ValueError(f"--resume: the checkpoint's algo is {resume_ck['args']['algo']!r}, this run's {args.algo!r}")
        # the optimizers' settings are loaded with their state, and the policy average with the networks
        if resume_ck["args"].get("fused_adam", False) != args.fused_adam or resume_ck["args"].get("update_graphs", False) != args.update_graphs:
            raise ValueError("--resume: --fused_adam and --update_graphs must match the checkpoint's run")
        check_resume(resume_ck["args"], args)
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")
    mapped = {"action_unit": ACTION_UNIT, "action_noise_on_a": ACTION_UNIT * args.action_noise,
              "st_imag_noise_on_a": ACTION_UNIT * args.st_imag_noise}  # fmt: skip
    json.dump({**vars(args), "mapped": mapped}, open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
    print(f"settings in action units, mapped to the muscle targets: {mapped}", flush=True)

    N = args.num_envs
    # --collect_graph: the control step's body runs in the collection's graph (capture_step=False)
    sim = ms.MuscleSim(N, dev, clips=tuple(args.mv_clips), muscle_model=args.muscle_model, random_count=ms.RANDOM_COUNT, seed=args.seed,
                       ball_joints=args.ball_joints, joint_damping=args.joint_damping, capture_step=not args.collect_graph)  # fmt: skip
    ref = mt.reference_from_sim(sim)
    algo = MUSCLE_ALGOS[args.algo](args, ref, dev)
    pi_lr = PolicyLR(args)  # --pi_lr_mult
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck)
        pi_lr.load_state_dict(resume_ck)
        start_it, wall_offset, sim_steps = resume_ck["iteration"], resume_ck["wall_time"], resume_ck["sim_steps"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, N, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    if args.update_graphs:  # sampling bounds on the device (--graph_early: the fill level too; --collect_graph: the write head)
        buffer.use_device_ptr(device_size=args.graph_early)
    modules = {"wm": algo.wm, "pi": algo.pi, "critic": getattr(algo, "critic", None), "bal_wm": getattr(algo, "bal_wm", None),
               "bal_pi": getattr(algo, "bal_pi", None)}  # fmt: skip
    stats_c = torch.compile(step_stats_fn(ref))
    z = lambda: torch.zeros((), device=dev)  # noqa: E731
    acc = {k: z() for k in ("r", "mpjpe", "n_ok", "rate", "n_rate", "ep_len", "done", "term", "bad")}
    # steps: since the last log row; a_prev: the previous action, written in place (the collection graph reads it there)
    state = {"a_prev": torch.zeros(N, NUM_MUSCLES, device=dev), "steps": 0, "sim_steps": sim_steps}
    graph = {"g": None}

    @torch.no_grad()
    def collect(n_steps: int):
        for _ in range(n_steps):
            if not args.collect_graph:
                collection_step(sim, algo, buffer, stats_c, acc, state["a_prev"])
            elif graph["g"] is None:  # the first step is the capture's warm-up (MuscleSim.capture)
                graph["g"] = sim.capture(lambda: collection_step(sim, algo, buffer, stats_c, acc, state["a_prev"], captured=True))
                print("captured one collection step (--collect_graph)", flush=True)
            else:
                graph["g"].replay()
            if args.collect_graph:
                buffer.advance_host()
            state["steps"] += 1
            state["sim_steps"] += N

    collect(args.warmup_steps)
    if args.no_sync:  # the updates without host synchronization; --update_graphs: as CUDA graphs once the replay is full
        if args.update_graphs:
            full = (lambda: True) if args.graph_early else (lambda: buffer.size == buffer.T)  # noqa: E731
            wm_update_ns = Graphed(lambda: algo.wm_update_graphable(buffer), full, "the world-model update")
            pi_update_ns = Graphed(lambda: algo.policy_update_graphable(buffer),
                                   lambda: full() and (algo.pi_avg is None or int(algo.pi_avg.n_averaged) > 0), "the policy update")  # fmt: skip
        else:
            wm_update_ns, pi_update_ns = lambda: algo.wm_update_graphable(buffer), lambda: algo.policy_update_graphable(buffer)  # noqa: E731
    light = LightCheckpoints() if args.ckpt_light else None
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:  # a resumed run continues its log without a second header
        log_f.write(header + "\n")
    t_start, t_log, step_acc = time.time() - wall_offset, time.time(), 0.0
    hours_left = sorted(h for h in args.ckpt_hours if h > wall_offset / 3600)
    last_time = wall_offset  # --last_every_min: the training time (s) of the last full last.pt (or of the start or resume point)
    profile = bool(int(os.environ.get("MUSCLE_PROFILE", "0")))
    prof = {"collect": 0.0, "model": 0.0, "policy": 0.0}

    def tick():
        if profile:
            torch.cuda.synchronize()
        return time.time()

    def save_last(iteration: int) -> dict:
        """The full checkpoint, which --resume continues from, as last.pt (written whole, then renamed); returns it."""
        ck = {**algo.state_dict(), **pi_lr.state_dict(), "iteration": iteration, "wall_time": time.time() - t_start, "sim_steps": state["sim_steps"],
              "args": vars(args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        return ck

    agg = {k: [] for k in ("wm", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival") + columns[len(LOG_COLUMNS) :]}
    for it in range(start_it, args.iterations):
        if args.lr_decay_hours is not None:
            lr = lr_at((time.time() - t_start) / 3600, args.lr, args.lr_decay_hours)
        else:
            lr = lr_at(it, args.lr, args.lr_decay_iters)
        for o in algo.optimizers:
            set_lr(o, lr)
        if args.wm_lr is not None:
            set_lr(algo.opt_wm, lr * args.wm_lr / args.lr)
        if pi_lr.on:
            set_lr(algo.opt_pi, lr * pi_lr.at(it))
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        t0 = tick()
        collect(n)
        t1 = tick()
        for _ in range(args.wm_batches):
            if args.no_sync:  # a skipped update's value is NaN, left out of the log's mean
                loss, ok = wm_update_ns()
                agg["wm"].append(torch.where(ok, loss, float("nan")))
                continue
            loss = algo.wm_update(buffer)
            if loss is not None:  # None: skipped (non-finite gradient), counted in wm_skipped
                agg["wm"].append(loss)
        t2 = tick()
        if it >= args.wm_only_iters:
            if args.no_sync:
                n_avg0 = int(algo.pi_avg.n_averaged) if algo.pi_avg is not None else 0
                *vals, ok = pi_update_ns()  # with the policy average's step
                if algo.pi_avg is not None and int(algo.pi_avg.n_averaged) == n_avg0:
                    algo.pi_avg.n_averaged += 1  # a step without a host read does not advance the host-side counter
                for k, val in zip(algo.graph_keys, vals):
                    agg[k].append(torch.where(ok, val, float("nan")))
            else:
                out = algo.policy_update(buffer)
                algo.update_pi_avg()  # --pi_ema
                if not out.pop("skipped"):
                    for k, val in out.items():
                        agg[k].append(val)
        t3 = tick()
        prof["collect"], prof["model"], prof["policy"] = prof["collect"] + t1 - t0, prof["model"] + t2 - t1, prof["policy"] + t3 - t2
        checked = False
        if (it + 1) % args.log_every == 0:
            if args.no_sync:  # the window's values read at once, the skipped updates' NaN left out
                agg = {k: [v for v in torch.stack(xs).tolist() if v == v] if xs else [] for k, xs in agg.items()}
                short = int(algo.short_starts)
                if short:
                    raise RuntimeError(f"iteration {it + 1}: {short} policy updates drew fewer than --pi_batch valid imagination starts among "
                                       "twice as many candidates (MuscleAgent.starts_nosync); no checkpoint written")  # fmt: skip
            mean = lambda xs: float(sum(float(v) for v in xs) / len(xs)) if xs else None  # noqa: E731  None: no value in the window
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            a_ = {k: float(v) for k, v in zip(acc, torch.stack(list(acc.values())).tolist())}
            div = lambda x, y: x / y if y > 0 else None  # noqa: E731
            n_bad, (wm_skipped, wm_run), (pi_skipped, pi_run) = int(a_["bad"]), algo.wm_skips.take(), algo.pi_skips.take()
            sim_r = div(a_["r"], a_["n_ok"])
            mpjpe = div(a_["mpjpe"], a_["n_ok"])
            row = [it + 1, (time.time() - t_start) / 3600, state["sim_steps"], mean(agg["wm"]), mean(agg["pi_loss"]), mean(agg["v_loss"]),
                   mean(agg["imag_r"]), mean(agg["pi_gradnorm"]), mean(agg["imag_survival"]), None if sim_r is None else sim_r / mt.MAX_STEP_REWARD,
                   None if mpjpe is None else 1e3 * mpjpe, div(a_["ep_len"], a_["done"]), div(a_["term"], a_["done"]), div(a_["rate"], a_["n_rate"]),
                   ips, lr, torch.cuda.max_memory_allocated() / 2**30, n_bad, wm_skipped, pi_skipped]  # fmt: skip
            row += [mean(agg[k]) for k in columns[len(LOG_COLUMNS) :]]
            log_f.write(",".join("" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
            log_f.flush()
            fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
            print(f"it {it + 1} | {row[1]:.3f} h | wm {fmt(row[3], '.4f')} | pi {fmt(row[4], '.4f')} v {fmt(row[5], '.4f')} "
                  f"imag_r {fmt(row[6], '.3f')} surv {fmt(row[8], '.3f')} | sim_r {fmt(row[9], '.3f')} mpjpe {fmt(row[10], '.1f')} mm "
                  f"ep_len {fmt(row[11], '.1f')} "
                  f"term {fmt(row[12], '.2f')} rate {fmt(row[13], '.4f')} | {ips:.2f} it/s, peak {row[16]:.1f} GB | nonfinite {n_bad}, skipped "
                  f"{wm_skipped} model / {pi_skipped} policy", flush=True)  # fmt: skip
            if profile:
                k = args.log_every
                print(f"[prof] per iteration: collect {1e3 * prof['collect'] / k:.0f} ms | model {1e3 * prof['model'] / k:.0f} ms | policy "
                      f"{1e3 * prof['policy'] / k:.0f} ms", flush=True)  # fmt: skip
            prof = {k: 0.0 for k in prof}
            check(it + 1, modules, n_flagged=n_bad, env_steps=state["steps"] * N, skipped={"model": wm_run, "policy": pi_run})
            checked, state["steps"] = True, 0
            msg = pi_lr.observe(it + 1, None)  # --lr_decay_by's deadline
            if msg:
                print(msg, flush=True)
            for k in acc:  # in place: the collection graph adds into these tensors
                acc[k].zero_()
            agg = {k: [] for k in agg}
        trained_h = (time.time() - t_start) / 3600
        out_of_time = args.max_hours is not None and trained_h >= args.max_hours
        marks = [h for h in hours_left if trained_h >= h]
        final_ck = it + 1 == args.iterations or out_of_time
        if (it + 1) % args.ckpt_every == 0 or final_ck or marks:
            if not checked:  # never a checkpoint with a non-finite parameter or normalizer
                check(it + 1, modules)
        if light is not None and not final_ck and ((it + 1) % args.ckpt_every == 0 or marks):  # --ckpt_light: the deployed policy only
            wall = time.time() - t_start
            light.save(algo, it + 1, wall, state["sim_steps"], args)
            if last_due(wall, last_time, args.last_every_min):  # --last_every_min: the full last.pt too
                save_last(it + 1)
                last_time = wall
                print(f"last.pt at it {it + 1} ({wall / 60:.1f} min of training, --last_every_min)", flush=True)
        elif (it + 1) % args.ckpt_every == 0 or final_ck or marks:
            ck = save_last(it + 1)
            small = {k: ck[k] for k in ("wm", "pi", "critic", "iteration", "wall_time", "sim_steps", "args", "saved_at") if k in ck}
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
        for h in marks:
            hours_left.remove(h)
            print(f"checkpoint at {h:g} h of training: it {it + 1}", flush=True)
        if out_of_time:
            print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
            break
    if light is not None:
        light.close()


if __name__ == "__main__":
    main()
