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

Run: ``python -m ndm_gym.train_muscles --algo supertrack --st_fail_w 5 --st_balance_w 4 --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import time

from ndm_gym.train import add_train_args

CLIPS = ("lafan_jump", "lafan_run", "lafan_walk")
LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "mpjpe_root_mm",
               "ep_len", "term_frac", "action_rate", "it_per_s", "lr", "peak_mem_gb", "nonfinite_envs", "wm_skipped", "pi_skipped")  # fmt: skip
# ndm_gym.train's arguments that muscle training does not implement (LAFAN1's tricks, data, losses, samplers and the G1
# RigidFormer's options); each is refused unless at its default. tests/test_muscle_training.py checks that every argument
# is either implemented or here.
UNSUPPORTED = ("st_no_root_terms", "st_no_task_regs", "st_substeps", "bptt_clip_k", "wm_bptt_fp32", "imag_fail_starts", "imag_fail_window",
               "sm_residual", "st_com", "st_quad_w", "st_fail_margin", "goal_frames", "hist", "rf_rope_detach", "rf_fused_rope", "wm_sample",
               "wm_prio_alpha", "wm_fail_windows", "wm_fail_window", "st_wm_loss", "no_critic", "motion_dir", "clips", "rf_anchors", "rf_anchor_mode",
               "rf_rope_center", "bptt_decay", "motions", "paired", "rf_merge")  # fmt: skip
# --wm_backbone rigidformer: the muscle RigidFormer (worldmodels.muscle.MuscleRigidFormerNet), sized by these; each
# backbone refuses the other's options unless at their defaults
RF_ARGS = ("rf_dim", "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "rf_muscle_tokens", "rf_muscle_dim", "rf_fused_train",
           "rf_fused_features")  # fmt: skip
MLP_ARGS = ("wm_hidden", "wm_act", "wm_no_ln")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    add_train_args(parser)
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
    args = parser.parse_args(argv)
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that muscle training does not implement")
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


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.common import lr_at
    from ndm_gym.algos.muscle_algos import ACTION_UNIT, MUSCLE_ALGOS, WINDOW_KEYS, replay_fields
    from ndm_gym.algos.muscle_guard import check, flagged
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.policies.muscle_policy import policy_observation
    from ndm_gym.robots.musclevae_char import NUM_MUSCLES
    from ndm_gym.sim.rigid import muscle_sim as ms
    from ndm_gym.tasks import muscle_tracking as mt

    if (ms.ERR_THRESHOLD, ms.ERR_LENGTH, ms.MIN_LENGTH, ms.OBS_LIMIT) != (mt.ERR_THRESHOLD, mt.ERR_LENGTH, mt.MIN_LENGTH, mt.OBS_LIMIT):
        raise ValueError("tasks.muscle_tracking's termination constants differ from ndm_gym.sim.rigid.muscle_sim's")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
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
    sim = ms.MuscleSim(N, dev, clips=tuple(args.mv_clips), muscle_model=args.muscle_model, random_count=ms.RANDOM_COUNT, seed=args.seed,
                       ball_joints=args.ball_joints, joint_damping=args.joint_damping)  # fmt: skip
    ref = mt.reference_from_sim(sim)
    algo = MUSCLE_ALGOS[args.algo](args, ref, dev)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck)
        start_it, wall_offset, sim_steps = resume_ck["iteration"], resume_ck["wall_time"], resume_ck["sim_steps"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, N, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    modules = {"wm": algo.wm, "pi": algo.pi, "critic": getattr(algo, "critic", None), "bal_wm": getattr(algo, "bal_wm", None),
               "bal_pi": getattr(algo, "bal_pi", None)}  # fmt: skip
    task_cfg = mt.MuscleRewardCfg()  # the logged reward keeps the task's widths

    def step_stats(s1, f, a, a_prev, fresh, bad):
        """Per env: the reward and the root-frame MPJPE of the state after the step (0 where flagged), the action rate
        (tasks.muscle_tracking.action_rate, the G1's units; 0 at an episode's first step)."""
        S1 = mt.unflatten_bodies(s1)
        ok = (~bad).float()
        rate = mt.action_rate(a, torch.where(fresh.unsqueeze(-1), a, a_prev))
        return mt.reward(S1, ref, f, task_cfg, rate) * ok, mt.root_frame_mpjpe(S1, ref, f) * ok, rate, (~fresh).float()

    stats_c = torch.compile(step_stats)
    z = lambda: torch.zeros((), device=dev)  # noqa: E731
    acc = {k: z() for k in ("r", "mpjpe", "n_ok", "rate", "n_rate", "ep_len", "done", "term", "bad")}
    state = {"a_prev": torch.zeros(N, NUM_MUSCLES, device=dev), "steps": 0, "sim_steps": sim_steps}  # steps: since the last log row

    @torch.no_grad()
    def collect(n_steps: int):
        v = sim.views[0]
        for _ in range(n_steps):
            fresh = sim.just_reset.clone()
            s = mt.body_states(v.body_q, v.body_qd, sim.before, fresh)
            fat = sim.part_fatigue()
            obs, f = policy_observation(mt.unflatten_bodies(s), fat), sim.ref_frame.clone()
            algo.update_pi_norms(obs, f)
            a, _ = algo.act(obs, f)
            a = a + algo.action_noise * torch.randn_like(a)
            ep_step, off = sim.ep_step.int(), sim.off_count.int()
            terminated, truncated = sim.step(a)
            s1, fat1 = mt.body_states(v.body_q, v.body_qd, sim.before, sim.just_reset), sim.part_fatigue()
            bad = flagged(s1, fat1)
            terminated = terminated | bad
            done = terminated | truncated
            tr = {"s": s, "s1": torch.where(bad[:, None, None], s, s1), "fat": fat, "fat1": torch.where(bad[:, None], fat, fat1), "a": a, "t": f,
                  "ep_step": ep_step, "off": off, "done": done, "term": terminated, "bad": bad}  # fmt: skip
            buffer.add(tr)
            algo.update_wm_norms(tr)
            r, e, rate, n_rate = stats_c(tr["s1"], f, a, state["a_prev"], fresh, bad)
            ok = (~bad).float()
            for k, x in (("r", r.sum()), ("mpjpe", e.sum()), ("n_ok", ok.sum()), ("rate", rate.sum()), ("n_rate", n_rate.sum()),
                         ("ep_len", (sim.ep_step * done).sum()), ("done", done.sum()), ("term", terminated.sum()), ("bad", bad.sum())):  # fmt: skip
                acc[k] += x
            state["a_prev"] = a
            sim.reset(done)
            state["steps"] += 1
            state["sim_steps"] += N

    collect(args.warmup_steps)
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:  # a resumed run continues its log without a second header
        log_f.write(header + "\n")
    t_start, t_log, step_acc = time.time() - wall_offset, time.time(), 0.0
    hours_left = sorted(h for h in args.ckpt_hours if h > wall_offset / 3600)
    profile = bool(int(os.environ.get("MUSCLE_PROFILE", "0")))
    prof = {"collect": 0.0, "model": 0.0, "policy": 0.0}

    def tick():
        if profile:
            torch.cuda.synchronize()
        return time.time()

    agg = {k: [] for k in ("wm", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival") + columns[len(LOG_COLUMNS) :]}
    for it in range(start_it, args.iterations):
        if args.lr_decay_hours is not None:
            lr = lr_at((time.time() - t_start) / 3600, args.lr, args.lr_decay_hours)
        else:
            lr = lr_at(it, args.lr, args.lr_decay_iters)
        for o in algo.optimizers:
            for group in o.param_groups:
                group["lr"] = lr
        if args.wm_lr is not None:
            for group in algo.opt_wm.param_groups:
                group["lr"] = lr * args.wm_lr / args.lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        t0 = tick()
        collect(n)
        t1 = tick()
        for _ in range(args.wm_batches):
            loss = algo.wm_update(buffer)
            if loss is not None:  # None: skipped (non-finite gradient), counted in wm_skipped
                agg["wm"].append(loss)
        t2 = tick()
        if it >= args.wm_only_iters:
            out = algo.policy_update(buffer)
            if not out.pop("skipped"):
                for k, val in out.items():
                    agg[k].append(val)
        t3 = tick()
        prof["collect"], prof["model"], prof["policy"] = prof["collect"] + t1 - t0, prof["model"] + t2 - t1, prof["policy"] + t3 - t2
        checked = False
        if (it + 1) % args.log_every == 0:
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
            for k in acc:
                acc[k] = z()
            agg = {k: [] for k in agg}
        trained_h = (time.time() - t_start) / 3600
        out_of_time = args.max_hours is not None and trained_h >= args.max_hours
        marks = [h for h in hours_left if trained_h >= h]
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or out_of_time or marks:
            if not checked:  # never a checkpoint with a non-finite parameter or normalizer
                check(it + 1, modules)
            ck = {**algo.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start, "sim_steps": state["sim_steps"], "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
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


if __name__ == "__main__":
    main()
