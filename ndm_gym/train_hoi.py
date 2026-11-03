"""Training loop for HDMI's human-object interaction tasks (docs/hoi_design.md): collect in HDMISim under HDMI's
training conditions -> fit the scene's neural dynamics model on replay windows -> policy update through imagined
rollouts (ndm_gym.algos.hoi_algos). The schedule and the algorithms' settings are LAFAN1's (ndm_gym.train's arguments);
those that only LAFAN1 training implements are refused unless left at their defaults (UNSUPPORTED).

Every collection step is guarded against non-finite or diverged simulator states (ndm_gym.algos.hoi_guard): a flagged
env is terminated and reset, its transition enters the replay sanitized and stays out of the model's windows and the
normalizers. Each log row counts the flagged envs and the skipped updates; a run stops at a log row, before writing a
checkpoint, when they or the parameters are out of bounds (hoi_guard.check). ``last.pt`` holds what ``--resume``
continues from: the networks with their normalizers, the optimizers' states and SuperTrack's loss weights. The replay is
not saved; a resumed run refills it with ``--warmup_steps`` of collection, as a new run starts, and continues its log.

Run: ``python -m ndm_gym.train_hoi --task move_suitcase --algo supermimic --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

from ndm_gym.train import add_train_args

WINDOW_KEYS = ("s", "s1", "o", "o1", "u", "ctx", "f_eef", "f_feet", "done", "bad")
LOG_COLUMNS = ("iteration", "wall_h", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "ep_len", "term_frac",
               "it_per_s", "lr", "peak_mem_gb", "nonfinite_envs", "wm_skipped", "pi_skipped")  # fmt: skip
# ndm_gym.train's arguments that HOI training does not implement (LAFAN1's tricks, data, losses and samplers; rf_fused_rope:
# HOIRigidFormerNet always uses the fused rotation; the G1 RigidFormer's anchor RoPE); each is refused unless at its
# default. tests/test_hoi_training.py checks that every argument is either implemented or here.
UNSUPPORTED = ("st_no_root_terms", "st_no_task_regs", "st_substeps", "st_balance_w", "imag_start_perturb", "bptt_clip_k", "wm_bptt_fp32",
               "imag_fail_starts", "imag_fail_window", "sm_residual", "st_com", "st_quad_w", "st_fail_margin", "goal_frames", "hist",
               "rf_no_tree_bias", "rf_no_rope", "rf_rope_detach", "wm_sample", "wm_prio_alpha", "wm_fail_windows", "wm_fail_window", "st_wm_loss",
               "reward_std_scale", "no_critic", "motion_dir", "rf_fused_rope", "rf_anchors", "rf_anchor_mode", "rf_rope_center",
               "bptt_decay", "motions", "paired", "rf_merge")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    parser.add_argument("--task", type=str, required=True, help="one of ndm_gym.tasks.hdmi_tasks.TASKS")
    parser.add_argument("--st_contact_force", action="store_true",
                        help="supertrack: the contact term is HDMI's contact reward as an error (distance and the predicted force's shortfall), "
                        "not the distance alone")  # fmt: skip
    parser.add_argument("--st_obj_w", type=float, default=1.0,
                        help="supertrack: the object's terms (position, orientation or hinge angle, contact) times this after calibration")  # fmt: skip
    parser.add_argument("--st_obj_imag_noise", type=float, default=0.0,
                        help="supertrack: Gaussian noise (m/s, rad/s) on the imagined object's velocities after every imagined step")  # fmt: skip
    parser.add_argument("--st_vel_terms", action="store_true",
                        help="supertrack: three more terms, the L1 errors of the velocities HDMI's tracking group tracks (its keypoints' linear "
                        "and angular velocities, its tracked joints' velocities), calibrated as the others")  # fmt: skip
    parser.add_argument("--st_task_penalties", action="store_true",
                        help="supertrack: HDMI's loco and feet penalties (survival left out) as a cost at the reward's ratio to its full tracking "
                        "score, as LAFAN1's SuperTrack")  # fmt: skip
    parser.add_argument("--st_ref_replay", type=float, default=0.0,
                        help="supertrack: DiffMimic's demonstration replay in the imagined window: a rollout whose keypoints are on average "
                        "more than this many metres from the reference continues from the reference's state (0: off)")  # fmt: skip
    parser.add_argument("--priv_obs", action="store_true",
                        help="the policy also reads HDMI's privileged observation group, as HDMI's PPO teacher does (both algorithms)")  # fmt: skip
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    st_options = (("st_contact_force", False), ("st_obj_w", 1.0), ("st_obj_imag_noise", 0.0), ("st_vel_terms", False), ("st_task_penalties", False),
                  ("st_late_w", 0.0), ("st_ref_replay", 0.0))  # fmt: skip
    for name, off in st_options:
        if getattr(args, name) != off and args.algo != "supertrack":
            parser.error(f"--{name} is an option of --algo supertrack")
    if args.st_obj_w <= 0 or args.st_obj_imag_noise < 0 or args.st_ref_replay < 0:
        parser.error("--st_obj_w must be > 0, --st_obj_imag_noise and --st_ref_replay >= 0")
    if args.clips:
        parser.error("--clips selects LAFAN1 clips; an HDMI task has its own clip")
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that HOI training does not implement")
    return args


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.common import lr_at
    from ndm_gym.algos.hoi_algos import GUARD_FIELDS, HOI_ALGOS, policy_groups, sanitize_transition, task_fields, task_record
    from ndm_gym.algos.hoi_guard import NonfiniteMonitor, check, guard_terms, nonfinite, record_layout
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim, make_hdmi_sim
    from ndm_gym.tasks.hdmi import NUM_ACTIONS, HDMITask
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.wmbench.collect_hoi import merge_states

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    columns = LOG_COLUMNS + (("replay_frac",) if args.st_ref_replay > 0 else ())  # --st_ref_replay: the replaced fraction of imagined steps
    log_path, header = os.path.join(args.log_dir, "train_log.csv"), ",".join(columns)
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=dev, weights_only=False)
        if "opt_wm" not in resume_ck:
            raise ValueError(f"--resume {args.resume}: no optimizer states; a run's last.pt holds them")
        for k in ("task", "algo"):
            if resume_ck["args"][k] != getattr(args, k):
                raise ValueError(f"--resume: the checkpoint's {k} is {resume_ck['args'][k]!r}, this run's {getattr(args, k)!r}")
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
    cfg = TASKS[args.task]
    sim_cfg = make_hdmi_sim(dev)
    with launch_simulation(sim_cfg, args):
        env = HDMISim(sim_cfg, cfg, args.num_envs)
        task = HDMITask(cfg, args.num_envs, env.joint_limits, env.robot_mass, noise=True, device=dev)
        algo = HOI_ALGOS[args.algo](args, env, dev)
        start_it, wall_offset = 0, 0.0
        if resume_ck is not None:
            algo.load_state_dict(resume_ck)
            start_it, wall_offset = resume_ck["iteration"], resume_ck["wall_time"]
            del resume_ck
        fields = {**env.transition_fields(), **task_fields(len(task.term_names)), **GUARD_FIELDS}
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, fields=fields)
        buffer.window_keys = WINDOW_KEYS
        modules = {"wm": algo.wm, "pi": algo.pi, "critic": getattr(algo, "critic", None), "bal": getattr(algo, "bal", None)}

        # the collection's task step and post-reset observations, compiled (a few fused kernels instead of hundreds of eager
        # ones per control step) to what collect reads: the policy's observation groups, rewards and terminations (HDMI's
        # privileged group and per-term values are left out), and the guard (hoi_guard.nonfinite), which terminates a
        # flagged env and zeroes its rewards; their observation noise is drawn outside (task.draw_noise), so that it is the
        # same random stream as without compilation
        groups = policy_groups(args.priv_obs)

        def step_task(x, noise, a, t):
            o = task.step(x, noise)
            obs = {k: o.obs[k] for k in groups}
            failed, record = nonfinite(x, obs, a, t)
            bad = failed.any(-1)
            return obs, torch.where(bad.unsqueeze(-1), 0.0, o.reward), o.terminated | bad, bad, failed, record

        def reset_obs(x, noise):
            return {k: v for k, v in task.observations(x, noise).items() if k in groups}

        task_step, task_obs, sanitize = torch.compile(step_task), torch.compile(reset_obs), torch.compile(sanitize_transition)
        all_ids = torch.arange(args.num_envs, device=dev)
        prev = env.read()
        task.reset(all_ids, prev)
        x0 = env.reset_input(prev)
        obs = task_obs(x0, task.draw_noise(push=False))
        a0 = torch.zeros(args.num_envs, NUM_ACTIONS, device=dev)
        monitor = NonfiniteMonitor(args.log_dir, record_layout(x0, obs, a0), [n for n, _, _ in guard_terms(x0, obs)], args.num_envs, dev)
        state = {"obs": obs, "prev": prev, "steps": 0}  # steps: collection steps since the last log row
        stats = {k: [] for k in ("r", "ep_len", "term", "done")}
        ep_len = torch.zeros(args.num_envs, device=dev)

        @torch.no_grad()
        def collect(n_steps: int, iteration: int):
            for _ in range(n_steps):
                before, rec, q_prev = env.before_step(), task_record(task), task.joint_pos_priv[:, 0]
                algo.update_pi_norms(state["obs"])
                a = algo.act(state["obs"], env.t)
                a = a + args.action_noise * torch.randn_like(a)
                x, truncated = env.step(a, state["prev"])
                new_obs, reward, terminated, bad, failed, record = task_step(x, task.draw_noise(), a, before["t"])
                monitor.step(bad, failed, record)
                tr = env.transition(before, a, x, terminated, truncated)
                tr.update(rec)
                tr = sanitize(tr, bad, q_prev)
                buffer.add(tr)
                algo.update_wm_norms(tr)
                done = terminated | truncated
                ep_len.add_(1)
                state["steps"] += 1
                stats["r"].append(reward.sum(-1).mean())
                stats["term"].append(terminated.float().sum()), stats["done"].append(done.float().sum())
                state["obs"], state["prev"] = new_obs, x.now
                if monitor.dumped:
                    any_done = bool(done.any())
                else:  # the run's first flagged env is written out at the step's synchronization
                    any_done, event = torch.stack((done.any(), monitor.captured)).tolist()
                    if event:
                        monitor.dump(iteration)
                if any_done:
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    stats["ep_len"].append(ep_len[ids].mean())
                    ep_len[ids] = 0
                    env.reset(ids)
                    env.refresh()
                    state["prev"] = merge_states(done, state["prev"], env.read())
                    task.reset(ids, state["prev"])
                    fresh = task_obs(env.reset_input(state["prev"]), task.draw_noise(push=False))
                    state["obs"] = {k: torch.where(done.view(-1, *([1] * (v.dim() - 1))), fresh[k], v) for k, v in state["obs"].items()}

        collect(args.warmup_steps, start_it)
        log_f = open(log_path, "a")
        if os.path.getsize(log_path) == 0:  # a resumed run continues its log without a second header
            log_f.write(header + "\n")
        t_start, t_log, step_acc = time.time() - wall_offset, time.time(), 0.0
        hours_left = sorted(h for h in args.ckpt_hours if h > wall_offset / 3600)
        profile = bool(int(os.environ.get("HOI_PROFILE", "0")))  # time collection, model and policy updates (synchronized)

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
            collect(n, it + 1)
            t1 = tick()
            for _ in range(args.wm_batches):
                loss = algo.wm_update(buffer)
                if loss is not None:  # None: skipped (non-finite gradient), counted in wm_skipped
                    agg["wm"].append(loss)
            t2 = tick()
            if it >= args.wm_only_iters:
                out = algo.policy_update(buffer)
                if not out.pop("skipped"):
                    for k, v in out.items():
                        agg[k].append(v)
                if profile:
                    t3 = tick()
                    print(f"[prof] collect {1e3 * (t1 - t0):.0f} ms ({n} steps) | model {1e3 * (t2 - t1):.0f} ms | policy {1e3 * (t3 - t2):.0f} ms", flush=True)
            checked = False
            if (it + 1) % args.log_every == 0:
                mean = lambda xs: float(sum(float(v) for v in xs) / len(xs)) if xs else None  # noqa: E731  None: no value in the window
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                n_bad, (wm_skipped, wm_run), (pi_skipped, pi_run) = monitor.take(), algo.wm_skips.take(), algo.pi_skips.take()
                row = [it + 1, (time.time() - t_start) / 3600, mean(agg["wm"]), mean(agg["pi_loss"]), mean(agg["v_loss"]), mean(agg["imag_r"]),
                       mean(agg["pi_gradnorm"]), mean(agg["imag_survival"]), mean(stats["r"]), mean(stats["ep_len"]), term_frac, ips, lr,
                       torch.cuda.max_memory_allocated() / 2**30, n_bad, wm_skipped, pi_skipped]  # fmt: skip
                row += [mean(agg[k]) for k in columns[len(LOG_COLUMNS) :]]
                log_f.write(",".join("" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
                log_f.flush()
                fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
                print(f"it {it + 1} | {row[1]:.3f} h | wm {fmt(row[2], '.4f')} | pi {fmt(row[3], '.4f')} v {fmt(row[4], '.4f')} "
                      f"imag_r {fmt(row[5], '.4f')} surv {fmt(row[7], '.3f')} | sim_r {fmt(row[8], '.4f')} ep_len {fmt(row[9], '.1f')} "
                      f"term {term_frac:.2f} | {ips:.2f} it/s, peak {row[13]:.1f} GB | nonfinite {n_bad}, skipped {wm_skipped} model / "
                      f"{pi_skipped} policy", flush=True)  # fmt: skip
                check(it + 1, modules, n_flagged=n_bad, env_steps=state["steps"] * args.num_envs, skipped={"model": wm_run, "policy": pi_run})
                checked, state["steps"] = True, 0
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            trained_h = (time.time() - t_start) / 3600
            out_of_time = args.max_hours is not None and trained_h >= args.max_hours
            marks = [h for h in hours_left if trained_h >= h]
            if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or out_of_time or marks:
                if not checked:  # never a checkpoint with a non-finite parameter or normalizer
                    check(it + 1, modules)
                ck = {**algo.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start, "args": vars(args),
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {k: ck[k] for k in ("wm", "pi", "critic", "iteration", "wall_time", "args") if k in ck}
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
