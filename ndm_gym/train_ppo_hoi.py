"""Model-free baseline for HDMI's human-object interaction tasks: HDMI's PPO-ROA (ndm_gym.algos.ppo_hoi) in HDMISim with
HDMITask under HDMI's training conditions, with HDMI's env count, rollout length and update schedule (TRN:116-188), and
the quantities train_hoi logs where they apply, for the wall-clock comparison with the model-based runs.

A step: the policy acts on the normalized observation, the simulator steps, HDMITask gives the observation groups,
rewards and terminations, the normalizer adds the observation; envs that are done (terminated, or truncated at the
clip's end or the episode length) are reset at once, their next observation HDMI's all-zero reset observation and their
residual base zero (ENV:371-389). After ``train_every`` steps the learner updates (PPOROA.update).

Deviations from HDMI's trainer, none of which reaches a loss:
* HDMI's second observation of an episode (step_count 1) is built from command buffers that Isaac Lab still holds from
  the previous episode, as no command update runs at a reset (§0); here from the reset state, as HDMISim and HDMITask
  produce it. Both losses drop these transitions (step_count > 1), so the episode's second action and the normalizer's
  statistics see the difference. The first observation is HDMI's, all zeros.
* HDMI's trainer takes one environment step before training to size its buffers and drops its result (TRN:120-123);
  omitted.
* HDMI's teacher rollout also runs the adaptation module and drops its output (PPO:409-412); omitted.

An engine deviation (docs/hoi_design.md §1): every step is guarded against non-finite or diverged simulator states
(ndm_gym.algos.hoi_guard), inside the compiled task step. A flagged env is terminated (no bootstrap) and reset; its
rewards and observation groups are zeroed before the normalizer adds them. The log counts the flagged envs per row, and a
run stops at a log row, before writing a checkpoint, when they or the parameters are out of bounds (hoi_guard.check).

Efficiency: no host synchronization or host copy in the rollout and the update. The rollout storage is preallocated on
the device; the policy, the normalizer and the episode statistics of a step are compiled, as are HDMITask's step and the
learner's losses; HDMISim runs each control step as one CUDA graph (``--eager_sim``: without), and the learner each
minibatch step (``--no_graphs``: without). HDMISim.reset takes env indices, whose number would need a synchronization
to read: the simulator runs one spare env beyond the N of the learner, reset at every step, whose index pads the done
envs' indices to a fixed count. ``--profile k`` counts kernels, launches and synchronizations of iteration k.

Run: ``python -m ndm_gym.train_ppo_hoi --task move_suitcase --log_dir logs/<run>`` (the teacher), then the student:
``python -m ndm_gym.train_ppo_hoi --task move_suitcase --phase finetune --teacher logs/<run>/ckpt_<it>.pt --log_dir logs/<run2>``
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

from ndm_gym.sim.launcher import add_launcher_args, launch_simulation

# Inductor options for HDMITask's step, the same computations in fewer kernels: a stack of up to 32 tensors (yaw_mat
# stacks 9) as one pointwise kernel instead of one kernel per tensor, independent kernels launched together
TASK_COMPILE_OPTIONS = {"max_pointwise_cat_inputs": 32, "combo_kernels": True}
LOG_COLUMNS = ("iteration", "wall_h", "env_steps", "pi_loss", "v_loss", "entropy", "kl", "clipfrac", "lr", "action_std", "adapt_loss",
               "distill_loss", "sim_r", "sim_r_tracking", "sim_r_object_tracking", "sim_r_loco", "sim_r_feet", "ret", "ret_tracking",
               "ret_object_tracking", "ret_loco", "ret_feet", "ep_len", "term_frac", "success", "it_per_s", "collect_s", "update_s",
               "peak_mem_gb", "nonfinite_envs")  # fmt: skip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, help="one of ndm_gym.tasks.hdmi_tasks.TASKS")
    p.add_argument("--phase", type=str, default="train", choices=["train", "finetune"],
                   help="HDMI's teacher (algo=ppo_roa_train) or its student (algo=ppo_roa_finetune, from --teacher)")  # fmt: skip
    p.add_argument("--teacher", type=str, default=None, help="finetune: the teacher's checkpoint")
    p.add_argument("--resume", type=str, default=None,
                   help="continue a run of this trainer from its checkpoint (the learner's state, the iteration and the training time; "
                   "episodes restart, as when HDMI loads a checkpoint), logging to the same --log_dir")  # fmt: skip
    p.add_argument("--num_envs", type=int, default=4096, help="HDMI's (Y:13)")
    p.add_argument("--train_every", type=int, default=32, help="rollout steps per env and iteration (PPO:43)")
    p.add_argument("--total_frames", type=int, default=150_000_000,
                   help="HDMI's (cfg/train.yaml:15): iterations = total_frames // (num_envs x train_every) (TRN:75-78)")  # fmt: skip
    p.add_argument("--iterations", type=int, default=None, help="instead of --total_frames")
    p.add_argument("--max_hours", type=float, default=None, help="stop with a checkpoint after this much training wall-clock time")
    p.add_argument("--ckpt_every", type=int, default=300, help="HDMI's save_interval (cfg/train.yaml:17)")
    p.add_argument("--ckpt_hours", type=float, nargs="+", default=[],
                   help="also checkpoint when the training wall-clock time first reaches each of these hours (as ndm_gym.train)")  # fmt: skip
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eager_sim", action="store_true", help="HDMISim without its CUDA graph (the same values)")
    p.add_argument("--no_compile", action="store_true", help="no torch.compile (the same computations)")
    p.add_argument("--no_graphs", action="store_true", help="the learner's minibatch steps without CUDA graphs (the same kernels)")
    p.add_argument("--profile", type=int, default=None,
                   help="profile iteration k (1-based): kernels, launches and synchronizations of iteration k, "
                   "synchronized times of its parts in iteration k + 1")  # fmt: skip
    add_launcher_args(p)
    args = p.parse_args()
    if args.resume is not None and args.teacher is not None:
        p.error("--resume continues a run; its checkpoint holds the student's teacher")
    if args.resume is None and (args.phase == "finetune") != (args.teacher is not None):
        p.error("--teacher is needed exactly for --phase finetune")
    return args


class Sections:
    """Context managers around the parts of an iteration: profiler ranges while profiling (``mode`` "profile"),
    synchronized wall-clock times while timing ("time", accumulated in ``times``), nothing otherwise."""

    def __init__(self):
        self.mode, self.times = None, {}

    def __call__(self, name: str):
        import torch

        if self.mode == "profile":
            return torch.profiler.record_function("ppo:" + name)
        return self._timed(name) if self.mode == "time" else contextlib.nullcontext()

    @contextlib.contextmanager
    def _timed(self, name: str):
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0


def profile_counts(fn, sections: Sections) -> dict[str, dict[str, int]]:
    """Counts of ``fn``'s work under torch.profiler: in total the GPU kernels (graph replays' included) and the
    device-host copies; in total and per section (the innermost range around the call) the kernel and graph launches,
    the host synchronizations and the scalar reads."""
    import bisect

    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    sections.mode = "profile"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    sections.mode = None
    calls = {"cudaLaunchKernel": "kernel_launches", "cuLaunchKernel": "kernel_launches", "cudaLaunchKernelExC": "kernel_launches",
             "cuLaunchKernelEx": "kernel_launches", "cudaGraphLaunch": "graph_launches", "cuGraphLaunch": "graph_launches",
             "cudaStreamSynchronize": "syncs", "cudaDeviceSynchronize": "syncs", "cudaEventSynchronize": "syncs", "cuStreamSynchronize": "syncs",
             "cuCtxSynchronize": "syncs", "cuEventSynchronize": "syncs", "aten::_local_scalar_dense": "item"}  # fmt: skip
    events = prof.events()
    ranges = sorted((e for e in events if e.device_type != DeviceType.CUDA and e.name.startswith("ppo:")), key=lambda e: e.time_range.start)
    starts = [r.time_range.start for r in ranges]
    out = {"total": {"gpu_kernels": 0, "memcpy_dtoh": 0, "memcpy_htod": 0}}
    for e in events:
        if e.device_type == DeviceType.CUDA:
            k = "memcpy_dtoh" if "Memcpy DtoH" in e.name else "memcpy_htod" if "Memcpy HtoD" in e.name else (
                None if e.name.startswith(("Memcpy", "Memset")) else "gpu_kernels")  # fmt: skip
            if k is not None:
                out["total"][k] += 1
            continue
        k = calls.get(e.name)
        if k is None:
            continue
        i = bisect.bisect_right(starts, e.time_range.start) - 1
        while i >= 0 and ranges[i].time_range.end < e.time_range.start:  # the innermost range that still contains it
            i -= 1
        for owner in ("total", ranges[i].name[4:] if i >= 0 else "other"):
            d = out.setdefault(owner, {})
            d[k] = d.get(k, 0) + 1
    return out


def main():
    args = parse_args()

    import torch

    from ndm_gym.algos.hoi_guard import NonfiniteMonitor, check, guard_terms, nonfinite, record_layout
    from ndm_gym.algos.ppo_hoi import STD_MIN, PPOConfig, PPOROA, Rollout, residual_base
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim, make_hdmi_sim
    from ndm_gym.tasks.hdmi import GROUPS as TASK_GROUPS
    from ndm_gym.tasks.hdmi import HDMITask
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.wmbench.collect_hoi import merge_states

    torch.backends.cuda.matmul.allow_tf32 = True  # TRN:27-28
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    dev = "cuda:0"
    N, T = args.num_envs, args.train_every
    iterations = args.iterations if args.iterations is not None else args.total_frames // (N * T)
    if iterations < 1:
        raise ValueError(f"{args.total_frames} frames are less than one iteration of {N} x {T}")
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
    cfg = TASKS[args.task]
    teacher_ck = resume_ck = None
    if args.teacher is not None:
        teacher_ck = torch.load(args.teacher, map_location=dev, weights_only=False)
        if teacher_ck["args"]["task"] != args.task:
            raise ValueError(f"the teacher was trained on {teacher_ck['args']['task']}, not {args.task}")
    if args.resume is not None:
        resume_ck = torch.load(args.resume, map_location=dev, weights_only=False)
        for k in ("task", "phase", "num_envs", "train_every"):
            if resume_ck["args"][k] != getattr(args, k):
                raise ValueError(f"--resume: the checkpoint's {k} is {resume_ck['args'][k]!r}, this run's {getattr(args, k)!r}")
    log_path = os.path.join(args.log_dir, "train_log.csv")
    if os.path.exists(log_path):  # a resumed run continues its log with the same columns
        with open(log_path) as f:
            if f.readline().strip() != ",".join(LOG_COLUMNS):
                raise ValueError(f"{log_path} has other columns than this trainer writes; continue in another --log_dir")
    sim_cfg = make_hdmi_sim(dev)
    with launch_simulation(sim_cfg, args):
        # the simulator runs one env more than the learner: a spare env (index N) with action 0 that is reset at every
        # step, whose index pads the done envs' indices to a fixed count (torch.nonzero_static), so that sizing the reset
        # needs no host synchronization; the learner, its normalizer and the statistics see the first N envs only
        Ns = N + 1
        env = HDMISim(sim_cfg, cfg, Ns, graph=not args.eager_sim)
        task = HDMITask(cfg, Ns, env.joint_limits, env.robot_mass, noise=True, device=dev)
        pcfg = PPOConfig(phase=args.phase, train_every=T)
        compile_ = not args.no_compile
        agent = PPOROA(pcfg, task.obs_dims, dev, compile=compile_, graphs=not args.no_graphs)
        if teacher_ck is not None:
            agent.load_teacher(teacher_ck["agent"])
        buf = Rollout(T, N, agent.obs_dim, pcfg.latent_dim, not pcfg.teacher, dev)
        cmp = (lambda f, **kw: torch.compile(f, dynamic=False, **kw)) if compile_ else (lambda f, **kw: f)  # noqa: E731

        def step_task(x, noise, a, t):
            """HDMITask's step down to what PPO reads (the four observation groups, the group rewards, the terminations)
            and the guard of the step (hoi_guard.nonfinite, ``t`` the frame counter after the step); the observation noise
            drawn outside (HDMITask.draw_noise), so that compiled it is eager's random stream."""
            o = task.step(x, noise)
            failed, record = nonfinite(x, o.obs, a, t - 1)
            return o.obs, o.reward, o.terminated, failed.any(-1), failed, record

        task_step = cmp(step_task, options=TASK_COMPILE_OPTIONS)
        q0, scale, clip_q, clip_len = env.q0[env.j_act].clone(), env.scale.clone(), env.clip.joint_pos, env.L
        n_task = len(TASK_GROUPS)

        def post_step(obs_groups, reward, terminated, bad, truncated, t_frame, step_count, ep_ret, acc):
            """The learner's envs' normalized observation, residual base, rewards, termination and done flags, step
            counts and episode statistics: ``acc`` sums the group rewards (n_task), the finished episodes' group returns
            (n_task), their lengths, their number, the terminations, the successes, HDMI's episodes that reach the clip's
            end (CMD:225-227: the frame counter at L - 1 when the rewards are computed, at L after the step), and the envs
            the guard flagged (``bad``), which are terminated with their rewards and observation groups zeroed before the
            normalizer adds them."""
            bad = bad[:N]
            keep = ~bad.unsqueeze(-1)
            obs_groups = {k: torch.where(keep, v[:N], 0.0) for k, v in obs_groups.items()}
            reward, terminated, truncated, t_frame = torch.where(keep, reward[:N], 0.0), terminated[:N] | bad, truncated[:N], t_frame[:N]
            done = terminated | truncated
            obs = agent.observe(obs_groups)
            ref = residual_base(clip_q, q0, scale, t_frame)
            step_count = step_count + 1
            d = done.float()
            ep_ret = ep_ret + reward
            acc = acc + torch.cat((reward.sum(0), (ep_ret * d.unsqueeze(-1)).sum(0), (step_count * d).sum(0, keepdim=True), d.sum(0, keepdim=True),
                                   terminated.float().sum(0, keepdim=True), (d * (t_frame >= clip_len)).sum(0, keepdim=True),
                                   bad.float().sum(0, keepdim=True)))  # fmt: skip
            return obs, ref, reward, terminated, done, step_count, ep_ret * (1.0 - d).unsqueeze(-1), acc

        def reset_carry(done, obs, ref, step_count):
            """The done envs' next observation, HDMI's reset observation (ENV:385-386) after the normalizer added the
            reset's zero rows if any env is done, their residual base zero and their step count 0."""
            m = done.unsqueeze(-1)
            zero_obs = agent.reset_observation(N, done.any())
            return torch.where(m, zero_obs, obs), torch.where(m, 0.0, ref), torch.where(done, 0, step_count)

        post_step, reset_carry = cmp(post_step), cmp(reset_carry)
        env.refresh()
        prev = env.read()
        task.reset(torch.arange(Ns, device=dev), prev)
        if resume_ck is not None:  # HDMI loads the normalizer's statistics over its fake tensordict's (HLP:146-151)
            agent.load_state_dict(resume_ck["agent"])
        elif pcfg.teacher:  # the normalizer's first batch: the environment's fake tensordict, all zeros (HLP:146-147)
            agent.vecnorm.absorb_zeros(N)
        obs = agent.reset_observation(N).expand(N, -1).clone()  # the first reset (TRN:117)
        ref = torch.zeros(N, buf.ref.shape[-1], device=dev)
        step_count = torch.zeros(N, dtype=torch.int32, device=dev)
        ep_ret = torch.zeros(N, n_task, device=dev)
        acc = torch.zeros(2 * n_task + 5, device=dev)
        a_sim = torch.zeros(Ns, buf.act.shape[-1], device=dev)  # the spare env's action stays 0
        done_sim = torch.ones(Ns, dtype=torch.bool, device=dev)  # the spare env is reset at every step
        # the guard's records cover the simulator's envs, its counts and the first event the learner's (hoi_guard)
        x0, obs_like = env.reset_input(prev), {k: torch.zeros(Ns, d, device=dev) for k, d in task.obs_dims.items()}
        monitor = NonfiniteMonitor(args.log_dir, record_layout(x0, obs_like, a_sim), [n for n, _, _ in guard_terms(x0, obs_like)], Ns, dev,
                                   compile=compile_)  # fmt: skip
        del x0, obs_like

        sec = Sections()

        @torch.no_grad()
        def collect():
            nonlocal prev, obs, ref, step_count, ep_ret, acc
            buf.std.copy_(agent.policy_actor.std.detach().clamp_min(STD_MIN))
            for t in range(T):
                with sec("act"):
                    a, logp, loc, pred = agent.act(obs, ref)
                    buf.obs[t].copy_(obs)
                    buf.ref[t].copy_(ref)
                    buf.act[t].copy_(a)
                    buf.logp[t].copy_(logp)
                    buf.loc[t].copy_(loc)
                    buf.step[t].copy_(step_count)
                    if pred is not None:
                        buf.pred[t].copy_(pred)
                    a_sim[:N].copy_(a)
                with sec("sim_step"):
                    x, truncated = env.step(a_sim, prev)
                with sec("task_step"):
                    obs_groups, reward, terminated, bad, failed, record = task_step(x, task.draw_noise(), a_sim, env.t)
                    monitor.step(bad[:N], failed[:N], record)
                with sec("post_step"):
                    outs = post_step(obs_groups, reward, terminated, bad, truncated, env.t, step_count, ep_ret, acc)
                    obs, ref, reward, terminated, done, step_count, ep_ret, acc = outs
                    buf.reward[t, :, :n_task].copy_(reward)  # the debug group's column stays zero
                    buf.term[t].copy_(terminated)
                    buf.done[t].copy_(done)
                with sec("reset_sim"):
                    done_sim[:N].copy_(done)
                    ids = torch.nonzero_static(done_sim, size=Ns, fill_value=N).squeeze(-1)  # the done envs, then the spare env
                    env.reset(ids)
                with sec("reset_refresh"):
                    env.refresh()
                with sec("reset_read"):
                    prev = merge_states(done_sim, x.now, env.read())
                with sec("reset_task"):
                    task.reset(ids, prev)
                with sec("reset_obs"):
                    obs, ref, step_count = reset_carry(done, obs, ref, step_count)

        new_log = not os.path.exists(log_path)
        log_f = open(log_path, "a")
        if new_log:
            log_f.write(",".join(LOG_COLUMNS) + "\n")
        print(f"{args.task}: {N} envs x {T} steps, {iterations} iterations, obs {task.obs_dims}, phase {args.phase}", flush=True)
        start = 0 if resume_ck is None else resume_ck["iteration"]
        t_start = time.time() - (0.0 if resume_ck is None else resume_ck["wall_time"])  # the training time continues
        t_log = time.time()
        hours_left = sorted(h for h in args.ckpt_hours if h > (time.time() - t_start) / 3600)
        events, infos = [], []
        guarded, guarded_tensors, checked = {"nets": agent.nets, "adapt_ema": agent.adapt_ema, "vecnorm": agent.vecnorm}, {"lr": agent.lr_policy}, 0
        for it in range(start, iterations):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            if args.profile == it + 1:  # counts
                cc = profile_counts(collect, sec)
                cu = profile_counts(lambda: infos.append(agent.update(buf, obs, sec)), sec)
                for k, v in cc.items():
                    per_step = ", ".join(f"{n} {c / T:.1f}" for n, c in v.items())
                    print(f"[profile] it {it + 1} collection, {k}: {v} | per step: {per_step}", flush=True)
                for k, v in cu.items():
                    print(f"[profile] it {it + 1} update, {k}: {v}", flush=True)
            elif args.profile is not None and args.profile + 1 == it + 1:  # synchronized times of the parts
                sec.mode, sec.times = "time", {}
                t0 = time.perf_counter()
                collect()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                infos.append(agent.update(buf, obs, sec))
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                sec.mode = None
                print(f"[profile] it {it + 1} synchronized: collection {t1 - t0:.3f} s, update {t2 - t1:.3f} s | "
                      + ", ".join(f"{k} {v:.3f} s" for k, v in sec.times.items()), flush=True)  # fmt: skip
            else:
                ev[0].record()
                collect()
                ev[1].record()
                infos.append(agent.update(buf, obs))
                ev[2].record()
                events.append(ev)
            if (it + 1) % args.log_every == 0 or it + 1 == iterations:
                torch.cuda.synchronize()
                now = time.time()
                ips = len(infos) / (now - t_log)
                t_log = now
                info = {k: torch.stack([i[k] for i in infos]).mean() for k in infos[0]}
                a = acc.tolist()
                steps = len(infos) * T * N
                r, ret = a[:n_task], a[n_task : 2 * n_task]
                eps, lens, term, succ = a[2 * n_task + 1], a[2 * n_task], a[2 * n_task + 2], a[2 * n_task + 3]
                n_bad = int(a[2 * n_task + 4])
                if not monitor.dumped and bool(monitor.captured):  # the run's first flagged env, captured at its step
                    monitor.dump(start + monitor.event_step() // T + 1)
                nan = float("nan")
                row = {"iteration": it + 1, "wall_h": (now - t_start) / 3600, "env_steps": (it + 1) * T * N, "sim_r": sum(r) / steps,
                       "ret": sum(ret) / eps if eps else nan, "ep_len": lens / eps if eps else nan, "term_frac": term / eps if eps else nan,
                       "success": succ / eps if eps else nan,
                       "it_per_s": ips, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                       "collect_s": sum(e[0].elapsed_time(e[1]) for e in events) / 1e3 / max(len(events), 1),
                       "update_s": sum(e[1].elapsed_time(e[2]) for e in events) / 1e3 / max(len(events), 1), "nonfinite_envs": n_bad}  # fmt: skip
                for g, name in enumerate(TASK_GROUPS):
                    row[f"sim_r_{name}"] = r[g] / steps
                    row[f"ret_{name}"] = ret[g] / eps if eps else nan
                for k in ("pi_loss", "v_loss", "entropy", "kl", "clipfrac", "lr", "action_std", "adapt_loss", "distill_loss"):
                    row[k] = float(info[k]) if k in info else nan
                log_f.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in LOG_COLUMNS) + "\n")
                log_f.flush()
                print(f"it {it + 1} | {row['wall_h']:.3f} h | sim_r {row['sim_r']:.4f} ret {row['ret']:.3f} ep_len {row['ep_len']:.1f} "
                      f"term {row['term_frac']:.2f} succ {row['success']:.2f} | pi {row['pi_loss']:.4f} v {row['v_loss']:.4f} kl {row['kl']:.4f} "
                      f"lr {row['lr']:.2e} "
                      f"std {row['action_std']:.3f} adapt {row['adapt_loss']:.4f} | collect {row['collect_s']:.2f} s update {row['update_s']:.2f} s "
                      f"| {ips:.3f} it/s, peak {row['peak_mem_gb']:.1f} GB | nonfinite {n_bad}", flush=True)  # fmt: skip
                acc.zero_()
                events, infos = [], []
                check(it + 1, guarded, guarded_tensors, n_flagged=n_bad, env_steps=steps)
                checked = it + 1
            trained_h = (time.time() - t_start) / 3600
            out_of_time = args.max_hours is not None and trained_h >= args.max_hours
            marks = [h for h in hours_left if trained_h >= h]
            if (it + 1) % args.ckpt_every == 0 or it + 1 == iterations or out_of_time or marks:
                if checked != it + 1:  # never a checkpoint with a non-finite parameter or normalizer
                    check(it + 1, guarded, guarded_tensors)
                ck = {"agent": agent.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start, "env_steps": (it + 1) * T * N,
                      "args": vars(args)}  # fmt: skip
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(ck, path + ".tmp")
                os.replace(path + ".tmp", path)
            for h in marks:
                hours_left.remove(h)
                print(f"checkpoint at {h:g} h of training: it {it + 1}", flush=True)
            if out_of_time:
                print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
                break


if __name__ == "__main__":
    main()
