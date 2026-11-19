"""Evaluate a quadruped tracking policy (train_quadruped's last.pt), or the reference pose alone as the position targets,
on every clip in the simulator.

    python -m ndm_gym.evaluate_quadruped --ckpt logs/<name>/last.pt --out logs/<name>/evaluation
    python -m ndm_gym.evaluate_quadruped --open_loop --clips dog_quad_walk_002_1 --out logs/open_loop
    python -m ndm_gym.evaluate_quadruped --ckpts logs/<name> --out logs/<name>/evaluation

Each clip is rolled out ``--num_envs`` times (256; with many clips at most ``--max_worlds`` worlds in all, shared evenly),
each in a world of its own, from its first frame with the training's perturbation of the start (the joints and the
velocities; evaluation seed ``--seed``), without exploration noise, to its last frame. The clips are rolled out in
groups of similar length where that is cheaper (``length_buckets``), a simulator per group, so that the short clips'
worlds do not run on to the longest clip's end; the policies evaluated together (the checkpoints of ``--ckpts``, the
policy, the reference targets and the pushes of ``--ckpt``) share each group's simulator, up to ``--batch_worlds``
worlds in it (``rollout_sums``): a control step costs about as much for few worlds as for many. A rollout succeeds if it never
terminates (tasks.quadruped_tracking's termination); a clip's ``success`` is that of all its rollouts, ``success_rate``
their share, and the summary's ``success_rate`` the mean of the clips'. The errors are means over the rollouts and the
frames before a rollout's termination, if any: the trunk's position (mm) and rotation (rad), the feet in the world and
about the trunk in its heading frame (mm), the joints (rad and degrees, RMS).

The return and the failed clips are ndm_gym.evaluate's (the G1's tracking), over several rollouts of a clip in place of
one: ``norm_return`` is a rollout's undiscounted reward sum, the steps after a termination counting nothing, over the
most the clip's steps can give (the reward's positive weights), its mean over a clip's rollouts and then over the clips
(``norm_return_framew``: over all steps alike); ``failed_clips`` is the mean number of clips failed by a pass of one
rollout per clip (the sum of the clips' failed shares), ``failed`` the clips with a success rate below ``--clip_success``
(1: any rollout failed) and ``n_failed`` their number. ``--push <m/s> [<m/s> ...]``: the policy's rollouts again with a
horizontal velocity of that size added to the trunk, in a random direction, every ``--push_every`` seconds (2; seed
``--push_seed``, the same pushes for every policy), as ndm_gym.evaluate's root pushes; ``push/<m/s>`` in metrics.json.

The same rollouts with the reference pose as the
targets in the policy's place (``reference_targets``, the open-loop baseline) are evaluated beside them. Writes
``metrics.json`` (per clip and over all, frames weighted alike) and with ``--save_rollout`` ``rollout.npz``, of one more
rollout per clip from the unperturbed start (per clip the scene's qpos at every control step, the reference's, and the
frame of the termination or -1: ``<clip>/qpos``, ``<clip>/reference_qpos``, ``<clip>/reference_targets_qpos``,
``<clip>/terminated_at``).

``--ckpts <run directory>``: the same evaluation (the policy only) of every ``ckpt_<iteration>.pt`` that train_quadruped
``--keep_ckpts`` kept, from the same perturbed starts, with at most ``--ckpt_worlds`` worlds in all (2048: a clip's 256
rollouts with up to 8 clips, 31 each with 65); writes ``checkpoints.json``: per checkpoint its iteration, the run's
wall time there (minutes) and the summary, and ``minutes_to_success``: the wall time of the first checkpoint from which
every later one has a success rate of ``--success`` or more and, with ``--success_trunk_mm``, the trunk within that
distance on average (null: none).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time

ERROR_KEYS = ("trunk_mm", "trunk_ori_rad", "feet_world_mm", "feet_local_mm", "joint_rad", "joint_deg")


def max_step_reward(cfg) -> float:
    """The most a control step's reward can be: its tracking terms' weights, each kernel at 1."""
    from ndm_gym.state import CONTROL_DT

    return (cfg.w_feet + cfg.w_joint + cfg.w_trunk_pos + cfg.w_trunk_ori + cfg.w_lin_vel + cfg.w_ang_vel) * CONTROL_DT


def length_buckets(lengths: list[int], per_clip: int = 1, overhead: int = 9000) -> list[list[int]]:
    """The clips' indices in groups of consecutive lengths, shortest first, each group rolled out in a simulator of its
    own to its longest clip, so that the short clips' worlds do not run on to the longest clip's end. The grouping is
    the one of least cost, a group's being its longest clip's steps times (``overhead`` + its worlds, ``per_clip`` per
    clip): a control step costs about as much as ``overhead`` worlds whatever the number of worlds (measured: about
    19 ms and 2 microseconds a world), so few worlds are not worth a group of their own."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    best = [(0.0, 0)] + [(float("inf"), 0)] * len(order)  # best[j]: the least cost of the j shortest clips, its last group's start
    for j in range(1, len(order) + 1):
        best[j] = min((best[i][0] + lengths[order[j - 1]] * (overhead + per_clip * (j - i)), i) for i in range(j))
    groups, j = [], len(order)
    while j > 0:
        groups.insert(0, order[best[j][1] : j])
        j = best[j][1]
    return groups


def rollout(sim, task, algo=None, noise: bool = False, push: tuple | None = None, clips=None, keep_qpos: bool = True):
    """Every world (world i on clip i modulo the clips; ``clips``: on the i modulo their number-th of these clip indices,
    a tensor) from its clip's first frame to the last frame of the longest of them, under the policy or
    (``algo`` None) the reference pose as the position targets; ``noise``: the start perturbed as training's. Returns the
    scene's qpos (T, N, nq; None without ``keep_qpos``), the per-step errors, the step each world terminated at (-1: never) and the per-step rewards
    (T, N). ``push``: (m/s, control steps between pushes, a generator): the trunk's horizontal velocity kicked by that much
    in a random direction at every such step but the first."""
    import torch

    unflatten_state = task.unflatten
    ref, N, dev = task.ref, sim.N, sim.device
    clip = torch.arange(N, device=dev) % ref.num_clips if clips is None else clips.to(dev)[torch.arange(N, device=dev) % len(clips)]
    sim.reset(torch.ones(N, dtype=torch.bool, device=dev), frames=ref.clip_first[clip], noise=noise)
    steps = int((ref.clip_last - ref.clip_first)[clip].max())
    a_prev = torch.zeros(N, task.kin.q_lo.shape[0], device=dev)
    term_at = torch.full((N,), -1, dtype=torch.long, device=dev)
    qpos, errs, rew = [sim.qpos.clone()] if keep_qpos else [], [], []
    for t in range(steps):
        if push is not None and t > 0 and t % push[1] == 0:
            ang = 2 * torch.pi * torch.rand(N, generator=push[2], device=dev)
            sim.qvel[:, 0] += push[0] * torch.cos(ang)  # the free joint's linear velocity is in the world
            sim.qvel[:, 1] += push[0] * torch.sin(ang)
        f = sim.f.clone()
        S = unflatten_state(sim.state())
        a = algo.act(S, a_prev, f) if algo is not None else torch.zeros_like(a_prev)
        s1, term, _ = sim.step(task.setpoint(a, f), reset=False)
        S1 = unflatten_state(s1)
        fk = task.kin.forward(S1[0])
        errs.append(task.errors(S1, fk, sim.f))
        rew.append(task.reward(S1, fk, sim.f, a, a_prev))
        term_at = torch.where(term & (term_at < 0) & (f < ref.last[f]), t + 1, term_at)
        a_prev = a
        if keep_qpos:
            qpos.append(sim.qpos.clone())
    return torch.stack(qpos) if keep_qpos else None, {k: torch.stack([e[k] for e in errs]) for k in errs[0]}, term_at, torch.stack(rew)


def step_metrics(errs: dict) -> dict:
    """The reported errors of tasks.quadruped_tracking's ``errors`` (any leading shape): the trunk's position (mm) and
    rotation (rad), the feet in the world and about the trunk (mm, their mean), the joints (rad and degrees)."""
    import torch

    return {"trunk_mm": 1000 * errs["trunk_pos"], "trunk_ori_rad": errs["trunk_ori_sq"].sqrt(), "feet_world_mm": 1000 * errs["feet_world"].mean(-1),
            "feet_local_mm": 1000 * errs["feet"].mean(-1), "joint_rad": errs["joint"], "joint_deg": torch.rad2deg(errs["joint"])}  # fmt: skip


def task_metrics(task):
    """The reported errors of a step: ``step_metrics`` for the Go2's task, a task's own ``step_metrics`` (a humanoid's)."""
    return getattr(task, "step_metrics", step_metrics)


def rollout_sums(sim, task, pis: list, goal_frames: int, noise: bool = False, push: tuple | None = None, clips=None, segments=None,
                 compiled: bool = False, graph: bool = False):
    """``rollout`` of several policies at once, reduced per world as ``summarize_sums`` reads it. The worlds are
    ``len(pis)`` blocks of ``sim.N / len(pis)``, block ``k`` under ``pis[k]`` (a policy network, None: the reference pose
    as the targets), each laid out as ``rollout``'s (``clips``) and started from the same perturbed starts, those of
    ``rollout`` with one block. ``push``: ``rollout``'s, its size per block (a list), the same directions in every block.
    A step's physics costs about the same for few worlds as for many, so the blocks share it. Returns per world the sums
    of ``step_metrics`` and of the reward over the steps before a termination (that step included), and the step of the
    termination (-1: never).

    ``segments``: (first frame, steps) per world of a block (tensors) in place of the clips' whole lengths (evaluate's
    --window_s). The loop runs on Warp's stream without a host synchronization per step (QuadrupedSim.step_nosync
    without its resets: the same kernels as ``step``), computes the forward kinematics of a state once (the termination's
    serve the errors, the reward and the next step's policy input: a push changes velocities only) and ends once every
    world has terminated or passed its last step (checked every 100 steps). ``compiled``: the policy inputs and the step's
    errors and reward as compiled functions (equal up to rounding). ``graph``: the step (policies, physics, termination,
    errors and sums) captured as one CUDA graph at the third step and replayed (the pushes between replays)."""
    import torch

    unflatten_state, step_metrics = task.unflatten, task_metrics(task)
    ref, N, dev, K = task.ref, sim.N, sim.device, len(pis)
    if N % K:
        raise ValueError(f"{N} worlds in {K} blocks")
    B = N // K
    if segments is None:
        clip = torch.arange(N, device=dev) % ref.num_clips if clips is None else clips.to(dev)[torch.arange(N, device=dev) % len(clips)]
        first_b, length = ref.clip_first[clip[:B]], (ref.clip_last - ref.clip_first)[clip]
    else:
        first_b, length = segments[0].to(dev), segments[1].to(dev).repeat(K)
    first = torch.zeros(N, dtype=torch.bool, device=dev)
    first[:B] = True
    sim.reset(first, frames=first_b, noise=noise)  # the first block's starts, drawn as rollout draws them
    for x in (sim.qpos, sim.qvel, sim.ctrl, sim.warmstart, sim.f):
        x[B:] = x[:B].repeat((K - 1,) + (1,) * (x.dim() - 1))
    steps = int(length.max())
    mag = None if push is None else torch.tensor(push[0], dtype=torch.float32, device=dev).reshape(-1).expand(K).repeat_interleave(B)
    a_prev = torch.zeros(N, task.kin.q_lo.shape[0], device=dev)
    term_at = torch.full((N,), -1, dtype=torch.long, device=dev)
    ret, tt = torch.zeros(N, device=dev), torch.zeros((), dtype=torch.long, device=dev)  # tt: the step's index on the device
    fk_fn, inputs, after = _step_functions(task, goal_frames, step_metrics, compiled)
    stream = sim.torch_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fk = [x.clone() for x in fk_fn(sim.qpos.clone())]  # the forward kinematics of the current state, kept in place
        sums = {k: torch.zeros_like(v) for k, v in step_metrics(task.errors(unflatten_state(sim.state()), tuple(fk), sim.f)).items()}

    def body(physics=None):
        """One step, its results written in place (the same computations as a step of ``rollout``)."""
        cache = {}

        def fk_cached(qpos):
            cache["fk"] = fk_fn(qpos)
            return cache["fk"]

        f = sim.f.clone()
        S = unflatten_state(sim.state())
        a = torch.zeros_like(a_prev)
        if any(pi is not None for pi in pis):
            obs, goal = inputs(S, tuple(fk), a_prev, f)
            for k, pi in enumerate(pis):
                if pi is not None:
                    a[k * B : (k + 1) * B] = pi(obs[k * B : (k + 1) * B], goal[k * B : (k + 1) * B])
        s1, term, _ = sim.step_nosync(task.setpoint(a, f), physics=physics, fk=fk_cached, reset=False)
        for x, y in zip(fk, cache["fk"], strict=True):  # the termination's forward kinematics of the next state
            x.copy_(y)
        term_at.copy_(torch.where(term & (term_at < 0) & (f < ref.last[f]), tt + 1, term_at))
        valid = (((term_at < 0) | (term_at == tt + 1)) & (tt + 1 <= length)).float()  # summarize's steps of a world
        m, r = after(unflatten_state(s1), tuple(fk), sim.f, a, a_prev)
        for k, v in m.items():
            sums[k].add_(v * valid)
        ret.add_(r * valid)
        a_prev.copy_(a)
        tt.add_(1)

    g, t0 = None, time.time()
    with torch.cuda.stream(stream):
        for t in range(steps):
            if t > 0 and t % 100 == 0 and bool(((term_at >= 0) | (length <= t)).all()):
                if push is not None:  # the remaining steps' pushes are drawn, so that the next rollouts' do not depend on this end
                    for _ in range(sum(u % push[1] == 0 for u in range(t, steps))):
                        torch.rand(B, generator=push[2], device=dev)
                break  # every world has terminated or passed its last step: the rest adds nothing
            if push is not None and t > 0 and t % push[1] == 0:
                ang = (2 * torch.pi * torch.rand(B, generator=push[2], device=dev)).repeat(K)
                sim.qvel[:, 0] += mag * torch.cos(ang)
                sim.qvel[:, 1] += mag * torch.sin(ang)
            if not graph or t < 2:  # a graph is captured at the third step (kernels loaded, workspaces allocated before)
                body()
                continue
            if g is None:
                g = _capture(sim, stream, lambda: body(sim.physics))
            g[0].replay()
    torch.cuda.current_stream().wait_stream(stream)
    n = int(tt)
    print(f"rollout_sums: {N} worlds, {n} steps, {1e3 * (time.time() - t0) / max(n, 1):.1f} ms per step", flush=True)
    return sums, term_at, ret


def _capture(sim, stream, fn):
    """``fn`` (a step with the physics as kernels) as a CUDA graph: torch's capture on Warp's stream with Warp's registered
    as external, as the training's collection step (algos.hand_fast.Collector.capture). Returns the graph and Warp's record
    of the capture (kept alive with it); the capture does not run ``fn``'s work, a replay does."""
    import torch
    import warp as wp

    wstream = wp.get_stream(sim.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        wp.capture_begin(stream=wstream, external=True, capture_mode=wp.CaptureMode.THREAD_LOCAL)
        try:
            fn()
        finally:
            record = wp.capture_end(stream=wstream)
    return graph, record


def _step_functions(task, goal_frames: int, step_metrics, compiled: bool):
    """rollout_sums' forward kinematics, policy inputs and per-step metrics and reward, compiled with ``compiled``."""
    import torch

    def inputs(S, fk, a_prev, f):
        return task.obs(S, fk, a_prev), task.goal(S, fk, f, goal_frames)

    def after(S1, fk, f, a, a_prev):
        return step_metrics(task.errors(S1, fk, f)), task.reward(S1, fk, f, a, a_prev)

    fns = (task.kin.forward, inputs, after)
    return tuple(torch.compile(fn, dynamic=False) for fn in fns) if compiled else fns


def summarize(task, errs: dict, term_at, clip_names: list[str], rew=None, clip_success: float = 1.0) -> dict:
    """Per clip (over its worlds: world i is of clip i modulo the clips) and over all: success, survival, the mean errors
    over the frames tracked before a termination and, with the per-step rewards ``rew`` (T, N), the normalized return;
    the failed clips (the module's docstring; ``clip_success``: the success rate below which a clip is listed)."""
    import torch

    length = (task.ref.clip_last - task.ref.clip_first)[torch.arange(term_at.shape[0], device=term_at.device) % len(clip_names)]
    T = next(iter(errs.values())).shape[0]
    steps = torch.arange(1, T + 1, device=length.device).unsqueeze(-1)
    end = torch.where(term_at >= 0, term_at, length)
    valid = (steps <= end).float()  # (T, N): the steps of each world's clip, up to its termination
    sums = {k: (v * valid).sum(0) for k, v in task_metrics(task)(errs).items()}
    return summarize_sums(task, sums, term_at, clip_names, None if rew is None else (rew * valid).sum(0), clip_success)


def summarize_sums(task, sums: dict, term_at, clip_names: list[str], ret=None, clip_success: float = 1.0) -> dict:
    """``summarize`` from per-world sums over the steps before a termination (that step included): of ``step_metrics``
    (``sums``) and of the reward (``ret``)."""
    import torch

    ref, C = task.ref, len(clip_names)
    N = term_at.shape[0]
    clip = torch.arange(N, device=term_at.device) % C
    length = (ref.clip_last - ref.clip_first)[clip]  # (N,)
    end = torch.where(term_at >= 0, term_at, length)  # the steps summed per world
    ret = None if ret is None else ret / (length * task.max_step_reward)  # (N,), a terminated rollout's later steps: 0
    clips = {}
    for i, name in enumerate(clip_names):
        m = clip == i
        n = float(end[m].sum().clamp(min=1))
        rate = float((term_at[m] < 0).float().mean())
        clips[name] = {"frames": int(length[m][0]), "rollouts": int(m.sum()), "success": rate == 1.0, "success_rate": rate,
                       "survived_frames": float(end[m].float().mean()), **{k: float(v[m].sum() / n) for k, v in sums.items()}}  # fmt: skip
        if ret is not None:
            clips[name]["norm_return"] = float(ret[m].mean())
    total = float(end.sum().clamp(min=1))
    failed = [n for n, c in clips.items() if c["success_rate"] < clip_success]
    out = {"success_rate": sum(c["success_rate"] for c in clips.values()) / len(clips),
           "survived_share": float(end.sum() / length.sum()),
           **{k: float(v.sum() / total) for k, v in sums.items()}}  # fmt: skip
    if ret is not None:
        out["norm_return"] = sum(c["norm_return"] for c in clips.values()) / len(clips)
        out["norm_return_framew"] = float((ret * length).sum() / length.sum())
    return {**out, "failed_clips": sum(1 - c["success_rate"] for c in clips.values()), "n_failed": len(failed), "failed": failed, "clips": clips}


def minutes_to_success(rows: list[dict], success: float, trunk_mm: float | None = None, norm_return: float | None = None) -> float | None:
    """The wall time (minutes) of the first checkpoint of ``rows`` (in training order) from which every later one has a
    success rate of ``success`` or more (and, with ``trunk_mm``, a mean trunk error of at most that; with ``norm_return``,
    a normalized return of at least that); None if the last one has not."""
    first = None
    for r in rows:
        if r["success_rate"] >= success and (trunk_mm is None or r["trunk_mm"] <= trunk_mm) and (norm_return is None or r["norm_return"] >= norm_return):
            first = r["wall_min"] if first is None else first
        else:
            first = None
    return first


def window_segments(ref, steps: int):
    """The clips cut into windows of ``steps`` control steps (a clip's last one shorter): per window its first frame, its
    steps and its clip, (W,) each."""
    import torch

    first, length, clip = [], [], []
    for c, (a, b) in enumerate(zip(ref.clip_first.tolist(), ref.clip_last.tolist(), strict=True)):
        for s in range(a, b, steps):
            first.append(s), length.append(min(steps, b - s)), clip.append(c)
    return torch.tensor(first), torch.tensor(length), torch.tensor(clip)


def evaluate_windows(task, make_sim, pis: list, goal_frames: int, args, targs) -> list[dict]:
    """--window_s: per policy of ``pis`` (None: the reference targets) the rollouts of every window of the clips
    (``window_segments``), ``--num_envs`` per window (fewer where --batch_worlds takes not all the policies' worlds at
    once), from the reference's perturbed state at the window's start (evaluation seed --seed), all in one simulator.
    Returns per policy: ``window_success`` (the windows' mean share of rollouts without a termination),
    ``window_return`` (the windows' mean normalized return, a terminated rollout's later steps counting nothing),
    ``window_success_framew`` (the success weighted by the windows' steps), the clips with a failed window and the mean
    errors over the steps before a termination."""
    import torch

    from ndm_gym.state import CONTROL_DT

    first, length, clip = window_segments(task.ref, max(1, round(args.window_s / CONTROL_DT)))
    W, K = first.shape[0], len(pis)
    per = max(1, min(args.num_envs, args.batch_worlds // (K * W)))
    sim = make_sim(K * W * per, getattr(targs, "joint_noise", 0.05))
    torch.manual_seed(args.seed)
    seg = (first.repeat(per), length.repeat(per))  # world i: window i modulo W
    with torch.no_grad():
        sums, term_at, ret = rollout_sums(sim, task, pis, goal_frames, noise=True, segments=seg, compiled=args.compile, graph=args.graph)
    dev, B = term_at.device, W * per
    L, wid = seg[1].to(dev).float(), torch.arange(B, device=dev) % W
    out = []
    for k in range(K):
        ta, r = term_at[k * B : (k + 1) * B], ret[k * B : (k + 1) * B]
        ok = (ta < 0).float()
        succ = torch.zeros(W, device=dev).index_add_(0, wid, ok) / per
        rets = torch.zeros(W, device=dev).index_add_(0, wid, r / (L * task.max_step_reward)) / per
        end = torch.where(ta >= 0, ta.float(), L)
        n = float(end.sum().clamp(min=1))
        failed = sorted({int(c) for c in clip[(succ < 1).cpu()]})
        out.append({"window_success": float(succ.mean()), "window_return": float(rets.mean()),
                    "window_success_framew": float((succ * length.to(dev)).sum() / length.sum()), "windows": W, "rollouts_per_window": per,
                    "clips_with_failed_window": len(failed),
                    **{key: float((v[k * B : (k + 1) * B]).sum() / n) for key, v in sums.items()}})  # fmt: skip
    return out


def go2_setup(targs, clips, dev: str):
    """The Go2's scene, kinematics, task and clip names (train_quadruped.build, with a checkpoint's servos), its agents and
    its simulator's factory ``make_sim(worlds, joint_noise)``."""
    from ndm_gym.algos.quadruped_algos import QUADRUPED_ALGOS
    from ndm_gym.robots.go2 import JOINT_DAMPING
    from ndm_gym.sim.rigid.quadruped_sim import QuadrupedSim
    from ndm_gym.train_quadruped import build

    damping = JOINT_DAMPING  # a checkpoint's own: none recorded (runs before the option) or negative is the model's own
    if targs is not None:
        damping = getattr(targs, "joint_damping", None)
        damping = None if damping is None or damping < 0 else damping
    model, kin, task, names = build(clips, dev, kp=getattr(targs, "kp", None), kd=getattr(targs, "kd", None), joint_damping=damping)

    def make_sim(worlds: int, joint_noise: float = 0.05):
        return QuadrupedSim(model, task, worlds, dev, joint_noise=joint_noise)

    return model, kin, task, names, QUADRUPED_ALGOS, make_sim


def main(setup=go2_setup, description: str | None = None):
    """The evaluation (the module's docstring) of the robot of ``setup(checkpoint arguments or None, clips, device)`` (``go2_setup``'s
    results); evaluate_humanoid runs it with a humanoid's."""
    parser = argparse.ArgumentParser(description=description or __doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpts", type=str, default=None, help="a run's directory: evaluate every ckpt_<iteration>.pt (train_quadruped --keep_ckpts)")
    parser.add_argument("--run", type=str, default=None,
                        help="a run's directory: its last.pt as --ckpt does (with the reference targets and --push) and every kept "
                        "ckpt_<iteration>.pt as --ckpts does, all as blocks of one simulator in one pass (metrics.json, checkpoints.json)")
    parser.add_argument("--open_loop", action="store_true", help="no policy: the reference pose as the position targets only")
    parser.add_argument("--window_s", type=float, default=None,
                        help="instead of whole clips: every clip cut into windows of this many seconds, each rolled out from the "
                        "reference's state at its start (perturbed) in worlds of its own, all at once (windows.json; window success "
                        "and window return, for scanning checkpoints, not the whole-clip metrics)")
    parser.add_argument("--graph", action="store_true", help="the rollouts' control step as one CUDA graph (rollout_sums)")
    parser.add_argument("--compile", action="store_true", help="the rollouts' policy inputs, errors and rewards as compiled functions (equal up to rounding)")
    parser.add_argument("--clips", type=str, default=None, help="the clips to evaluate (default: the checkpoint's; --open_loop: every clip)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256, help="rollouts per clip, from perturbed starts")
    parser.add_argument("--max_worlds", type=int, default=16384, help="at most this many worlds in all: fewer rollouts per clip with many clips")
    parser.add_argument("--success", type=float, default=0.99, help="--ckpts: the success rate of minutes_to_success")
    parser.add_argument("--success_trunk_mm", type=float, default=None, help="--ckpts: minutes_to_success also asks the trunk within this on average (mm)")
    parser.add_argument("--success_return", type=float, default=None, help="--ckpts: minutes_to_success also asks a normalized return of this or more")
    parser.add_argument("--ckpt_worlds", type=int, default=2048, help="--ckpts: at most this many worlds in all (fewer rollouts per clip with many clips)")
    parser.add_argument("--batch_worlds", type=int, default=32768, help="at most this many worlds in a simulator: the policies evaluated at once")
    parser.add_argument("--clip_success", type=float, default=1.0, help="a clip is listed as failed below this success rate (1: any rollout failed)")
    parser.add_argument("--push", type=float, nargs="+", default=None, help="also the policy's rollouts under trunk velocity pushes of these sizes (m/s)")
    parser.add_argument("--push_every", type=float, default=2.0, help="seconds between pushes")
    parser.add_argument("--push_seed", type=int, default=0, help="seed of the pushes' directions (the same pushes for every policy)")
    parser.add_argument("--save_rollout", action="store_true", help="also one rollout per clip from the unperturbed start, saved (rollout.npz)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if sum((args.ckpt is not None, args.ckpts is not None, args.run is not None, args.open_loop)) != 1:
        parser.error("give one of --ckpt, --ckpts, --run and --open_loop")
    if args.push is not None and args.ckpts:
        parser.error("--push: with --ckpt, --run or --open_loop")
    if args.window_s is not None and (args.push is not None or args.save_rollout or args.window_s <= 0):
        parser.error("--window_s: a positive length, without --push and --save_rollout")
    if args.success_return is not None and not (args.ckpts or args.run):
        parser.error("--success_return: with --ckpts or --run")
    whole = ("max_worlds", "ckpt_worlds", "clip_success", "success", "success_trunk_mm", "success_return")
    if args.window_s is not None and any(getattr(args, k) != parser.get_default(k) for k in whole):
        parser.error("--window_s: --num_envs rollouts per window within --batch_worlds; not with --" + ", --".join(whole))

    import numpy as np
    import torch

    from ndm_gym.state import CONTROL_DT

    dev = "cuda:0"
    run_dir = args.ckpts or args.run
    kept = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt"))) if run_dir else []
    if run_dir and not kept:
        parser.error(f"no ckpt_*.pt in {run_dir} (train_quadruped --keep_ckpts)")
    paths = [args.ckpt] if args.ckpt else [os.path.join(args.run, "last.pt")] if args.run else kept
    ck = torch.load(paths[-1], map_location=dev, weights_only=False) if paths else None
    targs = argparse.Namespace(**ck["args"]) if ck else None
    clips = args.clips if args.clips is not None or targs is None else targs.clips
    model, kin, task, names, algos, make_sim = setup(targs, clips, dev)
    if args.run:  # every block in one pass: the last checkpoint, the reference targets, the pushes and the kept checkpoints
        blocks = 2 + len(args.push or []) + len(kept)
        per_clip = max(1, min(args.num_envs, args.batch_worlds // (blocks * len(names))))
    else:
        per_clip = max(1, min(args.num_envs, (args.ckpt_worlds if args.ckpts else args.max_worlds) // len(names)))
    C, lengths = len(names), (task.ref.clip_last - task.ref.clip_first).tolist()
    groups = [torch.tensor(g, device=dev) for g in length_buckets(lengths, per_clip)]
    algo = algos[targs.algo](targs, model, kin, task, dev) if ck else None
    goal_frames = algo.goal_frames if algo is not None else 0
    sims = {}  # (group, blocks): its simulator

    def evaluate(pis: list, push_mags: list | None = None) -> list[dict]:
        """The summaries of ``pis`` (policies, None: the reference pose as the targets), each from the same perturbed
        starts (evaluation seed ``--seed``) and, with ``push_mags`` (one per policy, m/s), under the same pushes; as many at
        once as ``--batch_worlds`` takes."""
        K = max(1, min(len(pis), args.batch_worlds // (per_clip * max(len(g) for g in groups))))
        if K < len(pis):
            return [m for i in range(0, len(pis), K) for m in evaluate(pis[i : i + K], None if push_mags is None else push_mags[i : i + K])]
        K, N = len(pis), per_clip * C
        for key in [key for key in sims if key[1] != K]:  # one set of simulators at a time
            del sims[key]
        for gi, idx in enumerate(groups):  # built before the seed: a simulator's construction draws its own starts
            if (gi, K) not in sims:
                sims[(gi, K)] = make_sim(K * per_clip * len(idx), getattr(targs, "joint_noise", 0.05))
        torch.manual_seed(args.seed)  # the same perturbed starts for every policy
        push = None
        if push_mags is not None and any(push_mags):
            push = (push_mags, max(1, round(args.push_every / CONTROL_DT)), torch.Generator(device=dev).manual_seed(args.push_seed))
        sums, ret, term_at = None, torch.zeros(K, N, device=dev), torch.full((K, N), -1, dtype=torch.long, device=dev)
        for gi, idx in enumerate(groups):  # into summarize's layout: world i on clip i modulo the clips
            B = per_clip * len(idx)
            i = torch.arange(B, device=dev)
            cols = (i // len(idx)) * C + idx[i % len(idx)]
            with torch.no_grad():
                s, ta, r = rollout_sums(sims[(gi, K)], task, pis, goal_frames, noise=True, push=push, clips=idx, compiled=args.compile, graph=args.graph)
            if sums is None:
                sums = {k: torch.zeros(K, N, device=dev) for k in s}
            for k, v in s.items():
                sums[k][:, cols] = v.view(K, B)
            ret[:, cols], term_at[:, cols] = r.view(K, B), ta.view(K, B)
        return [summarize_sums(task, {k: v[j] for k, v in sums.items()}, term_at[j], names, ret[j], args.clip_success) for j in range(K)]

    os.makedirs(args.out, exist_ok=True)
    kept_cks, kept_pis = [], []
    for path in kept:
        c = torch.load(path, map_location=dev, weights_only=False)
        pi = copy.deepcopy(algo.pi)
        pi.load_state_dict(c["pi"])
        kept_cks.append(c), kept_pis.append(pi.eval())
    if ck is not None:
        algo.pi.load_state_dict(ck["pi"])  # the policy evaluated (with --pi_ema the average)
    runs = () if args.ckpts else (("reference_targets", None),) if algo is None else (("policy", algo), ("reference_targets", None))
    pushes = args.push or []  # the policy (--open_loop: the reference targets) under pushes, in the same simulator
    pis = [None if agent is None else algo.pi.eval() for _, agent in runs] + ([None if runs[0][1] is None else algo.pi] * len(pushes) if pushes else [])
    if args.window_s is not None:  # windows of the clips instead of whole clips
        names_w = [key for key, _ in runs] + [os.path.basename(p) for p in kept]
        res = evaluate_windows(task, make_sim, pis + kept_pis, goal_frames, args, targs)
        result = {"window_s": args.window_s, "seed": args.seed, **{n: r for n, r in zip(names_w, res, strict=True)}}
        for n, r in zip(names_w, res, strict=True):
            print(n, " ".join(f"{k}={v:.4g}" for k, v in r.items() if isinstance(v, float)), flush=True)
        with open(os.path.join(args.out, "windows.json"), "w") as f:
            json.dump(result, f, indent=2)
        return
    out = evaluate(pis + kept_pis, [0.0] * len(runs) + list(pushes) + [0.0] * len(kept_pis) if pushes else None)
    if kept:
        rows = []
        for path, c, m in zip(kept, kept_cks, out[len(pis) :], strict=True):
            rows.append({"ckpt": os.path.basename(path), "iteration": c["iteration"], "wall_min": c["wall_time"] / 60,
                         **{k: v for k, v in m.items() if k != "clips"}})  # fmt: skip
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items() if k != "failed"), flush=True)
        result = {"run": run_dir, "rollouts_per_clip": per_clip, "seed": args.seed, "success": args.success,
                  "success_trunk_mm": args.success_trunk_mm, "success_return": args.success_return,
                  "minutes_to_success": minutes_to_success(rows, args.success, args.success_trunk_mm, args.success_return), "checkpoints": rows}  # fmt: skip
        with open(os.path.join(args.out, "checkpoints.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"minutes to a success of {args.success}: {result['minutes_to_success']}")
        if args.ckpts:
            return
        out = out[: len(pis)]
    kp, kd = model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 2]  # one number where the servos share it (the Go2), or per servo
    metrics = {"ckpt": paths[0] if paths else None, "iteration": ck["iteration"] if ck else None, "control_dt": CONTROL_DT,
               **{k: float(v[0]) if (v == v[0]).all() else v.tolist() for k, v in (("kp", kp), ("kd", kd))}, "rollouts_per_clip": per_clip,
               "seed": args.seed}  # fmt: skip
    for (key, _), m in zip(runs, out, strict=False):
        metrics[key] = m
    if args.push is not None:  # the per-clip success rates only
        metrics["push_every_s"], metrics["push_seed"], metrics["push"] = args.push_every, args.push_seed, {}
        for mag, m in zip(pushes, out[len(runs) :], strict=True):
            metrics["push"][f"{mag:g}"] = {**{k: v for k, v in m.items() if k != "clips"},
                                           "clip_success_rate": {n: c["success_rate"] for n, c in m["clips"].items()}}  # fmt: skip
    if args.save_rollout:  # one rollout per clip from the unperturbed start, for rendering
        out, ref = {}, task.ref
        one = make_sim(len(names))
        for key, agent in runs:
            qpos, _, term_at, _ = rollout(one, task, agent)
            for i, name in enumerate(names):
                n = int(ref.clip_last[i] - ref.clip_first[i]) + 1
                out[f"{name}/{'qpos' if key == 'policy' else 'reference_targets_qpos'}"] = qpos[:n, i].cpu().numpy()
                out[f"{name}/{'terminated_at' if key == 'policy' else 'reference_targets_terminated_at'}"] = np.array(int(term_at[i]))
        for i, name in enumerate(names):
            out[f"{name}/reference_qpos"] = ref.qpos[ref.clip_first[i] : ref.clip_last[i] + 1].cpu().numpy()
        np.savez(os.path.join(args.out, "rollout.npz"), control_dt=CONTROL_DT, **out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    brief = {k: {m: v for m, v in metrics[k].items() if m != "clips"} for k, _ in runs}
    brief.update({f"push {k}": {m: v for m, v in p.items() if m != "clip_success_rate"} for k, p in metrics.get("push", {}).items()})
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
