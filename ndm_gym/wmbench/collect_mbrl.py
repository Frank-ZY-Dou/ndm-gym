"""Record CPU MuJoCo transitions of one of the environments of benchmarks/mbrl_lib at its control step (the environment's
step; dm_control: its own control step, ``repeat`` of which make one environment step at PlaNet's action repeat), for the
world model of MJCF models (``worldmodels.mjcf``):

- (a) ``rollout``: the optimizer's candidate rollouts at recorded plan starts. mbrl-lib's control_env.py runs as
  ``plan_mbrl --sim cpu`` runs it (unmodified, its pool, the run-time patches of ``sim.mbrl_envs``); at every ``--record_every``-th plan the first
  ``--record`` candidates of each iteration (CEM's draws, in random order) are kept with the plan's start, and after the run they
  are replayed on a CPU MuJoCo environment built as control_env.py builds it, from the same state, over the whole horizon (the
  physics past a termination too; up to dm_control's time limit, where its environment would reset). With ``--sim ndm`` the
  planner plans on a world model (``--ndm``, as ``plan_mbrl --sim ndm``) and every plan still executes on CPU MuJoCo: the
  plan starts are then the states that planner reaches, and the kept candidates are replayed on CPU MuJoCo all the same
  (recordings for a further fit of the world model on its own planner's states);
- (b) ``executed``: the closed loop's trajectory, control_env.py's evaluation environment step by step. control_env.py steps on
  after a termination; ``--until_termination`` keeps the plan starts and the executed steps up to the run's first termination
  only (the noisy episodes then replay the kept actions);
- (c) ``random``: ``--episodes`` episodes of uniform random actions from the environment's resets (at most the environment's
  episode, ``MbrlEnv.episode``), and ``noisy``:
  ``--episodes`` replays of the executed actions from the executed trajectory's start with Gaussian noise of ``--noise`` times
  the action range's half-width (clipped to it); each episode until the environment terminates or truncates it.

Each is a recording of ``wmbench/fit_mjcf.py`` (qpos (T + 1, N, nq), qvel, ctrl (T, N, nu), reset (T, N), dt), consecutive
rollouts or episodes joined by a reset step with zero controls, beside ``<out>_model.mjb`` (the model as the environment
simulates it, for the fits and MuJoCo Warp), ``<out>_plans.npz`` (the recorded plan starts: step, state, steps left, the
recorded candidates' actions and their values under the planner's simulator) and ``<out>.json`` (the settings, the versions, the run's
per-step rewards and planning times, the step and its physics steps, the rows per file). Seeds split into fit (0, 1) and
held-out (2) runs. pets_reacher is refused (``mbrl_envs.refusal``); ``--counter_fix``, ``--own_margins`` and ``--own_mujoco`` as
plan_mbrl's.

Run: ``python -m ndm_gym.wmbench.collect_mbrl --env pets_halfcheetah --seed 0 --num_processes 8 --out data/mbrl/hc_s0``
"""

import argparse
import contextlib
import json
import os

import numpy as np

from ..sim.mbrl_envs import ENVS, control_step_seconds, make_env, patch_envs, physics, recording_dm_control, unpatch_envs
from .plan_mbrl import batched_rollouts, check_options, run_control_env


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENVS))
    p.add_argument("--setting", default="control_env", choices=["control_env", "pets", "planet"], help="the planner's (plan_mbrl's option)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=None, help="executed steps (default: the environment's episode)")
    p.add_argument("--num_processes", type=int, default=1, help="control_env.py's pool")
    p.add_argument("--local_models", default=None,
                   help="the directory of hopper_local.xml and walker2d_local.xml (tools/convert_mbrl_models.py makes them, into data/mbrl_models)")
    p.add_argument("--counter_fix", action="store_true", help="dm_control: mbrl-lib's step counter restored in the pool")
    p.add_argument("--own_margins", action="store_true", help="the contact margins of mbrl-lib's own MuJoCo (sim.mbrl_envs.set_own_margins)")
    p.add_argument("--own_mujoco", action="store_true",
                   help="the models as mbrl-lib's own MuJoCo steps them: its margins, inverse weights and convex collision (sim.mbrl_envs.set_own_mujoco)")
    p.add_argument("--record", type=int, default=64, help="(a) candidates kept per iteration of a recorded plan")
    p.add_argument("--record_every", type=int, default=10, help="(a) every this many plans, the first included")
    p.add_argument("--episodes", type=int, default=10, help="(c) episodes of each kind")
    p.add_argument("--noise", type=float, default=0.3, help="(c) the noisy replays' std, in half-widths of the action range")
    p.add_argument("--until_termination", action="store_true", help="keep the plan starts and executed steps up to the first termination only")
    p.add_argument("--control_env", default=None, help="control_env.py (default: the imported mbrl's)")
    p.add_argument("--sim", default="cpu", choices=["cpu", "ndm"], help="where the planner's candidate rollouts run (plan_mbrl's option)")
    p.add_argument("--ndm", default=None, help="ndm: a fit_mjcf --save file of the environment's step")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="ndm: the inference copy's network")
    p.add_argument("--device", default="cuda:0", help="ndm: the device (cpu: torch's CPU)")
    p.add_argument("--no_compile", action="store_true", help="ndm: uncompiled (needed on the CPU)")
    p.add_argument("--fused", action="store_true",
                   help="ndm: the world model's kinematics and contact gaps on the fused path (worldmodels.mjcf.MjcfWorldModel.fused)")
    p.add_argument("--out", required=True, help="the output files' prefix")
    return p.parse_args(argv)


def recording(segments: list, dt: float) -> dict:
    """[(qpos (T_i + 1, N, nq), qvel, ctrl (T_i, N, nu)), ...] as one fit_mjcf recording: the segments one after another, a
    reset step with zero controls between two (its next state starts the next segment)."""
    q, v, u, r = [], [], [], []
    for i, (qi, vi, ui) in enumerate(segments):
        if i:
            u.append(np.zeros_like(ui[:1]))
            r.append(np.ones((1, ui.shape[1]), bool))
        q.append(qi)
        v.append(vi)
        u.append(ui)
        r.append(np.zeros(ui.shape[:2], bool))
    return {"qpos": np.concatenate(q), "qvel": np.concatenate(v), "ctrl": np.concatenate(u), "reset": np.concatenate(r), "dt": np.float64(dt)}


def episode(spec, handler, env, actions, start=None, stop=True) -> tuple:
    """One episode: ``actions(t, env)`` gives each step's action (None: the end), from ``start`` (a handler state; None: the
    env as it is), until the environment terminates or truncates it (``stop``; always at dm_control's time limit, after which
    its environment resets). (qpos (T + 1, 1, nq), qvel, ctrl (T, 1, nu)) at the control step: dm_control's inner steps."""
    if start is not None:
        handler.set_env_state(start, env)
        if spec.dm_control:  # mbrl-lib's set_env_state leaves gym's TimeLimit counting on: set it, so earlier replays do not truncate this one
            env.unwrapped.gym_env._elapsed_steps = start[1]
    _, d, _, _ = physics(spec, env)
    q, v, u = [d.qpos.copy()], [d.qvel.copy()], []
    for t in range(10**6):
        a = actions(t, env)
        if a is None:
            break
        inner = []
        with recording_dm_control(spec, env, inner, ("qpos", "qvel")) if spec.dm_control else contextlib.nullcontext():
            _, _, term, trunc, _ = env.step(a)
        for qi, vi in [(f.qpos[0].numpy(), f.qvel[0].numpy()) for f, _, _ in inner] if spec.dm_control else [(d.qpos.copy(), d.qvel.copy())]:
            q.append(qi)
            v.append(vi)
            u.append(np.asarray(a, dtype=np.float64))
        if (stop and (term or trunc)) or (spec.dm_control and term):
            break
    return np.stack(q)[:, None], np.stack(v)[:, None], np.stack(u)[:, None]


def replay(spec, handler, env, start, acts: np.ndarray) -> tuple:
    """The candidates ``acts`` (K, H, A) from the handler state ``start``, each over the whole horizon (dm_control: up to its
    time limit): (qpos (T + 1, K, nq), qvel, ctrl (T, K, nu))."""
    out = [episode(spec, handler, env, lambda t, e, k=k: acts[k, t] if t < acts.shape[1] else None, start, stop=False) for k in range(len(acts))]
    if len({o[2].shape[0] for o in out}) != 1:
        raise RuntimeError(f"collect_mbrl: a plan's candidates replayed to different lengths {[o[2].shape[0] for o in out]}")
    return tuple(np.concatenate([o[i] for o in out], 1) for i in range(3))


def main(argv=None):
    args = parse_args(argv)
    spec = ENVS[args.env]
    check_options(spec, args)
    settings = spec.settings()
    if args.setting not in settings:
        raise ValueError(f"collect_mbrl: {spec.name} has no {args.setting!r} setting ({sorted(settings)})")
    if (args.sim == "ndm") != bool(args.ndm):
        raise ValueError("collect_mbrl: --sim ndm needs --ndm (a fit_mjcf --save file), and --ndm needs --sim ndm")
    plans, executed, actions, inner = [], [], [], []  # executed: per control step; actions and inner (its control steps): per environment step

    def on_plan(t, hstate, st, callback):
        if t % args.record_every:
            return None
        plan = {"t": t, "start": hstate, "state": st, "acts": [], "values": []}
        plans.append(plan)

        def kept(population, values, i):
            plan["acts"].append(population[: args.record].numpy().copy())
            plan["values"].append(values[: args.record].numpy().copy())
            if callback is not None:
                callback(population, values, i)

        return kept

    def on_step(pre, states, action, reward, term, trunc):
        a = np.asarray(action, dtype=np.float64)
        actions.append(a)
        inner.append(len(states))
        for post in states:  # dm_control: its inner control steps, the same action
            executed.append((pre, post, a))
            pre = post

    run_args = argparse.Namespace(**{**vars(args), "out": args.out + ".json"})
    patch_envs(args.local_models, args.counter_fix, args.own_margins, args.own_mujoco)
    try:
        run = run_control_env(spec, run_args, settings[args.setting], rollouts_for=batched_rollouts(spec, args), on_plan=on_plan, on_step=on_step)
        ends = [i for i, s in enumerate(run["steps"]) if s["terminated"]]
        kept_steps = ends[0] + 1 if args.until_termination and ends else len(run["steps"])
        plans, executed, actions = [p for p in plans if p["t"] < kept_steps], executed[: sum(inner[:kept_steps])], actions[:kept_steps]
        handler, env = make_env(spec, args.seed)
        m, _, substeps, _ = physics(spec, env)
        dt = control_step_seconds(spec, env)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        import mujoco

        mujoco.mj_saveModel(m, f"{args.out}_model.mjb", None)
        files = {}

        def save(kind, segments):
            rec = recording(segments, dt)
            np.savez_compressed(f"{args.out}_{kind}.npz", **rec)
            files[f"{args.out}_{kind}.npz"] = {"steps": int(rec["ctrl"].shape[0]), "worlds": int(rec["ctrl"].shape[1]),
                                               "transitions": int((~rec["reset"]).sum())}  # fmt: skip

        blocks = [replay(spec, handler, env, p["start"], a) for p in plans for a in p["acts"]]
        save("rollout", blocks)
        segs, cur = [], None  # the executed trajectory, cut wherever a step does not start where the last one ended
        for pre, post, a in executed:
            if cur is None or not (np.array_equal(cur[0][-1], pre[0]) and np.array_equal(cur[1][-1], pre[1])):
                cur = ([pre[0]], [pre[1]], [])
                segs.append(cur)
            cur[0].append(post[0])
            cur[1].append(post[1])
            cur[2].append(a)
        save("executed", [(np.stack(q)[:, None], np.stack(v)[:, None], np.stack(u)[:, None]) for q, v, u in segs])
        rng = np.random.default_rng(args.seed)
        lo, hi = env.action_space.low, env.action_space.high
        rand = []
        for i in range(args.episodes):
            np.random.seed(args.seed + 1000 + i)  # the PETS environments reset from global numpy
            env.reset(seed=args.seed + 1000 + i)
            rand.append(episode(spec, handler, env, lambda t, e: rng.uniform(lo, hi).astype(np.float32) if t < spec.episode else None))
        save("random", rand)
        if not plans or plans[0]["t"] != 0:
            raise RuntimeError("collect_mbrl: the first plan's start was not recorded")
        run_start = plans[0]["start"]  # the executed trajectory's
        exec_acts = np.stack(actions).astype(np.float32)
        half = (hi - lo) / 2

        def noisy(t, e):
            if t >= len(exec_acts):
                return None
            return np.clip(exec_acts[t] + args.noise * half * rng.standard_normal(len(half)), lo, hi).astype(np.float32)

        save("noisy", [episode(spec, handler, env, noisy, run_start) for _ in range(args.episodes)])
    finally:
        unpatch_envs()
    np.savez_compressed(f"{args.out}_plans.npz", t=np.array([p["t"] for p in plans]), qpos=np.stack([p["state"]["qpos"] for p in plans]),
                        qvel=np.stack([p["state"]["qvel"] for p in plans]), steps_left=np.array([p["state"]["steps_left"] for p in plans]),
                        acts=np.stack([np.stack(p["acts"]) for p in plans]), values=np.stack([np.stack(p["values"]) for p in plans]))  # fmt: skip
    meta = {**{k: v for k, v in run.items() if k != "steps"}, "setting": args.setting, "record": args.record, "record_every": args.record_every,
            "episodes": args.episodes, "noise": args.noise, "local_models": args.local_models, "dt": dt, "substeps": substeps,
            "repeat": spec.repeat, "model": f"{args.out}_model.mjb", "sim": args.sim, "ndm": os.path.abspath(args.ndm) if args.ndm else None,
            "dtype": args.dtype if args.sim == "ndm" else None, "fused": args.fused if args.sim == "ndm" else None,
            "until_termination": args.until_termination, "kept_steps": kept_steps,
            "files": files, "rewards": [s["reward"] for s in run["steps"]], "plan_s": [s["plan_s"] for s in run["steps"]]}  # fmt: skip
    with open(f"{args.out}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[collect_mbrl] {spec.name} seed {args.seed}: " + ", ".join(f"{os.path.basename(k)} {v['transitions']} transitions"
                                                                       for k, v in files.items()))  # fmt: skip


if __name__ == "__main__":
    main()
