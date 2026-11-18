"""Run mbrl-lib's true-dynamics controller (``mbrl/diagnostics/control_env.py`` at 3f93ccc, unmodified) on one of the environments
of benchmarks/mbrl_lib, its optimizer's candidate rollouts on CPU MuJoCo as control_env.py ships them (its multiprocessing
pool), on MuJoCo Warp or on the NDM (``sim.mbrl_ndm``); every plan executes on CPU MuJoCo, control_env.py's evaluation
environment. Patches at run time, none in mbrl-lib's files:

- the environments as the benchmark runs them (``sim.mbrl_envs.patch_envs``: Hopper's and Walker2d's local-coordinate
  copies, dm_control at PlaNet's action repeat, with ``--own_margins`` the contact margins of mbrl-lib's own MuJoCo, with
  ``--own_mujoco`` the models as it steps them, and with ``--counter_fix`` mbrl-lib's dm_control step counter restored on the
  right wrapper), in this process and in the pool's workers (their initializer);
- the optimizer's population and iterations at ``--setting``'s: control_env.py's defaults (512, horizon 30, 5), the
  environment's PETS override, or PlaNet's (1,000, 12, 10); the horizon through control_env.py's own ``--control_horizon``;
  the rest is control_env.py's (CEM, elite ratio 0.1, alpha 0.1, the best solution over the iterations);
- ``--sim warp`` or ``ndm``: ``TrajectoryOptimizer.optimize`` scores the population with the batched rollouts from the
  evaluation environment's current state instead of the pool, whose workers are then not started;
- the evaluation environment's ``step`` records each executed step, and each plan (``optimize``) is timed.

An environment a simulator cannot score is refused with the reason (``mbrl_envs.refusal``; pets_reacher is not run at all).
control_env.py runs ``--num_steps`` steps (default: the environment's episode) and, as shipped, does not stop when the
environment terminates. The json holds per step the planning time (``optimize``), control_env.py's own step time, the reward
and the flags; per episode control_env.py's total and the return up to the first termination or truncation.

Run: ``python -m ndm_gym.wmbench.plan_mbrl --env pets_halfcheetah --sim ndm --ndm fits/hc.pt --setting pets --seed 0
       --out results/plan_hc_ndm_s0.json``
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import runpy
import subprocess
import sys
import time

import numpy as np

from ..sim.mbrl_envs import ENVS, patch_envs, physics, recording_dm_control, refusal, state, unpatch_envs, worker_init
from .collect_robokit import _Tee

STEP_LINE = re.compile(r"^step: (\d+), time: +([-+.\deE]+),")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENVS))
    p.add_argument("--sim", default="cpu", choices=["cpu", "warp", "ndm"], help="where the candidate rollouts run")
    p.add_argument("--setting", default="control_env", choices=["control_env", "pets", "planet"], help="population, horizon, iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=None, help="executed steps (default: the environment's episode)")
    p.add_argument("--num_processes", type=int, default=1, help="cpu: control_env.py's pool")
    p.add_argument("--local_models", default=None,
                   help="the directory of hopper_local.xml and walker2d_local.xml (tools/convert_mbrl_models.py makes them, into data/mbrl_models)")
    p.add_argument("--counter_fix", action="store_true", help="cpu, dm_control: mbrl-lib's step counter restored in the pool")
    p.add_argument("--own_margins", action="store_true", help="the contact margins of mbrl-lib's own MuJoCo (sim.mbrl_envs.set_own_margins)")
    p.add_argument("--own_mujoco", action="store_true",
                   help="the models as mbrl-lib's own MuJoCo steps them: its margins, inverse weights and convex collision (sim.mbrl_envs.set_own_mujoco)")
    p.add_argument("--ndm", default=None, help="ndm: a fit_mjcf --save file of the environment's step")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="ndm: the inference copy's network")
    p.add_argument("--device", default="cuda:0", help="warp and ndm: the device (cpu: Warp's CPU device, torch's CPU)")
    p.add_argument("--no_compile", action="store_true", help="ndm: uncompiled (needed on the CPU)")
    p.add_argument("--fused", action="store_true",
                   help="ndm: the world model's kinematics and contact gaps on the fused path (worldmodels.mjcf.MjcfWorldModel.fused)")
    p.add_argument("--control_env", default=None, help="control_env.py (default: the imported mbrl's)")
    p.add_argument("--out", required=True, help=".json of the run")
    return p.parse_args(argv)


class _NoPool:
    """control_env.py's pool when the rollouts are batched: never used, so no worker starts."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _handler_classes(spec):
    from mbrl.util.mujoco import MujocoEnvHandler

    if not spec.dm_control:
        return (MujocoEnvHandler,)
    from mbrl.util.dmcontrol import DmcontrolEnvHandler

    return (DmcontrolEnvHandler,)


def run_control_env(spec, args, setting, rollouts_for=None, on_plan=None, on_step=None) -> dict:
    """control_env.py run as ``__main__`` with the patches. ``rollouts_for(mjm, substeps, population)``: the batched rollouts
    (None: the pool); ``on_plan(t, handler state, sim.mbrl_envs.state, callback)`` returns plan t's callback in place of
    control_env.py's, or None; ``on_step(pre, states, action, reward, terminated, truncated)`` sees each executed step (pre:
    qpos and qvel before it; states: after each of its control steps, one but for dm_control's repeat)."""
    import multiprocessing as mp

    import mbrl
    import mbrl.planning
    import mbrl.util

    pop, horizon, iters = setting
    path = args.control_env or os.path.join(os.path.dirname(mbrl.__file__), "diagnostics", "control_env.py")
    commit = _commit(path)  # before the run: a run whose mbrl-lib cannot be named does not start
    procs = args.num_processes if rollouts_for is None else 1
    num_steps = args.num_steps or spec.episode
    out_dir = os.path.splitext(os.path.abspath(args.out))[0] + "_control_env"
    rec = {"plan_s": [], "steps": [], "optimizer": None, "eval": None}
    Opt = mbrl.planning.TrajectoryOptimizer
    real_init, real_opt = Opt.__init__, Opt.optimize
    handlers = _handler_classes(spec)
    real_make = [h.__dict__["make_env_from_str"] for h in handlers]
    real_pool = mp.Pool
    my_handler = mbrl.util.create_handler_from_str(spec.env)

    def opt_init(self, optimizer_cfg, *a, **kw):
        optimizer_cfg.population_size, optimizer_cfg.num_iterations = pop, iters
        real_init(self, optimizer_cfg, *a, **kw)
        from omegaconf import OmegaConf

        rec["optimizer"] = {k: v for k, v in OmegaConf.to_container(optimizer_cfg).items() if k not in ("lower_bound", "upper_bound")}
        rec["optimizer"]["horizon"] = self.horizon

    def optimize(self, trajectory_eval_fn, callback=None):
        env = rec["eval"]
        t = len(rec["plan_s"])
        if rec.get("rollouts") is not None:
            st = state(spec, my_handler, env)
            trajectory_eval_fn = lambda acts: rec["rollouts"].values(st, acts)  # noqa: E731
        if on_plan is not None:
            callback = on_plan(t, my_handler.get_current_state(env), state(spec, my_handler, env), callback) or callback
        t0 = time.perf_counter()
        out = real_opt(self, trajectory_eval_fn, callback=callback)
        rec["plan_s"].append(time.perf_counter() - t0)
        return out

    def make_wrapper(real):
        def make_env_from_str(env_name):
            env = real.__func__(env_name)
            if rec["eval"] is None:  # the first in this process: control_env.py's evaluation environment
                rec["eval"] = env
                _, d, substeps, _ = physics(spec, env)
                step = env.step

                def recorded(action):
                    pre, inner = (d.qpos.copy(), d.qvel.copy()), []
                    with recording_dm_control(spec, env, inner, ("qpos", "qvel")) if on_step and spec.dm_control else contextlib.nullcontext():
                        obs, r, term, trunc, info = step(action)
                    rec["steps"].append({"reward": float(r), "terminated": bool(term), "truncated": bool(trunc)})
                    if on_step is not None:
                        states = [(d.qpos.copy(), d.qvel.copy())]
                        if spec.dm_control:  # its control steps inside the repeat (none on a step that resets the environment)
                            states = [(f.qpos[0].numpy(), f.qvel[0].numpy()) for f, _, _ in inner]
                        on_step(pre, states, np.asarray(action), r, term, trunc)
                    return obs, r, term, trunc, info

                env.step = recorded
                if rollouts_for is not None:
                    t0 = time.perf_counter()
                    rec["rollouts"] = rollouts_for(physics(spec, env)[0], substeps, pop)
                    rec["build_s"] = time.perf_counter() - t0
            return env

        return staticmethod(make_env_from_str)

    def pool(processes=None, initializer=None, initargs=(), *a, **kw):
        if rollouts_for is not None:
            return _NoPool()
        return real_pool(processes, worker_init, (args.local_models, args.counter_fix, args.own_margins, args.own_mujoco, initializer, tuple(initargs)),
                         *a, **kw)  # fmt: skip

    saved_argv = sys.argv
    sys.argv = [path, "--env", spec.env, "--seed", str(args.seed), "--control_horizon", str(horizon), "--num_processes", str(procs),
                "--num_steps", str(num_steps), "--samples_per_process", str(math.ceil(pop / procs)), "--optimizer_type", "cem",
                "--output_dir", out_dir]  # fmt: skip
    Opt.__init__, Opt.optimize, mp.Pool = opt_init, optimize, pool
    for h in handlers:
        h.make_env_from_str = make_wrapper(h.__dict__["make_env_from_str"])
    log, t0 = io.StringIO(), time.time()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            runpy.run_path(path, run_name="__main__")
    finally:  # the patches undone, for whatever runs next in this process
        Opt.__init__, Opt.optimize, mp.Pool, sys.argv = real_init, real_opt, real_pool, saved_argv
        for h, f in zip(handlers, real_make):
            h.make_env_from_str = f
    lines = log.getvalue().splitlines()
    step_s = [float(m.group(2)) for m in map(STEP_LINE.match, lines) if m]
    total = [float(x.split(":", 1)[1]) for x in lines if x.startswith("total_reward: ")]
    if len(rec["plan_s"]) != num_steps or len(rec["steps"]) != num_steps or len(step_s) != num_steps or len(total) != 1:
        raise RuntimeError(f"run_control_env: {len(rec['plan_s'])} plans, {len(rec['steps'])} steps, {len(step_s)} step lines and "
                           f"{len(total)} totals for {num_steps} steps")  # fmt: skip
    for s, p, c in zip(rec["steps"], rec["plan_s"], step_s):
        s["plan_s"], s["step_s"] = p, c
    ends = [i for i, s in enumerate(rec["steps"]) if s["terminated"] or s["truncated"]]
    n = ends[0] + 1 if ends else num_steps
    return {"env": spec.name, "mbrl_env": spec.env, "seed": args.seed, "num_steps": num_steps, "num_processes": procs,
            "counter_fix": args.counter_fix, "own_margins": args.own_margins, "own_mujoco": args.own_mujoco, "population": pop, "horizon": horizon,
            "iterations": iters, "optimizer": rec["optimizer"],
            "steps": rec["steps"], "total_reward": total[0], "episode_return": float(sum(s["reward"] for s in rec["steps"][:n])),
            "episode_steps": n, "plan_s_median": float(np.median(rec["plan_s"])), "build_s": rec.get("build_s"), "seconds": time.time() - t0,
            "control_env": os.path.abspath(path), "mbrl_commit": commit, "output": lines[-20:], "versions": versions()}  # fmt: skip


def batched_rollouts(spec, args):
    """``run_control_env``'s ``rollouts_for`` of the options ``--sim``, ``--ndm``, ``--dtype``, ``--device`` and ``--no_compile``:
    the builder of the batched rollouts on MuJoCo Warp or on the world model, or None for control_env.py's pool (``--sim cpu``)."""
    if args.sim == "cpu":
        return None

    def rollouts_for(mjm, substeps, population):
        import torch

        from ..sim.mbrl_ndm import NdmRollouts, WarpRollouts, load_ndm_copy

        if args.sim == "warp":
            return WarpRollouts(spec, mjm, substeps, population, device=args.device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
        wm = load_ndm_copy(args.ndm, mjm, torch.device(args.device), dtype)
        if args.fused:  # the same values to rounding in fewer kernels; off: the path the fit ran
            wm.fused = True
        return NdmRollouts(spec, mjm, substeps, wm, population, compile=not args.no_compile)

    return rollouts_for


def check_options(spec, args):
    """The refusals of an environment on a simulator (``mbrl_envs.refusal``), and --counter_fix only where it applies."""
    why = refusal(spec, args.sim)
    if why:
        raise ValueError(f"{spec.name} on {args.sim}: refused, {why}")
    if args.counter_fix and not (spec.dm_control and args.sim == "cpu"):
        raise ValueError("--counter_fix: for a dm_control environment's pool (--sim cpu) only")


def versions() -> dict:
    """The versions of the packages a run reads (None: not imported)."""
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "mujoco", "gymnasium", "gym", "hydra", "omegaconf", "warp", "mujoco_warp", "dm_control"):
        mod = sys.modules.get(name)
        out[name] = None if mod is None else getattr(mod, "__version__", getattr(getattr(mod, "config", None), "version", "?"))
    if out.get("dm_control") == "?":
        from importlib import metadata

        out["dm_control"] = metadata.version("dm_control")
    return out


def _commit(path: str) -> str:
    """The commit of the git checkout that holds ``path`` (mbrl-lib's control_env.py), which a run's json records. A checkout
    that git cannot read is an error: no checkout, no git, or git's refusal of a checkout that another user owns ("dubious
    ownership": a ``safe.directory`` entry in the git configuration of the user who runs it admits it)."""
    where = os.path.dirname(os.path.abspath(path))
    try:
        run = subprocess.run(["git", "-C", where, "rev-parse", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"plan_mbrl: no git on the path to read the commit of mbrl-lib's checkout at {where}") from e
    if run.returncode != 0:
        raise RuntimeError(f"plan_mbrl: git cannot read the commit of mbrl-lib's checkout at {where}: {run.stderr.strip()}")
    return run.stdout.strip()


def main(argv=None):
    args = parse_args(argv)
    spec = ENVS[args.env]
    settings = spec.settings()
    if args.setting not in settings:
        raise ValueError(f"plan_mbrl: {spec.name} has no {args.setting!r} setting ({sorted(settings)})")
    check_options(spec, args)
    if args.sim == "ndm" and not args.ndm:
        raise ValueError("plan_mbrl --sim ndm needs --ndm (a fit_mjcf --save file)")
    if args.sim != "ndm" and args.ndm:
        raise ValueError(f"plan_mbrl: --ndm with --sim {args.sim}")
    patch_envs(args.local_models, args.counter_fix, args.own_margins, args.own_mujoco)
    try:
        out = run_control_env(spec, args, settings[args.setting], rollouts_for=batched_rollouts(spec, args))
    finally:
        unpatch_envs()
    out.update(sim=args.sim, setting=args.setting, ndm=os.path.abspath(args.ndm) if args.ndm else None,
               named_difference=spec.named if args.sim == "ndm" else None,
               dtype=args.dtype if args.sim == "ndm" else None, fused=args.fused if args.sim == "ndm" else None,
               device=args.device if args.sim != "cpu" else "cpu",
               compile=(not args.no_compile) if args.sim == "ndm" else None, local_models=args.local_models)  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[plan_mbrl] {spec.name} on {args.sim} ({args.setting}: {out['population']} x {out['horizon']} x {out['iterations']}), "
          f"{out['num_steps']} steps: control_env total {out['total_reward']:.4g}, episode return {out['episode_return']:.4g} over "
          f"{out['episode_steps']} steps, median plan {out['plan_s_median']:.3g} s")  # fmt: skip


if __name__ == "__main__":
    main()
