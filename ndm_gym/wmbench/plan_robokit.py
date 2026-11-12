"""Run one of RoboKit's SimMPC examples, unmodified, with its planner's rollouts on a world model (``sim.robokit_ndm``) and its
execution on MuJoCo Warp. SimMPC is handed an NdmSim twin of the example's simulator (the same model, data, substeps and
contact capacities, with the world models of ``--ndm``), so its rollout and validation worlds step the world model, while the
example's own simulator executes the plans. Without ``--ndm`` the example runs as it is (the baseline); in both cases only
viser's viewer, which the examples open after planning, is replaced so that the run ends. ``--stages fine`` gives a planner
that has no stages (and would try 2 and 4 timesteps on its first plan) the stages at its timestep, as MuJoCo Warp's own runs
chose them. The example's output lines (its pass or fail line, its planning time) and the run's settings
are written to ``--out`` (.json).

Run: ``python -m ndm_gym.wmbench.plan_robokit --example .../sim_mpc/04_g1_carry.py --ndm 0.02=fits/g1_carry.pt --dtype bf16
       --derived kinematics --out results/plan_g1_carry.json``
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import runpy
import shlex
import sys
import time

import numpy as np

from .collect_robokit import _commit, _Stop, _Tee, _versions


def null_sim():
    """A MjWarpSim whose step does nothing: plans on it cost SimMPC's own work only (sampling, controls, costs)."""
    from robokit.sim import MjWarpSim

    class NullSim(MjWarpSim):
        def step(self):
            pass

        def resized(self, nworld: int):
            scenes = self._args["model"]
            n = nworld if not isinstance(scenes, (list, tuple)) else [nworld // len(scenes)] * len(scenes)
            return type(self)(nworld=n, **self._args)

    return NullSim


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--example", required=True, help="the path of one of RoboKit's SimMPC examples")
    p.add_argument("--example_args", action="append", default=None, help="the example's own options, as one string; repeatable, in order")
    p.add_argument("--ndm", nargs="*", default=[], help="step length=fit_mjcf --save file, one per step length the planner takes")
    p.add_argument("--physics", type=float, nargs="*", default=[],
                   help="step lengths (s) the planner runs on MuJoCo Warp's own step beside the world models' (a hybrid planner)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="the world models' inference copies")
    p.add_argument("--derived", nargs="*", default=["kinematics"], help="MuJoCo Warp stages the costs read, run before each step")
    p.add_argument("--stages", default="as_is", choices=["as_is", "fine"], help="fine: a planner without stages at its timestep")
    p.add_argument("--seed", type=int, default=None, help="the planner's sampling seed (default: the example's)")
    p.add_argument("--no_compile", action="store_true", help="the world models uncompiled (needed on Warp's CPU device)")
    p.add_argument("--null", action="store_true", help="plan on a simulator whose step does nothing (SimMPC's own cost, the floor)")
    p.add_argument("--save_executed", default=None,
                   help=".npz of the executed trajectory (each instance's time, qpos, qvel and ctrl per sim step, every execute() in order), and .mjb of the executed model beside it")
    p.add_argument("--out", required=True, help=".json of the run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import mujoco
    import robokit.helpers.sim_mpc as sim_mpc_pkg
    import torch
    import viser
    from robokit.helpers.sim_mpc import sim_mpc as mod

    from ..sim.robokit_ndm import load_models, ndm_sim, planning

    paths = {}
    for spec in args.ndm:
        dt, path = spec.split("=", 1)
        paths[float(dt)] = path
    if args.null and paths:
        raise ValueError("--null plans on a simulator whose step does nothing; --ndm's world models would go unused")
    if args.physics and not paths:
        raise ValueError("--physics names the step lengths beside the world models of --ndm; without --ndm the planner is MuJoCo Warp's")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    NdmSim = ndm_sim() if paths else None
    init, execute = mod.SimMPC.__init__, mod.SimMPC.execute
    planners, executed = [], []

    def patched_init(self, config, sim, *a, **kw):
        if args.seed is not None:
            config = dataclasses.replace(config, seed=args.seed)
        if args.stages == "fine" and not config.stages:
            config = dataclasses.replace(config, stages=[mod.SimMPCStage(config.max_iterations)])
        if args.null:
            sim = null_sim()(nworld=sim.nworld, **sim._args)
        elif NdmSim is not None:
            models = load_models(paths, sim.mjm, torch.device(sim.device.alias), dtype)
            sim = NdmSim(nworld=sim.nworld, ndm=models, derived=tuple(args.derived), compile=not args.no_compile, physics=args.physics,
                         **sim._args)  # fmt: skip
        init(self, config, sim, *a, **kw)
        planners.append({"stages": repr(self.stages), "worlds": {k: int(v.nworld) for k, v in self.sims.items()},
                         "sim": type(self.sims["rollout"]).__name__})  # fmt: skip

    def patched_execute(self, sim, num_steps, *a, **kw):
        rows, plans = execute(self, sim, num_steps, *a, **kw)
        if args.save_executed is not None:
            if not executed:  # the executed model, as it simulates, for rendering
                mujoco.mj_saveModel(sim.mjm, os.path.splitext(args.save_executed)[0] + ".mjb", None)
            executed.append([{k: np.asarray(v) for k, v in r.items() if k in ("time", "qpos", "qvel", "ctrl")} for r in rows])
        return rows, plans

    def no_viewer(*a, **kw):
        raise _Stop

    saved = (mod.SimMPC.__init__, mod.SimMPC.execute, viser.ViserServer, sys.argv)
    mod.SimMPC.__init__, mod.SimMPC.execute = patched_init, patched_execute
    sim_mpc_pkg.SimMPC = mod.SimMPC
    viser.ViserServer = no_viewer
    sys.argv = [args.example] + [x for a in args.example_args or [] for x in shlex.split(a)]  # the example parses its own options (tyro)
    log, t0 = io.StringIO(), time.time()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), planning() if paths else contextlib.nullcontext():
            try:
                runpy.run_path(args.example, run_name="__main__")
            except _Stop:
                pass
    finally:  # the patches undone, for whatever runs next in this process
        mod.SimMPC.__init__, mod.SimMPC.execute, viser.ViserServer, sys.argv = saved
    if not planners:
        raise RuntimeError(f"{args.example}: no SimMPC was built")
    out = {"example": os.path.abspath(args.example), "example_args": args.example_args, "ndm": {str(k): v for k, v in paths.items()},
           "dtype": args.dtype if paths else None, "derived": args.derived if paths else None, "physics": args.physics, "stages": args.stages,
           "seed": args.seed,
           "planners": planners, "seconds": time.time() - t0, "output": log.getvalue().splitlines()[-20:],
           "versions": {"mujoco": mujoco.__version__, **_versions()}, "commit": _commit(args.example)}  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    if args.save_executed is not None:
        if not executed:
            raise RuntimeError(f"{args.example}: SimMPC.execute never ran; nothing to save")
        n = len(executed[0])  # instances
        np.savez_compressed(args.save_executed, **{f"{k}_{i}": np.concatenate([call[i][k] for call in executed])
                                                   for i in range(n) for k in executed[0][i]})  # fmt: skip
    print(f"[plan_robokit] {planners[0]['sim']} planner; {out['output'][-1] if out['output'] else ''}")


if __name__ == "__main__":
    main()
