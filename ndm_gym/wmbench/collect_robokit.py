"""Record the transitions of one of RoboKit's SimMPC examples, run unmodified, for the world model of MJCF models
(``worldmodels.mjcf``). Three patches at run time, none in RoboKit's files: its simulator class becomes the recording one
(``sim.robokit_rec``), so the planner's rollout worlds, built with ``resized()``, and the executing simulator record;
``SimMPC.execute`` drains every recording simulator after each plan (after the example's own callback); and viser's server,
which the examples open after planning to replay the motion, ends the run instead. ``--seed`` sets the planner's sampling seed.

Writes ``<out>_rollout_dt<ms>.npz``, one recording per step length of the rollouts (the coarse stages step longer), and
``<out>_executed.npz``, the closed loop's trajectory; each a recording of ``wmbench/fit_mjcf.py``, beside ``<out>.json`` (the
example, its commit, the versions, the rows per file, the example's output lines).

Run: ``python -m ndm_gym.wmbench.collect_robokit --example .../sim_mpc/04_g1_carry.py --out data/robokit/g1_carry_s0 --record 64``
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import runpy
import shlex
import subprocess
import sys
import weakref

import numpy as np


class _Tee(io.TextIOBase):
    def __init__(self, *outs):
        self.outs = outs

    def write(self, s):
        for o in self.outs:
            o.write(s)
        return len(s)

    def flush(self):
        for o in self.outs:
            o.flush()


class _Stop(Exception):
    """Raised by the patched viewer: the example is past its planning."""


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--example", required=True, help="the path of one of RoboKit's examples (a script with a main())")
    p.add_argument("--out", required=True, help="the output files' prefix")
    p.add_argument("--record", type=int, default=64, help="rollout worlds recorded per simulator (the first ones)")
    p.add_argument("--capacity", type=int, default=2048, help="steps a recording ring holds between two drains")
    p.add_argument("--seed", type=int, default=None, help="the planner's sampling seed (default: the example's)")
    p.add_argument("--example_args", action="append", default=None, help="the example's own options, as one string (e.g. the notebook's --arctic-root); repeatable, in order")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import mujoco
    import robokit.helpers.sim_mpc as sim_mpc_pkg
    import robokit.sim as sim_pkg
    from robokit.helpers.sim_mpc import sim_mpc as sim_mpc_mod

    from ndm_gym.sim.robokit_rec import recording, recording_sim

    Base = recording_sim()
    live = weakref.WeakSet()
    kinds = weakref.WeakKeyDictionary()  # each recording simulator's role: rollout, validation or executing

    class Recording(Base):
        def __init__(self, *a, **kw):
            kw.setdefault("record", args.record)
            kw.setdefault("capacity", args.capacity)
            super().__init__(*a, **kw)
            live.add(self)

    rows = {"rollout": [], "executed_returned": None, "executed_dt": None}
    execute, init = sim_mpc_mod.SimMPC.execute, sim_mpc_mod.SimMPC.__init__

    planners = []  # each SimMPC's planning settings, for the throughput runs (speed_mjcf.py)

    def patched_init(self, config, sim, *a, **kw):
        if args.seed is not None:  # the config is a frozen dataclass: a copy with the seed
            config = dataclasses.replace(config, seed=args.seed)
        init(self, config, sim, *a, **kw)
        for name, s in getattr(self, "sims", {}).items():
            if isinstance(s, Recording):
                kinds[s] = name
        path = f"{args.out}_planner{len(planners)}.mjb"  # the model as the planner simulates it (the example's scene edits)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        mujoco.mj_saveModel(sim.mjm, path, None)
        planners.append({"model": path, "worlds": {k: int(v.nworld) for k, v in getattr(self, "sims", {}).items()},
                         "substeps": int(sim.substeps), "njmax": sim._args.get("njmax"), "nconmax": sim._args.get("nconmax"),
                         "timestep": float(sim.mjm.opt.timestep), "stages": repr(getattr(self, "stages", None)),
                         "num_samples": int(config.num_samples), "num_replicas": int(config.num_replicas),
                         "horizon_steps": int(config.horizon_steps), "max_iterations": int(config.max_iterations)})  # fmt: skip

    def patched_execute(self, sim, num_steps, *a, callback=None, **kw):
        if rows["executed_returned"] is not None:
            raise RuntimeError("collect_robokit: SimMPC.execute ran a second time; a run records one executed trajectory")

        def drain(infos):
            if callback is not None:
                callback(infos)
            for s in list(live):
                if str(kinds.get(s, "")).startswith("rollout"):  # a coarse-to-fine planner has one rollout simulator per stage
                    rows["rollout"].append(s.drain())
                else:
                    s.drain()  # the validation worlds and the executing simulator: emptied, kept apart below

        out = execute(self, sim, num_steps, *a, callback=drain, **kw)
        rows["executed_returned"] = out[0]
        rows["executed_dt"] = float(sim.mjm.opt.timestep) * sim.substeps  # the executing simulator's own step
        return out

    import viser

    def no_viewer(*a, **kw):
        raise _Stop

    saved = (sim_pkg.MjWarpSim, sim_mpc_mod.SimMPC.__init__, sim_mpc_mod.SimMPC.execute, viser.ViserServer, sys.argv)
    sim_pkg.MjWarpSim = Recording
    sim_mpc_mod.SimMPC.__init__, sim_mpc_mod.SimMPC.execute = patched_init, patched_execute
    sim_mpc_pkg.SimMPC = sim_mpc_mod.SimMPC
    viser.ViserServer = no_viewer
    sys.argv = [args.example] + [x for a in args.example_args or [] for x in shlex.split(a)]  # the example parses its own options (tyro), not this script's
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):  # the example's output, also kept for the record
            try:
                runpy.run_path(args.example, run_name="__main__")
            except _Stop:
                pass
    finally:  # the patches undone, for whatever runs next in this process
        sim_pkg.MjWarpSim, sim_mpc_mod.SimMPC.__init__, sim_mpc_mod.SimMPC.execute, viser.ViserServer, sys.argv = saved
    if not rows["rollout"]:
        raise RuntimeError(f"{args.example}: no SimMPC rollout was recorded")
    allrows = {k: np.concatenate([r[k] for r in rows["rollout"]]) for k in rows["rollout"][0]}
    meta = {"example": os.path.abspath(args.example), "example_args": args.example_args, "record": args.record, "seed": args.seed, "files": {},
            "versions": {"mujoco": mujoco.__version__, **_versions()}, "commit": _commit(args.example),
            "output": log.getvalue().splitlines()[-20:], "planners": planners}  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    for dt in sorted(set(np.round(allrows["dt"][:, 0].astype(np.float64), 6).tolist())):  # the nominal step, to 1 us (the ring holds float32)
        rec = recording(allrows, dt)
        path = f"{args.out}_rollout_dt{1e3 * dt:g}.npz"
        np.savez_compressed(path, **{k: rec[k] for k in ("qpos", "qvel", "ctrl", "reset", "dt")})
        meta["files"][path] = {"steps": int(rec["ctrl"].shape[0]), "worlds": int(rec["ctrl"].shape[1])}
    for i, r in enumerate(rows["executed_returned"] or []):  # the closed loop: rows before each step, the next row after it
        q, v, u = r["qpos"], r["qvel"], r["ctrl"]
        path = f"{args.out}_executed{i}.npz"
        np.savez_compressed(path, qpos=q[:, None], qvel=v[:, None], ctrl=u[:-1, None], reset=np.zeros((len(u) - 1, 1), bool),
                            dt=np.float64(_executed_dt(r, rows["executed_dt"])))  # fmt: skip
        meta["files"][path] = {"steps": len(u) - 1, "worlds": 1}
    with open(f"{args.out}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[collect_robokit] {sum(x['steps'] * x['worlds'] for x in meta['files'].values())} transitions in {len(meta['files'])} files")


def _executed_dt(rows: dict, nominal: float) -> float:
    """The executing simulator's step, checked against the trajectory's float32 time column (an error if they differ)."""
    dt = np.diff(rows["time"].reshape(len(rows["time"]), -1)[:, 0].astype(np.float64))
    if len(dt) == 0:
        raise ValueError(f"the executed trajectory has {len(rows['time'])} rows: no step to check against the simulator's {nominal} s")
    if np.abs(dt - nominal).max() > 1e-6 + 1e-4 * nominal:
        raise ValueError(f"the executed trajectory's steps ({dt.min():.6g}-{dt.max():.6g} s) are not the simulator's {nominal} s")
    return nominal


def _versions() -> dict:
    out = {}
    for name in ("warp", "mujoco_warp", "robokit", "torch"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", getattr(getattr(mod, "config", None), "version", "?"))
        except ImportError:
            out[name] = None
    return out


def _commit(path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(path)), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == "__main__":
    main()
