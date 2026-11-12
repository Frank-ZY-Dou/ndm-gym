"""Throughput of a MuJoCo model's step as RoboKit's MuJoCo Warp simulator runs it and of its world model (``worldmodels.mjcf``)
at the same number of worlds and the same step: world-steps per second, 20 warm-up steps, then 200 timed steps between two
synchronizations, the median of three repeats (as ``wmbench/speed_hoi.py``).

``physics``: ``robokit.sim.MjWarpSim`` with the model, the worlds, the substeps and the contact capacities the planner used
(``collect_robokit``'s ``<out>.json``), its step a cached CUDA graph, every world started from a recorded state (so its contacts
are the task's) and holding its recorded controls. ``ndm``: the world model's step (``inference_copy`` in ``--dtype``) on the
same states and controls, compiled (``--compile``) and replayed as one CUDA graph per step, its outputs fed back as the next
inputs. The world model's weights do not change its cost; ``--ckpt`` loads a fit's, else the weights are random. ``ndmsim``:
the simulator SimMPC plans on (``sim.robokit_ndm.NdmSim``), its whole step (the ``--derived`` MuJoCo Warp stages, then the world
model) captured into one Warp graph as SimMPC captures it, and replayed.

Run: ``python -m ndm_gym.wmbench.speed_mjcf --meta data/robokit/g1_carry_s0.json --planner 0 --states data/.._rollout_dt20.npz
       --what physics ndm --out results/robokit_ndm/speed.csv``
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np

NDMSIM_COMPILE = "max-autotune-no-cudagraphs"  # sim.robokit_ndm.COMPILE_MODE, named here without importing Warp
COLUMNS = "what,model,nworld,substeps,dt,dtype,compile,steps,seconds,world_steps_per_s,ms_per_step,device,detail"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="collect_robokit's <out>.json (the planner's model and settings)")
    p.add_argument("--planner", type=int, default=0, help="which of the run's planners")
    p.add_argument("--states", required=True, help="a recording whose states and controls start the worlds")
    p.add_argument("--nworld", type=int, nargs="+", default=None, help="worlds (default: the planner's rollout worlds)")
    p.add_argument("--what", nargs="+", default=["physics", "ndm"], choices=["physics", "ndm", "ndmsim"])
    p.add_argument("--derived", nargs="*", default=["kinematics"], help="ndmsim: the MuJoCo Warp stages the task's costs read")
    p.add_argument("--stage_dt", type=float, default=None, help="a coarse stage's step (default: the planner's timestep); physics at it")
    p.add_argument("--reset_every", type=int, default=0,
                   help="copy the start states back every this many steps, as a planner restarts its rollouts (0: never; every simulator alike)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="ndm: the network's parameters")
    p.add_argument("--compile", default="max-autotune-no-cudagraphs", help="ndm: torch.compile's mode, or none")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024], help="ndm without --ckpt")
    p.add_argument("--wm_param", default="accel", choices=["accel", "increment"], help="ndm without --ckpt (fit_mjcf's option)")
    p.add_argument("--no_ln", action="store_true", help="ndm without --ckpt: no LayerNorm (fit_mjcf's option)")
    p.add_argument("--ckpt", default=None, help="ndm: a fit_mjcf --save file")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True, help="csv; one row per measurement is appended")
    return p.parse_args(argv)


def start_states(path: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n states and controls drawn from a recording's steps (with replacement when it holds fewer)."""
    z = np.load(path)
    q, v, u = z["qpos"][:-1].reshape(-1, z["qpos"].shape[-1]), z["qvel"][:-1].reshape(-1, z["qvel"].shape[-1]), z["ctrl"].reshape(-1, z["ctrl"].shape[-1])
    idx = np.random.default_rng(seed).choice(len(q), size=n, replace=n > len(q))
    return q[idx], v[idx], u[idx]


def timed(step, sync, warmup: int, steps: int, repeats: int, reset=None, every: int = 0) -> list[float]:
    """Seconds for ``steps`` calls of ``step`` between two syncs, ``repeats`` times, after ``warmup`` calls; with ``every``,
    ``reset()`` (the start states copied back on the device) before every ``every``-th call, inside the timed loop too."""
    calls = 0

    def one():
        nonlocal calls
        if every and calls % every == 0:
            reset()
        step()
        calls += 1

    for _ in range(warmup):
        one()
    sync()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            one()
        sync()
        out.append(time.perf_counter() - t0)
    return out


def physics(model, planner: dict, n: int, states, args) -> tuple[list[float], str]:
    import mujoco
    import warp as wp
    from robokit.sim import MjWarpSim

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = MjWarpSim(model, n, data, substeps=planner["substeps"], njmax=planner["njmax"], nconmax=planner["nconmax"])
    if args.stage_dt is not None:  # a coarse stage: SimMPC writes the longer timestep before each of its steps
        sim.opt.timestep.fill_(args.stage_dt / planner["substeps"])
    q, v, u = states
    sim.data.qpos.assign(q.astype(np.float32))
    sim.data.qvel.assign(v.astype(np.float32))
    sim.data.ctrl.assign(u.astype(np.float32))
    sim.forward()
    q0, v0 = wp.clone(sim.data.qpos), wp.clone(sim.data.qvel)

    def reset():
        wp.copy(sim.data.qpos, q0)
        wp.copy(sim.data.qvel, v0)

    runs = timed(sim.step, wp.synchronize_device, args.warmup, args.steps, args.repeats, reset, args.reset_every)
    return runs, f"MjWarpSim, njmax {planner['njmax']}, nconmax {planner['nconmax']}, reset every {args.reset_every}, warp {wp.config.version}"


def ndmsim(model, planner: dict, n: int, states, args) -> tuple[list[float], str]:
    import mujoco
    import torch
    import warp as wp

    from ndm_gym.sim.robokit_ndm import ndm_sim
    from ndm_gym.worldmodels.mjcf import MjcfLayout, inference_copy

    dev = torch.device("cuda")
    wm = load_ndm(MjcfLayout.from_model(model), args, dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = ndm_sim()(model, n, data, substeps=planner["substeps"], njmax=planner["njmax"], nconmax=planner["nconmax"],
                    ndm={wm.lay.dt: inference_copy(wm, dtype)}, derived=tuple(args.derived))  # fmt: skip
    if args.stage_dt is not None and abs(wm.lay.dt - args.stage_dt) > 1e-9:
        raise ValueError(f"ndmsim: the world model steps {wm.lay.dt} s, --stage_dt is {args.stage_dt} s")
    if abs(wm.lay.dt - planner["timestep"] * planner["substeps"]) > 1e-9:
        sim.host_dt = round(wm.lay.dt, 9)  # a coarse stage's model: the planner's timestep times its ticks
        sim.opt.timestep.fill_(wm.lay.dt / planner["substeps"])
    q, v, u = states

    def start():
        sim.data.qpos.assign(q.astype(np.float32))
        sim.data.qvel.assign(v.astype(np.float32))
        sim.data.ctrl.assign(u.astype(np.float32))

    start()
    sim.step()  # once outside the capture first, as SimMPC runs each step body before capturing it (the kernels load here)
    wp.synchronize_device()
    start()
    q0, v0 = wp.clone(sim.data.qpos), wp.clone(sim.data.qvel)

    def reset():
        wp.copy(sim.data.qpos, q0)
        wp.copy(sim.data.qvel, v0)

    with wp.ScopedDevice(sim.device):
        with wp.ScopedCapture() as capture:  # as SimMPC captures a step
            sim.step()
    runs = timed(lambda: wp.capture_launch(capture.graph), wp.synchronize_device, args.warmup, args.steps, args.repeats, reset, args.reset_every)
    sim.check()
    return runs, (f"NdmSim (derived {'+'.join(args.derived) or 'none'}), MjcfWorldModel({wm.kind}) in {args.dtype}, reset every {args.reset_every}, "
                  f"warp {wp.config.version}")


def ndm(model, n: int, states, args) -> tuple[list[float], str]:
    import torch

    from ndm_gym.worldmodels.mjcf import MjcfLayout, inference_copy

    dev = torch.device("cuda")
    wm = load_ndm(MjcfLayout.from_model(model), args, dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    m = inference_copy(wm, dtype)
    q, v, u = (torch.tensor(x, dtype=torch.float32, device=dev) for x in states)
    q0, v0 = q.clone(), v.clone()
    fn = m.step if args.compile == "none" else torch.compile(m.step, mode=args.compile, dynamic=False)

    def body():
        q1, v1 = fn(q, v, u)
        q.copy_(q1)
        v.copy_(v1)

    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        runs = timed(g.replay, torch.cuda.synchronize, args.warmup, args.steps, args.repeats, lambda: (q.copy_(q0), v.copy_(v0)), args.reset_every)
    return runs, f"MjcfWorldModel({wm.kind}) {list(wm.net[0].weight.shape)[::-1]} in {args.dtype}, reset every {args.reset_every}, torch {torch.__version__}"


def load_ndm(lay, args, device):
    """The world model to time: a fit_mjcf --save file's (its parameterization, widths, activation, LayerNorm and step), or one
    of random weights with this script's options. A checkpoint written before fit_mjcf had --wm_param is an acceleration model.
    A checkpoint fitted on another layout (its state's sizes, its anchor or its objects, as with fit_mjcf --anchor) is refused."""
    import torch

    from ndm_gym.worldmodels.mjcf import MjcfWorldModel

    if not args.ckpt:
        return MjcfWorldModel(lay, hidden=tuple(args.hidden), layer_norm=not args.no_ln, kind=args.wm_param).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    saved = json.loads(ck["layout"])
    for k in ("nq", "nv", "nu", "anchor", "objects"):  # the frame and the state the network was fitted in (fit_mjcf --anchor)
        if json.loads(json.dumps(getattr(lay, k))) != saved[k]:
            raise ValueError(f"load_ndm: {args.ckpt} was fitted on a layout with {k} {saved[k]}, this model's is {getattr(lay, k)}")
    if a.get("dt") is not None:  # fitted to a coarse stage's longer step
        lay = dataclasses.replace(lay, dt=a["dt"])
    wm = MjcfWorldModel(lay, hidden=tuple(a["hidden"]), act=a["act"], layer_norm=not a["no_ln"], kind=a.get("wm_param", "accel")).to(device)
    wm.load_state_dict(ck["state"])
    return wm


def main(argv=None):
    args = parse_args(argv)
    import mujoco

    meta = json.load(open(args.meta))
    planner = meta["planners"][args.planner]
    model = mujoco.MjModel.from_binary_path(planner["model"])
    worlds = args.nworld or [planner["worlds"]["rollout"]]
    dt = args.stage_dt if args.stage_dt is not None else planner["timestep"] * planner["substeps"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if new:
            f.write(COLUMNS + "\n")
    for n in worlds:
        states = start_states(args.states, n)
        for what in args.what:
            if what == "physics":
                runs, detail = physics(model, planner, n, states, args)
            elif what == "ndm":
                runs, detail = ndm(model, n, states, args)
            else:
                runs, detail = ndmsim(model, planner, n, states, args)
            sec = sorted(runs)[len(runs) // 2]
            row = [what, os.path.basename(planner["model"]), n, planner["substeps"], dt, args.dtype if what != "physics" else "fp32",
                   {"physics": "graph", "ndmsim": NDMSIM_COMPILE}.get(what, args.compile), args.steps, f"{sec:.5f}", f"{n * args.steps / sec:.1f}",
                   f"{1e3 * sec / args.steps:.4f}", _device(), detail.replace(",", ";")]  # fmt: skip
            with open(args.out, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")
            print(f"[speed_mjcf] {what} N={n}: {n * args.steps / sec:,.0f} world-steps/s ({1e3 * sec / args.steps:.3f} ms per step)", flush=True)


def _device() -> str:
    import torch

    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
