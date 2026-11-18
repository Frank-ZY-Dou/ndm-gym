"""Throughput of one of the models of benchmarks/mbrl_lib at its control step: CPU MuJoCo, MuJoCo Warp and
the world model (``worldmodels.mjcf``) step the same number of worlds from recorded states, each world holding its recorded
controls: world-steps per second, 20 warm-up steps, then 200 timed steps between two synchronizations, the median of three
repeats (``wmbench/speed_mjcf.py``'s timing loop and its world-model row).

``cpu``: one process and one MjData; per world the state is copied in (qpos, qvel, the constraint solver's warm start),
``mj_step`` runs the control step's physics steps and the state is copied out. ``warp``: MuJoCo Warp worlds of the same model
with its default capacities, the control step's physics steps captured as one Warp graph and replayed (``sim.mbrl_ndm``'s
control step without the stages a reward reads); a row is refused if a world ran out of contacts or constraints. ``ndm``: the
world model's step (``inference_copy`` in ``--dtype``), compiled and replayed as one CUDA graph, its outputs fed back as the next
inputs; ``--ckpt`` loads a fit's, else the weights are random, which do not change the cost. The model, its control step and
its physics steps come from ``collect_mbrl``'s ``<out>.json``.

Run: ``python -m ndm_gym.wmbench.speed_mbrl --meta data/mbrl/hc_s2.json --states data/mbrl/hc_s2_executed.npz --reset_every 30
       --ckpt fits/hc_acc_ln.pt --out results/mbrl_lib/speed.csv``
"""

import argparse
import json
import os

import numpy as np

from .speed_mjcf import COLUMNS, ndm, start_states, timed


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="collect_mbrl's <out>.json (the model, its control step and physics steps)")
    p.add_argument("--states", required=True, help="a recording whose states and controls start the worlds")
    p.add_argument("--nworld", type=int, nargs="+", default=[1, 64, 512, 4096])
    p.add_argument("--what", nargs="+", default=["cpu", "warp", "ndm"], choices=["cpu", "warp", "ndm"])
    p.add_argument("--reset_every", type=int, default=0,
                   help="copy the start states back every this many steps, as a planner restarts its rollouts (0: never; every simulator alike)")
    p.add_argument("--device", default="cuda:0", help="warp: Warp's device (cpu: its CPU device, stepped without a graph)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="ndm: the network's parameters")
    p.add_argument("--fused", action="store_true",
                   help="ndm: the world model's kinematics and contact gaps on the fused path (worldmodels.mjcf.MjcfWorldModel.fused)")
    p.add_argument("--compile", default="max-autotune-no-cudagraphs", help="ndm: torch.compile's mode, or none")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024], help="ndm without --ckpt")
    p.add_argument("--wm_param", default="accel", choices=["accel", "increment"], help="ndm without --ckpt (fit_mjcf's option)")
    p.add_argument("--no_ln", action="store_true", help="ndm without --ckpt: no LayerNorm (fit_mjcf's option)")
    p.add_argument("--ckpt", default=None, help="ndm: a fit_mjcf --save file of the model's control step")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--warm_seconds", type=float, default=0.0,
                   help="after --warmup, keep calling this many seconds before the first timed run (wmbench/warm.py: an idle GPU's clock)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True, help="csv; one row per measurement is appended")
    return p.parse_args(argv)


def cpu_step(model, d, substeps: int, q: np.ndarray, v: np.ndarray, u: np.ndarray, w: np.ndarray):
    """One control step of every world on the MjData ``d``, in place: the rows of ``q``, ``v`` and ``w`` (qpos, qvel and the
    constraint solver's warm start) under the controls ``u``."""
    import mujoco

    for i in range(len(q)):
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q[i], v[i], u[i], w[i]
        mujoco.mj_step(model, d, substeps)
        q[i], v[i], w[i] = d.qpos, d.qvel, d.qacc_warmstart


def cpu(model, substeps: int, n: int, states, args) -> tuple[list[float], str]:
    import mujoco

    d = mujoco.MjData(model)
    q, v, u = (np.array(x, dtype=np.float64) for x in states)
    w = np.zeros_like(v)
    q0, v0 = q.copy(), v.copy()

    def reset():
        q[:], v[:] = q0, v0

    runs = timed(lambda: cpu_step(model, d, substeps, q, v, u, w), lambda: None, args.warmup, args.steps, args.repeats, reset, args.reset_every)
    return runs, f"mj_step x {substeps} per world in one process, reset every {args.reset_every}, mujoco {mujoco.__version__}"


def warp(model, substeps: int, n: int, states, args) -> tuple[list[float], str]:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    device = wp.get_device(args.device)
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)
    q, v, u = (x.astype(np.float32) for x in states)
    with wp.ScopedDevice(device):
        m = mjw.put_model(model)
        d = mjw.put_data(model, mjd, nworld=n)
        d.qpos.assign(q)
        d.qvel.assign(v)
        d.ctrl.assign(u)
        q0, v0 = wp.clone(d.qpos), wp.clone(d.qvel)

        def control_step():
            for _ in range(substeps):
                mjw.step(m, d)

        def reset():
            wp.copy(d.qpos, q0)
            wp.copy(d.qvel, v0)

        step = control_step
        if device.is_cuda:
            control_step()  # the kernels load outside the capture
            wp.synchronize_device()
            reset()
            with wp.ScopedCapture() as capture:
                control_step()
            step = lambda: wp.capture_launch(capture.graph)  # noqa: E731
        runs = timed(step, wp.synchronize_device, args.warmup, args.steps, args.repeats, reset, args.reset_every, args.warm_seconds)
        nacon, nefc = int(d.nacon.numpy().max()), int(d.nefc.numpy().max())
    if nacon > d.naconmax or nefc > d.njmax:
        raise RuntimeError(f"speed_mbrl warp: {nacon} contacts for a capacity of {d.naconmax}, {nefc} constraints of a world for {d.njmax}")
    return runs, (f"MuJoCo Warp {mjw.__version__} x {substeps}, naconmax {d.naconmax}, njmax {d.njmax}, reset every {args.reset_every}, "
                  f"warp {wp.config.version}")  # fmt: skip


def cpu_name() -> str:
    with open("/proc/cpuinfo") as f:
        names = [x.split(":", 1)[1].strip() for x in f if x.startswith("model name")]
    if not names:
        raise RuntimeError("speed_mbrl: /proc/cpuinfo names no CPU model")
    return names[0]


def main(argv=None):
    args = parse_args(argv)
    import mujoco

    meta = json.load(open(args.meta))
    model = mujoco.MjModel.from_binary_path(meta["model"])
    substeps = int(meta["substeps"])
    if abs(float(model.opt.timestep) * substeps - float(meta["dt"])) > 1e-12:
        raise ValueError(f"{args.meta}: a control step of {meta['dt']} s, the model's {substeps} physics steps are {model.opt.timestep * substeps} s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write(COLUMNS + "\n")
    for n in args.nworld:
        states = start_states(args.states, n)
        for what in args.what:
            if what == "cpu":
                runs, detail = cpu(model, substeps, n, states, args)
                dtype, how, device = "fp64", "none", cpu_name()
            elif what == "warp":
                runs, detail = warp(model, substeps, n, states, args)
                dtype, how, device = "fp32", "graph" if args.device != "cpu" else "none", _gpu() if args.device != "cpu" else cpu_name()
            else:
                runs, detail = ndm(model, n, states, args)
                dtype, how, device = args.dtype, args.compile, _gpu()
            sec = sorted(runs)[len(runs) // 2]
            row = [what, os.path.basename(meta["model"]), n, substeps, meta["dt"], dtype, how, args.steps, f"{sec:.5f}",
                   f"{n * args.steps / sec:.1f}", f"{1e3 * sec / args.steps:.4f}", device, detail.replace(",", ";")]  # fmt: skip
            with open(args.out, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")
            print(f"[speed_mbrl] {meta['env']} {what} N={n}: {n * args.steps / sec:,.0f} world-steps/s ({1e3 * sec / args.steps:.3f} ms per step)",
                  flush=True)  # fmt: skip


def _gpu() -> str:
    import torch

    return torch.cuda.get_device_name()


if __name__ == "__main__":
    main()
