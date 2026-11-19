"""The manipulation tracking task's world model as the simulator: its throughput beside the training's MuJoCo-Warp scene, for a
checkpoint of train_robomimic (a kept checkpoint or last.pt).

Throughput in world-steps per second, one step being one 50 ms control step: ``physics``, RobomimicSim's control step (the
scene's physics steps with the checkpoint's solver settings, as the training runs them, one CUDA graph); ``ndm_fp32``, the
checkpoint's world model (algos.robomimic_algos.world_model), its step one CUDA graph; ``ndm_bf16``, its inference copy in
bf16 (worldmodels.mjcf.inference_copy, checked there against the fp32 network), compiled (``--compile``) and one CUDA graph.
Every world starts from a frame of a rollout (``--rollout``: evaluate_robomimic's rollout_<demo>.npz of the checkpoint's
demonstration, a frame drawn per world) and holds the reference's targets of that frame. Each number is the median of
``--repeats`` runs of ``--steps`` steps after ``--warmup`` steps and ``--warm_seconds`` of calls (wmbench.speed_mjcf.timed). A
count of worlds that does not fit is written with its error.

Run: ``python -m ndm_gym.wmbench.speed_robomimic --ckpt RUN/ckpt_002500.pt --rollout RUN/final/rollout_0.npz --out speed.json``
"""

from __future__ import annotations

import argparse
import json
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rollout", required=True, help="evaluate_robomimic's rollout_<demo>.npz: the worlds' start states")
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default=None, help="the demonstrations (default: the checkpoint's)")
    p.add_argument("--nworld", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--what", nargs="+", default=["physics", "ndm_fp32", "ndm_bf16"], choices=["physics", "ndm_fp32", "ndm_bf16"])
    p.add_argument("--compile", default="max-autotune-no-cudagraphs", help="ndm_bf16: torch.compile's mode, or none")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--warm_seconds", type=float, default=5.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import mujoco_warp
    import numpy as np
    import torch
    import warp as wp

    from ndm_gym.algos.robomimic_algos import world_model
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim
    from ndm_gym.tasks.robomimic_tracking import CONTROL_DT
    from ndm_gym.train_robomimic import build
    from ndm_gym.wmbench.speed_mjcf import timed
    from ndm_gym.worldmodels.mjcf import fused_compile, inference_copy

    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    t = argparse.Namespace(**ck["args"])
    demos = t.demo if isinstance(t.demo, list) else [t.demo]
    if len(demos) != 1:
        raise ValueError("one demonstration's checkpoint: its rollout's frames are that demonstration's")
    model, task, (nconmax, njmax) = build(t.task, demos, a.data_dir or t.data_dir, dev, solver_iterations=t.solver_iterations,
                                          solver_ls_iterations=t.solver_ls_iterations)  # fmt: skip
    substeps = round(CONTROL_DT / model.opt.timestep)
    wm = world_model(model, t, "accel" if t.algo == "supertrack" else "increment", substeps).to(dev)
    wm.load_state_dict(ck["wm"])
    wm = wm.eval().requires_grad_(False)
    states = np.load(a.rollout)["state"]  # (frames, nq + nv), frame f of the demonstration at row f
    out = {"ckpt": a.ckpt, "task": t.task, "algo": t.algo, "iteration": ck["iteration"], "gpu": torch.cuda.get_device_name(0),
           "wm_kind": wm.kind, "wm_hidden": list(t.wm_hidden), "substeps": substeps, "solver_iterations": [int(model.opt.iterations),
           int(model.opt.ls_iterations)], "nconmax": nconmax, "njmax": njmax, "torch": torch.__version__,
           "mujoco_warp": getattr(mujoco_warp, "__version__", None), "warp": wp.config.version, "rows": []}  # fmt: skip
    rng = np.random.default_rng(a.seed)
    for n in a.nworld:
        f = rng.integers(0, len(states) - 1, n)
        q0, v0 = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (states[f, : task.nq], states[f, task.nq :]))
        frames = torch.tensor(f, device=dev)
        u = task.setpoint(torch.zeros(n, task.nu, device=dev), frames)  # the reference's targets of each world's frame
        for what in a.what:
            row = {"what": what, "nworld": n}
            try:
                if what == "physics":
                    sim = RobomimicSim(model, task, n, dev, nconmax=nconmax, njmax=njmax)
                    sim.qpos.copy_(q0), sim.qvel.copy_(v0), sim.ctrl.copy_(u), sim.warmstart.zero_()
                    with wp.ScopedDevice(dev):
                        runs = timed(lambda: wp.capture_launch(sim.graph), wp.synchronize_device, a.warmup, a.steps, a.repeats,  # noqa: B023
                                     warm_seconds=a.warm_seconds)  # fmt: skip
                    row["detail"] = f"RobomimicSim, {substeps} physics steps one CUDA graph, nconmax {nconmax}, njmax {njmax} per world"
                    del sim
                else:
                    m = wm if what == "ndm_fp32" else inference_copy(wm, torch.bfloat16)
                    q, v = q0.clone(), v0.clone()
                    fn = m.step if what == "ndm_fp32" or a.compile == "none" else torch.compile(m.step, mode=a.compile, dynamic=False)

                    def body():
                        q1, v1 = fn(q, v, u)  # noqa: B023
                        q.copy_(q1), v.copy_(v1)  # noqa: B023

                    with torch.no_grad():
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s), fused_compile(m):
                            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture
                                body()
                        torch.cuda.current_stream().wait_stream(s)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            body()
                        q.copy_(q0), v.copy_(v0)
                        runs = timed(g.replay, torch.cuda.synchronize, a.warmup, a.steps, a.repeats, warm_seconds=a.warm_seconds)
                    row["detail"] = ("fp32, its step one CUDA graph" if what == "ndm_fp32" else
                                     f"bf16 inference copy, compiled ({a.compile}), one CUDA graph")  # fmt: skip
                sec = statistics.median(runs)
                row.update(world_steps_per_s=a.steps * n / sec, ms_per_step=1e3 * sec / a.steps, runs_s=runs)
            except Exception as error:  # a count that does not fit, for instance: written, not hidden
                row["error"] = f"{type(error).__name__}: {error}"
                torch.cuda.empty_cache()
            out["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "runs_s"}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
