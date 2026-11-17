"""The quadrotor's world model as the simulator: its throughput beside MuJoCo-Warp's and its open-loop error, for a
checkpoint of quadrotor training (train_quadrotor's last.pt).

Throughput: control steps (10 ms) per second times the number of worlds, the command held fixed. ``physics``: QuadSim's
control step, its four physics steps one CUDA graph. ``ndm``: one step of the checkpoint's world model as one CUDA graph, in
fp32 and with the MLP's products in bf16. Each is the median of ``--repeats`` runs of ``--steps`` steps after ``--warmup``,
the GPU synchronized around the timed loop. Development numbers only on a shared GPU.

Open-loop error: the policy flies ``--num_envs`` worlds in the simulator from random frames of the checkpoint's flights
(training's reset noise, exploration noise ``--action_noise``); the world model is rolled out from each start under the same
commands, and the position (mm), rotation (deg) and velocity (m/s) errors of its state against the simulator's are averaged
at the horizons ``--horizons``, over the worlds whose episodes did not end.

Run: ``python -m ndm_gym.wmbench.speed_quadrotor --ckpt logs/<run>/last.pt --out wm_speed.json``
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5, help="timed runs; the median is reported")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32, 64, 100])
    p.add_argument("--action_noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.quadrotor_algos import QUAD_ALGOS
    from ndm_gym.robots.quadrotor import NUM_U
    from ndm_gym.sim.rigid.quadrotor_sim import QuadSim
    from ndm_gym.tasks.quadrotor_tracking import flatten_state, unflatten_state
    from ndm_gym.train_quadrotor import build

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    dev, N = "cuda:0", args.num_envs
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    targs.update_graphs = False
    model, _, task = build(targs.flight, targs.data_dir, dev, goal_stride=targs.goal_stride)
    sim = QuadSim(model, task, N, dev, episode=max(args.horizons) + 1)
    algo = QUAD_ALGOS[targs.algo](targs, task, dev, substeps=sim.substeps)
    algo.wm.load_state_dict(ck["wm"]), algo.pi.load_state_dict(ck["pi"])
    wm = algo.wm.eval()
    out = {"ckpt": args.ckpt, "gpu": torch.cuda.get_device_name(0), "num_envs": N, "wm_hidden": targs.wm_hidden, "wm_kind": wm.kind}

    # ---- open-loop error (before the timing, which leaves the simulator's worlds anywhere)
    with torch.no_grad():
        H = max(args.horizons)
        s0, f = sim.state(), sim.f.clone()
        a_prev, alive = torch.zeros(N, NUM_U, device=dev), torch.ones(N, dtype=torch.bool, device=dev)
        S_hat, s, rows = unflatten_state(s0), s0, {}
        for h in range(1, H + 1):
            a = algo.act(unflatten_state(s), a_prev, f, noise=args.action_noise)
            u = task.setpoint(a, f)
            s, _, done = sim.step(u, reset=False)
            S_hat = wm.step(S_hat, u)
            alive &= ~done | (h == H)  # (the episodes last H + 1 steps: none ends by its length before the last horizon)
            f, a_prev = task.ref.next_frame(f), a
            if h in args.horizons:
                S = unflatten_state(s)
                e = wm.increments(S, S_hat)[alive]
                rows[h] = {"worlds": int(alive.sum()), "pos_mm": 1000 * float((S_hat[0] - S[0])[alive].norm(dim=-1).mean()),
                           "rot_deg": math.degrees(float(e[:, 3:6].norm(dim=-1).mean())), "vel_m_s": float(e[:, 6:9].norm(dim=-1).mean())}  # fmt: skip
        out["open_loop"] = rows
        del S_hat

    # ---- throughput
    def timed(step) -> float:
        rates = []
        for _ in range(args.repeats):
            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                step()
            torch.cuda.synchronize()
            rates.append(args.steps * N / (time.perf_counter() - t0))
        return statistics.median(rates)

    import warp as wp

    u = task.setpoint(torch.zeros(N, NUM_U, device=dev), sim.f)
    sim.ctrl.copy_(u)

    def physics():
        wp.capture_launch(sim.graph)

    out["physics_steps_per_s"] = timed(physics)
    wp.synchronize_device(dev)
    state = flatten_state(task.ref.state(sim.f)).clone()

    @torch.no_grad()
    def ndm_step():
        state.copy_(flatten_state(wm.step(unflatten_state(state), u)))

    for name, dtype in (("ndm_fp32", None), ("ndm_bf16", torch.bfloat16)):
        nets.AMP_DTYPE = dtype
        for _ in range(3):
            ndm_step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ndm_step()
        out[name + "_steps_per_s"] = timed(graph.replay)
        out[name + "_speedup"] = out[name + "_steps_per_s"] / out["physics_steps_per_s"]
        state.copy_(flatten_state(task.ref.state(sim.f)))
    nets.AMP_DTYPE = None
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
