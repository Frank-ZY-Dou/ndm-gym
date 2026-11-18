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

``--copy fp32 bf16`` (default none: the above alone) also times the world model's inference copies (``worldmodels.quad_infer``: the
network in fp32 with TF32 products, as quadrotor training runs it, or in bf16 end to end), each checked against the model first (refused on
a mismatch), with its own open-loop errors on the model's course. ``--fused``: the copies' features and integration as two Triton
kernels around the network, the state kept as four tensors (p, R, v, w) in two buffers and each step one CUDA graph from one buffer
into the other (nothing packed or copied); without it, the copy's step and the copy of the next state into the state as one graph.
``--compile``: torch.compile's modes for the fused path's network or the whole step otherwise (compiled outside the capture), each
copy timed in each; ``kernel`` (with ``--fused``): the fused path's network as one Triton kernel (``kernels.mlp_fused``, the fp32 copy's
with TF32 products, the bf16 copy's in bf16), checked on the copy's first inputs against its reference (``mlp_fused.check``: TF32
products emulated in fp64, torch's bf16 run) and refused on a mismatch, its distances reported, with its own open-loop errors. ``--warm_seconds``: every path keeps stepping this long before its timed runs (``wmbench/warm.py``).
``--profile``: each part of a step as its own CUDA graph, timed alone, in milliseconds per step at ``--num_envs``: for the step above
the state's unpacking (quaternion to matrix), the world model's step, and the packing (matrix to quaternion, the copy into the
state); for the fused copies the state's packing into the network's input (the features with the input's normalization), the
network, and the integration (the output's de-normalization, the next state). ``--data_dir``: the flights' directory on this machine
(default: the checkpoint's).

Run: ``python -m ndm_gym.wmbench.speed_quadrotor --ckpt logs/<run>/last.pt --out wm_speed.json``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
import types


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
    p.add_argument("--copy", type=str, nargs="*", default=[], choices=["fp32", "bf16"], help="also time the world model's inference copies")
    p.add_argument("--fused", action="store_true", help="with --copy: the features and the integration as Triton kernels")
    p.add_argument("--compile", type=str, nargs="+", default=["none"], choices=["none", "default", "max-autotune-no-cudagraphs", "kernel"],
                   help="with --copy: torch.compile's modes for the copies' network (--fused) or their whole step, each timed; kernel: "
                        "the network as one Triton kernel (--fused)")  # fmt: skip
    p.add_argument("--data_dir", type=str, default=None, help="the flights' directory (default: the checkpoint's)")
    p.add_argument("--warm_seconds", type=float, default=0.0,
                   help="before each path's timed runs, keep stepping this many seconds (wmbench/warm.py: an idle GPU's clock)")  # fmt: skip
    p.add_argument("--profile", action="store_true", help="each part of a step as its own CUDA graph, timed alone")
    args = p.parse_args()
    if not args.copy and (args.fused or args.compile != ["none"]):
        p.error("--fused and --compile build the inference copies: give --copy fp32 and/or bf16")
    if "kernel" in args.compile and not args.fused:
        p.error("--compile kernel runs the fused path's network as one kernel: give --fused")

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.quadrotor_algos import QUAD_ALGOS
    from ndm_gym.robots.quadrotor import NUM_U
    from ndm_gym.sim.rigid.quadrotor_sim import QuadSim
    from ndm_gym.tasks.quadrotor_tracking import flatten_state, unflatten_state
    from ndm_gym.train_quadrotor import build
    from ndm_gym.wmbench.warm import keep_busy

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    dev, N = "cuda:0", args.num_envs
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    targs.update_graphs = False
    model, _, task = build(targs.flight, args.data_dir or targs.data_dir, dev, goal_stride=targs.goal_stride)
    sim = QuadSim(model, task, N, dev, episode=max(args.horizons) + 1)
    algo = QUAD_ALGOS[targs.algo](targs, task, dev, substeps=sim.substeps)
    algo.wm.load_state_dict(ck["wm"]), algo.pi.load_state_dict(ck["pi"])
    wm = algo.wm.eval()
    out = {"ckpt": args.ckpt, "gpu": torch.cuda.get_device_name(0), "num_envs": N, "wm_hidden": targs.wm_hidden, "wm_kind": wm.kind,
           "torch": torch.__version__, "tf32": torch.backends.cuda.matmul.allow_tf32, "warm_seconds": args.warm_seconds}  # fmt: skip

    # ---- open-loop error (before the timing, which leaves the simulator's worlds anywhere)
    with torch.no_grad():
        H = max(args.horizons)
        s0, f = sim.state(), sim.f.clone()
        a_prev, alive = torch.zeros(N, NUM_U, device=dev), torch.ones(N, dtype=torch.bool, device=dev)
        S_hat, s, rows = unflatten_state(s0), s0, {}
        record = {"u": [], "s": [], "alive": {}}  # --copy: the simulator's course, for the copy's open-loop errors
        for h in range(1, H + 1):
            a = algo.act(unflatten_state(s), a_prev, f, noise=args.action_noise)
            u = task.setpoint(a, f)
            s, _, done = sim.step(u, reset=False)
            S_hat = wm.step(S_hat, u)
            alive &= ~done | (h == H)  # (the episodes last H + 1 steps: none ends by its length before the last horizon)
            f, a_prev = task.ref.next_frame(f), a
            if args.copy:
                record["u"].append(u)
                record["s"].append(s)
            if h in args.horizons:
                S = unflatten_state(s)
                e = wm.increments(S, S_hat)[alive]
                rows[h] = {"worlds": int(alive.sum()), "pos_mm": 1000 * float((S_hat[0] - S[0])[alive].norm(dim=-1).mean()),
                           "rot_deg": math.degrees(float(e[:, 3:6].norm(dim=-1).mean())), "vel_m_s": float(e[:, 6:9].norm(dim=-1).mean())}  # fmt: skip
                record["alive"][h] = alive.clone()
        out["open_loop"] = rows
        del S_hat

    @torch.no_grad()
    def open_loop(model) -> dict:  # a model's open-loop errors on the recorded course, as the model's above
        S_hat, res = unflatten_state(s0), {}
        for h in range(1, H + 1):
            S_hat = model.step(S_hat, record["u"][h - 1])
            if h in args.horizons:
                S, al = unflatten_state(record["s"][h - 1]), record["alive"][h]
                e = wm.increments(S, S_hat)[al]
                res[h] = {"worlds": int(al.sum()), "pos_mm": 1000 * float((S_hat[0] - S[0])[al].norm(dim=-1).mean()),
                          "rot_deg": math.degrees(float(e[:, 3:6].norm(dim=-1).mean())), "vel_m_s": float(e[:, 6:9].norm(dim=-1).mean())}  # fmt: skip
        return res

    # ---- throughput
    def timed(step) -> float:
        rates = []
        keep_busy(step, torch.cuda.synchronize, args.warm_seconds)
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

    def graphed(fn):  # fn as one CUDA graph (three calls first on a side stream: compilation, the allocator), and its replay
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g.replay

    def ms(fn) -> float:  # milliseconds per call of a graph's replay
        return N / timed(fn) * 1000

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
        if args.profile:  # the step above part by part, each part its own graph
            with torch.no_grad():
                S0 = unflatten_state(state)
                nxt = wm.step(S0, u)
                parts = {"unpack (quaternion to matrix)": lambda: unflatten_state(state), "world model step": lambda: wm.step(S0, u),  # noqa: B023
                         "pack (matrix to quaternion, copy)": lambda: state.copy_(flatten_state(nxt))}  # noqa: B023  # fmt: skip
                out[f"profile_{name}_ms"] = {k: ms(graphed(f)) for k, f in parts.items()}
                out[f"profile_{name}_ms"]["whole step"] = ms(graph.replay)
            state.copy_(flatten_state(task.ref.state(sim.f)))
    nets.AMP_DTYPE = None
    from ndm_gym.worldmodels.quad_infer import inference_copy

    bufs = [torch.empty(N, 18, device=dev) for _ in range(2)]  # two states (p, R, v, w) as views of a buffer each
    Sb = [(b[:, 0:3], b[:, 3:12].view(N, 3, 3), b[:, 12:15], b[:, 15:18]) for b in bufs]
    S0 = unflatten_state(flatten_state(task.ref.state(sim.f)))

    def start():
        for t, t0 in zip(Sb[0], S0, strict=True):
            t.copy_(t0)

    for kind, mode in itertools.product(args.copy, args.compile):
        cp = inference_copy(wm, torch.bfloat16 if kind == "bf16" else None, fused=args.fused)
        name = f"copy_{kind}" + ("_fused" if args.fused else "") + {"none": "", "kernel": "_kernel"}.get(mode, "_compiled")
        with torch.no_grad():
            start()
            if args.fused and mode == "kernel":  # the network as one kernel, checked on the copy's inputs at the start
                from ndm_gym.kernels.mlp_fused import FusedMLP, check

                seq = cp.wm.net.net if cp.bf16 else cp.wm.net
                net = FusedMLP(seq, "bf16" if cp.bf16 else "tf32")
                out[name + "_check"] = check(seq, net, cp.features(Sb[0], u))
                out[name + "_config"] = dict(zip(("bm", "ch"), net.config(N), strict=True))
            elif args.fused:
                net = cp.wm.net if mode == "none" else torch.compile(cp.wm.net, mode=mode, dynamic=False)
            if args.fused:  # two graphs, each a step from one buffer into the other: nothing is copied
                graphs = [graphed(lambda: cp.step_into(Sb[0], u, Sb[1], net)), graphed(lambda: cp.step_into(Sb[1], u, Sb[0], net))]  # noqa: B023
            else:  # one graph: the step and the copy of the next state into the state

                def body(p_, R_, v_, w_, u_):
                    for t, t1 in zip((p_, R_, v_, w_), cp.step((p_, R_, v_, w_), u_), strict=True):  # noqa: B023
                        t.copy_(t1)

                fn = body if mode == "none" else torch.compile(body, mode=mode, dynamic=False)
                graphs = [graphed(lambda: fn(*Sb[0], u))]  # noqa: B023
            start()
            turn = itertools.cycle(graphs)

            def replay():
                next(turn)()  # noqa: B023

            out[name + "_steps_per_s"] = timed(replay)
            out[name + "_speedup"] = out[name + "_steps_per_s"] / out["physics_steps_per_s"]
            if args.profile and args.fused:  # each part of the fused step its own graph
                start()
                x = cp.features(Sb[0], u)
                y = net(x)
                parts = {"state packing (features, normalization)": lambda: cp.features(Sb[0], u), "network": lambda: net(x),  # noqa: B023
                         "integration (de-normalization, next state)": lambda: cp.integrate_into(y, Sb[0], Sb[1])}  # noqa: B023  # fmt: skip
                out[f"profile_{name}_ms"] = {k: ms(graphed(f)) for k, f in parts.items()}
                out[f"profile_{name}_ms"]["whole step"] = ms(replay)
        # the copy's open-loop errors on the course the model's were measured on (the compiled network is not used there; the kernel is)
        if mode == args.compile[0]:
            out[f"open_loop_copy_{kind}" + ("_fused" if args.fused else "")] = open_loop(cp)
        if mode == "kernel":
            out[f"open_loop_copy_{kind}_fused_kernel"] = open_loop(types.SimpleNamespace(step=lambda S, u_: cp.integrate(net(cp.features(S, u_)), S)))  # noqa: B023
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
