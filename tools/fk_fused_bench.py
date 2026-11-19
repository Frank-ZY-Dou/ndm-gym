"""Speed of the G1's forward kinematics as Triton kernels (ndm_gym/kernels/fk.py, train.py --fk_fused) against
ArticulationKinematics.forward compiled by torch.compile, in the shapes training runs them: forward and backward of all
bodies at 16,384 states (SuperTrack's world-model loss), the forward alone at 16,384 (its targets, no gradient), forward
and backward of the tracked bodies at 32 x 2048 and 32 x 1024 imagined states (the rewards of a policy update), and of all
bodies at 32 x 2048 (SuperTrack's policy loss). Each path is compiled by torch.compile, its forward and backward captured
in one CUDA graph and replayed: GPU time per call (CUDA events over --iters replays; --rounds rounds, the paths
interleaved, the median). Then the largest absolute and relative differences of the two paths' values and gradients on the
same random states and upstream gradients. The kernels compile at their first call unless Triton's cache holds them: that
time is printed first. Times on other GPUs than A100/H100-class ones are development numbers.

  python tools/fk_fused_bench.py [--cases all16k all16k_fwd tracked2048x32 tracked1024x32 all2048x32] [--rounds 7] [--iters 50]
"""

import argparse
import os
import statistics
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots import g1

CASES = {  # name: (batch shape, all bodies, backward)
    "all16k": ((16384,), True, True),
    "all16k_fwd": ((16384,), True, False),
    "tracked2048x32": ((32, 2048), False, True),
    "tracked1024x32": ((32, 1024), False, True),
    "all2048x32": ((32, 2048), True, True),
}


def states(kin, shape: tuple, seed: int) -> list[torch.Tensor]:
    """Random root states and joint angles past the limits (as tests/test_fk_fused.py), on the GPU."""
    g = torch.Generator().manual_seed(seed)
    nj, lo, hi = kin.q_lo.numel(), kin.q_lo.cpu(), kin.q_hi.cpu()
    S = [torch.randn(*shape, 3, generator=g), so3_exp(2.0 * torch.randn(*shape, 3, generator=g)), 2.0 * torch.randn(*shape, 3, generator=g),
         3.0 * torch.randn(*shape, 3, generator=g), lo + (hi - lo) * (1.6 * torch.rand(*shape, nj, generator=g) - 0.3),
         4.0 * torch.randn(*shape, nj, generator=g)]  # fmt: skip
    return [x.cuda() for x in S]


def captured(kin, S: list[torch.Tensor], gs: list[torch.Tensor], all_bodies: bool, backward: bool):
    """kin.forward compiled by torch.compile and, with ``backward``, the gradients of S under the upstream gradients gs,
    captured in one CUDA graph. Returns the graph and its static outputs (values, gradients)."""
    fwd = torch.compile(lambda *x: kin.forward(*x, all_bodies=all_bodies), fullgraph=True)

    def step():
        with torch.set_grad_enabled(backward):
            out = fwd(*S)
            return out, (torch.autograd.grad(out, S, gs) if backward else ())

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, grads = step()
    return graph, out, grads


def gpu_ms(graph: torch.cuda.CUDAGraph, iters: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | triton {triton.__version__}", flush=True)
    kin, kin_f = g1.kinematics("cuda"), g1.kinematics("cuda", fused=True)
    for name in a.cases:  # the kernels' first calls: their compilation unless Triton's cache holds them
        shape, all_bodies, backward = CASES[name]
        S = [x.requires_grad_(backward) for x in states(kin, shape, 0)]
        torch.cuda.synchronize()
        t0 = time.time()
        out = kin_f.forward(*S, all_bodies=all_bodies)
        torch.cuda.synchronize()
        t1 = time.time()
        if backward:
            torch.autograd.grad(out, S, [torch.ones_like(o) for o in out])
            torch.cuda.synchronize()
        print(f"{name}: first eager call of the kernels {t1 - t0:.1f} s forward" + (f", {time.time() - t1:.1f} s backward" if backward else ""),
              flush=True)  # fmt: skip
    for name in a.cases:
        shape, all_bodies, backward = CASES[name]
        S = [x.requires_grad_(backward) for x in states(kin, shape, 1)]
        g = torch.Generator(device="cuda").manual_seed(2)
        gs = [torch.randn(o.shape, generator=g, device="cuda") for o in kin.forward(*(x.detach() for x in S), all_bodies=all_bodies)]
        runs = {path: captured(k, S, gs, all_bodies, backward) for path, k in (("default", kin), ("fused", kin_f))}
        times = {path: [] for path in runs}
        for _ in range(a.rounds):
            for path, (graph, _, _) in runs.items():
                times[path].append(gpu_ms(graph, a.iters))
        ms = {path: statistics.median(t) for path, t in times.items()}
        print(f"{name} {shape} {'all bodies' if all_bodies else 'tracked'}, {'forward + backward' if backward else 'forward'}: default "
              f"{ms['default']:.3f} ms, fused {ms['fused']:.3f} ms ({ms['default'] / ms['fused']:.2f}x)", flush=True)  # fmt: skip
        for path, (graph, _, _) in runs.items():  # both on the same inputs, then compared
            graph.replay()
        torch.cuda.synchronize()
        (_, out_d, grad_d), (_, out_f, grad_f) = runs["default"], runs["fused"]
        labels = ["P", "R", "V", "W"] + (["dp", "dR", "dv", "dw", "dq", "dqd"] if backward else [])
        for label, d, f in zip(labels, list(out_d) + list(grad_d), list(out_f) + list(grad_f), strict=True):
            diff, scale = float((f - d).abs().max()), float(d.abs().max())
            print(f"    {label:4s} max |fused - default| {diff:.2e}, relative to the largest |default| ({scale:.3g}) {diff / max(scale, 1e-30):.2e}",
                  flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
