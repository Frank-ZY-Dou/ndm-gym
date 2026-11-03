"""Time one muscle-force evaluation of MuscleVAE's character for batches of environments: one call of
ndm_gym.robots.muscles.MuscleModel.substep (anchor positions, lengths and rates, Hill curves, PD force within the fatigue
bounds, Hill inversion, fatigue update, anchor forces and per-body wrenches), the work done per environment and physics
substep. Body poses are the T-pose with every body turned and moved at random; needs the converted assets
(tools/convert_musclevae.py).

  python tools/bench_muscles.py --device cpu --envs 1 64 1024
  python tools/bench_muscles.py --device cuda --envs 1 64 1024 4096
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots import musclevae_char as mc


def inputs(model, n: int, device: torch.device, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    com = torch.tensor([b["com_tpose"] for b in mc.load_character()["bodies"]], dtype=torch.float64)
    pos = com + 0.02 * torch.randn(n, mc.NUM_BODIES, 3, generator=g, dtype=torch.float64)
    rot = so3_exp(0.2 * torch.randn(n, mc.NUM_BODIES, 3, generator=g, dtype=torch.float64))
    x = -torch.log(torch.rand(n, mc.NUM_MUSCLES, 3, generator=g, dtype=torch.float64))
    fatigue = (x / x.sum(-1, keepdim=True)).unbind(-1)
    to = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
    pos, rot = to(pos), to(rot)
    length = model.lengths(model.anchor_positions(pos, rot))
    prev = length * to(1 + 0.01 * torch.randn(n, mc.NUM_MUSCLES, generator=g, dtype=torch.float64))
    target = model.target_length(to(0.2 * torch.randn(n, mc.NUM_MUSCLES, generator=g, dtype=torch.float64)))
    return pos, rot, prev, target, *(to(f) for f in fatigue)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 64, 1024])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--model", default="released", choices=["released", "paper"])
    ap.add_argument("--calls", type=int, default=20, help="calls per timed round")
    ap.add_argument("--rounds", type=int, default=15, help="timed rounds; the median is reported")
    ap.add_argument("--compile", action="store_true", help="torch.compile the substep (one graph per batch size)")
    a = ap.parse_args()
    device, dtype = torch.device(a.device), getattr(torch, a.dtype)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    model = mc.muscle_model(a.model, device, dtype)
    step = torch.compile(model.substep, dynamic=False) if a.compile else model.substep
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else f"{torch.get_num_threads()} threads"
    mode = " (torch.compile)" if a.compile else ""
    print(f"MuscleModel.substep{mode}, {a.model} parameters, {a.dtype}, {device} ({name}), torch {torch.__version__}")
    print(f"{'envs':>6s} {'ms / call':>10s} {'us / env':>9s} {'env-substeps / s':>17s}")
    for n in a.envs:
        args = inputs(model, n, device, dtype)
        with torch.no_grad():
            for _ in range(5):
                step(*args)
            times = []
            for _ in range(a.rounds):
                sync()
                t0 = time.perf_counter()
                for _ in range(a.calls):
                    step(*args)
                sync()
                times.append((time.perf_counter() - t0) / a.calls)
        t = statistics.median(times)
        print(f"{n:6d} {1e3 * t:10.3f} {1e6 * t / n:9.2f} {n / t:17.0f}")


if __name__ == "__main__":
    main()
