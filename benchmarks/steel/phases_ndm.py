"""Per-phase time of an NDM control step (benchmarks/steel/results.md: input features, network, integration) from an Nsight
Systems trace of speed_pair.py (``nsys profile --trace cuda --cuda-graph-trace node``; then ``nsys stats --report cuda_gpu_kern_sum
--format csv``), as phases.py does for Steel. Kernels are assigned to phases by the name patterns below (first match wins); the
control steps are counted by the integration kernel, which runs once per step. Prints ms per control step per phase, their
sum, and the kernels no pattern matched (setup and restarts, reported apart). Standard library only.
Usage: python phases_ndm.py <kern_sum.csv>"""

import argparse
import csv
import re

PHASES = [  # (phase, name pattern)
    ("integration", r"g1_integrate"),  # the fused integration and the compiled output step that feeds it
    ("input features", r"features_kernel|fused_.*cat_mul"),  # the features kernel and the compiled normalization of its output
    ("network", r"gemm|cutlass|_net_kernel|layer_norm|silu|addmm"),  # products, the whole-network kernel, norms, activations, biases
    ("graph input copies", r"multi_tensor_apply_kernel"),  # the compiled graph's copies of its inputs, once per step
]
STEP = "_g1_integrate_kernel"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    steps = sum(int(r["Instances"]) for r in rows if r["Name"] == STEP)
    total, unmatched, per = 0.0, [], {name: 0.0 for name, _ in PHASES}
    for r in rows:
        t = float(r["Total Time (ns)"])
        for name, pat in PHASES:
            if re.search(pat, r["Name"]):
                per[name] += t
                total += t
                break
        else:
            unmatched.append((t, int(r["Instances"]), r["Name"][:90]))
    print(f"control steps in the trace: {steps}")
    for name, t in per.items():
        print(f"{name:22s} {t / steps / 1e6:8.3f} ms per control step  {100 * t / total:5.1f}%")
    print(f"{'sum':22s} {total / steps / 1e6:8.3f} ms per control step")
    rest = sum(t for t, _, _ in unmatched)
    print(f"unmatched (setup, restarts): {rest / 1e6:.1f} ms in all, {len(unmatched)} kernels; largest:")
    for t, n, name in sorted(unmatched, reverse=True)[:6]:
        print(f"  {t / 1e6:8.2f} ms  {n:5d}x  {name}")


if __name__ == "__main__":
    main()
