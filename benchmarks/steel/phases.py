"""Per-phase time of a Steel control step (benchmarks/steel/results.md) from an Nsight Systems trace of steel_g1.py
(``nsys profile --trace cuda --cuda-graph-trace node``; then ``nsys stats --report cuda_gpu_kern_sum --format csv``).
Kernels are assigned to phases by the name patterns below (first match wins); the control steps are counted by
the collision's AABB kernel, which runs once per collision. Prints ms per control step per phase, their sum, and the kernels
no pattern matched (setup and restarts, reported apart). Standard library only.
Usage: python phases.py <kern_sum.csv> [--collisions_per_step 1]"""

import argparse
import csv
import re

PHASES = [  # (phase, name pattern)
    ("collision", r"narrow_phase|broadphase|compute_shape_aabbs|verify_narrow_phase_buffers"),
    ("articulation dynamics", r"tree_fk|tree_tau|augmented_drive|crba|sparse_mass_matrix|sparse_factor|composite_inertia|velocity_predictor|"
                              r"free_root_transport|update_qdd|masked_body_inertia|mass_update_mask|update_body_qd_from_featherstone"),
    ("row build", r"contact_response|contact_slots|contact_rows|contact_bias|joint_limit_rows|contact_restitution|diag_cfm|"
                  r"prepare_world_impulses|pack_mf_meta|clear_dense_row_state|constraint_counts"),
    ("sweep", r"pgs_solve"),
    ("integration", r"integrate_generalized_joints"),
    ("status checks", r"constraint_status|disabled_tree_joints|constraint_row_overflow|memtile_value_kernel<int>"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--collisions_per_step", type=int, default=1, help="1 with --collide control, the substeps with --collide step")
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    steps = sum(int(r["Instances"]) for r in rows if "compute_shape_aabbs" in r["Name"]) / args.collisions_per_step
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
    print(f"control steps in the trace: {steps:.0f}")
    for name, t in per.items():
        print(f"{name:22s} {t / steps / 1e6:8.3f} ms per control step  {100 * t / total:5.1f}%")
    print(f"{'sum':22s} {total / steps / 1e6:8.3f} ms per control step")
    rest = sum(t for t, _, _ in unmatched)
    print(f"unmatched (setup, restarts): {rest / 1e6:.1f} ms in all, {len(unmatched)} kernels; largest:")
    for t, n, name in sorted(unmatched, reverse=True)[:6]:
        print(f"  {t / 1e6:8.2f} ms  {n:5d}x  {name}")


if __name__ == "__main__":
    main()
