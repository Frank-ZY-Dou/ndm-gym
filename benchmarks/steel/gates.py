"""The accuracy criteria of benchmarks/steel/results.md for every Steel (and MuJoCo Warp in Newton) setting scored by
``replay_engine score``: (1) no dropped row or contact in the accuracy runs (their .json beside the rollouts; the timed
runs' own check is in their rows), (2) the 8- and 32-step body-position errors against the recording at most 1.1x the
engine's reference setting's, (3) no more terminated windows than the reference beyond two standard errors, (4) mean and
peak foot penetration at most the recording's (for MuJoCo Warp, the recording's own engine, within 0.1 mm). Prints one
line per source with the errors beside the verdicts. Standard library only.
Usage: python gates.py <scores.jsonl> <acc_dir> [--ref steel_reference] [--mujoco_ref mujoco_newton] (each source's accuracy run
left <acc_dir>/<source><--suffix>.json beside its rollouts, as run.sh names them)"""

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("acc_dir")
    p.add_argument("--ref", default="steel_reference")
    p.add_argument("--mujoco_ref", default="mujoco_newton")
    p.add_argument("--suffix", default="", help="after the source's name in its accuracy run's file name")
    args = p.parse_args()
    rows = {}
    for line in open(args.scores):
        r = json.loads(line)
        rows[r["source"]] = r  # the last line of a source wins
    rec = rows["recording"]
    print(f"recording: penetration mean {rec['penetration_mean_mm']:.2f} mm, peak {rec['penetration_peak_mm']:.2f} mm, "
          f"terminated {rec['terminated_windows'][0]:.4f}")
    for name, r in rows.items():
        if name == "recording":
            continue
        engine = name.startswith("steel") or name.startswith("mujoco")  # world models are not gated
        ref = rows.get(args.mujoco_ref if name.startswith("mujoco") else args.ref) if engine else None
        e8, e32 = r["body_mm@8"][0], r["body_mm@32"][0]
        line = f"{name:24s} body mm @1 {r['body_mm@1'][0]:7.2f} @8 {e8:7.2f} @32 {e32:7.2f}  joint rad @32 {r['joint_rad@32'][0]:.4f}  " \
               f"pen mean/peak {r['penetration_mean_mm']:.2f}/{r['penetration_peak_mm']:.2f} mm  term {r['terminated_windows'][0]:.4f}"
        if ref is not None:
            js = os.path.join(args.acc_dir, f"{name}{args.suffix}.json")  # the accuracy run's record beside its rollouts
            dropped = any(c["dropped"] for c in json.load(open(js))["chunks"]) if os.path.exists(js) else None
            g1 = dropped is False and r["finite_windows"] == r["windows"]
            g2 = e8 <= 1.1 * ref["body_mm@8"][0] and e32 <= 1.1 * ref["body_mm@32"][0]
            t, t_ref = r["terminated_windows"], ref["terminated_windows"]
            g3 = t[0] <= t_ref[0] + 2 * (t[1] ** 2 + t_ref[1] ** 2) ** 0.5
            tol = 0.1 if name.startswith("mujoco") else 0.0  # the recording's own engine, within 0.1 mm
            g4 = r["penetration_mean_mm"] <= rec["penetration_mean_mm"] + tol and r["penetration_peak_mm"] <= rec["penetration_peak_mm"] + tol
            line += f"  gates {int(g1)}{int(g2)}{int(g3)}{int(g4)} {'PASS' if g1 and g2 and g3 and g4 else 'fail'}"
        print(line)


if __name__ == "__main__":
    main()
