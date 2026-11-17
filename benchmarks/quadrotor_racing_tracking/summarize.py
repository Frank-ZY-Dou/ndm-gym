"""The benchmark's tables from its runs (run.sh): per configuration and task, the mean and the sample SD over the training
seeds of each evaluation's numbers (ndm_gym.evaluate_quadrotor's metrics.json) and of the training time.

    python benchmarks/quadrotor_racing_tracking/summarize.py NAME=logs/quad/<configuration> [NAME=...] [--tasks ellipse ...]

A configuration's directory holds ``<task>/seed_<s>/`` with ``evaluation/metrics.json`` and ``train_log.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

COLUMNS = (("success_rate", "success", 100, "%", 1), ("pos_mm", "position", 1, "mm", 2), ("pos_max_mm", "position, largest after 1 s", 1, "mm", 0),
           ("rot_deg", "rotation", 1, "deg", 2), ("vel_m_s", "velocity", 1, "m/s", 3))  # fmt: skip


def cell(xs: list[float], digits: int, unit: str) -> str:
    """Mean ± sample SD of the seeds' values (one seed: the value)."""
    if not xs:
        return ""
    mean = statistics.fmean(xs)
    sd = f" ± {statistics.stdev(xs):.{digits}f}" if len(xs) > 1 else ""
    return f"{mean:.{digits}f}{sd} {unit}"


def load(root: str, task: str) -> list[dict]:
    """A task's seeds: each the evaluation's mean over the task's flights (the policy from perturbed starts, and under the
    reference's command), the training time in minutes and the iterations per second of the log's last row."""
    out = []
    for run in sorted(glob.glob(os.path.join(root, task, "seed_*"))):
        path = os.path.join(run, "evaluation", "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        with open(os.path.join(run, "train_log.csv")) as f:
            rows = list(csv.DictReader(f))
        flights = list(m["flights"].values())
        mean = lambda key, k: statistics.fmean(fl[key][k] for fl in flights if fl[key][k] == fl[key][k])  # noqa: E731  (NaN: no success)
        seed = {"seed": int(run.rsplit("_", 1)[1]), "minutes": m["wall_time_s"] / 60, "it_per_s": float(rows[-1]["it_per_s"]),
                "iterations": m["iteration"], "policy": {}, "reference_command": {}}  # fmt: skip
        for key in ("policy", "reference_command"):
            if all(key in fl for fl in flights):
                seed[key] = {"flown_s": mean(key, "flown_s"), "success_rate": mean(key, "success_rate")}
                if seed[key]["success_rate"] > 0:
                    seed[key].update({k: mean(key, k) for k, *_ in COLUMNS[1:]})
        out.append(seed)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="+", help="NAME=directory, in the table's order")
    parser.add_argument("--tasks", nargs="+", default=["ellipse", "lemniscate", "race_track"])
    args = parser.parse_args()
    print("| task | | seeds | " + " | ".join(c[1] for c in COLUMNS) + " | training | iterations/s |")
    print("|---" * (len(COLUMNS) + 5) + "|")
    for task in args.tasks:
        first, reference = True, None
        for spec in args.configs:
            name, root = spec.split("=", 1)
            seeds = load(root, task)
            if not seeds:
                continue
            reference = reference or next((s["reference_command"] for s in seeds if s["reference_command"]), None)
            if first and reference:  # the open-loop rows of the first configuration's first seed
                print(f"| `{task}` | reference's command, open loop | | {100 * reference['success_rate']:.1f} % | "
                      f"(in the air for {reference['flown_s']:.1f} s) | | | | | |")  # fmt: skip
            cells = [cell([s["policy"][k] * scale for s in seeds if k in s["policy"]], digits, unit) for k, _, scale, unit, digits in COLUMNS]
            print(f"| {'' if reference or not first else f'`{task}`'} | {name} | {', '.join(str(s['seed']) for s in seeds)} | " + " | ".join(cells)
                  + f" | {cell([s['minutes'] for s in seeds], 1, 'min')} | {cell([s['it_per_s'] for s in seeds], 1, '')} |")  # fmt: skip
            first = False


if __name__ == "__main__":
    main()
