"""The benchmark's tables from its runs (run.sh): per configuration and task, the mean and the sample SD over the training
seeds of the last checkpoint's evaluation (ndm_gym.evaluate_quadruped's metrics.json), the time to the target
(checkpoints.json), the training time and the iterations per second.

    python benchmarks/go2_lifelike_tracking/summarize.py NAME=logs/go2_sm [NAME=...] [--tasks all walk run play]

A configuration's directory holds ``<task>/seed_<s>/`` with ``evaluation/metrics.json``, ``evaluation/checkpoints.json``
and ``train_log.csv``. Iterations per second: the mean of the log's rows after the first 1,000 iterations (the start
compiles and captures graphs).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

# metrics.json's policy key, the column, the scale, the unit, the digits
COLUMNS = (("success_rate", "success", 1, "", 4), ("norm_return", "return", 1, "", 4), ("feet_local_mm", "feet (trunk frame)", 1, "mm", 2),
           ("feet_world_mm", "feet (world)", 1, "mm", 2), ("trunk_mm", "trunk", 1, "mm", 2), ("joint_deg", "joints", 1, "°", 2),
           ("n_failed", "failed clips", 1, "", 1))  # fmt: skip


def cell(xs: list[float], digits: int, unit: str = "") -> str:
    """Mean ± sample SD of the seeds' values (one seed: the value); "not reached" if a seed never reached it."""
    if not xs:
        return ""
    if any(x is None for x in xs):
        return f"not reached ({sum(x is None for x in xs)} of {len(xs)})"
    mean = statistics.fmean(xs)
    sd = f" ± {statistics.stdev(xs):.{digits}f}" if len(xs) > 1 else ""
    return f"{mean:.{digits}f}{sd}{' ' + unit if unit and unit != '°' else unit}"


def load(root: str, task: str) -> list[dict]:
    """A task's seeds: the policy's and the open loop's numbers, the pushes' failed clips, the time to the target, the
    training time in minutes and the iterations per second."""
    out = []
    for run in sorted(glob.glob(os.path.join(root, task, "seed_*"))):
        path = os.path.join(run, "evaluation", "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        with open(os.path.join(run, "train_log.csv")) as f:
            rows = list(csv.DictReader(f))
        ckpts = os.path.join(run, "evaluation", "checkpoints.json")
        to = json.load(open(ckpts))["minutes_to_success"] if os.path.exists(ckpts) else None
        its = [float(r["it_per_s"]) for r in rows if float(r["iteration"]) > 1000] or [float(rows[-1]["it_per_s"])]
        out.append({"seed": int(run.rsplit("_", 1)[1]), "policy": m["policy"], "open_loop": m.get("reference_targets"),
                    "push": {k: v["failed_clips"] for k, v in m.get("push", {}).items()}, "time_to": to,
                    "has_time_to": os.path.exists(ckpts), "minutes": float(rows[-1]["wall_h"]) * 60,
                    "it_per_s": statistics.fmean(its)})  # fmt: skip
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="+", help="NAME=directory, in the table's order")
    parser.add_argument("--tasks", nargs="+", default=["all", "walk", "run", "play"])
    args = parser.parse_args()
    head = ["task", "", "seeds", *(c[1] for c in COLUMNS), "failed clips, push 1 / 2 m/s", "time to target", "training", "iterations/s"]
    print("| " + " | ".join(head) + " |")
    print("|---" * len(head) + "|")
    for task in args.tasks:
        first = True
        for spec in args.configs:
            name, root = spec.split("=", 1)
            seeds = load(root, task)
            if not seeds:
                continue
            ol = seeds[0]["open_loop"]
            if first and ol:  # the open loop of the first configuration's first seed (the same rollouts for every policy)
                print(f"| `{task}` | reference's targets, open loop | | {ol['success_rate']:.3f} | | | | {ol['trunk_mm']:.0f} mm |" + " |" * 6)
            cells = [cell([s["policy"][k] * scale for s in seeds], digits, unit) for k, _, scale, unit, digits in COLUMNS]
            push = " / ".join(cell([s["push"][k] for s in seeds if k in s["push"]], 1) for k in ("1", "2") if any(k in s["push"] for s in seeds))
            to = cell([s["time_to"] for s in seeds if s["has_time_to"]], 2, "min")
            label = f"`{task}`" if first and not ol else ""
            print(f"| {label} | {name} | {', '.join(str(s['seed']) for s in seeds)} | " + " | ".join(cells)
                  + f" | {push} | {to} | {cell([s['minutes'] for s in seeds], 1, 'min')} | {cell([s['it_per_s'] for s in seeds], 2)} |")  # fmt: skip
            first = False


if __name__ == "__main__":
    main()
