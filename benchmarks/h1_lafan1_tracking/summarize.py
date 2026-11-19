"""The tables of results.md from the benchmark's runs (run.sh), with the Go2 benchmark's loader and cells
(benchmarks/go2_lifelike_tracking/summarize.py: per configuration and task the mean and sample SD over the training seeds)
and the humanoid evaluation's columns (ndm_gym.evaluate_humanoid).

    python benchmarks/h1_lafan1_tracking/summarize.py ST=logs/h1/h1-supertrack SM=logs/h1/h1-supermimic
        [--tasks selected all train] [--each selected] [--kinds all] [--tipped selected] [--heldout] [--curve selected]
        [--time ST=<dir> SM=<dir>]

The main table: per task, the last checkpoint's evaluation (``<task>/seed_<s>/evaluation/metrics.json``) and, for the first
configuration's first seed, the open loop's row (the reference's joint positions as the targets). ``--each <task>``: that
task's seeds one by one. ``--kinds <task>``: that task's return and success by the kind of motion (the clip name's leading
letters). ``--tipped <task>``: that task's clips in which the reference's pelvis tips past the horizontal (the z component of
its z axis below 0 in some frame; read from ``--clips_dir`` with the library's loader) and its other clips. ``--heldout``: the clips of subject 5,
by the policies that trained without them (``train/seed_<s>/heldout``) and by those that trained with them (``selected``),
and the training clips of the former. ``--curve <task>``: every kept checkpoint (``checkpoints/checkpoints.json``).
``--time``: the training time and the iterations per second of training-only runs (``<dir>/seed_<s>/train_log.csv``).
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import statistics

_GO2 = os.path.join(os.path.dirname(__file__), "..", "go2_lifelike_tracking", "summarize.py")
_spec = importlib.util.spec_from_file_location("go2_summarize", _GO2)
go2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(go2)

# metrics.json's policy key, the column, the unit, the digits
COLUMNS = (("norm_return", "return", "", 3), ("success_rate", "success", "", 3), ("n_failed", "clips with a failed rollout", "", 1),
           ("mpjpe_g_mm", "MPJPE (world)", "mm", 0), ("mpjpe_l_mm", "MPJPE (pelvis)", "mm", 1), ("root_mm", "pelvis", "mm", 0),
           ("joint_deg", "joints", "°", 2))  # fmt: skip
HELD = re.compile(r"subject5$")


def table(head: list[str]):
    print("\n| " + " | ".join(head) + " |")
    print("|---" * len(head) + "|")


def seeds_of(root: str, task: str, sub: str = "evaluation") -> list[dict]:
    """The policy's numbers (metrics.json) of a task's seeds in a folder of their runs."""
    paths = sorted(glob.glob(os.path.join(root, task, "seed_*", sub, "metrics.json")))
    return [json.load(open(p))["policy"] for p in paths]


def over_clips(policy: dict, keep) -> dict:
    """The return and the success over the clips ``keep`` accepts (the evaluation's own means are over all its clips)."""
    clips = [c for name, c in policy["clips"].items() if keep(name)]
    return {k: statistics.fmean(c[k] for c in clips) if clips else float("nan") for k in ("norm_return", "success_rate")} | {"n": len(clips)}


def with_runs(configs: list, task: str, sub: str = "evaluation") -> list:
    """(name, the seeds' numbers) of the configurations that have runs of the task."""
    rows = [(name, seeds_of(root, task, sub)) for name, root in configs]
    return [(name, seeds) for name, seeds in rows if seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="+", help="NAME=directory, in the tables' order")
    parser.add_argument("--tasks", nargs="+", default=["selected", "all", "train"])
    parser.add_argument("--each", type=str, default=None, help="a task: every seed's return, success and failed clips")
    parser.add_argument("--kinds", type=str, default=None, help="a task: its return and success by the kind of motion")
    parser.add_argument("--tipped", type=str, default=None, help="a task: its clips whose reference pelvis tips past the horizontal, and the others")
    parser.add_argument("--clips_dir", type=str, default="data/lafan1_h1", help="--tipped: the H1's clips (tools/download_h1.sh)")
    parser.add_argument("--heldout", action="store_true", help="subject 5's clips, unseen (train) and seen (selected)")
    parser.add_argument("--curve", type=str, default=None, help="a task: every kept checkpoint")
    parser.add_argument("--time", nargs="+", default=None, help="NAME=directory of training-only runs (seed_<s>/train_log.csv)")
    args = parser.parse_args()
    configs = [spec.split("=", 1) for spec in args.configs]

    table(["task", "", "seeds", *(c[1] for c in COLUMNS), "success, push 1 / 2 m/s"])
    for task in args.tasks:
        first = True
        for name, root in configs:
            seeds = go2.load(root, task)
            if not seeds:
                continue
            ol = seeds[0]["open_loop"]
            if first and ol:  # the same rollouts' starts for every policy
                print(f"| `{task}` | reference's targets, open loop | | " + " | ".join(go2.cell([ol[k]], d, u) for k, _, u, d in COLUMNS) + " | |")
            push = [json.load(open(p)).get("push", {}) for p in sorted(glob.glob(os.path.join(root, task, "seed_*", "evaluation", "metrics.json")))]
            pushed = " / ".join(go2.cell([p[k]["success_rate"] for p in push], 3) for k in ("1", "2") if all(k in p for p in push))
            cells = [go2.cell([s["policy"][k] for s in seeds], d, u) for k, _, u, d in COLUMNS]
            label = "" if ol or not first else f"`{task}`"
            print(f"| {label} | {name} | {', '.join(str(s['seed']) for s in seeds)} | " + " | ".join(cells) + f" | {pushed} |")
            first = False

    if args.each:
        table(["", "seed", "return", "success", "clips with a failed rollout", "clips with no successful rollout"])
        for name, root in configs:
            for s in go2.load(root, args.each):
                p = s["policy"]
                never = ", ".join(f"`{c}`" for c, v in p["clips"].items() if v["success_rate"] == 0)
                print(f"| {name} | {s['seed']} | {p['norm_return']:.3f} | {p['success_rate']:.3f} | {p['n_failed']} | {never} |")

    if args.kinds:
        rows = with_runs(configs, args.kinds)
        kinds = dict.fromkeys(re.match(r"[A-Za-z]+", clip).group() for _, seeds in rows[:1] for clip in seeds[0]["clips"])
        table(["kind", "clips", *(f"{name}: {what}" for what in ("return", "success") for name, _ in rows)])
        for kind in kinds:
            per = [[over_clips(s, lambda c, kind=kind: re.match(r"[A-Za-z]+", c).group() == kind) for s in seeds] for _, seeds in rows]
            cells = [go2.cell([s[k] for s in p], 3) for k in ("norm_return", "success_rate") for p in per]
            print(f"| `{kind}` | {per[0][0]['n']} | " + " | ".join(cells) + " |")

    if args.tipped:
        from ndm_gym.motion.sources import lafan1

        up = {}  # per clip, the lowest z component of the pelvis' z axis: 1 - 2 (x^2 + y^2) of its quaternion (w, x, y, z)
        for clip in lafan1.load_dir(args.clips_dir, 19):
            up[clip.name] = float((1 - 2 * (clip.root_quat[:, 1] ** 2 + clip.root_quat[:, 2] ** 2)).min())
        rows = with_runs(configs, args.tipped)
        table(["clips", "", *(f"{name}: {what}" for what in ("return", "success") for name, _ in rows)])
        for label, keep in (("the pelvis tips past the horizontal", lambda c: up[c] < 0), ("the others", lambda c: up[c] >= 0)):
            per = [[over_clips(s, keep) for s in seeds] for _, seeds in rows]
            if not rows or not per[0][0]["n"]:
                continue
            cells = [go2.cell([s[k] for s in p], 3) for k in ("norm_return", "success_rate") for p in per]
            names = ", ".join(f"`{c}`" for c in rows[0][1][0]["clips"] if keep(c)) if label != "the others" else ""
            print(f"| {per[0][0]['n']} | {label}{': ' + names if names else ''} | " + " | ".join(cells) + " |")

    if args.heldout:
        table(["", "clips", "trained on", *(f"{name}: {what}" for what in ("return", "success") for name, _ in configs)])
        rows = (("subject 5's", "subjects 1 to 4 (not these clips)", "train", "heldout", lambda c: True),
                ("subject 5's", "all five subjects", "selected", "evaluation", lambda c: bool(HELD.search(c))),
                ("subjects 1 to 4's", "subjects 1 to 4", "train", "evaluation", lambda c: True))  # fmt: skip
        for clips, trained, task, sub, keep in rows:
            per = [[over_clips(s, keep) for s in seeds_of(root, task, sub)] for _, root in configs]
            if not all(per) or not per[0][0]["n"]:
                continue
            cells = [go2.cell([s[k] for s in p], 3) for k in ("norm_return", "success_rate") for p in per]
            print(f"| {clips} | {per[0][0]['n']} | {trained} | " + " | ".join(cells) + " |")

    if args.curve:
        runs = {name: [json.load(open(p))["checkpoints"] for p in sorted(glob.glob(os.path.join(root, args.curve, "seed_*", "checkpoints", "checkpoints.json")))]
                for name, root in configs}  # fmt: skip
        runs = {name: [{ck["iteration"]: ck for ck in seed} for seed in seeds] for name, seeds in runs.items() if seeds}
        table(["iteration", *(f"{name}: {what}" for what in ("return", "success") for name in runs)])
        every = [set(seed) for seeds in runs.values() for seed in seeds]
        for it in sorted(set.intersection(*every)) if every else ():  # the checkpoints that every run kept
            cells = [go2.cell([seed[it][k] for seed in seeds], 3) for k in ("norm_return", "success_rate") for seeds in runs.values()]
            print(f"| {it} | " + " | ".join(cells) + " |")

    if args.time:
        table(["", "seeds", "training", "iterations/s"])
        for spec in args.time:
            name, root = spec.split("=", 1)
            minutes, its, seeds = [], [], []
            for run in sorted(glob.glob(os.path.join(root, "seed_*"))):
                with open(os.path.join(run, "train_log.csv")) as f:
                    log = list(csv.DictReader(f))
                seeds.append(run.rsplit("_", 1)[1])
                minutes.append(float(log[-1]["wall_h"]) * 60)
                late = [float(r["it_per_s"]) for r in log if float(r["iteration"]) > 1000]  # after the start's compilation
                its.append(statistics.fmean(late or [float(r["it_per_s"]) for r in log]))
            print(f"| {name} | {', '.join(seeds)} | {go2.cell(minutes, 1, 'min')} | {go2.cell(its, 2)} |")


if __name__ == "__main__":
    main()
