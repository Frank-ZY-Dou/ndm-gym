"""The animal tracking benchmark's stop rule (README.md): one run's kept checkpoints evaluated as they appear, in iteration
order, on the run's clips (each from its first frame and from 9 perturbed starts, the rat's from 19, evaluation seed 1:
evaluate_animal's Evaluator), until the rule is met or the cap is evaluated.

    python benchmarks/mimic_mjx_animal_tracking/watch.py <run dir> [--data_dir <data>] [--max_hours 10]

Writes ``<run>/quick.csv`` (iteration, minutes, ended share, the bodies' error of the completed episodes in mm), each
evaluation's ``<run>/quick/<iteration>/metrics.json`` and, at the end, ``<run>/stop.json``: the stop iteration and why
("rule", "cap", or "end": the training ended without either). A checkpoint is read once its file exists (train_animal
writes it under its name complete). The training is not touched here: run.sh ends it on ``stop.json``.

The rule, the same for both algorithms (``decide``): with S(i) and E(i) the means over the evaluations at i, i - every and
i - 2 every of the ended share and of the bodies' error, a run stops at the first evaluated iteration i of at least
``first`` at which E(i) and E(i - window) exist, E(i) > 0.97 E(i - window) (the error improved by less than 3 % over the
window) and either S(i) <= 0.01 or S(i) > S(i - window) - 0.01 (the ended share at most 1 % or lowered by less than one
point over the window); else at the cap. An evaluation in which no episode completed has no error and stops no run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

# per body: the kept checkpoints' spacing, the window and the first iteration at which a run may stop (a quarter of the cap)
RULE = {"stick": {"every": 250, "window": 1000, "first": 2000}, "fly": {"every": 250, "window": 1000, "first": 3000},
        "rat": {"every": 500, "window": 2000, "first": 10000}}  # fmt: skip
SMOOTH, ERROR_RATIO, ENDED_OK, ENDED_STEP = 3, 0.97, 0.01, 0.01
# the quick evaluations' starts, other starts than the final evaluations': per clip its first frame and this many perturbed
# starts (100 episodes for ten clips; the rat's 200, which halves the noise of its ended share's three-evaluation means)
QUICK_REPEATS, QUICK_SEED = {"stick": 9, "fly": 9, "rat": 19}, 1
COLUMNS = ("iteration", "minutes", "ended_share", "body_mm")
REASON = {"STOP": "rule", "CAP": "cap"}  # stop.json's reason for decide's verdicts (final.py's --reason; "end" without either)


def read_rows(path: str) -> dict[int, tuple[float, float]]:
    """quick.csv's rows: iteration -> (ended share, bodies' error in mm; NaN when no episode completed)."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            e = r["body_mm"].strip()
            out[int(r["iteration"])] = (float(r["ended_share"]), float(e) if e and e.lower() not in ("nan", "none") else math.nan)
    return out


def means(q: dict, i: int, every: int, smooth: int = SMOOTH) -> tuple[float, float] | None:
    """The means of the ended share and of the error over the evaluations at i, i - every, ... (``smooth`` of them); None
    when one is missing or has no error."""
    its = [i - k * every for k in range(smooth)]
    if any(j not in q for j in its) or not all(math.isfinite(q[j][1]) for j in its):
        return None
    return sum(q[j][0] for j in its) / smooth, sum(q[j][1] for j in its) / smooth


def decide(q: dict, animal: str, cap: int) -> tuple[str, int | None]:
    """The rule over the evaluations ``q`` (quick.csv's rows): ("STOP", i), ("CAP", cap) or ("CONTINUE", None)."""
    r = RULE[animal]
    for i in sorted(q):
        if i < r["first"]:
            continue
        now, before = means(q, i, r["every"]), means(q, i - r["window"], r["every"])
        if now is None or before is None:
            continue
        (s, e), (s0, e0) = now, before
        if e > ERROR_RATIO * e0 and (s <= ENDED_OK or s > s0 - ENDED_STEP):
            return "STOP", i
    if cap in q:
        return "CAP", cap
    return "CONTINUE", None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run")
    p.add_argument("--data_dir", default=None, help="the clips (default: the run's)")
    p.add_argument("--max_hours", type=float, default=10.0, help="give up (exit 1) after this long without a decision")
    p.add_argument("--poll", type=float, default=5.0)
    a = p.parse_args()

    import torch

    from ndm_gym.evaluate_animal import Evaluator

    t0 = time.time()
    args_path = os.path.join(a.run, "args.json")
    while not os.path.exists(args_path):  # train_animal writes it at its start
        if time.time() - t0 > a.max_hours * 3600:
            sys.exit(f"{args_path} did not appear")
        time.sleep(a.poll)
    run = json.load(open(args_path))
    animal, cap, every = run["animal"], int(run["iterations"]), int(run["ckpt_every"])
    if every != RULE[animal]["every"] or cap % every or not run["ckpt_keep"]:
        sys.exit(f"{a.run}: the rule reads a checkpoint kept every {RULE[animal]['every']} iterations up to the cap; the run keeps "
                 f"{'every ' + str(every) if run['ckpt_keep'] else 'none'} to {cap}")  # fmt: skip
    csv_path, stop_path = os.path.join(a.run, "quick.csv"), os.path.join(a.run, "stop.json")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
    ev = None
    while True:
        q = read_rows(csv_path)
        what, i = decide(q, animal, cap)
        if what != "CONTINUE":
            json.dump({"stop": i, "reason": REASON[what], "evaluated": max(q)}, open(stop_path, "w"))
            print(json.dumps({"stop": i, "reason": REASON[what]}), flush=True)
            return
        done = os.path.exists(os.path.join(a.run, "train.done"))  # the training ended (run.sh writes it)
        kept = sorted(int(os.path.basename(c)[5:11]) for c in glob.glob(os.path.join(a.run, "ckpt_*.pt")))
        todo = [k for k in kept if k not in q]
        if todo:
            it = todo[0]
            if q and it != max(q) + every:
                sys.exit(f"{a.run}: the evaluations are not consecutive ({max(q)}, then {it})")
            path = os.path.join(a.run, f"ckpt_{it:06d}.pt")
            ck = torch.load(path, map_location="cuda:0", weights_only=False)
            if ev is None:
                ev = Evaluator(ck["args"], None, QUICK_REPEATS[animal], 0.02, QUICK_SEED, a.data_dir)
            m, _ = ev({**ck, "path": os.path.abspath(path)})
            os.makedirs(os.path.join(a.run, "quick", str(it)), exist_ok=True)
            json.dump(m, open(os.path.join(a.run, "quick", str(it), "metrics.json"), "w"), indent=1)
            with open(csv_path, "a") as f:
                f.write(f"{it},{m['minutes']:.4f},{m['ended_share']},{m['body_mm'] if m['body_mm'] is not None else 'nan'}\n")
            print(json.dumps({k: m[k] for k in ("iteration", "minutes", "ended_share", "body_mm")}), flush=True)
            continue
        if done:
            json.dump({"stop": max(kept) if kept else None, "reason": "end", "evaluated": max(q) if q else None}, open(stop_path, "w"))
            print(json.dumps({"stop": max(kept) if kept else None, "reason": "end"}), flush=True)
            return
        if time.time() - t0 > a.max_hours * 3600:
            sys.exit(f"{a.run}: no decision after {a.max_hours} h")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
