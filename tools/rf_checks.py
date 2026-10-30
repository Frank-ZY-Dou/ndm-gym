"""The RigidFormer checks, by their declared rules: per algorithm, RigidFormer minus MLP in return, failed clips and
MPJPE-local with Welch 95% intervals over three seeds, and the verdicts.

  check 1  dynamics shifts: <root>/results/rfcheck/shift_<condition>_<run>.json of the MLP finals (final_<algo>_s1-3)
           and SuperTrack-RF / SuperMimic-RF (abl_<algo>_rf_s1-3), one evaluation each; their standard evaluation of
           the same checkpoint (eval.csv) as context
  check 2  sample efficiency: <root>/logs/c2_<algo>[_rf]_s4-6/eval.csv at iterations 4000, 6000 and 8000
  check 3  unseen motions: <root>/logs/h200_c3_<algo>[_rf]_s1-3/ckpt_<last>_eval.json, split into the 7 held-out clips
           and the 33 training clips of benchmarks/g1_lafan1_tracking/splits; every checkpoint as context

Usage: python tools/rf_checks.py <check> --root <directory with logs/ and results/> [--md <out.md>] [--csv <out.csv>]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablations import welch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "benchmarks", "g1_lafan1_tracking", "splits")
ALGOS = (("st", "SuperTrack"), ("sm", "SuperMimic"))
# metric, label, scale, sign (+1: higher is better)
METRICS = (("norm_return", "return", 1.0, 1), ("n_failed", "failed clips", 1.0, -1), ("mpjpe_l", "MPJPE-local (mm)", 1e3, -1))
CONDITIONS = (("f02", "friction 0.2"), ("f05", "friction 0.5"), ("f20", "friction 2.0"), ("p5", "payload 5 kg"), ("p10", "payload 10 kg"))
C2_ITERS = (4000, 6000, 8000)
C3_ITERS = {"st": 12670, "sm": 13540}


def read_list(path):
    with open(path) as f:
        return [n for n in (line.split("#", 1)[0].strip() for line in f) if n]


def compare(mlp, rf):
    """RigidFormer minus MLP for each metric over seeds (lists of per-seed dicts), with Welch 95% intervals, the
    metrics significantly better / worse for RigidFormer, and the declared verdict."""
    rows, better, worse = [], [], []
    for key, label, scale, sign in METRICS:
        a, b = [scale * s[key] for s in mlp], [scale * s[key] for s in rf]
        d, lo, hi = welch(a, b)
        if lo > 0 or hi < 0:  # the interval excludes 0
            (better if sign * d > 0 else worse).append(label)
        rows.append({"metric": label, "mlp": statistics.mean(a), "rf": statistics.mean(b), "diff": d, "lo": lo, "hi": hi})
    if better and not worse:
        verdict = "RigidFormer better (" + ", ".join(better) + ")"
    elif worse and not better:
        verdict = "RigidFormer worse (" + ", ".join(worse) + ")"
    elif better and worse:
        verdict = "mixed (better: " + ", ".join(better) + "; worse: " + ", ".join(worse) + ")"
    else:
        verdict = "no significant difference"
    return rows, verdict, bool(better and not worse)


def eval_rows(run_dir):
    with open(os.path.join(run_dir, "eval.csv")) as f:
        return {int(r["iteration"]): r for r in csv.DictReader(f)}


def one(path):
    with open(path) as f:
        res = json.load(f)
    if len(res) != 1:
        raise ValueError(f"{path}: {len(res)} evaluations, expected one")
    return res[0]


def check1(root):
    out = []
    for a, name in ALGOS:
        mlp_runs, rf_runs = [f"final_{a}_s{s}" for s in (1, 2, 3)], [f"abl_{a}_rf_s{s}" for s in (1, 2, 3)]
        its = {}
        for cond, label in CONDITIONS:
            res = {r: one(os.path.join(root, "results", "rfcheck", f"shift_{cond}_{r}.json")) for r in mlp_runs + rf_runs}
            for r, x in res.items():
                if its.setdefault(r, x["iteration"]) != x["iteration"]:
                    raise ValueError(f"{r}: conditions evaluated different checkpoints")
            rows, verdict, _ = compare([res[r] for r in mlp_runs], [res[r] for r in rf_runs])
            out.append({"algo": name, "case": label, "rows": rows, "verdict": verdict})
        std = {r: eval_rows(os.path.join(root, "logs", r))[its[r]] for r in mlp_runs + rf_runs}  # the same checkpoints
        std = {r: {k: float(v[k]) for k, *_ in METRICS} for r, v in std.items()}
        rows, _, _ = compare([std[r] for r in mlp_runs], [std[r] for r in rf_runs])
        out.insert(len(out) - len(CONDITIONS), {"algo": name, "case": "no shift: the watcher's evaluation (context)", "rows": rows,
                                                "verdict": "context, no verdict"})  # fmt: skip
    return out, "Check 1: dynamics shifts (RigidFormer minus MLP, seeds 1-3, the equal-iteration checkpoints)"


def check2(root):
    out = []
    for a, name in ALGOS:
        runs = {k: [os.path.join(root, "logs", f"c2_{a}{k}_s{s}") for s in (4, 5, 6)] for k in ("", "_rf")}
        evals = {k: [eval_rows(d) for d in ds] for k, ds in runs.items()}
        holds = 0
        for it in C2_ITERS:
            per = {k: [{m: float(e[it][m]) for m, *_ in METRICS} for e in es] for k, es in evals.items()}
            rows, verdict, better = compare(per[""], per["_rf"])
            holds += better
            out.append({"algo": name, "case": f"iteration {it}", "rows": rows, "verdict": verdict})
        out.append({"algo": name, "case": "hypothesis (2 of 3 checkpoints)", "rows": [],
                    "verdict": f"{'holds' if holds >= 2 else 'does not hold'} ({holds} of 3)"})  # fmt: skip
    return out, "Check 2: sample efficiency (RigidFormer minus MLP, seeds 4-6, fresh runs on A100 80GB GPUs)"


def split_scores(res, names):
    """Return, failed clips and MPJPE-local of an all-clip evaluation over a subset of its clips."""
    missing = [n for n in names if n not in res["per_clip"]]
    if missing:
        raise ValueError(f"clips not in the evaluation: {missing}")
    return {"norm_return": statistics.mean(res["per_clip"][n] for n in names), "n_failed": sum(n in res["failed"] for n in names),
            "mpjpe_l": statistics.mean(res["per_clip_err"][n]["mpjpe_l"] for n in names)}  # fmt: skip


def check3(root):
    train, test = read_list(os.path.join(SPLITS, "lafan1_train_subjects1-4.txt")), read_list(os.path.join(SPLITS, "lafan1_test_subject5.txt"))
    out, curves = [], []
    for a, name in ALGOS:
        scores = {}
        for k in ("", "_rf"):
            for s in (1, 2, 3):
                d = os.path.join(root, "logs", f"h200_c3_{a}{k}_s{s}")
                with open(os.path.join(d, "clips.txt")) as f:
                    trained = f.read().split()
                if len(trained) != len(train) or set(trained) != set(train):
                    raise ValueError(f"{d}: clips.txt is not the training list")
                evals = sorted(glob.glob(os.path.join(d, "ckpt_*_eval.json")))
                its = [int(os.path.basename(p)[5:11]) for p in evals]
                if not its or its[-1] != C3_ITERS[a]:
                    raise ValueError(f"{d}: last evaluated checkpoint {its[-1:] or None}, declared {C3_ITERS[a]}")
                for it, p in zip(its, evals):
                    with open(p) as f:
                        res = json.load(f)
                    if len(res["per_clip"]) != len(train) + len(test):
                        raise ValueError(f"{p}: {len(res['per_clip'])} clips, expected {len(train) + len(test)}")
                    scores[(k, s, it)] = {"held-out": split_scores(res, test), "training": split_scores(res, train)}
        last = C3_ITERS[a]
        for part in ("held-out", "training"):
            rows, verdict, _ = compare([scores[("", s, last)][part] for s in (1, 2, 3)], [scores[("_rf", s, last)][part] for s in (1, 2, 3)])
            out.append({"algo": name, "case": f"{part} clips, it {last}" + ("" if part == "held-out" else " (context)"), "rows": rows,
                        "verdict": verdict if part == "held-out" else "context, no verdict"})  # fmt: skip
        gap = {k: [{m: scores[(k, s, last)]["held-out"][m] - scores[(k, s, last)]["training"][m] for m, *_ in METRICS} for s in (1, 2, 3)]
               for k in ("", "_rf")}  # fmt: skip
        rows, _, _ = compare(gap[""], gap["_rf"])
        out.append({"algo": name, "case": "held-out minus training (context)", "rows": rows, "verdict": "context, no verdict"})
        for it in sorted({it for (_, _, it) in scores}):
            if all(("", s, it) in scores and ("_rf", s, it) in scores for s in (1, 2, 3)):
                for k, label in (("", "MLP"), ("_rf", "RigidFormer")):
                    v = [scores[(k, s, it)]["held-out"] for s in (1, 2, 3)]
                    curves.append({"algo": name, "model": label, "iteration": it, "held-out return": statistics.mean(x["norm_return"] for x in v),
                                   "held-out failed": statistics.mean(x["n_failed"] for x in v),
                                   "held-out MPJPE-local (mm)": 1e3 * statistics.mean(x["mpjpe_l"] for x in v)})  # fmt: skip
    return out, "Check 3: unseen motions (RigidFormer minus MLP, seeds 1-3, LAFAN1 subject 5 held out)", curves


def markdown(title, out, curves=None):
    lines = [f"### {title}", "", "| algorithm | case | metric | MLP | RigidFormer | RF - MLP [95% CI] | verdict |", "|---|---|---|---|---|---|---|"]
    for o in out:
        if not o["rows"]:
            lines.append(f"| {o['algo']} | {o['case']} | | | | | {o['verdict']} |")
        for i, r in enumerate(o["rows"]):
            f = ".3f" if r["metric"] == "return" else ".2f" if r["metric"] == "failed clips" else ".1f"
            ci = f"{r['diff']:+{f}} [{r['lo']:+{f}}, {r['hi']:+{f}}]"
            lines.append(f"| {o['algo'] if i == 0 else ''} | {o['case'] if i == 0 else ''} | {r['metric']} | {r['mlp']:{f}} | {r['rf']:{f}} | {ci} | "
                         f"{o['verdict'] if i == 0 else ''} |")  # fmt: skip
    if curves:
        lines += ["", "Held-out clips at every checkpoint (mean over seeds 1-3; context):", "",
                  "| algorithm | model | iteration | return | failed clips | MPJPE-local (mm) |", "|---|---|---|---|---|---|"]  # fmt: skip
        for c in curves:
            lines.append(f"| {c['algo']} | {c['model']} | {c['iteration']} | {c['held-out return']:.3f} | {c['held-out failed']:.2f} | "
                         f"{c['held-out MPJPE-local (mm)']:.1f} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("check", type=int, choices=[1, 2, 3])
    p.add_argument("--root", required=True, help="directory holding the runs' logs/ (and results/ for check 1)")
    p.add_argument("--md", default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    curves = None
    if args.check == 1:
        out, title = check1(args.root)
    elif args.check == 2:
        out, title = check2(args.root)
    else:
        out, title, curves = check3(args.root)
    md = markdown(title, out, curves)
    print(md)
    if args.md:
        open(args.md, "w").write(md)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algorithm", "case", "metric", "mlp", "rf", "diff", "lo", "hi", "verdict"])
            for o in out:
                for r in o["rows"]:
                    w.writerow([o["algo"], o["case"], r["metric"], r["mlp"], r["rf"], r["diff"], r["lo"], r["hi"], o["verdict"]])


if __name__ == "__main__":
    main()
