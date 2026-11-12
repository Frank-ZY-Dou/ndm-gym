"""Each task's best world model, from the fits' scores (``fit_mjcf`` csv files named ``rfit_<task>_dt<ms>_<arm>.csv``, and
``lwfit_`` for the fits with windows to the horizon): per task and step length, the fit with the lowest ``norm`` on the
held-out rollouts (``held``) at the horizon, the largest k scored (the horizon in that step, as the fits were run); fits within
2 % of the best tie, and a tie goes to the faster network, in the order of the networks' work (no LayerNorm before LayerNorm,
and at equal LayerNorm the acceleration model, nv outputs, before the increment model, 2 nv), then to the 8-step windows.
Prints the table of every arm (held-out and executed errors at the horizon, at the last update) and writes the choice as json.

With ``--rl``, the RL tasks' fits instead (``rlfit_<task>_ts<timestep in ms>x<substeps>_<arm>.csv``), per task and
simulator kind, at k = PPO's horizon (the largest k scored); the choice's key is ``<task>_<kind>``.

With ``--fast_out``, the fastest setting per task and step length: the chosen arm's 3 x 512 copy (``nfit_``) where
its held-out ``norm`` at the horizon is within 5 % of the chosen fit's, the chosen fit otherwise.

Run: ``python benchmarks/robokit_ndm/select_arms.py fits/ --out results/robokit_ndm/best_arms.json``
"""

import argparse
import csv
import glob
import json
import os
import re

ARMS = ("acc_noln", "inc_noln", "acc_ln", "inc_ln")  # the speed order: fastest first
NAME = re.compile(r"(?P<win>rfit|lwfit)_(?P<task>.+)_dt(?P<ms>[0-9.]+)_(?P<arm>acc_ln|acc_noln|inc_ln|inc_noln)\.csv$")
NAME_RL = re.compile(r"(?P<win>rlfit)_(?P<task>.+)_(?P<kind>ts(?P<ts>[0-9.]+)x(?P<sub>[0-9]+))_(?P<arm>acc_ln|acc_noln|inc_ln|inc_noln)\.csv$")


def last_scores(path: str) -> tuple[dict, int]:
    """({(test, k): row} at the last update of a fit's csv, that update)."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"{path}: no scores")
    last = max(int(r["step"]) for r in rows)
    return {(r["test"], int(r["k"])): r for r in rows if int(r["step"]) == last}, last


def select(paths: list) -> tuple[dict, list]:
    groups = {}
    for p in paths:
        if m := NAME.search(os.path.basename(p)):
            groups.setdefault((m["task"], float(m["ms"]), ""), {})[(m["arm"], m["win"])] = p
        elif m := NAME_RL.search(os.path.basename(p)):  # an RL task's simulator kind: its control step, and the kind in the key
            groups.setdefault((m["task"], float(m["ts"]) * int(m["sub"]), m["kind"]), {})[(m["arm"], m["win"])] = p
    choice, table = {}, []
    for (task, ms, kind), arms in sorted(groups.items()):
        where = f"{task} {kind}" if kind else f"{task} at {ms:g} ms"
        wins = sorted({w for _, w in arms}, key=lambda w: w != "rfit")  # the 8-step fits first
        missing = [(a, w) for w in wins for a in ARMS if (a, w) not in arms]
        if missing:
            raise ValueError(f"{where}: no fit for {missing}")
        running = [arms[c] for c in arms if not os.path.exists(arms[c][: -len(".csv")] + ".pt")]  # a fit saves its model when it ends
        if running:
            raise ValueError(f"{where}: fits without their saved model (still running?): {running}")
        order = [(a, w) for a in ARMS for w in wins]  # the tie order: the network's speed, then the 8-step windows
        scores = {c: last_scores(arms[c]) for c in order}
        steps = {c: s[1] for c, s in scores.items()}
        if len(set(steps.values())) != 1:
            raise ValueError(f"{where}: the fits stopped at different updates {steps}")
        k = max(kk for (t, kk) in scores[order[0]][0] if t == "held")
        norm = {c: float(scores[c][0][("held", k)]["norm"]) for c in order}
        best = min(norm.values())
        tied = [c for c in order if norm[c] <= 1.02 * best]
        pick = tied[0]  # the fastest of the tied fits
        choice[f"{task}_{kind}" if kind else f"{task}_dt{ms:g}"] = {
            "task": task, "dt": ms / 1e3, "k": k, "arm": pick[0], "windows": pick[1], "fit": arms[pick], **({"kind": kind} if kind else {}),
            "norm": {f"{a}_{w}": v for (a, w), v in norm.items()}, "tied": [f"{a}_{w}" for a, w in tied]}  # fmt: skip
        for c in order:
            ex = scores[c][0].get(("executed", k)) or scores[c][0].get(("eval", k))  # the executed trajectory; RL: the trained policy
            table.append([f"{task} {kind}" if kind else task, f"{ms:g}", k, f"{c[0]} ({c[1]})", f"{norm[c]:.4f}", ex["norm"] if ex else "",
                          "best" if c == pick else ("tied" if c in tied else "")])  # fmt: skip
    return choice, table


def fastest(choice: dict, fits: str) -> tuple[dict, list]:
    """``--fast_out``: {task_dt: the choice with "fit" the 3 x 512 copy where it is within 5 % at the horizon}, and its table."""
    out, table = {}, []
    for key, c in choice.items():
        path = os.path.join(fits, f"nfit_{c['task']}_dt{1e3 * c['dt']:g}_{c['arm']}.csv")
        if not os.path.exists(path):
            raise ValueError(f"{key}: no 3 x 512 fit {path}")
        scores, step = last_scores(path)
        if step != last_scores(c["fit"])[1]:
            raise ValueError(f"{key}: {path} stopped at update {step}, the chosen fit elsewhere")
        narrow, wide = float(scores[("held", c["k"])]["norm"]), c["norm"][f"{c['arm']}_{c['windows']}"]
        narrow_ok = narrow <= 1.05 * wide
        out[key] = {**c, "fit": path if narrow_ok else c["fit"], "hidden": [512] * 3 if narrow_ok else [1024] * 3, "norm_512": narrow}
        table.append([c["task"], f"{1e3 * c['dt']:g}", c["k"], f"{c['arm']} ({c['windows']})", f"{wide:.4f}", f"{narrow:.4f}",
                      f"{narrow / wide - 1:+.1%}", "3 x 512" if narrow_ok else "3 x 1024"])  # fmt: skip
    return out, table


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("fits", help="the folder of the rfit_*.csv files")
    p.add_argument("--out", required=True, help="json of each task's choice")
    p.add_argument("--tasks", nargs="+", default=None, help="only these tasks (default: every task with fits)")
    p.add_argument("--fast_out", default=None, help="json of the fastest setting per task (needs the nfit_ files)")
    p.add_argument("--rl", action="store_true", help="the RL tasks' fits (rlfit_*.csv), per task and simulator kind")
    args = p.parse_args(argv)
    if args.rl and args.fast_out:
        p.error("--fast_out is for the MPC tasks' fits")
    pats = ["rlfit_*.csv"] if args.rl else ["rfit_*.csv", "lwfit_*.csv"]
    paths = sorted(x for pat in pats for x in glob.glob(os.path.join(args.fits, pat)))
    if args.tasks:
        paths = [x for x in paths if (m := NAME.search(os.path.basename(x)) or NAME_RL.search(os.path.basename(x))) and m["task"] in args.tasks]
    choice, table = select(paths)
    print("| task | step (ms) | k | arm | held-out norm | executed (RL: the trained policy's) norm | |")
    print("|---|---|---|---|---|---|---|")
    for row in table:
        print("| " + " | ".join(str(x) for x in row) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(choice, f, indent=1)
    if args.fast_out:
        fast, table = fastest(choice, args.fits)
        print()
        print("| task | step (ms) | k | arm | held-out norm, 3 x 1024 | 3 x 512 | difference | fastest setting |")
        print("|---|---|---|---|---|---|---|---|")
        for row in table:
            print("| " + " | ".join(str(x) for x in row) + " |")
        with open(args.fast_out, "w") as f:
            json.dump(fast, f, indent=1)


if __name__ == "__main__":
    main()
