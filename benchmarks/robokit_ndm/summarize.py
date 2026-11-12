"""The timing runs' results per task: from each speed run's log (``speed_mjcf``'s output) the world-steps per second of
MuJoCo Warp's step and of the world model's step inside NdmSim, for the task's chosen fit (``best_arms.json``) at the planner's
rollout worlds and at 8,192, each step length apart; from each plan's json the example's own outcome line and planning time,
per planner seed, as it is, on the world models and on the do-nothing simulator. Writes one json and prints the tables.

Run: ``python benchmarks/robokit_ndm/summarize.py <study folder> --out results/summary.json``
"""

import argparse
import glob
import json
import os
import re
import statistics

LINE = re.compile(r"\[speed_mjcf\] (?P<what>physics|ndm|ndmsim) N=(?P<n>\d+): (?P<sps>[\d,]+) world-steps/s \((?P<ms>[\d.]+) ms per step\)")
PLANNED = re.compile(r"planned in (?P<s>[\d.]+) s")
VERDICT = re.compile(r"\b(pass|fail)\b", re.IGNORECASE)  # the examples' own line; the pen prints its turns and drops instead
RUN = re.compile(r"plan_(?P<task>.+)_(?P<kind>base|ndm|null)_s(?P<seed>\d+)\.json$")


def speeds(log: str) -> dict:
    """{(what, worlds): (world-steps per second, ms per step)} from a speed run's log."""
    out = {}
    for line in open(log):
        m = LINE.search(line)
        if m:
            out[m["what"], int(m["n"])] = (float(m["sps"].replace(",", "")), float(m["ms"]))
    return out


def plan(path: str) -> dict:
    run = json.load(open(path))
    lines = [x for x in run["output"] if x.strip()]
    seconds = [float(m["s"]) for x in lines for m in [PLANNED.search(x)] if m]
    if not seconds:
        raise ValueError(f"{path}: no planning time in the example's output")
    outcome = next((x for x in reversed(lines) if not PLANNED.search(x) and not x.startswith("[")), "").strip()
    v = VERDICT.search(outcome)
    return {"planned_s": seconds[-1], "outcome": outcome, "passed": None if v is None else v[1].lower() == "pass",
            "sim": run["planners"][0]["sim"]}  # fmt: skip


def plans_of(results: str, task: str) -> dict:
    """{kind: {"runs": {seed: run}, "seeds", "passes" (None where the example prints no verdict), "median_s", "min_s", "max_s"}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results, f"plan_{task}_*_s*.json"))):
        m = RUN.search(os.path.basename(path))
        if m and m["task"] == task:
            out.setdefault(m["kind"], {"runs": {}})["runs"][int(m["seed"])] = plan(path)
    for kind, k in out.items():
        runs = [k["runs"][s] for s in sorted(k["runs"])]
        t = [r["planned_s"] for r in runs]
        verdicts = [r["passed"] for r in runs]
        k.update(seeds=sorted(k["runs"]), median_s=statistics.median(t), min_s=min(t), max_s=max(t),
                 passes=None if None in verdicts else sum(verdicts))  # fmt: skip
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("study", help="the folder with logs/timing, results/ and best_arms.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    best = json.load(open(os.path.join(args.study, "best_arms.json")))
    L = os.path.join(args.study, "logs", "timing")
    summary = {}
    for key, choice in sorted(best.items()):
        task, ms = choice["task"], f"{1e3 * choice['dt']:g}"
        sp = speeds(os.path.join(L, f"speed_{task}_{ms}.log"))
        rows = {}
        for n in sorted({n for (_, n) in sp}):
            if ("physics", n) not in sp or ("ndmsim", n) not in sp:
                raise ValueError(f"{task} {ms} ms: N={n} lacks MuJoCo Warp's or NdmSim's row")
            phys, sim, alone = sp["physics", n], sp["ndmsim", n], sp.get(("ndm", n))
            rows[n] = {"physics_ms": phys[1], "ndmsim_ms": sim[1], "ndm_ms": alone[1] if alone else None, "speedup": phys[1] / sim[1]}
        summary[key] = {"task": task, "dt": choice["dt"], "arm": choice["arm"], "windows": choice.get("windows", "rfit"),
                        "norm": choice["norm"][f"{choice['arm']}_{choice.get('windows', 'rfit')}"], "speed": rows}  # fmt: skip
    for task in sorted({c["task"] for c in best.values()}):
        plans = plans_of(os.path.join(args.study, "results"), task)
        for key in summary:
            if summary[key]["task"] == task:
                summary[key]["plans"] = plans
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("| task | step (ms) | fit | held-out norm | worlds | MuJoCo Warp (ms) | NdmSim (ms) | NDM alone (ms) | step speedup |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in summary.values():
        for n, r in s["speed"].items():
            alone = f"{r['ndm_ms']:.3f}" if r["ndm_ms"] is not None else "-"
            print(f"| {s['task']} | {1e3 * s['dt']:g} | {s['arm']} ({s['windows']}) | {s['norm']:.3g} | {n} | {r['physics_ms']:.3f} | "
                  f"{r['ndmsim_ms']:.3f} | {alone} | {r['speedup']:.1f}x |")  # fmt: skip
    print()
    print("| task | simulator | seeds | passes | planning: median (min-max) |")
    print("|---|---|---|---|---|")
    for task in sorted({s["task"] for s in summary.values()}):
        pl = next(s for s in summary.values() if s["task"] == task).get("plans", {})
        for kind in ("base", "ndm", "null"):
            if kind in pl:
                k = pl[kind]
                passes = f"{k['passes']} of {len(k['seeds'])}" if k["passes"] is not None else "-"
                print(f"| {task} | {kind} | {','.join(map(str, k['seeds']))} | {passes} | {k['median_s']:g} s ({k['min_s']:g}-{k['max_s']:g}) |")
    for task in sorted({s["task"] for s in summary.values()}):
        pl = next(s for s in summary.values() if s["task"] == task).get("plans", {})
        for kind, k in pl.items():
            for seed, r in sorted(k["runs"].items()):
                print(f"{task} {kind} s{seed}: {r['outcome']} ({r['planned_s']:g} s)")


if __name__ == "__main__":
    main()
