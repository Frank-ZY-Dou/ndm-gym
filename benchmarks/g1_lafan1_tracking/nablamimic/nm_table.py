"""NablaMimic's own runs (run.sh): nm_table.py [--target 0.82] <run directory>... prints, per run, what its own files say: config.json
(its commit, GPU, configuration file and the options given after it), log.jsonl (its training clock wall_s per update, seconds per
update, GPU memory) and eval_nominal.json of its scripts/evaluate_run.sh (per evaluated checkpoint its full-clip normalized return
and the clips with a fall). Columns: seconds per update (the median from update 50 on); the first evaluated checkpoint at or above
the target (update / minutes on its clock / minutes from update 50, as this study counts from iteration 50); that crossing
interpolated linearly between the two evaluations around it (its paper's rule, here per seed); the first of three evaluations in a
row at or above the target without a fall (this study's stable); the best and the last evaluation. Then per arm (the run's name
without _s<seed>) the seeds' mean and sample standard deviation. NablaMimic's evaluator, simulator and G1 model score these runs, not
this library's: the rows stand beside this library's, not in one ranking (docs/algorithms/nablamimic.md, section 4)."""

import json
import os
import re
import statistics
import sys

FROM = 50  # minutes are also counted from this update (this study's iteration 50); its own clock starts after its compilation
MEM = ("peak_bytes_in_use", "bytes_limit")  # of log.jsonl's gpu_mem_gb (JAX's allocator, GB)


def read_log(run):
    """{update: record} of log.jsonl (one record per 10 updates; a resumed run repeats updates, the later record counts), and
    whether its last line is cut (a run that was killed while writing)."""
    out, cut = {}, False
    path = os.path.join(run, "log.jsonl")
    if not os.path.exists(path):
        return out, cut
    lines = open(path).read().splitlines()
    for n, line in enumerate(lines, 1):
        if not line.startswith('{"iter"'):
            continue
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            if n != len(lines):
                raise SystemExit(f"{path}:{n}: not a JSON record")
            cut = True
            continue
        out[int(x["iter"])] = x
    return out, cut


def read_eval(run):
    """[(update, wall_s, full_norm_return, clips with a fall)] of eval_nominal.json by update, its number of clips and its GPU."""
    path = os.path.join(run, "eval_nominal.json")
    if not os.path.exists(path):
        return [], 0, None
    ev = json.load(open(path))
    rows = []
    for key, rec in ev.items():
        if not re.fullmatch(r"it\d+/nominal", key):
            continue
        if len(rec["per_clip"]) != len(ev["clips"]):
            raise SystemExit(f"{path}: {key} has {len(rec['per_clip'])} clips, the file names {len(ev['clips'])}")
        falls = sum(c["fall_s"] is not None for c in rec["per_clip"])
        rows.append((int(rec["iter"]), float(rec["wall_s"]), float(rec["full_norm_return"]), falls, rec["gpu"]))
    rows.sort()
    return [r[:4] for r in rows], len(ev["clips"]), (rows[0][4] if rows else None)


def first_at(ev, target):
    return next(((it, w) for it, w, r, _ in ev if r >= target), None)


def crossing(ev, target):
    """Seconds on its clock at which the return first crosses the target, linear between the two evaluations around it; the first
    evaluation's own time when that one is already at the target."""
    for i, (_, w, r, _) in enumerate(ev):
        if r >= target:
            if i == 0:
                return w
            w0, r0 = ev[i - 1][1], ev[i - 1][2]
            return w0 + (target - r0) * (w - w0) / (r - r0)
    return None


def held(ev, target, n=3):
    ok = [r >= target and f == 0 for _, _, r, f in ev]
    return next(((ev[i][0], ev[i][1]) for i in range(len(ev) - n + 1) if all(ok[i : i + n])), None)


def run_state(run, log):
    """Its end marker and what the marker holds (DONE: the last update; DIVERGED: the update of the non-finite loss; HALTED: SHAC's
    guard), or how far the log of a run without one goes."""
    for name in ("DIVERGED", "HALTED", "DONE"):
        path = os.path.join(run, name)
        if os.path.exists(path):
            return f"{name} {open(path).read().strip()[:60]}"
    return "not finished" + (f", log to update {max(log)}" if log else "")


def read_run(run):
    cfg_path = os.path.join(run, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{run}: no config.json (not a run directory of nablamimic.train)")
    cfg = json.load(open(cfg_path))
    log, cut = read_log(run)
    ev, n_clips, eval_gpu = read_eval(run)
    notes = []
    if cfg.get("git_dirty") or any(s.get("dirty") for s in cfg.get("segments", [])):
        notes.append("its source was modified: not NablaMimic's code")
    if cfg.get("gpu") == "cpu":
        notes.append("ran on the CPU: a test, not a result")
    if len(cfg.get("segments", [])) > 1:
        notes.append(f"resumed ({len(cfg['segments'])} segments): not one timed run")
    launch = os.path.normpath(run) + ".launch.txt"  # run.sh's record, beside the run directory
    if os.path.exists(launch):
        rec = dict(line.split(" ", 1) for line in open(launch).read().splitlines() if " " in line)
        if rec.get("pin") and cfg.get("git_commit") != rec["pin"]:
            notes.append(f"its commit is not the pinned {rec['pin'][:7]}")
    if cut:
        notes.append("the log's last line is cut")
    over = cfg["argv"][len(cfg["config_args"]) :] if "config_args" in cfg else []
    conf = (os.path.basename(str(cfg.get("config_file") or "no --config")) + " " + " ".join(over)).strip()
    windows = f"{cfg['num_envs']} x {cfg['horizon']} windows {'kept' if cfg['no_remat'] else 'rematerialized'}"
    steady = [x["iter_s"] for it, x in log.items() if it >= FROM]
    mem = [x["gpu_mem_gb"] for x in log.values() if x.get("gpu_mem_gb")]
    return {
        "name": os.path.basename(os.path.normpath(run)),
        "conf": f"{conf}; {windows}, replay {cfg['replay_steps']} x {cfg['collect_envs'] or cfg['num_envs']}",
        "commit": (cfg.get("git_commit") or "no git")[:7],
        "gpu": str(cfg.get("gpu")) + (f" (evaluated on {eval_gpu})" if eval_gpu not in (None, cfg.get("gpu")) else ""),
        "state": run_state(run, log),
        "notes": notes,
        "ev": ev,
        "n_clips": n_clips,
        "w_from": log[FROM]["wall_s"] if FROM in log else None,
        "s_update": statistics.median(steady) if steady else None,
        "mem": {k: max(m[k] for m in mem) for k in MEM if mem and all(k in m for m in mem)},
    }


def cells(r, target):
    """The run's evaluation columns: first at the target, interpolated crossing, held, best, last."""
    ev, w0 = r["ev"], r["w_from"]
    if not ev:
        return ["not evaluated"] * 5
    from_update = lambda w: "–" if w0 is None or w < w0 else f"{(w - w0) / 60:.2f}"  # noqa: E731
    both = lambda t: "–" if t is None else f"{t[0]} / {t[1] / 60:.2f} / {from_update(t[1])}"  # noqa: E731
    x = crossing(ev, target)
    best, last = max(ev, key=lambda e: e[2]), ev[-1]
    return [
        both(first_at(ev, target)),
        "–" if x is None else f"{x / 60:.2f}",
        both(held(ev, target)),
        f"{best[2]:.3f} ({best[0]})",
        f"{last[2]:.3f}, {last[3]} of {r['n_clips']}, {last[1] / 3600:.2f} h ({last[0]})",
    ]


def main(argv):
    target = 0.82
    if argv[:1] == ["--target"]:
        target, argv = float(argv[1]), argv[2:]
    if not argv:
        raise SystemExit(__doc__)
    runs = [read_run(run) for run in argv]
    print("NablaMimic's own code and evaluation (MuJoCo Warp, its G1 model; the full-clip normalized return of the actor it evaluates).")
    print(f"first ≥ {target:g} and held: update / minutes on its clock / minutes from update {FROM}.\n")
    print(f"| run | configuration | code | GPU | s per update | first ≥ {target:g} | crossing, min (interpolated) "
          f"| ≥ {target:g} without a fall, 3 in a row | best return (update) "
          "| last evaluation: return, clips with a fall, training time (update) | peak GPU memory | state |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    arms = {}
    for r in runs:
        mem = " of ".join(f"{r['mem'][k]:.1f}" for k in MEM if k in r["mem"])
        s = "–" if r["s_update"] is None else f"{r['s_update']:.3f}"
        row = [r["name"], r["conf"], r["commit"], r["gpu"], s, *cells(r, target), mem + " GB" if mem else "–", "; ".join([r["state"]] + r["notes"])]
        print("| " + " | ".join(row) + " |")
        arms.setdefault(re.sub(r"_s\d+$", "", r["name"]), []).append(r)
    pm = lambda v: f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"  # noqa: E731

    def every(v, n):  # the seeds' values and their mean ± std, when every seed has one
        if len(v) < n:
            return f"{len(v)} of {n} seeds" if v else "not reached"
        return ", ".join(f"{x:.2f}" for x in v) + (f" ({pm(v)})" if n > 1 else "")

    print(f"\n| arm | seeds | s per update | first ≥ {target:g}: updates | minutes on its clock | minutes from update {FROM} "
          "| crossing, min (interpolated) | clips with a fall at the last evaluation |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|")
    for arm, rs in arms.items():
        firsts = [first_at(r["ev"], target) for r in rs]
        ups = ", ".join("–" if f is None else str(f[0]) for f in firsts)
        clock = [f[1] / 60 for f in firsts if f]
        from50 = [(f[1] - r["w_from"]) / 60 for f, r in zip(firsts, rs) if f and r["w_from"] is not None and f[1] >= r["w_from"]]
        cross = [x / 60 for x in (crossing(r["ev"], target) for r in rs) if x is not None]
        su = [r["s_update"] for r in rs if r["s_update"] is not None]
        falls = ", ".join(str(r["ev"][-1][3]) if r["ev"] else "–" for r in rs)
        f50 = every(from50, len(rs)) if len(from50) == len(clock) else "–"  # –: a run without that update's record, or at the target before it
        row = [arm, str(len(rs)), pm(su) if su else "–", ups, every(clock, len(rs)), f50, every(cross, len(rs)), falls]
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
