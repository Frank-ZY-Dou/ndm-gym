"""The tables of benchmarks/steel/results.md from run.sh's output (<dir>/one, <dir>/four): every process's result line in its log
(logs/<tag>.log) is joined to its CSV row by N and throughput; a row is the median of its three repeats, with the mean and the
range beside it. Standard library only. Usage: python benchmarks/steel/table.py <dir> [--json out.json]"""

import argparse
import csv
import glob
import json
import os
import re
import statistics

ROWS = ["st-fastest", "sm-fastest", "st-as-trained", "sm-as-trained", "fastest-passing", "task-physics", "g1-preset", "mujoco-newton",
        "mujoco-isaaclab"]
NAMES = {"st-fastest": "`lafan1-supertrack-accurate`'s world model, bf16 copy with the fused step",
         "sm-fastest": "`lafan1-supermimic-accurate`'s world model, bf16 copy with the fused step",
         "st-as-trained": "`lafan1-supertrack-accurate`'s world model, as trained (compiled)",
         "sm-as-trained": "`lafan1-supermimic-accurate`'s world model, as trained (compiled)",
         "fastest-passing": "Steel, fastest passing setting", "task-physics": "Steel at the task's physics", "g1-preset": "Steel's G1 preset",
         "mujoco-newton": "MuJoCo Warp in Newton", "mujoco-isaaclab": "MuJoCo Warp in Isaac Lab"}  # fmt: skip
FIXED_N = [1024, 4096, 16384, 65536]
PEAK_ROWS = ["st-fastest", "sm-fastest", "fastest-passing"]
FOUR_ROWS = ["st-fastest", "sm-fastest", "fastest-passing", "mujoco-newton"]
LINE = re.compile(r"^\[(steel_g1|speed_pair)\] (?:speed (\S+)|(\S+)) (\d+) (?:worlds|envs): ([\d,]+) "
                  r"(?:control|robot)-?\s?steps/s.*?(?:valid (\d))?\s*$")
END = re.compile(r"^(\S+Z) end (\S+): exit (\d+)")
START = re.compile(r"^(\S+Z) start (\S+) on GPU (\d+)")


def stamp(s):  # 2026-11-22T06:19:54Z -> seconds since the epoch
    import calendar
    import time
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def logs(part_dir):
    """tag -> {sps, valid, rc, start, end} from every process log of a part."""
    out = {}
    for f in sorted(glob.glob(os.path.join(part_dir, "logs", "*.log"))):
        tag = os.path.basename(f)[:-4]
        if tag == "venv":
            continue
        e = {"sps": None, "valid": None, "rc": None, "start": None, "end": None}
        for line in open(f, errors="replace"):
            m = LINE.match(line.rstrip("\n"))
            if m:
                e["sps"] = int(m.group(5).replace(",", ""))
                e["valid"] = None if m.group(6) is None else int(m.group(6))
            m = END.match(line)
            if m and m.group(2) == tag:
                e["end"], e["rc"] = stamp(m.group(1)), int(m.group(3))
            m = START.match(line)
            if m and m.group(2) == tag:
                e["start"] = stamp(m.group(1))
        out[tag] = e
    return out


def csv_rows(part_dir):
    rows = []
    for name, key in (("steel_rows.csv", "steps_per_s"), ("pair_rows.csv", "robot_steps_per_s")):
        p = os.path.join(part_dir, name)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                n = int(r.get("worlds") or r.get("envs"))
                rows.append({"n": n, "sps": round(float(r[key])), "seconds": float(r["seconds"]), "peak_mem_gb": float(r["peak_mem_gb"]),
                             "valid": int(r["valid"]) if "valid" in r else None, "build_s": float(r["build_s"]) if "build_s" in r else None,
                             "gpu": r["gpu"], "driver": r["driver"], "steps": int(r["steps"]), "used": False})
    return rows


def join(part_dir, parse_tag):
    """Each successful process: its log entry, its CSV row (by N and rounded throughput, each row used once)."""
    L, R = logs(part_dir), csv_rows(part_dir)
    out = []
    for tag, e in L.items():
        k = parse_tag(tag)
        if k is None:
            continue
        if e["rc"] != 0:
            out.append({**k, "tag": tag, "rc": e["rc"], "sps": None})
            continue
        assert e["sps"] is not None, f"{tag}: exit 0 but no result line"
        cand = [r for r in R if not r["used"] and r["n"] == k["n"] and abs(r["sps"] - e["sps"]) <= 1]
        assert len(cand) >= 1, f"{tag}: no CSV row with N {k['n']} and {e['sps']} steps/s"
        cand[0]["used"] = True
        out.append({**k, "tag": tag, "rc": 0, "sps": e["sps"], "valid": e["valid"] if e["valid"] is not None else cand[0]["valid"],
                    "peak_mem_gb": cand[0]["peak_mem_gb"], "build_s": cand[0]["build_s"], "seconds": cand[0]["seconds"], "steps": cand[0]["steps"],
                    "gpu": cand[0]["gpu"], "driver": cand[0]["driver"], "start": e["start"], "end": e["end"]})
    left = [r for r in R if not r["used"]]
    assert not left, f"{len(left)} CSV rows without a log: {left[:2]}"
    return out


def one_tag(tag):
    m = re.match(r"^(.+)_n(\d+)_r(\d)_g(\d)$", tag)
    if not m or tag.startswith("prof_"):
        return None
    return {"row": m.group(1), "n": int(m.group(2)), "rep": int(m.group(3)), "card": int(m.group(4))}


def four_tag(tag):
    m = re.match(r"^(.+)_n(\d+)_r(\d)k(\d)_g(\d)$", tag)
    return None if not m else {"row": m.group(1), "n": int(m.group(2)), "rep": int(m.group(3)), "k": int(m.group(4)), "card": int(m.group(5))}


def stats(v):
    return {"median": statistics.median(v), "mean": statistics.mean(v), "min": min(v), "max": max(v), "n": len(v)}


def fmt(s, scale=1e6, d=3):
    return f"{s['median'] / scale:.{d}f} M; {s['mean'] / scale:.{d}f} ({s['min'] / scale:.{d}f}–{s['max'] / scale:.{d}f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    one = join(os.path.join(a.dir, "one"), one_tag) if os.path.isdir(os.path.join(a.dir, "one")) else []
    four = join(os.path.join(a.dir, "four"), four_tag) if os.path.isdir(os.path.join(a.dir, "four")) else []
    res = {"one": {}, "four": {}, "doubling_end": {}, "barrier": []}
    gpus = sorted({(e["gpu"], e["driver"]) for e in one + four if e["rc"] == 0})
    print(f"processes: one GPU {len(one)} ({sum(e['rc'] != 0 for e in one)} non-zero), several GPUs {len(four)}; GPUs and drivers: {gpus}")
    by = {}
    for e in one:
        by.setdefault((e["row"], e["n"]), []).append(e)
    for (row, n), es in by.items():
        ok = [e for e in es if e["rc"] == 0]
        bad = [e for e in es if e["rc"] != 0]
        if bad:
            res["doubling_end"][row] = {"n": n, "rc": [e["rc"] for e in bad]}
        if ok:
            res["one"][f"{row}|{n}"] = {**stats([e["sps"] for e in ok]), "reps": sorted(e["rep"] for e in ok),
                                        "valid": [e["valid"] for e in sorted(ok, key=lambda e: e["rep"])],
                                        "peak_mem_gb": max(e["peak_mem_gb"] for e in ok),
                                        "values": [e["sps"] for e in sorted(ok, key=lambda e: e["rep"])]}
    rows = res["one"]
    for n in FIXED_N:
        if not any(k.endswith(f"|{n}") for k in rows):
            continue
        print(f"\n**N = {n:,}** (control steps per second over all worlds: median; mean (range) of the three runs)\n")
        print("| row | median; mean (range) | peak memory (GB) | no drops, per run | / Steel's fastest passing | / MuJoCo Warp in Isaac Lab |")
        print("|---|---|---|---|---|---|")
        for row in ROWS:
            s_ = rows.get(f"{row}|{n}")
            if s_ is None:
                print(f"| {NAMES[row]} | not run | | | | |")
                continue
            valid = "" if s_["valid"][0] is None else ", ".join(map(str, s_["valid"]))
            r1 = r2 = ""
            if row.endswith("-fastest") and f"fastest-passing|{n}" in rows and f"mujoco-isaaclab|{n}" in rows:
                r1 = f"{s_['median'] / rows[f'fastest-passing|{n}']['median']:.1f}x"
                r2 = f"{s_['median'] / rows[f'mujoco-isaaclab|{n}']['median']:.1f}x"
            print(f"| {NAMES[row]} | {fmt(s_)} | {s_['peak_mem_gb']:.2f} | {valid} | {r1} | {r2} |")
    doubled = sorted({int(k.split("|")[1]) for k in rows if int(k.split("|")[1]) > max(FIXED_N)})
    if doubled:
        print("\n**The largest batches** (median; mean (range))\n")
        print("| row | " + " | ".join(f"N = {n:,}" for n in doubled) + " | largest N | ended at |")
        print("|---|" + "---|" * (len(doubled) + 2))
        for row in PEAK_ROWS:
            cells = ["" if f"{row}|{n}" not in rows else fmt(rows[f"{row}|{n}"]) for n in doubled]
            ns = sorted(int(k.split("|")[1]) for k in rows if k.startswith(row + "|"))
            de = res["doubling_end"].get(row, {})
            print(f"| {NAMES[row]} | " + " | ".join(cells) + f" | {ns[-1]:,} | {de.get('n', 0):,} (exit {', '.join(map(str, de.get('rc', [])))}) |")
    if rows:
        print("\n**Each row at its own peak** (medians)\n")
        print("| row | peak (its N) | / Steel's peak | / MuJoCo Warp in Isaac Lab's peak |")
        print("|---|---|---|---|")
        peak = {r: max((rows[k]["median"], int(k.split("|")[1])) for k in rows if k.startswith(r + "|"))
                for r in ROWS if any(k.startswith(r + "|") for k in rows)}  # fmt: skip
        for r, (v, n) in peak.items():
            r1 = f"{v / peak['fastest-passing'][0]:.1f}x" if "fastest-passing" in peak and r.endswith("-fastest") else ""
            r2 = f"{v / peak['mujoco-isaaclab'][0]:.1f}x" if "mujoco-isaaclab" in peak and r != "mujoco-isaaclab" else ""
            print(f"| {NAMES[r]} | {v / 1e6:.3f} M ({n:,}) | {r1} | {r2} |")
    meet = {}
    for e in four:
        meet.setdefault((e["row"], e["k"], e["rep"]), []).append(e)
    tot = {}
    for (row, k, rep), es in sorted(meet.items()):
        if len(es) != k or any(e["rc"] != 0 for e in es):
            print(f"{row} on {k} GPUs, run {rep}: {len(es)} processes, statuses {[e['rc'] for e in es]}: left out")
            continue
        tot.setdefault((row, k), []).append(sum(e["sps"] for e in es))
        bar = os.path.join(a.dir, "four", "barrier", f"{row}_k{k}_r{rep}")
        t_go = float(open(os.path.join(bar, "go")).read())
        res["barrier"].append({"row": row, "k": k, "rep": rep, "ready": len([f for f in os.listdir(bar) if f.startswith("ready.")]),
                               "go_files": sorted(f for f in os.listdir(bar) if f.startswith("go")), "t_go": t_go,
                               "end_minus_window_end_s": [e["end"] - (t_go + e["seconds"]) for e in es],
                               "end_spread_s": max(e["end"] for e in es) - min(e["end"] for e in es), "windows_s": [e["seconds"] for e in es]})
    if tot:
        print("\n**Several GPUs** (16,384 worlds per GPU, 1,000 timed steps; the total is the sum over the GPUs; median; mean (range); "
              "ratio: the total's median over k times the one-GPU median)\n")
        print("| row | k = 1 | k = 2 | k = 4 | k = 2 / (2 x k = 1) | k = 4 / (4 x k = 1) |")
        print("|---|---|---|---|---|---|")
        for row in FOUR_ROWS:
            if any((row, k) not in tot for k in (1, 2, 4)):
                continue
            s_ = {k: stats(tot[(row, k)]) for k in (1, 2, 4)}
            res["four"][row] = {str(k): {**s_[k], "values": tot[(row, k)]} for k in (1, 2, 4)}
            print(f"| {NAMES[row]} | {fmt(s_[1])} | {fmt(s_[2])} | {fmt(s_[4])} | {s_[2]['median'] / (2 * s_[1]['median']):.3f} |"
                  f" {s_[4]['median'] / (4 * s_[1]['median']):.3f} |")
        B = res["barrier"]
        multi = [b for b in B if b["k"] > 1]
        print(f"\n**Barrier:** {len(B)} meetings ({len(multi)} with k > 1); ready files equal k in {sum(b['ready'] == b['k'] for b in B)}; one go "
              f"file in {sum(b['go_files'] == ['go'] for b in B)}; every process's end stamp at least its timed window's end after the common "
              f"start, within the stamps' 1 s, in {sum(min(b['end_minus_window_end_s']) >= -1 for b in B)}; largest spread of a meeting's end "
              f"stamps {max((b['end_spread_s'] for b in multi), default=0)} s.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
