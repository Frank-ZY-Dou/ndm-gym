"""The tables of results.md from the runs' result.json (final.py): per body and algorithm every seed and the mean ± sample SD,
SuperTrack first; the ended share first, then the errors of the completed episodes, and the options of each run that change
its result (``options``: the smoothness term's weight, empty without it). With ``--speed``: the speed rows
(wmbench/speed_animal.py's <body>_<algorithm>.json in that directory).

    python benchmarks/mimic_mjx_animal_tracking/summarize.py <log dir> [--seeds 0 1 2] [--speed <dir>]
"""

import argparse
import json
import math
import os

BODIES = (("stick", "stick insect"), ("fly", "fruit fly"), ("rat", "rat"))
ALGOS = (("supertrack", "SuperTrack"), ("supermimic", "SuperMimic"))


def mean_sd(xs: list, fmt: str = "{:.3g}") -> str:
    """Mean ± sample SD of the values given (None: a run without the value, e.g. no completed episode, counted apart)."""
    v = [x for x in xs if x is not None and math.isfinite(x)]
    if not v:
        return "-"
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return f"{fmt.format(m)} ± {fmt.format(sd)}" + ("" if len(v) == len(xs) else f" ({len(v)} of {len(xs)})")


def num(x, fmt: str = "{:.3g}") -> str:
    return "-" if x is None else fmt.format(x)


def run_rate(run: str) -> float:
    """The weight of the smoothness term (--target_rate_w) a run trained with, for a result that does not record it (final.py of before
    the option): from its args.json; a run without the option or without its arguments: 0."""
    f = os.path.join(run, "args.json")
    return float((json.load(open(f)) if os.path.exists(f) else {}).get("target_rate_w", 0.0))


def table(logs: str, seeds: list[int]) -> str:
    out = [
        "| body | algorithm | seed | stop (reason) | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | "
        "held-out: bodies ↓ | to the target ↓ | it/s | options |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for animal, body in BODIES:
        for algo, name in ALGOS:
            rates = {}  # each seed's smoothness weight: one row averages seeds of one weight only
            for s in seeds:
                f = os.path.join(logs, f"{animal}_{algo}_s{s}", "result.json")
                if os.path.exists(f):
                    r = json.load(open(f))
                    rates[s] = float(r["target_rate_w"]) if "target_rate_w" in r else run_rate(os.path.dirname(f))
            if len(set(rates.values())) > 1:
                raise SystemExit(f"summarize: the {body}'s {name} seeds trained with different weights of the smoothness term "
                                 f"({', '.join(f'seed {s}: {x:g}' for s, x in sorted(rates.items()))}): no row of them")
            cols = {k: [] for k in ("ended", "body", "ee", "joint", "h_ended", "h_body", "ttt", "rate")}
            for s in seeds:
                f = os.path.join(logs, f"{animal}_{algo}_s{s}", "result.json")
                if not os.path.exists(f):
                    out.append(f"| {body} | {name} | {s} | missing | | | | | | | | | |")
                    continue
                r = json.load(open(f))
                t, h, ttt = r["final"]["train"], r["final"]["heldout"], r["time_to_target"]["minutes"]
                vals = {"ended": 100 * t["ended_share"], "body": t["body_mm"], "ee": t["end_eff_mm"], "joint": t["joint_deg"],
                        "h_ended": 100 * h["ended_share"], "h_body": h["body_mm"], "ttt": ttt, "rate": r["it_per_s_301_to_stop"]}  # fmt: skip
                for k, v in vals.items():
                    cols[k].append(v)
                out.append(
                    f"| {body} | {name} | {s} | {r['stop_iteration']:,} ({r['stop_reason']}) | {vals['ended']:.1f} % | "
                    f"{num(vals['body'])} mm | {num(vals['ee'])} mm | {num(vals['joint'])}° | {vals['h_ended']:.1f} % | "
                    f"{num(vals['h_body'])} mm | {'not reached' if ttt is None else f'{ttt:.1f} min'} | {num(vals['rate'], '{:.2f}')} | "
                    + ("" if not rates[s] else f"target_rate_w {rates[s]:g}") + " |"
                )
            out.append(
                f"| {body} | {name} | mean ± SD | | {mean_sd(cols['ended'], '{:.1f}')} % | {mean_sd(cols['body'])} mm | "
                f"{mean_sd(cols['ee'])} mm | {mean_sd(cols['joint'])}° | {mean_sd(cols['h_ended'], '{:.1f}')} % | "
                f"{mean_sd(cols['h_body'])} mm | {mean_sd(cols['ttt'], '{:.1f}')} min | {mean_sd(cols['rate'], '{:.2f}')} | |"
            )
    return "\n".join(out)


def speed_table(d: str) -> str:
    """Per body and world model: world-steps per second of the MuJoCo-Warp body and of the world model in fp32 and as its bf16
    copy, at each count of worlds, their ratio, and the open-loop error of the bodies after 1, 8 and 16 control steps."""
    out = ["| body | world model | worlds | MuJoCo-Warp | fp32 | bf16 | bf16 / MuJoCo-Warp | open loop, 1 / 8 / 16 steps |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for animal, body in BODIES:
        for algo, name in ALGOS:
            f = os.path.join(d, f"{animal}_{algo}.json")
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            model = f"{name}'s {'acceleration' if r['wm_kind'] == 'accel' else 'increment'} model"
            by = {}
            for row in r["rows"]:
                by.setdefault(row["nworld"], {})[row["what"]] = row.get("world_steps_per_s")
            ol = r.get("open_loop", {}).get("bodies_mm", {})
            loop = " / ".join(num(ol.get(k)) for k in ("1", "8", "16")) + " mm" if ol else "-"
            for n, w in sorted(by.items()):
                ph, f32, b16 = w.get("physics"), w.get("ndm_fp32"), w.get("ndm_bf16")
                ratio = f"{b16 / ph:.1f}x" if ph and b16 else "-"
                out.append(f"| {body} | {model} | {n:,} | {num(ph, '{:,.0f}')} | {num(f32, '{:,.0f}')} | {num(b16, '{:,.0f}')} | {ratio} | {loop} |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("logs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--speed", default=None, help="the speed rows' directory (<body>_<algorithm>.json)")
    a = p.parse_args()
    print(table(a.logs, a.seeds))
    if a.speed:
        print()
        print(speed_table(a.speed))


if __name__ == "__main__":
    main()
