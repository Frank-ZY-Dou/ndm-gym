"""Tables of the policy-gradient diagnostic (tools/grad_stats.py via benchmarks/g1_lafan1_tracking/grad_diag.sh): one row
per run and setting from the summaries, as markdown.

Inputs: run directories holding <setting>.json (the diagnostic's outputs), and/or files of summary lines as the script
prints them (one JSON object per line with "run" and "set"), e.g. copied from another machine's output.

Usage: python tools/grad_diag_report.py logs/grad_diag/<run> ... [--lines summaries.jsonl ...] [--md out.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

SETTINGS = ("base", "rf_rope_detach=1", "bptt_clip_k=3", "bptt_clip_k=10", "wm_bptt_fp32=1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="directories with the diagnostic's <setting>.json files")
    p.add_argument("--lines", nargs="*", default=[], help="files of printed summary lines")
    p.add_argument("--md", default=None)
    return p.parse_args()


def rows(args) -> list[dict]:
    out = []
    for d in args.runs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            j = json.load(open(f))
            out.append({"run": j["meta"]["run"], "set": ",".join(j["meta"]["set"]) or "base", **j["summary"]})
    for path in args.lines:
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                j = json.loads(line)
                out.append({**j, "set": ",".join(j["set"]) or "base"})
    order = {s: i for i, s in enumerate(SETTINGS)}
    return sorted(out, key=lambda r: (r["run"], order.get(r["set"], len(order))))


def main():
    args = parse_args()
    lines = [
        "| run | setting | median norm | p99 | max | > 3x median | > 10x median | cos to mean direction | cos of the > 3x ones | "
        "consistency | mean direction against base | s per update |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows(args):
        spk = r["cos_to_ref_spikes_mean"]
        lines.append(f"| {r['run']} | {r['set']} | {r['gradnorm_median']:.2f} | {r['gradnorm_p99']:.1f} | {r['gradnorm_max']:.1f} | "
                     f"{r['above_3x_median']} | {r['above_10x_median']} | {r['cos_to_ref_mean']:.3f} | {'-' if spk is None else f'{spk:.2f}'} | "
                     f"{r['mean_unit_gradient_length']:.3f} | {r['mean_unit_gradient_cos_to_ref']:.3f} | {r['sec_per_update_median']:.2f} |")  # fmt: skip
    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
