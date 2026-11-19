"""Summarize final checkpoints, measured key masks and synchronized timers from a three-seed Ode to Joy run."""

import argparse
import json
from pathlib import Path

import numpy as np


def onsets(keys):
    return keys & ~np.concatenate((np.zeros_like(keys[..., :1, :]), keys[..., :-1, :]), axis=-2)


def chord_statistics(played, goal):
    """At each two-note score onset, match both pitches within 50 ms. Count attacks and measure inter-key delay."""
    expected = onsets(goal)
    observed = onsets(played)
    gaps = []
    simultaneous = 0
    beat_frames = np.flatnonzero(expected.sum(-1) == 2)
    for frame in beat_frames:
        pitches = np.flatnonzero(expected[frame])
        lo, hi = max(0, frame - 1), min(len(goal), frame + 2)
        candidates = observed[:, lo:hi, :][:, :, pitches]
        for window in candidates:
            if window.any(0).all():
                times = [min(np.flatnonzero(col) + lo, key=lambda t: abs(t - frame)) for col in window.T]
                gap = abs(times[0] - times[1])
                gaps.append(gap * 50)
                simultaneous += gap <= 1
    return dict(matched=len(gaps), within_one_step=int(simultaneous), total=len(played) * len(beat_frames), gaps_ms=gaps)


def summarize(run):
    rows = []
    for seed in (0, 1, 2):
        folder = run / f"seed{seed}"
        results = json.loads((folder / "results.json").read_text())
        with np.load(folder / "evaluation.npz") as trace:
            goal, down = trace["goal"], trace["played"]
        totals = down.sum(-1) + goal.sum(-1)
        expected_f1 = np.divide(2 * (down & goal).sum(-1), totals, out=np.ones_like(totals, float), where=totals != 0).mean()
        np.testing.assert_allclose(expected_f1, results["test"]["f1"], atol=1e-6)
        pick = next((r for r in results["history"] if r["f1"] >= 0.95), None)
        row = {
            **results["test"],
            "seed": seed,
            "training_s": results["training_s"],
            "eval_s": results["evaluation_s"],
            "baseline_f1": results["baseline"]["f1"],
            "target_s": pick["seconds"] if pick else None,
            "chords": chord_statistics(down, goal),
            "throughput": results["settings"]["worlds"] * results["settings"]["steps"] * results["round"] / results["training_s"],
        }
        for part in ("collect_s", "wm_s", "pi_s", "validate_s"):
            row[part] = sum(r[part] for r in results["history"])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    rows = summarize(args.run)
    print(
        "| Seed | Baseline F1 | Precision | Recall | Final F1 | Both notes down | Exact two-note frames | All 19 notes reached | Time to .95 | Training | Final evaluation |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        when = "unreached" if r["target_s"] is None else f"{r['target_s']:.1f} s"
        print(
            f"| {r['seed']} | {r['baseline_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['both_due']:.1%} | "
            f"{r['exact_poly']:.1%} | {r['all_notes']}/{r['starts']} | {when} | {r['training_s']:.1f} s | {r['eval_s']:.1f} s |"
        )

    def stat(k, d=3, scale=1):
        x = np.array([r[k] for r in rows]) * scale
        return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}"

    print(
        f"\nAcross training seeds, final F1 was **{stat('f1',4)}**, precision {stat('precision',4)}, recall {stat('recall',4)}, "
        f"and training time **{stat('training_s',1)} s** (mean ± sample SD). Total timed training was {sum(r['training_s'] for r in rows)/60:.2f} GPU-minutes. "
        f"Mean end-to-end training throughput was {stat('throughput',1)} collected control transitions per second."
    )
    print(
        f"\nAll due pitches were down on {stat('both_due',1,100)}% of the two-note frames. The entire keyboard matched the score on "
        f"{stat('exact_poly',1,100)}% of those frames and {stat('exact_frames',1,100)}% of all frames. "
        f"The policies reached {stat('notes_reached',2)} of 19 note events per episode. "
        f"{sum(r['perfect'] for r in rows)} of {sum(r['starts'] for r in rows)} episodes matched all 230 frames."
    )
    matched = sum(r["chords"]["matched"] for r in rows)
    total = sum(r["chords"]["total"] for r in rows)
    close = sum(r["chords"]["within_one_step"] for r in rows)
    gaps = [g for r in rows for g in r["chords"]["gaps_ms"]]
    print(
        f"\nAt the four written two-key attacks per episode, both measured onsets were within ±50 ms of the score on {matched:,}/{total:,} attacks. "
        f"On {close:,}/{total:,} attacks the two keys also started within 50 ms of each other."
    )
    if gaps:
        print(f"The median inter-key delay among the matched attacks was {np.median(gaps):.0f} ms, sampled at 20 Hz.")
    print("\n| Training component | Seconds per seed, mean ± SD |\n|---|---:|")
    for text, key in (("Physics collection", "collect_s"), ("World-model updates", "wm_s"), ("Policy updates", "pi_s"), ("Validation", "validate_s")):
        print(f"| {text} | {stat(key,1)} |")


if __name__ == "__main__":
    main()
