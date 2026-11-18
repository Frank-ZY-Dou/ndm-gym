"""Score measured 20 Hz key masks by pitch and onset, independent of the simulator."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from piano_score import DT, LOWEST_PITCH

TICK_MS = round(1000 * DT)


def intervals(mask):
    """(pitch index, onset, first released frame). An interval ending at len(mask) is right-censored."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), ((1, 1), (0, 0)))
    edges = np.diff(padded, axis=0)
    notes = []
    for pitch in np.flatnonzero(np.any(edges, axis=0)):
        on = np.flatnonzero(edges[:, pitch] == 1)
        off = np.flatnonzero(edges[:, pitch] == -1)
        notes.extend((int(pitch), int(a), int(b)) for a, b in zip(on, off, strict=True))
    return sorted(notes, key=lambda n: (n[1], n[0]))


def match_notes(target, played, tolerance=1):
    """Maximum number of same-pitch matches, then minimum total absolute onset error."""
    pairs = []
    for pitch in sorted({n[0] for n in target} & {n[0] for n in played}):
        a = [i for i, n in enumerate(target) if n[0] == pitch]
        b = [i for i, n in enumerate(played) if n[0] == pitch]
        distance = np.abs(np.subtract.outer([target[i][1] for i in a], [played[i][1] for i in b]))
        penalty = (min(len(a), len(b)) + 1) * (tolerance + 1)
        rows, cols = linear_sum_assignment(np.where(distance <= tolerance, distance, penalty))
        pairs.extend((a[i], b[j]) for i, j in zip(rows, cols, strict=True) if distance[i, j] <= tolerance)
    return pairs


def event_stats(goal, masks):
    target = intervals(goal)
    counts = Counter(n[1] for n in target)
    simultaneous = {i for i, n in enumerate(target) if counts[n[1]] > 1}
    repeated = set()
    for bass in (False, True):
        voice = [i for i, n in enumerate(target) if (n[0] + LOWEST_PITCH < 60) == bass]
        repeated.update(j for i, j in zip(voice, voice[1:]) if target[i][0] == target[j][0])
    onset, release = [], []
    hits = extras = repeat_hits = chord_hits = 0
    for mask in masks:
        played = intervals(mask)
        pairs = match_notes(target, played)
        hits += len(pairs)
        extras += len(played) - len(pairs)
        for i, j in pairs:
            _, a, b = target[i]
            _, c, d = played[j]
            onset.append(abs(a - c) * TICK_MS)
            if max(b, d) < len(goal):
                release.append(abs(b - d) * TICK_MS)
            repeat_hits += i in repeated
            chord_hits += i in simultaneous
    total = len(masks) * len(target)
    missed = total - hits

    def ratio(n, d):
        return n / d if d else 1.0

    return dict(
        episodes=len(masks),
        matched=hits,
        extra=extras,
        missed=missed,
        precision=ratio(hits, hits + extras),
        recall=ratio(hits, total),
        f1=ratio(2 * hits, 2 * hits + missed + extras),
        onset_median_ms=float(np.median(onset)) if onset else None,
        release_median_ms=float(np.median(release)) if release else None,
        observed_releases=len(release),
        repeat_hits=repeat_hits,
        repeat_total=len(repeated) * len(masks),
        paired_note_hits=chord_hits,
        paired_note_total=len(simultaneous) * len(masks),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluations", type=Path, nargs="+", help="evaluation.npz from a final policy or the script")
    args = p.parse_args()
    for path in args.evaluations:
        with np.load(path) as traces:
            print(json.dumps(dict(file=str(path), **event_stats(traces["goal"], traces["played"]))))


if __name__ == "__main__":
    main()
