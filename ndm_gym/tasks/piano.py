"""The Wuji piano task's physical sampling constants, Eden's framewise scoring and the timed, unassigned Eden Ode to Joy
score (benchmarks/wuji_piano).

The score is adapted from Eden's ``eden/tasks/piano/songs.py`` (``ode_to_joy``; Apache-2.0, Copyright 2026 Kashu Yamazaki;
https://github.com/embodied-ai-nexus/Eden at 148d972): the same melody, bass notes, durations, rests and tempo, written here
as note intervals sampled at the 20 Hz control rate after a one-second lead-in, without Eden's ``Song`` and ``Note``
types. ``frame_metrics`` scores as Eden's ``custom_terms/metrics.py``: framewise precision, recall and F1 over all 88
keys, a zero denominator counting as 1."""

import numpy as np
import torch

PHYSICS_STEP = 0.005
CONTROL_SUBSTEPS = 10
DT = CONTROL_SUBSTEPS * PHYSICS_STEP
KEY_COUNT = 88
LOWEST_PITCH = 21
KEY_DOWN_FRACTION = 0.87
MELODY = (64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62)


def piano_roll():
    edges = np.cumsum((0, *([1.0] * 12), 1.5, 0.5, 2.0))
    notes = [(p, t, u - t - 0.25) for p, t, u in zip(MELODY, edges[:-1], edges[1:], strict=True)]
    notes += [(p, 4 * i, 3.75) for i, p in enumerate((48, 43, 48, 43))]

    def boundary(beat):
        return int(np.ceil((1 + beat * 60 / 90) / DT - 1e-9))

    goal = np.zeros((boundary(max(t + d for _, t, d in notes)), KEY_COUNT), bool)
    for p, t, d in notes:
        goal[boundary(t) : boundary(t + d), p - LOWEST_PITCH] = True
    return goal


def frame_metrics(played, goal):
    hits = (played & goal).sum(-1)
    down = played.sum(-1)
    due = goal.sum(-1)

    def ratio(count, total):
        return torch.where(total > 0, count / total.clamp_min(1), 1.0)

    return torch.stack((ratio(hits, down), ratio(hits, due), ratio(2 * hits, down + due)), dim=-1)
