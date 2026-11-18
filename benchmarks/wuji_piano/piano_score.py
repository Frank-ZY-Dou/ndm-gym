"""Physical sampling constants and the timed, unassigned Eden Ode to Joy score."""

import numpy as np

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
