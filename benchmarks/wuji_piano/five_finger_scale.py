"""Eden's monophonic five-finger exercise and a split-keyboard Wuji Hand 2 scene."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from check_contact import CONTROL_SUBSTEPS, KEY_COUNT, LOWEST_PITCH, PHYSICS_STEP

DT = CONTROL_SUBSTEPS * PHYSICS_STEP
PITCHES = (60, 62, 64, 65, 67)
MELODY = (*PITCHES, 65, 64, 62, 60)
DIGITS = (("thumb", "THJ"), ("index_finger", "FFJ"), ("middle_finger", "MFJ"), ("ring_finger", "RFJ"), ("pinky", "LFJ"))
# Listed in score pitch order, from C4 to G4.
PLAYERS = (("left", 1), ("left", 0), ("right", 0), ("right", 1), ("right", 2))
MIDDLE_KEYS = {"left": 59, "right": 67}


def piano_roll():
    """Eden songs.py at 148d972, including the one-second lead-in and the last note's full duration."""
    seconds_per_beat = 60 / 80

    def boundary(beats):
        return int(np.ceil((1 + beats * seconds_per_beat) / DT - 1e-9))

    roll = np.zeros((boundary(len(MELODY) - 0.25), KEY_COUNT), dtype=bool)
    for beat, pitch in enumerate(MELODY):
        roll[boundary(beat) : boundary(beat + 0.75), pitch - LOWEST_PITCH] = True
    return roll


def build(hand_dir: Path, keyboard: Path):
    spec = mujoco.MjSpec.from_file(str(keyboard))
    spec.option.timestep = PHYSICS_STEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for key in range(KEY_COUNT):
        spec.body(f"key_{key}").gravcomp = 1
    for side, middle in MIDDLE_KEYS.items():
        hand = mujoco.MjSpec.from_file(str(hand_dir / f"{side}.xml"))
        hand.option = spec.option
        for body in hand.bodies[1:]:
            body.gravcomp = 1
        sign = 1 if side == "left" else -1
        orientation = Rotation.from_matrix([[0, 0, 1], [sign, 0, 0], [0, sign, 0]])
        frame = spec.worldbody.add_frame(
            pos=[0.158, spec.body(f"key_{middle - LOWEST_PITCH}").pos[1], 0.073], quat=orientation.as_quat(scalar_first=True)
        )
        spec.attach(hand, prefix=side + "/", frame=frame)
    model = spec.compile()
    model.actuator_gainprm[:, 0] *= 7
    model.actuator_biasprm[:, 1] *= 7
    model.actuator_biasprm[:, 2] *= np.sqrt(7)
    return model


def finger_ranges(model):
    """Solve static hover and press postures once. No timing or per-note action trajectory enters this map."""
    data = mujoco.MjData(model)
    digit_ids = {}
    hover = np.zeros(model.nu, np.float32)
    press = hover.copy()
    for side, middle in MIDDLE_KEYS.items():
        order = 1 if side == "right" else -1
        for digit, (finger, prefix) in enumerate(DIGITS):
            ids = np.array([model.actuator(f"{side}/{side[0]}_{prefix}{j}").id for j in range(4)])
            digit_ids[side, digit] = ids
            addr = model.jnt_qposadr[model.actuator_trnid[ids, 0]]
            lo, hi = model.actuator_ctrlrange[ids].T
            site = model.site(f"{side}/{side[0]}_{finger}_tip").id
            y = model.body(f"key_{middle - LOWEST_PITCH}").pos[1] + order * ((digit - 2) * 0.0235 + (0.0045 if digit == 0 else 0))
            x = (0.050, 0.036, 0.037, 0.035, 0.050)[digit]
            preferred = np.array([0.25, 0, 0.40, 0.25])

            def solve(z):
                target = np.array([x, y, z])

                def error(q):
                    data.qpos[addr] = q
                    mujoco.mj_kinematics(model, data)
                    return np.r_[1000 * (data.site_xpos[site] - target), 0.5 * (q - preferred)]

                return least_squares(error, preferred, bounds=(lo + 0.002, hi - 0.002), max_nfev=180).x

            active = (side, digit) in PLAYERS
            hover[ids] = solve(0.045 if active else 0.058)
            press[ids] = solve(0.005) if active else hover[ids]
    return hover, press, np.stack([digit_ids[key] for key in PLAYERS])
