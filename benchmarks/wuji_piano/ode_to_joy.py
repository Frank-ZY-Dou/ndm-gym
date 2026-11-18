"""The unassigned Eden Ode to Joy piano roll and its two freely moving Wuji Hand 2 hands."""

from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from piano_score import KEY_COUNT, PHYSICS_STEP
from piano_score import LOWEST_PITCH as LOWEST_PITCH, piano_roll as piano_roll
from five_finger_scale import DIGITS

SIDES = ("left", "right")
AXES = ("x", "y", "z", "rx", "ry", "rz")
WRIST_SCALE = (0.04, 0.10, 0.03, 0.15, 0.15, 0.30)
REST = ((0.35, -0.17, 0.80, 0), (0.38, -0.08, 1.15, 0.80), (0.56, 0.02, 0.95, 0.70), (0.56, 0, 0.90, 0.55), (0.72, 0.12, 0.45, 0.26))


def build(hand_dir: Path, keyboard: Path):
    spec = mj.MjSpec.from_file(str(keyboard))
    spec.option.timestep = PHYSICS_STEP
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    for key in range(KEY_COUNT):
        spec.body(f"key_{key}").gravcomp = 1
    for side in SIDES:
        hand = mj.MjSpec.from_file(str(hand_dir / f"{side}.xml"))
        hand.option = spec.option
        for body in hand.bodies[1:]:
            body.gravcomp = 1
        sign = 1 if side == "left" else -1
        palm = Rotation.from_matrix([[0, 0, 1], [sign, 0, 0], [0, sign, 0]])
        orientation = Rotation.from_euler("x", -sign * 15, degrees=True) * palm
        center = [0.145, -0.160 if side == "left" else -0.027, 0.10]
        root = spec.worldbody.add_body(name=side + "_base", pos=center)
        root.add_freejoint(name=side + "_free")
        # Both sites have world-aligned axes at rest. Six orthogonal motors apply a Cartesian wrench.
        root.add_site(name=side + "_target", size=[0.001] * 3)
        spec.worldbody.add_site(name=side + "_reference", size=[0.001] * 3, pos=center)
        spec.attach(hand, prefix=side + "/", frame=root.add_frame(quat=orientation.as_quat(scalar_first=True)))
        for i, axis in enumerate(AXES):
            limit = 100 if i < 3 else 10
            actuator = spec.add_actuator(
                name=f"{side}_{axis}",
                trntype=mj.mjtTrn.mjTRN_SITE,
                target=side + "_target",
                refsite=side + "_reference",
                gear=np.eye(6)[i],
                forcerange=[-limit, limit],
            )
            actuator.set_to_position(kp=2000 if i < 3 else 40, kv=80 if i < 3 else 2)
    return spec.compile()


def action_layout(model):
    ids, rest, scale = [], [], []
    for side in SIDES:
        ids.extend(model.actuator(f"{side}_{axis}").id for axis in AXES)
        rest.extend([0] * 6)
        scale.extend(WRIST_SCALE)
        for (_, prefix), angles in zip(DIGITS, REST, strict=True):
            ids.extend(model.actuator(f"{side}/{side[0]}_{prefix}{i}").id for i in range(4))
            rest.extend(angles)
            scale.extend([0.5] * 4)
    assert len(set(ids)) == model.nu == 52
    return np.array(ids), np.float32(rest), np.float32(scale)
