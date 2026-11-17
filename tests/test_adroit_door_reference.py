"""Native Door reference schema, transition alignment and independent opening score."""

import numpy as np
import pytest

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference


def reference():
    return DoorReference(np.zeros((3, 30)), np.zeros((3, 30)), np.zeros((2, 28)),
                         {"kind": "adroit-door-physical-replay-v1", "control_dt": .01,
                          "joint_names": list(JOINT_NAMES)})


def test_roundtrip_and_transition_alignment(tmp_path):
    ref = reference()
    ref.qpos[1, 28] = 1.5
    path = tmp_path / "reference.npz"
    ref.save(path)
    restored = DoorReference.load(path)
    assert restored.frames == len(restored.actions) + 1
    np.testing.assert_array_equal(restored.qpos, ref.qpos)
    assert restored.metadata == ref.metadata
    assert restored.errors(ref.qpos[1], 1) == {"hand_rms": 0., "door_rad": 0., "latch_rad": 0.}


def test_tracking_does_not_imply_opening():
    ref = reference()
    assert ref.errors(ref.qpos[0], 0)["door_rad"] == 0
    assert not ref.opened(ref.qpos[0])
    state = ref.qpos[0].copy()
    state[28:] = [1.5, .2]
    assert ref.opened(state)
    assert ref.errors(state, 0)["door_rad"] == 1.5
    assert ref.errors(state, 0)["latch_rad"] == .2


@pytest.mark.parametrize("field,value", [("qpos", np.zeros((3, 29))), ("qvel", np.zeros((2, 30))),
                                        ("actions", np.zeros((3, 28))), ("qpos", np.full((3, 30), np.nan))])
def test_invalid_arrays(field, value):
    ref = reference()
    with pytest.raises(ValueError):
        DoorReference(**{**ref.__dict__, field: value})


@pytest.mark.parametrize("change", [{"kind": "historical"}, {"control_dt": 0}, {"control_dt": float("nan")},
                                     {"joint_names": []}])
def test_invalid_metadata(change):
    ref = reference()
    with pytest.raises(ValueError):
        DoorReference(ref.qpos, ref.qvel, ref.actions, {**ref.metadata, **change})


def test_frame_bounds():
    ref = reference()
    for frame in [-1, 3]:
        with pytest.raises(ValueError):
            ref.errors(ref.qpos[0], frame)


def test_import_without_simulator():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", "import sys; import ndm_gym.motion.adroit_door; "
                    "assert not any(n in sys.modules for n in ('mujoco', 'gymnasium_robotics', 'isaaclab', 'torch'))"], check=True)
