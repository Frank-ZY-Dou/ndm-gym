"""Physical transition contracts on a small native-coordinate model, no assets."""

import numpy as np
import pytest
import torch

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.policies.adroit_door import DoorPolicy
from ndm_gym.sim.rigid.adroit_door import DoorPhysics


def setup():
    mujoco = pytest.importorskip("mujoco")
    # Separate bodies keep the test tiny; real native asset replay is an integration check.
    bodies = ''.join(f'<body pos="{i * .1} 0 0"><joint name="{name}" type="hinge"/>'
                     '<geom size=".01" mass=".1"/></body>' for i, name in enumerate(JOINT_NAMES))
    actuators = ''.join(f'<motor joint="{name}" ctrllimited="true" ctrlrange="-2 4"/>'
                        for name in JOINT_NAMES[:28])
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002" gravity="0 0 0"/>'
                                         f'<worldbody><body name="frame">{bodies}</body></worldbody>'
                                         f'<actuator>{actuators}</actuator></mujoco>')
    q = np.zeros((3, 30))
    ref = DoorReference(q, q.copy(), np.zeros((2, 28)),
                        {"kind": "adroit-door-physical-replay-v1", "joint_names": list(JOINT_NAMES),
                         "control_dt": .01, "frame_skip": 5, "scene_door_body_pos": [0, 0, 0]})
    return model, ref


def test_controls_terminal_and_explicit_reset():
    model, ref = setup()
    sim = DoorPhysics(model, ref)
    (q, _), done = sim.step(np.zeros(28))
    assert not done and np.any(q[:28] != 0) and not q[28:].any()
    np.testing.assert_array_equal(sim.data.ctrl, np.ones(28))
    state, done = sim.step(np.zeros(28))
    assert done
    with pytest.raises(RuntimeError, match="Reset"):
        sim.step(np.zeros(28))
    np.testing.assert_array_equal(sim.state()[0], state[0])
    sim.reset(qpos_offset=np.full(30, .001))
    np.testing.assert_allclose(sim.state()[0], .001)
    sim.reset()
    np.testing.assert_array_equal(sim.state()[0], ref.qpos[0])


def test_reject_assists_bad_actions_and_mismatched_dt():
    model, ref = setup()
    sim = DoorPhysics(model, ref)
    for a in (np.zeros(27), np.full(28, np.nan)):
        with pytest.raises(ValueError):
            sim.step(a)
    sim.data.qfrc_applied[28] = 1
    with pytest.raises(RuntimeError, match="forces"):
        sim.step(np.zeros(28))
    model.opt.timestep *= 2
    with pytest.raises(ValueError, match="timestep"):
        DoorPhysics(model, ref)


def test_policy_dimensions_and_gradients():
    p = DoorPolicy(goal_frames=2, hidden=(32, 32))
    a = p(torch.zeros(4, 88), torch.ones(4, 120))
    assert a.shape == (4, 28) and torch.isfinite(a).all() and (a.abs() <= 1).all()
    a.sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.net.parameters())
    with pytest.raises(ValueError):
        DoorPolicy(goal_frames=0)
