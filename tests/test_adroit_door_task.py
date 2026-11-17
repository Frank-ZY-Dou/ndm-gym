"""Native coordinate units, time alignment and differentiable Door heads."""

import numpy as np
import pytest
import torch

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.tasks.adroit_door import DoorRewardCfg, DoorTask


def task():
    q = np.zeros((3, 30))
    q[1, 28], q[2, 28] = 0.5, 1.5
    ref = DoorReference(q, np.zeros_like(q), np.zeros((2, 28)),
                        {"kind": "adroit-door-physical-replay-v1", "control_dt": .01,
                         "joint_names": list(JOINT_NAMES)})
    return DoorTask(ref, np.tile([-2., 4.], (28, 1)))


def test_alignment_and_independent_opening():
    t = task()
    f = torch.tensor([0, 1, 2])
    a = torch.zeros(3, 28)
    s = (t.q.clone(), t.v.clone())
    assert t.obs(s, a).shape == (3, t.obs_dim)
    g = t.goal(s, f)
    torch.testing.assert_close(g[:, 28], torch.tensor([.5, 1., 0.]))
    torch.testing.assert_close(t.reward(s, f, a, a), torch.full((3,), .04))
    assert t.opened(s).tolist() == [False, False, True]
    assert t.termination(s, f).tolist() == [False, False, True]
    assert t.next_frame(f).tolist() == [1, 2, 2]
    torch.testing.assert_close(t.setpoint(torch.full((28,), -2.)), torch.full((28,), -2.))
    torch.testing.assert_close(t.setpoint(torch.full((28,), 2.)), torch.full((28,), 4.))
    torch.testing.assert_close(t.setpoint(torch.zeros(28)), torch.ones(28))


def test_units_and_finite_gradient_at_reference():
    t = task()
    q = t.q.clone().requires_grad_()
    a = torch.zeros(3, 28, requires_grad=True)
    t.reward((q, t.v), torch.arange(3), a, a.detach()).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(a.grad).all()
    q = t.q.clone()
    q[:, 0] += .02
    e = t.errors((q, t.v), torch.arange(3))
    torch.testing.assert_close(e["translation_m"], torch.full((3,), .02))
    assert not e["hand_rad"].any()
    q[0, 0] = float("nan")
    assert t.termination((q, t.v), torch.zeros(3, dtype=torch.long))[0]
    assert not t.opened((q, t.v))[0]


@pytest.mark.parametrize("kwargs", [{"hand_std": 0}, {"door_std": float("nan")}, {"action_rate_weight": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DoorRewardCfg(**kwargs)


@pytest.mark.parametrize("ranges", [np.zeros((28, 2)), np.zeros((27, 2)), np.full((28, 2), np.nan)])
def test_invalid_actuator_ranges(ranges):
    t = task()
    ref = DoorReference(t.q.numpy(), t.v.numpy(), np.zeros((2, 28)),
                        {"kind": "adroit-door-physical-replay-v1", "control_dt": .01,
                         "joint_names": list(JOINT_NAMES)})
    with pytest.raises(ValueError, match="actuator ranges"):
        DoorTask(ref, ranges)
