"""Transition alignment and a real optimizer/recovery path on a synthetic fixture."""

import copy

import numpy as np
import pytest
import torch

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.policies.adroit_door import DoorPolicy
from ndm_gym.policies.door_bootstrap import bootstrap_batch, bootstrap_update
from ndm_gym.tasks.adroit_door import DoorTask


def test_alignment_fit_and_optimizer_restore():
    torch.manual_seed(2905)
    q = np.zeros((5, 30))
    q[:, 28] = np.arange(5) * .1
    a = np.full((4, 28), .25)
    r = DoorReference(q, q * 2, a, {"kind": "adroit-door-physical-replay-v1",
                     "control_dt": .01, "joint_names": list(JOINT_NAMES)})
    t = DoorTask(r, np.tile([-1., 1.], (28, 1)))
    batch = bootstrap_batch(t, r)
    obs, goal, target = batch
    torch.testing.assert_close(obs[0, -28:], torch.zeros(28))
    torch.testing.assert_close(obs[1:, -28:], target[:-1])
    torch.testing.assert_close(goal[:, 28], torch.full((4,), .1))
    assert len(obs) == 4  # no terminal action target
    p = DoorPolicy(hidden=(16, 16))
    p.norm.update(torch.cat((obs, goal), -1))
    opt = torch.optim.Adam(p.parameters(), lr=.01)
    initial = (p(obs, goal) - target).square().mean().item()
    for _ in range(12):
        bootstrap_update(p, opt, batch)
    assert (p(obs, goal) - target).square().mean().item() < initial * .3
    clone = DoorPolicy(hidden=(16, 16))
    clone.load_state_dict(copy.deepcopy(p.state_dict()))
    opt2 = torch.optim.Adam(clone.parameters(), lr=.01)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    bootstrap_update(p, opt, batch)
    bootstrap_update(clone, opt2, batch)
    for x, y in zip(p.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    before = copy.deepcopy(p.state_dict())
    with pytest.raises(FloatingPointError):
        bootstrap_update(p, opt, (obs, goal, target * float("nan")))
    for name, x in p.state_dict().items():
        torch.testing.assert_close(x, before[name], rtol=0, atol=0)
