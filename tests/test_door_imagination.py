"""Verify policy credit assignment and transactional rejection with frozen dynamics."""

import copy
from types import SimpleNamespace

import pytest
import torch

from ndm_gym.policies.adroit_door import DoorResidualPolicy
from ndm_gym.policies.door_imagination import residual_update


class Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.))
        self.fixed = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.bad = False

    def step(self, q, v, ctrl, dt):
        assert dt == .02
        return q + ctrl * self.gain + (float('nan') if self.bad else 0), v


def fixture():
    torch.manual_seed(42)
    task = SimpleNamespace(
        frames=6, action_dim=28, dt=.02,
        obs=lambda s, a: torch.cat((s[0], s[1], a, s[0][..., :4]), -1),
        goal=lambda s, f: torch.cat((s[0], s[1], s[0][..., :4]), -1),
        setpoint=lambda a: 2 * a,
        reward=lambda s, f, a, p: -(s[0] - .1).square().mean(-1),
    )
    policy, wm = DoorResidualPolicy(hidden=(16,)), Dynamics()
    opt = torch.optim.Adam(policy.parameters(), lr=.01)
    return policy, opt, wm, task, (torch.zeros(2, 28), torch.zeros(2, 28)), torch.zeros(2, dtype=torch.long), torch.zeros(2, 28), torch.zeros(5, 28)


def test_credit_assignment_frozen_weights_and_restored_flags():
    p, opt, wm, *args = fixture()
    before = copy.deepcopy(wm.state_dict())
    loss0, grad = residual_update(p, opt, wm, *args, horizon=2)
    loss1, _ = residual_update(p, opt, wm, *args, horizon=2)
    assert grad > 0 and loss1 < loss0
    assert wm.gain.requires_grad and not wm.fixed.requires_grad
    assert all(x.grad is None for x in wm.parameters())
    for key, value in before.items():
        torch.testing.assert_close(wm.state_dict()[key], value, rtol=0, atol=0)


def test_nonfinite_and_out_of_clip_do_not_update():
    p, opt, wm, *args = fixture()
    residual_update(p, opt, wm, *args, horizon=2)
    before = copy.deepcopy(p.state_dict())
    steps = [v['step'].clone() for v in opt.state.values()]
    wm.bad = True
    with pytest.raises(FloatingPointError):
        residual_update(p, opt, wm, *args, horizon=2)
    assert wm.gain.requires_grad and not wm.fixed.requires_grad
    for key, value in before.items():
        torch.testing.assert_close(p.state_dict()[key], value, rtol=0, atol=0)
    assert all(torch.equal(v['step'], step) for v, step in zip(opt.state.values(), steps))
    with pytest.raises(ValueError, match='within the reference'):
        residual_update(p, opt, wm, *args, horizon=6)
