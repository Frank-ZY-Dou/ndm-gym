"""The control prior is explicit, zero-initialized, and differentiable."""

import pytest
import torch

from ndm_gym.policies.adroit_door import DoorResidualPolicy


def test_zero_residual_preserves_controls_and_learns_correction():
    policy = DoorResidualPolicy(hidden=(16,))
    obs, goal = torch.randn(3, 88), torch.randn(3, 60)
    prior = torch.linspace(-.8, .8, 84).reshape(3, 28)
    torch.testing.assert_close(policy(obs, goal, prior), prior, rtol=0, atol=0)
    opt = torch.optim.SGD(policy.parameters(), lr=.1)
    before = (policy(obs, goal, prior) - (prior + .02)).square().mean()
    before.backward()
    assert torch.isfinite(policy.net[-1].weight.grad).all()
    assert policy.net[-1].weight.grad.abs().sum() > 0
    opt.step()
    after = (policy(obs, goal, prior) - (prior + .02)).square().mean()
    assert after < before


def test_action_bounds_shape_and_configuration():
    p = DoorResidualPolicy(hidden=(16,), residual_scale=.5)
    with torch.no_grad():
        p.net[-1].bias.fill_(10)
    a = p(torch.zeros(2, 88), torch.zeros(2, 60), torch.full((2, 28), .9))
    torch.testing.assert_close(a, torch.ones_like(a))
    with pytest.raises(ValueError, match="shape"):
        p(torch.zeros(2, 88), torch.zeros(2, 60), torch.zeros(28))
    for scale in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match="residual_scale"):
            DoorResidualPolicy(residual_scale=scale)
