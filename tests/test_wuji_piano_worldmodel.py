"""The Wuji piano task's own world model (ndm_gym.worldmodels.piano) on the CPU, with a synthetic fingertip fit (no assets,
no GPU): its step's shapes, its first-order actuator step at initialization (both residual networks start at zero), key
depths from the contact constants, and the gradients that train it."""

import torch

from ndm_gym.worldmodels.piano import KEY_COUNT, NUM_ACTUATORS as A, PianoDynamics, TipGeometry


def model():
    g = torch.Generator().manual_seed(914)
    geometry = TipGeometry(torch.randn(2, 5, 15, 3, generator=g) * 0.01, torch.zeros(2, 3), torch.ones(6) * 0.1,
                           torch.randn(KEY_COUNT, 3, generator=g) * 0.03, torch.ones(KEY_COUNT, 3) * 0.01)  # fmt: skip
    return PianoDynamics(geometry)


def test_tip_geometry_shapes():
    geometry = model().geom
    tips = geometry(torch.rand(4, A) * 2 - 1)
    assert tips.shape == (4, 10, 3) and tips.isfinite().all()
    dx, dy, dz = geometry.distances(tips)
    assert dx.shape == dy.shape == dz.shape == (4, 10, KEY_COUNT) and (dx >= 0).all()


def test_step_at_initialization():
    torch.manual_seed(0)
    wm = model()
    state, action = torch.rand(6, 2 * A + KEY_COUNT), torch.rand(6, A) * 2 - 1
    out = wm(state, action)
    assert out.shape == state.shape and out.isfinite().all()
    q, v, last = state.split([A, A, KEY_COUNT], -1)
    vel = 0.5 * (action - q) + 0.12 * v  # the residual networks' last layers start at zero
    torch.testing.assert_close(out[:, A : 2 * A], vel)
    torch.testing.assert_close(out[:, :A], q + vel)
    keys = out[:, 2 * A :] - 0.09 * last  # 0.91 times a contact term in [0, 1]
    assert (keys >= -1e-6).all() and (keys <= 0.91 + 1e-6).all()


def test_gradients_reach_the_trained_parameters():
    torch.manual_seed(0)
    wm = model()
    state, action = torch.rand(6, 2 * A + KEY_COUNT), torch.rand(6, A) * 2 - 1
    wm(state, action).square().sum().backward()
    for net in (wm.actuator, wm.key_residual):
        assert net[-1].weight.grad.abs().sum() > 0 and net[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is not None for p in wm.parameters())
    assert not any(b.requires_grad for b in wm.geom.buffers())  # the fit is fixed: buffers, not parameters
