"""Quaternion geometry and episode semantics without a simulator."""
import math

import pytest
import torch

from ndm_gym.tasks.reorientation import GoalTracker, ReorientationCfg, orientation_error, sample_goals


def test_known_angles_and_signs():
    q = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0], [2**-.5, 0, 0, 2**-.5]])
    goal = torch.tensor([[1., 0, 0, 0]]).expand_as(q)
    expected = torch.tensor([0., math.pi, math.pi / 2])
    torch.testing.assert_close(orientation_error(q, goal), expected)
    torch.testing.assert_close(orientation_error(-q, goal), expected)


def test_gradient_at_match_is_finite():
    q = torch.tensor([[1., 0, 0, 0]], requires_grad=True)
    orientation_error(q, q.detach()).sum().backward()
    assert torch.isfinite(q.grad).all()


def test_hold_drop_timeout_and_partial_reset():
    tracker = GoalTracker(3, ReorientationCfg(hold_steps=2, episode_steps=3))
    tracker.step(torch.tensor([0., 1., 1.]), torch.zeros(3))
    out = tracker.step(torch.tensor([0., 0., 1.]), torch.tensor([0., 1., 0.]))
    assert out['success'].tolist() == [True, False, False]
    assert out['dropped'].tolist() == [False, True, False]
    tracker.reset([0, 1])
    out = tracker.step(torch.ones(3), torch.zeros(3))
    assert out['truncated'].tolist() == [False, False, True]
    assert not out['terminated'].any()


def test_interrupted_hold():
    tracker = GoalTracker(1, ReorientationCfg(hold_steps=2))
    for e in (0., 1., 0.):
        assert not tracker.step(torch.tensor([e]), torch.zeros(1))['success'].item()
    assert tracker.step(torch.zeros(1), torch.zeros(1))['success'].item()


def test_goals_reproducible_and_isotropic():
    a = sample_goals(10000, generator=torch.Generator().manual_seed(7))
    b = sample_goals(10000, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(10000))
    assert (a.square().mean(0) - .25).abs().max() < .015


@pytest.mark.parametrize('kw', [dict(tolerance=0), dict(hold_steps=0), dict(drop_distance=-1), dict(episode_steps=2, hold_steps=3)])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        ReorientationCfg(**kw)


def test_reference_simulator():
    from pathlib import Path
    from ndm_gym.sim.rigid.reorientation import ReorientationSim

    model_dir = Path(__file__).resolve().parents[1] / 'data/menagerie/wonik_allegro'
    if not model_dir.exists():
        pytest.skip('run bash tools/download_allegro.sh for external Allegro assets')
    sim = ReorientationSim(model_dir, seed=4)
    a = sim.reset()
    b, r, info = sim.step(torch.zeros(16))
    assert a.shape == b.shape == (1, 45)
    assert torch.isfinite(b).all() and torch.isfinite(r).all()
    assert sim.tracker.steps.item() == 1
    with pytest.raises(ValueError):
        sim.step(torch.full((16,), float('nan')))
    assert sim.tracker.steps.item() == 1
    assert not info['truncated'].item()


def test_drop_overrides_success_and_nonfinite_is_terminal():
    tracker = GoalTracker(3, ReorientationCfg(hold_steps=1))
    out = tracker.step(torch.tensor([0., float('nan'), 0.]), torch.tensor([1., 0., float('inf')]))
    assert out['dropped'].all() and out['terminated'].all()
    assert not out['success'].any() and not out['truncated'].any()


def test_reward_orders_orientation_and_penalizes_actions():
    from ndm_gym.tasks.reorientation import reward

    q = torch.tensor([[1., 0, 0, 0]])
    opposite = torch.tensor([[0., 1, 0, 0]])
    cfg = ReorientationCfg()
    zero = torch.zeros(1, 16)
    assert reward(q, q, zero, cfg) > reward(q, opposite, zero, cfg)
    torch.testing.assert_close(reward(q, q, zero, cfg) - reward(q, q, torch.ones_like(zero), cfg), torch.tensor([.001]))
