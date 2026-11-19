"""The piano benchmark's tuning options on its SuperTrack and SuperMimic adapters, on the CPU with a synthetic fingertip
geometry (no assets, no GPU): the defaults are the benchmark's configuration, and every option reaches the updates."""

import math
from pathlib import Path

import pytest
import torch

from ndm_gym.algos.hand_fast import update_functions
from ndm_gym.tasks.piano import piano_roll
from ndm_gym.worldmodels.piano import KEY_COUNT, TipGeometry


@pytest.fixture
def pu(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks" / "wuji_piano"))
    import piano_updates

    return piano_updates


def geometry():
    g = torch.Generator().manual_seed(914)
    return TipGeometry(torch.randn(2, 5, 15, 3, generator=g) * 0.01, torch.zeros(2, 3), torch.ones(6) * 0.1,
                       torch.randn(KEY_COUNT, 3, generator=g) * 0.03, torch.ones(KEY_COUNT, 3) * 0.01)  # fmt: skip


def train(pu, algo):
    """Three model and three policy updates on a synthetic replay: the losses, and whether every trained network changed."""
    algo.args.pi_batch = algo.args.wm_batch = 8
    buffer = pu.make_replay(16, 2, "cpu")
    g = torch.Generator().manual_seed(1)
    for frame in range(16):
        buffer.add(dict(s=torch.rand(2, pu.D, generator=g), s1=torch.rand(2, pu.D, generator=g), u=torch.rand(2, pu.A, generator=g) * 2 - 1,
                        a_prev=torch.rand(2, pu.A, generator=g) * 2 - 1, f=torch.full((2,), 22 + frame), done=torch.zeros(2, dtype=torch.bool)))  # fmt: skip
    nets = [algo.wm, algo.pi] + ([algo.critic] if isinstance(algo, pu.OdeSuperMimic) else [])
    before = [torch.cat([p.detach().flatten().clone() for p in n.parameters()]) for n in nets]
    fit, improve = update_functions(algo, buffer, algo.args)
    losses = [float(fit()) for _ in range(3)] + [float(improve()[0]) for _ in range(3)]
    changed = [not torch.equal(b, torch.cat([p.detach().flatten() for p in n.parameters()])) for b, n in zip(before, nets, strict=True)]
    return losses, changed


def test_defaults_are_the_benchmarks_configuration(pu):
    roll = torch.from_numpy(piano_roll())
    st, sm = pu.OdeSuperTrack(geometry(), roll), pu.OdeSuperMimic(geometry(), roll)
    assert st.horizon == sm.horizon == 3 and len(st.goal) == len(roll) + 11 + 3
    assert torch.equal(st.bal.w, torch.tensor([1.0, 0.85, 0.20]) / 3) and (st.imag_noise, st.reg, st.fail_w) == (0.0, 0.0, 0.0)
    assert (sm.args.gamma, sm.args.lam, sm.args.critic_epochs, sm.args.critic_minibatches, sm.bptt_decay) == (0.99, 0.95, 5, 4, 1.0)
    assert torch.equal(sm.weights, st.weights) and torch.equal(st.weights, torch.tensor([1.0, 0.85, 0.20]))


def test_options_reach_the_updates(pu):
    roll = torch.from_numpy(piano_roll())
    torch.manual_seed(0)
    st = pu.OdeSuperTrack(geometry(), roll, horizon=8, imag_noise=0.1, reg=0.01, reach_w=0.4)
    assert st.horizon == 8 and torch.allclose(st.bal.w, torch.tensor([1.0, 0.85, 0.40]) / 8) and (st.imag_noise, st.reg) == (0.1, 0.01)
    assert torch.equal(st.weights, torch.tensor([1.0, 0.85, 0.20]))  # the reach weight of SuperTrack's loss alone
    sm = pu.OdeSuperMimic(geometry(), roll, horizon=8, gamma=0.95, critic_epochs=10, critic_minibatches=4, bptt_decay=0.95)
    assert (sm.horizon, sm.args.gamma, sm.args.critic_epochs, sm.bptt_decay) == (8, 0.95, 10, 0.95)
    for algo in (st, sm):
        losses, changed = train(pu, algo)
        assert all(math.isfinite(x) for x in losses) and all(changed)


def test_gradient_decay_keeps_the_imagined_states(pu):
    """--bptt-decay scales the gradient through each imagined state; the states themselves are unchanged."""
    roll = torch.from_numpy(piano_roll())
    torch.manual_seed(0)
    full = pu.OdeSuperMimic(geometry(), roll)
    torch.manual_seed(0)
    decayed = pu.OdeSuperMimic(geometry(), roll, bptt_decay=0.5)
    state, prev, f = torch.rand(4, pu.D), torch.zeros(4, pu.A), torch.full((4,), 30)
    for algo in (full, decayed):
        algo.wm.actuator[-1].weight.data.fill_(0.01)  # a non-zero residual, so the step depends on the actuator network
    a, b = full.imag_step(state, None, f, prev), decayed.imag_step(state, None, f, prev)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[4], b[4], rtol=0, atol=0)
    a[4].sum().backward(), b[4].sum().backward()
    ga, gb = full.wm.actuator[-1].weight.grad, decayed.wm.actuator[-1].weight.grad
    torch.testing.assert_close(gb, 0.5 * ga)
