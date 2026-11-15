"""SuperTrack calibration, frozen-model gradients and recovery."""

import copy

import pytest
import torch

from ndm_gym.algos.door_supertrack import DoorSuperTrack
from ndm_gym.algos.door_supermimic import DoorSuperMimic
from test_door_algo import build


@pytest.mark.parametrize("method", [DoorSuperTrack, DoorSuperMimic])
@pytest.mark.parametrize("upstream", [False, True])
def test_model_through_time_and_recovery(method, upstream):
    base = build()
    algo = method(base.task, base.layout, base.reference_actions,
                          hidden=(16,), wm_hidden=(16,), horizon=2, upstream_policy_optimizer=upstream)
    state = (torch.ones(3, 30) * .01, torch.zeros(3, 30))
    frame, prev = torch.zeros(3, dtype=torch.long), torch.zeros(3, 28)
    # The upstream zero output head initially has no action derivative. Fit it
    # before testing policy gradients, as the actual training pipeline must.
    q = torch.randn(2, 8, 30) * .01
    v = torch.randn(2, 8, 30) * .01
    ctrl = torch.randn(1, 8, 28) * .1
    wm_before = [p.detach().clone() for p in algo.wm.parameters()]
    loss_wm = algo.wm_loss(q, v, ctrl)
    algo.opt_wm.zero_grad(set_to_none=True)
    loss_wm.backward()
    algo.opt_wm.step()
    assert torch.isfinite(loss_wm)
    assert any(not torch.equal(a, b) for a, b in zip(wm_before, algo.wm.parameters()))
    before = copy.deepcopy(algo.wm.state_dict())
    policy = [p.detach().clone() for p in algo.pi.parameters()]
    loss, grad = algo.policy_update(state, frame, prev)
    assert torch.isfinite(loss) and grad > 0
    if method is DoorSuperTrack:
        assert algo.bal.calibrated
    else:
        assert torch.isfinite(algo.last_value_loss)
        assert algo.opt_v.state
        assert not algo.bal.calibrated
    assert any(not torch.equal(a, b) for a, b in zip(policy, algo.pi.parameters()))
    for key, value in algo.wm.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    resumed = method(base.task, base.layout, base.reference_actions,
                             hidden=(16,), wm_hidden=(16,), horizon=2, upstream_policy_optimizer=upstream)
    resumed.load_state_dict(copy.deepcopy(algo.state_dict()))
    for a, b in zip(algo.policy_update(state, frame, prev), resumed.policy_update(state, frame, prev)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("lam,expected", [(0., [6., 2.]), (1., [2., 2.]), (.5, [4., 2.])])
def test_lambda_return_terminal_cut(lam, expected):
    from ndm_gym.algos.door_supermimic import lambda_returns

    rewards = torch.tensor([[1.], [2.]], requires_grad=True)
    continuation = torch.tensor([[1.], [0.]])
    values = torch.tensor([[99.], [10.], [1000.]], requires_grad=True)
    target = lambda_returns(rewards, continuation, values, .5, lam)
    torch.testing.assert_close(target[:, 0], torch.tensor(expected))
    assert not target.requires_grad
    # Interior window must bootstrap from its final value.
    continuing = lambda_returns(rewards, torch.ones_like(continuation), values, .5, 1.)
    torch.testing.assert_close(continuing[:, 0], torch.tensor([252., 502.]))


def test_mimic_critic_step_has_no_policy_gradient():
    base = build()
    algo = DoorSuperMimic(base.task, base.layout, base.reference_actions,
                         hidden=(16,), wm_hidden=(16,), horizon=2)
    policy_at_value_step = []

    def before_value_step(*args, **kwargs):
        policy_at_value_step.extend(p.grad.detach().clone() if p.grad is not None else None
                                    for p in algo.pi.parameters())

    hook = algo.opt_pi.register_step_post_hook(before_value_step)
    algo.policy_update((torch.ones(3, 30)*.01, torch.zeros(3, 30)),
                       torch.zeros(3, dtype=torch.long), torch.zeros(3, 28))
    hook.remove()
    # The critic's backward must leave the policy-step gradients untouched.
    for old, parameter in zip(policy_at_value_step, algo.pi.parameters()):
        if old is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(old, parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("method,kind", [(DoorSuperTrack, "accel"), (DoorSuperMimic, "increment")])
def test_squared_window_contract(method, kind):
    base = build()
    algo = method(base.task, base.layout, base.reference_actions,
                  hidden=(16,), wm_hidden=(16,), horizon=2)
    assert algo.wm.kind == kind
    q, v = torch.randn(3, 2, 30), torch.randn(3, 2, 30)
    ctrl = torch.randn(2, 2, 28)
    valid = torch.tensor([[True, False], [False, True]])
    # Independently: only the first transition of the first window is valid.
    q1, v1 = algo.wm.step(q[0], v[0], ctrl[0])
    expected = algo.wm.error(q1, v1, q[1], v[1]).square().mean(-1)[0] / 2
    torch.testing.assert_close(algo.wm_loss(q, v, ctrl, valid), expected)
    checkpoint = copy.deepcopy(algo.state_dict())
    checkpoint.pop("model_contract")
    before = copy.deepcopy(algo.state_dict())
    with pytest.raises(ValueError, match="model contract mismatch"):
        algo.load_state_dict(checkpoint)
    for key, value in algo.wm.state_dict().items():
        torch.testing.assert_close(value, before["wm"][key], rtol=0, atol=0)


def test_mimic_critic_schedule_counts_and_recovery():
    base = build()
    kwargs = dict(hidden=(16,), wm_hidden=(16,), horizon=2,
                  critic_epochs=5, critic_minibatches=4)
    algo = DoorSuperMimic(base.task, base.layout, base.reference_actions, **kwargs)
    state = (torch.ones(4, 30)*.01, torch.zeros(4, 30))
    frame, prev = torch.zeros(4, dtype=torch.long), torch.zeros(4, 28)
    algo.policy_update(state, frame, prev)
    assert {int(s['step']) for s in algo.opt_v.state.values()} == {20}
    saved, rng = copy.deepcopy(algo.state_dict()), torch.get_rng_state()
    resumed = DoorSuperMimic(base.task, base.layout, base.reference_actions, **kwargs)
    resumed.load_state_dict(saved)
    torch.set_rng_state(rng)
    expected = algo.policy_update(state, frame, prev)
    torch.set_rng_state(rng)
    actual = resumed.policy_update(state, frame, prev)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(algo.critic.parameters(), resumed.critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    wrong = DoorSuperMimic(base.task, base.layout, base.reference_actions,
                          hidden=(16,), wm_hidden=(16,), horizon=2)
    before = copy.deepcopy(wrong.wm.state_dict())
    with pytest.raises(ValueError, match='schedule mismatch'):
        wrong.load_state_dict(saved)
    for key, value in wrong.wm.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("method", [DoorSuperTrack, DoorSuperMimic])
def test_squared_multistep_velocity_and_gradient_contract(method):
    base = build()
    algo = method(base.task, base.layout, base.reference_actions,
                  hidden=(16,), wm_hidden=(16,), horizon=2)
    torch.manual_seed(2966)
    q, v = torch.randn(4, 2, 30) * .01, torch.randn(4, 2, 30) * .03
    ctrl = torch.randn(3, 2, 28) * .1
    actual = algo.wm_loss(q, v, ctrl)
    # Native Door has only scalar joints: independent tangent-space expression.
    scale = torch.maximum(algo.wm.err_norm.std(), algo.wm.err_floor)
    qp, vp = q[0], v[0]
    expected = q.new_zeros(())
    for t in range(3):
        qp, vp = algo.wm.step(qp, vp, ctrl[t])
        error = torch.cat((qp - q[t+1], vp - v[t+1]), -1) / scale
        expected = expected + error.square().mean(-1).mean()
    torch.testing.assert_close(actual, expected)
    params = tuple(algo.wm.parameters())
    actual_grad = torch.autograd.grad(actual, params)
    expected_grad = torch.autograd.grad(expected, params)
    assert any(g.abs().sum() > 0 for g in actual_grad)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
    # Velocity targets participate in the loss, even with identical qpos.
    changed = v.clone()
    changed[1:] += .2
    assert not torch.isclose(actual, algo.wm_loss(q, changed, ctrl))


@pytest.mark.parametrize("method", [DoorSuperTrack, DoorSuperMimic])
def test_upstream_optimizer_recovery_contract(method):
    base = build()
    def make(enabled=False):
        return method(base.task, base.layout, base.reference_actions,
                      hidden=(16,), wm_hidden=(16,), horizon=2,
                      upstream_policy_optimizer=enabled)
    legacy = make()
    assert legacy.opt_pi.param_groups[0]["betas"] == (.9, .999)
    historical = copy.deepcopy(legacy.state_dict())
    historical.pop("upstream_policy_optimizer")
    make().load_state_dict(historical)
    aligned = make(True)
    assert aligned.opt_pi.param_groups[0]["betas"] == (.9, .95)
    saved = copy.deepcopy(aligned.state_dict())
    make(True).load_state_dict(saved)
    before = copy.deepcopy(legacy.pi.state_dict())
    with pytest.raises(ValueError, match="optimizer configuration"):
        legacy.load_state_dict(saved)
    for key, value in legacy.pi.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="optimizer configuration"):
        aligned.load_state_dict(historical)


def test_correction_regularizer_excludes_reference_and_has_gradient():
    base = build()
    kwargs = dict(hidden=(16,), wm_hidden=(16,), horizon=2)
    plain = DoorSuperTrack(base.task, base.layout, base.reference_actions, **kwargs)
    regularized = DoorSuperTrack(base.task, base.layout, base.reference_actions,
                                correction_reg=.01, **kwargs)
    with torch.no_grad():
        plain.pi.net[-1].bias.fill_(.2)
    checkpoint = copy.deepcopy(plain.state_dict())
    checkpoint['correction_reg'] = .01
    regularized.load_state_dict(checkpoint)
    # Disable stepping to compare the objective and gradients at the same policy.
    plain.opt_pi.step = lambda: None
    regularized.opt_pi.step = lambda: None
    state = (torch.ones(3, 30)*.01, torch.zeros(3, 30))
    frame, previous = torch.zeros(3, dtype=torch.long), torch.zeros(3, 28)
    loss0, _ = plain.policy_update(state, frame, previous)
    loss1, _ = regularized.policy_update(state, frame, previous)
    c = torch.tensor(.2).tanh()
    torch.testing.assert_close(loss1-loss0, .01*(c.square()+c), atol=1e-6, rtol=1e-4)
    assert regularized.pi.net[-1].bias.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='correction configuration'):
        plain.load_state_dict(regularized.state_dict())


def test_imagination_noise_units_and_clean_previous_action(monkeypatch):
    base = build()
    algo = DoorSuperTrack(base.task, base.layout, base.reference_actions,
                         hidden=(16,), wm_hidden=(16,), horizon=2,
                         correction_imag_noise=.2)
    controls, previous = [], []
    original_step, original_obs = algo.wm.step, algo.task.obs
    def step(q, v, ctrl, dt):
        controls.append(ctrl.detach().clone())
        return original_step(q, v, ctrl, dt)
    def obs(state, action):
        previous.append(action.detach().clone())
        return original_obs(state, action)
    monkeypatch.setattr(algo.wm, 'step', step)
    monkeypatch.setattr(algo.task, 'obs', obs)
    monkeypatch.setattr(torch, 'randn_like', lambda x: torch.ones_like(x))
    algo.policy_update((torch.ones(3, 30)*.01, torch.zeros(3, 30)),
                       torch.zeros(3, dtype=torch.long), torch.zeros(3, 28))
    for t, ctrl in enumerate(controls):
        clean = algo.reference_actions[t].expand(3, -1)
        expected = algo.setpoint((clean + algo.pi.residual_scale*.2).clamp(-1, 1))
        torch.testing.assert_close(ctrl, expected)
    torch.testing.assert_close(previous[1], algo.reference_actions[0].expand(3, -1))


@pytest.mark.parametrize('value', [-1., float('inf'), float('nan')])
def test_invalid_correction_options(value):
    base = build()
    with pytest.raises(ValueError, match='finite and nonnegative'):
        DoorSuperTrack(base.task, base.layout, base.reference_actions, correction_reg=value)
    with pytest.raises(ValueError, match='SuperTrack-only'):
        DoorSuperMimic(base.task, base.layout, base.reference_actions, correction_imag_noise=.1)


def test_task_weights_follow_native_reward_widths_and_recover():
    base = build()
    kwargs = dict(hidden=(16,), wm_hidden=(16,), horizon=2)
    algo = DoorSuperTrack(base.task, base.layout, base.reference_actions, task_weights=True, **kwargs)
    c = base.task.cfg
    widths = torch.tensor([c.translation_std, c.hand_std, c.door_std, c.latch_std])
    torch.testing.assert_close(algo.bal.w, 1 / (2 * widths))
    torch.testing.assert_close(algo.bal.scale, widths)
    assert algo.bal.calibrated
    resumed = DoorSuperTrack(base.task, base.layout, base.reference_actions, task_weights=True, **kwargs)
    resumed.load_state_dict(copy.deepcopy(algo.state_dict()))
    torch.testing.assert_close(resumed.bal.w, algo.bal.w, rtol=0, atol=0)
    legacy = DoorSuperTrack(base.task, base.layout, base.reference_actions, **kwargs)
    with pytest.raises(ValueError, match="weight configuration"):
        legacy.load_state_dict(algo.state_dict())
    old = copy.deepcopy(legacy.state_dict())
    old.pop("task_weights")
    legacy.load_state_dict(old)
    with pytest.raises(ValueError, match="SuperTrack-only"):
        DoorSuperMimic(base.task, base.layout, base.reference_actions, task_weights=True, **kwargs)
    with pytest.raises(ValueError, match="boolean"):
        DoorSuperTrack(base.task, base.layout, base.reference_actions, task_weights=1, **kwargs)
