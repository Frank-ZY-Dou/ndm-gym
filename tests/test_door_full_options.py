"""train_door_full's opt-in tuning settings: absent, the original configuration; present, recorded and rebuilt."""

import numpy as np
import pytest

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.train_door_full import make_algorithm
from test_door_native_collector import fixture


def reference(frames=24):
    q = np.zeros((frames, 30))
    return DoorReference(q, q.copy(), np.zeros((frames - 1, 28)),
                         dict(kind='adroit-door-physical-replay-v1', joint_names=list(JOINT_NAMES),
                              control_dt=.01, frame_skip=5, scene_door_body_pos=[0, 0, 0]))


@pytest.mark.parametrize('method', ['supertrack', 'supermimic'])
def test_defaults_are_the_pull_requests_configuration(method):
    model, _ = fixture()
    algo, rate = make_algorithm(method, model, reference(), 'cpu')
    assert rate == (4e-4 if method == 'supertrack' else 2e-4)
    assert {g['lr'] for opt in (algo.opt_wm, algo.opt_pi) for g in opt.param_groups} == {rate}
    assert algo.pi.residual_scale == .1 and algo.horizon == 16 and algo.task.goal_frames == 5
    assert algo.upstream_policy_optimizer
    if method == 'supertrack':
        assert (algo.task_weights, algo.correction_reg, algo.correction_imag_noise) == (True, .01, .2)
    else:
        assert (algo.gamma, algo.lam, algo.critic_epochs, algo.critic_minibatches) == (.99, .95, 5, 4)


def test_settings_are_applied_and_checked_on_loading():
    model, _ = fixture()
    ref = reference()
    algo, rate = make_algorithm('supertrack', model, ref, 'cpu', dict(lr=2e-4, residual_scale=.2, correction_imag_noise=.4))
    assert rate == 2e-4 and algo.pi.residual_scale == .2 and algo.correction_imag_noise == .4
    state = algo.state_dict()
    same, _ = make_algorithm('supertrack', model, ref, 'cpu', dict(lr=2e-4, residual_scale=.2, correction_imag_noise=.4))
    same.load_state_dict(state)
    other, _ = make_algorithm('supertrack', model, ref, 'cpu')
    with pytest.raises(ValueError, match='correction'):
        other.load_state_dict(state)
    sm, _ = make_algorithm('supermimic', model, ref, 'cpu', dict(gamma=.995))
    assert sm.gamma == .995
    default_sm, _ = make_algorithm('supermimic', model, ref, 'cpu')
    with pytest.raises(ValueError, match='return contract'):
        default_sm.load_state_dict(sm.state_dict())


@pytest.mark.parametrize('method,options', [('supermimic', dict(correction_imag_noise=.4)), ('supertrack', dict(gamma=.995)),
                                            ('supertrack', dict(horizon=8))])
def test_other_methods_or_unknown_settings_are_rejected(method, options):
    model, _ = fixture()
    with pytest.raises(ValueError):
        make_algorithm(method, model, reference(), 'cpu', options)
