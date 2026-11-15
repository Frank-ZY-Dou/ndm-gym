"""Round recovery must preserve subsequent stochastic optimizer updates."""

import copy

import pytest
import torch

from ndm_gym.algos.door_supermimic import DoorSuperMimic
from ndm_gym.algos.door_supertrack import DoorSuperTrack
from ndm_gym.motion.door_replay import DoorReplay
from ndm_gym.train_door_online import configure_fused_adam, normalize, optimizer_counts, restore, updates
from test_door_algo import build


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device required'))])
@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_round_recovery_preserves_updates_and_rejects_wrong_inputs(method, device, fused):
    if fused and device == 'cpu':
        pytest.skip("Fused CUDA path")
    base = build()
    for name in ('q', 'v', 'ctrl_mid', 'ctrl_half'):
        setattr(base.task, name, getattr(base.task, name).to(device))

    def create():
        agent = method(base.task, base.layout, base.reference_actions,
                       hidden=(16,), wm_hidden=(16,), horizon=2, device=device)
        configure_fused_adam(agent, fused)
        return agent

    torch.manual_seed(2944)
    algo = create()
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    updates(algo, replay, 4, 2, 2, 2)
    protocol = {'reference_sha256': 'original', 'method': method.__name__}
    checkpoint = copy.deepcopy(dict(algo=algo.state_dict(), optimizer_counts=optimizer_counts(algo),
                                    rng=torch.get_rng_state(),
                                    protocol=protocol, completed_rounds=1,
                                    cuda_rng=(torch.cuda.get_rng_state(device)
                                              if device.startswith('cuda') else None)))
    updates(algo, replay, 4, 2, 2, 2)
    expected = copy.deepcopy(algo.state_dict())
    resumed = create()
    with pytest.raises(ValueError, match='identity mismatch'):
        restore(resumed, checkpoint, {**protocol, 'reference_sha256': 'different'})
    assert restore(resumed, checkpoint, {**protocol, 'upstream_policy_optimizer': False}) == 1
    updates(resumed, replay, 4, 2, 2, 2)

    def equal(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        elif isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b

    equal(expected, resumed.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_canonical_normalizers_match_without_changing_learned_weights(method):
    base = build()
    torch.manual_seed(2947)
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    algos = []
    for device in ('cpu', 'cuda:0'):
        task = copy.deepcopy(base.task)
        for name in ('q', 'v', 'ctrl_mid', 'ctrl_half'):
            setattr(task, name, getattr(task, name).to(device))
        algo = method(task, base.layout, base.reference_actions,
                      hidden=(16,), wm_hidden=(16,), horizon=2, device=device)
        weights = [p.detach().clone() for p in algo.wm.parameters()]
        normalize(algo, replay)
        assert all(torch.equal(p, old) for p, old in zip(algo.wm.parameters(), weights))
        assert not algo.opt_wm.state
        algos.append(algo)
    for name in ('in_norm', 'out_norm', 'err_norm'):
        left, right = [getattr(a.wm, name).state_dict() for a in algos]
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key].cpu(), right[key].cpu())


def test_fused_adam_rejects_cpu_without_changing_optimizers():
    algo = build()
    before = copy.deepcopy(algo.opt_pi.param_groups)
    with pytest.raises(ValueError, match='CUDA'):
        configure_fused_adam(algo, True)
    assert algo.opt_pi.param_groups[0]['fused'] == before[0]['fused']


@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_reported_budget_is_verified_against_optimizer_state(method):
    from ndm_gym.train_door_online import verify_budget
    base = build()
    kwargs = dict(critic_epochs=2, critic_minibatches=2) if method is DoorSuperMimic else {}
    algo = method(base.task, base.layout, base.reference_actions,
                  hidden=(16,), wm_hidden=(16,), horizon=2, **kwargs)
    expected = dict(wm=0, policy=0)
    if method is DoorSuperMimic:
        expected['critic'] = 0
    assert optimizer_counts(algo) == expected
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    updates(algo, replay, 4, 2, 3, 2)
    counts = verify_budget(algo, 1, 1, 1, 2)
    assert counts['wm'] == 3 and counts['policy'] == 2
    if method is DoorSuperMimic:
        assert counts['critic'] == 8
    with pytest.raises(ValueError, match='budget mismatch'):
        verify_budget(algo, 1, 1, 1, 3)
    # A single missing update must not be hidden by taking max(step).
    parameter = next(iter(algo.opt_pi.state))
    algo.opt_pi.state[parameter]['step'] -= 1
    with pytest.raises(ValueError, match='Inconsistent policy'):
        optimizer_counts(algo)


def test_wm_compile_is_opt_in_and_wraps_original_loss(monkeypatch):
    from ndm_gym.train_door_online import configure_wm_compile
    algo = build()
    called = []
    def compile_loss(fn, **kwargs):
        called.append((fn, kwargs))
        return fn
    monkeypatch.setattr(torch, 'compile', compile_loss)
    configure_wm_compile(algo)
    assert not called and not hasattr(algo, 'training_wm_loss')
    configure_wm_compile(algo, True)
    assert called == [(algo.wm_loss, {'mode': 'default'})]
    assert algo.training_wm_loss == algo.wm_loss


def test_upstream_optimizer_cli_is_opt_in():
    from ndm_gym.train_door_online import parser
    required = ['--model', 'm.xml', '--reference', 'r.npz', '--bootstrap', 'b.pt',
                '--output', 'run', '--method', 'supertrack']
    assert not parser().parse_args(required).upstream_policy_optimizer
    assert parser().parse_args(required + ['--upstream-policy-optimizer']).upstream_policy_optimizer
