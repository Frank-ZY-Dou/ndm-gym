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


@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_independent_model_windows_and_policy_batches_with_critic_schedule(method):
    """Exercise distinct replay routes and count actual critic optimizer steps."""
    base = build()
    kwargs = dict(critic_epochs=5, critic_minibatches=4) if method is DoorSuperMimic else {}
    algo = method(base.task, base.layout, base.reference_actions,
                  hidden=(24, 12), wm_hidden=(32, 16), horizon=2, **kwargs)
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    normalize(algo, replay)
    requests = []
    original = replay.sample

    def sample(batch, window):
        requests.append((batch, window))
        return original(batch, window)

    replay.sample = sample
    before = {name: [p.detach().clone() for p in getattr(algo, name).parameters()]
              for name in ('wm', 'pi') + (('critic',) if method is DoorSuperMimic else ())}
    updates(algo, replay, 8, 2, 4, 1, wm_batch=6, wm_window=3)
    assert requests == [(6, 3)]*4 + [(8, 2)]
    expected = dict(wm=4, policy=1)
    if method is DoorSuperMimic:
        expected['critic'] = 20
    assert optimizer_counts(algo) == expected
    for name, old in before.items():
        parameters = list(getattr(algo, name).parameters())
        assert all(torch.isfinite(p).all() for p in parameters)
        assert any(not torch.equal(p, previous) for p, previous in zip(parameters, old))
    restored = method(base.task, base.layout, base.reference_actions,
                      hidden=(24, 12), wm_hidden=(32, 16), horizon=2, **kwargs)
    restored.load_state_dict(copy.deepcopy(algo.state_dict()))
    assert optimizer_counts(restored) == expected
    for name in before:
        for p, q in zip(getattr(algo, name).parameters(), getattr(restored, name).parameters()):
            assert torch.equal(p, q)


@pytest.mark.parametrize('method', ['supertrack', 'supermimic'])
@pytest.mark.parametrize('preset', [False, True])
def test_real_cli_valid_budget_reaches_asset_loading(monkeypatch, tmp_path, method, preset):
    """Exercise main validation, not only parser and internal optimizer helpers."""
    from ndm_gym import train_door_online as trainer
    from ndm_gym.door_configs import NAME

    class AssetBoundaryReached(Exception):
        pass

    def load(path):
        assert path == tmp_path / 'reference.npz'
        raise AssetBoundaryReached

    monkeypatch.setattr(trainer.DoorReference, 'load', load)
    argv = ['--method', method, '--model', str(tmp_path / 'model.xml'),
            '--reference', str(tmp_path / 'reference.npz'), '--bootstrap', str(tmp_path / 'bootstrap.npz'),
            '--output', str(tmp_path / 'output')]
    if preset:
        argv += ['--config', NAME]
    with pytest.raises(AssetBoundaryReached):
        trainer.main(argv)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('option,value', [
    ('--wm-updates', '-1'), ('--policy-updates', '-1'), ('--bootstrap-updates', '-1'),
    ('--wm-batch-size', '0'), ('--wm-batch-size', '3'), ('--wm-window', '0'),
    ('--batch-size', '3'), ('--learning-rate', 'nan'), ('--policy-lr', 'inf'),
    ('--collection-worlds', '0'), ('--correction-reg', 'nan'), ('--correction-imag-noise', '-1'),
])
def test_real_cli_rejects_invalid_budget_before_assets(monkeypatch, tmp_path, option, value):
    from ndm_gym import train_door_online as trainer

    def load(path):
        pytest.fail('Invalid budget reached asset loading')

    monkeypatch.setattr(trainer.DoorReference, 'load', load)
    with pytest.raises(ValueError, match='Invalid'):
        trainer.main(['--method', 'supertrack', '--model', 'unused', '--reference', 'unused',
                      '--bootstrap', 'unused', '--output', str(tmp_path / 'output'), option, value])
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('flags', [
    ['--task-weights'], ['--correction-reg', '.01'],
    ['--correction-imag-noise', '.2'], ['--critic-minibatches', '3'],
    ['--critic-minibatches', '0'], ['--fused-adam'],
])
def test_real_mimic_cli_rejects_unsupported_semantics_before_assets(monkeypatch, tmp_path, flags):
    from ndm_gym import train_door_online as trainer

    def load(path):
        pytest.fail('Unsupported configuration reached asset loading')

    monkeypatch.setattr(trainer.DoorReference, 'load', load)
    with pytest.raises(ValueError):
        trainer.main(['--method', 'supermimic', '--model', 'unused', '--reference', 'unused',
                      '--bootstrap', 'unused', '--output', str(tmp_path / 'output'), *flags])
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_bootstrap_checkpoint_recovers_first_stochastic_update(tmp_path, method):
    from ndm_gym.train_door_online import save_checkpoint

    base = build()

    def create():
        return method(base.task, base.layout, base.reference_actions,
                      hidden=(16,), wm_hidden=(16,), horizon=2)

    torch.manual_seed(3026)
    algo = create()
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    normalize(algo, replay)
    updates(algo, replay, 4, 2, 2, 0)
    torch.manual_seed(103026)
    save_checkpoint(algo, tmp_path, {}, 2, 1, 1, 0)
    assert not (tmp_path / 'checkpoint.tmp').exists()
    checkpoint = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert checkpoint['completed_rounds'] == 0
    assert checkpoint['optimizer_counts']['wm'] == 2
    assert checkpoint['optimizer_counts']['policy'] == 0
    updates(algo, replay, 4, 2, 1, 1)
    expected_rng = torch.get_rng_state().clone()
    recovered = create()
    assert restore(recovered, checkpoint, {}) == 0
    updates(recovered, replay, 4, 2, 1, 1)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert optimizer_counts(recovered) == optimizer_counts(algo)
    for name in ('wm', 'pi') + (('critic',) if method is DoorSuperMimic else ()):
        for p, q in zip(getattr(algo, name).parameters(), getattr(recovered, name).parameters()):
            assert torch.equal(p, q)


@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_round_timing_checkpoint_does_not_change_rng_or_updates(tmp_path, method):
    from ndm_gym.train_door_online import save_checkpoint

    base = build()
    algo = method(base.task, base.layout, base.reference_actions,
                  hidden=(16,), wm_hidden=(16,), horizon=2)
    replay = DoorReplay([(torch.randn(6, 30)*.01, torch.randn(6, 30)*.01,
                          torch.randn(5, 28)*.01)])
    updates(algo, replay, 4, 2, 2, 1)
    before = torch.get_rng_state().clone()
    metrics = [dict(completed_rounds=1, update_seconds=.25, optimizer_counts=optimizer_counts(algo))]
    save_checkpoint(algo, tmp_path, {}, 1, 1, 1, 1, round_metrics=metrics)
    assert torch.equal(before, torch.get_rng_state())
    checkpoint = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert checkpoint['round_metrics'] == metrics
    recovered = method(base.task, base.layout, base.reference_actions,
                       hidden=(16,), wm_hidden=(16,), horizon=2)
    assert restore(recovered, checkpoint, {}) == 1
    assert torch.equal(before, torch.get_rng_state())
    assert optimizer_counts(recovered) == metrics[0]['optimizer_counts']
