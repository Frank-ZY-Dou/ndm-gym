"""Deployment rejects invalid evaluation inputs before simulator construction."""

import pytest

from ndm_gym.eval_door import main


@pytest.mark.parametrize('offsets', [[], [float('nan')], [float('inf')]])
def test_rejects_invalid_saved_offsets(monkeypatch, tmp_path, offsets):
    monkeypatch.setattr('ndm_gym.eval_door.torch.load', lambda *a, **k: {
        'protocol': {'offsets': offsets}})
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='Offsets'):
        main(['--checkpoint', 'trusted.pt', '--model', 'model.xml',
              '--reference', 'reference.npz', '--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize("method", ["supertrack", "supermimic"])
def test_deployment_recovers_saved_model_method_options(method):
    import torch
    from ndm_gym.eval_door import checkpoint_options
    from ndm_gym.train_door import DOOR_METHODS
    from test_door_algo import build

    base = build()
    options = {"upstream_policy_optimizer": True}
    if method == "supertrack":
        options.update(task_weights=True, correction_reg=.01, correction_imag_noise=.1)
    else:
        options.update(critic_epochs=2, critic_minibatches=2)
    shape = dict(hidden=(16,), wm_hidden=(16,), horizon=2)
    original = DOOR_METHODS[method](base.task, base.layout, base.reference_actions, **shape, **options)
    saved = original.state_dict()
    protocol = dict(method=method, **options)
    recovered = DOOR_METHODS[method](base.task, base.layout, base.reference_actions, **shape,
                                   **checkpoint_options(protocol, saved))
    recovered.load_state_dict(saved)
    obs = torch.zeros(2, base.task.obs_dim)
    goal = torch.zeros(2, base.task.goal_frame_dim)
    reference = base.reference_actions[:2]
    torch.testing.assert_close(original.pi(obs, goal, reference), recovered.pi(obs, goal, reference),
                               rtol=0, atol=0)
    assert recovered.upstream_policy_optimizer
    if method == "supertrack":
        assert recovered.task_weights and recovered.correction_reg == .01
        assert recovered.correction_imag_noise == .1


def test_deployment_historical_flags_default_off():
    from ndm_gym.eval_door import checkpoint_options

    assert checkpoint_options({"method": "residual"}, {}) == {}
    assert checkpoint_options({"method": "supertrack"}, {}) == dict(
        task_weights=False, upstream_policy_optimizer=False, correction_reg=0., correction_imag_noise=0.)


@pytest.mark.parametrize("changed", ["model", "reference", "missing_digest"])
def test_deployment_rejects_wrong_physical_identity_before_simulator(monkeypatch, tmp_path, changed):
    from ndm_gym.motion.door_identity import input_identity

    model, reference = tmp_path / "model.xml", tmp_path / "reference.npz"
    model.write_bytes(b"native model")
    reference.write_bytes(b"physical reference")
    protocol = dict(offsets=[0.], **input_identity(model, reference))
    if changed == "missing_digest":
        del protocol["model_sha256"]
    else:
        {"model": model, "reference": reference}[changed].write_bytes(b"different")
    monkeypatch.setattr('ndm_gym.eval_door.torch.load', lambda *a, **k: {'protocol': protocol})
    # These bytes cannot construct a simulator/reference. Identity must fail first.
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="identity mismatch|lacks model_sha256"):
        main(['--checkpoint', 'trusted.pt', '--model', str(model),
              '--reference', str(reference), '--output', str(output)])
    assert not output.exists()
