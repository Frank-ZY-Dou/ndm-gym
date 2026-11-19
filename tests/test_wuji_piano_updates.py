"""Exercise the piano adapters through native replay, model fitting and policy updates."""

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from ndm_gym.algos.hand_fast import update_functions
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.worldmodels.piano import NUM_ACTUATORS as A, KEY_COUNT, TipGeometry


@pytest.mark.parametrize(
    ("method", "device"),
    [
        ("scale", "cpu"),
        ("supertrack", "cpu"),
        ("supermimic", "cpu"),
        *[
            pytest.param(algorithm, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay"))
            for algorithm in ("supertrack", "supermimic")
        ],
    ],
)
def test_piano_replay_update(method, device, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks" / "wuji_piano"))
    from ndm_gym.tasks.piano import piano_roll
    from piano_updates import D, OdeSuperMimic, OdeSuperTrack, make_replay

    # Synthetic actuator and key states exercise both wrist kinematics and key-travel gradients without physics or assets.
    rng = torch.Generator().manual_seed(914)
    geometry = TipGeometry(
        torch.randn(2, 5, 15, 3, generator=rng) * 0.01,
        torch.zeros(2, 3),
        torch.ones(6) * 0.1,
        torch.randn(KEY_COUNT, 3, generator=rng) * 0.03,
        torch.ones(KEY_COUNT, 3) * 0.01,
    )
    trainer = OdeSuperTrack if method == "supertrack" else OdeSuperMimic
    graphs = device == "cuda"
    monkeypatch.setattr(RunningNorm, "no_sync", graphs)
    stream = torch.cuda.stream(torch.cuda.Stream()) if graphs else nullcontext()
    with stream, torch.random.fork_rng(devices=[0] if graphs else []):
        torch.manual_seed(613)
        if method == "scale":
            pytest.importorskip("scipy")
            pytest.importorskip("mujoco_warp")
            import train_five_finger_scale as scale

            algo = scale.PianoSuperTrack(torch.from_numpy(scale.piano_roll()))
            state_dim, num_actions = scale.STATE_DIM, scale.N
        else:
            algo = trainer(geometry, torch.from_numpy(piano_roll()).to(device), update_graphs=graphs)
            state_dim, num_actions = D, A
        algo.args.pi_batch = algo.args.wm_batch = 8
        buffer = make_replay(8, 2, device, state_dim=state_dim, num_actions=num_actions, update_graphs=graphs)
        state = torch.rand(2, state_dim, device=device)
        prev = torch.zeros(2, num_actions, device=device)
        for frame in range(8):
            action = torch.rand(2, num_actions, device=device) - 0.5
            next_state = state * 0.7
            buffer.add(
                dict(
                    s=state,
                    s1=next_state,
                    u=action,
                    a_prev=prev,
                    f=torch.full((2,), 22 + frame, device=device),
                    done=torch.zeros(2, dtype=torch.bool, device=device),
                )
            )
            state, prev = next_state, action

        def weights(module):
            return torch.cat([p.detach().flatten() for p in module.parameters()])

        fit, improve = update_functions(algo, buffer, algo.args)
        if graphs:
            # Three warmups and one capture. The assertions below exercise actual graph replay.
            for _ in range(4):
                fit()
                improve()
        initial_model = weights(algo.wm)
        assert fit().isfinite()
        fitted = weights(algo.wm)
        assert not torch.equal(fitted, initial_model)
        initial_policy = weights(algo.pi)
        if method == "supermimic":
            initial_critic = weights(algo.critic)
        loss, value, reward, grad_norm, survival = improve()
        assert torch.stack((loss, reward, grad_norm, survival)).isfinite().all()
        assert grad_norm > 0
        assert not torch.equal(weights(algo.pi), initial_policy)
        assert torch.equal(weights(algo.wm), fitted)
        assert all(p.requires_grad for p in algo.wm.parameters())
        if method == "supermimic":
            assert value.isfinite()
            assert not torch.equal(weights(algo.critic), initial_critic)
