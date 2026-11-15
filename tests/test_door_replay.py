"""Reset boundaries and episode isolation are essential to model training."""

import pytest
import torch

from ndm_gym.motion.door_replay import DoorReplay


def episode(value):
    q = torch.arange(6.)[:, None].expand(6, 30) + value
    return q, q + 10, torch.full((5, 28), value / 100)


def test_windows_preserve_resets_frames_and_previous_control():
    replay = DoorReplay([episode(0), episode(50)])
    q, v, a, frame, previous = replay.sample(200, 4, torch.Generator().manual_seed(9))
    assert q.shape == (5, 200, 30)
    assert torch.equal(q[1:] - q[:-1], torch.ones_like(q[1:]))
    assert torch.equal(v, q + 10)
    assert torch.equal(q[0, :, 0] % 50, frame.float())
    assert torch.equal(previous[frame == 0], torch.zeros_like(previous[frame == 0]))
    assert torch.equal(previous[frame > 0], a[0, frame > 0])
    with pytest.raises(ValueError):
        replay.sample(1, 6)


def test_split_is_episode_disjoint_and_copies_inputs():
    source = episode(0)
    replay = DoorReplay([source, episode(50), episode(80)])
    train, held = replay.split([1])
    source[0].zero_()
    assert train.episodes[0][0][-1, 0] == 5
    assert {float(x[0][0, 0]) for x in train.episodes} == {0, 80}
    assert held.episodes[0][0][0, 0] == 50
    for invalid in ([], [0, 1, 2], [-1]):
        with pytest.raises(ValueError):
            replay.split(invalid)


def test_collection_executes_controls_and_replay_drives_multistep_model_update():
    import numpy as np
    from ndm_gym.motion.door_replay import collect_physical_replay
    from test_door_algo import build

    algo = build()

    class Physics:
        def reset(self, qpos_offset):
            self.q = qpos_offset.copy()
            return self.q.copy(), np.zeros(30)

        def step(self, action):
            self.q[:28] += action * .001
            return (self.q.copy(), np.zeros(30)), False

    collected = collect_physical_replay(algo, Physics(), [0., .01, -.01], .02, 13)
    repeat = collect_physical_replay(algo, Physics(), [0., .01, -.01], .02, 13)
    assert all(torch.equal(x, y) for e, r in zip(collected.episodes, repeat.episodes)
               for x, y in zip(e, r))
    train, _ = collected.split([2])
    q, v, a, _, _ = train.sample(8, 3)
    before = [p.detach().clone() for p in algo.wm.parameters()]
    loss = algo.wm_loss(q, v, algo.task.setpoint(a))
    assert torch.isfinite(loss)
    algo.opt_wm.zero_grad()
    loss.backward()
    algo.opt_wm.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, algo.wm.parameters()))


def test_load_physical_checkpoint_episodes(tmp_path):
    import numpy as np

    path = tmp_path / "physical.npz"
    saved = {f"episode{i}_{name}": value.numpy()
             for i, e in enumerate((episode(0), episode(50)))
             for name, value in zip(("qpos", "qvel", "actions"), e)}
    np.savez(path, **saved)
    replay = DoorReplay.load(path)
    train, heldout = replay.split([1])
    assert torch.equal(train.episodes[0][0], episode(0)[0])
    assert torch.equal(heldout.episodes[0][2], episode(50)[2])
    np.savez(path, episode1_qpos=episode(0)[0].numpy())
    with pytest.raises(ValueError, match="contiguous"):
        DoorReplay.load(path)


def test_balanced_sources_preserve_windows_and_holdouts():
    from ndm_gym.motion.door_replay import BalancedDoorReplay

    success, success_held = DoorReplay([episode(0), episode(10)]).split([1])
    failure, failure_held = DoorReplay([episode(50), episode(80)]).split([1])
    replay = BalancedDoorReplay([success, failure])
    q, v, a, frames, previous = replay.sample(20, 4, torch.Generator().manual_seed(7))
    assert (q[0, :, 0] < 50).sum() == 10
    assert (q[0, :, 0] >= 50).sum() == 10
    assert torch.equal(q[1:] - q[:-1], torch.ones_like(q[1:]))
    assert torch.equal(q[0, :, 0] % 50, frames.float())
    assert len(success_held.episodes) == len(failure_held.episodes) == 1
    assert torch.equal(previous[frames == 0], torch.zeros_like(previous[frames == 0]))
    with pytest.raises(ValueError, match="even"):
        replay.sample(3, 4)


def test_policy_collection_matches_deployment_and_resets_previous_action():
    import numpy as np
    from ndm_gym.motion.door_replay import collect_policy_replay
    from ndm_gym.train_door import evaluate
    from test_door_algo import build

    algo = build()
    # Nonzero feedback makes this distinguish policy collection from replay.
    with torch.no_grad():
        list(algo.pi.net.parameters())[-1].fill_(.2)

    class Physics:
        def reset(self, qpos_offset):
            self.q = algo.task.q[0].numpy().copy() + qpos_offset
            return self.q.copy(), np.zeros(30)

        def step(self, action):
            self.q[:28] += action * .001
            return (self.q.copy(), np.zeros(30)), False

    offsets = [0., .01, 0.]
    replay = collect_policy_replay(algo, Physics(), offsets)
    from types import SimpleNamespace
    reference = SimpleNamespace(qpos=algo.task.q.numpy())
    _, deployed = evaluate(algo, Physics(), reference, offsets)
    for i, episode_data in enumerate(replay.episodes):
        for name, value in zip(("qpos", "qvel", "actions"), episode_data):
            assert np.array_equal(value.numpy(), np.asarray(deployed[f"episode{i}_{name}"], dtype=np.float32))
    assert not torch.equal(replay.episodes[0][2], algo.reference_actions)
    assert all(torch.equal(a, b) for a, b in zip(replay.episodes[0], replay.episodes[2]))


def test_cached_sampling_matches_original_rng_for_unequal_episodes():
    """Optimization preserves window probabilities, values and RNG consumption."""
    short = tuple(x[:3] if i < 2 else x[:2] for i, x in enumerate(episode(20)))
    replay = DoorReplay([episode(0), short, episode(50)])
    for horizon in (1, 4, 2, 4):
        expected_rng = torch.Generator().manual_seed(2941)
        actual_rng = torch.Generator().manual_seed(2941)
        windows = [(i, f) for i, (_, _, actions) in enumerate(replay.episodes)
                   for f in range(len(actions)-horizon+1)]
        selected = torch.randint(len(windows), (200,), generator=expected_rng)
        actual = replay.sample(200, horizon, actual_rng)
        for column, index in enumerate(selected.tolist()):
            ep, frame = windows[index]
            q, v, actions = replay.episodes[ep]
            for value, source in zip(actual[:3], (q, v, actions)):
                length = horizon if source is actions else horizon+1
                assert torch.equal(value[:, column], source[frame:frame+length])
            assert actual[3][column] == frame
            assert torch.equal(actual[4][column], actions[frame-1] if frame else torch.zeros(28))
        assert torch.equal(expected_rng.get_state(), actual_rng.get_state())


@pytest.mark.parametrize('worlds', [1, 2, 4])
def test_batched_policy_collection_preserves_feedback_order_and_chunk_resets(worlds):
    import numpy as np
    from ndm_gym.motion.door_replay import collect_policy_replay, collect_policy_replay_batched
    from test_door_algo import build

    algo = build()
    with torch.no_grad():
        list(algo.pi.net.parameters())[-1].fill_(.2)

    class Physics:
        def reset(self, qpos_offset):
            self.q = algo.task.q[0].numpy().copy() + qpos_offset
            return self.q.copy(), np.zeros(30)

        def step(self, action):
            self.q[:28] += action * .001
            return (self.q.copy(), np.zeros(30)), False

    offsets = [0., .01, -.01, 0., .02]
    serial = collect_policy_replay(algo, Physics(), offsets)
    batched = collect_policy_replay_batched(algo, Physics, offsets, worlds)
    for expected, actual in zip(serial.episodes, batched.episodes):
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=1e-6)
    for x, y in zip(batched.episodes[0], batched.episodes[3]):
        assert torch.equal(x, y)
    with pytest.raises(ValueError, match='Positive'):
        collect_policy_replay_batched(algo, Physics, offsets, 0)
    shared = Physics()
    with pytest.raises(ValueError, match='independent'):
        collect_policy_replay_batched(algo, lambda: shared, offsets, 2)
