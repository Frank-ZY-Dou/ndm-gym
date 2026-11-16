"""Replay retains terminal states and never splices through reset or ring wrap."""
import numpy as np
import pytest
import torch

from ndm_gym.motion.door_ring import DoorReplayRing


def test_ring_wrap_and_reset_safe_windows():
    ring = DoorReplayRing(5, 2)
    for step in range(13):
        frames = np.array([step % 3, step % 4])
        q = np.repeat(frames[:, None], 30, axis=1)
        actions = np.repeat((frames / 10)[:, None], 28, axis=1)
        previous = np.where(frames[:, None] == 0, 0, actions - .1)
        ring.append(dict(qpos=q, qvel=q, next_qpos=q+1, next_qvel=q+1,
                         actions=actions, previous_actions=previous, frames=frames,
                         done=frames == np.array([2, 3])))
    q, v, a, frames, previous = ring.sample(200, 3)
    np.testing.assert_array_equal(q[1:].numpy(), q[:-1].numpy() + 1)
    np.testing.assert_array_equal(v.numpy(), q.numpy())
    np.testing.assert_allclose(a.numpy(), q[:-1, :, :28].numpy() / 10)
    np.testing.assert_allclose(previous.numpy(),
                               np.repeat(np.where(frames.numpy() == 0, 0,
                                                  (frames.numpy()-1)/10)[:, None], 28, axis=1))
    with pytest.raises(ValueError, match='reset-safe'):
        ring.sample(10, 5)
    restored = DoorReplayRing(5, 2)
    restored.load_state_dict(ring.state_dict())
    for a, b in zip(ring.sample(50, 3, torch.Generator().manual_seed(8)),
                    restored.sample(50, 3, torch.Generator().manual_seed(8))):
        assert torch.equal(a, b)
