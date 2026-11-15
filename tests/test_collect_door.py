"""Public bootstrap episode format and reset boundary preservation."""

import numpy as np
import torch

from ndm_gym.collect_door import save_replay
from ndm_gym.motion.door_replay import DoorReplay


def test_saved_bootstrap_preserves_episode_boundaries(tmp_path):
    episodes = [(np.full((4, 30), i), np.zeros((4, 30)), np.full((3, 28), i / 10))
                for i in range(3)]
    original = DoorReplay(episodes)
    path = tmp_path / 'episodes.npz'
    save_replay(original, path)
    loaded = DoorReplay.load(path)
    train, heldout = loaded.split([2])
    assert len(train.episodes) == 2 and len(heldout.episodes) == 1
    for expected, actual in zip(original.episodes, loaded.episodes):
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    q, _, _, _, _ = train.sample(32, 3)
    assert torch.equal(q[0], q[-1])
