"""Physical/replay/RNG recovery must reproduce subsequent learning, not just weights."""
import copy

import numpy as np
import pytest
import torch

from ndm_gym.motion.door_ring import DoorReplayRing
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.train_door_full import restore, save
from ndm_gym.train_door_online import updates
from test_door_algo import build
from test_door_native_collector import fixture


def test_full_checkpoint_restores_physics_replay_and_next_optimizer_updates(tmp_path):
    model, ref = fixture()
    algo = build()
    ring = DoorReplayRing(5, 3)
    torch.manual_seed(26)
    with ParallelNativeDoorCollector(model, ref, 3, 2) as collector:
        for _ in range(7):
            ring.append(collector.step(np.full((3, 28), .1)))
        updates(algo, ring, 4, 2, 2, 1)
        path = tmp_path/'latest.pt'
        protocol = {'identity': 'fixture'}
        save(path, algo, collector, ring, protocol, 1, {})
        checkpoint = torch.load(path, weights_only=False)
        next_action = torch.rand(3, 28).numpy()
        expected_batch = collector.step(next_action)
        updates(algo, ring, 4, 2, 2, 1)
        expected = copy.deepcopy(algo.state_dict())
        fresh = build()
        with ParallelNativeDoorCollector(model, ref, 3, 1) as restored:
            replay = DoorReplayRing(5, 3)
            with pytest.raises(ValueError, match='protocol'):
                restore(checkpoint, fresh, restored, replay, {})
            assert restore(checkpoint, fresh, restored, replay, protocol) == (1, {})
            actual_batch = restored.step(torch.rand(3, 28).numpy())
            for key in expected_batch:
                np.testing.assert_array_equal(actual_batch[key], expected_batch[key])
            updates(fresh, replay, 4, 2, 2, 1)
            for network in ('wm', 'pi'):
                for key, value in expected[network].items():
                    assert torch.equal(value, fresh.state_dict()[network][key]), (network, key)
