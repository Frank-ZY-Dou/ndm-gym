"""Named perturbation and physical metric checks for the development runner."""

import numpy as np

from ndm_gym.train_door import evaluate
from test_door_algo import build


def test_evaluation_uses_arrz_and_scores_actual_opening():
    algo = build()
    reference = type('Reference', (), {'qpos': np.zeros((6, 30))})()

    class Physics:
        def reset(self, qpos_offset):
            assert qpos_offset[3] == .01
            assert np.count_nonzero(qpos_offset) == 1
            self.frame = 0
            return np.zeros(30), np.zeros(30)

        def step(self, action):
            assert action.shape == (28,)
            self.frame += 1
            q = np.zeros(30)
            q[28] = 1.4
            return (q, np.zeros(30)), self.frame == 5

    metrics, saved = evaluate(algo, Physics(), reference, [.01])
    assert metrics[0]['opened_final']
    assert metrics[0]['door_rmse_rad'] == 1.4
    assert metrics[0]['hand_rmse_rad'] == 0
    assert saved['episode0_qpos'].shape == (6, 30)
    assert len(saved['episode0_actions']) == 5


def test_teacher_collection_uses_controls_and_bounded_targets():
    import torch
    from ndm_gym.train_door import collect_teacher

    algo = build()

    class Physics:
        steps = 0

        def reset(self, qpos_offset):
            assert qpos_offset[3] == .01
            return np.ones(30) * .01, np.zeros(30)

        def step(self, action):
            assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            self.steps += 1
            return (np.ones(30) * .01, np.zeros(30)), False

    sim = Physics()
    obs, goal, target = collect_teacher(algo, sim, [.01], .3)
    assert sim.steps == 5
    assert obs.shape[0] == goal.shape[0] == target.shape[0] == 5
    assert torch.isfinite(target).all() and target.abs().max() <= 1
    assert torch.count_nonzero(target) > 0
