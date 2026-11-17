"""Multivariate evaluation starts, full physical episodes and independent scoring."""
from types import SimpleNamespace

import numpy as np
import torch

from ndm_gym.eval_door_full import evaluation_offsets, initialize_evaluation, evaluate, score
from ndm_gym.sim.rigid.adroit_door import DoorPhysics
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.tasks.adroit_door import DoorTask
from test_door_native_collector import fixture


def test_fixed_perturbations_are_multivariate_and_leave_passive_coordinates_unchanged():
    q, v = evaluation_offsets()
    q2, v2 = evaluation_offsets()
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(v, v2)
    assert not q[0].any() and not v[0].any()
    assert not q[:, 28:].any() and not v[:, 28:].any()
    assert np.all(np.std(q[1:, :28], axis=0) > 0)
    assert abs(q[:, 0]).max() <= .001 and abs(q[:, 1:28]).max() <= .015
    assert abs(v[:, 0]).max() <= .001 and abs(v[:, 1:28]).max() <= .01


def test_evaluation_records_actual_terminal_states_and_scores_all_frames():
    model, ref = fixture()
    ref.actions[:] = np.random.default_rng(3).uniform(-.1, .1, ref.actions.shape)
    task = DoorTask(ref, model.actuator_ctrlrange)
    algo = SimpleNamespace(task=task, reference_actions=torch.tensor(ref.actions, dtype=torch.float32))
    dq, dv = evaluation_offsets()
    with ParallelNativeDoorCollector(model, ref, 256, 2) as collector:
        initialize_evaluation(collector, dq, dv)
        q, v, actions = evaluate(algo, collector, ref, prior=True)
    np.testing.assert_array_equal(q[0], ref.qpos[0] + dq)
    np.testing.assert_array_equal(v[0], ref.qvel[0] + dv)
    direct = DoorPhysics(model, ref)
    for frame, a in enumerate(ref.actions):
        (expected_q, expected_v), _ = direct.step(a)
        np.testing.assert_array_equal(q[frame+1, 0], expected_q)
        np.testing.assert_array_equal(v[frame+1, 0], expected_v)
    rows = score(q, v, actions, ref, task.cfg)
    expected = torch.zeros(256)
    prev = torch.zeros((256, 28))
    for frame in range(len(actions)):
        a = torch.tensor(actions[frame], dtype=torch.float32)
        state = tuple(torch.tensor(x[frame+1], dtype=torch.float32) for x in (q, v))
        expected += task.reward(state, torch.full((256,), frame+1), a, prev)
        prev = a
    np.testing.assert_allclose([r['tracking_return'] for r in rows], expected.numpy(), rtol=1e-6)
    assert len(rows) == 256 and all(r['finite'] for r in rows)
