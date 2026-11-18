"""Opt-in door and latch start offsets: the defaults keep the original starts and collection."""

import numpy as np
import pytest

from ndm_gym.eval_door_full import evaluation_offsets, evaluation_starts
from ndm_gym.sim.rigid.adroit_door import DoorPhysics
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.train_door_full import door_latch_restarts
from test_door_native_collector import fixture


def test_default_starts_are_the_original_ones():
    for (a, b) in zip(evaluation_starts(), evaluation_offsets()):
        np.testing.assert_array_equal(a, b)
    for (a, b) in zip(evaluation_starts(0.), evaluation_offsets()):
        np.testing.assert_array_equal(a, b)


def test_offset_starts_raise_only_the_door_and_latch_of_the_perturbed_starts():
    q0, v0 = evaluation_offsets()
    q, v = evaluation_starts(.15)
    np.testing.assert_array_equal(v, v0)
    np.testing.assert_array_equal(q[:, :28], q0[:, :28])
    np.testing.assert_array_equal(q[0], q0[0])
    u = np.random.default_rng(20261116).uniform(0., 1., (256, 2))
    np.testing.assert_array_equal(q[1:, 28:30], .15 * u[1:])
    assert (q[1:, 28:30] >= 0).all() and (q[1:, 28:30] <= .15).all()


def test_no_restarts_without_an_offset():
    model, ref = fixture()
    with ParallelNativeDoorCollector(model, ref, 3, 1) as collector:
        assert door_latch_restarts(collector, 0., 7) is None


def _native_library(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    if shutil.which('c++') is None:
        pytest.skip('Optional native loop requires a C++ compiler')
    library = tmp_path / 'door_native.so'
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'tools/build_door_native.py'),
                    '--output', str(library)], check=True)
    return library


@pytest.mark.parametrize('native', [False, True])
def test_restarted_worlds_continue_as_physics_from_the_offset_start(tmp_path, native):
    model, ref = fixture()
    library = _native_library(tmp_path) if native else None
    rng = np.random.default_rng(3)
    with ParallelNativeDoorCollector(model, ref, 5, 2, library) as collector:
        for _ in range(ref.frames - 1):
            batch = collector.step(rng.uniform(-1, 1, (5, 28)))
        assert batch['done'].all()
        collector.step(np.zeros((5, 28)))
        with pytest.raises(ValueError, match='just reset'):
            collector.restart_worlds(np.arange(5), np.zeros((5, 30)))
    with ParallelNativeDoorCollector(model, ref, 5, 2, library) as collector:
        restart = door_latch_restarts(collector, .2, 11)
        for _ in range(ref.frames - 1):
            batch = collector.step(rng.uniform(-1, 1, (5, 28)))
        expected_offsets = .2 * np.random.default_rng([11, 20261116]).uniform(0., 1., (5, 2))
        restart(batch['done'])
        observed = collector.observe()
        np.testing.assert_array_equal(observed['qpos'][:, 28:30], ref.qpos[0, 28:30] + expected_offsets)
        np.testing.assert_array_equal(observed['frames'], np.zeros(5, dtype=np.int64))
        action = rng.uniform(-1, 1, (5, 28))
        actual = collector.step(action)
        for i in range(5):
            sim = DoorPhysics(model, ref)
            dq = np.zeros(30)
            dq[28:30] = expected_offsets[i]
            sim.reset(qpos_offset=dq)
            (q, v), _ = sim.step(action[i])
            np.testing.assert_array_equal(actual['next_qpos'][i], q)
            np.testing.assert_array_equal(actual['next_qvel'][i], v)
