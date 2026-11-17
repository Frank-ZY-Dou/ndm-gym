"""Native collection boundaries and exact direct-physics comparison; no Torch."""

import numpy as np
import pytest

from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.sim.rigid.adroit_door import DoorPhysics
from ndm_gym.sim.rigid.door_collector import NativeDoorCollector, PackedNativeDoorCollector


def test_compiled_chunks_preserve_full_state_and_reject_external_forces(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
    if shutil.which('c++') is None:
        pytest.skip('Optional native loop requires a C++ compiler')
    library = tmp_path / 'door_native.so'
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'tools/build_door_native.py'),
                    '--output', str(library)], check=True)
    model, ref = fixture()
    serial = PackedNativeDoorCollector(model, ref, 7)
    rng = np.random.default_rng(81)
    with ParallelNativeDoorCollector(model, ref, 7, 2, library) as parallel:
        for _ in range(11):
            actions = rng.uniform(-2, 2, (7, 28))
            expected, actual = serial.step(actions), parallel.step(actions)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(np.concatenate([c.states for c in parallel.chunks]),
                                          serial.states)
        # Applied forces live in integration state, not just the temporary arena.
        import mujoco
        c = parallel.chunks[0]
        mujoco.mj_setState(model, c.sim.data, c.states[0], c.spec)
        c.sim.data.qfrc_applied[28] = 1
        mujoco.mj_getState(model, c.sim.data, c.states[0], c.spec)
        with pytest.raises(RuntimeError, match='code 2'):
            parallel.step(np.zeros((7, 28)))
        with pytest.raises(RuntimeError, match='failed'):
            parallel.step(np.zeros((7, 28)))


def fixture():
    import mujoco

    bodies = ''.join(f'<body pos="{i * .1} 0 0"><joint name="{name}"/>'
                     '<geom size=".01" mass=".1"/></body>'
                     for i, name in enumerate(JOINT_NAMES))
    actuators = ''.join(f'<motor joint="{name}" ctrlrange="-1 1" ctrllimited="true"/>'
                        for name in JOINT_NAMES[:28])
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002" gravity="0 0 0" noslip_iterations="2"/>'
        f'<worldbody><body name="frame">{bodies}</body></worldbody>'
        f'<actuator>{actuators}</actuator></mujoco>')
    q = np.zeros((4, 30))
    ref = DoorReference(q, q.copy(), np.zeros((3, 28)),
                        dict(kind='adroit-door-physical-replay-v1', joint_names=list(JOINT_NAMES),
                             control_dt=.01, frame_skip=5, scene_door_body_pos=[0, 0, 0]))
    return model, ref


def test_stagger_and_restore_preserve_the_next_closed_loop_transition():
    from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
    model, ref = fixture()
    with ParallelNativeDoorCollector(model, ref, 7, 2) as first:
        first.stagger_from_physical_replay(5)
        assert set(first.observe()['frames']) == {0, 1, 2}
        first.step(np.random.default_rng(7).uniform(-1, 1, (7, 28)))
        ck = first.state_dict()
        action = np.random.default_rng(8).uniform(-1, 1, (7, 28))
        expected = first.step(action)
        with ParallelNativeDoorCollector(model, ref, 7, 4) as second:
            second.load_state_dict(ck)
            actual = second.step(action)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            assert first.transitions == second.transitions


def test_persistent_control_steps_match_direct_native_physics():
    model, ref = fixture()
    collector = NativeDoorCollector(model, ref, 2)
    direct = [DoorPhysics(model, ref) for _ in range(2)]
    rng = np.random.default_rng(3)
    for step in range(8):
        actions = rng.uniform(-2, 2, (2, 28))
        batch = collector.step(actions)
        assert batch['frames'].tolist() == [step % 3] * 2
        for i, sim in enumerate(direct):
            state, done = sim.step(actions[i])
            np.testing.assert_array_equal(batch['next_qpos'][i], state[0])
            np.testing.assert_array_equal(batch['next_qvel'][i], state[1])
            assert batch['done'][i] == done
            if done:
                assert np.any(batch['next_qpos'][i] != ref.qpos[0])
                sim.reset()
        np.testing.assert_array_equal(collector.observe()['qpos'],
                                      np.stack([sim.state()[0] for sim in direct]))
    assert collector.transitions == 16
    assert model.opt.noslip_iterations == 2
    assert collector.worlds[0].model is collector.worlds[1].model
    assert collector.worlds[0].data is not collector.worlds[1].data


def test_invalid_batch_does_not_advance_and_physics_failure_is_not_retried():
    model, ref = fixture()
    collector = NativeDoorCollector(model, ref, 2)
    for actions in (np.zeros((1, 28)), np.full((2, 28), np.nan)):
        with pytest.raises(ValueError):
            collector.step(actions)
        assert collector.transitions == 0
        assert collector.observe()['frames'].tolist() == [0, 0]
    collector.worlds[1].data.qfrc_applied[28] = 1
    with pytest.raises(RuntimeError, match='forces'):
        collector.step(np.zeros((2, 28)))
    with pytest.raises(RuntimeError, match='failed'):
        collector.step(np.zeros((2, 28)))


def test_packed_integration_state_matches_separate_native_worlds():
    model, ref = fixture()
    separate = NativeDoorCollector(model, ref, 3)
    packed = PackedNativeDoorCollector(model, ref, 3)
    rng = np.random.default_rng(19)
    for _ in range(11):
        actions = rng.uniform(-1, 1, (3, 28))
        expected, actual = separate.step(actions), packed.step(actions)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        for key in separate.observe():
            np.testing.assert_array_equal(packed.observe()[key], separate.observe()[key])
    assert packed.transitions == 33


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_parallel_chunks_match_serial_integration_and_previous_actions(workers):
    from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector

    model, ref = fixture()
    serial = PackedNativeDoorCollector(model, ref, 7)
    rng = np.random.default_rng(3063)
    previous = np.zeros((7, 28))
    with ParallelNativeDoorCollector(model, ref, 7, workers) as parallel:
        for _ in range(11):
            actions = rng.uniform(-2, 2, (7, 28))
            expected, actual = serial.step(actions), parallel.step(actions)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(actual['previous_actions'], previous)
            previous = np.clip(actions, -1, 1)
            previous[expected['done']] = 0
            np.testing.assert_array_equal(parallel.previous_actions, previous)
            for key in serial.observe():
                np.testing.assert_array_equal(parallel.observe()[key], serial.observe()[key])
            np.testing.assert_array_equal(np.concatenate([c.states for c in parallel.chunks]),
                                          serial.states)
        assert parallel.transitions == 77
    with pytest.raises(RuntimeError, match='closed'):
        parallel.step(actions)
