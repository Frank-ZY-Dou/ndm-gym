"""Cross-task shapes, wrapped errors, hand-only collection and exact recovery."""
import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from ndm_gym.motion.adroit import AdroitReference, DIMS
from ndm_gym.motion.door_ring import DoorReplayRing
from ndm_gym.sim.rigid.adroit import AdroitPhysics, scene_identity
from ndm_gym.sim.rigid.door_collector import PackedNativeDoorCollector, ParallelNativeDoorCollector
from ndm_gym.tasks.adroit import AdroitTask
from ndm_gym.algos.door_supertrack import DoorSuperTrack
from ndm_gym.algos.door_supermimic import DoorSuperMimic
from ndm_gym.worldmodels.mjcf import MjcfLayout


def fixture(task):
    import mujoco
    nq, nv, nu = DIMS[task]
    names = [('ARTx' if task == 'relocate' and i == 0 else f'hand{i}') for i in range(nu)]
    if task == 'hammer':
        names.append('nail_dir')
    names += ['OBJTx', 'OBJTy', 'OBJTz', 'OBJRx', 'OBJRy', 'OBJRz']
    types = [2 if n in ('ARTx', 'nail_dir') or n.startswith('OBJT') else 3 for n in names]
    bodies = ''.join(f'<body pos="{i*.1} 0 0"><joint name="{n}" type="{("slide" if t == 2 else "hinge")}"/>'
                     '<geom size=".01" mass=".1"/></body>' for i, (n, t) in enumerate(zip(names, types)))
    actuators = ''.join(f'<motor joint="{n}" ctrlrange="-1 1" ctrllimited="true"/>' for n in names[:nu])
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002" gravity="0 0 0"/>'
                                         f'<worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>')
    ref = AdroitReference(np.zeros((21, nq)), np.zeros((21, nv)), np.zeros((20, nu)),
                         dict(kind='adroit-suite-physical-replay-v1', task=task, joint_names=names,
                              joint_types=types, joint_periodic=[t == 3 for t in types],
                              control_dt=.01, frame_skip=5, scene=scene_identity(model)))
    return model, ref


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp('native')/'adroit.so'
    subprocess.run([sys.executable, str(Path(__file__).parents[1]/'tools/build_door_native.py'),
                    '--output', str(out)], check=True, capture_output=True)
    return out


@pytest.mark.parametrize('task', DIMS)
def test_cross_task_physics_and_recovery(task, library):
    model, ref = fixture(task)
    serial = PackedNativeDoorCollector(model, ref, 5, physics_cls=AdroitPhysics)
    ring = DoorReplayRing(32, 5, nq=model.nq, nv=model.nv, nu=model.nu)
    rng = np.random.default_rng(7)
    with ParallelNativeDoorCollector(model, ref, 5, 2, library, physics_cls=AdroitPhysics) as parallel:
        for _ in range(23):  # includes terminal/reset boundary
            actions = rng.normal(0, .1, (5, model.nu))
            expected, actual = serial.step(actions), parallel.step(actions)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            ring.append(actual)
        q, v, a, _, previous = ring.sample(8, 3)
        assert q.shape == (4, 8, model.nq) and v.shape == (4, 8, model.nv)
        assert a.shape == (3, 8, model.nu) and previous.shape == (8, model.nu)
        saved = parallel.state_dict()
        action = rng.normal(0, .1, (5, model.nu))
        first = parallel.step(action)
        parallel.load_state_dict(saved)
        second = parallel.step(action)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize('task', DIMS)
@pytest.mark.parametrize('method', [DoorSuperTrack, DoorSuperMimic])
def test_shared_learners_train_and_restore(task, method):
    torch.manual_seed(42)
    model, ref = fixture(task)
    head = AdroitTask(ref, model.actuator_ctrlrange, goal_frames=2)
    layout = replace(MjcfLayout.from_model(model), dt=head.dt)
    kwargs = dict(hidden=(16,), wm_hidden=(16,), horizon=2)
    algo = method(head, layout, ref.actions, **kwargs)
    q, v = torch.randn(3, 8, model.nq)*.01, torch.randn(3, 8, model.nv)*.01
    ctrl = torch.randn(2, 8, model.nu)*.1
    algo.wm.update_norms(q[0], v[0], ctrl[0], q[1], v[1])
    loss = algo.wm_loss(q, v, ctrl)
    loss.backward()
    algo.opt_wm.step()
    state, frames, prev = (q[0], v[0]), torch.zeros(8, dtype=torch.long), torch.zeros(8, model.nu)
    original = copy.deepcopy(algo.wm.state_dict())
    loss, grad = algo.policy_update(state, frames, prev)
    assert torch.isfinite(loss) and grad > 0
    for k, value in original.items():
        torch.testing.assert_close(algo.wm.state_dict()[k], value, rtol=0, atol=0)
    restored = method(head, layout, ref.actions, **kwargs)
    restored.load_state_dict(copy.deepcopy(algo.state_dict()))
    for a, b in zip(algo.policy_update(state, frames, prev), restored.policy_update(state, frames, prev)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_periodic_error_and_exact_target_gradients():
    m, ref = fixture('pen')
    task = AdroitTask(ref, m.actuator_ctrlrange)
    q = torch.zeros(1, m.nq, requires_grad=True)
    f = torch.zeros(1, dtype=torch.long)
    a = torch.zeros(1, m.nu)
    task.reward((q, q*0), f, a, a).sum().backward()
    assert torch.isfinite(q.grad).all()
    q2 = q.detach().clone()
    q2[0, -1] = 2*torch.pi
    torch.testing.assert_close(task.tracking_terms((q2, q2*0), f), torch.zeros(1, 3), atol=1e-6, rtol=0)
    assert task.identity()['task'] == 'pen'


@pytest.mark.parametrize('task_name', DIMS)
def test_evaluation_reward_matches_training_and_passive_starts(task_name):
    from ndm_gym.eval_adroit import evaluation_offsets, score
    model, ref = fixture(task_name)
    task = AdroitTask(ref, model.actuator_ctrlrange)
    dq, dv = evaluation_offsets(ref)
    assert not dq[0].any() and not dv[0].any()
    assert not dq[:, model.nu:].any() and not dv[:, model.nu:].any()
    rng = np.random.default_rng(92)
    q = rng.normal(0, .01, (ref.frames, 3, model.nq))
    v = rng.normal(0, .01, q.shape)
    a = rng.normal(0, .01, (ref.frames-1, 3, model.nu))
    native = np.zeros((ref.frames-1, 3), dtype=bool)
    result = score(q, v, a, ref, task, native)
    previous = np.concatenate((np.zeros_like(a[:1]), a[:-1]))
    expected = sum(task.reward((torch.tensor(q[f+1]), torch.tensor(v[f+1])),
                              torch.full((3,), f+1, dtype=torch.long),
                              torch.tensor(a[f]), torch.tensor(previous[f])) for f in range(len(a)))
    np.testing.assert_allclose([x['tracking_return'] for x in result], expected.numpy(), rtol=1e-7)
    assert all(not x['native_success_final'] for x in result)  # reward is not success


def test_task_identity_rejects_changed_coordinate_convention():
    model, ref = fixture('pen')
    task = AdroitTask(ref, model.actuator_ctrlrange)
    layout = replace(MjcfLayout.from_model(model), dt=task.dt)
    algo = DoorSuperTrack(task, layout, ref.actions, hidden=(8,), wm_hidden=(8,), horizon=2)
    ck = copy.deepcopy(algo.state_dict())
    ck['task_identity']['adroit_task']['periodic'][0] = not ck['task_identity']['adroit_task']['periodic'][0]
    with pytest.raises(ValueError, match='identity'):
        algo.load_state_dict(ck)


@pytest.mark.parametrize('fault', ['nonfinite', 'action_shape', 'kind', 'periodicity', 'timestep'])
def test_reference_rejects_corrupt_contract(fault):
    _, ref = fixture('pen')
    q, v, a, meta = ref.qpos.copy(), ref.qvel.copy(), ref.actions.copy(), copy.deepcopy(ref.metadata)
    if fault == 'nonfinite':
        q[0, 0] = np.nan
    elif fault == 'action_shape':
        a = a[:, :-1]
    elif fault == 'kind':
        meta['kind'] = 'kinematic-target'
    elif fault == 'periodicity':
        meta['joint_periodic'].pop()
    elif fault == 'timestep':
        meta['control_dt'] = 0
    with pytest.raises(ValueError):
        AdroitReference(q, v, a, meta)


def test_physics_rejects_scene_and_object_actuation_changes():
    model, ref = fixture('hammer')
    model.body_pos[1, 0] += .01
    with pytest.raises(ValueError, match='scene'):
        AdroitPhysics(model, ref)
    model, ref = fixture('hammer')
    model.actuator_trnid[-1, 0] = model.nu  # would actuate the passive nail
    with pytest.raises(ValueError, match='hand-only'):
        AdroitPhysics(model, ref)


def test_precision_is_opt_in_and_checkpoint_bound():
    from ndm_gym.tasks.adroit import AdroitRewardCfg, training_arguments
    cfg, remaining = training_arguments(['--nail-tracking-std', '.001', '--seed', '12'])
    assert cfg.nail_std == .001 and remaining == ['--seed', '12']
    assert training_arguments([])[0] == AdroitRewardCfg()
    model, ref = fixture('hammer')
    default = AdroitTask(ref, model.actuator_ctrlrange)
    precise = AdroitTask(ref, model.actuator_ctrlrange, cfg=cfg)
    kwargs = dict(hidden=(16,), wm_hidden=(16,), horizon=2)
    layout = replace(MjcfLayout.from_model(model), dt=default.dt)
    old = DoorSuperMimic(default, layout, ref.actions, **kwargs)
    new = DoorSuperMimic(precise, layout, ref.actions, **kwargs)
    with pytest.raises(ValueError, match='task identity'):
        new.load_state_dict(old.state_dict())
    restored = DoorSuperMimic(precise, layout, ref.actions, **kwargs)
    restored.load_state_dict(copy.deepcopy(new.state_dict()))
    q = torch.zeros(1, model.nq)
    q[:, 26] = .0002
    frame = torch.zeros(1, dtype=torch.long)
    act = torch.zeros(1, model.nu)
    grad = []
    for task in (default, precise):
        state = q.clone().requires_grad_()
        task.reward((state, state*0), frame, act, act).sum().backward()
        grad.append(state.grad[0, 26])
    assert grad[0] < 0 and grad[1] < 0 and abs(grad[1]/grad[0]) > 90


@pytest.mark.parametrize('task_name', ['pen', 'relocate'])
def test_nail_precision_rejected_for_other_tasks(task_name):
    from ndm_gym.tasks.adroit import AdroitRewardCfg
    model, ref = fixture(task_name)
    with pytest.raises(ValueError, match='Hammer-only'):
        AdroitTask(ref, model.actuator_ctrlrange, cfg=AdroitRewardCfg(nail_std=.001))


def test_evaluation_starts_are_independent_of_precision():
    from ndm_gym.eval_adroit import evaluation_offsets
    _, ref = fixture('hammer')
    a, b = evaluation_offsets(ref), evaluation_offsets(ref, seed=3065)
    for x, y in zip(a, evaluation_offsets(ref)):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0][1:], b[0][1:])
    np.testing.assert_array_equal(a[0][:, 26:], b[0][:, 26:])


@pytest.mark.parametrize('option', [
    ['--door-latch-offset', '.3'], ['--lr', '.001'], ['--residual-scale', '.2'],
    ['--action-noise', '.2'], ['--wm-only-iters', '100'],
    ['--correction-reg', '.02'], ['--gamma', '.95'],
])
def test_suite_rejects_door_only_tuning_before_loading_inputs(option):
    from ndm_gym.train_adroit import main
    with pytest.raises(SystemExit) as error:
        main(['--model', 'absent.xml', '--reference', 'absent.npz',
              '--native-library', 'absent.so', '--output', 'absent-output',
              '--method', 'supermimic', *option])
    assert error.value.code == 2
