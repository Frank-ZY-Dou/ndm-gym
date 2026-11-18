"""Door registration, physical-control loss and exact optimizer recovery."""

import copy

import numpy as np
import pytest
import torch

from ndm_gym.algos import ALGOS, DOOR_ALGOS
from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference
from ndm_gym.tasks.adroit_door import DoorTask
from ndm_gym.worldmodels.mjcf import MjcfLayout


def build():
    q = np.zeros((6, 30))
    ref = DoorReference(q, q.copy(), np.zeros((5, 28)),
                        {"kind": "adroit-door-physical-replay-v1", "control_dt": .02,
                         "joint_names": list(JOINT_NAMES)})
    task = DoorTask(ref, np.tile([-2., 4.], (28, 1)))
    lay = MjcfLayout(30, 30, 28, .02, None, (), (),
                     tuple((i, i, False) for i in range(30)), (0,),
                     tuple(range(28)), tuple(range(28)), (1.,)*28, (1.,)*28,
                     ((0., -1., 0.),)*28, ((-2., 4.),)*28, (None,)*28)
    return DOOR_ALGOS["residual"](task, lay, np.zeros((5, 28)),
                                  hidden=(16,), wm_hidden=(16,), horizon=2)


def test_registration_loss_and_exact_next_update():
    assert {"supermimic", "supertrack"} <= set(ALGOS) and not set(DOOR_ALGOS.values()) & set(ALGOS.values())
    torch.manual_seed(2914)
    algo = build()
    state = (torch.zeros(2, 30), torch.zeros(2, 30))
    frame, prev = torch.zeros(2, dtype=torch.long), torch.zeros(2, 28)
    q = torch.zeros(2, 2, 30)
    ctrl = algo.setpoint(prev)[None]
    assert torch.isfinite(algo.wm_loss(q, q, ctrl))
    algo.policy_update(state, frame, prev)
    checkpoint = copy.deepcopy(algo.state_dict())
    resumed = build()
    resumed.load_state_dict(checkpoint)
    a = algo.policy_update(state, frame, prev)
    b = resumed.policy_update(state, frame, prev)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(algo.pi.parameters(), resumed.pi.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    checkpoint['reference_actions'][0, 0] = .1
    with pytest.raises(ValueError, match='reference or horizon'):
        resumed.load_state_dict(checkpoint)


@pytest.mark.parametrize("field", ["q", "v", "ctrl_mid", "ctrl_half", "dt", "reward", "layout", "missing"])
def test_task_mismatch_rejected_before_mutation(field):
    original, resumed = build(), build()
    checkpoint = copy.deepcopy(original.state_dict())
    identity = checkpoint["task_identity"]
    if field == "missing":
        del checkpoint["task_identity"]
    elif field in ("q", "v", "ctrl_mid", "ctrl_half"):
        identity[field].view(-1)[0] += .01
    elif field == "dt":
        identity[field] *= 2
    elif field == "reward":
        identity[field]["opening_rad"] += .1
    else:
        identity[field]["gain"] = (2.,) * 28
    before = copy.deepcopy(resumed.state_dict())
    with pytest.raises(ValueError, match="task identity"):
        resumed.load_state_dict(checkpoint)
    for name, tensor in resumed.pi.state_dict().items():
        torch.testing.assert_close(tensor, before["pi"][name], rtol=0, atol=0)
    for name, tensor in resumed.wm.state_dict().items():
        torch.testing.assert_close(tensor, before["wm"][name], rtol=0, atol=0)


def test_supervised_update_recovery_and_invalid_targets():
    torch.manual_seed(2921)
    algo = build()
    state = (torch.zeros(4, 30), torch.zeros(4, 30))
    obs = algo.task.obs(state, torch.zeros(4, 28))
    goal = algo.task.goal(state, torch.zeros(4, dtype=torch.long))
    target = torch.full((4, 28), .2)
    algo.pi.norm.update(torch.cat((obs, goal), -1))
    algo.supervised_update(obs, goal, target)
    resumed = build()
    resumed.load_state_dict(copy.deepcopy(algo.state_dict()))
    a = algo.supervised_update(obs, goal, target)
    b = resumed.supervised_update(obs, goal, target)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for x, y in zip(algo.pi.parameters(), resumed.pi.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    before = copy.deepcopy(algo.state_dict())
    for bad in (target * float("nan"), target * 6, target[:0]):
        with pytest.raises(ValueError):
            algo.supervised_update(obs, goal, bad)
    for name, value in algo.pi.state_dict().items():
        torch.testing.assert_close(value, before["pi"][name], rtol=0, atol=0)


def test_door_methods_share_public_registry():
    from ndm_gym.train_door import DOOR_METHODS

    assert DOOR_ALGOS == DOOR_METHODS
    assert {"supermimic", "supertrack"} <= set(ALGOS) and not set(DOOR_ALGOS.values()) & set(ALGOS.values())
