"""Five-frame goals reach both the policy and critic and are checkpoint identity."""
import pytest
import torch

from ndm_gym.algos.door_supermimic import DoorSuperMimic
from test_door_algo import build


def test_five_frame_goals_are_used_by_imagination_and_checkpoint_identity():
    base = build()
    base.task.goal_frames = 5
    algo = DoorSuperMimic(base.task, base.layout, base.reference_actions,
                          hidden=(16,), wm_hidden=(16,), horizon=2)
    state = (base.task.q[:2], base.task.v[:2])
    frame = torch.zeros(2, dtype=torch.long)
    previous = torch.zeros(2, 28)
    assert algo.task.goal(state, frame).shape == (2, 300)
    assert algo.features(state, frame, previous).shape == (2, 388)
    loss, grad = algo.policy_update(state, frame, previous)
    assert torch.isfinite(loss) and torch.isfinite(grad)
    ck = algo.state_dict()
    base.task.goal_frames = 1
    other = DoorSuperMimic(base.task, base.layout, base.reference_actions,
                           hidden=(16,), wm_hidden=(16,), horizon=2)
    with pytest.raises(ValueError, match='identity'):
        other.load_state_dict(ck)
