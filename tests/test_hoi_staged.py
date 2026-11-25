"""train_hoi's staged options on stand-ins: --pi_fixed_joints zeroes the policy's residual on those joints (the reference's
action alone), --init loads another run's networks and --init_zero_joints zeroes the policy's last-layer rows of those joints."""

import types

import pytest
import torch
import torch.nn as nn

from ndm_gym.algos.hoi_algos import HOIAgent, action_indices

BODY = types.SimpleNamespace(joint_names=("left_hip_pitch_joint", "left_knee_joint", "left_index_1_joint", "left_thumb_2_joint", "extra"), num_actions=4)
FINGERS = ".*_(thumb|index|middle|ring|pinky)_.*_joint"


def test_action_indices():
    assert action_indices(BODY, FINGERS) == [2, 3] and action_indices(BODY, "") == []
    with pytest.raises(ValueError):
        action_indices(BODY, "extra")  # not an action


class _Pi(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(nn.Linear(3, 4))

    def forward(self, obs):
        return self.dec(obs["x"])


def agent(mask_pattern=""):
    a = types.SimpleNamespace(body=BODY, pi=_Pi(), wm=nn.Linear(2, 2), critic=nn.Linear(2, 1), act_mask=None)
    if mask_pattern:
        a.act_mask = torch.ones(4)
        a.act_mask[action_indices(BODY, mask_pattern)] = 0.0
    a.base_action = lambda t: torch.full((t.shape[0], 4), 0.5)
    return a


def test_fixed_joints_follow_the_reference():
    torch.manual_seed(0)
    obs, t = {"x": torch.randn(5, 3)}, torch.zeros(5, dtype=torch.long)
    free, fixed = agent(), agent(FINGERS)
    fixed.pi.load_state_dict(free.pi.state_dict())
    a, b = HOIAgent.act(free, obs, t), HOIAgent.act(fixed, obs, t)
    assert torch.equal(b[:, 2:], torch.full((5, 2), 0.5)) and torch.equal(b[:, :2], a[:, :2]) and not torch.equal(a[:, 2:], b[:, 2:])


def test_init_from_zeroes_the_given_rows():
    torch.manual_seed(1)
    source, target = agent(), agent()
    ck = {"wm": source.wm.state_dict(), "pi": source.pi.state_dict(), "critic": source.critic.state_dict()}
    HOIAgent.init_from(target, ck, FINGERS)
    w, b = target.pi.dec[-1].weight, target.pi.dec[-1].bias
    assert torch.equal(w[:2], source.pi.dec[-1].weight[:2]) and torch.equal(b[:2], source.pi.dec[-1].bias[:2])
    assert not w[2:].any() and not b[2:].any()
    assert torch.equal(target.wm.weight, source.wm.weight) and torch.equal(target.critic.weight, source.critic.weight)
    obs = {"x": torch.randn(4, 3)}
    assert torch.equal(HOIAgent.act(target, obs, torch.zeros(4, dtype=torch.long))[:, 2:], torch.full((4, 2), 0.5))
