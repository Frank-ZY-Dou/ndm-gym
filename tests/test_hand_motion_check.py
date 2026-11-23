"""tools/hand_motion_check.py's measures on made-up signals and ndm_gym.evaluate_hand's recording of a rollout's targets."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import hand_motion_check as hmc  # noqa: E402

from ndm_gym.evaluate_hand import rollout  # noqa: E402

DT = 0.02


def test_joint_measures():
    """A slow sine: small acceleration and jerk, no high-frequency share, no limit hits. Noise added: larger acceleration, jerk
    and high-frequency share. Held at a limit for half the steps: that share of the pairs is a hit."""
    t = np.arange(200)[:, None] * DT
    smooth = 0.3 * np.sin(2 * np.pi * 1.0 * t) * np.ones((1, 4))
    lo, hi = -np.ones(4), np.ones(4)
    a = hmc.joint_measures(smooth, DT, 5.0, lo, hi, 0.01)
    assert a["hf_share"] < 1e-3 and a["limit_hits"] == 0.0
    assert abs(a["acc"] - 0.3 * (2 * np.pi) ** 2 / np.sqrt(2)) < 0.05 * a["acc"]  # the RMS of a sine's second derivative
    noisy = smooth + 0.01 * np.random.default_rng(0).standard_normal(smooth.shape)
    b = hmc.joint_measures(noisy, DT, 5.0, lo, hi, 0.01)
    assert b["acc"] > 3 * a["acc"] and b["jerk"] > 10 * a["jerk"] and b["hf_share"] > 100 * a["hf_share"] and b["hf_share"] > 0.1
    held = smooth.copy()
    held[100:, 0] = hi[0]
    assert abs(hmc.joint_measures(held, DT, 5.0, lo, hi, 0.01)["limit_hits"] - 100 / 800) < 1e-9
    assert abs(hmc.change(np.arange(10.0)[:, None] * 0.5) - 0.5) < 1e-12


def test_slips():
    """Two fingertips touching at every step: a fingertip still in the object's frame has no slip, one moving 1 mm a step has 1
    mm; the object still in the palm's frame has none, moving 2 mm a step 2 mm; no contact: None."""
    T = 10
    touch = np.ones((T, 2), bool)
    tips = np.zeros((T, 2, 3))
    tips[:, 1, 0] = np.arange(T) * 0.001
    obj = np.zeros((T, 3))
    obj[:, 2] = np.arange(T) * 0.002
    s = hmc.slips(touch, tips, obj)
    assert abs(s["fingertip_slip_mm"] - 0.5) < 1e-9 and abs(s["object_slip_mm"] - 2.0) < 1e-9 and s["held_steps"] == T
    none = hmc.slips(np.zeros((T, 2), bool), tips, obj)
    assert none["fingertip_slip_mm"] is None and none["object_slip_mm"] is None


class _Sim:
    """A stand-in simulator: the state is the last target plus a step count; it never terminates."""

    def __init__(self, n, j):
        self.N, self.device, self.x, self.f = n, "cpu", torch.zeros(n, j), torch.zeros(n, dtype=torch.long)
        self.qpos = self.x

    def state(self):
        return self.x.clone()

    def step(self, target, reset=False):
        self.x = target.clone()
        self.qpos = self.x
        self.f = self.f + 1
        return self.x.clone(), torch.zeros(self.N, dtype=torch.bool), None


class _Task:
    def __init__(self, j, frames):
        self.kin = type("K", (), {"num_joints": j})()
        self.ref = type("R", (), {"frames": frames})()

    def setpoint(self, a, f):
        return a + f.unsqueeze(-1).float()


class _Algo:
    def act(self, s, a_prev, f):
        return a_prev + 0.1


def test_rollout_record(monkeypatch):
    """With ``record`` the rollout is the same and its lists hold each step's action and position targets."""
    import ndm_gym.tasks.hand_tracking as ht

    monkeypatch.setattr(ht, "unflatten_state", lambda s: s)
    out = [rollout(_Sim(3, 2), _Algo(), _Task(2, 5), policy=p) for p in (True, False)]
    rec, rec0 = {}, {}
    with_rec = [
        rollout(_Sim(3, 2), _Algo(), _Task(2, 5), policy=True, record=rec),
        rollout(_Sim(3, 2), _Algo(), _Task(2, 5), policy=False, record=rec0),
    ]
    for a, b in zip(out, with_rec, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert len(rec["actions"]) == len(rec["targets"]) == 4 and torch.allclose(rec["actions"][-1], torch.full((3, 2), 0.4))
    assert torch.equal(rec["targets"][2], rec["actions"][2] + 2) and all(
        torch.equal(t, torch.full((3, 2), float(i))) for i, t in enumerate(rec0["targets"])
    )
