# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/mdp.py and motion.py) under its
# MIT License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""InstantTrack's contact reward (``--contact_reward``; ``--ref_contact_fix`` is BlinkMimic's): a term of SuperMimic's
imagined reward only, which matches a smooth foot-contact indicator to the reference's contact state and rewards low
horizontal foot speed while the reference foot is down. Evaluation keeps the task's reward."""

from __future__ import annotations

import torch

from ..motion.library import MotionLib
from ..robots.g1 import TRACKED_BODIES

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact


def foot_bodies() -> list[int]:
    """The feet's indices among the tracked bodies."""
    return [TRACKED_BODIES.index(b) for b in FEET]


def reference_contact(motion: MotionLib, feet: list[int], calibrated: bool) -> torch.Tensor:
    """The reference's foot contact per frame (num_frames, len(feet)): the foot below ``CONTACT_Z``; with ``calibrated``
    (``--ref_contact_fix``) below its clip's own floor (the 5th percentile of the foot's height in the clip) + 1.5 cm and
    slower than 0.5 m/s horizontally, since the retargeted feet of some clips sit about 1 cm higher."""
    if motion.body_p.shape[1] != len(TRACKED_BODIES):
        raise ValueError("the motion library's bodies must be the G1's tracked bodies")
    z = motion.body_p[:, feet, 2]
    if not calibrated:
        return (z < CONTACT_Z).float()
    vh = motion.body_v[:, feet, :2].norm(dim=-1)
    floor = torch.zeros_like(z)
    for c in range(len(motion.names)):
        s = motion.clip_id == c
        floor[s] = z[s].quantile(0.05, dim=0)
    return ((z < floor + 0.015) & (vh < 0.5)).float()


def contact_term(foot_p: torch.Tensor, foot_v: torch.Tensor, c_ref: torch.Tensor) -> torch.Tensor:
    """Per sample, the mean over the feet (positions and velocities (..., feet, 3), reference contact (..., feet)) of the
    contact match 1 - (c - c_ref)^2, c = sigmoid((CONTACT_Z - z) / 5 mm), plus c_ref exp(-|v_xy|^2 / (0.3 m/s)^2)."""
    c_sim = torch.sigmoid((CONTACT_Z - foot_p[..., 2]) / 0.005)
    slip = (foot_v[..., :2] ** 2).sum(-1)
    return ((1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)).mean(-1)
