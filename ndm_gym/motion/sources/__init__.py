"""Motion sources. Each loader returns a list of :class:`Clip` (root pose and joint angles at the source rate, and for
human-object interaction the object's pose and the scene's other objects)."""

from __future__ import annotations

from typing import NamedTuple

import torch


class Clip(NamedTuple):
    name: str
    root_p: torch.Tensor  # (T, 3) world root position
    root_quat: torch.Tensor  # (T, 4) world root orientation, wxyz
    joints: torch.Tensor  # (T, J) joint angles in the robot's joint order
    fps: float
    obj_name: str | None = None  # the interacted object, or None for a robot-only clip
    obj_p: torch.Tensor | None = None  # (T, 3) world object position
    obj_quat: torch.Tensor | None = None  # (T, 4) world object orientation, wxyz
    obj_joint: torch.Tensor | None = None  # (T, k) joint of an articulated object (door hinge), if any
    contact: torch.Tensor | None = None  # (T, c) reference contact flags, if any
    extra_names: tuple[str, ...] | None = None  # the scene's other objects (static props), if any
    extra_p: torch.Tensor | None = None  # (T, n, 3) their world positions
    extra_quat: torch.Tensor | None = None  # (T, n, 4) their world orientations, wxyz
    meta: dict | None = None  # the source's description of the clip (the converted format's sidecar), if any
    # the source's own reference velocities, if it stores them (HDMI: not finite differences of the poses):
    # keypoint_names, keypoint_lin_vel / keypoint_ang_vel (T, b, 3), joint_vel (T, J), object_joint_vel (T, k) if articulated
    src_vel: dict | None = None


def read_clip_list(path: str) -> list[str]:
    """Clip names from a list file, one per line; text after ``#`` and blank lines are ignored."""
    with open(path) as f:
        names = [n for n in (line.split("#", 1)[0].strip() for line in f) if n]
    if not names:
        raise ValueError(f"{path} lists no clips")
    if len(set(names)) < len(names):
        raise ValueError(f"{path} lists a clip more than once")
    return names


def select(clips: list[Clip], names: list[str]) -> list[Clip]:
    """The clips with these names, in the order of ``clips``; a name that matches no clip is an error."""
    missing = sorted(set(names) - {c.name for c in clips})
    if missing:
        raise ValueError(f"no clip named {missing}")
    keep = set(names)
    return [c for c in clips if c.name in keep]
