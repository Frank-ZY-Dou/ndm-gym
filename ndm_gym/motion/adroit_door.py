"""Replay-derived native Adroit Door references (not reconstructed historical states).

Arrays include the restored initial state at index zero and each subsequent control
boundary. Actions[i] causes the transition from state i to state i + 1. No simulator
is imported by this module. Asset and demonstration files are supplied separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "ARTz", "ARRx", "ARRy", "ARRz", "WRJ1", "WRJ0",
    "FFJ3", "FFJ2", "FFJ1", "FFJ0", "MFJ3", "MFJ2", "MFJ1", "MFJ0",
    "RFJ3", "RFJ2", "RFJ1", "RFJ0", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0",
    "THJ4", "THJ3", "THJ2", "THJ1", "THJ0", "door_hinge", "latch",
)


@dataclass(frozen=True)
class DoorReference:
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray
    metadata: dict

    def __post_init__(self):
        q, v, a = self.qpos, self.qvel, self.actions
        if q.ndim != 2 or q.shape[1] != 30 or v.shape != q.shape:
            raise ValueError("Expected native Door qpos/qvel with shape (T + 1, 30)")
        if a.ndim != 2 or a.shape != (len(q) - 1, 28) or len(a) == 0:
            raise ValueError("Expected nonempty actions with shape (T, 28)")
        if not all(np.isfinite(x).all() for x in (q, v, a)):
            raise ValueError("Reference contains nonfinite values")
        if self.metadata.get("kind") != "adroit-door-physical-replay-v1":
            raise ValueError("Reference must identify physical replay derivation")
        dt = self.metadata.get("control_dt", 0)
        if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        names = self.metadata.get("joint_names", [])
        if tuple(names) != JOINT_NAMES:
            raise ValueError("Reference requires the native named joint order")

    @property
    def frames(self):
        return len(self.qpos)

    def errors(self, qpos, frame):
        """Hand RMS and door/latch absolute errors in native generalized coordinates.

        Hand root coordinates include translations: do not label the aggregate as
        radians. No periodic wrap is applied to these limited native joints.
        """
        qpos = np.asarray(qpos)
        if qpos.shape[-1:] != (30,) or not np.isfinite(qpos).all():
            raise ValueError("Expected finite native generalized coordinates")
        if np.any(np.asarray(frame) < 0) or np.any(np.asarray(frame) >= self.frames):
            raise ValueError("Reference frame out of range")
        delta = qpos - self.qpos[frame]
        return {"hand_rms": np.sqrt(np.mean(delta[..., :28] ** 2, axis=-1)),
                "door_rad": np.abs(delta[..., 28]), "latch_rad": np.abs(delta[..., 29])}

    @staticmethod
    def opened(qpos, threshold=1.35):
        """Actual door opening, scored separately from reference tracking."""
        return np.asarray(qpos)[..., 28] >= threshold

    def save(self, path: str | Path):
        np.savez_compressed(path, qpos=self.qpos, qvel=self.qvel, actions=self.actions,
                            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path):
        """Load a non-pickle archive; reject invalid states and schema."""
        with np.load(path, allow_pickle=False) as data:
            return cls(data["qpos"], data["qvel"], data["actions"], json.loads(str(data["metadata"])))
