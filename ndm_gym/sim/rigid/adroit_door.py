"""Single-world native CPU Door transitions; no simulator import until construction.

The caller supplies the pinned native model (including controller gains and scene).
No object forces, reference feed-forward controls or per-step state restoration.
This correctness path is not a measured replacement for GPU training simulation.
"""

from __future__ import annotations

import numpy as np

from ...motion.adroit_door import JOINT_NAMES, DoorReference


class DoorPhysics:
    def __init__(self, model, reference: DoorReference):
        import mujoco

        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt))
        if (model.nq, model.nv, model.nu) != (30, 30, 28) or names != JOINT_NAMES:
            raise ValueError("Expected native Door joint layout")
        if not np.array_equal(model.actuator_trnid[:, 0], np.arange(28)):
            raise ValueError("Expected hand-only actuators in native joint order")
        if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Expected joint transmissions")
        skip = reference.metadata.get("frame_skip")
        if not isinstance(skip, int) or skip <= 0:
            raise ValueError("Reference requires positive integer frame_skip")
        if not np.isclose(model.opt.timestep * skip, reference.metadata["control_dt"], rtol=0, atol=1e-12):
            raise ValueError("Model/reference timestep mismatch")
        scene = reference.metadata.get("scene_door_body_pos")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
        if scene is None or body < 0 or not np.allclose(model.body_pos[body], scene, rtol=0, atol=1e-12):
            raise ValueError("Model/reference initial door scene mismatch")
        ranges = model.actuator_ctrlrange.copy()
        if not np.isfinite(ranges).all() or not (ranges[:, 1] > ranges[:, 0]).all():
            raise ValueError("Invalid native actuator ranges")
        self.model, self.reference, self.frame_skip = model, reference, skip
        self.data = mujoco.MjData(model)
        self.mid, self.half = ranges.mean(-1), np.diff(ranges, axis=-1)[:, 0] / 2
        self.reset()

    def state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def reset(self, qpos_offset=None, qvel_offset=None):
        """Restore only the initial state, with optional explicit evaluation offsets.

        Offsets do not change the reference or scene and are not silently clipped.
        Terminal states remain available until this method is called explicitly.
        """
        import mujoco

        offsets = []
        for value in (qpos_offset, qvel_offset):
            x = np.zeros(30) if value is None else np.asarray(value, dtype=float)
            if x.shape != (30,) or not np.isfinite(x).all():
                raise ValueError("Expected finite offset (30,)")
            offsets.append(x)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.reference.qpos[0] + offsets[0]
        self.data.qvel[:] = self.reference.qvel[0] + offsets[1]
        mujoco.mj_forward(self.model, self.data)
        self.frame = 0
        self.done = False
        return self.state()

    def step(self, action):
        """Apply normalized hand controls; return next state and horizon termination."""
        import mujoco

        a = np.asarray(action, dtype=float)
        if a.shape != (28,) or not np.isfinite(a).all():
            raise ValueError("Expected finite normalized hand action (28,)")
        if self.done:
            raise RuntimeError("Reset required after terminal transition")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError("External forces are forbidden")
        self.data.ctrl[:] = self.mid + np.clip(a, -1, 1) * self.half
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.done = True
                raise FloatingPointError("Nonfinite native Door transition")
        self.frame += 1
        self.done = self.frame >= self.reference.frames - 1
        return self.state(), self.done
