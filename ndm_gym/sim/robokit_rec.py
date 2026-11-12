"""Transitions recorded from RoboKit's MuJoCo Warp simulator (robokit.sim.MjWarpSim) as RoboKit runs it, for the world model
of MJCF models (worldmodels.mjcf). ``recording_sim()`` returns a subclass of RoboKit's simulator whose ``step()`` writes, for the
first ``record`` worlds, the state and the controls before the step, the step's length (``substeps`` times the timestep the
caller set) and the state after it into a ring buffer on the device. Its kernels are plain launches, so inside an enclosing
CUDA graph capture (SimMPC's rollout graphs) they are recorded with the step's own kernels and run at every replay; the ring's
cursor lives on the device. ``resized()`` keeps the subclass, as SimMPC builds its rollout worlds with it. RoboKit's files stay
unmodified: a run swaps the simulator class at run time.

``drain()`` takes the rows written since the last drain (an error if the ring went round, so no row is lost silently) and
``recording()`` turns them into the recordings ``wmbench/fit_mjcf.py`` reads: one step length at a time, a reset wherever a
world's next row does not start from the state its previous row ended in (a new rollout)."""

import numpy as np  # no `from __future__ import annotations`: warp reads the kernels' annotations as objects

_KERNELS = None


def _kernels():
    """The record kernels, built on first use (warp is RoboKit's dependency, not this library's)."""
    global _KERNELS
    if _KERNELS is None:
        import warp as wp

        @wp.kernel
        def record_pre(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
                       timestep: wp.array(dtype=float), cursor: wp.array(dtype=int), substeps: int, buf: wp.array3d(dtype=float)):  # fmt: skip
            k = wp.tid()
            row = cursor[0] % buf.shape[0]
            o = int(0)
            for i in range(qpos.shape[1]):
                buf[row, k, o + i] = qpos[k, i]
            o += qpos.shape[1]
            for i in range(qvel.shape[1]):
                buf[row, k, o + i] = qvel[k, i]
            o += qvel.shape[1]
            for i in range(ctrl.shape[1]):
                buf[row, k, o + i] = ctrl[k, i]
            o += ctrl.shape[1]
            dt = timestep[0]
            if timestep.shape[0] > 1:
                dt = timestep[k]
            buf[row, k, o] = dt * float(substeps)

        @wp.kernel
        def record_post(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), cursor: wp.array(dtype=int), offset: int,
                        buf: wp.array3d(dtype=float)):  # fmt: skip
            k = wp.tid()
            row = cursor[0] % buf.shape[0]
            for i in range(qpos.shape[1]):
                buf[row, k, offset + i] = qpos[k, i]
            for i in range(qvel.shape[1]):
                buf[row, k, offset + qpos.shape[1] + i] = qvel[k, i]

        @wp.kernel
        def advance(cursor: wp.array(dtype=int)):
            cursor[0] = cursor[0] + 1

        _KERNELS = (record_pre, record_post, advance)
    return _KERNELS


def recording_sim():
    """The recording subclass of robokit.sim.MjWarpSim (``RecordingSim(model, nworld, ..., record=8, capacity=4096)``)."""
    import mujoco
    import warp as wp
    from robokit.sim import MjWarpSim

    class RecordingSim(MjWarpSim):
        """MjWarpSim that records the first ``record`` worlds' transitions into a device ring of ``capacity`` steps."""

        def __init__(self, *args, record: int = 8, capacity: int = 4096, **kwargs):
            super().__init__(*args, **kwargs)
            self.record_request, self.capacity = record, capacity
            self.record = min(record, self.nworld)
            m = self.mjm
            self.width = 2 * m.nq + 2 * m.nv + m.nu + 1  # qpos, qvel, ctrl, dt, qpos', qvel'
            with wp.ScopedDevice(self.device):
                self.cursor = wp.zeros(1, dtype=int)
                self.buf = wp.zeros((capacity, self.record, self.width), dtype=float)
            self.drained = 0  # the cursor's value at the last drain

        def resized(self, nworld: int) -> "RecordingSim":
            scenes = self._args["model"]
            n = nworld if isinstance(scenes, mujoco.MjModel) else [nworld // len(scenes)] * len(scenes)
            return type(self)(nworld=n, record=self.record_request, capacity=self.capacity, **self._args)  # a subclass stays itself

        def step(self):
            pre, post, advance = _kernels()
            m, d = self.mjm, self._d
            with wp.ScopedDevice(self.device):
                wp.launch(pre, dim=self.record, inputs=[d.qpos, d.qvel, d.ctrl, self._m.opt.timestep, self.cursor, self.substeps], outputs=[self.buf])
            super().step()
            with wp.ScopedDevice(self.device):
                wp.launch(post, dim=self.record, inputs=[d.qpos, d.qvel, self.cursor, m.nq + m.nv + m.nu + 1], outputs=[self.buf])
                wp.launch(advance, dim=1, inputs=[self.cursor])

        def drain(self) -> dict:
            """The rows written since the last drain, (rows, record, .) per field, oldest first; a host sync."""
            n = int(self.cursor.numpy()[0])
            new = n - self.drained
            if new > self.capacity:
                raise RuntimeError(f"RecordingSim: {new} steps since the last drain, the ring holds {self.capacity}; raise capacity")
            rows = (np.arange(self.drained, n) % self.capacity) if new else np.zeros(0, dtype=int)
            b = self.buf.numpy()[rows]
            self.drained = n
            m = self.mjm
            cut = np.cumsum([m.nq, m.nv, m.nu, 1, m.nq])
            q, v, u, dt, q1, v1 = np.split(b, cut, axis=-1)
            return {"qpos": q, "qvel": v, "ctrl": u, "dt": dt[..., 0], "qpos1": q1, "qvel1": v1}

    return RecordingSim


def recording(rows: dict, dt: float, tol: float = 1e-6) -> dict:
    """The rows of one step length ``dt`` (``RecordingSim.drain``'s, concatenated) as a recording of ``wmbench/fit_mjcf.py``:
    qpos and qvel (T + 1, K, .), ctrl (T, K, nu), reset (T, K) where the next row does not start from this row's end state
    (or the step length changes), and dt. Rows of other step lengths are left out; none of this length is an error."""
    keep = np.flatnonzero(np.abs(rows["dt"][:, 0] - dt) < 1e-9)
    if len(keep) == 0:
        raise ValueError(f"recording: no rows of step length {dt}")
    q, v, u, q1, v1 = (rows[k][keep] for k in ("qpos", "qvel", "ctrl", "qpos1", "qvel1"))
    follows = np.zeros(q.shape[:2], dtype=bool)  # row t + 1 starts where row t ended, and is the next row of this length
    follows[:-1] = (np.abs(q[1:] - q1[:-1]).max(-1) <= tol) & (np.abs(v[1:] - v1[:-1]).max(-1) <= tol) & (np.diff(keep) == 1)[:, None]
    follows[-1] = True  # the last row's successor is its own end state, appended below
    # the states: each row's start, then the last row's end; a reset after row t where row t + 1 does not follow
    return {"qpos": np.concatenate((q, q1[-1:])), "qvel": np.concatenate((v, v1[-1:])), "ctrl": u, "reset": ~follows,
            "dt": np.float64(dt), "end_qpos": q1, "end_qvel": v1}  # fmt: skip
