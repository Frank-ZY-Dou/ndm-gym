"""Native Adroit hand-only actuation and original task success predicates."""
import numpy as np

from ...motion.adroit import DIMS
from .adroit_door import DoorPhysics


def scene_identity(model):
    return {k: getattr(model, k).tolist() for k in ('body_pos', 'body_quat', 'site_pos', 'site_quat')}


def success(model, data, task):
    """Gymnasium-Robotics 1.3.1 predicates, independent of tracking reward."""
    import mujoco

    def site(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError('Missing task site: '+name)
        return data.site_xpos[i]

    def body(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError('Missing task body: '+name)
        return data.xpos[i]

    if task == 'pen':
        obj = site('object_top') - site('object_bottom')
        target = site('target_top') - site('target_bottom')
        cosine = np.dot(obj, target) / (np.linalg.norm(obj)*np.linalg.norm(target))
        return bool(np.linalg.norm(body('Object')-site('eps_ball')) < .075 and cosine > .95)
    if task == 'hammer':
        return bool(np.linalg.norm(site('S_target')-site('nail_goal')) < .01)
    if task == 'relocate':
        return bool(np.linalg.norm(body('Object')-site('target')) < .1)
    raise ValueError('Unknown Adroit task')


class AdroitPhysics(DoorPhysics):
    def __init__(self, model, reference):
        import mujoco
        self.model, self.reference = model, reference
        m = reference.metadata
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
        if ((model.nq, model.nv, model.nu) != DIMS[m['task']] or names != m['joint_names']
                or model.jnt_type.tolist() != m['joint_types']):
            raise ValueError('Native task layout mismatch')
        if (not np.array_equal(model.actuator_trnid[:, 0], np.arange(model.nu))
                or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)):
            raise ValueError('Expected native hand-only joint actuators')
        if any(not np.allclose(getattr(model, k), np.asarray(v).reshape(getattr(model, k).shape), rtol=0, atol=1e-12)
               for k, v in m['scene'].items()):
            raise ValueError('Reference scene mismatch')
        self.frame_skip = m['frame_skip']
        if not np.isclose(model.opt.timestep*self.frame_skip, m['control_dt'], rtol=0, atol=1e-12):
            raise ValueError('Control timestep mismatch')
        ranges = model.actuator_ctrlrange
        if not np.isfinite(ranges).all() or not (ranges[:, 1] > ranges[:, 0]).all():
            raise ValueError('Invalid actuator ranges')
        self.mid, self.half = ranges.mean(-1), np.diff(ranges, axis=-1)[:, 0]/2
        self.data = mujoco.MjData(model)
        self.reset()

    def reset(self, qpos_offset=None, qvel_offset=None):
        import mujoco
        offsets = []
        for value, dim in ((qpos_offset, self.model.nq), (qvel_offset, self.model.nv)):
            x = np.zeros(dim) if value is None else np.asarray(value, dtype=float)
            if x.shape != (dim,) or not np.isfinite(x).all():
                raise ValueError('Invalid evaluation offset')
            offsets.append(x)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.reference.qpos[0]+offsets[0]
        self.data.qvel[:] = self.reference.qvel[0]+offsets[1]
        mujoco.mj_forward(self.model, self.data)
        self.frame, self.done = 0, False
        return self.state()

    def step(self, action):
        import mujoco
        a = np.asarray(action, dtype=float)
        if a.shape != (self.model.nu,) or not np.isfinite(a).all():
            raise ValueError('Invalid native action')
        if self.done:
            raise RuntimeError('Reset required after terminal state')
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError('External forces are forbidden')
        self.data.ctrl[:] = self.mid + np.clip(a, -1, 1)*self.half
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.done = True
                raise FloatingPointError('Nonfinite native Adroit transition')
        self.frame += 1
        self.done = self.frame >= self.reference.frames-1
        return self.state(), self.done
