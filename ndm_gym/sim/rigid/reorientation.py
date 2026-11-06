"""MuJoCo reference simulator for Allegro reorientation (no Isaac Lab needed).

The external Menagerie model is never redistributed. This serial CPU reference
backend is intended for correctness checks and small experiments.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from ...tasks.reorientation import GoalTracker, ReorientationCfg, orientation_error, reward, sample_goals


class ReorientationSim:
    """Fixed palm, 16 absolute position targets, one freely moving 5 cm cube.

    State: joint positions/velocities (32), object position (3), wxyz (4),
    linear/angular velocity (6). Observation appends target wxyz (4).
    No automatic reset: the returned state always belongs to the transition.
    """

    def __init__(self, model_dir, seed=0, cfg=None):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg or ReorientationCfg()
        root = ET.parse(Path(model_dir) / 'right_hand.xml').getroot()
        root.find('compiler').set('meshdir', str(Path(model_dir).resolve() / 'assets'))
        root.find('option').set('timestep', '0.002')
        world = root.find('worldbody')
        obj = ET.SubElement(world, 'body', name='reorientation_object', pos='0 0 0.1')
        ET.SubElement(obj, 'freejoint', name='object_free')
        ET.SubElement(obj, 'geom', type='box', size='.025 .025 .025', mass='.05',
                      friction='1 .005 .0001', condim='4', rgba='.8 .3 .1 1')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        if self.model.nu != 16 or self.model.nq != 23 or self.model.nv != 22:
            raise ValueError('expected the pinned 16-actuator Allegro model plus one free object')
        self.lo, self.hi = self.model.actuator_ctrlrange.T.copy()
        self.rng = torch.Generator().manual_seed(seed)
        self.tracker = GoalTracker(1, self.cfg)
        self.anchor = np.array([0., 0., .1])
        self.reset()

    def state(self):
        return torch.from_numpy(np.concatenate((self.data.qpos[:16], self.data.qvel[:16],
                                                self.data.qpos[16:], self.data.qvel[16:])).astype(np.float32)).unsqueeze(0)

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:16] = (self.lo + self.hi) / 2
        self.data.ctrl[:] = self.data.qpos[:16]
        self.data.qpos[16:19] = self.anchor
        self.data.qpos[19:23] = [1, 0, 0, 0]
        self.mj.mj_forward(self.model, self.data)
        self.goal = sample_goals(1, generator=self.rng)
        self.tracker.reset([0])
        return self.state()

    def step(self, action):
        a = np.asarray(torch.as_tensor(action).detach().cpu(), dtype=np.float64).reshape(-1)
        if a.shape != (16,) or not np.isfinite(a).all():
            raise ValueError('action must contain 16 finite values')
        a = np.clip(a, -1, 1)
        self.data.ctrl[:] = self.lo + (a + 1) * .5 * (self.hi - self.lo)
        self.mj.mj_step(self.model, self.data, nstep=10)
        state = self.state()
        error = orientation_error(state[:, 35:39], self.goal)
        distance = (state[:, 32:35] - torch.tensor(self.anchor, dtype=state.dtype)).norm(dim=-1)
        distance = torch.where(torch.isfinite(state).all(-1), distance, torch.full_like(distance, float("inf")))
        info = self.tracker.step(error, distance)
        info.update(angle=error, distance=distance)
        score = reward(state[:, 35:39], self.goal, torch.tensor(a).unsqueeze(0), self.cfg)
        score = torch.where(info['dropped'], -torch.ones_like(score), score)
        return state, score, info
