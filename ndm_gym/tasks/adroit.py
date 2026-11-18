"""Dense tracking in named native Adroit scalar coordinates.

Object angular errors are wrapped joint-coordinate errors, not a Cartesian
SO(3) orientation metric. Native task success is evaluated independently in
physics. No target-state assignment or object actuation is used in training.
"""
from dataclasses import dataclass

import torch

from ..motion.adroit import DIMS
from .adroit_door import DoorTask


@dataclass(frozen=True)
class AdroitRewardCfg:
    hand_translation_std: float = .02
    hand_rotation_std: float = .3
    object_translation_std: float = .02
    object_rotation_std: float = .3
    nail_std: float = .01
    action_rate_weight: float = .01

    def __post_init__(self):
        import math
        if any(not math.isfinite(x) or x < 0 or (k.endswith("std") and x == 0)
               for k, x in vars(self).items()):
            raise ValueError("Invalid Adroit tracking kernel")


class AdroitTask(DoorTask):
    def __init__(self, reference, ctrl_range, device='cpu', cfg=None, goal_frames=5):
        if type(goal_frames) is not int or goal_frames < 1:
            raise ValueError('Invalid goal horizon')
        self.name = reference.metadata['task']
        self.nq, self.nv, self.action_dim = DIMS[reference.metadata['task']]
        self.obs_dim = self.nq + self.nv + self.action_dim
        self.goal_frame_dim = self.nq+self.nv
        self.goal_frames, self.cfg = goal_frames, cfg or AdroitRewardCfg()
        if self.name != 'hammer' and self.cfg.nail_std != AdroitRewardCfg().nail_std:
            raise ValueError('Nail tracking precision is Hammer-only')
        self.q = torch.tensor(reference.qpos, dtype=torch.float32, device=device)
        self.v = torch.tensor(reference.qvel, dtype=torch.float32, device=device)
        self.dt, self.frames = reference.metadata['control_dt'], reference.frames
        ranges = torch.tensor(ctrl_range, dtype=torch.float32, device=device)
        if ranges.shape != (self.action_dim, 2) or not torch.isfinite(ranges).all() or not (ranges[:, 1] > ranges[:, 0]).all():
            raise ValueError('Invalid actuator ranges')
        self.ctrl_mid, self.ctrl_half = ranges.mean(-1), (ranges[:, 1]-ranges[:, 0])/2
        names, types = reference.metadata['joint_names'], reference.metadata['joint_types']
        self.groups = {}
        for i, (name, typ) in enumerate(zip(names, types)):
            group = ('nail' if name == 'nail_dir' else
                     ('object_' if name.startswith('OBJ') else 'hand_') + ('translation' if typ == 2 else 'rotation'))
            self.groups.setdefault(group, []).append(i)
        self.periodic = torch.tensor(reference.metadata['joint_periodic'], device=device, dtype=torch.bool)
        self.tracking_widths = tuple(getattr(self.cfg, name+'_std') for name in self.groups)

    @property
    def tracking_widths(self):
        return self._tracking_widths

    @tracking_widths.setter
    def tracking_widths(self, value):
        self._tracking_widths = value

    def identity(self):
        return dict(task=self.name, groups=self.groups, periodic=self.periodic.cpu().tolist())

    def delta(self, q, frame):
        d = q-self.q[frame]
        return torch.where(self.periodic, torch.atan2(torch.sin(d), torch.cos(d)), d)

    def goal(self, state, frame, goal_frames=None):
        parts = []
        for k in range(1, (self.goal_frames if goal_frames is None else goal_frames)+1):
            f = (frame+k).clamp(max=self.frames-1)
            parts.extend((-self.delta(state[0], f), .1*(self.v[f]-state[1])))
        return torch.cat(parts, -1)

    def tracking_terms(self, state, frame):
        d = self.delta(state[0], frame)
        return torch.stack([d[..., ids].abs().mean(-1) for ids in self.groups.values()], -1)

    def errors(self, state, frame):
        d = self.delta(state[0], frame)
        return {name: d[..., ids].square().mean(-1).sqrt() for name, ids in self.groups.items()}

    def reward(self, state, frame, action, a_prev):
        d = self.delta(state[0], frame)
        terms = [torch.exp(-d[..., ids].square().mean(-1)/width**2)
                 for ids, width in zip(self.groups.values(), self.tracking_widths)]
        return (torch.stack(terms).sum(0)-self.cfg.action_rate_weight*(action-a_prev).square().mean(-1))*self.dt

    def opened(self, state):
        raise NotImplementedError('Adroit success requires the native task-specific physical predicate')


TRAINING_HELP = '--nail-tracking-std METRES: Hammer-only reward width; changes training objective; default .01 preserves v1. Evaluation uses fixed v1 widths.'


def training_arguments(argv=None):
    """Opt-in reward precision; leave all shared runner arguments untouched."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--nail-tracking-std', type=float, default=.01,
                   help='Hammer nail tracking width in metres; changes training reward, default .01 preserves v1')
    a, remaining = p.parse_known_args(argv)
    return AdroitRewardCfg(nail_std=a.nail_tracking_std), remaining
