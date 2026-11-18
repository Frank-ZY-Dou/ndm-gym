"""Manipulation demonstration tracking: Panda arms with their grippers (robots.robomimic_scenes) follow a robomimic
demonstration, the arms, the grippers and the objects they handle.

State tuple ``S = (qpos, qvel)``: the scene's whole generalized state, MuJoCo's own, the arms', the grippers' and every free
object's (a free joint's angular velocity in its body's frame, as MuJoCo keeps it); flat, their concatenation. Observation,
goal, reward and termination are analytic functions of the state, its forward kinematics ``fk`` (the grippers' keypoints,
``SceneKinematics``, computed once per state) and the reference frame, shared by the simulator's transitions and the world
model's imagined rollouts. The arms stand on fixed bases, so world-frame positions are what the policy reads.

The reference is one or more demonstrations of one scene at their control rate (20 Hz), one after the other, each with its
last state held for ``HOLD`` frames (at rest): a step taken at frame ``f`` tracks frame ``f + 1`` of the same demonstration,
and imagined rollouts stay on a demonstration's last frame once they reach it (``RobomimicReference.end``). The action is an
offset of each actuator's position target about the reference's joint position at the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from ..core.rotations import mat_to_6d, quat_to_mat, rot_angle_sq
from ..robots.robomimic_scenes import (BIN_FLOOR, BIN_HALF_XY, BIN_TOP, KEYPOINT_BODIES, LIFT_HEIGHT, LIFT_TABLE_TOP, SQUARE_TABLE_TOP, SQUARE_XY,
                                       SQUARE_Z, RobomimicScene)  # fmt: skip
from ..worldmodels.mjcf import MjcfLayout, MjcfWorldModel

CONTROL_DT = 0.05  # robosuite's control_freq 20
HOLD = 10  # frames of the last state appended to a demonstration (0.5 s)


def flatten_state(S) -> torch.Tensor:
    return torch.cat(S, -1)


def unflatten_state(x: torch.Tensor, nq: int):
    return x[..., :nq], x[..., nq:]


class SceneKinematics(nn.Module):
    """The keypoint bodies' world positions (..., K, 3) from qpos: MjcfWorldModel's forward kinematics (MuJoCo's convention,
    body by body, which torch.compile fuses) on the scene's layout, without its network."""

    fused = True
    _init_kinematics = MjcfWorldModel._init_kinematics
    _fk_lists = MjcfWorldModel._fk_lists
    body_poses = MjcfWorldModel.body_poses

    def __init__(self, lay: MjcfLayout, bodies: tuple[int, ...]):
        super().__init__()
        self._init_kinematics(lay)
        self.register_buffer("bodies", torch.tensor(bodies, dtype=torch.long), persistent=False)

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        P, _ = self.body_poses(qpos)
        return P[..., self.bodies, :]


class RobomimicReference(NamedTuple):
    """Demonstrations at the control rate, (T, ...) per field: the state, the actuators' joint positions (the servos'
    reference targets), the keypoints, the tracked objects' positions and rotations, and the first and the last frame of each
    frame's demonstration."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    keypoints: torch.Tensor
    obj_p: torch.Tensor
    obj_R: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor

    @property
    def frames(self) -> int:
        return self.qpos.shape[0]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f + 1, self.end[f])

    def ahead(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """The frame ``k`` steps after ``f``, at most its demonstration's last."""
        return torch.minimum(f + k, self.end[f])


def object_poses(scene: RobomimicScene, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The tracked objects' positions (..., O, 3) and rotation matrices (..., O, 3, 3)."""
    p = torch.stack([qpos[..., q : q + 3] for q, _ in scene.objects], -2)
    R = quat_to_mat(torch.stack([qpos[..., q + 3 : q + 7] for q, _ in scene.objects], -2))
    return p, R


def build_reference(qpos, qvel, scene: RobomimicScene, kin: SceneKinematics, device: str) -> RobomimicReference:
    """The reference of demonstrations' states, (T, nq) and (T, nv) arrays or lists of them (one per demonstration, one after
    the other), each demonstration's last state held for ``HOLD`` frames at rest."""
    if not isinstance(qpos, (list, tuple)):
        qpos, qvel = [qpos], [qvel]
    qs, vs, starts, ends, offset = [], [], [], [], 0
    for q, v in zip(qpos, qvel, strict=True):
        q, v = torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(v, dtype=torch.float32)
        qs.append(torch.cat((q, q[-1:].expand(HOLD, -1)), 0))
        vs.append(torch.cat((v, torch.zeros(HOLD, v.shape[-1])), 0))
        n = len(qs[-1])
        starts.append(torch.full((n,), offset)), ends.append(torch.full((n,), offset + n - 1))
        offset += n
    qpos, qvel = torch.cat(qs).to(device), torch.cat(vs).to(device)
    ctrl = qpos[:, list(scene.act_q)]
    obj_p, obj_R = object_poses(scene, qpos)
    with torch.no_grad():
        keypoints = kin(qpos)
    return RobomimicReference(qpos, qvel, ctrl, keypoints, obj_p, obj_R, torch.cat(starts).to(device), torch.cat(ends).to(device))


@dataclass
class RewardCfg:
    keypoint_std: float = 0.04  # m
    joint_std: float = 0.4  # rad (the arms' joints, RMS)
    object_pos_std: float = 0.04  # m
    object_ori_std: float = 0.5  # rad
    w_keypoint: float = 1.0
    w_joint: float = 0.5
    w_object_pos: float = 2.0
    w_object_ori: float = 1.0
    w_action_rate: float = -0.05
    term_object: float = 0.15  # m, a tracked object's distance from its reference
    term_keypoint: float = 0.2  # m, a keypoint's

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("keypoint_std", "joint_std", "object_pos_std", "object_ori_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class RobomimicTask:
    def __init__(self, scene: RobomimicScene, kin: SceneKinematics, ref: RobomimicReference, cfg: RewardCfg = RewardCfg()):
        self.scene, self.kin, self.ref, self.cfg = scene, kin, ref, cfg
        dev = ref.qpos.device
        long = lambda x: torch.tensor(x, dtype=torch.long, device=dev)  # noqa: E731
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
        self.nq, self.nv, self.nu = scene.nq, scene.nv, scene.nu
        self.act_q, self.act_d = long(scene.act_q), long(scene.act_d)
        self.u_lo, self.u_hi, self.scale = f32(scene.act_lo), f32(scene.act_hi), f32(scene.act_scale)
        self.arm = long([i for i, a in enumerate(scene.arm) if a])  # the actuators of arm joints, and of gripper fingers
        self.fingers = long([i for i, a in enumerate(scene.arm) if not a])
        self.obj_d = [d for _, d in scene.objects]
        self.success_xy = f32(scene.success_xy)  # (on the device: success() runs inside the collection's CUDA graph)
        self.num_keypoints, self.num_objects, self.num_arms = len(scene.keypoint_bodies), len(scene.objects), scene.num_arms
        self.state_dim = scene.nq + scene.nv
        kps, objs, arms, acts = self.num_keypoints, self.num_objects, self.num_arms, self.nu
        self.obs_dim = acts + acts + 3 * kps + objs * (3 * arms + 6 + 3 + 3) + acts
        self.goal_frame_dim = acts + objs * (3 + 6) + 3 * kps
        self.max_step_reward = (cfg.w_keypoint + cfg.w_joint + cfg.w_object_pos + cfg.w_object_ori) * CONTROL_DT

    def fk(self, qpos: torch.Tensor) -> torch.Tensor:
        return self.kin(qpos)

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """The actuators' position targets for policy output ``a`` in [-1, 1] at frame ``f``: offsets about the next frame's
        joint positions."""
        return torch.clamp(self.ref.ctrl[self.ref.next_frame(f)] + self.scale * a, self.u_lo, self.u_hi)

    def _objects(self, S):
        qpos, qvel = S
        p, R = object_poses(self.scene, qpos)
        v = torch.stack([qvel[..., d : d + 3] for d in self.obj_d], -2)
        w = torch.stack([qvel[..., d + 3 : d + 6] for d in self.obj_d], -2)  # in the object's frame
        return p, R, v, w

    def obs(self, S, fk: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The actuated joints and their velocities, the keypoints, each tracked object about each arm's end effector, its
        rotation and velocities, the previous action."""
        qpos, qvel = S
        p, R, v, w = self._objects(S)
        eef = fk[..., 0 :: len(KEYPOINT_BODIES), :]  # each gripper's end effector, its first keypoint
        rel =(p.unsqueeze(-2) - eef.unsqueeze(-3)).flatten(-3)  # (..., O * A * 3)
        return torch.cat((qpos[..., self.act_q], 0.1 * qvel[..., self.act_d], fk.flatten(-2), rel, mat_to_6d(R).flatten(-2), v.flatten(-2),
                          0.1 * w.flatten(-2), a_prev), -1)  # fmt: skip

    def goal(self, S, fk: torch.Tensor, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """The next ``goal_frames`` reference frames about the state: the actuated joints' offsets, the tracked objects'
        offsets and relative rotations, the keypoints' offsets."""
        qpos, _ = S
        p, R = object_poses(self.scene, qpos)
        Rt, q, ref = R.transpose(-1, -2), qpos[..., self.act_q], self.ref
        parts = []
        for k in range(1, goal_frames + 1):
            g = ref.ahead(f, k)
            parts += [ref.ctrl[g] - q, (ref.obj_p[g] - p).flatten(-2), mat_to_6d(Rt @ ref.obj_R[g]).flatten(-2), (ref.keypoints[g] - fk).flatten(-2)]
        return torch.cat(parts, -1)

    def errors(self, S, fk: torch.Tensor, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: keypoints (..., K) and tracked objects' positions (..., O) in m, the arms'
        joints (rad, RMS), the grippers' fingers (m, RMS) and the objects' squared rotation angles (..., O)."""
        qpos, _ = S
        p, R = object_poses(self.scene, qpos)
        dq = qpos[..., self.act_q] - self.ref.ctrl[f]
        return {
            "keypoint": torch.linalg.vector_norm(fk - self.ref.keypoints[f], dim=-1),
            "joint": dq[..., self.arm].pow(2).mean(-1).sqrt(),
            "finger": dq[..., self.fingers].pow(2).mean(-1).sqrt(),
            "object_pos": torch.linalg.vector_norm(p - self.ref.obj_p[f], dim=-1),
            "object_ori_sq": rot_angle_sq(R, self.ref.obj_R[f]),
        }

    def reward(self, S, fk: torch.Tensor, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev``."""
        c, e = self.cfg, self.errors(S, fk, f)
        r = (
            c.w_keypoint * torch.exp(-e["keypoint"].pow(2).mean(-1) / c.keypoint_std**2)
            + c.w_joint * torch.exp(-e["joint"].pow(2) / c.joint_std**2)
            + c.w_object_pos * torch.exp(-e["object_pos"].pow(2) / c.object_pos_std**2).mean(-1)
            + c.w_object_ori * torch.exp(-e["object_ori_sq"] / c.object_ori_std**2).mean(-1)
            + c.w_action_rate * (a - a_prev).pow(2).mean(-1)
        )
        return r * CONTROL_DT

    def termination(self, S, fk: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        e = self.errors(S, fk, f)
        return (e["object_pos"].amax(-1) > self.cfg.term_object) | (e["keypoint"].amax(-1) > self.cfg.term_keypoint)

    def soft_termination(self, S, fk: torch.Tensor, f: torch.Tensor, width: float = 0.015) -> torch.Tensor:
        """The termination's probability with the thresholds as sigmoids, differentiable in the state."""
        e = self.errors(S, fk, f)
        alive = (1 - torch.sigmoid((e["object_pos"] - self.cfg.term_object) / width)).prod(-1) * (
            1 - torch.sigmoid((e["keypoint"] - self.cfg.term_keypoint) / width)).prod(-1)  # fmt: skip
        return 1 - alive

    def success(self, S) -> torch.Tensor:
        """robosuite's success test of the task as a function of the state (bool, ...)."""
        p, _ = object_poses(self.scene, S[0])
        kind = self.scene.success_kind
        if kind == "lift":
            return p[..., 0, 2] > LIFT_TABLE_TOP + LIFT_HEIGHT
        if kind == "square":
            return ((p[..., 0, :2] - self.success_xy).abs() < SQUARE_XY).all(-1) & (p[..., 0, 2] < SQUARE_TABLE_TOP + SQUARE_Z)
        if kind == "transport":  # the payload and the trash inside their bins (robots.robomimic_scenes.BIN_*); the lid has none
            ok = torch.ones(p.shape[:-2], dtype=torch.bool, device=p.device)
            for o, (q, _) in enumerate(self.scene.bins):
                c, R = S[0][..., q : q + 3], quat_to_mat(S[0][..., q + 3 : q + 7])
                rel = ((p[..., o, :] - c).unsqueeze(-2) @ R).squeeze(-2)  # in the bin's frame
                ok = ok & (rel[..., :2].abs() < BIN_HALF_XY).all(-1) & (rel[..., 2] > BIN_FLOOR) & (rel[..., 2] < BIN_TOP)
            return ok
        raise NotImplementedError(f"success of {kind!r}")

    def critic_features(self, S, fk: torch.Tensor, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, fk, a_prev), self.goal(S, fk, f, goal_frames)), -1)
