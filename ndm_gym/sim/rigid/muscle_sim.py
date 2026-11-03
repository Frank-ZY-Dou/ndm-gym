"""MuscleVAE's muscle-actuated character in N environments on Newton (docs/musclevae_design.md §2): the physics baseline
of the muscle character's throughput benchmark (benchmarks/ndm_speed) and the source of its NDM datasets.

The character (data/musclevae/character.xml, :mod:`ndm_gym.robots.musclevae_char`) is replicated once per environment
in Newton directly, not through Isaac Lab (whose pinned version cannot load this character's spatial muscles, and the
muscle forces are applied here anyway), on a ground plane, and stepped by Newton's MuJoCo-Warp solver at 120 Hz. A
control step (20 Hz, :meth:`MuscleSim.step`) is six physics substeps, each preceded by MuscleVAE's muscle substep
(ENV:373-381, :meth:`ndm_gym.robots.muscles.MuscleModel.substep`): anchor forces from the body poses, the previous
substep's lengths, the PD targets and the fatigue state, summed into one wrench per body and written to Newton's
per-body external wrenches (``State.body_f``: the force in the world frame at the body's center of mass, which is its
origin here, and the torque in the world frame, MuJoCo's ``xfrc_applied``); the fatigue state advances with the
activation. On CUDA a control step (the action's PD targets; six muscle substeps in torch, through torch.compile,
which fuses the model's gathers and elementwise operations; six solver steps in warp; the done flags; the reference's
next frames) is one CUDA graph, and so is a random reset: torch captures, warp joins the capture (the solver's
convergence loop becomes a conditional node), and the random draws come from a generator registered with the graphs.
No Python runs per substep and nothing synchronizes with the host; transitions are written straight into contiguous
device buffers. A step runs about 1,650 to 1,830 GPU kernels: the solver's loop runs until the slowest world converges,
12.4 to 14.5 iterations per substep, about 14 kernels an iteration.

Not run-to-run deterministic on the GPU: MuJoCo-Warp's solver accumulates with atomics, so two runs with the same seed
differ (by 3.4e-4 rad/s after one step and about 0.5 m after 20), and Newton's DeterministicMode.RUN_TO_RUN fails in
warp's sort-reduce (from MuJoCo-Warp's _update_gradient) at 64 to 4,096 environments. The muscle model itself is
deterministic (ndm_gym.robots.muscles). Equivalence checks of the simulator therefore run on the CPU or compare
discrete outcomes (reset frames, fatigue draws, reference frames, terminations).

Physics, against MuscleVAE's ODE build (ModifyODE in double precision, its damped step):
* 1/120 s steps, MuJoCo-Warp's Newton solver at MuJoCo's default iterations and tolerance, for their exact (Dantzig)
  LCP. Joint damping, 10 on every degree of freedom (``joint_damping``):
  - "implicit" (the default): MuJoCo's, implicitfast integration, which applies the damping implicitly after the
    constraint solve, so contacts and limits see the bodies' own inertia;
  - "ode": their damped step (dWorldDampedStep, DS:26-180, 1115-1160), which damps each joint's relative angular
    velocity w_rel = w_child - w_parent: (I + h D) w' = I w + h (tau + constraint torques), the constraints solved
    with I + h D. In joint coordinates that is an armature h D = 1/12 on every degree of freedom of the joints,
    MuJoCo's damping off, and the damping force of the substep's start as an explicit generalized force (Newton's
    Control.joint_f, written before each solver step): -D times the joint velocity for the hinges and ball joints,
    whose velocities are w_rel's components, and -D axis_i . w_rel for each of an Euler-hinge ball joint's three hinges
    (axis_i in the world, w_rel from the bodies' angular velocities), since w_rel is not the vector of its Euler rates
    (|w_rel|^2 = sum of the squared rates - 2 sin(a1) a0' a2', large at the hips' and shoulders' |a1| of 0.9 to 1.5
    rad). The armature stays the constant h D, an approximation of h D J^T J (J the Euler rates' map to w_rel), exact
    at a1 = 0 and for hinges and ball joints. The difference from "implicit" is large for the feet and toes, whose own
    inertia (1e-4 to 1e-2) is far below h D.
* Contacts with the ground only (self-collision off, as theirs): friction 0.8 (ODE takes the smaller of the two geoms',
  MuJoCo the larger; 0.8 for the character and the floor in both), a pyramidal friction cone (their dContactApprox1
  pyramid), sliding friction only (condim 3: ODE has no torsional or rolling friction), no restitution (their contacts
  have no bounce mode), contacts only between overlapping geoms (margin and gap 0, as dCollide).
* Contact and hinge-limit softness: ODE's constraints are hard (global CFM 1e-10) and remove the fraction ERP = 0.2 of a
  penetration per step (target velocity ERP depth / h). MuJoCo's soft constraints take that form with the direct
  reference parameters solref = (-ERP / h^2, -1 / h) and the impedance solimp = (0.99, 0.99): the constraint acceleration
  is -(v / h) - (ERP / h^2) r up to 1 % of the unconstrained one, so the next velocity is ERP depth / h. Set on every
  geom, the floor's included, and on the hinge limits (ODE's stops use the same ERP and CFM).
* Ball joints (``ball_joints``): their simulator limits each ball joint's three Euler angles separately (an ODE AMotor
  in Euler mode, BE:17727-17750, CFG's character UseAngleLimit), which MuJoCo's ball joints (cone limits only) cannot.
  - "ball" (the default): MuJoCo ball joints, the Euler limits not enforced; how far a pose exceeds them is
    :meth:`MuscleSim.ball_limit_excess`;
  - "euler": each ball joint as three hinges at its anchor, about their z, y and x axes (our x, z, y) in that order,
    which compose to Rot(z, a2) Rot(y, a1) Rot(x, a0), the AMotor's decomposition (z fixed in the parent, x in the
    child, musclevae_char.ball_euler_angles), with their Euler limits as hinge limits (the limit parameters below); the
    coordinates are those angles (Newton imports the three hinges as a D6 joint). The decomposition is singular at
    a1 = +-pi / 2, which the hips' and shoulders' +-2 rad allow: the three hinges then lose a degree of freedom, their
    ball joint (with its AMotor) does not. The references reach |a1| = 1.56 rad (Head) and 1.53 (LeftArm); a few
    reset frames lie within GIMBAL_MARGIN of the singularity, and many start beyond their Euler limits (the Head's in
    over a third), which the hinge limits then push back. ``reset_exclude`` ("gimbal", "limits", off by default) drops
    those frames from the resets; :meth:`MuscleSim.reset_frame_stats` counts them.
* Single precision (MuJoCo-Warp) where theirs is double.
Contact buffers: 64 contacts and 264 constraint rows per world (4 rows per contact of the pyramidal cone, 8 hinge
limits; 42 more limits with the ball joints as hinges), above what the character's 25 geoms make on the floor (one per
sphere, two per capsule, four per box lying flat: 61); :meth:`MuscleSim.check_solver` raises if a buffer overflowed
anyway.

The environment, as MuscleVAE's (ENV) with its config.yml (CFG):
* Reset (ENV:282-329) of an environment to a reference frame, drawn uniformly from the frames their reset draws from
  (IC:23-31: neither a clip's first frame nor its last 25; their training reweights them by returns, ENV:220-235, the
  algorithm's business), unless given: the clip's pose in that frame (root pose and joint positions on our skeleton,
  ndm_gym.motion.sources.musclevae) with velocities by backward differences at 20 Hz (their dataset's,
  BE:14001-14067), raised to 0.1 mm above the floor if a geom is below it (ENV:251-256); the muscle
  lengths 1/120 s earlier, for the first substep's rates, from the pose 5/6 of the way from the previous frame
  (ENV:258-270). Theirs interpolates each body's position linearly and orientation by slerp (MU:96-154), which dislocates
  the joints of fast limbs (forearm muscles then come out up to 2 cm shorter than in both frames); here the joint
  positions are interpolated (rotations by slerp), on our skeleton like the reset pose. The fatigue state is drawn as
  theirs (:meth:`ndm_gym.robots.muscles.MuscleModel.random_fatigue`): MA is about 0.5 in all but the first 0.1 s of the
  curve, so the 3CC-r lower bound makes every muscle start at an activation of at least 0.29, a co-contraction that
  decays by a factor 0.58 per substep (in theirs as here). The reference is at the next frame.
* Reference frame (ENV:332-354, play_random_mode 1): advances by one frame per control step, and jumps to a random
  frame (as a reset's) at a clip's last frame and every ``random_count`` steps (CFG env_random_count 96).
* Termination (ENV:193-218, CFG): a step counts as off when the head's height differs from the reference's (their
  dataset's placement of the head) by more than 0.5 m, and an off count decreases by one per step on track; terminated
  when the state is not finite or its observation exceeds 50 in magnitude (MU:204-234, with the state's velocities by
  backward differences over the control step as their env_recompute_velocity, MU:39-50), or after 26 steps when the
  off count reaches 20; truncated after 512 steps.
* Data policy (:meth:`MuscleSim.data_policy`): muscle-space PD towards the reference pose, the action
  a = l_ref / l_Tpose - 1 with l_ref the muscle model's lengths in the reference frame the step tracks, plus Gaussian
  noise. Its PD error is the tracking error alone, and kp is about f0max per 1.5 m, so its forces are a small fraction
  of the muscles' strength: open loop and nearly passive, the character falls within about half a second.

References: ENV = MuscleVAECore/Env/muscle_env.py, MU = MuscleVAECore/Utils/motion_utils.py, IC =
MuscleVAECore/Utils/index_counter.py, BE = ModifyODESrc/VclSimuBackend.pyx, DS = ModifyODESrc/src/dampedstep.cpp and CFG
= Data/NNModel/config.yml of wangchek/MuscleVAE_ @ 54c001c, which has no license: nothing is copied from it.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import NamedTuple

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

from ...core.rotations import quat_slerp, quat_to_mat, so3_log
from ...motion.sources import musclevae as clip_source
from ...robots import musclevae_char as mc
from ...robots.muscles import MuscleModel

SIM_DT, CONTROL_DT = 1.0 / mc.SIM_HZ, 1.0 / mc.CONTROL_HZ
GROUND_FRICTION = 0.8  # their floor's and the character's
ODE_ERP = 0.2  # ODE's default global ERP (CFM 1e-10 in double precision)
CONSTRAINT_SOLREF = (-ODE_ERP / SIM_DT**2, -1.0 / SIM_DT)  # direct stiffness and damping
CONSTRAINT_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)
NCONMAX, NJMAX = 64, 264  # contacts and constraint rows per world (4 per contact, the 8 hinge limits)
NJMAX_EULER = NJMAX + 3 * len(mc.BALL_JOINTS)  # and the limits of the ball joints' hinges
NQ, NV = 7 + mc.JOINT_QPOS_DIM, 6 + mc.JOINT_QVEL_DIM  # Newton's coordinates per environment with ball joints: 71 and 56
NQ_EULER = 7 + sum(1 if t == "hinge" else 3 for t in mc.JOINT_TYPES)  # 57, with ball joints as three hinges (velocities: NV)
BALL_JOINT_MODES, JOINT_DAMPING_MODES = ("ball", "euler"), ("implicit", "ode")
EULER_HINGES = (("z", 2, "1 0 0"), ("y", 1, "0 0 1"), ("x", 0, "0 1 0"))  # in order: their axis, its Euler angle, its axis in ours
SOLVER_OPTIONS = {"solver": "newton", "integrator": "implicitfast", "cone": "pyramidal", "nconmax": NCONMAX, "use_mujoco_contacts": True}
# MuscleVAE's environment (CFG)
ERR_THRESHOLD, ERR_LENGTH, MIN_LENGTH, MAX_LENGTH = 0.5, 20, 26, 512
OBS_LIMIT = 50.0
RANDOM_COUNT = 96
RESET_TAIL = 25  # a reset frame keeps this many frames after it in its clip (IC:23-31 with ENV:83's 24 + 1)
RESET_LIFT = 1e-4  # m (ENV:253-255)
PREV_FRACTION = (mc.SUBSTEPS - 1) / mc.SUBSTEPS  # the substep before a frame, between the previous frame and it
GIMBAL_MARGIN = 0.2  # rad: a reset pose this close to an Euler hinge chain's singularity, |a1| = pi / 2, counts as near it
RESET_EXCLUSIONS = ("gimbal", "limits")  # reset frames MuscleSim(reset_exclude=...) can drop

GEOM_DEFAULT = '<geom contype="1" conaffinity="0" condim="3"/>'
JOINT_DEFAULT = f'<joint damping="{mc.JOINT_DAMPING:g}" armature="0"/>'
_fmt = lambda xs: " ".join(repr(float(x)) for x in xs)  # noqa: E731
CONSTRAINT_ATTRS = f'solref="{_fmt(CONSTRAINT_SOLREF)}" solimp="{_fmt(CONSTRAINT_SOLIMP)}"'
LIMIT_ATTRS = f'solreflimit="{_fmt(CONSTRAINT_SOLREF)}" solimplimit="{_fmt(CONSTRAINT_SOLIMP)}"'
FLOOR_MJCF = (f'<mujoco model="floor"><worldbody><geom name="floor" type="plane" size="0 0 1" condim="3" contype="1" conaffinity="1" '
              f'friction="{GROUND_FRICTION!r} 0.005 0.0001" {CONSTRAINT_ATTRS}/></worldbody></mujoco>')  # fmt: skip


def _check_mode(ball_joints: str, joint_damping: str) -> None:
    if ball_joints not in BALL_JOINT_MODES or joint_damping not in JOINT_DAMPING_MODES:
        raise ValueError(f"ball_joints must be one of {BALL_JOINT_MODES} and joint_damping one of {JOINT_DAMPING_MODES}, got {ball_joints!r}, "
                         f"{joint_damping!r}")  # fmt: skip


def character_mjcf(path: str = mc.MJCF_PATH, ball_joints: str = "ball", joint_damping: str = "implicit", char: dict | None = None) -> str:
    """The character's MJCF with the contact and hinge-limit parameters above in its default class; with ``ball_joints``
    "euler" each ball joint as three limited hinges (module docstring; limits from character.json, ``char``), with
    ``joint_damping`` "ode" no damping and the armature h D on every joint."""
    _check_mode(ball_joints, joint_damping)
    with open(path) as f:
        xml = f.read()
    for line, attrs in ((GEOM_DEFAULT, CONSTRAINT_ATTRS), (JOINT_DEFAULT, LIMIT_ATTRS)):
        if xml.count(line) != 1:
            raise ValueError(f"{path}: expected the default {line!r} once (tools/convert_musclevae.py writes it)")
        new = line[:-2] + f" {attrs}/>"
        if line == JOINT_DEFAULT and joint_damping == "ode":
            new = new.replace(f'damping="{mc.JOINT_DAMPING:g}" armature="0"', f'damping="0" armature="{mc.JOINT_DAMPING * SIM_DT!r}"')
        xml = xml.replace(line, new)
    if ball_joints == "euler":
        char = mc.load_character() if char is None else char
        limits = {j["name"]: j["euler_limits"] for j in char["joints"] if j["type"] == "ball"}
        parts = re.split(r'(<joint name="[^"]+" type="ball" pos="[^"]+"/>)', xml)
        names = []
        for i in range(1, len(parts), 2):
            name, pos = re.fullmatch(r'<joint name="([^"]+)" type="ball" pos="([^"]+)"/>', parts[i]).groups()
            lo, hi = limits[name]["lo"], limits[name]["hi"]
            parts[i] = "\n".join(f'<joint name="{name}_{ax}" type="hinge" pos="{pos}" axis="{axis}" range="{lo[k]!r} {hi[k]!r}"/>'
                                 for ax, k, axis in EULER_HINGES)  # fmt: skip
            names.append(name)
        if names != [mc.JOINT_NAMES[j] for j in mc.BALL_JOINTS]:
            raise ValueError(f"{path}: ball joints {names}, expected those of ndm_gym.robots.musclevae_char.BALL_JOINTS")
        xml = "".join(parts)
    return xml


def build_model(num_envs: int, device: str, char: dict, ball_joints: str = "ball", joint_damping: str = "implicit") -> newton.Model:
    """``num_envs`` copies of the character, each in its own world, and a floor shared by all."""
    gravity = tuple(char["gravity"])

    def builder() -> newton.ModelBuilder:
        b = newton.ModelBuilder(gravity=gravity)
        b.rigid_gap = 0.0
        SolverMuJoCo.register_custom_attributes(b)
        return b

    one = builder()
    one.add_mjcf(character_mjcf(ball_joints=ball_joints, joint_damping=joint_damping, char=char), enable_self_collisions=False)
    scene = builder()
    scene.replicate(one, num_envs)
    scene.add_mjcf(FLOOR_MJCF)
    return scene.finalize(device=device)


def check_model(model: newton.Model, num_envs: int, ball_joints: str = "ball", joint_damping: str = "implicit", char: dict | None = None) -> None:
    """The layout this module relies on: bodies and joints in musclevae_char's order in every world, Newton's coordinate
    counts, body origins at the centers of mass (body_f's reference point) and joint frames aligned with the bodies'; the
    ball joints' three hinges (a D6 joint) about their z, y, x axes with their Euler limits; the damping and armature."""
    _check_mode(ball_joints, joint_damping)
    char = mc.load_character() if char is None else char
    NB = mc.NUM_BODIES
    nq = NQ if ball_joints == "ball" else NQ_EULER
    if (model.world_count, model.articulation_count, model.body_count) != (num_envs, num_envs, num_envs * NB):
        raise ValueError(f"expected {num_envs} worlds and articulations of {NB} bodies, got {model.world_count} worlds, "
                         f"{model.articulation_count} articulations, {model.body_count} bodies")  # fmt: skip
    if (model.joint_coord_count, model.joint_dof_count) != (num_envs * nq, num_envs * NV):
        raise ValueError(f"expected {nq} coordinates and {NV} degrees of freedom per world")
    names = [str(label).rsplit("/", 1)[-1] for label in model.body_label[:NB]]
    if tuple(names) != mc.BODY_NAMES:
        raise ValueError(f"Newton's bodies {names} differ from ndm_gym.robots.musclevae_char.BODY_NAMES")
    jt = model.joint_type.numpy()[:NB]
    ball = newton.JointType.BALL if ball_joints == "ball" else newton.JointType.D6
    want = [int(newton.JointType.FREE)] + [int(ball if t == "ball" else newton.JointType.REVOLUTE) for t in mc.JOINT_TYPES]
    if jt.tolist() != want or model.joint_parent.numpy()[1:NB].tolist() != list(mc.BODY_PARENTS[1:]):
        raise ValueError("Newton's joints differ from ndm_gym.robots.musclevae_char.JOINT_TYPES / BODY_PARENTS")
    dof_start = model.joint_qd_start.numpy()[:NB]
    damping, armature = model.joint_damping.numpy()[:NV], model.joint_armature.numpy()[:NV]
    want_damping, want_armature = (mc.JOINT_DAMPING, 0.0) if joint_damping == "implicit" else (0.0, mc.JOINT_DAMPING * SIM_DT)
    if not (np.allclose(damping[6:], want_damping) and np.allclose(armature[6:], want_armature) and not damping[:6].any() and not armature[:6].any()):
        raise ValueError(f"joint damping {np.unique(damping[6:])} and armature {np.unique(armature[6:])}, expected {want_damping} and {want_armature}")
    if ball_joints == "euler":
        axes, lower, upper = model.joint_axis.numpy(), model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        dim = model.joint_dof_dim.numpy()[:NB]
        want_axes = np.array([[float(x) for x in axis.split()] for _, _, axis in EULER_HINGES])
        for j in mc.BALL_JOINTS:
            d, lim = dof_start[j + 1], char["joints"][j]["euler_limits"]
            if tuple(dim[j + 1]) != (0, 3):
                raise ValueError(f"{mc.JOINT_NAMES[j]}: expected a D6 joint of three angular axes, got dof dims {tuple(dim[j + 1])}")
            if not np.allclose(axes[d : d + 3], want_axes):
                raise ValueError(f"{mc.JOINT_NAMES[j]}: hinge axes {axes[d : d + 3].tolist()}, expected their z, y, x axes {want_axes.tolist()}")
            k = [i for _, i, _ in EULER_HINGES]
            if not (np.allclose(lower[d : d + 3], np.array(lim["lo"])[k], atol=1e-6) and np.allclose(upper[d : d + 3], np.array(lim["hi"])[k], atol=1e-6)):
                raise ValueError(f"{mc.JOINT_NAMES[j]}: hinge limits differ from character.json's Euler limits")
    if np.abs(model.body_com.numpy()).max() > 1e-9:
        raise ValueError("body origins must be the centers of mass")
    for k in ("joint_X_p", "joint_X_c"):
        q = getattr(model, k).numpy()[:NB, 3:]
        if np.abs(np.abs(q[:, 3]) - 1.0).max() > 1e-6:
            raise ValueError(f"{k}: joint frames must be aligned with the bodies' (T-pose body frames along the world)")


# ---- reference motion
class Reference(NamedTuple):
    """The reference clips, concatenated frame by frame (F frames), as the simulator uses them."""

    names: tuple[str, ...]
    start: torch.Tensor  # (C,) first frame of each clip
    clip: torch.Tensor  # (F,) clip of each frame
    is_last: torch.Tensor  # (F,) a clip's last frame
    reset_frames: torch.Tensor  # (K,) frames a reset may draw
    joint_q: torch.Tensor  # (F, 71) Newton's coordinates, raised above the floor (lift included); (F, 57) with Euler hinges
    joint_qd: torch.Tensor  # (F, 56) Newton's velocities by backward differences
    lift: torch.Tensor  # (F,) m
    length: torch.Tensor  # (F, M) muscle lengths in the frame's pose
    length_prev: torch.Tensor  # (F, M) muscle lengths 1/120 s before the frame
    head_z: torch.Tensor  # (F,) their dataset's height of the head's center of mass
    ball_euler: torch.Tensor  # (F, 14, 3) the ball joints' Euler angles (musclevae_char.ball_euler_angles), BALL_JOINTS order


def euler_coords(joint_qpos: torch.Tensor, j: int) -> torch.Tensor:
    """The three hinge angles (..., 3) of ball joint ``j`` in the clips' layout, in the hinges' order (EULER_HINGES): their
    Euler angles about their z, y, x axes (musclevae_char.ball_euler_angles)."""
    a = mc.JOINT_QPOS_ADR[j]
    angles = mc.ball_euler_angles(quat_to_mat(joint_qpos[..., a : a + 4]))
    return torch.stack([angles[..., k] for _, k, _ in EULER_HINGES], -1)


def newton_coords(root_pos: torch.Tensor, root_quat: torch.Tensor, joint_qpos: torch.Tensor, ball_joints: str = "ball") -> torch.Tensor:
    """Newton's coordinates (..., 71) of joint positions in the clips' layout: quaternions xyzw instead of wxyz; with
    ``ball_joints`` "euler" (..., 57), each ball joint's three hinge angles (:func:`euler_coords`)."""
    parts = [root_pos, root_quat[..., [1, 2, 3, 0]]]
    for j, a in enumerate(mc.JOINT_QPOS_ADR):
        if mc.JOINT_TYPES[j] != "ball":
            parts.append(joint_qpos[..., a : a + 1])
        elif ball_joints == "ball":
            parts.append(joint_qpos[..., [a + 1, a + 2, a + 3, a]])
        else:
            parts.append(euler_coords(joint_qpos, j))
    return torch.cat(parts, -1)


def backward_velocities(root_pos: torch.Tensor, root_quat: torch.Tensor, joint_qpos: torch.Tensor, fps: float, ball_joints: str = "ball") -> torch.Tensor:
    """Newton's velocities (T, 56) of one clip (T frames) by backward differences, the first frame's those of the second
    (their dataset's, BE:14001-14067): the root's linear velocity and angular velocity in the world frame, each ball
    joint's angular velocity relative to its parent in the parent's frame (dL/dt = [u] L), or with ``ball_joints``
    "euler" its three hinge angles' rates, each hinge's rate."""
    R = quat_to_mat(root_quat)
    parts = [(root_pos[1:] - root_pos[:-1]) * fps, so3_log(R[1:] @ R[:-1].transpose(-1, -2)) * fps]
    for j, a in enumerate(mc.JOINT_QPOS_ADR):
        if mc.JOINT_TYPES[j] != "ball":
            parts.append((joint_qpos[1:, a : a + 1] - joint_qpos[:-1, a : a + 1]) * fps)
        elif ball_joints == "ball":
            L = quat_to_mat(joint_qpos[:, a : a + 4])
            parts.append(so3_log(L[1:] @ L[:-1].transpose(-1, -2)) * fps)
        else:
            e = euler_coords(joint_qpos, j)
            parts.append((e[1:] - e[:-1]) * fps)
    v = torch.cat(parts, -1)
    return torch.cat((v[:1], v), 0)


def interpolate_previous(root_pos: torch.Tensor, root_quat: torch.Tensor, joint_qpos: torch.Tensor):
    """The joint positions of one clip (T frames) 1/120 s before each frame: 5/6 of the way from the previous frame, the
    root's position linearly, rotations by slerp, hinge angles linearly; a clip's first frame is kept."""
    T = root_pos.shape[0]
    prev = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(T - 1)))
    alpha = torch.full((T,), PREV_FRACTION, dtype=root_pos.dtype)
    alpha[0] = 1.0
    joints = []
    for j, a in enumerate(mc.JOINT_QPOS_ADR):
        if mc.JOINT_TYPES[j] == "ball":
            joints.append(quat_slerp(joint_qpos[prev, a : a + 4], joint_qpos[:, a : a + 4], alpha))
        else:
            joints.append(joint_qpos[prev, a : a + 1] + alpha[:, None] * (joint_qpos[:, a : a + 1] - joint_qpos[prev, a : a + 1]))
    return (root_pos[prev] + alpha[:, None] * (root_pos - root_pos[prev]), quat_slerp(root_quat[prev], root_quat, alpha),
            torch.cat(joints, -1))  # fmt: skip


def geom_bottom(body_pos: torch.Tensor, body_rot: torch.Tensor, char: dict) -> torch.Tensor:
    """The lowest height (...) of the character's geoms (spheres, capsules along their z, boxes) in body poses (..., 23, 3)
    and (..., 23, 3, 3): the bottom of their bounding box, as ODE's AABB."""
    lows = []
    for b, body in enumerate(char["bodies"]):
        R, p = body_rot[..., b, :, :], body_pos[..., b, :]
        for g in body["geoms"]:
            gR = R @ quat_to_mat(body_pos.new_tensor(g["quat_wxyz"]))
            z = p[..., 2] + (R[..., 2, :] * body_pos.new_tensor(g["pos"])).sum(-1)
            size = body_pos.new_tensor(g["size"])
            if g["type"] == "sphere":
                ext = size[0].expand(z.shape)
            elif g["type"] == "capsule":
                ext = gR[..., 2, 2].abs() * size[1] + size[0]
            elif g["type"] == "box":
                ext = (gR[..., 2, :].abs() * size).sum(-1)
            else:
                raise ValueError(f"geom {g['name']}: unsupported type {g['type']!r}")
            lows.append(z - ext)
    return torch.stack(lows, -1).amin(-1)


def build_reference(names: tuple[str, ...], muscles: MuscleModel, char: dict, device: torch.device, chunk: int = 2048,
                    ball_joints: str = "ball") -> Reference:  # fmt: skip
    """The reference tables of the named clips (ndm_gym.motion.sources.musclevae), computed in double precision on the
    CPU with the character's forward kinematics and a double-precision copy of the muscle model, stored in single
    precision on ``device``; Newton's coordinates and velocities for ``ball_joints`` ("ball" or "euler")."""
    if not names:
        raise ValueError("no reference clip")
    exact = MuscleModel(muscles.table, muscles.params, "cpu", torch.float64)
    rows = {k: [] for k in ("joint_q", "joint_qd", "lift", "length", "length_prev", "head_z", "ball_euler")}
    start, clip_id, is_last, reset_frames, total = [], [], [], [], 0
    for c, name in enumerate(names):
        path = os.path.join(mc.CLIPS_DIR, f"{name}.npz")
        clip = clip_source.load_clip(path)
        if clip.fps != mc.CONTROL_HZ:
            raise ValueError(f"{path}: {clip.fps} fps, expected the control rate {mc.CONTROL_HZ}")
        T = clip.root_p.shape[0]
        if T < RESET_TAIL + 2:
            raise ValueError(f"{path}: {T} frames, a reset needs at least {RESET_TAIL + 2}")
        root_p, root_q, jq = clip.root_p.double(), clip.root_quat.double(), clip.joints.double()
        pos, rot = mc.forward_kinematics(root_p, root_q, jq, char)
        lift = geom_bottom(pos, rot, char)
        lift = torch.where(lift < 0.0, RESET_LIFT - lift, torch.zeros_like(lift))
        q = newton_coords(root_p, root_q, jq, ball_joints)
        q[:, 2] += lift
        pos_b, rot_b = mc.forward_kinematics(*interpolate_previous(root_p, root_q, jq), char)
        lengths = [torch.cat([exact.lengths(exact.anchor_positions(p[i : i + chunk], r[i : i + chunk])) for i in range(0, T, chunk)])
                   for p, r in ((pos, rot), (pos_b, rot_b))]  # fmt: skip
        body_pos, _ = clip_source.load_placement(path)
        qd = backward_velocities(root_p, root_q, jq, clip.fps, ball_joints)
        euler = torch.stack([mc.ball_euler_angles(quat_to_mat(jq[:, mc.JOINT_QPOS_ADR[j] : mc.JOINT_QPOS_ADR[j] + 4])) for j in mc.BALL_JOINTS], 1)
        for k, v in zip(rows, (q, qd, lift, *lengths, body_pos[:, mc.HEAD, 2].double(), euler)):
            rows[k].append(v)
        start.append(total)
        clip_id.append(torch.full((T,), c, dtype=torch.long))
        last = torch.zeros(T, dtype=torch.bool)
        last[-1] = True
        is_last.append(last)
        reset_frames.append(total + torch.arange(1, T - RESET_TAIL))
        total += T
    f32 = lambda x: torch.cat(x).to(device=device, dtype=torch.float32)  # noqa: E731
    i64 = lambda x: torch.as_tensor(x).to(device)  # noqa: E731
    return Reference(tuple(names), i64(start), i64(torch.cat(clip_id)), i64(torch.cat(is_last)), i64(torch.cat(reset_frames)),
                     *(f32(rows[k]) for k in rows))  # fmt: skip


# ---- the simulator
def _quat_mul_xyzw(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pv, pw, qv, qw = p[..., :3], p[..., 3:], q[..., :3], q[..., 3:]
    return torch.cat((pw * qv + qw * pv + torch.cross(pv, qv, dim=-1), pw * qw - (pv * qv).sum(-1, keepdim=True)), -1)


def done_flags(body_q, body_qd, before, head_ref, off_count, ep_step, xyzw_to_wxyz):
    """MuscleVAE's done flags of a control step (ENV:193-218) from the bodies' poses (N, 23, 7) and velocities (N, 23, 6)
    after it, their poses ``before`` it, the reference's head heights (N,), and the off-track counts and episode steps
    (N,) before it: (terminated, truncated, the updated off-track counts). The state's velocities are backward
    differences over the control step (MU:39-50); its observation (MU:204-234), the positions, velocities and angular
    velocities in the root's frame and the heights."""
    pos, quat, pos0, quat0 = body_q[..., :3], body_q[..., 3:], before[..., :3], before[..., 3:]
    off = (pos[:, mc.HEAD, 2] - head_ref).abs() > ERR_THRESHOLD
    off_count = torch.where(off, off_count + 1, (off_count - 1).clamp(min=0))
    vel = (pos - pos0) / CONTROL_DT
    dq = (quat * torch.sign((quat * quat0).sum(-1, keepdim=True)) - quat0) / CONTROL_DT
    avel = 2.0 * _quat_mul_xyzw(dq, torch.cat((-quat0[..., :3], quat0[..., 3:]), -1))[..., :3]
    root = quat_to_mat(quat[:, 0, xyzw_to_wxyz])[:, None]  # (N, 1, 3, 3)
    local = torch.cat([(v.unsqueeze(-1) * root).sum(-2) for v in (pos - pos[:, :1], vel, avel)], -1)  # R^T v
    obs_max = torch.maximum(local.abs().flatten(1).amax(-1), pos[..., 2].abs().amax(-1))
    finite = torch.isfinite(torch.cat((body_q, vel, avel, body_qd), -1)).flatten(1).all(-1)
    old = ep_step >= MIN_LENGTH
    terminated = ~finite | (obs_max > OBS_LIMIT) | (old & (off_count >= ERR_LENGTH))
    return terminated, ~terminated & old & (ep_step >= MAX_LENGTH), off_count


def next_frames(ref_frame, ep_step, is_last, random_count: int, random_frames):
    """The reference's next frames (ENV:332-354, play_random_mode 1): one frame on, or ``random_frames`` at a clip's end
    (when the next frame would be its last) and every ``random_count`` episode steps (0: never)."""
    nxt = ref_frame + 1
    switch = is_last[nxt]
    if random_count > 0:
        switch = switch | ((ep_step % random_count == 0) & (ep_step != 0))
    return torch.where(switch, random_frames, nxt)


class _Views(NamedTuple):
    """Torch views of one Newton state's arrays, one row per environment."""

    body_q: torch.Tensor  # (N, 23, 7): origin (the center of mass) and quaternion xyzw
    body_qd: torch.Tensor  # (N, 23, 6): center-of-mass velocity and angular velocity, world frame
    body_f: torch.Tensor  # (N, 23, 6): force at the center of mass and torque, world frame
    joint_q: torch.Tensor  # (N, 71)
    joint_qd: torch.Tensor  # (N, 56)


class MuscleSim:
    """``num_envs`` copies of MuscleVAE's character on ``device`` (CUDA, or the CPU for tests), with the muscle model
    ``muscle_model`` ("released" or "paper", ndm_gym.robots.muscles) and the reference clips ``clips`` (all three by
    default). ``graph``: the control step and the random reset as captured CUDA graphs (the default on CUDA; without them
    every solver step synchronizes with the host once per iteration). ``compiled``: the step's torch parts (the muscle
    substep; the PD targets; the done flags and reference frames), each working in place on the buffers, through
    torch.compile, which fuses their gathers, elementwise operations and writes (the default on CUDA; the same operations,
    so the results agree to float rounding). ``random_count``: steps between random reference switches, 0 for none.
    ``ball_joints`` ("ball" or "euler") and ``joint_damping`` ("implicit" or "ode"): the ball joints' Euler limits and
    the joint damping's integration (module docstring); MuscleVAE's simulator is "euler" and "ode". ``reset_exclude``:
    reset frames left out of the random resets, any of RESET_EXCLUSIONS ("gimbal": a ball joint within GIMBAL_MARGIN of
    the Euler singularity; "limits": a ball joint beyond its Euler limits), none by default. ``solver_iterations``: the
    solver's iteration limit (MuJoCo's default, 100, unless given; fewer is faster and changes some solves).
    ``capture_step``: with graphs, capture the control step as a graph of its own (:meth:`step`); False when a caller
    captures :meth:`step_body` in a graph of its own (ndm_gym.policies.musclevae_pretrained.PolicyRunner). Every
    environment is reset at construction.

    The graphs read and write fixed buffers in place (the action, the muscle state, the reference frames, the episode
    steps, the flags, the reset mask, ``just_reset``: the environments reset since the last step), and their random draws
    come from ``gen``, registered with them: a step or a reset is one graph launch and a few copies. Data are recorded by
    :meth:`before_step` and :meth:`transition`, into fresh tensors or, with ``out``, straight into the rows of contiguous
    buffers (:meth:`transition_buffers`), one write per field."""

    def __init__(self, num_envs: int, device: str = "cuda:0", clips: tuple[str, ...] = ("lafan_jump", "lafan_run", "lafan_walk"),
                 muscle_model: str = "released", graph: bool | None = None, compiled: bool | None = None, random_count: int = RANDOM_COUNT,
                 seed: int = 0, ball_joints: str = "ball", joint_damping: str = "implicit", reset_exclude: tuple[str, ...] = (),
                 solver_iterations: int | None = None, capture_step: bool = True):  # fmt: skip
        if mc.SUBSTEPS % 2:
            raise ValueError("the control step alternates two Newton states and must end in the first")
        _check_mode(ball_joints, joint_damping)
        if set(reset_exclude) - set(RESET_EXCLUSIONS):
            raise ValueError(f"reset_exclude must be drawn from {RESET_EXCLUSIONS}, got {reset_exclude}")
        if solver_iterations is not None and solver_iterations < 1:
            raise ValueError("solver_iterations must be at least 1")
        self.ball_joints, self.joint_damping = ball_joints, joint_damping
        self.nq = NQ if ball_joints == "ball" else NQ_EULER
        self.N, self.device = num_envs, torch.device(device)
        cuda = self.device.type == "cuda"
        if graph is None:
            graph = cuda
        if graph and not cuda:
            raise ValueError("CUDA graphs need a CUDA device")
        self.use_graph = graph  # the step and the reset as CUDA graphs
        compiled = cuda if compiled is None else compiled
        self.random_count = random_count
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.char = mc.load_character()
        self.muscles = mc.muscle_model(muscle_model, self.device, torch.float32)
        self.ref = build_reference(tuple(clips), self.muscles, self.char, self.device, ball_joints=ball_joints)
        lo, hi = mc.ball_euler_limits(self.char)
        self.euler_lo, self.euler_hi = (torch.tensor(x, dtype=torch.float32, device=self.device) for x in (lo, hi))
        self.reset_exclude, self.reset_candidates = tuple(reset_exclude), self.ref.reset_frames
        if self.reset_exclude:
            near, beyond, _ = self._euler_flags(self.reset_candidates)
            drop = torch.zeros_like(near[:, 0])
            for name, flags in (("gimbal", near), ("limits", beyond)):
                if name in self.reset_exclude:
                    drop |= flags.any(-1)
            if bool(drop.all()):
                raise ValueError(f"reset_exclude={self.reset_exclude} leaves no reset frame")
            self.ref = self.ref._replace(reset_frames=self.reset_candidates[~drop])
        self.ball_child = torch.tensor([j + 1 for j in mc.BALL_JOINTS], device=self.device)
        self.ball_parent = torch.tensor([mc.BODY_PARENTS[j + 1] for j in mc.BALL_JOINTS], device=self.device)
        # body_q's quaternion as wxyz, and a quaternion xyzw as wxyz: indices on the device (no copies in the graph or the step)
        self.wxyz, self.xyzw_to_wxyz = torch.tensor([6, 3, 4, 5], device=self.device), torch.tensor([3, 0, 1, 2], device=self.device)

        wp.init()
        self._wp_device = "cpu" if not cuda else f"cuda:{self.device.index or 0}"
        self.model = build_model(num_envs, self._wp_device, self.char, ball_joints, joint_damping)
        check_model(self.model, num_envs, ball_joints, joint_damping, self.char)
        options = {**SOLVER_OPTIONS, "njmax": NJMAX if ball_joints == "ball" else NJMAX_EULER}
        if solver_iterations is not None:
            options["iterations"] = solver_iterations
        self.solver = SolverMuJoCo(self.model, **options)
        self.states = (self.model.state(), self.model.state())
        N, NB, M, dev = num_envs, mc.NUM_BODIES, mc.NUM_MUSCLES, self.device
        self.views = tuple(_Views(wp.to_torch(s.body_q).view(N, NB, 7), wp.to_torch(s.body_qd).view(N, NB, 6), wp.to_torch(s.body_f).view(N, NB, 6),
                                  wp.to_torch(s.joint_q).view(N, self.nq), wp.to_torch(s.joint_qd).view(N, NV)) for s in self.states)  # fmt: skip
        # joint_damping "ode": the damping force, written into Newton's generalized forces before each solver step: -D qd for
        # the hinges and ball joints (gain), -D axis_i . w_rel for the Euler hinges (their columns of joint_f, euler_dofs)
        self.control = self.joint_f = self.damping_gain = self.euler_dofs = self.euler_q0 = None
        if joint_damping == "ode":
            self.control = self.model.control()
            self.joint_f = wp.to_torch(self.control.joint_f).view(N, NV)
            gain = torch.full((NV,), -mc.JOINT_DAMPING)
            gain[:6] = 0.0  # the root's free joint has no damping
            if ball_joints == "euler":
                dofs = [6 + mc.JOINT_QVEL_ADR[j] + i for j in mc.BALL_JOINTS for i in range(3)]
                gain[dofs] = 0.0
                q0 = np.cumsum([7] + [1 if t == "hinge" else 3 for t in mc.JOINT_TYPES])[:-1]  # each joint's first coordinate
                self.euler_dofs, self.euler_q0 = torch.tensor(dofs, device=dev), torch.tensor(q0[list(mc.BALL_JOINTS)], device=dev)
            self.damping_gain = gain.to(dev)
        # the fixed buffers the graphs read and write in place
        self.action, self.length_target, self.length_prev = (torch.zeros(N, M, device=dev) for _ in range(3))
        self.MA, self.MR, self.MF = torch.zeros(N, M, device=dev), torch.ones(N, M, device=dev), torch.zeros(N, M, device=dev)
        self.before = torch.zeros(N, NB, 7, device=dev)  # the bodies' poses before the step
        self.terminated, self.truncated, self.reset_mask = (torch.zeros(N, dtype=torch.bool, device=dev) for _ in range(3))
        self.just_reset = torch.zeros(N, dtype=torch.bool, device=dev)  # reset since the last step
        self.ref_frame, self.ep_step, self.off_count = (torch.zeros(N, dtype=torch.long, device=dev) for _ in range(3))
        self.world_mask = torch.zeros(N + 1, dtype=torch.bool, device=dev)  # Newton's reset mask: the worlds, then the global world
        self._wp_reset_mask, self._wp_world_mask = (wp.from_torch(x, dtype=wp.bool) for x in (self.reset_mask, self.world_mask))
        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device)) if cuda else None
        # the step's and the reset's torch parts, each working in place on the buffers: compiled (fused kernels) unless compiled is False
        fuse = (lambda f: torch.compile(f, dynamic=False)) if compiled else (lambda f: f)  # noqa: E731
        self.muscle_substep, self._begin, self._finish = fuse(self._muscle_substep), fuse(self._begin_step), fuse(self._finish_step)
        self._damping, self._reset_write = fuse(self._damping_force), fuse(self._reset_writes)
        self.step_graph = self.reset_graph = None
        self.reset()
        if graph:
            self.reset_mask.fill_(True)
            self.reset_graph = self.capture(lambda: self._reset_body(self.sample_frames(N)))
            if capture_step:
                self.step_graph = self.capture(self.step_body)
            self.reset()

    # ---- warp on torch's stream, graphs
    def _on_torch_stream(self):
        if self._wp_stream is None:
            return contextlib.nullcontext()
        return wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False)

    def capture(self, fn) -> torch.cuda.CUDAGraph:
        """``fn`` (torch and warp work) as a CUDA graph, captured by torch with warp joining the capture and the generator
        registered, after one warm-up call (which loads every kernel and compiles; it changes the state, which the caller
        resets)."""
        stream = torch.cuda.Stream(self.device)
        ws = wp.stream_from_torch(stream)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
            fn()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self.gen)
        with torch.cuda.graph(graph, stream=stream):
            wp.capture_begin(stream=ws, external=True, capture_mode=wp.CaptureMode.GLOBAL)
            try:
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    fn()
            finally:
                wp.capture_end(stream=ws)
        return graph

    # ---- control step
    def _muscle_substep(self, body_q: torch.Tensor, body_f: torch.Tensor):
        """One muscle substep in place: the wrenches of the bodies' poses ``body_q`` (N, 23, 7) into ``body_f`` (N, 23, 6),
        and the next lengths and fatigue state into their buffers."""
        out = self.muscles.substep(body_q[..., :3], quat_to_mat(body_q[..., self.wxyz]), self.length_prev, self.length_target, self.MA, self.MR,
                                   self.MF)  # fmt: skip
        body_f.copy_(torch.cat((out.body_force, out.body_torque), -1))
        self.length_prev.copy_(out.length)
        self.MA.copy_(out.MA)
        self.MR.copy_(out.MR)
        self.MF.copy_(out.MF)

    def _begin_step(self):
        """The bodies' poses before the step, and the action's PD targets (1 + a) l_Tpose (ENV:431-433), in place."""
        self.before.copy_(self.views[0].body_q)
        self.length_target.copy_(self.muscles.target_length(self.action))

    def _finish_step(self, random_frames: torch.Tensor):
        """The done flags, the reference's next frames (``random_frames`` where it switches) and the episode steps, in
        place; no environment counts as just reset any more."""
        terminated, truncated, off_count = done_flags(self.views[0].body_q, self.views[0].body_qd, self.before, self.ref.head_z[self.ref_frame],
                                                      self.off_count, self.ep_step, self.xyzw_to_wxyz)  # fmt: skip
        self.terminated.copy_(terminated)
        self.truncated.copy_(truncated)
        self.off_count.copy_(off_count)
        self.ref_frame.copy_(next_frames(self.ref_frame, self.ep_step, self.ref.is_last, self.random_count, random_frames))
        self.ep_step.add_(1)
        self.just_reset.zero_()

    def _damping_force(self, body_q: torch.Tensor, body_qd: torch.Tensor, joint_q: torch.Tensor, joint_qd: torch.Tensor):
        """joint_damping "ode": the explicit damping force of a state into the generalized forces (joint_f), in place: -D
        qd for the hinges and ball joints; for the Euler hinges -D axis_i . (w_child - w_parent), with the hinges' world
        axes our x in the parent, our z turned by the first hinge's angle about x, our y in the child (EULER_HINGES)."""
        f = joint_qd * self.damping_gain
        if self.euler_dofs is not None:
            R = quat_to_mat(body_q[..., self.wxyz])
            Rp, Rc = R[:, self.ball_parent], R[:, self.ball_child]  # (N, 14, 3, 3)
            w = body_qd[:, self.ball_child, 3:] - body_qd[:, self.ball_parent, 3:]
            q0 = joint_q[:, self.euler_q0].unsqueeze(-1)
            axes = (Rp[..., 0], torch.cos(q0) * Rp[..., 2] - torch.sin(q0) * Rp[..., 1], Rc[..., 1])
            torque = torch.stack([(ax * w).sum(-1) for ax in axes], -1) * -mc.JOINT_DAMPING  # (N, 14, 3)
            f = f.index_copy(1, self.euler_dofs, torque.flatten(1))
        self.joint_f.copy_(f)

    def _control_step(self):
        """Six substeps: the muscle substep on the current poses, its wrenches into the input state, (joint_damping "ode")
        the damping force of the input state into the generalized forces, a solver step into the other state. The states
        alternate, so the step ends in states[0]."""
        for i in range(mc.SUBSTEPS):
            a = i % 2
            v = self.views[a]
            self.muscle_substep(v.body_q, v.body_f)
            if self.control is not None:
                self._damping(v.body_q, v.body_qd, v.joint_q, v.joint_qd)
            self.solver.step(self.states[a], self.states[1 - a], self.control, None, SIM_DT)

    def step_body(self):
        """The control step on the fixed buffers: the PD targets, six substeps, the done flags, the reference's next frames,
        the episode steps. Without a graph of its own, so that a caller's capture (:meth:`capture`) can add work around
        it, e.g. a policy that writes the action buffer from the state (ndm_gym.policies.musclevae_pretrained)."""
        self._begin()
        self._control_step()
        self._finish(self.sample_frames(self.N))

    def muscle_step(self):
        """The muscle model's substep on the current state, for analysis (forces, activations, Hill factors: a
        :class:`~ndm_gym.robots.muscles.MuscleStep`); it changes nothing."""
        q = self.views[0].body_q
        return self.muscles.substep(q[..., :3], quat_to_mat(q[..., self.wxyz]), self.length_prev, self.length_target, self.MA, self.MR, self.MF)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One control step with muscle actions (N, 284); returns MuscleVAE's termination and truncation flags (N,) of the
        step. Resets are the caller's."""
        if self.use_graph and self.step_graph is None:
            raise RuntimeError("constructed with capture_step=False: the caller's graph runs the control step (step_body)")
        self.action.copy_(action)
        if self.step_graph is not None:
            self.step_graph.replay()
        else:
            with self._on_torch_stream():
                self.step_body()
        return self.terminated.clone(), self.truncated.clone()

    def sample_frames(self, n: int) -> torch.Tensor:
        """``n`` frames drawn uniformly from the frames a reset may start at."""
        k = torch.randint(0, len(self.ref.reset_frames), (n,), generator=self.gen, device=self.device)
        return self.ref.reset_frames[k]

    # ---- reset
    def _reset_writes(self, frames: torch.Tensor, MA: torch.Tensor, MR: torch.Tensor, MF: torch.Tensor):
        """The reset's writes, in place, for the environments in reset_mask: the reference's coordinates, velocities and
        previous lengths of ``frames`` (N,), the fatigue state (MA, MR, MF) (N, 284), the reference's next frame, zero
        episode steps and off counts, the just-reset and Newton's world masks."""
        m = self.reset_mask.unsqueeze(-1)
        v = self.views[0]
        v.joint_q.copy_(torch.where(m, self.ref.joint_q[frames], v.joint_q))
        v.joint_qd.copy_(torch.where(m, self.ref.joint_qd[frames], v.joint_qd))
        self.length_prev.copy_(torch.where(m, self.ref.length_prev[frames], self.length_prev))
        for x, new in zip((self.MA, self.MR, self.MF), (MA, MR, MF)):
            x.copy_(torch.where(m, new, x))
        self.ref_frame.copy_(torch.where(self.reset_mask, frames + 1, self.ref_frame))
        self.ep_step.masked_fill_(self.reset_mask, 0)
        self.off_count.masked_fill_(self.reset_mask, 0)
        self.just_reset.logical_or_(self.reset_mask)
        self.world_mask[: self.N].copy_(self.reset_mask)

    def _reset_body(self, frames: torch.Tensor):
        """MuscleVAE's reset (module docstring) of the environments in reset_mask to ``frames`` (N,): the fatigue draw, the
        writes (:meth:`_reset_writes`, compiled as the step's torch parts), the bodies' poses and the solver's reset."""
        self._reset_write(frames, *self.muscles.random_fatigue((self.N,), self.gen))
        newton.eval_fk(self.model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0], mask=self._wp_reset_mask)
        self.solver.reset(self.states[0], world_mask=self._wp_world_mask, flags=0)  # MuJoCo's warm start and applied forces

    def reset(self, mask: torch.Tensor | None = None, frames: torch.Tensor | None = None):
        """MuscleVAE's reset of the environments where ``mask`` (N,) is true, all if None, at the reference frames
        ``frames`` (N,) or at random ones: one graph launch, or eager with given frames (they are checked, a host
        synchronization)."""
        N = self.N
        if mask is None:
            self.reset_mask.fill_(True)
        elif mask.shape != (N,) or mask.dtype != torch.bool:
            raise ValueError(f"mask must be a ({N},) bool tensor")
        else:
            self.reset_mask.copy_(mask)
        if frames is None and self.reset_graph is not None:
            self.reset_graph.replay()
            return
        if frames is None:
            frames = self.sample_frames(N)
        else:
            frames = frames.to(self.device)
            pos_in_clip = frames - self.ref.start[self.ref.clip[frames.clamp(0, len(self.ref.clip) - 1)]]
            if frames.shape != (N,) or bool(((frames < 0) | (frames >= len(self.ref.clip)) | (pos_in_clip < 1)).any()) or \
                    bool(self.ref.is_last[frames.clamp(0, len(self.ref.clip) - 1)].any()):  # fmt: skip
                raise ValueError("reset frames must be (N,) frames with a previous one in their clip, none a clip's last")
        with self._on_torch_stream():
            self._reset_body(frames)

    # ---- policy, state, context
    def reference_action(self) -> torch.Tensor:
        """a = l_ref / l_Tpose - 1 (N, 284) of the reference frame the next step tracks."""
        return self.ref.length[self.ref_frame] / self.muscles.rest_length - 1.0

    def data_policy(self, noise_std: float) -> torch.Tensor:
        """The data policy's action: :meth:`reference_action` plus Gaussian noise of std ``noise_std``."""
        a = self.reference_action()
        return a + noise_std * torch.randn(a.shape, generator=self.gen, device=self.device)

    def body_state(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 23, 13): each body's position (its center of mass), orientation (quaternion wxyz), center-of-mass velocity
        and angular velocity, world frame; written into ``out`` if given."""
        q = self.views[0].body_q
        return torch.cat((q[..., :3], q[..., self.wxyz], self.views[0].body_qd), -1, out=out)

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Newton's joint coordinates (N, 71), the clips' layout but for quaternions xyzw, and velocities (N, 56): the
        root's linear and angular velocity in the world frame, ball joints' relative angular velocity in the parent's
        frame, hinge rates; with ball_joints "euler" (N, 57) and (N, 56), each ball joint's three hinge angles and rates
        (:func:`newton_coords`)."""
        return self.views[0].joint_q.clone(), self.views[0].joint_qd.clone()

    def fatigue_state(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 284, 3): each muscle's (MA, MR, MF), the 3CC-r compartments (activated, resting, fatigued); written into
        ``out`` if given."""
        return torch.stack((self.MA, self.MR, self.MF), -1, out=out)

    def part_fatigue(self) -> torch.Tensor:
        """(N, 15): their fatigue observation, the body parts' f0-weighted MA, MR, MF (CHR:333-337)."""
        return self.muscles.part_fatigue(self.MA, self.MR, self.MF)

    def context(self) -> torch.Tensor:
        """Everything that stays fixed within an episode and shapes the dynamics: nothing, (N, 0); MuscleVAE randomizes
        no physical parameter, and the reference frame and initial fatigue are state."""
        return torch.zeros(self.N, 0, device=self.device)

    def _euler_flags(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Of the reference frames ``frames`` (K,), per ball joint (K, 14): near the Euler hinges' singularity (|a1| within
        GIMBAL_MARGIN of pi / 2), beyond the Euler limits, and by how much (rad, the largest of the three angles')."""
        angles = self.ref.ball_euler[frames]  # (K, 14, 3)
        excess = ((angles - self.euler_hi).clamp(min=0.0) + (self.euler_lo - angles).clamp(min=0.0)).amax(-1)
        return angles[..., 1].abs() > np.pi / 2 - GIMBAL_MARGIN, excess > 0.0, excess

    def reset_frame_stats(self) -> dict:
        """The reference frames a reset may start at (before ``reset_exclude``) near the Euler hinges' singularity and
        beyond the ball joints' Euler limits (module docstring): the frame counts, those left after ``reset_exclude``, and
        per ball joint the counts, the largest |a1| and the largest excess (rad). A host synchronization."""
        near, beyond, excess = (x.cpu() for x in self._euler_flags(self.reset_candidates))
        a1 = self.ref.ball_euler[self.reset_candidates][..., 1].abs().amax(0).cpu()
        joints = {mc.JOINT_NAMES[j]: {"near_gimbal": int(near[:, i].sum()), "beyond_limits": int(beyond[:, i].sum()), "max_abs_a1": float(a1[i]),
                                      "max_excess": float(excess[:, i].max())} for i, j in enumerate(mc.BALL_JOINTS)}  # fmt: skip
        return {"reset_frames": len(self.reset_candidates), "near_gimbal": int(near.any(-1).sum()), "beyond_limits": int(beyond.any(-1).sum()),
                "gimbal_margin": GIMBAL_MARGIN, "reset_exclude": list(self.reset_exclude), "kept": len(self.ref.reset_frames), "joints": joints}  # fmt: skip

    def ball_limit_excess(self) -> torch.Tensor:
        """(N, 14, 3): by how much (rad) each ball joint's Euler angles (ndm_gym.robots.musclevae_char.ball_euler_angles)
        exceed the limits their simulator enforces, 0 within them; ball joints in BALL_JOINTS order."""
        q = self.views[0].body_q
        R = quat_to_mat(q[..., self.wxyz])
        angles = mc.ball_euler_angles(R[:, self.ball_parent].transpose(-1, -2) @ R[:, self.ball_child])
        return (angles - self.euler_hi).clamp(min=0.0) + (self.euler_lo - angles).clamp(min=0.0)

    # ---- transitions
    def transition_fields(self) -> dict:
        """One transition's fields for ``ndm_gym.buffer.ReplayBuffer`` (``transition``): the bodies' states before and
        after the control step (:meth:`body_state`), the action, the muscles' fatigue states before and after
        (:meth:`fatigue_state`), the context, the reference frame and episode step before the step, the flags."""
        f32, NB, M = torch.float32, mc.NUM_BODIES, mc.NUM_MUSCLES
        return {"s": ((NB, 13), f32), "s1": ((NB, 13), f32), "a": (M, f32), "fatigue": ((M, 3), f32), "fatigue1": ((M, 3), f32),
                "ctx": (self.context().shape[-1], f32), "t": ((), torch.long), "ep_step": ((), torch.int32), "done": ((), torch.bool),
                "term": ((), torch.bool)}  # fmt: skip

    def transition_buffers(self, T: int) -> dict:
        """Contiguous device buffers (T, N, ...) of T transitions, laid out as :meth:`transition_fields` (and as
        ReplayBuffer's data); their rows ``{k: v[t]}`` are ``out`` arguments of :meth:`before_step` and :meth:`transition`."""
        return {k: torch.empty((T, self.N) + ((s,) if isinstance(s, int) else s), dtype=d, device=self.device)
                for k, (s, d) in self.transition_fields().items()}  # fmt: skip

    def before_step(self, out: dict | None = None) -> dict:
        """The transition's fields read before ``step`` (s, fatigue, ctx, t, ep_step), new tensors or written into ``out``."""
        if out is None:
            return {"s": self.body_state(), "fatigue": self.fatigue_state(), "ctx": self.context(), "t": self.ref_frame.clone(),
                    "ep_step": self.ep_step.int()}  # fmt: skip
        self.body_state(out=out["s"])
        self.fatigue_state(out=out["fatigue"])
        out["t"].copy_(self.ref_frame)
        out["ep_step"].copy_(self.ep_step)
        return out

    def transition(self, before: dict, action: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, out: dict | None = None) -> dict:
        """The transition of the control step just taken; ``before`` is :meth:`before_step`, read before ``step``. With
        ``out`` (a row of :meth:`transition_buffers`, usually the one ``before`` was written into) the fields are written
        into it."""
        if out is None:
            return {"s": before["s"], "s1": self.body_state(), "a": action, "fatigue": before["fatigue"], "fatigue1": self.fatigue_state(),
                    "ctx": before["ctx"], "t": before["t"], "ep_step": before["ep_step"], "done": terminated | truncated, "term": terminated}  # fmt: skip
        if before is not out:
            for k in ("s", "fatigue", "ctx", "t", "ep_step"):
                out[k].copy_(before[k])
        self.body_state(out=out["s1"])
        out["a"].copy_(action)
        self.fatigue_state(out=out["fatigue1"])
        torch.logical_or(terminated, truncated, out=out["done"])
        out["term"].copy_(terminated)
        return out

    # ---- checks
    def check_solver(self) -> dict:
        """MuJoCo-Warp's overflow flags since construction (a host synchronization): raises if a contact or constraint
        buffer overflowed, which drops contacts; returns how many worlds hit the solver's or the line search's
        iteration limit."""
        import mujoco_warp

        flags = mujoco_warp.OverflowType
        ov = self.solver.mjw_data.overflow.numpy()
        lost = flags.NEFC | flags.NJMAX_NNZ | flags.BROADPHASE | flags.NARROWPHASE
        if (ov & int(lost)).any():
            raise RuntimeError(f"MuJoCo-Warp buffers overflowed in {int(((ov & int(lost)) != 0).sum())} worlds (flags "
                               f"{sorted({int(x) for x in ov if x & int(lost)})}): raise NCONMAX / NJMAX")  # fmt: skip
        return {"iterations": int(((ov & int(flags.ITERATIONS)) != 0).sum()), "ls_iterations": int(((ov & int(flags.LS_ITERATIONS)) != 0).sum())}


__all__ = ["MuscleSim", "Reference", "build_reference", "build_model", "character_mjcf", "done_flags", "next_frames"]
