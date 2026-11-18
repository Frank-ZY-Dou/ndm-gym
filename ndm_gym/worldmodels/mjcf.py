"""A world model of a MuJoCo model in its generalized coordinates, for models without actuator activation: qpos and qvel are
the whole state, so the model reads any MJCF that passes ``MjcfLayout``'s checks.

The layout is read once from the ``MjModel``: its joints (free, ball, slide, hinge), the anchor (the free joint at the root of
the actuated tree, if there is one), the other free joints (objects), the actuators (joint or fixed-tendon transmissions with
a fixed gain and an affine or no bias, the position and velocity servos of most robot models) and the joint equalities (a
joint held at a polynomial of another, as hands couple their finger joints), which each step then satisfies exactly: the
coupled joint's position and velocity are set from the other's. The inputs do not depend on the anchor's
heading and horizontal position: vectors in the world frame are read in the anchor's heading frame, heights stay absolute (the
floor). A planar model's root has no anchor but slides: with ``planar_root`` its horizontal slides' positions (rootx) are left
out of the inputs in the same way, their velocities kept. Two parameterizations: SuperTrack's (``kind="accel"``,
worldmodels.accel, the default): an MLP predicts the generalized accelerations, with the free joints' linear parts in the
heading frame, and one step integrates them in MuJoCo's own convention
(mj_integratePos: the velocity first, then positions add and free and ball quaternions turn by the body-frame angular
velocity), so qpos and qvel round-trip through MuJoCo unchanged; SuperMimic's (``kind="increment"``, worldmodels.increment):
the MLP predicts the step's increments directly, the tangent from qpos to the next qpos and the velocity's change, the free
joints' linear parts again in the heading frame, and the step applies them (mj_integratePos with dt = 1). The error of a
prediction is taken in the tangent space (mj_differentiatePos), in units of the spread of one true step's increments, as the
hand's world model takes it (worldmodels.hand). With ``geometry=True`` the inputs also hold the contact geometry: forward
kinematics in MuJoCo's convention (mj_kinematics) places the robot's collision geoms, and each one's centre is read in each
object's frame, with its signed distance to the object's primitive collision geoms. With ``contacts`` (``contact_spheres``'
result) they hold the contact inputs: the robot's collision geoms as spheres, and per geom and object its gap, the nearest
point, the normal there and the relative velocity (``contact_inputs``); the objects are the free objects and, where the spheres
name them, bodies that ride on slides or hinges and static geoms of the world (a floor). With ``backbone="rigidformer"`` a body-token
Transformer (``MjcfRigidFormerNet``, after worldmodels.rigidformer) takes the MLP's place: one token per body, with its pose,
velocity and joints in the inputs' frame, and per body the outputs of its own degrees of freedom in the MLP's output layout,
so that the targets, the integration and the error are the same."""

from __future__ import annotations

import contextlib
import copy
import functools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..core.kinematics import mat_mat, mat_vec
from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import axis_angle_to_mat, mat_to_6d, quat_to_mat, yaw_mat

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3  # mjtJoint
TRN_JOINT, TRN_TENDON, DYN_NONE, GAIN_FIXED, BIAS_NONE, BIAS_AFFINE = 0, 3, 0, 0, 0, 1  # mjtTrn, mjtDyn, mjtGain, mjtBias
EQ_JOINT, WRAP_JOINT = 2, 1  # mjtEq, mjtWrap
SPHERE, CAPSULE, CYLINDER, BOX = 2, 3, 5, 6  # mjtGeom: the objects' collision geoms the contact geometry reads
ELLIPSOID, MESH = 4, 7  # mjtGeom: with the four above, the robot's collision geoms the contact inputs cover with spheres
PLANE = 0  # mjtGeom: with the four primitives, the world's static geoms the contact inputs can read gaps to
CONTACT_FEATURES = 13  # the contact inputs per robot collision geom and object (MjcfWorldModel.contact_inputs)
UNUSED_PLACE = 10.0  # contact_spheres: each coordinate (m), in its body's frame, of a geom's unused sphere places; their radius is 0
# the revision of the contact gaps, which a fit with static geoms is saved with ("contact_gaps", beside its layout and its
# arguments): 2 keeps a geom's unused sphere places out of the gaps; a fit saved before (9418a03d) has no mark
GAPS_REVISION = 2
# error floors (per tangent or velocity component): rad, m, rad/s, m/s
FLOOR_ANGLE, FLOOR_POS, FLOOR_ANG_VEL, FLOOR_LIN_VEL = 1e-3, 5e-4, 5e-2, 1e-2
HIDDEN = (1024, 1024, 1024)  # the MLP's widths
# the RigidFormer backbone's options (MjcfWorldModel's ``rf``): width, blocks, heads, register tokens, the rotary frequencies'
# top (rad/m; from 1), and the token's point ("com": the body's centre of mass, "origin": its frame's origin)
RF_DEFAULTS = {"dim": 128, "layers": 4, "heads": 4, "registers": 2, "rope_max_freq": 10.0, "point": "com"}


@dataclass(frozen=True)
class MjcfLayout:
    """What the world model reads of an ``MjModel``. Index tuples address qpos (``q``) and qvel (``d``)."""

    nq: int
    nv: int
    nu: int
    dt: float
    anchor: tuple[int, int] | None  # (qpos address, qvel address) of the anchor's free joint
    objects: tuple[tuple[int, int], ...]  # the other free joints
    balls: tuple[tuple[int, int], ...]
    scalars: tuple[tuple[int, int, bool], ...]  # hinges and slides: (qpos address, qvel address, angle with no limits)
    slides: tuple[int, ...]  # the slides' qvel addresses (metres; the other scalars are radians)
    act_q: tuple[int, ...]  # each actuator's joint, qpos address (-1: a tendon transmission, in ``tendons``)
    act_d: tuple[int, ...]  # its qvel address
    gear: tuple[float, ...]
    gain: tuple[float, ...]
    bias: tuple[tuple[float, float, float], ...]
    ctrl_range: tuple[tuple[float, float] | None, ...]
    force_range: tuple[tuple[float, float] | None, ...]
    # fixed-tendon actuators: (actuator, ((qpos address, qvel address, gear times the joint's coefficient), ...))
    tendons: tuple[tuple[int, tuple[tuple[int, int, float], ...]], ...] = ()
    # joint equalities y - y0 = a0 + a1 (x - x0) + ... + a4 (x - x0)^4, y and x hinges or slides, y0 and x0 their qpos0:
    # (y's qpos address, its qvel address, x's (-1: none, y = y0 + a0), x's qvel address, y0, x0, (a0, ..., a4))
    couplings: tuple[tuple[int, int, int, int, float, float, tuple[float, ...]], ...] = ()
    # the kinematic tree, for the contact geometry inputs: per body after the world, in MuJoCo's order (parents first), (parent
    # body, position (3) and quaternion (4) in the parent's frame, a mocap body, joints), each joint (type, qpos address, axis
    # (3), anchor (3) in the body's frame, qpos0 of a hinge or slide)
    tree: tuple = ()
    # the robot's collision geoms (contype or conaffinity set, on a body outside the world and the objects' subtrees): (body,
    # position (3) in the body's frame)
    robot_geoms: tuple = ()
    # per object, in ``objects``' order, its body's collision geoms of a primitive type (sphere, capsule, cylinder, box): (type,
    # position (3) and quaternion (4) in the body's frame, size (3))
    object_geoms: tuple = ()
    # per object, the names of the other collision geoms of its subtree (another type, such as a mesh or an ellipsoid, or on a
    # body below the object's): no input reads them, so the contact geometry and the contact inputs refuse such an object
    object_unread: tuple = ()
    # per body after the world, its centre of mass in its frame (body_ipos), for the RigidFormer's tokens
    ipos: tuple = ()
    planar: tuple[int, ...] = ()  # qpos addresses of the root's horizontal slides, whose positions the inputs leave out

    @staticmethod
    def from_model(m, anchor: str | None = None, planar_root: bool = False) -> MjcfLayout:
        """The layout of ``m`` (mujoco.MjModel). ``anchor``: the name of the anchor's free joint; by default the free joint at the
        root of the actuated joints' tree, or none (a fixed base) if that root has none. Features the model does not support are
        an error: actuator activation (na > 0); equality constraints other than joint equalities active from the start, and
        chains of them; flexes; mocap bodies with geoms; actuators other than joint or fixed-tendon transmissions (to hinges and
        slides) with a fixed gain and an affine or no bias. ``planar_root``: the inputs leave out the positions of the slides that
        move the actuated tree's root horizontally (``planar_slides``)."""
        import mujoco

        bad = []
        if m.na > 0:
            bad.append(f"{m.na} actuator activation states")
        scalar = lambda j: j >= 0 and m.jnt_type[j] in (SLIDE, HINGE)  # noqa: E731
        couplings = []
        for e in range(m.neq):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or str(e)
            y, x = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if m.eq_type[e] != EQ_JOINT:  # (MuJoCo allows joint equalities between hinges and slides only)
                bad.append(f"equality {name}: type {int(m.eq_type[e])} (only joint equalities)")
            elif not m.eq_active0[e]:
                bad.append(f"equality {name}: inactive at the start (switched at run time)")
            else:
                ya, xa = (int(m.jnt_qposadr[y]), int(m.jnt_dofadr[y])), (int(m.jnt_qposadr[x]), int(m.jnt_dofadr[x])) if x >= 0 else (-1, -1)
                couplings.append((*ya, *xa, float(m.qpos0[ya[0]]), float(m.qpos0[xa[0]]) if x >= 0 else 0.0,
                                  tuple(float(c) for c in m.eq_data[e, :5])))  # fmt: skip
        dep = [c[0] for c in couplings]
        if len(set(dep)) < len(dep) or set(dep) & {c[2] for c in couplings}:
            bad.append("joint equalities: a joint coupled twice, or a coupled joint that drives another (a chain)")
        if getattr(m, "nflex", 0) > 0:
            bad.append(f"{m.nflex} flexes")
        mocap_geoms = [b for b in range(m.nbody) if m.body_mocapid[b] >= 0 and m.body_geomnum[b] > 0]
        if mocap_geoms:
            bad.append(f"mocap bodies with geoms ({len(mocap_geoms)})")
        tendons, tendon_joints = [], []
        for a in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)
            if m.actuator_trntype[a] == TRN_TENDON:
                t = int(m.actuator_trnid[a, 0])
                wraps = range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t])
                if any(m.wrap_type[w] != WRAP_JOINT or not scalar(int(m.wrap_objid[w])) for w in wraps):
                    bad.append(f"actuator {name}: a tendon that is not fixed to hinges and slides")
                else:
                    g = float(m.actuator_gear[a, 0])
                    tendon_joints += [int(m.wrap_objid[w]) for w in wraps]
                    tendons.append((a, tuple((int(m.jnt_qposadr[m.wrap_objid[w]]), int(m.jnt_dofadr[m.wrap_objid[w]]), g * float(m.wrap_prm[w]))
                                             for w in wraps)))  # fmt: skip
            elif m.actuator_trntype[a] != TRN_JOINT:
                bad.append(f"actuator {name}: transmission {int(m.actuator_trntype[a])} (only joint and fixed-tendon transmissions)")
            elif m.jnt_type[m.actuator_trnid[a, 0]] not in (SLIDE, HINGE):
                bad.append(f"actuator {name}: drives a free or ball joint")
            if m.actuator_dyntype[a] != DYN_NONE:
                bad.append(f"actuator {name}: dynamics type {int(m.actuator_dyntype[a])}")
            if m.actuator_gaintype[a] != GAIN_FIXED or m.actuator_biastype[a] not in (BIAS_NONE, BIAS_AFFINE):
                bad.append(f"actuator {name}: gain type {int(m.actuator_gaintype[a])}, bias type {int(m.actuator_biastype[a])}")
        if bad:
            raise ValueError("MjcfLayout: not supported: " + "; ".join(bad))
        free = [j for j in range(m.njnt) if m.jnt_type[j] == FREE]
        if anchor is not None:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, anchor)
            if j < 0 or m.jnt_type[j] != FREE:
                raise ValueError(f"MjcfLayout: no free joint named {anchor!r}")
            ja = j
        else:
            joints = [int(m.actuator_trnid[a, 0]) for a in range(m.nu) if m.actuator_trntype[a] == TRN_JOINT] + tendon_joints
            roots = {int(m.body_rootid[m.jnt_bodyid[j]]) for j in joints}
            rooted = [j for j in free if int(m.body_rootid[m.jnt_bodyid[j]]) in roots and m.jnt_bodyid[j] in roots]
            if len(rooted) > 1:
                raise ValueError(f"MjcfLayout: the actuated joints hang from {len(rooted)} free roots; name the anchor")
            ja = rooted[0] if rooted else None
        addr = lambda j: (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))  # noqa: E731
        act = [int(m.actuator_trnid[a, 0]) if m.actuator_trntype[a] == TRN_JOINT else -1 for a in range(m.nu)]
        fl = lambda x: tuple(float(v) for v in x)  # noqa: E731
        tree = tuple((int(m.body_parentid[b]), fl(m.body_pos[b]), fl(m.body_quat[b]), bool(m.body_mocapid[b] >= 0),
                      tuple((int(m.jnt_type[j]), int(m.jnt_qposadr[j]), fl(m.jnt_axis[j]), fl(m.jnt_pos[j]),
                             float(m.qpos0[m.jnt_qposadr[j]]) if m.jnt_type[j] in (SLIDE, HINGE) else 0.0)
                            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])))
                     for b in range(1, m.nbody))  # fmt: skip
        obj_bodies = [int(m.jnt_bodyid[j]) for j in free if j != ja]

        def under(b, root):  # b in root's subtree
            while b > 0 and b != root:
                b = int(m.body_parentid[b])
            return b == root

        collides = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])  # noqa: E731
        robot_geoms = tuple((int(m.geom_bodyid[g]), fl(m.geom_pos[g])) for g in range(m.ngeom)
                            if collides(g) and m.geom_bodyid[g] > 0 and not any(under(int(m.geom_bodyid[g]), o) for o in obj_bodies))  # fmt: skip
        object_geoms = tuple(tuple((int(m.geom_type[g]), fl(m.geom_pos[g]), fl(m.geom_quat[g]), fl(m.geom_size[g])) for g in range(m.ngeom)
                                   if m.geom_bodyid[g] == o and collides(g) and m.geom_type[g] in (SPHERE, CAPSULE, CYLINDER, BOX))
                             for o in obj_bodies)  # fmt: skip
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"  # noqa: E731
        object_unread = tuple(tuple(gname(g) for g in range(m.ngeom) if collides(g) and under(int(m.geom_bodyid[g]), o)
                                    and not (m.geom_bodyid[g] == o and m.geom_type[g] in (SPHERE, CAPSULE, CYLINDER, BOX)))
                              for o in obj_bodies)  # fmt: skip
        return MjcfLayout(
            nq=m.nq, nv=m.nv, nu=m.nu, dt=float(m.opt.timestep),
            anchor=addr(ja) if ja is not None else None,
            objects=tuple(addr(j) for j in free if j != ja),
            balls=tuple(addr(j) for j in range(m.njnt) if m.jnt_type[j] == BALL),
            scalars=tuple(addr(j) + (bool(m.jnt_type[j] == HINGE and not m.jnt_limited[j]),) for j in range(m.njnt) if m.jnt_type[j] in (SLIDE, HINGE)),
            slides=tuple(int(m.jnt_dofadr[j]) for j in range(m.njnt) if m.jnt_type[j] == SLIDE),
            act_q=tuple(int(m.jnt_qposadr[j]) if j >= 0 else -1 for j in act), act_d=tuple(int(m.jnt_dofadr[j]) if j >= 0 else -1 for j in act),
            gear=tuple(float(m.actuator_gear[a, 0]) for a in range(m.nu)),
            gain=tuple(float(m.actuator_gainprm[a, 0]) for a in range(m.nu)),
            bias=tuple(tuple(float(x) for x in m.actuator_biasprm[a, :3]) if m.actuator_biastype[a] == BIAS_AFFINE else (0.0, 0.0, 0.0)
                       for a in range(m.nu)),
            ctrl_range=tuple(tuple(float(x) for x in m.actuator_ctrlrange[a]) if m.actuator_ctrllimited[a] else None for a in range(m.nu)),
            force_range=tuple(tuple(float(x) for x in m.actuator_forcerange[a]) if m.actuator_forcelimited[a] else None for a in range(m.nu)),
            tendons=tuple(tendons), couplings=tuple(couplings), tree=tree, robot_geoms=robot_geoms, object_geoms=object_geoms,
            object_unread=object_unread,
            ipos=tuple(fl(m.body_ipos[b]) for b in range(1, m.nbody)),
            planar=planar_slides(m, ja, tendon_joints, couplings) if planar_root else (),
        )  # fmt: skip

    @property
    def free(self) -> tuple[tuple[int, int], ...]:
        """Every free joint, the anchor first."""
        return ((self.anchor,) if self.anchor is not None else ()) + self.objects

    def floors(self) -> np.ndarray:
        """(2 nv,) the error floors of the tangent position and the velocity, in qvel's layout."""
        fl = np.zeros(2 * self.nv)
        for _, d in self.free:
            fl[d : d + 3], fl[d + 3 : d + 6] = FLOOR_POS, FLOOR_ANGLE
            fl[self.nv + d : self.nv + d + 3], fl[self.nv + d + 3 : self.nv + d + 6] = FLOOR_LIN_VEL, FLOOR_ANG_VEL
        for _, d in self.balls:
            fl[d : d + 3], fl[self.nv + d : self.nv + d + 3] = FLOOR_ANGLE, FLOOR_ANG_VEL
        for _, d, _ in self.scalars:  # a slide's metres and a hinge's radians share the angle's floor
            fl[d], fl[self.nv + d] = FLOOR_ANGLE, FLOOR_ANG_VEL
        if (fl == 0).any():
            raise ValueError("MjcfLayout: a degree of freedom belongs to no joint")
        return fl


def planar_slides(m, anchor_joint, tendon_joints, couplings) -> tuple[int, ...]:
    """The qpos addresses of the slides that move the actuated tree's root body horizontally (a planar model's rootx): on a
    plane its dynamics do not depend on their positions. An error unless the model has no anchor, gravity points down z, the
    actuated joints hang from one root body and it has such a slide, unlimited, not driven, not in a joint equality and not
    turned by a hinge before it on that body."""
    import mujoco

    if anchor_joint is not None:
        raise ValueError("MjcfLayout(planar_root): the root is a free anchor, whose horizontal position the inputs already leave out")
    g = np.asarray(m.opt.gravity, dtype=float)
    if g[0] != 0 or g[1] != 0 or g[2] >= 0:
        raise ValueError(f"MjcfLayout(planar_root): gravity {g.tolist()} does not point down the z axis")
    driven = {int(m.actuator_trnid[a, 0]) for a in range(m.nu) if m.actuator_trntype[a] == TRN_JOINT} | set(tendon_joints)
    roots = {int(m.body_rootid[m.jnt_bodyid[j]]) for j in driven}
    if len(roots) != 1:
        raise ValueError(f"MjcfLayout(planar_root): the actuated joints hang from {len(roots)} root bodies")
    root = roots.pop()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.body_quat[root])
    coupled = {c[0] for c in couplings} | {c[2] for c in couplings}
    out, turned = [], False
    for j in range(m.body_jntadr[root], m.body_jntadr[root] + m.body_jntnum[root]):
        if m.jnt_type[j] != SLIDE:
            turned = True
            continue
        axis = rot.reshape(3, 3) @ m.jnt_axis[j]
        if abs(axis[2]) > 1e-9 * np.linalg.norm(axis):
            continue  # it moves the root up or down: its position matters
        why = [w for w, bad in (("turned by a hinge before it", turned), ("limited", m.jnt_limited[j]), ("driven", j in driven),
                                ("in a joint equality", int(m.jnt_qposadr[j]) in coupled)) if bad]  # fmt: skip
        if why:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or str(j)
            raise ValueError(f"MjcfLayout(planar_root): the root's horizontal slide {name} is {', '.join(why)}")
        out.append(int(m.jnt_qposadr[j]))
    if not out:
        raise ValueError("MjcfLayout(planar_root): the root body has no horizontal slide")
    return tuple(out)


# ---------------- quaternions (wxyz, MuJoCo's convention) ----------------
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)  # fmt: skip


def quat_exp(v: torch.Tensor) -> torch.Tensor:
    """The unit quaternion of the rotation vector ``v``."""
    th = v.norm(dim=-1, keepdim=True)
    half = 0.5 * th
    k = torch.where(th > 1e-8, torch.sin(half) / th.clamp_min(1e-8), 0.5 - th**2 / 48)
    return torch.cat((torch.cos(half), k * v), -1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
    """The rotation vector of the unit quaternion ``q``, the short way round."""
    q = torch.where(q[..., :1] < 0, -q, q)
    s = q[..., 1:].norm(dim=-1, keepdim=True)
    th = 2 * torch.atan2(s, q[..., :1])
    return torch.where(s > 1e-8, th / s.clamp_min(1e-8), 2.0 / q[..., :1].clamp_min(1e-8)) * q[..., 1:]


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def integrate_pos(lay: MjcfLayout, qpos: torch.Tensor, qvel: torch.Tensor, dt: float, index=None) -> torch.Tensor:
    """mj_integratePos: positions add the velocity times dt; free and ball quaternions turn by the body-frame angular velocity
    (right multiplication), then are normalized; hinges and slides add. ``index``: the hinges' and slides' qpos and qvel
    addresses as tensors on qpos's device (a world model's own, which torch.compile reads as buffers), else built once."""
    out = qpos.clone()
    for qa, da in lay.free:
        out[..., qa : qa + 3] = qpos[..., qa : qa + 3] + dt * qvel[..., da : da + 3]
        out[..., qa + 3 : qa + 7] = _turn(qpos[..., qa + 3 : qa + 7], dt * qvel[..., da + 3 : da + 6])
    for qa, da in lay.balls:
        out[..., qa : qa + 4] = _turn(qpos[..., qa : qa + 4], dt * qvel[..., da : da + 3])
    if lay.scalars:
        qi, di = _scalar_index(lay, qpos.device) if index is None else index
        out[..., qi] = qpos[..., qi] + dt * qvel[..., di]
    return out


def differentiate_pos(lay: MjcfLayout, qpos1: torch.Tensor, qpos2: torch.Tensor, index=None) -> torch.Tensor:
    """mj_differentiatePos with dt = 1: the tangent from qpos1 to qpos2 in qvel's layout (rotations in qpos1's body frame);
    ``index`` as integrate_pos's."""
    out = qpos1.new_zeros(qpos1.shape[:-1] + (lay.nv,))
    for qa, da in lay.free:
        out[..., da : da + 3] = qpos2[..., qa : qa + 3] - qpos1[..., qa : qa + 3]
        out[..., da + 3 : da + 6] = _between(qpos1[..., qa + 3 : qa + 7], qpos2[..., qa + 3 : qa + 7])
    for qa, da in lay.balls:
        out[..., da : da + 3] = _between(qpos1[..., qa : qa + 4], qpos2[..., qa : qa + 4])
    if lay.scalars:
        qi, di = _scalar_index(lay, qpos1.device) if index is None else index
        out[..., di] = qpos2[..., qi] - qpos1[..., qi]
    return out


def _turn(q: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    q1 = quat_mul(q, quat_exp(rot))
    return q1 / q1.norm(dim=-1, keepdim=True)


def _between(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return quat_log(quat_mul(quat_conj(q1 / q1.norm(dim=-1, keepdim=True)), q2 / q2.norm(dim=-1, keepdim=True)))


@functools.lru_cache(maxsize=64)
def _scalar_index(lay: MjcfLayout, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The hinges' and slides' qpos and qvel addresses on ``device``, built once per layout and device."""
    return (torch.tensor([s[0] for s in lay.scalars], device=device), torch.tensor([s[1] for s in lay.scalars], device=device))


def _norm(x: torch.Tensor) -> torch.Tensor:
    """|x| over the last axis with a finite gradient at zero."""
    return (x.square().sum(-1) + 1e-20).sqrt()


def signed_distance(kind: int, x: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """The signed distance (..., ) of points x (..., 3), in a geom's frame, to its surface (negative inside): a sphere of
    radius size[0], a capsule or cylinder of radius size[0] and half length size[1] along z, a box of half sizes size, or a
    plane (the frame's z = 0, unbounded; negative below it)."""
    if kind == PLANE:
        return x[..., 2]
    if kind == SPHERE:
        return _norm(x) - size[0]
    if kind == CAPSULE:
        return _norm(x - torch.nn.functional.pad(x[..., 2:].clamp(-size[1], size[1]), (2, 0))) - size[0]
    if kind == CYLINDER:
        d = torch.stack((_norm(x[..., :2]) - size[0], x[..., 2].abs() - size[1]), -1)
        return d.amax(-1).clamp_max(0.0) + _norm(d.clamp_min(0.0))
    if kind == BOX:
        d = x.abs() - size
        return d.amax(-1).clamp_max(0.0) + _norm(d.clamp_min(0.0))
    raise ValueError(f"signed_distance: geom type {kind}")


def distance_normal(kind: int, x: torch.Tensor, size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The signed distance (...,) of points x (..., 3), in a geom's frame, to its surface (negative inside) and the surface's unit
    normal (..., 3) at its point nearest to them (outwards): a sphere, a capsule or a cylinder along z, a box or a plane, with
    signed_distance's sizes."""
    if kind == PLANE:
        n = torch.zeros_like(x)
        n[..., 2] = 1.0
        return x[..., 2], n
    if kind == CYLINDER:
        rho = _norm(x[..., :2])
        radial = torch.nn.functional.pad(x[..., :2] / rho.unsqueeze(-1), (0, 1))  # away from the axis
        axial = torch.zeros_like(x)
        axial[..., 2] = torch.where(x[..., 2] >= 0, 1.0, -1.0)  # away from the middle plane
        dr, dz = rho - size[0], x[..., 2].abs() - size[1]
        out = torch.stack((dr, dz), -1).clamp_min(0.0)
        lo, deepest = _norm(out), torch.maximum(dr, dz)
        outside = (out[..., :1] * radial + out[..., 1:] * axial) / lo.unsqueeze(-1)  # from the side, a cap or the rim between them
        inside = torch.where((dr > dz).unsqueeze(-1), radial, axial)  # inside: the nearer of the side and the caps
        return deepest.clamp_max(0.0) + lo, torch.where((deepest > 0).unsqueeze(-1), outside, inside)
    if kind == SPHERE:
        far = _norm(x)
        return far - size[0], x / far.unsqueeze(-1)
    if kind == CAPSULE:
        a = x - torch.nn.functional.pad(x[..., 2:].clamp(-size[1], size[1]), (2, 0))
        far = _norm(a)
        return far - size[0], a / far.unsqueeze(-1)
    if kind == BOX:
        d = x.abs() - size
        out = d.clamp_min(0.0)
        lo, deepest = _norm(out), d.amax(-1)
        face = torch.nn.functional.one_hot(d.argmax(-1), 3).to(x.dtype)  # inside: the nearest face's
        n = torch.where((deepest > 0).unsqueeze(-1), out / lo.unsqueeze(-1), face) * torch.where(x >= 0, 1.0, -1.0)
        return deepest.clamp_max(0.0) + lo, n
    raise ValueError(f"distance_normal: geom type {kind} (a sphere, a capsule, a cylinder, a box or a plane)")


def _farthest(x: np.ndarray, n: int) -> np.ndarray:
    """The indices of n of the points x (N, 3), each the farthest from those before it; the first the nearest to their mean."""
    n = min(n, len(x))
    idx = np.empty(n, dtype=int)
    idx[0] = int(np.linalg.norm(x - x.mean(0), axis=1).argmin())
    d = np.linalg.norm(x - x[idx[0]], axis=1)
    for i in range(1, n):
        idx[i] = int(d.argmax())
        d = np.minimum(d, np.linalg.norm(x - x[idx[i]], axis=1))
    return idx


def _hull_spheres(pts: np.ndarray, spacing: float, rmax: float, cap: int, rng, tol: float = 2e-5) -> tuple[np.ndarray, np.ndarray]:
    """Spheres inscribed in the convex hull of pts (V, 3) that cover its surface: at points about ``spacing`` apart over its
    faces, each the largest sphere that touches the face there and passes no other face's plane by more than ``tol`` (at most
    ``rmax``; the tolerance keeps a finely faceted round surface's spheres at its own radius), and points (radius 0) along the
    edges where two faces meet at more than 25 degrees. More than ``cap`` spheres at that spacing: a wider spacing."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError as e:  # no other hull stands in for it
        raise ImportError("the contact inputs' spheres on a mesh, box, cylinder or ellipsoid need scipy's convex hull: "
                          "pip install 'ndm_gym[contacts]'") from e  # fmt: skip

    hull = ConvexHull(pts)
    nf, df = hull.equations[:, :3], -hull.equations[:, 3]  # n . x <= d inside
    tri = pts[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = {}
    for f, t in enumerate(hull.simplices):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            faces.setdefault((min(a, b), max(a, b)), []).append(f)
    sharp = [e for e, fs in faces.items() if len(fs) == 2 and nf[fs[0]] @ nf[fs[1]] < np.cos(np.radians(25.0))]
    length = float(sum(np.linalg.norm(pts[a] - pts[b]) for a, b in sharp))
    total = float(area.sum())
    s = max(spacing, (length + np.sqrt(length**2 + 4 * cap * total)) / (2 * cap))  # total / s^2 + length / s <= cap
    n = max(int(np.ceil(total / s**2)), 4)
    f = rng.choice(len(area), size=8 * n, p=area / total)  # eight times the points, drawn uniformly, then the farthest apart
    u, v = np.sqrt(rng.random(8 * n)), rng.random(8 * n)
    x = (1 - u)[:, None] * tri[f, 0] + (u * (1 - v))[:, None] * tri[f, 1] + (u * v)[:, None] * tri[f, 2]
    keep = _farthest(x, n)
    x, nk = x[keep], nf[f[keep]]
    slack, turn = np.maximum(df[None] - x @ nf.T, 0.0), 1.0 - nk @ nf.T  # each plane's room at x, and how far it turns from x's
    r = np.minimum(np.where(turn > 1e-6, (slack + tol) / np.maximum(turn, 1e-6), np.inf).min(1), rmax)
    c = x - r[:, None] * nk
    if sharp:
        e = np.concatenate([pts[a] + np.linspace(0, 1, max(int(np.ceil(np.linalg.norm(pts[b] - pts[a]) / s)), 1) + 1)[:, None] * (pts[b] - pts[a])
                            for a, b in sharp])  # fmt: skip
        e = e[_farthest(e, int(np.ceil(length / s)) + 8)]
        c, r = np.concatenate((c, e)), np.concatenate((r, np.zeros(len(e))))
    if (c @ nf.T + r[:, None] > df[None] + tol + 1e-6).any():
        raise RuntimeError("_hull_spheres: a sphere leaves the hull")
    return c, r


def contact_spheres(m, lay: MjcfLayout, spacing: float = 0.0025, rmax: float = 0.02, cap: int = 384, seed: int = 0, tol: float = 2e-5,
                    objects: tuple = (), static: tuple | None = None, pair_margin: bool = False) -> dict:  # fmt: skip
    """The robot's collision geoms as spheres, for the contact inputs (``MjcfWorldModel(contacts=...)``): a mesh's convex hull (as
    MuJoCo collides it), a box, a cylinder or an ellipsoid through ``_hull_spheres``, a capsule as spheres along its axis, a
    sphere as itself. Returns ``center`` (G, K, 3) in each geom's body's frame and ``radius`` (G, K), a geom's unused places a
    point 10 m off along every axis with radius 0 (the mark by which a model, or a saved fit, keeps them out of every gap);
    ``body`` (G,), each geom's body; ``count`` (G,); ``geom`` (G,), the geoms' ids in the model, in ``lay.robot_geoms``' order;
    and ``margin``, the largest contact margin among the objects' and the robot's collision geoms; with ``pair_margin`` the
    largest margin of a contact between one of the robot's collision geoms and one of the objects' or the static geoms, as the
    installed MuJoCo sets it from the two geoms' margins (``_pair_margin``): the margin of the simulator's contacts.

    ``objects``: names of bodies that are objects too although no free joint carries them (one that rides on slides or hinges):
    their subtrees' geoms are not the robot's, and the result holds ``tb``, their body ids, and ``tg`` (rows of 12: the body's
    place in ``tb``, then the type, position (3), quaternion (4) and size (3) of each of its collision geoms, a sphere, a capsule,
    a cylinder or a box). ``static``: names of collision geoms of the world body the robot's geoms' gaps are also read to (an
    empty tuple: every one of them): the result holds ``sg`` (rows of 11: type, position, quaternion, size in the world's
    frame; a plane, a sphere, a capsule, a cylinder or a box). By default neither: the layout's free objects alone."""
    import mujoco

    rng = np.random.default_rng(seed)
    obj_q = {qa for qa, _ in lay.objects}
    free_objects = [int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == FREE and int(m.jnt_qposadr[j]) in obj_q]
    names, named = tuple(objects), []
    for name in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b <= 0 or b in free_objects or b in named:
            raise ValueError(f"contact_spheres: object body {name!r}: no such body, the world, a free object or named twice")
        named.append(int(b))
    objects = free_objects + named

    def under(b, root):  # b in root's subtree
        while b > 0 and b != root:
            b = int(m.body_parentid[b])
        return b == root

    collides = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])  # noqa: E731
    geoms = [g for g in range(m.ngeom) if collides(g) and m.geom_bodyid[g] > 0 and not any(under(int(m.geom_bodyid[g]), o) for o in objects)]
    theirs = sum(any(under(b, o) for o in named) for b, _ in lay.robot_geoms)  # the layout counts the named bodies' geoms as the robot's
    if not geoms or len(geoms) != len(lay.robot_geoms) - theirs:
        raise ValueError(f"contact_spheres: {len(geoms)} robot collision geoms, the layout has {len(lay.robot_geoms) - theirs}")
    spheres = []
    for g in geoms:
        kind, size = int(m.geom_type[g]), m.geom_size[g]
        if kind == SPHERE:
            c, r = np.zeros((1, 3)), np.array([size[0]])
        elif kind == CAPSULE:
            n = max(int(np.ceil(2 * size[1] / (0.5 * spacing))) + 1, 2)
            c, r = np.zeros((n, 3)), np.full(n, size[0])
            c[:, 2] = np.linspace(-size[1], size[1], n)
        else:
            if kind == MESH:
                i = int(m.geom_dataid[g])
                pts = np.array(m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]], dtype=np.float64)
            elif kind == BOX:
                pts = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.float64) * size
            elif kind == CYLINDER:
                a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
                ring = np.stack((size[0] * np.cos(a), size[0] * np.sin(a)), 1)
                pts = np.concatenate([np.concatenate((ring, np.full((96, 1), z)), 1) for z in (-size[1], size[1])])
            elif kind == ELLIPSOID:
                k = np.arange(400) + 0.5
                z, a = 1 - 2 * k / 400, np.pi * (1 + 5**0.5) * k
                pts = np.stack((np.sqrt(1 - z**2) * np.cos(a), np.sqrt(1 - z**2) * np.sin(a), z), 1) * size
            else:
                raise ValueError(f"contact_spheres: geom {g} of type {kind} (a mesh, box, cylinder, ellipsoid, capsule or sphere)")
            c, r = _hull_spheres(pts, spacing, rmax, cap, rng, tol)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        spheres.append((m.geom_pos[g] + c @ R.reshape(3, 3).T, r))
    K = max(len(r) for _, r in spheres)
    center, radius = np.full((len(geoms), K, 3), UNUSED_PLACE), np.zeros((len(geoms), K))
    for i, (c, r) in enumerate(spheres):
        center[i, : len(r)], radius[i, : len(r)] = c, r
    og = [g for g in range(m.ngeom) if collides(g) and int(m.geom_bodyid[g]) in objects]
    row = lambda g: [float(m.geom_type[g]), *m.geom_pos[g], *m.geom_quat[g], *m.geom_size[g]]  # noqa: E731
    more, sgeoms = {}, []
    if named:
        tg = [[float(i)] + row(g) for i, b in enumerate(named) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and collides(g)]
        for i, name in enumerate(names):
            kinds = [int(r[1]) for r in tg if int(r[0]) == i]
            if not kinds or any(k not in (SPHERE, CAPSULE, CYLINDER, BOX) for k in kinds):
                raise ValueError(f"contact_spheres: object body {name!r} needs collision geoms of its own, each a sphere, a capsule, "
                                 "a cylinder or a box")  # fmt: skip
        below = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}" for g in range(m.ngeom)
                 if collides(g) and int(m.geom_bodyid[g]) not in named and any(under(int(m.geom_bodyid[g]), o) for o in named)]
        if below:  # never left out silently
            raise ValueError(f"contact_spheres: collision geoms {below} on bodies below the named objects are not read")
        more["tb"], more["tg"] = np.array(named), np.array(tg, dtype=np.float32)
    if static is not None:
        world = [g for g in range(m.ngeom) if collides(g) and m.geom_bodyid[g] == 0]
        sgeoms = world if not static else [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in static]
        bad = [name for name, g in zip(static, sgeoms) if g not in world] if static else []
        if bad or not sgeoms:
            raise ValueError(f"contact_spheres: static geoms {bad or 'none'}: each a collision geom of the world body (the model has {len(world)})")
        for g in sgeoms:
            if int(m.geom_type[g]) not in (PLANE, SPHERE, CAPSULE, CYLINDER, BOX):
                raise ValueError(f"contact_spheres: static geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or g} of type {int(m.geom_type[g])} "
                                 "(a plane, a sphere, a capsule, a cylinder or a box)")  # fmt: skip
        more["sg"] = np.array([row(g) for g in sgeoms], dtype=np.float32)
    margin = _pair_margin(m, geoms, og + sgeoms) if pair_margin else float(max(m.geom_margin[geoms + og + sgeoms].max(), 0.0))
    return {"center": center.astype(np.float32), "radius": radius.astype(np.float32), "body": np.array([int(m.geom_bodyid[g]) for g in geoms]),
            "count": np.array([len(r) for _, r in spheres]), "geom": np.array(geoms), "margin": margin} | more  # fmt: skip


def _pair_margin(m, geoms: list, targets: list) -> float:
    """The largest margin of a contact between one of ``geoms`` and one of ``targets`` (geom ids of the model ``m``), as the
    installed MuJoCo sets a contact's margin from its two geoms' (the larger in MuJoCo 2.x, their sum since:
    ``sim.mbrl_envs.contact_margin_rule``). Refused where a contact's margin, or the distance at which it acts, is not set that
    way: explicit contact pairs, the contact override, geoms with a gap."""
    import mujoco

    from ..sim.mbrl_envs import contact_margin_rule

    if not targets:
        raise ValueError("contact_spheres(pair_margin=True): no object's geom and no static geom for the robot's geoms to meet")
    if m.npair or int(m.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE) or m.geom_gap[geoms + targets].any():
        raise ValueError(f"contact_spheres(pair_margin=True): {m.npair} explicit contact pairs, the contact override or geoms with a gap: "
                         "the contacts' margin is not the two geoms' margins put together")  # fmt: skip
    ours, theirs = float(m.geom_margin[geoms].max()), float(m.geom_margin[targets].max())
    return max(ours + theirs if contact_margin_rule() == "sum" else max(ours, theirs), 0.0)


def check_gaps_revision(ck: dict, path: str):
    """Refuse a saved fit (``ck``: a fit_mjcf --save file's contents) whose contact inputs read static geoms and whose mark of
    the gaps' revision (``contact_gaps``, beside ``layout`` and ``args``) is not ``GAPS_REVISION``. A fit without the mark was
    fitted on the gaps before 9418a03d, where a geom's unused sphere places stood for its nearest point to a floor: it must be
    made again. A fit without static geoms, whose gaps are the same in both revisions, is not concerned, marked or not."""
    if "cs_sg" not in ck["state"]:
        return
    mark = ck.get("contact_gaps")
    if mark is None:
        raise ValueError(f"{path} has contact inputs against static geoms and no mark of their gaps' revision: it was fitted on the gaps "
                         "before 9418a03d (a geom's unused sphere places entered them) and must be made again")  # fmt: skip
    if mark != GAPS_REVISION:
        raise ValueError(f"{path} was fitted on revision {mark} of the contact gaps, this code's is {GAPS_REVISION}: it must be made again")


def mean_velocity_lead(m, substeps: int) -> float:
    """``MjcfWorldModel(pos_lead=...)`` for a control step of ``substeps`` physics steps of the model ``m``: where between the
    velocities before and after the step, v and v', the mean velocity of its position update lies under a constant
    acceleration. The integrators that advance the position with each physics step's new velocity (Euler, implicit,
    implicitfast) add up k velocities, v + i (v' - v) / k: (k + 1) / (2 k), which is 1 for one step. RK4 advances it with the
    mean of v and v': 1 / 2."""
    import mujoco

    k, integrator = int(substeps), int(m.opt.integrator)
    if k < 1:
        raise ValueError(f"mean_velocity_lead: {substeps} physics steps a control step")
    kinds = mujoco.mjtIntegrator
    if integrator == int(kinds.mjINT_RK4):
        return 0.5
    if integrator in [int(getattr(kinds, n)) for n in ("mjINT_EULER", "mjINT_IMPLICIT", "mjINT_IMPLICITFAST") if hasattr(kinds, n)]:
        return (k + 1) / (2 * k)
    raise ValueError(f"mean_velocity_lead: integrator {integrator} (Euler, RK4, implicit or implicitfast)")


# ---------------- the world model ----------------
class MjcfWorldModel(nn.Module):
    """Deterministic dynamics (qpos', qvel') = f(qpos, qvel, ctrl) over one step of ``dt`` (the model's timestep by default).
    ``backbone``: the network, "mlp" (``hidden``, ``act``, ``layer_norm``) or "rigidformer" (``MjcfRigidFormerNet`` on the token
    features of ``token_inputs``; ``rf``: its options, RF_DEFAULTS' where not given). Each backbone's options stay at their
    defaults with the other, and the contact geometry (``geometry``) is the MLP's: the tokens hold every body's pose.

    ``pos_lead`` (the accelerations' only): the next position is the position advanced by dt times v + pos_lead (v' - v). At
    1, the default, that is the next velocity, one semi-implicit Euler step of dt (SuperTrack's integration). A control step
    of several physics steps, or of RK4, advances the position with the mean of the velocities it passes through, which an
    exact v' does not give at 1 (``mean_velocity_lead``); the value is kept in the model's state (``pos_lead``) when it is
    not 1, and the step is then the model's own dt only."""

    KINDS = ("accel", "increment")  # SuperTrack's accelerations, SuperMimic's increments
    # body_poses and contact_gaps body by body, with elementwise 3 x 3 products and no scatter: the same values to rounding in
    # far fewer kernels once compiled (a planner's step at a few hundred worlds). Off: the level-by-level path every fit ran.
    fused = False
    BACKBONES = ("mlp", "rigidformer")

    def __init__(self, lay: MjcfLayout, hidden=HIDDEN, act: str = "silu", layer_norm: bool = True, amp: bool = True,
                 kind: str = "accel", geometry: bool = False, backbone: str = "mlp", rf: dict | None = None,
                 contacts: dict | None = None, pos_lead: float = 1.0):  # fmt: skip
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"MjcfWorldModel: kind {kind!r}, one of {self.KINDS}")
        if not 0.0 < pos_lead <= 1.0 or (pos_lead != 1.0 and kind != "accel"):
            raise ValueError(f"MjcfWorldModel(pos_lead={pos_lead}): within (0, 1], and 1 unless kind='accel' (the increments hold their positions)")
        self._lead = pos_lead != 1.0
        if self._lead:  # kept with the state: a saved fit names its own
            self.register_buffer("pos_lead", torch.tensor(float(pos_lead), dtype=torch.float32))
        if backbone not in self.BACKBONES:
            raise ValueError(f"MjcfWorldModel: backbone {backbone!r}, one of {self.BACKBONES}")
        if backbone == "mlp" and rf is not None:
            raise ValueError("MjcfWorldModel: rf configures the RigidFormer backbone (backbone='rigidformer'), not the MLP")
        if backbone == "rigidformer":
            if tuple(hidden) != HIDDEN or act != "silu" or not layer_norm:
                raise ValueError("MjcfWorldModel(backbone='rigidformer'): hidden, act and layer_norm configure the MLP; the RigidFormer takes rf")
            if geometry:
                raise ValueError("MjcfWorldModel(backbone='rigidformer'): the contact geometry inputs are the MLP's (tokens hold the bodies' poses)")
            rf = {**RF_DEFAULTS, **(rf or {})}
            if set(rf) != set(RF_DEFAULTS) or rf["point"] not in ("com", "origin"):
                raise ValueError(f"MjcfWorldModel: rf {rf}: the options are {sorted(RF_DEFAULTS)}, point 'com' or 'origin'")
        self.lay, self.amp, self.kind = lay, amp, kind  # amp: the network under bf16 autocast on a GPU (core.nets.run_mlp)
        self.geometry, self.backbone, self.contacts = geometry, backbone, contacts is not None
        if geometry:
            self._init_geometry(lay)
        if self.contacts:
            self._init_contacts(lay, contacts)
        gear = [0.0 if q < 0 else g for q, g in zip(lay.act_q, lay.gear)]  # a tendon actuator's gear is in its tendon's terms
        self.register_buffer("gear", torch.tensor(gear, dtype=torch.float32))
        self.register_buffer("gain", torch.tensor(lay.gain, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(lay.bias, dtype=torch.float32).reshape(lay.nu, 3))
        inf = float("inf")
        self.register_buffer("ctrl_lo", torch.tensor([r[0] if r else -inf for r in lay.ctrl_range], dtype=torch.float32))
        self.register_buffer("ctrl_hi", torch.tensor([r[1] if r else inf for r in lay.ctrl_range], dtype=torch.float32))
        self.register_buffer("force_lo", torch.tensor([r[0] if r else -inf for r in lay.force_range], dtype=torch.float32))
        self.register_buffer("force_hi", torch.tensor([r[1] if r else inf for r in lay.force_range], dtype=torch.float32))
        self.register_buffer("act_q", torch.tensor([max(x, 0) for x in lay.act_q], dtype=torch.long))
        self.register_buffer("act_d", torch.tensor([max(x, 0) for x in lay.act_d], dtype=torch.long))
        terms = [(a, qa, da, c) for a, ws in lay.tendons for qa, da, c in ws]  # the fixed tendons' joints, per actuator
        for k, i, dt in (("ten_act", 0, torch.long), ("ten_q", 1, torch.long), ("ten_d", 2, torch.long), ("ten_coef", 3, torch.float32)):
            self.register_buffer(k, torch.tensor([t[i] for t in terms], dtype=dt), persistent=False)
        cp = lay.couplings  # the joint equalities: y's addresses, x's (index 0 and a zero weight where there is no x), y0, x0, a
        for k, i, dt in (("cp_yq", 0, torch.long), ("cp_yd", 1, torch.long), ("cp_xq", 2, torch.long), ("cp_xd", 3, torch.long)):
            self.register_buffer(k, torch.tensor([max(c[i], 0) for c in cp], dtype=dt), persistent=False)
        self.register_buffer("cp_has", torch.tensor([float(c[2] >= 0) for c in cp], dtype=torch.float32), persistent=False)
        self.register_buffer("cp_y0", torch.tensor([c[4] for c in cp], dtype=torch.float32), persistent=False)
        self.register_buffer("cp_x0", torch.tensor([c[5] for c in cp], dtype=torch.float32), persistent=False)
        self.register_buffer("cp_a", torch.tensor([c[6] for c in cp], dtype=torch.float32).reshape(len(cp), 5), persistent=False)
        # the hinges' and slides' features gathered at once, in the order of a loop over the joints: per joint sin and cos
        # of an angle with no limits, else the position (none for a planar root's horizontal slide), then the velocity
        # (columns of [q, sin q, cos q, qdot])
        S = len(lay.scalars)
        order = []
        for i, (qa, _, wrap) in enumerate(lay.scalars):
            if qa not in lay.planar:
                order += [S + i, 2 * S + i] if wrap else [i]
            order.append(3 * S + i)
        self.register_buffer("sc_q", torch.tensor([x[0] for x in lay.scalars], dtype=torch.long), persistent=False)
        self.register_buffer("sc_d", torch.tensor([x[1] for x in lay.scalars], dtype=torch.long), persistent=False)
        self.register_buffer("sc_order", torch.tensor(order, dtype=torch.long), persistent=False)
        if backbone == "rigidformer":
            self.rf_point = rf.pop("point")
            if not hasattr(self, "_levels"):
                self._init_kinematics(lay)
            self._init_tokens(lay)
        q0 = torch.zeros(1, lay.nq)
        q0[:, self._quat_w(lay)] = 1.0  # unit quaternions
        self.n_in = self.inputs(q0, torch.zeros(1, lay.nv), torch.zeros(1, lay.nu)).shape[-1]
        n_out = lay.nv if kind == "accel" else 2 * lay.nv
        if backbone == "rigidformer":  # its normalizer also passes the token positions on, for the rotary attention
            self.net = MjcfRigidFormerNet(lay, kind, self.tok_dim, **rf)
            self.in_norm = self.net.make_input_norm()
        else:
            self.net = mlp(self.n_in, list(hidden), n_out, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.n_in)
        self.out_norm = RunningNorm(n_out, eps=1e-4 / lay.dt if kind == "accel" else 1e-4)  # increments: SuperMimic's floor
        self.err_norm = RunningNorm(2 * lay.nv, eps=1e-4)  # the spread of one true step's increments: the error's units
        self.register_buffer("err_floor", torch.tensor(lay.floors(), dtype=torch.float32))

    @staticmethod
    def _quat_w(lay: MjcfLayout) -> list[int]:
        return [qa + 3 for qa, _ in lay.free] + [qa for qa, _ in lay.balls]

    def heading(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The anchor's heading frame H (..., 3, 3) and horizontal position (..., 3); the world's with no anchor."""
        if self.lay.anchor is None:
            eye = torch.eye(3, device=qpos.device, dtype=qpos.dtype).expand(qpos.shape[:-1] + (3, 3))
            return eye, qpos.new_zeros(qpos.shape[:-1] + (3,))
        qa = self.lay.anchor[0]
        H = yaw_mat(quat_to_mat(qpos[..., qa + 3 : qa + 7]))
        return H, torch.cat((qpos[..., qa : qa + 2], torch.zeros_like(qpos[..., :1])), -1)

    def servo(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each actuator's force (MuJoCo's: gain ctrl + b0 + b1 length + b2 velocity, ctrl and force clipped to their ranges)
        and the clipped control's offset from the actuator's length, gear times the joint's position."""
        u = torch.maximum(torch.minimum(ctrl, self.ctrl_hi), self.ctrl_lo)
        length, vel = self.gear * qpos[..., self.act_q], self.gear * qvel[..., self.act_d]
        if self.lay.tendons:  # a fixed tendon's actuator: its gear times the sum of its joints' coefficient times their values
            length = length.index_add(-1, self.ten_act, qpos[..., self.ten_q] * self.ten_coef)
            vel = vel.index_add(-1, self.ten_act, qvel[..., self.ten_d] * self.ten_coef)
        f = self.gain * u + self.bias[:, 0] + self.bias[:, 1] * length + self.bias[:, 2] * vel
        return torch.maximum(torch.minimum(f, self.force_hi), self.force_lo), u - length

    def inputs(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """The features of a state and its controls, invariant to the anchor's heading and horizontal position (the RigidFormer's:
        ``token_inputs``)."""
        if self.backbone == "rigidformer":
            return self.token_inputs(qpos, qvel, ctrl)
        lay = self.lay
        H, p0 = self.heading(qpos)
        Ht = H.transpose(-1, -2)
        rot = lambda x: (Ht @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
        parts = []
        for qa, da in lay.free:
            R = quat_to_mat(qpos[..., qa + 3 : qa + 7])
            parts += [rot(qpos[..., qa : qa + 3] - p0), mat_to_6d(Ht @ R), rot(qvel[..., da : da + 3]), qvel[..., da + 3 : da + 6]]
        for qa, da in lay.balls:
            parts += [mat_to_6d(quat_to_mat(qpos[..., qa : qa + 4])), qvel[..., da : da + 3]]
        if lay.scalars:
            q = qpos[..., self.sc_q]
            parts.append(torch.cat((q, torch.sin(q), torch.cos(q), qvel[..., self.sc_d]), -1)[..., self.sc_order])
        if lay.nu:
            force, offset = self.servo(qpos, qvel, ctrl)
            parts += [offset, force]
        if self.geometry:
            parts.append(self.contact_geometry(qpos))
        if self.contacts:
            parts.append(self.contact_inputs(qpos, qvel))
        return torch.cat(parts, -1)

    def _init_geometry(self, lay: MjcfLayout):
        """The buffers of the contact geometry inputs, refused where they would not hold: no object, no robot collision geom, an
        object without a primitive collision geom, or a robot geom on a mocap body's subtree (its pose is not in qpos)."""
        if not lay.objects:
            raise ValueError("MjcfWorldModel(geometry=True): the model has no object (a free joint other than the anchor)")
        if not lay.robot_geoms:
            raise ValueError("MjcfWorldModel(geometry=True): the robot has no collision geom")
        if not all(lay.object_geoms):
            raise ValueError("MjcfWorldModel(geometry=True): an object has no collision geom of a primitive type (sphere, capsule, cylinder, box)")
        if any(lay.object_unread):  # never left out silently
            raise ValueError(f"MjcfWorldModel(geometry=True): the objects' collision geoms {[n for o in lay.object_unread for n in o]} are not "
                             "read (only the spheres, capsules, cylinders and boxes of an object's own body are)")  # fmt: skip
        mocap = [False] + [bool(b[3]) for b in lay.tree]  # per body, the world first: on a mocap body's subtree
        for i, b in enumerate(lay.tree):
            mocap[i + 1] = mocap[i + 1] or mocap[b[0]]
        if any(mocap[b] for b, _ in lay.robot_geoms):
            raise ValueError("MjcfWorldModel(geometry=True): a robot collision geom hangs from a mocap body")
        self._init_kinematics(lay)
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        rot = lambda q: quat_to_mat(torch.tensor(q, dtype=torch.float64)).float()  # rounded once  # noqa: E731
        self.register_buffer("rg_body", torch.tensor([b for b, _ in lay.robot_geoms], dtype=torch.long), persistent=False)
        self.register_buffer("rg_pos", f32([p for _, p in lay.robot_geoms]), persistent=False)
        geoms = [g for o in lay.object_geoms for g in o]
        self._og = [[(g[0], sum(len(x) for x in lay.object_geoms[:i]) + c) for c, g in enumerate(o)] for i, o in enumerate(lay.object_geoms)]
        self.register_buffer("og_pos", f32([g[1] for g in geoms]), persistent=False)
        self.register_buffer("og_rot", rot([g[2] for g in geoms]), persistent=False)
        self.register_buffer("og_size", f32([g[3] for g in geoms]), persistent=False)

    def _init_kinematics(self, lay: MjcfLayout):
        """The buffers of ``body_poses``: the free bodies' qpos and qvel addresses, and the other bodies level by level."""
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        rot = lambda q: quat_to_mat(torch.tensor(q, dtype=torch.float64)).float()  # rounded once  # noqa: E731
        long = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
        qv = {qa: da for qa, da in lay.free + lay.balls} | {qa: da for qa, da, _ in lay.scalars}  # each joint's qvel address
        tree, self._nbody = lay.tree, len(lay.tree) + 1
        depth = [0] * self._nbody
        for i, b in enumerate(tree):
            depth[i + 1] = depth[b[0]] + 1
        free = [i + 1 for i, b in enumerate(tree) if b[4] and b[4][0][0] == FREE]  # a free joint is its body's only joint
        self.register_buffer("kin_free", long(free), persistent=False)
        self.register_buffer("kin_free_q", long([tree[b - 1][4][0][1] for b in free]), persistent=False)
        self.register_buffer("kin_free_d", long([qv[tree[b - 1][4][0][1]] for b in free]), persistent=False)
        # the other bodies level by level (parents first), each level's joints in slots (a body's first joint, second, ...) and
        # each slot's joints grouped by type, so that one level costs a few batched operations
        self._levels = []
        for depth_l in range(1, max(depth) + 1):
            bodies = [b for b in range(1, self._nbody) if depth[b] == depth_l and b not in free]
            if not bodies:
                continue
            k, groups = len(self._levels), []
            for name, x in (("body", long(bodies)), ("parent", long([tree[b - 1][0] for b in bodies])),
                            ("pos", f32([tree[b - 1][1] for b in bodies])), ("rot", rot([tree[b - 1][2] for b in bodies]))):  # fmt: skip
                self.register_buffer(f"kin{k}_{name}", x, persistent=False)
            for slot in range(max(len(tree[b - 1][4]) for b in bodies)):
                for jt in (HINGE, SLIDE, BALL):
                    rows = [i for i, b in enumerate(bodies) if len(tree[b - 1][4]) > slot and tree[b - 1][4][slot][0] == jt]
                    if not rows:
                        continue
                    js, g = [tree[bodies[i] - 1][4][slot] for i in rows], len(groups)
                    for name, x in (("rows", long(rows)), ("q", long([j[1] for j in js])), ("d", long([qv[j[1]] for j in js])),
                                    ("axis", f32([j[2] for j in js])), ("anchor", f32([j[3] for j in js])),
                                    ("q0", f32([j[4] for j in js]))):  # fmt: skip
                        self.register_buffer(f"kin{k}_{g}_{name}", x, persistent=False)
                    groups.append(jt)
            self._levels.append((k, groups))
        # the same tree body by body, for the fused path: (parent, ((type, qpos address, qvel address, joint row), ...))
        joints = [j for b in tree for j in b[4]]
        self._fk, n = [], 0
        for b in tree:
            self._fk.append((b[0], tuple((j[0], j[1], qv[j[1]], n + i) for i, j in enumerate(b[4]))))
            n += len(b[4])
        for name, x in (("pos", f32([[0.0] * 3] + [b[1] for b in tree])), ("rot", rot([[1.0, 0.0, 0.0, 0.0]] + [b[2] for b in tree])),
                        ("axis", f32([j[2] for j in joints]).reshape(-1, 3)), ("anchor", f32([j[3] for j in joints]).reshape(-1, 3)),
                        ("q0", f32([j[4] for j in joints]))):  # fmt: skip
            self.register_buffer(f"fk_{name}", x, persistent=False)

    def body_poses(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Every body's world position (..., nbody, 3) and rotation (..., nbody, 3, 3), the world's first, as mj_kinematics
        computes them from qpos (a mocap body at its model pose), one level of the tree at a time. With ``qvel``, also every
        body's linear velocity at its origin and angular velocity (..., nbody, 3), in the world frame, as mj_objectVelocity gives
        them for mjOBJ_XBODY: a free joint's linear velocity is qvel's and its angular one qvel's turned from the body's frame;
        on a body's way out from its parent's frame, each joint in turn adds its own, about its anchor: a hinge its axis times its
        rate, a ball its velocity in the frame after its rotation, a slide its axis times its rate as a linear velocity."""
        if self.fused:
            out = self._fk_lists(qpos, qvel)
            return tuple(torch.stack(x, -3 if i == 1 else -2) for i, x in enumerate(out))
        lead = qpos.shape[:-1]
        P = qpos.new_zeros(lead + (self._nbody, 3))
        # contiguous: torch 2.11's Inductor miscompiles an index_copy into the expanded identity (the compiled poses were wrong)
        R = torch.eye(3, dtype=qpos.dtype, device=qpos.device).expand(lead + (self._nbody, 3, 3)).contiguous()
        mv = lambda M, x: (M @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
        vel = qvel is not None
        if vel:
            V, W = qpos.new_zeros(lead + (self._nbody, 3)), qpos.new_zeros(lead + (self._nbody, 3))
        if len(self.kin_free):
            qa = self.kin_free_q.unsqueeze(-1)
            P = P.index_copy(-2, self.kin_free, qpos[..., qa + torch.arange(3, device=qpos.device)])
            R = R.index_copy(-3, self.kin_free, quat_to_mat(qpos[..., qa + 3 + torch.arange(4, device=qpos.device)]))
            if vel:
                da = self.kin_free_d.unsqueeze(-1)
                V = V.index_copy(-2, self.kin_free, qvel[..., da + torch.arange(3, device=qpos.device)])
                W = W.index_copy(-2, self.kin_free, mv(R[..., self.kin_free, :, :], qvel[..., da + 3 + torch.arange(3, device=qpos.device)]))
        for k, groups in self._levels:
            parent = getattr(self, f"kin{k}_parent")
            p = P[..., parent, :] + mv(R[..., parent, :, :], getattr(self, f"kin{k}_pos"))
            r = R[..., parent, :, :] @ getattr(self, f"kin{k}_rot")
            if vel:  # the parent's motion carried to the body's origin
                w = W[..., parent, :]
                v = V[..., parent, :] + torch.cross(w, p - P[..., parent, :], dim=-1)
            for g, jt in enumerate(groups):
                rows, qa, axis, anchor, q0 = (getattr(self, f"kin{k}_{g}_{n}") for n in ("rows", "q", "axis", "anchor", "q0"))
                pr, rr = p[..., rows, :], r[..., rows, :, :]
                if vel:
                    da, vr, wr, p_in = getattr(self, f"kin{k}_{g}_d"), v[..., rows, :], w[..., rows, :], pr
                if jt == SLIDE:
                    pr = pr + mv(rr, axis) * (qpos[..., qa] - q0).unsqueeze(-1)
                    if vel:
                        vr = vr + torch.cross(wr, pr - p_in, dim=-1) + mv(rr, axis) * qvel[..., da].unsqueeze(-1)
                else:
                    a = pr + mv(rr, anchor)
                    loc = quat_to_mat(qpos[..., qa.unsqueeze(-1) + torch.arange(4, device=qpos.device)]) if jt == BALL else \
                        axis_angle_to_mat(axis, qpos[..., qa] - q0)  # fmt: skip
                    rr = rr @ loc
                    pr = a - mv(rr, anchor)
                    if vel:
                        wj = mv(rr, qvel[..., da.unsqueeze(-1) + torch.arange(3, device=qpos.device)]) if jt == BALL else \
                            mv(rr, axis) * qvel[..., da].unsqueeze(-1)  # fmt: skip
                        vr = vr + torch.cross(wr, pr - p_in, dim=-1) + torch.cross(wj, pr - a, dim=-1)
                        wr = wr + wj
                p, r = p.index_copy(-2, rows, pr), r.index_copy(-3, rows, rr)
                if vel:
                    v, w = v.index_copy(-2, rows, vr), w.index_copy(-2, rows, wr)
            body = getattr(self, f"kin{k}_body")
            P, R = P.index_copy(-2, body, p), R.index_copy(-3, body, r)
            if vel:
                V, W = V.index_copy(-2, body, v), W.index_copy(-2, body, w)
        return (P, R, V, W) if vel else (P, R)

    def _fk_lists(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> tuple[list, ...]:
        """``body_poses``' values as one tensor per body (the world's first), body by body: each body a short chain of
        elementwise operations on its parent's tensors (``core.kinematics.mat_vec``, ``mat_mat``), which torch.compile fuses."""
        vel = qvel is not None
        zero = qpos.new_zeros(qpos.shape[:-1] + (3,))
        P, R = [zero], [torch.eye(3, dtype=qpos.dtype, device=qpos.device).expand(qpos.shape[:-1] + (3, 3))]
        V, W = [zero], [zero]
        for b, (parent, joints) in enumerate(self._fk, start=1):
            if joints and joints[0][0] == FREE:  # a free joint is its body's only joint
                _, qa, da, _ = joints[0]
                p, r = qpos[..., qa : qa + 3], quat_to_mat(qpos[..., qa + 3 : qa + 7])
                if vel:
                    v, w = qvel[..., da : da + 3], mat_vec(r, qvel[..., da + 3 : da + 6])
            else:
                p, r = P[parent] + mat_vec(R[parent], self.fk_pos[b]), mat_mat(R[parent], self.fk_rot[b])
                if vel:  # the parent's motion carried to the body's origin
                    w = W[parent]
                    v = V[parent] + torch.cross(w, p - P[parent], dim=-1)
                for jt, qa, da, j in joints:
                    axis, anchor, p_in = self.fk_axis[j], self.fk_anchor[j], p
                    if jt == SLIDE:
                        p = p + mat_vec(r, axis) * (qpos[..., qa] - self.fk_q0[j]).unsqueeze(-1)
                        if vel:
                            v = v + torch.cross(w, p - p_in, dim=-1) + mat_vec(r, axis) * qvel[..., da].unsqueeze(-1)
                    else:
                        a = p + mat_vec(r, anchor)
                        loc = quat_to_mat(qpos[..., qa : qa + 4]) if jt == BALL else axis_angle_to_mat(axis, qpos[..., qa] - self.fk_q0[j])
                        r = mat_mat(r, loc)
                        p = a - mat_vec(r, anchor)
                        if vel:
                            wj = mat_vec(r, qvel[..., da : da + 3]) if jt == BALL else mat_vec(r, axis) * qvel[..., da].unsqueeze(-1)
                            v = v + torch.cross(w, p - p_in, dim=-1) + torch.cross(wj, p - a, dim=-1)
                            w = w + wj
            P.append(p)
            R.append(r)
            if vel:
                V.append(v)
                W.append(w)
        return (P, R, V, W) if vel else (P, R)

    def _init_tokens(self, lay: MjcfLayout):
        """The RigidFormer's token constants, one token per body after the world in the tree's order: the body's centre of mass in
        its frame, and per slot (its hinges and slides in order; ``n_slot`` slots on every body, the most any body has) the
        joint's addresses, the kind of its position features and the actuators on it (``n_act`` per slot, the most any joint
        has): a joint transmission's, or a fixed tendon's through the joint. Refused for a model with mocap bodies, whose poses
        are not in qpos."""
        if any(b[3] for b in lay.tree):
            raise ValueError("MjcfWorldModel(backbone='rigidformer'): the model has mocap bodies, whose poses are not in qpos")
        if len(lay.ipos) != len(lay.tree):
            raise ValueError(f"MjcfWorldModel(backbone='rigidformer'): the layout holds {len(lay.ipos)} centres of mass for {len(lay.tree)} bodies")
        acts = {}  # each hinge's or slide's actuators, by qpos address
        for a, qa in enumerate(lay.act_q):
            if qa >= 0:
                acts.setdefault(qa, []).append(a)
        for a, ws in lay.tendons:
            for qa, _, _ in ws:
                if a not in acts.setdefault(qa, []):
                    acts[qa].append(a)
        scal = {qa: (da, wrap) for qa, da, wrap in lay.scalars}
        slots = [[j[1] for j in b[4] if j[0] in (SLIDE, HINGE)] for b in lay.tree]
        B, J, A = len(lay.tree), max((len(s) for s in slots), default=0), max((len(x) for x in acts.values()), default=0)
        q, d, act = np.zeros((B, J), np.int64), np.zeros((B, J), np.int64), np.zeros((B, J, A), np.int64)
        value, wrap, has, act_has = (np.zeros(s, np.float32) for s in ((B, J), (B, J), (B, J), (B, J, A)))
        for t, s in enumerate(slots):
            for i, qa in enumerate(s):
                q[t, i], d[t, i], has[t, i] = qa, scal[qa][0], 1.0
                (wrap if scal[qa][1] else value)[t, i] = 1.0  # sin and cos of an angle with no limits, else the position
                for c, a in enumerate(acts.get(qa, ())):
                    act[t, i, c], act_has[t, i, c] = a, 1.0
        for name, x in (("tok_ipos", np.asarray(lay.ipos, np.float32)), ("tok_q", q), ("tok_d", d), ("tok_value", value), ("tok_wrap", wrap),
                        ("tok_has", has), ("tok_act", act), ("tok_act_has", act_has)):  # fmt: skip
            self.register_buffer(name, torch.from_numpy(x), persistent=False)
        self.n_slot, self.n_act = J, A
        self.tok_dim = 3 + 6 + 3 + 3 + 3 + J * (4 + 2 * A)
        if self.contacts:  # each body's token also holds the contact inputs of its collision geom nearest to each object
            G = len(self.cs_body)
            per = [[g for g in range(G) if int(self.cs_body[g]) == t + 1] for t in range(B)]
            tab = np.full((B, max(max(len(x) for x in per), 1)), G, np.int64)  # G: no geom, read as one far from every object
            for t, x in enumerate(per):
                tab[t, : len(x)] = x
            n_obj = len(self._cs_obj)  # the movable objects; the static geoms add 13 columns each and no summed terms
            extra = CONTACT_FEATURES * self.n_targets + (12 * n_obj if self.contact_basis else 0)
            own = np.zeros((B, n_obj), np.float32)  # the summed contact terms go to the object's own token
            for i, b in enumerate(self._cs_obj):
                own[b - 1, i] = 1.0
            cols = (np.arange(B)[:, None] * (self.tok_dim + extra) + self.tok_dim + np.arange(extra)[None]).reshape(-1)
            for name, x in (("tok_geom", tab), ("tok_own", own), ("tok_contact_cols", cols)):
                self.register_buffer(name, torch.from_numpy(x), persistent=False)
            self.tok_dim += extra

    def token_inputs(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """The RigidFormer's raw features (..., B F): B tokens, one per body after the world, of F = ``tok_dim`` features each, in
        the frame of ``inputs`` (the anchor's heading frame about its ground projection, heights absolute): the body's position (its
        centre of mass; with the option point="origin" its frame's origin), its 6-D rotation, its linear velocity there and its
        angular velocity, and the angular velocity in the body's own frame (the free and ball joints' velocities' frame); then
        per slot the joint's position features as the MLP takes them (q, or sin q and cos q for a hinge without limits, the
        other columns zero), its velocity, and per actuator on it the offset and the force of ``servo``; empty slots are zeros."""
        H, p0 = self.heading(qpos)
        Ht = H.transpose(-1, -2).unsqueeze(-3)
        mv = lambda M, x: (M @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
        P, R, V, W = self.body_poses(qpos, qvel)
        P, R, V, W = P[..., 1:, :], R[..., 1:, :, :], V[..., 1:, :], W[..., 1:, :]  # the world is no token
        if self.rf_point == "com":
            c = mv(R, self.tok_ipos)
            P, V = P + c, V + torch.cross(W, c, dim=-1)
        parts = [mv(Ht, P - p0.unsqueeze(-2)), mat_to_6d(Ht @ R), mv(Ht, V), mv(Ht, W), mv(R.transpose(-1, -2), W)]
        if self.n_slot:
            q, wrap = qpos[..., self.tok_q], self.tok_wrap
            s = [torch.stack((q * self.tok_value, torch.sin(q) * wrap, torch.cos(q) * wrap, qvel[..., self.tok_d] * self.tok_has), -1)]
            if self.n_act:
                force, offset = self.servo(qpos, qvel, ctrl)
                s.append((torch.stack((offset[..., self.tok_act], force[..., self.tok_act]), -1) * self.tok_act_has.unsqueeze(-1)).flatten(-2))
            parts.append(torch.cat(s, -1).flatten(-2))
        if self.contacts:
            x, basis = self.contact_features(qpos, qvel)  # (..., O, G, 13), (..., O, 12)
            far = torch.zeros_like(x[..., :1, :])
            far[..., 0] = 4.0  # 20 mm off, in the gap's units
            xg = torch.cat((x, far), -2)[..., self.tok_geom, :]  # (..., O, B, M, 13): each body's geoms
            j = xg[..., 0].argmin(-1)  # its geom nearest to the object
            xb = xg.gather(-2, j[..., None, None].expand(j.shape + (1, x.shape[-1]))).squeeze(-2)
            parts.append(xb.transpose(-3, -2).flatten(-2))  # (..., B, O 13)
            if basis is not None:
                parts.append((self.tok_own.unsqueeze(-1) * basis.unsqueeze(-3)).flatten(-2))  # (..., B, O 12)
        return torch.cat(parts, -1).flatten(-2)

    def contact_geometry(self, qpos: torch.Tensor) -> torch.Tensor:
        """Per object: each robot collision geom's centre in the object's frame (3) and its signed distance to the nearest of the
        object's primitive collision geoms (1)."""
        P, R = self.body_poses(qpos)
        g = P[..., self.rg_body, :] + (R[..., self.rg_body, :, :] @ self.rg_pos.unsqueeze(-1)).squeeze(-1)  # (..., G, 3), world
        out = []
        for i, (qa, _) in enumerate(self.lay.objects):
            Ro = quat_to_mat(qpos[..., qa + 3 : qa + 7]).unsqueeze(-3)
            rel = ((g - qpos[..., None, qa : qa + 3]).unsqueeze(-2) @ Ro).squeeze(-2)  # Ro^T (g - p): the object's frame
            d = torch.stack([signed_distance(t, ((rel - self.og_pos[c]).unsqueeze(-2) @ self.og_rot[c]).squeeze(-2), self.og_size[c])
                             for t, c in self._og[i]], -1).amin(-1)  # fmt: skip
            out += [rel.flatten(-2), d]
        return torch.cat(out, -1)

    def _init_contacts(self, lay: MjcfLayout, spheres: dict):
        """The buffers of the contact inputs: the robot's collision geoms as spheres (``contact_spheres``), kept with the model's
        state, and the objects' bodies and primitive collision geoms: the layout's free objects, then the bodies the spheres name
        as objects (``tb``, ``tg``), then the world's static geoms they name (``sg``); see ``contact_spheres``. Refused without
        any of them or where a free object has no primitive collision geom."""
        tb, tg, sg = (None if spheres.get(k) is None else torch.as_tensor(np.asarray(spheres[k].cpu() if torch.is_tensor(spheres[k]) else spheres[k]))
                      for k in ("tb", "tg", "sg"))  # fmt: skip
        named = [] if tb is None else [int(b) for b in tb]
        if not lay.objects and not named and sg is None:
            raise ValueError("MjcfWorldModel(contacts=...): the model has no object (a free joint other than the anchor), and the spheres name "
                             "no object body and no static geom")  # fmt: skip
        if not all(lay.object_geoms):
            raise ValueError("MjcfWorldModel(contacts=...): every object needs collision geoms, each a sphere, a capsule, a cylinder or a box")
        if any(lay.object_unread):  # never left out silently
            raise ValueError(f"MjcfWorldModel(contacts=...): the objects' collision geoms {[n for o in lay.object_unread for n in o]} are not "
                             "read (only the spheres, capsules, cylinders and boxes of an object's own body are)")  # fmt: skip
        if (named or sg is not None) and self.geometry:
            raise ValueError("MjcfWorldModel(contacts=...): object bodies and static geoms named by the spheres do not go with geometry=True")
        if not hasattr(self, "_levels"):
            self._init_kinematics(lay)
        t = lambda k, kind: torch.as_tensor(np.asarray(spheres[k].cpu() if torch.is_tensor(spheres[k]) else spheres[k])).to(kind)  # noqa: E731
        center, radius, body = t("center", torch.float32), t("radius", torch.float32), t("body", torch.long)
        if center.shape[:2] != radius.shape or center.shape[0] != len(body) or not 0 < len(body) <= len(lay.robot_geoms):
            raise ValueError(f"MjcfWorldModel(contacts=...): spheres of {tuple(center.shape)}, {tuple(radius.shape)}, {len(body)} geoms; the robot has "
                             f"{len(lay.robot_geoms)} collision geoms")  # fmt: skip
        self.register_buffer("cs_center", center)
        self.register_buffer("cs_radius", radius)
        self.register_buffer("cs_unused", torch.zeros_like(radius, dtype=torch.bool), persistent=False)  # kept out of every gap
        self._mark_unused()
        count = spheres.get("count")  # contact_spheres' own; a saved fit's spheres have none
        if count is not None and not torch.equal(self.cs_unused, torch.arange(radius.shape[1])[None] >= t("count", torch.long)[:, None]):
            raise ValueError(f"MjcfWorldModel(contacts=...): the places marked unused (radius 0, {UNUSED_PLACE:g} m along every axis) are not "
                             "the places after each geom's count")  # fmt: skip
        self.register_load_state_dict_post_hook(MjcfWorldModel._mark_unused)  # a loaded state's spheres stand from then on
        self.register_buffer("cs_body", body)
        self._cs_bodies = body.tolist()  # the same on the host, for the fused path: read there, a step would copy from the device
        self.register_buffer("cs_margin", t("margin", torch.float32).reshape(()))
        free = {b[4][0][1]: i + 1 for i, b in enumerate(lay.tree) if b[4] and b[4][0][0] == FREE}  # qpos address: body
        self._cs_obj = [free[qa] for qa, _ in lay.objects] + named  # the movable objects' bodies
        if not self.geometry:  # the objects' primitive geoms, as _init_geometry keeps them
            f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
            rot = lambda q: quat_to_mat(torch.tensor(q, dtype=torch.float64)).float()  # noqa: E731
            per = list(lay.object_geoms)
            if named:  # the named bodies' geoms after the free objects', in the same form
                rows = tg.double().tolist()
                per += [tuple((int(r[1]), tuple(r[2:5]), tuple(r[5:9]), tuple(r[9:12])) for r in rows if int(r[0]) == i) for i in range(len(named))]
            geoms = [g for o in per for g in o]
            self._og = [[(g[0], sum(len(x) for x in per[:i]) + c) for c, g in enumerate(o)] for i, o in enumerate(per)]
            if geoms:
                self.register_buffer("og_pos", f32([g[1] for g in geoms]), persistent=False)
                self.register_buffer("og_rot", rot([g[2] for g in geoms]), persistent=False)
                self.register_buffer("og_size", f32([g[3] for g in geoms]), persistent=False)
        self._sg = []  # the static geoms: (type, row)
        if sg is not None:
            rows = sg.double().tolist()
            self._sg = [(int(r[0]), k) for k, r in enumerate(rows)]
            self.register_buffer("sg_pos", torch.tensor([r[1:4] for r in rows], dtype=torch.float32), persistent=False)
            self.register_buffer("sg_rot", quat_to_mat(torch.tensor([r[4:8] for r in rows], dtype=torch.float64)).float(), persistent=False)
            self.register_buffer("sg_size", torch.tensor([r[8:11] for r in rows], dtype=torch.float32), persistent=False)
            self.register_buffer("cs_sg", sg.float())  # kept with the state, so that a saved fit names its static geoms
        if named:
            self.register_buffer("cs_tb", tb.long())
            self.register_buffer("cs_tg", tg.float())
        self.contact_basis = bool(spheres.get("basis", False))  # the summed push and drag terms beside the geoms' own inputs
        self.register_buffer("cs_basis", torch.tensor(self.contact_basis))
        if self.contact_basis and not self._cs_obj:
            raise ValueError("MjcfWorldModel(contacts=...): the summed terms (basis) are a movable object's; static geoms have none")
        self.n_targets = len(self._cs_obj) + len(self._sg)  # per robot geom: 13 inputs for each
        self.n_contact = CONTACT_FEATURES * len(body) * self.n_targets + (12 * len(self._cs_obj) if self.contact_basis else 0)

    @torch.no_grad()
    def contact_gaps(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Per object (..., O, G): each robot collision geom's smallest gap to the object's primitive collision geoms (negative:
        they overlap), the smallest over the geom's own spheres of the centre's signed distance less the radius; the object's surface
        point nearest to that sphere and the unit normal there, in the object's frame (..., O, G, 3); and with ``qvel`` the
        velocity of the geom's body at that point less the object's own there, in the object's frame. After the objects, one
        entry per static geom the spheres name: the gap to it; its surface point nearest to the geom's nearest sphere, from the
        origin of the geom's own body; the normal there; and the velocity of that body at the point: these three in the frame
        of the geom's body (a floor does not move with the robot: its own frame would hold the robot's place in the world)."""
        if self.fused:
            return self._contact_gaps_fused(qpos, qvel)
        b = self.cs_body
        if qvel is None:
            P, R = self.body_poses(qpos)
        else:
            P, R, V, W = self.body_poses(qpos, qvel)
        c = P[..., b, None, :] + torch.einsum("...gij,gkj->...gki", R[..., b, :, :], self.cs_center)  # (..., G, K, 3), world
        gaps, points, normals, vels = [], [], [], []
        for i, ob in enumerate(self._cs_obj):
            po, Ro = P[..., ob, :], R[..., ob, :, :]
            rel = torch.einsum("...gki,...ij->...gkj", c - po[..., None, None, :], Ro)  # Ro^T (c - po): the object's frame
            d = torch.stack([signed_distance(t, (rel - self.og_pos[k]) @ self.og_rot[k], self.og_size[k]) for t, k in self._og[i]], -1).amin(-1)
            gap, j = self._nearest_sphere(d - self.cs_radius)  # (..., G)
            cj = rel.gather(-2, j[..., None, None].expand(j.shape + (1, 3))).squeeze(-2)  # the nearest sphere's centre
            dn, n = None, None
            for t, k in self._og[i]:  # the object's geom nearest to that centre: its distance and normal
                dk, nk = distance_normal(t, (cj - self.og_pos[k]) @ self.og_rot[k], self.og_size[k])
                nk = nk @ self.og_rot[k].T
                dn, n = (dk, nk) if dn is None else (torch.minimum(dn, dk), torch.where((dk < dn).unsqueeze(-1), nk, n))
            p = cj - dn.unsqueeze(-1) * n
            gaps.append(gap)
            points.append(p)
            normals.append(n)
            if qvel is not None:
                xw = po[..., None, :] + torch.einsum("...ij,...gj->...gi", Ro, p)
                vr = V[..., b, :] + torch.cross(W[..., b, :], xw - P[..., b, :], dim=-1)
                vo = V[..., ob, None, :] + torch.cross(W[..., ob, None, :].expand_as(xw), xw - po[..., None, :], dim=-1)
                vels.append(torch.einsum("...gi,...ij->...gj", vr - vo, Ro))
        for t, k in self._sg:
            rel = (c - self.sg_pos[k]) @ self.sg_rot[k]  # (..., G, K, 3): the static geom's frame
            gap, j = self._nearest_sphere(signed_distance(t, rel, self.sg_size[k]) - self.cs_radius)
            cj = rel.gather(-2, j[..., None, None].expand(j.shape + (1, 3))).squeeze(-2)
            dn, n = distance_normal(t, cj, self.sg_size[k])
            arm = (cj - dn.unsqueeze(-1) * n) @ self.sg_rot[k].T + self.sg_pos[k] - P[..., b, :]  # the surface point from the body's origin, world
            Rb = R[..., b, :, :]
            gaps.append(gap)
            points.append(torch.einsum("...gji,...gj->...gi", Rb, arm))
            normals.append(torch.einsum("...gji,...gj->...gi", Rb, n @ self.sg_rot[k].T))
            if qvel is not None:
                vels.append(torch.einsum("...gji,...gj->...gi", Rb, V[..., b, :] + torch.cross(W[..., b, :], arm, dim=-1)))
        out = (torch.stack(gaps, -2), torch.stack(points, -3), torch.stack(normals, -3))
        return out if qvel is None else out + (torch.stack(vels, -3),)

    def _contact_gaps_fused(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """``contact_gaps`` on ``_fk_lists``' per-body tensors, its small matrix products written elementwise: the same values
        to rounding."""
        vel = qvel is not None
        fk = self._fk_lists(qpos, qvel)
        bodies = self._cs_bodies
        P, R = torch.stack([fk[0][i] for i in bodies], -2), torch.stack([fk[1][i] for i in bodies], -3)  # the geoms' bodies: (..., G, .)
        if vel:
            V, W = torch.stack([fk[2][i] for i in bodies], -2), torch.stack([fk[3][i] for i in bodies], -2)
        vm = lambda x, M: (x.unsqueeze(-1) * M).sum(-2)  # the row vector x times M: M^T x  # noqa: E731
        c = P.unsqueeze(-2) + (R.unsqueeze(-3) * self.cs_center.unsqueeze(-2)).sum(-1)  # (..., G, K, 3), world
        gaps, points, normals, vels = [], [], [], []
        for i, ob in enumerate(self._cs_obj):
            po, Ro = fk[0][ob], fk[1][ob]
            rel = vm(c - po[..., None, None, :], Ro[..., None, None, :, :])  # Ro^T (c - po): the object's frame
            d = torch.stack([signed_distance(t, vm(rel - self.og_pos[k], self.og_rot[k]), self.og_size[k]) for t, k in self._og[i]], -1).amin(-1)
            gap, j = self._nearest_sphere(d - self.cs_radius)  # (..., G)
            cj = rel.gather(-2, j[..., None, None].expand(j.shape + (1, 3))).squeeze(-2)  # the nearest sphere's centre
            dn, n = None, None
            for t, k in self._og[i]:  # the object's geom nearest to that centre: its distance and normal
                dk, nk = distance_normal(t, vm(cj - self.og_pos[k], self.og_rot[k]), self.og_size[k])
                nk = mat_vec(self.og_rot[k], nk)
                dn, n = (dk, nk) if dn is None else (torch.minimum(dn, dk), torch.where((dk < dn).unsqueeze(-1), nk, n))
            p = cj - dn.unsqueeze(-1) * n
            gaps.append(gap)
            points.append(p)
            normals.append(n)
            if vel:
                xw = po[..., None, :] + mat_vec(Ro.unsqueeze(-3), p)
                vr = V + torch.cross(W, xw - P, dim=-1)
                vo = fk[2][ob][..., None, :] + torch.cross(fk[3][ob][..., None, :].expand_as(xw), xw - po[..., None, :], dim=-1)
                vels.append(vm(vr - vo, Ro.unsqueeze(-3)))
        for t, k in self._sg:
            rel = vm(c - self.sg_pos[k], self.sg_rot[k])  # (..., G, K, 3): the static geom's frame
            gap, j = self._nearest_sphere(signed_distance(t, rel, self.sg_size[k]) - self.cs_radius)
            cj = rel.gather(-2, j[..., None, None].expand(j.shape + (1, 3))).squeeze(-2)
            dn, n = distance_normal(t, cj, self.sg_size[k])
            arm = mat_vec(self.sg_rot[k], cj - dn.unsqueeze(-1) * n) + self.sg_pos[k] - P  # the surface point from the body's origin, world
            gaps.append(gap)
            points.append(vm(arm, R))
            normals.append(vm(mat_vec(self.sg_rot[k], n), R))
            if vel:
                vels.append(vm(V + torch.cross(W, arm, dim=-1), R))
        out = (torch.stack(gaps, -2), torch.stack(points, -3), torch.stack(normals, -3))
        return out if qvel is None else out + (torch.stack(vels, -3),)

    def _mark_unused(self, *_):
        """``cs_unused``: the sphere places ``contact_spheres`` left unused, read off the spheres themselves (radius 0 at 10 m along
        every axis), which is all a saved fit keeps of the counts; read again whenever a state is loaded. A fit saved before the
        unused places were left out of the gaps loads as it is; one fitted against a static geom saw other gaps then and is to
        be fitted again."""
        unused = (self.cs_radius == 0) & (self.cs_center == UNUSED_PLACE).all(-1)
        if unused.all(-1).any():
            raise ValueError("MjcfWorldModel(contacts=...): a geom has no sphere (each of its places is marked unused)")
        self.cs_unused, self._cs_pad = unused, bool(unused.any())
        if "cs_body" in self._buffers:  # a loaded state's geoms' bodies, on the host too
            self._cs_bodies = self.cs_body.tolist()

    def _nearest_sphere(self, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The smallest of each geom's sphere gaps d (..., G, K) and its place, over the geom's own spheres. The unused places
        (``contact_spheres``' point 10 m off) are left out: a floor plane or a wide box is nearer to that point than to the
        geom whenever the body's rotation carries it below the surface. A set without unused places takes the plain minimum."""
        return (d.masked_fill(self.cs_unused, torch.inf) if self._cs_pad else d).min(-1)

    @torch.no_grad()
    def contact_inputs(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Per object and robot collision geom, 13 values of order one (the input normalizer leaves them as they are, unless
        ``update_norms`` was given ``contact_scale``): the gap
        beyond the contact margin, clipped to -2 .. 20 mm, in units of 5 mm; exp(-gap / 0.5 mm), 1 in contact; the overlap in
        units of 0.1 mm, at most 20; and, each times exp(-gap / 2 mm), the object's nearest point in its frame in units of 3 cm,
        the normal there, and the geom's velocity relative to the object at that point in units of 0.2 m/s, with its normal part.
        No gradient passes through them. With the spheres' ``basis`` set, 12 more per object (see below). The objects are the
        free objects, then the bodies the spheres name as objects, then the static geoms they name, whose point, normal and
        velocity are in the frame of the robot geom's body (``contact_gaps``) and which have no 12 more."""
        x, basis = self.contact_features(qpos, qvel)
        return x.flatten(-3) if basis is None else torch.cat((x.flatten(-3), basis.flatten(-2)), -1)

    @torch.no_grad()
    def contact_features(self, qpos: torch.Tensor, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``contact_inputs``' values per object and geom (..., O, G, 13), and with the spheres' ``basis`` the 12 summed terms per
        object (..., O, 12), else None."""
        gap, p, n, v = self.contact_gaps(qpos, qvel)
        s = (gap - self.cs_margin).unsqueeze(-1)
        near = torch.exp(-s.clamp_min(0.0) / 0.002)
        touch = torch.exp(-s.clamp_min(0.0) / 0.0005)
        x = torch.cat((s.clamp(-0.002, 0.02) / 0.005, touch, (-s).clamp(0.0, 0.002) / 1e-4,
                       near * p / 0.03, near * n, near * v / 0.2, near * (v * n).sum(-1, keepdim=True) / 0.2), -1)  # fmt: skip
        if not self.contact_basis:
            return x, None
        if self._sg:  # the summed terms are the movable objects'
            p, n, v, touch = (y[..., : len(self._cs_obj), :, :] for y in (p, n, v, touch))
        # per object, summed over the geoms in contact: the drag of their relative velocity and its moment about the object's
        # origin, and the push of their approach along the normal and its moment (what a rigid contact passes on, to first order)
        pu, vu = p / 0.03, v / 0.2
        closing = (-(vu * n).sum(-1, keepdim=True)).clamp_min(0.0)
        return x, (touch * torch.cat((vu, torch.cross(pu, vu, dim=-1), closing * n, closing * torch.cross(pu, n, dim=-1)), -1)).sum(-2)

    def _rotate_linear(self, y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """``y`` with the free joints' linear parts turned by M (..., 3, 3): the first three components of each free joint in
        each of y's nv-long halves (one for the accelerations, two for the increments)."""
        if not self.lay.free:
            return y
        out = y.clone()
        for h in range(y.shape[-1] // self.lay.nv):
            for _, da in self.lay.free:
                o = h * self.lay.nv + da
                out[..., o : o + 3] = (M @ y[..., o : o + 3].unsqueeze(-1)).squeeze(-1)
        return out

    def targets(self, qpos: torch.Tensor, qvel: torch.Tensor, qpos1: torch.Tensor, qvel1: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        """The regression targets, what ``out_norm`` normalizes, the free joints' linear parts in the heading frame: the mean
        generalized accelerations over the step (``accel``), or the step's increments (``increment``: ``increments``)."""
        dt = self.lay.dt if dt is None else dt
        y = (qvel1 - qvel) / dt if self.kind == "accel" else self.increments(qpos, qvel, qpos1, qvel1)
        return self._rotate_linear(y, self.heading(qpos)[0].transpose(-1, -2))

    def increments(self, qpos: torch.Tensor, qvel: torch.Tensor, qpos1: torch.Tensor, qvel1: torch.Tensor) -> torch.Tensor:
        """One step's increments in the error's layout: the tangent from qpos to qpos1, then qvel1 - qvel."""
        return torch.cat((differentiate_pos(self.lay, qpos, qpos1, (self.sc_q, self.sc_d)), qvel1 - qvel), -1)

    def step(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor, dt: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The next state: the predicted accelerations integrated as MuJoCo's semi-implicit Euler integrates them (with
        ``pos_lead``, the position advanced with the velocity that far from the present one to the next), or the predicted
        increments applied (``increment``); with ``pos_lead`` or the increments the step is the model's timestep only."""
        if self.kind == "increment" and dt is not None and dt != self.lay.dt:
            raise ValueError(f"MjcfWorldModel(kind='increment'): a step of {dt} s; its increments are of {self.lay.dt} s")
        if self._lead and dt is not None and dt != self.lay.dt:
            raise ValueError(f"MjcfWorldModel(pos_lead=...): a step of {dt} s; its mean velocity is that of the model's {self.lay.dt} s")
        dt = self.lay.dt if dt is None else dt
        y = run_mlp(self.net, self.in_norm(self.inputs(qpos, qvel, ctrl)), self.amp)
        y = self._rotate_linear(y * self.out_norm.std() + self.out_norm.mean, self.heading(qpos)[0])
        if self.kind == "increment":
            nv = self.lay.nv
            return self.couple(integrate_pos(self.lay, qpos, y[..., :nv], 1.0, (self.sc_q, self.sc_d)), qvel + y[..., nv:])
        qvel1 = qvel + dt * y
        if self._lead:  # the mean of the velocities the control step passes through, not its last
            return self.couple(integrate_pos(self.lay, qpos, qvel + self.pos_lead * (qvel1 - qvel), dt, (self.sc_q, self.sc_d)), qvel1)
        return self.couple(integrate_pos(self.lay, qpos, qvel1, dt, (self.sc_q, self.sc_d)), qvel1)

    def couple(self, qpos: torch.Tensor, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The joint equalities satisfied exactly: each coupled joint's position set to the polynomial of the other's and its
        velocity to the polynomial's slope times the other's velocity (the state unchanged in a model without them)."""
        if not self.lay.couplings:
            return qpos, qvel
        a = self.cp_a
        x = (qpos[..., self.cp_xq] - self.cp_x0) * self.cp_has
        y = self.cp_y0 + a[:, 0] + x * (a[:, 1] + x * (a[:, 2] + x * (a[:, 3] + x * a[:, 4])))
        dy = (a[:, 1] + x * (2 * a[:, 2] + x * (3 * a[:, 3] + x * 4 * a[:, 4]))) * qvel[..., self.cp_xd] * self.cp_has
        return qpos.index_copy(-1, self.cp_yq, y), qvel.index_copy(-1, self.cp_yd, dy)

    def error(self, qpos_hat: torch.Tensor, qvel_hat: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """The prediction's error in the increments' layout, in units of the increments' spread (with floors)."""
        e = torch.cat((differentiate_pos(self.lay, qpos, qpos_hat, (self.sc_q, self.sc_d)), qvel_hat - qvel), -1)
        return e / torch.maximum(self.err_norm.std(), self.err_floor)

    @torch.no_grad()
    def update_norms(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor, qpos1: torch.Tensor, qvel1: torch.Tensor,
                     contact_scale: bool = False):  # fmt: skip
        """The normalizers from transitions (..., dims): the inputs, the targets and the increments' spread. The contact inputs
        keep their units, made for order one (a hand at its object). With ``contact_scale`` each of their columns is divided
        by its spread over the transitions of every such call so far where that exceeds one, and stays as it is where not (a
        foot on a floor: points and velocities of tens of units); the scales sit in the input normalizer, so that a saved
        fit holds them."""
        if contact_scale and not self.contacts:
            raise ValueError("MjcfWorldModel.update_norms(contact_scale=True): the model has no contact inputs")
        if self.contacts:  # the spheres of every transition at once would not fit: the normalizer updated chunk by chunk
            cols = self.tok_contact_cols if self.backbone == "rigidformer" else slice(self.n_in - self.n_contact, self.n_in)
            for i in range(0, qpos.flatten(0, -2).shape[0], 4096):
                x = self.inputs(*(t.flatten(0, -2)[i : i + 4096] for t in (qpos, qvel, ctrl)))
                self.in_norm.update(x)
                if contact_scale:  # each contact column's count, sum and sum of squares, over the calls
                    c = x[:, cols].double()
                    n, s1, s2 = getattr(self, "_cs_moments", None) or (0, 0.0, 0.0)
                    self._cs_moments = (n + len(c), s1 + c.sum(0), s2 + c.square().sum(0))
            self.in_norm.mean[cols] = 0.0  # the contact inputs are of order one by construction
            self.in_norm.var[cols] = 1.0
            if contact_scale:
                n, s1, s2 = self._cs_moments
                self.in_norm.var[cols] = (s2 / n - (s1 / n).square()).clamp_min(1.0).to(self.in_norm.var.dtype)
        elif self.geometry:  # the kinematics' intermediates of every transition at once would not fit: the inputs in chunks
            x = torch.cat([self.inputs(*(t.flatten(0, -2)[i : i + 65536] for t in (qpos, qvel, ctrl)))
                           for i in range(0, qpos.flatten(0, -2).shape[0], 65536)])  # fmt: skip
            self.in_norm.update(x)
        elif self.backbone == "rigidformer":  # nor would the token features: the normalizer updated chunk by chunk
            for i in range(0, qpos.flatten(0, -2).shape[0], 65536):
                self.in_norm.update(self.inputs(*(t.flatten(0, -2)[i : i + 65536] for t in (qpos, qvel, ctrl))))
        else:
            self.in_norm.update(self.inputs(qpos, qvel, ctrl))
        self.out_norm.update(self.targets(qpos, qvel, qpos1, qvel1))
        self.err_norm.update(self.increments(qpos, qvel, qpos1, qvel1))

    def window_loss(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor, valid: torch.Tensor | None = None,
                    weights: torch.Tensor | None = None, loss: str = "l1") -> torch.Tensor:  # fmt: skip
        """The open-loop loss of windows: ``qpos`` (T + 1, B, nq), ``qvel`` (T + 1, B, nv), ``ctrl`` (T, B, nu); ``valid`` (T, B)
        bool, False from a step whose next state does not follow (a reset) on. The mean L1 error over the valid steps; with
        ``weights`` (2 nv,), each step's error components averaged with those weights instead of equally; ``loss``: "l1", "l2"
        (squared) or "huber" (squared within one step's spread, absolute beyond)."""
        T = ctrl.shape[0]
        alive = torch.ones_like(qpos[0, :, 0], dtype=torch.bool) if valid is None else None
        q, v = qpos[0], qvel[0]
        total, count = qpos.new_zeros(()), qpos.new_zeros(())
        for t in range(T):
            ok = alive if valid is None else valid[: t + 1].all(0)
            q, v = self.step(q, v, ctrl[t])
            e = self.error(q, v, qpos[t + 1], qvel[t + 1])
            if loss == "l1":
                e = e.abs()
            elif loss == "l2":  # the squared error: its minimum is the mean outcome, where the absolute error's is the median one
                e = e.square()
            elif loss == "huber":  # squared within one step's spread, absolute beyond
                e = torch.where(e.abs() < 1.0, 0.5 * e.square(), e.abs() - 0.5)
            else:
                raise ValueError(f"window_loss: loss {loss!r} ('l1', 'l2' or 'huber')")
            e = e.mean(-1) if weights is None else (e * weights).sum(-1) / weights.sum()
            w = ok.to(e.dtype)
            total, count = total + (e * w).sum(), count + w.sum()
        return total / count.clamp_min(1.0)


class MjcfRigidFormerNet(nn.Module):
    """The RigidFormer backbone of ``MjcfWorldModel`` (``backbone="rigidformer"``), worldmodels.rigidformer's design for any MJCF.
    One token per body after the world (in the tree's order): its features (``MjcfWorldModel.token_inputs``) through a shared
    embedding, plus a learned embedding of the body; ``registers`` learned tokens at the origin. ``layers`` pre-norm blocks
    (rigidformer.Block: gated attention with QK-RMSNorm and 3-D rotary positions, then SwiGLU; no FiLM, as there is no global
    context) with a learned per-head bias by relation: the number of links between two bodies of one tree (each body that hangs
    from the world starts a tree), one class for two bodies of different trees, one for pairs with a register. The rotary
    positions are the token positions, rotated by frequencies from 1 to ``rope_max_freq`` rad/m. The read-out fuses each
    token's features after the embedding and after the last block (SwiGLU), and every body with degrees of freedom gives the
    outputs of its own, with one head per joint type (a free joint 6, a ball 3, a hinge or slide 1; once for the accelerations,
    twice for the increments: the tangent's and the velocity's), gathered into the MLP's output layout (qvel's, twice for the
    increments). The heads start at zero, as the MLP's last layer.

    x (..., B F + 3 B) from ``make_input_norm`` (the normalized features, then the token positions over POS_SCALE) -> (..., nv)
    or (..., 2 nv), fp32. The features run in the parameters' dtype (bf16 in a bf16 inference copy; fp32 parameters under the
    caller's autocast, with an fp32 residual stream), the rotary positions in fp32."""

    def __init__(self, lay: MjcfLayout, kind: str, tok_dim: int, *, dim: int, layers: int, heads: int, registers: int, rope_max_freq: float,
                 ffn_mult: float = 2.5):  # fmt: skip
        super().__init__()
        from .rigidformer import POS_SCALE, Block, Rope3d, tree_hops

        if dim < 1 or heads < 1 or dim % heads or dim // heads < 16:  # a narrower head has one rotary frequency, 1 rad/m
            raise ValueError(f"MjcfRigidFormerNet: width {dim} in {heads} heads; a multiple of the heads, at least 16 channels per head (two "
                             "rotary frequencies per coordinate)")  # fmt: skip
        if layers < 1 or registers < 0 or not rope_max_freq >= 1.0:
            raise ValueError(f"MjcfRigidFormerNet: {layers} blocks (at least 1), {registers} registers (at least 0), rotary frequencies up to "
                             f"{rope_max_freq} rad/m (at least 1)")  # fmt: skip
        B = len(lay.tree)
        self.n_tok, self.tok_dim, self.n_reg, self.in_dim, self.pos_scale = B, tok_dim, registers, B * tok_dim, POS_SCALE
        self.infer = False  # an inference copy's network (inference_copy): forward only
        self.embed = nn.Linear(tok_dim, dim)
        self.body_emb = nn.Parameter(0.5 * torch.randn(B, dim))
        self.registers = nn.Parameter(0.02 * torch.randn(1, registers, dim)) if registers else None
        self.blocks = nn.ModuleList(Block(dim, heads, ffn_mult, 0, fused_rope=True) for _ in range(layers))
        self.rope = Rope3d(6 * ((3 * (dim // heads) // 4) // 6), rope_max_freq)
        root = []  # each body's tree: the body that hangs from the world
        for b in range(1, B + 1):
            while lay.tree[b - 1][0] != 0:
                b = lay.tree[b - 1][0]
            root.append(b)
        same = torch.tensor(root)[:, None] == torch.tensor(root)[None, :]
        hops = tree_hops([-1] + [b[0] for b in lay.tree])[1:, 1:]  # the world as the trees' common root
        top = int(hops[same].max())
        rel = torch.full((B + registers, B + registers), top + 2, dtype=torch.long)  # pairs with a register
        rel[:B, :B] = torch.where(same, hops, top + 1)  # two bodies of different trees: one class
        self.register_buffer("rel", rel, persistent=False)
        self.rel_bias = nn.Parameter(torch.zeros(heads, top + 3))
        hidden = int(ffn_mult * dim)
        self.head_norm = nn.RMSNorm(2 * dim, eps=1e-6)
        self.head_w12, self.head_w3 = nn.Linear(2 * dim, 2 * hidden), nn.Linear(hidden, dim)
        # the read-out: per joint type, its bodies and its outputs per body; each output's place in the heads' outputs side by side
        k, nv = (1 if kind == "accel" else 2), lay.nv
        qv = {qa: da for qa, da in lay.free + lay.balls} | {qa: da for qa, da, _ in lay.scalars}
        joints = {name: [[j[1] for j in b[4] if j[0] in types] for b in lay.tree] for name, types in (("free", (FREE,)), ("ball", (BALL,)),
                                                                                                    ("joint", (SLIDE, HINGE)))}  # fmt: skip
        width = {"free": 6, "ball": 3, "joint": 1}
        read = sorted({t for js in joints.values() for t in range(B) if js[t]})
        out_index, base, self.head_names = [-1] * (k * nv), 0, []
        for name, js in joints.items():
            toks, n = [t for t in range(B) if js[t]], max(len(x) for x in js)
            if not toks:
                continue
            per = k * width[name] * n  # outputs per body: per joint, the tangent's (or acceleration's) components, then the velocity's
            for i, t in enumerate(toks):
                for s, qa in enumerate(js[t]):
                    for h in range(k):
                        for c in range(width[name]):
                            out_index[h * nv + qv[qa] + c] = base + i * per + (s * k + h) * width[name] + c
            self.head_names.append(name)
            setattr(self, f"{name}_head", nn.Linear(dim, per))
            self.register_buffer(f"{name}_rows", torch.tensor([read.index(t) for t in toks]), persistent=False)
            base += len(toks) * per
        if min(out_index) < 0:
            raise ValueError("MjcfRigidFormerNet: a degree of freedom belongs to no body's joint")
        for name in self.head_names:
            nn.init.zeros_(getattr(self, f"{name}_head").weight)
            nn.init.zeros_(getattr(self, f"{name}_head").bias)
        self.register_buffer("read", torch.tensor(read), persistent=False)  # the tokens read out: the bodies with degrees of freedom
        self.register_buffer("out_index", torch.tensor(out_index), persistent=False)

    def make_input_norm(self):
        """The normalizer of the token features (rigidformer.TokenInputNorm), which also passes the token positions on."""
        from .rigidformer import TokenInputNorm

        return TokenInputNorm(self.in_dim, (torch.arange(self.n_tok).unsqueeze(-1) * self.tok_dim + torch.arange(3)).flatten())

    def describe(self) -> str:
        """Width x blocks, heads and tokens, for the logs."""
        n = self.n_reg
        return f"RigidFormer {self.embed.out_features} x {len(self.blocks)} ({self.blocks[0].heads} heads; {self.n_tok} bodies, {n} register{'s' * (n != 1)})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.infer and torch.is_grad_enabled():
            raise RuntimeError("MjcfRigidFormerNet: an inference copy's network is forward only; run it under torch.no_grad()")
        lead, B = x.shape[:-1], self.n_tok
        x = x.reshape(-1, x.shape[-1])
        dt = self.embed.weight.dtype
        h = self.embed(x[:, : self.in_dim].unflatten(-1, (B, self.tok_dim)).to(dt)).to(dt) + self.body_emb  # an fp32 residual under autocast
        pos = self.pos_scale * x[:, self.in_dim :].unflatten(-1, (B, 3))
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(h.shape[0], -1, -1)), 1)
            pos = torch.cat((pos, pos.new_zeros(h.shape[0], self.n_reg, 3)), 1)
        cos, sin = self.rope(pos)
        bias = self.rel_bias[:, self.rel].unsqueeze(0)
        h0 = h
        for blk in self.blocks:
            h = blk(h, cos, sin, bias)
        u, g = self.head_w12(self.head_norm(torch.cat((h0, h), -1)[:, self.read])).chunk(2, -1)
        return self.heads(self.head_w3(u * nn.functional.silu(g))).float().reshape(lead + (-1,))

    def heads(self, z: torch.Tensor) -> torch.Tensor:
        """The outputs (N, nv) or (N, 2 nv) from the read tokens' fused features z (N, R, dim): each joint type's head on its
        bodies' tokens, gathered into the MLP's layout."""
        y = [getattr(self, f"{n}_head")(z[:, getattr(self, f"{n}_rows")]).flatten(1) for n in self.head_names]
        return torch.cat(y, -1)[:, self.out_index]


def inference_copy(wm: MjcfWorldModel, dtype: torch.dtype | None = torch.bfloat16) -> MjcfWorldModel:
    """A forward-only copy for simulation throughput: the network's parameters in ``dtype`` (bf16 or fp16: the network runs in that
    type end to end; the features, normalizers and integration stay fp32, and so do the RigidFormer's rotary positions), the
    MLP with its input cast to them, the RigidFormer casting its token features itself; or the fp32 model (None). On a GPU the
    copy's network is compared with the model's in fp32 and refused if it differs by more than twice the model's own
    bf16-autocast error on average or four times at the largest (worldmodels.g1_infer's rule)."""
    from .g1_infer import CastNet, network_errors

    m = copy.deepcopy(wm).eval().requires_grad_(False)
    if dtype is None:
        return m
    rf = wm.backbone == "rigidformer"
    if rf:
        for mod in m.net.modules():
            for name, prm in list(mod.named_parameters(recurse=False)):
                setattr(mod, name, nn.Parameter(prm.detach().to(dtype), requires_grad=False))
        m.net.infer, m.amp = True, False
    else:
        m.net, m.amp = CastNet(m.net.to(dtype), dtype), False
    dev = wm.in_norm.mean.device
    if dev.type == "cuda":
        width = wm.n_in + (3 * wm.net.n_tok if rf else 0)  # the RigidFormer's input also holds the token positions
        x = torch.randn(256, width, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
        with torch.no_grad():
            (fmax, fmean), (rmax, rmean) = network_errors(wm.net, m.net, x, amp=False)
        if not (fmean <= 2 * rmean + 1e-7 and fmax <= 4 * rmax + 1e-6):
            raise RuntimeError(f"inference_copy({dtype}): the network differs from the fp32 model by {fmax:.3g} at most, {fmean:.3g} on "
                               f"average; the model under bf16 autocast by {rmax:.3g}, {rmean:.3g}: refused")  # fmt: skip
    return m


@contextlib.contextmanager
def fused_compile(wm):
    """Around the first calls of a torch.compile'd step of ``wm`` (the calls that compile it): on the fused path the compiler
    runs a graph's independent kernels as one launch (``torch._inductor.config.combo_kernels``, the trainers'
    ``--combo_kernels``): the path's step is a few dozen small kernels, and a planner's step at a few hundred worlds is paid
    by the launch. The kernels' code is the same. Off the fused path nothing is set."""
    if not getattr(wm, "fused", False):
        yield
        return
    import torch._inductor.config as config

    with config.patch(combo_kernels=True):
        yield


def backbone_kwargs(a: dict) -> dict:
    """``MjcfWorldModel``'s ``backbone`` and ``rf`` from fit_mjcf's arguments ``a`` (``--backbone``, ``--rf_*``), or from a fit's
    saved ``args`` (a fit from before ``--backbone`` is an MLP's). Each backbone's options must stay at their defaults with the
    other, which would otherwise ignore them: the RigidFormer's ``--rf_*`` with the MLP; the MLP's ``--hidden``, ``--act`` and
    ``--no_ln``, and the contact geometry inputs (``--geometry``), with the RigidFormer."""
    backbone = a.get("backbone", "mlp")
    rf = {k: a.get(f"rf_{k}", v) for k, v in RF_DEFAULTS.items()}
    if backbone == "mlp":
        changed = [f"--rf_{k}" for k, v in RF_DEFAULTS.items() if rf[k] != v]
        if changed:
            raise ValueError(f"{', '.join(changed)}: options of the RigidFormer backbone (--backbone rigidformer), not of the MLP")
        return {"backbone": "mlp"}
    if backbone != "rigidformer":
        raise ValueError(f"backbone {backbone!r}: 'mlp' or 'rigidformer'")
    mlp_defaults = {"hidden": list(HIDDEN), "act": "silu", "no_ln": False, "geometry": False}
    changed = [f"--{k}" for k, v in mlp_defaults.items() if k in a and (list(a[k]) if k == "hidden" else a[k]) != v]
    if changed:
        raise ValueError(f"{', '.join(changed)}: options of the MLP backbone; the RigidFormer backbone takes --rf_*")
    return {"backbone": "rigidformer", "rf": rf}
