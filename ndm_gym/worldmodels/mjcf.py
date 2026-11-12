"""A world model of a MuJoCo model in its generalized coordinates, for models without actuator activation: qpos and qvel are
the whole state, so the model reads any MJCF that passes ``MjcfLayout``'s checks.

The layout is read once from the ``MjModel``: its joints (free, ball, slide, hinge), the anchor (the free joint at the root of
the actuated tree, if there is one), the other free joints (objects), the actuators (joint or fixed-tendon transmissions with
a fixed gain and an affine or no bias, the position and velocity servos of most robot models) and the joint equalities (a
joint held at a polynomial of another, as hands couple their finger joints), which each step then satisfies exactly: the
coupled joint's position and velocity are set from the other's. The inputs do not depend on the anchor's
heading and horizontal position: vectors in the world frame are read in the anchor's heading frame, heights stay absolute (the
floor). Two parameterizations: SuperTrack's (``kind="accel"``, worldmodels.accel, the default): an MLP predicts the generalized
accelerations, with the free joints' linear parts in the heading frame, and one step integrates them in MuJoCo's own convention
(mj_integratePos: the velocity first, then positions add and free and ball quaternions turn by the body-frame angular
velocity), so qpos and qvel round-trip through MuJoCo unchanged; SuperMimic's (``kind="increment"``, worldmodels.increment):
the MLP predicts the step's increments directly, the tangent from qpos to the next qpos and the velocity's change, the free
joints' linear parts again in the heading frame, and the step applies them (mj_integratePos with dt = 1). The error of a
prediction is taken in the tangent space (mj_differentiatePos), in units of the spread of one true step's increments, as the
hand's world model takes it (worldmodels.hand)."""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, quat_to_mat, yaw_mat

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3  # mjtJoint
TRN_JOINT, TRN_TENDON, DYN_NONE, GAIN_FIXED, BIAS_NONE, BIAS_AFFINE = 0, 3, 0, 0, 0, 1  # mjtTrn, mjtDyn, mjtGain, mjtBias
EQ_JOINT, WRAP_JOINT = 2, 1  # mjtEq, mjtWrap
# error floors (per tangent or velocity component): rad, m, rad/s, m/s
FLOOR_ANGLE, FLOOR_POS, FLOOR_ANG_VEL, FLOOR_LIN_VEL = 1e-3, 5e-4, 5e-2, 1e-2


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

    @staticmethod
    def from_model(m, anchor: str | None = None) -> MjcfLayout:
        """The layout of ``m`` (mujoco.MjModel). ``anchor``: the name of the anchor's free joint; by default the free joint at the
        root of the actuated joints' tree, or none (a fixed base) if that root has none. Features the model does not support are
        an error: actuator activation (na > 0); equality constraints other than joint equalities active from the start, and
        chains of them; flexes; mocap bodies with geoms; actuators other than joint or fixed-tendon transmissions (to hinges and
        slides) with a fixed gain and an affine or no bias."""
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
            tendons=tuple(tendons), couplings=tuple(couplings),
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


# ---------------- the world model ----------------
class MjcfWorldModel(nn.Module):
    """Deterministic dynamics (qpos', qvel') = f(qpos, qvel, ctrl) over one step of ``dt`` (the model's timestep by default)."""

    KINDS = ("accel", "increment")  # SuperTrack's accelerations, SuperMimic's increments

    def __init__(self, lay: MjcfLayout, hidden=(1024, 1024, 1024), act: str = "silu", layer_norm: bool = True, amp: bool = True,
                 kind: str = "accel"):  # fmt: skip
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"MjcfWorldModel: kind {kind!r}, one of {self.KINDS}")
        self.lay, self.amp, self.kind = lay, amp, kind  # amp: the MLP under bf16 autocast on a GPU (core.nets.run_mlp)
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
        # of an angle with no limits, else the position, then the velocity (columns of [q, sin q, cos q, qdot])
        S = len(lay.scalars)
        order = []
        for i, (_, _, wrap) in enumerate(lay.scalars):
            order += [S + i, 2 * S + i] if wrap else [i]
            order.append(3 * S + i)
        self.register_buffer("sc_q", torch.tensor([x[0] for x in lay.scalars], dtype=torch.long), persistent=False)
        self.register_buffer("sc_d", torch.tensor([x[1] for x in lay.scalars], dtype=torch.long), persistent=False)
        self.register_buffer("sc_order", torch.tensor(order, dtype=torch.long), persistent=False)
        q0 = torch.zeros(1, lay.nq)
        q0[:, self._quat_w(lay)] = 1.0  # unit quaternions
        self.n_in = self.inputs(q0, torch.zeros(1, lay.nv), torch.zeros(1, lay.nu)).shape[-1]
        n_out = lay.nv if kind == "accel" else 2 * lay.nv
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
        """The features of a state and its controls, invariant to the anchor's heading and horizontal position."""
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
        return torch.cat(parts, -1)

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
        """The next state: the predicted accelerations integrated as MuJoCo's semi-implicit Euler integrates them, or the
        predicted increments applied (``increment``, whose step is the model's timestep only)."""
        if self.kind == "increment" and dt is not None and dt != self.lay.dt:
            raise ValueError(f"MjcfWorldModel(kind='increment'): a step of {dt} s; its increments are of {self.lay.dt} s")
        dt = self.lay.dt if dt is None else dt
        y = run_mlp(self.net, self.in_norm(self.inputs(qpos, qvel, ctrl)), self.amp)
        y = self._rotate_linear(y * self.out_norm.std() + self.out_norm.mean, self.heading(qpos)[0])
        if self.kind == "increment":
            nv = self.lay.nv
            return self.couple(integrate_pos(self.lay, qpos, y[..., :nv], 1.0, (self.sc_q, self.sc_d)), qvel + y[..., nv:])
        qvel1 = qvel + dt * y
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
    def update_norms(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor, qpos1: torch.Tensor, qvel1: torch.Tensor):
        """The normalizers from transitions (..., dims): the inputs, the targets and the increments' spread."""
        self.in_norm.update(self.inputs(qpos, qvel, ctrl))
        self.out_norm.update(self.targets(qpos, qvel, qpos1, qvel1))
        self.err_norm.update(self.increments(qpos, qvel, qpos1, qvel1))

    def window_loss(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """The open-loop loss of windows: ``qpos`` (T + 1, B, nq), ``qvel`` (T + 1, B, nv), ``ctrl`` (T, B, nu); ``valid`` (T, B)
        bool, False from a step whose next state does not follow (a reset) on. The mean L1 error over the valid steps."""
        T = ctrl.shape[0]
        alive = torch.ones_like(qpos[0, :, 0], dtype=torch.bool) if valid is None else None
        q, v = qpos[0], qvel[0]
        total, count = qpos.new_zeros(()), qpos.new_zeros(())
        for t in range(T):
            ok = alive if valid is None else valid[: t + 1].all(0)
            q, v = self.step(q, v, ctrl[t])
            e = self.error(q, v, qpos[t + 1], qvel[t + 1]).abs().mean(-1)
            w = ok.to(e.dtype)
            total, count = total + (e * w).sum(), count + w.sum()
        return total / count.clamp_min(1.0)


def inference_copy(wm: MjcfWorldModel, dtype: torch.dtype | None = torch.bfloat16) -> MjcfWorldModel:
    """A forward-only copy for simulation throughput: the MLP's parameters in ``dtype`` with its input cast to them (bf16 or
    fp16: the network runs in that type end to end; the features, normalizers and integration stay fp32), or the fp32 model
    (None). On a GPU the copy's network is compared with the model's in fp32 and refused if it differs by more than twice the
    model's own bf16-autocast error on average or four times at the largest (worldmodels.g1_infer's rule)."""
    from .g1_infer import CastNet, network_errors

    m = copy.deepcopy(wm).eval().requires_grad_(False)
    if dtype is None:
        return m
    m.net, m.amp = CastNet(m.net.to(dtype), dtype), False
    dev = wm.in_norm.mean.device
    if dev.type == "cuda":
        x = torch.randn(256, wm.n_in, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
        with torch.no_grad():
            (fmax, fmean), (rmax, rmean) = network_errors(wm.net, m.net, x, amp=False)
        if not (fmean <= 2 * rmean + 1e-7 and fmax <= 4 * rmax + 1e-6):
            raise RuntimeError(f"inference_copy({dtype}): the network differs from the fp32 model by {fmax:.3g} at most, {fmean:.3g} on "
                               f"average; the model under bf16 autocast by {rmax:.3g}, {rmean:.3g}: refused")  # fmt: skip
    return m
