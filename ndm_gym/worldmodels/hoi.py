"""Neural dynamics models of HDMI's human-object interaction scenes (docs/hoi_design.md §3): one 20 ms control step of
the robot and the tracked object, and the contact forces, from the state, the four physics substeps' filtered actions
and the per-env context (``HDMISim.context``).

Two parameterizations of the step, as the LAFAN1 world models: "accel" (SuperTrack's, ``accel.py``: accelerations in
the bodies' frames, integrated with semi-implicit Euler) and "increment" (SuperMimic's, ``increment.py``: frame-local
increments). A free object is integrated like the robot's root; a hinged object's base is fixed, so only its hinge
angle and rate change. Both are fit with the same loss, ``window_loss``: SuperMimic's normalized frame-local state error
over a k-step rollout, with the object's and the forces' errors appended. The frame-local targets, increments and errors
are those of ``accel.py`` and ``increment.py`` with the 3 x 3 products written elementwise, so they stay fp32 where TF32
matmuls are allowed.

Shapes (N rollouts): S robot state (ndm_gym.state), So object state (p, R, v, w) or (angle (N, 1), rate (N, 1)),
u (N, 4, 23) filtered actions, ctx (N, C); forces F = (end effectors (N, E, 3) in the contact body's frame, feet
(N, 4, 2, 3) world frame after each substep). The inputs are invariant to a yaw of the scene, so the network predicts the
feet's forces in the heading frame of the step's start state (``feet_heading``) and ``step`` turns them to the world
frame; the contact body's frame turns with the scene.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..core.kinematics import mat_mat, mat_vec
from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, so3_exp, so3_log, yaw_mat
from ..robots.g1 import NUM_J
from ..state import CONTROL_DT, INV_DIM
from ..tasks.hdmi import DECIMATION, NUM_ACTIONS

KINDS = ("accel", "increment")
FEET_FORCE_DIM = DECIMATION * 2 * 3
# floors of the loss's normalizer: SuperMimic's per-component floors for the robot and a free object's root, a hinge's
# like a joint's, 1 N for the forces
ROOT_FLOOR = (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3
JOINT_FLOOR = (3e-3, 5e-2)
FORCE_FLOOR = 1.0
# format of HOIWorldModel's state dict: 2 predicts the feet's forces in the heading frame of the step's start state; 1 (no
# version in the state dict) predicted them in the world frame. Loading another format fails.
FORMAT_VERSION = 2


class ContextLayout(NamedTuple):
    """Layout of ``HDMISim.context``: the startup part, the reset part (29 joint offsets, delay, filter constant, hinge
    friction and damping), then 12 numbers (position, rotation matrix) per pose: a fixed base's, then each prop's."""

    startup: int
    base: bool
    props: int

    @property
    def scalars(self) -> int:
        """Length of the leading part without poses."""
        return self.startup + NUM_J + 4

    @property
    def dim(self) -> int:
        return self.scalars + 12 * (int(self.base) + self.props)

    def poses(self, ctx: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        i, out = self.startup + NUM_J + 4, []
        for _ in range(int(self.base) + self.props):
            out.append((ctx[..., i : i + 3], ctx[..., i + 3 : i + 12].unflatten(-1, (3, 3))))
            i += 12
        return out


def object_increment(So, So1) -> torch.Tensor:
    """Frame-local increments of a free body (p, R, v, w), the object or the robot's root, as the root's in
    ``IncrementWorldModel.targets``: the displacement in the heading frame of R, the rotation and the velocity changes
    in R's frame. The 3 x 3 products are elementwise (``mat_mat``, ``mat_vec``), so they stay fp32 where TF32 is allowed
    (train_hoi), where the batched product R^T R1 leaves rotation errors of about 5e-4."""
    p, R, v, w = So
    p1, R1, v1, w1 = So1
    H, Rt = yaw_mat(R), R.transpose(-1, -2)
    return torch.cat((mat_vec(H.transpose(-1, -2), p1 - p), so3_log(mat_mat(Rt, R1)), mat_vec(Rt, v1 - v), mat_vec(Rt, w1 - w)), -1)


def robot_increment(S, S1) -> torch.Tensor:
    """``IncrementWorldModel.targets(S, S1)``, also its ``state_error(S1, S)``, with the products elementwise."""
    return torch.cat((object_increment(S[:4], S1[:4]), S1[4] - S[4], S1[5] - S[5]), -1)


def robot_accel(S, S1) -> torch.Tensor:
    """``AccelWorldModel.targets(S, S1)``, the mean accelerations over one control step, with the products elementwise."""
    Rt = S[1].transpose(-1, -2)
    return torch.cat((mat_vec(Rt, S1[2] - S[2]), mat_vec(Rt, S1[3] - S[3]), S1[5] - S[5]), -1) / CONTROL_DT


def feet_heading(R, F) -> torch.Tensor:
    """The feet's forces F (..., 4, 2, 3) in the heading frame of the root's rotation R (..., 3, 3): H^T F, H = yaw_mat(R)."""
    return mat_vec(yaw_mat(R).transpose(-1, -2)[..., None, None, :, :], F)


def invariant_features(S) -> torch.Tensor:
    """``ndm_gym.state.invariant_features`` with the 3 x 3 products written out elementwise (``mat_mat``, ``mat_vec``):
    torch.compile fuses them with the rest of the step instead of launching batched matmuls, which also run in TF32
    when it is allowed. Equal up to float rounding."""
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(mat_mat(yaw_mat(R).transpose(-1, -2), R)), mat_vec(Rt, v), mat_vec(Rt, w), q, qd), -1)


def rigidformer_args(constants: dict, dim: int = 128, layers: int = 3, heads: int = 4, registers: int = 2) -> dict:
    """HOIRigidFormerNet's keyword arguments from a task's ``ndm_gym.sim.rigid.hdmi_sim.ndm_constants`` and a size."""
    return dict(eef_bodies=list(constants["eef_bodies"]), q0=torch.tensor(constants["q0"]), act_scale=torch.tensor(constants["act_scale"]),
                act_of_joint=torch.tensor(constants["act_of_joint"]), dim=dim, layers=layers, heads=heads, registers=registers)  # fmt: skip


class HOIWorldModel(nn.Module):
    """Deterministic one-step model (S, So, u, ctx) -> (S', So', F') of one HDMI task's scene."""

    def __init__(self, kind: str, hinged: bool, num_eefs: int, layout: ContextLayout, hidden=(2048, 2048, 2048), act: str = "silu",
                 layer_norm: bool = True, rigidformer: dict | None = None):  # fmt: skip
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.hinged, self.E, self.layout = kind, hinged, num_eefs, layout
        self.robot_out = (6 + NUM_J) if kind == "accel" else (12 + 2 * NUM_J)
        self.obj_out = (1 if hinged else 6) if kind == "accel" else (2 if hinged else 12)
        self.force_out = 3 * num_eefs + FEET_FORCE_DIM
        out = self.robot_out + self.obj_out + self.force_out
        if rigidformer is None:
            self.in_dim = INV_DIM + (2 if hinged else 15) + 9 * (int(layout.base) + layout.props) + DECIMATION * NUM_ACTIONS + layout.scalars
            self.net = mlp(self.in_dim, list(hidden), out, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.in_dim)
        else:  # token Transformer (HOIRigidFormerNet), same output layout
            if num_eefs != len(rigidformer["eef_bodies"]):
                raise ValueError(f"{num_eefs} end effectors but eef_bodies {rigidformer['eef_bodies']}")
            self.net = HOIRigidFormerNet(kind, hinged, layout, **rigidformer)
            self.in_dim, self.in_norm = self.net.in_dim, self.net.make_input_norm()
        self.out_norm = RunningNorm(out, eps=1e-4 / CONTROL_DT if kind == "accel" else 1e-4)
        # scale of one true step's frame-local increments (robot, object) and of the forces: the loss's normalizer
        obj_floor = JOINT_FLOOR if hinged else ROOT_FLOOR
        floor = ROOT_FLOOR + (JOINT_FLOOR[0],) * NUM_J + (JOINT_FLOOR[1],) * NUM_J + obj_floor + (FORCE_FLOOR,) * self.force_out
        self.err_norm = RunningNorm(len(floor), eps=1e-4)
        self.register_buffer("err_floor", torch.tensor(floor), persistent=False)
        self.register_buffer("format_version", torch.tensor(FORMAT_VERSION))
        self.amp = True  # bf16 autocast in step (if enabled)
        self.net_dtype, self.fused = None, False  # inference_copy
        for name in ("in_scale", "out_std"):  # inference_copy: the frozen normalizers' 1 / std and std
            self.register_buffer(name, None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Refuses a state dict of another format (``FORMAT_VERSION``) instead of loading it into a step that would read
        its outputs differently."""
        v = state_dict.get(prefix + "format_version")
        if v is None or int(v) != FORMAT_VERSION:
            raise RuntimeError(f"HOIWorldModel: a state dict of format {'1 (no version)' if v is None else int(v)}, this code reads format "
                               f"{FORMAT_VERSION}, which predicts the feet's forces in the heading frame of the step's start state (format 1 "
                               "predicted them in the world frame); models trained before are not compatible, fit the model again")  # fmt: skip
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def inference_copy(self, dtype: torch.dtype | None = torch.bfloat16, attention: str | None = "token", fused: bool = False) -> HOIWorldModel:
        """A copy for inference only (simulation throughput; no parameter requires gradients). ``dtype``: the network's
        parameters in that precision and ``step`` without autocast, so the network runs in it end to end (bf16: the
        residual stream, norms, attention and heads; the input features, kinematics, normalizers and integration stay
        fp32); None keeps the fp32 parameters under ``step``'s autocast. A RigidFormer's attention by ``attention``
        ("token": ``ndm_gym.core.token_attention``, "sdpa"), its read-out heads merged into one product and its relation
        bias gathered once (``HOIRigidFormerNet.freeze_tables``); the MLP takes ``attention=None``. ``fused``: the
        features (the forward kinematics and token features) and the integration as two Triton kernels
        (``worldmodels/hoi_fused.py``), the normalizers fused with the network's casts, CUDA only; equal to the unfused
        step up to float rounding (A100, bf16, token attention: RigidFormer 0.75 -> 0.58 ms GPU time per step at N=256
        and 1.80 -> 1.59 ms at 1024, MLP 0.17 -> 0.13 and 0.31 -> 0.26 ms). On a trained
        RigidFormer, the network output deviates from its fp32 eager value by 1.4e-4 on average and 7.4e-3 at most (in
        normalized units; ``step`` under bf16 autocast: 1.1e-4 and 1.7e-2)."""
        rf = isinstance(self.net, HOIRigidFormerNet)
        if rf != (attention is not None):
            raise ValueError(f"attention={attention!r}: the RigidFormer takes 'token' or 'sdpa', the MLP None")
        wm = copy.deepcopy(self).requires_grad_(False)
        if dtype is not None:
            for m in wm.net.modules():
                for name, prm in list(m.named_parameters(recurse=False)):
                    setattr(m, name, nn.Parameter(prm.detach().to(dtype), requires_grad=False))
            wm.net_dtype = dtype
        if rf:
            wm.net.freeze_tables()
            for blk in wm.net.blocks:
                blk.attention = attention
        if fused and wm.in_norm.mean.device.type != "cuda":
            raise ValueError("fused=True runs Triton kernels: move the model to a CUDA device first")
        if fused and rf:  # the kinematic tree and the joints' constants, compile-time constants of hoi_fused.rf_features
            from . import hoi_fused

            net = wm.net
            net.fk_tree = hoi_fused.kinematic_tree(net.kin)
            net.fk_q0, net.fk_gain, net.fk_aidx = net.q0.tolist(), net.act_gain.tolist(), net.act_idx.tolist()
        wm.fused = fused
        with torch.no_grad():
            wm.in_scale, wm.out_std = 1 / torch.sqrt(wm.in_norm.var + wm.in_norm.eps**2), wm.out_norm.std()
        return wm

    # ------------------------------------------------------------------ features, targets, errors
    def inputs(self, S, So, u, ctx):
        """Raw features, invariant to a shift and yaw of the whole scene: the robot's invariant features, the object and
        the fixed poses in the robot's heading frame, the filtered actions and the context's scalars."""
        if isinstance(self.net, HOIRigidFormerNet):
            return self.net.inputs(S, So, u, ctx)
        p, R = S[0], S[1]
        Ht = yaw_mat(R).transpose(-1, -2)
        feats = [invariant_features(S)]
        if self.hinged:
            feats.append(torch.cat(So, -1))
        else:
            po, Ro, vo, wo = So
            feats += [mat_vec(Ht, po - p), mat_to_6d(mat_mat(Ht, Ro)), mat_vec(Ht, vo), mat_vec(Ht, wo)]
        for pp, Rp in self.layout.poses(ctx):
            feats += [mat_vec(Ht, pp - p), mat_to_6d(mat_mat(Ht, Rp))]
        return torch.cat(feats + [u.flatten(-2), ctx[..., : self.layout.scalars]], -1)

    def targets(self, S, So, S1, So1, F1) -> torch.Tensor:
        """The output that maps (S, So) exactly to (S1, So1), then the forces F1 (the feet's in the heading frame of S):
        what ``out_norm`` normalizes."""
        if self.kind == "accel":
            robot = robot_accel(S, S1)
            if self.hinged:
                obj = (So1[1] - So[1]) / CONTROL_DT
            else:
                Rt = So[1].transpose(-1, -2)
                obj = torch.cat((mat_vec(Rt, So1[2] - So[2]), mat_vec(Rt, So1[3] - So[3])), -1) / CONTROL_DT
        else:
            robot = robot_increment(S, S1)
            obj = torch.cat((So1[0] - So[0], So1[1] - So[1]), -1) if self.hinged else object_increment(So, So1)
        return torch.cat((robot, obj, F1[0].flatten(-2), feet_heading(S[1], F1[1]).flatten(-3)), -1)

    def errors(self, pred: tuple, true: tuple) -> torch.Tensor:
        """Frame-local error components of a prediction (S, So, F) against the truth, in ``err_norm``'s layout: in the
        true state's frames, as ``IncrementWorldModel.state_error``; the feet's forces in its heading frame (the step's
        end; the start state is not passed, and the heading turns by one step in between)."""
        (Sh, Soh, Fh), (S, So, F) = pred, true
        obj = torch.cat((Soh[0] - So[0], Soh[1] - So[1]), -1) if self.hinged else object_increment(So, Soh)
        return torch.cat((robot_increment(S, Sh), obj, (Fh[0] - F[0]).flatten(-2), feet_heading(S[1], Fh[1] - F[1]).flatten(-3)), -1)

    def increments(self, S, So, S1, So1, F1) -> torch.Tensor:
        """One true step's frame-local increments, then its forces (the feet's in the heading frame of S): the data
        ``err_norm`` is fit on."""
        obj = torch.cat((So1[0] - So[0], So1[1] - So[1]), -1) if self.hinged else object_increment(So, So1)
        return torch.cat((robot_increment(S, S1), obj, F1[0].flatten(-2), feet_heading(S[1], F1[1]).flatten(-3)), -1)

    # ------------------------------------------------------------------ prediction
    def step(self, S, So, u, ctx, input_noise: float = 0.0):
        """``input_noise``: std of Gaussian noise on the normalized inputs (fitting only, as a regularizer: train_hoi's
        --wm_input_noise); for the RigidFormer it also jitters the rotary positions, by input_noise * POS_SCALE metres."""
        if self.fused:
            if input_noise != 0:
                raise ValueError("an inference copy's fused step takes no input noise (a fitting option)")
            return self._fused_step(S, So, u, ctx)
        x = self.in_norm(self.inputs(S, So, u, ctx))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        if self.net_dtype is None:
            y = run_mlp(self.net, x, amp=self.amp)
        else:  # inference_copy: the network in its parameters' precision (the RigidFormer casts its token features itself)
            y = self.net(x if isinstance(self.net, HOIRigidFormerNet) else x.to(self.net_dtype)).float()
        y = y * self.out_norm.std() + self.out_norm.mean
        yr, yo, yf = y.split((self.robot_out, self.obj_out, self.force_out), -1)
        return self._robot_next(S, yr), self._object_next(So, yo), self._forces(S, yf)

    def _fused_step(self, S, So, u, ctx):
        """``step`` of an inference copy with ``fused``: one kernel for the features (``hoi_fused``), their copies and
        normalization fused with the network's first cast, the network, the output normalizer fused with its last cast,
        one kernel for the integration and the feet's forces."""
        from . import hoi_fused

        lay, n_in, rf = self.layout, self.in_norm, isinstance(self.net, HOIRigidFormerNet)
        poses = int(lay.base) + lay.props
        if rf:
            net, d = self.net, self.in_dim - lay.scalars
            raw = hoi_fused.rf_features(*S, list(So), u, ctx, net.fk_tree, net.fk_q0, net.fk_gain, net.fk_aidx, lay.startup, poses, self.hinged,
                                        d, d + 3 * (net.n_body + 1 + net.n_props)).t()  # fmt: skip
            x = torch.cat(((torch.cat((raw[:, :d], ctx[:, : lay.scalars]), -1) - n_in.mean) * self.in_scale, raw[:, d:]), -1)
        else:
            f = hoi_fused.mlp_features(S[0], S[1], S[2], S[3], list(So), ctx, lay.startup, poses, self.hinged).t()
            obj = torch.cat(So, -1) if self.hinged else f[:, 13:28]
            x = torch.cat((f[:, :13], S[4], S[5], obj, f[:, 13 + (0 if self.hinged else 15) :], u.flatten(-2), ctx[..., : lay.scalars]), -1)
            x = (x - n_in.mean) * self.in_scale
        if self.net_dtype is None:
            y = run_mlp(self.net, x, amp=self.amp)
        else:
            y = self.net(x if rf else x.to(self.net_dtype)).float()
        y = y * self.out_std + self.out_norm.mean
        i = self.robot_out + self.obj_out
        out = hoi_fused.integrate(y, *S, list(So), self.kind == "accel", self.robot_out, i + 3 * self.E)
        return tuple(out[:6]), tuple(out[6:-1]), (y[:, i : i + 3 * self.E].unflatten(-1, (self.E, 3)), out[-1])

    def rollout(self, S, So, u, ctx, input_noise: float = 0.0) -> list[tuple]:
        """Open-loop k-step prediction from (S, So) with u (N, k, 4, 23) and ctx (N, k, C): k tuples (S, So, F);
        ``input_noise`` as ``step``'s."""
        out = []
        for j in range(u.shape[1]):
            S, So, F = self.step(S, So, u[:, j], ctx[:, j], input_noise)
            out.append((S, So, F))
        return out

    def window_loss(self, preds: list[tuple], trues: list[tuple], mask: torch.Tensor):
        """SuperMimic's world-model loss over a window: squared errors normalized by the per-component scale of one
        true step (with floors), averaged over components, summed over the valid steps (``mask`` (N, k)), averaged over
        windows. Also returns each window's last-step error and its sum (for prioritized sampling)."""
        std = torch.maximum(self.err_norm.std(), self.err_floor)
        e = torch.stack([((self.errors(p, t) / std) ** 2).mean(-1) for p, t in zip(preds, trues)], 1)  # (N, k)
        em = torch.where(mask > 0, e, 0.0) * mask  # a masked step's error may be non-finite, and NaN * 0 is NaN
        return em.sum() / mask.shape[0], e[:, -1], em.sum(1)

    def _robot_next(self, S, y):
        p, R, v, w, q, qd = S
        dt = CONTROL_DT
        if self.kind == "accel":  # AccelWorldModel.step with one substep
            av, aw, aq = mat_vec(R, y[..., :3]), mat_vec(R, y[..., 3:6]), y[..., 6:]
            v1, w1, qd1 = v + dt * av, w + dt * aw, qd + dt * aq
            return (p + dt * v1, mat_mat(so3_exp(dt * w1, elementwise=True), R), v1, w1, q + dt * qd1, qd1)
        dp, dth, dv, dw, dq, dqd = y.split((3, 3, 3, 3, NUM_J, NUM_J), -1)  # IncrementWorldModel.step
        return (p + mat_vec(yaw_mat(R), dp), mat_mat(R, so3_exp(dth, elementwise=True)), v + mat_vec(R, dv), w + mat_vec(R, dw), q + dq, qd + dqd)

    def _object_next(self, So, y):
        dt = CONTROL_DT
        if self.hinged:
            angle, rate = So
            if self.kind == "accel":
                rate1 = rate + dt * y
                return (angle + dt * rate1, rate1)
            return (angle + y[..., :1], rate + y[..., 1:2])
        p, R, v, w = So
        if self.kind == "accel":
            v1, w1 = v + dt * mat_vec(R, y[..., :3]), w + dt * mat_vec(R, y[..., 3:6])
            return (p + dt * v1, mat_mat(so3_exp(dt * w1, elementwise=True), R), v1, w1)
        dp, dth, dv, dw = y.split((3, 3, 3, 3), -1)
        return (p + mat_vec(yaw_mat(R), dp), mat_mat(R, so3_exp(dth, elementwise=True)), v + mat_vec(R, dv), w + mat_vec(R, dw))

    def _forces(self, S, yf):
        """The forces from their outputs: the end effectors' as predicted, the feet's turned from the heading frame of S
        to the world frame."""
        feet = yf[..., 3 * self.E :].unflatten(-1, (DECIMATION, 2, 3))
        return yf[..., : 3 * self.E].unflatten(-1, (self.E, 3)), mat_vec(yaw_mat(S[1])[..., None, None, :, :], feet)


class HOIRigidFormerNet(nn.Module):
    """RigidFormer for HDMI's scenes: one token per robot body (G1 kinematic-tree order, as ``RigidFormerNet``), one for
    the object, one per static prop, and register tokens; 3-D rotary attention on the tokens' positions, a learned
    per-head bias by relation (the kinematic-tree hop count between two bodies; one class each for body-object,
    body-prop, object/prop pairs and register pairs), FiLM on the context's scalars, and read-out heads on the pelvis
    (root), the joint tokens, the object token, the end effectors' and the ankles' tokens (contact forces).

    Token features, in the robot's heading frame with the origin at the pelvis's ground projection (height absolute):
    body = position, 6-D rotation, linear and angular velocity, and its parent joint's angle, rate, offset, the four
    substeps' PD-target offsets from the default pose and PD errors (zeros on the pelvis); object = position, 6-D
    rotation, velocities, or for a hinge its base's position and 6-D rotation, angle, rate, Coulomb friction and
    damping; prop = position and 6-D rotation. ``inputs`` returns them flat with the context's scalars, ``forward`` takes
    them through ``TokenInputNorm`` (positions passed on for the rotary attention) and returns the flat output in
    ``HOIWorldModel``'s layout.

    ``checkpoint``: activation checkpointing of everything after the input split (recomputed in the backward pass), as
    ``RigidFormerNet``'s: "never", "always", or "frozen" = only while the parameters do not require gradients, i.e. in
    the policy's BPTT through the frozen world model, where the horizon's activations dominate the memory.
    """

    def __init__(self, kind: str, hinged: bool, layout: ContextLayout, eef_bodies: list[str], q0: torch.Tensor, act_scale: torch.Tensor,
                 act_of_joint: torch.Tensor, dim: int = 128, layers: int = 3, heads: int = 4, ffn_mult: float = 2.5, registers: int = 2,
                 rope_max_freq: float = 10.0, checkpoint: str = "frozen"):  # fmt: skip
        super().__init__()
        from ..robots.g1 import kinematics
        from .rigidformer import Block, Rope3d, TokenInputNorm, TreeKinematics, tree_hops

        if checkpoint not in ("never", "always", "frozen"):
            raise ValueError(f"checkpoint must be 'never', 'always' or 'frozen', got {checkpoint!r}")
        kin = kinematics("cpu")
        B, P = kin.num_bodies, layout.props
        self.kind, self.hinged, self.layout, self.n_body, self.n_props, self.n_reg = kind, hinged, layout, B, P, registers
        self.checkpoint = checkpoint
        self.kin = TreeKinematics(kin)
        self.register_buffer("joint_of_body", torch.tensor(kin.joint_idx[1:]), persistent=False)
        self.register_buffer("body_of_joint", torch.argsort(self.joint_of_body), persistent=False)
        self.register_buffer("q0", q0.detach().float().cpu().clone(), persistent=False)  # (29,) G1 order
        self.register_buffer("act_scale", act_scale.detach().float().cpu().clone(), persistent=False)  # (23,) HDMI's action order
        self.register_buffer("act_of_joint", act_of_joint.detach().long().cpu().clone(), persistent=False)  # (29,) action index or -1
        # each joint's action index (0 on the wrists) and action gain (0 on the wrists), built once
        self.register_buffer("act_idx", self.act_of_joint.clamp(min=0), persistent=False)
        self.register_buffer("act_gain", self.act_scale[self.act_idx] * (self.act_of_joint >= 0).float(), persistent=False)
        self.eef_tok = [kin.body_names.index(b) for b in eef_bodies]
        self.feet_tok = [kin.body_names.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.body_dim, self.obj_dim, self.prop_dim = 15 + 11, (13 if hinged else 15), 9
        self.in_dim = B * self.body_dim + self.obj_dim + P * self.prop_dim + layout.scalars
        pos = [b * self.body_dim + torch.arange(3) for b in range(B)] + [B * self.body_dim + torch.arange(3)]
        pos += [B * self.body_dim + self.obj_dim + k * self.prop_dim + torch.arange(3) for k in range(P)]
        self.register_buffer("pos_index", torch.cat(pos), persistent=False)
        self._norm_cls = TokenInputNorm
        self.ctx_net = nn.Sequential(nn.Linear(layout.scalars, 64), nn.SiLU())
        self.embed_body, self.embed_obj = nn.Linear(self.body_dim, dim), nn.Linear(self.obj_dim, dim)
        self.embed_prop = nn.Linear(self.prop_dim, dim) if P else None
        T = B + 1 + P + registers
        self.tok_emb = nn.Parameter(0.5 * torch.randn(B + 1 + P, dim))
        self.registers = nn.Parameter(0.02 * torch.randn(1, registers, dim)) if registers else None
        self.blocks = nn.ModuleList(Block(dim, heads, ffn_mult, 64, fused_rope=True) for _ in range(layers))
        self.rope = Rope3d(6 * ((3 * (dim // heads) // 4) // 6), rope_max_freq)
        hops = tree_hops(kin.parent)
        m = int(hops.max())
        rel = torch.full((T, T), m + 3, dtype=torch.long)  # object and prop pairs among themselves
        rel[:B, :B] = hops
        rel[:B, B], rel[B, :B] = m + 1, m + 1  # body-object
        rel[:B, B + 1 : B + 1 + P], rel[B + 1 : B + 1 + P, :B] = m + 2, m + 2  # body-prop
        rel[B + 1 + P :, :], rel[:, B + 1 + P :] = m + 4, m + 4  # registers
        self.register_buffer("rel", rel, persistent=False)
        self.rel_bias = nn.Parameter(torch.zeros(heads, m + 5))
        hidden = int(ffn_mult * dim)
        self.head_norm = nn.RMSNorm(2 * dim, eps=1e-6)
        self.head_w12, self.head_w3 = nn.Linear(2 * dim, 2 * hidden), nn.Linear(hidden, dim)
        root_out, joint_out = (6, 1) if kind == "accel" else (12, 2)
        obj_out = (1 if hinged else 6) if kind == "accel" else (2 if hinged else 12)
        self.root_head, self.joint_head, self.obj_head = nn.Linear(dim, root_out), nn.Linear(dim, joint_out), nn.Linear(dim, obj_out)
        self.eef_head, self.feet_head = nn.Linear(dim, 3), nn.Linear(dim, DECIMATION * 3)
        for h in (self.root_head, self.joint_head, self.obj_head, self.eef_head, self.feet_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.head_all = None  # freeze_tables (inference copies)
        for name in ("head_index", "bias_table"):
            self.register_buffer(name, None, persistent=False)
        self.fk_tree = None  # fused inference copies: the kinematic tree's key in hoi_fused, the joints' constants

    def freeze_tables(self):
        """For inference copies (parameters frozen): the relation bias gathered once, and the five read-out heads as one
        linear layer over the read tokens followed by a gather with an index table built here (one product instead of
        five; the same outputs up to rounding)."""
        heads = (self.root_head, self.joint_head, self.obj_head, self.eef_head, self.feet_head)
        off = [0]
        for h in heads:
            off.append(off[-1] + h.out_features)
        C, B, w = off[-1], self.n_body, self.root_head.weight
        self.head_all = nn.Linear(w.shape[1], C, device=w.device, dtype=w.dtype).requires_grad_(False)
        with torch.no_grad():
            self.head_all.weight.copy_(torch.cat([h.weight for h in heads]))
            self.head_all.bias.copy_(torch.cat([h.bias for h in heads]))
            self.bias_table = self.rel_bias[:, self.rel].unsqueeze(0).detach()
        idx = [off[0] + r for r in range(heads[0].out_features)]  # flat (token, column) of every output in forward's layout
        idx += [(1 + int(self.body_of_joint[j])) * C + off[1] + k for k in range(heads[1].out_features) for j in range(NUM_J)]
        idx += [B * C + off[2] + r for r in range(heads[2].out_features)]
        idx += [t * C + off[3] + r for t in self.eef_tok for r in range(3)]
        idx += [t * C + off[4] + 3 * s + r for s in range(DECIMATION) for t in self.feet_tok for r in range(3)]
        self.head_index = torch.tensor(idx, device=w.device)

    def make_input_norm(self):
        return self._norm_cls(self.in_dim, self.pos_index.clone())

    def inputs(self, S, So, u, ctx):
        p, R, v, w, q, qd = S
        P_, Rb, V, W = self.kin(p, R, v, w, q, qd)
        Ht = yaw_mat(R).transpose(-1, -2).unsqueeze(-3)
        origin = torch.cat((p[..., :2], torch.zeros_like(p[..., 2:])), -1)
        rel = lambda x: mat_vec(Ht.squeeze(-3), x - origin)  # noqa: E731  (N, 3) points
        off = ctx[..., self.layout.startup : self.layout.startup + NUM_J]
        d = u[..., self.act_idx] * self.act_gain  # (N, 4, 29): each substep's PD-target offset from q0 + offset, 0 on wrists
        tgt = self.q0 + off.unsqueeze(-2) + d
        joint = torch.cat((torch.stack((q, qd, off), -1), d.transpose(-1, -2), (tgt - q.unsqueeze(-2)).transpose(-1, -2)), -1)
        joint = joint.index_select(-2, self.joint_of_body)
        joint = torch.cat((torch.zeros_like(joint[..., :1, :]), joint), -2)
        body = torch.cat((mat_vec(Ht, P_ - origin.unsqueeze(-2)), mat_to_6d(mat_mat(Ht, Rb)), mat_vec(Ht, V), mat_vec(Ht, W), joint), -1)
        Ht2 = Ht.squeeze(-3)
        if self.hinged:
            (bp, bR), f, c = self.layout.poses(ctx)[0], ctx[..., self.layout.startup + NUM_J + 2 :][..., :1], ctx[..., self.layout.startup + NUM_J + 3 :][..., :1]
            obj = torch.cat((rel(bp), mat_to_6d(mat_mat(Ht2, bR)), So[0], So[1], f, c), -1)
        else:
            po, Ro, vo, wo = So
            obj = torch.cat((rel(po), mat_to_6d(mat_mat(Ht2, Ro)), mat_vec(Ht2, vo), mat_vec(Ht2, wo)), -1)
        props = [torch.cat((rel(pp), mat_to_6d(mat_mat(Ht2, Rp))), -1) for pp, Rp in self.layout.poses(ctx)[int(self.hinged) :]]
        return torch.cat([body.flatten(-2), obj] + props + [ctx[..., : self.layout.scalars]], -1)

    def forward(self, x):
        from .rigidformer import POS_SCALE

        B, P = self.n_body, self.n_props
        dt = self.tok_emb.dtype  # fp32 (under autocast), or an inference copy's dtype; the rotary positions stay fp32
        feats, pos = x[:, : self.in_dim].to(dt), POS_SCALE * x[:, self.in_dim :].unflatten(-1, (B + 1 + P, 3))
        recompute = self.checkpoint == "always" or (self.checkpoint == "frozen" and not self.embed_body.weight.requires_grad)
        if recompute and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._trunk, feats, pos, use_reentrant=False, preserve_rng_state=False)
        return self._trunk(feats, pos)

    def _trunk(self, feats, pos):
        N, B, P = feats.shape[0], self.n_body, self.n_props
        dt = self.tok_emb.dtype
        i = B * self.body_dim
        toks = [self.embed_body(feats[:, :i].unflatten(-1, (B, self.body_dim))), self.embed_obj(feats[:, i : i + self.obj_dim]).unsqueeze(1)]
        i += self.obj_dim
        if P:
            toks.append(self.embed_prop(feats[:, i : i + P * self.prop_dim].unflatten(-1, (P, self.prop_dim))))
            i += P * self.prop_dim
        c = self.ctx_net(feats[:, i:])
        h = torch.cat(toks, 1).to(dt) + self.tok_emb
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(N, -1, -1)), 1)
            pos = torch.cat((pos, pos.new_zeros(N, self.n_reg, 3)), 1)
        cos, sin = self.rope(pos)
        bias = self.rel_bias[:, self.rel].unsqueeze(0) if self.bias_table is None else self.bias_table
        h0 = h
        for blk in self.blocks:
            h = blk(h, cos, sin, bias, c)
        g12 = self.head_w12(self.head_norm(torch.cat((h0[:, : B + 1], h[:, : B + 1]), -1)))  # the tokens read out: bodies, object
        a, gate = g12.chunk(2, -1)
        z = self.head_w3(a * torch.nn.functional.silu(gate))
        if self.head_index is not None:  # freeze_tables: one product, then the outputs gathered
            return self.head_all(z).flatten(1)[:, self.head_index]
        y_joint = self.joint_head(z[:, 1:B]).index_select(1, self.body_of_joint)  # (N, 29, joint_out), G1 order
        eef = self.eef_head(z[:, self.eef_tok]).flatten(1)
        feet = self.feet_head(z[:, self.feet_tok]).unflatten(-1, (DECIMATION, 3)).transpose(1, 2).flatten(1)  # (N, 4, 2, 3)
        return torch.cat((self.root_head(z[:, 0]), y_joint.transpose(1, 2).flatten(1), self.obj_head(z[:, B]), eef, feet), -1)
