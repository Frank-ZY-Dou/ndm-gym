"""Differentiable forward kinematics of a floating-base articulation in torch, built from its compiled MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import axis_angle_to_mat, quat_to_mat


def mat_vec(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """M @ x for (..., 3, 3) and (..., 3), written as an elementwise product: torch.compile fuses it with its
    neighbours, whereas a batched 3x3 ``@`` stays a separate small matmul (3.6x faster fwd+bwd FK on an A100)."""
    return (M * x.unsqueeze(-2)).sum(-1)


def mat_mat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices as an elementwise product (see mat_vec)."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


class ArticulationKinematics:
    def __init__(self, mjcf_path: str, joint_names: list[str], tracked: list[str], device: str):
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in joint_names order)
        lo, hi = np.zeros(len(joint_names)), np.zeros(len(joint_names))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)
        # mass and local center of mass of every moving body, with fixed descendants (head, hands, sensors) merged in
        mass = np.array([m.body_mass[b] for b in mj_to_local], dtype=np.float64)
        mc = np.array([m.body_mass[b] * m.body_ipos[b] for b in mj_to_local], dtype=np.float64)
        for b in range(1, m.nbody):
            if m.body_jntnum[b] > 0:
                continue
            pos = m.body_ipos[b].copy()  # fixed body's CoM, walked up to its nearest moving ancestor
            a = b
            while a not in mj_to_local:
                R_a = np.zeros(9)
                mujoco.mju_quat2Mat(R_a, m.body_quat[a])
                pos = m.body_pos[a] + R_a.reshape(3, 3) @ pos
                a = m.body_parentid[a]
                assert a != 0, f"fixed body {m.body(b).name} is not attached to a moving body"
            i = mj_to_local[a]
            mass[i] += m.body_mass[b]
            mc[i] += m.body_mass[b] * pos
        self.body_mass = torch.tensor(mass, dtype=torch.float32, device=device)
        self.body_com = torch.tensor(mc / mass[:, None], dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,J) in joint_names order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mat_vec(Rp, self.off_p[i])
            R_off = mat_mat(Rp, self.off_R[i])
            Ri = mat_mat(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            axis_w = mat_vec(R_off, self.axes[i])
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

    def com(self, bp, bR, bv, bw):
        """Whole-body center of mass position and velocity from all-body FK (``forward(..., all_bodies=True)``)."""
        r = (bR @ self.body_com.unsqueeze(-1)).squeeze(-1)
        m = (self.body_mass / self.body_mass.sum()).unsqueeze(-1)
        return ((bp + r) * m).sum(-2), ((bv + torch.cross(bw, r, dim=-1)) * m).sum(-2)
