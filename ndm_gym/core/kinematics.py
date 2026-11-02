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


def farthest_point_sample(vertices: np.ndarray, k: int) -> np.ndarray:
    """Indices (k,) of RigidFormer's deterministic farthest point sampling of vertices (n, 3) with its rule
    'centroid_farthest': MultiObjectDynamics._farthest_point_sample_np in the official code (github.com/Frank-ZY-Dou/
    Dynamics-Modeling, RigidFormer/src/data/multi_dataloader.py:997-1042). Start at the vertex farthest from the
    centroid, then repeatedly take the vertex farthest from those taken; ties within a relative 1e-9 go to the
    lexicographically smallest coordinates, so the result depends on the vertex set only, not on its order; a taken
    vertex is never taken again (coincident duplicates come last). In float64."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected vertices of shape (n, 3), got {vertices.shape}")
    n = vertices.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}] for {n} vertices, got {k}")
    V = vertices.astype(np.float64)

    def argmax(d):  # multi_dataloader.py:1014-1021
        cand = np.flatnonzero(d >= d.max() - 1e-9 * max(1.0, abs(float(d.max()))))
        if len(cand) == 1:
            return int(cand[0])
        return int(cand[np.lexsort((V[cand, 2], V[cand, 1], V[cand, 0]))[0]])

    sel = np.empty(k, dtype=np.int64)
    sel[0] = argmax(((V - V.mean(0)) ** 2).sum(1))
    dist = np.full(n, np.inf)
    for i in range(1, k):  # multi_dataloader.py:1031-1041
        delta = V - V[sel[i - 1]]
        dist = np.minimum(dist, (delta * delta).sum(1))
        dist[sel[:i]] = -np.inf
        sel[i] = argmax(dist)
    return sel


class ArticulationKinematics:
    def __init__(self, mjcf_path: str, joint_names: list[str], tracked: list[str], device: str):
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        self.device, self.mjcf_path = device, mjcf_path
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

    def visual_vertices(self) -> list[np.ndarray]:
        """Every moving body's visual-mesh vertices (mesh geoms of group 1), with those of the fixed bodies attached to
        it, in its link frame: one (n_b, 3) float32 array per body, its vertex set for the anchors."""
        m = mujoco.MjModel.from_xml_path(self.mjcf_path)
        names = [m.body(b).name for b in range(m.nbody)]
        moving = {names.index(n): i for i, n in enumerate(self.body_names)}
        verts = [[] for _ in self.body_names]
        R = np.zeros(9)
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
                continue
            d = m.geom_dataid[g]
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            x = m.mesh_vert[m.mesh_vertadr[d] : m.mesh_vertadr[d] + m.mesh_vertnum[d]].astype(np.float64) @ R.reshape(3, 3).T + m.geom_pos[g]
            b = m.geom_bodyid[g]
            while b not in moving:  # a fixed body's vertices into its moving ancestor's frame
                mujoco.mju_quat2Mat(R, m.body_quat[b])
                x = x @ R.reshape(3, 3).T + m.body_pos[b]
                b = m.body_parentid[b]
                if b == 0:
                    raise RuntimeError(f"geom {g} is not attached to a moving body")
            verts[moving[b]].append(x)
        for name, v in zip(self.body_names, verts):
            if not v:
                raise RuntimeError(f"body {name} has no visual mesh to place anchors on")
        return [np.concatenate(v).astype(np.float32) for v in verts]

    def anchors(self, k: int) -> torch.Tensor:
        """k anchor points of every moving body in its link frame, (num_bodies, k, 3) float32: RigidFormer's
        deterministic farthest point sampling (``farthest_point_sample``) over the body's visual-mesh vertices
        (``visual_vertices``), as the official code's pregen_cf_anchors.py:15 builds an object's anchors from its
        float32 vertices; they are the first k of the body's full sampling order (pregen_fps_order.py)."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        out = []
        for name, v in zip(self.body_names, self.visual_vertices()):
            if len(v) < k:
                raise RuntimeError(f"body {name} has {len(v)} visual-mesh vertices, fewer than {k} anchors")
            out.append(v[farthest_point_sample(v, k)])
        return torch.tensor(np.stack(out), dtype=torch.float32, device=self.device)

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
