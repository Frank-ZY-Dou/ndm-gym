"""Reference motion library: clips from any source resampled to 50 Hz, with FK of tracked bodies, and for
human-object interaction the object's reference."""

from __future__ import annotations

import numpy as np
import torch

from ..core.kinematics import ArticulationKinematics
from ..core.rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat
from ..state import CONTROL_DT
from .sources import Clip

BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


class MotionLib:
    def __init__(self, clips: list[Clip], kin: ArticulationKinematics, device: str):
        self.device = device
        self.names = [c.name for c in clips]
        objects = {c.obj_name for c in clips}
        if len(objects) > 1:
            raise ValueError(f"one object per motion library (a task), or none; got {sorted(map(str, objects))}")
        self.obj_name = objects.pop()
        self.has_object = self.obj_name is not None
        for field in ("obj_joint", "contact"):
            if len({getattr(c, field) is None for c in clips}) > 1:
                raise ValueError(f"{field} is given for some clips and missing for others")
        ps, qs, js = [], [], []
        ops, oqs, ojs, cts = [], [], [], []
        lens = []
        for c in clips:
            p, quat, j, src_fps = c.root_p, c.root_quat, c.joints, c.fps
            # resample to 50 Hz
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
            if self.has_object:
                if c.obj_p is None or c.obj_quat is None or c.obj_p.shape[0] != T_src or c.obj_quat.shape[0] != T_src:
                    raise ValueError(f"clip {c.name}: the object pose must cover every frame")
                ops.append(c.obj_p[i0] * (1 - a[:, None]) + c.obj_p[i0 + 1] * a[:, None])
                oqs.append(quat_slerp(c.obj_quat[i0], c.obj_quat[i0 + 1], a))
                if c.obj_joint is not None:
                    ojs.append(c.obj_joint[i0] * (1 - a[:, None]) + c.obj_joint[i0 + 1] * a[:, None])
                if c.contact is not None:
                    cts.append(c.contact[torch.round(t).long().clamp(max=T_src - 1)])  # flags: nearest source frame
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        if self.has_object:
            self.obj_p = cat(ops)
            self.obj_R = cat([quat_to_mat(q) for q in oqs])
            self.obj_v = cat([_gradient(op, CONTROL_DT) for op in ops])
            self.obj_w = cat([_angular_velocity(quat_to_mat(q)) for q in oqs])
            self.obj_joint = cat(ojs) if ojs else None
            self.contact = cat(cts) if cts else None
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """K future reference frames, holding the final pose with zero velocity beyond the clip."""
        fut = self.future_frames(f, K)  # (B,K)
        requested = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        recorded = (requested <= self.clip_end[self.clip_id[f]].unsqueeze(-1)).unsqueeze(-1)
        # Keep the recorded last-frame velocity; only the synthetic held-pose
        # extension is stationary. Do not change velocities in ref_state/rewards.
        v = torch.where(recorded, self.root_v_local[fut], 0.0)
        w = torch.where(recorded, self.root_w_local[fut], 0.0)
        qd = torch.where(recorded, self.qd[fut], 0.0)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, v, w, self.q[fut], qd * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def ref_object(self, f: torch.Tensor):
        """The object's reference state at frames f: position, rotation matrix, world linear and angular velocity."""
        if not self.has_object:
            raise ValueError("this motion library has no object")
        return self.obj_p[f], self.obj_R[f], self.obj_v[f], self.obj_w[f]

    def object_goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """The object's K future reference frames in the reference root's heading frame at f: position relative to the
        root, rotation (6D), linear and angular velocity, and the object's joint if it has one; beyond the clip the last
        pose is held with zero velocity, as in :meth:`goal`."""
        if not self.has_object:
            raise ValueError("this motion library has no object")
        fut = self.future_frames(f, K)
        requested = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        recorded = (requested <= self.clip_end[self.clip_id[f]].unsqueeze(-1)).unsqueeze(-1)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.obj_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.obj_R[fut])
        v = (Ht @ torch.where(recorded, self.obj_v[fut], 0.0).unsqueeze(-1)).squeeze(-1)
        w = (Ht @ torch.where(recorded, self.obj_w[fut], 0.0).unsqueeze(-1)).squeeze(-1)
        parts = [dp, rot, v, w] + ([self.obj_joint[fut]] if self.obj_joint is not None else [])
        return torch.cat(parts, -1).reshape(f.shape[0], -1)


def _angular_velocity(R: torch.Tensor) -> torch.Tensor:
    """World angular velocity of a rotation sequence (T, 3, 3) by central differences, one-sided at the ends."""
    w = torch.zeros(R.shape[0], 3, dtype=R.dtype)
    w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
    w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
    return w


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
