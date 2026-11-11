"""A hand-object scene (motion.sources.spider) in N worlds on MuJoCo-Warp, for hand-object tracking.

The scene is stepped by MuJoCo-Warp directly, as SPIDER does: Isaac Lab adds nothing to one articulation with position
actuators and a free object. A control step (50 Hz) holds the joints' position targets for the physics steps of
CONTROL_DT, replayed as one CUDA graph, then advances every world's reference frame. Episodes start at the reference's
first frame, the hand's joints and the object's horizontal position perturbed, and end at a termination
(tasks.hand_tracking) or at the reference's last frame; ended worlds are reset in the same step. With ``stagger``, the
first episodes end early, at random frames.

``step_nosync`` (train_hand --no_sync, --collect_graph; algos.hand_fast) is ``step`` without host synchronization: the
caller runs it on Warp's stream (``torch_stream``), where torch's writes, the physics and torch's reads follow each other,
or inside a capture with the physics as kernels (``physics``); the ended worlds are reset by mask (``reset_masked``) and
the frame counter is advanced in place, so that a CUDA graph reads the tensors it captured.
"""

from __future__ import annotations

import mujoco
import torch

from .. import require_simulator

require_simulator(__name__, modules=("warp", "mujoco_warp"))

import mujoco_warp as mjw
import warp as wp

from ...core.rotations import mat_to_quat, quat_to_mat
from ...robots.xhand import NUM_J, scene_indices
from ...state import CONTROL_DT
from ...tasks.hand_tracking import HandTask, unflatten_state


class HandSim:
    def __init__(self, model: mujoco.MjModel, task: HandTask, num_envs: int, device: str, joint_noise: float = 0.02,
                 object_noise: float = 0.005, stagger: bool = False, nconmax: int = 128, njmax: int = 512):  # fmt: skip
        self.task, self.ref, self.N, self.device = task, task.ref, num_envs, device
        self.joint_noise, self.object_noise = joint_noise, object_noise
        self.substeps = round(CONTROL_DT / model.opt.timestep)
        if abs(self.substeps * model.opt.timestep - CONTROL_DT) > 1e-9:
            raise ValueError(f"the physics step {model.opt.timestep} does not divide the control step {CONTROL_DT}")
        idx = scene_indices(model)
        self.qpos_idx, self.qvel_idx = (torch.as_tensor(idx[k], device=device) for k in ("qpos", "qvel"))
        self.ctrl_idx = torch.as_tensor(idx["ctrl"], device=device)
        self.object_qpos, self.object_qvel = idx["object_qpos"], idx["object_qvel"]
        wp.config.quiet = True
        wp.init()
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(model, mujoco.MjData(model), nworld=num_envs, nconmax=nconmax, njmax=njmax)
            with wp.ScopedCapture() as capture:  # a control step's physics as one CUDA graph
                for _ in range(self.substeps):
                    mjw.step(self.model, self.data)
        self.graph = capture.graph
        self.qpos, self.qvel, self.ctrl = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel), wp.to_torch(self.data.ctrl)
        self.warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last = torch.zeros(num_envs, dtype=torch.long, device=device)  # the frame each world's episode ends at
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        # the first episodes end at random frames, so that the worlds do not run through the clip in step
        self.last = torch.randint(1, self.ref.frames, (num_envs,), device=device) if stagger else self.last

    @torch.no_grad()
    def reset(self, mask: torch.Tensor):
        """Start the worlds of ``mask`` at the reference's first frame, at rest."""
        n = int(mask.sum())
        if n == 0:
            return
        q = self.ref.q[0] + self.joint_noise * torch.randn(n, NUM_J, device=self.device)
        q = torch.clamp(q, self.task.kin.q_lo, self.task.kin.q_hi)
        p = self.ref.p[0].repeat(n, 1)
        p[:, :2] += self.object_noise * (2 * torch.rand(n, 2, device=self.device) - 1)
        qpos = self.qpos[mask]
        qpos[:, self.qpos_idx] = q
        qpos[:, self.object_qpos : self.object_qpos + 3] = p
        qpos[:, self.object_qpos + 3 : self.object_qpos + 7] = mat_to_quat(self.ref.R[0])
        ctrl = self.ctrl[mask]
        ctrl[:, self.ctrl_idx] = q
        self.qpos[mask], self.ctrl[mask] = qpos, ctrl
        self.qvel[mask] = 0
        self.warmstart[mask] = 0
        self.f[mask] = 0
        self.last[mask] = self.ref.frames - 1

    @torch.no_grad()
    def reset_masked(self, mask: torch.Tensor):
        """``reset`` of the worlds of ``mask`` without a host read (train_hand --reset_masked): the same writes at fixed
        shapes, the random draws made for every world (another random stream than ``reset``'s, which draws for the worlds
        it resets)."""
        N, o, m = self.N, self.object_qpos, mask.unsqueeze(-1)
        q = self.ref.q[0] + self.joint_noise * torch.randn(N, NUM_J, device=self.device)
        q = torch.clamp(q, self.task.kin.q_lo, self.task.kin.q_hi)
        p = self.ref.p[0].repeat(N, 1)
        p[:, :2] += self.object_noise * (2 * torch.rand(N, 2, device=self.device) - 1)
        self.qpos[:, self.qpos_idx] = torch.where(m, q, self.qpos[:, self.qpos_idx])
        self.qpos[:, o : o + 3] = torch.where(m, p, self.qpos[:, o : o + 3])
        self.qpos[:, o + 3 : o + 7] = torch.where(m, mat_to_quat(self.ref.R[0]), self.qpos[:, o + 3 : o + 7])
        self.ctrl[:, self.ctrl_idx] = torch.where(m, q, self.ctrl[:, self.ctrl_idx])
        self.qvel.masked_fill_(m, 0.0)
        self.warmstart.masked_fill_(m, 0.0)
        self.f.masked_fill_(mask, 0)
        self.last.masked_fill_(mask, self.ref.frames - 1)

    def torch_stream(self) -> torch.cuda.Stream:
        """Warp's stream of the device as a torch stream: torch's work on it is ordered with the physics graph's."""
        return torch.cuda.ExternalStream(wp.get_stream(self.device).cuda_stream, device=self.device)

    def physics(self):
        """A control step's physics as kernels, for a capture that holds the whole collection step (train_hand
        --collect_graph); ``step`` and ``step_nosync`` replay them as the graph captured at the start."""
        with wp.ScopedDevice(self.device):
            for _ in range(self.substeps):
                mjw.step(self.model, self.data)

    @torch.no_grad()
    def step_nosync(self, u: torch.Tensor, physics=None, fk=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``step`` without host synchronization (the module's docstring): on Warp's stream, or with ``physics`` (the
        kernels) inside a capture. The ended worlds are reset by mask. ``fk``: the forward kinematics of the termination
        (train_hand --compile_eager's compiled ones; None: the task's)."""
        s = self.state()
        self.ctrl[:, self.ctrl_idx] = u
        if physics is None:
            wp.capture_launch(self.graph)
        else:
            physics()
        self.f.copy_(self.ref.next_frame(self.f))
        s1 = self.state()
        bad = ~torch.isfinite(s1).all(-1)
        s1 = torch.where(bad.unsqueeze(-1), s, s1)
        S1 = unflatten_state(s1)
        term = bad | self.task.termination(S1, (self.task.kin.forward if fk is None else fk)(S1[0]), self.f)
        done = term | (self.f >= self.last)
        self.reset_masked(done)
        return s1, term, done

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        """The flat state (N, STATE_DIM); MuJoCo keeps a free joint's angular velocity in the body's frame."""
        quat = self.qpos[:, self.object_qpos + 3 : self.object_qpos + 7]
        w = (quat_to_mat(quat) @ self.qvel[:, self.object_qvel + 3 : self.object_qvel + 6].unsqueeze(-1)).squeeze(-1)
        return torch.cat((self.qpos[:, self.qpos_idx], self.qvel[:, self.qvel_idx], self.qpos[:, self.object_qpos : self.object_qpos + 3], quat,
                          self.qvel[:, self.object_qvel : self.object_qvel + 3], w), -1)  # fmt: skip

    @torch.no_grad()
    def step(self, u: torch.Tensor, reset: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hold the position targets ``u`` (N, NUM_J) for one control step. Returns the next state, whether the episode
        terminated (also for a non-finite state, returned as the state before the step) and whether it ended; ended
        worlds are reset unless ``reset`` is False."""
        s = self.state()
        self.ctrl[:, self.ctrl_idx] = u
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        self.f = self.ref.next_frame(self.f)
        s1 = self.state()
        bad = ~torch.isfinite(s1).all(-1)
        s1 = torch.where(bad.unsqueeze(-1), s, s1)
        S1 = unflatten_state(s1)
        term = bad | self.task.termination(S1, self.task.kin.forward(S1[0]), self.f)
        done = term | (self.f >= self.last)
        if reset:
            self.reset(done)
        return s1, term, done
