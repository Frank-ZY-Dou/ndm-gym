"""The Go2 scene (robots.go2.load_scene) in N worlds on MuJoCo-Warp, for quadruped motion tracking; also the humanoids'
(robots.humanoid.load_scene, tasks.humanoid_tracking): the task gives the state's layout (``task.unflatten``).

The scene is stepped by MuJoCo-Warp directly, as the hand-object task's (sim.rigid.hand_sim): one articulation with
position servos on a floor. A control step (50 Hz) holds the joints' position targets for the physics steps of
CONTROL_DT, replayed as one CUDA graph, then advances every world's reference frame. Each world tracks a frame of one
of the reference's clips (tasks.quadruped_tracking). Episodes start at a random frame of a random clip (every frame but
a clip's last equally likely, so longer clips start more episodes), at the reference's state with the joints
perturbed, and end at a termination or at their clip's last frame; ended worlds are reset in the same step.

``step_nosync`` (train_quadruped --no_sync, --collect_graph; algos.hand_fast) is ``step`` without host synchronization, as
HandSim's: the caller runs it on Warp's stream (``torch_stream``), or inside a capture with the physics as kernels
(``physics``); the ended worlds are reset by mask (``reset_masked``) and the frame counter is advanced in place.
"""

from __future__ import annotations

import mujoco
import torch

from .. import require_simulator

require_simulator(__name__, modules=("warp", "mujoco_warp"))

import mujoco_warp as mjw
import warp as wp

from ...state import CONTROL_DT
from ...tasks.quadruped_tracking import QuadrupedTask


class QuadrupedSim:
    def __init__(self, model: mujoco.MjModel, task: QuadrupedTask, num_envs: int, device: str, joint_noise: float = 0.05,
                 vel_noise: float = 0.1, nconmax: int = 32, njmax: int = 192):  # fmt: skip
        self.task, self.ref, self.N, self.device = task, task.ref, num_envs, device
        self.joint_noise, self.vel_noise = joint_noise, vel_noise
        self.substeps = round(CONTROL_DT / model.opt.timestep)
        if abs(self.substeps * model.opt.timestep - CONTROL_DT) > 1e-9:
            raise ValueError(f"the physics step {model.opt.timestep} does not divide the control step {CONTROL_DT}")
        # the frames an episode may start at: every frame but its clip's last
        self.starts = torch.nonzero(torch.arange(self.ref.frames, device=device) < self.ref.last).squeeze(-1)
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
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    @torch.no_grad()
    def reset(self, mask: torch.Tensor, frames: torch.Tensor | None = None, noise: bool = True):
        """Start the worlds of ``mask`` at the reference's state of random start frames (``frames``: these, one per world
        of the mask), the joints' positions and the velocities perturbed unless ``noise`` is False."""
        n = int(mask.sum())
        if n == 0:
            return
        f = self.sample_starts(n) if frames is None else frames
        qpos, qvel = self.ref.qpos[f].clone(), self.ref.qvel[f].clone()
        if noise:
            qpos, qvel = self.perturb(qpos, qvel)
        self.qpos[mask], self.qvel[mask], self.ctrl[mask] = qpos, qvel, qpos[:, 7:]
        self.warmstart[mask] = 0
        self.f[mask] = f
        self.started(mask, f)

    def sample_starts(self, n: int, gate: torch.Tensor | None = None) -> torch.Tensor:
        """``n`` episode start frames: every frame but a clip's last equally likely. ``gate``: (reset_masked) whether any
        world resets (a subclass's statistics advance only then; sim.rigid.humanoid_sim)."""
        return self.starts[torch.randint(0, self.starts.shape[0], (n,), device=self.device)]

    def perturb(self, qpos: torch.Tensor, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The start state perturbed (in place): Gaussian noise on the joints' positions (clamped to their ranges) and on
        every velocity."""
        qpos[:, 7:] = torch.clamp(qpos[:, 7:] + self.joint_noise * torch.randn_like(qpos[:, 7:]), self.task.kin.q_lo, self.task.kin.q_hi)
        qvel += self.vel_noise * torch.randn_like(qvel)
        return qpos, qvel

    def started(self, mask: torch.Tensor, f: torch.Tensor):
        """Called after the worlds of ``mask`` start at ``f`` (one per world of the mask, or, from reset_masked, one per
        world); a subclass's episode bookkeeping."""

    @torch.no_grad()
    def reset_masked(self, mask: torch.Tensor):
        """``reset`` of the worlds of ``mask`` without a host read (train_quadruped --reset_masked): the same writes at
        fixed shapes, the start frames and the perturbations drawn for every world (another random stream than ``reset``'s,
        which draws for the worlds it resets)."""
        m = mask.unsqueeze(-1)
        f = self.sample_starts(self.N, gate=mask.any())
        qpos, qvel = self.perturb(self.ref.qpos[f].clone(), self.ref.qvel[f].clone())
        self.qpos.copy_(torch.where(m, qpos, self.qpos))
        self.qvel.copy_(torch.where(m, qvel, self.qvel))
        self.ctrl.copy_(torch.where(m, qpos[:, 7:], self.ctrl))
        self.warmstart.masked_fill_(m, 0.0)
        self.f.copy_(torch.where(mask, f, self.f))
        self.started(mask, f)

    def torch_stream(self) -> torch.cuda.Stream:
        """Warp's stream of the device as a torch stream: torch's work on it is ordered with the physics graph's."""
        return torch.cuda.ExternalStream(wp.get_stream(self.device).cuda_stream, device=self.device)

    def physics(self):
        """A control step's physics as kernels, for a capture that holds the whole collection step (train_quadruped
        --collect_graph); ``step`` and ``step_nosync`` replay them as the graph captured at the start."""
        with wp.ScopedDevice(self.device):
            for _ in range(self.substeps):
                mjw.step(self.model, self.data)

    @torch.no_grad()
    def step_nosync(self, u: torch.Tensor, physics=None, fk=None, reset: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``step`` without host synchronization (the module's docstring): on Warp's stream, or with ``physics`` (the
        kernels) inside a capture. The ended worlds are reset by mask unless ``reset`` is False (evaluation). ``fk``: the
        forward kinematics of the termination (train_quadruped --compile_eager's compiled ones; None: the task's)."""
        s, f = self.state(), self.f.clone()
        self.ctrl.copy_(u)
        if physics is None:
            wp.capture_launch(self.graph)
        else:
            physics()
        self.f.copy_(self.ref.next_frame(self.f))
        s1, term, done = self.transition(s, f, self.task.kin.forward if fk is None else fk)
        if reset:
            self.reset_masked(done)
        return s1, term, done

    def transition(self, s: torch.Tensor, f: torch.Tensor, fk) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """After a control step from the state ``s`` at frame ``f`` (the frame counter advanced): the next state (``s`` for
        a non-finite one), the termination (the task's, or a non-finite state) and the episode's end (a termination or
        the clip's last frame). ``fk``: the forward kinematics of the termination."""
        s1 = self.state()
        bad = ~torch.isfinite(s1).all(-1)
        s1 = torch.where(bad.unsqueeze(-1), s, s1)
        S1 = self.task.unflatten(s1)
        term = bad | self.task.termination(S1, fk(S1[0]), self.f)
        return s1, term, term | (self.f >= self.ref.last[self.f])

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        """The flat state (N, STATE_DIM): qpos, then qvel."""
        return torch.cat((self.qpos, self.qvel), -1)

    @torch.no_grad()
    def step(self, u: torch.Tensor, reset: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hold the joint position targets ``u`` (N, NUM_J) for one control step. Returns the next state, whether the
        episode terminated (also for a non-finite state, returned as the state before the step) and whether it ended (a
        termination or its clip's last frame); ended worlds are reset unless ``reset`` is False."""
        s, f = self.state(), self.f
        self.ctrl.copy_(u)
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        self.f = self.ref.next_frame(self.f)
        s1, term, done = self.transition(s, f, self.task.kin.forward)
        if reset:
            self.reset(done)
        return s1, term, done
