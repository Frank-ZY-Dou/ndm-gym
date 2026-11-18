"""The racing quadrotor (ndm_gym.robots.quadrotor) in N worlds on MuJoCo-Warp, for trajectory tracking.

The model is stepped by MuJoCo-Warp directly, as the hand's scene is (sim.rigid.hand_sim). A control step (100 Hz) holds the
command (thrust, body rates) for the physics steps of CONTROL_DT, replayed as one CUDA graph, then advances every world's
reference frame. An episode starts at a random frame of the reference, at the reference's state with noise on it, and ends at
a termination (tasks.quadrotor_tracking), after ``episode`` control steps or at its flight's last frame; ended worlds are reset
in the same step. With ``stagger``, the first episodes end early, after random numbers of steps. ``reset(mask, frame)`` starts
worlds at given frames with their flights' ends as the episodes' (evaluation).

``step_nosync`` (train_quadrotor --no_sync, --collect_graph) is ``step`` without host synchronization, as HandSim's: on Warp's
stream (``torch_stream``) or inside a capture with the physics as kernels (``physics``); the ended worlds are reset by mask
(``reset_masked``) and the frame counter is advanced in place.
"""

from __future__ import annotations

import mujoco
import torch

from .. import require_simulator

require_simulator(__name__, modules=("warp", "mujoco_warp"))

import mujoco_warp as mjw
import warp as wp

from ...core.rotations import mat_to_quat, so3_exp
from ...robots.quadrotor import CONTROL_DT
from ...tasks.quadrotor_tracking import QuadTask, unflatten_state


class QuadSim:
    def __init__(self, model: mujoco.MjModel, task: QuadTask, num_envs: int, device: str, episode: int = 500, pos_noise: float = 0.1,
                 rot_noise: float = 0.1, vel_noise: float = 0.3, ang_noise: float = 0.5, stagger: bool = False):  # fmt: skip
        self.task, self.ref, self.N, self.device, self.episode = task, task.ref, num_envs, device, episode
        self.noise = (pos_noise, rot_noise, vel_noise, ang_noise)
        self.substeps = round(CONTROL_DT / model.opt.timestep)
        if abs(self.substeps * model.opt.timestep - CONTROL_DT) > 1e-9:
            raise ValueError(f"the physics step {model.opt.timestep} does not divide the control step {CONTROL_DT}")
        if (model.nq, model.nv, model.nu) != (7, 6, 4):
            raise ValueError("the model is not the quadrotor's (robots.quadrotor.load_model): one free body, four actuators")
        # the frames an episode may start at: every frame but a flight's last
        self.starts = torch.nonzero(torch.arange(self.ref.frames, device=device) < self.ref.end).squeeze(-1)
        wp.config.quiet = True
        wp.init()
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(model, mujoco.MjData(model), nworld=num_envs, nconmax=1, njmax=1)
            with wp.ScopedCapture() as capture:  # a control step's physics as one CUDA graph
                for _ in range(self.substeps):
                    mjw.step(self.model, self.data)
        self.graph = capture.graph
        self.qpos, self.qvel, self.ctrl = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel), wp.to_torch(self.data.ctrl)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last = torch.zeros(num_envs, dtype=torch.long, device=device)  # the frame each world's episode ends at
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        if stagger:  # the first episodes end after random numbers of steps, so that the worlds do not reset in step
            self.last = torch.minimum(self.f + torch.randint(1, episode + 1, (num_envs,), device=device), self.ref.end[self.f])

    def _starts(self, n: int, frame: torch.Tensor | None = None):
        """``n`` episode starts: the frames (random ones, or ``frame``), the reference's state there with noise on it (qpos,
        qvel) and the frames the episodes end at (``frame`` given: its flight's last)."""
        dev, (pn, rn, vn, an) = self.device, self.noise
        f = self.starts[torch.randint(0, self.starts.shape[0], (n,), device=dev)] if frame is None else frame
        p, R, v, w = self.ref.state(f)
        qpos = torch.cat((p + pn * torch.randn(n, 3, device=dev), mat_to_quat(R @ so3_exp(rn * torch.randn(n, 3, device=dev)))), -1)
        qvel = torch.cat((v + vn * torch.randn(n, 3, device=dev), w + an * torch.randn(n, 3, device=dev)), -1)
        last = torch.minimum(f + self.episode, self.ref.end[f]) if frame is None else self.ref.end[f]
        return f, qpos, qvel, last

    @torch.no_grad()
    def reset(self, mask: torch.Tensor, frame: torch.Tensor | None = None):
        """Start the worlds of ``mask`` (``frame``: at these frames, one per world of the mask, to their flights' ends)."""
        n = int(mask.sum())
        if n == 0:
            return
        self.f[mask], self.qpos[mask], self.qvel[mask], self.last[mask] = self._starts(n, frame)

    @torch.no_grad()
    def reset_masked(self, mask: torch.Tensor):
        """``reset`` of the worlds of ``mask`` without a host read (train_quadrotor --reset_masked): the same writes at fixed
        shapes, the random draws made for every world (another random stream than ``reset``'s)."""
        f, qpos, qvel, last = self._starts(self.N)
        m = mask.unsqueeze(-1)
        self.qpos.copy_(torch.where(m, qpos, self.qpos))
        self.qvel.copy_(torch.where(m, qvel, self.qvel))
        self.f.copy_(torch.where(mask, f, self.f))
        self.last.copy_(torch.where(mask, last, self.last))

    def torch_stream(self) -> torch.cuda.Stream:
        """Warp's stream of the device as a torch stream: torch's work on it is ordered with the physics graph's."""
        return torch.cuda.ExternalStream(wp.get_stream(self.device).cuda_stream, device=self.device)

    def physics(self):
        """A control step's physics as kernels, for a capture that holds the whole collection step (--collect_graph)."""
        with wp.ScopedDevice(self.device):
            for _ in range(self.substeps):
                mjw.step(self.model, self.data)

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        """The flat state (N, STATE_DIM): the free joint's qpos and qvel (MuJoCo keeps its angular velocity in the body's frame)."""
        return torch.cat((self.qpos, self.qvel), -1)

    def _after(self, s: torch.Tensor):
        """The next state (the state before the step where it is non-finite) and whether each episode terminated."""
        s1 = self.state()
        bad = ~torch.isfinite(s1).all(-1)
        s1 = torch.where(bad.unsqueeze(-1), s, s1)
        return s1, bad | self.task.termination(unflatten_state(s1), self.f)

    @torch.no_grad()
    def step_nosync(self, u: torch.Tensor, physics=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``step`` without host synchronization (the module's docstring); the ended worlds are reset by mask."""
        s = self.state()
        self.ctrl.copy_(u)
        if physics is None:
            wp.capture_launch(self.graph)
        else:
            physics()
        self.f.copy_(self.ref.next_frame(self.f))
        s1, term = self._after(s)
        done = term | (self.f >= self.last)
        self.reset_masked(done)
        return s1, term, done

    @torch.no_grad()
    def step(self, u: torch.Tensor, reset: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hold the command ``u`` (N, 4) for one control step. Returns the next state, whether the episode terminated (also
        for a non-finite state) and whether it ended; ended worlds are reset unless ``reset`` is False."""
        s = self.state()
        self.ctrl.copy_(u)
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        self.f = self.ref.next_frame(self.f)
        s1, term = self._after(s)
        done = term | (self.f >= self.last)
        if reset:
            self.reset(done)
        return s1, term, done
