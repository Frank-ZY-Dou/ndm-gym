"""A robomimic scene with its servos (robots.robomimic_scenes) in N worlds on MuJoCo-Warp, for demonstration tracking.

The scene is stepped by MuJoCo-Warp directly, as the hand's is (sim.rigid.hand_sim). A control step (20 Hz) holds the
actuators' position targets for the physics steps of CONTROL_DT (25 at robosuite's 2 ms), replayed as one CUDA graph, then
advances every world's reference frame. An episode starts at a random frame of the reference (but a demonstration's last), at the
reference's state there with noise on the arms' joints (the objects and the grippers as recorded, so that a grasp stays a
grasp), and ends at a termination (tasks.robomimic_tracking), after ``episode`` control steps or at its demonstration's
last frame; ended worlds are reset in the same step. With ``stagger``, the first episodes end early, after random numbers of steps.
``reset(mask, frame)`` starts worlds at given frames with their demonstrations' ends as the episodes' (evaluation).

``step_nosync`` (train_robomimic --no_sync, --collect_graph) is ``step`` without host synchronization, as HandSim's: on Warp's
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

from ...tasks.robomimic_tracking import RobomimicTask, unflatten_state


class RobomimicSim:
    def __init__(self, model: mujoco.MjModel, task: RobomimicTask, num_envs: int, device: str, nconmax: int = 64, njmax: int = 512,
                 episode: int = 200, joint_noise: float = 0.01, stagger: bool = False):  # fmt: skip
        self.task, self.ref, self.N, self.device, self.episode, self.joint_noise = task, task.ref, num_envs, device, episode, joint_noise
        self.substeps = round(task.control_dt / model.opt.timestep)
        if abs(self.substeps * model.opt.timestep - task.control_dt) > 1e-9:
            raise ValueError(f"the physics step {model.opt.timestep} does not divide the control step {task.control_dt}")
        if (model.nq, model.nv, model.nu) != (task.nq, task.nv, task.nu):
            raise ValueError("the model is not the task's scene")
        self.arm_q = task.act_q[task.arm]  # the arms' joints, which the starts perturb
        # the frames an episode may start at: every frame but a demonstration's last
        self.starts = torch.nonzero(torch.arange(self.ref.frames, device=device) < self.ref.end).squeeze(-1)
        wp.config.quiet = True
        wp.init()
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(model)
            if hasattr(mjw, "OverflowType") and hasattr(self.model.opt, "warn_overflow"):  # a capped line search ends early by design
                self.model.opt.warn_overflow &= ~int(mjw.OverflowType.LS_ITERATIONS)
            self.data = mjw.put_data(model, mujoco.MjData(model), nworld=num_envs, nconmax=nconmax, njmax=njmax)
            with wp.ScopedCapture() as capture:  # a control step's physics as one CUDA graph
                for _ in range(self.substeps):
                    mjw.step(self.model, self.data)
        self.graph = capture.graph
        self.qpos, self.qvel, self.ctrl = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel), wp.to_torch(self.data.ctrl)
        self.warmstart = wp.to_torch(self.data.qacc_warmstart)
        # the geom pairs of the scene's contact groups, each as one key (the smaller geom id first) with its group
        pairs = [(min(a, b) * model.ngeom + max(a, b), g) for g, group in enumerate(task.scene.contacts) for a, b in group]
        self.ngeom, self.pair_key = model.ngeom, torch.tensor([k for k, _ in pairs], dtype=torch.long, device=device)
        self.pair_group = torch.tensor([g for _, g in pairs], dtype=torch.long, device=device)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last = torch.zeros(num_envs, dtype=torch.long, device=device)  # the frame each world's episode ends at
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        if stagger:  # the first episodes end after random numbers of steps, so that the worlds do not reset in step
            self.last = torch.minimum(self.f + torch.randint(1, episode + 1, (num_envs,), device=device), self.ref.end[self.f])

    def _starts(self, n: int, frame: torch.Tensor | None = None):
        """``n`` episode starts: the frames (random ones but a demonstration's last, or ``frame``), the reference's state there
        with noise on the arms' joints (qpos, qvel), the servos' targets there, and the frames the episodes end at."""
        ref = self.ref
        f = self.starts[torch.randint(0, self.starts.shape[0], (n,), device=self.device)] if frame is None else frame
        qpos = ref.qpos[f].clone()
        qpos[:, self.arm_q] += self.joint_noise * torch.randn(n, len(self.arm_q), device=self.device)
        last = torch.minimum(f + self.episode, ref.end[f]) if frame is None else ref.end[f]
        return f, qpos, ref.qvel[f], ref.ctrl[f], last

    @torch.no_grad()
    def reset(self, mask: torch.Tensor, frame: torch.Tensor | None = None):
        """Start the worlds of ``mask`` (``frame``: at these frames, one per world of the mask, to their demonstrations' ends)."""
        n = int(mask.sum())
        if n == 0:
            return
        self.f[mask], self.qpos[mask], self.qvel[mask], self.ctrl[mask], self.last[mask] = self._starts(n, frame)
        self.warmstart[mask] = 0

    @torch.no_grad()
    def reset_masked(self, mask: torch.Tensor):
        """``reset`` of the worlds of ``mask`` without a host read (train_robomimic --reset_masked): the same writes at fixed
        shapes, the random draws made for every world (another random stream than ``reset``'s)."""
        f, qpos, qvel, ctrl, last = self._starts(self.N)
        m = mask.unsqueeze(-1)
        self.qpos.copy_(torch.where(m, qpos, self.qpos))
        self.qvel.copy_(torch.where(m, qvel, self.qvel))
        self.ctrl.copy_(torch.where(m, ctrl, self.ctrl))
        self.warmstart.masked_fill_(m, 0.0)
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
        """The flat state (N, nq + nv): MuJoCo's qpos and qvel."""
        return torch.cat((self.qpos, self.qvel), -1)

    @torch.no_grad()
    def touching(self) -> torch.Tensor:
        """Whether each world has a pair of each of the scene's contact groups in contact (N, G, bool), in a collision pass
        at the present state: MuJoCo's contact list of a state, which ACT's reward reads after a control step."""
        with wp.ScopedDevice(self.device):
            mjw.kinematics(self.model, self.data)
            mjw.collision(self.model, self.data)
        wp.synchronize_device(self.device)
        n = int(wp.to_torch(self.data.nacon)[0])
        geom, world = wp.to_torch(self.data.contact.geom)[:n].long(), wp.to_torch(self.data.contact.worldid)[:n].long()
        key = geom.amin(-1) * self.ngeom + geom.amax(-1)
        c, p = torch.nonzero(key.unsqueeze(-1) == self.pair_key, as_tuple=True)
        out = torch.zeros(self.N, len(self.task.scene.contacts), dtype=torch.bool, device=self.device)
        out[world[c], self.pair_group[p]] = True
        return out

    def _after(self, s: torch.Tensor, fk):
        """The next state (the state before the step where it is non-finite) and whether each episode terminated."""
        s1 = self.state()
        bad = ~torch.isfinite(s1).all(-1)
        s1 = torch.where(bad.unsqueeze(-1), s, s1)
        S1 = unflatten_state(s1, self.task.nq)
        return s1, bad | self.task.termination(S1, fk(S1[0]), self.f)

    @torch.no_grad()
    def step_nosync(self, u: torch.Tensor, physics=None, fk=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``step`` without host synchronization (the module's docstring); the ended worlds are reset by mask. ``fk``: the
        termination's forward kinematics (train_robomimic --compile_eager's compiled ones; None: the task's)."""
        s = self.state()
        self.ctrl.copy_(u)
        if physics is None:
            wp.capture_launch(self.graph)
        else:
            physics()
        self.f.copy_(self.ref.next_frame(self.f))
        s1, term = self._after(s, self.task.fk if fk is None else fk)
        done = term | (self.f >= self.last)
        self.reset_masked(done)
        return s1, term, done

    @torch.no_grad()
    def step(self, u: torch.Tensor, reset: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hold the targets ``u`` (N, nu) for one control step. Returns the next state, whether the episode terminated (also
        for a non-finite state) and whether it ended; ended worlds are reset unless ``reset`` is False."""
        s = self.state()
        self.ctrl.copy_(u)
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        self.f = self.ref.next_frame(self.f)
        s1, term = self._after(s, self.task.fk)
        done = term | (self.f >= self.last)
        if reset:
            self.reset(done)
        return s1, term, done
