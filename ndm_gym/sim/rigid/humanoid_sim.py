"""The humanoid's simulator for training (train_humanoid): QuadrupedSim on a humanoid's scene with the library G1's episode
starts and lengths (sim.rigid.isaaclab_mjwarp.TrackingSim), each an option, off by default (then it is QuadrupedSim's).

- ``max_episode_steps``: a training episode also ends after this many control steps (the G1's 500, 10 s), not only at
  a termination or its clip's last frame.
- ``bin_starts``: episodes start in 1 s bins of the clips (motion.library.BIN_FRAMES frames; a clip's last bin may be
  shorter) drawn by the G1's return-based sampling: half uniformly, 0.4 by a softmax of minus 25 times each bin's
  discounted return over its maximum (low returns more often), 0.1 by a slow average of that softmax; a uniform frame
  within the bin. A bin's return is an exponential average (0.8 on the old value) of the discounted returns of the
  episodes that started in it, a truncated episode's bootstrapped with its mean reward; the episodes of a step that share
  a bin are averaged first (BlinkMimic's fix, the G1's ``--bin_fix``). The return here is the task's reward without the
  action-rate term (the simulator does not see the actions; the term is small), with the training's discount ``gamma``.
- ``fail_reset``: this share of the start bins drawn from the bins before recent failure points, each bin's
  termination rate (an exponential average over the episodes that ended in it) moved to its previous bin (InstantTrack's,
  the G1's ``--fail_reset``; reset_stats.mix_failures).
- ``reset_noise="g1"``: the G1's start perturbation (TrackingSim.reset) in place of QuadrupedSim's Gaussian one: uniform
  noise of +-5 cm horizontally and +-1 cm vertically on the pelvis, a rotation of +-0.1 rad about the horizontal axes and
  +-0.2 rad about the vertical, +-0.5 m/s horizontal and +-0.2 m/s vertical velocity, +-0.52 rad/s (+-0.78 about the
  pelvis' z axis) angular velocity and +-0.1 rad on the joints; the joints' velocities are the reference's.

Evaluation (evaluate_humanoid) uses none of them: its rollouts run from given frames with QuadrupedSim's perturbation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ...core.rotations import mat_to_quat, quat_to_mat, so3_exp
from ...motion.library import BIN_FRAMES
from .quadruped_sim import QuadrupedSim
from .reset_stats import bin_ema_w, mix_failures, previous_bins


def clip_bins(clip_first: torch.Tensor, clip_last: torch.Tensor) -> SimpleNamespace:
    """The reference's 1 s bins as motion.library.MotionLib lays them out: per clip of n frames ceil(n / BIN_FRAMES) bins
    from its first frame; ``bin_start``, ``bin_clip``, ``num_bins`` and every frame's bin ``frame_bin``."""
    starts, clips = [], []
    for c, (a, b) in enumerate(zip(clip_first.tolist(), clip_last.tolist(), strict=True)):
        for s in range(a, b + 1, BIN_FRAMES):
            starts.append(s), clips.append(c)
    dev = clip_first.device
    bin_start, bin_clip = torch.tensor(starts, device=dev), torch.tensor(clips, device=dev)
    frames = int(clip_last.max()) + 1
    frame_bin = torch.searchsorted(bin_start, torch.arange(frames, device=dev), right=True) - 1
    return SimpleNamespace(bin_start=bin_start, bin_clip=bin_clip, num_bins=len(starts), frame_bin=frame_bin)


class HumanoidSim(QuadrupedSim):
    def __init__(self, model, task, num_envs: int, device: str, joint_noise: float = 0.05, nconmax: int = 32, njmax: int = 192,
                 max_episode_steps: int | None = None, bin_starts: bool = False, fail_reset: float = 0.0, reset_noise: str = "gaussian",
                 gamma: float = 0.99):  # fmt: skip
        if not 0 <= fail_reset < 1 or (fail_reset > 0 and not bin_starts):
            raise ValueError(f"fail_reset {fail_reset}: in [0, 1), with bin_starts")
        if reset_noise not in ("gaussian", "g1"):
            raise ValueError(f"reset_noise {reset_noise!r}: 'gaussian' or 'g1'")
        ref = task.ref
        self.max_episode_steps, self.bin_starts, self.fail_reset, self.reset_noise, self.gamma = max_episode_steps, bin_starts, fail_reset, reset_noise, gamma
        self.bins = clip_bins(ref.clip_first, ref.clip_last)
        nb = self.bins.num_bins
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_ret = torch.zeros(num_envs, device=device)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.start_bin = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.G_max = task.max_step_reward / (1 - gamma)
        self.bin_G = torch.zeros(nb, device=device)
        self.bin_w_avg = torch.full((nb,), 1.0 / nb, device=device)
        self.bin_fail = torch.zeros(nb, device=device)
        self.bin_prev = previous_bins(self.bins)
        self.a0 = torch.zeros(1, task.num_joints, device=device)
        # reset_noise="g1": the half-widths of the uniform noise on the pelvis' position, rotation vector, linear and angular
        # velocity, and the joints
        self.g1_noise = torch.tensor([0.05, 0.05, 0.01, 0.1, 0.1, 0.2, 0.5, 0.5, 0.2, 0.52, 0.52, 0.78] + [0.1] * task.num_joints, device=device)
        super().__init__(model, task, num_envs, device, joint_noise=joint_noise, nconmax=nconmax, njmax=njmax)

    # ------------------------------------------------------------------ starts
    def sample_starts(self, n: int, gate: torch.Tensor | None = None) -> torch.Tensor:
        """With ``bin_starts`` the G1's return-based draw (the module's docstring; TrackingSim.sample_frames); QuadrupedSim's
        otherwise. ``gate``: the running average advances only where it holds (reset_masked: some world resets)."""
        if not self.bin_starts:
            return super().sample_starts(n, gate)
        b = self.bins
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        avg = self.bin_w_avg * 0.999 + 0.001 * soft
        self.bin_w_avg.copy_(avg if gate is None else torch.where(gate, avg, self.bin_w_avg))
        probs = 0.5 / b.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            probs = mix_failures(probs, self.bin_fail, self.bin_prev, self.fail_reset)
        bins = torch.multinomial(probs, n, replacement=True)
        start, last = b.bin_start[bins], self.ref.clip_last[b.bin_clip[bins]]
        end = torch.minimum(start + BIN_FRAMES, last)
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return torch.minimum(f, last - 1)

    def perturb(self, qpos: torch.Tensor, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``reset_noise="g1"``: the G1's start perturbation (the module's docstring); QuadrupedSim's otherwise."""
        if self.reset_noise != "g1":
            return super().perturb(qpos, qvel)
        n, half = qpos.shape[0], self.g1_noise
        u = lambda a, b: (2 * torch.rand(n, b - a, device=qpos.device) - 1) * half[a:b]  # noqa: E731
        qpos[:, :3] += u(0, 3)
        qpos[:, 3:7] = mat_to_quat(so3_exp(u(3, 6)) @ quat_to_mat(qpos[:, 3:7]))
        qvel[:, :3] += u(6, 9)
        qvel[:, 3:6] += u(9, 12)  # MuJoCo's free joint: the angular velocity in the pelvis' frame
        qpos[:, 7:] = torch.clamp(qpos[:, 7:] + u(12, half.shape[0]), self.task.kin.q_lo, self.task.kin.q_hi)
        return qpos, qvel

    def started(self, mask: torch.Tensor, f: torch.Tensor):
        """The episodes' counters cleared and their start bins kept, for the worlds of ``mask``."""
        if f.shape[0] == self.N:  # reset_masked: a frame per world
            self.start_bin.copy_(torch.where(mask, self.bins.frame_bin[f], self.start_bin))
        else:
            self.start_bin[mask] = self.bins.frame_bin[f]
        for x in (self.ep_step, self.ep_ret, self.ep_disc_ret):
            x.masked_fill_(mask, 0)

    # ------------------------------------------------------------------ episode ends
    def transition(self, s: torch.Tensor, f: torch.Tensor, fk):
        """QuadrupedSim's, with the episode's length limit, and the bin statistics of the ended episodes (``bin_starts``)."""
        s1, term, done = super().transition(s, f, fk)
        if self.max_episode_steps is None and not self.bin_starts:
            return s1, term, done
        T = self.ep_step + 1
        if self.max_episode_steps is not None:
            done = done | (T >= self.max_episode_steps)
        if self.bin_starts:
            S1 = self.task.unflatten(s1)
            r = self.task.reward(S1, fk(S1[0]), self.f, self.a0, self.a0)
            self.ep_disc_ret.add_(self.gamma ** self.ep_step.float() * r)
            self.ep_ret.add_(r)
            Tf, dw = T.float(), done.float()
            boot = torch.where(term, 0.0, self.gamma**Tf * (self.ep_ret / Tf) / (1 - self.gamma))
            self.bin_G.copy_(bin_ema_w(self.bin_G, self.start_bin, self.ep_disc_ret + boot, dw, 0.8))
            if self.fail_reset > 0:  # the termination rate of the bin each episode ended in; bins no episode ends in fade out
                bf = torch.where(done.any(), self.bin_fail * 0.999, self.bin_fail)
                self.bin_fail.copy_(bin_ema_w(bf, self.bins.frame_bin[f], term.float(), dw, 0.9))
        self.ep_step.copy_(T)
        return s1, term, done
