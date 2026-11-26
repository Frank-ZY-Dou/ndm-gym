"""SuperTrack and SuperMimic for animal motion tracking (tasks.animal_tracking), as hand-object tracking and quadrotor tracking
have them (algos.hand_algos, quadrotor_algos): both fit the world model (worldmodels.animal: MjcfWorldModel over a control
step) on k-step windows of simulator transitions and train the policy through imagined rollouts of the frozen world model
from replayed states. The activations of a body with actuator filters (the rat) are not predicted: each imagined step computes
them in closed form from the control (AnimalTask.activation_step). The systems options are the hand task's
(algos.hand_fast), with a collection step of this task's (``AnimalCollector``).

SuperTrack (``AnimalSuperTrack``): the acceleration world model; the policy minimizes, over the imagined window, the L1 errors
of what the task tracks (``TERMS``: the root's position and rotation, the joints, the tracked bodies' and the end effectors'
positions), each term weighted once to 1 at calibration (supertrack.LossBalancer) or, with ``--st_task_weights``, by the
task's reward weight over its kernel's width; plus the failure penalty on the task's soft termination (``--st_fail_w``) and
the regularizer on the actions (``--st_reg``). ``--st_imag_noise`` adds Gaussian noise to the imagined actions. No reward, no
critic.

SuperMimic (``AnimalSuperMimic``; Liao et al., CoRL 2026): the increment world model; the policy maximizes the imagined
tracking return with a soft survival weight and a critic beyond the horizon.

Both take the main trainer's options on the policy's backward pass through the imagined rollout (ndm_gym.train), each off by
default: ``--bptt_decay lam`` scales the gradient through the imagined state by lam at every step, ``--bptt_clip_k k`` caps
each rollout's state gradient after every step at k times the batch's median, ``--pi_rollout_clip k`` caps each rollout's
part of the policy gradient at k times the median over the rollouts. On a body driven by torques a few rollouts that start
shortly before a fall can carry almost all of the gradient (benchmarks/mimic_mjx_animal_tracking); these keep them from
steering the update."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.hand_policy import HandPolicy
from ..tasks.animal_tracking import AnimalTask
from ..tasks.hdmi import relative_rot, rot_angle
from ..worldmodels.animal import AnimalWorldModel
from ..worldmodels.mjcf import MjcfLayout
from .common import set_requires_grad
from .hand_fast import Collector, finite_step, optimizer, survive
from .supertrack import LossBalancer

WINDOW_KEYS = ("s", "s1", "u", "done")
ACC_KEYS = ("r", "steps", "done", "term", "ep_len", "body_err")  # the log's sums over a row's collection steps


def replay_fields(task: AnimalTask) -> dict:
    """A transition: the state, the controls ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (task.state_dim, torch.float32),
        "s1": (task.state_dim, torch.float32),
        "u": (task.nu, torch.float32),
        "a_prev": (task.nu, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


def live_median(size: torch.Tensor) -> torch.Tensor:
    """The median of the positive entries of ``size`` (B,), the lower of the two middle ones as torch.median takes it; inf
    when none is positive. Without a host read: a sort and a gather at an index computed on the device (torch.nanmedian of
    the whole tensor reads its count of NaNs on the host, which an update's CUDA graph cannot hold)."""
    n = (size > 0).sum()
    middle = torch.where(size > 0, size, torch.inf).sort().values.gather(0, ((n - 1).clamp(min=0) // 2).view(1)).squeeze(0)
    return torch.where(n > 0, middle, torch.inf)


class _CapRows(torch.autograd.Function):
    """Identity; in the backward pass every row whose gradient norm exceeds k times the median over the rows with a nonzero
    gradient is scaled down to that norm (common._CapRowGrad, its median without a host read)."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        n = g.flatten(1).norm(dim=1)
        cap = ctx.k * live_median(n)  # inf when every row is zero: nothing is scaled
        s = torch.where(n > cap, cap / n, 1.0)
        return g * s.view(-1, *([1] * (g.dim() - 1))), None


def cap_rollout_grads(S: tuple, k: float) -> tuple:
    """``--bptt_clip_k k`` as the main trainer applies it (common.cap_rollout_grads), on the state after an imagined step:
    the same values; in the policy's backward pass each rollout's gradient on each part of the state is scaled down to at
    most k times the median over the batch, so that a few rollouts whose gradients grow through the world model cannot
    dominate the policy gradient. The other rollouts' gradients are unchanged."""
    return tuple(_CapRows.apply(x, k) for x in S)


def clip_rollout_grads(pi: HandPolicy, inputs: list, taps: list, k: float):
    """``--pi_rollout_clip k`` as common.clip_rollout_output_grads applies it, without a host read (--no_sync and the update's
    CUDA graph): each rollout's part of the policy gradient is scaled down to at most k times the median over the rollouts.
    A rollout's size is the norm over the horizon of the adjoints that reached the policy's output (``taps``: per imagined
    step the zero tensor (B, nu) added to the output, whose gradient is that adjoint); the rollouts without a gradient do
    not count in the median. The backward pass has put every rollout's part in the parameters' gradients; the policy runs
    again on every step's input (``inputs``: per step (B, D), detached) and adds (w_b - 1) times each rollout's part, w_b =
    min(1, k median / size_b): nothing for the rollouts under the cap, at fixed shapes."""
    if any(t.grad is None for t in taps):  # every step's output reaches a later loss: a missing gradient is a broken tap
        raise RuntimeError("--pi_rollout_clip: no gradient reached the policy's output at some imagined step")
    A = torch.stack([t.grad for t in taps])  # (H, B, nu)
    size = A.square().sum((0, 2)).sqrt()
    w = torch.clamp(k * live_median(size) / size.clamp_min(1e-30), max=1.0)  # (no rollout with a gradient: the median inf, every w 1)
    with torch.enable_grad():
        pi.from_input(torch.stack(inputs).flatten(0, 1)).backward(((w - 1).view(1, -1, 1) * A).flatten(0, 1))


class AnimalSuperMimic:
    kind = "increment"  # the world model's parameterization

    def __init__(self, args, task: AnimalTask, wm_layout: MjcfLayout, dev: str, pos_lead: float = 1.0):
        self.args, self.task, self.dev = args, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        self.bptt_clip_k = float(getattr(args, "bptt_clip_k", 0.0))  # --bptt_clip_k (0: off)
        self.pi_rollout_clip = float(getattr(args, "pi_rollout_clip", 0.0))  # --pi_rollout_clip (0: off)
        if self.bptt_clip_k < 0 or self.pi_rollout_clip < 0:
            raise ValueError(f"--bptt_clip_k >= 0 and --pi_rollout_clip >= 0 (0: off), got {self.bptt_clip_k}, {self.pi_rollout_clip}")
        goal_dim = task.goal_frame_dim * self.goal_frames
        lead = {"pos_lead": pos_lead} if self.kind == "accel" else {}
        self.wm = AnimalWorldModel(wm_layout, n_act=task.na, hidden=tuple(args.wm_hidden), act=args.wm_act, layer_norm=not args.wm_no_ln,
                                   kind=self.kind, **lead).to(dev)  # fmt: skip
        self.pi = HandPolicy(task.obs_dim, goal_dim, task.nu).to(dev)
        self.critic = Critic(task.obs_dim + goal_dim).to(dev)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
        self.no_sync = bool(getattr(args, "no_sync", False))
        self.fk = task.kin  # the collection's forward kinematics, run eagerly; --compile_eager compiles them (AnimalCollector)
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.imag_step = torch.compile(self._imag_step)
        self.imag_step_tap = torch.compile(self._imag_step_tap)  # --pi_rollout_clip's
        self.start_features = torch.compile(self._start_features)

    @property
    def optimizers(self):
        return self.opt_wm, self.opt_pi, self.opt_v

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])

    # ------------------------------------------------------------------ the world model's step on the task's state
    def wm_step(self, S, u: torch.Tensor):
        """The next state from ``S`` under the controls ``u``: qpos and qvel from the world model, the activations computed."""
        qpos, qvel, act = S
        q1, v1 = self.wm.step(qpos, qvel, self.task.wm_control(act, u))
        return q1, v1, (self.task.activation_step(act, u)[0] if self.task.na else act)

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer."""
        x = self.task.obs(S, self.fk(S[0]), a_prev), self.task.goal(S, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.update_norms(S[0], S[1], self.task.wm_control(S[2], u), S1[0], S1[1])

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, nu) against ``s1`` (B, k, D): the normalized squared error
        summed over the valid steps, mean over windows. The true activations are the closed form's: no error is taken on them."""
        task = self.task
        S, loss = task.unflatten(s0), 0.0
        for j in range(u.shape[1]):
            S = self.wm_step(S, u[:, j])
            q, v, _ = task.unflatten(s1[:, j])
            e = self.wm.error(S[0], S[1], q, v).pow(2).mean(-1)
            loss = loss + (e * mask[:, j]).sum() / s0.shape[0]
        return loss

    def wm_update(self, buffer: ReplayBuffer) -> torch.Tensor:
        """``--wm_batches`` steps on k-step windows; returns the last step's loss."""
        args = self.args
        for _ in range(args.wm_batches):
            w = buffer.sample_windows(args.wm_batch, args.wm_window)
            loss = self.wm_rollout(w["s"][:, 0], w["u"], w["s1"], w["mask"])
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
            self.opt_wm.step()
        return loss.detach()

    # ------------------------------------------------------------------ policy and critic
    def _start_features(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        fk = self.task.kin(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _bptt_state(self, S):
        """The imagined state as the rollout continues from it: the same values; in the policy's backward pass its gradient
        capped per rollout (--bptt_clip_k), then scaled (--bptt_decay). Nothing with both off."""
        if self.bptt_clip_k > 0:  # --bptt_clip_k: each rollout's gradient on each part of the state at most k times the batch's median
            S = cap_rollout_grads(S, self.bptt_clip_k)
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        return S

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor):
        """One imagined step: the policy's action, the world model's next state and its bodies, the reward, the termination
        probability and the critic's input there."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        S = self._bptt_state(self.wm_step(S, task.setpoint(a, f)))
        fk, f = task.kin(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def _imag_step_tap(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, tap: torch.Tensor):
        """``_imag_step`` for --pi_rollout_clip: ``tap`` (zeros, B x nu) is added to the policy's output, so that its gradient
        is the output's adjoint; the policy's input is returned too (clip_rollout_grads)."""
        task = self.task
        x = torch.cat((task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames)), -1)
        a = self.pi.from_input(x) + tap
        S = self._bptt_state(self.wm_step(S, task.setpoint(a, f)))
        fk, f = task.kin(S[0]), task.ref.next_frame(f)
        return (S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, f), task.critic_features(S, fk, f, a, self.goal_frames),
                x)  # fmt: skip

    def _starts(self, buffer: ReplayBuffer):
        age, env = buffer.sample_start_points(self.args.pi_batch)
        return self.task.unflatten(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        S, f, a_prev = self._starts(buffer)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        fk, x = self.start_features(S, f, a_prev)
        X, rews, dones = [x], [], []
        clip = self.pi_rollout_clip > 0  # --pi_rollout_clip: each step's policy input and a zero tensor added to its output
        taps, inputs = [], []
        for _ in range(horizon):
            if clip:
                taps.append(torch.zeros(B, self.task.nu, device=self.dev, requires_grad=True))
                S, fk, f, a_prev, r, d, x, x_in = self.imag_step_tap(S, fk, f, a_prev, taps[-1])
                inputs.append(x_in.detach())
            else:
                S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev)
            rews.append(r), dones.append(d), X.append(x)
        r, d = torch.stack(rews), torch.stack(dones)
        alive = survive(d, self.no_sync)  # c_{t+1}: cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(horizon, device=self.dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + args.gamma**horizon * alive[-1] * self.critic(X[-1])
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            clip_rollout_grads(self.pi, inputs, taps, self.pi_rollout_clip)
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_step(self.opt_pi, self.pi.parameters(), grad_norm, self.no_sync)  # the step, unless the gradient is non-finite
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)

        X = torch.stack(X).detach()  # (horizon + 1, B, D)
        r, d, c_t = r.detach(), d.detach(), c_t.detach()
        self.critic.norm.update(X)
        with torch.no_grad():
            V = self.critic(X)
            R, targets = V[-1], []
            for t in range(horizon - 1, -1, -1):
                R = r[t] + args.gamma * (1 - d[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1]).reshape(-1)
        inputs = X[:-1].reshape(-1, X.shape[-1])
        minibatch = inputs.shape[0] // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(inputs.shape[0], device=self.dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * minibatch : (i + 1) * minibatch]
                v_loss = (self.critic(inputs[idx]) - targets[idx]).pow(2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        imag_r = (c_t * r).sum(0).mean() / (horizon * self.task.max_step_reward)
        return loss.detach(), v_loss.detach(), imag_r, grad_norm.detach(), alive[-1].mean().detach()


class AnimalSuperTrack(AnimalSuperMimic):
    kind = "accel"
    TERMS = ("root_pos", "root_rot", "joint", "body_pos", "end_eff")

    def __init__(self, args, task: AnimalTask, wm_layout: MjcfLayout, dev: str, pos_lead: float = 1.0):
        super().__init__(args, task, wm_layout, dev, pos_lead)
        self.critic = self.opt_v = None
        self.bal = LossBalancer(len(self.TERMS)).to(dev)
        self.fail_w, self.reg, self.imag_noise = float(args.st_fail_w), float(args.st_reg), float(args.st_imag_noise)
        if self.fail_w < 0 or self.reg < 0 or self.imag_noise < 0:
            raise ValueError(f"--st_fail_w >= 0, --st_reg >= 0, --st_imag_noise >= 0, got {self.fail_w}, {self.reg}, {self.imag_noise}")
        # --st_rate_w (0: off): a cost on the action's change from step to step. This loss has none: its regularizer (--st_reg) is
        # on the action's magnitude, and the errors it sums do not see the action. The task's reward, which SuperMimic's return
        # holds, costs the change (RewardCfg.w_action_rate: 0.05 per unit of mean squared change against 1 per tracking term at
        # its best); with --st_task_weights this loss is in the same units (a tracking term at its kernel's width costs its
        # weight), so 0.05 is the reward's own proportion
        self.rate_w = float(getattr(args, "st_rate_w", 0.0))
        if self.rate_w < 0:
            raise ValueError(f"--st_rate_w >= 0 (0: off), got {self.rate_w}")
        if task.cfg.w_target_rate:  # --target_rate_w: the task's smoothness cost, the one term on the action's change
            if not getattr(args, "st_task_weights", False):
                raise ValueError("--target_rate_w enters SuperTrack's loss at the scale of its task-weighted terms: give --st_task_weights too")
            if self.rate_w:
                raise ValueError("--target_rate_w: without --st_rate_w (it stands in its place)")
        self.step_worth = float(len(self.TERMS))  # a step's worth of loss, the failure penalty's unit: every term at its calibrated 1
        if getattr(args, "st_task_weights", False):  # the task's weights and widths in the calibration's place
            c, L = task.cfg, task.length
            weights = (c.w_root_pos, c.w_root_rot, c.w_joint, c.w_body_pos, c.w_end_eff)
            widths = (c.root_pos_std * L, c.root_rot_std, c.joint_std, c.body_pos_std * L, c.end_eff_std * L)
            self.bal.w.copy_(torch.tensor([w / s for w, s in zip(weights, widths, strict=True)], device=dev) / self.horizon)
            self.bal.scale.copy_(torch.tensor(widths))
            self.bal.ready.fill_(True)
            self.step_worth = sum(weights)  # every error at its kernel's width
        self.st_step = torch.compile(self._st_step)
        self.st_step_tap = torch.compile(self._st_step_tap)  # --pi_rollout_clip's

    @property
    def optimizers(self):
        return self.opt_wm, self.opt_pi

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "bal": self.bal.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.bal.load_state_dict(ck["bal"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    def _errors(self, S, fk: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``."""
        task, ref = self.task, self.task.ref
        p, R = task.root(S[0])
        rp, rR = task.root(ref.qpos[f])
        body = (fk - ref.bodies[f]).abs().sum(-1)
        return torch.stack(((p - rp).abs().sum(-1), rot_angle(relative_rot(rR, R)), (S[0][..., task.jq] - ref.qpos[f][..., task.jq]).abs().mean(-1),
                            body.mean(-1), body[..., task.ee].mean(-1)), -1)  # fmt: skip

    def _st_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """One imagined step: the policy's action (with ``noise`` on it in the rollout), the world model's next state and its
        bodies, the errors against the frame it ends at, the termination probability, the reward."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        S = self._bptt_state(self.wm_step(S, task.setpoint((a + noise).clamp(-1, 1), f)))
        fk, f = task.kin(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, self._errors(S, fk, f), task.soft_termination(S, f), task.reward(S, fk, f, a, a_prev)

    def _st_step_tap(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor, tap: torch.Tensor):
        """``_st_step`` for --pi_rollout_clip: ``tap`` (zeros, B x nu) is added to the policy's output, so that its gradient is
        the output's adjoint; the policy's input is returned too (clip_rollout_grads)."""
        task = self.task
        x = torch.cat((task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames)), -1)
        a = self.pi.from_input(x) + tap
        S = self._bptt_state(self.wm_step(S, task.setpoint((a + noise).clamp(-1, 1), f)))
        fk, f = task.kin(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, self._errors(S, fk, f), task.soft_termination(S, f), task.reward(S, fk, f, a, a_prev), x

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B, nu = self.horizon, self.args.pi_batch, self.task.nu
        S, f, a_prev = self._starts(buffer)
        a_start = a_prev  # the action before the window's first (--st_rate_w)
        set_requires_grad(self.wm, False)
        fk = self.task.kin(S[0])
        errs, dones, acts, rews = [], [], [], []
        clip = self.pi_rollout_clip > 0  # --pi_rollout_clip: each step's policy input and a zero tensor added to its output
        taps, inputs = [], []
        for _ in range(horizon):
            noise = self.imag_noise * torch.randn(B, nu, device=self.dev) if self.imag_noise else torch.zeros(B, nu, device=self.dev)
            if clip:
                taps.append(torch.zeros(B, nu, device=self.dev, requires_grad=True))
                S, fk, f, a_prev, e, d, r, x_in = self.st_step_tap(S, fk, f, a_prev, noise, taps[-1])
                inputs.append(x_in.detach())
            else:
                S, fk, f, a_prev, e, d, r = self.st_step(S, fk, f, a_prev, noise)
            errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
        E, d = torch.stack(errs), torch.stack(dones)  # (H, B, terms), (H, B)
        terms = E.sum(0).mean(0)
        if not self.bal.calibrated:
            self.bal.calibrate(terms.detach().clamp(min=1e-6), torch.ones_like(terms))
        loss = (self.bal.w * terms).sum()
        alive = survive(d, self.no_sync)  # cumprod(1 - d, 0)
        if self.fail_w:  # every imagined step after a failure costs st_fail_w steps' worth of loss
            loss = loss + self.fail_w * (self.step_worth / horizon) * (1 - alive).sum(0).mean()
        A = torch.stack(acts)
        loss = loss + self.reg * ((A**2).mean() + A.abs().mean())
        if self.rate_w:  # --st_rate_w: the action's change from step to step, the window's first from the action before it
            loss = loss + self.rate_w * (A - torch.cat((a_start.unsqueeze(0), A[:-1]))).pow(2).mean()
        if self.task.cfg.w_target_rate:  # --target_rate_w: the smoothness cost's mean over the imagined steps and the batch
            loss = loss + self.task.smooth_cost(A, torch.cat((a_start.unsqueeze(0), A[:-1]))).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            clip_rollout_grads(self.pi, inputs, taps, self.pi_rollout_clip)
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_step(self.opt_pi, self.pi.parameters(), grad_norm, self.no_sync)  # the step, unless the gradient is non-finite
        set_requires_grad(self.wm, True)
        with torch.no_grad():
            c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
            imag_r = (c_t * torch.stack(rews)).sum(0).mean() / (horizon * self.task.max_step_reward)
        return loss.detach(), torch.full((), float("nan"), device=self.dev), imag_r, grad_norm.detach(), alive[-1].mean().detach()


ANIMAL_ALGOS = {"supertrack": AnimalSuperTrack, "supermimic": AnimalSuperMimic}


def collect_step(sim, task, algo, args, a_prev: torch.Tensor, ep_len: torch.Tensor, acc: dict, step, add, fk):
    """One collection step on the loop's state kept in fixed tensors (the previous action, the episode lengths, the log's
    sums ``acc``), without a host read: ``step(u)`` the simulator's control step with its resets, ``add(transition)`` the
    replay's write, ``fk`` the forward kinematics of the next state for the reward and the log."""
    s, f = sim.state(), sim.f.clone()
    S = task.unflatten(s)
    a = algo.act(S, a_prev, f, noise=args.action_noise)
    u = task.setpoint(a, f)
    s1, term, done = step(u)
    S1, f1 = task.unflatten(s1), task.ref.next_frame(f)
    algo.update_wm_norms(S, S1, u)
    add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done})
    fk1 = fk(S1[0])
    ep_len.add_(1)
    acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
    acc["body_err"] += task.errors(S1, fk1, f1)["body"].mean(-1).sum()
    acc["steps"] += sim.N
    acc["done"] += done.sum()
    acc["term"] += term.sum()
    acc["ep_len"] += torch.where(done, ep_len, 0.0).sum()
    ep_len.masked_fill_(done, 0.0)
    a_prev.copy_(torch.where(done.unsqueeze(-1), 0.0, a))


class AnimalCollector(Collector):
    """The collection step for --no_sync and --collect_graph: hand_fast.Collector's streams and capture around this task's
    step (``collect_step`` with AnimalSim.step_nosync); --compile_eager compiles the step's forward kinematics, the agent's too."""

    def __init__(self, sim, task, algo, buffer, args):
        self.sim, self.task, self.algo, self.buffer, self.args = sim, task, algo, buffer, args
        self.graphed = bool(args.collect_graph)
        if self.graphed and buffer.size_t is None:
            raise ValueError("--collect_graph writes at the replay's device-side head (ReplayBuffer.use_device_ptr(device_size=True))")
        self.a_prev = torch.zeros(sim.N, task.nu, device=sim.device)
        self.ep_len = torch.zeros(sim.N, device=sim.device)
        self.acc = {k: torch.zeros((), device=sim.device) for k in ACC_KEYS}
        self.fk = task.kin
        if args.compile_eager:
            self.fk = algo.fk = torch.compile(task.kin)
        self.calls, self.stream, self.graph, self.record = 0, None, None, None

    @torch.no_grad()
    def core(self, physics=None):
        add = self.buffer.add_device if self.graphed else self.buffer.add
        collect_step(self.sim, self.task, self.algo, self.args, self.a_prev, self.ep_len, self.acc,
                     lambda u: self.sim.step_nosync(u, physics), add, self.fk)  # fmt: skip
