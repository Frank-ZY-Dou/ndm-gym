"""SuperMimic (algos/supermimic.py; Liao et al., CoRL 2026) for hand-object tracking (tasks.hand_tracking): the world
model is fit on k-step windows of simulator transitions, the policy by backpropagating the imagined tracking return
through the frozen world model, with a soft survival weight and a critic beyond the horizon (paper §3, Eq. 1-3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.hand_policy import HandPolicy
from ..robots.xhand import NUM_J, HandKinematics
from ..tasks.hand_tracking import GOAL_FRAME_DIM, OBS_DIM, STATE_DIM, HandTask, unflatten_state
from ..worldmodels.hand import HandWorldModel
from .common import set_requires_grad

WINDOW_KEYS = ("s", "s1", "u", "done")


def replay_fields() -> dict:
    """A transition: the state, the position targets ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (STATE_DIM, torch.float32),
        "s1": (STATE_DIM, torch.float32),
        "u": (NUM_J, torch.float32),
        "a_prev": (NUM_J, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


class HandSuperMimic:
    def __init__(self, args, kin: HandKinematics, task: HandTask, dev: str):
        self.args, self.kin, self.task, self.dev = args, kin, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        goal_dim = GOAL_FRAME_DIM * self.goal_frames
        self.wm = HandWorldModel(hidden=args.wm_hidden, act=args.wm_act, layer_norm=not args.wm_no_ln).to(dev)
        self.pi = HandPolicy(OBS_DIM, goal_dim).to(dev)
        self.critic = Critic(OBS_DIM + goal_dim).to(dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=args.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.imag_step = torch.compile(self._imag_step)
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

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer."""
        x = self.task.obs(S, self.kin.forward(S[0]), a_prev), self.task.goal(S, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.in_norm.update(self.wm.inputs(S, self.kin.forward(S[0]), u))
        self.wm.out_norm.update(self.wm.targets(S, S1))

    # ------------------------------------------------------------------ world model (Eq. 1)
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, J) against ``s1`` (B, k, D): the normalized squared
        error summed over the valid steps, mean over windows."""
        S, loss = unflatten_state(s0), 0.0
        for j in range(u.shape[1]):
            S = self.wm.step(S, self.kin.forward(S[0]), u[:, j])
            e = self.wm.error(S, unflatten_state(s1[:, j])).pow(2).mean(-1)
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

    # ------------------------------------------------------------------ policy (Eq. 2) and critic (Eq. 3)
    def _start_features(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        fk = self.kin.forward(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor):
        """One imagined step: the policy's action, the world model's next state and its forward kinematics, the reward,
        the termination probability and the critic's input there."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        S = self.wm.step(S, fk, task.setpoint(a, f))
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, fk, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        age, env = buffer.sample_start_points(B)
        S, f, a_prev = unflatten_state(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        fk, x = self.start_features(S, f, a_prev)
        X, rews, dones = [x], [], []
        for _ in range(horizon):
            S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev)
            rews.append(r), dones.append(d), X.append(x)
        r, d = torch.stack(rews), torch.stack(dones)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(horizon, device=self.dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + args.gamma**horizon * alive[-1] * self.critic(X[-1])
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(grad_norm):
            self.opt_pi.step()
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


HAND_ALGOS = {"supermimic": HandSuperMimic}
