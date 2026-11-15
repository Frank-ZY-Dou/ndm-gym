"""Native Door imagined reward and bootstrapped critic, following HandSuperMimic.

Door has no error termination; finite interior windows have survival one. The
reference's final frame is a true finite-horizon terminal (no value bootstrap).
"""

import torch

from ..policies.critic import Critic
from .door_supertrack import DoorSuperTrack


def lambda_returns(rewards, continuation, values, gamma, lam):
    """Time-major detached critic targets; continuation zero cuts bootstrap."""
    with torch.no_grad():
        ret, targets = values[-1], []
        for t in range(len(rewards) - 1, -1, -1):
            ret = rewards[t] + gamma * continuation[t] * ((1-lam)*values[t+1] + lam*ret)
            targets.append(ret)
        return torch.stack(targets[::-1])


class DoorSuperMimic(DoorSuperTrack):
    """Increment model, imagined discounted reward, and lambda-return critic."""

    model_contract = "increment-squared-v1"

    def __init__(self, *args, critic_epochs=1, critic_minibatches=1, **kwargs):
        if any(type(x) is not int or x < 1 for x in (critic_epochs, critic_minibatches)):
            raise ValueError("Critic epochs and minibatches must be positive integers")
        if kwargs.get("task_weights", False):
            raise ValueError("Task tracking weights are SuperTrack-only")
        if kwargs.get("correction_reg", 0.) or kwargs.get("correction_imag_noise", 0.):
            raise ValueError("Correction regularization/noise are SuperTrack-only")
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        super().__init__(*args, **kwargs)
        # HandSuperMimic uses increments; SuperTrack uses accelerations.
        self.wm = self.world_model_cls(self.layout, hidden=kwargs.get("wm_hidden", (64, 64)),
                                       amp=False, kind="increment").to(self.reference_actions.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=kwargs.get("lr", 1e-3))
        self.critic = Critic(self.task.obs_dim + self.task.goal_frame_dim).to(self.reference_actions.device)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=kwargs.get("lr", 1e-3))
        self.gamma, self.lam = .99, .95
        self.last_value_loss = None

    def wm_loss(self, qpos, qvel, physical_ctrl, valid=None):
        """Upstream squared normalized error, summed over valid rollout steps.

        The divisor is the number of windows, as in HandSuperMimic; a reset
        invalidates that step and every later step in its window.
        """
        q, v = qpos[0], qvel[0]
        total = q.new_zeros(())
        for t in range(physical_ctrl.shape[0]):
            q, v = self.wm.step(q, v, physical_ctrl[t])
            error = self.wm.error(q, v, qpos[t+1], qvel[t+1]).square().mean(-1)
            mask = torch.ones_like(error) if valid is None else valid[:t+1].all(0).to(error.dtype)
            total = total + (error * mask).sum() / qpos.shape[1]
        return total

    def features(self, state, frame, previous):
        return torch.cat((self.task.obs(state, previous), self.task.goal(state, frame)), -1)

    def policy_update(self, state, frame, previous_action):
        if frame.ndim != 1 or torch.any(frame < 0) or torch.any(frame + self.horizon >= self.task.frames):
            raise ValueError("Imagined window must remain within the reference")
        if self.horizon * len(frame) % self.critic_minibatches:
            raise ValueError("Imagined sample count must be divisible by critic minibatches")
        state = tuple(x.detach() for x in state)
        previous = previous_action.detach()
        parameters = list(self.wm.parameters()) + list(self.critic.parameters())
        flags = [p.requires_grad for p in parameters]
        try:
            for p in parameters:
                p.requires_grad_(False)
            features, rewards, continuation = [self.features(state, frame, previous)], [], []
            for _ in range(self.horizon):
                action = self.pi(self.task.obs(state, previous), self.task.goal(state, frame),
                                 self.reference_actions[frame])
                state = self.wm.step(*state, self.setpoint(action), self.task.dt)
                frame = frame + 1
                rewards.append(self.task.reward(state, frame, action, previous))
                continuation.append((frame < self.task.frames - 1).to(state[0].dtype))
                previous = action
                features.append(self.features(state, frame, previous))
            r, c = torch.stack(rewards), torch.stack(continuation)
            alive = c.cumprod(0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
            discount = self.gamma ** torch.arange(self.horizon, device=r.device)[:, None]
            loss = -((discount * before * r).sum(0)
                     + self.gamma**self.horizon * alive[-1] * self.critic(features[-1])).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite imagined Door return")
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(self.pi.parameters(), .1)
            self.opt_pi.step()
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)
        x = torch.stack(features).detach()
        self.critic.norm.update(x)
        targets = lambda_returns(r.detach(), c, self.critic(x).detach(), self.gamma, self.lam)
        inputs = x[:-1].reshape(-1, x.shape[-1])
        targets = targets.reshape(-1)
        minibatch = len(inputs) // self.critic_minibatches
        for _ in range(self.critic_epochs):
            # Preserve the original deterministic full-batch recovery path.
            perm = (torch.randperm(len(inputs), device=x.device) if self.critic_minibatches > 1
                    else torch.arange(len(inputs), device=x.device))
            for index in perm.split(minibatch):
                value_loss = (self.critic(inputs[index]) - targets[index]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., error_if_nonfinite=True)
                self.opt_v.step()
        self.last_value_loss = value_loss.detach()
        return loss.detach(), grad.detach()

    def state_dict(self):
        return {**super().state_dict(), "method": "supermimic", "model_contract": "increment-squared-v1", "critic": self.critic.state_dict(),
                "opt_v": self.opt_v.state_dict(), "gamma": self.gamma, "lam": self.lam,
                "critic_epochs": self.critic_epochs, "critic_minibatches": self.critic_minibatches}

    def load_state_dict(self, state):
        if state.get("model_contract") != "increment-squared-v1":
            raise ValueError("Door SuperMimic model contract mismatch; legacy acceleration checkpoints require historical source")
        if state.get("method") != "supermimic" or state["gamma"] != self.gamma or state["lam"] != self.lam:
            raise ValueError("Door method or return contract mismatch")
        if (state.get("critic_epochs", 1), state.get("critic_minibatches", 1)) != (self.critic_epochs, self.critic_minibatches):
            raise ValueError("Door critic fitting schedule mismatch")
        super().load_state_dict({**state, "method": "supertrack"})
        self.critic.load_state_dict(state["critic"])
        self.opt_v.load_state_dict(state["opt_v"])
