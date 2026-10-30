"""SuperMimic (Liao et al., CoRL 2026; arXiv 2610.28480): BPTT of the imagined tracking return through the frozen
world model, with a soft survival weight and a critic bootstrap beyond the horizon (paper §3, Eq. 1-3)."""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..robots.g1 import G1_JOINT_NAMES, action_scale
from ..state import INV_DIM, normalize_xi, unflatten_state
from ..tasks.tracking import CRITIC_EXTRA
from ..worldmodels.increment import IncrementWorldModel
from ..worldmodels.rigidformer import rigidformer_kwargs
from .common import Algo, cap_rollout_grads, reference_setpoints, set_requires_grad

PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


class SuperMimic(Algo):
    def __init__(self, args, kin, motion, mdp, dev):
        super().__init__(args, kin, motion, mdp, dev)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * self.K).to(dev)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        # optional, for ablations of the action parameterization: offsets about the reference pose of the next frame
        # (SuperTrack's) instead of setpoints about the default pose
        self.residual = self.is_residual(vars(args))
        if self.residual:
            self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
            self.ref_u = reference_setpoints(motion, mdp.default_q, self.alpha)
        # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
        self.err_floor = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                                    torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip
        self.wm_err = torch.compile(self._wm_err)
        self.imag_step = torch.compile(self._imag_step)
        self.reward_term = torch.compile(self._reward_term)
        self.critic_input_c = torch.compile(self._critic_input)

    @staticmethod
    def world_model_cls(a):
        return IncrementWorldModel(hidden=a["wm_hidden"], act=a["wm_act"], layer_norm=not a["wm_no_ln"], rigidformer=rigidformer_kwargs(a))

    @classmethod
    def is_residual(cls, a):
        return a.get("sm_residual", False)

    def setpoint(self, a, f):
        return a + self.ref_u[f + 1] if self.residual else a

    @property
    def optimizers(self):
        return (self.opt_wm, self.opt_pi, self.opt_v)

    def state_dict(self):
        return {**super().state_dict(), "critic": self.critic.state_dict(), "opt_v": self.opt_v.state_dict()}

    def load_state_dict(self, ck):
        super().load_state_dict(ck)
        self.critic.load_state_dict(ck["critic"]), self.opt_v.load_state_dict(ck["opt_v"])

    # ------------------------------------------------------------------ world model (Eq. 1, state term)
    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1], (e * mask).sum(1)

    def wm_loss(self, S_hat, w):
        loss, e_last, per_window = self.wm_err(S_hat, w["s1"], w["mask"])
        m = w["mask"][:, -1]
        return loss, (e_last.detach() * m).sum() / m.sum().clamp(min=1), per_window.detach()

    # ------------------------------------------------------------------ policy (Eq. 2) and critic (Eq. 3)
    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g):
        a = self.setpoint(self.pi(o_hist, g), f)
        S1 = self.wm.step(S, a, xi_n)
        if self.args.bptt_clip_k > 0:
            S1 = cap_rollout_grads(S1, self.args.bptt_clip_k)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.K)

    def _reward_term(self, S, f, a, a_prev):
        return self.mdp.reward(S, f, a, a_prev), self.mdp.soft_termination(S, f)

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def policy_update(self, buffer: ReplayBuffer):
        args, dev, HZ, K = self.args, self.dev, self.HZ, self.K
        if PROFILE:
            torch.cuda.synchronize()
            tq0 = time.time()
        s, h = self.sample_valid_starts(buffer, args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        o_hist = self.start_obs_history(h)
        S = self.perturb_start(S)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        if args.wm_bptt_fp32:
            self.wm.amp = False
        B = args.pi_batch
        g0 = self.motion.goal(f, K)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not args.no_critic:
            J = J + (args.gamma**HZ) * c * self.critic(self._critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if PROFILE:
            torch.cuda.synchronize()
            tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        self.wm.amp = True
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros(())
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        if PROFILE:
            torch.cuda.synchronize()
            print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
        return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()
