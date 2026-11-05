"""SuperMimic (Liao et al., CoRL 2026; arXiv 2610.28480): BPTT of the imagined tracking return through the frozen
world model, with a soft survival weight and a critic bootstrap beyond the horizon (paper §3, Eq. 1-3)."""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..robots.g1 import G1_JOINT_NAMES, NUM_J, action_scale
from ..state import INV_DIM, normalize_xi, unflatten_state
from ..tasks.contact import foot_bodies, reference_contact
from ..tasks.tracking import CRITIC_EXTRA
from ..worldmodels.increment import IncrementWorldModel
from ..worldmodels.rigidformer import rigidformer_kwargs
from ..worldmodels.semi_implicit import SemiImplicitIncrementWorldModel
from .common import (Algo, cap_rollout_grads, clip_rollout_policy_grads, decay_rollout_grads, reference_setpoints, set_requires_grad, survival,
                     update_in_place, zero_nonfinite_grads)

PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


class SuperMimic(Algo):
    def __init__(self, args, kin, motion, mdp, dev):
        super().__init__(args, kin, motion, mdp, dev)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * self.K).to(dev)
        # optional (train.py --critic_cudagraphs): the critic's minibatch step as a CUDA graph, with Adam's capturable
        # implementation (its step count and learning rate on the device; not bitwise equal to the default one)
        self.critic_graphs = getattr(args, "critic_cudagraphs", False)
        if self.critic_graphs:
            if args.no_critic:
                raise ValueError("--critic_cudagraphs needs the critic (no --no_critic)")
            self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=torch.tensor(args.lr, device=dev), capturable=True, **self.adam_kw)
        else:
            self.opt_v = torch.optim.Adam(self.critic.parameters(), **self.optimizer_kw(args.lr))  # --update_graphs: capturable
        self.critic_graph = None  # with --critic_cudagraphs: the static inputs, then the captured step
        # optional, for ablations of the action parameterization: offsets about the reference pose of the next frame
        # (SuperTrack's) instead of setpoints about the default pose
        self.residual = self.is_residual(vars(args))
        if self.residual:
            self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
            self.ref_u = reference_setpoints(motion, mdp.default_q, self.alpha)
        # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
        self.err_floor = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                                    torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip
        # optional (--contact_reward w, InstantTrack; --ref_contact_fix, BlinkMimic): a foot-contact term in the imagined
        # reward only (tasks/contact.py); the simulator's reward, which the evaluation uses, is unchanged
        self.contact_w = getattr(args, "contact_reward", 0.0)
        if self.contact_w > 0:
            self.feet = foot_bodies()
            self.ref_contact = reference_contact(motion, self.feet, calibrated=args.ref_contact_fix)
        self.wm_err = torch.compile(self._wm_err, **self.wm_compile)
        self.imag_step = torch.compile(self._imag_step, **self.pi_compile)
        self.reward_term = torch.compile(self._reward_term, **self.pi_compile)
        self.critic_input_c = torch.compile(self._critic_input, **self.pi_compile)
        if self.pi_graphs:
            self.mark_static_tensors()

    @staticmethod
    def world_model_cls(a):
        cls = SemiImplicitIncrementWorldModel if a.get("wm_structured", False) else IncrementWorldModel  # --wm_structured (BlinkMimic)
        return cls(hidden=a["wm_hidden"], act=a["wm_act"], layer_norm=not a["wm_no_ln"], rigidformer=rigidformer_kwargs(a))

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
        norm = self.wm.inc_norm if getattr(self.wm, "structured", False) else self.wm.out_norm  # --wm_structured: the increments' scale
        std = torch.maximum(norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1], (e * mask).sum(1)

    def wm_loss(self, S_hat, w):
        loss, e_last, per_window = self.wm_err(S_hat, w["s1"], w["mask"])
        m = w["mask"][:, -1]
        return loss, (e_last.detach() * m).sum() / m.sum().clamp(min=1), per_window.detach()

    # ------------------------------------------------------------------ policy (Eq. 2) and critic (Eq. 3)
    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g, tap=None):
        a = self.pi(o_hist, g)
        if tap is not None:  # --pi_rollout_clip: the gradient that reaches the policy's output lands in tap.grad
            a = a + tap
        a = self.setpoint(a, f)
        S1 = self.wm.step(S, a, xi_n)
        if self.args.bptt_clip_k > 0:
            S1 = cap_rollout_grads(S1, self.args.bptt_clip_k)
        if self.bptt_decay != 1.0:
            S1 = decay_rollout_grads(S1, self.bptt_decay)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.K)

    def _reward_term(self, S, f, a, a_prev):
        contact = (self.contact_w, self.feet, self.ref_contact) if self.contact_w > 0 else None
        return self.mdp.reward(S, f, a, a_prev, contact=contact), self.mdp.soft_termination(S, f)

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def policy_update(self, buffer: ReplayBuffer):
        args, dev, HZ, K = self.args, self.dev, self.HZ, self.K
        if self.pi_graphs:
            torch.compiler.cudagraph_mark_step_begin()
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
        Ss, As, Aps, Fs, Gs, taps = [], [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            if self.pi_rollout_clip > 0:  # the step's policy inputs and a zero tensor added to its output (clip_rollout_policy_grads)
                taps.append((o_hist, g, torch.zeros(B, NUM_J, device=dev, requires_grad=True)))
                S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g, taps[-1][2])
            else:
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
            critic_input = self.critic_input_c if self.compile_eager else self._critic_input  # --compile_eager: the compiled one
            J = J + (args.gamma**HZ) * c * self.critic(critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if taps:
            clip_rollout_policy_grads(self.pi, taps, self.pi_rollout_clip)
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
            if self.critic_graphs:  # the captured step reads the normalizer's tensors
                update_in_place(self.critic.norm, X)
            else:
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
            if self.critic_graphs:
                v_loss = self._critic_steps_graph(Xf, Yf, mb)
            else:
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

    def policy_update_graphable(self, buffer: ReplayBuffer):
        """``policy_update`` for --update_graphs (BlinkMimic's --cuda_graph): the same update without host synchronization,
        so that all of it (start sampling, imagination, objective, backward, clipping, the optimizer step skipped on the
        device for a non-finite gradient, the policy average's step, the critic's targets and minibatch steps) can be
        captured in one CUDA graph. The survival products are a loop of multiplications (cumprod's backward tests for zeros
        on the host). Returns (policy loss, value loss, imagined reward, gradient norm, imagined survival)."""
        args, dev, HZ, K = self.args, self.dev, self.HZ, self.K
        s, h = self.sample_valid_starts_nosync(buffer, args.pi_batch)
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
        alive = survival(d)  # c_{t+1}: cumprod(1 - d, 0) as a loop
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not args.no_critic:
            critic_input = self.critic_input_c if self.compile_eager else self._critic_input
            J = J + (args.gamma**HZ) * c * self.critic(critic_input(S, f, xi_n, g))
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite = torch.isfinite(gn)
        self.nonfinite.add_((~finite).long())
        zero_nonfinite_grads(self.pi.parameters(), finite)
        self.optimizer_step(self.opt_pi, finite)
        if self.pi_avg is not None:  # n_averaged lives on the host: the first update copies the policy (eagerly, before capture)
            self.ema_update() if int(self.pi_avg.n_averaged) > 0 else self.pi_avg.update_parameters(self.pi)
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        self.wm.amp = True
        v_loss = torch.zeros((), device=dev)
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            self.critic.norm.update(X)  # --no_sync: in place
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
        return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()

    def _critic_steps_graph(self, Xf, Yf, mb):
        """--critic_cudagraphs: the default loop's minibatch steps (the same permutations, losses, gradient clipping and
        Adam steps) with each step a CUDA graph, replayed on the minibatch gathered into static tensors. The first policy
        update runs the steps eagerly on a side stream (Adam's state, the libraries' workspaces), the second captures one
        step. Returns the last minibatch's loss."""
        args, n = self.args, Xf.shape[0]
        if self.critic_graph is None:
            self.critic_graph = {"x": Xf.new_empty(mb, Xf.shape[1]), "y": Yf.new_empty(mb)}
        cg = self.critic_graph

        def step():
            loss = ((self.critic(cg["x"]) - cg["y"]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.opt_v.step()
            return loss.detach()

        warm = "warm" not in cg
        stream = torch.cuda.Stream() if warm else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=self.dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    torch.index_select(Xf, 0, idx, out=cg["x"]), torch.index_select(Yf, 0, idx, out=cg["y"])
                    if warm:
                        self.opt_v.zero_grad(set_to_none=True)
                        loss = step()
                        continue
                    if "graph" not in cg:  # the gradients are allocated by the capture and rewritten by every replay
                        self.opt_v.zero_grad(set_to_none=True)
                        cg["graph"] = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cg["graph"], capture_error_mode="thread_local"):
                            cg["loss"] = step()
                    cg["graph"].replay()
                    loss = cg["loss"]
        torch.cuda.current_stream().wait_stream(stream)
        cg["warm"] = True
        return loss.clone()
