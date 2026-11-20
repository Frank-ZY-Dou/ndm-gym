"""SuperMimic (Liao et al., CoRL 2026; arXiv 2610.28480): BPTT of the imagined tracking return through the frozen
world model, with a soft survival weight and a critic bootstrap beyond the horizon (paper §3, Eq. 1-3)."""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..core.nets import deferred_dtype, deferred_grads, mlp_taps, run_mlp, run_mlp_deferred
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
        # optional (train.py --critic_compile): what the critic's terms otherwise run eagerly, as compiled functions: the policy loss's
        # terms after the rollout (the survival products, the discounted rewards, the critic's value of the last state; forward and
        # backward), the critic's values and GAE(lambda) targets over the imagined transitions, and each minibatch step's loss (forward
        # and backward) on the transitions' inputs normalized once per update and cast once to the first product's dtype (the
        # normalizer does not change during the steps). Equal to the default up to float rounding.
        self.critic_compile = getattr(args, "critic_compile", False)
        if self.critic_compile:
            if args.no_critic or self.critic_graphs:
                raise ValueError("--critic_compile needs the critic (no --no_critic) and steps it itself (not with --critic_cudagraphs)")
            self.loss_terms_c = torch.compile(self._loss_terms)
            self.critic_targets_c = torch.compile(self._critic_targets)
            self.critic_loss_c = torch.compile(self._critic_loss)
        # optional (train.py --critic_fused, with --critic_compile): each minibatch step (the loss, its backward pass, the clipping and
        # Adam's step) by Triton kernels around cuBLAS's products (kernels/critic_fused.py), the critic's parameters, their
        # gradients and Adam's state in flat buffers. Equal to --critic_compile's steps up to float rounding.
        self.critic_fused = getattr(args, "critic_fused", False)
        if self.critic_fused:
            if not self.critic_compile:
                raise ValueError("--critic_fused needs --critic_compile (its steps read the inputs normalized once per update)")
            from ..kernels.critic_fused import CriticFused

            self.cfused = CriticFused(self.critic, self.opt_v)
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
        # optional (train.py --pi_encode_once; XMimic's --imag_hoist and --pi_deferred_grads, algos.xmimic): an imagined
        # step's goal depends on its reference frame only, so the policy's goal encoder runs once over every step's goal and
        # the steps run its decoder; with --imag_hoist the steps compute only what depends on the imagined state; with
        # --pi_deferred_grads the decoder's weight gradients are one product per layer over the steps
        self.encode_once = getattr(args, "pi_encode_once", False)
        self.hoist = getattr(args, "imag_hoist", False)
        self.pi_deferred = getattr(args, "pi_deferred_grads", False)
        if (self.hoist and not self.encode_once) or (self.pi_deferred and not self.hoist):
            raise ValueError("--imag_hoist needs --pi_encode_once, and --pi_deferred_grads needs --imag_hoist")
        if self.encode_once:
            if self.pi_rollout_clip > 0:
                raise ValueError("--pi_encode_once: not with --pi_rollout_clip (its rerun of the policy reads every step's goal)")
            self.encode_goals = torch.compile(self._encode_goals, **self.pi_compile)
            self.imag_step_z = torch.compile(self._imag_step_z, **self.pi_compile)
            self.imag_step_h = torch.compile(self._imag_step_h, **self.pi_compile)
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
        if self.critic_fused:  # Adam's state back into the flat buffers, the parameters' bf16 copies from the loaded ones
            self.cfused.attach(), self.cfused.refresh()

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

    def _encode_goals(self, f):
        """--pi_encode_once: the rollout's reference frames (H + 1, B) from the start frames f (B,) on, their goals
        (H + 1, B, 73 K), and the policy's goal latents (H, B, z) of the steps: step t's goal is that of frame f + t, as
        ``_imag_step`` passes it on; the last frame's goal is the critic's."""
        frames = f.unsqueeze(0) + torch.arange(self.HZ + 1, device=f.device).unsqueeze(-1)
        G = self.motion.goal(frames.flatten(), self.K).view(self.HZ + 1, f.shape[0], -1)
        return frames, G, self.pi.encode(G[:-1].flatten(0, 1)).view(self.HZ, f.shape[0], -1)

    def _imag_step_z(self, S, o_hist, f, xi, xi_n, z):
        """``_imag_step`` from the step's goal latent ``z`` (--pi_encode_once): the policy's decoder, no next goal."""
        a = self.setpoint(self.pi.decode(z, o_hist), f)
        S1 = self.wm.step(S, a, xi_n)
        if self.args.bptt_clip_k > 0:
            S1 = cap_rollout_grads(S1, self.args.bptt_clip_k)
        if self.bptt_decay != 1.0:
            S1 = decay_rollout_grads(S1, self.bptt_decay)
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f + 1))
        return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)

    def _imag_step_h(self, S, hn, z, ref_u, root_R, noise, xi, xi_n, taps=None):
        """--imag_hoist: ``_imag_step_z`` on what depends on the imagined state. ``hn`` (B, hist, obs): the observation
        history normalized, each observation once, when it is produced; ``ref_u``: the reference setpoint of the step's
        target frame (--sm_residual; else None); ``root_R``: the reference's root rotation there; ``noise``: the step's
        observation noise. ``taps`` (--pi_deferred_grads): the decoder through ``nets.run_mlp_deferred``. Returns the next
        state, the setpoint, the next normalized history and the decoder's Linear inputs (None without taps)."""
        x = torch.cat((z, hn.flatten(-2)), -1)
        if taps is None:
            a, ins = run_mlp(self.pi.dec, x), None
        else:
            a, ins = run_mlp_deferred(self.pi.dec, x, taps)
        if ref_u is not None:
            a = a + ref_u
        S1 = self.wm.step(S, a, xi_n)
        if self.args.bptt_clip_k > 0:
            S1 = cap_rollout_grads(S1, self.args.bptt_clip_k)
        if self.bptt_decay != 1.0:
            S1 = decay_rollout_grads(S1, self.bptt_decay)
        o1 = self.mdp.obs(S1, a, xi, None, root_R=root_R) + noise
        return S1, a, torch.cat((hn[:, 1:], self.pi.obs_norm(o1).unsqueeze(1)), 1), ins

    def _imagine_once(self, S, o_hist, a_prev, f, xi, xi_n):
        """The imagined rollout with --pi_encode_once: ``policy_update``'s loop over ``imag_step`` from goals encoded
        before it; with --imag_hoist the reference's terms of all frames, the observation noise of all steps (one draw:
        another random stream than a draw per step) and the start history's normalization before it too. Returns the
        steps' states, setpoints and previous setpoints, the frames (H + 1, B) and their goals, the last state, and with
        --pi_deferred_grads the decoder's (taps, Linear inputs (H, B, in) per layer)."""
        HZ, B = self.HZ, f.shape[0]
        F_all, G_all, Z = self.encode_goals(f)
        Zs = Z.unbind(0)
        Ss, As, Aps = [], [], []
        if not self.hoist:
            for t in range(HZ):
                Aps.append(a_prev)
                S, a_prev, o_hist = self.imag_step_z(S, o_hist, F_all[t], xi, xi_n, Zs[t])
                Ss.append(S), As.append(a_prev)
            return Ss, As, Aps, F_all, G_all, S, None
        ref_u = self.ref_u[F_all[1:]].unbind(0) if self.residual else [None] * HZ
        root_R = self.motion.root_R[F_all[1:]].unbind(0)
        noise = self.imag_noise(o_hist).unbind(0)
        hn = self.pi.obs_norm(o_hist)
        taps = mlp_taps(self.pi.dec, (HZ, B), o_hist.device) if self.pi_deferred else None
        steps = None if taps is None else [t.unbind(0) for t in taps]  # the steps' slices: their gradients come back stacked
        ins = []
        for t in range(HZ):
            Aps.append(a_prev)
            step_taps = None if steps is None else [u[t] for u in steps]
            S, a_prev, hn, x = self.imag_step_h(S, hn, Zs[t], ref_u[t], root_R[t], noise[t], xi, xi_n, step_taps)
            Ss.append(S), As.append(a_prev), ins.append(x)
        if taps is None:
            return Ss, As, Aps, F_all, G_all, S, None
        return Ss, As, Aps, F_all, G_all, S, (taps, [torch.stack([x[i] for x in ins]) for i in range(len(taps))])

    def _reward_term(self, S, f, a, a_prev):
        contact = (self.contact_w, self.feet, self.ref_contact) if self.contact_w > 0 else None
        return self.mdp.reward(S, f, a, a_prev, contact=contact), self.mdp.soft_termination(S, f)

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def _loss_terms(self, r, d, S, f, xi_n, g):
        """--critic_compile: the policy loss's terms after the rollout, from its rewards and termination probabilities r, d (H, B)
        and its last state S, frame f, context xi_n and goal g: the objective J (B,) with the critic's bootstrap value, each step's
        survival c_t (H, B), the last step's survival c and the survival-weighted reward sums (B,). The survival products are a loop
        of multiplications (``survival``, the graphable update's)."""
        args, (HZ, B) = self.args, r.shape
        alive = survival(d)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=r.device), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=r.device, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        J = J + (args.gamma**HZ) * c * self.critic(self._critic_input(S, f, xi_n, g))
        return J, c_t, c, (c_t * r).detach().sum(0)

    def _critic_targets(self, X, rews, dones):
        """--critic_compile: from the critic's inputs X (H + 1, B, D) of the imagined transitions and their rewards and
        terminations (H, B), the GAE(lambda) targets (H, B) as the default's loop computes them, and the first H steps' inputs
        normalized (H * B, D) in the dtype of the critic's first product (``run_mlp``'s autocast: its cast of them), which the
        minibatch steps read (``_critic_loss``)."""
        args = self.args
        Xn = self.critic.norm(X)
        V = run_mlp(self.critic.net, Xn).squeeze(-1)  # self.critic(X)
        R = V[-1]
        targets = []
        for t in range(self.HZ - 1, -1, -1):
            R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        return torch.stack(targets[::-1]), Xn[:-1].reshape(-1, X.shape[-1]).to(deferred_dtype(X.device))

    def _critic_loss(self, Xn, Y, idx):
        """--critic_compile: a minibatch step's loss on the rows ``idx`` of the normalized inputs (``_critic_targets``) and the
        targets: the default's ``((self.critic(X[idx]) - Y[idx]) ** 2).mean()``, the normalizer's work done once per update."""
        return ((run_mlp(self.critic.net, Xn[idx]).squeeze(-1) - Y[idx]) ** 2).mean()

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
        S0 = S
        Ss, As, Aps, Fs, Gs, taps, deferred = [], [], [], [], [], [], None
        if self.encode_once:  # --pi_encode_once (--imag_hoist, --pi_deferred_grads): the goals encoded before the rollout
            Ss, As, Aps, F_all, G_all, S, deferred = self._imagine_once(S, o_hist, a_prev, f, xi, xi_n)
            Fs, f, g = list(F_all[1:].unbind(0)), F_all[-1], G_all[-1]
        else:
            g0 = self.motion.goal(f, K)
            g = g0
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
        rews, dones = r.detach(), d.detach()
        if self.critic_compile:  # the survival products, the discounted rewards and the bootstrap value in one compiled function
            J, c_t, c, r_sum = self.loss_terms_c(r, d, S, f, xi_n, g)
        else:
            alive = torch.cumprod(1 - d, 0)  # c_{t+1}
            c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
            disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
            J = (disc * c_t * r).sum(0)
            c = alive[-1]
            r_sum = (c_t * r).detach().sum(0)
            if not args.no_critic:
                critic_input = self.critic_input_c if self.compile_eager else self._critic_input  # --compile_eager: the compiled one
                J = J + (args.gamma**HZ) * c * self.critic(critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if deferred is not None:  # --pi_deferred_grads: the decoder's gradients, one product per layer over the steps
            deferred_grads(self.pi.dec, deferred[1], deferred[0], self.deferred_bf16)
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
                if self.encode_once:  # the rollout's frames (H + 1, B) and goals, as read before it
                    f_c, g_c = F_all.flatten(), G_all.flatten(0, 1)
                else:
                    f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                    g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            if self.critic_graphs:  # the captured step reads the normalizer's tensors
                update_in_place(self.critic.norm, X)
            else:
                self.critic.norm.update(X)
            with torch.no_grad():
                if self.critic_compile:  # the values, their targets and the steps' normalized inputs in one compiled function
                    targets, Xf = self.critic_targets_c(X, rews, dones)
                else:
                    V = self.critic(X)
                    R = V[-1]
                    targets = []
                    for t in range(HZ - 1, -1, -1):
                        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                        targets.append(R)
                    targets = torch.stack(targets[::-1])  # (H, B)
                    Xf = X[:-1].reshape(-1, X.shape[-1])
            Yf = targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            if self.critic_graphs:
                v_loss = self._critic_steps_graph(Xf, Yf, mb)
            else:
                for _ in range(args.critic_epochs):
                    perm = torch.randperm(n, device=dev)
                    for i in range(args.critic_minibatches):
                        idx = perm[i * mb : (i + 1) * mb]
                        if self.critic_fused:  # the step by the kernels
                            v_loss = self.cfused.step(Xf, Yf, idx)
                            continue
                        v_loss = self.critic_loss_c(Xf, Yf, idx) if self.critic_compile else ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
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
        S0 = S
        Ss, As, Aps, Fs, Gs, deferred = [], [], [], [], [], None
        if self.encode_once:  # --pi_encode_once (--imag_hoist, --pi_deferred_grads): the goals encoded before the rollout
            Ss, As, Aps, F_all, G_all, S, deferred = self._imagine_once(S, o_hist, a_prev, f, xi, xi_n)
            Fs, f, g = list(F_all[1:].unbind(0)), F_all[-1], G_all[-1]
        else:
            g0 = self.motion.goal(f, K)
            g = g0
            for t in range(HZ):
                Aps.append(a_prev)
                S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
                Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        rews, dones = r.detach(), d.detach()
        if self.critic_compile:  # the survival products, the discounted rewards and the bootstrap value in one compiled function
            J, c_t, c, r_sum = self.loss_terms_c(r, d, S, f, xi_n, g)
        else:
            alive = survival(d)  # c_{t+1}: cumprod(1 - d, 0) as a loop
            c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
            disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
            J = (disc * c_t * r).sum(0)
            c = alive[-1]
            r_sum = (c_t * r).detach().sum(0)
            if not args.no_critic:
                critic_input = self.critic_input_c if self.compile_eager else self._critic_input
                J = J + (args.gamma**HZ) * c * self.critic(critic_input(S, f, xi_n, g))
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if deferred is not None:  # --pi_deferred_grads: the decoder's gradients, one product per layer over the steps
            deferred_grads(self.pi.dec, deferred[1], deferred[0], self.deferred_bf16)
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
                if self.encode_once:  # the rollout's frames (H + 1, B) and goals, as read before it
                    f_c, g_c = F_all.flatten(), G_all.flatten(0, 1)
                else:
                    f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                    g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            self.critic.norm.update(X)  # --no_sync: in place
            with torch.no_grad():
                if self.critic_compile:  # the values, their targets and the steps' normalized inputs in one compiled function
                    targets, Xf = self.critic_targets_c(X, rews, dones)
                else:
                    V = self.critic(X)
                    R = V[-1]
                    targets = []
                    for t in range(HZ - 1, -1, -1):
                        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                        targets.append(R)
                    targets = torch.stack(targets[::-1])  # (H, B)
                    Xf = X[:-1].reshape(-1, X.shape[-1])
            Yf = targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    if self.critic_fused:  # the step by the kernels
                        v_loss = self.cfused.step(Xf, Yf, idx)
                        continue
                    v_loss = self.critic_loss_c(Xf, Yf, idx) if self.critic_compile else ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
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
