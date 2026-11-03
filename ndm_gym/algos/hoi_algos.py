"""SuperTrack and SuperMimic on HDMI's human-object interaction tasks (docs/hoi_design.md §3), each with the settings
selected on LAFAN1: a neural dynamics model of the scene (worldmodels.hoi) fit on replay windows, and a policy
(policies.hoi_policy) trained by backpropagation through imagined rollouts, on which HDMI's own terms run
(algos.hoi.HOIImagination and an HDMITask of the imagination batch).

Imagination starts are replay states with HDMI's full per-env task state rebuilt: the action delay buffer and filter,
the observation histories (slot j: what the end of step t-1-j pushed; before the episode, the reset state's fill;
the policy's copies with fresh noise, as HDMI draws it, if the task has noise), the termination counters and the feet's
timers (recorded), a fresh episodic contact-target noise. A start's observations are those of the step that ended at
it, or at an episode's first row those after the reset (HDMISim.reset_input). Terminations are HDMI's (a term fires after
its run of consecutive steps over the threshold), with a gradient in imagination: a step's exceedance is HDMI's
1[e >= thr] in the forward pass and has the gradient of sigmoid((e - thr) / (0.1 thr)) (a straight-through estimator),
the run length counts on as exceed (count + 1) from the recorded count, so it is HDMI's counter exactly, and a term
fires by sigmoid((count - min_steps + 0.5) / 0.05), within 5e-5 of HDMI's 0 or 1 at every whole count.

Replay rows of a step the guard flagged (hoi_guard; ``bad``) hold finite stand-ins (``sanitize_transition``) and are
left out of the model's windows and normalizers; model and policy updates with a non-finite gradient are skipped and
counted (``wm_skips``, ``pi_skips``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.kinematics import mat_vec
from ..policies.critic import Critic
from ..policies.hoi_policy import HOIPolicy
from ..robots.g1 import NUM_J
from ..sim.rigid.hdmi_sim import ndm_constants
from ..state import unflatten_object, unflatten_state
from ..tasks.hdmi import (EPISODIC_CONTACT_NOISE, NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_JOINT_POS, NUM_ACTIONS, PRIV_HISTORY, HDMITask,
                          action_pipeline, contact_error, contact_reward, full_tracking_score, heading_pos, penalty_terms, relative_rot,
                          reward_selection, rot_angle)  # fmt: skip
from ..worldmodels.hoi import ContextLayout, HOIWorldModel, rigidformer_args
from .common import check_smoothness_args, clip_rollout_output_grads, set_requires_grad, smoothness_penalty
from .hoi import HOIImagination
from .hoi_guard import OBS_BOUND, SkippedUpdates
from .supertrack import LossBalancer

TASK_FIELDS = ("term_count", "feet_air", "feet_cnt", "feet_last")
POLICY_OBS = ("command", "policy", "object")  # the observation groups the policy reads ("priv" is the critic's)


def policy_groups(priv_obs: bool) -> tuple[str, ...]:
    """The observation groups the policy reads: POLICY_OBS, and with train_hoi's ``--priv_obs`` HDMI's privileged group too
    (as HDMI's PPO teacher reads it)."""
    return POLICY_OBS + (("priv",) if priv_obs else ())
GUARD_FIELDS = {"bad": ((), torch.bool)}  # whether the guard flagged the step (sanitize_transition)


def task_fields(n_terms: int) -> dict:
    """Replay fields of HDMITask's counters before each step, which imagination starts need."""
    return {"term_count": (n_terms, torch.int32), "feet_air": (2, torch.long), "feet_cnt": (2, torch.long), "feet_last": (2, torch.float32)}


def task_record(task: HDMITask) -> dict:
    """The counters before a step; HDMITask.step and reset replace them rather than write into them, so no copy is needed."""
    return {"term_count": task.term_count, "feet_air": task.air_count, "feet_cnt": task.contact_count, "feet_last": task.last_air_time}


def sanitize_transition(tr: dict, bad: torch.Tensor, q_prev: torch.Tensor) -> dict:
    """A step's transition with the rows the guard flagged (``bad`` (N,)) made finite for the replay: the robot's and the
    object's state after the step set to those before it, the contact forces to zero and the pushed joint observation to
    the previous one (``q_prev``, the history's newest slot before the step); by selection, since NaN x 0 is NaN. The
    flag is stored as ``bad``, which keeps the rows out of the model's windows and normalizers."""
    rows = lambda v: bad.view(-1, *([1] * (v.dim() - 1)))  # noqa: E731
    out = dict(tr, bad=bad)
    out["s1"], out["o1"] = torch.where(rows(tr["s1"]), tr["s"], tr["s1"]), torch.where(rows(tr["o1"]), tr["o"], tr["o1"])
    out["f_eef"], out["f_feet"] = torch.where(rows(tr["f_eef"]), 0.0, tr["f_eef"]), torch.where(rows(tr["f_feet"]), 0.0, tr["f_feet"])
    out["q_obs"] = torch.where(rows(tr["q_obs"]), q_prev, tr["q_obs"])
    return out


@torch.no_grad()
def masked_update(norm, x: torch.Tensor, keep: torch.Tensor):
    """``norm.update(x)`` (a core.normalizer.RunningNorm) from the rows where ``keep`` (N,) holds, written into the
    normalizer's buffers (a CUDA graph reading them sees the update) and without a host synchronization: the rows left out
    are replaced by zeros rather than multiplied by them (NaN x 0 is NaN); without rows, or once the normalizer is frozen
    at max_count, nothing changes."""
    x = x.reshape(-1, x.shape[-1]).float()
    k = keep.reshape(-1, 1)
    n = k.sum().float()
    m = torch.where(k, x, 0.0).sum(0) / n.clamp(min=1.0)
    v = torch.where(k, (x - m) ** 2, 0.0).sum(0) / n.clamp(min=1.0)
    tot = norm.count + n
    tc = tot.clamp(min=1.0)
    delta = m - norm.mean
    upd = (n > 0) & (norm.count < norm.max_count)
    mean = norm.mean + delta * n / tc
    var = (norm.var * norm.count + v * n + delta**2 * norm.count * n / tc) / tc
    norm.mean.copy_(torch.where(upd, mean, norm.mean))
    norm.var.copy_(torch.where(upd, var, norm.var))
    norm.count.copy_(torch.where(upd, tot, norm.count))


def mix_states(hit: torch.Tensor, now, ref):
    """--st_ref_replay's HDMIState of a step's end: the reference's ``ref`` where ``hit`` (B,), else the rollout's ``now``;
    a field the reference leaves out (None: the contact forces) is the rollout's. ``imagine`` mixes the next step's ``prev``
    with it outside the compiled step: with that mix an output of the step's graph, Inductor's backward returned non-finite
    gradients from the first policy update (move_stool-climb, and every task with --priv_obs; on the CPU too, where it also
    crashed), while the same graph on eager kernels (aot_eager, aot_eager_decomp_partition) and the eager step were finite."""
    sel = lambda r, y: torch.where(hit.view(-1, *([1] * (y.dim() - 1))), r, y)  # noqa: E731
    return type(now)(*(y if (r is None or y is None) else sel(r, y) for r, y in zip(ref, now)))


def soft_terminations(errors, thresholds, min_steps, count, rel_tau: float = 0.1):
    """HDMI's terminations for imagined rollouts (module docstring): the step's termination probability (N,) and the
    new run lengths (N, terms), HDMI's counts in the forward pass."""
    soft = torch.sigmoid((errors - thresholds) / (rel_tau * thresholds))
    exceed = (errors >= thresholds).to(soft.dtype) + (soft - soft.detach())  # HDMI's 0 or 1 (TRM:24), the sigmoid's gradient
    count = exceed * (count + 1)
    fired = torch.sigmoid((count - min_steps + 0.5) / 0.05)
    return 1 - torch.prod(1 - fired, -1), count


class HOIAgent:
    """One HDMI task's learner; ``kind`` is the dynamics model's parameterization (SuperTrack: "accel", SuperMimic:
    "increment"). Built from the training environment (HDMISim) for its task constants."""

    kind: str = ""
    imag_noise: float = 0.0  # std of Gaussian noise on imagined actions
    obj_imag_noise: float = 0.0  # std of Gaussian noise on the imagined object's velocities (m/s, rad/s; SuperTrack's --st_obj_imag_noise)
    replay: float = 0.0  # SuperTrack's --st_ref_replay threshold (m; 0: off)

    def __init__(self, args, env, dev: str):
        self.args, self.dev, self.cfg = args, dev, env.cfg
        self.layout = ContextLayout(*env.context_layout)
        consts = ndm_constants(env.cfg)
        rf = None
        if args.wm_backbone == "rigidformer":
            rf = rigidformer_args(consts, dim=args.rf_dim, layers=args.rf_layers, heads=args.rf_heads, registers=args.rf_registers)
        self.wm = HOIWorldModel(self.kind, env.hinged, env.cfg.num_eefs, self.layout, hidden=args.wm_hidden, act=args.wm_act,
                                layer_norm=not args.wm_no_ln, rigidformer=rf).to(dev)  # fmt: skip
        self.B = args.pi_batch
        self.task = HDMITask(env.cfg, self.B, env.joint_limits, env.robot_mass, noise=True, device=dev)  # the imagination batch
        dims = self.task.obs_dims
        self.groups = policy_groups(getattr(args, "priv_obs", False))
        self.pi = HOIPolicy(dims["command"], dims["policy"], dims["object"], priv_dim=dims["priv"] if "priv" in self.groups else 0).to(dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=args.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
        self.imag = HOIImagination(env.cfg, env.clip, consts, env.hinge_frames() if env.hinged else None, self.layout, dev)
        self.clip, self.L, self.hinged = env.clip, env.L, env.hinged
        self.q0_act, self.scale = env.q0[env.j_act].clone(), env.scale.clone()  # HDMI's action order
        self.kp_com_b, self.obj_scale = env.kp_com_b.clone(), env.obj_scale.clone()  # per env, fixed for the run
        self.obj = (lambda x: (x[..., :1], x[..., 1:2])) if env.hinged else unflatten_object  # noqa: E731
        self.wm_skips, self.pi_skips = SkippedUpdates(), SkippedUpdates()  # updates skipped for a non-finite gradient
        # optional, as on LAFAN1 (algos.common): the model's fit with noise on its normalized inputs (--wm_input_noise) and a
        # finite-difference Jacobian penalty (--wm_smooth_w); each rollout's part of the policy gradient at most k times the
        # median (--pi_rollout_clip)
        self.input_noise = getattr(args, "wm_input_noise", 0.0)
        if self.input_noise < 0:
            raise ValueError(f"--wm_input_noise must be >= 0, got {self.input_noise}")
        self.smooth_w = check_smoothness_args(args)
        self.pi_rollout_clip = getattr(args, "pi_rollout_clip", 0.0)
        if self.pi_rollout_clip < 0:
            raise ValueError(f"--pi_rollout_clip must be >= 0 (0: off), got {self.pi_rollout_clip}")
        # compiled as the LAFAN1 algorithms' world-model rollout and imagination step; also the normalizers' updates of
        # each collection step and the imagination's start (a few fused kernels instead of hundreds of eager ones).
        # The model update replays CUDA graphs (mode "reduce-overhead", the same kernels launched at once); the
        # normalizers it reads keep their buffers in place (masked_update)
        self.wm_window = torch.compile(self._wm_window, mode="reduce-overhead")
        self.wm_smoothness = torch.compile(self._wm_smoothness)
        self.imag_step = torch.compile(self._imag_step)
        self.wm_norm_update = torch.compile(self._wm_norm_update)
        self.pi_norm_update = torch.compile(self._pi_norm_update)
        self.imag_start = torch.compile(self._imag_start)

    @property
    def optimizers(self) -> tuple:
        return (self.opt_wm, self.opt_pi)

    def state_dict(self) -> dict:
        """Everything a resumed run continues from: the networks with their normalizers and the optimizers' states."""
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict()}

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    def _pi_norm_update(self, obs: dict):
        """The policy's normalizers updated from the observations it acts on, without rows beyond the guard's bound (a
        flagged env's observation is replaced at its reset; this also keeps out a reset that left one behind)."""
        rest = torch.cat((obs["policy"], obs["object"]), -1)
        keep = (obs["command"].abs() <= OBS_BOUND).all(-1) & (rest.abs() <= OBS_BOUND).all(-1)
        if "priv" in obs:
            keep = keep & (obs["priv"].abs() <= OBS_BOUND).all(-1)
            masked_update(self.pi.priv_norm, obs["priv"], keep)
        masked_update(self.pi.command_norm, obs["command"], keep)
        masked_update(self.pi.obs_norm, rest, keep)

    @torch.no_grad()
    def update_pi_norms(self, obs: dict):
        self.pi_norm_update(obs)

    def base_action(self, t: torch.Tensor) -> torch.Tensor:
        """HDMI's residual base at frame t: the reference's tracking joints as an action, (q_ref - q0) / scale."""
        return (self.clip.joint_pos[t.clamp(max=self.L - 1)] - self.q0_act) / self.scale

    def act(self, obs: dict, t: torch.Tensor) -> torch.Tensor:
        return self.pi(obs) + self.base_action(t)

    # ------------------------------------------------------------------ world model
    def _wm_norm_data(self, tr: dict) -> tuple:
        """What the model's normalizers are fit on, from one batch of transitions: its inputs, output targets and one
        true step's increments and forces."""
        S, So, S1, So1 = unflatten_state(tr["s"]), self.obj(tr["o"]), unflatten_state(tr["s1"]), self.obj(tr["o1"])
        F1 = (tr["f_eef"], tr["f_feet"])
        return self.wm.inputs(S, So, tr["u"], tr["ctx"]), self.wm.targets(S, So, S1, So1, F1), self.wm.increments(S, So, S1, So1, F1)

    def _wm_norm_update(self, tr: dict):
        x, y, inc = self._wm_norm_data(tr)
        keep = ~tr["bad"]
        masked_update(self.wm.in_norm, x, keep)
        masked_update(self.wm.out_norm, y, keep)
        masked_update(self.wm.err_norm, inc, keep)

    @torch.no_grad()
    def update_wm_norms(self, tr: dict):
        """The model's normalizers updated from a step's transitions (``sanitize_transition``), without the flagged ones."""
        self.wm_norm_update(tr)

    def _wm_window(self, w: dict) -> torch.Tensor:
        """The model's loss on a batch of replay windows (open-loop rollout from each window's first state); a flagged
        step (``bad``) and the window's steps after it are left out."""
        preds = self.wm.rollout(unflatten_state(w["s"][:, 0]), self.obj(w["o"][:, 0]), w["u"], w["ctx"], self.input_noise)
        trues = [(unflatten_state(w["s1"][:, j]), self.obj(w["o1"][:, j]), (w["f_eef"][:, j], w["f_feet"][:, j])) for j in range(w["u"].shape[1])]
        return self.wm.window_loss(preds, trues, w["mask"] * torch.cumprod((~w["bad"]).to(w["mask"].dtype), 1))[0]

    def _wm_smoothness(self, w: dict) -> torch.Tensor:
        """--wm_smooth_w's penalty (algos.common.smoothness_penalty, LAFAN1's) at the model's normalized inputs of the
        windows' first steps, those the window loss keeps."""
        x = self.wm.in_norm(self.wm.inputs(unflatten_state(w["s"][:, 0]), self.obj(w["o"][:, 0]), w["u"][:, 0], w["ctx"][:, 0]))
        a = self.args
        return smoothness_penalty(self.wm.net, x, a.wm_smooth_eps, a.wm_smooth_dirs, (w["mask"][:, 0] > 0) & ~w["bad"][:, 0])

    def wm_update(self, buffer) -> torch.Tensor | None:
        """One model update; skipped (and counted) if its gradient is not finite. Returns the loss per window step (the fit
        loss, without --wm_smooth_w's penalty), None when skipped."""
        a = self.args
        w = buffer.sample_windows(a.wm_batch, a.wm_window)
        loss = self.wm_window(w)
        objective = loss if self.smooth_w == 0 else loss + self.smooth_w * self.wm_smoothness(w)
        self.opt_wm.zero_grad(set_to_none=True)
        objective.backward()
        gn = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        ok = bool(torch.isfinite(gn))
        self.wm_skips(not ok)
        if not ok:
            return None
        self.opt_wm.step()
        return loss.detach() / a.wm_window

    # ------------------------------------------------------------------ imagination starts
    @torch.no_grad()
    def starts(self, buffer) -> dict:
        """B replay states whose clip covers the horizon, with HDMI's task state rebuilt (module docstring); the task of
        the imagination batch is set to it."""
        D, B, H, dev = buffer.data, self.B, self.args.horizon, self.dev
        ages, envs, n = [], [], 0
        while n < B:
            age = torch.randint(0, buffer.size, (2 * B,), device=dev)
            env = torch.randint(0, buffer.N, (2 * B,), device=dev)
            ok = (D["t"][buffer._row(age), env] + H < self.L).nonzero(as_tuple=False).squeeze(-1)[: B - n]
            ages.append(age[ok]), envs.append(env[ok])
            n += len(ok)
        age, env = torch.cat(ages), torch.cat(envs)
        row = buffer._row(age)
        ep = torch.minimum(D["ep_step"][row, env].long(), age)  # steps since the episode began (or the buffer did)

        def back(k: int):  # the row k steps earlier, clamped to the episode's first row; whether it is k steps earlier
            kk = torch.minimum(torch.full_like(ep, k), ep)
            return buffer._row(age - kk), (kk == k)

        g = lambda key, r: D[key][r, env]  # noqa: E731
        backs = [back(k) for k in range(1, PRIV_HISTORY + 1)]  # each once: the action buffer reads 1-3, the histories 1-9
        S, So, ctx, t = unflatten_state(g("s", row)), self.obj(g("o", row)), g("ctx", row), g("t", row)
        abuf = torch.zeros(B, NUM_ACTIONS, 3, device=dev)
        for k in range(1, 4):
            r, v = backs[k - 1]
            abuf[..., k - 1] = g("a", r) * v.unsqueeze(-1)
        r1, v1 = backs[0]
        vf = v1.float()
        u1 = g("u", r1)
        afilt = u1[:, -1] * vf.unsqueeze(-1)
        keep = lambda x, y: torch.where(v1.view(-1, *([1] * (x.dim() - 1))), x, y)  # noqa: E731  the previous row inside the episode
        S_prev = tuple(keep(x, y) for x, y in zip(unflatten_state(g("s", r1)), S))
        So_prev = tuple(keep(x, y) for x, y in zip(self.obj(g("o", r1)), So))
        f_eef_prev, f_feet_prev, u_prev = g("f_eef", r1) * vf[:, None, None], g("f_feet", r1) * vf[:, None, None, None], u1 * vf[:, None, None]
        # observation histories: slot j holds what the end of step t-1-j pushed, else the episode's reset fill; the 9 rows
        # gathered and their rotations made at once; the body-frame angular velocities R^T w as elementwise products
        # (mat_vec), which stay fp32 under train_hoi's allow_tf32, where a batched @ runs in TF32
        S0 = unflatten_state(g("s", buffer._row(age - ep)))
        jh = self.imag.j_hdmi
        fill = mat_vec(S0[1].transpose(-1, -2), S0[3]), -S0[1][:, 2, :], S0[4][:, jh]
        rows, pushed = torch.stack([r for r, _ in backs], 1), torch.stack([v for _, v in backs], 1)  # (B, 9)
        envs = env.unsqueeze(-1).expand_as(rows)
        S1h, vv = unflatten_state(D["s1"][rows, envs]), pushed.unsqueeze(-1)
        ang = mat_vec(S1h[1].transpose(-1, -2), S1h[3])  # (B, 9, 3)
        ang = torch.where(vv, ang, fill[0].unsqueeze(1))
        grav = torch.where(vv, -S1h[1][:, :, 2, :], fill[1].unsqueeze(1))
        q = torch.where(vv, D["q_obs"][rows, envs], fill[2].unsqueeze(1))
        T = self.task
        T.ang_vel_priv, T.gravity_priv, T.joint_pos_priv = ang, grav, q
        if T.noise:
            n3 = lambda x: torch.randn_like(x).clamp(-3.0, 3.0)  # noqa: E731  HDMI's clipped Gaussian
            T.ang_vel_policy = ang[:, :1] + NOISE_ANG_VEL * n3(ang[:, :1])  # noisy whether pushed or filled at reset
            gp = grav[:, :1] + NOISE_GRAVITY * n3(grav[:, :1])
            T.gravity_policy = torch.where(pushed[:, :1, None], gp / gp.norm(dim=-1, keepdim=True), grav[:, :1])
            T.joint_pos_policy = torch.where(pushed[..., None], q + NOISE_JOINT_POS * n3(q), q)
            T.contact_noise = (2 * torch.rand(B, self.cfg.num_eefs, 3, device=dev) - 1) * EPISODIC_CONTACT_NOISE
        else:  # the policy's histories are the privileged ones and the episodic noise is zero, as HDMITask keeps them without noise
            T.ang_vel_policy, T.gravity_policy, T.joint_pos_policy = ang[:, :1], grav[:, :1], q
            T.contact_noise = torch.zeros(B, self.cfg.num_eefs, 3, device=dev)
        T.term_count, T.air_count, T.contact_count, T.last_air_time = g("term_count", row), g("feet_air", row), g("feet_cnt", row), g("feet_last", row)
        delay, alpha = ctx[:, self.layout.startup + NUM_J].round().long(), ctx[:, self.layout.startup + NUM_J + 1 : self.layout.startup + NUM_J + 2]
        return {"S": S, "So": So, "ctx": ctx, "t": t, "env": env, "row": row, "v1": v1, "abuf": abuf, "afilt": afilt, "S_prev": S_prev,
                "So_prev": So_prev, "f_eef_prev": f_eef_prev, "f_feet_prev": f_feet_prev, "u_prev": u_prev, "delay": delay, "alpha": alpha,
                "count0": T.term_count.float()}  # fmt: skip

    # ------------------------------------------------------------------ imagined rollout
    def _imag_start(self, st: dict, kp: torch.Tensor, sc: torch.Tensor, noise: dict | None) -> tuple:
        """The start's state and observations: HDMI's step input of the replay step that ended at the start, observed
        with ``noise`` (HDMITask.draw_noise, drawn outside the compiled graph so that it is eager's random stream). At an
        episode's first row (``v1`` false) it is the reset's input, as HDMISim.reset_input builds it: the reference windows
        of the frame counter t itself and no actuator or hinge torque (the reset state as both states, no actions and no
        forces follow from the start's zeroed history)."""
        imag, ctx, t, v1 = self.imag, st["ctx"], st["t"], st["v1"]
        now = imag.state(st["S"], st["So"], st["f_eef_prev"], ctx, kp, sc)
        x0 = imag.step_input(now, imag.state(st["S_prev"], st["So_prev"], st["f_eef_prev"], ctx, kp, sc), torch.where(v1, t - 1, t), st["u_prev"],
                             st["abuf"], ctx, st["f_feet_prev"])  # fmt: skip
        hinge = None if x0.obj_joint_torque is None else torch.where(v1, x0.obj_joint_torque, 0.0)
        x0 = x0._replace(applied_torque=torch.where(v1.unsqueeze(-1), x0.applied_torque, 0.0), obj_joint_torque=hinge)
        return now, self.task.observations(x0, noise)

    def imagine(self, st: dict, taps: list | None = None):
        """H imagined steps from the starts: the observations at the start, and per step what the policy loss reads
        (``_record``). The dynamics model is frozen; gradients reach the policy through every step. ``taps`` (a list, for
        --pi_rollout_clip): per step the policy's observations and a zero tensor added to its output are appended."""
        e, ctx = st["env"], st["ctx"]
        kp, sc = self.kp_com_b[e], self.obj_scale[e]
        S, So, t = st["S"], st["So"], st["t"]
        prev, obs0 = self.imag_start(st, kp, sc, self.task.draw_noise(push=False))
        obs, abuf, afilt, count = {k: obs0[k] for k in self.groups}, st["abuf"], st["afilt"], st["count0"]
        delay, alpha, out = st["delay"], st["alpha"], []
        for _ in range(self.args.horizon):
            tap = None if taps is None else torch.zeros(self.B, NUM_ACTIONS, device=self.dev, requires_grad=True)
            if tap is not None:
                taps.append((obs, tap))
            S, So, abuf, afilt, prev, obs, count, rec = self.imag_step(S, So, obs, t, abuf, afilt, prev, count, ctx, kp, sc, delay, alpha, tap)
            if self.replay:  # --st_ref_replay: the next prev mixed outside the compiled step (_replay)
                prev = mix_states(rec[-1], *prev)
            out.append(rec)
            t = t + 1
        return obs0, out

    def clip_rollout_grads(self, taps: list) -> int:
        """--pi_rollout_clip after the policy loss's backward pass (algos.common.clip_rollout_output_grads): the policy run
        again on the taps' observations."""

        def rerun(idx):
            return self.pi({k: torch.stack([o[k][idx].detach() for o, _ in taps]).flatten(0, 1) for k in self.groups})

        return clip_rollout_output_grads(rerun, [t for _, t in taps], self.pi_rollout_clip)

    def _imag_step(self, S, So, obs, t, abuf, afilt, prev, count, ctx, kp, sc, delay, alpha, tap=None):
        """One imagined control step: the policy's action (with noise for SuperTrack), HDMI's delay and filter, the
        dynamics model, the step input rebuilt, the task's step and its soft terminations (``count``: the soft run
        lengths). Returns the next step's inputs and the step's record; whatever no record reads (SuperTrack's rewards but
        its --st_task_penalties terms, its privileged observations, every step input) stays out of the compiled graph's outputs.
        ``tap`` (--pi_rollout_clip): a zero tensor added to the policy's output, whose gradient is the output's adjoint.
        With SuperTrack's --st_ref_replay, the rollouts the replay replaces continue from the reference (``_replay``), the
        record ends with which ones they are, and the next ``prev`` is returned as the pair of the rollout's HDMIState and
        the reference's, for ``imagine`` to mix (mix_states)."""
        a = self.act(obs, t)
        if tap is not None:
            a = a + tap
        if self.imag_noise:
            a = a + self.imag_noise * torch.randn_like(a)
        u, abuf, afilt = action_pipeline(a, abuf, afilt, delay, alpha)
        S, So, (fe, ff) = self.wm.step(S, So, u, ctx)
        if self.obj_imag_noise:  # the object's velocities (a hinge's rate) perturbed after every imagined step
            n = self.obj_imag_noise
            if self.hinged:
                So = (So[0], So[1] + n * torch.randn_like(So[1]))
            else:
                So = (So[0], So[1], So[2] + n * torch.randn_like(So[2]), So[3] + n * torch.randn_like(So[3]))
        now = self.imag.state(S, So, fe, ctx, kp, sc)
        x = self.imag.step_input(now, prev, t, u, abuf, ctx, ff)
        T = self.task
        o = T.step(x)
        p, count = soft_terminations(T.termination_errors(x), T.term_thresholds, T.term_min_steps.float(), count)
        if not self.replay:
            return S, So, abuf, afilt, now, {k: o.obs[k] for k in self.groups}, count, self._record(o, x, a, t, p)
        rec = self._record(o, x, a, t, p)
        S, So, ref_now, obs, count, hit = self._replay(S, So, now, {k: o.obs[k] for k in self.groups}, count, x, prev, t, u, abuf, ctx, ff, kp, sc)
        return S, So, abuf, afilt, (now, ref_now), obs, count, rec + (hit,)

    def _replay(self, S, So, now, obs, count, x, prev, t, u, abuf, ctx, ff, kp, sc):
        """SuperTrack's --st_ref_replay (DiffMimic's demonstration replay, arXiv 2304.03274) after an imagined step whose
        losses are recorded: every rollout whose keypoints' mean world-space distance to the reference frame the step is
        measured against (CU(n-1)'s, index 0) exceeds the threshold continues from that frame's reference state
        (``reference_state``), with no gradient through it: its robot and object state, their HDMIState as the next step's
        ``prev`` (the contact forces the model predicted kept), its observation histories filled with that state as HDMI's
        reset fills them and its termination counters cleared (HDMITask.refill), its observations of it, its soft run
        lengths restarted. The action buffer and filter, the feet's timers and the episodic contact noise stay the
        rollout's, and so does the failure penalty: the step's termination probability is the rollout's, and a failure
        before a replacement still costs every later step of the window. Returns the next step's (S, So, obs, count),
        the reference's HDMIState without the contact forces, from which ``imagine`` mixes the next ``prev`` (mix_states),
        and which rollouts were replaced (B,)."""
        hit = (now.body_pos - x.ref.body_pos[:, 0]).detach().norm(dim=-1).mean(-1) > self.replay
        Sr, Sor = self.reference_state(t, kp)
        sel = lambda r, y: torch.where(hit.view(-1, *([1] * (y.dim() - 1))), r, y)  # noqa: E731
        S, So = tuple(sel(r, y) for r, y in zip(Sr, S)), tuple(sel(r, y) for r, y in zip(Sor, So))
        ref_now = self.imag.state(Sr, Sor, now.eef_force, ctx, kp, sc)
        mixed = mix_states(hit, now, ref_now)
        self.task.refill(hit, mixed)
        obs_r = self.task.observations(self.imag.step_input(mixed, prev, t, u, abuf, ctx, ff))
        obs = {k: sel(obs_r[k], v) for k, v in obs.items()}
        ref_now = type(now)(*[None if k == "eef_force" else v for k, v in zip(now._fields, ref_now)])
        return S, So, ref_now, obs, torch.where(hit.unsqueeze(-1), 0.0, count), hit

    def reference_state(self, t: torch.Tensor, kp: torch.Tensor) -> tuple:
        """--st_ref_replay's state of the reference frame ``t`` (B,) (the frame a step from counter t is measured against),
        as HDMI's reset sets it (§2, HDMISim.reset) without its noise: the pelvis's pose and angular velocity, its reference
        centre-of-mass velocity turned into the root link's with the env's centre of mass ``kp`` (B, 16, 3), every joint's
        reference clamped as the reset clamps it (G1 order), the object's reference pose at rest, or the hinge's reference
        angle (clamped to its limits) and rate."""
        c, f = self.clip, t.clamp(max=self.L - 1)
        R0, w0 = c.body_rot[f, 0], c.body_ang_vel[f, 0]
        v0 = c.body_lin_vel[f, 0] - torch.cross(w0, mat_vec(R0, kp[:, 0]), dim=-1)
        rt = self.ref_tables
        q = torch.minimum(torch.maximum(rt["q"][f], rt["q_lo"]), rt["q_hi"])
        qd = torch.minimum(torch.maximum(rt["qd"][f], -rt["qd_max"]), rt["qd_max"])
        if self.hinged:
            So = (c.obj_joint_pos[f].clamp(*rt["hinge_lim"]).unsqueeze(-1), rt["hinge_rate"][f].unsqueeze(-1))
        else:
            z = torch.zeros_like(c.obj_pos[f])
            So = (c.obj_pos[f], c.obj_rot[f], z, z)
        return (c.body_pos[f, 0], R0, v0, w0, q, qd), So

    def _record(self, o, x, a: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> tuple:
        """What the policy loss reads of one imagined step: from the task's output o, the step input x, the action a
        taken at frame t and the step's termination probability p (B,)."""
        raise NotImplementedError


class HOISuperMimic(HOIAgent):
    """SuperMimic: the policy maximizes HDMI's imagined return (the sum of its four reward groups, each times dt) with a
    critic on HDMI's privileged observations at the horizon; the critic regresses GAE(lambda) targets."""

    kind = "increment"

    def __init__(self, args, env, dev):
        super().__init__(args, env, dev)
        self.critic = Critic(self.task.obs_dims["priv"]).to(dev)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_targets = torch.compile(self._critic_targets)
        self.critic_loss = torch.compile(self._critic_loss)

    @property
    def optimizers(self) -> tuple:
        return (self.opt_wm, self.opt_pi, self.opt_v)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "critic": self.critic.state_dict(), "opt_v": self.opt_v.state_dict()}

    def load_state_dict(self, ck: dict):
        super().load_state_dict(ck)
        self.critic.load_state_dict(ck["critic"]), self.opt_v.load_state_dict(ck["opt_v"])

    def _record(self, o, x, a, t, p):
        return o.reward.sum(-1), p, o.obs["priv"]

    def _critic_targets(self, X: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """GAE(lambda) targets (H, B) of the imagined transitions, from the critic's values of their privileged
        observations X (H + 1, B, P), rewards r and termination probabilities d (H, B)."""
        a = self.args
        V = self.critic(X)
        R, targets = V[-1], []
        for h in range(a.horizon - 1, -1, -1):
            R = r[h] + a.gamma * (1 - d[h]) * ((1 - a.lam) * V[h + 1] + a.lam * R)
            targets.append(R)
        return torch.stack(targets[::-1])

    def _critic_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return ((self.critic(X) - Y) ** 2).mean()

    def policy_update(self, buffer):
        a, dev = self.args, self.dev
        st = self.starts(buffer)
        set_requires_grad(self.wm, False), set_requires_grad(self.critic, False)
        taps = [] if self.pi_rollout_clip > 0 else None
        obs0, steps = self.imagine(st, taps)
        r, d = torch.stack([s[0] for s in steps]), torch.stack([s[1] for s in steps])  # (H, B)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, self.B, device=dev), alive[:-1]), 0)
        disc = a.gamma ** torch.arange(a.horizon, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + (a.gamma**a.horizon) * alive[-1] * self.critic(steps[-1][2])
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if taps:
            self.clip_rollout_grads(taps)
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        set_requires_grad(self.wm, True), set_requires_grad(self.critic, True)
        with torch.no_grad():
            X = torch.stack([obs0["priv"]] + [s[2] for s in steps]).detach()  # (H + 1, B, P)
            ok = bool(torch.stack((torch.isfinite(gn), torch.isfinite(X).all(), torch.isfinite(r).all(), torch.isfinite(d).all())).all())
        # a non-finite gradient or imagined rollout skips the actor's and the critic's updates (counted)
        self.pi_skips(not ok)
        if not ok:
            return {"skipped": True}
        self.opt_pi.step()
        # critic: GAE(lambda) targets over the imagined transitions, as SuperMimic's on LAFAN1
        with torch.no_grad():
            self.critic.norm.update(X.flatten(0, 1))
            targets = self.critic_targets(X, r.detach(), d.detach())  # (H, B)
        Xf, Yf = X[:-1].flatten(0, 1), targets.flatten()
        mb = len(Xf) // a.critic_minibatches
        v_loss = torch.zeros((), device=dev)
        for _ in range(a.critic_epochs):
            perm = torch.randperm(len(Xf), device=dev)
            for i in range(a.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = self.critic_loss(Xf[idx], Yf[idx])
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        return {"pi_loss": loss.detach(), "v_loss": v_loss.detach(), "imag_r": r.detach().mean(), "pi_gradnorm": gn.detach(),
                "imag_survival": alive[-1].detach().mean(), "skipped": False}  # fmt: skip


class HOISuperTrack(HOIAgent):
    """SuperTrack: the policy minimizes L1 errors of HDMI's tracked quantities over the imagined window (keypoint
    positions in the world and the root's heading frame, keypoint orientations, the tracking joints, the object's
    position and orientation or the hinge angle, and the end effectors' distance to their contact targets where the
    reference is in contact; with ``--st_contact_force``, HDMI's contact reward as an error instead, which adds the
    predicted force's shortfall), each term weighted once to 1 at calibration (LossBalancer; ``--st_obj_w`` then weighs the
    object's three terms k times), plus the LAFAN1 failure penalty on HDMI's soft terminations; no reward, no critic.
    ``--st_obj_imag_noise`` perturbs the imagined object's velocities after every imagined step. ``--st_vel_terms`` adds
    three terms, calibrated as the others: the L1 errors of the velocities HDMI's tracking group tracks (its keypoints'
    linear and angular velocities, its tracked joints' velocities; VEL_TERMS). ``--st_task_penalties`` adds HDMI's own
    penalties (its loco and feet terms but survival; the tracking groups' kernels stay out) as a cost at the reward's ratio
    to its full tracking score, as LAFAN1's SuperTrack: their mean per imagined step times n / full score, n the number
    of tracking terms. ``--st_late_w`` weighs the tracking terms of later imagined steps more (LAFAN1's linear ramp, before
    the calibration). ``--st_ref_replay tau`` replays the demonstration inside the imagined window (``_replay``): a rollout
    whose keypoints are on average more than tau metres from the reference continues from the reference's state."""

    kind = "accel"
    TERMS = ("keypoint_pos", "keypoint_pos_local", "keypoint_ori", "joint_pos", "object_pos", "object_ori", "contact")
    VEL_TERMS = ("keypoint_lin_vel", "keypoint_ang_vel", "joint_vel")  # --st_vel_terms, after TERMS
    VEL_REWARDS = ("keypoint_lin_vel_tracking_product", "keypoint_ang_vel_tracking_product", "joint_vel_tracking_product")  # their HDMI terms

    def __init__(self, args, env, dev):
        super().__init__(args, env, dev)
        self.terms = self.TERMS + (self.VEL_TERMS if args.st_vel_terms else ())  # the tracking terms, LossBalancer's
        self.bal = LossBalancer(len(self.terms)).to(dev)
        self.imag_noise = args.st_imag_noise
        self.contact = contact_reward(env.cfg) if args.st_contact_force else None  # (kind, parameters) of HDMI's contact reward
        self.obj_imag_noise = args.st_obj_imag_noise
        # --st_obj_w: the object's terms (position, orientation or hinge angle, contact) weigh k times their calibrated weight
        scale = [1.0] * 4 + [float(args.st_obj_w)] * 3 + [1.0] * (len(self.terms) - len(self.TERMS))
        self.term_scale = None if args.st_obj_w == 1 else torch.tensor(scale, device=dev)
        # --st_vel_terms: the keypoints of HDMI's linear and angular velocity terms and the joints of its joint velocity term
        self.vel_idx = tuple(torch.tensor(reward_selection(env.cfg, k), device=dev) for k in self.VEL_REWARDS) if args.st_vel_terms else None
        # --st_task_penalties: HDMI's enabled loco and feet terms but survival (a constant), at the reward's ratio to its full
        # tracking score on the n unit-scaled tracking terms, as LAFAN1's SuperTrack weighs its task penalties
        self.penalties = penalty_terms(env.cfg) if args.st_task_penalties else []
        self.w_pen = len(self.terms) / full_tracking_score(env.cfg) if args.st_task_penalties else 0.0
        # --st_late_w (LAFAN1's): later imagined steps weigh more, 1 at the first step to 1 + w at the last
        late_w = getattr(args, "st_late_w", 0.0)
        self.late = 1 + late_w * torch.arange(args.horizon, device=dev) / max(args.horizon - 1, 1) if late_w != 0 else None
        # --st_ref_replay: DiffMimic's demonstration replay in the imagined window (_replay), from the reference's tables in G1
        # order and the limits HDMI's reset clamps to (HDMISim)
        self.replay = getattr(args, "st_ref_replay", 0.0)
        if self.replay < 0:
            raise ValueError(f"--st_ref_replay must be >= 0 (0: off), got {self.replay}")
        if self.replay:
            j = env.j_g1  # G1 order -> the simulator's joint index
            self.ref_tables = {"q": env.ref_joint_pos_all, "qd": env.ref_joint_vel_all, "q_lo": env.soft_lo[j], "q_hi": env.soft_hi[j],
                               "qd_max": env.vel_limit[j], "hinge_lim": env.hinge_lim, "hinge_rate": env.ref_obj_joint_vel}  # fmt: skip
        self.calibrated = False  # whether bal is known to be calibrated (its flag is read once, not at every update)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "bal": self.bal.state_dict()}

    def load_state_dict(self, ck: dict):
        super().load_state_dict(ck)
        self.bal.load_state_dict(ck["bal"])
        self.calibrated = False

    def _errors(self, x) -> torch.Tensor:
        """(B, terms) per-step L1 errors against CU(n-1)'s reference (index 0 = the frame the step ends at); with
        ``--st_vel_terms`` also those of the velocities HDMI tracks (world frame, its keypoints and joints)."""
        now, ref = x.now, x.ref
        e_pos = (now.body_pos - ref.body_pos[:, 0]).abs().sum(-1).mean(-1)
        e_loc = (heading_pos(now.body_pos, now.root_pos, now.root_rot) - heading_pos(ref.body_pos[:, 0], ref.body_pos[:, 0, 0], ref.body_rot[:, 0, 0]))
        e_loc = e_loc.abs().sum(-1).mean(-1)
        e_ori = rot_angle(relative_rot(ref.body_rot[:, 0], now.body_rot)).mean(-1)
        e_q = (now.joint_pos[:, :NUM_ACTIONS] - ref.joint_pos[:, 0]).abs().mean(-1)
        if self.hinged:
            e_op = (now.obj_joint_pos - ref.obj_joint_pos[:, 0]).abs()
            e_oo = torch.zeros_like(e_op)
        else:
            e_op = (now.obj_pos - ref.obj_pos[:, 0]).abs().sum(-1)
            e_oo = rot_angle(relative_rot(ref.obj_rot[:, 0], now.obj_rot))
        flag = ref.contact[:, 0].float()  # (B, C): C = 1 shares the flag
        if self.contact is not None:  # --st_contact_force: the distance and the predicted force's shortfall, as HDMI rewards them
            e_c = contact_error(now, flag, self.contact[0], **self.contact[1])
        else:
            dist = (now.eef_pos - now.target_pos).norm(dim=-1)  # (B, E)
            e_c = (dist * flag.expand_as(dist)).mean(-1)
        errs = [e_pos, e_loc, e_ori, e_q, e_op, e_oo, e_c]
        if self.vel_idx is not None:
            kv, kw, jv = self.vel_idx
            errs += [(now.body_lin_vel[:, kv] - ref.body_lin_vel[:, 0, kv]).abs().sum(-1).mean(-1),
                     (now.body_ang_vel[:, kw] - ref.body_ang_vel[:, 0, kw]).abs().sum(-1).mean(-1),
                     (now.joint_vel[:, jv] - ref.joint_vel[:, 0, jv]).abs().mean(-1)]  # fmt: skip
        return torch.stack(errs, -1)

    def _penalties(self, o) -> torch.Tensor:
        """(B,) --st_task_penalties' cost of a step: minus HDMI's penalty terms (``penalties``, hdmi.penalty_terms: weight x
        value, each <= 0), in the units of its tracking terms (before dt)."""
        return -torch.stack([o.terms[k] for k in self.penalties], -1).sum(-1)

    def _record(self, o, x, a, t, p):
        rec = (self._errors(x), p, a - self.base_action(t))
        return rec + (self._penalties(o),) if self.penalties else rec

    def policy_update(self, buffer):
        a = self.args
        st = self.starts(buffer)
        set_requires_grad(self.wm, False)
        taps = [] if self.pi_rollout_clip > 0 else None
        _, steps = self.imagine(st, taps)
        E = torch.stack([s[0] for s in steps])  # (H, B, terms)
        if self.late is not None:  # --st_late_w, before the calibration (LAFAN1's)
            E = E * self.late[:, None, None]
        terms = E.sum(0).mean(0)
        if not self.calibrated:
            if not self.bal.ready:
                self.bal.calibrate(terms.detach().clamp(min=1e-6), torch.ones_like(terms))
            self.calibrated = True
        loss = (self.bal.w * terms).sum() if self.term_scale is None else (self.bal.w * self.term_scale * terms).sum()
        alive = torch.cumprod(1 - torch.stack([s[1] for s in steps]), 0)
        n = len(self.terms)
        if a.st_fail_w != 0:  # the failure penalty: each imagined step after a failure costs st_fail_w steps' worth of loss
            loss = loss + a.st_fail_w * (n / a.horizon) * (1 - alive).sum(0).mean()
        A = torch.stack([s[2] for s in steps])  # (H, B, 23): the actions' residuals over the reference
        loss = loss + a.st_reg * ((A**2).mean() + A.abs().mean())
        if self.penalties:  # --st_task_penalties: the mean penalty per imagined step and rollout times n / full score
            loss = loss + self.w_pen * torch.stack([s[3] for s in steps]).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if taps:
            self.clip_rollout_grads(taps)
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        set_requires_grad(self.wm, True)
        ok = bool(torch.isfinite(gn))
        self.pi_skips(not ok)  # a non-finite gradient skips the update (counted)
        if not ok:
            return {"skipped": True}
        self.opt_pi.step()
        out = {"pi_loss": loss.detach(), "pi_gradnorm": gn.detach(), "imag_survival": alive[-1].detach().mean(), "skipped": False}
        if self.replay:  # --st_ref_replay: the fraction of imagined steps after which a rollout was replaced
            out["replay_frac"] = torch.stack([s[-1] for s in steps]).float().mean()
        return out


HOI_ALGOS = {"supertrack": HOISuperTrack, "supermimic": HOISuperMimic}
