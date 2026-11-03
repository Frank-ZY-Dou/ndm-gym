"""HDMI's model-free learner for its human-object interaction tasks, PPO with a privileged teacher and an adaptation
module ("PPO-ROA", LeCAR-Lab/HDMI @ 32282f6, active_adaptation/learning/ppo/ppo_roa.py), re-implemented from its
behaviour on our simulator and task (HDMISim + HDMITask): the model-free baseline of the wall-clock comparison with the
model-based algorithms (docs/hoi_design.md). No HDMI code is copied. References are ALIAS:line as in
docs/hdmi_task_spec.md, with PPO = learning/ppo/ppo_roa.py, PC = learning/ppo/common.py, DIST =
learning/modules/distributions.py, TRN = scripts/train.py, HLP = scripts/helpers.py and VN = the VecNorm of torchrl
0.7.0 (the version HDMI pins, setup.py:27), torchrl/envs/transforms/transforms.py:6082-6330.

HDMI's README trains a teacher (``algo=ppo_roa_train``) and then a student from its checkpoint
(``algo=ppo_roa_finetune``; cfg/train_sequential.yaml chains the two); the student, which observes no privileged group,
is the policy HDMI deploys. An iteration of either phase (TRN:143-188, PPO:441-465):

* Rollout: ``train_every`` = 32 steps of every env (PPO:43; 4096 envs, Y:13). Actions are drawn from a diagonal Gaussian
  whose std is a parameter, 1 at initialization (PC:145-164), neither squashed nor clipped (DIST:139-158). The command,
  policy, object and priv groups are normalized by torchrl's VecNorm (``VecNorm``), ``ref_joint_pos_`` is not.
* Values: the critic, one value per reward group (PPO:267-272), of the stored observations and of the observation after
  the last step; a done step's next value is its own value (TRN:153-159), not bootstrapped at all after a termination.
* GAE per group, gamma 0.99, lambda 0.95 (PC:195-225), over HDMI's five reward groups (the fifth, debug, is identically
  zero, §5.6); the groups' advantages are scaled by 1/5, summed and normalized over the batch (PPO:624-639). The value
  targets are the groups' returns; there is no value normalization (PPO:74, 173-177).
* PPO: 3 epochs of 8 minibatches (PPO:44-45, PC:99-117). The loss is the clipped surrogate (0.2), an entropy bonus of
  0.001 and the mean over the groups of the squared value errors, over the transitions with step_count > 1 (teacher)
  or > 5 (student) (PPO:653-693): HDMI's first observation of an episode is all zeros and its second one partly stale
  (§0). The actor's, the critic's and the encoder's gradients are clipped to norm 1 separately (PPO:698-701). Adam at
  3e-4; after every minibatch the policy's learning rate is divided by 1.5 if the KL exceeded 0.02 and multiplied by 1.5
  if it was in (0, 0.005), within [1e-5, 1e-2] (PPO:483-491); the critic's stays 3e-4.
* Adaptation (PPO:507-564): 2 epochs of 8 minibatches of envs. The adaptation module (policy, command and object groups
  -> 256) regresses the teacher's privileged features except at reset observations; in the teacher phase the student's
  action mean is also distilled from the teacher's, the student given the teacher's features ("residual distillation").
  An EMA (0.04 per iteration) of the module drives the student's rollouts.

Teacher ("train"): the actor's input is cat[command, policy, encoder(cat[priv, object])] and its mean the reference's
joints as an action, (q_ref - q0) / scale (OBS:37-50), plus the MLP's output (PPO:223-236); the encoder is trained with
the actor; the normalizer is updated (HLP:152-154). Student ("finetune"): loaded from the teacher's checkpoint with both
actors' std set to 0.5 (PC:166-171) and the policy's learning rate where the teacher's ended (PPO:769-774); its input is
cat[command, policy, the adaptation EMA's prediction], its mean the MLP's output alone; the normalizer is frozen
(HLP:155-157).

Implementation choices with the same values: time-major storage, preallocated on the device; masked means as masked
sums, the learning rate a device tensor and the normalizer's reset rows conditional on a device flag, so that neither
the rollout nor the update needs a host synchronization; a minibatch's rows gathered inside the compiled losses; with
``graphs``, every minibatch step (forward, backward, clipping, both optimizers, the learning rate) replayed as one CUDA
graph. Deviations from HDMI's trainer (all outside the losses) are listed in ndm_gym/train_ppo_hoi.py.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..tasks.hdmi import NUM_ACTIONS

GROUPS = 5  # HDMI's reward groups (ENV:259-289): tracking, object_tracking, loco, feet, and debug, which is always 0 (§5.6)
OBS_GROUPS = ("command", "policy", "object", "priv")  # the normalized groups, stored concatenated in this order
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
STD_MIN = 1e-6  # the distribution clamps its scale (DIST:144)


@dataclass(frozen=True)
class PPOConfig:
    """HDMI's PPO-ROA settings (PPOConfig, PPO:39-86) as its two stages set them (PPO:89-91)."""

    phase: str = "train"  # "train": the teacher; "finetune": the student, from a teacher's checkpoint
    train_every: int = 32  # rollout steps per env and iteration
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    lr: float = 3e-4
    desired_kl: float = 0.01
    entropy_coef: float = 0.001  # entropy_coef_start = entropy_coef_end in both stages
    init_noise_scale: float = 1.0
    load_noise_scale: float = 0.5
    latent_dim: int = 256
    max_grad_norm: float = 1.0
    adapt_epochs: int = 2  # PPO:514
    ema_tau: float = 0.04  # PPO:561
    vecnorm_decay: float = 0.9999  # HLP:146
    vecnorm_eps: float = 1e-4  # torchrl's default (VN)

    def __post_init__(self):
        if self.phase not in ("train", "finetune"):
            raise ValueError(f"phase {self.phase!r}: HDMI's README and train_sequential.yaml use 'train' (teacher) and 'finetune' (student)")

    @property
    def teacher(self) -> bool:
        return self.phase == "train"

    @property
    def min_step_count(self) -> int:
        """A transition enters the PPO losses when its step_count exceeds this (PPO:670-674)."""
        return 1 if self.teacher else 5


# ---------------- networks ----------------
def make_mlp(inp: int, units: tuple[int, ...]) -> nn.Sequential:
    """HDMI's MLP: Linear, LayerNorm, Mish per layer (PC:56-71 with ``layer_norm="before"``, PPO:73)."""
    layers, d = [], inp
    for n in units:
        layers += [nn.Linear(d, n), nn.LayerNorm(n), nn.Mish()]
        d = n
    return nn.Sequential(*layers)


def hdmi_init(module: nn.Module):
    """Every linear layer orthogonal with gain 0.01, zero bias (PPO:336-344); LayerNorms and stds keep theirs."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)


class GaussianActor(nn.Module):
    """HDMI's actor (PPO:238-254, PC:145-164): MLP [512, 256, 256], a linear mean of the 23 actions, and the std a free
    parameter (no exponential), clamped at 1e-6 by the distribution (DIST:143-146)."""

    def __init__(self, inp: int, init_std: float):
        super().__init__()
        self.body = make_mlp(inp, (512, 256, 256))
        self.mean = nn.Linear(256, NUM_ACTIONS)
        self.std = nn.Parameter(torch.full((NUM_ACTIONS,), init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean(self.body(x))


def gaussian_log_prob(a: torch.Tensor, loc: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """log N(a; loc, diag std^2), summed over the actions (torch's Normal under Independent)."""
    return (-(a - loc).square() / (2 * std.square()) - std.log() - LOG_SQRT_2PI).sum(-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Entropy of the diagonal Gaussian; the std does not depend on the state, so it is the batch's mean too."""
    return (0.5 + LOG_SQRT_2PI + std.log()).sum(-1)


def gaussian_kl(loc0: torch.Tensor, std0: torch.Tensor, loc1: torch.Tensor, std1: torch.Tensor) -> torch.Tensor:
    """KL(N(loc0, std0) || N(loc1, std1)) summed over the actions (torch's kl_divergence for Normal, PPO:717-718)."""
    var_ratio = (std0 / std1).square()
    return 0.5 * (var_ratio + ((loc0 - loc1) / std1).square() - 1 - var_ratio.log()).sum(-1)


class VecNorm(nn.Module):
    """torchrl 0.7.0's VecNorm as HDMI uses it (HLP:138-160, VN:6210-6249): per feature, decayed sums of x and x^2 and a
    decayed count; each update first decays them by ``decay`` and then adds a batch, and the batch is normalized with
    the statistics that include it: (x - mean) / std, std = sqrt(max(E[x^2] - mean^2, eps)).

    HDMI's environment returns all-zero observations for every env at a reset (ENV:385-386), and the transform updates
    its statistics with them before the partial reset keeps the reset envs' rows (VN:6210-6216, common.py:2672-2676):
    every step with a reset also adds N zero rows (``absorb_zeros``). The statistics start with one batch of zeros, the
    environment's fake tensordict (HLP:146-147)."""

    def __init__(self, dim: int, decay: float = 0.9999, eps: float = 1e-4):
        super().__init__()
        self.decay, self.eps = decay, eps
        self.register_buffer("sum", torch.zeros(dim))
        self.register_buffer("ssq", torch.zeros(dim))
        self.register_buffer("count", torch.zeros(1))

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """Add the batch x (N, dim) and return it normalized."""
        self.sum.mul_(self.decay).add_(x.sum(0))
        self.ssq.mul_(self.decay).add_(x.square().sum(0))
        self.count.mul_(self.decay).add_(x.shape[0])
        return self.normalize(x)

    def absorb_zeros(self, n: int, when: torch.Tensor | None = None):
        """Add n zero rows: the sums only decay. ``when`` (a device bool): only if true, without a host synchronization
        (otherwise every statistic is multiplied by 1 and gets 0 added, unchanged)."""
        d = self.decay if when is None else torch.where(when, torch.full((), self.decay, dtype=self.sum.dtype, device=self.sum.device), 1.0)
        self.sum.mul_(d)
        self.ssq.mul_(d)
        self.count.mul_(d).add_(n if when is None else n * when)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.sum / self.count
        std = (self.ssq / self.count - mean.square()).clamp_min(self.eps).sqrt()
        return (x - mean) / std.clamp_min(self.eps)


# ---------------- advantages ----------------
def next_values(value: torch.Tensor, last_value: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """The next state's values (T, N, G) of a rollout's values (T, N, G), the values of the observation after its last
    step (N, G) and its done flags (T, N): a done step takes its own value, as HDMI stores no terminal observation
    (TRN:153-159); after a termination GAE drops it."""
    nxt = torch.cat((value[1:], last_value.unsqueeze(0)), 0)
    return torch.where(done.unsqueeze(-1), value, nxt)


def gae(reward, terminated, done, value, next_value, gamma: float, lmbda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """HDMI's GAE (PC:195-225) per reward group, time-major: ``reward``, ``value``, ``next_value`` (T, N, G),
    ``terminated`` and ``done`` (T, N) bool. A termination drops the next value, any done step cuts the trace. HDMI's
    discount is 1: it is set every step (ENV:488) and only feet_air_time could change it, by its soft_discount, which
    is 1 in every task (RF:152-153). Returns the advantages and the returns, advantages + values."""
    nonterm = (~terminated).to(reward.dtype).unsqueeze(-1)
    nondone = (~done).to(reward.dtype).unsqueeze(-1)
    adv = torch.empty_like(reward)
    last = torch.zeros_like(reward[0])
    for t in range(reward.shape[0] - 1, -1, -1):
        delta = reward[t] + gamma * (next_value[t] * nonterm[t]) - value[t]
        last = delta + gamma * lmbda * nondone[t] * last
        adv[t] = last
    return adv, adv + value


def combine_advantages(adv: torch.Tensor) -> torch.Tensor:
    """The groups' advantages (T, N, G) scaled by 1/G (the reward scales, PPO:171-172), summed and normalized over the
    batch, its unbiased std plus 1e-8 (PPO:633-639): (T, N)."""
    s = (adv * (1.0 / adv.shape[-1])).sum(-1)
    return (s - s.mean()) / (s.std() + 1e-8)


# ---------------- rollout storage ----------------
class Rollout:
    """One iteration's transitions, time-major (T, N, ...) and preallocated on the device (TRN:129-134): the normalized
    observation groups concatenated (OBS_GROUPS), the teacher's residual base (``ref_joint_pos_``, zero at reset
    observations), the student's latent (the adaptation EMA's prediction), the action, its log-probability and the
    distribution's mean, the group rewards (debug's column zero), the termination and done flags and HDMI's step_count
    (steps since the reset when the action was taken); the actor's std during the rollout (state independent)."""

    def __init__(self, T: int, N: int, obs_dim: int, latent: int, student: bool, device):
        e = lambda *s, dtype=torch.float32: torch.empty(T, N, *s, dtype=dtype, device=device)  # noqa: E731
        self.T, self.N = T, N
        self.obs, self.ref = e(obs_dim), e(NUM_ACTIONS)
        self.pred = e(latent) if student else None
        self.act, self.loc, self.logp = e(NUM_ACTIONS), e(NUM_ACTIONS), e()
        self.reward = torch.zeros(T, N, GROUPS, device=device)
        self.term, self.done = e(dtype=torch.bool), e(dtype=torch.bool)
        self.step = e(dtype=torch.int32)
        self.std = torch.empty(NUM_ACTIONS, device=device)


class _GraphedStep:
    """``fn()``, which reads and writes only fixed tensors, replayed as one CUDA graph (whole-network capture: forward,
    backward, gradient clipping, optimizer steps). The first ``warmup`` calls run eagerly on a side stream, real steps
    that also compile and create the optimizers' states; the next one is captured and then replayed, later calls replay.
    A replay runs the captured kernels on the same tensors, so the values are those of running ``fn``."""

    def __init__(self, fn, warmup: int = 3):
        self.fn, self.warmup, self.calls, self.graph, self.stream = fn, warmup, 0, None, None

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
            return
        if self.calls < self.warmup:
            self.stream = self.stream or torch.cuda.Stream()
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.fn()
            torch.cuda.current_stream().wait_stream(self.stream)
            self.calls += 1
            return
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.fn()
        self.graph.replay()


# ---------------- the learner ----------------
class PPOROA:
    """HDMI's PPO-ROA for one task. ``dims``: HDMITask.obs_dims. ``compile``: torch.compile the rollout's forward pass,
    the values and advantages, and the losses with their backward passes (the same computations). ``graphs`` (CUDA):
    each PPO and adaptation minibatch step replayed as one CUDA graph (the same kernels on the same tensors); its
    optimizers are then capturable, and the update is bound to the first rollout buffer it is given."""

    def __init__(self, cfg: PPOConfig, dims: dict[str, int], device, compile: bool = False, graphs: bool = False):
        self.cfg, self.device = cfg, torch.device(device)
        if graphs and self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self.graphs = graphs
        self.dims = {k: dims[k] for k in OBS_GROUPS}
        off, self.sl = 0, {}
        for k in OBS_GROUPS:
            self.sl[k] = slice(off, off + self.dims[k])
            off += self.dims[k]
        self.obs_dim = off
        c, p, o, q = (self.dims[k] for k in OBS_GROUPS)
        L = cfg.latent_dim
        self.encoder = nn.Sequential(make_mlp(q + o, (L,)), nn.Linear(L, L))  # priv, object -> privileged features (PPO:190-205)
        self.actor = GaussianActor(c + p + L, cfg.init_noise_scale)  # the teacher: command, policy, features (PPO:259-260)
        self.adapt = nn.Sequential(make_mlp(p + c + o, (L, L)), nn.Linear(L, L))  # policy, command, object -> features (PPO:191-197, 213-218)
        self.actor_adapt = GaussianActor(c + p + L, cfg.init_noise_scale)  # the student: command, policy, predicted features (PPO:264-265)
        # the critic's inputs sorted by name (PPO:267-272, PC:369-370)
        self.critic = nn.Sequential(make_mlp(c + o + p + q, (512, 256, 128)), nn.Linear(128, GROUPS))
        self.nets = nn.ModuleDict({"encoder": self.encoder, "actor": self.actor, "adapt": self.adapt, "actor_adapt": self.actor_adapt,
                                   "critic": self.critic}).to(self.device)  # fmt: skip
        hdmi_init(self.nets)
        self.adapt_ema = copy.deepcopy(self.adapt).requires_grad_(False)  # after the initialization (PPO:344-345)
        self.vecnorm = VecNorm(self.obs_dim, cfg.vecnorm_decay, cfg.vecnorm_eps).to(self.device)
        # the policy's learning rate is a device tensor shared by its parameter groups, so that adapting it does not
        # synchronize; fused Adam reads it at every step
        self.lr_policy = torch.tensor(cfg.lr, device=self.device)
        pol = [self.actor, self.encoder] if cfg.teacher else [self.actor_adapt]  # PPO:347-356
        cuda = self.device.type == "cuda"
        adam = lambda groups, lr: torch.optim.Adam(groups, lr=lr, fused=cuda, capturable=graphs)  # noqa: E731
        self.opt_policy = adam([{"params": m.parameters()} for m in pol], self.lr_policy)
        for g in self.opt_policy.param_groups:
            g["lr"] = self.lr_policy
        self.opt_critic = adam(self.critic.parameters(), cfg.lr)
        self.opt_adapt = adam(self.adapt.parameters(), cfg.lr)
        self.opt_adapt_actor = adam(self.actor_adapt.parameters(), cfg.lr) if cfg.teacher else None  # PPO:375-381
        self.policy_actor = self.actor if cfg.teacher else self.actor_adapt
        # the parameter groups whose gradients are clipped separately (PPO:698-703): actor, critic, and the teacher's encoder
        self._clip_groups = [list(self.policy_actor.parameters()), list(self.critic.parameters())]
        if cfg.teacher:
            self._clip_groups.append(list(self.encoder.parameters()))
        cmp = (lambda f: torch.compile(f, dynamic=False)) if compile else (lambda f: f)  # noqa: E731
        self._act = cmp(self._act_impl)
        self._values = cmp(self._values_impl)
        self._advantages = cmp(self._advantages_impl)
        self._ppo_loss = cmp(self._ppo_loss_impl)
        self._adapt_loss = cmp(self._adapt_loss_impl)
        self._distill_loss = cmp(self._distill_loss_impl)
        self._zero = torch.zeros(1, self.obs_dim, device=self.device)
        self._st = None

    # ---------------- observations ----------------
    def cat_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs[k] for k in OBS_GROUPS], -1)

    def group(self, obs: torch.Tensor, k: str) -> torch.Tensor:
        return obs[..., self.sl[k]]

    def observe(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """A step's observation groups, normalized: the teacher's normalizer adds them first (HLP:152-154), the
        student's is frozen (HLP:155-157)."""
        x = self.cat_obs(obs)
        return self.vecnorm.update(x) if self.cfg.teacher else self.vecnorm.normalize(x)

    def reset_observation(self, num_envs: int, when: torch.Tensor | None = None) -> torch.Tensor:
        """The normalized observation HDMI's reset returns (all groups zero, ENV:385-386), (1, D); the teacher's
        normalizer first adds the reset's ``num_envs`` zero rows (``VecNorm``), if ``when`` (a device bool: whether any
        env is reset) is true or not given."""
        if self.cfg.teacher:
            self.vecnorm.absorb_zeros(num_envs, when)
        return self.vecnorm.normalize(self._zero)

    # ---------------- forward passes ----------------
    def _inputs(self, obs: torch.Tensor):
        g = lambda k: self.group(obs, k)  # noqa: E731
        return g("command"), g("policy"), g("object"), g("priv")

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """The teacher's privileged features, encoder(cat[priv, object])."""
        _, _, o, q = self._inputs(obs)
        return self.encoder(torch.cat((q, o), -1))

    def predicted_features(self, obs: torch.Tensor, ema: bool) -> torch.Tensor:
        """The adaptation module's (or its EMA's) prediction of the features from cat[policy, command, object]."""
        c, p, o, _ = self._inputs(obs)
        return (self.adapt_ema if ema else self.adapt)(torch.cat((p, c, o), -1))

    def teacher_mean(self, obs: torch.Tensor, ref: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        c, p, _, _ = self._inputs(obs)
        return ref + self.actor(torch.cat((c, p, feat), -1))

    def student_mean(self, obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        c, p, _, _ = self._inputs(obs)
        return self.actor_adapt(torch.cat((c, p, pred), -1))

    def _act_impl(self, obs: torch.Tensor, ref: torch.Tensor, deterministic: bool = False):
        """The rollout policy (PPO:406-439): action, its log-probability, the mean, and the student's latent (None for
        the teacher)."""
        if self.cfg.teacher:
            pred, loc = None, self.teacher_mean(obs, ref, self.features(obs))
        else:
            pred = self.predicted_features(obs, ema=True)
            loc = self.student_mean(obs, pred)
        std = self.policy_actor.std.clamp_min(STD_MIN)
        a = loc if deterministic else loc + std * torch.randn_like(loc)
        return a, gaussian_log_prob(a, loc, std), loc, pred

    @torch.no_grad()
    def act(self, obs: torch.Tensor, ref: torch.Tensor, deterministic: bool = False):
        return self._act(obs, ref, deterministic)

    def _values_impl(self, obs: torch.Tensor) -> torch.Tensor:
        c, p, o, q = self._inputs(obs)
        return self.critic(torch.cat((c, o, p, q), -1))

    @torch.no_grad()
    def values(self, obs: torch.Tensor) -> torch.Tensor:
        return self._values(obs)

    def _advantages_impl(self, buf_obs, last_obs, reward, term, done):
        v = self._values_impl(buf_obs)
        nv = next_values(v, self._values_impl(last_obs), done)
        adv, ret = gae(reward, term, done, v, nv, self.cfg.gamma, self.cfg.lmbda)
        return combine_advantages(adv), ret

    # ---------------- losses ----------------
    def _ppo_loss_impl(self, obs, ref, pred, act, logp_old, loc_old, std_old, adv, ret, valid, idx):
        """HDMI's PPO loss (PPO:653-693) on the rows ``idx`` of the iteration's flat data, and its diagnostics."""
        cfg = self.cfg
        rows = lambda x: x.index_select(0, idx)  # noqa: E731
        obs, act, logp_old, loc_old, adv, ret, valid = (rows(x) for x in (obs, act, logp_old, loc_old, adv, ret, valid))
        if cfg.teacher:  # the encoder is trained with the actor (PPO:656-658)
            loc = self.teacher_mean(obs, rows(ref), self.features(obs))
        else:  # the stored prediction of the rollout's adaptation EMA
            loc = self.student_mean(obs, rows(pred))
        std = self.policy_actor.std.clamp_min(STD_MIN)
        logp = gaussian_log_prob(act, loc, std)
        entropy = gaussian_entropy(std)
        log_ratio = logp - logp_old
        ratio = log_ratio.exp()
        surr = torch.minimum(adv * ratio, adv * ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param))
        w = valid.to(surr.dtype)
        n = w.sum()
        policy_loss = -(surr * w).sum() / n
        value_loss = ((ret - self._values_impl(obs)).square() * w.unsqueeze(-1)).sum(0) / n  # (G,), PPO:688-691
        loss = policy_loss - cfg.entropy_coef * entropy + value_loss.mean()
        with torch.no_grad():
            kl = gaussian_kl(loc_old, std_old.clamp_min(STD_MIN), loc, std).mean()
            clipfrac = ((ratio - 1.0).abs() > cfg.clip_param).float().mean()
        return loss, policy_loss.detach(), value_loss.detach(), entropy.detach(), kl, clipfrac

    def _adapt_loss_impl(self, obs, feat, keep, envs):
        """The adaptation module's squared error to the features on the envs ``envs`` (all their steps), zero at reset
        observations, averaged over all entries (PPO:516-518)."""
        obs, feat, keep = obs.index_select(1, envs), feat.index_select(1, envs), keep.index_select(1, envs)
        return ((self.predicted_features(obs, ema=False) - feat).square() * keep.unsqueeze(-1)).mean()

    def _distill_loss_impl(self, obs, feat, teacher_loc, envs):
        """Residual distillation (PPO:529-545): the student's mean, given the teacher's features, to the teacher's."""
        obs, feat, teacher_loc = obs.index_select(1, envs), feat.index_select(1, envs), teacher_loc.index_select(1, envs)
        return (teacher_loc - self.student_mean(obs, feat)).square().mean()

    # ---------------- update ----------------
    def _adapt_lr(self, kl: torch.Tensor):
        """HDMI's KL-adaptive learning rate (PPO:483-491), on the device."""
        k, lr = self.cfg.desired_kl, self.lr_policy
        down = (lr / 1.5).clamp_min(1e-5)
        up = (lr * 1.5).clamp_max(1e-2)
        lr.copy_(torch.where(kl > 2.0 * k, down, torch.where((kl < k / 2.0) & (kl > 0.0), up, lr)))

    def _static(self, buf: Rollout) -> dict:
        """The update's fixed tensors for the rollout buffer ``buf``: its flat views, the advantages, returns and loss
        mask, a minibatch's rows or envs, the adaptation targets and the diagnostics' sums, with the minibatch steps
        that read them (CUDA-graphed with ``graphs``, which binds them to this buffer)."""
        if self._st is not None:
            if self._st["buf"] is buf:
                return self._st
            if self.graphs:
                raise ValueError("the CUDA-graphed update is bound to the first rollout buffer it was given")
        cfg, T, N, dev = self.cfg, buf.T, buf.N, self.device
        nmb = cfg.num_minibatches
        if T * N // nmb == 0 or N // nmb == 0:
            raise ValueError(f"{N} envs x {T} steps cannot be split into {nmb} minibatches of transitions and of envs")
        flat = lambda x: None if x is None else x.view(T * N, *x.shape[2:])  # noqa: E731
        e = lambda *s, dtype=torch.float32: torch.empty(*s, dtype=dtype, device=dev)  # noqa: E731
        st = {"buf": buf, "obs": flat(buf.obs), "ref": flat(buf.ref), "pred": flat(buf.pred), "act": flat(buf.act), "logp": flat(buf.logp),
              "loc": flat(buf.loc), "adv": e(T * N), "ret": e(T * N, GROUPS), "valid": e(T * N, dtype=torch.bool),
              "idx": e(T * N // nmb, dtype=torch.long), "sums": torch.zeros(6 + GROUPS, device=dev), "feat": e(T, N, cfg.latent_dim),
              "tloc": e(T, N, NUM_ACTIONS) if cfg.teacher else None, "keep": e(T, N), "envs": e(N // nmb, dtype=torch.long),
              "asums": torch.zeros(2, device=dev)}  # fmt: skip
        run = _GraphedStep if self.graphs else (lambda f: f)
        st["ppo_step"] = run(lambda: self._ppo_step(st))
        st["adapt_step"] = run(lambda: self._adapt_step(st))
        self._st = st
        return st

    def _ppo_step(self, st: dict):
        """One PPO minibatch on the rows ``st["idx"]`` (PPO:653-705): the loss, its gradients clipped per module, both
        optimizers' steps, the learning rate's adaptation; the diagnostics added to ``st["sums"]``."""
        cfg, buf = self.cfg, st["buf"]
        loss, pl, vl, ent, kl, cf = self._ppo_loss(st["obs"], st["ref"], st["pred"], st["act"], st["logp"], st["loc"], buf.std, st["adv"], st["ret"],
                                                   st["valid"], st["idx"])  # fmt: skip
        self.opt_policy.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        gn = [nn.utils.clip_grad_norm_(ps, cfg.max_grad_norm, foreach=True) for ps in self._clip_groups]
        self.opt_policy.step()
        self.opt_critic.step()
        self._adapt_lr(kl)
        st["sums"].add_(torch.cat((torch.stack((pl, ent, kl, cf, gn[0], gn[1])), vl)))

    def _adapt_step(self, st: dict):
        """One adaptation minibatch on the envs ``st["envs"]`` (PPO:515-559): the module's regression, clipped, and
        for the teacher the residual distillation, not clipped (PPO:542-544); the losses added to ``st["asums"]``."""
        cfg, buf = self.cfg, st["buf"]
        la = self._adapt_loss(buf.obs, st["feat"], st["keep"], st["envs"])
        self.opt_adapt.zero_grad(set_to_none=True)
        la.backward()
        nn.utils.clip_grad_norm_(list(self.adapt.parameters()), cfg.max_grad_norm, foreach=True)
        self.opt_adapt.step()
        ld = torch.zeros((), device=self.device)
        if cfg.teacher:
            ld = self._distill_loss(buf.obs, st["feat"], st["tloc"], st["envs"])
            self.opt_adapt_actor.zero_grad(set_to_none=True)
            ld.backward()
            self.opt_adapt_actor.step()
        st["asums"].add_(torch.stack((la.detach(), ld.detach())))

    def update(self, buf: Rollout, last_obs: torch.Tensor, section=None) -> dict[str, torch.Tensor]:
        """One iteration's learning from the rollout ``buf`` and the observation after its last step (N, D): HDMI's
        train_op (PPO:441-465). Returns the minibatches' mean diagnostics as device tensors. ``section(name)``: a
        context manager around each part (the trainer's profiling), none by default."""
        cfg, dev = self.cfg, self.device
        section = section or (lambda name: contextlib.nullcontext())
        st = self._static(buf)
        with torch.no_grad(), section("advantages"):
            adv, ret = self._advantages(buf.obs, last_obs, buf.reward, buf.term, buf.done)
            st["adv"].copy_(adv.view(-1))
            st["ret"].copy_(ret.view(-1, GROUPS))
            st["valid"].copy_(buf.step.view(-1) > cfg.min_step_count)
            st["sums"].zero_()
        nmb, M = cfg.num_minibatches, st["idx"].numel()
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(M * nmb, device=dev).view(nmb, M)  # PC:111-117
            for i in range(nmb):
                with section("ppo_minibatch"):
                    st["idx"].copy_(perm[i])
                    st["ppo_step"]()
        out = st["sums"] / (cfg.ppo_epochs * nmb)
        info = {"pi_loss": out[0], "entropy": out[1], "kl": out[2], "clipfrac": out[3], "actor_gradnorm": out[4], "critic_gradnorm": out[5],
                "v_loss": out[6:].mean()}  # fmt: skip
        for g in range(GROUPS):
            info[f"v_loss_{g}"] = out[6 + g]
        with section("adaptation"):
            info.update(self._train_adapt(st))
        info["lr"] = self.lr_policy.clone()
        info["action_std"] = self.policy_actor.std.detach().mean()
        return info

    def _train_adapt(self, st: dict) -> dict[str, torch.Tensor]:
        """HDMI's train_adapt (PPO:507-564): the adaptation module and, for the teacher, the residual distillation, on
        minibatches of whole envs (sequences of train_every = T steps, PC:99-109), then the module's EMA."""
        cfg, buf, dev = self.cfg, st["buf"], self.device
        with torch.no_grad():  # the teacher's features (and means) of every transition, after the PPO epochs
            feat = self.features(buf.obs)
            st["feat"].copy_(feat)
            if cfg.teacher:
                st["tloc"].copy_(self.teacher_mean(buf.obs, buf.ref, feat))
            st["keep"].copy_(buf.step != 0)  # ~is_init
            st["asums"].zero_()
        nmb, M = cfg.num_minibatches, st["envs"].numel()
        for _ in range(cfg.adapt_epochs):
            perm = torch.randperm(M * nmb, device=dev).view(nmb, M)
            for i in range(nmb):
                st["envs"].copy_(perm[i])
                st["adapt_step"]()
        with torch.no_grad():  # PC:286-288
            torch._foreach_lerp_(list(self.adapt_ema.parameters()), list(self.adapt.parameters()), cfg.ema_tau)
        n = cfg.adapt_epochs * nmb
        out = {"adapt_loss": st["asums"][0] / n}
        if cfg.teacher:
            out["distill_loss"] = st["asums"][1] / n
        return out

    # ---------------- checkpoints ----------------
    def state_dict(self) -> dict:
        """HDMI's state_dict (PPO:736-754): in the teacher phase the student's std is set to the teacher's and the EMA
        to the adaptation module first, side effects that HDMI's training also has at every checkpoint."""
        if self.cfg.teacher:
            with torch.no_grad():
                self.actor_adapt.std.copy_(self.actor.std)
                for d, s in zip(self.adapt_ema.parameters(), self.adapt.parameters()):
                    d.copy_(s)
        return {"phase": self.cfg.phase, "nets": self.nets.state_dict(), "adapt_ema": self.adapt_ema.state_dict(),
                "vecnorm": self.vecnorm.state_dict(), "lr_policy": float(self.lr_policy), "opt_policy": self.opt_policy.state_dict(),
                "opt_critic": self.opt_critic.state_dict(), "opt_adapt": self.opt_adapt.state_dict(),
                "opt_adapt_actor": None if self.opt_adapt_actor is None else self.opt_adapt_actor.state_dict()}  # fmt: skip

    def load_teacher(self, sd: dict):
        """Start the student phase from a teacher's checkpoint (PPO:756-776, HLP:132-160): every network, both actors'
        std set to ``load_noise_scale``, the policy's learning rate where the teacher's ended, the normalizer's
        statistics (frozen in this phase); fresh optimizer states, as HDMI's (it saves none)."""
        if self.cfg.teacher:
            raise ValueError("load_teacher starts the student phase ('finetune')")
        if sd["phase"] != "train":
            raise ValueError(f"expected a teacher's checkpoint (phase 'train'), got phase {sd['phase']!r}")
        self.nets.load_state_dict(sd["nets"])
        self.adapt_ema.load_state_dict(sd["adapt_ema"])
        self.vecnorm.load_state_dict(sd["vecnorm"])
        with torch.no_grad():
            self.actor.std.fill_(self.cfg.load_noise_scale)
            self.actor_adapt.std.fill_(self.cfg.load_noise_scale)
            self.lr_policy.fill_(sd["lr_policy"])

    def load_state_dict(self, sd: dict):
        """Resume a run of the same phase."""
        if sd["phase"] != self.cfg.phase:
            raise ValueError(f"checkpoint of phase {sd['phase']!r}, this run is {self.cfg.phase!r}")
        self.nets.load_state_dict(sd["nets"])
        self.adapt_ema.load_state_dict(sd["adapt_ema"])
        self.vecnorm.load_state_dict(sd["vecnorm"])
        with torch.no_grad():
            self.lr_policy.fill_(sd["lr_policy"])
        for name in ("opt_policy", "opt_critic", "opt_adapt", "opt_adapt_actor"):
            if getattr(self, name) is not None:
                getattr(self, name).load_state_dict(sd[name])
        for g in self.opt_policy.param_groups:
            g["lr"] = self.lr_policy


def residual_base(clip_joint_pos: torch.Tensor, q0: torch.Tensor, scale: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """HDMI's ``ref_joint_pos_`` (OBS:37-50): the reference's action joints at frame t (the command update's index 0,
    held at the clip's last frame, MOT:333-338) as an action, (q_ref - q0) / scale, (N, 23)."""
    return (clip_joint_pos[t.clamp(max=clip_joint_pos.shape[0] - 1)] - q0) / scale


__all__ = ["PPOConfig", "PPOROA", "Rollout", "VecNorm", "GROUPS", "OBS_GROUPS", "gae", "combine_advantages", "next_values", "residual_base"]
