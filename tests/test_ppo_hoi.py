"""HDMI's PPO-ROA (ndm_gym.algos.ppo_hoi) on CPU, against hand-computed values and HDMI's formulas: GAE per reward group
with terminations and truncations, the advantages' combination, the next values of done steps, the loss masks, the
clipped surrogate and value loss, the teacher's residual action mean, the normalizer (torchrl 0.7.0's VecNorm, with the
zero rows of resets), the initialization, the KL-adaptive learning rate, a full update in both phases and the
checkpoint's side effects. A short GPU run of the trainer is marked slow and needs CUDA and Isaac Lab."""

import importlib.util
import math
import os
import subprocess
import sys

import pytest
import torch

from ndm_gym.algos import ppo_hoi
from ndm_gym.algos.ppo_hoi import GROUPS, PPOConfig, PPOROA, Rollout, VecNorm, combine_advantages, gae, next_values, residual_base
from ndm_gym.tasks.hdmi import NUM_ACTIONS

DIMS = {"command": 7, "policy": 5, "object": 4, "priv": 9}
G, L = 0.99, 0.95


def agent(phase="train", seed=0):
    torch.manual_seed(seed)
    return PPOROA(PPOConfig(phase=phase), DIMS, "cpu")


def test_gae_by_hand():
    """Two envs, two groups, three steps; env 1 terminates at t = 1 and is truncated at t = 2."""
    r = [[(1.0, 0.5), (0.2, -0.1)], [(0.0, 1.0), (0.3, 0.4)], [(2.0, -1.0), (-0.5, 0.6)]]
    v = [[(0.5, 0.1), (1.0, 2.0)], [(0.4, 0.2), (1.5, -0.5)], [(0.3, 0.3), (0.7, 0.8)]]
    last = [(0.9, -0.2), (5.0, 5.0)]
    term = [[False, False], [False, True], [False, False]]
    done = [[False, False], [False, True], [False, True]]
    T = lambda x, **k: torch.tensor(x, dtype=torch.float64, **k)  # noqa: E731
    nv = next_values(T(v), T(last), torch.tensor(done))
    adv, ret = gae(T(r), torch.tensor(term), torch.tensor(done), T(v), nv, G, L)
    for g in range(2):
        # env 0: no episode end, bootstrapped from the next step's value and, at the end, from the last observation's
        d2 = r[2][0][g] + G * last[0][g] - v[2][0][g]
        d1 = r[1][0][g] + G * v[2][0][g] - v[1][0][g]
        d0 = r[0][0][g] + G * v[1][0][g] - v[0][0][g]
        e0 = [d0 + G * L * (d1 + G * L * d2), d1 + G * L * d2, d2]
        # env 1: truncated at t = 2 (bootstrapped from its own value), terminated at t = 1 (no bootstrap, trace cut)
        a2 = r[2][1][g] + G * v[2][1][g] - v[2][1][g]
        a1 = r[1][1][g] - v[1][1][g]
        a0 = r[0][1][g] + G * v[1][1][g] - v[0][1][g] + G * L * a1
        e1 = [a0, a1, a2]
        for t in range(3):
            assert adv[t, 0, g].item() == pytest.approx(e0[t], abs=1e-12)
            assert adv[t, 1, g].item() == pytest.approx(e1[t], abs=1e-12)
            assert ret[t, 0, g].item() == pytest.approx(e0[t] + v[t][0][g], abs=1e-12)
            assert ret[t, 1, g].item() == pytest.approx(e1[t] + v[t][1][g], abs=1e-12)


def test_next_values():
    """TRN:155-159: a done step's next value is its own value; otherwise the next step's, after the last step the
    value of the observation after it."""
    v = torch.arange(3 * 2 * GROUPS, dtype=torch.float64).view(3, 2, GROUPS)
    last = -torch.ones(2, GROUPS, dtype=torch.float64)
    done = torch.tensor([[False, True], [False, False], [True, False]])
    nv = next_values(v, last, done)
    assert torch.equal(nv[0, 0], v[1, 0]) and torch.equal(nv[0, 1], v[0, 1])
    assert torch.equal(nv[1, 0], v[2, 0]) and torch.equal(nv[1, 1], v[2, 1])
    assert torch.equal(nv[2, 0], v[2, 0]) and torch.equal(nv[2, 1], last[1])


def test_truncation_bootstraps_termination_does_not():
    """One step of one env: truncated, delta = r + gamma V(s) - V(s); terminated, delta = r - V(s)."""
    r, v = torch.tensor([[[1.0] * GROUPS]], dtype=torch.float64), torch.tensor([[[2.0] * GROUPS]], dtype=torch.float64)
    done = torch.tensor([[True]])
    nv = next_values(v, torch.full((1, GROUPS), 100.0, dtype=torch.float64), done)
    adv_trunc, _ = gae(r, torch.tensor([[False]]), done, v, nv, G, L)
    adv_term, _ = gae(r, torch.tensor([[True]]), done, v, nv, G, L)
    assert torch.allclose(adv_trunc, torch.full_like(r, 1.0 + G * 2.0 - 2.0))
    assert torch.allclose(adv_term, torch.full_like(r, 1.0 - 2.0))


def test_combine_advantages():
    """PPO:633-639: scaled by 1/5, summed over the groups, normalized with the batch's unbiased std plus 1e-8."""
    adv = torch.randn(4, 3, GROUPS, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    s = [sum(adv[t, n, g].item() * 0.2 for g in range(GROUPS)) for t in range(4) for n in range(3)]
    m = sum(s) / len(s)
    sd = math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1))
    out = combine_advantages(adv).flatten().tolist()
    assert out == pytest.approx([(x - m) / (sd + 1e-8) for x in s], abs=1e-12)


def _minibatch(a: PPOROA, B=64, seed=0):
    """A minibatch as a rollout leaves it: actions drawn around a mean near the residual base."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    obs, ref = r(B, a.obs_dim), r(B, NUM_ACTIONS)
    std_old = 1.0 + 0.1 * torch.rand(NUM_ACTIONS, generator=g)
    loc_old = ref + 0.1 * r(B, NUM_ACTIONS)
    act = loc_old + std_old * r(B, NUM_ACTIONS)
    logp_old = ppo_hoi.gaussian_log_prob(act, loc_old, std_old)
    step = torch.randint(0, 8, (B,), generator=g)
    step[:2] = 7  # some valid transitions in either phase
    pred = r(B, a.cfg.latent_dim) if not a.cfg.teacher else None
    return [obs, ref, pred, act, logp_old, loc_old, std_old, r(B), r(B, GROUPS), step]


def _loss(a, mb):
    *xs, step = mb
    return a._ppo_loss_impl(*xs, step > a.cfg.min_step_count, torch.arange(len(step)))


@pytest.mark.parametrize("phase", ["train", "finetune"])
def test_loss_masks(phase):
    """PPO:670-674: only transitions with step_count > 1 (teacher) or > 5 (student) enter the policy and value losses;
    changing the others' data changes neither the losses nor the gradients. The losses are HDMI's: the clipped
    surrogate's mean and each group's mean squared value error over the valid transitions, then their mean."""
    a = agent(phase)
    mb = _minibatch(a)
    *xs, step = mb
    valid = step > a.cfg.min_step_count
    assert 0 < valid.sum() < len(valid)
    loss, pl, vl, ent, kl, cf = _loss(a, mb)
    # HDMI's form with boolean indexing
    obs, ref, pred, act, logp_old, loc_old, std_old, adv, ret = xs
    loc = a.teacher_mean(obs, ref, a.features(obs)) if a.cfg.teacher else a.student_mean(obs, pred)
    std = a.policy_actor.std
    ratio = (ppo_hoi.gaussian_log_prob(act, loc, std) - logp_old).exp()
    surr = torch.minimum(adv * ratio, adv * ratio.clamp(0.8, 1.2))
    assert pl.item() == pytest.approx(-surr[valid].mean().item(), rel=1e-5)
    v = a._values_impl(obs)
    assert torch.allclose(vl, (ret - v).square()[valid].mean(0), rtol=1e-5)
    h = (0.5 + 0.5 * math.log(2 * math.pi) + std.log()).sum()
    assert loss.item() == pytest.approx((pl - 0.001 * h + vl.mean()).item(), rel=1e-5)
    params = [p for p in a.nets.parameters()]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    # scramble the invalid transitions
    g = torch.Generator().manual_seed(9)
    mb2 = [x.clone() if x is not None else None for x in mb]
    for i in (0, 1, 3, 4, 5, 7, 8) + ((2,) if not a.cfg.teacher else ()):
        mb2[i][~valid] = torch.randn(mb2[i][~valid].shape, generator=g)
    loss2, pl2, vl2, _, _, _ = _loss(a, mb2)
    assert torch.equal(pl, pl2) and torch.equal(vl, vl2)
    grads2 = torch.autograd.grad(loss2, params, allow_unused=True)
    for x, y in zip(grads, grads2):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.allclose(x, y, rtol=1e-6, atol=1e-9)


def test_residual_action_mean():
    """The teacher's mean is the residual base plus the MLP's output on cat[command, policy, encoder(cat[priv,
    object])] (PPO:223-236); the student's has no residual and ignores the base."""
    a = agent()
    with torch.no_grad():
        for p in a.nets.parameters():  # leave the near-zero initialization, so that the MLP's output matters
            p.add_(0.3 * torch.randn_like(p))
    obs, ref = torch.randn(6, a.obs_dim), torch.randn(6, NUM_ACTIONS)
    _, _, loc, pred = a.act(obs, ref, deterministic=True)
    assert pred is None
    c, p, o, q = (obs[:, a.sl[k]] for k in ppo_hoi.OBS_GROUPS)
    with torch.no_grad():
        mlp = a.actor(torch.cat((c, p, a.encoder(torch.cat((q, o), -1))), -1))
    assert torch.allclose(loc, ref + mlp, atol=1e-6)
    _, _, loc2, _ = a.act(obs, ref + 1.0, deterministic=True)
    assert torch.allclose(loc2 - loc, torch.ones_like(loc), atol=1e-5)
    s = agent("finetune")
    s.load_teacher(a.state_dict())
    _, _, l1, pred = s.act(obs, ref, deterministic=True)
    _, _, l2, _ = s.act(obs, ref + 1.0, deterministic=True)
    assert torch.equal(l1, l2)
    with torch.no_grad():
        assert torch.allclose(l1, s.actor_adapt(torch.cat((c, p, s.adapt_ema(torch.cat((p, c, o), -1))), -1)))
        assert torch.equal(pred, s.adapt_ema(torch.cat((p, c, o), -1)))


def test_rollout_sample_log_prob():
    a = agent()
    obs, ref = torch.randn(4000, a.obs_dim), torch.zeros(4000, NUM_ACTIONS)
    act, logp, loc, _ = a.act(obs, ref)
    std = a.actor.std
    assert torch.allclose(logp, torch.distributions.Normal(loc, std).log_prob(act).sum(-1), atol=1e-4)
    z = (act - loc) / std  # unit Gaussian noise, std initialized to 1 (PC:153)
    assert torch.allclose(std, torch.ones_like(std)) and abs(z.mean().item()) < 0.01 and abs(z.std().item() - 1) < 0.01


def _torchrl_vecnorm(batches, decay=0.9999, eps=1e-4):
    """torchrl 0.7.0's VecNorm._update, written out: the decayed sums and count after each batch and the batch
    normalized with them."""
    s = q = c = 0.0
    outs = []
    for x in batches:
        s = s * decay + x.sum(0)
        q = q * decay + x.square().sum(0)
        c = c * decay + x.shape[0]
        mean = s / c
        std = (q / c - mean**2).clamp_min(eps).sqrt()
        outs.append((x - mean) / std.clamp_min(eps))
    return outs, (s, q, c)


def test_vecnorm():
    """The statistics after the fake tensordict's zeros, steps of data and the zero rows of resets (VecNorm's
    docstring) equal torchrl's updates with explicit zero batches; normalize leaves them unchanged."""
    g = torch.Generator().manual_seed(3)
    N, D = 16, 5
    data = [2.0 + 3.0 * torch.randn(N, D, generator=g, dtype=torch.float64) for _ in range(6)]
    zeros = torch.zeros(N, D, dtype=torch.float64)
    seq = [zeros, data[0], data[1], zeros, data[2], zeros, data[3], data[4], zeros, data[5]]
    ref, (s, q, c) = _torchrl_vecnorm(seq)
    vn = VecNorm(D).double()
    for x, r in zip(seq, ref):
        if x is zeros:
            vn.absorb_zeros(N)
            out = vn.normalize(zeros[:1])
            assert torch.allclose(out, r[:1], rtol=1e-12, atol=1e-12)
        else:
            assert torch.allclose(vn.update(x), r, rtol=1e-12, atol=1e-12)
    assert torch.allclose(vn.sum, s) and torch.allclose(vn.ssq, q) and vn.count.item() == pytest.approx(c, rel=1e-12)
    before = [b.clone() for b in vn.buffers()]
    vn.normalize(data[0])
    assert all(torch.equal(b, a) for b, a in zip(before, vn.buffers()))
    # the zero rows conditional on a device flag (no host synchronization): unchanged when false, as above when true
    vn.absorb_zeros(N, torch.tensor(False))
    assert all(torch.equal(b, a) for b, a in zip(before, vn.buffers()))
    twin = VecNorm(D).double()
    twin.load_state_dict(vn.state_dict())
    vn.absorb_zeros(N, torch.tensor(True))
    twin.absorb_zeros(N)
    assert all(torch.equal(b, a) for b, a in zip(twin.buffers(), vn.buffers()))


def test_observe_and_reset_observation():
    """The teacher's normalizer adds each step's observation and a reset's zero rows (HLP:152-154); the student's is
    frozen (HLP:155-157); the reset observation is the normalized zero."""
    a = agent()
    a.vecnorm.absorb_zeros(8)
    obs = {k: torch.randn(8, d) for k, d in DIMS.items()}
    c0 = a.vecnorm.count.clone()
    x = a.observe(obs)
    assert a.vecnorm.count.item() == pytest.approx(c0.item() * 0.9999 + 8)
    assert torch.allclose(x, a.vecnorm.normalize(a.cat_obs(obs)))
    z = a.reset_observation(8)
    assert a.vecnorm.count.item() == pytest.approx((c0.item() * 0.9999 + 8) * 0.9999 + 8)
    assert torch.allclose(z, a.vecnorm.normalize(torch.zeros(1, a.obs_dim)))
    s = agent("finetune")
    s.load_teacher(a.state_dict())
    before = [b.clone() for b in s.vecnorm.buffers()]
    y = s.observe(obs)
    s.reset_observation(8)
    assert all(torch.equal(b, c) for b, c in zip(before, s.vecnorm.buffers()))
    assert torch.allclose(y, a.vecnorm.normalize(a.cat_obs(obs)))


def test_initialization():
    """PPO:336-345: every linear layer orthogonal with gain 0.01 and zero bias; stds 1 (PC:153); the EMA a copy."""
    a = agent()
    n = 0
    for m in a.nets.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().double()
            WW = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            assert torch.allclose(WW, 1e-4 * torch.eye(len(WW), dtype=torch.float64), atol=1e-9)
            assert torch.equal(m.bias, torch.zeros_like(m.bias))
            n += 1
    assert n == 2 + 4 + 3 + 4 + 4  # encoder, actor, adaptation module, student actor, critic
    assert torch.equal(a.actor.std, torch.ones(NUM_ACTIONS)) and torch.equal(a.actor_adapt.std, torch.ones(NUM_ACTIONS))
    assert all(torch.equal(x, y) for x, y in zip(a.adapt.parameters(), a.adapt_ema.parameters()))
    assert a.critic[-1].out_features == GROUPS == 5


@pytest.mark.parametrize("kl, factor", [(0.03, 1 / 1.5), (0.0201, 1 / 1.5), (0.02, 1.0), (0.01, 1.0), (0.005, 1.0), (0.004, 1.5), (0.0, 1.0)])
def test_adaptive_lr(kl, factor):
    """PPO:483-491: KL > 2 x 0.01 divides the policy's learning rate by 1.5, 0 < KL < 0.01 / 2 multiplies it by 1.5."""
    a = agent()
    a._adapt_lr(torch.tensor(kl))
    assert a.lr_policy.item() == pytest.approx(3e-4 * factor, rel=1e-6)
    assert all(g["lr"] is a.lr_policy for g in a.opt_policy.param_groups)
    a.lr_policy.fill_(1.2e-5)
    a._adapt_lr(torch.tensor(0.5))
    assert a.lr_policy.item() == pytest.approx(1e-5)
    a.lr_policy.fill_(9e-3)
    a._adapt_lr(torch.tensor(1e-4))
    assert a.lr_policy.item() == pytest.approx(1e-2)


def _rollout(a: PPOROA, T=4, N=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    buf = Rollout(T, N, a.obs_dim, a.cfg.latent_dim, not a.cfg.teacher, "cpu")
    buf.obs.copy_(torch.randn(T, N, a.obs_dim, generator=g))
    buf.ref.copy_(torch.randn(T, N, NUM_ACTIONS, generator=g))
    with torch.no_grad():
        for t in range(T):
            act, logp, loc, pred = a.act(buf.obs[t], buf.ref[t])
            buf.act[t], buf.logp[t], buf.loc[t] = act, logp, loc
            if pred is not None:
                buf.pred[t] = pred
    buf.reward[..., :4] = torch.rand(T, N, 4, generator=g)
    buf.done.copy_(torch.rand(T, N, generator=g) < 0.2)
    buf.term.copy_(buf.done & (torch.rand(T, N, generator=g) < 0.5))
    buf.step.copy_(torch.randint(0, 30, (T, N), generator=g, dtype=torch.int32))  # mostly valid in either phase: no empty minibatch
    buf.std.copy_(a.policy_actor.std.detach())
    return buf, torch.randn(N, a.obs_dim, generator=g)


@pytest.mark.parametrize("phase", ["train", "finetune"])
def test_update(phase):
    """A full iteration's learning: finite diagnostics, the policy's and critic's parameters move, the adaptation EMA
    moves 0.04 of the way to the module; the student keeps the teacher's encoder and the normalizer."""
    a = agent()
    if phase == "finetune":
        t = a
        t.lr_policy.fill_(7e-4)
        a = agent("finetune", seed=1)
        a.load_teacher(t.state_dict())
        assert torch.equal(a.actor.std, torch.full((NUM_ACTIONS,), 0.5)) and torch.equal(a.actor_adapt.std, torch.full((NUM_ACTIONS,), 0.5))
        assert a.lr_policy.item() == pytest.approx(7e-4)
    buf, last = _rollout(a)
    snap = {k: [p.detach().clone() for p in m.parameters()] for k, m in a.nets.items()}
    ema0 = [p.clone() for p in a.adapt_ema.parameters()]
    info = a.update(buf, last)
    assert all(torch.isfinite(v).all() for v in info.values())
    assert set(info) >= {"pi_loss", "v_loss", "entropy", "kl", "clipfrac", "lr", "action_std", "adapt_loss"}
    assert ("distill_loss" in info) == (phase == "train")
    moved = {k: any(not torch.equal(p, q) for p, q in zip(m.parameters(), snap[k])) for k, m in a.nets.items()}
    expect = {"encoder": phase == "train", "actor": phase == "train", "adapt": True, "actor_adapt": True, "critic": True}
    assert moved == expect
    for e0, e1, p in zip(ema0, a.adapt_ema.parameters(), a.adapt.parameters()):
        assert torch.allclose(e1, e0 + 0.04 * (p - e0), atol=1e-7)


def test_checkpoint_side_effects():
    """PPO:737-746: a teacher's checkpoint first sets the student's std to the teacher's and the EMA to the module."""
    a = agent()
    with torch.no_grad():
        a.actor.std.fill_(0.7)
        for p in a.adapt.parameters():
            p.add_(1.0)
    sd = a.state_dict()
    assert torch.equal(a.actor_adapt.std, a.actor.std)
    assert all(torch.equal(x, y) for x, y in zip(a.adapt.parameters(), a.adapt_ema.parameters()))
    b = agent(seed=5)
    b.load_state_dict(sd)
    assert all(torch.equal(x, y) for x, y in zip(a.nets.parameters(), b.nets.parameters()))
    assert all(g["lr"] is b.lr_policy for g in b.opt_policy.param_groups)
    with pytest.raises(ValueError):
        agent().load_teacher(sd)
    with pytest.raises(ValueError):
        agent("finetune").load_state_dict(sd)


def test_residual_base():
    """OBS:37-50: (q_ref - q0) / scale at the frame counter, held at the clip's last frame."""
    clip = torch.arange(5 * NUM_ACTIONS, dtype=torch.float32).view(5, NUM_ACTIONS)
    q0, scale = torch.full((NUM_ACTIONS,), 1.0), torch.full((NUM_ACTIONS,), 0.5)
    out = residual_base(clip, q0, scale, torch.tensor([0, 3, 4, 9]))
    assert torch.equal(out, (clip[[0, 3, 4, 4]] - 1.0) / 0.5)


def test_minibatch_rows():
    """The loss on rows idx of the flat data equals the loss on those rows gathered beforehand."""
    a = agent()
    mb = _minibatch(a, B=64)
    idx = torch.randperm(64, generator=torch.Generator().manual_seed(2))[:24]
    *xs, step = mb
    sub = [x if x is None or x.dim() == 1 and x.shape[0] == NUM_ACTIONS else x[idx] for x in xs]
    full = a._ppo_loss_impl(*xs, step > 1, idx)
    part = a._ppo_loss_impl(*sub, step[idx] > 1, torch.arange(24))
    for x, y in zip(full, part):
        assert torch.allclose(x, y, rtol=1e-6, atol=1e-7)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("phase", ["train", "finetune"])
def test_update_graphs_match_eager(phase):
    """Two updates with every minibatch step replayed as a CUDA graph leave the networks, the optimizers' learning
    rate and the diagnostics where the eager steps leave them (the same kernels on the same tensors)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    outs = []
    for graphs in (False, True):
        torch.manual_seed(0)
        t = PPOROA(PPOConfig(), DIMS, "cuda")
        a = t
        if phase == "finetune":
            a = PPOROA(PPOConfig(phase="finetune"), DIMS, "cuda", graphs=graphs)
            a.load_teacher(t.state_dict())
        elif graphs:
            a = PPOROA(PPOConfig(), DIMS, "cuda", graphs=True)
            a.nets.load_state_dict(t.nets.state_dict())
            a.adapt_ema.load_state_dict(t.adapt_ema.state_dict())
        torch.manual_seed(1)
        buf, last = _rollout_cuda(a, T=8, N=64)
        infos = [a.update(buf, last) for _ in range(2)]
        outs.append((a, infos))
    (e, ie), (g, ig) = outs
    for x, y in zip(e.nets.parameters(), g.nets.parameters()):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6)
    assert torch.allclose(e.lr_policy, g.lr_policy, rtol=1e-6)
    for k in ie[-1]:
        assert torch.allclose(ie[-1][k], ig[-1][k], rtol=1e-4, atol=1e-6), k


def _rollout_cuda(a: PPOROA, T: int, N: int):
    """_rollout's synthetic iteration on the GPU."""
    buf, last = _rollout(a.__class__(a.cfg, DIMS, "cpu"), T=T, N=N)
    g = Rollout(T, N, a.obs_dim, a.cfg.latent_dim, not a.cfg.teacher, "cuda")
    for k in ("obs", "ref", "pred", "act", "loc", "logp", "reward", "term", "done", "step"):
        if getattr(buf, k) is not None:
            getattr(g, k).copy_(getattr(buf, k))
    with torch.no_grad():
        for t in range(T):  # the actions' statistics under this agent
            act, logp, loc, pred = a.act(g.obs[t], g.ref[t])
            g.act[t], g.logp[t], g.loc[t] = act, logp, loc
            if pred is not None:
                g.pred[t] = pred
    g.std.copy_(a.policy_actor.std.detach())
    return g, last.cuda()


HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
def test_trainer_gpu_smoke(tmp_path):
    """Three iterations of the teacher on move_suitcase with 64 envs, then one of the student from its checkpoint."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "ndm_gym.train_ppo_hoi", "--task", "move_suitcase", "--num_envs", "64", "--train_every", "8", "--log_every", "1"]
    run = lambda *a: subprocess.run(cmd + list(a), cwd=repo, capture_output=True, text=True, timeout=1800)  # noqa: E731
    t = run("--iterations", "3", "--ckpt_every", "3", "--log_dir", str(tmp_path / "teacher"))
    assert t.returncode == 0, t.stdout[-3000:] + t.stderr[-3000:]
    rows = (tmp_path / "teacher" / "train_log.csv").read_text().strip().splitlines()
    assert len(rows) == 4 and all("nan" not in r.split(",")[3] for r in rows[1:])
    ck = tmp_path / "teacher" / "ckpt_000003.pt"
    s = run("--phase", "finetune", "--teacher", str(ck), "--iterations", "1", "--log_dir", str(tmp_path / "student"))
    assert s.returncode == 0, s.stdout[-3000:] + s.stderr[-3000:]
    r = run("--resume", str(ck), "--iterations", "4", "--ckpt_every", "4", "--log_dir", str(tmp_path / "teacher"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = (tmp_path / "teacher" / "train_log.csv").read_text().strip().splitlines()
    assert len(rows) == 5 and rows[-1].startswith("4,") and (tmp_path / "teacher" / "ckpt_000004.pt").exists()
