"""The muscle character's tracking task, losses and guard on synthetic states (CPU, without MuscleVAE's assets):
SuperMimic's reward against its formula, its invariance to MuscleVAE's reference switches (any heading and place of the
reference), the kernel widths' scale; SuperTrack's terms; MuscleVAE's termination rule, hard against a step-by-step
count written here from its documented semantics, and soft (straight-through) against the hard one; the guard's flags
and masked normalizer updates."""

import math

import torch

from ndm_gym.algos.muscle_guard import SkippedUpdates, check, flagged, masked_update
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots import musclevae_char as mc
from ndm_gym.tasks import muscle_tracking as mt

DT, NB = mt.DT, mt.NB


def synthetic_reference(F: int = 40, seed: int = 0) -> mt.MuscleReference:
    """A reference of F frames: bodies moving smoothly, velocities as backward differences."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(F, dtype=torch.float32)[:, None, None] * DT
    base = torch.randn(1, NB, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, 1.0])
    p = base + torch.cat((0.8 * t, 0.1 * torch.sin(3 * t), 0.05 * torch.cos(5 * t)), -1)
    axis = torch.randn(1, NB, 3, generator=g)
    R = so3_exp(axis * (0.3 + t))
    v = torch.cat(((p[1:2] - p[:1]) / DT, (p[1:] - p[:-1]) / DT))
    w = axis.expand(F, NB, 3).clone()
    return mt.MuscleReference(p, R, v, w, p[:, mc.HEAD, 2].clone(), torch.zeros(F, mc.NUM_MUSCLES), torch.full((F,), F - 1, dtype=torch.long))


def turn(S, yaw: float, shift):
    c, s = math.cos(yaw), math.sin(yaw)
    Q = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p, R, v, w = S
    return p @ Q.T + torch.tensor([shift[0], shift[1], 0.0]), Q @ R, v @ Q.T, w @ Q.T


def test_reward_formula_and_switch_invariance():
    """At the reference the reward is its maximum, 4 dt, wherever and however turned the reference is (MuscleVAE's
    switches); a vertical offset d of every body costs exp(-d^2 / std^2) on the position term, with the widths scaled."""
    ref = synthetic_reference()
    f = torch.tensor([3, 10, 25])
    S = tuple(x[f] for x in ref.state)
    cfg = mt.MuscleRewardCfg()
    torch.testing.assert_close(mt.reward(S, ref, f, cfg), torch.full((3,), mt.MAX_STEP_REWARD), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mt.reward(turn(S, 2.0, (4.0, -7.0)), ref, f, cfg), torch.full((3,), mt.MAX_STEP_REWARD), rtol=1e-5, atol=1e-6)
    d = 0.1
    up = (S[0] + torch.tensor([0.0, 0.0, d]), S[1], S[2], S[3])
    for k in (1.0, 0.7):
        c = cfg.scaled(k)
        r, terms = mt.reward(up, ref, f, c, terms=True)
        torch.testing.assert_close(terms["e_body_pos"], torch.full((3,), d * d), rtol=1e-4, atol=1e-6)
        want = (3.0 + math.exp(-d * d / (k * cfg.body_pos_std) ** 2)) * DT
        torch.testing.assert_close(r, torch.full((3,), want), rtol=1e-4, atol=1e-6)
    # the G1 task's action-rate penalty in its units: a change of 0.05 on every muscle target is 0.1 on each of 29 setpoints
    rate = mt.action_rate(torch.full((3, mc.NUM_MUSCLES), 0.05), torch.zeros(3, mc.NUM_MUSCLES))
    torch.testing.assert_close(rate, torch.full((3,), 29 * 0.1**2))
    torch.testing.assert_close(mt.reward(S, ref, f, cfg, rate), (4.0 + cfg.w_action_rate * rate) * DT, rtol=1e-5, atol=1e-6)
    S_far = (S[0] + 0.3, S[1], S[2] + 1.0, S[3])
    assert (mt.reward(S_far, ref, f, cfg) < mt.reward(S, ref, f, cfg) - 1e-3).all()
    assert torch.allclose(mt.root_frame_mpjpe(turn(S, 1.0, (2.0, 2.0)), ref, f), torch.zeros(3), atol=1e-5)


def hard_rule(head_err: list[float], obs_max: list[float], count0: int, ep0: int) -> tuple[list[bool], list[int]]:
    """MuscleVAE's termination, step by step (ENV:193-218 as documented in ndm_gym.sim.rigid.muscle_sim): the off count
    +1 when the head is more than 0.5 m off, else -1 and at least 0; terminated when it reaches 20 with 26 steps behind
    the step, or when the observation exceeds 50."""
    count, out, counts = count0, [], []
    for k, (e, o) in enumerate(zip(head_err, obs_max)):
        count = count + 1 if e > 0.5 else max(count - 1, 0)
        out.append(o > 50.0 or (ep0 + k >= 26 and count >= 20))
        counts.append(count)
    return out, counts


def state_with(head_err: torch.Tensor, ref_head: torch.Tensor, obs_scale: torch.Tensor):
    """States (B, 23, ...) whose head is ``head_err`` above the reference's and whose largest observation entry is about
    ``obs_scale`` (the bodies' velocities)."""
    B = head_err.shape[0]
    p = torch.zeros(B, NB, 3)
    p[..., 2] = 1.0
    p[:, mc.HEAD, 2] = ref_head + head_err
    R = torch.eye(3).expand(B, NB, 3, 3).clone()
    v = torch.zeros(B, NB, 3)
    v[:, 5, 0] = obs_scale
    return p, R, v, torch.zeros(B, NB, 3)


def test_soft_termination_against_the_rule():
    """Random sequences of head-height errors around 0.5 m and observations around 50: the straight-through count is
    MuscleVAE's exactly, the termination probability within 5e-5 of the rule at every step, and the head error has a
    gradient near the threshold."""
    g = torch.Generator().manual_seed(0)
    B, T = 64, 60
    head = 0.5 + 0.2 * torch.randn(T, B, generator=g)
    head[:, :16] = 0.8 + 0.05 * torch.rand(T, 16, generator=g)  # long runs off track
    obs = 40.0 + 6.0 * torch.randn(T, B, generator=g)
    count0 = torch.randint(0, 25, (B,), generator=g)
    ep0 = torch.randint(0, 40, (B,), generator=g)
    ref_head = torch.full((B,), 1.5)
    count, ep = count0.float(), ep0.clone()
    hard_count, ep_h = count0.clone(), ep0.clone()
    fired_any = 0
    for k in range(T):
        S = state_with(head[k], ref_head, obs[k])
        d, count = mt.soft_termination(S, ref_head, count, ep)
        term, hard_count = mt.termination(S, ref_head, hard_count, ep_h)
        assert torch.equal(count, hard_count.float())
        assert (d - term.float()).abs().max() < 5e-5
        fired_any += int(term.sum())
        ep, ep_h = ep + 1, ep_h + 1
    assert fired_any > 0
    for b in range(4):  # the hard rule against the step-by-step count written above
        S_obs = [float(mt.observation_max(state_with(head[k, b : b + 1], ref_head[:1], obs[k, b : b + 1]))) for k in range(T)]
        want, counts = hard_rule((head[:, b]).tolist(), S_obs, int(count0[b]), int(ep0[b]))
        c, e = count0[b : b + 1].clone(), ep0[b : b + 1].clone()
        for k in range(T):
            term, c = mt.termination(state_with(head[k, b : b + 1], ref_head[:1], obs[k, b : b + 1]), ref_head[:1], c, e)
            assert bool(term) == want[k] and int(c) == counts[k]
            e = e + 1
    # the gradient reaches the head's height near the threshold
    err = torch.tensor([0.48, 0.52], requires_grad=True)
    S = state_with(err, torch.zeros(2), torch.zeros(2))
    d, c = mt.soft_termination(S, torch.zeros(2), torch.tensor([19.0, 19.0]), torch.tensor([30, 30]))
    assert d.tolist()[0] < 1e-4 and d.tolist()[1] > 1 - 1e-4 and c.tolist() == [18.0, 20.0]
    c.sum().backward()
    assert (err.grad.abs() > 0).all()


def test_guard():
    """Flags: non-finite or beyond the bounds; the masked normalizer update is the normalizer's own on the kept rows, and
    a NaN row left out changes nothing; a non-finite parameter stops training."""
    s1, fat = torch.zeros(4, NB, 13), torch.full((4, 15), 0.3)
    s1[1, 3, 8] = float("nan")
    s1[2, 0, 0] = 2e3
    fat[3, 2] = float("inf")
    assert flagged(s1, fat).tolist() == [False, True, True, True]
    x = torch.randn(10, 6)
    keep = torch.tensor([True] * 7 + [False] * 3)
    x[8] = float("nan")
    a, b = RunningNorm(6), RunningNorm(6)
    a.update(torch.randn(5, 6))
    b.load_state_dict(a.state_dict())
    masked_update(a, x, keep)
    b.update(x[keep])
    for k in ("mean", "var", "count"):
        torch.testing.assert_close(getattr(a, k), getattr(b, k))
    before = {k: v.clone() for k, v in a.state_dict().items()}
    masked_update(a, x, torch.zeros(10, dtype=torch.bool))
    assert all(torch.equal(before[k], v) for k, v in a.state_dict().items())
    skips = SkippedUpdates()
    for s in (True, True, False, True):
        skips(s)
    assert skips.take() == (3, 2)
    net = torch.nn.Linear(2, 2)
    check(1, {"net": net})
    with torch.no_grad():
        net.weight[0, 0] = float("nan")
    try:
        check(2, {"net": net})
        raise AssertionError("check did not stop on a non-finite parameter")
    except RuntimeError as e:
        assert "net.weight" in str(e)
