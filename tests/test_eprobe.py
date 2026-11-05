"""BlinkMimic's evaluation-guided resets (ndm_gym.algos.eprobe, ``--eprobe_*``): the probe worlds' tiles and sweep frames, the
failure counts and their window against BlinkMimic's own code (train.py @ d264658, restated below), the focus's share of
the reset distribution, the deterministic probe policy, the options' checks and the trainer's timing of the sweeps. The
probe simulator itself runs in the GPU smoke runs (docs/blinkmimic.md); at the defaults nothing changes
(test_algo_equivalence, test_supertrack_golden)."""

import argparse
import math
import types

import pytest
import torch

from conftest import DEV


def _blinkmimic_tiles(motion, tile):
    """BlinkMimic's EP_TILE (train.py)."""
    return torch.cat([torch.arange(int(motion.clip_start[c]), int(motion.clip_end[c]) - 1, tile) for c in range(len(motion.names))])


def _blinkmimic_focus(motion, counts, fail_f, win, decay):
    """BlinkMimic's eprobe_finish on EP_FAIL = ``counts`` (train.py, its EP_NEXT included); returns the counts and the focus."""
    nxt = torch.arange(motion.num_bins) + 1
    ep_next = torch.where((nxt < motion.num_bins) & (motion.bin_clip[nxt.clamp(max=motion.num_bins - 1)] == motion.bin_clip), nxt, nxt - 1)
    counts = counts * decay
    fb = motion.frame_bin[fail_f[fail_f >= 0]]
    counts = counts.index_add(0, fb, torch.ones_like(fb, dtype=counts.dtype))
    if not counts.sum() > 0:
        return counts, None
    w, q = counts.clone(), counts.clone()
    for _ in range(win):
        w = w[ep_next] * (ep_next != torch.arange(motion.num_bins)).float()
        q += w
    return counts, q / q.sum()


def test_tiles_and_sweep_frames(motion):
    from ndm_gym.algos.eprobe import probe_tiles, sweep_frames

    tiles = probe_tiles(motion, 300)
    assert torch.equal(tiles, _blinkmimic_tiles(motion, 300))
    cid = motion.clip_id[tiles]
    assert ((tiles >= motion.clip_start[cid]) & (tiles < motion.clip_end[cid] - 1)).all()
    for lead in (0, 200, 450, 10**6):
        f0, stop, cap = sweep_frames(motion, tiles, lead, 300)
        assert torch.equal(f0, torch.maximum(tiles - lead, motion.clip_start[cid]))  # BlinkMimic's f0, stop and fcap
        assert torch.equal(stop, torch.minimum(tiles + 300, motion.clip_end[cid]))
        assert torch.equal(cap, torch.minimum(stop, motion.clip_end[cid] - 1))
        assert (motion.clip_id[f0] == cid).all() and (cap + 1 <= motion.clip_end[cid]).all() and (f0 < stop).all()


def test_failure_counts_and_window_are_blinkmimics(motion):
    from ndm_gym.algos.eprobe import FailureFocus

    g = torch.Generator().manual_seed(0)
    focus = FailureFocus(motion, win=10, decay=0.7)
    ref = torch.zeros(motion.num_bins)
    assert focus.update(torch.full((50,), -1)) is None  # no failure yet: no focus
    ref, _ = _blinkmimic_focus(motion, ref, torch.full((50,), -1), 10, 0.7)
    for _ in range(6):
        fail_f = torch.randint(-1, motion.num_frames, (400,), generator=g)
        fail_f[torch.rand(400, generator=g) < 0.5] = -1
        q = focus.update(fail_f)
        ref, q_ref = _blinkmimic_focus(motion, ref, fail_f, 10, 0.7)
        torch.testing.assert_close(focus.counts, ref, rtol=0, atol=0)
        torch.testing.assert_close(q, q_ref, rtol=1e-6, atol=1e-9)
        assert abs(float(q.sum()) - 1) < 1e-5 and (q >= 0).all()


def test_window_reaches_back_within_the_clip(motion):
    """One failure in bin b weighs bins b - win .. b of its clip equally and nothing in other clips."""
    from ndm_gym.algos.eprobe import FailureFocus

    c = int(torch.argmax(motion.clip_len))
    bins = (motion.bin_clip == c).nonzero().flatten()
    b = int(bins[len(bins) // 2])
    focus = FailureFocus(motion, win=3, decay=0.7)
    q = focus.update(torch.tensor([int(motion.bin_start[b]) + 1]))
    want = torch.zeros(motion.num_bins)
    want[b - 3 : b + 1] = 0.25
    torch.testing.assert_close(q, want)
    first = int(bins[0])  # at a clip's start the window stops at its first bin
    q = FailureFocus(motion, win=3, decay=0.7).update(torch.tensor([int(motion.bin_start[first])]))
    assert int((q > 0).sum()) == 1 and float(q[first]) == 1.0


def test_focus_share_of_the_reset_distribution():
    from ndm_gym.sim.rigid.reset_stats import mix_focus

    g = torch.Generator().manual_seed(1)
    probs = torch.rand(60, generator=g)
    probs /= probs.sum()
    q = torch.zeros(60)
    q[10:20] = 0.1
    mixed = mix_focus(probs, q, 0.15)
    torch.testing.assert_close(mixed, 0.85 * probs + 0.15 * q)
    assert abs(float(mixed.sum()) - 1) < 1e-6
    assert float(mixed[10:20].sum()) == pytest.approx(0.85 * float(probs[10:20].sum()) + 0.15)


def test_static_focus_at_share_zero_leaves_the_distribution():
    """use_static_focus's fixed tensors before the first sweep (share 0): the reset distribution bit for bit."""
    from ndm_gym.sim.rigid.reset_stats import mix_focus

    g = torch.Generator().manual_seed(2)
    probs = torch.rand(80, generator=g)
    probs /= probs.sum()
    assert torch.equal(mix_focus(probs, torch.full((80,), 1 / 80), torch.zeros(())), probs)


def test_probe_policy_is_deterministic(motion, mdp):
    """The deployed policy in a probe world: no action noise; a residual policy's output about the next reference pose."""
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.eprobe import probe_action
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.tasks.tracking import OBS_DIM

    torch.manual_seed(0)
    K, H, n = 4, 4, 6
    pi = TrackingPolicy(OBS_DIM, 73 * K, hist=H).to(DEV).eval()
    f = torch.randint(0, motion.num_frames - 10, (n,))
    hist, goal = torch.randn(n, H, OBS_DIM), motion.goal(f, K)
    alpha = torch.ones(29)
    ref_u = reference_setpoints(motion, mdp.default_q, alpha)
    with torch.no_grad():
        a1 = probe_action(pi, hist, goal, False, ref_u, f)
        a2 = probe_action(pi, hist, goal, False, ref_u, f)
        r = probe_action(pi, hist, goal, True, ref_u, f)
        assert torch.equal(a1, a2) and torch.equal(a1, pi(hist, goal))
        torch.testing.assert_close(r, a1 + ref_u[f + 1])


def _args(**kw):
    a = dict(eprobe_share=0.15, eprobe_tile=300, eprobe_lead_min=200, eprobe_lead_max=450, eprobe_every=125, eprobe_start=1000, eprobe_win=10,
             eprobe_decay=0.7, eprobe_graph=True, eprobe_async=20, pi_ema=0.995, paired=False)  # fmt: skip
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.mark.parametrize(
    "kw",
    [dict(pi_ema=0.0), dict(eprobe_graph=False), dict(eprobe_async=5), dict(paired=True), dict(eprobe_share=1.0), dict(eprobe_tile=0),
     dict(eprobe_lead_min=500), dict(eprobe_share=0.0), dict(eprobe_decay=1.5)],
)  # fmt: skip
def test_unsupported_probe_settings_are_refused(kw):
    from ndm_gym.algos.eprobe import check_eprobe_args

    with pytest.raises(ValueError):
        check_eprobe_args(_args(**kw))


def test_probe_settings_accepted():
    from ndm_gym.algos.eprobe import check_eprobe_args

    check_eprobe_args(_args())  # BlinkMimic's PRESET[main]
    check_eprobe_args(_args(eprobe_graph=False, eprobe_async=0))  # synchronous, eager sweeps
    check_eprobe_args(_args(eprobe_share=0.0, eprobe_graph=False, eprobe_async=0))  # off


class _Env:
    focus_q, focus_share = None, 0.0

    def set_focus(self, q, share):
        self.focus_q, self.focus_share = q, share


def _probe(motion, args, tmp_path):
    """An EvalProbe without its worker: requests are answered at once with failures in the given clip's bins."""
    from ndm_gym.algos.eprobe import EvalProbe, FailureFocus, probe_tiles

    p = EvalProbe.__new__(EvalProbe)
    p.args, p.motion, p.env = args, motion, _Env()
    p.tiles = probe_tiles(motion, args.eprobe_tile)
    p.focus = FailureFocus(motion, args.eprobe_win, args.eprobe_decay)
    p.dir, p.k, p.pending = str(tmp_path), 0, None
    p.sent, p.applied = [], []

    def wait(self, path):
        k = int(path[-8:-3])
        fail_f = torch.full((len(self.tiles),), -1)
        fail_f[k % len(fail_f)] = int(self.tiles[k % len(fail_f)])
        return {"fail_f": fail_f, "sweep_s": 0.0}

    p._wait = types.MethodType(wait, p)
    real_apply = EvalProbe._apply

    def apply(self, it):
        self.applied.append((self.pending[1], it))
        open(self._path("res", self.pending[0]), "w").close()  # _apply removes the result file
        real_apply(self, it)

    p._apply = types.MethodType(apply, p)
    return p


def test_sweep_timing_is_blinkmimics(motion, tmp_path, monkeypatch):
    """--eprobe_async 20: a sweep started at iteration it applies its failures at it + ceil(steps / 20), steps = lead + tile
    (three fewer for the first sweep: BlinkMimic's graph warm-up), before that iteration's collection; without
    --eprobe_async, at once. A sweep still pending when the next one is due is an error."""
    leads = iter([200, 450, 333, 210])
    monkeypatch.setattr(torch, "randint", lambda lo, hi, size, **kw: torch.tensor([next(leads)]))
    pi = torch.nn.Linear(2, 2)
    p = _probe(motion, _args(eprobe_start=50, eprobe_every=50), tmp_path)  # sweeps where it >= start and it % every == 0
    for it in range(0, 200):
        p.tick(it, pi)
    assert p.applied == [(50, 50 + math.ceil((200 + 300 - 3) / 20)), (100, 100 + math.ceil(750 / 20)), (150, 150 + math.ceil(633 / 20))]
    assert p.env.focus_share == 0.15 and abs(float(p.env.focus_q.sum()) - 1) < 1e-5
    q = _probe(motion, _args(eprobe_start=40, eprobe_every=50, eprobe_graph=False, eprobe_async=0), tmp_path)
    leads = iter([250, 260, 270])
    for it in range(0, 160):
        q.tick(it, pi)
    assert q.applied == [(50, 50), (100, 100), (150, 150)]
    r = _probe(motion, _args(eprobe_start=20, eprobe_every=20), tmp_path)
    leads = iter([450, 450])
    with pytest.raises(RuntimeError):
        for it in range(0, 60):
            r.tick(it, pi)


def test_probe_state_round_trip(motion, tmp_path):
    p = _probe(motion, _args(), tmp_path)
    p.focus.counts[3] = 2.0
    p.env.set_focus(torch.full((motion.num_bins,), 1.0 / motion.num_bins), 0.15)
    sd = p.state_dict()
    q = _probe(motion, _args(), tmp_path)
    q.load_state_dict(sd)
    assert torch.equal(q.focus.counts, p.focus.counts) and torch.equal(q.env.focus_q, p.env.focus_q) and q.env.focus_share == 0.15
    with pytest.raises(ValueError):
        q.load_state_dict({})
