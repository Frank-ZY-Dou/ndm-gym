"""BlinkMimic's evaluation-guided resets (ndm_gym.algos.eprobe, ``--eprobe_*``): the probe worlds' tiles and sweep frames, the
failure counts and their window against BlinkMimic's own code (train.py @ d264658, restated below), the focus's share of
the reset distribution, the deterministic probe policy, the options' checks and the trainer's timing of the sweeps. The
probe simulator itself runs in the GPU smoke runs (docs/blinkmimic.md); at the defaults nothing changes
(test_algo_equivalence, test_supertrack_golden). --eprobe_lean (this library's): the sweep takes the simulator's lean step and
records the same failures (a stand-in simulator, on CPU); the lean step against the default one in the simulator (a process
of this file, ``--worker``, on the synthetic clips; marked slow, needs a GPU and Isaac Lab). The two-robot scene's sweeps: tiles,
offsets, focus and frames on CPU; its steps and sweeps in the simulator (``--worker --pairs``, slow)."""

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import types

import pytest
import torch

from conftest import DEV, REPO, synthetic_clips

HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None


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
    [dict(pi_ema=0.0), dict(eprobe_graph=False), dict(eprobe_async=5), dict(eprobe_share=1.0), dict(eprobe_tile=0),
     dict(eprobe_lead_min=500), dict(eprobe_share=0.0), dict(eprobe_decay=1.5),
     dict(eprobe_share=0.0, eprobe_graph=False, eprobe_async=0, eprobe_lean=True)],
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
    check_eprobe_args(_args(eprobe_lean=True))  # the lean sweep step
    check_eprobe_args(_args(paired=True))  # the two-robot scene


def test_lean_step_is_off_by_default():
    from ndm_gym.algos.eprobe import add_eprobe_args

    p = argparse.ArgumentParser()
    add_eprobe_args(p)
    assert p.parse_args([]).eprobe_lean is False and p.parse_args(["--eprobe_lean"]).eprobe_lean is True


class _SweepEnv:
    """A stand-in for the probe simulator: a step moves every world one frame on, to the reference state there, and
    terminates the worlds whose frame is in ``fail``; ``step`` (the default) and ``step_probe`` (--eprobe_lean) do the
    same and count their calls."""

    def __init__(self, motion, f0, fail):
        n = len(f0)
        self.N, self.device, self.motion, self.fail = n, DEV, motion, fail
        self.f, self.a_prev, self.xi = f0.clone(), torch.zeros(n, 29), torch.zeros(n, 33)
        self.done_eval, self.failed_eval = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        self.calls = {"step": 0, "step_probe": 0}

    def reset(self, ids, frames):
        self.f[ids] = frames

    def state(self, vel=None):
        return self.motion.ref_state(self.f)

    def _advance(self, a):
        self.f.copy_(self.f + 1)
        self.a_prev.copy_(a)
        term = torch.isin(self.f, self.fail)
        self.done_eval |= term
        return {"term": term}

    def step(self, a, S, inplace=False):
        assert inplace
        self.calls["step"] += 1
        return self._advance(a)

    def step_probe(self, a):
        self.calls["step_probe"] += 1
        return self._advance(a)


def test_lean_sweep_takes_the_lean_step_and_records_the_same_failures(motion, mdp):
    """ProbeSweep's step with --eprobe_lean calls the simulator's step_probe where the default calls step; given the same
    terminations both record the same first-failure frames, stop the probes at their tiles' ends and cap their frames."""
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.eprobe import ProbeSweep, probe_tiles, sweep_frames
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.tasks.tracking import OBS_DIM

    torch.manual_seed(0)
    K, H, tile, lead, n = 4, 4, 30, 10, 6
    tiles = probe_tiles(motion, 300)[:n]
    pi = TrackingPolicy(OBS_DIM, 73 * K, hist=H).to(DEV).eval()
    ref_u = reference_setpoints(motion, mdp.default_q, torch.ones(29))
    f0, stop, cap = sweep_frames(motion, tiles, lead, tile)
    fail = torch.stack((f0[1] + 3, f0[4] + 17))  # two probes fail, at their 3rd and 17th step
    res = {}
    for lean in (False, True):
        env = _SweepEnv(motion, f0, fail)
        sweep = ProbeSweep(env, pi, motion, mdp, K, H, True, ref_u, tiles, tile, graph=False, lean=lean)
        sweep.stop.copy_(stop), sweep.cap.copy_(cap)
        o = mdp.obs(motion.ref_state(env.f), env.a_prev, env.xi, env.f)
        sweep.hist = o.unsqueeze(1).repeat(1, H, 1)
        with torch.no_grad():
            for _ in range(lead + tile):
                sweep._step(motion.ref_state(env.f))
        assert env.calls == ({"step": 0, "step_probe": lead + tile} if lean else {"step": lead + tile, "step_probe": 0})
        res[lean] = (sweep.fail_f.clone(), env.f.clone(), env.done_eval.clone(), sweep.hist.clone())
    assert all(torch.equal(x, y) for x, y in zip(res[False], res[True]))
    fail_f = res[True][0]
    assert fail_f.tolist() == [-1, int(f0[1]) + 2, -1, -1, int(f0[4]) + 16, -1]  # the frame each failing step started at
    assert bool(res[True][2].all()) and torch.equal(res[True][1], cap)


HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None


@pytest.mark.skipif(not HAVE_ISAACLAB, reason="the pairing of the clips is the scene module's")
def test_paired_tiles_offsets_and_focus(kin, tmp_path):
    """The two-robot scene's sweeps: tiles on the pairs' first sides, each with the offset to the same time of its second
    side's clip; next_pair_bins is the inverse of previous_pair_bins within a pair; a failure of either robot counts in its
    own clip's bin, and the focus over the pair bins weighs each pair bin's two sides' counts and those of the win pair bins
    after it in its pair (BlinkMimic's focus over the pair bins)."""
    from test_paired_tracking import _pair_motion

    from ndm_gym.algos.eprobe import FailureFocus, next_pair_bins, pair_offsets, probe_tiles
    from ndm_gym.sim.rigid.isaaclab_mjwarp import motion_pairs, pair_bins, previous_pair_bins

    m = _pair_motion(kin, tmp_path)
    pc = motion_pairs(m)
    a, b, _ = pair_bins(m, pc)
    tiles = probe_tiles(m, 30, pc[0])
    assert torch.equal(tiles, torch.cat([torch.arange(int(m.clip_start[c]), int(m.clip_end[c]) - 1, 30) for c in pc[0].tolist()]))
    off = pair_offsets(m, pc, tiles)
    ca, cb = m.clip_id[tiles], m.clip_id[tiles + off]
    assert torch.equal(cb, pc[1][(pc[0].unsqueeze(1) == ca).float().argmax(0)])  # each tile's partner clip
    assert torch.equal(tiles + off - m.clip_start[cb], tiles - m.clip_start[ca])
    with pytest.raises(ValueError):
        pair_offsets(m, pc, tiles + off)  # second sides' frames are not tiles
    nxt, prev = next_pair_bins(m, a), previous_pair_bins(m, a)
    moved = nxt != torch.arange(len(a))
    assert torch.equal(prev[nxt[moved]], torch.arange(len(a))[moved]) and int((~moved).sum()) == pc.shape[1]  # one last bin per pair
    win, decay = 3, 0.7
    focus = FailureFocus(m, win, decay, (a, b))
    k = int(moved.nonzero()[0])  # a pair bin with a next one
    fail_f = torch.tensor([-1, int(m.bin_start[b[k]]) + 1, -1, int(m.bin_start[a[nxt[k]]])])  # the second robot at k, the first at k + 1
    q = focus.update(fail_f)
    c = torch.zeros(len(a))
    c[k] += 1
    c[nxt[k]] += 1
    exp = c.clone()
    w = c.clone()
    for _ in range(win):
        w = w[nxt] * (nxt != torch.arange(len(a))).float()
        exp += w
    assert torch.allclose(q, exp / exp.sum()) and q[k] > q[nxt[k]] > 0
    q2 = focus.update(torch.full((4,), -1))  # no failure: the counts decay, the focus stays
    assert torch.allclose(q2, q) and torch.isclose(focus.counts.sum(), torch.tensor(2 * decay))


@pytest.mark.skipif(not HAVE_ISAACLAB, reason="the sweep imports the scene module")
def test_paired_sweep_starts_both_robots_at_the_same_time(kin, tmp_path):
    """ProbeSweep in the two-robot scene: the second robots start, stop and are capped at their partner clips' same times
    (the first robots' rows first), each robot's first failure is its own, and a failing robot does not end its partner's
    probe."""
    from test_paired_tracking import _pair_motion

    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.eprobe import ProbeSweep, pair_offsets, probe_tiles, sweep_frames
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import motion_pairs
    from ndm_gym.tasks.tracking import MDP, OBS_DIM

    torch.manual_seed(0)
    m = _pair_motion(kin, tmp_path)
    mdp = MDP(kin, m, torch.tensor(g1.default_joint_pos_list(), device=DEV))
    pc = motion_pairs(m)
    K, H, tile, lead = 4, 4, 30, 10
    tiles = probe_tiles(m, 10**6, pc[0])  # one tile per pair, at its start: no two probes share a frame
    off = pair_offsets(m, pc, tiles)
    f0, stop, cap = sweep_frames(m, tiles, lead, tile)
    fail = torch.stack((f0[1] + 3, f0[0] + off[0] + 17))  # the first robot of probe 1 at its 3rd step, the second of probe 0 at its 17th
    T = len(tiles)
    env = _SweepEnv(m, torch.zeros(2 * T, dtype=torch.long), fail)
    pi = TrackingPolicy(OBS_DIM, 73 * K, hist=H).to(DEV).eval()
    sweep = ProbeSweep(env, pi, m, mdp, K, H, True, reference_setpoints(m, mdp.default_q, torch.ones(29)), tiles, tile, graph=False,
                       pair_off=off)  # fmt: skip
    fail_f = sweep.run(lead)
    expected = [-1] * (2 * T)
    expected[1], expected[T] = int(f0[1]) + 2, int(f0[0] + off[0]) + 16  # the frame each failing step started at
    assert fail_f.tolist() == expected
    assert torch.equal(env.f, torch.cat((cap, cap + off))) and bool(env.done_eval.all())
    with pytest.raises(ValueError):
        ProbeSweep(env, pi, m, mdp, K, H, True, None, tiles, tile, graph=False)  # 2T worlds for T one-robot probes


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


def worker(out: str):
    """An evaluation-mode TrackingSim on the synthetic clips, one probe world per tile of 40 frames, with termination
    thresholds of 2 cm and 0.02 so that probes fail within a few steps. (1) From the same reset and with the same actions,
    8 default steps and 8 lean steps (TrackingSim.step_probe). (2) Lean steps checked against the default step's rules on
    the states they leave. (3) A sweep of the default step and one of the lean step as CUDA graphs (ProbeSweep), 15 steps."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.eprobe import ProbeSweep, probe_tiles, sweep_frames
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP, OBS_DIM, RewardCfg

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, K, Hh, tile, lead = "cuda:0", 10, 10, 40, 10
    kin = g1.kinematics(dev)
    motion = MotionLib([Clip(**c) for c in synthetic_clips(sim=True)], kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), RewardCfg(term_height=0.02, term_tilt=0.02))
    tiles = probe_tiles(motion, tile)
    f0, _, _ = sweep_frames(motion, tiles, lead, tile)
    sim_cfg = make_sim(dev, launcher)
    res = {}
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        P = len(tiles)
        env = TrackingSim(sim_cfg, P, motion, mdp, eval_mode=True)
        ids = torch.arange(P, device=dev)
        acts = 0.5 * torch.randn(8, P, 29, device=dev)

        def flags():
            return {"f": env.f.clone(), "ep_step": env.ep_step.clone(), "a_prev": env.a_prev.clone(), "done": env.done_eval.clone(),
                    "failed": env.failed_eval.clone()}  # fmt: skip

        def run(lean):  # 8 steps from the reset: the flags after every step, the terminations, the states
            env.done_eval[:] = False
            env.failed_eval[:] = False
            env.reset(ids, f0)
            rec = {"term": [], "q": [], "p": [], **{k: [] for k in flags()}}
            for a in acts:
                S = env.state()
                tr = env.step_probe(a) if lean else env.step(a, S, inplace=True)
                S1 = env.state()
                rec["term"].append(tr["term"].clone()), rec["q"].append(S1[4].clone()), rec["p"].append(S1[0].clone())
                for k, v in flags().items():
                    rec[k].append(v)
            return {k: torch.stack(v).cpu() for k, v in rec.items()}

        res["default"], res["lean"] = run(False), run(True)
        # (2) a lean step against the default step's rules, on the state it leaves
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, f0)
        rules = []
        for a in acts:
            before = flags()
            tr = env.step_probe(a)
            S1, f1 = env.state(), before["f"] + 1
            term = mdp.termination(S1, f1)
            done = term | (f1 >= motion.clip_end[motion.clip_id[before["f"]]])
            rules.append(bool(torch.equal(tr["term"], term) and torch.equal(env.f, f1) and torch.equal(env.ep_step, before["ep_step"] + 1)
                              and torch.equal(env.a_prev, a) and torch.equal(env.failed_eval, before["failed"] | (term & ~before["done"]))
                              and torch.equal(env.done_eval, before["done"] | done)))  # fmt: skip
        res["rules"] = rules
        # (3) sweeps as CUDA graphs: 15 steps each (the probes stop 5 frames into their tiles), a policy with large outputs
        pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev).eval()
        pi.dec[-1].weight.mul_(50.0)
        ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
        for lean in (False, True):
            sweep = ProbeSweep(env, pi, motion, mdp, K, Hh, True, ref_u, tiles, 5, graph=True, lean=lean)
            res["sweep_lean" if lean else "sweep_default"] = {"fail_f": sweep.run(lead), "f": env.f.cpu(), "done": env.done_eval.cpu()}
    torch.save(res, out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_lean_step_is_the_default_step_without_its_reward(tmp_path):
    """TrackingSim.step_probe (--eprobe_lean): the frame counters and flags of the default evaluation step exactly, its
    terminations, and its states within the simulator's own run-to-run differences; a sweep of it as a CUDA graph records
    the default sweep's failures."""
    out = tmp_path / "lean.pt"
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--out", str(out)], cwd=REPO, capture_output=True, text=True,
                       timeout=1800)  # fmt: skip
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr  # Warp's teardown messages follow it
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    res = torch.load(out)
    d, lean = res["default"], res["lean"]
    for k in ("f", "ep_step", "a_prev", "done", "failed", "term"):
        assert torch.equal(lean[k], d[k]), k
    for k in ("q", "p"):
        torch.testing.assert_close(lean[k], d[k], rtol=1e-4, atol=1e-4)
    assert bool(d["term"].any()) and not bool(d["term"].all())  # the thresholds make some probes fail, not all at once
    assert all(res["rules"]), res["rules"]
    a, b = res["sweep_default"], res["sweep_lean"]
    assert torch.equal(b["fail_f"], a["fail_f"]) and torch.equal(b["f"], a["f"]) and torch.equal(b["done"], a["done"])
    assert int((a["fail_f"] >= 0).sum()) > 0


def worker_paired(pairs: str, out: str):
    """The two-robot scene in evaluation mode (PairedTrackingSim) on the synthetic pairs, one probe env per tile of 40 frames of
    the pairs' first sides, with termination thresholds of 2 cm and 0.02. (1) From the same reset and with the same actions,
    8 steps each of the evaluation step, the in-place step and the lean step (PairedTrackingSim.step_probe). (2) Sweeps of
    the default and the lean step, eager and as CUDA graphs (ProbeSweep), 15 steps."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.eprobe import ProbeSweep, pair_offsets, probe_tiles, sweep_frames
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, make_sim, motion_pairs
    from ndm_gym.tasks.tracking import MDP, OBS_DIM, RewardCfg

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, K, Hh, tile, lead = "cuda:0", 10, 10, 40, 10
    kin = g1.kinematics(dev)
    motion = MotionLib(interhuman.load(pairs), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), RewardCfg(term_height=0.02, term_tilt=0.02))
    pc = motion_pairs(motion).to(dev)
    tiles = probe_tiles(motion, tile, pc[0])
    off = pair_offsets(motion, pc, tiles)
    f0, _, _ = sweep_frames(motion, tiles, lead, tile)
    f0 = torch.cat((f0, f0 + off))
    sim_cfg = make_sim(dev, launcher, paired=True)
    res = {}
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = PairedTrackingSim(sim_cfg, 2 * len(tiles), motion, mdp, eval_mode=True)
        ids = torch.arange(env.N, device=dev)
        acts = 0.5 * torch.randn(8, env.N, 29, device=dev)

        def run(kind):  # 8 steps from the reset: the flags after every step, the terminations, the states
            env.done_eval[:] = False
            env.failed_eval[:] = False
            env.reset(ids, f0)
            rec = {k: [] for k in ("term", "q", "p", "f", "ep_step", "a_prev", "done", "failed")}
            for a in acts:
                S = env.state()
                tr = env.step(a, S) if kind == "eval" else env.step(a, S, inplace=True) if kind == "inplace" else env.step_probe(a)
                S1 = env.state()
                rec["term"].append(tr["term"].clone()), rec["q"].append(S1[4].clone()), rec["p"].append(S1[0].clone())
                for k, v in (("f", env.f), ("ep_step", env.ep_step), ("a_prev", env.a_prev), ("done", env.done_eval), ("failed", env.failed_eval)):
                    rec[k].append(v.clone())
            return {k: torch.stack(v).cpu() for k, v in rec.items()}

        for kind in ("eval", "inplace", "lean"):
            res[kind] = run(kind)
        pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev).eval()
        pi.dec[-1].weight.mul_(50.0)
        ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
        for graph in (False, True):
            for lean in (False, True):
                sweep = ProbeSweep(env, pi, motion, mdp, K, Hh, True, ref_u, tiles, 5, graph=graph, lean=lean, pair_off=off)
                res[("graph" if graph else "eager", lean)] = {"fail_f": sweep.run(lead), "f": env.f.cpu(), "done": env.done_eval.cpu()}
        res["P"] = len(tiles)
    torch.save(res, out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_paired_probe_steps_and_sweeps(tmp_path):
    """The two-robot scene's probe steps: the in-place step and the lean step keep the evaluation step's frame counters, flags
    and terminations exactly and its states within the simulator's run-to-run differences; the sweeps, eager and as CUDA
    graphs, of the default and the lean step record the same failures of either robot."""
    from test_paired_tracking import write_pairs

    out = tmp_path / "paired.pt"
    write_pairs(tmp_path / "pairs", 1.2)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--pairs", str(tmp_path / "pairs"), "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)  # fmt: skip
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    res = torch.load(out)
    d = res["eval"]
    for kind in ("inplace", "lean"):
        for k in ("f", "ep_step", "a_prev", "done", "failed", "term"):
            assert torch.equal(res[kind][k], d[k]), (kind, k)
        for k in ("q", "p"):
            torch.testing.assert_close(res[kind][k], d[k], rtol=1e-4, atol=1e-4)
    assert bool(d["term"].any()) and not bool(d["term"].all())
    P = res["P"]
    assert bool((d["term"][:, :P] != d["term"][:, P:]).any())  # a robot fails while its partner plays on
    ref = res[("graph", False)]
    assert int((ref["fail_f"] >= 0).sum()) > 0
    for key in (("graph", True), ("eager", False), ("eager", True)):
        assert torch.equal(res[key]["fail_f"], ref["fail_f"]), key
    for key in (("graph", True),):
        assert torch.equal(res[key]["f"], ref["f"]) and torch.equal(res[key]["done"], ref["done"])
    assert torch.equal(res[("eager", True)]["f"], res[("eager", False)]["f"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", required=True)
    ap.add_argument("--pairs", default=None, help="the two-robot scene's worker, on the two-person clips of this directory")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    if cli.pairs:
        worker_paired(cli.pairs, cli.out)
    else:
        worker(cli.out)
