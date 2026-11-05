"""BlinkMimic's options (ndm_gym.algos.blinkmimic, docs/blinkmimic.md): ``--algo blinkmimic`` as SuperMimic with BlinkMimic's
recipe, the options' checks, Muon's parameter split, the policy average, the mirror augmentation, the semi-implicit world
model (whose step with zero position corrections is SuperTrack's acceleration world model's with 4 substeps), the contact
reward, the reset statistics and the policy's learning-rate multiplier. At their defaults the algorithms are unchanged
(test_algo_equivalence, test_supertrack_golden, and test_defaults_are_the_unchanged_updates here)."""

import argparse
import copy
import importlib.util

import pytest
import torch
import torch.nn as nn

from conftest import DEV, random_states
from test_algo_equivalence import EXACT
from test_bptt_tricks import _algo

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
needs_isaaclab = pytest.mark.skipif(not HAVE_ISAACLAB, reason="ndm_gym.train imports Isaac Lab's launcher")

# BlinkMimic's PRESET[main], verbatim (scripts/presets.sh @ d264658), and how its flags map to this library's options
THEIRS = ("--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 "
          "--fail_reset 0.3 --contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip "
          "--wm_only_iters 0 --bin_fix --ref_contact_fix --ckpt_minutes 0 --cuda_graph --eprobe_share 0.15 --graph_early "
          "--collect_graph --reset_masked --pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --iterations 3500 "
          "--ckpt_every 100 --ckpt_light --eprobe_graph --eprobe_async 20 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7 "
          "--lr_decay_by 1400")  # fmt: skip
RENAMED = {"--fused_rot": ["--elementwise_3x3"], "--compile_more": ["--compile_eager"], "--pi_ref_skip": ["--sm_residual"],
           "--cuda_graph": ["--update_graphs"]}  # fmt: skip
LEFT_OUT = {"--ckpt_minutes"}


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _parser():
    from ndm_gym.algos.blinkmimic import add_blinkmimic_args
    from ndm_gym.train import add_speed_args, add_train_args

    parser = argparse.ArgumentParser()
    add_train_args(parser), add_speed_args(parser), add_blinkmimic_args(parser)
    return parser


def _snapshot(*modules):
    return [t.detach().clone() for m in modules for t in list(m.parameters()) + list(m.buffers())]


# ---------------------------------------------------------------------------------------------------- preset and checks
def test_preset_is_blinkmimics_recipe():
    """PRESET is BlinkMimic's PRESET[main] in this library's options: the same flags and values in the same order, four
    under other names (--cuda_graph as --update_graphs), --ckpt_minutes 0 left out (NOT_PORTED: no time-based checkpoints
    without --ckpt_hours)."""
    from ndm_gym.algos.blinkmimic import NOT_PORTED, PRESET

    tokens, mapped, i = THEIRS.split(), [], 0
    while i < len(tokens):
        flag = tokens[i]
        n = 1 + int(i + 1 < len(tokens) and not tokens[i + 1].startswith("--"))
        values = [] if n == 1 else tokens[i + 1].split(",")  # --wm_hidden takes its widths space-separated here
        if flag not in LEFT_OUT:
            mapped += RENAMED.get(flag, [flag]) + values
        else:
            assert flag in NOT_PORTED
        i += n
    assert mapped == PRESET


@needs_isaaclab
def test_algo_blinkmimic_is_supermimic_with_the_preset():
    from ndm_gym.algos.blinkmimic import PRESET, check_args, expand_preset

    parser = _parser()
    argv = expand_preset(["--algo", "blinkmimic", "--iterations", "100", "--seed", "7"])
    assert argv[:4] == ["--algo", "supermimic", "--preset", "blinkmimic"] and argv[4 : 4 + len(PRESET)] == PRESET
    a = parser.parse_args(argv)
    check_args(a)
    assert (a.algo, a.preset, a.iterations, a.seed) == ("supermimic", "blinkmimic", 100, 7)  # the command line's own options win
    want = dict(pi_ema=0.995, no_sync=True, fused_adam=True, elementwise_3x3=True, compile_eager=True, lr=4e-4, wm_batches=5, wm_hidden=[1024] * 3,
                fail_reset=0.3, contact_reward=0.25, wm_structured=True, wm_mirror=0.5, pi_opt="muon", wm_opt="muon", sm_residual=True,
                wm_only_iters=0, bin_fix=True, ref_contact_fix=True, update_graphs=True, graph_early=True, collect_graph=True,
                reset_masked=True, ckpt_light=True, wm_cudagraphs=False, pi_cudagraphs=False, critic_cudagraphs=False, pi_batch=2048,
                pi_lr_mult=4.0, lr_decay_ret=0.75, lr_decay_len=300, lr_decay_by=1400, ckpt_every=100, eprobe_share=0.15, eprobe_graph=True,
                eprobe_async=20, eprobe_start=1000, eprobe_every=125, eprobe_decay=0.7, eprobe_tile=300, eprobe_lead_min=200,
                eprobe_lead_max=450, eprobe_win=10)  # fmt: skip
    assert {k: getattr(a, k) for k in want} == want
    assert parser.parse_args(expand_preset(["--algo=blinkmimic"])).iterations == 3500
    plain = ["--algo", "supermimic", "--seed", "1"]
    assert expand_preset(plain) is plain  # other command lines are unchanged, and every option is off
    d = parser.parse_args(plain)
    assert (d.preset, d.pi_ema, d.pi_opt, d.wm_opt, d.wm_mirror, d.wm_structured, d.contact_reward, d.ref_contact_fix, d.fail_reset,
            d.bin_fix, d.pi_lr_mult, d.lr_decay_ret, d.lr_decay_by) == (None, 0.0, "adam", "adam", 0.0, False, 0.0, False, 0.0, False, 1.0, 0.0, 0)  # fmt: skip
    assert (d.eprobe_share, d.eprobe_graph, d.eprobe_async) == (0.0, False, 0)
    for bad in (["--algo", "blinkmimic", "--algo", "supertrack"], ["--algo", "supermimic", "--preset", "blinkmimic"]):
        with pytest.raises(ValueError):
            expand_preset(bad)


@needs_isaaclab
@pytest.mark.parametrize(
    "argv",
    [
        ["--algo", "supertrack", "--wm_structured"],
        ["--algo", "supermimic", "--wm_structured", "--wm_backbone", "rigidformer"],
        ["--algo", "supertrack", "--contact_reward", "0.25"],
        ["--algo", "supermimic", "--ref_contact_fix"],
        ["--algo", "supermimic", "--wm_mirror", "0.5", "--wm_sample", "prioritized"],
        ["--algo", "supermimic", "--fail_reset", "0.3", "--paired", "--motions", "interhuman"],
        ["--algo", "supermimic", "--pi_lr_mult", "4", "--lr_decay_by", "1400"],
        ["--algo", "supermimic", "--lr_decay_ret", "0.75"],
        ["--algo", "supermimic", "--pi_ema", "1"],
        ["--algo", "supermimic", "--update_graphs", "--fused_adam"],
        ["--algo", "supermimic", "--update_graphs", "--no_sync"],
        ["--algo", "supermimic", "--update_graphs", "--no_sync", "--fused_adam", "--pi_cudagraphs"],
        ["--algo", "supermimic", "--no_sync", "--imag_fail_starts", "0.1"],
        ["--algo", "supermimic", "--no_sync", "--pi_rollout_clip", "3"],
        ["--algo", "supermimic", "--graph_early"],
        ["--algo", "supermimic", "--no_sync", "--fused_adam", "--update_graphs", "--collect_graph"],
        ["--algo", "supermimic", "--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--reset_masked"],
        ["--algo", "supermimic", "--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--paired",
         "--motions", "interhuman"],
        ["--algo", "supermimic", "--eprobe_share", "0.15"],
        ["--algo", "supermimic", "--pi_ema", "0.995", "--eprobe_share", "0.15", "--eprobe_async", "20"],
        ["--algo", "supermimic", "--last_every_min", "30"],
        ["--algo", "supermimic", "--ckpt_light", "--last_every_min", "0"],
        ["--algo", "supermimic", "--ckpt_light", "--last_every_min", "-1"],
        ["--algo", "supermimic", "--ckpt_light", "--last_every_min", "nan"],
    ],
)
def test_unsupported_settings_are_refused(argv):
    from ndm_gym.algos.blinkmimic import check_args

    with pytest.raises(ValueError):
        check_args(_parser().parse_args(argv))


@needs_isaaclab
def test_algorithm_independent_options_apply_to_supertrack():
    from ndm_gym.algos.blinkmimic import check_args

    check_args(_parser().parse_args(["--algo", "supertrack", "--pi_ema", "0.995", "--pi_opt", "muon", "--wm_opt", "muon", "--wm_mirror",
                                     "0.5", "--fail_reset", "0.3", "--bin_fix", "--pi_lr_mult", "4", "--lr_decay_ret", "0.75"]))  # fmt: skip


@needs_isaaclab
def test_paired_collection_graph_with_masked_resets():
    from ndm_gym.algos.blinkmimic import check_args

    check_args(_parser().parse_args(["--algo", "supermimic", "--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph",
                                     "--reset_masked", "--paired", "--motions", "interhuman"]))  # fmt: skip


def test_resume_must_match():
    from ndm_gym.algos.blinkmimic import check_resume

    args = argparse.Namespace(pi_ema=0.995, pi_opt="muon", wm_opt="muon", wm_structured=True)
    check_resume(dict(vars(args)), args)
    with pytest.raises(ValueError):
        check_resume({}, args)  # a run from before the options: Adam, no average, the increment world model


# ---------------------------------------------------------------------------------------------------- Muon, average, defaults
def test_muon_on_the_hidden_matrices():
    """--pi_opt muon: Muon (matched to Adam's update RMS, no weight decay) on the encoder's and decoder's hidden weight
    matrices, Adam on every other parameter; set_lr reaches both; a resumed optimizer steps as the original."""
    from ndm_gym.algos.common import set_lr
    from ndm_gym.core.muon import MuonAdam
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.tasks.tracking import OBS_DIM

    torch.manual_seed(0)
    pi = TrackingPolicy(OBS_DIM, 73 * 10)
    opt = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-3, betas=(0.9, 0.95))
    linears = lambda m: [x for x in m.modules() if isinstance(x, nn.Linear)]  # noqa: E731
    muon = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adam = {id(p) for g in opt.adam.param_groups for p in g["params"]}
    assert muon == {id(lin.weight) for m in (pi.enc, pi.dec) for lin in linears(m)[1:-1]} and len(muon) == 6
    assert not muon & adam and muon | adam == {id(p) for p in pi.parameters()}
    assert opt.muon.defaults["adjust_lr_fn"] == "match_rms_adamw" and opt.muon.defaults["weight_decay"] == 0
    assert opt.adam.defaults["betas"] == (0.9, 0.95)
    set_lr(opt, 5e-4)
    assert [g["lr"] for g in opt.param_groups] == [5e-4] * len(opt.param_groups)
    x, g = torch.randn(8, 10, OBS_DIM), torch.randn(8, 730)

    def step(p, o):
        o.zero_grad()
        (p(x, g) ** 2).mean().backward()
        o.step()

    before = [p.detach().clone() for p in pi.parameters()]
    step(pi, opt)
    assert all(not torch.equal(a, b) for a, b in zip(before, pi.parameters()))
    pi2 = copy.deepcopy(pi)
    opt2 = MuonAdam(pi2, [pi2.enc, pi2.dec], lr=1e-3)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))  # a copy: loading keeps the tensors of a state on the same device
    step(pi, opt), step(pi2, opt2)
    for a, b in zip(pi.parameters(), pi2.parameters()):
        torch.testing.assert_close(b, a, **EXACT)


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_policy_average(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--pi_ema d: the first update copies the policy, every later one moves the average by 1 - d toward it (weights and
    normalizer statistics); checkpoints hold the average as "pi" and the trained policy as "pi_live", and a resumed run
    gets both back. Without the option there is no average and the checkpoint keys are as before."""
    algo, _ = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_ema=0.9)
    algo.update_pi_avg()
    for a, b in zip(_snapshot(algo.pi_avg.module), _snapshot(algo.pi)):
        torch.testing.assert_close(a, b, **EXACT)
    old = _snapshot(algo.pi_avg.module)
    with torch.no_grad():
        for p in algo.pi.parameters():
            p.add_(torch.randn_like(p))
    algo.pi.obs_norm.update(torch.randn(16, algo.pi.obs_norm.mean.shape[0]))
    algo.update_pi_avg()
    for avg, o, live in zip(_snapshot(algo.pi_avg.module), old, _snapshot(algo.pi)):
        torch.testing.assert_close(avg, o + 0.1 * (live - o))
    ck = algo.state_dict()
    for k, m in (("pi", algo.pi_avg.module), ("pi_live", algo.pi)):
        for a, b in zip(ck[k].values(), m.state_dict().values()):
            torch.testing.assert_close(a, b, **EXACT)
    other, _ = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_ema=0.9)
    other.load_state_dict(copy.deepcopy(ck))
    for a, b in zip(_snapshot(other.pi, other.pi_avg), _snapshot(algo.pi, algo.pi_avg)):
        torch.testing.assert_close(a, b, **EXACT)
    plain, _ = _algo(name, kin, synthetic_motion, synthetic_mdp)
    assert plain.pi_avg is None and not {"pi_live", "pi_avg"} & set(plain.state_dict())


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_defaults_are_the_unchanged_updates(eager, kin, synthetic_motion, synthetic_mdp, name):
    """The options at their defaults, as train.py passes them, give bit for bit the updates of a run's arguments without
    them (which test_algo_equivalence and test_supertrack_golden compare with the reference code)."""
    defaults = dict(pi_ema=0.0, pi_opt="adam", wm_opt="adam", wm_mirror=0.0, wm_structured=False, contact_reward=0.0, ref_contact_fix=False)
    out = {}
    for explicit in (False, True):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **(defaults if explicit else {}))
        torch.manual_seed(1)
        res = list(algo.wm_update(buf))
        torch.manual_seed(2)
        res += list(algo.policy_update(buf))
        algo.update_pi_avg()
        out[explicit] = res + _snapshot(algo.wm, algo.pi)
    for a, b in zip(out[False], out[True]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)


# ---------------------------------------------------------------------------------------------------- mirror
def test_mirror(kin, motion):
    """--wm_mirror's reflection is an involution; the G1's default pose and action scales are symmetric (so setpoints
    mirror as joints); the forward kinematics of a mirrored state are the mirrored bodies, left and right swapped."""
    from ndm_gym.algos.blinkmimic import Mirror
    from ndm_gym.robots import g1

    m = Mirror(DEV)
    f, S, xi = random_states(motion, 64, seed=0)
    for a, b in zip(m.state(m.state(S)), S):
        torch.testing.assert_close(a, b, **EXACT)
    torch.testing.assert_close(m.xi(m.xi(xi)), xi, **EXACT)
    q0 = torch.tensor(g1.default_joint_pos_list())
    alpha = torch.tensor([g1.action_scale(n) for n in g1.G1_JOINT_NAMES])
    assert torch.equal(m.joints(q0), q0) and torch.equal(alpha[m.perm], alpha)
    mx = m.xi(xi)
    assert torch.equal(mx[:, 0], xi[:, 0]) and torch.equal(mx[:, 31], -xi[:, 31]) and torch.equal(mx[:, [30, 32]], xi[:, [30, 32]])
    swap = [g1.TRACKED_BODIES.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in g1.TRACKED_BODIES]
    bp, bR, bv, bw = kin.forward(*S)
    mp, mR, mv, mw = kin.forward(*m.state(S))
    M = m.M
    torch.testing.assert_close(mp, (bp @ M)[:, swap], atol=3e-5, rtol=0)  # float32 rounding along the chains (1.0e-5 at most seen)
    torch.testing.assert_close(mR, (M @ bR @ M)[:, swap], atol=3e-5, rtol=0)
    torch.testing.assert_close(mv, (bv @ M)[:, swap], atol=1e-4, rtol=0)
    torch.testing.assert_close(mw, -(bw @ M)[:, swap], atol=1e-4, rtol=0)


def test_mirror_windows(eager, kin, synthetic_motion, synthetic_mdp):
    """The first share of the world-model windows is replaced by their mirror images (states, setpoints, context), the
    rest and the masks are untouched; Algo.wm_update fits on those windows."""
    from ndm_gym.state import flatten_state, unflatten_state

    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, wm_mirror=0.5)
    ref, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp)
    n = int(algo.args.wm_batch * 0.5)
    torch.manual_seed(1)
    w = buf.sample_windows(algo.args.wm_batch, algo.args.wm_window)
    mw = algo.mirror.windows(w, n)
    for k in ("mask", "done"):
        assert torch.equal(mw[k], w[k])
    for k in ("s", "s1", "a", "xi"):
        assert torch.equal(mw[k][n:], w[k][n:])
    torch.testing.assert_close(mw["a"][:n], algo.mirror.joints(w["a"][:n]), **EXACT)
    torch.testing.assert_close(mw["s1"][:n], flatten_state(algo.mirror.state(unflatten_state(w["s1"][:n]))), **EXACT)
    want = ref.wm_loss(ref.wm_rollout(mw["s"], mw["a"], mw["xi"]), mw)[0] / algo.args.wm_window
    torch.manual_seed(1)
    torch.testing.assert_close(algo.wm_update(buf)[0], want.detach(), **EXACT)


# ---------------------------------------------------------------------------------------------------- semi-implicit world model
def test_semi_implicit_round_trip(synthetic_motion):
    """The semi-implicit model's targets integrate back to the next state."""
    from ndm_gym.worldmodels.semi_implicit import SemiImplicitIncrementWorldModel

    from ndm_gym.core.rotations import so3_exp

    wm = SemiImplicitIncrementWorldModel(hidden=[64, 64])
    f, S, _ = random_states(synthetic_motion, 64, seed=0)
    torch.manual_seed(0)
    p, R, v, w, q, qd = S
    S1 = (p + 0.02 * torch.randn_like(p), so3_exp(0.05 * torch.randn_like(p)) @ R, v + 0.3 * torch.randn_like(v), w + 0.5 * torch.randn_like(w),
          q + 0.02 * torch.randn_like(q), qd + 0.5 * torch.randn_like(qd))  # fmt: skip
    for a, b in zip(wm.integrate(S, wm.targets(S, S1)), S1):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_semi_implicit_integration_is_supertracks(synthetic_motion):
    """With zero position corrections the semi-implicit world model steps as SuperTrack's acceleration world model with 4
    substeps (Fussell et al. 2021, §3.4): velocities gain dt a, positions follow from the old and new velocities with
    weight (n + 1) / (2 n) = 0.625 on the new one."""
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.semi_implicit import SemiImplicitIncrementWorldModel

    torch.manual_seed(0)
    st = AccelWorldModel(hidden=[64, 64], substeps=4)  # zero last layer: the network outputs out_norm's mean
    si = SemiImplicitIncrementWorldModel(hidden=[64, 64])
    acc = torch.randn(3 + 3 + 29)
    st.out_norm.mean = acc.clone()
    f, S, xi = random_states(synthetic_motion, 64, seed=1)
    dt, z = 0.02, torch.zeros
    d = torch.cat((z(6), dt * acc[:6], z(29), dt * acc[6:])).expand(64, -1)
    for a, b in zip(si.integrate(S, d), st.step(S, torch.randn(64, 29), normalize_xi(xi))):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_semi_implicit_in_supermimic(eager, kin, synthetic_motion, synthetic_mdp):
    """--wm_structured: SuperMimic builds the semi-implicit model, fits its output normalizer on its targets and its loss
    scale on the plain increments; the fused inference copy refuses it."""
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.state import unflatten_state
    from ndm_gym.worldmodels.g1_infer import inference_copy
    from ndm_gym.worldmodels.semi_implicit import SemiImplicitIncrementWorldModel

    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, wm_structured=True)
    assert isinstance(algo.wm, SemiImplicitIncrementWorldModel)
    w = {k: v.flatten(0, 1) for k, v in buf.data.items()}
    S0, S1 = unflatten_state(w["s"]), unflatten_state(w["s1"])
    algo.wm.out_norm, algo.wm.inc_norm = RunningNorm(algo.wm.out_norm.mean.shape[0], eps=1e-4), RunningNorm(algo.wm.out_norm.mean.shape[0], eps=1e-4)
    algo.update_wm_norms(S0, S1, w["a"], w["xi"])
    for norm, x in ((algo.wm.out_norm, algo.wm.targets(S0, S1)), (algo.wm.inc_norm, algo.wm.increments(S0, S1))):
        ref = RunningNorm(x.shape[-1], eps=1e-4)
        ref.update(x)
        torch.testing.assert_close(norm.var, ref.var, **EXACT)
    assert not torch.allclose(algo.wm.out_norm.mean, algo.wm.inc_norm.mean)
    loss, e8 = algo.wm_update(buf)
    assert torch.isfinite(loss) and torch.isfinite(e8)
    with pytest.raises(ValueError):
        inference_copy(algo.wm, dtype=None, attention=None, fused=True, check=False)
    with pytest.raises(ValueError):
        _algo("supertrack", kin, synthetic_motion, synthetic_mdp, wm_structured=True)


# ---------------------------------------------------------------------------------------------------- contact reward
def test_contact_reward(eager, kin, motion, mdp):
    """--contact_reward w adds w * dt times the contact term of the feet to the reward of SuperMimic's imagination only;
    the reference's contact is the fixed 4.5 cm rule, or with --ref_contact_fix each clip's own floor + 1.5 cm at a
    horizontal speed below 0.5 m/s."""
    from ndm_gym.robots.g1 import TRACKED_BODIES
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.contact import CONTACT_Z, contact_term, foot_bodies, reference_contact

    feet = foot_bodies()
    assert [TRACKED_BODIES[i] for i in feet] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    fixed = reference_contact(motion, feet, calibrated=False)
    assert torch.equal(fixed, (motion.body_p[:, feet, 2] < CONTACT_Z).float())
    f, S, _ = random_states(motion, 64, seed=0)
    a, a_prev = torch.randn(64, 29), torch.randn(64, 29)
    r0 = mdp.reward(S, f, a, a_prev)
    torch.testing.assert_close(mdp.reward(S, f, a, a_prev, contact=None), r0, **EXACT)
    bp, _, bv, _ = kin.forward(*S)
    r1 = mdp.reward(S, f, a, a_prev, contact=(0.25, feet, fixed))
    torch.testing.assert_close(r1 - r0, 0.25 * contact_term(bp[:, feet], bv[:, feet], fixed[f]) * CONTROL_DT)
    up = contact_term(torch.tensor([[[0.0, 0.0, 0.5]]]), torch.zeros(1, 1, 3), torch.zeros(1, 1))
    down = contact_term(torch.tensor([[[0.0, 0.0, 0.0]]]), torch.zeros(1, 1, 3), torch.ones(1, 1))
    torch.testing.assert_close(up, torch.ones(1))
    torch.testing.assert_close(down, 2 * torch.ones(1), atol=1e-6, rtol=0)
    cal = reference_contact(motion, feet, calibrated=True)
    c = motion.clip_id == 3
    z, vh = motion.body_p[c][:, feet, 2], motion.body_v[c][:, feet, :2].norm(dim=-1)
    assert torch.equal(cal[c], ((z < z.quantile(0.05, dim=0) + 0.015) & (vh < 0.5)).float())
    algo, _ = _algo("supermimic", kin, motion, mdp, contact_reward=0.25, ref_contact_fix=True)
    assert torch.equal(algo.ref_contact, cal)
    torch.testing.assert_close(algo._reward_term(S, f, a, a_prev)[0], mdp.reward(S, f, a, a_prev, contact=(0.25, feet, cal)), **EXACT)


# ---------------------------------------------------------------------------------------------------- resets, learning rate
def test_reset_statistics(motion):
    """--bin_fix averages a step's samples of one bin (index_put_ keeps one of them; without repeats both agree);
    --fail_reset moves a share of the reset distribution to the bin before each failing bin of the same clip."""
    from ndm_gym.sim.rigid.reset_stats import bin_ema, mix_failures, previous_bins

    nb = motion.num_bins
    prev = previous_bins(motion)
    first = motion.bin_start == motion.clip_start[motion.bin_clip]
    assert torch.equal(prev[first], torch.arange(nb)[first]) and torch.equal(prev[~first], torch.arange(nb)[~first] - 1)
    g = torch.Generator().manual_seed(0)
    stat = torch.rand(nb, generator=g)
    want = stat.clone()
    want[3], want[5] = 0.8 * stat[3] + 0.2 * 2.0, 0.8 * stat[5] + 0.2 * 4.0
    torch.testing.assert_close(bin_ema(stat, torch.tensor([3, 3, 5]), torch.tensor([1.0, 3.0, 4.0]), 0.8), want)
    b, x = torch.tensor([1, 4, 9]), torch.rand(3, generator=g)
    torch.testing.assert_close(bin_ema(stat, b, x, 0.8), stat.clone().index_put_((b,), 0.8 * stat[b] + 0.2 * x))
    probs = torch.softmax(torch.rand(nb, generator=g), 0)
    torch.testing.assert_close(mix_failures(probs, torch.zeros(nb), prev, 0.3), probs)
    k = int((~first).nonzero()[0])
    fail = torch.zeros(nb)
    fail[k] = 0.5
    want = 0.7 * probs
    want[k - 1] += 0.3
    torch.testing.assert_close(mix_failures(probs, fail, prev, 0.3), want)


def test_policy_learning_rate_multiplier():
    """--pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --lr_decay_by 1400: x4 until the logged training return
    reaches 0.75 (or iteration 1400), then a linear ramp to x1 over 300 iterations; the trigger is checkpointed."""
    from ndm_gym.algos.blinkmimic import PolicyLR

    a = argparse.Namespace(pi_lr_mult=4.0, lr_decay_ret=0.75, lr_decay_len=300, lr_decay_by=1400)
    s = PolicyLR(a)
    assert s.on and s.at(0) == 4.0 and s.observe(50, 0.5) is None and s.at(100) == 4.0
    assert "training return 0.800 >= 0.75 at it 100" in s.observe(100, 0.8)
    assert s.observe(150, 0.9) is None  # once
    assert (s.at(100), s.at(250), s.at(400), s.at(5000)) == (4.0, 2.5, 1.0, 1.0)
    late = PolicyLR(a)
    assert late.observe(1350, 0.1) is None and "deadline --lr_decay_by 1400 at it 1400" in late.observe(1400, 0.1)
    resumed = PolicyLR(a)
    resumed.load_state_dict(s.state_dict())
    assert resumed.at(250) == 2.5
    with pytest.raises(ValueError):
        PolicyLR(a).load_state_dict({})
    off = PolicyLR(argparse.Namespace(pi_lr_mult=1.0, lr_decay_ret=0.0, lr_decay_len=300, lr_decay_by=0))
    assert not off.on and off.state_dict() == {} and off.observe(1400, 0.9) is None


# ---------------------------------------------------------------------------------------------------- the recipe's updates
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_recipe_updates_run(eager, kin, synthetic_motion, synthetic_mdp, name):
    """Every option of the recipe that applies to the algorithm, together: the updates are finite and train the world
    model, the policy and its average."""
    kw = dict(pi_ema=0.995, pi_opt="muon", wm_opt="muon", wm_mirror=0.5)
    if name == "supermimic":
        kw.update(wm_structured=True, contact_reward=0.25, ref_contact_fix=True, sm_residual=True)
    algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **kw)
    params = lambda: [p.detach().clone() for m in (algo.wm, algo.pi) for p in m.parameters()]  # noqa: E731
    before = params()
    torch.manual_seed(1)
    wm_loss, e8 = algo.wm_update(buf)
    torch.manual_seed(2)
    pi_loss, v_loss, imag_r, gn, _ = algo.policy_update(buf)
    algo.update_pi_avg(), algo.update_pi_avg()
    finite = [wm_loss, pi_loss, gn] + ([e8, v_loss, imag_r] if name == "supermimic" else [])  # SuperTrack's are nan
    assert all(bool(torch.isfinite(torch.as_tensor(x))) for x in finite)
    assert all(not torch.equal(a, b) for a, b in zip(before, params()))
    assert int(algo.pi_avg.n_averaged) == 2
