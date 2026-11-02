"""Flag-gated tricks against policy-gradient spikes in the BPTT through the world model: ``--bptt_clip_k`` (cap each
rollout's state gradient after every imagined step), ``--rf_rope_detach`` (no gradient through RigidFormer's rotary
positions), ``--wm_bptt_fp32`` (no bf16 autocast in the world model during the BPTT) and ``--wm_smooth_w`` (a
smoothness penalty on the world model's network while fitting). At their defaults the algorithms are unchanged
(test_algo_equivalence, test_supertrack_golden)."""

import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import DEV
from test_algo_equivalence import EXACT, _args, _fill

ST_ARGS = dict(st_reg=0.01, st_imag_noise=0.1, st_no_root_terms=False, st_no_task_regs=False, st_substeps=1, st_balance_w=1.0, st_com=False,
               st_quad_w=0.0, st_late_w=0.0, st_fail_w=0.0, st_fail_margin=1.0, st_wm_loss="l1")  # fmt: skip


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _algo(name, kin, motion, mdp, **kw):
    """A small-MLP algorithm whose world model's last layer is random (zero at initialization, the policy gradient
    would not pass through the network), with normalizers fit on the replay data as collect() fits them."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.state import normalize_xi, unflatten_state

    args = _args()
    vars(args).update({"algo": name, "wm_hidden": [64, 64], **ST_ARGS, **kw})  # kw may replace the small MLP
    torch.manual_seed(0)
    algo = ALGOS[name](args, kin, motion, mdp, DEV)
    if args.wm_backbone == "rigidformer":  # its read-out heads start at zero as well
        for m in (algo.wm.net.root_head, algo.wm.net.joint_head):
            nn.init.normal_(m.weight, std=0.05)
    else:
        nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
    buf = _fill(ReplayBuffer, motion, seed=7)
    w = buf.data
    S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
    algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1))))
    algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    algo.pi.obs_norm.update(mdp.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten()))
    algo.pi.goal_norm.update(motion.goal(w["f"].flatten(), args.goal_frames))
    return algo, buf


def test_cap_rollout_grads():
    from ndm_gym.algos.common import cap_rollout_grads

    torch.manual_seed(0)
    x, R = torch.randn(64, 3, requires_grad=True), torch.randn(64, 3, 3, requires_grad=True)
    y, Ry = cap_rollout_grads((x, R), 3.0)
    assert torch.equal(y, x) and torch.equal(Ry, R)
    g, gR = torch.randn(64, 3), torch.randn(64, 3, 3)
    g[5] *= 100.0  # an exploding rollout
    g[7] = 0.0  # a rollout without gradient: not counted in the median
    torch.autograd.backward((y, Ry), (g, gR))
    n = g.norm(dim=1)
    cap = 3.0 * n[n > 0].median()
    keep = n <= cap
    assert not keep[5] and keep[7]
    torch.testing.assert_close(x.grad[keep], g[keep], **EXACT)  # every rollout below the cap is untouched
    torch.testing.assert_close(x.grad[~keep].norm(dim=1), cap.expand(int((~keep).sum())))  # the others are scaled to it
    torch.testing.assert_close(x.grad[5] / x.grad[5].norm(), g[5] / n[5])  # in their own direction
    nR = gR.flatten(1).norm(dim=1)
    torch.testing.assert_close(R.grad[nR <= 3.0 * nR.median()], gR[nR <= 3.0 * nR.median()], **EXACT)  # per component, per row
    z = torch.randn(8, 3, requires_grad=True)
    (zy,) = cap_rollout_grads((z,), 2.0)
    zy.backward(torch.zeros(8, 3))
    torch.testing.assert_close(z.grad, torch.zeros(8, 3), **EXACT)  # no gradient at all: nothing to scale


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_bptt_clip_k_in_policy_update(eager, kin, motion, mdp, name):
    """A cap no rollout reaches changes nothing, bit for bit; a low cap changes the policy gradient, not the loss."""
    out, params = {}, {}
    for k in (0.0, 1e9, 0.5):
        algo, buf = _algo(name, kin, motion, mdp, bptt_clip_k=k)
        torch.manual_seed(2)
        out[k] = algo.policy_update(buf)
        params[k] = [p.detach().clone() for p in algo.pi.parameters()]
    for a, b in zip(out[0.0], out[1e9]):
        torch.testing.assert_close(a, b, equal_nan=True, **EXACT)
    for a, b in zip(params[0.0], params[1e9]):
        torch.testing.assert_close(a, b, **EXACT)
    torch.testing.assert_close(out[0.5][0], out[0.0][0], **EXACT)  # the loss: same forward pass
    assert out[0.5][3] < out[0.0][3]  # the policy-gradient norm before clipping


def test_decay_rollout_grads():
    """--bptt_decay: identity in the forward pass; through a chain of steps x_{t+1} = f(x_t) with a loss on every state,
    the gradient at the start weighs the loss d steps later by lam^d."""
    from ndm_gym.algos.common import decay_rollout_grads

    torch.manual_seed(0)
    A, x0, lam = 0.5 * torch.randn(3, 3), torch.randn(5, 3, requires_grad=True), 0.7
    f = lambda x: torch.tanh(x @ A)  # noqa: E731
    xs, x = [], x0
    for _ in range(4):
        (x,) = decay_rollout_grads((f(x),), lam)
        xs.append(x)
    torch.testing.assert_close(xs[0], f(x0), **EXACT)
    g = torch.autograd.grad(sum((x**2).sum() for x in xs), x0)[0]
    ref, x = torch.zeros_like(x0), x0
    for d in range(1, 5):
        x = f(x)
        ref += lam**d * torch.autograd.grad((x**2).sum(), x0, retain_graph=True)[0]
    torch.testing.assert_close(g, ref)


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_bptt_decay_in_policy_update(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--bptt_decay 1 (the default; also a run's arguments from before the flag) is the unchanged update, bit for bit;
    lam < 1 changes the policy gradient, not the loss; lam outside (0, 1] is refused."""
    out, params = {}, {}
    for lam in (None, 1.0, 0.5):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **({} if lam is None else {"bptt_decay": lam}))
        torch.manual_seed(2)
        out[lam] = algo.policy_update(buf)
        params[lam] = [p.detach().clone() for p in algo.pi.parameters()]
    for a, b in zip(out[None] + tuple(params[None]), out[1.0] + tuple(params[1.0])):
        torch.testing.assert_close(a, b, equal_nan=True, **EXACT)
    torch.testing.assert_close(out[0.5][0], out[None][0], **EXACT)  # the loss: same forward pass
    assert abs(float(out[0.5][3]) / float(out[None][3]) - 1) > 1e-3  # the policy-gradient norm before clipping
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="bptt_decay"):
            _algo(name, kin, synthetic_motion, synthetic_mdp, bptt_decay=bad)


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_pi_rollout_clip_in_policy_update(eager, kin, synthetic_motion, synthetic_mdp, monkeypatch, name):
    """--pi_rollout_clip: 0 (the default; also a run's arguments from before the flag) and a cap no rollout reaches are
    the unchanged update, bit for bit; with k = 1 the policy gradient is sum_b w_b G_b with every rollout's part G_b its
    own vector-Jacobian product through the policy (brute force) and w_b = min(1, k median / size_b), size_b the norm of
    the gradient that reached the policy's output over the horizon; the loss is unchanged; k < 0 is refused."""
    import importlib

    from ndm_gym.algos.common import set_lr

    mod = importlib.import_module(f"ndm_gym.algos.{name}")
    clip, seen = mod.clip_rollout_policy_grads, []

    def spy(pi, taps, k):
        seen.append([(o.detach().clone(), g.detach().clone(), t.grad.clone()) for o, g, t in taps])
        return clip(pi, taps, k)

    monkeypatch.setattr(mod, "clip_rollout_policy_grads", spy)
    out, grads = {}, {}
    for k in (None, 0.0, 1e9, 1.0):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **({} if k is None else {"pi_rollout_clip": k}))
        for o in algo.optimizers:
            set_lr(o, 0.0)
        params = list(algo.pi.parameters())
        raw = [None] * len(params)
        for i, p in enumerate(params):  # the gradient before policy_update clips it (as tools/grad_stats.py captures it)
            p.register_post_accumulate_grad_hook(lambda p, i=i: raw.__setitem__(i, p.grad.detach().clone()))
        torch.manual_seed(2)
        out[k] = algo.policy_update(buf)
        grads[k] = torch.cat([g.flatten() for g in raw])
    assert len(seen) == 2  # only k = 1e9 and k = 1 tap the policy's output
    for k in (0.0, 1e9):
        torch.testing.assert_close(grads[k], grads[None], **EXACT)
        for a, b in zip(out[k], out[None]):
            torch.testing.assert_close(a, b, equal_nan=True, **EXACT)
    torch.testing.assert_close(out[1.0][0], out[None][0], **EXACT)  # the loss: same forward pass
    obs, goal, A = (torch.stack(x) for x in zip(*seen[1]))  # (H, B, ...)
    size = A.square().sum((0, 2)).sqrt()
    w = torch.clamp(size[size > 0].median() / size.clamp_min(1e-30), max=1.0)
    assert 0 < int((w < 1).sum()) < len(w)
    expect = 0
    for b in range(A.shape[1]):
        with torch.enable_grad():
            parts = torch.autograd.grad(algo.pi(obs[:, b], goal[:, b]), params, A[:, b])
        expect = expect + w[b] * torch.cat([x.flatten() for x in parts])
    torch.testing.assert_close(grads[1.0], expect, rtol=1e-4, atol=1e-6 * float(expect.abs().max()))
    with pytest.raises(ValueError, match="pi_rollout_clip"):
        _algo(name, kin, synthetic_motion, synthetic_mdp, pi_rollout_clip=-1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled imagination step and its CUDA graphs run on CUDA")
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_pi_rollout_clip_compiled(synthetic_motion, name):
    """--pi_rollout_clip in the compiled imagination step, without and with CUDA graphs (--pi_cudagraphs), as training
    runs it: the zero tensors added to the policy's output receive its adjoint there too, so the clip changes the compiled
    update as it changes the eager one. The compiled step rounds differently from the eager one, and the BPTT amplifies
    it to a few percent of the policy gradient with or without the clip (here 3-5%; eager runs repeat bit for bit), so the
    clip's effect, the gradient with the clip minus the gradient without it (28-61% of the latter), is compared across
    modes by direction and size; the CUDA graphs replay the compiled step's kernels (fp32, no autocast)."""
    from conftest import synthetic_clips
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import set_lr
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin_c = g1.kinematics("cuda")
    motion_c = MotionLib([Clip(**c) for c in synthetic_clips()], kin_c, "cuda")
    mdp_c = MDP(kin_c, motion_c, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    cpu = _fill(ReplayBuffer, synthetic_motion, seed=7)
    buf = ReplayBuffer(cpu.T, cpu.N, "cuda")
    for k, v in cpu.data.items():
        buf.data[k].copy_(v)
    buf.ptr, buf.size = cpu.ptr, cpu.size
    grads = {}
    for mode, clip in itertools.product(("eager", "compiled", "cudagraphs"), (0.0, 1.0)):
        args = _args()
        vars(args).update({"algo": name, "wm_hidden": [64, 64], **ST_ARGS, "pi_rollout_clip": clip, "pi_cudagraphs": mode == "cudagraphs"})
        torch.manual_seed(0)
        algo = ALGOS[name](args, kin_c, motion_c, mdp_c, "cuda")
        nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
        if mode == "eager":  # the imagination step and losses as written
            for fn in ("st_imag_step", "st_track", "imag_step", "reward_term"):
                if hasattr(algo, "_" + fn):
                    setattr(algo, fn, getattr(algo, "_" + fn))
        w = buf.data
        S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
        algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1))))
        algo.wm.out_norm.update(algo.wm.targets(S0, S1))
        algo.pi.obs_norm.update(mdp_c.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten()))
        algo.pi.goal_norm.update(motion_c.goal(w["f"].flatten(), args.goal_frames))
        for o in algo.optimizers:
            set_lr(o, 0.0)
        params = list(algo.pi.parameters())
        raw = [None] * len(params)
        for i, p in enumerate(params):
            p.register_post_accumulate_grad_hook(lambda p, i=i: raw.__setitem__(i, p.grad.detach().clone()))
        for _ in range(2):  # the second update replays the captured graphs
            torch.manual_seed(2)
            algo.policy_update(buf)
        grads[mode, clip] = torch.cat([g.flatten() for g in raw])
    effect = {mode: grads[mode, 1.0] - grads[mode, 0.0] for mode in ("eager", "compiled", "cudagraphs")}
    assert effect["eager"].norm() > 0.2 * grads["eager", 0.0].norm()  # the clip acts on half of the rollouts
    for mode in ("compiled", "cudagraphs"):
        assert F.cosine_similarity(effect[mode], effect["eager"], dim=0) > 0.99
        assert (effect[mode] - effect["eager"]).norm() < 0.15 * effect["eager"].norm()
        assert F.cosine_similarity(grads[mode, 1.0], grads["eager", 1.0], dim=0) > 0.995
    for clip in (0.0, 1.0):
        g = grads["compiled", clip]
        torch.testing.assert_close(grads["cudagraphs", clip], g, rtol=1e-5, atol=1e-7 * float(g.abs().max()))


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_wm_bptt_fp32_switches_autocast_only_in_the_policy_update(eager, kin, motion, mdp, monkeypatch, name):
    import ndm_gym.worldmodels.accel as accel
    import ndm_gym.worldmodels.increment as increment

    for flag in (False, True):
        algo, buf = _algo(name, kin, motion, mdp, wm_bptt_fp32=flag)
        seen = []
        mod = accel if name == "supertrack" else increment
        run = mod.run_mlp

        def spy(net, x, amp=True):
            seen.append(amp)
            return run(net, x, amp=amp)

        monkeypatch.setattr(mod, "run_mlp", spy)
        torch.manual_seed(1)
        algo.wm_update(buf)
        assert seen and all(seen)  # world-model fitting keeps its precision
        seen.clear()
        torch.manual_seed(2)
        algo.policy_update(buf)
        assert len(seen) == algo.HZ and all(a == (not flag) for a in seen)
        assert algo.wm.amp is True  # restored for everything else
        monkeypatch.setattr(mod, "run_mlp", run)


def test_rope_detach_cuts_only_the_rotary_path_in_the_frozen_backward(kin):
    from ndm_gym.algos.common import set_requires_grad
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(0)
    nets = [RigidFormerNet(kin, root_out=6, joint_out=1, checkpoint="never", rope_detach=d, dim=24, layers=2, heads=2, registers=2)
            for d in (False, True)]  # fmt: skip
    for m in (nets[0].root_head, nets[0].joint_head):  # zero at initialization: the output would not depend on the trunk
        nn.init.normal_(m.weight, std=0.1)
    nets[1].load_state_dict(nets[0].state_dict())
    D = nets[0].in_dim
    x0 = torch.randn(4, D + 3 * nets[0].n_body)
    for frozen in (False, True):
        grads, ys = [], []
        for net in nets:
            set_requires_grad(net, not frozen)
            x = x0.clone().requires_grad_(True)
            y = net(x)
            y.square().sum().backward()
            ys.append(y.detach()), grads.append(x.grad)
        torch.testing.assert_close(ys[1], ys[0], **EXACT)  # the forward pass is unchanged
        if not frozen:  # world-model training: unchanged
            torch.testing.assert_close(grads[1], grads[0], **EXACT)
        else:  # the policy's BPTT: the positions reach the output through the token features only
            assert grads[0][:, D:].abs().max() > 0
            torch.testing.assert_close(grads[1][:, D:], torch.zeros_like(grads[1][:, D:]), **EXACT)
            torch.testing.assert_close(grads[1][:, :D], grads[0][:, :D])
    with pytest.raises(ValueError):
        RigidFormerNet(kin, root_out=6, joint_out=1, rope=False, rope_detach=True, **dict(dim=24, layers=1, heads=2))


def test_rope_detach_training_argument():
    from ndm_gym.worldmodels.rigidformer import rigidformer_kwargs

    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=96, rf_layers=2, rf_heads=3,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    assert rigidformer_kwargs(a)["rope_detach"] is False  # checkpoints from before the flag
    assert rigidformer_kwargs({**a, "rf_rope_detach": True})["rope_detach"] is True
    with pytest.raises(ValueError):
        rigidformer_kwargs({**a, "wm_backbone": "mlp", "rf_rope_detach": True})


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_wm_smoothness_off_by_default(eager, kin, motion, mdp, name):
    """--wm_smooth_w 0, and the arguments of runs from before the flag, leave the world-model update unchanged bit for
    bit, without a random draw; a positive weight changes the parameters' update but not the reported fit loss."""
    out, params, rng = {}, {}, {}
    for w in (None, 0.0, 1.0):
        algo, buf = _algo(name, kin, motion, mdp, **({} if w is None else dict(wm_smooth_w=w, wm_smooth_eps=1e-2, wm_smooth_dirs=2)))
        torch.manual_seed(1)
        out[w] = algo.wm_update(buf)
        params[w], rng[w] = [p.detach().clone() for p in algo.wm.parameters()], torch.get_rng_state()
    for a, b in zip(list(out[None]) + params[None], list(out[0.0]) + params[0.0]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
    assert torch.equal(rng[0.0], rng[None])
    torch.testing.assert_close(out[1.0][0], out[0.0][0], **EXACT)  # the fit loss: the same forward pass
    assert any(not torch.equal(a, b) for a, b in zip(params[1.0], params[0.0]))


@pytest.mark.parametrize("backbone", ["mlp", "rigidformer"])
def test_wm_smoothness_penalty(eager, kin, motion, mdp, backbone):
    """The penalty is the mean squared directional derivative of the network along its random unit directions (against
    central differences in float64), for both backbones; its gradient reaches the parameters, finite and nonzero."""
    import copy

    from ndm_gym.state import normalize_xi, unflatten_state

    kw = dict(wm_smooth_w=1.0, wm_smooth_eps=1e-2, wm_smooth_dirs=3)
    if backbone == "rigidformer":
        kw.update(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], rf_dim=24, rf_layers=1, rf_heads=2, rf_registers=2,
                  rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    algo, buf = _algo("supermimic", kin, motion, mdp, **kw)
    w = buf.sample_windows(16, 8)
    s0, a0, xi0 = w["s"][:, 0], w["a"][:, 0], w["xi"][:, 0]
    torch.manual_seed(5)
    pen = algo._wm_smoothness(s0, a0, xi0)
    torch.manual_seed(5)  # the same directions
    x = algo.wm.in_norm(algo.wm.inputs(unflatten_state(s0), a0, normalize_xi(xi0))).detach().double()
    u = torch.randn((3,) + x.shape, dtype=torch.float32).double()
    u = u / u.norm(dim=-1, keepdim=True)
    net, h = copy.deepcopy(algo.wm.net).double(), 1e-4  # RigidFormer computes its rotary terms in float32
    ju = torch.stack([(net(x + h * ui) - net(x - h * ui)) / (2 * h) for ui in u])
    torch.testing.assert_close(pen.double(), (ju**2).sum(-1).mean(), rtol=2e-2, atol=0)
    pen.backward()
    grads = [p.grad for p in algo.wm.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and any(bool(g.abs().max() > 0) for g in grads)


def test_wm_smoothness_step_floor(eager, kin, motion, mdp):
    """--wm_smooth_eps below 0.01 is refused while the penalty is on (TF32 rounding would distort it); off, it is unused."""
    _algo("supertrack", kin, motion, mdp, wm_smooth_w=1.0, wm_smooth_eps=0.01, wm_smooth_dirs=1)
    _algo("supertrack", kin, motion, mdp, wm_smooth_w=0.0, wm_smooth_eps=1e-3, wm_smooth_dirs=1)
    for eps in (5e-3, 1e-3, 0.0, -0.1):
        with pytest.raises(ValueError, match="wm_smooth_eps"):
            _algo("supertrack", kin, motion, mdp, wm_smooth_w=1.0, wm_smooth_eps=eps, wm_smooth_dirs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast of the world model runs on CUDA")
@pytest.mark.parametrize("eps", [0.1, 0.01])
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_wm_smoothness_runs_in_fp32(eager, synthetic_motion, monkeypatch, name, eps):
    """The finite difference runs without the world model's bf16 autocast, whose rounding is of the order of the step (on
    this random network it gave 1.6 times the exact value at eps 0.1, 47 times at 0.01): under train.py's settings
    (AMP_DTYPE bf16, TF32 matmuls) the penalty is the one computed without autocast, bit for bit; with strict fp32 matmuls
    it is the float64 finite difference to 1e-3; at the default step TF32's rounding keeps it within 1%."""
    import copy

    import ndm_gym.core.nets as nets
    from conftest import random_states, synthetic_clips
    from ndm_gym.algos import ALGOS
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin_c = g1.kinematics("cuda")
    motion_c = MotionLib([Clip(**c) for c in synthetic_clips()], kin_c, "cuda")
    mdp_c = MDP(kin_c, motion_c, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    args = _args()
    vars(args).update({"algo": name, **ST_ARGS, "wm_smooth_w": 1.0, "wm_smooth_eps": eps, "wm_smooth_dirs": 4})
    torch.manual_seed(0)
    algo = ALGOS[name](args, kin_c, motion_c, mdp_c, "cuda")
    nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
    f, S, xi = random_states(synthetic_motion, 512, seed=9)
    s0, a0, xi0 = flatten_state(S).cuda(), torch.randn(512, 29).cuda(), xi.cuda()
    S1 = tuple(x.cuda() for x in synthetic_motion.ref_state(f + 1))
    algo.update_wm_norms(unflatten_state(s0), S1, a0, xi0)
    pens = {}
    for tf32, amp in ((True, torch.bfloat16), (True, None), (False, torch.bfloat16)):
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
        monkeypatch.setattr(nets, "AMP_DTYPE", amp)
        torch.manual_seed(5)
        with torch.no_grad():
            pens[tf32, amp] = algo._wm_smoothness(s0, a0, xi0).double()
    assert torch.equal(pens[True, torch.bfloat16], pens[True, None])
    torch.manual_seed(5)  # the same directions, in float64
    x = algo.wm.in_norm(algo.wm.inputs(unflatten_state(s0), a0, normalize_xi(xi0))).double()
    u = torch.randn((4,) + x.shape, device="cuda").double()
    u = u / u.norm(dim=-1, keepdim=True)
    net = copy.deepcopy(algo.wm.net).double()
    with torch.no_grad():
        y = net(torch.cat((x.unsqueeze(0), x + eps * u)).flatten(0, 1)).unflatten(0, (5, -1))
    exact = ((y[1:] - y[:1]) ** 2).sum(-1).mean() / eps**2
    torch.testing.assert_close(pens[False, torch.bfloat16], exact, rtol=1e-3, atol=0)
    if eps >= 0.1:
        torch.testing.assert_close(pens[True, None], exact, rtol=1e-2, atol=0)
