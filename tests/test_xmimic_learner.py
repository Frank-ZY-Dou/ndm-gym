"""XMimic's learner restructuring as options of ndm_gym.train (algos.xmimic, docs/algorithms/xmimic.md), each against the default
update on CPU in eager mode: --pi_encode_once in SuperMimic's imagination, --imag_hoist, --pi_deferred_grads and
--wm_deferred_grads (core.nets.run_mlp_deferred, deferred_grads), --muon_batched (core.muon) and --wm_presample
(ReplayBuffer.sample_windows_multi). The tolerances: exact where the option only moves or batches the same operations on the
CPU; 1e-5 or 1e-4 of a tensor's largest entry where sums are formed in another order in fp32. Under --amp bf16 the
deferred products' operands are bf16 and backpropagation rounds each step's product to bf16: the ``cpu_autocast`` tests run
that branch with the CPU's bf16 autocast in the place of CUDA's (2e-2; here 4e-3 at most). The tests at the end need a
GPU: the same under CUDA's autocast, the updates compiled, and (marked slow, with Isaac Lab) a training run that captures
them with --update_graphs."""

import argparse
import csv
import importlib.util
import math
import types

import pytest
import torch
import torch.nn as nn

from conftest import DEV, random_states, synthetic_clips
from test_algo_equivalence import EXACT
from test_bptt_tricks import ST_ARGS, _algo
from test_xmimic import _parse

H, B = 32, 64  # the imagined steps and rollouts of test_bptt_tricks._algo's policy update
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None
POLICY = dict(pi_encode_once=True, imag_hoist=True, pi_deferred_grads=True)  # the policy update's options together
RF = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], rf_dim=24, rf_layers=1, rf_heads=2, rf_registers=2, rf_no_tree_bias=False,
          rf_no_rope=False)  # fmt: skip


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


@pytest.fixture()
def cpu_autocast(monkeypatch):
    """--amp bf16 on the CPU: the CPU's bf16 autocast in the place of CUDA's in ``run_mlp`` and in the deferred MLP, whose
    test of where the autocast acts (``nets._autocasts``) then holds for CPU tensors."""
    from ndm_gym.core import nets

    real = torch.autocast
    monkeypatch.setattr(torch, "autocast", lambda device_type, dtype=None, **kw: real("cpu", dtype=dtype, **kw))
    monkeypatch.setattr(nets, "_autocasts", lambda device: True)
    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)


def _close(b, a, rel: float, msg=None):
    """b equals a up to ``rel`` times a's largest magnitude, entry by entry (a gradient's small entries round as its large
    ones do)."""
    torch.testing.assert_close(b, a, rtol=rel, atol=rel * float(a.abs().max()), msg=msg)


def _policy_update(algo, buf, graphable: bool, seed: int = 3):
    """One policy update: what its forward pass gives (loss, imagined reward, imagined survival), then its backward pass
    (the gradient norm before clipping and the policy's gradients after it)."""
    torch.manual_seed(seed)
    res = algo.policy_update_graphable(buf) if graphable else algo.policy_update(buf)
    return [res[0], res[2], res[4]], [res[3]] + [p.grad.clone() for p in algo.pi.parameters()]


class _SameNoise:
    """The same imagination noise for the default's draw per step and for --imag_hoist's one draw: observation noise
    (H, B, obs) and SuperTrack's exploration noise (H, B, J), fixed. ``per_step`` (for the update of an algorithm that
    draws per step) makes ``mdp.add_noise`` add the next step's rows to a step's observations (the start histories,
    (B, hist, obs), keep their own draw, in both) and ``torch.randn_like`` return the next step's exploration noise;
    ``one_draw`` makes the hoisting algorithm's two draws return them whole."""

    def __init__(self, monkeypatch, mdp, seed: int = 5):
        from ndm_gym.tasks.tracking import OBS_DIM

        g = torch.Generator().manual_seed(seed)
        self.obs = (2 * torch.rand(H, B, OBS_DIM, generator=g) - 1) * mdp.obs_noise
        self.act = torch.randn(H, B, 29, generator=g)
        self.monkeypatch, self.mdp = monkeypatch, mdp

    def per_step(self):
        rows, acts, add = iter(self.obs), iter(self.act), self.mdp.add_noise
        self.monkeypatch.setattr(self.mdp, "add_noise", lambda o: add(o) if o.dim() == 3 else o + next(rows))
        self.monkeypatch.setattr(torch, "randn_like", lambda x: next(acts))

    def one_draw(self, algo):
        algo.imag_noise = lambda o_hist: self.obs
        algo.imag_action_noise = lambda n: algo.args.st_imag_noise * self.act


# ---------------------------------------------------------------- arguments
def test_learner_options_are_off_by_default_and_parse_together():
    from ndm_gym.algos.xmimic import LEARNER, check_args

    a = _parse([])
    assert not any(getattr(a, n) for n in LEARNER)
    check_args(a)
    everything = ["--pi_encode_once", "--imag_hoist", "--pi_deferred_grads", "--wm_deferred_grads", "--deferred_bf16", "--muon_batched",
                  "--wm_presample", "--pi_opt", "muon", "--wm_opt", "muon"]  # fmt: skip
    for algo in ("supermimic", "supertrack"):
        a = _parse(["--algo", algo] + everything)
        assert all(getattr(a, n) for n in LEARNER)
        check_args(a)
    check_args(_parse(["--pi_encode_once"]))  # SuperMimic's imagination: no longer SuperTrack's alone


@pytest.mark.parametrize(
    "argv",
    [["--algo", "other", "--wm_presample"], ["--algo", "other", "--wm_deferred_grads"], ["--wm_backbone", "rigidformer", "--wm_deferred_grads"],
     ["--wm_backbone", "rigidformer", "--wm_presample"], ["--paired", "--wm_presample"], ["--paired", "--pi_encode_once"],
     ["--paired", "--muon_batched", "--pi_opt", "muon"], ["--imag_hoist"], ["--pi_encode_once", "--pi_deferred_grads"],
     ["--pi_encode_once", "--pi_rollout_clip", "2"], ["--deferred_bf16"], ["--wm_deferred_grads", "--deferred_bf16", "--amp", "none"],
     ["--muon_batched"], ["--wm_presample", "--wm_sample", "prioritized"], ["--wm_presample", "--wm_fail_windows", "0.25"],
     ["--pi_encode_once", "--imag_hoist", "--pi_cudagraphs"], ["--wm_deferred_grads", "--wm_cudagraphs"], ["--wm_presample", "--wm_cudagraphs"]],
)  # fmt: skip
def test_learner_options_are_refused_where_they_do_not_apply(argv):
    """Not with another algorithm, the RigidFormer world model or --paired; each option only with what it builds on; not with the
    compiled parts' CUDA graphs (--pi_cudagraphs, --wm_cudagraphs), under which they are not checked."""
    from ndm_gym.algos.xmimic import check_args

    with pytest.raises(ValueError):
        check_args(_parse(argv))
    check_args(_parse(["--pi_encode_once", "--pi_cudagraphs"]))  # as SuperTrack's before
    check_args(_parse(["--pi_encode_once", "--wm_backbone", "rigidformer"]))  # the policy's encoder: any world model


def test_train_parses_the_learner_options(monkeypatch):
    """ndm_gym.train's parser: the options with --algo blinkmimic (SuperMimic with the recipe), refused with another algorithm;
    --pi_encode_once with SuperMimic."""
    import sys

    pytest.importorskip("isaaclab")
    from ndm_gym import train

    opts = ["--pi_encode_once", "--imag_hoist", "--pi_deferred_grads", "--wm_deferred_grads", "--muon_batched", "--wm_presample"]
    monkeypatch.setattr(sys, "argv", ["ndm_gym.train", "--algo", "blinkmimic"] + opts)
    a = train.parse_args()
    assert a.algo == "supermimic" and a.imag_hoist and a.pi_deferred_grads and a.wm_deferred_grads and a.muon_batched and a.wm_presample
    for bad in (["--algo", "other", "--pi_encode_once"], ["--algo", "other", "--wm_presample"], ["--algo", "supermimic", "--imag_hoist"]):
        monkeypatch.setattr(sys, "argv", ["ndm_gym.train"] + bad)
        with pytest.raises(SystemExit):
            train.parse_args()


# ---------------------------------------------------------------- --pi_encode_once in SuperMimic's imagination
@pytest.mark.parametrize("graphable", [False, True])
@pytest.mark.parametrize("extra", [{}, {"sm_residual": True}, RF], ids=["setpoints", "residual", "rigidformer"])
def test_encode_once_is_supermimics_policy_update(eager, kin, synthetic_motion, synthetic_mdp, graphable, extra):
    """SuperMimic --pi_encode_once: the goals of the rollout's frames and the encoder's products over all steps at once, the
    steps the decoder. The same random stream (a noise draw per step, as the default). The policy update, default and
    graphable, gives the default's loss, imagined reward, gradient norm, survival and policy gradient up to float rounding
    (fp32: the encoder's products over H * B rows instead of B; here 3e-6 of each gradient's largest entry at most, the
    tolerance 1e-4), with either world model: the option is the policy's."""
    kw = dict(fused_adam=True, no_sync=True) if graphable else {}
    out = {}
    for once in (False, True):
        algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, pi_encode_once=once, **extra, **kw)
        assert algo.encode_once == once and not algo.hoist
        out[once] = _policy_update(algo, buf, graphable)
    for a, b in zip(out[False][0] + out[False][1], out[True][0] + out[True][1]):
        _close(b, a, 1e-4)


def test_encode_once_goals_and_refusals(eager, kin, synthetic_motion, synthetic_mdp):
    """The rollout's frames, their goals and the steps' latents, each step's that of its own frame; what the option and the
    ones that build on it refuse."""
    algo, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, pi_encode_once=True)
    f = synthetic_motion.clip_start[:3] + 50
    frames, G, Z = algo._encode_goals(f)
    assert frames.shape == (H + 1, 3) and torch.equal(frames[:, 0], f[0] + torch.arange(H + 1))
    torch.testing.assert_close(G, torch.stack([synthetic_motion.goal(f + t, 10) for t in range(H + 1)]), **EXACT)  # a goal per frame
    _close(Z.detach(), torch.stack([algo.pi.encode(synthetic_motion.goal(f + t, 10)) for t in range(H)]).detach(), 1e-5)  # step t: frame f + t's
    with pytest.raises(ValueError, match="pi_rollout_clip"):
        _algo("supermimic", kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, pi_rollout_clip=2.0)
    for name in ("supermimic", "supertrack"):
        with pytest.raises(ValueError, match="imag_hoist"):
            _algo(name, kin, synthetic_motion, synthetic_mdp, imag_hoist=True)
        with pytest.raises(ValueError, match="imag_hoist"):
            _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, pi_deferred_grads=True)
        with pytest.raises(ValueError, match="MLP world model"):
            _algo(name, kin, synthetic_motion, synthetic_mdp, wm_deferred_grads=True, **RF)
        with pytest.raises(ValueError, match="MLP world model"):
            _algo(name, kin, synthetic_motion, synthetic_mdp, wm_opt="muon", muon_batched=True, **RF)


# ---------------------------------------------------------------- --imag_hoist
@pytest.mark.parametrize("graphable", [False, True])
@pytest.mark.parametrize("name,extra", [("supermimic", {}), ("supermimic", {"sm_residual": True, "bptt_decay": 0.95}), ("supertrack", {}),
                                        ("supertrack", {"st_fail_w": 5.0, "bptt_clip_k": 3.0})])  # fmt: skip
def test_imag_hoist_given_the_same_noise(eager, monkeypatch, kin, synthetic_motion, synthetic_mdp, graphable, name, extra):
    """--imag_hoist against --pi_encode_once alone, with the same noise in both (the option draws each kind of noise once
    for all steps instead of once per step, another random stream; _SameNoise): the forward pass is the same arithmetic on
    the same values (the reference's terms read before the rollout; an observation's normalization does not depend on the
    step that computes it), so the loss and the other outputs are exact. The policy gradient is equal up to float rounding
    (fp32: an observation's normalization is backpropagated once, the sum over the histories that hold it formed first;
    here 2e-7 of each gradient's largest entry at most, the tolerance 1e-5)."""
    kw = dict(fused_adam=True, no_sync=True) if graphable else {}
    noise = _SameNoise(monkeypatch, synthetic_mdp)
    algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, **extra, **kw)
    hoisted, buf_h = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, imag_hoist=True, **extra, **kw)
    assert hoisted.hoist and not hoisted.pi_deferred
    noise.per_step()
    base = _policy_update(algo, buf, graphable)
    noise.one_draw(hoisted)
    out = _policy_update(hoisted, buf_h, graphable)
    for a, b in zip(base[0], out[0]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)  # SuperTrack's imagined reward is nan
    for a, b in zip(base[1], out[1]):
        _close(b, a, 1e-5)


def test_imag_hoist_draws_the_steps_noise_at_once(eager, kin, synthetic_motion, synthetic_mdp):
    """The one draw's noise is ``MDP.add_noise``'s: uniform within the sensor noise's half-widths, none where they are 0."""
    algo, _ = _algo("supertrack", kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, imag_hoist=True)
    torch.manual_seed(0)
    nz = algo.imag_noise(torch.zeros(B, 10, 99))
    assert nz.shape == (H, B, 99) and bool((nz.abs() <= synthetic_mdp.obs_noise).all())
    assert bool((nz[..., synthetic_mdp.obs_noise == 0] == 0).all()) and float(nz[..., :29].abs().max()) > 0.009
    an = algo.imag_action_noise(B)
    assert an.shape == (H, B, 29) and abs(float(an.std()) - algo.args.st_imag_noise) < 0.01


# ---------------------------------------------------------------- deferred weight gradients
@pytest.mark.parametrize("layer_norm", [False, True])
def test_deferred_mlp_is_the_mlp_and_its_gradients(layer_norm):
    """run_mlp_deferred gives run_mlp's values bit for bit (the same products, with zeros added), and deferred_grads the
    Linear layers' gradients of backpropagation up to float rounding (fp32: one product over all applications' rows instead
    of a sum of one per application; 1e-5 of each gradient's largest entry), over several applications of the MLP in a
    chain; LayerNorm's parameters keep backpropagation's gradients. A gradient that is there already is added to."""
    from ndm_gym.core import nets

    torch.manual_seed(0)
    net = nets.mlp(12, [32, 24], 12, act=nn.SiLU, layer_norm=layer_norm)
    x0, T = torch.randn(16, 12), 5
    x, outs = x0, []
    for _ in range(T):  # a recurrence, as an imagined rollout
        x = nets.run_mlp(net, x)
        outs.append(x)
    torch.stack(outs).square().sum().backward()
    ref = {n: p.grad.clone() for n, p in net.named_parameters()}
    net.zero_grad(set_to_none=True)
    taps = nets.mlp_taps(net, (T, 16), DEV)
    assert [t.shape for t in taps] == [(T, 16, 32), (T, 16, 24), (T, 16, 12)] and all(t.dtype == torch.float32 and t.requires_grad for t in taps)
    steps = [t.unbind(0) for t in taps]
    x, outs2, ins = x0, [], []
    for j in range(T):
        x, i = nets.run_mlp_deferred(net, x, [t[j] for t in steps])
        outs2.append(x), ins.append(i)
    for a, b in zip(outs, outs2):
        torch.testing.assert_close(b, a, **EXACT)
    torch.stack(outs2).square().sum().backward()
    lin = nets.linears(net)
    assert all(m.weight.grad is None and m.bias.grad is None for m in lin)  # nothing reached them in the backward pass
    stacked = [torch.stack([i[k] for i in ins]) for k in range(len(lin))]
    nets.deferred_grads(net, stacked, taps)
    for n, p in net.named_parameters():
        _close(p.grad, ref[n], 1e-5)
    nets.deferred_grads(net, stacked, taps)  # onto the gradients that are there
    for m in lin:
        _close(m.weight.grad, 2 * ref[[n for n, p in net.named_parameters() if p is m.weight][0]], 1e-5)
    with pytest.raises(RuntimeError, match="pre-activation"):
        nets.deferred_grads(net, stacked, nets.mlp_taps(net, (T, 16), DEV))


def test_deferred_products_of_bf16_operands():
    """Under the bf16 autocast the taps' gradients and the stored inputs are bf16. deferred_grads multiplies them in fp32,
    which holds bf16 exactly: the fp32 sum of the exact operands. --deferred_bf16: the bf16 product, its result rounded to
    bf16 (within 2^-7 of the fp32 sum's largest entry here)."""
    from ndm_gym.core import nets

    torch.manual_seed(0)
    net = nets.mlp(6, [], 4)
    x, dy = torch.randn(3, 8, 6).bfloat16(), torch.randn(3, 8, 4).bfloat16()
    tap = types.SimpleNamespace(grad=dy, flatten=None)
    want_w, want_b = dy.float().flatten(0, 1).T @ x.float().flatten(0, 1), dy.float().sum((0, 1))
    nets.deferred_grads(net, [x], [tap])
    assert net[0].weight.grad.dtype == torch.float32
    torch.testing.assert_close(net[0].weight.grad, want_w, **EXACT)
    torch.testing.assert_close(net[0].bias.grad, want_b, **EXACT)
    net.zero_grad(set_to_none=True)
    nets.deferred_grads(net, [x], [tap], bf16_product=True)
    assert net[0].weight.grad.dtype == torch.float32
    torch.testing.assert_close(net[0].weight.grad, (dy.flatten(0, 1).T @ x.flatten(0, 1)).float(), **EXACT)
    _close(net[0].weight.grad, want_w, 2.0**-7)
    torch.testing.assert_close(net[0].bias.grad, want_b, **EXACT)  # the bias's sum stays in fp32


def test_deferred_taps_follow_the_autocast(monkeypatch):
    """The taps have the pre-activations' dtype: the autocast's where run_mlp applies it (AMP_DTYPE set, on a CUDA device,
    amp not turned off), fp32 otherwise. On the CPU, where run_mlp's autocast does not act, the deferred MLP stays in fp32
    with AMP_DTYPE set, as run_mlp does."""
    from ndm_gym.core import nets

    net = nets.mlp(6, [8], 4)
    assert nets.deferred_dtype("cuda") == torch.float32 and nets.deferred_dtype("cpu") == torch.float32
    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    assert nets.deferred_dtype("cuda") == nets.deferred_dtype("cuda:1") == nets.deferred_dtype(torch.device("cuda", 0)) == torch.bfloat16
    assert nets.deferred_dtype("cuda", amp=False) == torch.float32 and nets.deferred_dtype("cpu") == torch.float32
    taps = nets.mlp_taps(net, (2, 3), DEV)
    assert all(t.dtype == torch.float32 for t in taps)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(0))
    y, ins = nets.run_mlp_deferred(net, x, [t[0] for t in taps])
    assert y.dtype == torch.float32 and all(i.dtype == torch.float32 for i in ins)
    torch.testing.assert_close(y, nets.run_mlp(net, x), **EXACT)


@pytest.mark.parametrize("graphable", [False, True])
@pytest.mark.parametrize("name,extra", [("supermimic", {}), ("supermimic", {"sm_residual": True, "bptt_decay": 0.95}), ("supertrack", {"st_fail_w": 5.0})])
def test_pi_deferred_grads_is_the_policy_gradient(eager, kin, synthetic_motion, synthetic_mdp, graphable, name, extra):
    """--pi_deferred_grads against --imag_hoist alone (the same random stream): the forward pass is the same (exact), the
    decoder's weight and bias gradients are one product per layer over the H * B rows instead of H products summed (fp32
    rounding; here 6e-7 of each gradient's largest entry at most, the tolerance 1e-5), the encoder's gradients
    backpropagation's."""
    kw = dict(fused_adam=True, no_sync=True) if graphable else {}
    out = {}
    for deferred in (False, True):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, imag_hoist=True, pi_deferred_grads=deferred, **extra, **kw)
        assert algo.pi_deferred == deferred
        out[deferred] = _policy_update(algo, buf, graphable)
    for a, b in zip(out[False][0], out[True][0]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
    for a, b in zip(out[False][1], out[True][1]):
        _close(b, a, 1e-5)
    with pytest.raises(ValueError, match="imag_hoist"):
        _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, pi_deferred_grads=True)


@pytest.mark.parametrize("graphable", [False, True])
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_policy_options_together_are_the_default_update(eager, monkeypatch, kin, synthetic_motion, synthetic_mdp, graphable, name):
    """--pi_encode_once --imag_hoist --pi_deferred_grads against the default policy update with the same noise: the loss,
    gradient norm and policy gradient up to float rounding (fp32, the three options' reorderings through 32 imagined steps;
    here 3e-6 of each gradient's largest entry at most, the tolerance 1e-4)."""
    kw = dict(fused_adam=True, no_sync=True) if graphable else {}
    noise = _SameNoise(monkeypatch, synthetic_mdp)
    algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **kw)
    fast, buf_f = _algo(name, kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, imag_hoist=True, pi_deferred_grads=True, **kw)
    noise.per_step()
    base = _policy_update(algo, buf, graphable)
    noise.one_draw(fast)
    out = _policy_update(fast, buf_f, graphable)
    for a, b in zip(base[0] + base[1], out[0] + out[1]):
        if torch.isnan(a).any():  # SuperTrack's imagined reward and, without --st_fail_w, its survival
            torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
        else:
            _close(b, a, 1e-4)


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("name,extra", [("supermimic", {}), ("supermimic", {"wm_structured": True, "wm_mirror": 0.5}), ("supertrack", {}),
                                        ("supermimic", {"wm_smooth_w": 0.1, "wm_smooth_eps": 0.1, "wm_smooth_dirs": 2}),
                                        ("supermimic", {"wm_input_noise": 0.05})])  # fmt: skip
def test_wm_deferred_grads_is_the_fit(eager, kin, synthetic_motion, synthetic_mdp, name, extra, probe):
    """--wm_deferred_grads: the world model's k-step fit with its Linear layers' gradients formed after the backward pass,
    for SuperMimic's increment model, its semi-implicit one and SuperTrack's acceleration model. The losses are the default's
    bit for bit (the same forward pass and random stream), the gradients up to float rounding (fp32: one product per layer
    over the window's k * B rows; here 7e-7 of each gradient's largest entry at most, the tolerance 1e-5), LayerNorm's
    included; with --wm_smooth_w the penalty's gradients, which backpropagation forms, are added to."""
    out = {}
    for deferred in (False, True):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, wm_deferred_grads=deferred, **extra)
        assert algo.wm_deferred == deferred
        torch.manual_seed(4)
        res = algo.wm_update(buf, probe=probe)
        out[deferred] = list(res), [p.grad.clone() for p in algo.wm.parameters()]
    for a, b in zip(out[False][0], out[True][0]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
    for a, b in zip(out[False][1], out[True][1]):
        _close(b, a, 1e-5)


def test_world_model_step_with_taps_is_the_step(kin, synthetic_motion):
    """The MLP world models' ``step`` with taps returns the default step's state bit for bit and the Linear layers' inputs."""
    from ndm_gym.core import nets
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel
    from ndm_gym.worldmodels.semi_implicit import SemiImplicitIncrementWorldModel

    f, S, xi = random_states(synthetic_motion, 8, seed=1)
    a = torch.randn(8, 29, generator=torch.Generator().manual_seed(2))
    for cls in (IncrementWorldModel, SemiImplicitIncrementWorldModel, AccelWorldModel):
        torch.manual_seed(0)
        wm = cls(hidden=(32, 32))
        nn.init.normal_(wm.net[-1].weight, std=0.05)
        S1 = wm.step(S, a, normalize_xi(xi))
        S2, ins = wm.step(S, a, normalize_xi(xi), taps=nets.mlp_taps(wm.net, (8,), DEV))
        assert len(ins) == 3 and ins[0].shape == (8, wm.in_dim)
        for x, y in zip(S1, S2):
            torch.testing.assert_close(y, x, **EXACT)


# ---------------------------------------------------------------- --muon_batched
def test_batched_newton_schulz_is_the_single_iteration():
    """newton_schulz_batched on a stack of matrices gives torch.optim.Muon's iteration of each, wide, tall and square, bit
    for bit on the CPU, whose batched bf16 product is the single product per matrix (on a GPU the two may round
    differently)."""
    from torch.optim._muon import _zeropower_via_newtonschulz

    from ndm_gym.core.muon import newton_schulz_batched

    coef = (3.4445, -4.7750, 2.0315)
    for shape in ((3, 32, 32), (2, 24, 48), (2, 48, 24)):
        G = torch.randn(shape, generator=torch.Generator().manual_seed(sum(shape)))
        out = newton_schulz_batched(G, coef, 5, 1e-7)
        assert out.dtype == torch.bfloat16 and out.shape == shape
        torch.testing.assert_close(out, torch.stack([_zeropower_via_newtonschulz(g, coef, 5, 1e-7) for g in G]), **EXACT)


def _muon_net():
    from ndm_gym.core.nets import mlp

    torch.manual_seed(0)
    # Muon's matrices: (32, 48) in both MLPs, (32, 32) and (16, 32) in the encoder alone, (24, 32) and (16, 24) in the decoder alone
    net = nn.ModuleDict({"enc": mlp(20, [48, 32, 32, 16], 8), "dec": mlp(30, [48, 32, 24, 16], 5)})
    return net, [0.01 * torch.randn(p.shape, generator=torch.Generator().manual_seed(i)) for i, p in enumerate(net.parameters())]


def _muon_steps(net, grads, oks=(None, None, None), **kw):
    import copy

    from ndm_gym.core.muon import MuonAdam

    m = copy.deepcopy(net)
    opt = MuonAdam(m, [m["enc"], m["dec"]], lr=1e-3, **kw)
    for step, ok in enumerate(oks):
        for p, g in zip(m.parameters(), grads):
            p.grad = g.clone() * (step + 1)
        opt.step() if ok is None else opt.step(ok=torch.tensor(ok))
    return [p.detach().clone() for p in m.parameters()], opt


def test_muon_batched_is_the_muon_step(monkeypatch):
    """MuonAdam(batched=True): the momenta first, then the matrices of each shape orthogonalized together. Against the
    same step one matrix at a time (the --update_graphs step, capturable), bit for bit on the CPU, also with steps gated
    off on the device; a matrix alone in its shape keeps the single iteration. Without --update_graphs the batched step
    replaces torch.optim.Muon's own: equal up to float rounding (1e-6: its in-place add scales the update inside the add)."""
    from ndm_gym.core import muon

    net, grads = _muon_net()
    ref, _ = _muon_steps(net, grads, capturable=True)
    sizes = []
    real = muon.newton_schulz_batched
    monkeypatch.setattr(muon, "newton_schulz_batched", lambda g, *a: sizes.append(tuple(g.shape)) or real(g, *a))
    out, opt = _muon_steps(net, grads, capturable=True, batched=True)
    assert opt.batched and sizes == [(2, 32, 48)] * 3  # the shape two matrices share, at each of the 3 steps; the four others alone
    for a, b in zip(ref, out):
        torch.testing.assert_close(b, a, **EXACT)
    gated = (True, False, True)
    for a, b in zip(_muon_steps(net, grads, gated, capturable=True)[0], _muon_steps(net, grads, gated, capturable=True, batched=True)[0]):
        torch.testing.assert_close(b, a, **EXACT)
    for a, b in zip(_muon_steps(net, grads, fused=True)[0], _muon_steps(net, grads, fused=True, batched=True)[0]):
        _close(b, a, 1e-6)
    for a, b in zip(_muon_steps(net, grads, gated, capturable=True)[0], _muon_steps(net, grads, gated, fused=True, batched=True)[0]):
        _close(b, a, 1e-6)  # gated without --update_graphs: the batched step's own gate


def test_muon_batched_reaches_the_trainers_optimizers(eager, kin, synthetic_motion, synthetic_mdp):
    algo, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, pi_opt="muon", wm_opt="muon", wm_hidden=[64, 64, 64], muon_batched=True)
    assert algo.opt_pi.batched and algo.opt_wm.batched
    algo, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, pi_opt="muon", wm_opt="muon", wm_hidden=[64, 64, 64])
    assert not algo.opt_pi.batched and not algo.opt_wm.batched


# ---------------------------------------------------------------- --wm_presample
@pytest.mark.parametrize("device_size", [False, True])
def test_presampled_windows_are_the_windows_one_batch_at_a_time(synthetic_motion, device_size):
    """sample_windows_multi(n, B, k): the windows and masks of n calls of sample_windows(B, k), bit for bit (the same draws
    in the same order, one read), also with the device-side fill level of --graph_early; their mirror images in one pass
    are the mirror images batch by batch."""
    from ndm_gym.algos.blinkmimic import Mirror
    from ndm_gym.buffer import ReplayBuffer
    from test_algo_equivalence import _fill

    def make():
        buf = _fill(lambda *a: ReplayBuffer(*a, priorities=False), synthetic_motion, seed=7)
        if device_size:
            buf.use_device_ptr(device_size=True)
        return buf

    buf, mirror = make(), Mirror(DEV)
    torch.manual_seed(1)
    one = [buf.sample_windows(16, 8) for _ in range(3)]
    after = torch.rand(4)
    torch.manual_seed(1)
    multi = buf.sample_windows_multi(3, 16, 8)
    torch.testing.assert_close(torch.rand(4), after, **EXACT)  # the generator is where three calls leave it
    assert set(multi) == set(one[0]) and multi["mask"].shape == (3, 16, 8)
    for i, w in enumerate(one):
        for key in w:
            torch.testing.assert_close(multi[key][i], w[key], **EXACT)
    mirrored = mirror.windows(multi, 8, stacked=True)
    for i, w in enumerate(one):
        m = mirror.windows(w, 8)
        for key in w:
            torch.testing.assert_close(mirrored[key][i], m[key], **EXACT)
    assert not torch.equal(mirrored["s"][:, :8], multi["s"][:, :8]) and torch.equal(mirrored["s"][:, 8:], multi["s"][:, 8:])


@pytest.mark.parametrize("name,extra", [("supermimic", {"wm_mirror": 0.5}), ("supertrack", {}), ("supermimic", {"wm_deferred_grads": True})])
def test_wm_presample_is_the_iterations_updates(eager, kin, synthetic_motion, synthetic_mdp, name, extra):
    """--wm_presample: the iteration's --wm_batches world-model updates on windows drawn before the first give the updates
    one at a time bit for bit (the fit itself draws nothing here: the same windows), the returned loss and k-step error and
    the probe's errors of the last update included."""
    out = {}
    for presample in (False, True):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, wm_batches=3, wm_presample=presample, **extra)
        torch.manual_seed(6)
        if presample:
            res = algo.wm_updates_presampled(buf, probe=True)
        else:
            for b in range(3):
                res = algo.wm_update(buf, probe=b == 2)
        out[presample] = list(res) + [p.detach().clone() for p in algo.wm.parameters()] + [torch.rand(3)]
    assert len(out[True]) == len(out[False]) and len(res) == 4
    for a, b in zip(out[False], out[True]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
    for bad in ({"wm_sample": "prioritized"}, {"wm_fail_windows": 0.25}):  # what the one pass does not draw
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, wm_batches=3, **bad)
        with pytest.raises(ValueError, match="wm_presample"):
            algo.wm_updates_presampled(buf)


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_graphable_updates_with_the_options_are_the_default_ones(eager, monkeypatch, kin, synthetic_motion, synthetic_mdp, name):
    """With every learner option on (and Muon, the policy average): the presampled world-model updates and
    policy_update_graphable, run eagerly, give the parameters of wm_update one at a time and of policy_update bit for bit,
    as test_systems' test of the default paths (the default's cumprod as the graphable's loop there too): the options'
    code is the same in both, the starts, the gated optimizer steps and the average's step as without them."""
    from test_systems import _cumprod_loop, _params, _tensors

    monkeypatch.setattr(torch, "cumprod", _cumprod_loop)
    kw = dict(fused_adam=True, no_sync=True, pi_ema=0.995, pi_opt="muon", wm_opt="muon", wm_hidden=[64, 64, 64], wm_batches=2, wm_mirror=0.5,
              wm_deferred_grads=True, muon_batched=True, wm_presample=True, **POLICY)  # fmt: skip
    if name == "supertrack":
        kw["st_fail_w"] = 5.0  # its survival products
    (a0, b0), (a1, b1) = [_algo(name, kin, synthetic_motion, synthetic_mdp, **kw) for _ in range(2)]
    assert a1.opt_pi.batched and a1.opt_wm.batched and a1.hoist and a1.pi_deferred and a1.wm_deferred
    for step in range(2):  # the second policy update steps the policy average (the first copies it)
        torch.manual_seed(10 + step)
        for _ in range(2):
            r0 = a0.wm_update(b0)
        torch.manual_seed(10 + step)
        r1 = a1.wm_updates_presampled(b1)
        torch.testing.assert_close(r1[0], r0[0], **EXACT)
        torch.manual_seed(20 + step)
        p0 = a0.policy_update(b0)
        a0.update_pi_avg()
        torch.manual_seed(20 + step)
        n_avg0 = int(a1.pi_avg.n_averaged)
        p1 = a1.policy_update_graphable(b1)
        if int(a1.pi_avg.n_averaged) == n_avg0:
            a1.pi_avg.n_averaged += 1  # as train.py does: the captured average's step does not advance the host-side counter
        torch.testing.assert_close(p1[0], p0[0], **EXACT)
        assert int(a1.nonfinite) == 0
    for x, y in zip(_params(a0) + _tensors(a0.pi_avg.module), _params(a1) + _tensors(a1.pi_avg.module)):
        torch.testing.assert_close(y, x, **EXACT)


def test_update_graphs_captures_the_presampled_updates():
    """train.py --update_graphs with --wm_presample: the world-model graph is the iteration's updates (one call), not one
    update; without it, one update."""
    pytest.importorskip("isaaclab")
    from ndm_gym import train

    calls = []
    algo = types.SimpleNamespace(wm_update_graphable=lambda buf: calls.append("one") or (0, 0), wm_updates_presampled=lambda buf: calls.append("all") or (0, 0),
                                 policy_update_graphable=None)  # fmt: skip
    buf = types.SimpleNamespace(size=0, T=1)  # not full: the graphed update runs eagerly
    train.update_graphs(algo, buf, argparse.Namespace(graph_early=False, wm_presample=True))[0]()
    train.update_graphs(algo, buf, argparse.Namespace(graph_early=False, wm_presample=False))[0]()
    train.update_graphs(algo, buf, argparse.Namespace(graph_early=False))[0]()
    assert calls == ["all", "one", "one"]


# ---------------------------------------------------------------- compiled
@pytest.mark.slow
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_learner_options_compile(monkeypatch, kin, synthetic_motion, synthetic_mdp, name):
    """The updates with the learner's options, their step functions compiled as training compiles them (here for the CPU;
    minutes): the presampled world-model updates and the graphable policy update give the eager ones' losses and gradients
    up to float rounding (here 6e-6 of each gradient's largest entry at most, the tolerance 1e-4). The compiled functions
    take the taps' slices and return the layers' inputs as the eager ones do."""
    kw = dict(fused_adam=True, no_sync=True, wm_batches=2, wm_deferred_grads=True, wm_presample=True, **POLICY)
    real, out = torch.compile, []
    for compiled in (False, True):
        monkeypatch.setattr(torch, "compile", real if compiled else (lambda fn, **k: fn))
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **kw)
        forward, backward = _policy_update(algo, buf, True)  # before the world model's updates, which it must not depend on
        torch.manual_seed(4)
        wm = algo.wm_updates_presampled(buf)
        out.append(forward + backward + [wm[0]] + [p.grad.clone() for p in algo.wm.parameters()])
    for a, b in zip(*out):
        if torch.isnan(a).any():  # SuperTrack's imagined reward and, without --st_fail_w, its survival
            torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
        else:
            _close(b, a, 1e-4)


# ---------------------------------------------------------------- the bf16 autocast (the CPU's, then on a GPU CUDA's)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA's bf16 autocast and the updates compiled for the GPU")


def _deferred_mlp_bf16(device: str, layer_norm: bool):
    """Under the bf16 autocast on ``device``, over several applications of the MLP in a chain: run_mlp_deferred gives
    run_mlp's values; the taps, their gradients and the stored inputs are bf16; deferred_grads' gradients are the sums of
    dy^T x and of dy over those operands (against float64: 1e-5 of each gradient's largest entry, the fp32 accumulation),
    with bf16_product the bf16 product (2^-7); and they are backpropagation's, which rounds every application's product
    to bf16 before summing, within 2e-2 of each gradient's largest entry (on the CPU 1.5e-3, and 4e-3 with bf16_product)."""
    from ndm_gym.core import nets

    torch.manual_seed(0)
    net = nets.mlp(12, [32, 24], 12, act=nn.SiLU, layer_norm=layer_norm).to(device)
    x0, T = torch.randn(16, 12, device=device), 5
    x, outs = x0, []
    for _ in range(T):  # a recurrence, as an imagined rollout
        x = nets.run_mlp(net, x)
        outs.append(x)
    torch.stack(outs).square().sum().backward()
    ref = {n: prm.grad.clone() for n, prm in net.named_parameters()}
    net.zero_grad(set_to_none=True)
    taps = nets.mlp_taps(net, (T, 16), device)
    steps = [t.unbind(0) for t in taps]
    x, outs2, ins = x0, [], []
    for j in range(T):
        x, i = nets.run_mlp_deferred(net, x, [t[j] for t in steps])
        outs2.append(x), ins.append(i)
    for a, b in zip(outs, outs2):
        torch.testing.assert_close(b, a, **EXACT)
    torch.stack(outs2).square().sum().backward()
    lin = nets.linears(net)
    stacked = [torch.stack([i[k] for i in ins]) for k in range(len(lin))]
    assert all(t.dtype == torch.bfloat16 and t.grad.dtype == torch.bfloat16 for t in taps) and all(x.dtype == torch.bfloat16 for x in stacked)
    want = [(t.grad.double().flatten(0, 1).T @ x.double().flatten(0, 1), t.grad.double().sum((0, 1))) for x, t in zip(stacked, taps)]
    nets.deferred_grads(net, stacked, taps)
    for m, (dw, db) in zip(lin, want):
        _close(m.weight.grad.double(), dw, 1e-5), _close(m.bias.grad.double(), db, 1e-5)
    for n, prm in net.named_parameters():
        _close(prm.grad, ref[n], 2e-2)
    for m in lin:
        m.weight.grad = m.bias.grad = None
    nets.deferred_grads(net, stacked, taps, bf16_product=True)
    for m, (dw, db) in zip(lin, want):
        _close(m.weight.grad.double(), dw, 2.0**-7), _close(m.bias.grad.double(), db, 1e-5)
        _close(m.weight.grad, ref[[n for n, prm in net.named_parameters() if prm is m.weight][0]], 2e-2)


def _deferred_updates_bf16(run):
    """Under the bf16 autocast, ``run(**options)`` one policy update and one world-model update -> (forward outputs,
    gradients): with the deferred gradients (--wm_deferred_grads against the default fit; --pi_deferred_grads against
    --imag_hoist alone, which shares its random stream) the forward outputs are the same and the gradients equal up to the
    bf16 rounding of backpropagation's per-step products (2e-2 of each gradient's largest entry; on the CPU 1.5e-3, and
    3.4e-3 with --deferred_bf16)."""
    base = run(pi_encode_once=True, imag_hoist=True)
    for extra in ({}, {"deferred_bf16": True}):
        out = run(wm_deferred_grads=True, **POLICY, **extra)
        for a, b in zip(base[0], out[0]):
            if torch.isnan(a).any():  # SuperTrack's imagined reward and, without --st_fail_w, its survival
                torch.testing.assert_close(b, a, equal_nan=True, **EXACT)
            else:
                _close(b, a, 1e-6)
        for a, b in zip(base[1], out[1]):
            _close(b, a, 2e-2)


@pytest.mark.parametrize("layer_norm", [False, True])
def test_deferred_mlp_under_the_cpus_autocast(cpu_autocast, layer_norm):
    _deferred_mlp_bf16("cpu", layer_norm)


@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_deferred_updates_under_the_cpus_autocast(cpu_autocast, eager, kin, synthetic_motion, synthetic_mdp, name):
    """``_deferred_updates_bf16`` with the CPU's autocast: the deferred MLPs cast their inputs to bf16 themselves, and the
    equal losses show that ``run_mlp``, in the update without them, ran in bf16 too."""
    from ndm_gym.core import nets

    assert nets.deferred_dtype(DEV) == torch.bfloat16

    def run(**kw):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **kw)
        forward, backward = _policy_update(algo, buf, False)  # before the world model's update, which it must not depend on
        torch.manual_seed(4)
        wm = algo.wm_update(buf)
        return forward + [wm[0]], backward + [prm.grad.clone() for prm in algo.wm.parameters()]

    _deferred_updates_bf16(run)


@GPU
@pytest.mark.parametrize("layer_norm", [False, True])
def test_deferred_mlp_under_the_autocast(monkeypatch, layer_norm):
    from ndm_gym.core import nets

    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    _deferred_mlp_bf16("cuda", layer_norm)


def _gpu_runs(synthetic_motion, name):
    """On the GPU: the synthetic motion library, a replay of noisy reference states, and ``run(**options)``: an agent with
    those options (the same initial weights and normalizers), one policy update and then one world-model update, each
    with a fixed seed -> ([the policy's loss, the world model's loss], [the policy's gradient norm and gradients, the
    world model's gradients])."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin = g1.kinematics("cuda")
    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    buf, g = ReplayBuffer(48, 64, "cuda", priorities=False), torch.Generator().manual_seed(7)
    for t in range(48):
        f, S, xi = random_states(synthetic_motion, 64, seed=100 + t)
        f = torch.minimum(f + t, synthetic_motion.clip_end[synthetic_motion.clip_id[f]] - 40)
        tr = {"s": flatten_state(S), "s1": flatten_state(synthetic_motion.ref_state(f + 1)), "a": torch.randn(64, 29, generator=g),
              "a_prev": torch.randn(64, 29, generator=g), "xi": xi, "r": torch.rand(64, generator=g), "f": f,
              "ep_step": torch.full((64,), t, dtype=torch.int32), "done": torch.zeros(64, dtype=torch.bool)}  # fmt: skip
        buf.add({k: v.cuda() for k, v in tr.items()})
    w = {k: v.flatten(0, 1) for k, v in buf.data.items()}
    S0, S1 = unflatten_state(w["s"]), unflatten_state(w["s1"])

    def run(**kw):
        args = _args()
        vars(args).update(algo=name, wm_hidden=[64, 64], **ST_ARGS, **kw)
        torch.manual_seed(0)
        algo = ALGOS[name](args, kin, motion, mdp, "cuda")
        nn.init.normal_(algo.wm.net[-1].weight, std=0.05)  # zero at initialization: no gradient through the network
        algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"], normalize_xi(w["xi"]))), algo.wm.out_norm.update(algo.wm.targets(S0, S1))
        algo.update_pi_norms(mdp.obs(S0, w["a_prev"], w["xi"], w["f"]), motion.goal(w["f"], algo.K))
        torch.manual_seed(3)
        res = algo.policy_update(buf)  # before the world model's update, which it must not depend on
        grads = [res[3]] + [prm.grad.clone() for prm in algo.pi.parameters()]
        torch.manual_seed(4)
        wm = algo.wm_update(buf)
        return [res[0], wm[0]], grads + [prm.grad.clone() for prm in algo.wm.parameters()]

    return run


@GPU
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_deferred_updates_under_the_autocast(monkeypatch, synthetic_motion, name):
    """``_deferred_updates_bf16`` on the GPU under CUDA's autocast, eagerly."""
    from ndm_gym.core import nets

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    _deferred_updates_bf16(_gpu_runs(synthetic_motion, name))


@pytest.mark.slow
@GPU
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_learner_options_compiled(synthetic_motion, name):
    """The updates compiled on the GPU, in fp32, as test_lafan1_training's test_goal_options_compiled: with --pi_encode_once
    and --wm_deferred_grads, which keep the default's random stream, the losses and gradients are the default's up to
    float rounding (1e-4 of each gradient's largest entry); so are --pi_deferred_grads' against --imag_hoist alone."""
    run = _gpu_runs(synthetic_motion, name)
    base, once = run(), run(pi_encode_once=True, wm_deferred_grads=True)
    for a, b in zip(base[0] + base[1], once[0] + once[1]):
        _close(b, a, 1e-4)
    hoisted, deferred = run(pi_encode_once=True, imag_hoist=True), run(**POLICY)
    for a, b in zip(hoisted[0] + hoisted[1], deferred[0] + deferred[1]):
        _close(b, a, 1e-4)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_learner_options_train(tmp_path, name):
    """A training run with the systems options and every learner option (Muon for the policy and the world model, the
    trainer's --amp bf16): the iteration's world-model updates are captured as one graph and the policy update as another,
    both replayed; no update is skipped and the logged losses are finite."""
    from test_lafan1_training import TRAIN, run, write_motions
    from test_systems import SYSTEMS

    motions, log = write_motions(tmp_path / "motions"), tmp_path / "run"
    learner = ["--pi_encode_once", "--imag_hoist", "--pi_deferred_grads", "--wm_deferred_grads", "--muon_batched", "--wm_presample"]
    args = [*TRAIN[2:], "--algo", name, "--pi_ema", "0.995", *SYSTEMS, "--pi_opt", "muon", "--wm_opt", "muon", "--wm_hidden", "256", "256", "256",
            "--wm_batches", "2", *learner, "--iterations", "8", "--motion_dir", motions, "--log_dir", str(log)]  # fmt: skip
    r = run("-m", "ndm_gym.train", *args)
    for what in ("one collection step", "the world-model updates", "the policy update"):
        assert f"captured {what}" in r.stdout, what
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == list(range(1, 9)) and all(x["nonfinite"] == "0" for x in rows)
    assert all(math.isfinite(float(x["wm_loss"])) for x in rows) and all(math.isfinite(float(x["pi_loss"])) for x in rows[1:])
