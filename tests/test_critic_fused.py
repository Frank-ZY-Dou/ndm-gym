"""SuperMimic's --critic_fused (train.py speed option, off by default; ndm_gym.kernels.critic_fused): each critic minibatch step by
Triton kernels around cuBLAS's products, against the default's step (--critic_compile's loss, autograd's backward pass,
clip_grad_norm_(1.0) and Adam's fused step) over several steps, in fp32 and under the bf16 autocast, with minibatches that are not
multiples of the kernels' tile of rows: the losses, the clipped gradients and Adam's moments to float rounding, the parameters'
changes as Adam turns them (a gradient entry within rounding of zero may change its sign, and with it the first step's change of
the learning rate's size). Also: Adam's state through state_dict and load_state_dict (the step count, the moments as
the flat buffers' views), the policy update with the option against without, without the option no kernels, and the refusals. On
the CPU the kernels run through Triton's interpreter, in a process started with TRITON_INTERPRET=1 (test_interpreter starts it;
the products are torch's there too); on a GPU compiled, also captured in a CUDA graph and replayed."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from conftest import DEV
from test_critic_compile import GRAPHABLE, PUBLISHED, _update
from test_xmimic_learner import _close

CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = 1e-3


@pytest.fixture()
def kernels():
    """ndm_gym.kernels.critic_fused with its kernels under Triton's interpreter, in a process started with TRITON_INTERPRET=1."""
    if not INTERPRET:
        pytest.skip("through Triton's interpreter, in a process started with TRITON_INTERPRET=1 (test_interpreter starts one)")
    from ndm_gym.kernels import critic_fused

    assert critic_fused.INTERPRETED
    return critic_fused


@pytest.mark.skipif(INTERPRET, reason="this process runs the interpreter's tests itself")
def test_interpreter():
    """This module's CPU tests through Triton's interpreter, in a process of their own (as test_st_wm_fused's)."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.abspath(__file__), "-k", "not test_interpreter and not test_gpu"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    passed = [int(w) for w, nxt in zip(r.stdout.split(), r.stdout.split()[1:]) if nxt.startswith("passed") and w.isdigit()]
    assert r.returncode == 0 and passed and passed[-1] >= 7, r.stdout[-6000:] + r.stderr[-3000:]


@pytest.fixture()
def cpu_autocast(monkeypatch):
    """--amp bf16 on the CPU: the CPU's bf16 autocast in the place of CUDA's in ``run_mlp``, and ``nets._autocasts`` then holds
    for CPU tensors (test_critic_compile's)."""
    from ndm_gym.core import nets

    real = torch.autocast
    monkeypatch.setattr(torch, "autocast", lambda device_type, dtype=None, **kw: real("cpu", dtype=dtype, **kw))
    monkeypatch.setattr(nets, "_autocasts", lambda device: True)
    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)


def _critics(D: int, dev, seed: int = 0):
    """Two equal critics (policies.critic: D -> 256 -> 128 -> 1) with Adam's fused step at LR."""
    from ndm_gym.policies.critic import Critic

    torch.manual_seed(seed)
    a = Critic(D).to(dev)
    b = Critic(D).to(dev)
    b.load_state_dict(a.state_dict())
    return [(c, torch.optim.Adam(c.parameters(), lr=LR, fused=True)) for c in (a, b)]


def _data(n: int, D: int, dt, dev, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (2.0 * torch.randn(n, D, generator=g)).to(dt).to(dev), torch.randn(n, generator=g).to(dev)


def _default_step(critic, opt, X, Y, idx):
    """The default's minibatch step (algos.supermimic with --critic_compile, eagerly) -> its loss"""
    from ndm_gym.core.nets import run_mlp

    loss = ((run_mlp(critic.net, X[idx]).squeeze(-1) - Y[idx]) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt.step()
    return loss.detach()


def _steps(fused, X, Y, mb: int, epochs: int, seed: int = 2):
    """``epochs`` passes of minibatches of ``mb`` rows (the update's loop) of the default (fused None: the first critic) or of the
    kernels -> per step (its loss, the clipped gradients, the parameters' change)"""
    critic, opt, F = fused
    g = torch.Generator().manual_seed(seed)
    out = []
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(X.device)
        for i in range(n // mb):
            idx = perm[i * mb : (i + 1) * mb]
            before = [p.detach().clone() for p in critic.parameters()]
            loss = _default_step(critic, opt, X, Y, idx) if F is None else F.step(X, Y, idx).clone()
            out.append((loss, [p.grad.detach().clone() for p in critic.parameters()],
                        [p.detach() - q for p, q in zip(critic.parameters(), before)]))  # fmt: skip
    return out


def _check(base, out, opt_a, opt_b, rel_first: float, rel_later: float):
    """The kernels' steps against the default's: the first step's loss and clipped gradients to rel_first (the same parameters);
    the later steps' to rel_later (their parameters differ by the earlier steps' rounding, which under bf16 moves some of the
    products' roundings by a unit or two: 1e-2 of the largest entry); Adam's moments to rel_later and the step count at the end.
    The parameters' changes: Adam's first step moves an entry by the learning rate times its gradient's sign (where the gradient is
    well above eps), so where the default's gradient exceeds rel_first of its largest entry (its sign beyond rounding) the two
    changes agree to 2 % of the learning rate, and elsewhere, where rounding may flip a sign, within twice the learning rate; later
    steps' changes within four times the learning rate (Adam's ratio of moments, bounded)"""
    for k, ((la, ga, da), (lb, gb, db)) in enumerate(zip(base, out)):
        rel = rel_first if k == 0 else rel_later
        torch.testing.assert_close(lb, la, rtol=rel, atol=0.0)
        for a, b in zip(ga, gb):
            _close(b, a, rel, msg=lambda m, k=k: f"step {k}: {m}")
        for g, a, b in zip(ga, da, db):
            diff = (b - a).abs()
            if k == 0:
                signed = g.abs() > rel_first * float(g.abs().max())
                worst = float(diff[signed].max()) if bool(signed.any()) else 0.0
                assert worst <= 0.02 * LR and float(diff.max()) <= 2 * LR + 1e-9, (k, worst, float(diff.max()))
            else:
                assert float(diff.max()) <= 4 * LR, (k, float(diff.max()))
    for pa, pb in zip(opt_a.param_groups[0]["params"], opt_b.param_groups[0]["params"]):
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        assert float(sb["step"]) == float(sa["step"]) == len(base)
        _close(sb["exp_avg"], sa["exp_avg"], rel_later)
        _close(sb["exp_avg_sq"], sa["exp_avg_sq"], 2 * rel_later)


@pytest.mark.parametrize("mb", [40, 64])
def test_steps_fp32(kernels, mb):
    """fp32 (no autocast): eight steps, minibatches of 40 rows (one tile and a part) and 64 (two tiles) of 128 rows."""
    (ca, oa), (cb, ob) = _critics(37, DEV)
    X, Y = _data(128, 37, torch.float32, DEV)
    base = _steps((ca, oa, None), X, Y, mb, 4 if mb == 64 else 3)
    out = _steps((cb, ob, kernels.CriticFused(cb, ob)), X, Y, mb, 4 if mb == 64 else 3)
    _check(base, out, oa, ob, 2e-5, 2e-5)


def test_steps_bf16(kernels, cpu_autocast):
    """Under the bf16 autocast (the CPU's in the place of CUDA's): every value the default rounds to bf16 rounded at the same point,
    the products accumulated in fp32 as the autocast's; eight steps of 40 rows."""
    (ca, oa), (cb, ob) = _critics(37, DEV)
    X, Y = _data(128, 37, torch.bfloat16, DEV)
    F = kernels.CriticFused(cb, ob)
    assert F.lowp == torch.bfloat16
    base = _steps((ca, oa, None), X, Y, 40, 3)
    out = _steps((cb, ob, F), X, Y, 40, 3)
    _check(base, out, oa, ob, 1e-5, 1e-2)


def test_state_dict_round_trip(kernels):
    """Three steps, the optimizer's and the critic's state_dict into a new critic, optimizer and kernels (the default's form: one
    step count and two moments per parameter), three more steps: the same bits as six steps of the first."""
    (ca, oa), (cb, ob) = _critics(37, DEV)
    X, Y = _data(128, 37, torch.float32, DEV)
    Fa = kernels.CriticFused(ca, oa)
    first = _steps((ca, oa, Fa), X, Y, 40, 1, seed=5)
    sd_c, sd_o = ca.state_dict(), oa.state_dict()
    assert all(float(s["step"]) == 3 for s in sd_o["state"].values())
    cb.load_state_dict(sd_c)
    ob.load_state_dict(sd_o)
    Fb = kernels.CriticFused(cb, ob)
    a = _steps((ca, oa, Fa), X, Y, 40, 1, seed=6)
    b = _steps((cb, ob, Fb), X, Y, 40, 1, seed=6)
    assert len(first) == len(a) == len(b) == 3
    for (la, ga, da), (lb, gb, db) in zip(a, b):
        torch.testing.assert_close(lb, la, rtol=0, atol=0)
        for x, y in zip(ga + da, gb + db):
            torch.testing.assert_close(y, x, rtol=0, atol=0)
    for x, y in zip(ca.parameters(), cb.parameters()):
        torch.testing.assert_close(y.detach(), x.detach(), rtol=0, atol=0)


def test_policy_update_with_the_option(kernels, kin, synthetic_motion, synthetic_mdp, monkeypatch):
    """The graphable policy update with --critic_fused against --critic_compile's (eagerly, torch.compile as the identity): the
    outputs (the value loss to rounding, the rest bit for bit: the option touches only the critic's steps) and the critic after the
    update's minibatch steps (2 % of the learning rate, as above)."""
    from test_bptt_tricks import _algo

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    kw = {**PUBLISHED, **GRAPHABLE}
    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, **kw)
    base = _update(algo, buf, True)
    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, critic_fused=True, **kw)
    assert algo.critic_fused and isinstance(algo.cfused, kernels.CriticFused)
    out = _update(algo, buf, True)
    for i, (a, b) in enumerate(zip(base[0], out[0])):
        if i == 1:
            torch.testing.assert_close(b, a, rtol=1e-4, atol=0)
        else:
            torch.testing.assert_close(b, a, rtol=0, atol=0)
    for a, b in zip(base[1], out[1]):
        torch.testing.assert_close(b, a, rtol=0, atol=0)
    lr = algo.args.lr
    for a, b in zip(base[2], out[2]):
        torch.testing.assert_close(b, a, rtol=0, atol=0.02 * lr)


def test_without_the_option_no_kernels(kin, synthetic_motion, synthetic_mdp, monkeypatch):
    """Without the option the update builds no kernels (the critic's parameters are its own tensors, Adam's state its own)."""
    from test_bptt_tricks import _algo

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    algo, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, **PUBLISHED, **GRAPHABLE)
    assert not algo.critic_fused and not hasattr(algo, "cfused")
    ptrs = [p.data_ptr() for p in algo.critic.parameters()]
    assert len(set(ptrs)) == 6 and all(p.grad is None for p in algo.critic.parameters())


def test_refusals(kernels, kin, synthetic_motion, synthetic_mdp, monkeypatch):
    """The option's refusals: without --critic_compile; a critic of other layers; Adam with weight decay; the training's
    command line."""
    from test_bptt_tricks import _algo

    from ndm_gym import train
    from ndm_gym.policies.critic import Critic

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    with pytest.raises(ValueError, match="critic_compile"):
        _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_fused=True, **GRAPHABLE)
    c = Critic(9)
    c.net[1] = nn.SiLU()
    with pytest.raises(ValueError, match="Linear, ELU"):
        kernels.CriticFused(c, torch.optim.Adam(c.parameters()))
    c = Critic(9)
    with pytest.raises(ValueError, match="weight decay"):
        kernels.CriticFused(c, torch.optim.Adam(c.parameters(), weight_decay=0.1))
    argv = sys.argv
    sys.argv = ["ndm_gym.train", "--algo", "supermimic", "--critic_fused"]
    try:
        with pytest.raises(SystemExit):
            train.parse_args()
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ the GPU: the compiled kernels


def _gpu_kernels():
    from ndm_gym.kernels import critic_fused

    assert not critic_fused.INTERPRETED
    return critic_fused


@pytest.mark.skipif(not CUDA or INTERPRET, reason="needs a GPU and the compiled kernels")
def test_gpu_steps_fp32(monkeypatch):
    """Compiled, fp32 products without TF32 (the default's then too): eight steps of 1,000 rows of 4,096, the training's widths
    and input size (914)."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    cf = _gpu_kernels()
    (ca, oa), (cb, ob) = _critics(914, "cuda")
    X, Y = _data(4096, 914, torch.float32, "cuda")
    base = _steps((ca, oa, None), X, Y, 1000, 2)
    out = _steps((cb, ob, cf.CriticFused(cb, ob)), X, Y, 1000, 2)
    _check(base, out, oa, ob, 1e-4, 1e-4)


@pytest.mark.skipif(not CUDA or INTERPRET, reason="needs a GPU and the compiled kernels")
def test_gpu_steps_bf16(monkeypatch):
    """Compiled, under CUDA's bf16 autocast (the training's --amp bf16), eight steps of 1,000 rows of 4,096: the first step's
    gradients within one bf16 unit of their largest entry (8e-3 of it: the products' fp32 sums rounded to bf16 at a different
    accumulation order move some entries by a unit)."""
    from ndm_gym.core import nets

    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    cf = _gpu_kernels()
    (ca, oa), (cb, ob) = _critics(914, "cuda")
    X, Y = _data(4096, 914, torch.bfloat16, "cuda")
    F = cf.CriticFused(cb, ob)
    assert F.lowp == torch.bfloat16
    base = _steps((ca, oa, None), X, Y, 1000, 2)
    out = _steps((cb, ob, F), X, Y, 1000, 2)
    _check(base, out, oa, ob, 8e-3, 1e-2)


@pytest.mark.skipif(not CUDA or INTERPRET, reason="needs a GPU and the compiled kernels")
def test_gpu_captured(monkeypatch):
    """Four steps captured in a CUDA graph (after a warm-up on a side stream, as the training's capture) and replayed: the same
    bits as the same four steps run eagerly from the same state."""
    from ndm_gym.core import nets

    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    cf = _gpu_kernels()
    (ca, oa), (cb, ob) = _critics(914, "cuda")
    X, Y = _data(4096, 914, torch.bfloat16, "cuda")
    Fa, Fb = cf.CriticFused(ca, oa), cf.CriticFused(cb, ob)
    idx = [torch.randperm(4096, generator=torch.Generator().manual_seed(k)).cuda()[:1024] for k in range(4)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # the warm-up: one step of each, eagerly
        Fa.step(X, Y, idx[0]), Fb.step(X, Y, idx[0])
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        losses = [Fb.step(X, Y, i).clone() for i in idx]
    eager = [Fa.step(X, Y, i).clone() for i in idx]
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, losses):
        torch.testing.assert_close(b, a, rtol=0, atol=0)
    for a, b in zip(ca.parameters(), cb.parameters()):
        torch.testing.assert_close(b.detach(), a.detach(), rtol=0, atol=0)
