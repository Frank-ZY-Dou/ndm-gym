"""SuperTrack's world-model window loss as Triton kernels (ndm_gym.kernels.wm_loss; train.py --st_wm_fused) against
algos.supertrack.wm_window_terms: the four terms and the gradients of the predicted state's six parts under random term
gradients, L1 and L2, for random states (rotations of any angle, joint angles past both limits, fast joints), predictions near
their targets (as a world model's), LAFAN1's recorded states, batch shapes of one robot and of two per sample and sizes that
are not multiples of the kernels' block, to fp32 rounding: values within RTOL |reference| + ATOL, gradients within RTOL
|reference| + ATOL times the gradient's largest magnitude. The rotation term alone (so3_log's four candidates, the
small-angle limit, near pi, the identity) against so3_log and autograd on given matrices, and the kernels' errors from float64
against the default's. SuperTrack's world-model loss and its parameters' gradients with the option against without. On the
CPU the kernels run through Triton's interpreter, in a process started with TRITON_INTERPRET=1 (test_interpreter starts
it); on a GPU compiled, also inside torch.compile and replayed as CUDA graphs (the first run compiles the kernels, minutes).
Also: without the option SuperTrack's loss does not call the kernels, and the option's refusals.

An L1 term's gradient is the sign of an error: where a predicted body's position, velocity or angle equals its target's to
fp32 rounding, the two implementations' signs differ by chance. The comparisons of L1 gradients therefore take states whose
errors are far from rounding (random states, predictions 0.02-0.05 off), as a world model's in training are."""

import argparse
import importlib
import os
import subprocess
import sys

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from ndm_gym.algos.supertrack import wm_window_terms  # noqa: E402
from ndm_gym.core.rotations import so3_exp, so3_log  # noqa: E402
from ndm_gym.kernels.fk import _ld9, _st9  # noqa: E402
from ndm_gym.kernels.wm_loss import _rel, _rot_term  # noqa: E402

CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_LAFAN1 = os.path.isdir(os.path.join(REPO, "data", "lafan1_g1"))
RTOL, ATOL = 1e-5, 1e-5
SIZES = (1, 37, 130)  # one sample, a part of the 64-sample block, two blocks and a part
TORCH_COMPILE = torch.compile  # the eager fixture replaces it


@pytest.fixture(scope="module")
def wm_interp():
    """ndm_gym.kernels.wm_loss with its kernels under Triton's interpreter, in a process started with TRITON_INTERPRET=1."""
    if not INTERPRET:
        pytest.skip("through Triton's interpreter, in a process started with TRITON_INTERPRET=1 (test_interpreter starts one)")
    wm_loss = importlib.import_module("ndm_gym.kernels.wm_loss")
    assert wm_loss.INTERPRETED
    return wm_loss


@pytest.mark.skipif(INTERPRET, reason="this process runs the interpreter's tests itself")
def test_interpreter():
    """This module's CPU tests through Triton's interpreter, in a process of their own (as test_fk_fused's)."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.abspath(__file__), "-k", "not test_interpreter and not test_gpu"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    passed = [int(w) for w, nxt in zip(r.stdout.split(), r.stdout.split()[1:]) if nxt.startswith("passed") and w.isdigit()]
    assert r.returncode == 0 and passed and passed[-1] >= 17, r.stdout[-6000:] + r.stderr[-3000:]


def random_states(kin, shape: tuple, seed: int, dev: str = "cpu") -> list[torch.Tensor]:
    """Root positions, rotations (any angle), linear and angular velocities, joint angles up to 30 % of the range past
    each limit and fast joint rates, of the batch shape ``shape`` (test_fk_fused's)."""
    g = torch.Generator().manual_seed(seed)
    nj = kin.q_lo.numel()
    lo, hi = kin.q_lo.cpu(), kin.q_hi.cpu()
    S = [torch.randn(*shape, 3, generator=g), so3_exp(2.0 * torch.randn(*shape, 3, generator=g)), 2.0 * torch.randn(*shape, 3, generator=g),
         3.0 * torch.randn(*shape, 3, generator=g), lo + (hi - lo) * (1.6 * torch.rand(*shape, nj, generator=g) - 0.3),
         4.0 * torch.randn(*shape, nj, generator=g)]  # fmt: skip
    return [x.to(dev) for x in S]


def near(S: list[torch.Tensor], seed: int, eps: float) -> list[torch.Tensor]:
    """A prediction of the states S off by about ``eps`` in every part (the root's rotation by an angle about eps)."""
    g = torch.Generator().manual_seed(seed)
    p, R, v, w, q, qd = (x.cpu() for x in S)
    out = [p + eps * torch.randn(p.shape, generator=g), R @ so3_exp(eps * torch.randn(p.shape, generator=g)),
           v + eps * torch.randn(v.shape, generator=g), w + eps * torch.randn(w.shape, generator=g), q + eps * torch.randn(q.shape, generator=g),
           qd + eps * torch.randn(qd.shape, generator=g)]  # fmt: skip
    return [x.to(S[0].device) for x in out]


def compare(kin, tree, S_hat, S, squared: bool, seed: int, make=lambda *S: S):
    """window_terms against wm_window_terms at the predictions ``make(*S_hat)`` and the targets S: values and the gradients of
    the leaves S_hat under random term gradients."""
    from ndm_gym.kernels.wm_loss import window_terms

    xs = [x.detach().clone().requires_grad_(True) for x in S_hat]
    ys = [x.detach().clone().requires_grad_(True) for x in S_hat]
    ref = wm_window_terms(kin, make(*xs), S, squared=squared)
    out = window_terms(tree, make(*ys), S, squared=squared)
    assert out.shape == ref.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    if not ref.numel():
        return
    gE = torch.randn(ref.shape, generator=torch.Generator().manual_seed(seed + 1)).to(ref.device)
    for a, b in zip(torch.autograd.grad(out, ys, gE), torch.autograd.grad(ref, xs, gE), strict=True):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))


@pytest.fixture(scope="module")
def tree(kin):
    from ndm_gym.kernels import fk

    return fk.register(kin)


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_random_states(wm_interp, kin, tree, n, squared):
    compare(kin, tree, random_states(kin, (n,), seed=n), random_states(kin, (n,), seed=n + 100), squared, seed=n)


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("eps", [0.05, 0.02])
def test_near_targets(wm_interp, kin, tree, eps, squared):
    """Predictions off by a few centimeters and hundredths of a radian, as a world model's in training."""
    S = random_states(kin, (130,), seed=3)
    compare(kin, tree, near(S, seed=4, eps=eps), S, squared, seed=5)


@pytest.mark.skipif(not HAVE_LAFAN1, reason="needs LAFAN1's converted motions (data/lafan1_g1)")
@pytest.mark.parametrize("squared", [False, True])
def test_recorded_states(wm_interp, kin, tree, motion, squared):
    """LAFAN1's reference states (the clips' poses with their finite-difference velocities) at random frames as the targets,
    predictions near them and the next frames' states as the predictions."""
    f = torch.randint(0, motion.num_frames - 1, (130,), generator=torch.Generator().manual_seed(13))
    S = list(motion.ref_state(f))
    compare(kin, tree, near(S, seed=14, eps=0.03), S, squared, seed=13)
    compare(kin, tree, list(motion.ref_state(f + 1)), S, squared, seed=15)


def test_shapes_and_layouts(wm_interp, kin, tree):
    """Leading batch dimensions (windows x steps, and two robots per sample), targets that are views of the flat replay state
    (rows of 71 values, as unflatten_state gives SuperTrack's), one unbatched sample and an empty batch."""
    from ndm_gym.state import unflatten_state

    compare(kin, tree, random_states(kin, (3, 2, 11), seed=1), random_states(kin, (3, 2, 11), seed=2), False, seed=1)
    compare(kin, tree, random_states(kin, (5, 3, 2), seed=3), random_states(kin, (5, 3, 2), seed=4), True, seed=3)
    p, R, v, w, q, qd = random_states(kin, (70,), seed=5)
    flat = torch.cat((p, torch.randn(70, 4, generator=torch.Generator().manual_seed(5)), v, w, q, qd), -1)
    S = unflatten_state(flat)
    assert S[0].stride(0) == S[4].stride(0) == flat.shape[-1] and not S[4].is_contiguous()
    compare(kin, tree, random_states(kin, (70,), seed=6), list(S), False, seed=6)
    compare(kin, tree, random_states(kin, (), seed=7), random_states(kin, (), seed=8), True, seed=7)
    compare(kin, tree, random_states(kin, (0,), seed=9), random_states(kin, (0,), seed=10), False, seed=9)


@triton.jit
def _rot_kernel(RT, RH, V, Y, N, SQUARED: tl.constexpr, BLOCK: tl.constexpr):
    """kernels.wm_loss._rot_term on rows of RT, RH (N, 3, 3): its value into V (N,), its gradient into Y (N, 3, 3)."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    rt = _ld9(RT, n * 9, m)
    value, y = _rot_term(_rel(rt, _ld9(RH, n * 9, m)), rt, SQUARED)
    tl.store(V + n, value, mask=m)
    _st9(Y, n * 9, m, y)


def rotation_cases(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(Rt, Rh) pairs: random rotations of any angle; then, with Rt the identity (so that Rt^T Rh is Rh exactly in both
    implementations), rotations by 1e-1 down to 1e-8 rad (the small-angle branch below 1e-6), by pi less 1e-1 down to 1e-4
    about each axis and about random axes (each of the four quaternion candidates), by pi exactly, and the identity."""
    g = torch.Generator().manual_seed(seed)
    Rt, Rh = [so3_exp(2.0 * torch.randn(64, 3, generator=g))], [so3_exp(2.0 * torch.randn(64, 3, generator=g))]
    axes = torch.cat((torch.eye(3), torch.nn.functional.normalize(torch.randn(5, 3, generator=g), dim=-1)))
    for angle in (1e-1, 1e-3, 1e-5, 1e-7, 1e-8, torch.pi - 1e-1, torch.pi - 1e-2, torch.pi - 1e-4, torch.pi):
        Rh.append(so3_exp(angle * axes))
        Rt.append(torch.eye(3).expand(len(axes), 3, 3))
    Rh.append(torch.eye(3).expand(4, 3, 3))
    Rt.append(torch.eye(3).expand(4, 3, 3))
    return torch.cat(Rt).contiguous(), torch.cat(Rh).contiguous()


@pytest.mark.parametrize("squared", [False, True])
def test_rotation_term(wm_interp, squared):
    """The rotation term alone on given matrices: |so3_log(Rt^T Rh)| summed (squared: its square) and its gradient with
    respect to Rh by autograd, to fp32 rounding, case by case (each case's gradients against their own largest)."""
    Rt, Rh = rotation_cases(seed=21)
    N = Rt.shape[0]
    V, Y = torch.empty(N), torch.empty(N, 3, 3)
    _rot_kernel[(triton.cdiv(N, 64),)](Rt, Rh, V, Y, N, SQUARED=squared, BLOCK=64)
    x = Rh.clone().requires_grad_(True)
    r = so3_log(Rt.transpose(-1, -2) @ x)
    ref = r.square().sum(-1) if squared else r.abs().sum(-1)
    (gref,) = torch.autograd.grad(ref.sum(), x)
    torch.testing.assert_close(V, ref.detach(), rtol=RTOL, atol=ATOL)
    for lo in range(0, N, 8):  # the cases come in groups of eight rows (the random ones too)
        a, b = Y[lo : lo + 8], gref[lo : lo + 8]
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * max(float(b.abs().max()), 1e-6))


def test_accuracy_against_float64(wm_interp, kin, tree):
    """The kernels' errors from the default in float64 are those of the default in fp32, at most a few times over: L1 on
    random states, L2 on predictions near their targets."""
    from ndm_gym.kernels.wm_loss import window_terms

    gen = torch.Generator().manual_seed(8)
    cases = (
        (False, (random_states(kin, (130,), seed=7), random_states(kin, (130,), seed=17))),
        (True, (near(random_states(kin, (130,), seed=9), seed=10, eps=0.03), random_states(kin, (130,), seed=9))),
    )
    for squared, (S_hat, S) in cases:
        gE = torch.randn(130, 4, generator=gen, dtype=torch.float64)
        runs = []
        for fused, dtype in ((False, torch.float64), (False, torch.float32), (True, torch.float32)):
            xs = [x.to(dtype).requires_grad_(True) for x in S_hat]
            St = [x.to(dtype) for x in S]
            out = window_terms(tree, xs, St, squared) if fused else wm_window_terms(kin, xs, St, squared=squared)
            runs.append([out.detach().double()] + [x.double() for x in torch.autograd.grad(out, xs, gE.to(dtype))])
        exact, default, fused = runs
        for e, a, b in zip(exact, default, fused, strict=True):
            err_default, err_fused = (a - e).abs().max(), (b - e).abs().max()
            assert err_fused <= max(4 * err_default, 4e-7 * e.abs().max()), (squared, float(err_fused), float(err_default))


def test_compiled_graph(wm_interp, kin, tree):
    """Under torch.compile (dynamo, AOTAutograd's joint graph from the fake and autograd registrations) the option's graph
    calls the kernels' operator, the default's does not; the compiled function's values and gradients equal the eager ones."""
    from torch._dynamo.testing import CompileCounterWithBackend

    from ndm_gym.kernels.wm_loss import window_terms

    S_hat, S = random_states(kin, (40,), seed=11), random_states(kin, (40,), seed=12)
    gE = torch.randn(40, 4, generator=torch.Generator().manual_seed(12))
    graphs = {}
    for name in ("default", "fused"):
        counter = CompileCounterWithBackend("aot_eager")

        def loss(p, R, v, w, q, qd, name=name):
            E = window_terms(tree, (p, R, v, w, q, qd), S) if name == "fused" else wm_window_terms(kin, (p, R, v, w, q, qd), S)
            return (E * gE).sum()

        torch._dynamo.reset()
        xs = [x.clone().requires_grad_(True) for x in S_hat]
        ys = [x.clone().requires_grad_(True) for x in S_hat]
        compiled = torch.compile(loss, backend=counter, fullgraph=True)(*xs)
        eager = loss(*ys)
        torch.testing.assert_close(compiled, eager, rtol=RTOL, atol=ATOL * float(eager.detach().abs()))
        for a, b in zip(torch.autograd.grad(compiled, xs), torch.autograd.grad(eager, ys)):
            torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))
        assert counter.frame_count == 1 and len(counter.graphs) == 1
        graphs[name] = counter.graphs[0].code
    torch._dynamo.reset()
    assert "ndm_gym.st_wm_terms" in graphs["fused"] and "st_wm_terms" not in graphs["default"]


def _supertrack(kin, motion, mdp, **kw):
    """test_bptt_tricks' small SuperTrack (random last layer, fitted normalizers) with the options kw, and its replay."""
    from test_bptt_tricks import _algo

    return _algo("supertrack", kin, motion, mdp, **kw)


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


@pytest.mark.skipif(not HAVE_LAFAN1, reason="needs LAFAN1's converted motions (data/lafan1_g1)")
@pytest.mark.parametrize("loss", ["l1", "l2"])
def test_supertrack_wm_loss(wm_interp, eager, kin, motion, mdp, loss):
    """SuperTrack's world-model loss on the same windows and predictions with the option and without: the loss, the
    per-window errors and the world model's parameters' gradients (L1: an untrained model's predictions are far from
    their targets)."""
    out = {}
    for fused in (False, True):
        algo, buf = _supertrack(kin, motion, mdp, st_wm_fused=fused, st_wm_loss=loss)
        assert algo.wm_fused is fused
        torch.manual_seed(1)
        w = buf.sample_windows(algo.args.wm_batch, algo.args.wm_window)
        S_hat = algo.wm_rollout(w["s"], w["a"], w["xi"])
        value, _, per_window = algo.wm_loss(S_hat, w)
        algo.wm.zero_grad(set_to_none=True)
        value.backward()
        out[fused] = (value.detach(), per_window, [p.grad.clone() for p in algo.wm.parameters()], algo.bal_wm.w.clone())
    (v0, pw0, g0, w0), (v1, pw1, g1, w1) = out[False], out[True]
    torch.testing.assert_close(w1, w0, rtol=RTOL, atol=0)  # the calibration on the first loss: the terms' reciprocals
    torch.testing.assert_close(v1, v0, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(pw1, pw0, rtol=RTOL, atol=ATOL * float(pw0.abs().max()))
    for a, b in zip(g1, g0, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4 * float(b.abs().max()))


@pytest.mark.skipif(not HAVE_LAFAN1, reason="needs LAFAN1's converted motions (data/lafan1_g1)")
def test_default_path_unchanged(kin, motion, mdp, eager):
    """Without the option SuperTrack holds no tree of the kernels, and its compiled window terms do not call them."""
    from torch._dynamo.testing import CompileCounterWithBackend

    algo, buf = _supertrack(kin, motion, mdp)
    assert algo.wm_fused is False and not hasattr(algo, "wm_tree") and not hasattr(algo, "wm_terms_fused")
    w = buf.sample_windows(8, algo.args.wm_window)
    with torch.no_grad():
        S_hat = algo.wm_rollout(w["s"], w["a"], w["xi"])
    counter = CompileCounterWithBackend("aot_eager")
    torch._dynamo.reset()
    TORCH_COMPILE(algo._st_wm_terms, backend=counter)(S_hat, w["s1"], w["mask"])
    assert counter.frame_count >= 1 and "ndm_gym.st_wm_terms" not in "".join(g.code for g in counter.graphs)
    torch._dynamo.reset()


def test_refusals(kin, tree, monkeypatch):
    """Other dtypes, mismatched shapes, unknown trees, and CPU tensors for compiled kernels raise."""
    from ndm_gym.kernels import fk, wm_loss

    S_hat, S = random_states(kin, (5,), seed=12), random_states(kin, (5,), seed=13)
    with pytest.raises(TypeError, match="fp32"):
        wm_loss.window_terms(tree, [x.double() for x in S_hat], S)
    with pytest.raises(ValueError, match="shapes"):
        wm_loss.window_terms(tree, S_hat[:4] + [S_hat[4][:, :20], S_hat[5]], S)
    with pytest.raises(ValueError, match="shapes"):
        wm_loss.window_terms(tree, S_hat, [x[:4] for x in S])
    with pytest.raises(KeyError, match="not registered"):
        wm_loss.window_terms("tree-unknown", S_hat, S)
    monkeypatch.setattr(fk, "INTERPRETED", False)  # the interpreter's kernels take CPU tensors, the compiled ones do not
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        wm_loss.window_terms(tree, S_hat, S)


def test_train_options(monkeypatch):
    """--st_wm_fused: off by default; an option of --algo supertrack only; refused by the trainers that have not checked it."""
    from ndm_gym import train, train_hand, train_muscles
    from ndm_gym.train import add_speed_args

    p = argparse.ArgumentParser()
    add_speed_args(p)
    assert p.parse_args([]).st_wm_fused is False and p.parse_args(["--st_wm_fused"]).st_wm_fused is True
    monkeypatch.setattr(sys, "argv", ["train", "--algo", "supertrack", "--log_dir", "/nonexistent", "--st_wm_fused"])
    assert train.parse_args().st_wm_fused is True
    for algo in ("supermimic", "other"):
        monkeypatch.setattr(sys, "argv", ["train", "--algo", algo, "--log_dir", "/nonexistent", "--st_wm_fused"])
        with pytest.raises(SystemExit):
            train.parse_args()
    with pytest.raises(SystemExit):
        train_hand.parse_args(["--clip", "nowhere", "--st_wm_fused"])
    assert "st_wm_fused" in train_muscles.SPEED_UNSUPPORTED
    with pytest.raises(SystemExit):
        train_muscles.parse_args(["--algo", "supertrack", "--st_wm_fused"])
    if importlib.util.find_spec("isaaclab") is not None:  # train_hoi imports Isaac Lab's launcher
        from ndm_gym import train_hoi

        with pytest.raises(SystemExit):
            train_hoi.parse_args(["--algo", "supertrack", "--task", "move_suitcase", "--st_wm_fused"])


@pytest.mark.skipif(not CUDA or INTERPRET, reason="needs a GPU and the compiled kernels")
def test_gpu_kernels(kin):
    """On a GPU (fp32 products, TF32 off): the checks above with the compiled kernels, up to one world-model batch of the
    configurations (2,048 windows x 8 steps, and 3,072 x 8 with the mirrored half), the rotation term on its cases, the
    kernels inside torch.compile (inductor) and replayed as CUDA graphs (reduce-overhead), and a CUDA graph captured by hand
    around the eager kernels."""
    from ndm_gym.kernels import fk
    from ndm_gym.kernels.wm_loss import window_terms
    from ndm_gym.robots import g1

    torch.backends.cuda.matmul.allow_tf32 = False
    kin_c = g1.kinematics("cuda")
    tree = fk.register(kin_c)
    for n in (1, 37, 130, 4097, 16384, 24576):
        compare(kin_c, tree, random_states(kin, (n,), seed=n, dev="cuda"), random_states(kin, (n,), seed=n + 1, dev="cuda"), False, seed=n)
        S = random_states(kin, (n,), seed=n + 2, dev="cuda")
        compare(kin_c, tree, near(S, seed=n + 3, eps=0.03), S, True, seed=n)
    Rt, Rh = (x.cuda() for x in rotation_cases(seed=21))
    for squared in (False, True):
        V, Y = torch.empty(Rt.shape[0], device="cuda"), torch.empty_like(Rt)
        _rot_kernel[(triton.cdiv(Rt.shape[0], 64),)](Rt, Rh, V, Y, Rt.shape[0], SQUARED=squared, BLOCK=64)
        x = Rh.clone().requires_grad_(True)
        r = so3_log(Rt.transpose(-1, -2) @ x)
        ref = r.square().sum(-1) if squared else r.abs().sum(-1)
        (gref,) = torch.autograd.grad(ref.sum(), x)
        torch.testing.assert_close(V, ref.detach(), rtol=RTOL, atol=ATOL)
        for lo in range(0, Rt.shape[0], 8):
            torch.testing.assert_close(Y[lo : lo + 8], gref[lo : lo + 8], rtol=RTOL, atol=ATOL * max(float(gref[lo : lo + 8].abs().max()), 1e-6))
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        window_terms(tree, random_states(kin, (4,), seed=5), random_states(kin, (4,), seed=6))
    S_t = random_states(kin, (512,), seed=7, dev="cuda")
    gE = torch.randn(512, 4, device="cuda")

    def loss(p, R, v, w, q, qd):
        return (window_terms(tree, (p, R, v, w, q, qd), S_t) * gE).sum()

    def ref_loss(p, R, v, w, q, qd):
        return (wm_window_terms(kin_c, (p, R, v, w, q, qd), S_t) * gE).sum()

    torch._dynamo.reset()
    for mode in ("default", "reduce-overhead"):
        compiled = torch.compile(loss, mode=mode, fullgraph=True)
        for it in range(4):  # warm-up, recording, replays with new values in the graph's inputs
            S = random_states(kin, (512,), seed=30 + it, dev="cuda")
            xs = [x.clone().requires_grad_(True) for x in S]
            ys = [x.clone().requires_grad_(True) for x in S]
            torch.compiler.cudagraph_mark_step_begin()
            out, ref = compiled(*xs), ref_loss(*ys)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL * float(ref.detach().abs()))
            out.backward()
            ref.backward()
            for a, b in zip(xs, ys):
                torch.testing.assert_close(a.grad, b.grad, rtol=RTOL, atol=ATOL * float(b.grad.abs().max()))
    torch._dynamo.reset()
    static = [x.clone().requires_grad_(True) for x in random_states(kin, (512,), seed=40, dev="cuda")]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            torch.autograd.grad(loss(*static), static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value = loss(*static)
        grads = torch.autograd.grad(value, static)
    S = random_states(kin, (512,), seed=41, dev="cuda")
    with torch.no_grad():
        for t, x in zip(static, S):
            t.copy_(x)
    graph.replay()
    ys = [x.clone().requires_grad_(True) for x in S]
    ref = ref_loss(*ys)
    torch.testing.assert_close(value, ref, rtol=RTOL, atol=ATOL * float(ref.detach().abs()))
    for a, b in zip(grads, torch.autograd.grad(ref, ys)):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))


@pytest.mark.skipif(not (CUDA and HAVE_LAFAN1) or INTERPRET, reason="needs a GPU, the compiled kernels and LAFAN1's converted motions")
def test_gpu_recorded_states(kin, motion):
    """On a GPU: LAFAN1's reference states at 4,097 random frames as the targets, predictions near them (L1 and L2)."""
    from ndm_gym.kernels import fk
    from ndm_gym.robots import g1

    torch.backends.cuda.matmul.allow_tf32 = False
    kin_c = g1.kinematics("cuda")
    f = torch.randint(0, motion.num_frames, (4097,), generator=torch.Generator().manual_seed(14))
    S = [x.cuda() for x in motion.ref_state(f)]
    for squared in (False, True):
        compare(kin_c, fk.register(kin_c), near(S, seed=15, eps=0.03), S, squared, seed=14)
