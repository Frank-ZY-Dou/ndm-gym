"""The G1's forward kinematics as Triton kernels (ndm_gym.kernels.fk; ArticulationKinematics(fused=True), train.py
--fk_fused) against ArticulationKinematics.forward: the four outputs and the gradients of all six inputs under random
upstream gradients on all four outputs, for all bodies and the tracked ones, random states (rotations of any angle, joint
angles past both limits, fast joints) and batch sizes that are not multiples of the kernels' block, to fp32 rounding:
values within ATOL + RTOL |reference|, gradients within RTOL |reference| + ATOL times the gradient's largest magnitude
(sums over up to 30 bodies). On the CPU the kernels run through Triton's interpreter, in a process started with
TRITON_INTERPRET=1 (test_interpreter starts it: triton.jit picks the interpreter when Triton is imported); on a GPU
compiled, also inside torch.compile and replayed as CUDA graphs (the first run compiles the kernels, minutes). Also:
without the option the class's own forward runs and the compiled graph has no kernel in it, and the option's refusals."""

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import types

import pytest
import torch

from ndm_gym.core.kinematics import ArticulationKinematics
from ndm_gym.core.rotations import so3_exp

CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_LAFAN1 = os.path.isdir(os.path.join(REPO, "data", "lafan1_g1"))  # recorded states: LAFAN1's reference frames
RTOL, ATOL = 1e-5, 1e-5
SIZES = (1, 37, 130)  # one sample, a part of the 64-sample block, two blocks and a part


@pytest.fixture(scope="module")
def fk_interp():
    """ndm_gym.kernels.fk with its kernels under Triton's interpreter, in a process started with TRITON_INTERPRET=1."""
    if not INTERPRET:
        pytest.skip("through Triton's interpreter, in a process started with TRITON_INTERPRET=1 (test_interpreter starts one)")
    fk = importlib.import_module("ndm_gym.kernels.fk")
    assert fk.INTERPRETED
    return fk


@pytest.mark.skipif(INTERPRET, reason="this process runs the interpreter's tests itself")
def test_interpreter():
    """This module's CPU tests through Triton's interpreter, in a process of their own: triton.jit decides between the
    interpreter and the compiler when Triton and the kernels are imported, so TRITON_INTERPRET=1 is set from the start."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.abspath(__file__), "-k", "not test_interpreter and not test_gpu"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    passed = [int(w) for w, nxt in zip(r.stdout.split(), r.stdout.split()[1:]) if nxt.startswith("passed") and w.isdigit()]
    assert r.returncode == 0 and passed and passed[-1] >= 13, r.stdout[-6000:] + r.stderr[-3000:]


def random_states(kin: ArticulationKinematics, shape: tuple, seed: int, dev: str = "cpu") -> list[torch.Tensor]:
    """Root positions, rotations (any angle), linear and angular velocities, joint angles up to 30 % of the range past
    each limit and fast joint rates, of the batch shape ``shape``."""
    g = torch.Generator().manual_seed(seed)
    nj = kin.q_lo.numel()
    lo, hi = kin.q_lo.cpu(), kin.q_hi.cpu()
    S = [torch.randn(*shape, 3, generator=g), so3_exp(2.0 * torch.randn(*shape, 3, generator=g)), 2.0 * torch.randn(*shape, 3, generator=g),
         3.0 * torch.randn(*shape, 3, generator=g), lo + (hi - lo) * (1.6 * torch.rand(*shape, nj, generator=g) - 0.3),
         4.0 * torch.randn(*shape, nj, generator=g)]  # fmt: skip
    return [x.to(dev) for x in S]


def compare(kin, kin_f, leaves, all_bodies: bool, seed: int, make=lambda *S: S):
    """kin_f.forward against kin.forward at the state ``make(*leaves)``: values and the gradients of the leaves."""
    xs = [x.detach().clone().requires_grad_(True) for x in leaves]
    ys = [x.detach().clone().requires_grad_(True) for x in leaves]
    ref = kin.forward(*make(*xs), all_bodies=all_bodies)
    out = kin_f.forward(*make(*ys), all_bodies=all_bodies)
    for a, b in zip(out, ref, strict=True):
        assert a.shape == b.shape and a.dtype == torch.float32 and a.is_contiguous()
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    g = torch.Generator().manual_seed(seed + 1)
    gs = [torch.randn(r.shape, generator=g).to(r.device) for r in ref]
    for a, b in zip(torch.autograd.grad(out, ys, gs), torch.autograd.grad(ref, xs, gs), strict=True):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))


@pytest.mark.parametrize("all_bodies", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_values_and_gradients(fk_interp, kin, n, all_bodies):
    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    compare(kin, kin_f, random_states(kin, (n,), seed=n), all_bodies, seed=n)


def test_accuracy_against_float64(fk_interp, kin):
    """The kernels' errors from forward in float64 are those of forward in fp32, at most a few times over."""
    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    S = random_states(kin, (130,), seed=7)
    gen = torch.Generator().manual_seed(8)
    runs = []
    for k, dtype in ((kin, torch.float64), (kin, torch.float32), (kin_f, torch.float32)):
        xs = [x.to(dtype).requires_grad_(True) for x in S]
        out = k.forward(*xs, all_bodies=True)
        if not runs:
            gs = [torch.randn(o.shape, generator=gen, dtype=torch.float64) for o in out]
        runs.append([o.detach().double() for o in out] + [x.double() for x in torch.autograd.grad(out, xs, [x.to(dtype) for x in gs])])
    exact, default, fused = runs
    for e, a, b in zip(exact, default, fused, strict=True):
        err_default, err_fused = (a - e).abs().max(), (b - e).abs().max()
        assert err_fused <= max(4 * err_default, 4e-7 * e.abs().max()), (float(err_fused), float(err_default))


@pytest.mark.skipif(not HAVE_LAFAN1, reason="needs LAFAN1's converted motions (data/lafan1_g1)")
@pytest.mark.parametrize("all_bodies", [True, False])
def test_recorded_states(fk_interp, kin, motion, all_bodies):
    """LAFAN1's reference states (the clips' poses with their finite-difference velocities) at random frames."""
    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    f = torch.randint(0, motion.num_frames, (130,), generator=torch.Generator().manual_seed(13))
    compare(kin, kin_f, list(motion.ref_state(f)), all_bodies, seed=13)


def test_shapes_and_layouts(fk_interp, kin):
    """Leading batch dimensions, inputs that are views of the flat replay state (rows of 71 values), a root broadcast
    to every sample (row stride 0), one unbatched sample and an empty batch."""
    from ndm_gym.robots import g1
    from ndm_gym.state import STATE_DIM, unflatten_state

    kin_f = g1.kinematics("cpu", fused=True)
    compare(kin, kin_f, random_states(kin, (2, 3, 11), seed=1), True, seed=1)
    p, R, v, w, q, qd = random_states(kin, (70,), seed=2)
    flat = torch.cat((p, torch.randn(70, 4), v, w, q, qd), -1)
    S = unflatten_state(flat)
    assert flat.shape[-1] == STATE_DIM and S[0].stride(0) == S[4].stride(0) == STATE_DIM and not S[4].is_contiguous()
    compare(kin, kin_f, [flat], False, seed=2, make=unflatten_state)
    S = random_states(kin, (40,), seed=3)
    S[0], S[1] = S[0][:1], S[1][:1]
    compare(kin, kin_f, S, True, seed=3, make=lambda p, R, *rest: (p.expand(40, 3), R.expand(40, 3, 3), *rest))
    compare(kin, kin_f, random_states(kin, (), seed=4), False, seed=4)
    S = random_states(kin, (0,), seed=5)
    for a, b in zip(kin_f.forward(*S, all_bodies=True), kin.forward(*S, all_bodies=True), strict=True):
        assert a.shape == b.shape and a.shape[0] == 0


def test_unused_outputs(fk_interp, kin):
    """A loss on the positions alone (the critic's input): the other outputs' adjoints are zero, and so are the gradients
    of the velocities, which the positions do not depend on."""
    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    S = random_states(kin, (50,), seed=9)
    xs = [x.clone().requires_grad_(True) for x in S]
    ys = [x.clone().requires_grad_(True) for x in S]
    ref = torch.autograd.grad(kin.forward(*xs)[0].square().sum(), xs, allow_unused=True, materialize_grads=True)
    out = torch.autograd.grad(kin_f.forward(*ys)[0].square().sum(), ys)
    for a, b in zip(out, ref, strict=True):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))
    assert not out[2].any() and not out[3].any() and not out[5].any()


def test_default_path_unchanged(kin):
    """Without the option the kinematics object runs the class's own forward; with it, ``forward`` is the kernels' and
    the class's forward is unchanged on the same object."""
    from ndm_gym.robots import g1

    assert kin.fused is False and "forward" not in vars(kin) and kin.forward.__func__ is ArticulationKinematics.forward
    kin_f = g1.kinematics("cpu", fused=True)
    assert kin_f.fused is True and kin_f.forward.__func__ is ArticulationKinematics.forward_fused
    S = random_states(kin, (20,), seed=10)
    for a, b in zip(ArticulationKinematics.forward(kin_f, *S, all_bodies=True), kin.forward(*S, all_bodies=True)):
        assert torch.equal(a, b)


def test_compiled_graphs(fk_interp, kin):
    """Under torch.compile (dynamo, AOTAutograd's joint graph from the fake and autograd registrations) the option's graph
    calls the kernels' operator, the default's does not: the two graphs, and so their compile-cache keys, differ. The
    compiled function's values and gradients equal the eager ones."""
    from torch._dynamo.testing import CompileCounterWithBackend

    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    S = random_states(kin, (40,), seed=11)
    gs = [torch.randn(s.shape, generator=torch.Generator().manual_seed(12)) for s in kin.forward(*S, all_bodies=True)]
    graphs = {}
    for name, k in (("default", kin), ("fused", kin_f)):
        counter = CompileCounterWithBackend("aot_eager")

        def loss(p, R, v, w, q, qd, k=k):
            return sum((o * g).sum() for o, g in zip(k.forward(p, R, v, w, q, qd, all_bodies=True), gs))

        torch._dynamo.reset()
        xs = [x.clone().requires_grad_(True) for x in S]
        ys = [x.clone().requires_grad_(True) for x in S]
        compiled = torch.compile(loss, backend=counter, fullgraph=True)(*xs)
        eager = loss(*ys)
        torch.testing.assert_close(compiled, eager, rtol=RTOL, atol=ATOL * float(eager.detach().abs()))
        for a, b in zip(torch.autograd.grad(compiled, xs), torch.autograd.grad(eager, ys)):
            torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL * float(b.abs().max()))
        assert counter.frame_count == 1 and len(counter.graphs) == 1
        graphs[name] = counter.graphs[0].code
    torch._dynamo.reset()
    assert "ndm_gym.fk_fused" in graphs["fused"] and "fk_fused" not in graphs["default"]


def test_refusals(kin, monkeypatch):
    """Other dtypes, mismatched shapes, unknown trees or bodies, trees the kernels do not cover, and CPU tensors for
    compiled kernels raise."""
    from ndm_gym.kernels import fk
    from ndm_gym.robots import g1

    kin_f = g1.kinematics("cpu", fused=True)
    S = random_states(kin, (5,), seed=12)
    with pytest.raises(TypeError, match="fp32"):
        kin_f.forward(*(x.double() for x in S))
    with pytest.raises(ValueError, match="shapes"):
        kin_f.forward(*S[:4], S[4][:, :20], S[5])
    with pytest.raises(KeyError, match="not registered"):
        fk.forward_kinematics("tree-unknown", [0], *S)
    with pytest.raises(ValueError, match="distinct indices"):
        fk.forward_kinematics(kin_f.fk_tree, [0, 0], *S)
    eye = torch.eye(3).expand(3, 3, 3)
    bad = types.SimpleNamespace(parent=[-1, 2, 0], joint_idx=[-1, 0, 1], q_lo=torch.zeros(2), off_p=torch.zeros(3, 3), off_R=eye, axes=eye[0])
    with pytest.raises(ValueError, match="parent"):
        fk.register(bad)
    bad.parent, bad.joint_idx = [-1, 0, 1], [-1, 0, 0]
    with pytest.raises(ValueError, match="exactly one body"):
        fk.register(bad)
    monkeypatch.setattr(fk, "INTERPRETED", False)  # the interpreter's kernels take CPU tensors, the compiled ones do not
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        kin_f.forward(*S)


def test_train_options():
    """--fk_fused: off by default in ndm_gym.train; refused by the trainers that have not checked it (hand training does
    not take ndm_gym.train's speed options at all)."""
    from ndm_gym import train_hand
    from ndm_gym.train import add_speed_args

    p = argparse.ArgumentParser()
    add_speed_args(p)
    assert p.parse_args([]).fk_fused is False and p.parse_args(["--fk_fused"]).fk_fused is True
    with pytest.raises(SystemExit):
        train_hand.parse_args(["--clip", "nowhere", "--fk_fused"])
    from ndm_gym import train_muscles

    assert "fk_fused" in train_muscles.SPEED_UNSUPPORTED
    with pytest.raises(SystemExit):
        train_muscles.parse_args(["--algo", "supertrack", "--fk_fused"])
    if importlib.util.find_spec("isaaclab") is not None:  # train_hoi imports Isaac Lab's launcher; it takes no speed options
        from ndm_gym import train_hoi

        with pytest.raises(SystemExit):
            train_hoi.parse_args(["--algo", "supertrack", "--task", "move_suitcase", "--fk_fused"])


@pytest.mark.skipif(not (CUDA and HAVE_LAFAN1) or INTERPRET, reason="needs a GPU, the compiled kernels and LAFAN1's converted motions")
@pytest.mark.parametrize("all_bodies", [True, False])
def test_gpu_recorded_states(kin, motion, all_bodies):
    """On a GPU: LAFAN1's reference states at 4,097 random frames."""
    from ndm_gym.robots import g1

    f = torch.randint(0, motion.num_frames, (4097,), generator=torch.Generator().manual_seed(14))
    compare(g1.kinematics("cuda"), g1.kinematics("cuda", fused=True), [x.cuda() for x in motion.ref_state(f)], all_bodies, seed=14)


@pytest.mark.skipif(not CUDA or INTERPRET, reason="needs a GPU and the compiled kernels")
@pytest.mark.parametrize("all_bodies", [True, False])
def test_gpu_kernels(kin, all_bodies):
    """On a GPU: the checks above with the compiled kernels (one batch of 65,536 = 2048 x 32 imagined states), CPU tensors
    refused, the kernels inside torch.compile (inductor) and replayed as CUDA graphs (reduce-overhead), against the
    eager default."""
    from ndm_gym.robots import g1

    kin_c, kin_f = g1.kinematics("cuda", fused=False), g1.kinematics("cuda", fused=True)
    for n in (1, 37, 130, 4097):
        compare(kin_c, kin_f, random_states(kin, (n,), seed=n, dev="cuda"), all_bodies, seed=n)
    compare(kin_c, kin_f, random_states(kin, (32, 2048), seed=20, dev="cuda"), all_bodies, seed=20)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        kin_f.forward(*random_states(kin, (4,), seed=21), all_bodies=all_bodies)
    gs = [torch.randn(o.shape, device="cuda") for o in kin_c.forward(*random_states(kin, (512,), seed=22, dev="cuda"), all_bodies=all_bodies)]

    def loss(p, R, v, w, q, qd):
        return sum((o * g).sum() for o, g in zip(kin_f.forward(p, R, v, w, q, qd, all_bodies=all_bodies), gs))

    def ref_loss(p, R, v, w, q, qd):
        return sum((o * g).sum() for o, g in zip(kin_c.forward(p, R, v, w, q, qd, all_bodies=all_bodies), gs))

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
    # a CUDA graph captured by hand around the eager kernels: forward and backward replayed on new inputs
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
