"""The fused training kernels of RigidFormer's block (ndm_gym/kernels/rf_train.py, ``--rf_fused_train``) against the Block
on the GPU (skipped without CUDA): the forward pass and every gradient (the input, the rotary tables, the attention bias,
FiLM's context and every parameter) over token counts, batch sizes (odd, not a multiple of a tile's environments) and
options, with the parameters trained (the world-model fit) and frozen (the policy's BPTT: gradients to the inputs only);
RigidFormerNet with the fused blocks, eager and compiled, its trunk recomputed in the BPTT or the blocks' tensors kept
(--rf_bptt_keep_blocks: the same bits); RigidFormer's anchor RoPE (--rf_anchors, both modes, --rf_rope_center) and
--rf_fit_keep_ops on the fused path, in the network and in both world models' rollouts (fit and BPTT); determinism; the
refusals; with NDM_RF_TRAIN_CKPT and NDM_RF_TRAIN_DATA set, a trained world model's fit on recorded windows. The
reference runs as the world models run it, under bf16 autocast; the fused path rounds to bf16 only at the matrix
products' inputs, so its distance to the fp32 reference must not exceed the autocast reference's own (both measured on
the same inputs). The flags' mapping and refusals are also checked on CPU.
"""

import os

import pytest
import torch

DEV = "cuda"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused training kernels need a CUDA GPU")
OFFICIAL = dict(anchors=4, anchor_mode="mean", rope_center=True)  # RigidFormer v1's ARoPE: --rf_anchors 4 --rf_rope_center
CONCAT = dict(anchors=4, anchor_mode="concat")


def _errors(a, b):
    """Maximum absolute error and relative error of the norm of a against the reference b."""
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.norm() / b.float().norm().clamp(min=1e-30))


def _assert_at_bf16_rounding(fused, ref16, ref32, what, rel_factor=1.5):
    """The fused path is about as close to the fp32 reference as the bf16-autocast reference is, or closer: the relative
    error of the norm at most ``rel_factor`` times the autocast's, the maximum error at most twice. Both round the
    products' inputs to bf16, at partly different places (both round q, k, v and the gate's products; gradients summed
    over many tokens, as the norms' and FiLM's, carry errors of either), so either can come out ahead."""
    f, r = _errors(fused, ref32), _errors(ref16, ref32)
    assert torch.isfinite(fused).all(), what
    assert f[1] <= rel_factor * r[1] and f[0] <= 2.0 * r[0], f"{what}: fused vs fp32 (max, relative) {f}, autocast vs fp32 {r}"
    return f, r


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _block(T, N, rope=True, bias=True, seed=0):
    from ndm_gym.worldmodels.rigidformer import Block, Rope3d

    torch.manual_seed(seed)
    blk = Block(128, 4, 2.5, 64).to(DEV)
    with torch.no_grad():  # FiLM starts at zero; norms at one
        blk.film.weight.normal_(0.0, 0.1)
        blk.film.bias.normal_(0.0, 0.1)
        for m in (blk.norm1, blk.q_norm, blk.k_norm, blk.norm2):
            m.weight.uniform_(0.5, 1.5)
    x = torch.randn(N, T, 128, device=DEV) * (1 + 3 * torch.rand(N, T, 1, device=DEV))  # tokens of different scales
    pos = torch.cat((2 * torch.rand(N, T, 2, device=DEV) - 1, 1.8 * torch.rand(N, T, 1, device=DEV)), -1)  # metres
    cos, sin = Rope3d(24, 10.0).to(DEV)(pos) if rope else (None, None)
    b = 0.5 * torch.randn(1, 4, T, T, device=DEV) if bias else None
    c = torch.nn.functional.silu(torch.randn(N, 64, device=DEV))
    return blk, [x, cos, sin, b, c]


def _grads(fn, blk, inputs, dy):
    """fn(*inputs) and the gradients of <fn, dy> to the inputs (None where an input is None or does not require one) and
    to the block's parameters (None for frozen ones)."""
    ins = [None if t is None else t.detach().clone().requires_grad_(True) for t in inputs]
    for p in blk.parameters():
        p.grad = None
    y = fn(*ins)
    y.backward(dy)
    return y.detach().float(), [None if t is None else t.grad for t in ins], [None if p.grad is None else p.grad.clone() for p in blk.parameters()]


def _compare(blk, inputs, what, **kw):
    """Forward pass and every gradient of the fused block against the Block in fp32, next to the Block under autocast."""
    from ndm_gym.kernels.rf_train import rf_block_train

    dy = torch.randn_like(inputs[0])
    names = ["x", "cos", "sin", "bias", "c"] + [n for n, _ in blk.named_parameters()]
    y32, gi32, gp32 = _fp32(lambda: _grads(blk, blk, inputs, dy))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y16, gi16, gp16 = _grads(blk, blk, inputs, dy)
        yf, gif, gpf = _grads(lambda *a: rf_block_train(blk, *a, **kw), blk, inputs, dy)  # autocast does not change the kernels
    _assert_at_bf16_rounding(yf, y16, y32, f"{what}: y")
    for n, a, b, c in zip(names, gi16 + gp16, gif + gpf, gi32 + gp32):
        assert (b is None) == (c is None), f"{what}: gradient of {n}"
        if c is not None:
            _assert_at_bf16_rounding(b, a, c, f"{what}: gradient of {n}")
    return yf, gif, gpf


@GPU
@pytest.mark.parametrize("T,N", [(32, 1), (32, 7), (32, 37), (32, 1000), (16, 50), (64, 21)])
def test_block_matches_reference(T, N):
    """At T=64 an environment needs the 64-row tiles in every kernel (the default attention-backward tile holds 32)."""
    blk, inputs = _block(T, N, seed=T + N)
    _compare(blk, inputs, f"T={T} N={N}", **(dict(block_m=(128, 64, 64), warps=(8, 8, 8)) if T == 64 else {}))


@GPU
@pytest.mark.parametrize("rope,bias", [(False, True), (True, False), (False, False)])
def test_block_options(rope, bias):
    blk, inputs = _block(32, 300, rope, bias, seed=1)
    _compare(blk, inputs, f"rope {rope} bias {bias}")


@GPU
@pytest.mark.parametrize(
    "kw", [dict(block_m=(128, 64, 64), warps=(8, 8, 4)), dict(block_m=(64, 32, 32), warps=(4, 4, 4)), dict(chunk=32), dict(stages=1)]
)
def test_block_knobs(kw):
    """The tile sizes, warps, hidden chunk and pipelining change the schedule, not the result's precision."""
    blk, inputs = _block(32, 99, seed=2)
    _compare(blk, inputs, str(kw), **kw)


@GPU
def test_frozen_parameters():
    """In the policy's BPTT the block's parameters are frozen: no parameter gradients, and the input gradients are the
    ones of the world-model fit's backward pass (the same kernels without the parameters' sums and operands)."""
    from ndm_gym.kernels.rf_train import rf_block_train

    blk, (x, cos, sin, b, c) = _block(32, 45, seed=3)
    dy = torch.randn_like(x)
    fn = lambda *a: rf_block_train(blk, *a)  # noqa: E731
    _, gi_train, _ = _grads(fn, blk, [x, cos, sin, b, c], dy)
    for p in blk.parameters():
        p.requires_grad_(False)
        p.grad = None
    ins = [t.detach().clone().requires_grad_(True) for t in (x, cos, sin)]
    fn(*ins, b, c).backward(dy)  # the bias and FiLM's context without gradients, as in the BPTT
    assert all(p.grad is None for p in blk.parameters())
    for a, t in zip(gi_train[:3], ins):
        torch.testing.assert_close(t.grad, a, rtol=1e-6, atol=1e-6)


@GPU
@pytest.mark.parametrize("keep_ops", [False, True])
def test_block_is_deterministic(keep_ops):
    from ndm_gym.kernels.rf_train import rf_block_train

    blk, inputs = _block(32, 600, seed=4)
    dy = torch.randn_like(inputs[0])
    fn = lambda *a: rf_block_train(blk, *a, keep_ops=keep_ops)  # noqa: E731
    a, b = _grads(fn, blk, inputs, dy), _grads(fn, blk, inputs, dy)
    assert torch.equal(a[0], b[0])
    assert all(torch.equal(x, y) for x, y in zip(a[1] + a[2], b[1] + b[2]) if x is not None)


@GPU
@pytest.mark.parametrize("T,N", [(32, 1), (32, 37), (32, 1000), (16, 50), (64, 21)])
def test_block_keep_ops_matches_reference(T, N):
    """--rf_fit_keep_ops: the forward pass keeps the products' operands (xn1, z, xn2) and the backward pass takes them,
    with the SwiGLU half as the hidden-chunk kernel, cuBLAS dxn2 and the row kernel: the output and every gradient
    against the fp32 block, next to the bf16-autocast block's own errors, as the default fused path; the output is the
    default fused path's bit for bit (the same forward kernel)."""
    from ndm_gym.kernels.rf_train import rf_block_train

    blk, inputs = _block(T, N, seed=T + N + 1)
    kw = dict(block_m=(128, 64, 64), warps=(8, 8, 8)) if T == 64 else {}
    yk = _compare(blk, inputs, f"keep_ops T={T} N={N}", keep_ops=True, **kw)[0]
    with torch.no_grad():
        torch.testing.assert_close(yk, rf_block_train(blk, *inputs, **kw).float(), rtol=0, atol=0)


@GPU
def test_keep_ops_only_for_the_fit():
    """keep_ops keeps the operands only when the parameters train: with them frozen (the policy's BPTT) the forward and
    backward passes are the default fused path's, bit for bit, and no operand is kept."""
    from ndm_gym.kernels import rf_train
    from ndm_gym.kernels.rf_train import rf_block_train

    blk, (x, cos, sin, b, c) = _block(32, 45, seed=5)
    for p in blk.parameters():
        p.requires_grad_(False)
    dy = torch.randn_like(x)
    shapes, fwd = [], rf_train.rf_train_fwd
    out = []
    for keep in (False, True):
        ins = [t.detach().clone().requires_grad_(True) for t in (x, cos, sin)]
        y = rf_block_train(blk, *ins, b, c, keep_ops=keep)
        y.backward(dy)
        out.append([y.detach()] + [t.grad for t in ins])
    assert all(torch.equal(u, v) for u, v in zip(*out))
    rf_train.rf_train_fwd = lambda *a: shapes.append([t.shape for t in fwd(*a)]) or fwd(*a)
    try:
        rf_block_train(blk, x.clone().requires_grad_(True), cos, sin, b, c, keep_ops=True)
    finally:
        rf_train.rf_train_fwd = fwd
    assert shapes and all(s.numel() == 0 for s in shapes[0][2:]), shapes


def _net(fused_train, checkpoint="frozen", seed=0, **kw):  # kw: RigidFormerNet's (anchors, keep_ops, ...)
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(seed)
    net = RigidFormerNet(kinematics("cpu"), 6, 1, checkpoint=checkpoint, fused_train=fused_train, **kw).to(DEV)
    with torch.no_grad():  # the heads, FiLM and the tree bias start at zero; give every path a signal
        for m in [net.root_head, net.joint_head] + [b.film for b in net.blocks]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        net.rel_bias.normal_(0.0, 0.3)
    return net


def _net_input(net, n, g=None):
    """n random network inputs (N, in_dim + 3BP): normalized features, then the P rotary points of every body within
    1 m of the origin, divided by POS_SCALE as TokenInputNorm passes them on."""
    pos = (2 * torch.rand(n, 3 * net.n_body * max(net.n_rot_pts, 1), device=DEV, generator=g) - 1) / 0.1
    return torch.cat((torch.randn(n, net.in_dim, device=DEV, generator=g), pos), 1)


@GPU
@pytest.mark.parametrize("arope", [{}, OFFICIAL, CONCAT, {"keep_ops": True}, {**OFFICIAL, "keep_ops": True}],
                         ids=["link_origin", "official", "concat", "keep_ops", "official_keep_ops"])  # fmt: skip
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_network_matches_reference(frozen, compiled, arope):
    """RigidFormerNet with the fused blocks (fused_train) against the same network's default path in fp32 and under
    autocast: the output and the gradients of the input and (trained) of every parameter, eager and compiled; with frozen
    parameters the trunk is recomputed in the backward pass (checkpoint="frozen"), as in the policy's BPTT; one rotary
    point per body, and the anchor RoPE (the official configuration, and concat); --rf_fit_keep_ops. The embedding, the
    context network and the read-out run under autocast in both, compiled or not, so their roundings are common and
    either can come out ahead (factor 1.5 on the relative error)."""
    ref, fused = _net(False, **{k: v for k, v in arope.items() if k != "keep_ops"}), _net(True, **arope)
    fused.load_state_dict(ref.state_dict())
    n = 333
    x = _net_input(ref, n)
    dy = torch.randn(n, ref.out_dim, device=DEV)
    for net in (ref, fused):
        for p in net.parameters():
            p.requires_grad_(not frozen)

    def run(net, fn):
        xi = x.clone().requires_grad_(True)
        for p in net.parameters():
            p.grad = None
        y = fn(xi)
        y.float().backward(dy)
        grads = [p.grad for p in net.parameters()] if not frozen else []
        return [y.detach().float(), xi.grad] + grads

    out32 = _fp32(lambda: run(ref, ref))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out16 = run(ref, ref)
        out = run(fused, torch.compile(fused) if compiled else fused)
    names = ["y", "x"] + ([n for n, _ in ref.named_parameters()] if not frozen else [])
    for name, a, b, c in zip(names, out16, out, out32):
        _assert_at_bf16_rounding(b, a, c, f"network {arope} frozen {frozen} compiled {compiled}: {name}", rel_factor=1.5)


@GPU
@pytest.mark.parametrize("arope", [{}, OFFICIAL], ids=["link_origin", "official"])
@pytest.mark.parametrize("compiled", [False, True])
def test_network_keeps_blocks(compiled, arope):
    """--rf_bptt_keep_blocks (checkpoint="frozen_readout"): in the policy's BPTT the fused blocks' saved tensors (x and
    x1) take the place of their recomputation; the output and the input's gradient are the default recomputation's;
    also with the official anchor RoPE."""
    g = torch.Generator(device=DEV).manual_seed(7)
    out = []
    for mode in ("frozen", "frozen_readout"):
        net = _net(True, checkpoint=mode, **arope)
        for p in net.parameters():
            p.requires_grad_(False)
        if not out:
            x = _net_input(net, 257, g)
            dy = torch.randn(257, net.out_dim, device=DEV, generator=g)
        xi = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = (torch.compile(net) if compiled else net)(xi)
        y.float().backward(dy)
        out.append((y.detach().float(), xi.grad))
    torch.testing.assert_close(out[1][0], out[0][0], rtol=0, atol=0)
    torch.testing.assert_close(out[1][1], out[0][1], rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("compiled", [False, True])
def test_network_fp32_passes_run_the_blocks(compiled):
    """Without autocast (the fp32 passes of --wm_bptt_fp32 and --wm_smooth_w) a fused_train network runs its blocks as
    without the option: the output and every gradient equal the default network's bit for bit; compiled, the tree
    bias's gradient only up to the order of its atomic sums (the compiled default network's own runs differ there)."""
    g = torch.Generator(device=DEV).manual_seed(8)
    out = []
    for fused in (False, True):
        net = _net(fused)
        if not out:
            x = _net_input(net, 129, g)
            dy = torch.randn(129, net.out_dim, device=DEV, generator=g)
        xi = x.clone().requires_grad_(True)
        y = (torch.compile(net) if compiled else net)(xi)
        y.backward(dy)
        out.append({"y": y.detach(), "x": xi.grad} | {n: p.grad for n, p in net.named_parameters()})
    for name, a in out[0].items():
        exact = not (compiled and name == "rel_bias")
        torch.testing.assert_close(out[1][name], a, **(dict(rtol=0, atol=0) if exact else {}), msg=lambda m, n=name: f"{n}: {m}")


@GPU
def test_compiled_network_dispatch():
    """Traced by torch.compile, a fused_train network calls the fused kernels in its passes under bf16 autocast (entered
    outside the compiled function or inside it, as the world models' step does) and the blocks in its fp32 passes."""
    net = _net(True)
    x = _net_input(net, 65)

    def inner(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(x)

    for fn, amp, fused in ((net, True, True), (inner, False, True), (net, False, False)):
        torch._dynamo.reset()
        ops = set()

        def backend(gm, _):
            ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
            return gm.forward

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            torch.compile(fn, backend=backend)(x)
        assert any("rf_train_fwd" in o for o in ops) == fused, (fn, amp, sorted(ops)[:20])


@GPU
@pytest.mark.parametrize("which", ["accel", "increment"])
@pytest.mark.parametrize("kw", [OFFICIAL, CONCAT, {"keep_ops": True}, {**OFFICIAL, "keep_ops": True}],
                         ids=["official", "concat", "keep_ops", "official_keep_ops"])  # fmt: skip
def test_world_model_rollout(motion, which, kw):
    """A world model's (SuperTrack's accelerations, SuperMimic's increments) 4-step open-loop rollout from noisy
    reference states with the fused blocks against the default path in fp32, next to the default path's own error
    under bf16 autocast: with the anchor RoPE (the official configuration and concat; it changes only the rotary tables
    the blocks take, so the fused blocks serve it) and with --rf_fit_keep_ops (the fused path only). With the parameters
    trained (the world-model fit) the rollout's states and the parameters' gradients of a linear functional of them (a
    loss); with the parameters frozen (the policy's BPTT, the trunk recomputed) the states' and the setpoints'
    gradients. Factor 1.5 on the relative error, as above (on 4 seeds the fused path's error with the anchor RoPE was
    0.79-0.89 times the autocast path's in every such quantity)."""
    import ndm_gym.core.nets as nets
    from conftest import random_states
    from ndm_gym.algos.common import set_requires_grad
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    n, steps = 300, 4
    _, S0, xi = random_states(motion, n, seed=11)
    g = torch.Generator().manual_seed(12)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    p, R, v, w, q, qd = S0
    S1 = (p + 0.02 * v, so3_exp(0.02 * w) @ R, v + 0.1 * rn(n, 3), w + 0.2 * rn(n, 3), q + 0.02 * qd, qd + 0.5 * rn(n, 29))
    u, dy = 0.5 * rn(steps, n, 29), rn(steps, n, 76)
    S0, S1, u, dy, xi_n = (tuple(t.to(DEV) for t in S0), tuple(t.to(DEV) for t in S1), u.to(DEV), dy.to(DEV), normalize_xi(xi).to(DEV))
    cls = AccelWorldModel if which == "accel" else IncrementWorldModel
    wms = []
    for fused in (False, True):
        torch.manual_seed(0)
        opts = {k: v for k, v in kw.items() if fused or k != "keep_ops"}  # keep_ops: an option of the fused path
        wms.append(cls(rigidformer=dict(kin=kinematics("cpu"), fused_train=fused, **opts)).to(DEV))
    ref, fus = wms
    with torch.no_grad():  # the heads, FiLM and the tree bias start at zero; give every path a signal
        for m in [ref.net.root_head, ref.net.joint_head] + [b.film for b in ref.net.blocks]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        ref.net.rel_bias.normal_(0.0, 0.3)
    ref.in_norm.update(ref.inputs(S0, u[0], xi_n))
    ref.out_norm.update(ref.targets(S0, S1))
    fus.load_state_dict(ref.state_dict())

    def run(wm, frozen):
        set_requires_grad(wm, not frozen)
        S = tuple(t.clone().requires_grad_(frozen) for t in S0)
        us = u.clone().requires_grad_(frozen)
        for prm in wm.parameters():
            prm.grad = None
        out, St = [], S
        for k in range(steps):
            St = wm.step(St, us[k], xi_n)
            out.append(torch.cat((St[0], St[1].flatten(-2), *St[2:]), -1))
        y = torch.stack(out)
        (y * dy).sum().backward()
        grads = [t.grad for t in S + (us,)] if frozen else [prm.grad for prm in wm.parameters()]
        return [y.detach()] + grads

    amp = nets.AMP_DTYPE
    try:
        for frozen in (False, True):
            nets.AMP_DTYPE = None
            out32 = _fp32(lambda: run(ref, frozen))
            nets.AMP_DTYPE = torch.bfloat16
            out16, out = run(ref, frozen), run(fus, frozen)
            assert all(t is not None and torch.isfinite(t).all() for t in out)
            if not frozen:  # all parameters' gradients together, then each of at least 1024 entries: the error of a
                # gradient of a few entries (the heads' biases) does not average out and comes out up to 6x either way
                big = lambda o: [o[0], torch.cat([g.flatten() for g in o[1:]])] + [g for g in o[1:] if g.numel() >= 1024]  # noqa: E731
                out16, out, out32 = big(out16), big(out), big(out32)
            for i, (a, b, c) in enumerate(zip(out16, out, out32)):
                _assert_at_bf16_rounding(b, a, c, f"{which} {kw} frozen {frozen}: output {i}", rel_factor=1.5)
    finally:
        nets.AMP_DTYPE = amp


@GPU
@pytest.mark.skipif(not (os.environ.get("NDM_RF_TRAIN_CKPT") and os.environ.get("NDM_RF_TRAIN_DATA")),
                    reason="NDM_RF_TRAIN_CKPT and NDM_RF_TRAIN_DATA not set")  # fmt: skip
def test_trained_world_model_update_on_recorded_windows():
    """A LAFAN1 RigidFormer run's world model (NDM_RF_TRAIN_CKPT, a checkpoint of ndm_gym.train) on windows recorded with
    its policy (NDM_RF_TRAIN_DATA, from ndm_gym.wmbench.collect): the world-model fit's loss and the gradient of every
    parameter (the run's open-loop rollout and loss, eager, on 4 batches of 512 windows) with the fused blocks against
    fp32, next to the default path's own error under bf16 autocast. Medians over the batches: in bf16 the gradient of a
    window now and then jumps on either path (rounding moves its rollout across a kink of the loss; the fp32 gradient is
    stable there), so a single batch can come out far on either side. The loss's error at most twice the default's, the
    gradients' relative error (all parameters together) at most 1.5 times."""
    import argparse
    import statistics

    import ndm_gym.core.nets as nets
    import ndm_gym.train as T
    from conftest import REPO
    from ndm_gym.algos import ALGOS
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP
    from ndm_gym.wmbench.fit import load_buffer

    ck = torch.load(os.environ["NDM_RF_TRAIN_CKPT"], map_location=DEV, weights_only=False)
    kin = g1.kinematics(DEV)
    motion = MotionLib(lafan1.load_dir(os.path.join(REPO, "data", "lafan1_g1")), kin, DEV)
    buf, _ = load_buffer([os.environ["NDM_RF_TRAIN_DATA"]], DEV)

    def build(fused):
        parser = argparse.ArgumentParser()
        T.add_train_args(parser)
        T.add_speed_args(parser)
        args = parser.parse_args(["--algo", ck["args"]["algo"]])
        for k, v in ck["args"].items():
            if hasattr(args, k) and k not in ("motion_dir", "log_dir", "resume", "clips"):
                setattr(args, k, v)
        assert args.wm_backbone == "rigidformer", args.wm_backbone
        args.rf_fused_train = fused
        mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=DEV), T.training_reward_cfg(args.reward_std_scale))
        algo = ALGOS[args.algo](args, kin, motion, mdp, DEV)
        st = {k: ck[k] for k in ("wm", "pi", "bal_wm", "bal_pi", "critic") if k in ck}
        st.update({k: getattr(algo, k).state_dict() for k in ("opt_wm", "opt_pi", "opt_v") if hasattr(algo, k)})
        algo.load_state_dict(st)
        algo.wm_step = algo.wm.step  # eager: the comparison needs no compilation
        for name in ("st_wm_terms", "wm_err"):
            if hasattr(algo, "_" + name):
                setattr(algo, name, getattr(algo, "_" + name))
        return algo

    def fit_grads(algo, w):
        algo.opt_wm.zero_grad(set_to_none=True)
        torch.manual_seed(1)  # the same input-noise draws on every path
        loss = algo.wm_loss(algo._wm_rollout(w["s"], w["a"], w["xi"]), w)[0]
        loss.backward()
        return float(loss.detach()), torch.cat([p.grad.detach().flatten() for p in algo.wm.parameters()])

    ref, fused = build(False), build(True)
    errs = []  # per batch: the loss's error of autocast and fused, the gradient's relative error of autocast and fused
    amp = nets.AMP_DTYPE
    try:
        for b in range(4):
            torch.manual_seed(b)
            w = buf.sample_windows(512, ref.args.wm_window)
            nets.AMP_DTYPE = torch.bfloat16
            l16, g16 = fit_grads(ref, w)
            lf, gf = fit_grads(fused, w)
            nets.AMP_DTYPE = None
            l32, g32 = _fp32(lambda: fit_grads(ref, w))
            assert torch.isfinite(gf).all()
            errs.append((abs(l16 - l32), abs(lf - l32), _errors(g16, g32)[1], _errors(gf, g32)[1]))
    finally:
        nets.AMP_DTYPE = amp
    med = [statistics.median(e[i] for e in errs) for i in range(4)]
    fmt = lambda i, spec: " ".join(format(e[i], spec) for e in errs)  # noqa: E731
    msg = f"loss error autocast {fmt(0, '.2e')}, fused {fmt(1, '.2e')}; gradient relative error autocast {fmt(2, '.3f')}, fused {fmt(3, '.3f')}"
    print(f"\n{ck['args']['algo']}, 4 x 512 windows, vs fp32 per batch: {msg}")
    assert med[1] <= 2.0 * med[0] + 1e-9, errs
    assert med[3] <= 1.5 * med[2], errs


def test_flag_mapping_and_refusals():
    """--rf_fused_train reaches RigidFormerNet; it is refused with the MLP and with --amp none (no bf16 passes to fuse),
    and with fused_rope or without FiLM; with --wm_bptt_fp32 and --wm_smooth_w (fp32 passes) and with the anchor RoPE
    (either mode, --rf_rope_center) it is accepted. --rf_fit_keep_ops reaches it, and is refused without
    --rf_fused_train."""
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet, rigidformer_kwargs

    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=128, rf_layers=3, rf_heads=4,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False, amp="bf16")  # fmt: skip
    assert rigidformer_kwargs(a)["fused_train"] is False and rigidformer_kwargs({**a, "rf_fused_train": True})["fused_train"] is True
    for bad in ({"wm_backbone": "mlp"}, {"amp": "none"}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**a, "rf_fused_train": True, **bad})
    arope = ({"rf_anchors": 4, "rf_rope_center": True}, {"rf_anchors": 4, "rf_anchor_mode": "concat"})
    for ok in ({"wm_bptt_fp32": True}, {"wm_smooth_w": 0.1}) + arope:
        assert rigidformer_kwargs({**a, "rf_fused_train": True, **ok})["fused_train"] is True
    assert not RigidFormerNet(kinematics("cpu"), 6, 1).fused_train
    for bad in ({"fused_rope": True}, {"film": False}):
        with pytest.raises(ValueError, match="fused_train"):
            RigidFormerNet(kinematics("cpu"), 6, 1, fused_train=True, **bad)
    # --rf_fit_keep_ops: with --rf_fused_train only
    assert rigidformer_kwargs(a)["keep_ops"] is False
    assert rigidformer_kwargs({**a, "rf_fused_train": True, "rf_fit_keep_ops": True})["keep_ops"] is True
    for bad in ({"rf_fit_keep_ops": True}, {"wm_backbone": "mlp", "rf_fit_keep_ops": True}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**a, **bad})
    with pytest.raises(ValueError, match="fused_train"):
        RigidFormerNet(kinematics("cpu"), 6, 1, keep_ops=True)


@GPU
def test_block_refuses_what_it_does_not_do():
    from ndm_gym.kernels.rf_train import rf_block_train
    from ndm_gym.worldmodels.rigidformer import Block

    blk, (x, cos, sin, b, c) = _block(32, 4)
    with pytest.raises(ValueError, match="powers of two"):
        rf_block_train(blk, x[:, :31].contiguous(), cos[:, :, :31], sin[:, :, :31], b[:, :, :31, :31], c)
    with pytest.raises(ValueError, match="hidden chunk"):
        rf_block_train(blk, x, cos, sin, b, c, chunk=48)
    with pytest.raises(ValueError, match="block_m"):
        rf_block_train(blk, x, cos, sin, b, c, block_m=(96, 64, 32))
    with pytest.raises(ValueError, match="fp32"):
        rf_block_train(blk, x.double(), cos, sin, b, c)
    with pytest.raises(ValueError, match="FiLM"):
        rf_block_train(Block(128, 4, 2.5, 0).to(DEV), x, cos, sin, b, c)
    with pytest.raises(ValueError, match="CUDA"):
        rf_block_train(Block(128, 4, 2.5, 64), x.cpu(), cos.cpu(), sin.cpu(), b.cpu(), c.cpu())
    with pytest.raises(ValueError, match="bf16 autocast"), torch.autocast("cuda", dtype=torch.float16):
        rf_block_train(blk, x, cos, sin, b, c)
