"""The fused read-out of RigidFormerNet (ndm_gym/kernels/rf_head.py, ``--rf_fused_readout``) against
``RigidFormerNet._readout`` on the GPU (skipped without CUDA): the output and the gradients of the tokens' features on the
two read-out layers, for one token per body and merged tokens (``--rf_merge``: from the tokens' rows, the default, and
per body), over batch sizes (one, odd, not a multiple of a tile's environments) and the kernels' tile knobs;
determinism; RigidFormerNet with the fused read-out
beside the fused blocks, eager and compiled, with frozen parameters (the policy's BPTT) and trained ones (the read-out
then runs in PyTorch: the same bits as without the option); the refusals. The reference runs as the world models run
it, under bf16 autocast; the fused kernels round to bf16 only at the matrix products' inputs and multiply by the product
of the output projection and the heads, so their distance to the fp32 reference must not exceed the autocast
reference's own (both measured on the same inputs). The flags' mapping and refusals are also checked on CPU.
"""

import pytest
import torch

DEV = "cuda"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused read-out needs a CUDA GPU")
NETS = [("none", 2, 3), ("joints", 2, 2), ("distal", 8, 2)]  # merge, registers, layers: 32 or 16 tokens


def _errors(a, b):
    """Maximum absolute error and relative error of the norm of a against the reference b."""
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.norm() / b.float().norm().clamp(min=1e-30))


def _assert_at_bf16_rounding(fused, ref16, ref32, what, rel_factor=1.5):
    """The fused path is about as close to the fp32 reference as the bf16-autocast reference is, or closer."""
    f, r = _errors(fused, ref32), _errors(ref16, ref32)
    assert torch.isfinite(fused).all(), what
    assert f[1] <= rel_factor * r[1] and f[0] <= 2.0 * r[0], f"{what}: fused vs fp32 (max, relative) {f}, autocast vs fp32 {r}"


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _net(merge="none", registers=2, layers=3, seed=0, **kw):
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(seed)
    net = RigidFormerNet(kinematics("cpu"), 6, 1, layers=layers, registers=registers, merge=merge, **kw).to(DEV)
    with torch.no_grad():  # the heads, FiLM and the tree bias start at zero, the norm's weight at one; give every path a signal
        for m in [net.root_head, net.joint_head] + [b.film for b in net.blocks]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        net.rel_bias.normal_(0.0, 0.3)
        net.head_norm.weight.uniform_(0.5, 1.5)
        if merge != "none":
            net.unmerge_emb.normal_(0.0, 0.5)
    return net


def _features(net, n, g):
    """The tokens' features on the two read-out layers, of varied norms, and an output gradient."""
    T, D = net.n_tok + net.n_reg, net.embed.weight.shape[0]
    hs = [torch.randn(n, T, D, device=DEV, generator=g) * (1 + 2 * torch.rand(n, T, 1, device=DEV, generator=g)) for _ in range(2)]
    return hs[0], hs[1], torch.randn(n, net.out_dim, device=DEV, generator=g)


def _readout_grads(fn, h0, h1, dy):
    a, b = h0.clone().requires_grad_(True), h1.clone().requires_grad_(True)
    y = fn(a, b)
    y.float().backward(dy)
    return y.detach().float(), a.grad, b.grad


def _check_readout(net, n, tables, what):
    from ndm_gym.kernels.rf_head import rf_readout

    net.requires_grad_(False)
    h0, h1, dy = _features(net, n, torch.Generator(device=DEV).manual_seed(n))
    r32 = _fp32(lambda: _readout_grads(net._readout, h0, h1, dy))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r16 = _readout_grads(net._readout, h0, h1, dy)
        fused = [_readout_grads(lambda a, b: rf_readout(net, a, b, tables), h0, h1, dy) for _ in range(2)]
    for name, a, b, b2, c in zip(("y", "dh0", "dh1"), r16, *fused, r32):
        _assert_at_bf16_rounding(b, a, c, f"{what} n {n}: {name}")
        assert torch.equal(b, b2), f"{what} n {n}: {name} differs between two runs"


@GPU
@pytest.mark.parametrize("merge,registers,layers,token_rows", [(*NETS[0], None), (*NETS[1], None), (*NETS[1], False), (*NETS[2], None),
                                                               (*NETS[2], False)])  # fmt: skip
@pytest.mark.parametrize("n", [1, 7, 333, 2048])
def test_readout_matches_reference(merge, registers, layers, token_rows, n):
    """The output and both gradients, twice (the same bits): a body per token, and merged tokens, read out from the
    tokens' rows (the default) or per body, whose bodies' gradients are summed per token; a register's gradient is zero."""
    from ndm_gym.kernels.rf_head import head_tables, rf_readout

    net = _net(merge, registers, layers)
    tables = head_tables(net, token_rows=token_rows)
    assert tables["token_rows"] == (merge != "none" and token_rows is None)
    _check_readout(net, n, tables, f"merge {merge} token_rows {token_rows}")
    h0, h1, dy = _features(net, n, torch.Generator(device=DEV).manual_seed(1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, d0, d1 = _readout_grads(lambda a, b: rf_readout(net, a, b, tables), h0, h1, dy)
    assert not d0[:, net.n_tok :].any() and not d1[:, net.n_tok :].any()


@GPU
@pytest.mark.parametrize("kw", [dict(block_m=128, token_rows=False), dict(block_m=32, chunk=16, token_rows=False),
                                dict(bwd_block_m=64, token_rows=False), dict(bwd_chunk=16), dict(tok_block_m=64), dict(chunk=16, bwd_chunk=16)],
                         ids=lambda kw: "_".join(f"{k}{v}" for k, v in kw.items()))  # fmt: skip
def test_readout_knobs(kw):
    """Other rows per tile and other chunks: the same function. The kernels per body (another chunk of the backward pass
    leaves only them), then the token rows' in tiles of 64 rows and in chunks of 16 columns."""
    from ndm_gym.kernels.rf_head import head_tables

    net = _net("joints", 2, 2)
    tables = head_tables(net, **kw)
    assert tables["token_rows"] == ("token_rows" not in kw and kw.get("bwd_chunk", 32) == kw.get("chunk", 32))
    _check_readout(net, 131, tables, f"knobs {kw}")


@GPU
@pytest.mark.parametrize("merge,registers,layers", NETS[:2])
@pytest.mark.parametrize("compiled", [False, True])
def test_network_matches_reference(merge, registers, layers, compiled):
    """RigidFormerNet with the fused blocks and the fused read-out, frozen (the policy's BPTT: the blocks' tensors kept,
    the read-out's kernels keep nothing), against the default network in fp32 and under autocast: the output and the
    input's gradient, eager and compiled."""
    ref = _net(merge, registers, layers)
    fused = _net(merge, registers, layers, fused_train=True, fused_readout=True, checkpoint="frozen_readout")
    fused.load_state_dict(ref.state_dict())
    n = 333
    g = torch.Generator(device=DEV).manual_seed(3)
    pos = (2 * torch.rand(n, 3 * ref.n_body * max(ref.n_rot_pts, 1), device=DEV, generator=g) - 1) / 0.1
    x = torch.cat((torch.randn(n, ref.in_dim, device=DEV, generator=g), pos), 1)
    dy = torch.randn(n, ref.out_dim, device=DEV, generator=g)
    for net in (ref, fused):
        net.requires_grad_(False)

    def run(fn):
        xi = x.clone().requires_grad_(True)
        y = fn(xi)
        y.float().backward(dy)
        return y.detach().float(), xi.grad

    out32 = _fp32(lambda: run(ref))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out16 = run(ref)
        out = run(torch.compile(fused) if compiled else fused)
    for name, a, b, c in zip(("y", "x"), out16, out, out32):
        _assert_at_bf16_rounding(b, a, c, f"network merge {merge} compiled {compiled}: {name}")


@GPU
def test_trained_parameters_run_the_default_readout():
    """With parameters that train (the world-model fit) and in fp32 passes the option changes nothing: the output and
    every gradient equal those of the network without it bit for bit."""
    g = torch.Generator(device=DEV).manual_seed(5)
    out = []
    for fused_readout in (False, True):
        net = _net("joints", 2, 2, fused_train=True, fused_readout=fused_readout)
        if not out:
            pos = (2 * torch.rand(65, 3 * net.n_body * max(net.n_rot_pts, 1), device=DEV, generator=g) - 1) / 0.1
            x = torch.cat((torch.randn(65, net.in_dim, device=DEV, generator=g), pos), 1)
            dy = torch.randn(65, net.out_dim, device=DEV, generator=g)
        res = {}
        for amp in (True, False):
            for p in net.parameters():
                p.grad = None
            xi = x.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y = net(xi)
            y.float().backward(dy)
            res |= {f"{amp} y": y.detach(), f"{amp} x": xi.grad} | {f"{amp} {n}": p.grad.clone() for n, p in net.named_parameters()}
        out.append(res)
    for name, a in out[0].items():
        assert torch.equal(out[1][name], a), name


@GPU
def test_compiled_network_dispatch():
    """Traced by torch.compile, a frozen network with the option (the blocks' tensors kept, as --rf_bptt_keep_blocks: no
    checkpoint around the trunk) calls the read-out's kernel under bf16 autocast, and the default read-out when its
    parameters train or without autocast."""
    net = _net("joints", 2, 2, fused_train=True, fused_readout=True, checkpoint="frozen_readout")
    pos = (2 * torch.rand(65, 3 * net.n_body * max(net.n_rot_pts, 1), device=DEV) - 1) / 0.1
    x = torch.cat((torch.randn(65, net.in_dim, device=DEV), pos), 1)
    for frozen, amp, fused in ((True, True, True), (False, True, False), (True, False, False)):
        net.requires_grad_(not frozen)
        torch._dynamo.reset()
        ops = set()

        def backend(gm, _):
            ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
            return gm.forward

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            torch.compile(net, backend=backend)(x)
        assert any("rf_head_tok_fwd" in o for o in ops) == fused, (frozen, amp, sorted(ops)[:20])  # merged tokens: from the tokens' rows


def test_flag_mapping_and_refusals():
    """--rf_fused_readout reaches RigidFormerNet; it is refused without --rf_fused_train; the kernels refuse the shapes
    they do not serve (a SwiGLU width that is not a multiple of the chunks, a tile that is not whole environments; the
    read-out from the tokens' rows without merged tokens, with two chunk sizes, in tiles that are not whole
    environments)."""
    from ndm_gym.kernels.rf_head import HEAD_CFG, TOK_CFG, head_tables
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet, rigidformer_kwargs

    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=128, rf_layers=3, rf_heads=4,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False, amp="bf16")  # fmt: skip
    assert rigidformer_kwargs(a)["fused_readout"] is False
    assert rigidformer_kwargs({**a, "rf_fused_train": True, "rf_fused_readout": True})["fused_readout"] is True
    with pytest.raises(ValueError, match="rf_fused_train"):
        rigidformer_kwargs({**a, "rf_fused_readout": True})
    with pytest.raises(ValueError, match="fused_train"):
        RigidFormerNet(kinematics("cpu"), 6, 1, fused_readout=True)
    net = RigidFormerNet(kinematics("cpu"), 6, 1)
    assert not net.fused_readout and not net._fused_readout_now()
    tables = head_tables(net)
    assert tables["out_size"] == net.out_dim and len(tables["config"]) == 2 * len(HEAD_CFG)
    for bad in (dict(block_m=48), dict(bwd_block_m=16), dict(block_m=256), dict(chunk=48), dict(bwd_chunk=8), dict(bwd_chunk=128)):
        with pytest.raises(ValueError, match="fused read-out"):
            head_tables(net, **bad)
    # merged tokens are read out from the tokens' rows where the shapes allow it; asked for where they do not, it is refused
    assert not tables["token_rows"]
    merged = RigidFormerNet(kinematics("cpu"), 6, 1, layers=2, merge="joints")
    tables = head_tables(merged)
    assert tables["token_rows"] and len(tables["tok_config"]) == 2 * len(TOK_CFG) and not head_tables(merged, bwd_chunk=16)["token_rows"]
    for n, bad in ((net, {}), (merged, dict(bwd_chunk=16)), (merged, dict(tok_block_m=48)), (merged, dict(tok_block_m=256))):
        with pytest.raises(ValueError, match="read-out from the tokens' rows"):
            head_tables(n, token_rows=True, **bad)
