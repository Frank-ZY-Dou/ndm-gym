"""The fused RigidFormer kernels (ndm_gym/kernels/rf_block.py) against the reference on the GPU (skipped without
CUDA): one block over token counts, batch sizes (odd, not a multiple of the tile's environments) and options, the whole
HOIRigidFormerNet forward (in one kernel, or with only its blocks fused) and the world model's step at size RF-S, the
self-check. The reference runs as the world model runs it, under bf16 autocast; the fused path rounds to bf16 only at
the matrix products' inputs, so its distance to the fp32 reference must not exceed the autocast reference's own (both
measured on the same inputs).

With NDM_RF_CKPT (a trained HOI world model, fit_hoi.py's _wm.pt) and NDM_RF_DATA (a collect_hoi.py dataset of its
task), the network is also compared on the trained weights and recorded states.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a CUDA GPU")
DEV = "cuda"
# the whole-network kernel is refused on compute capability 9.x (FusedHOIRigidFormerNet: Triton 3.6 miscompiled it there)
# unless NDM_RF_WHOLE_ON_SM90=1, which runs its tests there to verify it
SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9 and os.environ.get("NDM_RF_WHOLE_ON_SM90") != "1"
WHOLE_ON_SM90 = "the whole-network kernel is refused on compute capability 9.x (Triton 3.6: shared-memory reads past the allocation, NaN)"


def _skip_whole_on_sm90(whole: bool):
    if whole and SM90:
        pytest.skip(WHOLE_ON_SM90)


def _errors(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.mean())


def _assert_at_bf16_rounding(fused, ref16, ref32, what, max_factor=1.0):
    """The fused path is at least as close to the fp32 reference as the bf16-autocast reference is: the mean error no
    larger, the maximum error at most ``max_factor`` times larger. Through a whole network the read-out heads run under
    autocast in both, so their rounding is common to both and the maximum can land on either side (factor 1.5)."""
    f, r = _errors(fused, ref32), _errors(ref16, ref32)
    assert torch.isfinite(fused).all(), what
    assert f[0] <= max_factor * r[0] and f[1] <= r[1], f"{what}: fused vs fp32 (max, mean) {f}, autocast vs fp32 {r}"
    return f, r


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _block(T, N, rope=True, film=True, bias=True, seed=0):
    from ndm_gym.worldmodels.rigidformer import Block, Rope3d

    torch.manual_seed(seed)
    blk = Block(128, 4, 2.5, 64 if film else 0).to(DEV)
    with torch.no_grad():  # FiLM starts at zero; norms at one
        if film:
            blk.film.weight.normal_(0.0, 0.1)
            blk.film.bias.normal_(0.0, 0.1)
        for m in (blk.norm1, blk.q_norm, blk.k_norm, blk.norm2):
            m.weight.uniform_(0.5, 1.5)
    x = torch.randn(N, T, 128, device=DEV) * (1 + 3 * torch.rand(N, T, 1, device=DEV))  # tokens of different scales
    pos = torch.cat((2 * torch.rand(N, T, 2, device=DEV) - 1, 1.8 * torch.rand(N, T, 1, device=DEV)), -1)  # metres
    cos, sin = Rope3d(24, 10.0).to(DEV)(pos) if rope else (None, None)
    b = 0.5 * torch.randn(1, 4, T, T, device=DEV) if bias else None
    c = torch.nn.functional.silu(torch.randn(N, 64, device=DEV)) if film else None
    return blk, (x, cos, sin, b, c)


@pytest.mark.parametrize("T,N", [(33, 1), (33, 7), (33, 1000), (33, 4096), (32, 1001), (34, 999), (17, 100), (64, 50), (100, 20), (128, 9)])
def test_block_matches_reference(T, N):
    from ndm_gym.kernels.rf_block import FusedBlock

    blk, args = _block(T, N, seed=T + N)
    with torch.no_grad():
        ref32 = _fp32(lambda: blk(*args))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16 = blk(*args)
            fused = FusedBlock(blk)(*args)  # autocast does not change the kernel
    _assert_at_bf16_rounding(fused, ref16, ref32, f"T={T} N={N}")
    assert fused.shape == ref32.shape and fused.dtype == torch.float32


@pytest.mark.parametrize("rope,film,bias", [(False, True, True), (True, False, True), (True, True, False), (False, False, False)])
def test_block_options(rope, film, bias):
    from ndm_gym.kernels.rf_block import FusedBlock

    blk, args = _block(33, 300, rope, film, bias, seed=1)
    with torch.no_grad():
        ref32 = _fp32(lambda: blk(*args))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16 = blk(*args)
        fused = FusedBlock(blk)(*args)
    _assert_at_bf16_rounding(fused, ref16, ref32, f"rope {rope} film {film} bias {bias}")


@pytest.mark.parametrize("kw", [dict(block_m=64), dict(hidden_chunk=32), dict(stages=3), dict(num_warps=4), dict(num_warps=16), dict(merge=1),
                                dict(merge=1, hidden_chunk=32, stages=3), dict(fast_exp=1), dict(merge=1, fast_exp=1), dict(merge=1, prefetch=1),
                                dict(merge=1, prefetch=1, fast_exp=1), dict(merge=2), dict(merge=2, fast_exp=1)])
def test_block_knobs(kw):
    """The tuning knobs change the schedule, not the result's precision. (256-row tiles need more shared memory than an
    A100 or an H100 has: 254 and 295 KB with two stages.)"""
    from triton.errors import TritonError

    from ndm_gym.kernels.rf_block import FusedBlock

    blk, args = _block(33, 500, seed=2)
    with torch.no_grad():
        ref32 = _fp32(lambda: blk(*args))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16 = blk(*args)
        try:
            fused = FusedBlock(blk, **kw)(*args)
        except TritonError as e:  # a setting this GPU cannot compile: out of shared memory (OutOfResources) or registers (PTXASError)
            pytest.skip(f"{kw} does not fit this GPU: {type(e).__name__}: {e}")
    _assert_at_bf16_rounding(fused, ref16, ref32, str(kw))


def test_unpipelined_kernel_refused():
    """Without software pipelining (stages 1), Triton 3.6 miscompiles the kernel on an H100; it is refused everywhere."""
    from ndm_gym.kernels.rf_block import FusedBlock

    blk, _ = _block(33, 1)
    with pytest.raises(ValueError, match="stages must be at least 2"):
        FusedBlock(blk, stages=1)


def test_self_check():
    """fused_copy compares every fused block with its Block first; a wrong kernel is caught."""
    from ndm_gym.kernels.rf_block import FusedBlock, self_check

    blk, _ = _block(33, 1, seed=6)
    fb = FusedBlock(blk)
    self_check(blk, fb, 33, 12, True)
    with torch.no_grad():
        fb.wp.mul_(1.5)  # a kernel that computes something else
    with pytest.raises(RuntimeError, match="miscompiled"):
        self_check(blk, fb, 33, 12, True)


def test_block_is_deterministic():
    from ndm_gym.kernels.rf_block import FusedBlock

    blk, args = _block(33, 2000, seed=3)
    fb = FusedBlock(blk)
    with torch.no_grad():
        assert torch.equal(fb(*args), fb(*args))


def test_block_refuses_what_it_does_not_do():
    from ndm_gym.kernels.rf_block import FusedBlock

    blk, (x, cos, sin, b, c) = _block(33, 4, seed=4)
    fb = FusedBlock(blk)
    with pytest.raises(RuntimeError, match="inference-only"):
        fb(x, cos, sin, b, c)  # gradients enabled
    with torch.no_grad():
        with pytest.raises(ValueError, match="one bias for all environments"):
            fb(x, cos, sin, b.expand(4, -1, -1, -1), c)
        with pytest.raises(ValueError, match="FiLM"):
            fb(x, cos, sin, b, None)
        with pytest.raises(ValueError, match="do not fit"):
            FusedBlock(blk, block_m=32)(x, cos, sin, b, c)


def _world_model(seed=0, props=0, hinged=False):
    from ndm_gym.robots.g1 import G1_JOINT_NAMES, NUM_J
    from ndm_gym.tasks.hdmi import JOINT_NAMES, NUM_ACTIONS
    from ndm_gym.worldmodels.hoi import ContextLayout, HOIWorldModel, rigidformer_args

    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    consts = {"eef_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "q0": [0.05 * (i % 5) for i in range(NUM_J)],
              "act_scale": [0.4] * NUM_ACTIONS, "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip
    torch.manual_seed(seed)
    wm = HOIWorldModel("accel", hinged, 2, ContextLayout(55, hinged, props), rigidformer=rigidformer_args(consts, 128, 3, 4, 2))
    with torch.no_grad():  # the heads, FiLM and the relation bias start at zero; give every path a signal
        for m in [wm.net.root_head, wm.net.joint_head, wm.net.obj_head, wm.net.eef_head, wm.net.feet_head] + [b.film for b in wm.net.blocks]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        wm.net.rel_bias.normal_(0.0, 0.3)
    return wm.to(DEV)


def _scene(wm, n, seed=0):
    """Plausible states of a standing robot with an object near its hands, in metres and radians."""
    from ndm_gym.core.rotations import so3_exp, yaw_mat
    from ndm_gym.robots.g1 import NUM_J
    from ndm_gym.tasks.hdmi import DECIMATION, NUM_ACTIONS

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    R = yaw_mat(so3_exp(torch.cat((0.05 * r(n, 2), 3.0 * r(n, 1)), -1)))
    S = (torch.cat((r(n, 2), 0.78 + 0.02 * r(n, 1)), -1), R @ so3_exp(0.05 * r(n, 3)), 0.3 * r(n, 3), 0.3 * r(n, 3), 0.1 * r(n, NUM_J),
         0.5 * r(n, NUM_J))  # fmt: skip
    if wm.hinged:
        So = (0.3 * r(n, 1), 0.3 * r(n, 1))
    else:
        So = (S[0] + torch.tensor([0.4, 0.0, 0.1]) + 0.05 * r(n, 3), so3_exp(r(n, 3)), 0.1 * r(n, 3), 0.1 * r(n, 3))
    lay = wm.layout
    poses = [torch.cat((S[0] + r(n, 3), so3_exp(r(n, 3)).flatten(1)), -1) for _ in range(int(lay.base) + lay.props)]
    ctx = torch.cat([1.0 + 0.1 * r(n, lay.scalars)] + poses, -1)
    to = lambda x: x.to(DEV)  # noqa: E731
    return tuple(map(to, S)), tuple(map(to, So)), to(0.2 * r(n, DECIMATION, NUM_ACTIONS)), to(ctx)


def _network_inputs(wm, n, seed=0):
    """Normalized inputs at the scale the network sees: the input normalizer fit on the same kind of states."""
    with torch.no_grad():
        for k in range(4):
            wm.in_norm.update(wm.inputs(*_scene(wm, 2048, seed=100 + k)))
        return wm.in_norm(wm.inputs(*_scene(wm, n, seed)))


@pytest.mark.parametrize("whole", [True, False])
@pytest.mark.parametrize("n,props,hinged", [(1, 0, False), (333, 0, False), (2048, 0, False), (257, 2, False), (200, 1, True)])
def test_network_matches_reference(n, props, hinged, whole):
    """The whole network in one kernel (embedding, blocks, read-out: closer to fp32 than autocast), or only its blocks
    fused (the read-out under autocast in both: factor 1.5 on the maximum)."""
    from ndm_gym.kernels.rf_block import FusedBlock, FusedHOIRigidFormerNet, fused_copy
    from ndm_gym.worldmodels.hoi import HOIRigidFormerNet
    from ndm_gym.worldmodels.rigidformer import Block

    _skip_whole_on_sm90(whole)
    wm = _world_model(props=props, hinged=hinged)
    x = _network_inputs(wm, n)
    fused = fused_copy(wm.net, whole_network=whole)
    assert type(wm.net) is HOIRigidFormerNet and all(isinstance(b, Block) for b in wm.net.blocks)  # the original is left as it is
    assert isinstance(fused, FusedHOIRigidFormerNet) if whole else all(isinstance(b, FusedBlock) for b in fused.blocks)
    with torch.no_grad():
        ref32 = _fp32(lambda: wm.net(x))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16, y = wm.net(x).float(), fused(x).float()
        if whole:
            assert torch.equal(fused(x), y)  # deterministic
    _assert_at_bf16_rounding(y, ref16, ref32, f"network n={n} props={props} hinged={hinged} whole={whole}", max_factor=1.0 if whole else 1.5)


@pytest.mark.parametrize("merge,prefetch", [(1, 0), (1, 1), (2, 0)])
@pytest.mark.parametrize("whole", [True, False])
def test_network_merged_products(whole, merge, prefetch):
    """The merged products (merge 1: one per head for q, k, v and the gate; 2: two; one per chunk for the SwiGLU), with
    the next head's product issued ahead or not, through a network."""
    from ndm_gym.kernels.rf_block import fused_copy

    _skip_whole_on_sm90(whole)
    wm = _world_model(props=1)
    x = _network_inputs(wm, 500)
    fused = fused_copy(wm.net, whole_network=whole, merge=merge, prefetch=prefetch)
    with torch.no_grad():
        ref32 = _fp32(lambda: wm.net(x))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16, y = wm.net(x).float(), fused(x).float()
    _assert_at_bf16_rounding(y, ref16, ref32, f"merged products whole={whole} merge={merge} prefetch={prefetch}", max_factor=1.0 if whole else 1.5)


def test_fused_network_refuses_what_it_does_not_do():
    from ndm_gym.kernels.rf_block import FusedHOIRigidFormerNet, fused_copy, self_check

    wm = _world_model()
    if SM90:
        with pytest.raises(RuntimeError, match="refused on compute capability 9"):
            fused_copy(wm.net, whole_network=True)
        pytest.skip(WHOLE_ON_SM90)
    fused = fused_copy(wm.net, whole_network=True)
    x = _network_inputs(wm, 3)
    with pytest.raises(RuntimeError, match="inference-only"):
        fused(x)
    with torch.no_grad(), pytest.raises(ValueError, match="must be an fp32"):
        fused(x[:, 1:])
    with pytest.raises(TypeError, match="HOIRigidFormerNet"):
        FusedHOIRigidFormerNet.from_net(fused)
    with torch.no_grad():
        fused.k_wout.mul_(1.5)  # a kernel that computes something else
    with pytest.raises(RuntimeError, match="miscompiled"):
        self_check(wm.net, fused, 0, 0, True)


@pytest.mark.parametrize("whole", [False, True])
def test_world_model_step_matches_reference(whole):
    """HOIWorldModel.step of the fused copy (as wmbench/speed_hoi.py runs it, compiled with CUDA graphs) and of the
    original under bf16 autocast, both against the original in fp32, over a few open-loop steps (the read-out under
    autocast with only the blocks fused: factor 1.5 on the maximum)."""
    import ndm_gym.core.nets as nets
    from ndm_gym.kernels.rf_block import FusedBlock, FusedHOIRigidFormerNet, fused_copy

    _skip_whole_on_sm90(whole)
    wm = _world_model(seed=1)
    _network_inputs(wm, 1)  # fit the input normalizer
    fused = fused_copy(wm, whole_network=whole)
    assert isinstance(fused.net, FusedHOIRigidFormerNet) if whole else all(isinstance(b, FusedBlock) for b in fused.net.blocks)
    S, So, u, ctx = _scene(wm, 1500, seed=5)
    step = torch.compile(fused.step, mode="reduce-overhead")
    amp = nets.AMP_DTYPE
    try:
        with torch.no_grad():
            nets.AMP_DTYPE = None
            ref32 = _fp32(lambda: wm.rollout(S, So, u.unsqueeze(1).expand(-1, 3, -1, -1), ctx.unsqueeze(1).expand(-1, 3, -1)))
            nets.AMP_DTYPE = torch.bfloat16
            ref16 = wm.rollout(S, So, u.unsqueeze(1).expand(-1, 3, -1, -1), ctx.unsqueeze(1).expand(-1, 3, -1))
            Sf, Sof, out = S, So, []
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                Sf, Sof, Ff = step(Sf, Sof, u, ctx)
                Sf, Sof, Ff = (tuple(t.clone() for t in z) for z in (Sf, Sof, Ff))
                out.append((Sf, Sof, Ff))
    finally:
        nets.AMP_DTYPE = amp
    flat = lambda steps: torch.cat([t.flatten(1) for s in steps for part in s for t in part], 1)  # noqa: E731
    _assert_at_bf16_rounding(flat(out), flat(ref16), flat(ref32), f"world model rollout whole={whole}", max_factor=1.0 if whole else 1.5)


@pytest.mark.parametrize("whole", [False, True])
def test_inference_copy_with_fused_kernels(whole):
    """The world model's inference copy (fp32 parameters under autocast, token attention, merged read-out heads, the
    step outside the network as Triton kernels: HOIWorldModel.inference_copy(None, fused=True)) with its network or
    blocks fused, its compiled step against the original model in fp32 and under bf16 autocast; a bf16 copy refused."""
    import ndm_gym.core.nets as nets
    from ndm_gym.kernels.rf_block import FusedBlock, FusedHOIRigidFormerNet, fused_copy

    _skip_whole_on_sm90(whole)
    wm = _world_model(seed=2)
    _network_inputs(wm, 1)  # fit the input normalizer
    with pytest.raises(ValueError, match="fp32 parameters"):
        fused_copy(wm.inference_copy(torch.bfloat16, "token"), whole_network=whole)
    fused = fused_copy(wm.inference_copy(None, "token", fused=True), whole_network=whole)
    assert fused.fused and (isinstance(fused.net, FusedHOIRigidFormerNet) if whole else all(isinstance(b, FusedBlock) for b in fused.net.blocks))
    S, So, u, ctx = _scene(wm, 1500, seed=6)
    step = torch.compile(fused.step, mode="reduce-overhead")
    amp = nets.AMP_DTYPE
    try:
        with torch.no_grad():
            nets.AMP_DTYPE = None
            ref32 = _fp32(lambda: wm.rollout(S, So, u.unsqueeze(1).expand(-1, 3, -1, -1), ctx.unsqueeze(1).expand(-1, 3, -1)))
            nets.AMP_DTYPE = torch.bfloat16
            ref16 = wm.rollout(S, So, u.unsqueeze(1).expand(-1, 3, -1, -1), ctx.unsqueeze(1).expand(-1, 3, -1))
            Sf, Sof, out = S, So, []
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                Sf, Sof, Ff = step(Sf, Sof, u, ctx)
                Sf, Sof, Ff = (tuple(t.clone() for t in z) for z in (Sf, Sof, Ff))
                out.append((Sf, Sof, Ff))
    finally:
        nets.AMP_DTYPE = amp
    flat = lambda steps: torch.cat([t.flatten(1) for s in steps for part in s for t in part], 1)  # noqa: E731
    _assert_at_bf16_rounding(flat(out), flat(ref16), flat(ref32), f"inference copy rollout whole={whole}", max_factor=1.0 if whole else 1.5)


@pytest.mark.parametrize("rope,tree_bias,film", [(True, True, True), (False, False, False)])
def test_lafan1_backbone(rope, tree_bias, film):
    """The LAFAN1 world models' RigidFormerNet (30 bodies and 2 registers: 4 environments fill a 128-row tile) with its
    blocks fused, with and without rotation, tree bias and FiLM."""
    from ndm_gym.kernels.rf_block import FusedBlock, fused_copy
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(0)
    net = RigidFormerNet(kinematics("cpu"), 6, 1, rope=rope, tree_bias=tree_bias, film=film).to(DEV)
    with torch.no_grad():  # the heads, FiLM and the tree bias start at zero; give every path a signal
        for m in [net.root_head, net.joint_head] + ([b.film for b in net.blocks] if film else []):
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        if tree_bias:
            net.rel_bias.normal_(0.0, 0.3)
    n = 1001
    x = torch.cat((torch.randn(n, net.in_dim, device=DEV), (2 * torch.rand(n, 3 * net.n_body, device=DEV) - 1) / 0.1), 1)
    fused = fused_copy(net)
    assert all(isinstance(b, FusedBlock) for b in fused.blocks)
    with torch.no_grad():
        ref32 = _fp32(lambda: net(x))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16, y = net(x).float(), fused(x).float()
    _assert_at_bf16_rounding(y, ref16, ref32, f"RigidFormerNet rope={rope} tree_bias={tree_bias} film={film}", max_factor=1.5)


@pytest.mark.skipif(not (os.environ.get("NDM_RF_CKPT") and os.environ.get("NDM_RF_DATA")), reason="NDM_RF_CKPT and NDM_RF_DATA not set")
def test_trained_network_on_recorded_states():
    from ndm_gym.kernels.rf_block import fused_copy
    from ndm_gym.sim.rigid.hdmi_sim import ndm_constants
    from ndm_gym.state import unflatten_object, unflatten_state
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.worldmodels.hoi import ContextLayout, HOIWorldModel, rigidformer_args

    ck = torch.load(os.environ["NDM_RF_CKPT"], map_location="cpu")
    a, layout = ck["args"], ContextLayout(*ck["layout"])
    cfg = TASKS[a["task"]]
    rf = rigidformer_args(ndm_constants(cfg), a["rf_dim"], a["rf_layers"], a["rf_heads"], a["rf_registers"])
    hinged = cfg.object_joint is not None
    wm = HOIWorldModel(a["kind"], hinged, cfg.num_eefs, layout, rigidformer=rf)
    wm.load_state_dict(ck["wm"])
    wm = wm.to(DEV)
    d = torch.load(os.environ["NDM_RF_DATA"], map_location="cpu", mmap=True)["data"]
    t = torch.arange(8) * (d["s"].shape[0] // 8)  # 8 times x 512 environments of the recording
    pick = lambda k: d[k][t][:, :512].flatten(0, 1).clone().to(DEV)  # noqa: E731
    o = pick("o")
    So = (o[..., :1], o[..., 1:2]) if hinged else unflatten_object(o)
    blocks = fused_copy(wm.net)
    with torch.no_grad():
        x = wm.in_norm(wm.inputs(unflatten_state(pick("s")), So, pick("u"), pick("ctx")))
        ref32 = _fp32(lambda: wm.net(x))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16, yb = wm.net(x).float(), blocks(x).float()
    fb, r = _assert_at_bf16_rounding(yb, ref16, ref32, "trained network, blocks fused", max_factor=1.5)
    msg = f"blocks fused {fb[0]:.3e} {fb[1]:.3e}; autocast {r[0]:.3e} {r[1]:.3e}; output scale {float(ref32.abs().max()):.3f}"
    if not SM90:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = fused_copy(wm.net, whole_network=True)(x).float()
        f, _ = _assert_at_bf16_rounding(y, ref16, ref32, "trained network")
        msg = f"whole network fused {f[0]:.3e} {f[1]:.3e}; " + msg
    print(f"\ntrained network, {len(x)} recorded states, vs fp32 (max, mean): {msg}")
