"""The G1 world models' inference paths (ndm_gym/worldmodels/g1_infer.py, g1_fused.py; kernels/rf_block.py's
FusedG1RigidFormerNet and fused_copy on a RigidFormerNet): the fused features and integration against the models' own
``inputs`` and ``step`` (float rounding), the bf16 copy, the fused blocks and the whole-network kernel against the model as
trained (no further from its fp32 values than the model under bf16 autocast), the refusals, merged tokens (--rf_merge:
fused blocks; the whole-network kernel refuses them), and on the CPU an fp32 copy that computes the same function bit for
bit. The GPU tests are skipped without CUDA."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

DEV = "cuda"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the inference paths' kernels need a CUDA GPU")
needs_isaaclab = pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="speed_pair.py imports Isaac Lab's packages")
# the whole-network kernel is refused on compute capability 9.x (Triton 3.6 miscompiled it there) unless NDM_RF_WHOLE_ON_SM90=1
SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9 and os.environ.get("NDM_RF_WHOLE_ON_SM90") != "1"


def _skip_whole_on_sm90(whole: bool):
    if whole and SM90:
        pytest.skip("the whole-network kernel is refused on compute capability 9.x")


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _states(n, seed=0, dev=DEV):
    """Plausible states of a G1 standing or moving, setpoints and normalized contexts."""
    from ndm_gym.core.rotations import so3_exp, yaw_mat
    from ndm_gym.robots.g1 import NUM_J
    from ndm_gym.state import XI_DIM

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    R = yaw_mat(so3_exp(torch.cat((0.05 * r(n, 2), 3.0 * r(n, 1)), -1))) @ so3_exp(0.1 * r(n, 3))
    S = (torch.cat((r(n, 2), 0.78 + 0.03 * r(n, 1)), -1), R, 0.5 * r(n, 3), 0.5 * r(n, 3), 0.3 * r(n, NUM_J), r(n, NUM_J))
    return tuple(t.to(dev).contiguous() for t in S), (0.3 * r(n, NUM_J)).to(dev), r(n, XI_DIM).to(dev)


def _world_model(kind="accel", rf=True, seed=0, dev=DEV, **rf_kw):
    """A G1 world model with a signal on every path (heads, FiLM, relation bias, the MLP's last layer start at zero) and
    its normalizers fit on plausible states."""
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    torch.manual_seed(seed)
    rigid = dict(kin=kinematics("cpu"), **{"dim": 128, "layers": 3, "heads": 4, "registers": 2, **rf_kw}) if rf else None
    hidden = (256, 256)
    wm = AccelWorldModel(hidden=hidden, rigidformer=rigid) if kind == "accel" else IncrementWorldModel(hidden=hidden, rigidformer=rigid)
    with torch.no_grad():
        if rf:
            for m in [wm.net.root_head, wm.net.joint_head] + [b.film for b in wm.net.blocks]:
                m.weight.normal_(0.0, 0.05)
                m.bias.normal_(0.0, 0.05)
            wm.net.rel_bias.normal_(0.0, 0.3)
        else:
            wm.net[-1].weight.normal_(0.0, 0.02)
            wm.net[-1].bias.normal_(0.0, 0.02)
    wm = wm.to(dev)
    with torch.no_grad():
        for k in range(3):
            S, a, xi_n = _states(2048, seed=100 + k, dev=dev)
            wm.in_norm.update(wm.inputs(S, a, xi_n))
            S1 = tuple(t + 0.01 * torch.randn_like(t) for t in S)
            wm.out_norm.update(wm.targets(S, (S1[0], S[1], *S1[2:])))
    return wm.requires_grad_(False)


def _rollout(step, S, a, xi_n, k=3, graphs=False):
    out = []
    with torch.no_grad():
        for _ in range(k):
            if graphs:
                torch.compiler.cudagraph_mark_step_begin()
            S = tuple(t.clone() for t in step(S, a, xi_n))
            out.append(S)
    return torch.cat([t.flatten(1) for s in out for t in s], 1)


def _references(wm, S, a, xi_n, k=3):
    """The model's rollout in fp32 and under bf16 autocast (as trained)."""
    import ndm_gym.core.nets as nets

    amp = nets.AMP_DTYPE
    try:
        nets.AMP_DTYPE = None
        ref32 = _fp32(lambda: _rollout(wm.step, S, a, xi_n, k))
        nets.AMP_DTYPE = torch.bfloat16
        ref16 = _rollout(wm.step, S, a, xi_n, k)
    finally:
        nets.AMP_DTYPE = amp
    return ref32, ref16


def _assert_at_bf16_rounding(y, ref16, ref32, what, max_factor=1.0, mean_factor=1.0):
    d = (y - ref32).abs()
    r = (ref16 - ref32).abs()
    assert torch.isfinite(y).all(), what
    fd, fr = (float(d.max()), float(d.mean())), (float(r.max()), float(r.mean()))
    msg = f"{what}: path vs fp32 (max, mean) {fd[0]:.3g}, {fd[1]:.3g}; autocast vs fp32 {fr[0]:.3g}, {fr[1]:.3g}"
    assert fd[0] <= max_factor * fr[0] and fd[1] <= mean_factor * fr[1], msg


@gpu
@pytest.mark.parametrize("rf", [True, False])
def test_fused_features_match_inputs(rf):
    """The fused features kernel (forward kinematics and token features, or the MLP's invariant features) equals
    ``in_norm(inputs(...))`` up to float rounding, at batch sizes that are not multiples of the kernel's block."""
    from ndm_gym.worldmodels.g1_infer import inference_copy

    wm = _world_model(rf=rf)
    m = inference_copy(wm, None, "sdpa" if rf else None, fused=True)
    for n in (1, 63, 1000):
        S, a, xi_n = _states(n, seed=n)
        with torch.no_grad():
            ref = _fp32(lambda: wm.in_norm(wm.inputs(S, a, xi_n)))
            got = _fp32(lambda: m.features(S, a, xi_n))
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("kind,rf", [("accel", True), ("increment", True), ("accel", False), ("increment", False)])
def test_fused_step_matches_step(kind, rf):
    """With fp32 everywhere (no autocast, TF32 off, sdpa attention) the fused step (features, network, integration)
    equals the model's step up to float rounding over a few open-loop steps; also with SuperTrack's 4 substeps."""
    import ndm_gym.core.nets as nets
    from ndm_gym.worldmodels.g1_infer import inference_copy

    wm = _world_model(kind, rf)
    m = inference_copy(wm, None, "sdpa" if rf else None, fused=True)
    S, a, xi_n = _states(777, seed=3)
    amp = nets.AMP_DTYPE
    try:
        nets.AMP_DTYPE = None
        ref = _fp32(lambda: _rollout(wm.step, S, a, xi_n))
        got = _fp32(lambda: _rollout(m.step, S, a, xi_n))
        if kind == "accel":
            wm.pos_coef = m.wm.pos_coef = (4 + 1) / (2 * 4) * 0.02  # AccelWorldModel(substeps=4)'s
            ref4 = _fp32(lambda: _rollout(wm.step, S, a, xi_n))
            got4 = _fp32(lambda: _rollout(m.step, S, a, xi_n))
            torch.testing.assert_close(got4, ref4, rtol=1e-4, atol=1e-4)
    finally:
        nets.AMP_DTYPE = amp
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("kind,rf", [("accel", True), ("increment", True), ("accel", False)])
def test_bf16_copy(kind, rf):
    """The bf16 copy (the network bf16 end to end, token attention, the fused step), compiled with CUDA graphs: within
    the documented bound of the model under bf16 autocast (check_copy's: twice its mean, four times its largest error)."""
    from ndm_gym.worldmodels.g1_infer import inference_copy

    wm = _world_model(kind, rf, seed=1)
    m = inference_copy(wm, torch.bfloat16, "token" if rf else None, fused=True)
    S, a, xi_n = _states(1500, seed=5)
    ref32, ref16 = _references(wm, S, a, xi_n)
    y = _rollout(torch.compile(m.step, mode="reduce-overhead"), S, a, xi_n, graphs=True)
    _assert_at_bf16_rounding(y, ref16, ref32, f"bf16 copy {kind} rf={rf}", max_factor=4.0, mean_factor=2.0)


@gpu
@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_fused_kernels(whole, copy):
    """The RigidFormer's blocks or whole network as fused kernels (fused_copy) in the model as trained or in its fp32
    inference copy with the fused step, compiled with CUDA graphs, against the model as trained (the read-out under
    autocast with only the blocks fused: factor 1.5 on the maximum)."""
    from ndm_gym.kernels.rf_block import FusedBlock, FusedG1RigidFormerNet, fused_copy
    from ndm_gym.worldmodels.g1_infer import inference_copy

    _skip_whole_on_sm90(whole)
    wm = _world_model(seed=2)
    base = inference_copy(wm, None, "token", fused=True) if copy else wm
    m = fused_copy(base, whole_network=whole)
    net = m.wm.net if copy else m.net
    assert isinstance(net, FusedG1RigidFormerNet) if whole else all(isinstance(b, FusedBlock) for b in net.blocks)
    S, a, xi_n = _states(1001, seed=6)
    ref32, ref16 = _references(wm, S, a, xi_n)
    y = _rollout(torch.compile(m.step, mode="reduce-overhead"), S, a, xi_n, graphs=True)
    _assert_at_bf16_rounding(y, ref16, ref32, f"fused kernels whole={whole} copy={copy}", max_factor=1.5, mean_factor=1.0)


@gpu
def test_fused_blocks_with_merged_tokens():
    """Networks with merged tokens (--rf_merge joints: 14 + 2 tokens; 128 and 64 wide) in their fp32 inference copy with
    the fused step and the blocks as fused kernels, and in their bf16 copy, compiled with CUDA graphs, against the model
    as trained; the whole-network kernel, which embeds and reads out one body per token, refuses them."""
    from ndm_gym.kernels.rf_block import FusedBlock, FusedG1RigidFormerNet, fused_copy
    from ndm_gym.worldmodels.g1_infer import inference_copy

    S, a, xi_n = _states(1001, seed=7)
    for kw in ({}, {"dim": 64, "heads": 2, "layers": 2}):  # 64 wide: the SwiGLU's 160 columns in chunks of 32
        wm = _world_model(seed=3, merge="joints", **kw)
        m = fused_copy(inference_copy(wm, None, "token", fused=True))
        assert wm.net.n_tok == 14 and all(isinstance(b, FusedBlock) for b in m.wm.net.blocks)
        ref32, ref16 = _references(wm, S, a, xi_n)
        y = _rollout(torch.compile(m.step, mode="reduce-overhead"), S, a, xi_n, graphs=True)
        _assert_at_bf16_rounding(y, ref16, ref32, f"fused blocks, merged tokens {kw}", max_factor=1.5, mean_factor=1.0)
        # the bf16 copy (the network bf16 end to end, token attention, the fused step), as test_bf16_copy
        b16 = inference_copy(wm, torch.bfloat16, "token", fused=True)
        y = _rollout(torch.compile(b16.step, mode="reduce-overhead"), S, a, xi_n, graphs=True)
        _assert_at_bf16_rounding(y, ref16, ref32, f"bf16 copy, merged tokens {kw}", max_factor=4.0, mean_factor=2.0)
    if not SM90:
        with pytest.raises(ValueError, match="merged tokens"):
            FusedG1RigidFormerNet.from_net(wm.net)


@gpu
@pytest.mark.parametrize("n", [1, 333, 2048])
def test_whole_network_matches_reference(n):
    """FusedG1RigidFormerNet (one kernel: the embedding, the blocks, the read-out) against the network: closer to its fp32
    values than the network under bf16 autocast; deterministic."""
    from ndm_gym.kernels.rf_block import FusedG1RigidFormerNet, fused_copy
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    _skip_whole_on_sm90(True)
    wm = _world_model(seed=3)
    S, a, xi_n = _states(n, seed=n)
    with torch.no_grad():
        x = wm.in_norm(wm.inputs(S, a, xi_n))
    fused = fused_copy(wm.net, whole_network=True)
    assert isinstance(fused, FusedG1RigidFormerNet) and type(wm.net) is RigidFormerNet
    with torch.no_grad():
        ref32 = _fp32(lambda: wm.net(x))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16, y = wm.net(x).float(), fused(x).float()
        assert torch.equal(fused(x), y)
    _assert_at_bf16_rounding(y, ref16, ref32, f"whole network n={n}", max_factor=1.0, mean_factor=1.0)


@gpu
def test_fused_copy_turns_off_the_training_kernels():
    """A model trained with --rf_fused_train: its fused copy runs the FusedBlocks (fused_train off in the copy only)."""
    from ndm_gym.kernels.rf_block import FusedBlock, fused_copy

    wm = _world_model(seed=4, fused_train=True)
    m = fused_copy(wm)
    assert wm.net.fused_train and not m.net.fused_train and all(isinstance(b, FusedBlock) for b in m.net.blocks)
    S, a, xi_n = _states(100, seed=7)
    ref32, ref16 = _references(wm, S, a, xi_n)
    import ndm_gym.core.nets as nets

    amp = nets.AMP_DTYPE
    try:
        nets.AMP_DTYPE = torch.bfloat16
        y = _rollout(m.step, S, a, xi_n)
    finally:
        nets.AMP_DTYPE = amp
    _assert_at_bf16_rounding(y, ref16, ref32, "fused copy of a fused_train model", max_factor=1.5, mean_factor=1.0)


@gpu
def test_refusals():
    from ndm_gym.kernels.rf_block import FusedG1RigidFormerNet, fused_copy
    from ndm_gym.worldmodels.g1_infer import check_copy, inference_copy

    wm, mlp = _world_model(seed=5), _world_model(rf=False, seed=5)
    with pytest.raises(ValueError, match="fp32 parameters"):
        fused_copy(inference_copy(wm, torch.bfloat16, "token"))
    with pytest.raises(ValueError, match="the MLP None"):
        inference_copy(mlp, None, "token")
    with pytest.raises(ValueError, match="the RigidFormer takes"):
        inference_copy(wm, None, None)
    with pytest.raises(TypeError, match="AccelWorldModel or IncrementWorldModel"):
        inference_copy(wm.net)
    with pytest.raises(ValueError, match="fuses the RigidFormer|no RigidFormerNet"):
        fused_copy(mlp)
    m = inference_copy(wm, None, "token", fused=True)
    S, a, xi_n = _states(4)
    with pytest.raises(RuntimeError, match="inference-only"):
        m.step(S, a, xi_n)
    with torch.no_grad(), pytest.raises(ValueError, match="input noise"):
        m.step(S, a, xi_n, input_noise=0.1)
    anchors = _world_model(seed=5, anchors=4)
    with pytest.raises(ValueError, match="anchors"):
        inference_copy(anchors, None, "token", fused=True)
    if not SM90:
        with pytest.raises(ValueError, match="anchors"):
            FusedG1RigidFormerNet.from_net(anchors.net)
    m.in_scale.mul_(1.01)  # a features path that computes something else
    with pytest.raises(RuntimeError, match="fused features differ"):
        check_copy(wm, m)


def _path(**kw):
    return SimpleNamespace(**({"compile": "none", "inference_copy": "none", "rf_attention": "token", "fused": False, "rf_kernels": "none"} | kw))


@needs_isaaclab
def test_speed_pair_paths():
    """speed_pair.py's inference_model: --rf_kernels refused on the MLP, --fused without a copy refused, the paths
    described (on the CPU, an fp32 copy with sdpa attention: no kernels)."""
    from ndm_gym.wmbench.speed_pair import inference_model

    mlp, rf = _world_model(rf=False, dev="cpu"), _world_model(dev="cpu")
    for kernels in ("blocks", "whole"):
        with pytest.raises(ValueError, match="the MLP backbone has neither"):
            inference_model(mlp, _path(rf_kernels=kernels))
    with pytest.raises(ValueError, match="--fused needs --inference_copy"):
        inference_model(rf, _path(fused=True))
    m, _, path = inference_model(rf, _path())
    assert m is rf and path == "compile none"
    m, _, path = inference_model(rf, _path(inference_copy="amp", rf_attention="sdpa"))
    assert m is not rf and path == "compile none inference copy amp attention sdpa"


@gpu
@needs_isaaclab
@pytest.mark.parametrize("kernels", ["blocks", "whole"])
def test_speed_pair_kernels(kernels):
    """speed_pair.py's --rf_kernels on the model as trained and on its fp32 copy with the fused step; refused on a bf16
    copy (the kernels take fp32 parameters) and, for the whole network, on compute capability 9.x."""
    from ndm_gym.kernels.rf_block import FusedBlock, FusedG1RigidFormerNet
    from ndm_gym.wmbench.speed_pair import inference_model

    wm = _world_model(seed=7)
    with pytest.raises(ValueError, match="fp32 parameters"):
        inference_model(wm, _path(inference_copy="bf16", fused=True, rf_kernels=kernels))
    if kernels == "whole" and SM90:
        with pytest.raises(RuntimeError, match="refused on compute capability 9"):
            inference_model(wm, _path(rf_kernels=kernels))
        return
    for copy in ("none", "amp"):
        m, _, path = inference_model(wm, _path(inference_copy=copy, fused=copy == "amp", rf_kernels=kernels))
        assert isinstance(m.net, FusedG1RigidFormerNet) if kernels == "whole" else all(isinstance(b, FusedBlock) for b in m.net.blocks)
        assert path.endswith(f" rf_kernels {kernels}")


@pytest.mark.parametrize("merge", ["none", "joints"])
def test_fp32_copy_computes_the_same_function_on_cpu(merge):
    """On the CPU (no kernels): the fp32 inference copy with sdpa attention and without the fused step runs the model's
    own step on an InferenceRigidFormerNet, bit for bit the model's (also with merged tokens), and leaves the model as it
    was."""
    from ndm_gym.worldmodels.g1_infer import InferenceRigidFormerNet, inference_copy
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    for kind in ("accel", "increment"):
        wm = _world_model(kind, seed=6, dev="cpu", merge=merge)
        m = inference_copy(wm, None, "sdpa")
        assert type(wm.net) is RigidFormerNet and type(m.wm.net) is InferenceRigidFormerNet
        S, a, xi_n = _states(50, seed=8, dev="cpu")
        with torch.no_grad():
            ref, got = wm.step(S, a, xi_n), m.step(S, a, xi_n)
        assert all(torch.equal(x, y) for x, y in zip(ref, got)), kind
