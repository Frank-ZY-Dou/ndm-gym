"""The whole-network kernel of the MLP world models (ndm_gym/kernels/mlp_fused.py): each precision against the network computed in fp64,
held to its reference (``check``: at most 2x the reference's distance on average and 4x at the largest; TF32 products emulated in
fp64, torch's bf16 run, torch's fp32 run), over several network shapes, sample counts and input strides; the kernel kept whole in a
compiled function; the inference-only and unsupported-network refusals. The GPU tests are skipped without CUDA; TF32 and bf16 need
an Ampere or newer GPU."""

import pytest
import torch
import torch.nn as nn

DEV = "cuda"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on a CUDA GPU")
NETS = {  # in, hidden, out, activation, LayerNorm
    "quad_accel": (13, [256] * 3, 6, "silu", True),
    "quad_increment": (13, [256] * 3, 12, "silu", True),
    "elu_128": (37, [128] * 2, 20, "elu", False),
    "relu_512": (64, [512] * 2, 16, "relu", True),
    "one_layer": (13, [64], 3, "silu", True),
}


def _net(name, seed=0):
    from ndm_gym.core.nets import ACTIVATIONS, mlp

    inp, hidden, out, act, ln = NETS[name]
    torch.manual_seed(seed)
    net = mlp(inp, hidden, out, act=ACTIVATIONS[act], layer_norm=ln).to(DEV).eval()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.LayerNorm):  # away from the identity, as trained ones are
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)
    return net


def _dots():
    if not torch.cuda.is_available():
        return []
    return ["ieee"] + (["tf32", "bf16"] if torch.cuda.get_device_capability()[0] >= 8 else [])


@gpu
@pytest.mark.parametrize("name", list(NETS))
@pytest.mark.parametrize("dot", _dots())
def test_kernel_within_torch_precision(name, dot):
    from ndm_gym.kernels.mlp_fused import FusedMLP, check

    net = _net(name)
    fm = FusedMLP(net, dot)
    g = torch.Generator(device=DEV).manual_seed(1)
    for n in (1, 100, 4103):  # the small samples' tiles and the large ones' (config)
        x = 2 * torch.randn(n, net[0].in_features, device=DEV, generator=g)
        check(net, fm, x)
        check(net, fm, x.t().contiguous().t())  # features-major, as the quadrotor's fused features hand them on


@gpu
@pytest.mark.parametrize("bm,ch", [(32, 64), (64, 64), (16, 128), (32, 128)])
def test_tile_sizes_agree(bm, ch):
    """Every tile size computes the same values within fp32 rounding (exact products; the sums' order may differ)."""
    from ndm_gym.kernels.mlp_fused import FusedMLP

    net = _net("quad_accel")
    x = torch.randn(5000, 13, device=DEV, generator=torch.Generator(device=DEV).manual_seed(2))
    with torch.no_grad():
        ref = FusedMLP(net, "ieee", bm=16, ch=64)(x)
        assert torch.allclose(FusedMLP(net, "ieee", bm=bm, ch=ch)(x), ref, rtol=1e-5, atol=1e-5)


@gpu
def test_compiled_step_keeps_the_kernel_whole():
    """FusedMLP inside a torch.compile'd function with fullgraph (no graph break), as a planner's compiled step calls it: the kernel is one
    registered operator, and the compiled call gives the eager call's values."""
    from ndm_gym.kernels.mlp_fused import FusedMLP

    net = _net("quad_accel")
    fm = FusedMLP(net, "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "ieee")
    f = torch.compile(lambda x: fm(2.0 * x) + 1.0, fullgraph=True, dynamic=False)
    x = torch.randn(1000, 13, device=DEV, generator=torch.Generator(device=DEV).manual_seed(3))
    with torch.no_grad():
        assert torch.equal(f(x), fm(2.0 * x) + 1.0)


@gpu
def test_refusals():
    from ndm_gym.core.nets import mlp
    from ndm_gym.kernels.mlp_fused import FusedMLP

    net = _net("quad_accel")
    fm = FusedMLP(net, "ieee")
    with pytest.raises(RuntimeError, match="inference-only"):
        fm(torch.randn(4, 13, device=DEV))
    with torch.no_grad(), pytest.raises(ValueError, match="expected"):
        fm(torch.randn(4, 12, device=DEV))
    with pytest.raises(ValueError, match="dot must be"):
        FusedMLP(net, "fp16")
    for bad, match in [(mlp(13, [256, 128], 6).to(DEV), "equally wide"), (mlp(13, [96], 6).to(DEV), "multiple of 64"),
                       (mlp(13, [1024], 6).to(DEV), "at most 512"), (nn.Sequential(nn.Linear(13, 64), nn.GELU(), nn.Linear(64, 6)).to(DEV), "activations"),
                       (nn.Sequential(nn.Linear(13, 64, bias=False), nn.SiLU(), nn.Linear(64, 6)).to(DEV), "without bias")]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            FusedMLP(bad, "ieee")
    with pytest.raises(ValueError, match="CUDA"):
        FusedMLP(mlp(13, [64], 6), "ieee")
