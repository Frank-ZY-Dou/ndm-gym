"""The quadrotor world model's inference copies (ndm_gym/worldmodels/quad_infer.py, quad_fused.py), both parameterizations: the fp32 copy
computes the model's step exactly, the fused copy within float rounding over a rollout (also written into given buffers), the bf16 copy
passes its own check, and the refusals. The GPU tests are skipped without CUDA."""

import pytest
import torch

DEV = "cuda"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the inference copies are checked on a CUDA GPU")


def _model(kind, seed=0):
    """A QuadWorldModel with random weights (its last layer too) and normalizers of plausible scales."""
    from ndm_gym.worldmodels.quadrotor import QuadWorldModel

    torch.manual_seed(seed)
    wm = QuadWorldModel(hidden=(64, 64), kind=kind).to(DEV)
    with torch.no_grad():
        torch.nn.init.normal_(wm.net[-1].weight, std=0.05)
        wm.in_norm.mean.normal_(0, 0.5)
        wm.in_norm.var.uniform_(0.5, 4.0)
        wm.out_norm.mean.normal_(0, 0.1)
        wm.out_norm.var.uniform_(0.01, 1.0)
    return wm.eval()


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@gpu
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_fp32_copy_is_the_model(kind):
    from ndm_gym.worldmodels.quad_infer import inference_copy, random_states

    wm = _model(kind)
    cp = inference_copy(wm, None, fused=False)
    S, u = random_states(512, DEV, seed=2)
    with torch.no_grad():
        for a, b in zip(_fp32(lambda: wm.step(S, u)), _fp32(lambda: cp.step(S, u))):
            assert torch.equal(a, b)


@gpu
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_fused_copy_follows_the_model(kind):
    from ndm_gym.worldmodels.quad_infer import inference_copy, random_states

    wm = _model(kind)
    cp = inference_copy(wm, None, fused=True)
    S, u = random_states(1000, DEV, seed=3)  # (not a multiple of the kernels' block)
    S = (S[0], S[1], 0.3 * S[2], 0.3 * S[3])
    with torch.no_grad():
        Sa, Sb = S, S
        for _ in range(16):
            Sa = _fp32(lambda: wm.step(Sa, u))  # noqa: B023
            Sb = _fp32(lambda: cp.step(Sb, u))  # noqa: B023
        for a, b in zip(Sa, Sb):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), float((a - b).abs().max())
        # into the other of two buffers (rows of 18 values), as the speed benchmark steps
        buf = torch.empty(1000, 18, device=DEV)
        out = (buf[:, 0:3], buf[:, 3:12].view(1000, 3, 3), buf[:, 12:15], buf[:, 15:18])
        _fp32(lambda: cp.step_into(S, u, out))
        for a, b in zip(out, _fp32(lambda: cp.step(S, u))):
            assert torch.equal(a, b)


@gpu
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_bf16_copy(kind):
    from ndm_gym.worldmodels.quad_infer import inference_copy, random_states

    wm = _model(kind)
    cp = inference_copy(wm, torch.bfloat16, fused=True)  # checked against the model on the GPU, refused on a mismatch
    S, u = random_states(256, DEV, seed=4)
    with torch.no_grad():
        nxt = cp.step(S, u)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in nxt)


@gpu
def test_refusals():
    from ndm_gym.worldmodels.quad_infer import inference_copy, random_states

    wm = _model("accel")
    with pytest.raises(ValueError, match="dtype"):
        inference_copy(wm, torch.float16)
    cp = inference_copy(wm, None, fused=False)
    S, u = random_states(8, DEV)
    with pytest.raises(ValueError, match="fused path"):
        with torch.no_grad():
            cp.step_into(S, u, S)
    with pytest.raises(RuntimeError, match="inference-only"):
        cp.step(S, u)
