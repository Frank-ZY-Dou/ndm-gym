"""The fused muscle geometry kernels (ndm_gym.kernels.muscle_geometry) on a GPU, without MuscleVAE's assets skipped:
the muscles' lengths, length rates and per-body wrench against ndm_gym.robots.muscles in float64, every gradient against
autograd through it, and the body-token RigidFormer's features and step with and without the kernels."""

import os

import pytest
import torch

from ndm_gym.core.rotations import mat_to_quat, so3_exp

HAVE_TABLE = os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "musclevae", "muscles.json"))
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_TABLE), reason="needs a GPU and MuscleVAE's converted muscle table")
NB, M = 23, 284


def pose(n: int, seed: int, dev: str = "cuda"):
    """Body states (p, R, v, w) in float64 of poses about the T-pose, with velocities."""
    from ndm_gym.robots import musclevae_char as mc

    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(n, mc.JOINT_QPOS_DIM, dtype=torch.float64)
    for adr, size in zip(mc.JOINT_QPOS_ADR, mc.JOINT_QPOS_SIZE):
        if size == 4:
            q[:, adr : adr + 4] = mat_to_quat(so3_exp(0.3 * torch.randn(n, 3, generator=g, dtype=torch.float64)))
        else:
            q[:, adr] = 0.3 * torch.randn(n, generator=g, dtype=torch.float64)
    root = mat_to_quat(so3_exp(0.5 * torch.randn(n, 3, generator=g, dtype=torch.float64)))
    p, R = mc.forward_kinematics(torch.tensor([0.4, -0.3, 0.95], dtype=torch.float64).expand(n, 3), root, q, mc.load_character())
    v, w = torch.randn(n, NB, 3, generator=g, dtype=torch.float64), 2.0 * torch.randn(n, NB, 3, generator=g, dtype=torch.float64)
    return [x.to(dev) for x in (p, R, v, w)]


def test_geometry_against_muscle_model():
    """Lengths, rates and wrench within fp32 rounding of robots.muscles in float64 (relative to each quantity's scale),
    and the gradients to positions, rotations, velocities, angular velocities and muscle forces likewise."""
    from ndm_gym.kernels import muscle_geometry as mg
    from ndm_gym.robots import muscles as mu
    from ndm_gym.robots.musclevae_char import load_muscle_table

    table, params = load_muscle_table(), mu.muscle_params("released")
    mm64 = mu.MuscleModel(table, params, "cuda", torch.float64)
    tables = mg.MuscleTables(mu.MuscleModel(table, params, "cuda", torch.float32)).tensors
    p, R, v, w = pose(96, 1)
    force = torch.rand(96, M, generator=torch.Generator(device="cuda").manual_seed(2), device="cuda", dtype=torch.float64) * mm64.f0max
    x64 = [x.clone().requires_grad_(True) for x in (p, R, v, w, force)]
    x32 = [x.float().clone().requires_grad_(True) for x in (p, R, v, w, force)]
    an = mm64.anchor_positions(x64[0], x64[1])
    fb, tb = mm64.body_wrench(an, mm64.anchor_forces(an, x64[4]), x64[0])
    ref = (mm64.lengths(an), mm64.length_rates(an, mm64.anchor_velocities(*x64[:4])), torch.cat((fb, tb), -1))
    frames = mg.frames(*x32[:4])
    out = (*mg.muscle_lengths(frames, tables), mg.muscle_wrench(frames, x32[4], tables))
    for r, o in zip(ref, out):
        assert o.shape == r.shape and torch.isfinite(o).all()
        assert (o.double() - r).abs().max() < 2e-5 * r.abs().max()
    gs = [torch.randn_like(r) for r in ref]
    g64 = torch.autograd.grad(sum((r * g).sum() for r, g in zip(ref, gs)), x64)
    g32 = torch.autograd.grad(sum((o * g.float()).sum() for o, g in zip(out, gs)), x32)
    for a, b in zip(g64, g32):
        assert (b.double() - a).abs().max() < 2e-5 * a.abs().max()


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rigidformer_fused_features(kind):
    """The body-token RigidFormer's features and step with the kernels equal those without them up to fp32 rounding,
    and so do the step's gradients to the state, the action and the parameters."""
    from ndm_gym.worldmodels.muscle import MuscleWorldModel

    models = []
    for fused in (False, True):
        torch.manual_seed(0)
        wm = MuscleWorldModel(kind, rigidformer=dict(dim=48, layers=2, heads=4, checkpoint="never", tokens="body", muscle_dim=16,
                                                     fused_features=fused)).cuda()  # fmt: skip
        with torch.no_grad():
            for prm in wm.net.parameters():
                prm.normal_(0.0, 0.1)
            wm.out_norm.var.uniform_(0.1, 2.0)
        models.append(wm)
    p, R, v, w = (x.float() for x in pose(64, 3))
    F = torch.rand(64, 15, device="cuda")
    a = 0.1 * torch.randn(64, M, device="cuda")
    x0, x1 = (wm.inputs((p, R, v, w), F, a) for wm in models)
    scale = x0.abs().amax(0).clamp(min=1e-3)
    assert ((x1 - x0).abs() / scale).max() < 1e-3  # fp32 rounding through the Hill curves, the PD force's clips, the activation's division
    with torch.no_grad():
        models[0].in_norm.update(x0)
        models[1].in_norm.load_state_dict(models[0].in_norm.state_dict())
    grads = []
    for wm in models:
        vv, aa = v.clone().requires_grad_(True), a.clone().requires_grad_(True)
        S1, F1 = wm.step((p, R, vv, w), F, aa)
        loss = S1[0].square().sum() + S1[2].square().sum() + F1.square().sum()
        grads.append(torch.autograd.grad(loss, [vv, aa] + list(wm.net.parameters())))
    for g0, g1 in zip(*grads):
        assert (g1 - g0).abs().max() <= 1e-3 * g0.abs().max() + 1e-6
