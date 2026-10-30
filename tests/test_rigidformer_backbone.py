"""RigidFormer body-token backbone (ndm_gym/worldmodels/rigidformer.py) on CPU: shapes, token features, forward
kinematics, invariances, read-out layout, the zero-initialized start, differentiability, activation checkpointing, the
external-wrench channel, input noise, the training-argument mapping, and small overfitting runs of both world models
(SuperTrack accelerations, SuperMimic increments)."""

from types import SimpleNamespace

import pytest
import torch

from conftest import random_states

SMALL = dict(dim=32, layers=2, heads=2, registers=2)
N = 64


@pytest.fixture(scope="module")
def batch(motion):
    """States, setpoints, normalized sim context, raw context and next states of a fixed batch."""
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.state import normalize_xi

    f, S, xi = random_states(motion, N, seed=3)
    g = torch.Generator().manual_seed(4)
    u = torch.randn(N, 29, generator=g)
    p, R, v, w, q, qd = S
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    S1 = (p + 0.02 * v, so3_exp(0.02 * w) @ R, v + 0.1 * rn(N, 3), w + 0.2 * rn(N, 3), q + 0.02 * qd, qd + 0.5 * rn(N, 29))
    return SimpleNamespace(S=S, u=u, xi=xi, xi_n=normalize_xi(xi), S1=S1)


def rf_models(kin, **cfg):
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    torch.manual_seed(0)
    rf = dict(kin=kin, **{**SMALL, **cfg})
    return AccelWorldModel(rigidformer=rf), IncrementWorldModel(rigidformer=rf)


def fit_norms(wm, b):
    wm.in_norm.update(wm.inputs(b.S, b.u, b.xi_n))
    wm.out_norm.update(wm.targets(b.S, b.S1))


def randomize_zero_init(net, scale=0.1):
    """The heads, FiLM layers and tree bias start at zero; give them values so every path carries signal."""
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in [net.root_head, net.joint_head] + [blk.film for blk in net.blocks]:
            m.weight.copy_(scale * torch.randn(m.weight.shape, generator=g))
            m.bias.copy_(scale * torch.randn(m.bias.shape, generator=g))
        net.rel_bias.copy_(torch.randn(net.rel_bias.shape, generator=g))


def test_tree_kinematics_matches_articulation_kinematics(kin, batch):
    from ndm_gym.worldmodels.rigidformer import TreeKinematics

    for a, b in zip(TreeKinematics(kin)(*batch.S), kin.forward(*batch.S, all_bodies=True)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_xi_offset_scale_matches_normalize_xi(batch):
    from ndm_gym.worldmodels.rigidformer import XI_OFFSET_SCALE

    torch.testing.assert_close(XI_OFFSET_SCALE * batch.xi_n[:, 1:30], batch.xi[:, 1:30])


def test_shapes(kin, batch):
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.worldmodels.rigidformer import GLOBAL_DIM, POS_SCALE, TOKEN_DIM

    for wm, out in zip(rf_models(kin), (35, 70)):
        fit_norms(wm, batch)
        x = wm.inputs(batch.S, batch.u, batch.xi_n)
        assert x.shape == (N, 30 * TOKEN_DIM + GLOBAL_DIM) == (N, wm.in_dim)
        xin = wm.in_norm(x)
        assert xin.shape == (N, wm.in_dim + 90)
        torch.testing.assert_close(xin[:, : wm.in_dim], RunningNorm.forward(wm.in_norm, x))
        torch.testing.assert_close(POS_SCALE * xin[:, wm.in_dim :].view(N, 30, 3), x[:, : 30 * TOKEN_DIM].view(N, 30, TOKEN_DIM)[..., :3])
        assert wm.net(xin).shape == (N, out) == (N, wm.out_norm.mean.numel())
        for a, b in zip(wm.step(batch.S, batch.u, batch.xi_n), batch.S):
            assert a.shape == b.shape and torch.isfinite(a).all()


def test_input_noise(kin, batch):
    """step's input noise (normalized units) applies to the token features and, scaled, to the rotary positions."""
    for wm in rf_models(kin):
        randomize_zero_init(wm.net)
        fit_norms(wm, batch)
        clean = wm.step(batch.S, batch.u, batch.xi_n)
        torch.manual_seed(1)
        noisy = wm.step(batch.S, batch.u, batch.xi_n, 0.1)
        assert all(torch.isfinite(a).all() for a in noisy)
        assert (noisy[5] - clean[5]).abs().max() > 0


def test_token_features(kin, batch):
    from ndm_gym.core.rotations import mat_to_6d, yaw_mat
    from ndm_gym.robots.g1 import G1_JOINT_NAMES, action_scale, default_joint_pos_list
    from ndm_gym.worldmodels.rigidformer import TOKEN_DIM

    wm = rf_models(kin)[0]
    p, R, v, w, q, qd = batch.S
    x = wm.inputs(batch.S, batch.u, batch.xi_n)
    tok = x[:, : 30 * TOKEN_DIM].view(N, 30, TOKEN_DIM)
    # pelvis: at the origin's vertical, its tilt, no parent joint
    torch.testing.assert_close(tok[:, 0, :3], torch.cat((torch.zeros(N, 2), p[:, 2:]), -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(tok[:, 0, 3:9], mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R))
    assert (tok[:, 0, 15:] == 0).all()
    # heights stay absolute (height above the ground plane)
    P = kin.forward(*batch.S, all_bodies=True)[0]
    torch.testing.assert_close(tok[:, :, 2], P[..., 2])
    # every other body carries its own joint; the PD error is the simulator's target minus q
    alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES])
    target = torch.tensor(default_joint_pos_list()) + batch.xi[:, 1:30] + alpha * batch.u  # TrackingSim.step
    j = torch.tensor(kin.joint_idx[1:])
    torch.testing.assert_close(tok[:, 1:, 15], q[:, j])
    torch.testing.assert_close(tok[:, 1:, 16], qd[:, j])
    torch.testing.assert_close(tok[:, 1:, 17], batch.u[:, j])
    torch.testing.assert_close(tok[:, 1:, 18], (target - q)[:, j])
    assert (tok[:, 1:, 19].abs() <= 1).all()
    torch.testing.assert_close(x[:, 30 * TOKEN_DIM :], batch.xi_n[:, [0, 30, 31, 32]])


def test_heading_and_translation_invariance(kin, batch):
    from ndm_gym.core.rotations import axis_angle_to_mat

    Rz = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(1.1))
    t = torch.tensor([3.0, -2.0, 0.0])
    p, R, v, w, q, qd = batch.S
    rot = lambda x: (Rz @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    S2 = (rot(p) + t, Rz @ R, rot(v), rot(w), q, qd)
    for wm in rf_models(kin):
        randomize_zero_init(wm.net)
        fit_norms(wm, batch)
        x1, x2 = wm.inputs(batch.S, batch.u, batch.xi_n), wm.inputs(S2, batch.u, batch.xi_n)
        torch.testing.assert_close(x2, x1, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(wm.net(wm.in_norm(x2)), wm.net(wm.in_norm(x1)), atol=1e-4, rtol=1e-4)


def test_readout_layout(kin, batch):
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    net = RigidFormerNet(kin, root_out=12, joint_out=2, **SMALL)
    norm = net.make_input_norm()
    inp = norm(net.inputs(batch.S, batch.u, batch.xi_n))
    with torch.no_grad():
        net.root_head.bias.copy_(torch.arange(12.0))
        net.joint_head.bias.copy_(torch.tensor([100.0, 200.0]))
    y = net(inp)
    torch.testing.assert_close(y, torch.cat((torch.arange(12.0), torch.full((29,), 100.0), torch.full((29,), 200.0))).expand(N, -1))

    # joint j's outputs come from the token of the body that owns joint j
    class TokenIndex(torch.nn.Module):
        def forward(self, z):
            return torch.arange(1.0, z.shape[1] + 1).view(1, -1, 1).expand(z.shape[0], -1, 1)

    net = RigidFormerNet(kin, root_out=6, joint_out=1, **SMALL)
    net.joint_head = TokenIndex()
    y = net(inp)[:, 6:]
    owner = torch.tensor([kin.joint_idx.index(j) for j in range(29)], dtype=torch.float32)
    torch.testing.assert_close(y, owner.expand(N, -1))


def test_zero_init_step_matches_mlp(kin, batch):
    """With zero-initialized heads both backbones predict the output normalizer's mean: the read-out, normalizers and
    integration of the backbone path are those of the MLP world models."""
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    for rf, mlp in zip(rf_models(kin), (AccelWorldModel(hidden=(64, 64)), IncrementWorldModel(hidden=(64, 64)))):
        fit_norms(rf, batch)
        fit_norms(mlp, batch)
        torch.testing.assert_close(rf.out_norm.mean, mlp.out_norm.mean)
        for a, b in zip(rf.step(batch.S, batch.u, batch.xi_n), mlp.step(batch.S, batch.u, batch.xi_n)):
            torch.testing.assert_close(a, b)


def test_gradients_reach_inputs_and_parameters(kin, batch):
    from ndm_gym.state import flatten_state

    for wm in rf_models(kin):
        randomize_zero_init(wm.net)
        fit_norms(wm, batch)
        u, xi_n = batch.u.clone().requires_grad_(True), batch.xi_n.clone().requires_grad_(True)
        v, qd = batch.S[2].clone().requires_grad_(True), batch.S[5].clone().requires_grad_(True)
        S = (batch.S[0], batch.S[1], v, batch.S[3], batch.S[4], qd)
        S1 = wm.step(S, u, xi_n)
        (flatten_state(S1) * torch.randn(N, 71, generator=torch.Generator().manual_seed(6))).sum().backward()
        for name, t in (("u", u), ("xi_n", xi_n), ("v", v), ("qd", qd)):
            assert torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0, name
        for name, prm in wm.net.named_parameters():
            assert prm.grad is not None and torch.isfinite(prm.grad).all() and prm.grad.abs().sum() > 0, name


def test_checkpointing_is_exact(kin, batch):
    wm = rf_models(kin)[0]
    randomize_zero_init(wm.net)
    fit_norms(wm, batch)
    out = []
    for mode in ("never", "always"):
        wm.net.checkpoint = mode
        wm.zero_grad()
        u = batch.u.clone().requires_grad_(True)
        y = wm.net(wm.in_norm(wm.inputs(batch.S, u, batch.xi_n)))
        (y * torch.linspace(-1, 1, y.shape[-1])).sum().backward()
        out.append((y.detach(), u.grad, [p.grad.clone() for p in wm.net.parameters()]))
    torch.testing.assert_close(out[1][0], out[0][0])
    torch.testing.assert_close(out[1][1], out[0][1])
    for a, b in zip(out[1][2], out[0][2]):
        torch.testing.assert_close(a, b)


def test_checkpointing_frozen_mode(kin, batch, monkeypatch):
    """The default mode recomputes the trunk only while the parameters are frozen (policy BPTT)."""
    from ndm_gym.algos.common import set_requires_grad

    wm = rf_models(kin)[0]
    fit_norms(wm, batch)
    calls = []
    orig = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda *a, **k: calls.append(1) or orig(*a, **k))
    u = batch.u.clone().requires_grad_(True)
    wm.step(batch.S, u, batch.xi_n)
    assert not calls
    set_requires_grad(wm, False)
    wm.step(batch.S, u, batch.xi_n)[5].sum().backward()
    assert len(calls) == 1 and u.grad is not None


def test_external_wrench_channel(kin, batch):
    from ndm_gym.worldmodels.rigidformer import TOKEN_DIM, WRENCH_DIM

    # the channel lives in the backbone; the world models pass no forces yet
    net = rf_models(kin, ext_wrench=True)[0].net
    with pytest.raises(ValueError):
        net.inputs(batch.S, batch.u, batch.xi_n)
    with pytest.raises(ValueError):
        net.inputs(batch.S, batch.u, batch.xi_n, SimpleNamespace(wrench_h=torch.zeros(N, 30, 3)))
    f = SimpleNamespace(wrench_h=torch.randn(N, 30, WRENCH_DIM))
    x = net.inputs(batch.S, batch.u, batch.xi_n, f)
    F = TOKEN_DIM + WRENCH_DIM
    assert x.shape[-1] == net.in_dim == 30 * F + 4
    torch.testing.assert_close(x[:, : 30 * F].view(N, 30, F)[..., TOKEN_DIM:], f.wrench_h)
    with pytest.raises(ValueError):
        rf_models(kin)[0].net.inputs(batch.S, batch.u, batch.xi_n, f)


@pytest.mark.slow
@pytest.mark.parametrize("which", [0, 1])
def test_overfits_a_small_batch(kin, motion, which):
    """Each world model learns the one-step dynamics of a random teacher MLP world model on 128 states."""
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    torch.manual_seed(7)
    teacher = (AccelWorldModel, IncrementWorldModel)[which](hidden=(64, 64))
    with torch.no_grad():
        teacher.net[-1].weight.normal_(0.0, 0.3)
    _, S, xi = random_states(motion, 128, seed=8)
    xi_n, u = normalize_xi(xi), torch.randn(128, 29, generator=torch.Generator().manual_seed(9))
    teacher.in_norm.update(teacher.inputs(S, u, xi_n))
    with torch.no_grad():
        S1 = teacher.step(S, u, xi_n)
    wm = rf_models(kin)[which]
    wm.in_norm.update(wm.inputs(S, u, xi_n))
    wm.out_norm.update(wm.targets(S, S1))
    inp = wm.in_norm(wm.inputs(S, u, xi_n))
    target = (wm.targets(S, S1) - wm.out_norm.mean) / wm.out_norm.std()
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    losses = []
    for _ in range(400):
        loss = ((wm.net(inp) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 0.5 and losses[-1] < 0.02 * losses[0], (losses[0], losses[-1])
    with torch.no_grad():
        err = (wm.step(S, u, xi_n)[5] - S1[5]).abs().mean() / (S1[5] - S[5]).abs().mean()
    assert err < 0.2, float(err)


def test_rigidformer_kwargs_from_training_args():
    from ndm_gym.worldmodels.rigidformer import rigidformer_kwargs

    a = dict(wm_backbone="mlp", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=96, rf_layers=2, rf_heads=3,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=True)  # fmt: skip
    assert rigidformer_kwargs(a) is None
    kw = rigidformer_kwargs({**a, "wm_backbone": "rigidformer"})
    assert (kw["dim"], kw["layers"], kw["heads"], kw["registers"], kw["tree_bias"], kw["rope"]) == (96, 2, 3, 2, True, False)
    assert kw["kin"].num_bodies == 30
    for bad in ({"wm_act": "gelu"}, {"wm_no_ln": True}, {"wm_hidden": [512, 512]}, {"wm_backbone": "gnn"}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**a, "wm_backbone": "rigidformer", **bad})
