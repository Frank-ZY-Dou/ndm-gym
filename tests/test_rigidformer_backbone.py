"""RigidFormer body-token backbone (ndm_gym/worldmodels/rigidformer.py) on CPU: shapes, token features, forward
kinematics, invariances, read-out layout, the zero-initialized start, differentiability, activation checkpointing, the
external-wrench channel, input noise, the training-argument mapping, the fused rotary embedding, anchor-based rotary
positions (RigidFormer's farthest point sampling, both anchor modes, translation invariance of the query-key products),
and small overfitting runs of both world models (SuperTrack accelerations, SuperMimic increments);
with a GPU, the token attention kernel."""

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


def test_checkpointing_frozen_readout(kin, batch, monkeypatch):
    """checkpoint="frozen_readout" (--rf_bptt_keep_blocks): while the parameters are frozen only the read-out is
    recomputed, with the default mode's output and input gradients; while they are trained nothing is recomputed."""
    from ndm_gym.algos.common import set_requires_grad

    wm = rf_models(kin)[0]
    randomize_zero_init(wm.net)
    fit_norms(wm, batch)
    set_requires_grad(wm, False)
    calls, out = [], []
    orig = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda fn, *a, **k: calls.append(fn.__name__) or orig(fn, *a, **k))
    for mode in ("frozen", "frozen_readout"):
        wm.net.checkpoint = mode
        u = batch.u.clone().requires_grad_(True)
        y = wm.net(wm.in_norm(wm.inputs(batch.S, u, batch.xi_n)))
        (y * torch.linspace(-1, 1, y.shape[-1])).sum().backward()
        out.append((y.detach(), u.grad))
    assert calls == ["_trunk", "_readout"]
    torch.testing.assert_close(out[1][0], out[0][0], rtol=0, atol=0)
    torch.testing.assert_close(out[1][1], out[0][1], rtol=0, atol=0)
    set_requires_grad(wm, True)
    wm.step(batch.S, batch.u.clone().requires_grad_(True), batch.xi_n)[5].sum().backward()
    assert calls == ["_trunk", "_readout"]


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
    assert kw["checkpoint"] == "frozen"  # also for the checkpoints from before --rf_bptt_keep_blocks
    keep = {**a, "wm_backbone": "rigidformer", "rf_fused_train": True, "rf_bptt_keep_blocks": True}
    assert rigidformer_kwargs(keep)["checkpoint"] == "frozen_readout"
    for bad in ({"wm_backbone": "mlp"}, {"rf_fused_train": False}, {"wm_bptt_fp32": True}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**keep, **bad})
    for bad in ({"wm_act": "gelu"}, {"wm_no_ln": True}, {"wm_hidden": [512, 512]}, {"wm_backbone": "gnn"}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**a, "wm_backbone": "rigidformer", **bad})


def test_rope_rotate_fused(kin):
    """The rotary embedding as one elementwise expression equals rope_rotate up to float rounding at any head width; the
    G1 network keeps rope_rotate unless asked (the numerics of the runs trained before the flag)."""
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet, Rope3d, rigidformer_kwargs, rope_rotate, rope_rotate_fused

    g = torch.Generator().manual_seed(0)
    for hd in (16, 24, 32, 48):
        cos, sin = Rope3d(6 * ((3 * hd // 4) // 6), 10.0)(torch.randn(5, 7, 3, generator=g))
        x = torch.randn(5, 3, 7, hd, generator=g)
        torch.testing.assert_close(rope_rotate_fused(x, cos, sin), rope_rotate(x, cos, sin), atol=1e-6, rtol=1e-6)
    assert not any(b.fused_rope for b in RigidFormerNet(kin, root_out=6, joint_out=1, **SMALL).blocks)
    assert all(b.fused_rope for b in RigidFormerNet(kin, root_out=6, joint_out=1, fused_rope=True, **SMALL).blocks)
    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=96, rf_layers=2, rf_heads=3,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    assert rigidformer_kwargs(a)["fused_rope"] is False and rigidformer_kwargs({**a, "rf_fused_rope": True})["fused_rope"] is True
    with pytest.raises(ValueError):
        rigidformer_kwargs({**a, "wm_backbone": "mlp", "rf_fused_rope": True})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the token attention kernel is a Triton kernel (CUDA)")
def test_token_attention_matches_sdpa():
    """ndm_gym.core.token_attention against F.scaled_dot_product_attention with the Block's layouts (q, k contiguous, v a
    view of the fused projection), a bias shared by the batch, per sample or none, bf16 and fp32, 33 tokens; and the
    Block refuses it where a backward pass would be needed."""
    import torch.nn.functional as F

    from ndm_gym.core.token_attention import token_attention
    from ndm_gym.worldmodels.rigidformer import Block

    g = torch.Generator(device="cuda").manual_seed(0)
    for dtype, tol in ((torch.bfloat16, 2e-2), (torch.float32, 1e-5)):
        qkv = torch.randn(64, 33, 3, 4, 32, device="cuda", generator=g).to(dtype)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q, k = q.contiguous(), k.contiguous()
        biases = (torch.randn(1, 4, 33, 33, device="cuda", generator=g).to(dtype),
                  torch.randn(64, 4, 33, 33, device="cuda", generator=g).to(dtype), None)  # fmt: skip
        for bias in biases:
            torch.testing.assert_close(token_attention(q, k, v, bias), F.scaled_dot_product_attention(q, k, v, attn_mask=bias), atol=tol, rtol=tol)
    blk = Block(64, 2, 2.5, 0).cuda()
    blk.attention = "token"
    with pytest.raises(RuntimeError):  # no backward pass
        blk(torch.randn(4, 33, 64, device="cuda"))


def test_farthest_point_sample():
    """RigidFormer's farthest point sampling (rule 'centroid_farthest'): it starts at the vertex farthest from the
    centroid, breaks ties by the lexicographically smallest coordinates (a cube's corners), does not depend on the
    vertices' order, never takes a vertex twice (coincident duplicates), and its first k are the k-point sampling."""
    import numpy as np

    from ndm_gym.core.kinematics import farthest_point_sample

    rng = np.random.default_rng(0)
    cube = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)], dtype=np.float32)
    for v in (cube, cube[rng.permutation(8)]):
        np.testing.assert_array_equal(v[farthest_point_sample(v, 3)], [[-1, -1, -1], [1, 1, 1], [-1, -1, 1]])
    pts = rng.normal(size=(200, 3)).astype(np.float32)
    c = pts.astype(np.float64).mean(0)
    assert farthest_point_sample(pts, 1)[0] == np.argmax(((pts - c) ** 2).sum(1))
    perm = rng.permutation(200)
    np.testing.assert_array_equal(pts[perm][farthest_point_sample(pts[perm], 12)], pts[farthest_point_sample(pts, 12)])
    np.testing.assert_array_equal(farthest_point_sample(pts, 200)[:12], farthest_point_sample(pts, 12))
    dup = np.concatenate((pts, pts[:50]))
    assert sorted(farthest_point_sample(dup, 250).tolist()) == list(range(250))
    perm = rng.permutation(250)
    np.testing.assert_array_equal(dup[perm][farthest_point_sample(dup[perm], 12)], dup[farthest_point_sample(dup, 12)])
    for bad in (0, 201):
        with pytest.raises(ValueError):
            farthest_point_sample(pts, bad)


def test_body_anchors(kin):
    """Each body's anchors: RigidFormer's farthest point sampling over its visual-mesh vertices in the link frame (its
    own meshes and those of the fixed bodies attached to it); deterministic, distinct vertices of the body, on the body,
    and the first k of a longer sampling."""
    import numpy as np

    verts = kin.visual_vertices()
    a4, a8 = kin.anchors(4), kin.anchors(8)
    assert len(verts) == 30 and a4.shape == (30, 4, 3) and a8.shape == (30, 8, 3) and a8.dtype == torch.float32
    assert torch.equal(a4, kin.anchors(4)) and torch.equal(a8[:, :4], a4)
    for b, v in enumerate(verts):
        assert v.dtype == np.float32 and len(v) >= 8
        assert len(torch.unique(a8[b], dim=0)) == 8
        assert all((v == a.numpy()).all(1).any() for a in a8[b])  # vertices of the body
    assert float(a8.norm(dim=-1).max()) < 0.6  # metres from the link origin
    with pytest.raises(ValueError):
        kin.anchors(0)


def test_rope_translation_invariance():
    """Every channel pair turns by one angle, so translating all token positions (every anchor, the registers) leaves
    the query-key products unchanged up to float rounding: one point per token (also ARoPE's, the anchors' centroid)
    and four (concat), with rope_rotate and rope_rotate_fused (RigidFormer's original half-split angle layout did not
    have this property)."""
    from ndm_gym.worldmodels.rigidformer import Rope3d, rope_rotate, rope_rotate_fused

    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(2, 3, 2, 9, 32, generator=g).unbind(0)  # (N, H, T, head width)
    t = torch.tensor([1.3, -0.7, 0.4])
    for rope, shape in ((Rope3d(24, 10.0), (3, 9, 3)), (Rope3d(24, 10.0, 4), (3, 9, 4, 3))):
        pos = torch.randn(*shape, generator=g)
        for rot in (rope_rotate, rope_rotate_fused):
            logits = [rot(q, *rope(x)) @ rot(k, *rope(x)).transpose(-1, -2) for x in (pos, pos + t)]
            torch.testing.assert_close(logits[1], logits[0], atol=1e-4, rtol=1e-4)
            assert not torch.allclose(logits[0], q @ k.transpose(-1, -2), atol=1e-2)


def test_anchors_one_is_todays_network(kin, batch):
    """anchors=1 (the default, in either mode) is the network of the runs trained before the flag: same inputs,
    parameters and outputs."""
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    nets = []
    for kw in ({}, {"anchors": 1}, {"anchors": 1, "anchor_mode": "concat"}):
        torch.manual_seed(0)
        nets.append(RigidFormerNet(kin, root_out=6, joint_out=1, **SMALL, **kw))
    a = nets[0]
    for b in nets[1:]:
        assert b.in_dim == a.in_dim and torch.equal(b.pos_index, a.pos_index) and not hasattr(b, "anchor_pts")
        assert list(b.state_dict()) == list(a.state_dict())
        assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    for net in nets:
        randomize_zero_init(net)
    xa = a.inputs(batch.S, batch.u, batch.xi_n)
    norm = a.make_input_norm()
    norm.update(xa)
    for b in nets[1:]:
        xb = b.inputs(batch.S, batch.u, batch.xi_n)
        assert torch.equal(xa, xb) and torch.equal(a(norm(xa)), b(norm(xb)))


def _scene_moved(S):
    """S shifted by (2, 1, 0) m and turned by -0.8 rad about the vertical axis."""
    from ndm_gym.core.rotations import axis_angle_to_mat

    Rz = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(-0.8))
    rot = lambda v: (Rz @ v.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    p, R, v, w, q, qd = S
    return rot(p) + torch.tensor([2.0, 1.0, 0.0]), Rz @ R, rot(v), rot(w), q, qd


def _reference_rope_tables(net, kin, S):
    """The rotary tables (cos, sin (N, 1, T, R)) of RigidFormer's anchor RoPE for states S, computed directly with the
    official code's definitions (physformer_rope.py) in float64: every body's anchors (K = 1: its link origin) moved
    with it, in the heading frame at the pelvis's ground projection; with rope_center relative to their centroid over
    all bodies and anchors (decoder.py:248-262); the registers at zero; every coordinate x_c of every anchor turning its
    channel pairs by x_c * omega_f (omega_f log-spaced in [1, 10] rad/m, interleaved pairs), mean-pooled over the anchors
    (ARoPE, _forward_sa_rope) or each anchor coordinate with its own channels (concat, forward_multi_point)."""
    from ndm_gym.core.kinematics import mat_vec
    from ndm_gym.core.rotations import yaw_mat

    p, R = S[0], S[1]
    P, Rb = kin.forward(*S, all_bodies=True)[:2]
    pts = net.anchor_pts if net.n_anchor > 1 else torch.zeros(30, 1, 3)
    world = P.double().unsqueeze(-2) + mat_vec(Rb.double().unsqueeze(-3), pts.double())  # (N, B, K, 3)
    origin = torch.cat((p[:, :2], torch.zeros_like(p[:, 2:])), -1).double()
    anc = mat_vec(yaw_mat(R).double().transpose(-1, -2)[:, None, None], world - origin[:, None, None])
    if net.rope_center:
        anc = anc - anc.mean((1, 2), keepdim=True)
    anc = torch.cat((anc, anc.new_zeros(anc.shape[0], net.n_reg, *anc.shape[2:])), 1)
    n_rot, K = 24, anc.shape[2]  # rotary channels per head at head width 32
    if net.anchor_mode == "mean" or K == 1:
        ang = (anc.unsqueeze(-1) * torch.logspace(0, 1, n_rot // 6, dtype=torch.float64)).flatten(-2).mean(-2)
    else:
        ang = (anc.flatten(-2).unsqueeze(-1) * torch.logspace(0, 1, n_rot // (6 * K), dtype=torch.float64)).flatten(-2)
    return ang.cos().float().unsqueeze(1), ang.sin().float().unsqueeze(1)


@pytest.mark.parametrize("kw", [{}, {"rope_center": True}, {"anchors": 4}, {"anchors": 4, "rope_center": True},
                                {"anchors": 4, "anchor_mode": "concat"}, {"anchors": 4, "anchor_mode": "concat", "rope_center": True}])  # fmt: skip
def test_rotary_tables_match_reference(kin, batch, kw):
    """The rotary tables the blocks take (one point per body, ARoPE with 4 anchors per body, concat; with and without
    rope_center) against RigidFormer's definitions computed directly in float64 (``_reference_rope_tables``), through
    inputs, in_norm and the network's forward pass, and in the world model's step."""
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(0)
    net = RigidFormerNet(kin, root_out=6, joint_out=1, dim=64, layers=2, heads=2, registers=2, **kw)
    x = net.inputs(batch.S, batch.u, batch.xi_n)
    norm = net.make_input_norm()
    norm.update(x)
    tables = []
    net.rope.register_forward_hook(lambda m, i, o: tables.append(o))
    net(norm(x))
    ref = _reference_rope_tables(net, kin, batch.S)
    assert tables[0][0].shape == ref[0].shape == (N, 1, 32, 12)
    for a, b in zip(tables[0], ref):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    torch.manual_seed(0)
    wm = AccelWorldModel(rigidformer=dict(kin=kin, dim=64, layers=2, heads=2, registers=2, **kw))
    fit_norms(wm, batch)
    wm.net.rope.register_forward_hook(lambda m, i, o: tables.append(o))
    wm.step(batch.S, batch.u, batch.xi_n)
    for a, b in zip(tables[1], ref):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def test_anchors(kin, batch):
    """K = 4 anchors per body, mode 'mean' (ARoPE): the inputs carry the anchors' centroid moved with its body (the mean
    of the anchors' angles is its angles), in the heading frame as the token positions; in_norm passes it on; the
    outputs are invariant to the anchors' order; the anchors are saved with the model; the inputs are invariant to a
    shift and yaw of the scene; gradients reach the parameters; the training-argument mapping and its errors."""
    from ndm_gym.core.kinematics import mat_vec
    from ndm_gym.core.rotations import yaw_mat
    from ndm_gym.worldmodels.rigidformer import GLOBAL_DIM, POS_SCALE, TOKEN_DIM, RigidFormerNet, rigidformer_kwargs

    torch.manual_seed(0)
    net = RigidFormerNet(kin, root_out=6, joint_out=1, anchors=4, **SMALL)
    randomize_zero_init(net)
    D = 30 * TOKEN_DIM + GLOBAL_DIM
    x = net.inputs(batch.S, batch.u, batch.xi_n)
    assert net.n_rot_pts == 1 and net.in_dim == D + 30 * 3 == x.shape[-1]
    # the anchors' centroid in the heading frame, origin at the pelvis's ground projection, as the token positions
    p, R = batch.S[0], batch.S[1]
    P, Rb = kin.forward(*batch.S, all_bodies=True)[:2]
    world = P.unsqueeze(-2) + mat_vec(Rb.unsqueeze(-3), net.anchor_pts)
    origin = torch.cat((p[:, :2], torch.zeros_like(p[:, 2:])), -1)
    expect = mat_vec(yaw_mat(R).transpose(-1, -2)[:, None, None], world - origin[:, None, None])  # (N, 30, 4, 3)
    torch.testing.assert_close(x[:, D:].view(N, 30, 3), expect.mean(-2), atol=1e-5, rtol=1e-5)
    norm = net.make_input_norm()
    norm.update(x)
    xin = norm(x)
    torch.testing.assert_close(POS_SCALE * xin[:, net.in_dim :], x[:, D:])
    y = net(xin)
    assert torch.equal(net.state_dict()["anchor_pts"], kin.anchors(4))  # saved with the model
    net.anchor_pts = net.anchor_pts[:, torch.randperm(4)]  # invariant to the anchors' order
    torch.testing.assert_close(net(norm(net.inputs(batch.S, batch.u, batch.xi_n))), y, atol=1e-5, rtol=1e-5)
    # a shift and yaw of the whole scene
    x2 = net.inputs(_scene_moved(batch.S), batch.u, batch.xi_n)
    torch.testing.assert_close(x2, net.inputs(batch.S, batch.u, batch.xi_n), atol=2e-5, rtol=1e-5)
    # gradients
    net.checkpoint = "never"
    net(norm(net.inputs(batch.S, batch.u, batch.xi_n))).square().sum().backward()
    assert all(prm.grad is not None and prm.grad.abs().sum() > 0 for prm in net.parameters())
    # the training arguments
    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=96, rf_layers=2, rf_heads=3,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    kw = rigidformer_kwargs(a)
    assert kw["anchors"] == 1 and kw["anchor_mode"] == "mean" and kw["rope_center"] is False
    kw = rigidformer_kwargs({**a, "rf_anchors": 8, "rf_anchor_mode": "concat", "rf_rope_center": True})
    assert kw["anchors"] == 8 and kw["anchor_mode"] == "concat" and kw["rope_center"] is True
    fast = {**a, "rf_anchors": 4, "rf_rope_center": True, "rf_fused_train": True, "rf_bptt_keep_blocks": True}
    assert rigidformer_kwargs(fast)["fused_train"] is True and rigidformer_kwargs(fast)["checkpoint"] == "frozen_readout"
    for bad in ({"rf_anchors": 4}, {"rf_anchor_mode": "concat"}, {"rf_rope_center": True}):
        with pytest.raises(ValueError):
            rigidformer_kwargs({**a, "wm_backbone": "mlp", **bad})
    for bad in ({"anchors": 0}, {"anchors": 2.5}, {"anchors": 4, "rope": False}, {"anchor_mode": "set"}, {"rope_center": True, "rope": False}):
        with pytest.raises(ValueError):
            RigidFormerNet(kin, root_out=6, joint_out=1, **SMALL, **bad)


def test_anchors_concat(kin, batch):
    """K = 4 anchors per body, mode 'concat' (RigidFormer's forward_multi_point without set aggregation): the inputs
    carry the K anchors; the 3K anchor coordinates, anchor by anchor, each turn their own channels by R / 6K
    frequencies; the channel budget (R = 24 at head width 32: K = 4 fits with one frequency of 1 rad/m per coordinate,
    K = 3 and 8 are refused, K = 8 fits at head width 64 with R = 48); the anchors' order matters; outputs invariant to
    a shift and yaw of the scene; gradients."""
    from ndm_gym.worldmodels.rigidformer import GLOBAL_DIM, TOKEN_DIM, RigidFormerNet, Rope3d

    rope, pos = Rope3d(24, 10.0, 4), torch.randn(5, 7, 4, 3)
    c, s_ = rope(pos)
    assert c.shape == (5, 1, 7, 12)
    torch.testing.assert_close(c, pos.flatten(-2).cos().unsqueeze(1))  # one frequency, 1 rad/m, per coordinate
    torch.testing.assert_close(s_, pos.flatten(-2).sin().unsqueeze(1))
    c2 = Rope3d(24, 10.0, 2)(pos[:, :, :2])[0]  # K = 2: two frequencies, 1 and 10 rad/m
    torch.testing.assert_close(c2, (pos[:, :, :2].flatten(-2).unsqueeze(-1) * torch.tensor([1.0, 10.0])).flatten(-2).cos().unsqueeze(1))
    assert not torch.allclose(rope(pos[:, :, [1, 0, 2, 3]])[0], c)  # the anchors' order matters
    for bad in (pos[:, :, :3], pos[:, :, 0]):
        with pytest.raises(ValueError):
            rope(bad)
    for k in (3, 8):
        with pytest.raises(ValueError):
            Rope3d(24, 10.0, k)
        with pytest.raises(ValueError):
            RigidFormerNet(kin, root_out=6, joint_out=1, dim=64, layers=1, heads=2, anchors=k, anchor_mode="concat")
    wide = RigidFormerNet(kin, root_out=6, joint_out=1, dim=128, layers=1, heads=2, anchors=8, anchor_mode="concat")
    assert wide.rope.freqs.tolist() == [1.0]  # head width 64: 48 channels, 2 per coordinate
    torch.manual_seed(0)
    net = RigidFormerNet(kin, root_out=6, joint_out=1, dim=64, layers=2, heads=2, registers=2, anchors=4, anchor_mode="concat")
    randomize_zero_init(net)
    x = net.inputs(batch.S, batch.u, batch.xi_n)
    assert net.n_rot_pts == 4 and net.in_dim == 30 * TOKEN_DIM + GLOBAL_DIM + 30 * 4 * 3 == x.shape[-1]
    norm = net.make_input_norm()
    norm.update(x)
    y = net(norm(x))
    torch.testing.assert_close(net(norm(net.inputs(_scene_moved(batch.S), batch.u, batch.xi_n))), y, atol=1e-4, rtol=1e-4)
    net.anchor_pts = net.anchor_pts[:, [1, 0, 2, 3]]
    assert not torch.allclose(net(norm(net.inputs(batch.S, batch.u, batch.xi_n))), y, atol=1e-4)
    net.checkpoint = "never"
    y.square().sum().backward()
    assert all(prm.grad is not None and prm.grad.abs().sum() > 0 for prm in net.parameters())


def test_rope_center(kin, batch):
    """rope_center (RigidFormer's rope_center_positions): the rotary positions relative to their centroid over all
    bodies and anchors, so a translation of all of them changes nothing (with it the registers move along), whereas
    without it the registers at the origin see the translation; one point per body and both anchor modes."""
    from ndm_gym.worldmodels.rigidformer import POS_SCALE, RigidFormerNet

    t = torch.tensor([0.3, -0.2, 0.5]) / POS_SCALE
    for kw in ({}, {"anchors": 4}, {"anchors": 4, "anchor_mode": "concat"}):
        ys = []
        for center in (True, False):
            torch.manual_seed(0)
            net = RigidFormerNet(kin, root_out=6, joint_out=1, dim=64, layers=2, heads=2, registers=2, rope_center=center, **kw)
            randomize_zero_init(net)
            x = net.inputs(batch.S, batch.u, batch.xi_n)
            norm = net.make_input_norm()
            norm.update(x)
            xin = norm(x)
            moved = xin.clone()
            moved[:, net.in_dim :] += t.repeat(moved[:, net.in_dim :].shape[1] // 3)
            ys.append((net(xin), net(moved)))
        torch.testing.assert_close(ys[0][1], ys[0][0], atol=1e-4, rtol=1e-4)
        assert not torch.allclose(ys[1][1], ys[1][0], atol=1e-3)
