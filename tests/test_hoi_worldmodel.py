"""Neural dynamics models of HDMI's scenes (ndm_gym/worldmodels/hoi.py) on CPU: the context layout, outputs that
reproduce the next state, invariance of the inputs and equivariance of the step to a shift and yaw of the whole scene,
the frame-local quantities against the LAFAN1 models', the loss, gradients, the state dict's format, activation
checkpointing, the inference copies; with a GPU, the frame-local quantities under TF32 against float64, the fused step
kernels and the compiled RigidFormer at 33 tokens against eager."""

import math

import pytest
import torch

from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots.g1 import NUM_J
from ndm_gym.state import CONTROL_DT
from ndm_gym.tasks.hdmi import DECIMATION, NUM_ACTIONS
from ndm_gym.worldmodels.hoi import FORMAT_VERSION, ContextLayout, HOIWorldModel

N, E = 32, 2


def _rot(n, g):
    return so3_exp(torch.randn(n, 3, generator=g))


def _scene(hinged: bool, layout: ContextLayout, seed: int = 0, n: int = N):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    S = (r(n, 3), _rot(n, g), r(n, 3), r(n, 3), r(n, NUM_J), r(n, NUM_J))
    So = (r(n, 1), r(n, 1)) if hinged else (r(n, 3), _rot(n, g), r(n, 3), r(n, 3))
    poses = [torch.cat((r(n, 3), _rot(n, g).flatten(1)), -1) for _ in range(int(layout.base) + layout.props)]
    ctx = torch.cat([r(n, layout.scalars)] + poses, -1)
    return S, So, r(n, DECIMATION, NUM_ACTIONS), ctx


def _next(S, So, hinged, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: 0.1 * torch.randn(*s, generator=g)  # noqa: E731
    p, R, v, w, q, qd = S
    n = len(p)
    S1 = (p + r(n, 3), R @ so3_exp(r(n, 3)), v + r(n, 3), w + r(n, 3), q + r(n, NUM_J), qd + r(n, NUM_J))
    if hinged:
        So1 = (So[0] + r(n, 1), So[1] + r(n, 1))
    else:
        So1 = (So[0] + r(n, 3), So[1] @ so3_exp(r(n, 3)), So[2] + r(n, 3), So[3] + r(n, 3))
    F1 = (r(n, E, 3), r(n, DECIMATION, 2, 3))
    return S1, So1, F1


def _model(kind, hinged, layout):
    return HOIWorldModel(kind, hinged, E, layout, hidden=(32, 32))


def _close(a, b, **kw):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, **kw)


def _yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _turned(Rz, t, x):
    """A free body's (p, R, v, w) and the robot's joints after it, turned by Rz about z and shifted by t; a hinge's (angle,
    rate) as it is."""
    if len(x) == 2:
        return x
    return (x[0] @ Rz.T + t, Rz @ x[1], x[2] @ Rz.T, x[3] @ Rz.T) + tuple(x[4:])


def _turned_ctx(Rz, t, lay, ctx):
    """The context with its poses (a fixed base's, the props') turned by Rz about z and shifted by t."""
    parts, i = [ctx[:, : lay.scalars]], lay.scalars
    for _ in range(int(lay.base) + lay.props):
        parts += [ctx[:, i : i + 3] @ Rz.T + t, (Rz @ ctx[:, i + 3 : i + 12].unflatten(-1, (3, 3))).flatten(1)]
        i += 12
    return torch.cat(parts, -1)


def test_context_layout():
    lay = ContextLayout(startup=7, base=True, props=2)
    assert lay.scalars == 7 + NUM_J + 4 and lay.dim == lay.scalars + 36
    S, So, u, ctx = _scene(True, lay)
    poses = lay.poses(ctx)
    assert len(poses) == 3
    torch.testing.assert_close(poses[1][0], ctx[:, lay.scalars + 12 : lay.scalars + 15])
    torch.testing.assert_close(poses[2][1], ctx[:, lay.scalars + 27 : lay.scalars + 36].unflatten(-1, (3, 3)))


@pytest.mark.parametrize("hinged", [False, True])
def test_increments_reproduce_the_next_state(hinged):
    lay = ContextLayout(5, hinged, 1)
    wm = _model("increment", hinged, lay)
    S, So, _, _ = _scene(hinged, lay)
    S1, So1, F1 = _next(S, So, hinged)
    y = wm.targets(S, So, S1, So1, F1)
    yr, yo, yf = y.split((wm.robot_out, wm.obj_out, wm.force_out), -1)
    _close(wm._robot_next(S, yr), S1, atol=2e-5, rtol=2e-5)
    _close(wm._object_next(So, yo), So1, atol=2e-5, rtol=2e-5)
    _close(wm._forces(S, yf), F1, atol=2e-5, rtol=2e-5)  # the feet's through the heading frame of S and back
    torch.testing.assert_close(wm.increments(S, So, S1, So1, F1), y)  # the loss's scale is fit on the same quantities


@pytest.mark.parametrize("hinged", [False, True])
def test_accelerations_reproduce_the_next_velocities(hinged):
    """SuperTrack's parameterization: the velocities are exact, positions follow by semi-implicit Euler."""
    lay = ContextLayout(5, hinged, 0)
    wm = _model("accel", hinged, lay)
    S, So, _, _ = _scene(hinged, lay)
    S1, So1, F1 = _next(S, So, hinged)
    yr, yo, _ = wm.targets(S, So, S1, So1, F1).split((wm.robot_out, wm.obj_out, wm.force_out), -1)
    Sn = wm._robot_next(S, yr)
    _close((Sn[2], Sn[3], Sn[5]), (S1[2], S1[3], S1[5]), atol=2e-5, rtol=2e-5)
    _close((Sn[0], Sn[4]), (S[0] + CONTROL_DT * S1[2], S[4] + CONTROL_DT * S1[5]), atol=2e-5, rtol=2e-5)
    On = wm._object_next(So, yo)
    if hinged:
        _close(On, (So[0] + CONTROL_DT * So1[1], So1[1]), atol=2e-5, rtol=2e-5)
    else:
        _close((On[2], On[3]), (So1[2], So1[3]), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("hinged", [False, True])
def test_inputs_invariant_to_a_shift_and_yaw_of_the_scene(hinged):
    lay = ContextLayout(5, hinged, 2)
    wm = _model("increment", hinged, lay)
    S, So, u, ctx = _scene(hinged, lay)
    Rz, t = _yaw(0.7), torch.tensor([3.0, -2.0, 0.0])
    x2 = wm.inputs(_turned(Rz, t, S), _turned(Rz, t, So), u, _turned_ctx(Rz, t, lay, ctx))
    torch.testing.assert_close(x2, wm.inputs(S, So, u, ctx), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["accel", "increment"])
@pytest.mark.parametrize("hinged", [False, True])
def test_step_loss_and_gradients(kind, hinged):
    lay = ContextLayout(5, hinged, 1)
    torch.manual_seed(0)
    wm = _model(kind, hinged, lay)
    torch.nn.init.normal_(wm.net[-1].weight, std=0.1)  # zero at initialization
    S, So, u, ctx = _scene(hinged, lay)
    S1, So1, F1 = _next(S, So, hinged)
    for _ in range(3):
        wm.in_norm.update(wm.inputs(S, So, u, ctx)), wm.out_norm.update(wm.targets(S, So, S1, So1, F1))
        wm.err_norm.update(wm.increments(S, So, S1, So1, F1))
    assert wm.errors((S1, So1, F1), (S1, So1, F1)).abs().max() < 1e-5
    k = 3
    uk = u.unsqueeze(1).expand(N, k, DECIMATION, NUM_ACTIONS).clone().requires_grad_(True)
    preds = wm.rollout(S, So, uk, ctx.unsqueeze(1).expand(N, k, -1))
    assert len(preds) == k and preds[-1][2][0].shape == (N, E, 3) and preds[-1][2][1].shape == (N, DECIMATION, 2, 3)
    trues = [(S1, So1, F1)] * k
    mask = torch.ones(N, k)
    loss, last, per = wm.window_loss(preds, trues, mask)
    assert loss > 0 and last.shape == (N,) and per.shape == (N,)
    zero, _, _ = wm.window_loss(trues, trues, mask)
    assert float(zero) < 1e-8
    loss.backward()
    assert uk.grad is not None and uk.grad.abs().sum() > 0 and wm.net[0].weight.grad.abs().sum() > 0
    masked, _, _ = wm.window_loss(preds, trues, torch.cat((torch.ones(N, 1), torch.zeros(N, k - 1)), 1))
    first, _, _ = wm.window_loss(preds[:1], trues[:1], torch.ones(N, 1))
    torch.testing.assert_close(masked, first)  # steps after an episode end do not count


def test_frame_local_quantities_match_the_lafan1_models():
    """The robot's targets, increments and errors are the LAFAN1 models' (IncrementWorldModel.targets and state_error,
    AccelWorldModel.targets) with the 3 x 3 products elementwise: equal in float64 up to its rounding."""
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.hoi import robot_accel, robot_increment
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    S, So, _, _ = _scene(False, ContextLayout(5, False, 0))
    S1, _, _ = _next(S, So, False)
    S, S1 = tuple(x.double() for x in S), tuple(x.double() for x in S1)
    torch.testing.assert_close(robot_increment(S, S1), IncrementWorldModel.targets(S, S1))
    torch.testing.assert_close(robot_increment(S1, S), IncrementWorldModel.state_error(S, S1))
    torch.testing.assert_close(robot_accel(S, S1), AccelWorldModel.targets(S, S1))


def test_state_dict_format():
    """The state dict carries its format; a model saved before the feet's forces moved to the heading frame (no version)
    or of another version is refused, also by a non-strict load."""
    lay = ContextLayout(5, False, 0)
    wm, twin = _model("increment", False, lay), _model("increment", False, lay)
    sd = wm.state_dict()
    assert int(sd["format_version"]) == FORMAT_VERSION
    twin.load_state_dict(sd)
    old = {k: v for k, v in sd.items() if k != "format_version"}
    for bad in (old, {**sd, "format_version": torch.tensor(FORMAT_VERSION + 1)}):
        for strict in (True, False):
            with pytest.raises(RuntimeError, match="format"):
                twin.load_state_dict(bad, strict=strict)


def _rf_args(hinged):
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.tasks.hdmi import JOINT_NAMES
    from ndm_gym.worldmodels.hoi import rigidformer_args

    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    consts = {"eef_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"] if hinged else ["left_wrist_yaw_link", "right_wrist_yaw_link"],
              "q0": [0.1 * i for i in range(NUM_J)], "act_scale": [0.25 + 0.01 * i for i in range(NUM_ACTIONS)],
              "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip
    return rigidformer_args(consts, dim=24, layers=1, heads=2, registers=2)


@pytest.mark.parametrize("kind", ["accel", "increment"])
@pytest.mark.parametrize("hinged", [False, True])
def test_token_backbone(kind, hinged):
    """Output layout, invariance of the inputs to a shift and yaw of the scene, gradients to the actions."""
    lay = ContextLayout(5, hinged, 2)
    torch.manual_seed(0)
    wm = HOIWorldModel(kind, hinged, E, lay, rigidformer=_rf_args(hinged))
    S, So, u, ctx = _scene(hinged, lay)
    x = wm.inputs(S, So, u, ctx)
    assert x.shape == (N, wm.net.in_dim)
    y = wm.net(wm.in_norm(x))
    assert y.shape == (N, wm.robot_out + wm.obj_out + wm.force_out)
    Rz, t = _yaw(-1.1), torch.tensor([-1.0, 4.0, 0.0])
    torch.testing.assert_close(wm.inputs(_turned(Rz, t, S), _turned(Rz, t, So), u, _turned_ctx(Rz, t, lay, ctx)), x, atol=2e-5, rtol=2e-5)
    for h in (wm.net.root_head, wm.net.joint_head, wm.net.obj_head, wm.net.eef_head, wm.net.feet_head):  # zero at initialization
        torch.nn.init.normal_(h.weight, std=0.1)
    uk = u.clone().requires_grad_(True)
    S1, So1, F1 = wm.step(S, So, uk, ctx)
    assert F1[0].shape == (N, E, 3) and F1[1].shape == (N, DECIMATION, 2, 3)
    (S1[0].sum() + S1[4].sum() + So1[0].sum() + F1[1].sum()).backward()
    assert uk.grad is not None and uk.grad.abs().sum() > 0


def test_token_backbone_actions_reach_their_joints():
    """An action drives the PD-target features of exactly its joint's token (and no wrist's)."""
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.tasks.hdmi import JOINT_NAMES

    lay = ContextLayout(5, False, 0)
    wm = HOIWorldModel("accel", False, E, lay, rigidformer=_rf_args(False))
    S, So, _, ctx = _scene(False, lay)
    net, k = wm.net, 7
    u = torch.zeros(N, DECIMATION, NUM_ACTIONS)
    u[:, :, k] = 1.0
    x = net.inputs(S, So, u, ctx)
    body = x[:, : net.n_body * net.body_dim].unflatten(-1, (net.n_body, net.body_dim))
    tgt = body[..., 15 + 3 : 15 + 7]  # the four substeps' PD-target offsets from the default pose
    j = G1_JOINT_NAMES.index(JOINT_NAMES[k])
    tok = 1 + int((net.joint_of_body == j).nonzero())
    torch.testing.assert_close(tgt[:, tok], torch.full((N, DECIMATION), float(net.act_scale[k])))
    others = torch.ones(net.n_body, dtype=torch.bool)
    others[tok] = False
    assert tgt[:, others].abs().max() == 0


def _randomize(wm, std=0.1, seed=3):
    """Give the zero-initialized parameters (heads, FiLM, relation bias) values, so that every path carries signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in wm.net.parameters():
            if prm.abs().max() == 0:
                prm.copy_(std * torch.randn(prm.shape, generator=g))


@pytest.mark.parametrize("backbone", ["mlp", "rigidformer"])
@pytest.mark.parametrize("hinged", [False, True])
def test_inference_copy(backbone, hinged):
    """The inference copy: frozen, the original untouched; with fp32 parameters the same step up to float rounding
    (merged read-out heads, the relation bias gathered once); in bf16 end to end within bf16 rounding."""
    lay = ContextLayout(5, hinged, 1)
    torch.manual_seed(0)
    rf = _rf_args(hinged) if backbone == "rigidformer" else None
    wm = HOIWorldModel("accel", hinged, E, lay, hidden=(32, 32), rigidformer=rf)
    _randomize(wm)
    S, So, u, ctx = _scene(hinged, lay)
    att = "sdpa" if rf else None
    with pytest.raises(ValueError):
        wm.inference_copy(None, None if rf else "sdpa")
    c32, c16 = wm.inference_copy(None, att), wm.inference_copy(torch.bfloat16, att)
    assert not any(p.requires_grad for p in c16.parameters()) and all(p.requires_grad for p in wm.parameters())
    assert all(p.dtype == torch.bfloat16 for p in c16.net.parameters()) and all(p.dtype == torch.float32 for p in wm.net.parameters())
    with torch.no_grad():
        ref, a, b = wm.step(S, So, u, ctx), c32.step(S, So, u, ctx), c16.step(S, So, u, ctx)
    for x, y, z in zip([*ref[0], *ref[1], *ref[2]], [*a[0], *a[1], *a[2]], [*b[0], *b[1], *b[2]]):
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z, x, atol=0.05 * float(x.abs().max()) + 1e-3, rtol=0)


@pytest.mark.parametrize("backbone", ["mlp", "rigidformer"])
@pytest.mark.parametrize("kind", ["accel", "increment"])
@pytest.mark.parametrize("hinged", [False, True])
def test_step_turns_with_the_scene(backbone, kind, hinged):
    """A yaw of the whole scene about z (and a shift) turns the step's world-frame outputs with it: the robot's and a free
    object's positions, rotations and velocities, and the feet's forces. The frame-local quantities stay: the joints, a
    hinge, the end effectors' forces (contact body's frame), the targets, increments and errors. Random weights, fp32."""
    lay = ContextLayout(5, hinged, 1)
    torch.manual_seed(0)
    rf = _rf_args(hinged) if backbone == "rigidformer" else None
    wm = HOIWorldModel(kind, hinged, E, lay, hidden=(32, 32), rigidformer=rf)
    _randomize(wm)
    S, So, u, ctx = _scene(hinged, lay)
    S1, So1, F1 = _next(S, So, hinged)
    with torch.no_grad():
        wm.in_norm.update(wm.inputs(S, So, u, ctx)), wm.out_norm.update(wm.targets(S, So, S1, So1, F1))
    Rz, t = _yaw(2.0), torch.tensor([1.5, -0.5, 0.0])
    turn = lambda S, So, F: (_turned(Rz, t, S), _turned(Rz, t, So), (F[0], F[1] @ Rz.T))  # noqa: E731
    with torch.no_grad():
        pred = wm.step(S, So, u, ctx)
        pred2 = wm.step(_turned(Rz, t, S), _turned(Rz, t, So), u, _turned_ctx(Rz, t, lay, ctx))
    kw = dict(atol=1e-4, rtol=1e-4)
    for x, y in zip(pred2, turn(*pred)):
        _close(x, y, **kw)
    feet = pred[2][1]
    assert (feet @ Rz.T - feet).abs().max() > 0.1 * feet.abs().max()  # the feet's horizontal forces turn: not a trivial check
    true2 = turn(S1, So1, F1)
    _close(wm.targets(*turn(S, So, F1)[:2], *true2), wm.targets(S, So, S1, So1, F1), **kw)
    _close(wm.increments(*turn(S, So, F1)[:2], *true2), wm.increments(S, So, S1, So1, F1), **kw)
    _close(wm.errors(pred2, true2), wm.errors(pred, (S1, So1, F1)), **kw)


def test_token_backbone_checkpointing(monkeypatch):
    """Activation checkpointing of the HOI RigidFormer, as RigidFormerNet's: the same outputs and gradients as without it;
    by default ("frozen") only while the parameters are frozen, as in the policy's BPTT through the model."""
    lay = ContextLayout(5, False, 1)
    torch.manual_seed(0)
    wm = HOIWorldModel("increment", False, E, lay, rigidformer=_rf_args(False))
    _randomize(wm)
    S, So, u, ctx = _scene(False, lay)
    with torch.no_grad():
        wm.in_norm.update(wm.inputs(S, So, u, ctx))
    assert wm.net.checkpoint == "frozen"
    out = []
    for mode in ("never", "always"):
        wm.net.checkpoint = mode
        wm.zero_grad()
        uk = u.clone().requires_grad_(True)
        y = wm.net(wm.in_norm(wm.inputs(S, So, uk, ctx)))
        (y * torch.linspace(-1, 1, y.shape[-1])).sum().backward()
        out.append([y.detach(), uk.grad] + [p.grad.clone() for p in wm.net.parameters()])
    _close(out[1], out[0])
    wm.net.checkpoint = "frozen"
    calls = []
    orig = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda *a, **k: calls.append(1) or orig(*a, **k))
    uk = u.clone().requires_grad_(True)
    wm.step(S, So, uk, ctx)
    assert not calls
    wm.requires_grad_(False)
    with torch.no_grad():
        wm.step(S, So, uk, ctx)
    assert not calls
    wm.step(S, So, uk, ctx)[0][0].sum().backward()
    assert len(calls) == 1 and uk.grad is not None and uk.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="checkpoint"):
        HOIWorldModel("increment", False, E, lay, rigidformer={**_rf_args(False), "checkpoint": "sometimes"})


def _gpu_scene(hinged, lay, n, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
    S = (r(n, 3), so3_exp(r(n, 3)), r(n, 3), r(n, 3), r(n, NUM_J), r(n, NUM_J))
    So = (r(n, 1), r(n, 1)) if hinged else (r(n, 3), so3_exp(r(n, 3)), r(n, 3), r(n, 3))
    poses = [torch.cat((r(n, 3), so3_exp(r(n, 3)).flatten(1)), -1) for _ in range(int(lay.base) + lay.props)]
    return S, So, r(n, DECIMATION, NUM_ACTIONS), torch.cat([r(n, lay.scalars)] + poses, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is a mode of CUDA's matmuls")
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_frame_local_quantities_under_tf32(kind):
    """targets, increments and errors (the model's loss and its normalizers' data) on CUDA with TF32 matmuls allowed, as
    train_hoi runs them, eager and compiled: within fp32 rounding of float64. IncrementWorldModel.targets' batched
    rotation product is off by about 5e-4 there (A100), which shows that TF32 is in effect."""
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    wm = HOIWorldModel(kind, False, E, ContextLayout(5, False, 0), hidden=(32, 32))
    S, So, _, _ = _scene(False, wm.layout, seed=4, n=4096)
    S1, So1, F1 = _next(S, So, False, seed=5)
    Sh, Soh, Fh = _next(S1, So1, False, seed=6)  # a prediction near the truth
    data = (S, So, S1, So1, F1, Sh, Soh, Fh)

    def quantities(S, So, S1, So1, F1, Sh, Soh, Fh):
        return wm.targets(S, So, S1, So1, F1), wm.increments(S, So, S1, So1, F1), wm.errors((Sh, Soh, Fh), (S1, So1, F1))

    ref = quantities(*[tuple(t.double() for t in x) for x in data])
    shared_ref = IncrementWorldModel.targets(tuple(t.double() for t in S), tuple(t.double() for t in S1))
    cuda = [tuple(t.cuda() for t in x) for x in data]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        got = [quantities(*cuda), torch.compile(quantities)(*cuda)]
        shared = [IncrementWorldModel.targets(cuda[0], cuda[2]), torch.compile(IncrementWorldModel.targets)(cuda[0], cuda[2])]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch._dynamo.reset()
    floor = max(float((x[:, 3:6].double().cpu() - shared_ref[:, 3:6]).abs().max()) for x in shared)  # the rotation's
    if floor < 1e-4:
        pytest.skip(f"TF32 is not in effect on this GPU (batched products off by {floor:.1e})")
    for out in got:
        for x, y in zip(out, ref):
            err = float((x.double().cpu() - y).abs().max())
            assert err < 1e-5 * max(1.0, float(y.abs().max())), (err, floor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused step kernels are Triton kernels (CUDA)")
@pytest.mark.parametrize("kind", ["accel", "increment"])
@pytest.mark.parametrize("backbone", ["mlp", "rigidformer"])
def test_fused_step_matches_step(kind, backbone):
    """worldmodels/hoi_fused.py: the features and the integration as Triton kernels equal the step's torch code in fp32
    up to float rounding, for a free object, a free object with props and a hinge with a prop."""
    import ndm_gym.core.nets as nets

    amp, nets.AMP_DTYPE = nets.AMP_DTYPE, None
    try:
        for hinged, props in ((False, 0), (False, 2), (True, 1)):
            lay = ContextLayout(7, hinged, props)
            torch.manual_seed(0)
            rf = _rf_args(hinged) if backbone == "rigidformer" else None
            wm = HOIWorldModel(kind, hinged, E, lay, hidden=(64, 64), rigidformer=rf).cuda()
            _randomize(wm)
            S, So, u, ctx = _gpu_scene(hinged, lay, 300)
            att = "sdpa" if rf else None
            with torch.no_grad():
                wm.in_norm.update(wm.inputs(S, So, u, ctx))
                a, b = wm.inference_copy(None, att).step(S, So, u, ctx), wm.inference_copy(None, att, fused=True).step(S, So, u, ctx)
            for x, y in zip([*a[0], *a[1], *a[2]], [*b[0], *b[1], *b[2]]):
                torch.testing.assert_close(y, x, atol=1e-5 * float(x.abs().max()) + 1e-6, rtol=1e-5)
    finally:
        nets.AMP_DTYPE = amp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor's GPU code generation")
def test_compiled_token_backbone_matches_eager():
    """The HOI RigidFormer at its default size and 33 tokens (30 bodies, the object, 2 registers), compiled under bf16
    autocast, equals its eager forward up to bf16 rounding: torch 2.11's Inductor miscompiled rope_rotate's stack in this
    graph (errors of the order of the outputs), which the fused rotary embedding avoids."""
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.tasks.hdmi import JOINT_NAMES
    from ndm_gym.worldmodels.hoi import rigidformer_args

    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    consts = {"eef_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "q0": [0.1 * i for i in range(NUM_J)],
              "act_scale": [0.25 + 0.01 * i for i in range(NUM_ACTIONS)],
              "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip
    lay = ContextLayout(5, False, 0)
    torch.manual_seed(0)
    wm = HOIWorldModel("accel", False, E, lay, rigidformer=rigidformer_args(consts)).cuda()
    _randomize(wm, std=0.3)
    with torch.no_grad():
        wm.net.rel_bias.normal_(0.0, 1.0)
    S, So, u, ctx = _gpu_scene(False, lay, 1024)
    with torch.no_grad():
        wm.in_norm.update(wm.inputs(S, So, u, ctx))
        x = wm.in_norm(wm.inputs(S, So, u, ctx))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ye, yc = wm.net(x).float(), torch.compile(wm.net)(x).float()
    torch._dynamo.reset()
    assert float((yc - ye).abs().max()) < 0.05 * float(ye.abs().max())
