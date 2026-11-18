"""The RigidFormer backbone of the MJCF world model (ndm_gym.worldmodels.mjcf, backbone="rigidformer") against MuJoCo on CPU:
the bodies' velocities beside the forward kinematics against mj_objectVelocity; the token features (each body's centre of mass
or origin, rotation and velocities, the joint slots with their actuators, a fixed tendon's on each of its joints) and their
invariance to the anchor's heading and horizontal position, with the step's; the read-out, every degree of freedom's outputs
from its own body's token; the step with the heads at zero equal to the MLP's (the outer API, the integration and the
couplings untouched); a short fit of each parameterization that lowers the open-loop error; fit_mjcf --backbone rigidformer,
its checkpoint rebuilt by load_ndm, and the refused combinations; the inference copies in fp32 and bf16; and, marked slow,
the step and the window loss under torch.compile. The MLP's tests are tests/test_mjcf_wm.py."""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_mjcf_wm import COUPLED, KIN_XML, XML, model, random_state, simulate  # noqa: E402

from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel, inference_copy, quat_mul  # noqa: E402

RF = {"dim": 32, "layers": 2, "heads": 2}  # a small RigidFormer for the tests


def rigidformer(m, kind="accel", heads_std=None, **rf):
    """The RigidFormer world model of ``m`` (mujoco.MjModel); ``heads_std``: its read-out heads drawn at random instead of zero."""
    wm = MjcfWorldModel(MjcfLayout.from_model(m), kind=kind, backbone="rigidformer", rf={**RF, **rf})
    if heads_std is not None:
        for n in wm.net.head_names:
            torch.nn.init.normal_(getattr(wm.net, f"{n}_head").weight, std=heads_std)
    return wm


@pytest.mark.parametrize("xml", [XML, KIN_XML], ids=["free_base", "fixed_base"])
def test_body_velocities_match_mujoco(xml):
    """The velocity recursion beside the forward kinematics, against mj_objectVelocity of each body's frame (mjOBJ_XBODY: the
    angular velocity and the linear velocity at the frame's origin, in the world's orientation; mjOBJ_BODY would be the inertial
    frame's, at the centre of mass): a free base, hinges and a slide about off-centre anchors, a ball, a hinge with a reference
    angle, a fixed base turned in the world, free objects. The poses do not depend on the velocities."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    wm = rigidformer(m).double()  # its kinematic constants are float32, as the model runs
    qpos, qvel = random_state(m, 5, np.random.default_rng(12))
    P, R, V, W = wm.body_poses(torch.tensor(qpos), torch.tensor(qvel))
    for i in range(5):
        d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(m, d)
        for b in range(m.nbody):
            res = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            np.testing.assert_allclose(W[i, b].numpy(), res[:3], atol=1e-6, rtol=0)
            np.testing.assert_allclose(V[i, b].numpy(), res[3:], atol=1e-6, rtol=0)
    P0, R0 = wm.body_poses(torch.tensor(qpos))
    assert torch.equal(P0, P) and torch.equal(R0, R)


@pytest.mark.parametrize("point", ["com", "origin"])
def test_token_features_of_a_fixed_base(point):
    """The fixed-base model's tokens, in the world's frame (no anchor): one per body after the world; each body's position at its
    centre of mass (xipos) or its origin (xpos), its rotation's first two columns, the linear velocity there and the angular
    velocity (mj_objectVelocity's for mjOBJ_BODY or mjOBJ_XBODY), the angular velocity in its own frame; two joint slots (the
    finger holds a hinge and a slide), each the joint's position and velocity and its actuator's offset and force, zeros where
    there is none."""
    m = mujoco.MjModel.from_xml_string(KIN_XML)
    d = mujoco.MjData(m)
    wm = rigidformer(m, point=point).double()
    rng = np.random.default_rng(13)
    qpos, qvel = random_state(m, 3, rng)
    ctrl = rng.standard_normal((3, m.nu))
    x = wm.inputs(*(torch.tensor(a) for a in (qpos, qvel, ctrl))).unflatten(-1, (m.nbody - 1, wm.tok_dim)).numpy()
    force, offset = (a.numpy() for a in wm.servo(*(torch.tensor(a) for a in (qpos, qvel, ctrl))))
    assert (wm.n_slot, wm.n_act, wm.tok_dim, wm.net.n_tok) == (2, 1, 18 + 2 * (4 + 2), m.nbody - 1)
    joint = lambda name: (m.jnt_qposadr[m.joint(name).id], m.jnt_dofadr[m.joint(name).id])  # noqa: E731
    for i in range(3):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], ctrl[i]
        mujoco.mj_forward(m, d)
        for b in range(1, m.nbody):
            t, res, Rb = x[i, b - 1], np.zeros(6), d.xmat[b].reshape(3, 3)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY if point == "com" else mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            np.testing.assert_allclose(t[:3], (d.xipos if point == "com" else d.xpos)[b], atol=1e-6, rtol=0)
            np.testing.assert_allclose(t[3:9], Rb[:, :2].reshape(6), atol=1e-6, rtol=0)
            np.testing.assert_allclose(t[9:12], res[3:], atol=1e-6, rtol=0)
            np.testing.assert_allclose(t[12:15], res[:3], atol=1e-6, rtol=0)
            np.testing.assert_allclose(t[15:18], Rb.T @ res[:3], atol=1e-6, rtol=0)
        slot = lambda b, s: x[i, b - 1, 18 + 6 * s : 24 + 6 * s]  # noqa: E731
        (kq, kd), (rq, rd), (tq, td) = joint("knuckle"), joint("reach"), joint("thumb")
        np.testing.assert_allclose(slot(3, 0), [qpos[i, kq], 0, 0, qvel[i, kd], offset[i, 0], force[i, 0]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(slot(3, 1), [qpos[i, rq], 0, 0, qvel[i, rd], 0, 0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(slot(5, 0), [qpos[i, tq], 0, 0, qvel[i, td], offset[i, 1], force[i, 1]], atol=1e-12, rtol=0)
        assert not slot(5, 1).any() and not x[i, [0, 1, 3, 5], 18:].any()  # the mount, palm, tip (a ball) and cube have no slots


def test_token_slots_of_an_unlimited_hinge_and_a_fixed_tendon():
    """An angle with no limits takes sin and cos (the spinner); a fixed tendon's actuator is on each of its joints, beside the
    joint's own actuator (the shoulder and the slide: two actuators per slot); a free body's angular velocity in its own frame
    is its qvel's."""
    m = mujoco.MjModel.from_xml_string(COUPLED)
    wm = rigidformer(m).double()
    rng = np.random.default_rng(14)
    qpos, qvel = random_state(m, 4, rng)
    ctrl = rng.standard_normal((4, m.nu))
    x = wm.inputs(*(torch.tensor(a) for a in (qpos, qvel, ctrl))).unflatten(-1, (m.nbody - 1, wm.tok_dim)).numpy()
    force, offset = (a.numpy() for a in wm.servo(*(torch.tensor(a) for a in (qpos, qvel, ctrl))))
    assert (wm.n_slot, wm.n_act, wm.tok_dim) == (1, 2, 18 + 4 + 2 * 2)
    slot = lambda name: x[:, m.body(name).id - 1, 18:]  # noqa: E731
    pair = 3  # the tendon's actuator
    sq, sd = m.jnt_qposadr[m.joint("shoulder").id], m.jnt_dofadr[m.joint("shoulder").id]
    np.testing.assert_allclose(slot("arm"), np.stack([qpos[:, sq], np.zeros(4), np.zeros(4), qvel[:, sd], offset[:, 0], force[:, 0],
                                                      offset[:, pair], force[:, pair]], -1), atol=1e-12, rtol=0)  # fmt: skip
    eq, ed = m.jnt_qposadr[m.joint("extend").id], m.jnt_dofadr[m.joint("extend").id]
    np.testing.assert_allclose(slot("slider"), np.stack([qpos[:, eq], np.zeros(4), np.zeros(4), qvel[:, ed], offset[:, 1], force[:, 1],
                                                         offset[:, pair], force[:, pair]], -1), atol=1e-12, rtol=0)  # fmt: skip
    pq, pd = m.jnt_qposadr[m.joint("spin").id], m.jnt_dofadr[m.joint("spin").id]
    np.testing.assert_allclose(slot("spinner"), np.stack([np.zeros(4), np.sin(qpos[:, pq]), np.cos(qpos[:, pq]), qvel[:, pd], offset[:, 2], force[:, 2],
                                                          np.zeros(4), np.zeros(4)], -1), atol=1e-12, rtol=0)  # fmt: skip
    for body, da in (("base", 0), ("box", m.jnt_dofadr[m.joint("box").id])):
        np.testing.assert_allclose(x[:, m.body(body).id - 1, 15:18], qvel[:, da + 3 : da + 6], atol=1e-9, rtol=0)


def turned(lay, qpos, qvel, phi, shift):
    """The scene turned by phi about the world's z axis and moved by shift (3,) horizontally: the free joints' positions,
    quaternions and world-frame linear velocities (their angular velocities are the bodies')."""
    Q = torch.tensor([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]], dtype=qpos.dtype)
    qz = torch.tensor([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], dtype=qpos.dtype)
    q2, v2 = qpos.clone(), qvel.clone()
    for qa, da in lay.free:
        q2[:, qa : qa + 3] = qpos[:, qa : qa + 3] @ Q.T + shift
        q2[:, qa + 3 : qa + 7] = quat_mul(qz.expand(len(qpos), 4), qpos[:, qa + 3 : qa + 7])
        v2[:, da : da + 3] = qvel[:, da : da + 3] @ Q.T
    return q2, v2


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rf_inputs_ignore_the_anchors_heading_and_horizontal_position(kind):
    """The tokens (and so the rotary positions) are the same for a scene turned about the vertical and moved horizontally, and
    the step's next state turns and moves with it."""
    m, rng = model(), np.random.default_rng(2)
    lay = MjcfLayout.from_model(m)
    wm = rigidformer(m, kind, heads_std=0.3).double()
    qpos, qvel = (torch.tensor(x) for x in random_state(m, 6, rng))
    ctrl = torch.tensor(rng.standard_normal((6, m.nu)))
    phi, shift = 0.9, torch.tensor([0.7, -1.3, 0.0], dtype=torch.float64)
    q2, v2 = turned(lay, qpos, qvel, phi, shift)
    torch.testing.assert_close(wm.inputs(q2, v2, ctrl), wm.inputs(qpos, qvel, ctrl), atol=1e-9, rtol=0)
    with torch.no_grad():
        want, got = turned(lay, *wm.step(qpos, qvel, ctrl), phi, shift), wm.step(q2, v2, ctrl)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=0)


@pytest.mark.parametrize("kind", ["accel", "increment"])
@pytest.mark.parametrize("xml", [XML, KIN_XML], ids=["free_base", "fixed_base"])
def test_readout_takes_each_degree_of_freedom_from_its_own_bodys_token(kind, xml):
    """The read tokens are the bodies with degrees of freedom; a change of one read token's fused features moves exactly its own
    body's degrees of freedom's outputs (MuJoCo's body_dofadr and body_dofnum), once for the accelerations and twice (tangent
    and velocity) for the increments."""
    m = mujoco.MjModel.from_xml_string(xml)
    net = rigidformer(m, kind, heads_std=1.0).net
    bodies = [b for b in range(1, m.nbody) if m.body_dofnum[b]]
    assert net.read.tolist() == [b - 1 for b in bodies]
    z = torch.randn(3, len(bodies), net.head_w3.out_features)
    with torch.no_grad():
        y = net.heads(z)
        assert y.shape == (3, m.nv if kind == "accel" else 2 * m.nv)
        for r, b in enumerate(bodies):
            z2 = z.clone()
            z2[:, r] += 1.0
            moved = ((net.heads(z2) - y).abs() > 0).any(0).nonzero().flatten().tolist()
            own = list(range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]))
            assert moved == (own if kind == "accel" else own + [m.nv + d for d in own]), (b, moved)


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rf_step_with_its_heads_at_zero_is_the_mlps(kind):
    """Both networks' last layers start at zero, so both steps predict the output normalizer's mean: the same next state, with
    the same integration and joint equalities, in the same output layout (a model with couplings and a fixed tendon)."""
    m, rng = mujoco.MjModel.from_xml_string(COUPLED), np.random.default_rng(15)
    lay = MjcfLayout.from_model(m)
    mlp, rf = MjcfWorldModel(lay, hidden=(16,), kind=kind), MjcfWorldModel(lay, kind=kind, backbone="rigidformer", rf=RF)
    mean = torch.randn(mlp.out_norm.mean.shape)
    for wm in (mlp, rf):
        wm.out_norm.mean.copy_(mean)
    qpos, qvel = (torch.tensor(x, dtype=torch.float32) for x in random_state(m, 8, rng))
    ctrl = torch.tensor(rng.standard_normal((8, m.nu)), dtype=torch.float32)
    with torch.no_grad():
        for a, b in zip(rf.step(qpos, qvel, ctrl), mlp.step(qpos, qvel, ctrl)):
            assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_a_short_rf_fit_lowers_the_open_loop_error(kind):
    """120 updates on 4-step windows bring the error on every window of the episodes well below its start (seeds 0-2: to 0.56-0.62
    of it for the accelerations, 0.45-0.48 for the increments; the MLP of test_mjcf_wm reaches 0.59 and 0.47 in 300)."""
    m, rng = model(), np.random.default_rng(4)
    torch.manual_seed(0)
    qpos, qvel, ctrl = simulate(m, 32, 60, rng)
    wm = rigidformer(m, kind, dim=64, layers=1, heads=4)
    wm.update_norms(qpos[:-1], qvel[:-1], ctrl, qpos[1:], qvel[1:])
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    k = 4
    every = lambda x, n: torch.cat([x[s : s + n] for s in range(60 - k)], 1)  # noqa: E731

    def loss(starts=None):
        if starts is None:
            return wm.window_loss(every(qpos, k + 1), every(qvel, k + 1), every(ctrl, k))
        sl = lambda x, n: torch.stack([x[s : s + n, i] for i, s in enumerate(starts.tolist())], 1)  # noqa: E731
        return wm.window_loss(sl(qpos, k + 1), sl(qvel, k + 1), sl(ctrl, k))

    with torch.no_grad():
        before = loss().item()
    for _ in range(120):
        opt.zero_grad()
        x = loss(torch.randint(0, 60 - k, (32,)))
        x.backward()
        opt.step()
    with torch.no_grad():
        after = loss().item()
    assert after < 0.75 * before, (before, after)


def test_fit_mjcf_rigidformer_runs_saves_and_reloads(tmp_path):
    """fit_mjcf with --backbone rigidformer, for both parameterizations; load_ndm rebuilds the fitted model from its checkpoint (its
    options from the saved arguments, every tensor of its state equal); the refused combinations of the two backbones' options
    fail before any recording loads."""
    from types import SimpleNamespace

    from ndm_gym.wmbench.fit_mjcf import SCORE_COLUMNS, main
    from ndm_gym.wmbench.speed_mjcf import load_ndm

    m, rng = model(), np.random.default_rng(5)
    xml = tmp_path / "scene.xml"
    xml.write_text(XML)
    for name, n in (("train", 8), ("test", 4)):
        qpos, qvel, ctrl = simulate(m, n, 30, rng)
        np.savez(tmp_path / f"{name}.npz", qpos=qpos.numpy(), qvel=qvel.numpy(), ctrl=ctrl.numpy(), reset=np.zeros((30, n), bool), dt=m.opt.timestep)
    common = ["--model", str(xml), "--train", str(tmp_path / "train.npz"), "--test", f"held={tmp_path / 'test.npz'}", "--steps", "20",
              "--batch", "16", "--window", "4", "--amp", "none", "--eval_every", "20", "--eval_k", "1", "4", "--eval_windows", "16"]  # fmt: skip
    rf = ["--backbone", "rigidformer", "--rf_dim", "32", "--rf_layers", "1", "--rf_heads", "2", "--rf_registers", "1", "--rf_rope_max_freq", "30"]
    lay = MjcfLayout.from_model(m)
    for kind, point in (("accel", "com"), ("increment", "origin")):
        main(common + rf + ["--rf_point", point, "--wm_param", kind, "--out", str(tmp_path / f"{kind}.csv"), "--save", str(tmp_path / f"{kind}.pt")])
        rows = (tmp_path / f"{kind}.csv").read_text().strip().splitlines()
        assert rows[0] == "step,seconds,loss," + SCORE_COLUMNS and len(rows) == 1 + 2
        last = dict(zip(rows[0].split(","), rows[-1].split(",")))
        assert np.isfinite(float(last["loss"])) and float(last["norm"]) > 0
        wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / f"{kind}.pt")), "cpu")
        assert (wm.backbone, wm.kind, wm.rf_point) == ("rigidformer", kind, point)
        assert wm.net.describe() == "RigidFormer 32 x 1 (2 heads; 6 bodies, 1 register)" and wm.net.rope.freqs.tolist() == pytest.approx([1.0, 30.0])
        saved = torch.load(tmp_path / f"{kind}.pt", weights_only=False)["state"]
        state = wm.state_dict()
        assert state.keys() == saved.keys() and all(torch.equal(v, saved[k].cpu()) for k, v in state.items())
    missing = ["--train", str(tmp_path / "none.npz")]  # never read: the refusals come first
    with pytest.raises(ValueError, match="--rf_dim, --rf_point: options of the RigidFormer backbone"):
        main(common + missing + ["--rf_dim", "64", "--rf_point", "origin", "--out", str(tmp_path / "x.csv")])
    with pytest.raises(ValueError, match="--hidden, --no_ln: options of the MLP backbone"):
        main(common + missing + rf + ["--hidden", "32", "--no_ln", "--out", str(tmp_path / "x.csv")])
    with pytest.raises(ValueError, match="--geometry: options of the MLP backbone"):
        main(common + missing + rf + ["--geometry", "--out", str(tmp_path / "x.csv")])


def test_rf_refusals():
    m = model()
    lay = MjcfLayout.from_model(m)
    with pytest.raises(ValueError, match="not the MLP"):
        MjcfWorldModel(lay, rf={"dim": 32})
    with pytest.raises(ValueError, match="the options are"):
        MjcfWorldModel(lay, backbone="rigidformer", rf={"width": 32})
    with pytest.raises(ValueError, match="point 'com' or 'origin'"):
        MjcfWorldModel(lay, backbone="rigidformer", rf={"point": "geoms"})
    with pytest.raises(ValueError, match="configure the MLP"):
        MjcfWorldModel(lay, hidden=(64,), backbone="rigidformer")
    with pytest.raises(ValueError, match="contact geometry"):
        MjcfWorldModel(lay, geometry=True, backbone="rigidformer")
    for dim, heads in ((30, 4), (32, 4)):  # not a multiple; 8 channels per head, one rotary frequency
        with pytest.raises(ValueError, match="at least 16 channels per head"):
            MjcfWorldModel(lay, backbone="rigidformer", rf={"dim": dim, "heads": heads})
    with pytest.raises(ValueError, match="rad/m"):
        MjcfWorldModel(lay, backbone="rigidformer", rf={"rope_max_freq": 0.5})
    with pytest.raises(ValueError, match="one of"):
        MjcfWorldModel(lay, backbone="gnn")
    mocap = mujoco.MjModel.from_xml_string(XML.replace('<body name="box"', '<body name="target" mocap="true" pos="0 0 2"/>\n    <body name="box"'))
    assert MjcfLayout.from_model(mocap).nq == lay.nq  # the MLP takes it (a mocap body without geoms)
    with pytest.raises(ValueError, match="mocap bodies"):
        MjcfWorldModel(MjcfLayout.from_model(mocap), backbone="rigidformer")


def test_rf_inference_copy_on_cpu():
    """The fp32 copy steps as the model; the bf16 copy holds its parameters in bf16 (the rotary frequencies and the normalizers
    stay fp32), steps close to the model and refuses to run with gradients; the model keeps its fp32 parameters."""
    m, rng = model(), np.random.default_rng(6)
    wm = rigidformer(m, heads_std=0.1)
    qpos, qvel = (torch.tensor(x, dtype=torch.float32) for x in random_state(m, 16, rng))
    ctrl = torch.tensor(rng.standard_normal((16, m.nu)), dtype=torch.float32)
    with torch.no_grad():
        ref = wm.step(qpos, qvel, ctrl)
        same = inference_copy(wm, None).step(qpos, qvel, ctrl)
    assert all(torch.equal(a, b) for a, b in zip(ref, same))
    try:
        torch.ones(2, 2, dtype=torch.bfloat16) @ torch.ones(2, 2, dtype=torch.bfloat16)
    except RuntimeError as e:  # pragma: no cover
        pytest.skip(f"this CPU's torch runs no bf16 products ({e})")
    half = inference_copy(wm, torch.bfloat16)
    with torch.no_grad():
        got = half.step(qpos, qvel, ctrl)
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in half.net.parameters())
    assert half.net.rope.freqs.dtype == torch.float32 and half.in_norm.mean.dtype == torch.float32 and not half.amp
    assert all(p.dtype == torch.float32 for p in wm.net.parameters())
    for a, b in zip(ref, got):
        assert (a - b).abs().max() < 0.05
    with pytest.raises(RuntimeError, match="forward only"):
        half.step(qpos, qvel, ctrl)


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]


def compile_case(kind, device, seed=16):
    m, rng = model(), np.random.default_rng(seed)
    torch.manual_seed(0)
    qpos, qvel, ctrl = simulate(m, 8, 10, rng)
    wm = rigidformer(m, kind, heads_std=0.1)
    wm.update_norms(qpos[:-1], qvel[:-1], ctrl, qpos[1:], qvel[1:])
    return wm.to(device), qpos.to(device), qvel.to(device), ctrl.to(device)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rf_step_under_torch_compile(kind, device):
    """The step and its bf16 inference copy's (on a GPU with bf16 only) under torch.compile with static shapes, as NdmSim compiles
    them, against eager."""
    wm, qpos, qvel, ctrl = compile_case(kind, device)
    copies = [(wm, 1e-5 if device == "cpu" else 1e-4)]
    if device == "cpu" or torch.cuda.is_bf16_supported(including_emulation=False):
        copies.append((inference_copy(wm, torch.bfloat16), 2e-2))
    with torch.no_grad():
        for w, tol in copies:
            for a, b in zip(torch.compile(w.step, dynamic=False)(qpos[0], qvel[0], ctrl[0]), w.step(qpos[0], qvel[0], ctrl[0])):
                torch.testing.assert_close(a, b, atol=tol, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rf_window_loss_under_torch_compile(kind, device):
    """The window loss and its gradients under torch.compile (fit_mjcf --compile) against eager. torch 2.11's Inductor cannot build
    a window loss's backward pass for the CPU (a C++ compile error, the MLP's too): skipped there; it runs with torch 2.15."""
    from torch._inductor.exc import InductorError

    wm, qpos, qvel, ctrl = compile_case(kind, device)
    try:
        loss = torch.compile(wm.window_loss, dynamic=False)(qpos[:5], qvel[:5], ctrl[:4])
        grads = torch.autograd.grad(loss, list(wm.parameters()))
    except InductorError as e:
        if "CppCompileError" not in str(e):
            raise
        pytest.skip(f"torch {torch.__version__}'s Inductor does not build the window loss's backward pass for the CPU (C++ compile error)")
    ref = wm.window_loss(qpos[:5], qvel[:5], ctrl[:4])
    torch.testing.assert_close(loss, ref, atol=1e-5, rtol=1e-5)
    for g, r in zip(grads, torch.autograd.grad(ref, list(wm.parameters()))):
        torch.testing.assert_close(g, r, atol=1e-5 if device == "cpu" else 1e-4, rtol=1e-3)
