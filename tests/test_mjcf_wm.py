"""The world model of MJCF models in generalized coordinates (ndm_gym.worldmodels.mjcf) against MuJoCo on CPU: the layout
read from a model with every joint type, a free anchor, a free object and three kinds of servo; the refused features; the
integration and the tangent difference against mj_integratePos and mj_differentiatePos; the servo forces against
mj_forward's; the inputs' invariance to the anchor's heading and horizontal position; one step against mj_step's Euler
integration given the true accelerations, and the increment model's step against mj_step's given the true increments; and
a short fit of each parameterization that lowers the open-loop error."""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel, differentiate_pos, integrate_pos, quat_mul  # noqa: E402

XML = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <default><joint armature="0.02"/></default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.2 0.1 0.05" mass="2"/>
      <body name="arm" pos="0.2 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="slider" pos="0.3 0 0">
          <joint name="extend" type="slide" axis="1 0 0" range="-0.1 0.1"/>
          <geom type="sphere" size="0.04"/>
          <body name="wrist" pos="0.05 0 0">
            <joint name="wrist" type="ball"/>
            <geom type="box" size="0.03 0.02 0.01" mass="0.2"/>
            <body name="spinner" pos="0.05 0 0">
              <joint name="spin" type="hinge" axis="1 0 0"/>
              <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="box" pos="1 0 0.2">
      <freejoint name="box"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="50" ctrlrange="-1 1" forcerange="-20 20"/>
    <position joint="extend" kp="200" kv="5"/>
    <velocity joint="spin" kv="2"/>
  </actuator>
</mujoco>
"""


def model():
    return mujoco.MjModel.from_xml_string(XML)


def random_state(m, n, rng):
    """n random states (qpos with unit quaternions, qvel)."""
    qpos = np.tile(m.qpos0, (n, 1)) + 0.3 * rng.standard_normal((n, m.nq))
    for j in range(m.njnt):
        if m.jnt_type[j] in (0, 1):
            a = m.jnt_qposadr[j] + (3 if m.jnt_type[j] == 0 else 0)
            qpos[:, a : a + 4] /= np.linalg.norm(qpos[:, a : a + 4], axis=1, keepdims=True)
    return qpos, rng.standard_normal((n, m.nv))


def test_layout():
    m = model()
    lay = MjcfLayout.from_model(m)
    assert (lay.nq, lay.nv, lay.nu) == (7 + 1 + 1 + 4 + 1 + 7, 6 + 1 + 1 + 3 + 1 + 6, 3)
    assert lay.anchor == (0, 0) and lay.objects == ((14, 12),) and lay.balls == ((9, 8),)
    assert lay.scalars == ((7, 6, False), (8, 7, False), (13, 11, True))  # the shoulder, the slide, the unlimited spin
    assert lay.slides == (7,)
    assert lay.act_q == (7, 8, 13) and lay.ctrl_range[0] == (-1.0, 1.0) and lay.force_range[0] == (-20.0, 20.0)
    assert lay.ctrl_range[1] is None and lay.bias[1] == (0.0, -200.0, -5.0) and lay.bias[2] == (0.0, 0.0, -2.0)
    assert MjcfLayout.from_model(m, anchor="box").anchor == (14, 12)
    with pytest.raises(ValueError, match="no free joint"):
        MjcfLayout.from_model(m, anchor="shoulder")


REFUSED = [
    ('<equality><weld body1="arm" body2="slider"/></equality>', "only joint equalities"),
    ('<equality><joint joint1="extend" joint2="shoulder" active="false"/></equality>', "inactive"),
    ('<equality><joint joint1="extend" joint2="shoulder"/><joint joint1="spin" joint2="extend"/></equality>', "chain"),
    ('<actuator><general joint="extend" dyntype="filter" dynprm="0.1"/></actuator>', "activation"),
    ('<tendon><spatial name="s"><site site="s0"/><site site="s1"/></spatial></tendon><actuator><motor tendon="s"/></actuator>', "not fixed"),
]


@pytest.mark.parametrize("extra, match", REFUSED)
def test_refused(extra, match):
    xml = XML.replace('<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>',
                      '<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/><site name="s0"/><site name="s1" pos="0.2 0 0"/>')  # fmt: skip
    with pytest.raises(ValueError, match=match):
        MjcfLayout.from_model(mujoco.MjModel.from_xml_string(xml.replace("</mujoco>", extra + "</mujoco>")))


COUPLED = XML.replace("</mujoco>", """<equality><joint joint1="extend" joint2="shoulder" polycoef="0.01 0.05 -0.02 0.003 0.001"/>
  <joint joint1="spin" polycoef="0.4 0 0 0 0"/></equality>
  <tendon><fixed name="pair"><joint joint="shoulder" coef="0.5"/><joint joint="extend" coef="-2"/></fixed></tendon>
  <actuator><position tendon="pair" kp="30" kv="1" gear="1.5" ctrlrange="-0.5 0.5"/></actuator>
</mujoco>""")


def test_couplings_and_tendons_read_and_hold():
    m, rng = mujoco.MjModel.from_xml_string(COUPLED), np.random.default_rng(10)
    lay = MjcfLayout.from_model(m)
    assert lay.act_q[3] == -1 and lay.tendons == ((3, ((7, 6, 0.75), (8, 7, -3.0))),)
    (y, yd, x, xd, y0, x0, a), (y2, _, x2, _, _, _, a2) = lay.couplings
    assert (y, yd, x, xd) == (8, 7, 7, 6) and (y2, x2) == (13, -1) and a2 == (0.4, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(a, (0.01, 0.05, -0.02, 0.003, 0.001)) and (y0, x0) == (m.qpos0[8], m.qpos0[7])
    d = mujoco.MjData(m)
    wm = MjcfWorldModel(lay, hidden=(8,)).double()
    qpos, qvel = random_state(m, 8, rng)
    ctrl = 2.0 * rng.standard_normal((8, m.nu))
    force, _ = wm.servo(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(ctrl))
    for i in range(8):  # the tendon actuator's force, as MuJoCo computes it
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], ctrl[i]
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(force[i].numpy(), d.actuator_force, atol=1e-9)
    for kind in MjcfWorldModel.KINDS:  # every step leaves the coupled joints on their polynomials
        wm = MjcfWorldModel(lay, hidden=(16,), kind=kind).double()
        for p in wm.net.parameters():
            torch.nn.init.normal_(p, std=0.3)
        with torch.no_grad():
            q1, v1 = wm.step(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(ctrl))
        a, y0, x0 = wm.cp_a[0], wm.cp_y0[0], wm.cp_x0[0]  # the coefficients as the model holds them (float32, then double)
        t = q1[:, 7] - x0
        np.testing.assert_allclose(q1[:, 8].numpy(), (y0 + a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4).numpy(), atol=1e-12)
        np.testing.assert_allclose(v1[:, 7].numpy(), ((a[1] + 2 * a[2] * t + 3 * a[3] * t**2 + 4 * a[4] * t**3) * v1[:, 6]).numpy(), atol=1e-12)
        np.testing.assert_allclose(q1[:, 13].numpy(), (wm.cp_y0[1] + wm.cp_a[1, 0]).item(), atol=1e-12)
        np.testing.assert_allclose(q1[:, 13].numpy(), m.qpos0[13] + 0.4, atol=1e-7)
        assert (v1[:, 11] == 0).all()
    plain = MjcfWorldModel(MjcfLayout.from_model(model()), hidden=(8,)).double()
    q, v = torch.tensor(qpos), torch.tensor(qvel)
    assert all(a is b for a, b in zip(plain.couple(q, v), (q, v)))  # no equalities: the state as it was


def test_integration_and_tangent_match_mujoco():
    m, rng = model(), np.random.default_rng(0)
    lay = MjcfLayout.from_model(m)
    qpos, qvel = random_state(m, 16, rng)
    for dt in (0.01, 0.37):
        mine = integrate_pos(lay, torch.tensor(qpos), torch.tensor(qvel), dt).numpy()
        for i in range(len(qpos)):
            ref = qpos[i].copy()
            mujoco.mj_integratePos(m, ref, qvel[i], dt)
            np.testing.assert_allclose(mine[i], ref, atol=1e-10)
            tan = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, tan, 1.0, qpos[i], ref)
            np.testing.assert_allclose(differentiate_pos(lay, torch.tensor(qpos[i]), torch.tensor(ref)).numpy(), tan, atol=1e-9)
            np.testing.assert_allclose(tan, dt * qvel[i], atol=1e-9)  # the tangent of a step is its velocity times dt


def test_servo_forces_match_mujoco():
    m, rng = model(), np.random.default_rng(1)
    d = mujoco.MjData(m)
    wm = MjcfWorldModel(MjcfLayout.from_model(m), hidden=(8,)).double()
    qpos, qvel = random_state(m, 8, rng)
    ctrl = 2.0 * rng.standard_normal((8, m.nu))  # beyond the shoulder's range at times
    force, offset = wm.servo(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(ctrl))
    for i in range(8):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], ctrl[i]
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(force[i].numpy(), d.actuator_force, atol=1e-9)
        np.testing.assert_allclose(offset[i, 0].item(), np.clip(ctrl[i, 0], -1, 1) - d.actuator_length[0], atol=1e-9)


def test_inputs_ignore_the_anchors_heading_and_horizontal_position():
    m, rng = model(), np.random.default_rng(2)
    lay = MjcfLayout.from_model(m)
    wm = MjcfWorldModel(lay, hidden=(8,)).double()
    qpos, qvel = (torch.tensor(x) for x in random_state(m, 6, rng))
    ctrl = torch.tensor(rng.standard_normal((6, m.nu)))
    phi, shift = 0.9, torch.tensor([0.7, -1.3, 0.0], dtype=torch.float64)
    Q = torch.tensor([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    qz = torch.tensor([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], dtype=torch.float64)
    q2, v2 = qpos.clone(), qvel.clone()
    for qa, da in lay.free:  # the whole scene turned about the world's z axis and moved horizontally
        q2[:, qa : qa + 3] = qpos[:, qa : qa + 3] @ Q.T + shift
        q2[:, qa + 3 : qa + 7] = quat_mul(qz.expand(6, 4), qpos[:, qa + 3 : qa + 7])
        v2[:, da : da + 3] = qvel[:, da : da + 3] @ Q.T  # world-frame linear velocity; the angular one is the body's
    torch.testing.assert_close(wm.inputs(q2, v2, ctrl), wm.inputs(qpos, qvel, ctrl), atol=1e-9, rtol=0)


def test_scalar_features_are_the_per_joint_loops():
    """The hinges' and slides' features, gathered at once, are bit for bit those of a loop over the joints."""
    m, rng = model(), np.random.default_rng(9)
    lay = MjcfLayout.from_model(m)
    wm = MjcfWorldModel(lay, hidden=(8,))
    qpos, qvel = (torch.tensor(x, dtype=torch.float32) for x in random_state(m, 5, rng))
    parts = []
    for qa, da, wrap in lay.scalars:
        q = qpos[..., qa : qa + 1]
        parts += [torch.sin(q), torch.cos(q)] if wrap else [q]
        parts.append(qvel[..., da : da + 1])
    want = torch.cat(parts, -1)
    n_free = sum(9 + 6 for _ in lay.free) + sum(9 for _ in lay.balls)  # position, 6D rotation and the velocities; balls: 6D and 3
    got = wm.inputs(qpos, qvel, torch.zeros(5, m.nu))[..., n_free : n_free + want.shape[-1]]
    assert torch.equal(got, want)
    assert any(w for _, _, w in lay.scalars) and not all(w for _, _, w in lay.scalars)  # both kinds of scalar are in the model


def test_step_is_mujocos_euler_step_given_the_true_accelerations():
    m, rng = model(), np.random.default_rng(3)
    d = mujoco.MjData(m)
    lay = MjcfLayout.from_model(m)
    for k in range(4):
        qpos, qvel = random_state(m, 1, rng)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[0], qvel[0], rng.standard_normal(m.nu)
        mujoco.mj_step(m, d)
        q1, v1 = d.qpos.copy(), d.qvel.copy()
        wm = MjcfWorldModel(lay, hidden=(8,)).double()  # its last layer is zero: the step predicts out_norm's mean
        target = wm.targets(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(q1)[None], torch.tensor(v1)[None])
        wm.out_norm.mean.copy_(target[0])
        with torch.no_grad():
            qh, vh = wm.step(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(d.ctrl)[None])
        np.testing.assert_allclose(vh[0].numpy(), v1, atol=1e-9)
        np.testing.assert_allclose(qh[0].numpy(), q1, atol=1e-9)


@pytest.mark.parametrize("integrator", ["Euler", "RK4", "implicitfast"])
def test_increment_step_is_mujocos_step_given_the_true_increments(integrator):
    m, rng = mujoco.MjModel.from_xml_string(XML.replace('integrator="Euler"', f'integrator="{integrator}"')), np.random.default_rng(8)
    d = mujoco.MjData(m)
    lay = MjcfLayout.from_model(m)
    for k in range(4):
        qpos, qvel = random_state(m, 1, rng)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[0], qvel[0], rng.standard_normal(m.nu)
        mujoco.mj_step(m, d)
        q1, v1 = d.qpos.copy(), d.qvel.copy()
        wm = MjcfWorldModel(lay, hidden=(8,), kind="increment").double()  # the step predicts out_norm's mean
        target = wm.targets(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(q1)[None], torch.tensor(v1)[None])
        assert target.shape == (1, 2 * m.nv)
        wm.out_norm.mean.copy_(target[0])
        with torch.no_grad():
            qh, vh = wm.step(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(d.ctrl)[None])
        np.testing.assert_allclose(vh[0].numpy(), v1, atol=1e-9)
        np.testing.assert_allclose(qh[0].numpy(), q1, atol=1e-9)
    with pytest.raises(ValueError, match="increments are of"):
        wm.step(torch.tensor(qpos), torch.tensor(qvel), torch.tensor(d.ctrl)[None], dt=2 * m.opt.timestep)
    with pytest.raises(ValueError, match="kind"):
        MjcfWorldModel(lay, hidden=(8,), kind="velocity")


def simulate(m, n, steps, rng):
    """n episodes of ``steps`` steps from random poses near qpos0, controls a random walk: (qpos, qvel) (steps + 1, n, .), ctrl."""
    d = mujoco.MjData(m)
    qpos, qvel, ctrl = np.zeros((steps + 1, n, m.nq)), np.zeros((steps + 1, n, m.nv)), np.zeros((steps, n, m.nu))
    for i in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[7:14] += 0.2 * rng.standard_normal(7)
        d.qpos[9:13] /= np.linalg.norm(d.qpos[9:13])
        u = np.zeros(m.nu)
        for t in range(steps):
            qpos[t, i], qvel[t, i] = d.qpos, d.qvel
            u = np.clip(u + 0.2 * rng.standard_normal(m.nu), -1, 1)  # scaled per actuator below: the slide moves 0.1 m
            d.ctrl[:] = ctrl[t, i] = u * np.array([1.0, 0.1, 1.0])
            mujoco.mj_step(m, d)
        qpos[steps, i], qvel[steps, i] = d.qpos, d.qvel
    return (torch.tensor(x, dtype=torch.float32) for x in (qpos, qvel, ctrl))


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_a_short_fit_lowers_the_open_loop_error(kind):
    m, rng = model(), np.random.default_rng(4)
    torch.manual_seed(0)
    qpos, qvel, ctrl = simulate(m, 32, 60, rng)
    wm = MjcfWorldModel(MjcfLayout.from_model(m), hidden=(128, 128), kind=kind)
    wm.update_norms(qpos[:-1], qvel[:-1], ctrl, qpos[1:], qvel[1:])
    opt = torch.optim.Adam(wm.parameters(), lr=2e-3)

    def loss(k=4):
        starts = torch.randint(0, 60 - k, (32,))
        sl = lambda x, n: torch.stack([x[s : s + n, i] for i, s in enumerate(starts.tolist())], 1)  # noqa: E731
        return wm.window_loss(sl(qpos, k + 1), sl(qvel, k + 1), sl(ctrl, k))

    with torch.no_grad():
        before = loss().item()
    for _ in range(300):
        opt.zero_grad()
        x = loss()
        x.backward()
        opt.step()
    with torch.no_grad():
        after = loss().item()
    assert after < 0.5 * before, (before, after)


def test_fit_windows_never_cross_a_reset():
    from ndm_gym.wmbench.fit_mjcf import starts

    reset = torch.zeros(6, 2, dtype=torch.bool)
    reset[2, 0] = True  # env 0: the state after step 2 does not follow from it
    got = {tuple(x) for x in starts({"reset": reset}, 3).tolist()}
    assert got == {(3, 0), (0, 1), (1, 1), (2, 1), (3, 1)}  # env 0's windows 0-2, 1-3 and 2-4 hold step 2


def test_fit_mjcf_runs_and_scores(tmp_path):
    from ndm_gym.wmbench.fit_mjcf import SCORE_COLUMNS, main

    m, rng = model(), np.random.default_rng(5)
    xml = tmp_path / "scene.xml"
    xml.write_text(XML)
    for name, n in (("train", 8), ("test", 4)):
        qpos, qvel, ctrl = simulate(m, n, 30, rng)
        reset = np.zeros((30, n), dtype=bool)
        reset[14, :] = True  # as if every episode restarted after step 14
        np.savez(tmp_path / f"{name}.npz", qpos=qpos.numpy(), qvel=qvel.numpy(), ctrl=ctrl.numpy(), reset=reset, dt=m.opt.timestep)
    out = tmp_path / "fit.csv"
    main(["--model", str(xml), "--train", str(tmp_path / "train.npz"), "--test", f"held={tmp_path / 'test.npz'}", "--steps", "40",
          "--batch", "16", "--window", "4", "--hidden", "32", "--amp", "none", "--eval_every", "20", "--eval_k", "1", "4",
          "--eval_windows", "16", "--out", str(out), "--save", str(tmp_path / "wm.pt")])  # fmt: skip
    rows = out.read_text().strip().splitlines()
    assert rows[0] == "step,seconds,loss," + SCORE_COLUMNS and len(rows) == 1 + 2 * 2
    last = dict(zip(rows[0].split(","), rows[-1].split(",")))
    assert last["test"] == "held" and last["k"] == "4" and float(last["anchor_mm"]) > 0 and float(last["obj_mm"]) >= 0
    assert (tmp_path / "wm.pt").exists()
    main(["--model", str(xml), "--train", str(tmp_path / "train.npz"), "--test", f"held={tmp_path / 'test.npz'}", "--steps", "20",
          "--batch", "16", "--window", "4", "--hidden", "32", "--amp", "none", "--eval_every", "20", "--eval_k", "4",
          "--eval_windows", "16", "--wm_param", "increment", "--no_ln", "--out", str(tmp_path / "inc.csv"),
          "--save", str(tmp_path / "inc.pt")])  # fmt: skip
    from types import SimpleNamespace

    from ndm_gym.wmbench.speed_mjcf import load_ndm

    lay = MjcfLayout.from_model(m)
    for path, kind, ln in (("wm.pt", "accel", True), ("inc.pt", "increment", False)):  # the timed model is the fitted one
        wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / path)), "cpu")
        assert wm.kind == kind and any(isinstance(x, torch.nn.LayerNorm) for x in wm.net) == ln
        saved = torch.load(tmp_path / path, weights_only=False)["state"]
        assert all(torch.equal(v, saved[k]) for k, v in wm.state_dict().items())
    with pytest.raises(ValueError, match="fitted on a layout with anchor"):  # the box as the anchor: another frame
        load_ndm(MjcfLayout.from_model(m, anchor="box"), SimpleNamespace(ckpt=str(tmp_path / "wm.pt")), "cpu")
    bad = tmp_path / "bad.npz"
    np.savez(bad, qpos=qpos.numpy(), qvel=qvel.numpy(), ctrl=ctrl.numpy(), reset=reset, dt=0.005)
    with pytest.raises(ValueError, match="the world model's are 0.01 s"):
        main(["--model", str(xml), "--train", str(bad), "--test", f"held={bad}", "--steps", "1", "--amp", "none", "--out", str(tmp_path / "x.csv")])
    main(["--model", str(xml), "--train", str(bad), "--test", f"held={bad}", "--steps", "1", "--amp", "none", "--dt", "0.005", "--eval_k", "1",
          "--hidden", "8", "--batch", "4", "--window", "2", "--out", str(tmp_path / "x.csv"), "--save", str(tmp_path / "x.pt")])  # a stage's own step
    assert load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / "x.pt")), "cpu").lay.dt == 0.005


def test_inference_copy_on_cpu():
    from ndm_gym.worldmodels.mjcf import inference_copy

    m, rng = model(), np.random.default_rng(6)
    wm = MjcfWorldModel(MjcfLayout.from_model(m), hidden=(64, 64))
    for p in wm.net.parameters():
        torch.nn.init.normal_(p, std=0.1)
    qpos, qvel = (torch.tensor(x, dtype=torch.float32) for x in random_state(m, 16, rng))
    ctrl = torch.tensor(rng.standard_normal((16, m.nu)), dtype=torch.float32)
    with torch.no_grad():
        ref = wm.step(qpos, qvel, ctrl)
        same = inference_copy(wm, None).step(qpos, qvel, ctrl)
        half = inference_copy(wm, torch.bfloat16).step(qpos, qvel, ctrl)
    for a, b in zip(ref, same):
        assert torch.equal(a, b)
    assert all(p.dtype == torch.bfloat16 for p in inference_copy(wm, torch.bfloat16).net.parameters())
    for a, b in zip(ref, half):
        assert (a - b).abs().max() < 0.05
    assert not any(p.requires_grad for p in inference_copy(wm, None).parameters())


def test_speed_start_states(tmp_path):
    from ndm_gym.wmbench.speed_mjcf import start_states

    m, rng = model(), np.random.default_rng(7)
    qpos, qvel, ctrl = simulate(m, 2, 5, rng)
    np.savez(tmp_path / "r.npz", qpos=qpos.numpy(), qvel=qvel.numpy(), ctrl=ctrl.numpy(), reset=np.zeros((5, 2), bool), dt=m.opt.timestep)
    q, v, u = start_states(str(tmp_path / "r.npz"), 25)
    assert q.shape == (25, m.nq) and v.shape == (25, m.nv) and u.shape == (25, m.nu)
    pool = {tuple(np.round(x, 6)) for x in qpos[:-1].reshape(-1, m.nq).numpy()}
    assert all(tuple(np.round(x, 6)) in pool for x in q)  # every start is a recorded state (not a successor-only one)


def test_cut_blowups_cuts_a_rollout_from_its_first_outlier_to_its_reset():
    from ndm_gym.wmbench.fit_mjcf import cut_blowups

    m, rng = model(), np.random.default_rng(11)
    lay = MjcfLayout.from_model(m)
    qpos, qvel, ctrl = simulate(m, 2, 12, rng)
    reset = torch.zeros(12, 2, dtype=torch.bool)
    reset[8, 0] = True  # world 0: a new rollout after step 8
    qpos, qvel = qpos.clone(), qvel.clone()
    clean = {"qpos": qpos.clone(), "qvel": qvel.clone(), "ctrl": ctrl, "reset": reset.clone()}
    assert cut_blowups(clean, lay, 300.0) == 0.0 and torch.equal(clean["reset"], reset)  # the same rollouts without a blow-up
    qpos[6:9, 0, 2] += 1000.0  # world 0 blows up at step 5 (its height jumps by a kilometre) and recovers only with the reset
    rec = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl, "reset": reset.clone()}
    share = cut_blowups(rec, lay, 300.0)
    r = rec["reset"]
    assert r[5:9, 0].all() and not r[:5, 0].any() and not r[9:, 0].any()  # cut from the outlier to the reset, kept after it
    assert not r[:, 1].any()  # the other world untouched
    assert share == pytest.approx(3 / 23)  # steps 5, 6, 7 newly cut, of the 23 kept before
    from ndm_gym.wmbench.fit_mjcf import median_increments

    one = {"qpos": qpos[:, :1].clone(), "qvel": qvel[:, :1].clone(), "ctrl": ctrl[:, :1], "reset": torch.zeros(12, 1, dtype=torch.bool)}
    one["qpos"][6:, 0, 2] = qpos[6:, 1, 2]  # a long trajectory with one fast step, no blow-up at the scale of other recordings
    wide = median_increments([{"qpos": qpos * 100, "qvel": qvel * 100, "reset": torch.zeros(12, 2, dtype=torch.bool)}], lay)
    assert cut_blowups(one, lay, 300.0, wide) == 0.0  # judged by the given (training) medians, not by its own
