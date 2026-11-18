"""The velocity the acceleration model's step advances the position with (ndm_gym.worldmodels.mjcf:
``MjcfWorldModel(pos_lead=...)``, ``mean_velocity_lead``; fit_mjcf ``--accel_mean_velocity``) on CPU: by default the next
velocity, as before, with no entry in the model's state; a control step's weight against MuJoCo's own steps of a cart under a
constant force (Euler and implicitfast with one, two and five physics steps, RK4), where an exact next velocity then gives
MuJoCo's position and the default leaves a rest worked out by hand; the world model's step with that weight against the same
steps; what is refused; and a fit made with the option holds the weight in its saved state and loads without the option."""

import dataclasses
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_fit_mjcf_options import FIT, same, save  # noqa: E402

from ndm_gym.wmbench.fit_mjcf import main  # noqa: E402
from ndm_gym.wmbench.speed_mjcf import load_ndm  # noqa: E402
from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel, mean_velocity_lead  # noqa: E402

# a cart of 2 kg on a slide, pushed by a motor of gear 10: a control of 0.6 accelerates it by 3 m/s^2, whatever its state
CART = """
<mujoco>
  <option timestep="0.01" integrator="INTEGRATOR"/>
  <worldbody>
    <body name="cart"><joint name="x" type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.1 0.1" mass="2"/></body>
  </worldbody>
  <actuator><motor joint="x" gear="10"/></actuator>
</mujoco>
"""
Q0, V0, U, A = 0.3, -0.7, 0.6, 3.0
CASES = [("Euler", 1, 1.0), ("Euler", 2, 0.75), ("Euler", 5, 0.6), ("implicitfast", 2, 0.75), ("RK4", 1, 0.5), ("RK4", 4, 0.5)]


def cart(integrator: str):
    return mujoco.MjModel.from_xml_string(CART.replace("INTEGRATOR", integrator))


def stepped(m, k: int):
    """The cart's position and velocity after one control step of k physics steps from (Q0, V0) under the control U."""
    d = mujoco.MjData(m)
    d.qpos[0], d.qvel[0], d.ctrl[0] = Q0, V0, U
    for _ in range(k):
        mujoco.mj_step(m, d)
    return float(d.qpos[0]), float(d.qvel[0])


@pytest.mark.parametrize("integrator, k, lead", CASES)
def test_the_weight_of_a_control_step_against_mujocos_steps(integrator, k, lead):
    m = cart(integrator)
    dt = k * m.opt.timestep
    assert mean_velocity_lead(m, k) == pytest.approx(lead, abs=1e-15)
    q1, v1 = stepped(m, k)
    assert v1 == pytest.approx(V0 + A * dt, abs=1e-12)  # the constant acceleration
    assert q1 == pytest.approx(Q0 + dt * (V0 + lead * (v1 - V0)), abs=1e-12)  # the mean of the velocities the step passes through
    assert Q0 + dt * v1 - q1 == pytest.approx((1 - lead) * A * dt**2, abs=1e-12)  # what the next velocity alone leaves


@torch.no_grad()
@pytest.mark.parametrize("integrator, k, lead", CASES)
def test_the_models_step_with_the_weight_is_mujocos(integrator, k, lead):
    m = cart(integrator)
    lay = dataclasses.replace(MjcfLayout.from_model(m), dt=k * m.opt.timestep)
    q, v, u = (torch.tensor([[x]], dtype=torch.float64) for x in (Q0, V0, U))
    q1, v1 = stepped(m, k)
    plain = MjcfWorldModel(lay, hidden=(8,)).double()
    plain.out_norm.mean.fill_(A)  # the network starts at zero: the model's acceleration is the output normalizer's mean
    assert not plain._lead and "pos_lead" not in plain.state_dict()  # off unless asked for, and nothing in the state
    a, b = plain.step(q, v, u)
    assert float(b) == pytest.approx(v1, abs=1e-9) and float(a) == pytest.approx(Q0 + lay.dt * v1, abs=1e-9)  # the next velocity, as before
    if lead == 1.0:
        assert float(a) == pytest.approx(q1, abs=1e-9)
        return
    wm = MjcfWorldModel(lay, hidden=(8,), pos_lead=lead).double()
    wm.out_norm.mean.fill_(A)
    assert wm._lead and float(wm.state_dict()["pos_lead"]) == pytest.approx(lead, abs=1e-7)
    assert set(wm.state_dict()) == set(plain.state_dict()) | {"pos_lead"}
    a, b = wm.step(q, v, u)
    assert float(b) == pytest.approx(v1, abs=1e-9) and float(a) == pytest.approx(q1, abs=1e-8)  # MuJoCo's control step
    assert abs(float(plain.step(q, v, u)[0]) - q1) == pytest.approx((1 - lead) * A * lay.dt**2, abs=1e-9)
    with pytest.raises(ValueError, match="its mean velocity is that of the model's"):
        wm.step(q, v, u, dt=2 * lay.dt)
    plain.step(q, v, u, dt=2 * lay.dt)  # the default steps any dt, as before


def test_what_is_refused():
    m = cart("Euler")
    lay = MjcfLayout.from_model(m)
    for bad in (0.0, 1.5, -0.5):
        with pytest.raises(ValueError, match="within \\(0, 1\\]"):
            MjcfWorldModel(lay, hidden=(8,), pos_lead=bad)
    with pytest.raises(ValueError, match="1 unless kind='accel'"):
        MjcfWorldModel(lay, hidden=(8,), kind="increment", pos_lead=0.5)
    assert not MjcfWorldModel(lay, hidden=(8,), kind="increment")._lead
    with pytest.raises(ValueError, match="0 physics steps"):
        mean_velocity_lead(m, 0)
    m.opt.integrator = 7  # no such integrator
    with pytest.raises(ValueError, match="integrator 7"):
        mean_velocity_lead(m, 2)


def test_fit_mjcf_accel_mean_velocity_is_saved_with_the_fit(tmp_path, capsys):
    xml = CART.replace("INTEGRATOR", "Euler")
    (tmp_path / "cart.xml").write_text(xml)
    m, rng = mujoco.MjModel.from_xml_string(xml), np.random.default_rng(0)

    def record(name, k, n=6, steps=24):  # control steps of k physics steps under random controls
        d = mujoco.MjData(m)
        qpos, qvel, ctrl = np.zeros((steps + 1, n, 1)), np.zeros((steps + 1, n, 1)), rng.uniform(-1, 1, (steps, n, 1))
        for i in range(n):
            mujoco.mj_resetData(m, d)
            d.qvel[0] = rng.standard_normal()
            for t in range(steps):
                qpos[t, i], qvel[t, i], d.ctrl[:] = d.qpos, d.qvel, ctrl[t, i]
                for _ in range(k):
                    mujoco.mj_step(m, d)
            qpos[steps, i], qvel[steps, i] = d.qpos, d.qvel
        return save(tmp_path / f"{name}.npz", qpos, qvel, ctrl, k * m.opt.timestep)

    two, one = record("two", 2), record("one", 1)

    def fit(name, rec, *extra):
        main(["--model", str(tmp_path / "cart.xml"), "--train", rec, "--test", f"held={rec}", *FIT, *extra,
              "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
        return torch.load(tmp_path / f"{name}.pt", weights_only=False)

    a = fit("plain", two, "--substeps", "2")
    capsys.readouterr()
    b = fit("lead", two, "--substeps", "2", "--accel_mean_velocity")
    out = capsys.readouterr().out
    assert "--accel_mean_velocity: the position advances with v + 0.75 (v' - v) (integrator 0, 2 physics steps a control step)" in out
    assert a["args"]["accel_mean_velocity"] is False and "pos_lead" not in a["state"]
    assert b["args"]["accel_mean_velocity"] is True and float(b["state"]["pos_lead"]) == 0.75
    assert set(b["state"]) == set(a["state"]) | {"pos_lead"} and set(a) == set(b)
    lay = MjcfLayout.from_model(m)
    wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / "lead.pt")), torch.device("cpu"))  # nothing but the saved fit
    assert wm._lead and float(wm.pos_lead) == 0.75 and wm.lay.dt == pytest.approx(0.02) and same(wm.state_dict(), b["state"])
    assert not load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / "plain.pt")), torch.device("cpu"))._lead
    c = fit("single", one, "--accel_mean_velocity")  # one semi-implicit Euler step: the next velocity, as without the option
    assert "v + 1 (v' - v) (integrator 0, 1 physics steps a control step)" in capsys.readouterr().out and "pos_lead" not in c["state"]
    with pytest.raises(ValueError, match="--accel_mean_velocity is an option of --wm_param accel"):
        fit("x", two, "--substeps", "2", "--accel_mean_velocity", "--wm_param", "increment")
    with pytest.raises(ValueError, match="--accel_mean_velocity is an option of --wm_param accel, and not of --init"):
        fit("y", two, "--substeps", "2", "--accel_mean_velocity", "--init", str(tmp_path / "lead.pt"))
    with pytest.raises(ValueError, match="name the step as --substeps"):
        fit("z", two, "--dt", "0.02", "--accel_mean_velocity")
    again = fit("again", two, "--substeps", "2", "--init", str(tmp_path / "lead.pt"))  # a refit keeps the start's weight
    assert float(again["state"]["pos_lead"]) == 0.75 and set(again["state"]) == set(b["state"])
    assert "pos_lead" not in fit("again_plain", two, "--substeps", "2", "--init", str(tmp_path / "plain.pt"))["state"]
