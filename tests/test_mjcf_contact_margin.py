"""The margin the contact inputs read their gaps beyond (ndm_gym.worldmodels.mjcf: ``contact_spheres(pair_margin=True)``;
fit_mjcf ``--contact_pair_margin``) on CPU: by default the largest of the geoms' margins, as before; with the option the margin
MuJoCo gives a contact between a robot geom and an object's or a static geom, read off MuJoCo's own contacts of the same
models; what the option refuses; the first three inputs by hand with both margins; and a fit made with the option holds its
margin in its saved state and loads without the option."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_fit_mjcf_options import FIT, recorded, same  # noqa: E402
from test_mjcf_contacts import FOOT, PROBES  # noqa: E402

from ndm_gym.sim.mbrl_envs import contact_margin_rule  # noqa: E402
from ndm_gym.wmbench.fit_mjcf import main  # noqa: E402
from ndm_gym.wmbench.speed_mjcf import load_ndm  # noqa: E402
from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel, contact_spheres  # noqa: E402

F64 = dict(dtype=torch.float64)
# the capsule foot with a margin of 3 mm above the floor with 1 mm (the step keeps none)
FOOT_M = FOOT.replace('<geom name="floor" type="plane"', '<geom name="floor" type="plane" margin="0.001"').replace(
    '<geom name="foot" type="capsule"', '<geom name="foot" type="capsule" margin="0.003"')  # fmt: skip
# the three probes above a free box with a margin of 4 mm, the ball with 2 mm (the rod and the brick keep none)
PROBES_M = (PROBES.replace("KIND", "box").replace("SIZE", "0.1 0.1 0.05").replace("MARGIN", 'margin="0.004"')
            .replace('<geom name="ball" type="sphere"', '<geom name="ball" type="sphere" margin="0.002"'))  # fmt: skip


def put(rule, larger, total):
    return total if rule == "sum" else larger


def contact_margins(m, qpos):
    """The margins of MuJoCo's contacts at ``qpos``, by the two geoms' names."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))  # noqa: E731
    return {frozenset((name(c.geom[0]), name(c.geom[1]))): float(c.includemargin) for c in d.contact[: d.ncon]}


def test_the_default_margin_is_the_largest_of_the_geoms_as_before():
    m = mujoco.MjModel.from_xml_string(FOOT_M)
    lay = MjcfLayout.from_model(m)
    assert contact_spheres(m, lay, static=("floor",))["margin"] == pytest.approx(0.003, abs=1e-12)  # the foot's, whatever the floor's
    assert contact_spheres(m, lay, static=())["margin"] == pytest.approx(0.003, abs=1e-12)
    m = mujoco.MjModel.from_xml_string(PROBES_M)
    assert contact_spheres(m, MjcfLayout.from_model(m))["margin"] == pytest.approx(0.004, abs=1e-12)  # the object's


def test_pair_margin_is_the_margin_of_mujocos_contacts():
    rule = contact_margin_rule()
    assert rule in ("max", "sum")
    m = mujoco.MjModel.from_xml_string(FOOT_M)
    lay = MjcfLayout.from_model(m)
    theirs = contact_margins(m, [0.0, -0.46, 0.0])  # the foot's lowest line on the floor
    assert set(theirs) == {frozenset(("foot", "floor"))}
    floor = contact_spheres(m, lay, static=("floor",), pair_margin=True)["margin"]
    assert floor == theirs[frozenset(("foot", "floor"))] == pytest.approx(put(rule, 0.003, 0.004), abs=1e-12)
    step = contact_margins(m, [2.0, -0.26, 0.0])[frozenset(("foot", "step"))]  # on the step, 0.2 m up, which has no margin
    assert contact_spheres(m, lay, static=("step",), pair_margin=True)["margin"] == step == pytest.approx(0.003, abs=1e-12)
    assert contact_spheres(m, lay, static=(), pair_margin=True)["margin"] == floor  # the floor and the step: the larger of the two contacts'
    plain, pair = contact_spheres(m, lay, static=("floor",)), contact_spheres(m, lay, static=("floor",), pair_margin=True)
    assert plain.keys() == pair.keys() and all(np.array_equal(plain[k], pair[k]) for k in plain if k != "margin")  # nothing else changes

    m = mujoco.MjModel.from_xml_string(PROBES_M)
    lay = MjcfLayout.from_model(m)
    ball = contact_margins(m, [-0.18, 0.0, 0.0, 0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])[frozenset(("ball", "obj"))]  # the ball on the box's top
    rod = contact_margins(m, [0.0, -0.185, 0.0, 0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])[frozenset(("rod", "obj"))]  # the box under the rod
    assert ball == pytest.approx(put(rule, 0.004, 0.006), abs=1e-12) and rod == pytest.approx(0.004, abs=1e-12)
    assert contact_spheres(m, lay, pair_margin=True)["margin"] == max(ball, rod)  # the largest among the robot's geoms' contacts


def test_what_pair_margin_refuses():
    m = mujoco.MjModel.from_xml_string(FOOT_M)
    lay = MjcfLayout.from_model(m)
    with pytest.raises(ValueError, match="no object's geom and no static geom"):
        contact_spheres(m, lay, pair_margin=True)
    for change in (lambda x: x.replace("</worldbody>", '</worldbody><contact><pair geom1="foot" geom2="floor"/></contact>'),
                   lambda x: x.replace('<option timestep="0.01"/>', '<option timestep="0.01"><flag override="enable"/></option>'),
                   lambda x: x.replace('margin="0.003"', 'margin="0.003" gap="0.001"')):  # fmt: skip
        xml = change(FOOT_M)
        assert xml != FOOT_M
        m = mujoco.MjModel.from_xml_string(xml)
        lay = MjcfLayout.from_model(m)
        assert contact_spheres(m, lay, static=("floor",))["margin"] == pytest.approx(0.003, abs=1e-12)  # the default reads the geoms as before
        with pytest.raises(ValueError, match="explicit contact pairs, the contact override or geoms with a gap"):
            contact_spheres(m, lay, static=("floor",), pair_margin=True)


def test_the_gap_inputs_by_hand_with_both_margins():
    m = mujoco.MjModel.from_xml_string(FOOT_M)
    lay = MjcfLayout.from_model(m)
    heights = (0.0, 0.0035, 0.01)  # the foot's lowest line above the floor
    q = torch.tensor([[0.0, h - 0.46, 0.0] for h in heights], **F64)
    for more, margin in (({}, 0.003), ({"pair_margin": True}, put(contact_margin_rule(), 0.003, 0.004))):
        wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, static=("floor",), **more)).double()
        assert float(wm.cs_margin) == pytest.approx(margin, abs=1e-9)
        x = wm.contact_inputs(q, torch.zeros(3, 3, **F64))
        beyond = [h - margin for h in heights]  # the gap beyond the margin: in units of 5 mm between -2 and 20 mm; its gate; the overlap in 0.1 mm
        assert x[:, 0].tolist() == pytest.approx([min(max(s, -0.002), 0.02) / 0.005 for s in beyond], abs=1e-5)
        assert x[:, 1].tolist() == pytest.approx([math.exp(-max(s, 0.0) / 0.0005) for s in beyond], abs=1e-5)
        assert x[:, 2].tolist() == pytest.approx([min(max(-s, 0.0), 0.002) / 1e-4 for s in beyond], abs=1e-3)
    if contact_margin_rule() == "sum":  # 3.5 mm up: beyond the foot's 3 mm, inside the contact's 4 mm
        assert x[1, :3].tolist() == pytest.approx([-0.1, 1.0, 5.0], abs=1e-3)


def test_fit_mjcf_contact_pair_margin_is_saved_with_the_fit(tmp_path, capsys):
    drop = lambda rng: np.array([0.0, -0.36 + 0.05 * rng.random(), 0.3 * rng.standard_normal()])  # noqa: E731  the foot 0.1-0.15 m up: it lands
    (tmp_path / "foot.xml").write_text(FOOT_M)
    train, test = recorded(FOOT_M, tmp_path / "train.npz", drop), recorded(FOOT_M, tmp_path / "test.npz", drop, n=3, seed=1)

    def fit(name, *extra):
        main(["--model", str(tmp_path / "foot.xml"), "--train", train, "--test", f"held={test}", *FIT, *extra,
              "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
        return str(tmp_path / f"{name}.pt")

    floor = ("--contacts", "--contact_static", "floor")
    margin = put(contact_margin_rule(), 0.003, 0.004)
    a = torch.load(fit("plain", *floor), weights_only=False)
    assert "margin 3 mm" in capsys.readouterr().out
    b = torch.load(fit("pair", *floor, "--contact_pair_margin"), weights_only=False)
    assert f"margin {1e3 * margin:g} mm" in capsys.readouterr().out
    assert a["args"]["contact_pair_margin"] is False and float(a["state"]["cs_margin"]) == pytest.approx(0.003, abs=1e-9)
    assert b["args"]["contact_pair_margin"] is True and float(b["state"]["cs_margin"]) == pytest.approx(margin, abs=1e-9)
    assert a["state"].keys() == b["state"].keys() and set(a) == set(b)  # no key more: the margin is the state's
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"] if k.startswith("cs_") and k != "cs_margin")
    lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(FOOT_M))
    wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / "pair.pt")), torch.device("cpu"))  # nothing but the saved fit
    assert same(wm.state_dict(), b["state"]) and torch.equal(wm.cs_margin, b["state"]["cs_margin"].cpu())  # saved on the fit's device
    with pytest.raises(ValueError, match="--contact_pair_margin is an option of --contacts"):
        fit("x", "--contact_pair_margin")
    with pytest.raises(ValueError, match="--contact_pair_margin with --init"):
        fit("y", *floor, "--contact_pair_margin", "--init", str(tmp_path / "pair.pt"))
    again = torch.load(fit("again", *floor, "--init", str(tmp_path / "pair.pt")), weights_only=False)  # a refit keeps the start's margin
    assert torch.equal(again["state"]["cs_margin"], b["state"]["cs_margin"])
