"""The unused places of the contact inputs' spheres (ndm_gym.worldmodels.mjcf: ``contact_spheres`` fills a geom's places beyond
its own spheres with a point 10 m off in its body's frame) stay out of every gap, on CPU: two capsules of different lengths on
tilted bodies above a floor plane and above a wide box, with states that carry the unused point below the surface, each
against MuJoCo's own distance (``mj_geomDistance``) and one by hand; the gaps and inputs against a set whose unused places
hold copies of a real sphere; a saved fit's spheres, which keep the mark and no count, and a state loaded over other spheres;
and the sets that are refused. And the gaps' revision a fit with static geoms is saved with (fit_mjcf): such a fit carries it
and loads, one without it is refused by ``load_ndm`` and by ``--init``, and a fit without static geoms loads marked or not."""

import json
import os
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_fit_mjcf_options import FIT, XML, simulated  # noqa: E402

from ndm_gym.wmbench.fit_mjcf import main  # noqa: E402
from ndm_gym.wmbench.speed_mjcf import load_ndm  # noqa: E402
from ndm_gym.worldmodels.mjcf import GAPS_REVISION, PLANE, UNUSED_PLACE, MjcfLayout, MjcfWorldModel, contact_spheres, signed_distance  # noqa: E402

F64 = dict(dtype=torch.float64)
# a trunk on two slides and a pitch hinge with a long capsule (radius 50 mm, half length 0.3 m, along its z) and, 0.4 m below
# its origin, a limb on a roll hinge with a short one (radius 40 mm, half length 0.1 m, its centre 0.15 m down the limb's z)
PAIR = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    GROUND
    <body name="trunk" pos="0 0 1">
      <joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><joint name="pitch" type="hinge" axis="0 1 0"/>
      <geom name="trunk" type="capsule" size="0.05 0.3"/>
      <body name="limb" pos="0 0 -0.4">
        <joint name="roll" type="hinge" axis="1 0 0"/>
        <geom name="limb" type="capsule" size="0.04 0.1" pos="0 0 -0.15"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="pitch"/><motor joint="roll"/></actuator>
</mujoco>
"""
# the ground: an unbounded plane, or a box with its top at z = 0, wide and deep enough to hold the unused point
GROUNDS = {"plane": '<geom name="floor" type="plane" size="5 5 0.1"/>', "box": '<geom name="floor" type="box" size="30 30 10" pos="0 0 -10"/>'}
# x, z, pitch, roll: upright; then pitched, rolled, and both, far enough to carry the limb's unused point below z = 0
STATES = ((0.0, 0.0, 0.0, 0.0), (0.2, -0.1, 1.2, 0.0), (0.0, 0.0, 0.0, 2.5), (-0.3, 0.3, -0.8, -2.0), (0.1, -0.35, 0.4, 0.9), (0.0, -0.2, 2.0, 1.0))
BELOW = (1, 2, 3, 5)  # the states that do


def pair(ground: str):
    m = mujoco.MjModel.from_xml_string(PAIR.replace("GROUND", GROUNDS[ground]))
    lay = MjcfLayout.from_model(m)
    cs = contact_spheres(m, lay, 0.02, static=("floor",))
    return m, lay, cs


def gaps_over_every_place(wm, q):
    """The gap to the first static geom as the smallest over every sphere place, the unused ones too (as it was read before)."""
    P, R = wm.body_poses(q)
    c = P[..., wm.cs_body, None, :] + torch.einsum("...gij,gkj->...gki", R[..., wm.cs_body, :, :], wm.cs_center)
    return (signed_distance(wm._sg[0][0], (c - wm.sg_pos[0]) @ wm.sg_rot[0], wm.sg_size[0]) - wm.cs_radius).min(-1).values


@pytest.mark.parametrize("ground", ["plane", "box"])
def test_two_capsules_of_different_lengths_on_tilted_bodies_against_mujocos_distance(ground):
    if not hasattr(mujoco, "mj_geomDistance"):
        pytest.skip(f"MuJoCo {mujoco.__version__} has no mj_geomDistance to compare with")
    m, lay, cs = pair(ground)
    assert cs["count"].tolist() == [61, 21] and cs["center"].shape == (2, 61, 3)  # the limb's last 40 places are unused
    assert (cs["center"][1, 21:] == UNUSED_PLACE).all() and (cs["radius"][1, 21:] == 0).all()
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs).double()
    assert wm.cs_unused.sum(-1).tolist() == [0, 40] and "cs_unused" not in wm.state_dict()  # derived from the spheres, not kept
    floor, d = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), mujoco.MjData(m)
    assert wm._sg[0][0] == (PLANE if ground == "plane" else 6)
    for i, state in enumerate(STATES):
        d.qpos[:] = state
        mujoco.mj_forward(m, d)
        q = torch.tensor([state], **F64)
        gap = wm.contact_gaps(q)[0][0, 0]
        want = [mujoco.mj_geomDistance(m, d, int(g), floor, 50.0, None) for g in cs["geom"]]
        assert gap.tolist() == pytest.approx(want, abs=1e-6)
        P, R = wm.body_poses(q)
        point = P[0, 2] + R[0, 2] @ torch.full((3,), UNUSED_PLACE, **F64)  # the limb's unused place in the world
        every = gaps_over_every_place(wm, q)[0]
        assert every[0].item() == gap[0].item()  # the trunk has no unused place
        if i in BELOW:  # under the surface: the smallest over every place is the point's own distance, metres off
            depth = point[2] if ground == "plane" else (point[2] + 10.0).abs() - 10.0  # under the plane; inside the box, to its top or bottom
            assert point[2] < -1.0 and every[1] < gap[1] - 1.0 and every[1].item() == pytest.approx(depth.item(), abs=1e-6)
        else:
            assert point[2] > 1.0 and every[1].item() == gap[1].item()


def test_a_rolled_limb_above_the_plane_by_hand():
    m, lay, cs = pair("plane")
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs).double()
    roll, vz, w = 2.5, -0.3, 1.5  # the limb turned over about x, the trunk falling, the limb rolling on
    q, v = torch.tensor([[0.0, 0.0, 0.0, roll]], **F64), torch.tensor([[0.0, vz, 0.0, w]], **F64)
    gap, p, n, vel = wm.contact_gaps(q, v)
    c, s = np.cos(roll), np.sin(roll)
    Rb, Pb = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), np.array([0.0, 0.0, 0.6])  # the limb: turned about x, its origin 0.4 m under the trunk's
    low = Pb + Rb @ [0.0, 0.0, -0.05]  # the centre of the capsule's lower end: the one nearer the limb's origin, now that it is turned over
    arm = np.array([low[0], low[1], 0.0]) - Pb  # the floor's point under it, from the limb's origin
    assert low[2] < (Pb + Rb @ [0.0, 0.0, -0.25])[2]
    assert gap[0, 0, 1].item() == pytest.approx(low[2] - 0.04, abs=1e-6)  # the lower end's height less the radius
    assert np.allclose(p[0, 0, 1].numpy(), Rb.T @ arm, atol=1e-6)  # in the limb's frame
    assert np.allclose(n[0, 0, 1].numpy(), Rb.T @ [0, 0, 1], atol=1e-6)
    assert np.allclose(vel[0, 0, 1].numpy(), Rb.T @ (np.array([0.0, 0.0, vz]) + np.cross([w, 0.0, 0.0], arm)), atol=1e-6)
    assert gap[0, 0, 0].item() == pytest.approx(1.0 - 0.3 - 0.05, abs=1e-6)  # the trunk upright: its lower end's height less its radius
    x = wm.contact_inputs(q, v)
    assert x.shape == (1, 26) and x[0, 0].item() == pytest.approx(4.0) and x[0, 13].item() == pytest.approx(4.0)  # both beyond 20 mm
    assert x[0, [1, 2, 14, 15]].abs().max() < 1e-12 and x.abs().max() <= 4.0  # no touch, no overlap; nothing of the point 10 m off


@pytest.mark.parametrize("ground", ["plane", "box"])
@pytest.mark.parametrize("double", [False, True])
def test_gaps_and_inputs_are_those_of_a_set_without_unused_places(ground, double):
    m, lay, cs = pair(ground)
    filled = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cs.items() if k != "count"}
    filled["center"][1, 21:], filled["radius"][1, 21:] = cs["center"][1, 0], cs["radius"][1, 0]  # copies of the limb's first sphere
    wm, ref = (MjcfWorldModel(lay, hidden=(8,), contacts=c) for c in (cs, filled))
    wm, ref = (wm.double(), ref.double()) if double else (wm, ref)
    assert wm._cs_pad and not ref._cs_pad and not ref.cs_unused.any()  # the reference takes the plain minimum
    g = torch.Generator().manual_seed(0)
    q = torch.cat((torch.rand(256, 2, generator=g) - 0.5, 2 * np.pi * (torch.rand(256, 2, generator=g) - 0.5)), -1)  # down to and into the ground
    q[:, 1] -= 0.4 * torch.rand(256, generator=g)
    v = torch.randn(256, m.nv, generator=g)
    q, v = (q.double(), v.double()) if double else (q, v)
    for got, want in zip(wm.contact_gaps(q, v), ref.contact_gaps(q, v)):
        assert torch.equal(got, want)
    assert torch.equal(wm.contact_inputs(q, v), ref.contact_inputs(q, v))
    gap = wm.contact_gaps(q)[0]
    assert (gap < 0).any() and (gap > 0.02).any() and gap.min() > -1.0  # states in contact and states apart, none metres deep
    assert (gaps_over_every_place(wm, q)[:, 1] < -1.0).any()  # where the smallest over every place is


def test_a_saved_fits_spheres_keep_the_mark_and_load_as_before(tmp_path):
    m, lay, cs = pair("plane")
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs)
    state = wm.state_dict()
    assert {k for k in state if k.startswith("cs_")} == {"cs_center", "cs_radius", "cs_body", "cs_margin", "cs_basis", "cs_sg"}  # as before
    path = str(tmp_path / "fit.pt")
    saved = {"state": state, "layout": json.dumps(lay.__dict__), "args": {"hidden": [8], "act": "silu", "no_ln": False, "contacts": True}}
    torch.save(saved | {"contact_gaps": GAPS_REVISION}, path)  # as fit_mjcf saves a fit with static geoms
    back = load_ndm(lay, types.SimpleNamespace(ckpt=path), torch.device("cpu"))  # the spheres from the state alone: no count
    assert torch.equal(back.cs_unused, wm.cs_unused) and back.cs_unused.sum(-1).tolist() == [0, 40] and back._cs_pad
    q, v = torch.tensor([STATES[2], STATES[5]]), torch.zeros(2, m.nv)
    for got, want in zip(back.contact_gaps(q, v), wm.contact_gaps(q, v)):
        assert torch.equal(got, want)
    assert back.contact_gaps(q)[0][:, 0, 1].tolist() == pytest.approx([0.600057, 0.937701], abs=1e-5)  # the limb's, not the point's depth
    other = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cs.items() if k != "count"}
    other["center"][1, 21:], other["radius"][1, 21:] = cs["center"][1, 0], cs["radius"][1, 0]  # a set of the same shape with no unused place
    full = MjcfWorldModel(lay, hidden=(8,), contacts=other)
    assert not full._cs_pad and not full.cs_unused.any()
    full.load_state_dict(state)  # the loaded spheres stand, and so do their unused places
    assert full._cs_pad and torch.equal(full.cs_unused, wm.cs_unused) and torch.equal(full.contact_gaps(q)[0], wm.contact_gaps(q)[0])


def test_sets_whose_marks_do_not_follow_the_counts_are_refused():
    m, lay, cs = pair("plane")
    marked = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cs.items()}
    marked["center"][0, 3], marked["radius"][0, 3] = UNUSED_PLACE, 0.0  # one of the trunk's own spheres bears the mark
    with pytest.raises(ValueError, match="not the places after each geom's count"):
        MjcfWorldModel(lay, hidden=(8,), contacts=marked)
    short = dict(cs, count=np.array([61, 20]))  # a count that leaves out a real sphere
    with pytest.raises(ValueError, match="not the places after each geom's count"):
        MjcfWorldModel(lay, hidden=(8,), contacts=short)
    empty = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cs.items() if k != "count"}  # as a saved fit's: no count
    empty["center"][1], empty["radius"][1] = UNUSED_PLACE, 0.0
    with pytest.raises(ValueError, match="a geom has no sphere"):
        MjcfWorldModel(lay, hidden=(8,), contacts=empty)


def recorded_pair(path, n=6, steps=24, seed=0):
    """A recording of the pair dropped onto the plane from turned starts under random torques, without resets."""
    m, rng = mujoco.MjModel.from_xml_string(PAIR.replace("GROUND", GROUNDS["plane"])), np.random.default_rng(seed)
    d = mujoco.MjData(m)
    qpos, qvel, ctrl = np.zeros((steps + 1, n, m.nq)), np.zeros((steps + 1, n, m.nv)), np.zeros((steps, n, m.nu))
    for i in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [0.0, 0.1 * rng.random(), rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)]  # 0.2 m or more above the plane
        for t in range(steps):
            qpos[t, i], qvel[t, i] = d.qpos, d.qvel
            d.ctrl[:] = ctrl[t, i] = rng.uniform(-1.0, 1.0, m.nu)
            mujoco.mj_step(m, d)
        qpos[steps, i], qvel[steps, i] = d.qpos, d.qvel
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    np.savez(path, qpos=f32(qpos), qvel=f32(qvel), ctrl=f32(ctrl), reset=np.zeros((steps, n), dtype=bool), dt=m.opt.timestep)
    return str(path)


def fit_pair(tmp_path, name, *extra):
    """fit_mjcf on the pair above the plane with the floor's contact inputs; the saved file's path."""
    (tmp_path / "pair.xml").write_text(PAIR.replace("GROUND", GROUNDS["plane"]))
    train, test = recorded_pair(tmp_path / "pair_train.npz"), recorded_pair(tmp_path / "pair_test.npz", n=3, seed=1)
    main(["--model", str(tmp_path / "pair.xml"), "--train", train, "--test", f"held={test}", *FIT, "--contacts", "--contact_static", "floor",
          "--contact_spacing", "0.02", *extra, "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
    return str(tmp_path / f"{name}.pt")


def test_a_fit_with_static_geoms_carries_the_gaps_revision_and_loads(tmp_path):
    path = fit_pair(tmp_path, "floor")
    ck = torch.load(path, weights_only=False)
    assert ck["contact_gaps"] == GAPS_REVISION and set(ck) == {"state", "layout", "args", "contact_gaps"}  # beside the layout and the arguments
    assert "cs_sg" in ck["state"] and not any("gaps" in k for k in ck["state"])  # not among the state's keys
    lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(PAIR.replace("GROUND", GROUNDS["plane"])))
    wm = load_ndm(lay, types.SimpleNamespace(ckpt=path), torch.device("cpu"))
    assert wm.cs_unused.sum(-1).tolist() == [0, 40] and all(torch.equal(v, ck["state"][k].cpu()) for k, v in wm.state_dict().items())
    again = torch.load(fit_pair(tmp_path, "again", "--init", path), weights_only=False)  # a start for a refit, which carries it too
    assert again["contact_gaps"] == GAPS_REVISION


def test_a_fit_with_static_geoms_and_no_mark_of_the_gaps_is_refused(tmp_path):
    ck = torch.load(fit_pair(tmp_path, "floor"), weights_only=False)
    lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(PAIR.replace("GROUND", GROUNDS["plane"])))
    cases = (("unmarked", {}, "fitted on the gaps before 9418a03d"),  # as every such fit saved before the mark
             ("other", {"contact_gaps": GAPS_REVISION + 1}, f"fitted on revision {GAPS_REVISION + 1} of the contact gaps"))  # fmt: skip
    for name, mark, match in cases:
        path = str(tmp_path / f"{name}.pt")
        torch.save({k: v for k, v in ck.items() if k != "contact_gaps"} | mark, path)
        with pytest.raises(ValueError, match=match):
            load_ndm(lay, types.SimpleNamespace(ckpt=path), torch.device("cpu"))
        with pytest.raises(ValueError, match=match):
            fit_pair(tmp_path, f"from_{name}", "--init", path)
        assert not (tmp_path / f"from_{name}.pt").exists()


def test_a_fit_without_static_geoms_loads_marked_or_not(tmp_path):
    (tmp_path / "probes.xml").write_text(XML)
    train, test = simulated(tmp_path / "train.npz"), simulated(tmp_path / "test.npz", n=3, seed=1)

    def fit(name, *extra):
        main(["--model", str(tmp_path / "probes.xml"), "--train", train, "--test", f"held={test}", *FIT, "--contacts", *extra,
              "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
        return str(tmp_path / f"{name}.pt")

    ck = torch.load(fit("free"), weights_only=False)
    assert set(ck) == {"state", "layout", "args"} and "cs_sg" not in ck["state"]  # the free object alone: saved as before, without a mark
    lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(XML))
    for name, mark in (("bare", {}), ("marked", {"contact_gaps": GAPS_REVISION}), ("other", {"contact_gaps": GAPS_REVISION - 1})):
        path = str(tmp_path / f"{name}.pt")
        torch.save(ck | mark, path)
        wm = load_ndm(lay, types.SimpleNamespace(ckpt=path), torch.device("cpu"))
        assert all(torch.equal(v, ck["state"][k].cpu()) for k, v in wm.state_dict().items())
        assert set(torch.load(fit(f"from_{name}", "--init", path), weights_only=False)) == {"state", "layout", "args"}  # and is a start
