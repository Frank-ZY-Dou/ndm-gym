"""The contact inputs of the MJCF world model (ndm_gym.worldmodels.mjcf: ``contact_spheres``, ``MjcfWorldModel(contacts=...)``)
on CPU, against values worked out by hand: the spheres that stand for a sphere, a capsule and a box; their gaps, nearest points,
normals and relative velocities to an object's box, sphere, capsule and cylinder at known poses; the 13 inputs per geom and
their gates; the summed terms; the RigidFormer's contact columns; the models that are refused; scipy's absence; and the two
options beyond the free objects: static geoms of the world (a capsule foot above a floor plane and a box at a known height and
tilt) and an object that rides on slides, with the free objects' path compared bit for bit with the code before them."""

import os
import sys

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from ndm_gym.worldmodels.mjcf import CONTACT_FEATURES, MjcfLayout, MjcfWorldModel, _hull_spheres, contact_spheres  # noqa: E402

# three probes on vertical slides, 0.3 m up: a ball (radius 20 mm), a rod (a capsule of radius 15 mm and half length 50 mm, along x)
# and a brick (a box of half sizes 30, 20, 10 mm); and one free object
PROBES = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="ball" pos="0 0 0.3"><joint name="zs" type="slide" axis="0 0 1"/><geom name="ball" type="sphere" size="0.02"/></body>
    <body name="rod" pos="0.5 0 0.3"><joint name="zc" type="slide" axis="0 0 1"/>
      <geom name="rod" type="capsule" size="0.015 0.05" quat="0.7071067811865476 0 0.7071067811865476 0"/></body>
    <body name="brick" pos="1 0 0.3"><joint name="zb" type="slide" axis="0 0 1"/><geom name="brick" type="box" size="0.03 0.02 0.01"/></body>
    <body name="obj" pos="0 0 0.05"><freejoint name="obj"/><geom name="obj" type="KIND" size="SIZE" MARGIN/></body>
  </worldbody>
  <actuator><position joint="zs" kp="10"/><position joint="zc" kp="10"/><position joint="zb" kp="10"/></actuator>
</mujoco>
"""
BOX = ("box", "0.1 0.1 0.05")
RF = {"dim": 32, "layers": 1, "heads": 2}  # a small RigidFormer


def probes(kind="box", size="0.1 0.1 0.05", margin=""):
    m = mujoco.MjModel.from_xml_string(PROBES.replace("KIND", kind).replace("SIZE", size).replace("MARGIN", margin))
    return m, MjcfLayout.from_model(m)


def state(m, at=(0.0, 0.0, 0.05), quat=(1.0, 0.0, 0.0, 0.0), lift=(0.0, 0.0, 0.0), vel=None):
    """One state: the object at ``at`` with ``quat``, the three probes lifted by ``lift``; ``vel``: {degree of freedom: speed}."""
    q, v = torch.zeros(1, m.nq, dtype=torch.float64), torch.zeros(1, m.nv, dtype=torch.float64)
    q[0, :3] = torch.tensor(lift)
    q[0, 3:6], q[0, 6:10] = torch.tensor(at), torch.tensor(quat)
    for d, x in (vel or {}).items():
        v[0, d] = x
    return q, v


def test_hull_spheres_stay_inside_touch_the_faces_and_cover_a_box():
    size = np.array([0.03, 0.02, 0.01])
    pts = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.float64) * size
    c, r = _hull_spheres(pts, 0.0025, 0.02, 384, np.random.default_rng(0))
    assert c.shape == (len(r), 3) and (r >= 0).all() and r.max() <= 0.01 + 3e-5  # no sphere wider than the box's thinnest half
    out = np.abs(c) + r[:, None] - size  # how far each sphere passes each pair of faces
    assert out.max() <= 2e-5 + 1e-9  # inside, to the tolerance
    face, edge = r > 0, r == 0
    assert face.sum() > 200 and edge.sum() > 8 and len(r) <= 384 + 10  # the cap holds (faces and edges): at 2.5 mm, 1,408 on the faces
    assert np.abs(out[face].max(1)).max() <= 2e-5 + 1e-9  # each sphere touches a face
    assert ((np.abs(np.abs(c[edge]) - size) < 1e-12).sum(1) >= 2).all()  # the points of radius 0 lie on the box's edges
    rng = np.random.default_rng(1)  # the spheres come within their spacing of every point of the surface
    x = rng.uniform(-1, 1, (4000, 3)) * size
    k = rng.integers(0, 3, 4000)
    x[np.arange(4000), k] = np.sign(x[np.arange(4000), k]) * size[k]
    reach = (np.linalg.norm(x[:, None] - c[None], axis=-1) - r[None]).min(1)
    assert reach.max() < np.sqrt(0.0088 / face.sum())
    few, _ = _hull_spheres(pts, 0.0025, 0.02, 48, np.random.default_rng(0))
    assert len(few) < len(c) / 3  # a lower cap: a wider spacing
    again, r2 = _hull_spheres(pts, 0.0025, 0.02, 384, np.random.default_rng(0))
    assert np.array_equal(again, c) and np.array_equal(r2, r)  # the same spheres from the same seed


def test_contact_spheres_of_a_sphere_a_capsule_and_a_box():
    m, lay = probes(*BOX, margin='margin="0.004"')
    cs = contact_spheres(m, lay)
    ids = [m.geom(n).id for n in ("ball", "rod", "brick")]
    assert list(cs["geom"]) == ids and list(cs["body"]) == [m.body(n).id for n in ("ball", "rod", "brick")]
    K = int(cs["count"].max())
    assert cs["center"].shape == (3, K, 3) and cs["radius"].shape == (3, K) and cs["margin"] == pytest.approx(0.004)
    assert cs["count"][0] == 1 and np.allclose(cs["center"][0, 0], 0) and cs["radius"][0, 0] == pytest.approx(0.02)  # the ball as itself
    assert np.allclose(cs["center"][0, 1:], 10.0) and (cs["radius"][0, 1:] == 0).all()  # unused places: points 10 m off
    n = cs["count"][1]  # the rod: spheres of its radius along its axis (the body's x), half a spacing apart
    assert n == 81 and np.allclose(cs["radius"][1, :n], 0.015)
    assert np.allclose(cs["center"][1, :n, 0], np.linspace(-0.05, 0.05, 81), atol=1e-7) and np.allclose(cs["center"][1, :n, 1:], 0, atol=1e-7)
    n = cs["count"][2]  # the brick: spheres inside it
    assert (np.abs(cs["center"][2, :n]) + cs["radius"][2, :n, None] <= np.array([0.03, 0.02, 0.01]) + 3e-5).all()
    capped = contact_spheres(m, lay, cap=32)
    assert capped["count"][2] < cs["count"][2] / 3 and capped["count"][1] == 81 and capped["center"].shape[1] == 81
    assert contact_spheres(*probes(*BOX))["margin"] == 0.0


# the object, and its top above its centre
OBJECTS = [("box", "0.1 0.1 0.05", 0.05), ("sphere", "0.05", 0.05), ("capsule", "0.03 0.1", 0.13), ("cylinder", "0.05 0.04", 0.04)]


@pytest.mark.parametrize("kind, size, top", OBJECTS)
def test_gaps_by_hand(kind, size, top):
    m, lay = probes(kind, size)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay)).double()
    for i, (x, radius) in enumerate(((0.0, 0.02), (0.5, 0.015), (1.0, 0.01))):  # the object under the ball, the rod, the brick
        if kind not in ("box", "cylinder") and i > 0:  # the rod and the brick over a round top: their nearest sphere is not the one above it
            continue
        if kind == "cylinder" and i == 2:  # the brick is wider than the cylinder's cap is flat under its spheres' spacing
            continue
        for lift in (0.0, -0.1, -0.2 + top + radius - 0.001 - 0.05):  # the last: 1 mm above the object
            q, _ = state(m, at=(x, 0.0, 0.05), lift=(lift,) * 3)
            gap, p, n = wm.contact_gaps(q)
            assert gap.shape == (1, 1, 3) and p.shape == n.shape == (1, 1, 3, 3)
            want = 0.3 + lift - 0.05 - top - radius  # the probe's height above the object's centre, less the top and its own radius
            assert gap[0, 0, i].item() == pytest.approx(want, abs=3e-5 if i == 2 else 1e-6)  # the model's constants are float32
            assert torch.allclose(n[0, 0, i], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6)  # the top's normal
            assert p[0, 0, i, 2].item() == pytest.approx(top, abs=1e-6)  # the nearest point: on the top


def test_gaps_beside_an_edge_under_a_turn_and_inside():
    m, lay = probes(*BOX)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay)).double()
    q, _ = state(m)  # the object under the ball: the rod's nearest end is 0.35 m beside the box's edge and 0.2 m above it
    gap, p, n = wm.contact_gaps(q)
    assert gap[0, 0, 1].item() == pytest.approx(np.hypot(0.35, 0.2) - 0.015, abs=1e-6)
    assert torch.allclose(p[0, 0, 1], torch.tensor([0.1, 0.0, 0.05], dtype=torch.float64), atol=1e-6)  # the edge's point
    assert torch.allclose(n[0, 0, 1], torch.tensor([0.35, 0.0, 0.2], dtype=torch.float64) / np.hypot(0.35, 0.2), atol=1e-6)
    turned = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # a quarter turn about x: the box's y faces are its top and bottom
    q, _ = state(m, at=(0.0, 0.0, 0.1), quat=turned)
    gap, p, n = wm.contact_gaps(q)
    assert gap[0, 0, 0].item() == pytest.approx(0.3 - 0.1 - 0.1 - 0.02, abs=1e-6)
    assert torch.allclose(p[0, 0, 0], torch.tensor([0.0, 0.1, 0.0], dtype=torch.float64), atol=1e-6)  # in the object's frame
    assert torch.allclose(n[0, 0, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-6)
    q, _ = state(m, lift=(-0.2 + 0.015, 0.0, 0.0))  # the ball 5 mm into the top
    assert wm.contact_gaps(q)[0][0, 0, 0].item() == pytest.approx(-0.005, abs=1e-6)


def test_relative_velocity_at_the_nearest_point():
    m, lay = probes(*BOX)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay)).double()
    q, v = state(m, vel={0: -0.5})  # the ball comes down at 0.5 m/s
    assert torch.allclose(wm.contact_gaps(q, v)[3][0, 0, 0], torch.tensor([0.0, 0.0, -0.5], dtype=torch.float64), atol=1e-6)
    q, v = state(m, vel={0: -0.5, 3 + 2: 0.1, 3 + 0: 0.2})  # and the object rises at 0.1 m/s and slides at 0.2 m/s along x
    assert torch.allclose(wm.contact_gaps(q, v)[3][0, 0, 0], torch.tensor([-0.2, 0.0, -0.6], dtype=torch.float64), atol=1e-6)
    q, v = state(m, vel={3 + 4: 2.0})  # the object turns about its y at 2 rad/s: its top's point under the ball moves along x
    assert torch.allclose(wm.contact_gaps(q, v)[3][0, 0, 0], torch.tensor([-2.0 * 0.05, 0.0, 0.0], dtype=torch.float64), atol=1e-6)


def test_inputs_values_gates_and_summed_terms():
    m, lay = probes(*BOX)
    cs = contact_spheres(m, lay)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs).double()
    assert wm.n_contact == CONTACT_FEATURES * 3 and wm.n_in == MjcfWorldModel(lay, hidden=(8,)).n_in + wm.n_contact
    q, v = state(m, lift=(-0.2 + 0.02 + 0.001, 0.0, 0.0), vel={0: -0.4})  # the ball 1 mm above the top, coming down at 0.4 m/s
    x = wm.contact_inputs(q, v)
    assert x.shape == (1, 39) and not x.requires_grad
    near, touch = np.exp(-0.5), np.exp(-2.0)  # exp(-1 mm / 2 mm), exp(-1 mm / 0.5 mm)
    want = [0.2, touch, 0.0, 0.0, 0.0, near * 0.05 / 0.03, 0.0, 0.0, near, 0.0, 0.0, near * -2.0, near * -2.0]
    assert torch.allclose(x[0, :13], torch.tensor(want, dtype=torch.float64), atol=2e-4)
    far = x[0, 13:].reshape(2, 13)  # the rod and the brick, 0.2 m and more off: the gap's value at its limit, every gated value gone
    assert torch.allclose(far[:, 0], torch.tensor(4.0, dtype=torch.float64)) and far[:, 1:].abs().max() < 1e-30
    q, v = state(m, lift=(-0.2 + 0.02 - 0.0005, 0.0, 0.0))  # the ball 0.5 mm into the top
    x = wm.contact_inputs(q, v)
    assert torch.allclose(x[0, :3], torch.tensor([-0.1, 1.0, 5.0], dtype=torch.float64), atol=2e-4) and x[0, 8].item() == pytest.approx(1.0)
    inputs = wm.inputs(q, v, torch.zeros(1, 3, dtype=torch.float64))
    plain = MjcfWorldModel(lay, hidden=(8,)).double().inputs(q, v, torch.zeros(1, 3, dtype=torch.float64))
    assert torch.equal(inputs[:, :-39], plain) and torch.equal(inputs[:, -39:], x)  # the other inputs as without them, bit for bit
    wm.update_norms(q, v, torch.zeros(1, 3, dtype=torch.float64), q, v)  # the contact inputs pass the normalizer as they are
    assert (wm.in_norm.mean[-39:] == 0).all() and (wm.in_norm.var[-39:] == 1).all()
    both = MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"basis": True}).double()
    assert both.n_contact == 39 + 12 and bool(both.cs_basis) and not bool(wm.cs_basis)
    q, v = state(m, lift=(-0.2 + 0.02 + 0.001, 0.0, 0.0), vel={0: -0.4})
    y = both.contact_inputs(q, v)
    assert y.shape == (1, 51) and torch.equal(y[:, :39], wm.contact_inputs(q, v))
    # the drag (the relative velocity, 2 in units of 0.2 m/s, downwards), its moment (none: the point is above the centre), the push
    # (the approach along the normal) and its moment, each times the touch gate
    want = touch * torch.tensor([0, 0, -2.0, 0, 0, 0, 0, 0, 2.0, 0, 0, 0], dtype=torch.float64)
    assert torch.allclose(y[0, 39:], want, atol=2e-4)


def test_the_rigidformers_contact_columns():
    m, lay = probes(*BOX)
    cs = contact_spheres(m, lay)
    plain = MjcfWorldModel(lay, backbone="rigidformer", rf=RF).double()
    wm = MjcfWorldModel(lay, backbone="rigidformer", rf=RF, contacts=cs | {"basis": True}).double()
    assert wm.tok_dim == plain.tok_dim + 13 + 12
    q, v = state(m, lift=(-0.2 + 0.02 + 0.001, 0.0, 0.0), vel={0: -0.4})
    u = torch.zeros(1, 3, dtype=torch.float64)
    tok = wm.token_inputs(q, v, u).reshape(1, 4, wm.tok_dim)  # the ball's, the rod's, the brick's and the object's tokens
    assert torch.equal(tok[..., : plain.tok_dim], plain.token_inputs(q, v, u).reshape(1, 4, plain.tok_dim))  # the rest as without them
    x, basis = wm.contact_features(q, v)
    assert torch.equal(tok[0, :3, plain.tok_dim : plain.tok_dim + 13], x[0, 0])  # each probe's token: its geom's 13 inputs
    far = torch.zeros(13, dtype=torch.float64)
    far[0] = 4.0
    assert torch.equal(tok[0, 3, plain.tok_dim : plain.tok_dim + 13], far)  # the object's own token: no geom, read as one far off
    assert torch.equal(tok[0, 3, plain.tok_dim + 13 :], basis[0, 0]) and (tok[0, :3, plain.tok_dim + 13 :] == 0).all()  # the summed terms
    assert sorted(wm.tok_contact_cols.tolist()) == [t * wm.tok_dim + c for t in range(4) for c in range(plain.tok_dim, wm.tok_dim)]


def test_refused():
    m, lay = probes(*BOX)
    cs = contact_spheres(m, lay)
    with pytest.raises(ValueError, match="each a sphere, a capsule, a cylinder or a box"):
        MjcfWorldModel(probes("ellipsoid", "0.05 0.04 0.03")[1], hidden=(8,), contacts=cs)
    obj = '<body name="obj" pos="0 0 0.05"><freejoint name="obj"/><geom name="obj" type="KIND" size="SIZE" MARGIN/></body>'
    alone = mujoco.MjModel.from_xml_string(PROBES.replace(obj, ""))
    with pytest.raises(ValueError, match="has no object"):
        MjcfWorldModel(MjcfLayout.from_model(alone), hidden=(8,), contacts=cs)
    with pytest.raises(ValueError, match="the robot has 3 collision geoms"):
        MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"radius": cs["radius"][:, :-1]})


def test_without_scipy_the_hull_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)  # as if it were not installed
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    with pytest.raises(ImportError, match=r"ndm_gym\[contacts\]"):
        contact_spheres(*probes(*BOX))


# ---- beyond the free objects: static geoms of the world, and an object that rides on slides ----

# a capsule foot (radius 40 mm, half length 100 mm, along its body's x) that slides along x and z and tilts about y, 0.5 m above a
# floor plane; a box (a step) whose top is 0.2 m up, 2 m along x
FOOT = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="step" type="box" pos="2 0 0.1" size="0.3 0.3 0.1"/>
    <body name="foot" pos="0 0 0.5">
      <joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><joint name="tilt" type="hinge" axis="0 1 0"/>
      <geom name="foot" type="capsule" size="0.04 0.1" quat="0.7071067811865476 0 0.7071067811865476 0"/>
    </body>
  </worldbody>
  <actuator><motor joint="tilt"/></actuator>
</mujoco>
"""
# a tip (a ball of radius 20 mm) on two slides and a puck (a cylinder of radius and half height 50 mm) on two slides of its own, on a table
PUSHER = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.1"/>
    <body name="arm" pos="0 0 0.05"><joint name="ax" type="slide" axis="1 0 0"/><joint name="ay" type="slide" axis="0 1 0"/>
      <geom name="tip" type="sphere" size="0.02"/></body>
    <body name="object" pos="0.3 0 0.05"><joint name="oy" type="slide" axis="0 1 0"/><joint name="ox" type="slide" axis="1 0 0"/>
      <geom name="puck" type="cylinder" size="0.05 0.05"/></body>
  </worldbody>
  <actuator><motor joint="ax"/><motor joint="ay"/></actuator>
</mujoco>
"""
F64 = dict(dtype=torch.float64)


def test_distance_and_normal_of_a_cylinder_and_a_plane():
    from ndm_gym.worldmodels.mjcf import CYLINDER, PLANE, distance_normal, signed_distance

    size = torch.tensor([0.05, 0.04, 0.0], **F64)  # the radius and the half height
    x = torch.tensor([[0.2, 0.0, 0.01], [0.0, 0.03, 0.1], [0.08, 0.0, 0.08], [0.04, 0.0, 0.0], [0.0, 0.01, -0.035]], **F64)
    d, n = distance_normal(CYLINDER, x, size)
    # beside the side; above the top cap; off the rim (30 mm out, 40 mm up); inside, nearer the side; inside, nearer the bottom cap
    assert torch.allclose(d, torch.tensor([0.15, 0.06, 0.05, -0.01, -0.005], **F64), atol=1e-9)
    assert torch.allclose(n, torch.tensor([[1, 0, 0], [0, 0, 1], [0.6, 0, 0.8], [1, 0, 0], [0, 0, -1]], **F64), atol=1e-6)
    assert torch.allclose(signed_distance(CYLINDER, x, size), d, atol=1e-12)
    d, n = distance_normal(PLANE, x, size)
    assert torch.equal(d, x[:, 2]) and torch.equal(signed_distance(PLANE, x, size), d)
    assert torch.equal(n, torch.tensor([[0.0, 0, 1]], **F64).expand(5, 3))


def test_a_capsule_foot_above_a_floor_and_a_step_by_hand():
    m = mujoco.MjModel.from_xml_string(FOOT)
    lay = MjcfLayout.from_model(m)
    with pytest.raises(ValueError, match="has no object"):  # as before: no free object, and nothing else named
        MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay))
    cs = contact_spheres(m, lay, static=())  # every collision geom of the world body: the floor (a plane, type 0), the step (a box, type 6)
    assert "tb" not in cs and cs["sg"].shape == (2, 11) and cs["sg"][:, 0].tolist() == [0.0, 6.0]
    assert contact_spheres(m, lay, static=("step",))["sg"][:, 0].tolist() == [6.0]
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs).double()
    assert wm.n_targets == 2 and wm.n_contact == 2 * 13 and "cs_sg" in wm.state_dict() and "cs_tb" not in wm.state_dict()
    for x, dz, tilt, vz, w in ((0.0, -0.3, 0.3, -0.5, 2.0), (0.3, 0.0, -0.4, 0.1, -1.0)):  # the foot over the floor, tilted either way
        q, v = torch.tensor([[x, dz, tilt]], **F64), torch.tensor([[0.0, vz, w]], **F64)
        gap, p, n, vel = wm.contact_gaps(q, v)
        assert gap.shape == (1, 2, 1) and p.shape == n.shape == vel.shape == (1, 2, 1, 3)
        c, s = np.cos(tilt), np.sin(tilt)
        Rb, Pb = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), np.array([x, 0.0, 0.5 + dz])  # the body: turned about y, and its origin
        low = Pb + np.sign(s) * 0.1 * Rb[:, 0]  # the centre of the capsule's lower end (its axis is the body's x)
        arm = np.array([low[0], 0.0, 0.0]) - Pb  # the floor's point under it, from the body's origin
        assert gap[0, 0, 0].item() == pytest.approx(low[2] - 0.04, abs=1e-6)  # the lower end's height less the radius
        assert np.allclose(p[0, 0, 0].numpy(), Rb.T @ arm, atol=1e-6)  # in the body's frame
        assert np.allclose(n[0, 0, 0].numpy(), Rb.T @ [0, 0, 1], atol=1e-6)
        assert np.allclose(vel[0, 0, 0].numpy(), Rb.T @ (np.array([0.0, 0.0, vz]) + np.cross([0.0, w, 0.0], arm)), atol=1e-6)
        assert gap[0, 1, 0].item() > 1.0  # the step is far
    q, v = torch.tensor([[2.0, -0.1, 0.0]], **F64), torch.tensor([[0.0, 0.2, 0.0]], **F64)  # level, 0.4 m up, over the step
    gap, p, n, vel = wm.contact_gaps(q, v)
    assert gap[0, :, 0].tolist() == pytest.approx([0.4 - 0.04, 0.4 - 0.2 - 0.04], abs=1e-6)
    assert p[0, :, 0, 2].tolist() == pytest.approx([-0.4, -0.2], abs=1e-6)  # the floor's and the step's points, below the body's origin
    assert torch.allclose(n[0, :, 0], torch.tensor([[0.0, 0.0, 1.0]], **F64).expand(2, 3), atol=1e-6)
    assert torch.allclose(vel[0, :, 0], torch.tensor([[0.0, 0.0, 0.2]], **F64).expand(2, 3), atol=1e-6)
    x = wm.contact_inputs(q, v)
    assert x.shape == (1, 26) and x[0, 0].item() == pytest.approx(4.0) and x[0, 13].item() == pytest.approx(4.0)  # both beyond 20 mm
    q = torch.tensor([[2.0, -0.3 + 0.04 + 0.001, 0.0]], **F64)  # 1 mm above the step
    assert wm.contact_inputs(q, v)[0, 13:16].tolist() == pytest.approx([0.2, np.exp(-2.0), 0.0], abs=2e-4)
    with pytest.raises(ValueError, match="the summed terms"):  # a static geom takes no push: no summed terms without a movable object
        MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"basis": True})
    rf, bare = MjcfWorldModel(lay, backbone="rigidformer", rf=RF, contacts=cs).double(), MjcfWorldModel(lay, backbone="rigidformer", rf=RF).double()
    tok = rf.token_inputs(q, v, torch.zeros(1, 1, **F64)).reshape(1, 1, rf.tok_dim)  # the RigidFormer's one token: 13 more per static geom
    assert rf.tok_dim == bare.tok_dim + 26 and torch.equal(tok[0, 0, bare.tok_dim :], wm.contact_inputs(q, v)[0])
    with pytest.raises(ValueError, match="each a collision geom of the world body"):
        contact_spheres(m, lay, static=("foot",))


def test_an_object_on_slides_named_as_an_object():
    m = mujoco.MjModel.from_xml_string(PUSHER)
    lay = MjcfLayout.from_model(m)
    assert lay.objects == () and len(lay.robot_geoms) == 2  # no free joint: the layout counts the puck among the robot's geoms
    with pytest.raises(ValueError, match="has no object"):
        MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay))
    cs = contact_spheres(m, lay, objects=("object",))
    assert cs["geom"].tolist() == [m.geom("tip").id] and cs["tb"].tolist() == [m.body("object").id]  # the puck is no longer the robot's
    assert cs["tg"].shape == (1, 12) and cs["tg"][0, :2].tolist() == [0.0, 5.0] and "sg" not in cs  # its geom: the first body's, a cylinder
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"basis": True}).double()
    assert wm.n_targets == 1 and wm.n_contact == 13 + 12 and {"cs_tb", "cs_tg"} <= set(wm.state_dict()) and "cs_sg" not in wm.state_dict()
    q = torch.tensor([[0.1, 0.0, 0.0, 0.0]], **F64)  # the tip 0.2 m from the puck's axis, at its middle height
    v = torch.tensor([[0.3, 0.0, 0.0, 0.1]], **F64)  # the tip at 0.3 m/s towards it, the puck at 0.1 m/s away
    gap, p, n, vel = wm.contact_gaps(q, v)
    assert gap[0, 0, 0].item() == pytest.approx(0.2 - 0.05 - 0.02, abs=1e-6)
    assert torch.allclose(p[0, 0, 0], torch.tensor([-0.05, 0.0, 0.0], **F64), atol=1e-6)  # the side's point, in the puck's frame
    assert torch.allclose(n[0, 0, 0], torch.tensor([-1.0, 0.0, 0.0], **F64), atol=1e-6)
    assert torch.allclose(vel[0, 0, 0], torch.tensor([0.2, 0.0, 0.0], **F64), atol=1e-6)
    q = torch.tensor([[0.3, 0.0, 0.2, 0.0]], **F64)  # the puck moved 0.2 m along y, the tip beside it
    gap, p, n = wm.contact_gaps(q)
    assert gap[0, 0, 0].item() == pytest.approx(0.13, abs=1e-6) and torch.allclose(n[0, 0, 0], torch.tensor([0.0, -1.0, 0.0], **F64), atol=1e-6)
    both = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, objects=("object",), static=("table",)) | {"basis": True}).double()
    assert both.n_targets == 2 and both.n_contact == 2 * 13 + 12  # the puck, then the table; the summed terms are the puck's
    q = torch.tensor([[0.1, 0.0, 0.0, 0.0]], **F64)
    assert both.contact_gaps(q)[0][0, :, 0].tolist() == pytest.approx([0.13, 0.05 - 0.02], abs=1e-6)  # the tip's centre is 50 mm above the table
    x, basis = both.contact_features(q, v)
    assert x.shape == (1, 2, 1, 13) and torch.equal(x[:, :1], wm.contact_features(q, v)[0]) and torch.equal(basis, wm.contact_features(q, v)[1])
    for bad in ("nobody", "world"):
        with pytest.raises(ValueError, match="no such body, the world"):
            contact_spheres(m, lay, objects=(bad,))
    with pytest.raises(ValueError, match="needs collision geoms of its own"):
        contact_spheres(mujoco.MjModel.from_xml_string(PUSHER.replace('type="cylinder" size="0.05 0.05"', 'type="ellipsoid" size="0.05 0.05 0.05"')),
                        lay, objects=("object",))  # fmt: skip


def gaps_as_before(wm, qpos, qvel):
    """contact_gaps as it was with the free objects alone (before static geoms and named object bodies), word for word."""
    from ndm_gym.worldmodels.mjcf import distance_normal, signed_distance

    b = wm.cs_body
    P, R, V, W = wm.body_poses(qpos, qvel)
    c = P[..., b, None, :] + torch.einsum("...gij,gkj->...gki", R[..., b, :, :], wm.cs_center)
    gaps, points, normals, vels = [], [], [], []
    for i, ob in enumerate(wm._cs_obj):
        po, Ro = P[..., ob, :], R[..., ob, :, :]
        rel = torch.einsum("...gki,...ij->...gkj", c - po[..., None, None, :], Ro)
        d = torch.stack([signed_distance(t, (rel - wm.og_pos[k]) @ wm.og_rot[k], wm.og_size[k]) for t, k in wm._og[i]], -1).amin(-1)
        gap, j = (d - wm.cs_radius).min(-1)
        cj = rel.gather(-2, j[..., None, None].expand(j.shape + (1, 3))).squeeze(-2)
        dn, n = None, None
        for t, k in wm._og[i]:
            dk, nk = distance_normal(t, (cj - wm.og_pos[k]) @ wm.og_rot[k], wm.og_size[k])
            nk = nk @ wm.og_rot[k].T
            dn, n = (dk, nk) if dn is None else (torch.minimum(dn, dk), torch.where((dk < dn).unsqueeze(-1), nk, n))
        p = cj - dn.unsqueeze(-1) * n
        gaps.append(gap)
        points.append(p)
        normals.append(n)
        xw = po[..., None, :] + torch.einsum("...ij,...gj->...gi", Ro, p)
        vr = V[..., b, :] + torch.cross(W[..., b, :], xw - P[..., b, :], dim=-1)
        vo = V[..., ob, None, :] + torch.cross(W[..., ob, None, :].expand_as(xw), xw - po[..., None, :], dim=-1)
        vels.append(torch.einsum("...gi,...ij->...gj", vr - vo, Ro))
    return torch.stack(gaps, -2), torch.stack(points, -3), torch.stack(normals, -3), torch.stack(vels, -3)


def features_as_before(wm, qpos, qvel):
    """contact_features as it was, word for word, on ``gaps_as_before``."""
    gap, p, n, v = gaps_as_before(wm, qpos, qvel)
    s = (gap - wm.cs_margin).unsqueeze(-1)
    near = torch.exp(-s.clamp_min(0.0) / 0.002)
    touch = torch.exp(-s.clamp_min(0.0) / 0.0005)
    x = torch.cat((s.clamp(-0.002, 0.02) / 0.005, touch, (-s).clamp(0.0, 0.002) / 1e-4,
                   near * p / 0.03, near * n, near * v / 0.2, near * (v * n).sum(-1, keepdim=True) / 0.2), -1)  # fmt: skip
    pu, vu = p / 0.03, v / 0.2
    closing = (-(vu * n).sum(-1, keepdim=True)).clamp_min(0.0)
    return x, (touch * torch.cat((vu, torch.cross(pu, vu, dim=-1), closing * n, closing * torch.cross(pu, n, dim=-1)), -1)).sum(-2)


@pytest.mark.parametrize("double", [False, True])
def test_the_free_objects_path_is_as_before_bit_for_bit(double):
    m, lay = probes(*BOX)
    cs = contact_spheres(m, lay)
    assert set(cs) == {"center", "radius", "body", "count", "geom", "margin"}  # nothing new without the options
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"basis": True})
    wm = wm.double() if double else wm
    assert {k for k in wm.state_dict() if k.startswith("cs_")} == {"cs_center", "cs_radius", "cs_body", "cs_margin", "cs_basis"}
    assert wm.n_targets == 1 and wm.n_contact == 39 + 12
    g = torch.Generator().manual_seed(0)
    q = torch.zeros(64, m.nq)
    q[:, :3] = -0.25 * torch.rand(64, 3, generator=g)  # the probes down to and into the object
    q[:, 3:6] = torch.tensor([0.0, 0.0, 0.05]) + torch.tensor([1.0, 0.1, 0.05]) * torch.rand(64, 3, generator=g)  # the object under each of them
    q[:, 6:10] = torch.nn.functional.normalize(torch.randn(64, 4, generator=g), dim=-1)
    v = torch.randn(64, m.nv, generator=g)
    q, v = (q.double(), v.double()) if double else (q, v)
    for got, want in zip(wm.contact_gaps(q, v), gaps_as_before(wm, q, v)):
        assert torch.equal(got, want)
    assert all(torch.equal(a, b) for a, b in zip(wm.contact_gaps(q), gaps_as_before(wm, q, v)[:3]))
    x, basis = features_as_before(wm, q, v)
    got = wm.contact_features(q, v)
    assert torch.equal(got[0], x) and torch.equal(got[1], basis)
    assert torch.equal(wm.contact_inputs(q, v), torch.cat((x.flatten(-3), basis.flatten(-2)), -1))
    assert (wm.contact_gaps(q)[0] < 0).any() and (wm.contact_gaps(q)[0] > 0.02).any()  # states in contact and states apart


def test_static_geoms_beside_a_free_object_and_in_the_rigidformers_columns():
    xml = PROBES.replace("KIND", "box").replace("SIZE", "0.1 0.1 0.05").replace("MARGIN", "")
    m = mujoco.MjModel.from_xml_string(xml.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -1"/>'))
    lay = MjcfLayout.from_model(m)
    cs, plain = contact_spheres(m, lay, static=()), contact_spheres(m, lay)
    assert cs["sg"].shape == (1, 11) and all(np.array_equal(cs[k], plain[k]) for k in ("center", "radius", "body", "count", "geom"))
    wm, free = (MjcfWorldModel(lay, hidden=(8,), contacts=c | {"basis": True}).double() for c in (cs, plain))
    assert wm.n_targets == 2 and wm.n_contact == 2 * 39 + 12 and free.n_contact == 39 + 12
    q, v = state(m, lift=(-0.2 + 0.02 + 0.001, 0.0, 0.0), vel={0: -0.4})
    x, basis = wm.contact_features(q, v)
    x0, basis0 = free.contact_features(q, v)
    assert torch.equal(x[:, :1], x0) and torch.equal(basis, basis0)  # the object's entries and the summed terms as without the floor
    assert wm.contact_gaps(q)[0][0, 1].tolist() == pytest.approx([1.3 - 0.2 + 0.021 - 0.02, 1.3 - 0.015, 1.3 - 0.01], abs=3e-5)  # the probes' heights
    with pytest.raises(ValueError, match="do not go with geometry=True"):
        MjcfWorldModel(lay, hidden=(8,), geometry=True, contacts=cs)
    rf, rf0 = (MjcfWorldModel(lay, backbone="rigidformer", rf=RF, contacts=c | {"basis": True}).double() for c in (cs, plain))
    assert rf.tok_dim == rf0.tok_dim + 13
    u = torch.zeros(1, 3, **F64)
    tok, tok0 = rf.token_inputs(q, v, u).reshape(1, 4, rf.tok_dim), rf0.token_inputs(q, v, u).reshape(1, 4, rf0.tok_dim)
    base = rf0.tok_dim - 13 - 12
    assert torch.equal(tok[..., : base + 13], tok0[..., : base + 13]) and torch.equal(tok[..., base + 26 :], tok0[..., base + 13 :])
    assert torch.equal(tok[0, :3, base + 13 : base + 26], x[0, 1])  # each probe's token: its 13 inputs to the floor


def test_an_objects_other_collision_geoms_are_refused_by_name():
    """A free object's collision geom that no input reads (a mesh or an ellipsoid beside a primitive one, or one on a body
    below the object's) is an error that names it where the object's geoms are read, not a geom left out."""
    geom = '<geom name="obj" type="KIND" size="SIZE" MARGIN/>'
    box = '<geom name="obj" type="box" size="0.1 0.1 0.05"/>'
    m = mujoco.MjModel.from_xml_string(PROBES.replace(geom, box + '<geom name="knob" type="ellipsoid" size="0.02 0.02 0.03" pos="0 0 0.08"/>'))
    lay = MjcfLayout.from_model(m)
    assert lay.object_unread == (("knob",),) and len(lay.object_geoms[0]) == 1
    MjcfWorldModel(lay, hidden=(8,))  # a model that reads no object geom is built as before
    with pytest.raises(ValueError, match=r"contacts=\.\.\.\): the objects' collision geoms \['knob'\] are not read"):
        MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay))
    with pytest.raises(ValueError, match=r"geometry=True\): the objects' collision geoms \['knob'\] are not read"):
        MjcfWorldModel(lay, hidden=(8,), geometry=True)
    lid = '<body name="lid" pos="0 0 0.06"><geom type="box" size="0.05 0.05 0.01"/></body>'  # an unnamed geom: named by its number
    m = mujoco.MjModel.from_xml_string(PROBES.replace(geom, box + lid))
    lay = MjcfLayout.from_model(m)
    assert lay.object_unread == ((f"geom {m.ngeom - 1}",),)
    with pytest.raises(ValueError, match=rf"the objects' collision geoms \['geom {m.ngeom - 1}'\] are not read"):
        MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay))
    assert probes(*BOX)[1].object_unread == ((),)  # an object of primitive geoms on its own body: nothing unread
    puck = '<geom name="puck" type="cylinder" size="0.05 0.05"/>'
    m = mujoco.MjModel.from_xml_string(PUSHER.replace(puck, puck + '<body name="cap" pos="0 0 0.06"><geom name="cap" type="sphere" size="0.02"/></body>'))
    with pytest.raises(ValueError, match=r"collision geoms \['cap'\] on bodies below the named objects are not read"):
        contact_spheres(m, MjcfLayout.from_model(m), objects=("object",))
