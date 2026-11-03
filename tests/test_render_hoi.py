"""tools/render_hoi.py without a GPU: the --states npz format (written, read back, checked against the task's scene), the
recorder on a stand-in for HDMISim, the converted clips' loading, resampling, the interaction window and the hinged
objects' kinematics; with pxr (Isaac Lab's environment, else skipped), the USD to OBJ conversion of a small stage."""

import os
import sys

import numpy as np
import pytest
import torch

from ndm_gym.robots.g1 import G1_JOINT_NAMES, NUM_J

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import convert_hdmi as ch  # noqa: E402
import render_hoi as rh  # noqa: E402


def _quats(g, *shape):
    q = g.normal(size=(*shape, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _motion(task: str, T: int = 7, seed: int = 0) -> rh.Motion:
    """Random states of a task's scene in the Motion layout of recorded states."""
    g = np.random.default_rng(seed)
    cfg = rh.hdmi_tasks.get(task)
    n = len(cfg.extra_objects)
    return rh.Motion(task, 50.0, g.normal(size=(T, 3)), _quats(g, T), g.normal(size=(T, NUM_J)), g.normal(size=(T, 3)), _quats(g, T),
                     g.normal(size=T) if cfg.object_joint else None, g.normal(size=(T, n, 3)) if n else None, _quats(g, T, n) if n else None,
                     None, np.arange(10, 10 + T))  # fmt: skip


@pytest.mark.parametrize("task", ["open_door-feet", "carry_and_place_bread_box", "move_suitcase"])
def test_states_roundtrip(tmp_path, task):
    m = _motion(task)
    path = str(tmp_path / "states.npz")
    rh.save_states(path, m)
    back = rh.load_states(path)
    assert back.task == task and back.fps == 50.0 and back.contact is None
    for k in rh.Motion._fields:
        a, b = getattr(m, k), getattr(back, k)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=k)
    with np.load(path) as z:
        assert set(z.files) == set(rh.STATE_KEYS) | ({"object_joint_pos"} if m.hinge is not None else set()) | (
            {"extra_pos", "extra_quat_wxyz"} if m.extra_pos is not None else set())


def test_states_checked(tmp_path):
    path = str(tmp_path / "s.npz")

    def write(name="open_door-feet", drop=(), **change):
        rh.save_states(path, _motion(name))
        with np.load(path) as z:
            a = {k: z[k] for k in z.files if k not in drop}
        a.update(change)
        np.savez(path, **a)

    write(drop=("frame",))
    with pytest.raises(ValueError, match="missing"):
        rh.load_states(path)
    write(joint_pos=np.zeros((7, 23)))
    with pytest.raises(ValueError, match="joint_pos"):
        rh.load_states(path)
    write(root_quat_wxyz=np.ones((7, 4)))
    with pytest.raises(ValueError, match="unit quaternions"):
        rh.load_states(path)
    write(drop=("object_joint_pos",))  # the door's hinge angle is required
    with pytest.raises(ValueError, match="object_joint_pos"):
        rh.load_states(path)
    write("move_suitcase", object_joint_pos=np.zeros((7, 1)))  # a suitcase has none
    with pytest.raises(ValueError, match="do not belong"):
        rh.load_states(path)
    write("carry_and_place_bread_box", extra_pos=np.zeros((7, 1, 3)))  # two supports
    with pytest.raises(ValueError, match="extra_pos"):
        rh.load_states(path)
    write(task=np.array("open_door"))
    with pytest.raises(KeyError, match="unknown HDMI task"):
        rh.load_states(path)
    write(object_pos=np.full((7, 3), np.nan))
    with pytest.raises(ValueError, match="finite"):
        rh.load_states(path)


class _Buf:  # an Isaac Lab 3 data buffer: a torch view as .torch
    def __init__(self, x):
        self.torch = x


class _Asset:
    def __init__(self, pose, joint_pos=None):
        self.data = type("Data", (), {"root_link_pose_w": _Buf(pose), "joint_pos": _Buf(joint_pos)})()


def test_recorder(tmp_path):
    """StateRecorder on a stand-in with HDMISim's buffers: Isaac Lab 3 poses (xyzw) relative to the env origins, joints
    from the robot's order to G1's, the hinge angle, the props, the frame counter."""
    g = torch.Generator().manual_seed(0)
    N, env = 3, 1
    unit = lambda *s: torch.nn.functional.normalize(torch.randn(*s, 4, generator=g, dtype=torch.float64), dim=-1)  # noqa: E731
    pose = lambda: torch.cat((torch.randn(N, 3, generator=g, dtype=torch.float64), unit(N)), -1)  # noqa: E731
    perm = torch.randperm(NUM_J, generator=g)  # robot joint i is G1 joint perm[i]
    j_g1 = torch.argsort(perm)  # G1 joint k sits at robot index j_g1[k]
    sim = type("Sim", (), {})()
    sim.cfg = rh.hdmi_tasks.get("carry_and_place_bread_box")
    sim.scene = type("Scene", (), {"env_origins": torch.randn(N, 3, generator=g, dtype=torch.float64)})()
    sim.j_g1, sim.hinged, sim.t = j_g1, False, torch.tensor([4, 9, 2])
    rec = rh.StateRecorder(sim, env)
    rows = []
    for step in range(3):
        sim.robot, sim.obj = _Asset(pose(), torch.randn(N, NUM_J, generator=g, dtype=torch.float64)), _Asset(pose())
        sim.extras = [_Asset(pose()), _Asset(pose())]
        sim.t = sim.t + 1
        rec.add()
        rows.append((sim.robot, sim.obj, sim.extras, int(sim.t[env])))
    m = rec.motion()
    org = sim.scene.env_origins[env].numpy()
    wxyz = lambda p: p[env, [6, 3, 4, 5]].numpy()  # noqa: E731
    for t, (robot, obj, extras, frame) in enumerate(rows):
        np.testing.assert_allclose(m.root_pos[t], robot.data.root_link_pose_w.torch[env, :3].numpy() - org)
        np.testing.assert_allclose(m.root_quat[t], wxyz(robot.data.root_link_pose_w.torch))
        np.testing.assert_allclose(m.joint_pos[t], robot.data.joint_pos.torch[env, j_g1].numpy())
        np.testing.assert_allclose(m.object_pos[t], obj.data.root_link_pose_w.torch[env, :3].numpy() - org)
        np.testing.assert_allclose(m.object_quat[t], wxyz(obj.data.root_link_pose_w.torch))
        for k, e in enumerate(extras):
            np.testing.assert_allclose(m.extra_pos[t, k], e.data.root_link_pose_w.torch[env, :3].numpy() - org)
            np.testing.assert_allclose(m.extra_quat[t, k], wxyz(e.data.root_link_pose_w.torch))
        assert m.frame[t] == frame
    assert m.hinge is None and m.fps == 50.0
    rec.save(str(tmp_path / "rec.npz"))
    back = rh.load_states(str(tmp_path / "rec.npz"))
    np.testing.assert_array_equal(back.extra_quat, m.extra_quat)


def _converted_clip(out, name: str, T: int, seed: int = 0) -> dict:
    """A synthetic HDMI clip of a task's scene, converted by tools/convert_hdmi.py into ``out``; returns its arrays."""
    g = np.random.default_rng(seed)
    t = ch.TASKS[name]
    joints = list(G1_JOINT_NAMES) + ([t.object_joint] if t.object_joint else [])
    bodies = list(ch.KEYPOINTS) + [t.object_name, *t.extras]
    contact = np.zeros((T, len(t.segments)), bool)
    contact[2:5] = True
    motion = {"body_pos_w": g.normal(size=(T, len(bodies), 3)), "body_quat_w": _quats(g, T, len(bodies)),
              "body_lin_vel_w": g.normal(size=(T, len(bodies), 3)), "body_ang_vel_w": g.normal(size=(T, len(bodies), 3)),
              "joint_pos": g.normal(size=(T, len(joints))), "joint_vel": g.normal(size=(T, len(joints))), "object_contact": contact}  # fmt: skip
    task = t._replace(frames=T, segments=((((2, 5),),) * len(t.segments)))
    arrays, side = ch.convert(name, task, motion, {"body_names": bodies, "joint_names": joints, "fps": 50.0})
    ch.write_clip(out, name, arrays, side)
    return arrays


@pytest.mark.parametrize("task", ["open_door-feet", "carry_and_place_bread_box", "traverse_door"])
def test_clip_motion(tmp_path, task):
    """A converted clip loads as the Motion of its arrays: G1 joint order, the hinge angle, the props, the contact flags."""
    a = _converted_clip(str(tmp_path), task, T=9)
    m = rh.clip_motion(task, str(tmp_path))
    f64 = lambda k: a[k].astype(np.float64)  # noqa: E731
    np.testing.assert_array_equal(m.root_pos, f64("root_pos"))
    np.testing.assert_array_equal(m.root_quat, f64("root_quat_wxyz"))
    np.testing.assert_array_equal(m.joint_pos, f64("joint_pos"))
    np.testing.assert_array_equal(m.object_pos, f64("object_pos"))
    np.testing.assert_array_equal(m.object_quat, f64("object_quat_wxyz"))
    assert (m.hinge is None) == ("object_joint_pos" not in a) and (m.extra_pos is None) == ("extra_pos" not in a)
    if m.hinge is not None:
        np.testing.assert_array_equal(m.hinge, f64("object_joint_pos")[:, 0])
    if m.extra_pos is not None:
        np.testing.assert_array_equal(m.extra_pos, f64("extra_pos"))
        np.testing.assert_array_equal(m.extra_quat, f64("extra_quat_wxyz"))
    np.testing.assert_array_equal(m.contact, a["contact"])
    np.testing.assert_array_equal(m.frame, np.arange(9))
    assert m.task == task and m.fps == 50.0
    os.rename(tmp_path / f"{task}.npz", tmp_path / "move_suitcase.npz")
    os.rename(tmp_path / f"{task}.json", tmp_path / "move_suitcase.json")
    with pytest.raises(ValueError, match="is not the clip of move_suitcase"):
        rh.clip_motion("move_suitcase", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="convert_hdmi"):
        rh.clip_motion(task, str(tmp_path))


def test_resample():
    """Positions, joints and the hinge linear between steps, rotations slerped; contact and frames from the nearer step."""
    T = 5
    ang = np.linspace(0.0, 1.0, T)
    quat = np.stack((np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)), -1)  # about z
    m = rh.Motion("open_door-feet", 4.0, np.outer(ang, [1.0, 2.0, 3.0]), quat, np.outer(ang, np.ones(NUM_J)), np.zeros((T, 3)), quat, ang, None, None,
                  np.arange(T)[:, None] % 2 == 0, np.arange(T))  # fmt: skip
    r = rh.resample(m, np.array([0.0, 0.375, 0.625]), 100.0)  # steps 0, 1.5, 2.5
    np.testing.assert_allclose(r.root_pos[:, 0], [0.0, 0.375, 0.625])
    np.testing.assert_allclose(r.hinge, [0.0, 0.375, 0.625])
    np.testing.assert_allclose(2 * np.arctan2(r.root_quat[:, 3], r.root_quat[:, 0]), [0.0, 0.375, 0.625], atol=1e-12)
    assert r.fps == 100.0 and list(r.frame) == [0, 2, 3] and list(r.contact[:, 0]) == [True, True, False]


def test_fit_view_and_key_light():
    """fit_view puts every point within MARGIN of the half frame (one at its edge, or the distance is MIN_DISTANCE); the
    soft key light's directions are unit vectors within SOFT_ANGLE of the key's."""
    P = np.random.default_rng(3).normal(size=(200, 3)) * [1.5, 0.6, 0.5] + [2.0, -1.0, 0.8]
    for az in (0.0, 75.0, 230.0):
        c, d = rh.fit_view(P, az, 16 / 9)
        f, r, u = rh.camera_axes(az)
        rel = P - (c - d * f)
        x, y = (rel @ r) / (rel @ f), (rel @ u) / (rel @ f)
        tv = np.tan(np.radians(rh.FOVY) / 2) * rh.MARGIN
        edge = max(np.abs(x).max() / (tv * 16 / 9), np.abs(y).max() / tv)
        assert edge <= 1 + 1e-9 and (edge > 1 - 1e-9 or d == rh.MIN_DISTANCE)
    keys = rh.soft_key_dirs(30.0)
    center = rh.light_dir(30.0, rh.KEY[2], rh.KEY[3])
    assert keys.shape == (rh.SOFT_SAMPLES, 3)
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0)
    assert (np.degrees(np.arccos(np.clip(keys @ center, -1, 1))) <= rh.SOFT_ANGLE + 1e-9).all()


def test_interaction_window():
    moving = np.zeros(100, bool)
    moving[40:50] = True
    contact = np.zeros((100, 2), bool)
    contact[60:70, 1] = True
    assert rh.interaction_window(moving, contact, 40) == 35  # all 20 interaction steps fit in windows 30..40: the middle
    assert rh.interaction_window(moving, None, 20) == 35  # windows 30..40 hold all 10 moving steps
    assert rh.interaction_window(moving, contact, 100) == 0 and rh.interaction_window(moving, contact, 150) == 0


def _door_asset(panel_pose=None) -> dict:
    """A door like HDMI's: a fixed frame and a panel on a revolute joint about z at (0, 0.5, 0) of the frame."""
    P = np.eye(4)
    P[:3, 3] = [0.0, 0.5, 0.0]
    joint = {"name": "door_joint", "type": "revolute", "body0": "door_frame", "body1": "door_panel", "pos0": [0.0, 0.5, 0.0], "quat0": [1.0, 0, 0, 0],
             "pos1": [0.0, 0.0, 0.0], "quat1": [1.0, 0, 0, 0], "axis": 2, "range": [-2.618, 0.0]}  # fmt: skip
    bodies = [{"name": "door_frame", "pose": np.eye(4).tolist(), "geoms": []},
              {"name": "door_panel", "pose": (P if panel_pose is None else panel_pose).tolist(),
               "geoms": [{"obj": "p.obj", "prim": "/p", "material": {"name": "Beadboard", "diffuse": None}, "aabb": [[0.0, -0.9, 0.0], [0.05, 0.0, 2.0]]}]}]  # fmt: skip
    return {"usd": "door.usd", "root": "door_frame", "bodies": bodies, "joints": [joint]}


def test_hinged_object():
    """The panel's pose follows the base and the hinge angle through the joint frames; its contact body's corners."""
    door = _door_asset()
    rh.check_joint_frames(door, "door_joint")
    yaw = np.pi / 2  # the base turned about z, at (1, 2, 0)
    base_q = np.array([[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]] * 2)
    poses = rh.asset_poses(door, np.array([[1.0, 2.0, 0.0]] * 2), base_q, np.array([0.0, -np.pi / 2]), "door_joint")
    np.testing.assert_allclose(poses["door_panel"][:, :3, 3], [[0.5, 2.0, 0.0]] * 2, atol=1e-12)  # the hinge at (0, 0.5) of the base
    np.testing.assert_allclose(poses["door_panel"][1, :3, :3], np.eye(3), atol=1e-12)  # yaw and the angle cancel
    c = rh.contact_body_corners(door, poses, "door_panel")
    assert c.shape == (2, 8, 3)
    np.testing.assert_allclose(c[1].min(0), [0.5, 1.1, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="revolute joints"):
        rh.asset_poses(door, np.zeros((1, 3)), base_q[:1], None, None)
    moved = np.eye(4)
    moved[:3, 3] = [0.0, 0.6, 0.0]
    with pytest.raises(ValueError, match="away from its authored pose"):
        rh.check_joint_frames(_door_asset(moved), "door_joint")


def _read_obj(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vertices, normals, texture coordinates and triangles of an OBJ whose face corners index all three alike."""
    out = {"v": [], "vn": [], "vt": []}
    f = []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if tok and tok[0] in out:
                out[tok[0]].append([float(x) for x in tok[1:]])
            elif tok and tok[0] == "f":
                refs = [[int(r) for r in c.split("/") if r] for c in tok[1:]]
                assert all(len(set(r)) == 1 for r in refs), line
                f.append([r[0] - 1 for r in refs])
    return np.array(out["v"]), np.array(out["vn"]), np.array(out["vt"]), np.array(f)


def test_usd_to_obj(tmp_path):
    """A stage like HDMI's door: visible prims of each rigid body in its frame at their authored scale (a guide prim left
    out), triangles, normals and a mesh's texture coordinates, the bound material, the joint frames; cached by the USD's
    hash."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    usd = str(tmp_path / "door.usd")
    stage = Usd.Stage.CreateNew(usd)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/door").GetPrim())
    mat = UsdShade.Material.Define(stage, "/door/Looks/Oak")
    shader = UsdShade.Shader.Define(stage, "/door/Looks/Oak/Shader")
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("Oak")
    frame = UsdGeom.Xform.Define(stage, "/door/door_frame")
    UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
    post = UsdGeom.Cube.Define(stage, "/door/door_frame/post")
    post.GetSizeAttr().Set(1.0)
    post.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 1.0))
    post.AddScaleOp().Set(Gf.Vec3f(0.1, 1.0, 2.0))
    UsdShade.MaterialBindingAPI.Apply(post.GetPrim()).Bind(mat)
    guide = UsdGeom.Cube.Define(stage, "/door/door_frame/guide")
    guide.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    panel = UsdGeom.Xform.Define(stage, "/door/door_panel")
    panel.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.5, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(panel.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/door/door_panel/board")
    mesh.GetPointsAttr().Set([(0, 0, 0), (0, -0.9, 0), (0, -0.9, 2), (0, 0, 2), (0.05, 0, 0)])
    mesh.GetFaceVertexCountsAttr().Set([4, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 0, 4, 1])
    mesh.GetNormalsAttr().Set([(-1, 0, 0)] * 4 + [(0, 0, -1)] * 3)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0.5, 0), (1, 0)])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/door/door_frame/door_joint")
    joint.CreateBody0Rel().SetTargets(["/door/door_frame"])
    joint.CreateBody1Rel().SetTargets(["/door/door_panel"])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.5, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(-150.0)
    joint.CreateUpperLimitAttr().Set(0.0)
    root = UsdPhysics.FixedJoint.Define(stage, "/door/root_joint")
    root.CreateBody1Rel().SetTargets(["/door/door_frame"])
    stage.GetRootLayer().Save()

    cache = str(tmp_path / "cache")
    a = rh.usd_asset(usd, cache)
    assert a["root"] == "door_frame" and [b["name"] for b in a["bodies"]] == ["door_frame", "door_panel"]
    (post_geom,), (board,) = (b["geoms"] for b in a["bodies"])
    assert post_geom["material"] == {"name": "Oak", "diffuse": None} and board["material"] == {"name": "default", "diffuse": None}
    assert not post_geom["texcoords"] and board["texcoords"]
    v, n, uv, f = _read_obj(os.path.join(a["dir"], post_geom["obj"]))
    np.testing.assert_allclose([v.min(0), v.max(0)], [[-0.05, 0.5, 0.0], [0.05, 1.5, 2.0]], atol=1e-6)  # scaled, in the frame's frame
    assert f.shape == (12, 3) and len(n) == len(v) == 24 and len(uv) == 0
    v, n, uv, f = _read_obj(os.path.join(a["dir"], board["obj"]))  # the panel's own frame: authored where the joint puts it
    assert f.shape == (3, 3) and len(v) == len(uv) == 7  # a quad and a triangle; points 0 and 1 twice, with each face's normal
    np.testing.assert_allclose(uv[np.argmin(np.abs(v - [0.05, 0.0, 0.0]).sum(1))], [0.5, 0.0])  # point 4's st
    tri = v[f]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_n = n[f[:, 0]]
    assert (np.sum(normal * face_n, -1) > 0).all()  # windings agree with the normals
    (j,) = a["joints"]
    assert (j["type"], j["body0"], j["body1"], j["axis"]) == ("revolute", "door_frame", "door_panel", 2)
    np.testing.assert_allclose(j["range"], np.radians([-150.0, 0.0]))
    rh.check_joint_frames(a, "door_joint")
    ball = {"prim": "/ball", "material": {"name": "Bamboo_Planks", "diffuse": None}, "texcoords": True}
    assert rh.material_of(ball)[1] == (1.0, 1.0, 1.0, 1.0)  # its texture carries the color
    with pytest.raises(ValueError, match="needs texture coordinates"):
        rh.material_of({**ball, "texcoords": False})
    mtime = os.path.getmtime(os.path.join(a["dir"], board["obj"]))
    assert rh.usd_asset(usd, cache) == a and os.path.getmtime(os.path.join(a["dir"], board["obj"])) == mtime  # from the cache
    UsdGeom.Cylinder.Define(stage, "/door/door_panel/knob")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="not supported"):
        rh.usd_asset(usd, cache)
