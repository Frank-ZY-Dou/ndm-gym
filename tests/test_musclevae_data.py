"""MuscleVAE's converted assets (tools/convert_musclevae.py): the converter's pieces on small synthetic inputs (ODE's geom
masses, MASS's body names, the anchor binding with its blend weights and the root's shift, BVH parsing and forward
kinematics, the joint-space clips, rotations), and the converted character, muscles and clips in data/musclevae when
they exist, checked with MuJoCo's forward kinematics."""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from ndm_gym.robots import musclevae_char as mc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import convert_musclevae as cm  # noqa: E402

HAVE_DATA = all(os.path.isfile(p) for p in (mc.MJCF_PATH, mc.CHARACTER_PATH, mc.MUSCLES_PATH)) and os.path.isdir(mc.CLIPS_DIR)


def random_rot(g: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    v = scale * g.normal(size=(n, 3))
    angle = np.linalg.norm(v, axis=-1)
    return np.stack([cm.axis_angle(a / t, t) for a, t in zip(v, angle)])


def test_rotations():
    g = np.random.default_rng(0)
    R = np.concatenate((random_rot(g, 200, 2.0), random_rot(g, 50, 1e-4), [cm.axis_angle(np.array([0.0, 0.6, 0.8]), np.pi - 1e-9), np.eye(3)]))
    q = cm.mat_to_quat(R)
    assert np.all(q[:, 0] >= 0) and np.allclose(np.linalg.norm(q, axis=-1), 1)
    back = cm.quat_xyzw_to_mat(np.concatenate((q[:, 1:], q[:, :1]), -1))
    assert np.abs(back - R).max() < 1e-12
    assert np.allclose(cm.quat_angle(q, q), 0) and np.isclose(cm.quat_angle(q[:1], -q[:1])[0], 0)
    turned = cm.mat_to_quat(R @ cm.axis_angle(np.array([0.0, 0.0, 1.0]), np.full(len(R), 0.7)))
    assert np.allclose(cm.quat_angle(q, turned), 0.7) and np.allclose(cm.quat_angle(q, -turned), 0.7)
    e = np.array([[10.0, -20.0, 30.0], [170.0, 89.0, -45.0]])
    for row, Rx in zip(e, cm.euler_xyz_deg(e)):  # intrinsic x, y, z: Rx Ry Rz
        a = np.deg2rad(row)
        rx = np.array([[1, 0, 0], [0, math.cos(a[0]), -math.sin(a[0])], [0, math.sin(a[0]), math.cos(a[0])]])
        ry = np.array([[math.cos(a[1]), 0, math.sin(a[1])], [0, 1, 0], [-math.sin(a[1]), 0, math.cos(a[1])]])
        rz = np.array([[math.cos(a[2]), -math.sin(a[2]), 0], [math.sin(a[2]), math.cos(a[2]), 0], [0, 0, 1]])
        assert np.abs(Rx - rx @ ry @ rz).max() < 1e-12
    # the twist about an axis: exact for rotations about it, the closest one otherwise
    axis = np.array([0.0, 1.0, 0.0])
    th = np.linspace(-3.1, 3.1, 25)
    assert np.abs(cm.hinge_angle(cm.axis_angle(axis, th), axis) - th).max() < 1e-12
    L = cm.axis_angle(axis, 0.7) @ cm.axis_angle(np.array([1.0, 0.0, 0.0]), 0.3)
    grid = np.linspace(-np.pi, np.pi, 20001)
    best = grid[np.argmin(cm.quat_angle(cm.mat_to_quat(cm.axis_angle(axis, grid)), np.broadcast_to(cm.mat_to_quat(L), (len(grid), 4))))]
    assert abs(cm.hinge_angle(L, axis) - best) < 1e-3
    assert np.array_equal(cm.C @ np.array([1.0, 2.0, 3.0]), [3.0, 1.0, 2.0]) and np.isclose(np.linalg.det(cm.C), 1)  # (x, y, z) -> (z, x, y)


def test_mass_body_names():
    """MASS's right side is the character's left (their loader mirrors x)."""
    assert [cm.mass_body(n) for n in ("Pelvis", "Spine", "FemurR", "FemurL", "TalusR", "ShoulderL", "ForeArmR", "FootThumbR", "FootPinkyL")] == [
        "pelvis", "lowerback", "lUpperLeg", "rUpperLeg", "lFoot", "rClavicle", "lLowerArm", "FootThumbL", "FootPinkyR"]  # fmt: skip
    assert {cm.mass_body(b + s) for b in cm.MASS_SIDED for s in "RL"} | set(cm.MASS_UNSIDED.values()) == set(mc.BODY_NAMES)
    for bad in ("Femur", "FemurX", "Tail"):
        with pytest.raises(ValueError):
            cm.mass_body(bad)


def test_geom_mass():
    """ODE's dMassSetSphere, dMassSetCapsule (along z) and dMassSetBox at a density, and a body's center of mass and
    inertia about it (parallel axes)."""
    rho = 1000.0
    m, inertia, t, size = cm.geom_mass({"GeomType": "Sphere", "Scale": [0.1, 0.1, 0.1]}, rho)
    assert t == "sphere" and size == [0.1] and np.isclose(m, 4 / 3 * math.pi * 1e-3 * rho) and np.allclose(inertia, np.eye(3) * 0.4 * m * 0.01)
    m, inertia, t, size = cm.geom_mass({"GeomType": "Capsule", "Scale": [0.1, 0.4, 0.0]}, rho)
    m1, m2 = math.pi * 0.01 * 0.4 * rho, 4 / 3 * math.pi * 1e-3 * rho
    ia = m1 * (0.25 * 0.01 + 0.16 / 12) + m2 * (0.4 * 0.01 + 0.375 * 0.04 + 0.25 * 0.16)
    assert t == "capsule" and size == [0.1, 0.2] and np.isclose(m, m1 + m2) and np.allclose(np.diag(inertia), [ia, ia, (0.5 * m1 + 0.4 * m2) * 0.01])
    m, inertia, t, size = cm.geom_mass({"GeomType": "Cube", "Scale": [0.1, 0.2, 0.3]}, rho)
    assert t == "box" and np.allclose(size, [0.05, 0.1, 0.15]) and np.isclose(m, 6.0)
    assert np.allclose(np.diag(inertia), [0.5 * 0.13, 0.5 * 0.10, 0.5 * 0.05])
    with pytest.raises(ValueError):
        cm.geom_mass({"GeomType": "Plane", "Scale": [1, 1, 1]}, rho)
    def sphere(gid, p, r):
        return {"GeomID": gid, "Name": f"g{gid}", "GeomType": "Sphere", "Position": p, "Quaternion": [0, 0, 0, 1], "Scale": [r] * 3,
                "Friction": 0.8, "Collidable": True}  # fmt: skip

    body = {"Name": "b", "ParentBodyID": -1, "MassMode": "MassValue", "InertiaMode": "Density", "Density": rho, "Mass": 15.0,
            "Quaternion": [0, 0, 0, 1], "Position": [0, 1, 0], "Geoms": [sphere(0, [0, 1, 0], 0.1), sphere(1, [0.2, 0.9, 0], 0.05)]}  # fmt: skip
    b = cm.parse_body(body)
    ma, mb = 4 / 3 * math.pi * 1e-3 * rho, 4 / 3 * math.pi * 1.25e-4 * rho
    com = (ma * np.array([0, 1, 0]) + mb * np.array([0.2, 0.9, 0])) / (ma + mb)
    want = 0.4 * ma * 0.01 * np.eye(3) + 0.4 * mb * 0.0025 * np.eye(3)
    for m_, p in ((ma, np.array([0, 1, 0])), (mb, np.array([0.2, 0.9, 0]))):
        r = p - com
        want += m_ * (r @ r * np.eye(3) - np.outer(r, r))
    assert b.mass == 15.0 and np.isclose(b.geom_mass, ma + mb) and np.allclose(b.com, com) and np.allclose(b.inertia, want)  # inertia not rescaled


def toy_character() -> cm.Character:
    """pelvis (root, two spheres: its first geom 2 cm above its center of mass) -> lUpperLeg -> lLowerLeg -> lFoot."""
    def body(name, parent, com, geom0):
        return cm.Body(name, parent, 1.0, 1.0, np.array(com, float), np.eye(3), [{"pos": np.array(geom0, float)}], np.array(geom0, float))

    bodies = [body("pelvis", -1, [0, 1.0, 0], [0, 1.02, 0]), body("lUpperLeg", 0, [0.1, 0.75, 0], [0.1, 0.75, 0]),
              body("lLowerLeg", 1, [0.1, 0.3, 0], [0.1, 0.3, 0]), body("lFoot", 2, [0.1, 0.05, 0.05], [0.1, 0.05, 0.05])]  # fmt: skip
    joints = [cm.Joint(n, "ball", p, c, np.array(a, float), None, None, None, 10.0)
              for n, p, c, a in (("hip", 0, 1, [0.1, 0.95, 0]), ("knee", 1, 2, [0.1, 0.5, 0]), ("ankle", 2, 3, [0.1, 0.1, 0]))]  # fmt: skip
    return cm.Character(bodies, joints, [], np.array([0, 1.02, 0.0]), np.array([0, -9.8, 0.0]), {})


def test_bind_anchors():
    """Their loader's binding (LDR:238-375): ends stay on their bodies; an intermediate waypoint within 8 cm of the nearest
    reference point (the root's center of mass, each other body's parent joint) is blended between that body and its
    parent with weights 1/sqrt(distance), or moves with the root; anchors on the root shift by its first geom's offset."""
    char = toy_character()
    mirror = lambda p: [-p[0], p[1], p[2]]  # noqa: E731  MASS's x is mirrored (LDR:244)
    wps = [("Pelvis", [0.12, 0.98, 0.0]),  # origin within 8 cm of the hip, kept on the pelvis
           ("FemurR", [0.1, 0.53, 0.01]),  # 3.2 cm from the knee: lLowerLeg / lUpperLeg (reference: the hip)
           ("FemurR", [0.12, 0.9, 0.0]),  # 5.4 cm from the hip, 15.6 cm from the root: lUpperLeg / pelvis (reference: its center of mass)
           ("Pelvis", [0.03, 1.0, 0.0]),  # nearest to the root's center of mass: moves with the root
           ("TibiaR", [0.1, 0.3, 0.1]),  # farther than 8 cm from every reference point: its own body
           ("TalusR", [0.1, 0.12, 0.0])]  # insertion within 8 cm of the ankle, kept on lFoot  # fmt: skip
    out = cm.bind_anchors(char, [{"BodyName": b, "Position": mirror(p)} for b, p in wps])
    shift = np.array([0, -0.02, 0])
    q = [np.array(p) for _, p in wps]
    inv_sqrt = lambda p, ref: 1 / math.sqrt(math.dist(p, ref))  # noqa: E731
    w1 = inv_sqrt(q[1], [0.1, 0.5, 0]) / (inv_sqrt(q[1], [0.1, 0.5, 0]) + inv_sqrt(q[1], [0.1, 0.95, 0]))  # knee, hip
    w2 = inv_sqrt(q[2], [0.1, 0.95, 0]) / (inv_sqrt(q[2], [0.1, 0.95, 0]) + inv_sqrt(q[2], [0, 1.0, 0]))  # hip, root
    want = [(0, 0, 1.0), (2, 1, w1), (1, 0, w2), (0, 0, 1.0), (2, 2, 1.0), (3, 3, 1.0)]
    assert [(a.body0, a.body1) for a in out] == [w[:2] for w in want]
    assert np.allclose([a.weight0 for a in out], [w[2] for w in want], rtol=1e-14) and 0.5 < w1 < 1 and 0.5 < w2 < 1
    assert [a.source_body for a in out] == [0, 1, 1, 0, 2, 3]
    coms = [b.com for b in char.bodies]
    for a, qq in zip(out, q):
        p0, p1 = qq + (shift if a.body0 == 0 else 0), qq + (shift if a.body1 == 0 else 0)
        assert np.allclose(a.local0, p0 - coms[a.body0]) and np.allclose(a.local1, p1 - coms[a.body1])
        assert np.allclose(a.tpose, a.weight0 * p0 + (1 - a.weight0) * p1)
    assert np.allclose(out[0].tpose, q[0] + shift) and np.allclose(out[2].tpose, q[2] + (1 - w2) * shift)
    assert cm.polyline_length(np.array([a.tpose for a in out])) > 0


BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
  JOINT hip
  {
    OFFSET 0.1 -0.05 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    JOINT knee
    {
      OFFSET 0 -0.45 0
      CHANNELS 3 Xrotation Yrotation Zrotation
      JOINT ankle
      {
        OFFSET 0 -0.4 0
        CHANNELS 3 Xrotation Yrotation Zrotation
      }
    }
  }
}
MOTION
Frames: {T}
Frame Time: 0.01
"""


def test_bvh_and_joint_space(tmp_path):
    g = np.random.default_rng(3)
    T = 23
    vals = np.concatenate((g.normal(size=(T, 3)), 40 * g.normal(size=(T, 12))), -1)
    path = tmp_path / "toy.bvh"
    path.write_text(BVH.replace("{T}", str(T)) + "\n".join(" ".join(f"{x:.6f}" for x in row) for row in vals) + "\n")
    bvh = cm.parse_bvh(str(path))
    assert bvh.names == ["Hips", "hip", "knee", "ankle"] and bvh.parents == [-1, 0, 1, 2] and bvh.frame_time == 0.01
    frames = np.arange(0, T, 5)
    G, X = cm.bvh_fk(bvh, frames)
    v = np.round(vals[frames], 6)
    for k, f in enumerate(frames):  # pymotionlib's FK: x_j = x_parent + G_parent offset_j, G_j = G_parent Rx Ry Rz
        Gs = [cm.euler_xyz_deg(v[k, 3:6])]
        Xs = [v[k, :3]]
        for j in (1, 2, 3):
            Xs.append(Xs[-1] + Gs[-1] @ bvh.offsets[j])
            Gs.append(Gs[-1] @ cm.euler_xyz_deg(v[k, 3 + 3 * j : 6 + 3 * j]))
        assert np.abs(G[k] - np.array(Gs)).max() < 1e-12 and np.abs(X[k] - np.array(Xs)).max() < 1e-12
    # the character: hip a ball joint, knee a hinge about x, ankle a ball joint; joint names as the BVH's
    char = toy_character()
    joints = [char.joints[0]._replace(name="hip"), char.joints[1]._replace(name="knee", type="hinge", axis=np.array([1.0, 0, 0]), range=(0.0, 2.3)),
              char.joints[2]._replace(name="ankle")]  # fmt: skip
    char = char._replace(joints=joints)
    idx = cm.bvh_bodies(char, bvh)
    pos0, rot0 = cm.their_placement(char, G, X, idx)
    assert np.allclose(rot0[:, 0], G[:, 0]) and np.allclose(pos0[:, 0], X[:, 0] + G[:, 0] @ (char.bodies[0].com - char.root_info_pos))
    assert np.allclose(pos0[:, 2] + np.einsum("tij,j->ti", G[:, 2], joints[1].anchor - char.bodies[2].com), X[:, 2])  # anchor on the BVH joint
    rot, pos, local, theta, dropped = cm.joint_space(char, G, rot0[:, 0], pos0[:, 0], idx)
    assert np.allclose(rot[:, 1], G[:, 1]) and np.allclose(rot[:, 3], G[:, 3])  # ball joints: the BVH's orientations
    assert np.allclose(rot[:, 2], rot[:, 1] @ cm.axis_angle(np.array([1.0, 0, 0]), theta[2]))  # the hinge: a rotation about its axis
    assert np.allclose(theta[2], cm.hinge_angle(np.swapaxes(G[:, 1], -1, -2) @ G[:, 2], np.array([1.0, 0, 0])))
    for b, j in ((1, 0), (2, 1), (3, 2)):  # every joint anchor coincides on both of its bodies
        a = joints[j].anchor
        p = char.bodies[b].parent
        assert np.allclose(pos[:, p] + rot[:, p] @ (a - char.bodies[p].com), pos[:, b] + rot[:, b] @ (a - char.bodies[b].com))
    assert (dropped[2] > 0.01).any()
    # BVH rotations of the knee about x only: the joint-space clip is the BVH's own pose
    vals[:, 9:12] = vals[:, 9:10] * np.array([1.0, 0.0, 0.0])
    path.write_text(BVH.replace("{T}", str(T)) + "\n".join(" ".join(f"{x:.6f}" for x in row) for row in vals) + "\n")
    G, X = cm.bvh_fk(cm.parse_bvh(str(path)), frames)
    rot, pos, _, theta, dropped = cm.joint_space(char, G, G[:, 0], X[:, 0], idx)
    assert np.abs(rot - G).max() < 1e-9 and dropped[2].max() < 1e-6
    for bad in (BVH.replace("Zrotation\n    JOINT knee", "Zrotation Xposition\n    JOINT knee"), BVH.replace("JOINT ankle", "End Site")):
        path.write_text(bad.replace("{T}", "1") + "0 " * 15 + "\n")
        with pytest.raises(ValueError):
            cm.parse_bvh(str(path))


def test_source_checksums(tmp_path):
    with pytest.raises(FileNotFoundError):
        cm.read_sources(str(tmp_path))
    for rel in cm.SOURCE_SHA256:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("not theirs")
    with pytest.raises(ValueError):
        cm.read_sources(str(tmp_path))


@pytest.mark.skipif(not HAVE_DATA, reason="MuscleVAE's assets are optional data: run tools/download_musclevae.sh, then "
                                          "python tools/convert_musclevae.py")
def test_converted_musclevae():
    """The converted character, muscles and clips: the MJCF's T-pose, masses and qpos layout, the muscle model's T-pose
    lengths (convert_musclevae.check), the clips' bodies against their placement, and the tables' contents."""
    import mujoco

    from ndm_gym.motion.sources import musclevae as src

    rows = cm.check()  # raises with the failures
    assert [(r["clip"], r["frames"]) for r in rows] == [("lafan_walk", 5227), ("lafan_run", 4757), ("lafan_jump", 4889)]  # paper Table 1
    assert all(r["orientation_max_rad"] < 1e-5 and r["root_distance_max_m"] < 1e-6 for r in rows)
    char = mc.load_character()
    assert abs(char["mass"] - mc.MASS) < 1e-6 and [j["type"] for j in char["joints"]].count("ball") == 14 and char["nq"] == 71
    assert all(j["damping"] == mc.JOINT_DAMPING for j in char["joints"])
    m = mujoco.MjModel.from_xml_path(mc.MJCF_PATH)
    assert (m.geom_contype[1:] == 1).all() and (m.geom_conaffinity[1:] == 0).all()  # no self-collision
    assert all(m.jnt_limited[i + 1] == (t == "hinge") for i, t in enumerate(mc.JOINT_TYPES))  # ball joints unlimited
    assert np.allclose(m.dof_damping[6:], mc.JOINT_DAMPING) and np.isclose(m.opt.timestep, 1 / 120) and np.allclose(m.opt.gravity, [0, 0, -9.8])
    with open(mc.MUSCLES_PATH) as f:
        d = json.load(f)
    assert (d["counts"]["muscles"], d["counts"]["anchors"], d["counts"]["blended_anchors"]) == (284, 1212, 438)
    t = mc.load_muscle_table()
    assert np.allclose(t.kp, t.f0 * np.array([mm["kp_ratio"] for mm in d["muscles"]]))
    assert np.allclose(t.f0max, 1.5 * np.array([mm["force_file"] for mm in d["muscles"]]))
    assert set(np.unique(t.part)) == set(range(len(mc.PART_NAMES))) and (t.rest_length > 0.01).all() and (t.rest_length < 1.5).all()
    assert d["models"]["released"]["fatigue_F"] == 0.1 and d["models"]["paper"]["fatigue_F"] == 0.01
    model = mc.muscle_model("paper", dtype=torch.float64)
    assert model.num_anchors == 1212 and torch.allclose(model.kd, 0.1 * model.kp)
    clips = src.load_dir()
    assert [c.name for c in clips] == ["lafan_jump", "lafan_run", "lafan_walk"] and all(c.fps == 20.0 and c.joints.shape[1] == 64 for c in clips)
    for c in clips:
        assert c.meta["source_commit"] == mc.SOURCE_COMMIT
        for a, s in zip(mc.JOINT_QPOS_ADR, mc.JOINT_QPOS_SIZE):  # unit quaternions for the ball joints
            assert s == 1 or (c.joints[:, a : a + 4].norm(dim=-1) - 1).abs().max() < 1e-6
    pos, quat = src.load_placement(os.path.join(mc.CLIPS_DIR, "lafan_walk.npz"))
    assert pos.shape == (5227, 23, 3) and quat.shape == (5227, 23, 4)


def test_ball_euler_angles():
    """The angles of their ball joints' Euler AMotors: local = Rot(z, a2) Rot(y, a1) Rot(x, a0) in their axes (our y, z,
    x), each recovered from the rotation; a rotation about one of their axes moves only its angle."""
    from ndm_gym.core.rotations import axis_angle_to_mat

    g = torch.Generator().manual_seed(0)
    a = 3.0 * torch.rand(500, 3, generator=g, dtype=torch.float64) - 1.5  # the middle angle stays within (-pi/2, pi/2)
    theirs = [torch.tensor(v, dtype=torch.float64).expand(500, 3) for v in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))]
    L = axis_angle_to_mat(theirs[2], a[:, 2]) @ axis_angle_to_mat(theirs[1], a[:, 1]) @ axis_angle_to_mat(theirs[0], a[:, 0])
    torch.testing.assert_close(mc.ball_euler_angles(L), a, rtol=0, atol=1e-12)
    for k in range(3):
        one = torch.zeros(500, 3, dtype=torch.float64)
        one[:, k] = a[:, k]
        torch.testing.assert_close(mc.ball_euler_angles(axis_angle_to_mat(theirs[k], a[:, k])), one, rtol=0, atol=1e-12)
    assert torch.equal(mc.ball_euler_angles(torch.eye(3, dtype=torch.float64)), torch.zeros(3, dtype=torch.float64))
    assert len(mc.BALL_JOINTS) == 14 and mc.JOINT_QVEL_DIM == 50 and mc.BODY_NAMES[mc.HEAD] == "Head"


@pytest.mark.skipif(not HAVE_DATA, reason="MuscleVAE's assets are optional data: run tools/download_musclevae.sh, then "
                                          "python tools/convert_musclevae.py")
def test_forward_kinematics():
    """The character's forward kinematics against MuJoCo's of character.xml on clip frames, and the T-pose; the Euler
    limits in ball-joint order."""
    import mujoco

    from ndm_gym.motion.sources import musclevae as src

    clip = src.load_clip(os.path.join(mc.CLIPS_DIR, "lafan_run.npz"))
    frames = torch.arange(0, clip.root_p.shape[0], 61)
    root_p, root_q, jq = clip.root_p[frames].double(), clip.root_quat[frames].double(), clip.joints[frames].double()
    pos, rot = mc.forward_kinematics(root_p, root_q, jq)
    m = mujoco.MjModel.from_xml_path(mc.MJCF_PATH)
    d = mujoco.MjData(m)
    for i in range(len(frames)):
        d.qpos[:] = torch.cat((root_p[i], root_q[i], jq[i])).numpy()
        mujoco.mj_kinematics(m, d)
        assert np.abs(d.xpos[1:] - pos[i].numpy()).max() < 1e-10 and np.abs(d.xmat[1:].reshape(-1, 3, 3) - rot[i].numpy()).max() < 1e-10
    char = mc.load_character()
    com = torch.tensor([b["com_tpose"] for b in char["bodies"]], dtype=torch.float64)
    ident = torch.cat([torch.tensor([1.0, 0, 0, 0] if t == "ball" else [0.0], dtype=torch.float64) for t in mc.JOINT_TYPES])
    pos, rot = mc.forward_kinematics(com[0], torch.tensor([1.0, 0, 0, 0], dtype=torch.float64), ident)
    assert torch.allclose(pos, com, atol=1e-12) and torch.allclose(rot, torch.eye(3, dtype=torch.float64).expand(23, 3, 3), atol=1e-12)
    lo, hi = mc.ball_euler_limits(char)
    assert lo.shape == hi.shape == (14, 3) and np.allclose(lo, -hi) and np.isclose(hi[0, 0], 2.0) and np.allclose(hi[4], [0.4, 0.2, 0.4])
