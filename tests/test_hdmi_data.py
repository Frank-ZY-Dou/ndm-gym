"""HDMI's human-object interaction clips (tools/convert_hdmi.py): the converter on small synthetic clips, the converted
format through the loader and the motion library (50 Hz frames kept as recorded, other objects, metadata), the contact
flags read one frame ahead as HDMI reads them, and the 13 converted clips in data/hdmi/clips checked with ndm_gym's G1
forward kinematics."""

import json
import os
import re
import sys

import numpy as np
import pytest
import torch

from ndm_gym.motion.library import MotionLib
from ndm_gym.motion.sources import Clip, hoi
from ndm_gym.robots.g1 import G1_JOINT_NAMES, NUM_J

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import convert_hdmi as ch  # noqa: E402

WRIST_PY = [n for n in G1_JOINT_NAMES if re.fullmatch(r".*_wrist_(pitch|yaw)_joint", n)]
FUTURE = np.array([1, 2, 8, 16, 32])  # HDMI's future steps (Y:93)


def _hdmi_clip(T=60, wrists=True, float_contact=False, seed=0):
    """A synthetic HDMI clip with a door (hinge) and one support: the task row, motion.npz arrays and meta.json."""
    g = np.random.default_rng(seed)
    joints = [n for n in G1_JOINT_NAMES if wrists or n not in WRIST_PY][::-1] + ["door_joint"]  # not ndm_gym's order, hinge last
    bodies = list(ch.KEYPOINTS)[::-1] + ["door", "support0"]
    B = len(bodies)
    quat = g.normal(size=(T, B, 4))
    contact = np.zeros((T, 1))
    contact[10:20] = 1
    motion = {"body_pos_w": g.normal(size=(T, B, 3)), "body_quat_w": quat / np.linalg.norm(quat, axis=-1, keepdims=True),
              "body_lin_vel_w": g.normal(size=(T, B, 3)), "body_ang_vel_w": g.normal(size=(T, B, 3)),
              "joint_pos": g.normal(size=(T, len(joints))), "joint_vel": g.normal(size=(T, len(joints))),
              "object_contact": contact if float_contact else contact.astype(bool)}  # fmt: skip
    meta = {"body_names": bodies, "joint_names": joints, "fps": 50.0}
    task = ch.Task("G1Test", "data/motion/test", T, ch.ROBOT_BOX, "door", "door_panel", "door_joint", ("support0",), ch.WRISTS, (((10, 20),),))
    return task, motion, meta


def test_remap_joints():
    names = [n for n in G1_JOINT_NAMES if n not in WRIST_PY][::-1] + ["foldchair_joint"]
    x = np.random.default_rng(1).normal(size=(5, len(names)))
    out, absent = ch.remap_joints(x, names)
    assert out.shape == (5, 29) and absent == WRIST_PY
    for i, n in enumerate(G1_JOINT_NAMES):
        assert np.array_equal(out[:, i], np.zeros(5) if n in WRIST_PY else x[:, names.index(n)]), n


def test_contact_segments():
    flags = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 1]], dtype=bool)
    assert ch.contact_segments(flags) == (((1, 3), (4, 5)), ((0, 2), (4, 5)))
    assert ch.contact_segments(np.ones((4, 1), bool)) == (((0, 4),),)
    assert ch.contact_segments(np.zeros((4, 1), bool)) == ((),)


def test_match_clips(tmp_path):
    for d in ("data/motion/a/b", "data/motion/a/bc", "data/motion/x/b"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "motion.npz").touch()
    assert ch.match_clips(tmp_path, "data/motion/a/b") == [tmp_path / "data/motion/a/b"]  # fullmatch, not a prefix
    assert ch.match_clips(tmp_path, "data/motion/./b") == [tmp_path / "data/motion/a/b", tmp_path / "data/motion/x/b"]  # a regex


@pytest.mark.parametrize("wrists, float_contact", [(True, False), (False, True)])
def test_convert(wrists, float_contact):
    task, motion, meta = _hdmi_clip(wrists=wrists, float_contact=float_contact)
    arrays, side = ch.convert("test", task, motion, meta)
    b, j = meta["body_names"].index, meta["joint_names"]
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    kp = [b(k) for k in ch.KEYPOINTS]
    expected = {"root_pos": f32(motion["body_pos_w"][:, b("pelvis")]), "root_quat_wxyz": f32(motion["body_quat_w"][:, b("pelvis")]),
                "object_pos": f32(motion["body_pos_w"][:, b("door")]), "object_quat_wxyz": f32(motion["body_quat_w"][:, b("door")]),
                "object_joint_pos": f32(motion["joint_pos"][:, [j.index("door_joint")]]),
                "object_joint_vel": f32(motion["joint_vel"][:, [j.index("door_joint")]]),
                "extra_pos": f32(motion["body_pos_w"][:, [b("support0")]]), "extra_quat_wxyz": f32(motion["body_quat_w"][:, [b("support0")]]),
                "keypoint_pos": f32(motion["body_pos_w"][:, kp]), "keypoint_quat_wxyz": f32(motion["body_quat_w"][:, kp]),
                "keypoint_lin_vel": f32(motion["body_lin_vel_w"][:, kp]), "keypoint_ang_vel": f32(motion["body_ang_vel_w"][:, kp]),
                "contact": motion["object_contact"] != 0}  # fmt: skip
    for k, v in expected.items():
        assert arrays[k].dtype == v.dtype and np.array_equal(arrays[k], v), k
    for i, n in enumerate(G1_JOINT_NAMES):  # by name; joints the clip lacks are 0, as HDMI fills them
        for key in ("joint_pos", "joint_vel"):
            want = np.zeros(task.frames, np.float32) if n not in j else f32(motion[key][:, j.index(n)])
            assert np.array_equal(arrays[key][:, i], want), (key, n)
    assert sorted(arrays) == sorted([*expected, "joint_pos", "joint_vel"])
    assert side["absent_joints"] == ([] if wrists else WRIST_PY) and (side["absent_joints_note"] is None) == wrists
    assert side["contact_segments"] == [[[10, 20]]] and side["contact_lead_frames"] == 1 and side["contact_columns"] == 1
    assert side["contact_source_dtype"] == ("float64" if float_contact else "bool")
    assert (side["object_name"], side["object_body_name"], side["object_joint_name"]) == ("door", "door_panel", "door_joint")
    assert side["extra_objects"] == ["support0"] and side["contact_eef_bodies"] == list(ch.WRISTS) and side["keypoint_names"] == list(ch.KEYPOINTS)


def test_convert_refuses_mismatches():
    task, motion, meta = _hdmi_clip()
    for bad_task in (task._replace(frames=61), task._replace(segments=(((10, 21),),)), task._replace(object_joint=None),
                     task._replace(object_name="box"), task._replace(extras=("support1",))):  # fmt: skip
        with pytest.raises(ValueError):
            ch.convert("test", bad_task, motion, meta)
    with pytest.raises(ValueError):
        ch.convert("test", task, motion, {**meta, "fps": 30.0})
    with pytest.raises(ValueError):
        ch.convert("test", task, {**motion, "object_contact": np.zeros((60, 3), bool)}, meta)


def _write(tmp_path, name="test", **kw):
    task, motion, meta = _hdmi_clip(**kw)
    arrays, side = ch.convert(name, task, motion, meta)
    ch.write_clip(tmp_path, name, arrays, side)
    return arrays, side


def test_converted_clip_in_motion_lib(tmp_path, kin):
    """At 50 Hz every frame is kept as recorded, also where the frame count of resampling would drop the last one (928)."""
    arrays, _ = _write(tmp_path, T=928)
    clip = hoi.load_clip(str(tmp_path / "test.npz"))
    assert clip.extra_names == ("support0",) and clip.meta["task"] == "test" and clip.fps == 50.0
    m = MotionLib([clip], kin, "cpu")
    assert m.num_frames == 928 and m.extra_names == ("support0",) and m.meta[0]["contact_lead_frames"] == 1
    t = torch.from_numpy
    for got, key in ((m.root_p, "root_pos"), (m.q, "joint_pos"), (m.obj_p, "object_pos"), (m.obj_joint, "object_joint_pos"),
                     (m.extra_p, "extra_pos")):  # fmt: skip
        assert torch.equal(got, t(arrays[key])), key
    assert torch.equal(m.contact, t(arrays["contact"]).float())
    p, R = m.ref_extras(torch.tensor([0, 927]))
    assert p.shape == (2, 1, 3) and R.shape == (2, 1, 3, 3) and torch.equal(p, t(arrays["extra_pos"])[[0, 927]])
    m30 = MotionLib([clip._replace(fps=30.0, src_vel=None)], kin, "cpu")  # another rate is resampled, the other objects with the rest
    assert m30.num_frames == int(np.floor(927 / 30.0 / 0.02)) + 1 and m30.extra_p.shape == (m30.num_frames, 1, 3)
    assert torch.equal(m30.ref_extras(torch.tensor([0]))[0], p[:1])


def test_robot_only_50hz_clip_kept(kin):
    T = 928
    g = torch.Generator().manual_seed(0)
    quat = torch.nn.functional.normalize(torch.randn(T, 4, generator=g), dim=-1)
    clip = Clip("walk", torch.randn(T, 3, generator=g), quat, 0.1 * torch.randn(T, 29, generator=g), 50.0)
    m = MotionLib([clip], kin, "cpu")
    assert m.num_frames == T and torch.equal(m.root_p, clip.root_p) and torch.equal(m.q, clip.joints) and m.extra_names is None
    with pytest.raises(ValueError):
        m.ref_extras(torch.tensor([0]))


def test_contact_read_one_frame_ahead(tmp_path, kin):
    """HDMI's command update at counter t loads the pose references of frames t + F and, after t += 1, the flags of rows
    t + 1 + F, all clamped to the last frame (CMD:247, 290, 565-568). From the motion library: with pose frame f, the flags
    of row min(f + contact_lead_frames, last frame)."""
    task, motion, meta = _hdmi_clip(T=50)
    motion["object_contact"] = np.random.default_rng(3).random((50, 2)) < 0.5
    task = task._replace(eefs=("right_wrist_yaw_link", "left_ankle_roll_link"), segments=ch.contact_segments(motion["object_contact"]))
    arrays, side = ch.convert("test", task, motion, meta)
    ch.write_clip(tmp_path, "test", arrays, side)
    m = MotionLib([hoi.load_clip(str(tmp_path / "test.npz"))], kin, "cpu")
    end, lead = int(m.clip_end[0]), m.meta[0]["contact_lead_frames"]
    for t in range(55):
        poses = np.minimum(t + FUTURE, 49)  # HDMI's pose reference frames
        flags = motion["object_contact"][np.minimum(t + 1 + FUTURE, 49)]  # and its flags
        assert np.array_equal(m.contact[torch.clamp(torch.from_numpy(poses) + lead, max=end)].numpy() > 0.5, flags), t


def test_libraries_and_files_refused(tmp_path, kin):
    _write(tmp_path)
    clip = hoi.load_clip(str(tmp_path / "test.npz"))
    for bad in ([clip, clip._replace(contact=torch.zeros(60, 2))], [clip, clip._replace(extra_names=("support1",))],
                [clip._replace(obj_name=None, obj_p=None, obj_quat=None, obj_joint=None, contact=None)]):  # fmt: skip
        with pytest.raises(ValueError):
            MotionLib(bad, kin, "cpu")
    with np.load(tmp_path / "test.npz") as z:
        good = {k: z[k] for k in z.files}
    side = json.loads((tmp_path / "test.json").read_text())
    for arrays, meta in (({**good, "body_pos_w": good["root_pos"]}, side), (good, {**side, "extra_objects": []}),
                         (good, {k: v for k, v in side.items() if k != "keypoint_names"}),
                         ({k: v for k, v in good.items() if k != "extra_quat_wxyz"}, side)):  # fmt: skip
        np.savez(tmp_path / "bad.npz", **arrays)
        (tmp_path / "bad.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError):
            hoi.load_clip(str(tmp_path / "bad.npz"))


def test_check_configs(tmp_path):
    """The task table against HDMI's task configs, here written from the table itself and then changed."""
    import yaml

    cfg = tmp_path / "cfg" / "task"
    (cfg / "G1" / "hdmi").mkdir(parents=True)
    (cfg / "base").mkdir()
    # Y:52-61
    pats = [".*_hip_(pitch|yaw)_link", ".*_knee_link", ".*_ankle_roll_link", "pelvis", "torso_link", ".*_shoulder_pitch_link", ".*_elbow_link",
            ".*_wrist_yaw_link"]  # fmt: skip
    base = {"command": {"tracking_keypoint_names": pats, "contact_eef_body_name": list(ch.WRISTS), "extra_object_names": []}}
    (cfg / "base" / "hdmi-base.yaml").write_text(yaml.safe_dump(base))

    def write(tasks):
        for name, t in tasks.items():
            cmd = {"data_path": t.data_path, "object_asset_name": t.object_name, "object_body_name": t.object_body}
            cmd.update({"object_joint_name": t.object_joint} if t.object_joint else {})
            cmd.update({"extra_object_names": list(t.extras)} if t.extras else {})
            cmd.update({"contact_eef_body_name": list(t.eefs)} if t.eefs != ch.WRISTS else {})
            doc = {"defaults": ["base/hdmi-base", "_self_"], "name": t.hdmi_name, "robot": {"name": "g1", "robot_type": t.robot}, "command": cmd}
            (cfg / "G1" / "hdmi" / f"{name}.yaml").write_text(yaml.safe_dump(doc))

    links = list(ch.KEYPOINTS) + ["left_hip_roll_link", "left_wrist_pitch_link", "waist_yaw_link"]
    write(ch.TASKS)
    ch.check_configs(tmp_path, links)
    write({"traverse_door": ch.TASKS["traverse_door"]._replace(eefs=ch.WRISTS)})
    with pytest.raises(ValueError, match="traverse_door.eefs"):
        ch.check_configs(tmp_path, links)
    write(ch.TASKS)
    with pytest.raises(ValueError, match="KEYPOINTS"):
        ch.check_configs(tmp_path, links + ["middle_knee_link"])


@pytest.mark.skipif(not os.path.isdir(os.path.join(ch.HDMI_DIR, "clips")),
                    reason="HDMI's clips are optional data: run tools/download_hdmi.sh, then python tools/convert_hdmi.py")  # fmt: skip
def test_converted_hdmi_clips():
    """The 13 converted clips: ndm_gym's G1 FK of each clip's root and joints reproduces HDMI's stored keypoints, the
    motion library keeps every frame as converted, the contact segments and the sidecars are the table's (spec §8.1)."""
    clips_dir = os.path.join(ch.HDMI_DIR, "clips")
    rows = ch.check(clips_dir)  # raises with the failures
    assert [r["task"] for r in rows] == list(ch.TASKS)
    for name, t in ch.TASKS.items():
        with open(os.path.join(clips_dir, f"{name}.json")) as f:
            side = json.load(f)
        assert (side["hdmi_task_name"], side["data_path"], side["frames"], side["robot_type"]) == (t.hdmi_name, t.data_path, t.frames, t.robot)
        assert (side["object_name"], side["object_body_name"], side["object_joint_name"]) == (t.object_name, t.object_body, t.object_joint)
        assert side["extra_objects"] == list(t.extras) and side["contact_eef_bodies"] == list(t.eefs) and side["source_commit"] == ch.COMMIT
        assert side["contact_segments"] == [[list(s) for s in col] for col in t.segments] and side["contact_lead_frames"] == 1
        wristless = name in ("open_foldchair-sit", "push_box", "roll_ball-hand")
        assert side["absent_joints"] == (WRIST_PY if wristless else []), name
        with np.load(os.path.join(clips_dir, f"{name}.npz")) as z:
            if wristless:
                cols = [G1_JOINT_NAMES.index(n) for n in WRIST_PY]
                assert not z["joint_pos"][:, cols].any() and not z["joint_vel"][:, cols].any(), name


def test_source_velocities(kin):
    """A clip's own reference velocities (HDMI's) are kept frame by frame, for every clip or none, and only at 50 Hz."""
    T, b = 6, 2
    base = Clip("a", torch.zeros(T, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4), torch.zeros(T, NUM_J), 50.0)
    vel = {"keypoint_names": ["pelvis", "torso_link"], "keypoint_lin_vel": torch.rand(T, b, 3), "keypoint_ang_vel": torch.rand(T, b, 3),
           "joint_vel": torch.rand(T, NUM_J)}  # fmt: skip
    m = MotionLib([base._replace(src_vel=vel), base._replace(name="b", src_vel=vel)], kin, "cpu")
    assert m.src_vel["keypoint_names"] == ["pelvis", "torso_link"] and m.src_vel["joint_vel"].shape == (2 * T, NUM_J)
    torch.testing.assert_close(m.src_vel["keypoint_lin_vel"][T:], vel["keypoint_lin_vel"], atol=0, rtol=0)
    assert MotionLib([base], kin, "cpu").src_vel is None
    with pytest.raises(ValueError, match="some clips"):
        MotionLib([base._replace(src_vel=vel), base._replace(name="b")], kin, "cpu")
    with pytest.raises(ValueError, match="50 Hz"):
        MotionLib([base._replace(src_vel=vel, fps=30.0)], kin, "cpu")
