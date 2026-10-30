"""Motion library with an object track: resampling, the object's reference and goal, the
converted-format loader, and robot-only clips left exactly as they were."""

import json

import numpy as np
import pytest
import torch

from ndm_gym.core.rotations import quat_to_mat
from ndm_gym.motion.library import MotionLib
from ndm_gym.motion.sources import Clip
from ndm_gym.motion.sources import hoi
from ndm_gym.robots.g1 import NUM_J
from ndm_gym.state import CONTROL_DT

FPS = 30.0


def _yaw_quat(theta):
    return torch.stack((torch.cos(theta / 2), torch.zeros_like(theta), torch.zeros_like(theta), torch.sin(theta / 2)), -1)


def _clip(name, T=61, obj=True, joint=False):
    t = torch.arange(T, dtype=torch.float32) / FPS
    root_p = torch.stack((0.3 * t, torch.zeros_like(t), torch.full_like(t, 0.78)), -1)
    root_quat = _yaw_quat(0.2 * t)
    joints = 0.1 * torch.sin(t)[:, None].expand(T, NUM_J).clone()
    if not obj:
        return Clip(name, root_p, root_quat, joints, FPS)
    obj_p = root_p + torch.tensor([0.4, 0.0, 0.1])  # carried ahead of the root
    obj_quat = _yaw_quat(0.5 * t)
    return Clip(name, root_p, root_quat, joints, FPS, obj_name="box", obj_p=obj_p, obj_quat=obj_quat,
                obj_joint=(0.3 * t)[:, None] if joint else None)  # fmt: skip


def test_robot_only_library_unchanged(kin):
    """A library without objects has exactly the robot arrays it had before the object track existed."""
    clips = [_clip("a", obj=False), _clip("b", T=45, obj=False)]
    m = MotionLib(clips, kin, "cpu")
    assert not m.has_object and m.obj_name is None
    with pytest.raises(ValueError):
        m.ref_object(torch.tensor([0]))
    # the robot part is identical to a library built from the same clips with objects attached
    m_obj = MotionLib([_clip("a"), _clip("b", T=45)], kin, "cpu")
    for k in ("root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R", "body_v", "body_w"):
        assert torch.equal(getattr(m, k), getattr(m_obj, k)), k
    f = torch.tensor([0, 10, 40])
    assert torch.equal(m.goal(f, 4), m_obj.goal(f, 4))


def test_object_track_resampled(kin):
    m = MotionLib([_clip("a")], kin, "cpu")
    n = m.num_frames
    assert m.has_object and m.obj_name == "box" and m.obj_p.shape == (n, 3) and m.obj_R.shape == (n, 3, 3)
    t = torch.arange(n, dtype=torch.float32) * CONTROL_DT
    # linear object motion: constant velocity, yaw rate 0.5 rad/s
    torch.testing.assert_close(m.obj_p[:, 0], 0.4 + 0.3 * t, atol=1e-5, rtol=0)
    torch.testing.assert_close(m.obj_v, torch.tensor([0.3, 0.0, 0.0]).expand(n, 3), atol=1e-4, rtol=0)
    torch.testing.assert_close(m.obj_w, torch.tensor([0.0, 0.0, 0.5]).expand(n, 3), atol=1e-3, rtol=0)
    torch.testing.assert_close(m.obj_R, quat_to_mat(_yaw_quat(0.5 * t)), atol=1e-5, rtol=0)
    po, Ro, vo, wo = m.ref_object(torch.tensor([3]))
    assert torch.equal(po, m.obj_p[[3]]) and torch.equal(Ro, m.obj_R[[3]])


def test_object_goal(kin):
    m = MotionLib([_clip("a", joint=True)], kin, "cpu")
    f = torch.tensor([0, m.num_frames - 1])
    g = m.object_goal(f, 3).reshape(2, 3, -1)
    assert g.shape[-1] == 3 + 6 + 3 + 3 + 1
    # beyond the clip end: last pose held, zero velocity
    assert torch.all(g[1, :, 9:15] == 0)
    # relative position in the root heading frame: at frame 0 the root's yaw is 0, so the next frame's object sits
    # 0.4 m ahead of the root at frame 0 plus one step of the 0.3 m/s motion, 0.1 m up
    torch.testing.assert_close(g[0, 0, :3], torch.tensor([0.4 + 0.3 * CONTROL_DT, 0.0, 0.1]), atol=1e-4, rtol=0)


def test_mixed_and_inconsistent_clips_refused(kin):
    with pytest.raises(ValueError):
        MotionLib([_clip("a"), _clip("b", obj=False)], kin, "cpu")
    other = _clip("b")._replace(obj_name="suitcase")
    with pytest.raises(ValueError):
        MotionLib([_clip("a"), other], kin, "cpu")
    with pytest.raises(ValueError):
        MotionLib([_clip("a", joint=True), _clip("b")], kin, "cpu")


def test_loader_round_trip(tmp_path):
    c = _clip("sub1_box_001", joint=True)
    np.savez(tmp_path / "sub1_box_001.npz", root_pos=c.root_p.numpy(), root_quat_wxyz=c.root_quat.numpy(), joint_pos=c.joints.numpy(),
             object_pos=c.obj_p.numpy(), object_quat_wxyz=c.obj_quat.numpy(), object_joint_pos=c.obj_joint.numpy(), fps=np.float32(FPS))  # fmt: skip
    (tmp_path / "sub1_box_001.json").write_text(json.dumps({"object_name": "box", "source": "test"}))
    (clip,) = hoi.load_dir(str(tmp_path))
    assert clip.obj_name == "box" and clip.fps == FPS
    for a, b in ((clip.root_p, c.root_p), (clip.obj_quat, c.obj_quat), (clip.obj_joint, c.obj_joint)):
        torch.testing.assert_close(a, b)
    with pytest.raises(FileNotFoundError):
        hoi.load_dir(str(tmp_path), object_name="suitcase")
    bad = tmp_path / "bad"
    bad.mkdir()
    np.savez(bad / "x.npz", root_pos=c.root_p.numpy(), root_quat_wxyz=c.root_quat.numpy(), joint_pos=c.joints[:, :23].numpy(),
             object_pos=c.obj_p.numpy(), object_quat_wxyz=c.obj_quat.numpy(), fps=np.float32(FPS))  # fmt: skip
    (bad / "x.json").write_text(json.dumps({"object_name": "box"}))
    with pytest.raises(ValueError):
        hoi.load_dir(str(bad))
