"""Render HDMI's human-object interaction scenes (docs/hdmi_task_spec.md §8-§10) with MuJoCo (EGL), in the look of HDMI's
simulation videos: a task's G1 variant with HDMI's own visual meshes, the task's object and the scene's other objects, on
HDMI's blue floor with a white 0.5 m grid under a pale sky, a following camera, and the small NDM Gym mark of tools/render.py.

The look follows HDMI's project page (https://hdmi-humanoid.github.io: a simulation video per task, rendered by Isaac Sim at
1280 x 720) and paper figures (arXiv:2509.16757, Fig. 1d, 3, 7, 9), measured on frames of those videos:
* floor: steel blue, lit (70, 118, 153), white grid lines about 1.5 cm wide every 0.5 m (a 2 m platform spans four squares),
  fading into a haze toward the horizon;
* sky: pale blue, (188, 191, 197) at the horizon to about (140, 168, 198) higher up; their clouds are left out;
* robot: white, as the materials of HDMI's G1 USDs (OmniPBR, diffuse 1.0; the chest logo 0.5);
* objects: colored after the Omniverse materials their USDs bind (cardboard, oak, beadboard, birch, plywood, parquet, bamboo,
  rubber, aluminum, gold, brass), as those look in the videos; unbound prims in Isaac Sim's default light grey. One color
  stands for each material's texture (MATERIALS), but the ball's bamboo planks are drawn as planks (plank_png, on the
  mesh's own texture coordinates) so that its rolling shows. The videos' yellow dot at the env origin, a debug marker, is
  not drawn.
MuJoCo stands in for their path tracer: a warm key light whose shadows are softened by averaging renders lit from points of a
2 degree disk, a cool fill and a rim light, a matte floor, 16x multisampling and an 8192 shadow map; previews downscale the
1920 x 1080 frames.

Geometry:
* robot: ndm_gym's G1 description (data/g1_description/g1_29dof.xml) for the kinematics, with the visual meshes of the
  task's HDMI G1 USD (HDMI's variants share the kinematics and differ in collision shapes; push_box's has no hands but L
  plates). Every USD link is checked to sit where the MJCF's body does at the zero pose.
* objects and props: the rigid bodies of HDMI's object USDs (§9) with their visible prims (meshes, cubes, spheres) in the
  body's frame at the USD's authored scale; prims of purpose guide (collision shapes, the door frame's second post) are not
  drawn, as in HDMI's videos. A door's panel and a folding chair's seat and backrest follow from the fixed base's pose and
  the hinge angle through the USD's joint frames. Props (supports, platforms, a wall, the foam's support) at the clip's
  poses (§8.4). Spawned unscaled, as HDMISim spawns them (its per-env object scale is pending).
* conversion: read with pxr once (Isaac Lab's environment) and kept as OBJ files in .cache/render_hoi/<usd>-<sha256>/.

Inputs, one of:
* --task NAME: the task's converted reference clip (data/hdmi/clips/NAME.npz, tools/convert_hdmi.py);
* --states FILE.npz: states recorded from ndm_gym.sim.rigid.hdmi_sim.HDMISim (StateRecorder below), one environment, per
  control step, with the converted clips' keys:
    task              ()         str, one of ndm_gym.tasks.hdmi_tasks.TASKS
    fps               ()         steps per second (50)
    root_pos          (T, 3)     the robot's root link (pelvis), m, relative to the env origin
    root_quat_wxyz    (T, 4)
    joint_pos         (T, 29)    rad, ndm_gym's G1 (Unitree) joint order
    object_pos        (T, 3)     the object's root link: the object, or a door's or folding chair's fixed base
    object_quat_wxyz  (T, 4)
    object_joint_pos  (T, 1)     rad, the hinge angle; only for a door or the folding chair
    extra_pos         (T, n, 3)  the task's props (its extra_objects, in that order); only when it has props
    extra_quat_wxyz   (T, n, 4)
    frame             (T,)       int, HDMISim's reference frame counter t after the step
Camera: a fixed heading per video, 20 degrees down, following the robot and the object's contact body so that both fit with
a margin: it holds a view while they stay framed without zooming out far, then glides to the next (camera_path). The default
heading is the best of every 10 degrees over the video's steps: the object hides little of the robot (a door or a pushed box
would from the far side), the robot little of the object, the robot faces the camera, and the two fit at a short distance.

EGL opens its context on MUJOCO_EGL_DEVICE_ID, or on the first GPU: it ignores CUDA_VISIBLE_DEVICES. When CUDA_VISIBLE_DEVICES
names GPUs, this tool requires MUJOCO_EGL_DEVICE_ID to be the EGL device of one of them (tools/render.py).

Usage: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 python tools/render_hoi.py out.mp4 --task move_suitcase --start 2
       ... python tools/render_hoi.py frame.png --states rollout.npz --start 1.5
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from convert_hdmi import OBJECT_USD  # tools/convert_hdmi.py: each object asset's USD, under data/hdmi
from ndm_gym.core.rotations import mat_to_quat, quat_slerp, quat_to_mat
from ndm_gym.motion.sources import hoi
from ndm_gym.robots.g1 import G1_JOINT_NAMES, MJCF_PATH
from ndm_gym.state import CONTROL_DT
from ndm_gym.tasks import hdmi_tasks

HDMI_DIR = os.path.join(ROOT, "data", "hdmi")
CLIPS_DIR = os.path.join(HDMI_DIR, "clips")
CACHE_DIR = os.path.join(ROOT, ".cache", "render_hoi")
CACHE_VERSION = 2

# the look (sRGB 0-255 where not stated)
FLOOR_RGB, LINE_RGB = (63, 106, 138), (236, 239, 242)  # texture colors, lit to about HDMI's (70, 118, 153) and white
GRID, GRID_LINE, GRID_PX = 0.5, 0.015, 1024  # m per square, m per line, texels per square
SKY = ((0.0, (188, 191, 197)), (1.5, (184, 190, 199)), (4.0, (176, 188, 202)), (8.0, (165, 183, 203)), (14.0, (154, 177, 203)),
       (22.0, (146, 171, 201)), (35.0, (139, 166, 198)))  # (elevation in degrees, color), constant beyond
SKY_SIZE = 1024
# materials: (sRGB base color, specular, shininess). HDMI's G1 materials are OmniPBR with a diffuse constant, drawn at
# ROBOT_ALBEDO times it; the objects' Omniverse materials by name, matched to how they look in HDMI's videos
ROBOT_ALBEDO, ROBOT_FINISH = 0.94, (0.35, 0.5)
MATERIALS = {
    "Cardboard": ((200, 164, 118), 0.08, 0.1), "Oak": ((126, 90, 58), 0.15, 0.3), "Beadboard": ((222, 224, 226), 0.12, 0.3),
    "Birch": ((228, 214, 190), 0.1, 0.2), "Plywood": ((232, 208, 158), 0.08, 0.15), "Parquet_Floor": ((206, 156, 102), 0.18, 0.35),
    "Bamboo_Planks": ((196, 156, 78), 0.3, 0.45), "Rubber_Textured": ((30, 31, 34), 0.15, 0.25), "Aluminum_Cast": ((142, 152, 168), 0.5, 0.5),
    "Aluminum_Polished": ((190, 196, 204), 0.7, 0.7), "Gold": ((206, 166, 82), 0.6, 0.6), "Brass": ((190, 160, 96), 0.55, 0.55),
    "default": ((190, 192, 196), 0.1, 0.2),  # an unbound prim: Isaac Sim's default material
}  # fmt: skip
# materials drawn with a texture on their meshes' texture coordinates (plank_png): HDMI's ball shows its bamboo planks
# roll, which one color would hide
PLANKS = {"Bamboo_Planks": 12}  # planks around the ball
# lights: (diffuse, specular) linear; heading relative to the camera's (degrees) and elevation. The key light is a disk of
# SOFT_ANGLE degrees: each frame averages SOFT_SAMPLES renders lit from points of it (a Vogel spiral), for soft shadows
HEADLIGHT_AMBIENT, HEADLIGHT_DIFFUSE = (0.37, 0.38, 0.41), (0.10, 0.10, 0.10)
KEY = ((0.74, 0.71, 0.65), (0.30, 0.30, 0.28), 45.0, 42.0)  # warm, from the camera's front right; casts the shadows
FILL = ((0.19, 0.21, 0.26), (0.0, 0.0, 0.0), -60.0, 25.0)  # cool, from the front left
RIM = ((0.22, 0.22, 0.24), (0.45, 0.45, 0.48), -135.0, 32.0)  # from behind, on the left
SOFT_SAMPLES, SOFT_ANGLE = 12, 2.0
MARK_RGBA = (255, 255, 255, 150)  # tools/render.py's NDM Gym mark, in translucent white on HDMI's dark floor
FOVY, ELEVATION, MARGIN, MIN_DISTANCE = 38.0, -20.0, 0.72, 2.4  # degrees, degrees, fraction of the half frame, m
FOG = (3.0, 20.0)  # where the fog starts and ends, in model extents (2 m)
# the default heading: the best mean score over the frames, -occlusion + FRONT x facing - SPREAD x distance / MIN_DISTANCE
FRONT, SPREAD = 0.45, 0.1
# the camera holds a view as long as it frames the steps so far within ZOOM_LIMIT times the distance they need one by one,
# then glides to the next hold in about GLIDE s; ZOOM_HOLD s: the window of the running max of the steps' distances
ZOOM_LIMIT, GLIDE, ZOOM_HOLD = 1.35, 1.2, 0.5
MOVING = 0.05  # m/s: the object moves when a corner of its contact body is this fast


class Motion(NamedTuple):
    """Poses at ``fps`` per step (clip frame or control step): the robot's root (wxyz) and joints (G1 order), the object's
    root link, a door's or folding chair's hinge angle, the props; a clip's contact flags; the reference frames."""

    task: str
    fps: float
    root_pos: np.ndarray  # (T, 3)
    root_quat: np.ndarray  # (T, 4)
    joint_pos: np.ndarray  # (T, 29)
    object_pos: np.ndarray  # (T, 3)
    object_quat: np.ndarray  # (T, 4)
    hinge: np.ndarray | None  # (T,)
    extra_pos: np.ndarray | None  # (T, n, 3)
    extra_quat: np.ndarray | None  # (T, n, 4)
    contact: np.ndarray | None  # (T, c) bool
    frame: np.ndarray  # (T,) int


class Scene(NamedTuple):
    robot_usd: str
    object_usd: str
    object_body: str  # the body that carries the contact targets: framed with the robot
    hinge: str | None  # the object's hinge joint
    props: tuple[tuple[str, str], ...]  # (name, usd)


def scene_of(task: str) -> Scene:
    """The USDs of a task's scene (its config, ndm_gym.tasks.hdmi_tasks; the files of tools/download_hdmi.sh)."""
    cfg = hdmi_tasks.get(task)
    usd = lambda rel: os.path.join(HDMI_DIR, rel)  # noqa: E731
    return Scene(usd(f"active_adaptation/assets/g1/{cfg.robot_type}.usd"), usd(OBJECT_USD[cfg.object_asset]), cfg.object_body,
                 cfg.object_joint, tuple((n, usd(OBJECT_USD[n])) for n in cfg.extra_objects))  # fmt: skip


# ---------------------------------------------------------------------------------------------------- inputs
def clip_motion(task: str, clips_dir: str = CLIPS_DIR) -> Motion:
    """A task's converted reference clip (ndm_gym.motion.sources.hoi)."""
    path = os.path.join(clips_dir, task + ".npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: run tools/download_hdmi.sh, then python tools/convert_hdmi.py")
    c = hoi.load_clip(path)
    cfg = hdmi_tasks.get(task)
    if c.meta.get("task") != task or c.obj_name != cfg.object_asset or tuple(c.extra_names or ()) != cfg.extra_objects:
        raise ValueError(f"{path} is not the clip of {task} (object {cfg.object_asset}, props {cfg.extra_objects})")
    if (c.obj_joint is None) != (cfg.object_joint is None):
        raise ValueError(f"{path}: a hinge angle goes with a hinged object ({cfg.object_joint})")
    f64 = lambda x: None if x is None else x.double().numpy()  # noqa: E731
    return Motion(task, float(c.fps), f64(c.root_p), f64(c.root_quat), f64(c.joints), f64(c.obj_p), f64(c.obj_quat),
                  None if c.obj_joint is None else f64(c.obj_joint)[:, 0], f64(c.extra_p), f64(c.extra_quat),
                  None if c.contact is None else c.contact.numpy() > 0.5, np.arange(len(c.root_p)))  # fmt: skip


STATE_KEYS = ("task", "fps", "root_pos", "root_quat_wxyz", "joint_pos", "object_pos", "object_quat_wxyz", "frame")
STATE_OPTIONAL = ("object_joint_pos", "extra_pos", "extra_quat_wxyz")


def save_states(path: str, m: Motion):
    """Write a Motion in the --states format (the module docstring)."""
    out = {"task": np.array(m.task), "fps": np.array(m.fps), "root_pos": m.root_pos, "root_quat_wxyz": m.root_quat, "joint_pos": m.joint_pos,
           "object_pos": m.object_pos, "object_quat_wxyz": m.object_quat, "frame": np.asarray(m.frame, dtype=np.int64)}  # fmt: skip
    if m.hinge is not None:
        out["object_joint_pos"] = m.hinge[:, None]
    if m.extra_pos is not None:
        out.update(extra_pos=m.extra_pos, extra_quat_wxyz=m.extra_quat)
    np.savez(path, **out)


def load_states(path: str) -> Motion:
    """Recorded states (the --states format), checked against the task's scene."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in STATE_KEYS if k not in z.files]
        unknown = sorted(set(z.files) - set(STATE_KEYS) - set(STATE_OPTIONAL))
        if missing or unknown:
            raise ValueError(f"{path}: missing arrays {missing}, unknown arrays {unknown}")
        a = {k: z[k] for k in z.files}
    task, fps = str(a["task"]), float(a["fps"])
    cfg = hdmi_tasks.get(task)
    T, n = len(a["root_pos"]), len(cfg.extra_objects)
    shapes = {"root_pos": (T, 3), "root_quat_wxyz": (T, 4), "joint_pos": (T, len(G1_JOINT_NAMES)), "object_pos": (T, 3),
              "object_quat_wxyz": (T, 4), "frame": (T,)}  # fmt: skip
    if cfg.object_joint is not None:
        shapes["object_joint_pos"] = (T, 1)
    if n:
        shapes.update(extra_pos=(T, n, 3), extra_quat_wxyz=(T, n, 4))
    extra = sorted(set(a) - set(shapes) - {"task", "fps"})
    if extra:
        raise ValueError(f"{path}: {extra} do not belong to {task} (hinge {cfg.object_joint}, props {cfg.extra_objects})")
    for k, shape in shapes.items():
        if k not in a or a[k].shape != shape or not np.isfinite(a[k]).all():
            raise ValueError(f"{path}: {k} must be finite with shape {shape}, got {None if k not in a else a[k].shape}")
    if T == 0 or not fps > 0:
        raise ValueError(f"{path}: no steps, or fps {fps} not positive")
    for k in ("root_quat_wxyz", "object_quat_wxyz", "extra_quat_wxyz"):
        if k in a and np.abs(np.linalg.norm(a[k], axis=-1) - 1.0).max() > 1e-3:
            raise ValueError(f"{path}: {k} must hold unit quaternions (wxyz)")
    f64 = lambda k: a[k].astype(np.float64) if k in a else None  # noqa: E731
    hinge = f64("object_joint_pos")
    return Motion(task, fps, f64("root_pos"), f64("root_quat_wxyz"), f64("joint_pos"), f64("object_pos"), f64("object_quat_wxyz"),
                  None if hinge is None else hinge[:, 0], f64("extra_pos"), f64("extra_quat_wxyz"), None, a["frame"].astype(np.int64))  # fmt: skip


class StateRecorder:
    """Records environment ``env`` of a running HDMISim (ndm_gym.sim.rigid.hdmi_sim) in the --states format: add() after a
    reset (and refresh()) and after every step, then save(path). It only reads the simulator's buffers."""

    def __init__(self, sim, env: int = 0):
        self.sim, self.env, self.rows = sim, int(env), []

    def _pose(self, pose_w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """An Isaac Lab 3 pose (x, y, z, qx, qy, qz, qw) of this env: position relative to the env origin, quaternion wxyz."""
        p = (pose_w[self.env].double() - torch.cat((self.sim.scene.env_origins[self.env].double(), pose_w.new_zeros(4).double()))).cpu().numpy()
        return p[:3], p[[6, 3, 4, 5]]

    def add(self):
        s = self.sim
        root, obj = self._pose(s.robot.data.root_link_pose_w.torch), self._pose(s.obj.data.root_link_pose_w.torch)
        props = [self._pose(e.data.root_link_pose_w.torch) for e in s.extras]
        row = {"root": root, "q": s.robot.data.joint_pos.torch[self.env, s.j_g1].double().cpu().numpy(), "obj": obj, "props": props,
               "frame": int(s.t[self.env])}  # fmt: skip
        if s.hinged:
            row["hinge"] = float(s.obj.data.joint_pos.torch[self.env, 0])
        self.rows.append(row)

    def motion(self) -> Motion:
        if not self.rows:
            raise ValueError("no states recorded")
        r, st = self.rows, lambda f: np.stack([f(x) for x in self.rows])  # noqa: E731
        props = bool(r[0]["props"])
        return Motion(self.sim.cfg.name, 1.0 / CONTROL_DT, st(lambda x: x["root"][0]), st(lambda x: x["root"][1]), st(lambda x: x["q"]),
                      st(lambda x: x["obj"][0]), st(lambda x: x["obj"][1]), st(lambda x: x["hinge"]) if "hinge" in r[0] else None,
                      st(lambda x: [p for p, _ in x["props"]]) if props else None, st(lambda x: [q for _, q in x["props"]]) if props else None,
                      None, st(lambda x: x["frame"]))  # fmt: skip

    def save(self, path: str):
        save_states(path, self.motion())


def resample(m: Motion, times: np.ndarray, fps: float) -> Motion:
    """The motion at ``times`` (s from its start, within it; ``fps`` their rate): positions, joints and the hinge linear,
    rotations slerped."""
    s = np.clip(np.asarray(times, dtype=np.float64) * m.fps, 0.0, len(m.root_pos) - 1)
    i0 = np.floor(s).astype(np.int64)
    i1, w = np.minimum(i0 + 1, len(m.root_pos) - 1), s - i0
    lerp = lambda x: None if x is None else x[i0] + (x[i1] - x[i0]) * w.reshape(-1, *[1] * (x.ndim - 1))  # noqa: E731

    def slerp(q):
        if q is None:
            return None
        tw = torch.from_numpy(w).reshape(-1, *[1] * (q.ndim - 2)).expand(q[i0].shape[:-1])
        return quat_slerp(torch.from_numpy(q[i0]), torch.from_numpy(q[i1]), tw).numpy()

    near = np.where(w < 0.5, i0, i1)
    return Motion(m.task, fps, lerp(m.root_pos), slerp(m.root_quat), lerp(m.joint_pos), lerp(m.object_pos), slerp(m.object_quat),
                  lerp(m.hinge), lerp(m.extra_pos), slerp(m.extra_quat), None if m.contact is None else m.contact[near], m.frame[near])  # fmt: skip


# ---------------------------------------------------------------------------------------------------- USD -> OBJ
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def usd_asset(usd: str, cache_dir: str = CACHE_DIR) -> dict:
    """A USD asset's rigid bodies with their visible geometry (OBJ files in each body's frame, the bound material, the
    bounds), its joints and its root body; converted with pxr once, then read from <cache_dir>/<name>-<sha256>/."""
    if not os.path.isfile(usd):
        raise FileNotFoundError(f"{usd}: run tools/download_hdmi.sh")
    digest = _sha256(usd)
    out = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(usd))[0]}-{digest[:12]}")
    path = os.path.join(out, "manifest.json")
    if os.path.isfile(path):
        with open(path) as f:
            asset = json.load(f)
        if asset.get("version") == CACHE_VERSION and asset.get("sha256") == digest:
            return {**asset, "dir": out}
    os.makedirs(out, exist_ok=True)
    asset = {"version": CACHE_VERSION, "usd": os.path.basename(usd), "sha256": digest, **convert_usd(usd, out)}
    with open(path + ".tmp", "w") as f:
        json.dump(asset, f, indent=1)
    os.replace(path + ".tmp", path)
    return {**asset, "dir": out}


def convert_usd(usd: str, out_dir: str) -> dict:
    """Write each rigid body's visible gprims as one OBJ per prim (triangles, normals, a mesh's st as texture coordinates)
    in the body's frame into out_dir.
    Visible: not invisible and of purpose default or render, as Isaac Sim draws them (guide prims are not drawn)."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as e:
        raise ImportError(f"{usd} is not in the OBJ cache yet; converting it needs pxr (Isaac Lab's environment)") from e
    stage = Usd.Stage.Open(usd)
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"{usd}: {UsdGeom.GetStageMetersPerUnit(stage)} m per unit and {UsdGeom.GetStageUpAxis(stage)} up; 1 m and Z expected")
    xf = UsdGeom.XformCache()
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    names = {b.GetPath(): b.GetName() for b in bodies}
    if not bodies or len(set(names.values())) != len(bodies):
        raise ValueError(f"{usd}: no rigid bodies, or two with one name: {sorted(names.values())}")

    def owner(p):  # the nearest rigid body at or above a prim
        while p and p.GetPath() not in names:
            p = p.GetParent()
        return names.get(p.GetPath()) if p else None

    out_bodies = []
    for b in bodies:
        Mb = _gf_matrix(xf.GetLocalToWorldTransform(b))
        if not np.allclose(np.linalg.norm(Mb[:3, :3], axis=0), 1.0, atol=1e-6):
            raise ValueError(f"{usd}: the rigid body {b.GetPath()} is scaled")
        geoms = []
        for p in Usd.PrimRange(b, Usd.TraverseInstanceProxies()):
            img = UsdGeom.Imageable(p)
            if not p.IsA(UsdGeom.Gprim) or owner(p) != b.GetName() or img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            v, n, uv, tri = gprim_triangles(p)
            M = np.linalg.inv(Mb) @ _gf_matrix(xf.GetLocalToWorldTransform(p))
            v = v @ M[:3, :3].T + M[:3, 3]
            if n is not None:
                n = n @ np.linalg.inv(M[:3, :3])
                n /= np.linalg.norm(n, axis=-1, keepdims=True)
            if np.linalg.det(M[:3, :3]) < 0:
                tri = tri[:, ::-1]
            name = f"{b.GetName()}_{len(geoms)}.obj"
            write_obj(os.path.join(out_dir, name), v, n, uv, tri)
            geoms.append({"obj": name, "prim": p.GetPath().pathString, "material": bound_material(p), "texcoords": uv is not None,
                          "aabb": [v.min(0).tolist(), v.max(0).tolist()]})  # fmt: skip
        out_bodies.append({"name": b.GetName(), "pose": Mb.tolist(), "geoms": geoms})
    joints, child = [], set()
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = ([names[t] for t in rel.GetTargets()] for rel in (j.GetBody0Rel(), j.GetBody1Rel()))
        kind = "revolute" if p.IsA(UsdPhysics.RevoluteJoint) else "fixed" if p.IsA(UsdPhysics.FixedJoint) else None
        if kind is None or len(b1) != 1 or len(b0) > 1:
            raise ValueError(f"{usd}: joint {p.GetPath()} ({p.GetTypeName()}, bodies {b0} {b1}) is not supported")
        if not b0:  # fixes the root to the world
            continue
        q = lambda r: [r.GetReal(), *r.GetImaginary()]  # noqa: E731
        joint = {"name": p.GetName(), "type": kind, "body0": b0[0], "body1": b1[0], "pos0": list(j.GetLocalPos0Attr().Get()),
                 "quat0": q(j.GetLocalRot0Attr().Get()), "pos1": list(j.GetLocalPos1Attr().Get()), "quat1": q(j.GetLocalRot1Attr().Get())}  # fmt: skip
        if kind == "revolute":
            r = UsdPhysics.RevoluteJoint(p)
            joint.update(axis="XYZ".index(r.GetAxisAttr().Get()), range=np.radians([r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()]).tolist())
        joints.append(joint)
        child.add(b1[0])
    roots = [b["name"] for b in out_bodies if b["name"] not in child]
    if len(roots) != 1 or len(child) != len(joints):
        raise ValueError(f"{usd}: the joints do not form a tree with one root (roots {roots})")
    return {"root": roots[0], "bodies": out_bodies, "joints": joints}


def _gf_matrix(m) -> np.ndarray:
    """A Gf.Matrix4d (row vectors, translation in the last row) as a 4 x 4 matrix acting on column vectors."""
    return np.array(m, dtype=np.float64).T


def gprim_triangles(prim) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """A mesh's, cube's or sphere's triangles in the prim's frame: vertices (n, 3), their normals (n, 3) and texture
    coordinates (n, 2) (a mesh's st), each None when absent, and triangles (m, 3); a vertex per distinct combination."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Cube):
        h = 0.5 * UsdGeom.Cube(prim).GetSizeAttr().Get()
        v, n, tri = [], [], []
        for axis in range(3):
            for sign in (1.0, -1.0):
                a, b = (axis + 1) % 3, (axis + 2) % 3
                if sign < 0:
                    a, b = b, a
                k = len(v)
                for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = np.zeros(3)
                    p[axis], p[a], p[b] = sign * h, s * h, t * h
                    v.append(p)
                    n.append(np.eye(3)[axis] * sign)
                tri += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
        return np.array(v), np.array(n), None, np.array(tri, dtype=np.int64)
    if prim.IsA(UsdGeom.Sphere):
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        th, ph = np.meshgrid(np.linspace(0.0, np.pi, 33), np.linspace(0.0, 2 * np.pi, 65), indexing="ij")
        n = np.stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), -1).reshape(-1, 3)
        i = np.arange(32)[:, None] * 65 + np.arange(64)[None, :]
        tri = np.concatenate((np.stack((i, i + 65, i + 66), -1), np.stack((i, i + 66, i + 1), -1))).reshape(-1, 3)
        return r * n, n, None, tri
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"{prim.GetPath()}: {prim.GetTypeName()} prims are not supported (meshes, cubes and spheres)")
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if len(counts) == 0 or counts.min() < 3 or counts.sum() != len(idx) or len(m.GetHoleIndicesAttr().Get() or []):
        raise ValueError(f"{prim.GetPath()}: faces must be polygons without holes")
    start, k = np.cumsum(counts) - counts, counts - 2  # fans: corner indices of each face's triangles
    face = np.repeat(np.arange(len(counts)), k)
    j = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    corners = np.stack((start[face], start[face] + j + 1, start[face] + j + 2), -1)
    if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        corners = corners[:, ::-1]

    def per_corner(vals, interp, what):
        vals = np.asarray(vals, dtype=np.float64)
        per = {UsdGeom.Tokens.faceVarying: lambda: vals, UsdGeom.Tokens.vertex: lambda: vals[idx], UsdGeom.Tokens.varying: lambda: vals[idx],
               UsdGeom.Tokens.uniform: lambda: vals[np.repeat(np.arange(len(counts)), counts)],
               UsdGeom.Tokens.constant: lambda: np.repeat(vals[:1], len(idx), 0)}  # fmt: skip
        if interp not in per:
            raise ValueError(f"{prim.GetPath()}: {what} of interpolation {interp}")
        out = per[interp]()
        if len(out) != len(idx):
            raise ValueError(f"{prim.GetPath()}: {len(out)} {what} for {len(idx)} face corners")
        return out

    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
    vals, interp = (pv.ComputeFlattened(), pv.GetInterpolation()) if pv and pv.HasValue() else (m.GetNormalsAttr().Get(), m.GetNormalsInterpolation())
    n = None if vals is None or len(vals) == 0 else per_corner(vals, interp, "normals")
    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    uv = per_corner(st.ComputeFlattened(), st.GetInterpolation(), "st") if st and st.HasValue() else None
    keys, inv = np.unique(np.concatenate([idx[:, None].astype(np.float64)] + [np.round(x, 6) for x in (n, uv) if x is not None], 1), axis=0,
                          return_inverse=True)  # fmt: skip
    cols = np.cumsum([1, 3 if n is not None else 0, 2 if uv is not None else 0])
    return (pts[keys[:, 0].astype(np.int64)], None if n is None else keys[:, cols[0] : cols[1]], None if uv is None else keys[:, cols[1] : cols[2]],
            inv.reshape(-1)[corners])  # fmt: skip


def bound_material(prim) -> dict:
    """The material bound to a prim or its nearest ancestor that has one (material:binding): its MDL name and, for OmniPBR,
    the diffuse constant; Isaac Sim's default material without a binding."""
    p = prim
    while p and not p.IsPseudoRoot():
        rel = p.GetRelationship("material:binding")
        if rel and rel.GetTargets():
            mat = prim.GetStage().GetPrimAtPath(rel.GetTargets()[0])
            shaders = [s for s in mat.GetChildren() if s.GetTypeName() == "Shader"]
            if len(shaders) != 1:
                raise ValueError(f"{mat.GetPath()}: {len(shaders)} shaders, one expected")
            name = shaders[0].GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
            color = shaders[0].GetAttribute("inputs:diffuse_color_constant").Get()
            return {"name": str(name), "diffuse": None if color is None else [float(c) for c in color]}
        p = p.GetParent()
    return {"name": "default", "diffuse": None}


def write_obj(path: str, v: np.ndarray, n: np.ndarray | None, uv: np.ndarray | None, tri: np.ndarray):
    """An OBJ file of vertices, their normals and texture coordinates (either may be None) and triangles."""
    refs = "/".join(["%d", "%d" if uv is not None else "", "%d" if n is not None else ""]).rstrip("/")
    per = 1 + (uv is not None) + (n is not None)
    with open(path + ".tmp", "w") as f:
        f.write("# converted by tools/render_hoi.py\n")
        np.savetxt(f, v, fmt="v %.7g %.7g %.7g")
        if uv is not None:
            np.savetxt(f, uv, fmt="vt %.6g %.6g")
        if n is not None:
            np.savetxt(f, n, fmt="vn %.6g %.6g %.6g")
        np.savetxt(f, np.repeat(tri + 1, per, axis=1), fmt="f " + " ".join([refs] * 3))
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------------------------------- object kinematics
def _transform(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(torch.tensor(quat_wxyz, dtype=torch.float64)).numpy()
    T[:3, 3] = pos
    return T


def _axis_rotation(axis: int, q: np.ndarray) -> np.ndarray:
    """Rotations (T, 4, 4) about the joint frame's x, y or z axis by angles q (T,)."""
    c, s = np.cos(q), np.sin(q)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    R = np.tile(np.eye(4), (len(q), 1, 1))
    R[:, a, a], R[:, a, b], R[:, b, a], R[:, b, b] = c, -s, s, c
    return R


def asset_poses(asset: dict, root_pos: np.ndarray, root_quat: np.ndarray, hinge: np.ndarray | None, hinge_name: str | None) -> dict:
    """World poses (T, 4, 4) of an asset's bodies: the root at (root_pos, root_quat), every other body through its joint,
    T_1 = T_0 X(pos0, quat0) J(q) X(pos1, quat1)^-1, J the hinge's rotation by ``hinge`` (rad) or the identity."""
    T = len(root_pos)
    root = np.tile(np.eye(4), (T, 1, 1))
    root[:, :3, :3] = quat_to_mat(torch.from_numpy(np.asarray(root_quat, dtype=np.float64))).numpy()
    root[:, :3, 3] = root_pos
    poses, todo = {asset["root"]: root}, list(asset["joints"])
    revolute = [j["name"] for j in todo if j["type"] == "revolute"]
    if revolute != ([hinge_name] if hinge_name else []) or (hinge is None) != (hinge_name is None):
        raise ValueError(f"{asset['usd']}: revolute joints {revolute}, the scene's hinge {hinge_name} (angle given: {hinge is not None})")
    while todo:
        ready = [j for j in todo if j["body0"] in poses]
        if not ready:
            raise ValueError(f"{asset['usd']}: joints {[j['name'] for j in todo]} do not connect to the root")
        for j in ready:
            J = _axis_rotation(j["axis"], hinge) if j["type"] == "revolute" else np.eye(4)
            poses[j["body1"]] = poses[j["body0"]] @ _transform(j["pos0"], j["quat0"]) @ J @ np.linalg.inv(_transform(j["pos1"], j["quat1"]))
            todo.remove(j)
    return poses


def check_joint_frames(asset: dict, hinge_name: str | None):
    """The joint frames place every body where the USD authors it, relative to the root, at hinge angle 0."""
    authored = {b["name"]: np.array(b["pose"]) for b in asset["bodies"]}
    root = authored[asset["root"]]
    q = mat_to_quat(torch.from_numpy(root[None, :3, :3])).numpy()
    zero = np.zeros(1) if hinge_name else None
    for name, P in asset_poses(asset, root[None, :3, 3], q, zero, hinge_name).items():
        if np.abs(P[0] - authored[name]).max() > 1e-5:
            raise ValueError(f"{asset['usd']}: the joint frames place {name} away from its authored pose")


def box_corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.array([[(hi if i & 1 else lo)[0], (hi if i & 2 else lo)[1], (hi if i & 4 else lo)[2]] for i in range(8)])


def contact_body_corners(asset: dict, poses: dict, body: str) -> np.ndarray:
    """World corners (T, 8 k, 3) of the bounds of a body's k geoms."""
    geoms = next(b["geoms"] for b in asset["bodies"] if b["name"] == body)
    if not geoms:
        raise ValueError(f"{asset['usd']}: {body} has no visible geometry")
    c = np.concatenate([box_corners(*g["aabb"]) for g in geoms])
    P = poses[body]
    return np.einsum("tij,kj->tki", P[:, :3, :3], c) + P[:, None, :3, 3]


def interaction_window(moving: np.ndarray, contact: np.ndarray | None, length: int) -> int:
    """The start of the ``length``-frame window with the most interaction frames (the object moving, or a contact flag
    set), centered on them among windows with as many; the whole input from 0 when it is not longer."""
    T = len(moving)
    if length >= T:
        return 0
    act = moving.astype(bool) | (np.zeros(T, bool) if contact is None else np.asarray(contact).reshape(T, -1).any(-1))
    score = np.convolve(act.astype(np.int64), np.ones(length, np.int64), mode="valid")
    best = np.flatnonzero(score == score.max())
    return int(best[len(best) // 2])


# ---------------------------------------------------------------------------------------------------- the scene model
def _fmt(xs) -> str:
    return " ".join(f"{float(x):.7g}" for x in np.ravel(xs))


def material_of(geom: dict) -> tuple[str, tuple, float, float]:
    """(name, linear rgba, specular, shininess) of a converted geom's material; a textured one (PLANKS) is white, its
    texture carries the color, and needs the geom's texture coordinates."""
    mat = geom["material"]
    if mat["diffuse"] is not None:  # OmniPBR with a diffuse constant: HDMI's G1
        rgb = ROBOT_ALBEDO * np.array(mat["diffuse"])
        return "pbr_" + "_".join(f"{c:.3f}" for c in mat["diffuse"]), (*rgb, 1.0), *ROBOT_FINISH
    if mat["name"] not in MATERIALS:
        raise KeyError(f"{geom['prim']}: no color for the material {mat['name']}; add it to MATERIALS")
    rgb, spec, shin = MATERIALS[mat["name"]]
    if mat["name"] in PLANKS:
        if not geom["texcoords"]:
            raise ValueError(f"{geom['prim']}: the textured material {mat['name']} needs texture coordinates (a mesh's st)")
        return mat["name"], (1.0, 1.0, 1.0, 1.0), spec, shin
    return mat["name"], (*(np.array(rgb) / 255.0), 1.0), spec, shin


def robot_xml(robot: dict) -> str:
    """ndm_gym's G1 body tree with its geoms replaced by the HDMI USD's visual meshes (mesh robot/<body>_<k>)."""
    pelvis = ET.parse(MJCF_PATH).getroot().find("worldbody/body")
    usd = {b["name"]: b for b in robot["bodies"]}
    placed = set()
    for body in pelvis.iter("body"):
        for g in body.findall("geom") + body.findall("site"):
            body.remove(g)
        for k, g in enumerate(usd.get(body.get("name"), {"geoms": []})["geoms"]):
            ET.SubElement(body, "geom", type="mesh", mesh=f"robot/{os.path.splitext(g['obj'])[0]}", material=material_of(g)[0])
            placed.add(body.get("name"))
    missing = sorted(n for n, b in usd.items() if b["geoms"] and n not in placed)
    if missing:
        raise ValueError(f"{robot['usd']}: links {missing} with visual meshes are not bodies of {MJCF_PATH}")
    return ET.tostring(pelvis, encoding="unicode")


def scene_xml(robot: dict, assets: list[tuple[str, dict]], width: int, height: int, cache_dir: str = CACHE_DIR) -> str:
    """The robot (ndm_gym's G1 tree, HDMI's meshes), every asset body with geometry as a mocap body <prefix>/<body>, the
    floor, the sky and the lights (the key light on a mocap rig at the camera's target, which centers its shadow map)."""
    meshes, materials, bodies = [], {}, []
    for prefix, a in [("robot", robot)] + assets:
        sub = os.path.basename(a["dir"])
        for b in a["bodies"]:
            for g in b["geoms"]:
                meshes.append(f'    <mesh name="{prefix}/{os.path.splitext(g["obj"])[0]}" file="{sub}/{g["obj"]}" inertia="shell"/>')
                name, rgba, spec, shin = material_of(g)
                tex = f' texture="{name}"' if name in PLANKS else ""
                materials[name] = f'    <material name="{name}" rgba="{_fmt(rgba)}" specular="{spec}" shininess="{shin}" reflectance="0"{tex}/>'
            if prefix != "robot" and b["geoms"]:
                geoms = "".join(f'<geom type="mesh" mesh="{prefix}/{os.path.splitext(g["obj"])[0]}" material="{material_of(g)[0]}"/>' for g in b["geoms"])
                bodies.append(f'    <body name="{prefix}/{b["name"]}" mocap="true">{geoms}</body>')
    textures = "".join(f'\n    <texture name="{n}" type="2d" file="{n}.png"/>' for n in materials if n in PLANKS)
    light = lambda name, spec, shadow: (f'<light name="{name}" directional="true" castshadow="{shadow}" pos="0 0 4" dir="0 0 -1" '  # noqa: E731
                                        f'diffuse="{_fmt(spec[0])}" specular="{_fmt(spec[1])}"/>')  # fmt: skip
    return f"""<mujoco model="hoi_render">
  <compiler angle="radian" meshdir="{cache_dir}" inertiafromgeom="false" autolimits="true"/>
  <statistic extent="2" center="0 0 0.8"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="{FOVY}"/>
    <quality shadowsize="8192" offsamples="16" numquads="8"/>
    <headlight ambient="{_fmt(HEADLIGHT_AMBIENT)}" diffuse="{_fmt(HEADLIGHT_DIFFUSE)}" specular="0 0 0"/>
    <map znear="0.01" zfar="60" shadowclip="2.5" haze="0.3" fogstart="{FOG[0]}" fogend="{FOG[1]}"/>
    <rgba haze="{_fmt(np.array(SKY[0][1]) / 255.0)} 1" fog="{_fmt(np.array(SKY[0][1]) / 255.0)} 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0" group="1"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" width="{SKY_SIZE}" height="{SKY_SIZE}"/>
    <texture name="grid" type="2d" file="grid.png"/>{textures}
    <material name="floor" texture="grid" texrepeat="{1 / GRID:g} {1 / GRID:g}" texuniform="true" reflectance="0" specular="0" shininess="0"/>
{chr(10).join(materials.values())}
{chr(10).join(meshes)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {light("fill", FILL, "false")}
    {light("rim", RIM, "false")}
    <body name="rig" mocap="true">
      {light("key", KEY, "true")}
    </body>
{robot_xml(robot)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def grid_png() -> bytes:
    """One floor square: FLOOR_RGB with LINE_RGB lines along its edges (half a line on each, so the squares tile)."""
    from PIL import Image

    d = np.minimum((np.arange(GRID_PX) + 0.5), GRID_PX - (np.arange(GRID_PX) + 0.5))  # texels to the nearest edge
    a = np.clip(0.5 * GRID_LINE / GRID * GRID_PX + 0.5 - d, 0.0, 1.0)  # line coverage, a texel of anti-aliasing
    cov = np.maximum(a[:, None], a[None, :])[..., None]
    rgb = np.array(FLOOR_RGB) * (1.0 - cov) + np.array(LINE_RGB) * cov
    buf = io.BytesIO()
    Image.fromarray(np.round(rgb).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def plank_png(rgb, planks: int, size: tuple[int, int] = (1024, 512), seed: int = 0) -> bytes:
    """A wood texture: ``planks`` planks side by side along u, each running along v (on the ball's spherical coordinates:
    around it, and from pole to pole), each a tone of rgb with grain along it and darker seams; tiles along u."""
    from PIL import Image

    g = np.random.default_rng(seed)
    W, H = size
    u, v = np.meshgrid((np.arange(W) + 0.5) / W * planks, (np.arange(H) + 0.5) / H)
    k = np.floor(u).astype(np.int64) % planks
    a = u - np.floor(u)  # across the plank
    tone, freq, phase = 1.0 + 0.06 * g.standard_normal(planks), g.uniform(5.0, 9.0, planks), g.uniform(0.0, 2 * np.pi, planks)
    x = a + 0.04 * np.sin(2 * np.pi * (2.0 * v + a))  # wavy grain
    grain = 0.05 * np.sin(2 * np.pi * freq[k] * x + phase[k]) + 0.025 * np.sin(2 * np.pi * 3.1 * freq[k] * x + 2.0 * phase[k])
    seam = np.exp(-((np.minimum(a, 1.0 - a) / 0.02) ** 2))
    img = np.array(rgb, dtype=np.float64) * (tone[k] * (1.0 + grain) * (1.0 - 0.3 * seam))[..., None]
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(img, 0, 255)).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def sky_cube(size: int = SKY_SIZE) -> np.ndarray:
    """The sky as a MuJoCo skybox texture: 6 size x size faces stacked (+x, -x, up, down, +z, -z as MuJoCo lays out its
    builtin gradient; side faces' rows from the top), colored by elevation; uint8 (6 size, size, 3)."""
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    v, u = np.meshgrid(-c, c, indexing="ij")
    side = np.degrees(np.arctan2(v, np.sqrt(1.0 + u * u)))
    top = np.degrees(np.arctan2(1.0, np.sqrt(u * u + v * v)))
    el = np.concatenate([side, side, top, -top, side, side])
    knots = np.array([e for e, _ in SKY])
    rgb = np.stack([np.interp(el, knots, [col[k] for _, col in SKY]) for k in range(3)], -1)
    return np.round(rgb).astype(np.uint8)


class SceneModel:
    """The MuJoCo model of a task's scene and how to pose it from a Motion."""

    def __init__(self, scene: Scene, width: int, height: int, cache_dir: str = CACHE_DIR):
        import mujoco

        self.scene = scene
        self.robot = usd_asset(scene.robot_usd, cache_dir)
        self.obj = usd_asset(scene.object_usd, cache_dir)
        self.props = [(n, usd_asset(u, cache_dir)) for n, u in scene.props]
        check_joint_frames(self.obj, scene.hinge)
        assets = [("object", self.obj)] + [(f"prop_{n}", a) for n, a in self.props]
        textures = {"grid.png": grid_png(), **{f"{n}.png": plank_png(MATERIALS[n][0], k) for n, k in PLANKS.items()}}
        self.model = m = mujoco.MjModel.from_xml_string(scene_xml(self.robot, assets, width, height, cache_dir), textures)
        sky, cube = m.texture("sky"), sky_cube()
        if (int(sky.height[0]), int(sky.width[0]), int(sky.nchannel[0])) != cube.shape:
            raise ValueError(f"MuJoCo's skybox texture is {sky.height[0]} x {sky.width[0]} x {sky.nchannel[0]}, expected {cube.shape}")
        m.tex_data[int(sky.adr[0]) : int(sky.adr[0]) + cube.size] = cube.ravel()
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.joint(n).qposadr[0] for n in G1_JOINT_NAMES])
        self.mocap = {name: m.body(name).mocapid[0] for name in (m.body(i).name for i in range(m.nbody)) if m.body(name).mocapid[0] >= 0}
        self.robot_geoms = np.flatnonzero(np.isin(m.geom_bodyid, [m.body(b["name"]).id for b in self.robot["bodies"] if b["geoms"]]))
        self.framed_geoms = np.flatnonzero(m.geom_bodyid == m.body(f"object/{scene.object_body}").id)
        self.object_geoms = np.flatnonzero(np.isin(m.geom_bodyid, [m.body(f"object/{b['name']}").id for b in self.obj["bodies"] if b["geoms"]]))
        self._check_robot_links()

    def _check_robot_links(self):
        """HDMI's G1 USD and ndm_gym's MJCF place every link alike at the zero pose (root at the origin), to 0.1 mm and
        1 mrad (the MJCF's quaternions are rounded: they differ by up to 0.2 mrad)."""
        import mujoco

        d = self.data
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        mujoco.mj_kinematics(self.model, d)
        for b in self.robot["bodies"]:
            P, i = np.array(b["pose"]), self.model.body(b["name"]).id
            cos = (np.trace(d.xmat[i].reshape(3, 3).T @ P[:3, :3]) - 1.0) / 2.0
            if np.abs(d.xpos[i] - P[:3, 3]).max() > 1e-4 or cos < np.cos(1e-3):
                raise ValueError(f"{self.robot['usd']}: {b['name']} is not where {MJCF_PATH} puts it at the zero pose")

    def object_poses(self, m: Motion) -> dict:
        """World poses (T, 4, 4) of the object's bodies and the props', keyed by mocap body name."""
        out = {f"object/{k}": v for k, v in asset_poses(self.obj, m.object_pos, m.object_quat, m.hinge, self.scene.hinge).items()}
        for i, (n, a) in enumerate(self.props):
            out.update({f"prop_{n}/{k}": v for k, v in asset_poses(a, m.extra_pos[:, i], m.extra_quat[:, i], None, None).items()})
        return out

    def pose(self, m: Motion, poses: dict, t: int):
        """Set step t of the motion (``poses``: object_poses(m)) and update the kinematics."""
        import mujoco

        d = self.data
        d.qpos[0:3], d.qpos[3:7], d.qpos[self.qadr] = m.root_pos[t], m.root_quat[t], m.joint_pos[t]
        for name, P in poses.items():
            if name in self.mocap:
                d.mocap_pos[self.mocap[name]] = P[t, :3, 3]
                d.mocap_quat[self.mocap[name]] = mat_to_quat(torch.from_numpy(P[t, :3, :3])).numpy()
        mujoco.mj_kinematics(self.model, d)

    def corners(self, g: np.ndarray) -> np.ndarray:
        """World corners (8 len(g), 3) of the bounds of geoms g (after pose())."""
        m, d = self.model, self.data
        aabb = m.geom_aabb[g]  # center and half sizes in the geom's frame
        signs = np.array([[(i & 1) * 2 - 1, (i >> 1 & 1) * 2 - 1, (i >> 2 & 1) * 2 - 1] for i in range(8)], dtype=np.float64)
        local = aabb[:, None, :3] + signs[None] * aabb[:, None, 3:]
        return (np.einsum("gij,gkj->gki", d.geom_xmat[g].reshape(-1, 3, 3), local) + d.geom_xpos[g][:, None]).reshape(-1, 3)


# ---------------------------------------------------------------------------------------------------- camera and lights
def camera_axes(azimuth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MuJoCo's free camera: forward, right and up for an azimuth (degrees) at ELEVATION."""
    a, e = np.radians(azimuth), np.radians(ELEVATION)
    f = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    r = np.array([np.sin(a), -np.cos(a), 0.0])
    return f, r, np.cross(r, f)


def fit_view(P: np.ndarray, azimuth: float, aspect: float) -> tuple[np.ndarray, float]:
    """A target and distance that fit the points P (n, 3) within MARGIN of the half frame: the target centers their bounds
    on the camera's axes, and every point satisfies |x| <= MARGIN tan(fov / 2) (z + d) on each axis."""
    f, r, u = camera_axes(azimuth)
    tv = np.tan(np.radians(FOVY) / 2) * MARGIN
    th = tv * aspect
    a = P @ np.stack((r, u, f), 1)  # coordinates along right, up, forward
    c = 0.5 * (a.min(0) + a.max(0))
    rel = a - c
    d = np.max(np.maximum(np.abs(rel[:, 0]) / th, np.abs(rel[:, 1]) / tv) - rel[:, 2])
    return c @ np.stack((r, u, f)), max(float(d), MIN_DISTANCE)


def choose_azimuth(sm: SceneModel, m: Motion, aspect: float, samples: int = 12) -> float:
    """The camera heading (every 10 degrees) with the best mean score over ``samples`` steps of the motion, each step framed
    as camera_path frames it: -(the share of the robot's image bounds that the nearer object's covers, all its bodies, or
    half the share of the object's that the nearer robot covers) + FRONT x (how far the robot faces the camera) - SPREAD x
    distance / MIN_DISTANCE."""
    poses = sm.object_poses(m)
    ts = np.unique(np.linspace(0, len(m.root_pos) - 1, samples).round().astype(np.int64))
    heads = quat_to_mat(torch.from_numpy(m.root_quat[ts])).numpy()[:, :2, 0]
    heads /= np.linalg.norm(heads, axis=1, keepdims=True)
    parts = []
    for t in ts:
        sm.pose(m, poses, t)
        parts.append((sm.corners(sm.robot_geoms), sm.corners(sm.framed_geoms), sm.corners(sm.object_geoms)))

    def bounds(P, e, f, r, u):  # image bounds (x0, y0, x1, y1) and mean depth of points seen from e
        z = (P - e) @ f
        x, y = ((P - e) @ r) / z, ((P - e) @ u) / z
        return np.array([x.min(), y.min(), x.max(), y.max()]), z.mean()

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    best, best_score = 0.0, -np.inf
    for az in np.arange(0.0, 360.0, 10.0):
        f, r, u = camera_axes(az)
        fh = f[:2] / np.linalg.norm(f[:2])
        score = 0.0
        for (Pr, Pf, Po), h in zip(parts, heads):
            c, d = fit_view(np.concatenate((Pr, Pf)), az, aspect)
            (br, zr), (bo, zo) = bounds(Pr, c - d * f, f, r, u), bounds(Po, c - d * f, f, r, u)
            inter = area(np.array([max(br[0], bo[0]), max(br[1], bo[1]), min(br[2], bo[2]), min(br[3], bo[3])]))
            hidden = inter / area(br) if zo < zr else 0.5 * inter / area(bo)
            score += -hidden + FRONT * max(0.0, -(fh @ h)) - SPREAD * d / MIN_DISTANCE
        if score > best_score + 1e-9:
            best, best_score = float(az), score
    return best


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the first axis (sigma in samples), edges held."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, k, mode="valid")


def camera_path(sm: SceneModel, m: Motion, azimuth: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Per step of the motion: the camera's target (T, 3) and distance (T,), holding and gliding. Each step needs the
    distance at which fit_view frames the robot and the object's contact body (its running max over ZOOM_HOLD s, so that
    fast limbs stay in). A hold is the longest run of steps, from where the last ended, that one view frames, half a glide
    beyond each end included, within ZOOM_LIMIT times the largest need in the run; the holds' views, smoothed both ways
    over a quarter of GLIDE, glide from one to the next."""
    poses = sm.object_poses(m)
    points = []
    for t in range(len(m.root_pos)):
        sm.pose(m, poses, t)
        points.append(sm.corners(np.concatenate((sm.robot_geoms, sm.framed_geoms))))
    need = np.array([fit_view(P, azimuth, aspect)[1] for P in points])
    hold = int(round(ZOOM_HOLD * m.fps))
    need = np.array([need[max(0, t - hold) : t + hold + 1].max() for t in range(len(need))])
    T, g = len(points), int(round(GLIDE * m.fps / 2))
    stride = max(1, int(round(m.fps / 10)))  # 10 steps per second suffice to fit a run
    target, dist = np.empty((T, 3)), np.empty(T)
    a = 0
    while a < T:
        view = fit_view(np.concatenate(points[max(0, a - g) : a + g + 1 : stride]), azimuth, aspect)
        b = a
        while b + 1 < T:
            v = fit_view(np.concatenate(points[max(0, a - g) : b + g + 2 : stride]), azimuth, aspect)
            if v[1] > ZOOM_LIMIT * need[a : b + 2].max():
                break
            view, b = v, b + 1
        target[a : b + 1], dist[a : b + 1] = view
        a = b + 1
    return _smooth(target, GLIDE / 4 * m.fps), _smooth(dist, GLIDE / 4 * m.fps)


def light_dir(azimuth: float, heading: float, elevation: float) -> np.ndarray:
    """The direction of a light at a heading relative to the camera's position (azimuth + 180 degrees) and an elevation."""
    h, e = np.radians(azimuth + 180.0 + heading), np.radians(elevation)
    return -np.array([np.cos(e) * np.cos(h), np.cos(e) * np.sin(h), np.sin(e)])


def soft_key_dirs(azimuth: float) -> np.ndarray:
    """The key light's directions (SOFT_SAMPLES, 3): points of a disk of SOFT_ANGLE degrees around it (a Vogel spiral)."""
    d = light_dir(azimuth, KEY[2], KEY[3])
    e1 = np.cross(d, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    k = np.arange(SOFT_SAMPLES)
    r, phi = np.tan(np.radians(SOFT_ANGLE)) * np.sqrt((k + 0.5) / SOFT_SAMPLES), k * np.pi * (3.0 - np.sqrt(5.0))
    out = d + r[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def aim_lights(model, azimuth: float, key_dir: np.ndarray | None = None):
    """Key (``key_dir``, default its center), fill and rim lights at their headings relative to the camera's position."""
    for name, spec in (("key", KEY), ("fill", FILL), ("rim", RIM)):
        i = model.light(name).id
        model.light_dir[i] = key_dir if name == "key" and key_dir is not None else light_dir(azimuth, spec[2], spec[3])
        model.light_pos[i] = -4.0 * model.light_dir[i]  # the key's: relative to the rig, the shadow map's center


# ---------------------------------------------------------------------------------------------------- rendering
def render(m: Motion, out: str, start: float = 0.0, seconds: float = 10.0, fps: float | None = None, width: int = 1920, height: int = 1080,
           azimuth: float | None = None, cache_dir: str = CACHE_DIR) -> dict:
    """Render ``seconds`` of the motion from ``start`` (an .mp4 at ``fps``, default the motion's rate) or the frame at
    ``start`` (a .png). Returns the window, the frame count and the camera heading."""
    import mujoco
    from PIL import Image, ImageDraw
    from render import _font, check_egl_device, draw_mark  # tools/render.py: importing it first sets up EGL as it does

    ext = os.path.splitext(out)[1].lower()
    if ext not in (".mp4", ".png"):
        raise ValueError(f"{out}: the output must be an .mp4 or a .png")
    check_egl_device()
    duration = (len(m.root_pos) - 1) / m.fps
    if not 0.0 <= start <= duration:
        raise ValueError(f"--start {start} s is outside the input's {duration:.2f} s")
    fps = fps or m.fps
    n = 1 if ext == ".png" else max(1, int(np.floor(min(seconds, duration - start) * fps + 1e-6)))
    times = start + np.arange(n) / fps
    sm = SceneModel(scene_of(m.task), width, height, cache_dir)
    # the camera: fitted on the input's own steps over the window and the smoothing's reach, then sampled at the frames
    pad = GLIDE + ZOOM_HOLD
    i0, i1 = int(max(0, np.floor((times[0] - pad) * m.fps))), int(min(len(m.root_pos), np.ceil((times[-1] + pad) * m.fps) + 1))
    win = m._replace(**{k: getattr(m, k)[i0:i1] for k in Motion._fields if isinstance(getattr(m, k), np.ndarray)})
    if azimuth is None:
        azimuth = choose_azimuth(sm, win, width / height)
    target, dist = camera_path(sm, win, azimuth, width / height)
    s = times * m.fps - i0
    target = np.stack([np.interp(s, np.arange(len(dist)), target[:, k]) for k in range(3)], 1)
    dist = np.interp(s, np.arange(len(dist)), dist)
    frames = resample(m, times, fps)
    poses = sm.object_poses(frames)

    model = sm.model
    renderer = mujoco.Renderer(model, height, width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, ELEVATION
    rig = model.body("rig").mocapid[0]
    bold = _font(0, round(20 * (height / 432)))
    ff = None
    if ext == ".mp4":
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", out],
            stdin=subprocess.PIPE,
        )  # fmt: skip
    keys = soft_key_dirs(azimuth)
    acc = np.zeros((height, width, 3), np.float32)
    for k in range(len(times)):
        sm.pose(frames, poses, k)
        sm.data.mocap_pos[rig] = target[k]
        mujoco.mj_kinematics(model, sm.data)
        cam.lookat[:], cam.distance = target[k], dist[k]
        acc[:] = 0.0
        for key in keys:  # soft shadows: the mean of renders lit from points of the key light's disk
            aim_lights(model, azimuth, key)
            mujoco.mj_camlight(model, sm.data)
            renderer.update_scene(sm.data, camera=cam)
            acc += renderer.render()
        img = Image.fromarray(np.round(acc / len(keys)).astype(np.uint8)).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_mark(ImageDraw.Draw(over), width, height, bold, MARK_RGBA)
        img = Image.alpha_composite(img, over).convert("RGB")
        if ff is None:
            img.save(out)
        else:
            ff.stdin.write(np.asarray(img).tobytes())
    renderer.close()
    if ff is not None:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")
    return {"start": float(times[0]), "seconds": float(times[-1] - times[0]), "frames": len(times), "fps": fps, "azimuth": azimuth}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help=".mp4 (--seconds from --start) or .png (the frame at --start)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", help="a task's converted reference clip (ndm_gym.tasks.hdmi_tasks.TASKS)")
    src.add_argument("--states", help="an npz of states recorded from HDMISim (the docstring)")
    ap.add_argument("--start", type=float, default=0.0, help="s")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=None, help="video frame rate; default the input's (50), else poses are interpolated")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--azimuth", type=float, default=None, help="the camera's heading (degrees, MuJoCo's); default from the scene")
    ap.add_argument("--cache", default=CACHE_DIR, help="the OBJ cache of the converted USDs")
    a = ap.parse_args()
    m = clip_motion(a.task) if a.task else load_states(a.states)
    info = render(m, a.out, a.start, a.seconds, a.fps, a.width, a.height, a.azimuth, a.cache)
    print(f"wrote {a.out}: {info['frames']} frame(s) from {info['start']:.2f} s at {info['fps']:g} fps, camera heading {info['azimuth']:.0f} deg")


if __name__ == "__main__":
    main()
