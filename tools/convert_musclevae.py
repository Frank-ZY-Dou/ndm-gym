"""Convert MuscleVAE's muscle-actuated character, its muscles and its reference motions (wangchek/MuscleVAE_ @ 54c001c,
placed in data/musclevae/src by tools/download_musclevae.sh) into ndm_gym's assets in data/musclevae/, git-ignored like
the source (MuscleVAE has no license, so neither is redistributed):

* character.xml: the character as an MJCF; character.json: its bodies, joints and conventions, read by
  ndm_gym.robots.musclevae_char;
* muscles.json: the muscle table (anchors, rest lengths, Hill and PD constants), read into
  ndm_gym.robots.muscles.MuscleTable, with the released code's and the paper's parameters side by side;
* clips/<name>.npz and <name>.json: their three reference motions at 20 Hz in ndm_gym's clip format for this character
  (ndm_gym.motion.sources.musclevae).

References are to MuscleVAE's source: LDR = Muscle/muscle_scene_loader.py, CHR = Muscle/muscle_character.py, ENV =
MuscleVAECore/Env/muscle_env.py, BE = ModifyODESrc/VclSimuBackend.pyx (their ODE wrapper, loaders and muscle kernels),
CFG = Data/NNModel/config.yml.

Inputs, checked against their SHA-256: the character of CFG's env_scene_fname (Data/Misc/muscle_character.json; of its
two characters the first, the second is their identical kinematic reference copy), the kp and force files of ENV:98-103,
CFG (its rates, file names and fatigue ratio are checked), the three BVHs of CFG's bvh_folder, and
Data/Misc/muscle_character_full.json, used only to cross-check the anchors: it lists the bodies and blend weights their
loader gave each waypoint.

Frames. Their world is Y-up with the character facing +z; ours is Z-up with it facing +x: (x, y, z) -> (z, x, y), a
rotation C. Body frames turn with the world, so every body keeps the identity orientation of the T-pose; a geom keeps
its own axes (a capsule along its local z), so its orientation is C times theirs.

Bodies (their JSON loader, BE:20784-20910). A body's origin is the center of mass of its geoms at its density (1000
kg/m^3; ODE's sphere, capsule and box formulas), as their loader places it (BE:18464-18490): the pelvis's three spheres
put it 1.72 cm below its JSON position, the other bodies have one geom on their JSON position. The mass is the JSON's
value, but the inertia stays that of the geoms at 1000 kg/m^3, because their loader sets Mass.mass without rescaling
the inertia (BE:20880-20890, 572-575): the pelvis has 15 kg with the inertia of 5.4 kg of spheres. The MJCF keeps both
as their simulator has them, in explicit <inertial> elements (inertiafromgeom="false"); total mass 61.4 kg.

Joints. Ball and hinge joints at the JSON anchors, damping 10 on every degree of freedom (ODE's implicit joint damping,
as MuJoCo's). Hinge axes: their x (knees, toes) or y (elbows) in the T-pose, our y and z; hinge ranges from the JSON.
The left toes' hinges are kept as the JSON places them, each next to the other toe (FootThumbL's by FootPinkyL and the
reverse, unlike the right foot's).
Ball joints: their simulator limits the three Euler angles of an ODE AMotor separately (Euler mode, their x axis
anchored to the child body and z to the parent, BE:17727-17750): +-2 rad about each axis at the hips and shoulders,
+-0.4 / +-0.2 / +-0.4 rad at the spine, +-0.4 rad at the neck and head, +-0.5 at the clavicles, +-0.7 at the wrists,
+-1 at the ankles. MuJoCo's ball joints only take a cone limit on the rotation angle, which cannot express per-axis
Euler limits: the MJCF's ball joints are unlimited, and character.json keeps their limits (radians, about their x, y, z
axes, which are our y, z, x) for the simulator to enforce or measure.

Collisions. Self-collision is off (WorldAttr SelfCollision false, ENV:62): the character's geoms have contype 1 and
conaffinity 0, so they touch anything with conaffinity 1 (a floor) but not each other. The MJCF has no floor; the scene
adds it, as for the G1. Friction 0.8 as in the JSON.

Muscles (their muscle loader, LDR:131-392). Waypoints are MASS's (Lee et al. 2019) with x mirrored and MASS's body
names mapped to the character's, MASS's right side to the character's left (LDR:70-97, 241-244). The origin and the
insertion stay on their bodies. An intermediate waypoint within 8 cm of the nearest of the root's center of mass and the
joints that join each other body to its parent is blended between that body and its parent, with weights proportional
to 1 / sqrt of each one's distance (the parent's distance is to its own parent joint, or to the root's center of mass),
and its force goes entirely to the nearer body; nearest to the root's center of mass, it moves with the root; farther
than 8 cm it stays on its body (LDR:285-375). Anchors on the root sit 1.72 cm lower than MASS's waypoint: their loader
stores them relative to the pelvis's first geom, and the anchor update shifts them back by the geom's distance from the
root's center of mass along the root's y axis (CHR:147-179, BE:2930-2970), so they move with the root as if relative to
its center of mass. The rest length is the T-pose polyline length (LDR:392). kp = f0 x manual_muscle_kp4.txt (ENV:101),
f0max = 1.5 x MuscleVAE_force.txt (ENV:103), the body part (fatigue state) by the insertion (LDR:253-265), the group by
group_name (LDR:143-145). The bodies and weights must match muscle_character_full.json's (weights within 1e-6: float32
there).

Clips. Their LAFAN1 walk, run and jump BVHs (100 fps) at 20 Hz: every fifth frame, which is what their resampling
(linear and slerp at the 20 Hz ticks, BE:13845-13890) evaluates to. Their dataset places each body by its BVH joint
(BE:18761-18860): the child body of a joint takes that joint's global BVH orientation and the position that puts its
joint anchor on the BVH joint; the root body takes the BVH root's orientation and its position plus the root's center
of mass offset. The BVH skeleton's joint offsets differ from the character's (by 1.1 cm on average, 5.3 cm at most, at
the right big toe), so their bodies are dislocated, and the BVHs turn the knees and elbows about all three axes. The
clips hold joint positions for the character's own skeleton instead: the root body's pose as theirs; each ball joint
the rotation that gives its child the BVH's global orientation, relative to the converted parent (so a ball joint below
a hinge also takes up what the hinge cannot); each hinge the twist, about its axis, of the BVH's rotation relative to the
converted parent (the closest rotation about that axis). Every body then has the BVH's orientation but the hinges'
children (lower legs, forearms; the toes' BVH rotations are zero), and the positions follow the character's skeleton.
Each npz also keeps their placement of the bodies (body_pos, body_quat_wxyz) for checks and for MuscleVAE's own
observations; their mirrored copies (MotionDataSet's mirror augment) are not made here.

Usage, from the repository root, in ndm_gym's environment:
  tools/download_musclevae.sh
  python tools/convert_musclevae.py           # data/musclevae/src -> data/musclevae
  python tools/convert_musclevae.py --check   # MuJoCo FK of the MJCF, the T-pose muscle lengths, the clips vs their placement
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from typing import NamedTuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ndm_gym.robots import musclevae_char as mc  # noqa: E402
from ndm_gym.robots.muscles import MUSCLE_MODELS, damping_gain  # noqa: E402

SRC_DIR = os.path.join(mc.DATA_DIR, "src")
CHARACTER_FILE = "Data/Misc/muscle_character.json"
FULL_FILE = "Data/Misc/muscle_character_full.json"
KP_FILE = "Data/Misc/manual_muscle_kp4.txt"
FORCE_FILE = "Data/Misc/MuscleVAE_force.txt"
CONFIG_FILE = "Data/NNModel/config.yml"
CLIP_FILES = {name: f"Data/ReferenceData/walk_run_jump/{name}.bvh" for name in ("lafan_walk", "lafan_run", "lafan_jump")}
SOURCE_SHA256 = {
    CHARACTER_FILE: "a67ede529911b525b784465837a2ca41b21160b1a1056de3e1907a5509834e29",
    FULL_FILE: "1d4d2519191ae846e9010afe88b9c7e6bbd6f83e1cb4801932462b5968f933c7",
    KP_FILE: "bf5f54c6602b30e010bdfdd8502cb8570b7c21bba60438d991831954baeaf5f6",
    FORCE_FILE: "3aee3f8b0640b620e05de61343fc0ee43ae09efda7fe471a664ecca1bcb00ebf",
    CONFIG_FILE: "e81e1c7c20632d2e6265556bc84e2594f8666aeb0ad45d43198a800637e28758",
    CLIP_FILES["lafan_walk"]: "9bc46c54c717726935fa15de820c58ea27003697577b0201be235f8c8cad7d3d",
    CLIP_FILES["lafan_run"]: "a415ff56a149a8fb6f2a384cc00c5cafc50a5023ea47daa6b8fbe398a5fdcd15",
    CLIP_FILES["lafan_jump"]: "d28054b7105c180007cbf231555c33d3fa004acc1ab7be322b5a5f53b38f4dcf",
}
# CFG entries this conversion relies on
CONFIG_EXPECTED = {"env_fps": mc.CONTROL_HZ, "env_substep": mc.SUBSTEPS, "env_scene_fname": "./" + CHARACTER_FILE, "kp_coff_ratio": "./" + KP_FILE,
                   "fatigue_adjust_ratio": 2.0, "bvh_folder": "./Data/ReferenceData/walk_run_jump", "env_self_collision": 0}  # fmt: skip
C = np.array(mc.Y_UP_TO_Z_UP)
BLEND_RADIUS = 0.08  # m (LDR:296)
F0MAX_SCALE = 1.5  # ENV:103
FATIGUE_F, FATIGUE_R = 0.05, 0.01  # 1/s, times CFG's fatigue_adjust_ratio (ENV:111-113)
CLIP_FPS = 20
FULL_WEIGHT_TOL = 1e-6  # muscle_character_full.json stores float32 weights
CHECK_TOL = 1e-6  # m and rad: MuJoCo's FK of the converted MJCF and clips (float32) vs the converted values
# MASS's body names (LDR:70-97): unsided ones, and sided ones whose R / L is the character's left / right
MASS_UNSIDED = {"Pelvis": "pelvis", "Spine": "lowerback", "Torso": "torso", "Neck": "Neck", "Head": "Head"}
MASS_SIDED = {"Femur": "{s}UpperLeg", "Tibia": "{s}LowerLeg", "Talus": "{s}Foot", "Shoulder": "{s}Clavicle", "Arm": "{s}UpperArm",
              "ForeArm": "{s}LowerArm", "Hand": "{s}Hand", "FootThumb": "FootThumb{S}", "FootPinky": "FootPinky{S}"}  # fmt: skip
PART_BODIES = {1: ("lUpperLeg", "lLowerLeg", "lFoot", "FootThumbL", "FootPinkyL"), 2: ("rUpperLeg", "rLowerLeg", "rFoot", "FootThumbR", "FootPinkyR"),
               3: ("lClavicle", "lUpperArm", "lLowerArm", "lHand"), 4: ("rClavicle", "rUpperArm", "rLowerArm", "rHand")}  # by insertion; else 0
GEOM_TYPES = {"Sphere": "sphere", "Capsule": "capsule", "Cube": "box"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mass_body(name: str) -> str:
    """The character's body for a MASS waypoint body name; MASS's right side is the character's left (LDR:70-97)."""
    if name in MASS_UNSIDED:
        return MASS_UNSIDED[name]
    base, side = name[:-1], name[-1:]
    if base not in MASS_SIDED or side not in ("R", "L"):
        raise ValueError(f"unknown MASS body {name!r}")
    ours = "l" if side == "R" else "r"
    return MASS_SIDED[base].format(s=ours, S=ours.upper())


# ---- rotations (numpy, rotation matrices act on column vectors, quaternions are wxyz unless named xyzw)
def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64) / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(np.shape(q)[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4), wxyz with w >= 0 (from the largest diagonal term)."""
    R = np.asarray(R, dtype=np.float64)
    m = R.reshape(-1, 3, 3)
    tr = np.trace(m, axis1=1, axis2=2)
    cand = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], -1)
    k = cand.argmax(-1)
    q = np.empty((len(m), 4))
    for i in range(4):
        s = k == i
        if not s.any():
            continue
        r = m[s]
        if i == 0:
            t = np.sqrt(1 + tr[s]) * 2
            q[s] = np.stack([t / 4, (r[:, 2, 1] - r[:, 1, 2]) / t, (r[:, 0, 2] - r[:, 2, 0]) / t, (r[:, 1, 0] - r[:, 0, 1]) / t], -1)
        else:
            a, b, c = i - 1, i % 3, (i + 1) % 3
            t = np.sqrt(1 + r[:, a, a] - r[:, b, b] - r[:, c, c]) * 2
            v = np.empty((len(r), 4))
            v[:, 0] = (r[:, c, b] - r[:, b, c]) / t
            v[:, 1 + a] = t / 4
            v[:, 1 + b] = (r[:, b, a] + r[:, a, b]) / t
            v[:, 1 + c] = (r[:, c, a] + r[:, a, c]) / t
            q[s] = v
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(R.shape[:-2] + (4,))


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angles between the rotations of quaternions (..., 4), accurate near 0 (unlike arccos of their dot product):
    |q1 -+ q2| = 2 sin(angle / 4), |q1 +- q2| = 2 cos(angle / 4)."""
    q1, q2 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True), q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    d1, d2 = np.linalg.norm(q1 - q2, axis=-1), np.linalg.norm(q1 + q2, axis=-1)
    return 4.0 * np.arctan2(np.minimum(d1, d2), np.maximum(d1, d2))


def axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotations (..., 3, 3) by angles (...) about one unit axis (3,)."""
    x, y, z = np.asarray(axis, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    return np.stack([c + x * x * t, x * y * t - z * s, x * z * t + y * s, y * x * t + z * s, c + y * y * t, y * z * t - x * s,
                     z * x * t - y * s, z * y * t + x * s, c + z * z * t], -1).reshape(np.shape(angle) + (3, 3))  # fmt: skip


def euler_xyz_deg(angles: np.ndarray) -> np.ndarray:
    """BVH rotation channels "Xrotation Yrotation Zrotation" in degrees (..., 3) -> Rx Ry Rz (intrinsic x, y, z)."""
    a = np.deg2rad(np.asarray(angles, dtype=np.float64))
    e = np.eye(3)
    return axis_angle(e[0], a[..., 0]) @ axis_angle(e[1], a[..., 1]) @ axis_angle(e[2], a[..., 2])


def to_z_up_rot(R: np.ndarray) -> np.ndarray:
    """A body frame's rotation in our frame: C R C^T (body frames turn with the world)."""
    return C @ R @ C.T


# ---- the character
class Body(NamedTuple):
    name: str
    parent: int
    mass: float  # kg, the JSON's
    geom_mass: float  # kg, of the geoms at the body's density
    com: np.ndarray  # (3,) their world frame, T-pose: the body's origin
    inertia: np.ndarray  # (3, 3) about the center of mass, their world axes
    geoms: list  # dicts: name, type (MuJoCo), size (MuJoCo), pos and rot (their world frame), friction
    json_pos: np.ndarray  # (3,)


class Joint(NamedTuple):
    name: str
    type: str  # "ball" or "hinge"
    parent: int
    child: int
    anchor: np.ndarray  # (3,) their world frame, T-pose
    axis: np.ndarray | None  # hinge axis, their world frame
    range: tuple[float, float] | None  # hinge range (rad)
    euler_limits: np.ndarray | None  # ball: (3, 2) lo / hi (rad) of the AMotor's Euler angles about their x, y, z axes
    damping: float


def geom_mass(g: dict, density: float) -> tuple[float, np.ndarray, str, list[float]]:
    """ODE's mass and inertia of one geom at a density (dMassSetSphere / Capsule along z / Box), and its MuJoCo type and size."""
    t, s = g["GeomType"], [float(x) for x in g["Scale"]]
    if t == "Sphere":
        r = s[0]
        m = (4.0 / 3.0) * np.pi * r * r * r * density
        return m, np.diag([0.4 * m * r * r] * 3), "sphere", [r]
    if t == "Capsule":  # radius, cylinder length (without the caps), along the geom's z axis
        r, length = s[0], s[1]
        m1, m2 = np.pi * r * r * length * density, (4.0 / 3.0) * np.pi * r * r * r * density
        ia = m1 * (0.25 * r * r + length * length / 12.0) + m2 * (0.4 * r * r + 0.375 * r * length + 0.25 * length * length)
        ib = (m1 * 0.5 + m2 * 0.4) * r * r
        return m1 + m2, np.diag([ia, ia, ib]), "capsule", [r, length / 2]
    if t == "Cube":  # side lengths
        lx, ly, lz = s
        m = lx * ly * lz * density
        return m, np.diag([m / 12 * (ly * ly + lz * lz), m / 12 * (lx * lx + lz * lz), m / 12 * (lx * lx + ly * ly)]), "box", [lx / 2, ly / 2, lz / 2]
    raise ValueError(f"unsupported geom type {t!r}")


def parse_body(b: dict) -> Body:
    """One body as their loader builds it (BE:20835-20910, 18464-18490)."""
    if b["MassMode"] != "MassValue" or b["InertiaMode"] != "Density" or not b["Density"] > 0:
        raise ValueError(f"body {b['Name']}: MassMode {b['MassMode']}, InertiaMode {b['InertiaMode']}, Density {b['Density']} are not supported")
    if not np.allclose(b["Quaternion"], [0, 0, 0, 1], atol=0):
        raise ValueError(f"body {b['Name']}: a T-pose orientation other than the identity is not supported")
    geoms = sorted(b["Geoms"], key=lambda g: g["GeomID"])
    masses, inertias, out = [], [], []
    for g in geoms:
        m, inertia_g, gtype, size = geom_mass(g, float(b["Density"]))
        R = quat_xyzw_to_mat(g["Quaternion"])
        masses.append(m)
        inertias.append(R @ inertia_g @ R.T)
        out.append({"name": g["Name"], "type": gtype, "size": size, "pos": np.array(g["Position"], dtype=np.float64), "rot": R,
                    "friction": float(g["Friction"]), "collidable": bool(g["Collidable"])})  # fmt: skip
    ms = np.array(masses)
    com = ms @ np.array([g["pos"] for g in out]) / ms.sum()
    inertia = np.zeros((3, 3))
    for m, inertia_g, g in zip(ms, inertias, out):
        t = com - g["pos"]
        inertia += inertia_g + m * (t @ t * np.eye(3) - np.outer(t, t))  # parallel axes
    return Body(b["Name"], int(b["ParentBodyID"]), float(b["Mass"]), float(ms.sum()), com, inertia, out, np.array(b["Position"], dtype=np.float64))


def parse_joint(j: dict) -> Joint:
    if not np.allclose(j["Quaternion"], [0, 0, 0, 1], atol=0) or not np.allclose(j.get("EulerAxisLocalRot") or [0, 0, 0, 1], [0, 0, 0, 1], atol=0):
        raise ValueError(f"joint {j['Name']}: rotated joint frames are not supported")
    lo, hi = np.deg2rad(np.asarray(j["AngleLoLimit"], dtype=np.float64)), np.deg2rad(np.asarray(j["AngleHiLimit"], dtype=np.float64))
    common = dict(name=j["Name"], parent=int(j["ParentBodyID"]), child=int(j["ChildBodyID"]), anchor=np.array(j["Position"], dtype=np.float64),
                  damping=float(j["Damping"]))  # fmt: skip
    if j["JointType"] == "BallJoint":
        if j["EulerOrder"] != "XYZ" or lo.shape != (3,) or hi.shape != (3,):
            raise ValueError(f"joint {j['Name']}: ball joints need EulerOrder XYZ and three limits")
        return Joint(type="ball", axis=None, range=None, euler_limits=np.stack([lo, hi], -1), **common)
    if j["JointType"] == "HingeJoint":
        if len(j["EulerOrder"]) != 1 or j["EulerOrder"] not in "XYZ" or lo.shape != (1,) or hi.shape != (1,):
            raise ValueError(f"joint {j['Name']}: hinge joints need a one-axis EulerOrder and one limit")
        axis = np.eye(3)["XYZ".index(j["EulerOrder"])]  # BE:20925-20929
        return Joint(type="hinge", axis=axis, range=(float(lo[0]), float(hi[0])), euler_limits=None, **common)
    raise ValueError(f"joint {j['Name']}: unsupported type {j['JointType']!r}")


class Character(NamedTuple):
    bodies: list[Body]
    joints: list[Joint]  # joint j moves body j + 1
    ends: list[dict]  # end effectors: name, body, pos (their world frame)
    root_info_pos: np.ndarray  # (3,) their RootInfo position: where the BVH root goes in the T-pose
    gravity: np.ndarray  # (3,) their world frame
    raw: dict  # the JSON character


def parse_character(scene: dict) -> Character:
    """Their first character, checked against ndm_gym.robots.musclevae_char's names and tree."""
    fixed = scene["WorldAttr"]["FixedAttr"]
    if fixed["SimulateFPS"] != mc.SIM_HZ or fixed["SelfCollision"] or not fixed["UseHinge"] or not fixed["UseAngleLimit"]:
        raise ValueError(f"unexpected WorldAttr {fixed}")
    chars = scene["CharacterList"]["Characters"]
    same = lambda k: json.dumps(chars[0][k], sort_keys=True) == json.dumps(chars[1][k], sort_keys=True)  # noqa: E731
    if len(chars) != 2 or not all(same(k) for k in ("Bodies", "Joints", "Muscles")):
        raise ValueError("expected their character and its identical reference copy")
    raw = chars[0]
    bodies = [parse_body(b) for b in sorted(raw["Bodies"], key=lambda b: b["BodyID"])]
    joints = [parse_joint(j) for j in sorted(raw["Joints"], key=lambda j: j["JointID"])]
    if tuple(b.name for b in bodies) != mc.BODY_NAMES or tuple(b.parent for b in bodies) != mc.BODY_PARENTS:
        raise ValueError("bodies differ from ndm_gym.robots.musclevae_char")
    if tuple(j.name for j in joints) != mc.JOINT_NAMES or tuple(j.type for j in joints) != mc.JOINT_TYPES:
        raise ValueError("joints differ from ndm_gym.robots.musclevae_char")
    if any(j.child != i + 1 or j.parent != mc.BODY_PARENTS[i + 1] for i, j in enumerate(joints)):
        raise ValueError("joint j must move body j + 1 relative to its parent")
    if any(j.damping != mc.JOINT_DAMPING for j in joints):
        raise ValueError(f"joint damping other than {mc.JOINT_DAMPING}")
    if abs(sum(b.mass for b in bodies) - mc.MASS) > 1e-6:
        raise ValueError(f"total mass {sum(b.mass for b in bodies)}, expected {mc.MASS}")
    ends = [{"name": e["Name"], "body": joints[e["ParentJointID"]].child, "pos": np.array(e["Position"], dtype=np.float64)} for e in raw["EndJoints"]]
    return Character(bodies, joints, ends, np.array(raw["RootInfo"]["Position"], dtype=np.float64),
                     np.array(scene["WorldAttr"]["ChangeAttr"]["Gravity"], dtype=np.float64), raw)  # fmt: skip


def root_anchor_shift(char: Character) -> np.ndarray:
    """Their anchor update's offset of anchors on the root, in the T-pose (their world frame): minus the distance of the
    root's first geom from its center of mass, along the root's y axis (CHR:147-155, 166-179)."""
    root = char.bodies[0]
    return np.array([0.0, -np.linalg.norm(root.geoms[0]["pos"] - root.com), 0.0])


# ---- muscles
class Anchor(NamedTuple):
    body0: int
    body1: int
    weight0: float
    local0: np.ndarray  # (3,) body0's frame (their axes, T-pose: world axes)
    local1: np.ndarray
    tpose: np.ndarray  # (3,) their world frame
    source_body: int  # the waypoint's own body
    mass_pos: np.ndarray  # (3,) MASS's waypoint, x mirrored (their world frame)


def bind_anchors(char: Character, waypoints: list[dict]) -> list[Anchor]:
    """One muscle's anchors as their loader binds them (LDR:238-375), with the root shift of the anchor update."""
    names = [b.name for b in char.bodies]
    parent = [b.parent for b in char.bodies]
    # each body's reference point: the root's center of mass, else the joint to its parent (LDR:285-288)
    ref = np.array([char.bodies[0].com] + [char.joints[b - 1].anchor for b in range(1, len(names))])
    shift = root_anchor_shift(char)
    out = []
    for i, w in enumerate(waypoints):
        q = np.array(w["Position"], dtype=np.float64)
        q[0] = -q[0]  # LDR:244
        own = names.index(mass_body(w["BodyName"]))
        b0 = b1 = own
        w0 = 1.0
        if 0 < i < len(waypoints) - 1:
            d = np.linalg.norm(ref - q, axis=-1)
            k = int(np.argmin(d))
            if d[k] < BLEND_RADIUS:
                b0 = b1 = k
                if parent[k] != -1:
                    b1 = parent[k]
                    a0, a1 = 1.0 / np.sqrt(d[k]), 1.0 / np.sqrt(d[b1])
                    w0 = a0 / (a0 + a1)
        p0, p1 = q + (shift if b0 == 0 else 0.0), q + (shift if b1 == 0 else 0.0)
        out.append(Anchor(b0, b1, float(w0), p0 - char.bodies[b0].com, p1 - char.bodies[b1].com, w0 * p0 + (1.0 - w0) * p1, own, q))
    return out


def muscle_part(char: Character, insertion_body: int) -> int:
    name = char.bodies[insertion_body].name
    return next((p for p, bodies in PART_BODIES.items() if name in bodies), 0)


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())


def check_against_full(char: Character, anchors: list[list[Anchor]], full: dict) -> int:
    """The anchors' bodies and weights against muscle_character_full.json (their loader's dump); returns the number of
    blended anchors. Raises on any difference."""
    names = [b.name for b in char.bodies]
    ms = sorted(full["CharacterList"]["Characters"][0]["Muscles"], key=lambda m: m["muscleID"])
    raw = sorted(char.raw["Muscles"], key=lambda m: m["muscleID"])
    if [m["Name"] for m in ms] != [m["Name"] for m in raw] or [len(m["Waypoints"]) for m in ms] != [len(a) for a in anchors]:
        raise ValueError(f"{FULL_FILE}: muscles differ from {CHARACTER_FILE}'s")
    errors, blended = [], 0
    for m, am in zip(ms, anchors):
        for w, a in zip(m["Waypoints"], am):
            b1 = names.index(w["Body1"]) if w["Body1"] != "None" else a.body0
            blended += a.body1 != a.body0
            if names.index(w["Body0"]) != a.body0 or b1 != a.body1 or abs(w["Weight0"] - a.weight0) > FULL_WEIGHT_TOL:
                errors.append(f"{m['Name']} waypoint {w['waypointID']}: theirs {w['Body0']} / {w['Body1']} {w['Weight0']:.7f}, "
                              f"ours {names[a.body0]} / {names[a.body1]} {a.weight0:.7f}")  # fmt: skip
    if errors:
        raise ValueError(f"{len(errors)} anchors differ from {FULL_FILE}:\n  " + "\n  ".join(errors[:20]))
    return blended


# ---- outputs
def fmt(xs) -> str:
    return " ".join("0" if abs(float(x)) < 1e-15 else f"{float(x):.12g}" for x in np.ravel(xs))


def write_mjcf(char: Character) -> str:
    """The character in our frame: body origins at the centers of mass, frames aligned with the world in the T-pose."""
    children = {i: [b for b in range(len(char.bodies)) if char.bodies[b].parent == i] for i in range(len(char.bodies))}
    lines = [
        '<mujoco model="musclevae_character">',
        f"  <!-- MuscleVAE's character (wangchek/MuscleVAE_ @ {mc.SOURCE_COMMIT[:7]}), written by tools/convert_musclevae.py: see its docstring",
        "       for the conventions. Body origins are the centers of mass; ball joints carry no limits (their per-axis Euler",
        "       limits are in character.json); the geoms do not collide with each other. -->",
        '  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>',
        f'  <option timestep="{1.0 / mc.SIM_HZ!r}" gravity="{fmt(C @ char.gravity)}"/>',
        "  <default>",
        '    <default class="musclevae">',
        f'      <joint damping="{mc.JOINT_DAMPING:g}" armature="0"/>',
        '      <geom contype="1" conaffinity="0" condim="3"/>',
        '      <site size="0.01" group="4"/>',
        "    </default>",
        "  </default>",
        "  <worldbody>",
    ]

    def body(i: int, depth: int):
        b, ind = char.bodies[i], "  " * depth
        pos = C @ (b.com if b.parent < 0 else b.com - char.bodies[b.parent].com)
        cls = ' childclass="musclevae"' if b.parent < 0 else ""
        lines.append(f'{ind}<body name="{b.name}" pos="{fmt(pos)}"{cls}>')
        if b.parent < 0:
            lines.append(f'{ind}  <freejoint name="root"/>')
        else:
            j = char.joints[i - 1]
            joint = f'{ind}  <joint name="{j.name}" type="{j.type}" pos="{fmt(C @ (j.anchor - b.com))}"'
            lines.append(joint + ("/>" if j.type == "ball" else f' axis="{fmt(C @ j.axis)}" range="{fmt(j.range)}"/>'))
        inr = C @ b.inertia @ C.T
        full = fmt([inr[0, 0], inr[1, 1], inr[2, 2], inr[0, 1], inr[0, 2], inr[1, 2]])
        lines.append(f'{ind}  <inertial pos="0 0 0" mass="{b.mass!r}" fullinertia="{full}"/>')
        for g in b.geoms:
            if not g["collidable"]:
                raise ValueError(f"geom {g['name']}: non-collidable geoms are not supported")
            lines.append(f'{ind}  <geom name="{g["name"]}" type="{g["type"]}" size="{fmt(g["size"])}" pos="{fmt(C @ (g["pos"] - b.com))}" '
                         f'quat="{fmt(mat_to_quat(C @ g["rot"]))}" friction="{g["friction"]!r} 0.005 0.0001"/>')  # fmt: skip
        for e in char.ends:
            if e["body"] == i:
                lines.append(f'{ind}  <site name="end_{e["name"]}" pos="{fmt(C @ (e["pos"] - b.com))}"/>')
        for c in children[i]:
            body(c, depth + 1)
        lines.append(f"{ind}</body>")

    body(0, 2)
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def character_json(char: Character, sources: dict) -> dict:
    zb = lambda v: (C @ np.asarray(v)).tolist()  # noqa: E731
    bodies = [{"name": b.name, "parent": b.parent, "mass": b.mass, "geom_mass": b.geom_mass, "com_tpose": zb(b.com),
               "inertia": (C @ b.inertia @ C.T).tolist(), "source_position": b.json_pos.tolist(),
               "geoms": [{"name": g["name"], "type": g["type"], "size": g["size"], "pos": zb(g["pos"] - b.com),
                          "quat_wxyz": mat_to_quat(C @ g["rot"]).tolist(), "friction": g["friction"]} for g in b.geoms]}
              for b in char.bodies]  # fmt: skip
    joints = []
    for i, j in enumerate(char.joints):
        d = {"name": j.name, "type": j.type, "parent": j.parent, "child": j.child, "anchor_tpose": zb(j.anchor),
             "pos": zb(j.anchor - char.bodies[j.child].com), "damping": j.damping, "qpos_adr": 7 + mc.JOINT_QPOS_ADR[i],
             "qpos_size": mc.JOINT_QPOS_SIZE[i]}  # fmt: skip
        if j.type == "hinge":
            d.update(axis=zb(j.axis), range=list(j.range))
        else:
            d.update(euler_limits={"lo": j.euler_limits[:, 0].tolist(), "hi": j.euler_limits[:, 1].tolist(), "axes_theirs": ["x", "y", "z"],
                                   "axes_ours": ["y", "z", "x"]})  # fmt: skip
        joints.append(d)
    return {
        "format": "ndm_gym/musclevae-character/1", "source": sources, "frame": {"up": "z", "forward": "x", "from_theirs": C.tolist()},
        "sim_hz": mc.SIM_HZ, "control_hz": mc.CONTROL_HZ, "gravity": zb(char.gravity), "mass": sum(b.mass for b in char.bodies),
        "bodies": bodies, "joints": joints, "nq": 7 + mc.JOINT_QPOS_DIM,
        "end_effectors": [{"name": e["name"], "body": e["body"], "pos_tpose": zb(e["pos"]), "pos": zb(e["pos"] - char.bodies[e["body"]].com)}
                          for e in char.ends],
        "bvh_root_tpose": zb(char.root_info_pos), "root_anchor_shift_tpose": zb(root_anchor_shift(char)),
        "notes": {
            "frames": "Z-up, facing +x: theirs (x, y, z) -> ours (z, x, y). Body origins at the centers of mass of their geoms; body frames aligned "
                      "with the world in the T-pose (all positions here are in the T-pose).",
            "inertia": "mass: the JSON's; inertia about the center of mass: of the geoms at the body's density (1000 kg/m^3), not rescaled to the "
                       "mass, as their loader leaves it (Mass.mass is set without dMassAdjust).",
            "ball_limits": "Their simulator limits each ball joint's three ODE AMotor Euler angles (Euler mode: their x axis anchored to the child, "
                           "z to the parent). MuJoCo's ball joints only take a cone limit, so the MJCF's ball joints are unlimited; euler_limits "
                           "keeps theirs (rad).",
            "self_collision": "off: the geoms have contype 1, conaffinity 0.",
        },
    }  # fmt: skip


def muscles_json(char: Character, muscles: list[dict], anchors: list[list[Anchor]], kp_ratio: np.ndarray, force: np.ndarray, sources: dict,
                 blended: int) -> dict:  # fmt: skip
    groups = []
    for m in muscles:  # by first appearance (LDR:143-145)
        if m["group_name"] not in groups:
            groups.append(m["group_name"])
    out = []
    for m, am, r, f in zip(muscles, anchors, kp_ratio, force):
        tpose = np.array([a.tpose for a in am])
        out.append({
            "name": m["Name"], "group": groups.index(m["group_name"]), "part": muscle_part(char, am[-1].source_body), "f0": float(m["f0"]),
            "lm": float(m["lm"]), "lt": float(m["lt"]), "kp_ratio": float(r), "kp": float(m["f0"]) * float(r),
            "kd": {k: float(damping_gain(float(m["f0"]) * float(r), p)) for k, p in MUSCLE_MODELS.items()}, "force_file": float(f),
            "f0max": F0MAX_SCALE * float(f), "rest_length": polyline_length(tpose),
            "anchors": [{"body0": a.body0, "local0": (C @ a.local0).tolist(), "body1": a.body1, "local1": (C @ a.local1).tolist(),
                         "weight0": a.weight0, "tpose": (C @ a.tpose).tolist(), "source_body": a.source_body} for a in am],
        })  # fmt: skip
    models = {k: dataclasses.asdict(v) for k, v in MUSCLE_MODELS.items()}
    return {
        "format": "ndm_gym/musclevae-muscles/1", "source": sources, "frame": {"up": "z", "forward": "x", "from_theirs": C.tolist()},
        "bodies": list(mc.BODY_NAMES), "parts": list(mc.PART_NAMES), "groups": groups, "models": models, "blend_radius": BLEND_RADIUS,
        "counts": {"muscles": len(out), "anchors": sum(len(a) for a in anchors), "blended_anchors": blended,
                   "root_anchors": sum(a.body0 == 0 and a.body1 == 0 for am in anchors for a in am)},
        "muscles": out,
        "notes": {
            "anchors": "Anchor position = weight0 (R_body0 local0 + p_body0) + (1 - weight0) (R_body1 local1 + p_body1), with body origins at the "
                       "centers of mass; the force acts on body0 at that position. body1 = body0 and weight0 = 1 unless the anchor is blended. "
                       "tpose: the T-pose position; source_body: the waypoint's own body.",
            "constants": "kp = f0 x kp_ratio (manual_muscle_kp4.txt); f0max = 1.5 x force_file (MuscleVAE_force.txt): the released code's, used "
                         "by both models; kd per model. f0 (MASS's) weighs the part fatigue state. rest_length: the T-pose length.",
            "models": "released: the released code's behavior (the default); paper: kd 0.1 kp, F 0.01, R 0.002, length rates per dt (App. C.1, "
                      "Eq. 24). See ndm_gym.robots.muscles.",
            "unused": "pen_angle (0), lmax, activation, fatigue and flexibility of the JSON are not used by their muscle model.",
        },
    }  # fmt: skip


# ---- clips
class Bvh(NamedTuple):
    names: list[str]
    parents: list[int]
    offsets: np.ndarray  # (J, 3)
    channels: list[list[str]]
    frame_time: float
    values: np.ndarray  # (T, total channels)


ROOT_CHANNELS = ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"]
JOINT_CHANNELS = ["Xrotation", "Yrotation", "Zrotation"]


def parse_bvh(path: str) -> Bvh:
    names, parents, offsets, channels, stack = [], [], [], [], []
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            tok = line.split()
            if not tok or tok[0] == "HIERARCHY":
                continue
            if tok[0] in ("ROOT", "JOINT"):
                names.append(tok[1])
                parents.append(stack[-1] if stack else -1)
                offsets.append(None)
                channels.append(None)
            elif tok[0] == "{":
                stack.append(len(names) - 1)
            elif tok[0] == "}":
                stack.pop()
            elif tok[0] == "OFFSET":
                offsets[-1] = [float(x) for x in tok[1:4]]
            elif tok[0] == "CHANNELS":
                channels[-1] = tok[2:]
                if int(tok[1]) != len(channels[-1]):
                    raise ValueError(f"{path}: malformed {line.strip()!r}")
            elif tok[0] == "MOTION":
                break
            else:  # End Site among others: none in their clips
                raise ValueError(f"{path}: unsupported line {line.strip()!r}")
        frames = int(next(lines).split(":")[1])
        frame_time = float(next(lines).split(":")[1])
        values = np.loadtxt(lines, dtype=np.float64, ndmin=2)
    if stack or not names or any(o is None for o in offsets):
        raise ValueError(f"{path}: malformed hierarchy")
    if channels[0] != ROOT_CHANNELS or any(c != JOINT_CHANNELS for c in channels[1:]):
        raise ValueError(f"{path}: channels must be {ROOT_CHANNELS} at the root and {JOINT_CHANNELS} elsewhere")
    if values.shape != (frames, 6 + 3 * (len(names) - 1)):
        raise ValueError(f"{path}: motion data has shape {values.shape}, expected ({frames}, {6 + 3 * (len(names) - 1)})")
    return Bvh(names, parents, np.array(offsets), channels, frame_time, values)


def bvh_fk(bvh: Bvh, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global joint orientations (T, J, 3, 3) and positions (T, J, 3) of the given frames, as pymotionlib computes them
    (the root's OFFSET ignored, BE:13235-13236)."""
    v = bvh.values[frames]
    J = len(bvh.names)
    G, X = np.empty((len(frames), J, 3, 3)), np.empty((len(frames), J, 3))
    for j in range(J):
        p = bvh.parents[j]
        col = 3 + 3 * j
        L = euler_xyz_deg(v[:, col : col + 3])
        if p < 0:
            G[:, j], X[:, j] = L, v[:, 0:3]
        else:
            G[:, j] = G[:, p] @ L
            X[:, j] = X[:, p] + G[:, p] @ bvh.offsets[j]
    return G, X


def bvh_bodies(char: Character, bvh: Bvh) -> list[int]:
    """The BVH joint of each body: the root, else the joint named like the body's parent joint; checks the tree."""
    idx = [0] + [bvh.names.index(j.name) if j.name in bvh.names else -1 for j in char.joints]
    if bvh.parents[0] != -1 or -1 in idx or len(set(idx)) != len(idx) or len(bvh.names) != len(idx):
        raise ValueError(f"the BVH joints {bvh.names} do not match the character's joints")
    if any(bvh.parents[idx[b]] != idx[char.bodies[b].parent] for b in range(1, len(idx))):
        raise ValueError("the BVH hierarchy differs from the character's")
    return idx


def their_placement(char: Character, G: np.ndarray, X: np.ndarray, idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Body positions and orientations (their world frame) as their dataset places them (BE:18776-18800, 18846-18860)."""
    T, N = len(G), len(char.bodies)
    pos, rot = np.empty((T, N, 3)), np.empty((T, N, 3, 3))
    rot[:, 0] = G[:, idx[0]]
    pos[:, 0] = X[:, idx[0]] + G[:, idx[0]] @ (char.bodies[0].com - char.root_info_pos)
    for b in range(1, N):
        j = char.joints[b - 1]
        rot[:, b] = G[:, idx[b]]
        pos[:, b] = X[:, idx[b]] - G[:, idx[b]] @ (j.anchor - char.bodies[b].com)
    return pos, rot


def hinge_angle(L: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The twist about a unit axis of rotations (..., 3, 3): the angle of the closest rotation about it, in [-pi, pi]."""
    q = mat_to_quat(L)
    return 2.0 * np.arctan2(q[..., 1:] @ axis, q[..., 0])


def joint_space(char: Character, G: np.ndarray, rot0: np.ndarray, pos0: np.ndarray, idx: list[int]):
    """The clip in joint space for the character's skeleton (their world frame): body orientations and positions, each
    joint's rotation relative to its parent, and for each hinge the dropped rotation angle."""
    T, N = len(G), len(char.bodies)
    rot, pos = np.empty((T, N, 3, 3)), np.empty((T, N, 3))
    rot[:, 0], pos[:, 0] = rot0, pos0
    local, theta, dropped = {}, {}, {}
    for b in range(1, N):
        j, p = char.joints[b - 1], char.bodies[b].parent
        L = np.swapaxes(rot[:, p], -1, -2) @ G[:, idx[b]]
        if j.type == "hinge":
            theta[b] = hinge_angle(L, j.axis)
            Lh = axis_angle(j.axis, theta[b])
            dropped[b] = 2.0 * np.arccos(np.clip(np.abs(mat_to_quat(np.swapaxes(Lh, -1, -2) @ L)[..., 0]), 0.0, 1.0))
            L = Lh
        local[b] = L
        rot[:, b] = rot[:, p] @ L
        pos[:, b] = pos[:, p] + rot[:, p] @ (j.anchor - char.bodies[p].com) + rot[:, b] @ (char.bodies[b].com - j.anchor)
    return rot, pos, local, theta, dropped


def convert_clip(char: Character, bvh: Bvh, name: str) -> tuple[dict, dict]:
    """One BVH as clip arrays (float32, our frame) and the sidecar's fields."""
    fps = round(1.0 / bvh.frame_time)  # as pymotionlib rounds it (BE:13209)
    if fps % CLIP_FPS:
        raise ValueError(f"{name}: {fps} fps is not a multiple of {CLIP_FPS}")
    step = fps // CLIP_FPS
    frames = np.arange(0, len(bvh.values), step)
    idx = bvh_bodies(char, bvh)
    G, X = bvh_fk(bvh, frames)
    their_pos, their_rot = their_placement(char, G, X, idx)
    rot, pos, local, theta, dropped = joint_space(char, G, their_rot[:, 0], their_pos[:, 0], idx)
    qpos = []
    for b in range(1, len(char.bodies)):
        qpos.append(theta[b][:, None] if char.joints[b - 1].type == "hinge" else mat_to_quat(to_z_up_rot(local[b])))
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    arrays = {"root_pos": f32(pos[:, 0] @ C.T), "root_quat_wxyz": f32(mat_to_quat(to_z_up_rot(rot[:, 0]))),
              "joint_qpos": f32(np.concatenate(qpos, -1)), "body_pos": f32(their_pos @ C.T),
              "body_quat_wxyz": f32(mat_to_quat(to_z_up_rot(their_rot)))}  # fmt: skip
    hinges = {}
    for b, th in theta.items():
        j = char.joints[b - 1]
        lo, hi = j.range
        out = np.maximum(lo - th, th - hi)
        hinges[j.name] = {"dropped_rotation_max": float(dropped[b].max()), "dropped_rotation_mean": float(dropped[b].mean()),
                          "frames_outside_range": int((out > 0).sum()), "outside_range_max": float(max(out.max(), 0.0))}  # fmt: skip
    dist = np.linalg.norm(pos - their_pos, axis=-1)
    side = {
        "source": "musclevae", "fps": float(CLIP_FPS), "frames": len(frames), "source_fps": fps, "source_frames": len(bvh.values),
        "source_step": step, "root_body": mc.BODY_NAMES[0], "body_names": list(mc.BODY_NAMES), "joint_names": list(mc.JOINT_NAMES),
        "joint_types": list(mc.JOINT_TYPES), "joint_qpos_adr": list(mc.JOINT_QPOS_ADR), "quat_convention": "wxyz",
        "frame": {"up": "z", "forward": "x", "from_theirs": C.tolist()},
        "hinges": hinges, "placement_distance": {"max": float(dist.max()), "mean": float(dist.mean()),
                                                 "max_per_body": dict(zip(mc.BODY_NAMES, dist.max(0).tolist()))},
        "notes": {
            "joint_qpos": "the MJCF's qpos after the free joint: per joint in joint order, a ball joint's rotation relative to its parent "
                          "(w, x, y, z) or a hinge angle (rad). With root_pos and root_quat_wxyz (the pelvis's center of mass and "
                          "orientation) it is the full qpos.",
            "conversion": "ball joints: the BVH's global orientation of the child; hinges: the twist about the axis of the BVH's rotation relative "
                          "to the converted parent (hinges.dropped_rotation: the rest, rad). Positions follow the character's skeleton.",
            "body_pos": "body_pos and body_quat_wxyz: their dataset's placement of each body from its BVH joint (dislocated by the BVH skeleton's "
                        "offsets: placement_distance, m), as MuscleVAE's own reference states.",
        },
    }  # fmt: skip
    return arrays, side


def write_clip(out_dir: str, name: str, arrays: dict, side: dict) -> None:
    out = {**arrays, "fps": np.asarray(CLIP_FPS, dtype=np.float32)}
    np.savez(os.path.join(out_dir, f"{name}.npz"), **out)
    side = {**side, "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in out.items()}}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        f.write(json.dumps(side, indent=1) + "\n")


# ---- conversion and checks
def read_sources(src_dir: str) -> dict:
    """The input files, each checked against its SHA-256."""
    for rel, want in SOURCE_SHA256.items():
        path = os.path.join(src_dir, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing; run tools/download_musclevae.sh")
        if sha256(path) != want:
            raise ValueError(f"{path} differs from MuscleVAE @ {mc.SOURCE_COMMIT[:7]}; remove {src_dir} and run tools/download_musclevae.sh")
    return {"repo": mc.SOURCE_REPO, "commit": mc.SOURCE_COMMIT, "files": dict(SOURCE_SHA256)}


def check_config(path: str) -> None:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    bad = {k: (cfg.get(k), v) for k, v in CONFIG_EXPECTED.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"{path}: entries differ from what this conversion assumes (theirs, expected): {bad}")


def load_mjcf_model(path: str):
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    nv = 6 + 3 * mc.JOINT_TYPES.count("ball") + mc.JOINT_TYPES.count("hinge")
    if (m.nbody, m.njnt, m.nq, m.nv) != (mc.NUM_BODIES + 1, mc.NUM_JOINTS + 1, 7 + mc.JOINT_QPOS_DIM, nv):
        raise ValueError(f"{path}: unexpected model size nbody {m.nbody}, njnt {m.njnt}, nq {m.nq}, nv {m.nv}")
    bodies, joints = [m.body(i + 1).name for i in range(mc.NUM_BODIES)], [m.joint(i + 1).name for i in range(mc.NUM_JOINTS)]
    if bodies != list(mc.BODY_NAMES) or joints != list(mc.JOINT_NAMES):
        raise ValueError(f"{path}: body or joint order differs from ndm_gym.robots.musclevae_char")
    if list(m.jnt_qposadr[1:]) != [7 + a for a in mc.JOINT_QPOS_ADR]:
        raise ValueError(f"{path}: joint qpos addresses differ from ndm_gym.robots.musclevae_char.JOINT_QPOS_ADR")
    return m


def convert_all(src_dir: str = SRC_DIR, out_dir: str = mc.DATA_DIR) -> None:
    sources = read_sources(src_dir)
    check_config(os.path.join(src_dir, CONFIG_FILE))
    with open(os.path.join(src_dir, CHARACTER_FILE)) as f:
        char = parse_character(json.load(f))
    with open(os.path.join(src_dir, FULL_FILE)) as f:
        full = json.load(f)
    muscles = sorted(char.raw["Muscles"], key=lambda m: m["muscleID"])
    kp_ratio, force = np.loadtxt(os.path.join(src_dir, KP_FILE)), np.loadtxt(os.path.join(src_dir, FORCE_FILE))
    if kp_ratio.shape != (len(muscles),) or force.shape != (len(muscles),) or len(muscles) != mc.NUM_MUSCLES:
        raise ValueError(f"{len(muscles)} muscles, {kp_ratio.shape} kp ratios, {force.shape} forces; expected {mc.NUM_MUSCLES} each")
    anchors = [bind_anchors(char, m["Waypoints"]) for m in muscles]
    blended = check_against_full(char, anchors, full)
    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    with open(os.path.join(out_dir, "character.xml"), "w") as f:
        f.write(write_mjcf(char))
    with open(os.path.join(out_dir, "character.json"), "w") as f:
        f.write(json.dumps(character_json(char, sources), indent=1) + "\n")
    with open(os.path.join(out_dir, "muscles.json"), "w") as f:
        f.write(json.dumps(muscles_json(char, muscles, anchors, kp_ratio, force, sources, blended), indent=1) + "\n")
    load_mjcf_model(os.path.join(out_dir, "character.xml"))
    mass = sum(b.mass for b in char.bodies)
    print(f"character: {len(char.bodies)} bodies, {len(char.joints)} joints, {mass:.1f} kg -> {out_dir}/character.xml, character.json")
    print(f"muscles: {len(muscles)}, {sum(len(a) for a in anchors)} anchors ({blended} blended, as in {FULL_FILE}) -> {out_dir}/muscles.json")
    for name, rel in CLIP_FILES.items():
        arrays, side = convert_clip(char, parse_bvh(os.path.join(src_dir, rel)), name)
        side.update(source_file=rel, source_sha256=SOURCE_SHA256[rel], source_repo=mc.SOURCE_REPO, source_commit=mc.SOURCE_COMMIT)
        write_clip(os.path.join(out_dir, "clips"), name, arrays, side)
        worst = max(side["hinges"].items(), key=lambda kv: kv[1]["dropped_rotation_max"])
        print(f"clip {name}: {side['frames']} frames at {CLIP_FPS} Hz from {side['source_frames']} at {side['source_fps']}; largest dropped "
              f"hinge rotation {np.rad2deg(worst[1]['dropped_rotation_max']):.1f} deg ({worst[0]}); bodies within "
              f"{100 * side['placement_distance']['max']:.1f} cm of their placement")  # fmt: skip


def check(data_dir: str = mc.DATA_DIR) -> list[dict]:
    """The converted assets with MuJoCo's forward kinematics: the MJCF's T-pose and masses against character.json, the
    muscle model's T-pose lengths against the rest lengths, and per clip the bodies of joint_qpos against their placement
    (orientations equal but for the hinges' children; positions within the dislocation). Returns per-clip maxima; raises
    on any failure."""
    import mujoco
    import torch

    from ndm_gym.motion.sources import musclevae as clips_src
    from ndm_gym.robots.muscles import MuscleModel, muscle_params

    char = mc.load_character(os.path.join(data_dir, "character.json"))
    table = mc.load_muscle_table(os.path.join(data_dir, "muscles.json"))
    with open(os.path.join(data_dir, "muscles.json")) as f:
        tpose = np.array([a["tpose"] for mm in json.load(f)["muscles"] for a in mm["anchors"]])
    m = load_mjcf_model(os.path.join(data_dir, "character.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    errors = []
    com = np.array([b["com_tpose"] for b in char["bodies"]])
    if np.abs(d.xpos[1:] - com).max() > CHECK_TOL or np.abs(d.xquat[1:] - [1, 0, 0, 0]).max() > CHECK_TOL:
        errors.append("the MJCF's T-pose differs from character.json")
    if abs(m.body_mass.sum() - mc.MASS) > 1e-6:  # the JSON's masses are float32
        errors.append(f"the MJCF's mass is {m.body_mass.sum()}")
    model = MuscleModel(table, muscle_params("released"), dtype=torch.float64)
    rot = torch.tensor(d.xmat[1:].reshape(-1, 3, 3)).unsqueeze(0)
    anchor = model.anchor_positions(torch.tensor(d.xpos[1:]).unsqueeze(0), rot)
    rel = np.abs(model.lengths(anchor)[0].numpy() / table.rest_length - 1).max()
    if np.abs(anchor[0].numpy() - tpose).max() > CHECK_TOL or rel > 1e-9:
        errors.append(f"T-pose anchors differ from muscles.json (length error {rel:.1e})")
    hinge_child = {mc.JOINT_NAMES.index(n) + 1 for n in mc.HINGE_JOINTS}
    rows = []
    for name in CLIP_FILES:
        clip = clips_src.load_clip(os.path.join(data_dir, "clips", f"{name}.npz"))
        ref_pos, ref_quat = clips_src.load_placement(os.path.join(data_dir, "clips", f"{name}.npz"))
        qpos = torch.cat((clip.root_p, clip.root_quat, clip.joints), -1).double().numpy()
        xpos, xquat = np.empty((len(qpos), mc.NUM_BODIES, 3)), np.empty((len(qpos), mc.NUM_BODIES, 4))
        for t in range(len(qpos)):
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            xpos[t], xquat[t] = d.xpos[1:], d.xquat[1:]
        ang = quat_angle(xquat, ref_quat.double().numpy())
        dist = np.linalg.norm(xpos - ref_pos.double().numpy(), axis=-1)
        keep = [b for b in range(mc.NUM_BODIES) if b not in hinge_child]
        row = {"clip": name, "frames": len(qpos), "orientation_max_rad": float(ang[:, keep].max()),
               "hinge_child_orientation_max_rad": float(ang[:, sorted(hinge_child)].max()), "root_distance_max_m": float(dist[:, 0].max()),
               "distance_max_m": float(dist.max()), "distance_mean_m": float(dist.mean())}  # fmt: skip
        rows.append(row)
        if row["orientation_max_rad"] > 1e-5 or row["root_distance_max_m"] > 1e-5:
            errors.append(f"{name}: orientation {row['orientation_max_rad']:.1e} rad, root {row['root_distance_max_m']:.1e} m from their placement")
    if errors:
        raise AssertionError("\n".join(errors))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-dir", default=SRC_DIR, help="MuscleVAE's repository as tools/download_musclevae.sh places it")
    ap.add_argument("--out-dir", default=mc.DATA_DIR)
    ap.add_argument("--check", action="store_true", help="check the converted assets with MuJoCo's forward kinematics")
    a = ap.parse_args()
    if not a.check:
        convert_all(a.src_dir, a.out_dir)
        return
    print(f"{'clip':12s} {'frames':>6s} {'orient. max (rad)':>17s} {'hinge children (rad)':>20s} {'root (m)':>9s} {'bodies max (m)':>14s} "
          f"{'mean (m)':>9s}")
    for r in check(a.out_dir):
        print(f"{r['clip']:12s} {r['frames']:6d} {r['orientation_max_rad']:17.2e} {r['hinge_child_orientation_max_rad']:20.3f} "
              f"{r['root_distance_max_m']:9.1e} {r['distance_max_m']:14.4f} {r['distance_mean_m']:9.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
