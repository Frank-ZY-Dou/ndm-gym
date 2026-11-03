"""Render MuscleVAE's muscle-actuated character (ndm_gym.robots.musclevae_char) in MuscleVAE's own look, with MuJoCo (EGL):
MASS's bone meshes on the 23 bodies, the 284 muscles as tubes through their anchors (ndm_gym.robots.muscles), a checker
floor, a sky gradient, a following camera, optionally the reference beside it as a blue skeleton, and the small NDM Gym
mark of tools/render.py.

The look follows MuscleVAE's project page and videos (https://pku-mocca.github.io/MuscleVAE-page, rendered in Unity at
1280 x 720), measured on frames of its walk, run and random-generation videos:
* bones: MASS's skeleton meshes (Lee et al. 2019; lsw9021/MASS @ e44c2ef, Apache-2.0, fetched by
  tools/download_mass_meshes.sh into data/mass), off-white. MuscleVAE's character is MASS's skeleton (its bodies and
  joints sit where MASS's do), and MASS's OBJs are in MASS's T-pose world frame in centimetres, so a vertex v of a
  body's mesh is C v / 100 - com_tpose in that body's frame, with C and com_tpose the conversion's (Y-up to Z-up, body
  origins at the centers of mass; tools/convert_musclevae.py). Each mesh goes on the body it lies on in the T-pose.
* muscles: tubes along each muscle's anchor polyline, from the body poses by MuscleModel.anchor_positions, colored and
  sized by activation a in [0, 1] with their renderer's rule: color (1, 1 - a, 1 - a), white at rest and red at full
  activation, width 1 + 3a times the resting width (MuscleVAE_ @ 54c001c, ModifyODESrc/drawstuff/src/drawstuffWrapper.cpp:
  850-859); the resting width is a 3.5 mm tube, about one pixel at the default framing, as their thin lines, and the
  anchors are small grey spheres, the dots along their lines. Without activations (reference motions) every muscle is
  at rest. Their lines' widths are in pixels (glLineWidth): --muscle_scale widens the tubes and the anchors for a video
  shown at another scale, by (--distance / 2.42) x (720 / the height it is shown at), e.g. 1.85 for 1920 x 1080 at
  --distance 2.8 shown 800 px wide.
* reference (--ref): the reference's bones in their blue beside the character, 1.5 m to the camera's right of its
  pelvis (anchored to the character's position, so only the poses differ), the camera centered between the two, as in
  their tracking videos.
* floor: 0.5 m checker squares; lit, their squares are (227, 241, 241) and (208, 221, 221), shadow cores (100, 117, 140).
* sky: their sky's color by elevation above the horizon, sampled up to 17 degrees (the top of their frames; constant
  above), as a skybox: from (214, 241, 243) at the horizon to (110, 134, 171).
* light: a warm key light from the camera's side, 77 degrees high and a little to the right, so that shadows fall
  behind the character and to the left, about 0.4 m long for the head, as theirs; their sky's light as directional
  lights without shadows from above (bluish), the horizon and the ground, and an ambient. Lit floor keeps the squares'
  colors and shadows reach their darkness. Their shadows are solid: the character's collision shapes (character.json)
  cast them, drawn all but transparent.
* camera: their vertical field of view, 60 degrees, pitched 13.4 degrees down, 2.42 m from the point 0.78 m above the
  pelvis: 1.34 m high and 2.36 m from the character (the floor's vanishing lines and squares and the character's height
  in their frames); it follows the pelvis (a 0.2 s lag) at a fixed heading along the floor's grid, as theirs. --distance
  moves it along its line of sight, e.g. for room around jumps and the reference.
* the NDM Gym mark of tools/render.py (draw_mark): its font, size and place.
MuJoCo renders it rather than their Unity: the EGL renderer of tools/render.py, which draws the meshes, the shadows and an
arbitrary sky texture, with the colors above matched under its lighting.

Inputs, one of:
* --clip NAME: a converted reference clip (data/musclevae/clips/NAME.npz, ndm_gym.motion.sources.musclevae), the body
  poses by musclevae_char.forward_kinematics;
* --states FILE.npz: recorded states as ndm_gym.sim.rigid.muscle_sim.MuscleSim gives them for one environment, per step:
  body_state (T, 23, 13 or 7), the bodies' positions (their centers of mass) and quaternions wxyz first; fps, the
  steps' rate; fatigue_state (T, 284, 3), whose first column (MA) is the activation, unless --muscles rest or none;
  ref_body_state (T, 23, 13 or 7), the reference's bodies at each step, for --ref.
The video has one frame per step (or per --stride steps); a .png output is the frame at --start. --interp k draws k frames
per step, those between two steps interpolated (positions and activations linearly, rotations by slerp), for inputs
without resets or reference switches (tools/replay_musclevae.py's without --switch); with --stride s, a frame every s / k
steps (k 3 and s 2: 30 fps from 20 Hz).

EGL opens its context on MUJOCO_EGL_DEVICE_ID, or on the first GPU: it ignores CUDA_VISIBLE_DEVICES. When
CUDA_VISIBLE_DEVICES names GPUs, this tool requires MUJOCO_EGL_DEVICE_ID to be the EGL device of one of them.

Usage: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 python tools/render_musclevae.py walk.mp4 --clip lafan_walk --start 5
       ... python tools/render_musclevae.py rollout.mp4 --states rollout.npz --ref
"""

import argparse
import os
import subprocess
import sys

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.core.rotations import mat_to_quat, quat_slerp, quat_to_mat
from ndm_gym.motion.sources import musclevae as clip_source
from ndm_gym.robots import musclevae_char as mc

MASS_DIR = os.path.join(ROOT, "data", "mass")
MASS_COMMIT = "e44c2ef184cb55d45afd16b000414c84092ae18e"
# each body's MASS mesh: the one that lies on it in the T-pose (MASS's left toes' nodes have each other's places; the
# character's bodies and these meshes agree)
BODY_MESHES = {
    "pelvis": "Pelvis", "rUpperLeg": "R_Femur", "rLowerLeg": "R_Tibia", "rFoot": "R_Talus", "FootThumbR": "R_FootThumb",
    "FootPinkyR": "R_FootPinky", "lUpperLeg": "L_Femur", "lLowerLeg": "L_Tibia", "lFoot": "L_Talus", "FootThumbL": "L_FootThumb",
    "FootPinkyL": "L_FootPinky", "lowerback": "Spine", "torso": "Torso", "Neck": "Neck", "Head": "Skull", "rClavicle": "R_Shoulder",
    "rUpperArm": "R_Humerus", "rLowerArm": "R_Radius", "rHand": "R_Hand", "lClavicle": "L_Shoulder", "lUpperArm": "L_Humerus",
    "lLowerArm": "L_Radius", "lHand": "L_Hand",
}  # fmt: skip
# their sky: elevation (degrees) -> sRGB, averaged over frames of their videos
SKY = ((0.0, (214, 241, 243)), (0.5, (211, 240, 243)), (1.0, (207, 236, 241)), (1.5, (201, 230, 238)), (2.0, (194, 224, 234)), (2.5, (188, 218, 232)),
       (3.0, (181, 211, 228)), (3.5, (175, 205, 225)), (4.0, (171, 201, 222)), (5.0, (163, 193, 218)), (6.0, (155, 184, 212)), (7.0, (147, 176, 207)),
       (8.0, (143, 170, 203)), (9.0, (136, 164, 198)), (10.0, (132, 159, 194)), (11.0, (128, 154, 191)), (12.0, (124, 149, 186)),
       (13.0, (121, 146, 183)), (14.0, (117, 142, 179)), (15.0, (115, 141, 178)), (16.0, (112, 136, 174)), (17.0, (110, 134, 171)))  # fmt: skip
SKY_SIZE = 1024  # texels per cube face
FLOOR_RGB = ((227, 241, 241), (208, 221, 221))  # lit checker squares
# lights, as fractions of the lit floor's: floor in shadow gets ambient + zenith, the measured (0.46, 0.51, 0.61)
AMBIENT = (0.28, 0.29, 0.31)
ZENITH = (0.18, 0.22, 0.30)  # from above, bluish
HORIZON = (0.55, 0.58, 0.58)  # from the horizon on the camera's side
HORIZON_SIDE = (0.08, 0.08, 0.08)  # from the horizon 120 degrees to either side
GROUND = (0.10, 0.10, 0.10)  # from below
KEY = (0.554, 0.503, 0.400)  # warm, casting the shadows: lit floor = ambient + zenith + key x sin(77 deg) = 1, the squares' own colors
KEY_ELEVATION, KEY_HEADING = 77.0, 16.0  # degrees; the rays' heading from the camera's: away from it, a little to its left
BONE_RGBA, REF_RGBA = (0.95, 0.95, 0.95, 1.0), (0.52, 0.56, 0.86, 1.0)  # off-white; their reference's blue
BONE_SPECULAR, BONE_SHININESS = 0.5, 0.5
CASTER_ALPHA = 0.004  # the collision shapes that cast the solid shadows; MuJoCo skips geoms of alpha 0, shadows included
MUSCLE_RADIUS = 0.00175  # m, at rest; (1 + 3a) times at activation a
MUSCLE_SPECULAR, MUSCLE_SHININESS = 0.3, 0.3
ANCHOR_RADIUS, ANCHOR_RGBA = 0.0025, (0.55, 0.57, 0.60, 1.0)  # m: the grey dots along their muscle lines
FOVY, PITCH, CAM_DISTANCE, LOOK_HEIGHT = 60.0, 13.4, 2.42, 0.78  # degrees, degrees, m, m
FOLLOW_TAU = 0.2  # s
REF_OFFSET = 1.5  # m, the reference's pelvis to the camera's right of the character's
def read_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """An OBJ's vertices (n, 3) and its faces as triangles (m, 3), polygons split into fans; other elements ignored."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == "f":
                idx = [int(t.split("/")[0]) for t in tok[1:]]
                if len(idx) < 3 or min(idx) < 1:
                    raise ValueError(f"{path}: unsupported face {line.strip()!r} (polygons with positive vertex indices only)")
                faces += [(idx[0] - 1, idx[i] - 1, idx[i + 1] - 1) for i in range(1, len(idx) - 1)]
    v, fc = np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)
    if len(v) == 0 or len(fc) == 0 or fc.max() >= len(v):
        raise ValueError(f"{path}: no vertices or faces, or a face indexes a missing vertex")
    return v, fc


def body_meshes(char: dict, mass_dir: str = MASS_DIR) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each body's bone mesh in the body's frame: vertices (n, 3) in m and triangles (m, 3), in BODY_NAMES order."""
    commit_file = os.path.join(mass_dir, "COMMIT")
    if not os.path.isfile(commit_file):
        raise FileNotFoundError(f"{commit_file} is missing: run tools/download_mass_meshes.sh")
    with open(commit_file) as f:
        if f.read().strip() != MASS_COMMIT:
            raise ValueError(f"{mass_dir} is not MASS @ {MASS_COMMIT[:7]}: remove it and run tools/download_mass_meshes.sh")
    C = np.array(char["frame"]["from_theirs"], dtype=np.float64)
    out = []
    for body in char["bodies"]:
        v, faces = read_obj(os.path.join(mass_dir, "data", "OBJ", BODY_MESHES[body["name"]] + ".obj"))
        out.append(((v / 100.0) @ C.T - np.array(body["com_tpose"]), faces))
    return out


def sky_cube(size: int = SKY_SIZE) -> np.ndarray:
    """Their sky as a MuJoCo skybox texture (6 size x size faces stacked: +x, -x, up, down, +z, -z as MuJoCo's builtin
    gradient lays them out, rows from top to bottom on the side faces), uint8 (6 size, size, 3)."""
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # texel centers on a face, in [-1, 1]
    v, u = np.meshgrid(-c, c, indexing="ij")  # side faces: v up (row 0 at the top), u across
    side = np.degrees(np.arctan2(v, np.sqrt(1.0 + u * u)))
    top = np.degrees(np.arctan2(1.0, np.sqrt(u * u + v * v)))
    el = np.concatenate([side, side, top, -top, side, side])
    knots = np.array([e for e, _ in SKY])
    rgb = np.stack([np.interp(el, knots, [col[k] for _, col in SKY]) for k in range(3)], -1)  # constant outside the knots
    return np.round(rgb).astype(np.uint8)


def fmt(xs) -> str:
    return " ".join(f"{float(x):.7g}" for x in np.ravel(xs))


def scene_xml(char: dict, meshes: list[tuple[np.ndarray, np.ndarray]], width: int, height: int, ref: bool) -> str:
    """The bones on 23 mocap bodies (BODY_NAMES order; the reference's prefixed ref_), each with its collision shapes as
    shadow casters, the lights (the key light on a mocap rig that follows the camera's target), the floor and the sky."""
    rgb = lambda c: fmt(np.array(c) / 255.0)  # noqa: E731
    mesh_assets = "\n".join(f'    <mesh name="{n}" vertex="{fmt(v)}" face="{fmt(f)}" smoothnormal="true" inertia="shell"/>'
                            for n, (v, f) in zip(mc.BODY_NAMES, meshes))  # fmt: skip
    bodies = []
    for prefix, material in (("", "bone"), ("ref_", "ref")) if ref else (("", "bone"),):
        for b, n in enumerate(mc.BODY_NAMES):
            casters = "".join(f'<geom type="{g["type"]}" size="{fmt(g["size"])}" pos="{fmt(g["pos"])}" quat="{fmt(g["quat_wxyz"])}" '
                              f'rgba="1 1 1 {CASTER_ALPHA}"/>' for g in char["bodies"][b]["geoms"])  # fmt: skip
            bodies.append(f'    <body name="{prefix}{n}" mocap="true"><geom type="mesh" mesh="{n}" material="{material}"/>{casters}</body>')
    horizon = "\n".join(f'    <light name="horizon{k}" directional="true" castshadow="false" dir="1 0 0" '
                        f'diffuse="{fmt(HORIZON if k == 0 else HORIZON_SIDE)}" specular="0 0 0"/>' for k in range(3))  # fmt: skip
    return f"""<mujoco model="musclevae_render">
  <compiler inertiafromgeom="false"/>
  <statistic extent="2" center="0 0 1"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="{FOVY}"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="{fmt(AMBIENT)}" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.01" zfar="500" shadowclip="1.5" haze="0.15"/>
    <rgba haze="{rgb(SKY[0][1])} 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0" group="1"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" width="{SKY_SIZE}" height="{SKY_SIZE}"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="{rgb(FLOOR_RGB[0])}" rgb2="{rgb(FLOOR_RGB[1])}" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="2 2" texuniform="true" reflectance="0" specular="0" shininess="0"/>
    <material name="bone" rgba="{fmt(BONE_RGBA)}" specular="{BONE_SPECULAR}" shininess="{BONE_SHININESS}"/>
    <material name="ref" rgba="{fmt(REF_RGBA)}" specular="{BONE_SPECULAR}" shininess="{BONE_SHININESS}"/>
{mesh_assets}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <light name="zenith" directional="true" castshadow="false" dir="0 0 -1" diffuse="{fmt(ZENITH)}" specular="0 0 0"/>
    <light name="ground" directional="true" castshadow="false" dir="0 0 1" diffuse="{fmt(GROUND)}" specular="0 0 0"/>
{horizon}
    <body name="rig" mocap="true">
      <light name="key" directional="true" castshadow="true" pos="0 0 4" dir="0 0 -1" diffuse="{fmt(KEY)}" specular="0.3 0.3 0.3"/>
    </body>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def build_model(width: int, height: int, ref: bool) -> mujoco.MjModel:
    char = mc.load_character()
    model = mujoco.MjModel.from_xml_string(scene_xml(char, body_meshes(char), width, height, ref))
    sky = model.texture("sky")
    cube = sky_cube()
    if (int(sky.height[0]), int(sky.width[0]), int(sky.nchannel[0])) != cube.shape:
        raise ValueError(f"MuJoCo's skybox texture is {sky.height[0]} x {sky.width[0]} x {sky.nchannel[0]}, expected {cube.shape}")
    model.tex_data[int(sky.adr[0]) : int(sky.adr[0]) + cube.size] = cube.ravel()
    return model


def aim_lights(model: mujoco.MjModel, azimuth: float):
    """Directions of the key and horizon lights for a camera heading (degrees, MuJoCo's azimuth)."""
    ray, el = np.radians(azimuth + KEY_HEADING), np.radians(KEY_ELEVATION)
    key = model.light("key").id
    model.light_dir[key] = [np.cos(el) * np.cos(ray), np.cos(el) * np.sin(ray), -np.sin(el)]
    model.light_pos[key] = -4.0 * model.light_dir[key]  # relative to the rig at the camera's target: the shadow map's center
    for k in range(3):
        h = np.radians(azimuth + 120.0 * k)
        model.light_dir[model.light(f"horizon{k}").id] = [np.cos(h), np.sin(h), 0.0]


def clip_poses(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """A converted reference clip's body positions (T, 23, 3) and rotations (T, 23, 3, 3), and its rate."""
    clip = clip_source.load_clip(os.path.join(mc.CLIPS_DIR, name + ".npz"))
    pos, rot = mc.forward_kinematics(clip.root_p.double(), clip.root_quat.double(), clip.joints.double())
    return pos.numpy(), rot.numpy(), float(clip.fps)


def body_poses(state: np.ndarray, path: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions (T, 23, 3) and rotations (T, 23, 3, 3) of body states (T, 23, >= 7): position, quaternion wxyz, ..."""
    if state.ndim != 3 or state.shape[1:2] != (mc.NUM_BODIES,) or state.shape[2] < 7 or not np.isfinite(state[..., :7]).all():
        raise ValueError(f"{path}: {key} must be finite (T, {mc.NUM_BODIES}, 13 or 7), got {state.shape}")
    quat = state[..., 3:7]
    if np.abs(np.linalg.norm(quat, axis=-1) - 1.0).max() > 1e-3:
        raise ValueError(f"{path}: {key}[..., 3:7] must be unit quaternions (wxyz)")
    return state[..., :3], quat_to_mat(torch.tensor(quat)).numpy()


def recorded_states(path: str, need_activation: bool, need_ref: bool):
    """Recorded states (see the module docstring): positions, rotations, rate, activations or None, reference poses or None."""
    with np.load(path) as d:
        need = ["body_state", "fps"] + ["fatigue_state"] * need_activation + ["ref_body_state"] * need_ref
        missing = [k for k in need if k not in d.files]
        if missing:
            raise KeyError(f"{path}: missing arrays {missing}")
        arr = {k: d[k].astype(np.float64) for k in need}
    T, fps = len(arr["body_state"]), float(arr["fps"])
    if T == 0 or not fps > 0:
        raise ValueError(f"{path}: no steps, or fps {fps} not positive")
    pos, rot = body_poses(arr["body_state"], path, "body_state")
    act = ref = None
    if need_activation:
        fat = arr["fatigue_state"]
        if fat.shape != (T, mc.NUM_MUSCLES, 3) or not (fat[..., 0].min() >= 0.0 and fat[..., 0].max() <= 1.0):
            raise ValueError(f"{path}: fatigue_state must be (T, {mc.NUM_MUSCLES}, 3) with MA in [0, 1], got {fat.shape}")
        act = fat[..., 0]
    if need_ref:
        if arr["ref_body_state"].shape[0] != T:
            raise ValueError(f"{path}: ref_body_state has {arr['ref_body_state'].shape[0]} steps, body_state {T}")
        ref = body_poses(arr["ref_body_state"], path, "ref_body_state")
    return pos, rot, fps, act, ref


def lerp_steps(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-step values x (T, ...) at fractional steps s (n,), linearly."""
    i0 = np.minimum(np.floor(s).astype(np.int64), len(x) - 2)
    w = (s - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] + w * (x[i0 + 1] - x[i0])


def interpolate_poses(pos: np.ndarray, rot: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body positions (T, B, 3) and rotations (T, B, 3, 3) at fractional steps s (n,): positions linearly, rotations by slerp."""
    i0 = np.minimum(np.floor(s).astype(np.int64), len(pos) - 2)
    q = mat_to_quat(torch.tensor(rot))
    w = torch.tensor(s - i0)[:, None].expand(-1, rot.shape[1])
    return lerp_steps(pos, s), quat_to_mat(quat_slerp(q[i0], q[i0 + 1], w)).numpy()


def muscle_segments(table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each polyline segment's two anchors and its muscle: (S,), (S,), (S,)."""
    count = np.asarray(table.anchor_count)
    start = np.cumsum(count) - count
    a0 = np.concatenate([np.arange(s, s + c - 1) for s, c in zip(start, count)])
    return a0, a0 + 1, np.repeat(np.arange(len(count)), count - 1)


def heading(R: np.ndarray) -> float:
    """The yaw (degrees) of a body's forward axis (+x: the character faces +x in the T-pose)."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help=".mp4 (a video of --seconds from --start) or .png (the frame at --start)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="a converted reference clip: lafan_walk, lafan_run or lafan_jump")
    src.add_argument("--states", help="an npz of recorded states (see the docstring)")
    ap.add_argument("--muscles", choices=("rest", "activation", "none"), default=None,
                    help="rest: at rest (white, thin); activation: by the states' activations; none: bones only. Default: rest for --clip, "
                         "activation for --states")  # fmt: skip
    ap.add_argument("--muscle_scale", type=float, default=1.0, help="the muscles' and anchors' radii times this (see the docstring)")
    ap.add_argument("--ref", action="store_true", help="--states: the reference beside the character, in their blue")
    ap.add_argument("--start", type=float, default=0.0, help="s")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--stride", type=int, default=1, help="steps per video frame (with --interp k, k-ths of a step)")
    ap.add_argument("--interp", type=int, default=1, help="frames per step, those between two steps interpolated (see the docstring); "
                    "1: the steps only")  # fmt: skip
    ap.add_argument("--azimuth", type=float, default=None, help="the camera's heading (degrees, MuJoCo's); default: along the floor's grid, "
                    "as theirs, the grid direction closest to facing the character at the first frame")  # fmt: skip
    ap.add_argument("--distance", type=float, default=CAM_DISTANCE, help="m, the camera's distance from the point it looks at; default: theirs")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    ext = os.path.splitext(a.out)[1].lower()
    if ext not in (".mp4", ".png"):
        raise ValueError(f"{a.out}: the output must be an .mp4 or a .png")
    muscles_mode = a.muscles or ("rest" if a.clip else "activation")
    if a.clip and (muscles_mode == "activation" or a.ref):
        raise ValueError("reference clips have no activations and no reference of their own: --muscles activation and --ref need --states")
    check_egl_device()
    if a.clip:
        pos, rot, fps = clip_poses(a.clip)
        act = ref = None
    else:
        pos, rot, fps, act, ref = recorded_states(a.states, muscles_mode == "activation", a.ref)
    t0 = int(round(a.start * fps))
    t1 = t0 + 1 if ext == ".png" else min(len(pos), t0 + int(round(a.seconds * fps)))
    if not 0 <= t0 < len(pos) or a.stride < 1 or a.interp < 1 or not a.muscle_scale > 0 or not a.distance > 0:
        raise ValueError(f"--start {a.start} s is outside the input's {len(pos) / fps:.1f} s, or --stride or --interp < 1, or --muscle_scale or "
                         "--distance not positive")  # fmt: skip
    start_s = t0 / fps
    if a.interp > 1:  # the steps t0 .. t1 - 1 with interp - 1 interpolated frames between two
        if ext == ".png":
            raise ValueError("--interp needs an .mp4 output: a .png is the frame at --start")
        s = t0 + np.arange((t1 - 1 - t0) * a.interp + 1) / a.interp
        pos, rot = interpolate_poses(pos, rot, s)
        act = None if act is None else lerp_steps(act, s)
        ref = None if ref is None else interpolate_poses(*ref, s)
        fps, t0, t1 = fps * a.interp, 0, len(s)
    frames = np.arange(t0, t1, a.stride)

    model = build_model(a.width, a.height, ref is not None)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width, max_geom=4000)
    scene = renderer.scene
    quat = mat_to_quat(torch.tensor(rot[frames])).numpy()
    mocap = [model.body(n).mocapid[0] for n in mc.BODY_NAMES]
    if ref is not None:
        ref_quat = mat_to_quat(torch.tensor(ref[1][frames])).numpy()
        ref_mocap = [model.body("ref_" + n).mocapid[0] for n in mc.BODY_NAMES]
    if muscles_mode != "none":
        muscles = mc.muscle_model(dtype=torch.float64)
        anchors = muscles.anchor_positions(torch.tensor(pos[frames]), torch.tensor(rot[frames])).numpy()
        seg_a, seg_b, seg_m = muscle_segments(muscles.table)
        level = act[frames] if muscles_mode == "activation" else np.zeros((len(frames), mc.NUM_MUSCLES))
        rgba = np.stack([np.ones_like(level), 1.0 - level, 1.0 - level, np.ones_like(level)], -1).astype(np.float32)  # their rule
        width = a.muscle_scale * MUSCLE_RADIUS * (1.0 + 3.0 * level)
        anchor_size, anchor_rgba = np.array([a.muscle_scale * ANCHOR_RADIUS, 0.0, 0.0]), np.array(ANCHOR_RGBA, dtype=np.float32)
        if scene.maxgeom < model.ngeom + len(seg_m) + mc.NUM_ANCHORS:
            raise ValueError(f"the scene holds {scene.maxgeom} geoms, the muscles need {model.ngeom + len(seg_m) + mc.NUM_ANCHORS}")

    cam = mujoco.MjvCamera()
    azimuth = a.azimuth if a.azimuth is not None else 90.0 * round((heading(rot[frames[0], 0]) + 180.0) / 90.0)
    cam.distance, cam.azimuth, cam.elevation = a.distance, azimuth, -PITCH
    right = np.array([np.sin(np.radians(azimuth)), -np.cos(np.radians(azimuth)), 0.0])  # the camera's right, horizontal
    aim_lights(model, azimuth)
    rig = model.body("rig").mocapid[0]
    bold = _font(0, round(20 * (a.height / 432)))
    follow = 1.0 - np.exp(-a.stride / fps / FOLLOW_TAU)
    look = ff = None
    if ext == ".mp4":
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", f"{fps / a.stride:g}",
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out],
            stdin=subprocess.PIPE,
        )  # fmt: skip
    for i, t in enumerate(frames):
        data.mocap_pos[mocap] = pos[t]
        data.mocap_quat[mocap] = quat[i]
        target = np.array([pos[t, 0, 0], pos[t, 0, 1], LOOK_HEIGHT])
        if ref is not None:  # the reference's pelvis REF_OFFSET to the camera's right of the character's
            shift = pos[t, 0] + REF_OFFSET * right - ref[0][t, 0]
            shift[2] = 0.0
            data.mocap_pos[ref_mocap] = ref[0][t] + shift
            data.mocap_quat[ref_mocap] = ref_quat[i]
            target += 0.5 * REF_OFFSET * right
        look = target if look is None else look + follow * (target - look)  # smooth follow, no jitter
        data.mocap_pos[rig] = look
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        cam.lookat[:] = look
        renderer.update_scene(data, camera=cam)
        if muscles_mode != "none":
            for p, q, m in zip(anchors[i, seg_a], anchors[i, seg_b], seg_m):
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba[i, m])
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width[i, m], p, q)
                g.specular, g.shininess = MUSCLE_SPECULAR, MUSCLE_SHININESS
                scene.ngeom += 1
            for p in anchors[i]:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, anchor_size, p, np.eye(3).ravel(), anchor_rgba)
                scene.ngeom += 1
        img = Image.fromarray(renderer.render()).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_mark(ImageDraw.Draw(over), a.width, a.height, bold)
        img = Image.alpha_composite(img, over).convert("RGB")
        if ff is None:
            img.save(a.out)
        else:
            ff.stdin.write(np.asarray(img).tobytes())
    if ff is not None:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {a.out}")
    print(f"wrote {a.out}: {len(frames)} frame(s) from {start_s:.2f} s at {fps / a.stride:g} fps")


if __name__ == "__main__":
    main()
