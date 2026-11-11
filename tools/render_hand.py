"""Render a hand-object rollout (ndm_gym.evaluate_hand's rollout.npz) in the look of tools/render.py: the simulated hand
and object solid, the kinematic reference a translucent ghost at its own place, a camera that follows them over the floor,
the NDM Gym mark (``--no_mark``: none).

    python tools/render_hand.py --clip data/spider/processed/<dataset>/xhand/right/<task>/0 \
        --rollout logs/<name>/evaluation/rollout.npz --out logs/<name>/evaluation/policy.mp4

Frames interpolate the saved control steps. ``--source reference_targets`` renders the rollout without the policy, the
reference pose as the position targets, over the same ghost; ``--source reference`` the kinematic reference alone, solid. Needs
ffmpeg (or the imageio-ffmpeg package's).
"""

import argparse
import os
import shutil
import subprocess
import sys

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.core.rotations import mat_to_quat
from ndm_gym.robots.xhand import scene_indices

FLOOR_CELL = 0.1  # the side of a floor tile (m)
GHOST_RGBA = (0.25, 0.58, 0.95, 0.18)
SHOWN_GROUPS = 3  # geom groups 0-2: the floor, the hand's and the object's visual meshes; not the collision shapes


def studio_scene(scene_xml: str, width: int, height: int, ghost: bool) -> mujoco.MjModel:
    """The clip's scene with tools/render.py's studio look (a gradient sky, a softly reflecting floor with a fine grid, a
    shadow-casting key light and a fill light, haze). With ``ghost``, a second, translucent hand and object for the
    reference, their names prefixed ``ref_``."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    for light in spec.lights:
        spec.delete(light)
    if ghost:
        copy = mujoco.MjSpec.from_file(scene_xml)
        for light in copy.lights:
            copy.delete(light)
        for geom in copy.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                copy.delete(geom)
            else:
                geom.material, geom.rgba, geom.contype, geom.conaffinity = "", GHOST_RGBA, 0, 0
        spec.attach(copy, prefix="ref_", frame=spec.worldbody.add_frame())

    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(tex)
    spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=(0.985, 0.987, 0.99), rgb2=(0.86, 0.88, 0.91), width=512, height=512)  # fmt: skip
    spec.add_texture(name="studio_floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=(0.80, 0.81, 0.83), rgb2=(0.77, 0.78, 0.80), mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=(0.66, 0.68, 0.72),
                     width=512, height=512)  # fmt: skip
    floor = spec.add_material(name="studio_floor")
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "studio_floor"
    floor.texrepeat, floor.texuniform = (1 / FLOOR_CELL, 1 / FLOOR_CELL), True
    floor.reflectance, floor.shininess, floor.specular = 0.10, 0.2, 0.1
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material, geom.rgba, geom.size = "studio_floor", (1, 1, 1, 1), (0, 0, 0.05)
    spec.worldbody.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0.8, -0.6, 3), dir=(-0.3, 0.25, -1),
                             castshadow=True, diffuse=(0.58, 0.57, 0.55), specular=(0.2, 0.2, 0.2))  # fmt: skip
    spec.worldbody.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0, 0, 5), dir=(0.5, 0.6, -0.6),
                             castshadow=False, diffuse=(0.30, 0.31, 0.33), specular=(0, 0, 0))  # fmt: skip
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = (0.34, 0.34, 0.35), (0.16,) * 3, (0.05,) * 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.quality.shadowsize, spec.visual.quality.offsamples = 8192, 8
    model = spec.compile()
    model.stat.extent = 1.0  # the unit of the clipping planes, the shadow's range and the haze
    model.vis.map.znear, model.vis.map.shadowclip, model.vis.map.shadowscale, model.vis.map.haze = 0.01, 1.0, 0.6, 0.4
    model.vis.rgba.haze[:] = (0.95, 0.96, 0.97, 1.0)
    return model


def interpolate(qpos: np.ndarray, quat_at: int, x: np.ndarray) -> np.ndarray:
    """qpos (T, nq) at fractional steps ``x``: linear, the object's quaternion at ``quat_at`` normalized (short arc)."""
    i0 = np.minimum(np.floor(x).astype(int), len(qpos) - 2)
    t = (x - i0)[:, None]
    a, b = qpos[i0], qpos[i0 + 1].copy()
    q = slice(quat_at, quat_at + 4)
    b[:, q] *= np.where((a[:, q] * b[:, q]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    out = a + (b - a) * t
    out[:, q] /= np.linalg.norm(out[:, q], axis=-1, keepdims=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", type=str, required=True)
    parser.add_argument("--rollout", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--source", type=str, default="policy", choices=["policy", "reference_targets", "reference"])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=0.5, help="playback speed (1: real time)")
    parser.add_argument("--azimuth", type=float, default=None, help="the camera's (degrees; default: facing the final grasp)")
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--no_mark", action="store_true", help="frames without the NDM Gym mark")
    args = parser.parse_args()
    check_egl_device()

    scene_xml = os.path.join(os.path.dirname(os.path.normpath(args.clip)), "scene.xml")
    indices, rollout = scene_indices(mujoco.MjModel.from_xml_path(scene_xml)), np.load(args.rollout)
    object_qpos, nq = indices["object_qpos"], rollout["qpos"].shape[1]
    reference = np.zeros((len(rollout["reference_q"]), nq))
    reference[:, indices["qpos"]], reference[:, object_qpos : object_qpos + 3] = rollout["reference_q"], rollout["reference_p"]
    reference[:, object_qpos + 3 : object_qpos + 7] = mat_to_quat(torch.as_tensor(rollout["reference_R"])).numpy()
    solid = reference if args.source == "reference" else rollout["qpos" if args.source == "policy" else "reference_targets_qpos"].astype(np.float64)
    steps = np.arange(0, len(solid) - 1 + 1e-9, args.speed / (args.fps * float(rollout["control_dt"])))  # control steps per video frame
    frames, followed = interpolate(solid, object_qpos + 3, steps), interpolate(reference[: len(solid)], object_qpos + 3, steps)
    if args.source != "reference":  # the ghost's joints follow the scene's own in qpos
        frames = np.concatenate((frames, followed), -1)
    model = studio_scene(scene_xml, args.width, args.height, ghost=args.source != "reference")
    if model.nq != frames.shape[1]:
        raise ValueError("the studio scene's qpos does not match the rollout's")

    # the camera follows the reference's palm and object over the floor at a fixed height, whatever the source, by default
    # facing the palm of the final grasp from its side
    palm_positions, object_positions = followed[:, indices["qpos"][:3]], followed[:, object_qpos : object_qpos + 3]
    track = np.concatenate((palm_positions, object_positions))
    lowest, highest = track.min(0), track.max(0)
    grasp_direction = object_positions[-1, :2] - palm_positions[-1, :2]  # from the palm through the object
    camera = mujoco.MjvCamera()
    camera.distance = 0.45 + 1.0 * float(max(highest[2] - lowest[2], np.linalg.norm((palm_positions - object_positions)[:, :2], axis=-1).max()))
    camera.azimuth = np.degrees(np.arctan2(-grasp_direction[1], -grasp_direction[0])) + 70.0 if args.azimuth is None else args.azimuth
    camera.elevation = args.elevation
    follow = 1 - (1 - 0.18) ** (25 / args.fps)  # tools/render.py's smoothing, per frame at 25 fps
    look = np.array([*(palm_positions[0, :2] + object_positions[0, :2]) / 2, (lowest[2] + highest[2]) / 2])
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:SHOWN_GROUPS] = 1
    option.sitegroup[:] = 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    data, bold = mujoco.MjData(model), _font(0, round(20 * args.height / 432))
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for qpos, palm_at, object_at in zip(frames, palm_positions, object_positions, strict=True):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            look[:2] += follow * ((palm_at[:2] + object_at[:2]) / 2 - look[:2])
            camera.lookat[:] = look
            model.light("key").pos[:2] = look[:2] + (0.8, -0.6)  # the shadow map stays on the scene in view
            renderer.update_scene(data, camera, option)
            image = renderer.render()
            if not args.no_mark:
                over = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
                draw_mark(ImageDraw.Draw(over), args.width, args.height, bold)
                image = np.asarray(Image.alpha_composite(Image.fromarray(image).convert("RGBA"), over).convert("RGB"))
            encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"{args.out}: {len(frames)} frames, {args.width}x{args.height} at {args.fps} fps")


if __name__ == "__main__":
    main()
