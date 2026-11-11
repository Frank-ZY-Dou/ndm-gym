"""Render a hand-object rollout (ndm_gym.evaluate_hand's rollout.npz) in the look of tools/render.py: the simulated hand
and object solid, the kinematic reference a translucent ghost at its own place, a camera that follows them over the floor,
the NDM Gym mark (``--no_mark``: none).

    python tools/render_hand.py --clip data/spider/processed/<dataset>/xhand/right/<task>/0 \
        --rollout logs/<name>/evaluation/rollout.npz --out logs/<name>/evaluation/policy.mp4

Frames interpolate the saved control steps. ``--source reference_targets`` renders the rollout without the policy, the
reference pose as the position targets, over the same ghost; ``--source reference`` the kinematic reference alone, solid. Needs
ffmpeg (or the imageio-ffmpeg package's).

``--look noir``: a dark stage instead of the studio's light one (a dark gradient sky, a glossy dark floor with a faint grid
that fades into the haze, a warm key light, a cool rim light from behind, satin materials on the hand and the object, the
reference as a faintly glowing cyan ghost, the mark in white). ``--orbit d``: the camera turns by d degrees over the clip;
``--zoom z``: the camera's distance times z; ``--follow_z``: the camera follows the height too; ``--no_ghost``: the
solid scene alone; ``--ghost_overlay``: the reference rendered alone and laid over the frame as a translucent blue
silhouette with an outline, so that it does not interleave with the solid hand where the two nearly coincide.
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
NOIR_GHOST_RGBA = (0.40, 0.85, 1.0, 0.12)
GHOST_GROUP = 5  # --ghost_overlay: the ghost's geoms, drawn in a pass of their own
OVERLAY_RGBA = {"studio": (0.20, 0.50, 0.98, 1.0), "noir": (0.32, 0.62, 1.0, 1.0)}
OVERLAY_ALPHA = {"studio": (0.26, 0.75), "noir": (0.26, 0.85)}  # the silhouette's and the outline's opacity
NOIR_SKY = ((0.115, 0.125, 0.15), (0.012, 0.013, 0.017))  # the gradient's top and bottom
NOIR_HAZE = (0.016, 0.017, 0.021, 1.0)
LOOKS = ("studio", "noir")
SHOWN_GROUPS = 3  # geom groups 0-2: the floor, the hand's and the object's visual meshes; not the collision shapes


def studio_scene(scene_xml: str, width: int, height: int, ghost: bool, look: str = "studio", overlay: bool = False) -> mujoco.MjModel:
    """The clip's scene with tools/render.py's studio look (a gradient sky, a softly reflecting floor with a fine grid, a
    shadow-casting key light and a fill light, haze), or with ``look="noir"`` the dark stage of the module's docstring. With
    ``ghost``, a second, translucent hand and object for the reference, their names prefixed ``ref_``; with ``overlay`` the
    ghost is opaque, in geom group GHOST_GROUP, for a pass of its own (main's --ghost_overlay)."""
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}, got {look!r}")
    noir = look == "noir"
    spec = mujoco.MjSpec.from_file(scene_xml)
    for light in spec.lights:
        spec.delete(light)
    if noir:  # satin materials on the hand's and the object's visual geoms that have none (their colors kept)
        satin = spec.add_material(name="noir_satin")
        satin.specular, satin.shininess, satin.reflectance = 0.55, 0.75, 0.0
        for geom in spec.geoms:
            if geom.type != mujoco.mjtGeom.mjGEOM_PLANE and not geom.material and geom.group < SHOWN_GROUPS:
                geom.material = "noir_satin"
    if ghost:
        copy = mujoco.MjSpec.from_file(scene_xml)
        for light in copy.lights:
            copy.delete(light)
        rgba = OVERLAY_RGBA[look] if overlay else NOIR_GHOST_RGBA if noir else GHOST_RGBA
        if noir or overlay:
            glow = copy.add_material(name="hologram")
            glow.rgba, glow.emission, glow.specular, glow.shininess = rgba, 0.55 if overlay else 0.5, 0.3 if overlay else 0.0, 0.5 if overlay else 0.0
        for geom in copy.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                copy.delete(geom)
            else:
                geom.material = "hologram" if noir or overlay else ""
                geom.rgba, geom.contype, geom.conaffinity = rgba, 0, 0
                if overlay:
                    geom.group = GHOST_GROUP if geom.group < SHOWN_GROUPS else geom.group
        spec.attach(copy, prefix="ref_", frame=spec.worldbody.add_frame())

    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(tex)
    sky = NOIR_SKY if noir else ((0.985, 0.987, 0.99), (0.86, 0.88, 0.91))
    spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=sky[0], rgb2=sky[1], width=512, height=512)  # fmt: skip
    if noir:
        checker = dict(rgb1=(0.078, 0.081, 0.092), rgb2=(0.070, 0.073, 0.083), mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=(0.150, 0.160, 0.188))
    else:
        checker = dict(rgb1=(0.80, 0.81, 0.83), rgb2=(0.77, 0.78, 0.80), mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=(0.66, 0.68, 0.72))
    spec.add_texture(name="studio_floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     width=512, height=512, **checker)  # fmt: skip
    floor = spec.add_material(name="studio_floor")
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "studio_floor"
    floor.texrepeat, floor.texuniform = (1 / FLOOR_CELL, 1 / FLOOR_CELL), True
    floor.reflectance, floor.shininess, floor.specular = (0.35, 0.85, 0.45) if noir else (0.10, 0.2, 0.1)
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material, geom.rgba, geom.size = "studio_floor", (1, 1, 1, 1), (0, 0, 0.05)
    if noir:  # a warm key light, a cool rim light from behind the hand (turned with the camera), a faint fill
        spec.worldbody.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0.8, -0.6, 3), dir=(-0.35, 0.3, -1),
                                 castshadow=True, diffuse=(1.00, 0.95, 0.88), specular=(0.60, 0.57, 0.52))  # fmt: skip
        spec.worldbody.add_light(name="rim", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0, 0, 3), dir=(1, 0, -0.35),
                                 castshadow=False, diffuse=(0.36, 0.50, 0.74), specular=(0.65, 0.80, 1.00))  # fmt: skip
        spec.worldbody.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0, 0, 5), dir=(0.5, 0.6, -0.6),
                                 castshadow=False, diffuse=(0.17, 0.18, 0.21), specular=(0, 0, 0))  # fmt: skip
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = (0.17, 0.17, 0.18), (0.14,) * 3, (0.05,) * 3
    else:
        spec.worldbody.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0.8, -0.6, 3), dir=(-0.3, 0.25, -1),
                                 castshadow=True, diffuse=(0.58, 0.57, 0.55), specular=(0.2, 0.2, 0.2))  # fmt: skip
        spec.worldbody.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0, 0, 5), dir=(0.5, 0.6, -0.6),
                                 castshadow=False, diffuse=(0.30, 0.31, 0.33), specular=(0, 0, 0))  # fmt: skip
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = (0.34, 0.34, 0.35), (0.16,) * 3, (0.05,) * 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.quality.shadowsize, spec.visual.quality.offsamples = 8192, 8
    model = spec.compile()
    model.stat.extent = 1.0  # the unit of the clipping planes, the shadow's range and the haze
    model.vis.map.znear, model.vis.map.shadowscale = (0.05, 0.6) if noir else (0.01, 0.6)
    model.vis.map.shadowclip = 3.0 if noir else 1.0
    if noir:  # the depth buffer's precision over the floor in view: with znear 0.01 the far floor's shadow test speckles
        model.vis.map.zfar = 8.0
    model.vis.map.haze = 0.9 if noir else 0.4
    model.vis.rgba.haze[:] = NOIR_HAZE if noir else (0.95, 0.96, 0.97, 1.0)
    return model


def _eroded(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` without the pixels within ``r`` of its border (a 4-neighbour erosion by shifts)."""
    inner = mask.copy()
    for dy, dx in ((r, 0), (-r, 0), (0, r), (0, -r)):
        inner &= np.roll(np.roll(mask, dy, 0), dx, 1)
    return inner


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
    parser.add_argument("--look", type=str, default="studio", choices=LOOKS, help="studio: tools/render.py's light studio; noir: a dark stage")
    parser.add_argument("--orbit", type=float, default=0.0, help="the camera turns by this many degrees over the clip")
    parser.add_argument("--zoom", type=float, default=1.0, help="the camera's distance times this")
    parser.add_argument("--follow_z", action="store_true", help="the camera also follows the height of the palm and the object")
    parser.add_argument("--no_ghost", action="store_true", help="without the reference's ghost")
    parser.add_argument("--ghost_overlay", action="store_true",
                        help="the reference drawn over the frame: rendered alone, then laid on as a translucent silhouette with an outline")  # fmt: skip
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
    ghost = args.source != "reference" and not args.no_ghost
    if ghost:  # the ghost's joints follow the scene's own in qpos
        frames = np.concatenate((frames, followed), -1)
    overlay = ghost and args.ghost_overlay
    model = studio_scene(scene_xml, args.width, args.height, ghost=ghost, look=args.look, overlay=overlay)
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
    camera.distance *= args.zoom
    camera.azimuth = np.degrees(np.arctan2(-grasp_direction[1], -grasp_direction[0])) + 70.0 if args.azimuth is None else args.azimuth
    azimuths = camera.azimuth + args.orbit * (np.linspace(-0.5, 0.5, len(frames)) if len(frames) > 1 else np.zeros(1))
    camera.elevation = args.elevation
    follow = 1 - (1 - 0.18) ** (25 / args.fps)  # tools/render.py's smoothing, per frame at 25 fps
    look = np.array([*(palm_positions[0, :2] + object_positions[0, :2]) / 2, (lowest[2] + highest[2]) / 2])
    if args.follow_z:
        look[2] = (palm_positions[0, 2] + object_positions[0, 2]) / 2
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:SHOWN_GROUPS] = 1
    option.sitegroup[:] = 0
    ghost_option = mujoco.MjvOption()  # --ghost_overlay: the ghost's pass, the ghost's geoms alone
    ghost_option.geomgroup[:] = 0
    ghost_option.geomgroup[GHOST_GROUP] = 1
    ghost_option.sitegroup[:] = 0
    alpha, outline = OVERLAY_ALPHA[args.look]

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
    mark_fill = (236, 238, 242, 120) if args.look == "noir" else (28, 30, 34, 110)
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for qpos, palm_at, object_at, azimuth in zip(frames, palm_positions, object_positions, azimuths, strict=True):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            look[:2] += follow * ((palm_at[:2] + object_at[:2]) / 2 - look[:2])
            if args.follow_z:
                look[2] += follow * ((palm_at[2] + object_at[2]) / 2 - look[2])
            camera.lookat[:], camera.azimuth = look, azimuth
            model.light("key").pos[:2] = look[:2] + (0.8, -0.6)  # the shadow map stays on the scene in view
            if args.look == "noir":  # the rim light shines from behind the hand, toward the camera
                a = np.radians(azimuth)
                model.light("rim").dir[:] = (-np.cos(a), -np.sin(a), -0.35)
            renderer.update_scene(data, camera, option)
            image = renderer.render()
            if overlay:  # the ghost alone, then laid over the frame where it is: a translucent silhouette and an outline
                renderer.update_scene(data, camera, ghost_option)
                shade = renderer.render().astype(np.float32)
                renderer.enable_segmentation_rendering()
                mask = renderer.render()[..., 0] >= 0
                renderer.disable_segmentation_rendering()
                edge = mask & ~_eroded(mask, max(1, round(args.height / 400)))
                weight = (alpha * mask + (outline - alpha) * edge)[..., None]
                image = (image * (1 - weight) + shade * weight).round().clip(0, 255).astype(np.uint8)
            if not args.no_mark:
                over = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
                draw_mark(ImageDraw.Draw(over), args.width, args.height, bold, fill=mark_fill)
                image = np.asarray(Image.alpha_composite(Image.fromarray(image).convert("RGBA"), over).convert("RGB"))
            encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"{args.out}: {len(frames)} frames, {args.width}x{args.height} at {args.fps} fps")


if __name__ == "__main__":
    main()
