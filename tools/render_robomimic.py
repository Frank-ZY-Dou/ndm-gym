"""Render a manipulation tracking rollout (ndm_gym.evaluate_robomimic's rollout_<demo>.npz) in robosuite's own scene, the one
the demonstration was recorded in (its materials, textures and lights; its visual geoms): the simulated arms, grippers and
objects solid, the demonstration's a translucent blue ghost laid over the frame with an outline (tools/render_hand.py's
--ghost_overlay), a fixed camera per task that follows the tracked objects a little, a label and the NDM Gym mark
(tools/render.py's).

    MUJOCO_GL=egl python tools/render_robomimic.py --rollout logs/<name>/evaluation/rollout_0.npz --out lift.webp --title SuperTrack

An ``--out`` ending in ``.webp`` is an animated WebP at ``--fps`` and ``--quality`` (the README's renders, gathered by
tools/make_grid.py); ``.png`` or ``.jpg`` a contact sheet of ``--frames`` frames, each captioned with its time; any other a video
through ffmpeg (or the imageio-ffmpeg package's).

``--speed``: playback speed (Transport's demonstration lasts 36 s: 4 shows it in 9); ``--first``, ``--seconds``: a window of the
rollout, in its seconds. A rollout that terminated is drawn up to its termination. ``--ghost``: the demonstration's arms and
objects (``all``), its objects alone (``objects``) or nothing (``none``). ``--source reference_targets``: the rollout under the
reference's joint positions as the servos' targets (open loop) over the same ghost. The camera: ``--azimuth``, ``--elevation``,
``--distance`` (m) and ``--lookat`` override the task's; ``--follow`` how far it leans toward the tracked objects (0: fixed).
The scene comes from the demonstration (``--data_dir``: tools/download_robomimic.py's output); robosuite is not needed.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

from render import _font, _pill, check_egl_device, draw_mark  # tools/render.py: labels and the mark; importing it sets up EGL
from render_hand import _eroded

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.motion.sources.robomimic import demo_path, load_demo
from ndm_gym.robots.robomimic_scenes import TASKS, load_scene

GHOST_RGB = np.array((0.25, 0.58, 0.95)) * 255
# the ghost's silhouette's and outline's opacity: the arms' faint (they track closely and would tint robosuite's white Panda
# blue), the objects' as tools/render_hand.py's --ghost_overlay
GHOST_ALPHA = {1: (0.10, 0.55), 2: (0.30, 0.85)}
VISUAL_GROUP = 1  # robosuite's visual geoms (group 0: the collision shapes, which its renders hide)
# per task the free camera: lookat (robosuite's world: the table's centre at the origin, 0.8 m high), distance (m), azimuth,
# elevation (degrees); Lift and Square seen from the front, beside the arm, Transport from the side of its two tables
VIEWS = {
    "lift": ((-0.2, 0.02, 1.17), 1.35, 150.0, -12.0),
    "square": ((-0.12, -0.02, 1.12), 1.5, 155.0, -16.0),
    "transport": ((0.0, 0.0, 1.1), 1.85, 180.0, -27.0),
}
INK, MUTED, RED = (28, 30, 34), (92, 98, 108), (200, 45, 45)
MARK_LIGHT = (255, 255, 255, 150)  # the mark in white where the first frame is dark under it (Transport's pedestals)


def ghost_geoms(m: mujoco.MjModel, task: str) -> np.ndarray:
    """Each geom's part of the ghost (ngeom,): 1 the robots' (their arms and grippers; not their fixed bases and mounts), 2 the
    tracked objects', 0 none; of the visual geoms that move (not welded to the world)."""
    objects = {int(m.body_rootid[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]]) for j in TASKS[task]["objects"]}
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    robots = {int(m.body_rootid[b]) for b, name in enumerate(names) if re.match(r"(robot|gripper)\d+_", name)}
    body = m.geom_bodyid
    moving = (m.geom_group == VISUAL_GROUP) & (m.body_weldid[body] != 0)
    return np.where(moving & np.isin(m.body_rootid[body], list(objects)), 2, np.where(moving & np.isin(m.body_rootid[body], list(robots)), 1, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "robomimic"))
    ap.add_argument("--task", default=None, help="the rollout's task (default: the rollout's own)")
    ap.add_argument("--demo", type=int, default=None, help="its demonstration (default: the rollout's own)")
    ap.add_argument("--source", default="state", choices=["state", "reference_targets"])
    ap.add_argument("--ghost", default="all", choices=["all", "objects", "none"])
    ap.add_argument("--title", default="", help="bold label, e.g. the method")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--first", type=float, default=0.0, help="the window's start in the rollout's seconds")
    ap.add_argument("--seconds", type=float, default=None, help="the window's length in the rollout's seconds (default: to its end)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--lookat", type=float, nargs=3, default=None)
    ap.add_argument("--follow", type=float, default=0.25, help="the share of the tracked objects' mean position in the camera's lookat")
    ap.add_argument("--frames", type=int, default=8, help="a contact sheet's frames")
    ap.add_argument("--cols", type=int, default=4, help="a contact sheet's columns")
    ap.add_argument("--quality", type=int, default=70, help=".webp: the lossy WebP's quality")
    ap.add_argument("--no_labels", action="store_true", help="the NDM Gym mark only")
    ap.add_argument("--no_mark", action="store_true")
    a = ap.parse_args()
    check_egl_device()

    with np.load(a.rollout) as d:
        nq, dt = int(d["nq"]), float(d["control_dt"])
        qpos = d["state" if a.source == "state" else "reference_targets_state"][:, :nq]
        ref = d["reference_qpos"]
        task = a.task or (str(d["task"]) if "task" in d else None)
        demo = a.demo if a.demo is not None else int(d["demo"]) if "demo" in d else 0
        term = int(d["terminated_at"]) if "terminated_at" in d and a.source == "state" else -1
    if task is None:
        ap.error("the rollout does not name its task: --task")
    xml = load_demo(demo_path(a.data_dir, task), demo)["xml"]
    model = load_scene(xml, os.path.abspath(os.path.join(a.data_dir, "robosuite_assets")))
    if model.nq != nq:
        raise ValueError(f"the rollout's {nq} coordinates, the scene of {task} demo {demo} has {model.nq}")
    model.vis.global_.offwidth, model.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
    model.vis.quality.offsamples = 8
    data, ref_data = mujoco.MjData(model), mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[VISUAL_GROUP] = 1
    opt.sitegroup[:] = 0
    ghost = None if a.ghost == "none" else ghost_geoms(model, task)
    if a.ghost == "objects":
        ghost[ghost == 1] = 0
    look0, dist, az, el = VIEWS[task]
    cam = mujoco.MjvCamera()
    cam.distance = a.distance or dist
    cam.azimuth = az if a.azimuth is None else a.azimuth
    cam.elevation = el if a.elevation is None else a.elevation
    look0 = np.array(a.lookat or look0, dtype=float)
    obj_q = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in TASKS[task]["objects"]]
    s = a.height / 432  # label sizes scale with the frame
    bold, small = _font(0, round(20 * s)), _font(1, round(15 * s))
    mark_fill = None  # the mark's color, chosen at the first frame by the brightness under it

    T = len(qpos) if term < 0 else term + 1  # the steps drawn: up to the termination
    if term >= 0:
        print(f"the rollout terminated after {term * dt:.2f} s")
    if a.seconds is not None:
        T = min(T, int(round((a.first + a.seconds) / dt)) + 1)
    first = int(round(a.first / dt))
    if first >= T:
        ap.error(f"--first {a.first} s: the rollout's drawn steps end at {(T - 1) * dt:.2f} s")

    def target(k: int) -> np.ndarray:
        objects = np.mean([qpos[k, q : q + 3] for q in obj_q], 0)
        return (1 - a.follow) * look0 + a.follow * objects

    def frame(k: int, lookat: np.ndarray, caption: str) -> Image.Image:
        nonlocal mark_fill
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = lookat
        renderer.update_scene(data, cam, scene_option=opt)
        image = renderer.render()
        if ghost is not None:  # the demonstration's scene, its ghost's pixels laid over the frame: a translucent silhouette and an outline
            ref_data.qpos[:] = ref[min(k, len(ref) - 1)]
            mujoco.mj_forward(model, ref_data)
            renderer.update_scene(ref_data, cam, scene_option=opt)
            lum = renderer.render().astype(np.float32).mean(-1, keepdims=True) / 255
            renderer.enable_segmentation_rendering()
            seg = renderer.render()
            renderer.disable_segmentation_rendering()
            part = np.where((seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] >= 0), ghost[seg[..., 0].clip(0)], 0)
            weight = np.zeros(part.shape, np.float32)
            for p, (alpha, outline) in GHOST_ALPHA.items():
                mask = part == p
                edge = mask & ~_eroded(mask, max(1, round(a.height / 400)))
                weight = np.maximum(weight, alpha * mask + (outline - alpha) * edge)
            weight = weight[..., None]
            shade = GHOST_RGB * (0.55 + 0.6 * lum)
            image = (image * (1 - weight) + shade * weight).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(image).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(over)
        m = round(12 * s)
        if not a.no_labels:
            if a.title:
                _pill(draw, (m, m), [(a.title, bold, INK)])
            done = 0 <= term <= k
            _pill(draw, (a.width - m, m), [(caption, small, MUTED)] + ([("terminated", small, RED)] if done else []), anchor="ra")
        if not a.no_mark:
            if mark_fill is None:
                corner = image[-round(a.height * 0.12) :, -round(a.width * 0.2) :]
                mark_fill = MARK_LIGHT if corner.mean() < 70 else (28, 30, 34, 110)
            draw_mark(draw, a.width, a.height, bold, fill=mark_fill)
        return Image.alpha_composite(img, over).convert("RGB")

    name = {"lift": "Lift", "square": "Square", "transport": "Transport"}[task]
    speed = "" if a.speed == 1 else f"   {a.speed:g}x"
    ext = os.path.splitext(a.out)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if ext in (".png", ".jpg"):  # a contact sheet
        steps = np.linspace(first, T - 1, a.frames).round().astype(int)
        tiles = [frame(k, target(k), f"{name}   {k * dt:.2f} s") for k in steps]
        rows = (len(tiles) + a.cols - 1) // a.cols
        sheet = Image.new("RGB", (a.cols * a.width, rows * a.height), (255, 255, 255))
        for i, img in enumerate(tiles):
            sheet.paste(img, ((i % a.cols) * a.width, (i // a.cols) * a.height))
        sheet.save(a.out)
        print("wrote", a.out)
        return
    webp, frames = ext == ".webp", []
    if not webp:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:  # the imageio-ffmpeg package's binary
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
                               "-r", f"{a.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], stdin=subprocess.PIPE)  # fmt: skip
    step = a.speed / (a.fps * dt)  # control steps per video frame
    look, t = None, float(first)
    while t < T:
        k = int(t)
        look = target(k) if look is None else look + 0.15 * (target(k) - look)
        img = frame(k, look, f"{name}   {k * dt:5.2f} s{speed}")
        if webp:
            frames.append(img)
        else:
            ff.stdin.write(np.asarray(img).tobytes())
        t += step
    if webp:
        frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=round(1000 / a.fps), loop=0, quality=a.quality, method=4)
    else:
        ff.stdin.close()
        ff.wait()
    print("wrote", a.out, f"({len(frames) if webp else ''} frames, steps {first}-{T - 1})")


if __name__ == "__main__":
    main()
