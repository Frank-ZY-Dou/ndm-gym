"""Render a quadruped tracking rollout (ndm_gym.evaluate_quadruped's rollout.npz, saved with --save_rollout): the simulated
Go2 solid and the retargeted reference a blue ghost at its own place, laid over the frame as a translucent silhouette with an
outline so that it shows where it overlaps the robot (tools/render_hand.py's --ghost_overlay; ``--plain_ghost``: a faint
ghost in the scene instead), in the studio look of tools/render_hand.py, a camera that follows the robot's trunk from the side, with the NDM Gym mark (``--no_mark``: none). An ``--out`` ending in
``.webp`` is an animation of the whole clip in real time at ``--fps`` (the README's renders, gathered by tools/make_grid.py);
any other is a contact sheet: ``--frames`` frames at even steps in a grid, each captioned with its time.

    MUJOCO_GL=egl python tools/render_quadruped.py --rollout logs/<name>/evaluation/rollout.npz --clip dog_quad_walk_002_1 \
        --out logs/<name>/evaluation/dog_quad_walk_002_1.png      # or .webp

``--source reference_targets``: the rollout without the policy (the reference pose as the position targets) over the same
ghost. A rollout that terminated is drawn up to its end; frames after a termination are marked.
"""

import argparse
import os
import sys

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does
from render_hand import GHOST_GROUP, OVERLAY_ALPHA, _eroded, studio_scene

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.robots.go2 import NQ, SCENE_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default="qpos", choices=["qpos", "reference_targets"])
    parser.add_argument("--frames", type=int, default=12, help="frames in the sheet")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--no_mark", action="store_true")
    parser.add_argument("--plain_ghost", action="store_true", help="the reference as a faint ghost in the scene, not laid over the frame")
    parser.add_argument("--fps", type=float, default=25.0, help=".webp: frames per second, in real time")
    parser.add_argument("--quality", type=int, default=70, help=".webp: the lossy WebP's quality")
    args = parser.parse_args()
    check_egl_device()

    with np.load(args.rollout) as d:
        key = f"{args.clip}/{'qpos' if args.source == 'qpos' else 'reference_targets_qpos'}"
        qpos, ref = d[key], d[f"{args.clip}/reference_qpos"]
        term = int(d[f"{args.clip}/{'terminated_at' if args.source == 'qpos' else 'reference_targets_terminated_at'}"])
        dt = float(d["control_dt"])
    overlay = not args.plain_ghost
    model = studio_scene(SCENE_PATH, args.width, args.height, ghost=True, overlay=overlay)
    if model.nq != 2 * NQ:
        raise ValueError(f"expected the Go2 and its ghost, {2 * NQ} coordinates; the scene has {model.nq}")
    model.vis.map.shadowclip = 6.0  # the robot walks metres from the origin: a shadow map over all of it
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.6, -18.0, 90.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # the visual meshes, not the collision shapes (nor the overlay's ghost)
    ghost_opt = mujoco.MjvOption()  # the overlay's pass: the ghost's geoms alone
    ghost_opt.geomgroup[:] = 0
    ghost_opt.geomgroup[GHOST_GROUP] = 1
    ghost_opt.sitegroup[:] = 0
    alpha, outline = OVERLAY_ALPHA["studio"]
    bold = _font(0, round(20 * args.height / 432))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def frame(t: int, lookat: np.ndarray) -> Image.Image:
        data.qpos[:NQ], data.qpos[NQ:] = qpos[min(t, len(qpos) - 1)], ref[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = lookat
        renderer.update_scene(data, cam, scene_option=opt)
        image = renderer.render()
        if overlay:  # the ghost alone, then laid over the frame where it is: a translucent silhouette and an outline
            renderer.update_scene(data, cam, scene_option=ghost_opt)
            shade = renderer.render().astype(np.float32)
            renderer.enable_segmentation_rendering()
            mask = renderer.render()[..., 0] >= 0
            renderer.disable_segmentation_rendering()
            edge = mask & ~_eroded(mask, max(1, round(args.height / 400)))
            weight = (alpha * mask + (outline - alpha) * edge)[..., None]
            image = (image * (1 - weight) + shade * weight).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)

    if args.out.endswith(".webp"):  # every frame at --fps, the camera following the trunk smoothly
        steps = np.arange(0, len(ref), max(1, round(1 / (dt * args.fps))))
        frames, look = [], None
        for t in steps:
            target = qpos[min(t, len(qpos) - 1), :3] * (1, 1, 0) + (0, 0, 0.25)
            look = target if look is None else look + 0.25 * (target - look)
            img = frame(t, look)
            if not args.no_mark:
                draw_mark(ImageDraw.Draw(img, "RGBA"), img.width, img.height, _font(0, round(20 * img.height / 432)))
            frames.append(img)
        ms = round(1000 * dt * (steps[1] - steps[0] if len(steps) > 1 else 1))
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=ms, loop=0, quality=args.quality, method=4)
        print(args.out)
        return
    steps = np.linspace(0, len(ref) - 1, args.frames).round().astype(int)
    tiles = []
    for t in steps:
        img = frame(t, qpos[min(t, len(qpos) - 1), :3] * (1, 1, 0) + (0, 0, 0.25))
        draw = ImageDraw.Draw(img, "RGBA")
        label = f"{t * dt:.2f} s" + ("  (terminated)" if 0 <= term <= t else "")
        draw.text((10, 8), label, font=bold, fill=(28, 30, 34, 220))
        tiles.append(img)
    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.width, rows * args.height), (255, 255, 255))
    for i, img in enumerate(tiles):
        sheet.paste(img, ((i % args.cols) * args.width, (i // args.cols) * args.height))
    if not args.no_mark:
        draw_mark(ImageDraw.Draw(sheet, "RGBA"), sheet.width, sheet.height, _font(0, round(20 * sheet.height / 432 / rows)))
    sheet.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
