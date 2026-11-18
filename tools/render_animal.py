"""Render an animal tracking rollout (ndm_gym.evaluate_animal --record's rollout.npz) in MIMIC-MJX's arena: the simulated body
solid, the reference a translucent blue ghost at its own place, a camera that follows them, a clip and time label and the
NDM Gym mark (tools/render.py's).

    MUJOCO_GL=egl python tools/render_animal.py --rollout logs/<run>/eval/rollout.npz --out rat.mp4 [--world 0] [--speed 0.5]

``--world``: which of the rollout's episodes (the clips in order, then their perturbed repeats); ``--speed``: playback speed
(the fly's clips last 1.2 s: 0.25 shows them at a quarter). The episode's frames after it ended are not drawn. Needs ffmpeg (or the imageio-ffmpeg package's).
"""

import argparse
import os
import shutil
import subprocess
import sys

from render import _font, _pill, check_egl_device, draw_mark  # tools/render.py: labels and the mark; importing it sets up EGL

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.robots.animals import ANIMALS, build_model

GHOST_RGBA = (0.25, 0.58, 0.95, 0.25)
INK, MUTED = (28, 30, 34), (92, 98, 108)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "mimic_mjx"))
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", default="")
    ap.add_argument("--distance", type=float, default=None, help="the camera's distance in body sizes (default 3.5)")
    a = ap.parse_args()
    check_egl_device()
    d = np.load(a.rollout)
    animal = str(d["animal"])
    spec = ANIMALS[animal]
    model = build_model(animal, a.data_dir, ghost_rgba=GHOST_RGBA)
    nq = (model.nq) // 2
    data = mujoco.MjData(model)
    qpos, frame, ref = d["qpos"][a.world], d["frame"][a.world], d["ref_qpos"]
    alive = np.ones(len(frame), dtype=bool)
    alive[1:] = frame[1:] != frame[:-1]  # an ended episode's frame stops at its clip's last
    T = int(alive.sum()) if not alive.all() else len(frame)
    size = float(np.linalg.norm(np.ptp(ref[:, :3], 0))) or 0.1
    reach = {"rat": 0.16, "fly": 0.26, "stick": 0.018}[animal]
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = (a.distance or 3.5) * reach, 140.0, -18.0
    s = a.height / 432
    bold, small = _font(0, round(20 * s)), _font(1, round(15 * s))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:  # the imageio-ffmpeg package's binary
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], stdin=subprocess.PIPE)  # fmt: skip
    step = 1.0 / spec.ctrl_dt * a.speed / a.fps  # control steps per video frame
    look = None
    t = 0.0
    start = int(d["start"][a.world])
    while t < T:
        k = int(t)
        data.qpos[:nq] = qpos[k]
        data.qpos[nq:] = ref[frame[k]]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        target = np.array([qpos[k, 0], qpos[k, 1], qpos[k, 2]])
        look = target if look is None else look + 0.15 * (target - look)
        cam.lookat[:] = look
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render()).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(over)
        m = round(12 * s)
        if a.title:
            _pill(draw, (m, m), [(a.title, bold, INK)])
        _pill(draw, (a.width - m, m), [(f"{animal}   {(frame[k] - start) * spec.ctrl_dt:4.2f} s", small, MUTED)], anchor="ra")
        draw_mark(draw, a.width, a.height, bold)
        ff.stdin.write(np.asarray(Image.alpha_composite(img, over).convert("RGB")).tobytes())
        t += step
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out, f"({size:.3g} root span)")


if __name__ == "__main__":
    main()
