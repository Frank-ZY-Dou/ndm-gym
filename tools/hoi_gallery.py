"""The README's gallery of HDMI's 13 human-object interaction tasks: each task's reference clip, rendered by
tools/render_hoi.py and cut to the window of at most --seconds with the most interaction (steps where a corner of the
bounds of the object's contact body moves faster than render_hoi.MOVING, or the clip flags contact), as an MP4 and an
animated WebP preview (GitHub shows it inline; the gallery shows the previews, the MP4s stay local). With --readme, a grid
of --cols previews per row, captioned with the task names, replaces the part of that file between <!-- hoi:begin --> and
<!-- hoi:end -->.

The previews are lossy WebP with a key frame at least every KMAX frames (Pillow's libwebp encoder). Between key frames,
libwebp keeps every 8 x 8 block of a frame that is within its quality's tolerance of the previous frame (8 levels at
quality 60), so shading that changes slowly (a door turning, a soft shadow passing) drifts in stale blocks until the next
key frame; ffmpeg's encoder inserts none.

``MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 python tools/hoi_gallery.py --out docs/media/hoi --readme README.md``;
``--previews_only`` re-encodes the previews from the MP4s in --out, ``--layout_only`` rewrites the grid from its previews.
"""

import argparse
import os
import subprocess

import numpy as np
from PIL import Image

import render_hoi as rh
from ndm_gym.tasks.hdmi_tasks import TASKS

HDMI_URL = "https://github.com/LeCAR-Lab/HDMI"
CAPTION = ("The reference motions of HDMI's 13 tasks with their objects: per task the 10 s (or the whole clip) with the most object "
           "motion or contact, in the look of HDMI's simulation videos.")  # fmt: skip
ATTRIBUTION = (f'Renders of HDMI\'s motions and objects (<a href="{HDMI_URL}">{HDMI_URL}</a> @ 32282f6), cut to at most 10 s and rendered; '
               "not covered by this repository's MIT license.")  # fmt: skip
KMAX = 10  # frames between WebP key frames at most


def interaction(task: str) -> tuple[rh.Motion, np.ndarray]:
    """A task's reference clip and, per step, whether a corner of its object's contact body moves faster than MOVING."""
    m = rh.clip_motion(task)
    scene = rh.scene_of(task)
    obj = rh.usd_asset(scene.object_usd)
    poses = rh.asset_poses(obj, m.object_pos, m.object_quat, m.hinge, scene.hinge)
    corners = rh.contact_body_corners(obj, poses, scene.object_body)
    speed = np.linalg.norm(np.diff(corners, axis=0), axis=-1).max(-1) * m.fps
    return m, np.concatenate((speed[:1], speed)) > rh.MOVING


def webp(mp4: str, out: str, width: int, height: int, fps: float, quality: int):
    """The MP4's frames at fps, scaled to width x height (Lanczos), as a looping lossy WebP of that quality with a key frame
    at least every KMAX frames; frame durations in whole ms that add up to the frames' times."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf", f"fps={fps:g},scale={width}:{height}:flags=lanczos", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout  # fmt: skip
    frames = [Image.fromarray(f) for f in np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)]
    ms = np.round(np.arange(len(frames) + 1) * 1000.0 / fps)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=np.diff(ms).astype(int).tolist(), loop=0, lossless=False,
                   quality=quality, method=6, kmin=KMAX - 1, kmax=KMAX)  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="directory for the MP4s and the WebP previews")
    p.add_argument("--tasks", nargs="*", default=list(TASKS), help="default: all 13, in ndm_gym.tasks.hdmi_tasks' order")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=30.0, help="video and preview frame rate (poses interpolated from 50 Hz)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--preview_width", type=int, default=800)
    p.add_argument("--preview_quality", type=int, default=60, help="WebP quality, 0-100")
    p.add_argument("--azimuth", nargs="*", default=[], help="TASK=DEGREES: a camera heading instead of render_hoi's choice")
    p.add_argument("--readme", default=None, help="markdown file whose hoi section is replaced")
    p.add_argument("--cols", type=int, default=4, help="previews per gallery row")
    p.add_argument("--previews_only", action="store_true", help="re-encode the previews from the MP4s in --out, render nothing")
    p.add_argument("--layout_only", action="store_true", help="reuse the previews in --out, render nothing")
    a = p.parse_args()
    unknown = sorted(set(a.tasks) - set(TASKS))
    if unknown:
        raise KeyError(f"unknown tasks {unknown}")
    azimuth = {k: float(v) for k, v in (s.split("=") for s in a.azimuth)}
    preview_height = int(round(a.preview_width * a.height / a.width / 2)) * 2
    os.makedirs(a.out, exist_ok=True)
    for task in a.tasks:
        mp4, preview = os.path.join(a.out, f"{task}.mp4"), os.path.join(a.out, f"{task}.webp")
        if a.layout_only:
            if not os.path.isfile(preview):
                raise FileNotFoundError(f"--layout_only: {preview} is missing")
            continue
        line = f"{task:28s} "
        if not a.previews_only:
            m, moving = interaction(task)
            length = min(int(round(a.seconds * m.fps)), len(m.root_pos))
            s = rh.interaction_window(moving, m.contact, length)
            info = rh.render(m, mp4, s / m.fps, length / m.fps, a.fps, a.width, a.height, azimuth.get(task))
            act = moving[s : s + length] | (m.contact[s : s + length].any(-1) if m.contact is not None else False)
            line += (f"{s / m.fps:5.1f}-{(s + length) / m.fps:5.1f} s of {len(m.root_pos) / m.fps:5.1f} s, {act.mean():4.0%} interaction, "
                     f"heading {info['azimuth']:3.0f}: ")  # fmt: skip
        elif not os.path.isfile(mp4):
            raise FileNotFoundError(f"--previews_only: {mp4} is missing")
        webp(mp4, preview, a.preview_width, preview_height, a.fps, a.preview_quality)
        print(line + f"{os.path.getsize(mp4) / 1e6:5.1f} MB mp4, {os.path.getsize(preview) / 1e6:4.2f} MB webp", flush=True)
    if a.readme:
        rel = lambda path: os.path.relpath(path, os.path.dirname(os.path.abspath(a.readme)))  # noqa: E731
        cells = [f'<td align="center"><img src="{rel(os.path.join(a.out, t + ".webp"))}" alt="{t}"/><br><sub>{t}</sub></td>' for t in a.tasks]
        lines = ["<table>"]
        for i in range(0, len(cells), a.cols):
            lines += ["<tr>"] + cells[i : i + a.cols] + ["</tr>"]
        lines += ["</table>", f'<p align="center"><i>{CAPTION}</i></p>', f'<p align="center"><sub>{ATTRIBUTION}</sub></p>']
        text = open(a.readme).read()
        begin, end = "<!-- hoi:begin -->", "<!-- hoi:end -->"
        i, j = text.index(begin) + len(begin), text.index(end)
        open(a.readme, "w").write(text[:i] + "\n" + "\n".join(lines) + "\n" + text[j:])
        print(f"gallery of {len(cells)} tasks -> {a.readme}")


if __name__ == "__main__":
    main()
