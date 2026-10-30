"""Side-by-side videos of two policies on clips both tracked to the end, for the README.

From two recorded evaluations of the same clips (``python -m ndm_gym.evaluate --ckpt ... --record traj.npz``): one
clip per motion category that both policies completed, each cut to the window with the most reference joint motion,
rendered with ``tools/render.py`` (MuJoCo, EGL) and stacked left | right as an MP4 plus an animated WebP preview
(GitHub shows it inline, at a fraction of a GIF's size; the gallery shows the previews, the MP4s stay local). With
``--readme``, the gallery replaces the part of that file between ``<!-- videos:begin -->`` and ``<!-- videos:end -->``
(``--marker`` names another section, e.g. ``videos_rf``).

``MUJOCO_GL=egl python tools/compare_videos.py traj_st.npz traj_sm.npz --labels SuperTrack SuperMimic --seed 2
--out docs/media/final --readme README.md``; ``--layout_only`` rewrites the gallery from the files already in ``--out``.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 50  # control and reference frame rate
CATEGORIES = ["dance", "fightAndSports", "jumps", "sprint", "fallAndGetUp", "fight", "run", "walk"]


def category(clip):
    return re.match(r"([A-Za-z]+?)\d", clip).group(1)


def completed(d):
    """Clips whose episode lasted the whole clip (alive also turns off when a clip finishes, after clip_len - 1 steps)."""
    return {str(c) for c, k, n in zip(d["names"], d["alive"].sum(0), d["clip_len"]) if k >= n - 1}


def busiest_window(d, clip, seconds):
    """(start, length) in seconds and the summed reference joint speed of the clip's window with the most motion."""
    c = list(d["names"]).index(clip)
    s, n = int(d["clip_start"][c]), int(d["clip_len"][c])
    speed = np.abs(np.diff(d["ref_q"][s : s + n], axis=0)).sum(1)
    w = min(int(seconds * FPS), len(speed))
    acc = np.convolve(speed, np.ones(w), mode="valid")
    i = int(np.argmax(acc))
    return i / FPS, w / FPS, float(acc[i])


def pick(d_a, d_b, n, seconds):
    """Up to n clips both completed, one per category in CATEGORIES' order: the one whose busiest window moves most."""
    both = completed(d_a) & completed(d_b)
    out = []
    for cat in CATEGORIES:
        cands = sorted(c for c in both if category(c) == cat)
        if cands:
            out.append(max(cands, key=lambda c: busiest_window(d_a, c, seconds)[2]))
        if len(out) == n:
            break
    return out


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traj", nargs=2, help="recorded evaluations: left, right")
    p.add_argument("--labels", nargs=2, required=True)
    p.add_argument("--seed", type=int, default=None, help="shown in the titles")
    p.add_argument("--n", type=int, default=6, help="number of clips")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", required=True, help="directory for the MP4s and GIFs")
    p.add_argument("--width", type=int, default=640, help="width of each side")
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--preview_width", type=int, default=800, help="width of the side-by-side WebP preview")
    p.add_argument("--preview_fps", type=int, default=12)
    p.add_argument("--preview_quality", type=int, default=60, help="WebP quality, 0-100")
    p.add_argument("--readme", default=None, help="markdown file whose videos section is replaced")
    p.add_argument("--caption", default="", help="caption under the gallery")
    p.add_argument("--marker", default="videos", help="the --readme section to replace: <!-- <marker>:begin --> to <!-- <marker>:end -->")
    p.add_argument("--cols", type=int, default=2, help="previews per gallery row")
    p.add_argument("--layout_only", action="store_true", help="reuse the MP4s and previews in --out, render nothing")
    a = p.parse_args()
    d = [np.load(t) for t in a.traj]
    assert list(d[0]["names"]) == list(d[1]["names"]), "the recordings must cover the same clips"
    os.makedirs(a.out, exist_ok=True)
    entries = []
    clips = pick(d[0], d[1], a.n, a.seconds)
    if not clips:
        sys.exit("no clip was completed by both policies; nothing to render")
    for clip in clips:
        start, length, _ = busiest_window(d[0], clip, a.seconds)
        mp4, preview = os.path.join(a.out, f"{clip}.mp4"), os.path.join(a.out, f"{clip}.webp")
        if a.layout_only:
            assert os.path.exists(mp4) and os.path.exists(preview), f"--layout_only: {mp4} or {preview} is missing"
        else:
            sides = []
            for traj, label, side in zip(a.traj, a.labels, ("a", "b")):
                path = os.path.join(a.out, f"{clip}_{side}.mp4")
                sub = [] if a.seed is None else ["--subtitle", f"seed {a.seed}"]
                run([sys.executable, os.path.join(HERE, "render.py"), traj, path, "--clips", clip, "--start", f"{start:.2f}",
                     "--seconds", f"{length:.2f}", "--title", label, *sub, "--width", str(a.width), "--height", str(a.height)])
                sides.append(path)
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", sides[0], "-i", sides[1], "-filter_complex", "hstack=inputs=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4])
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", f"fps={a.preview_fps},scale={a.preview_width}:-1:flags=lanczos",
                 "-c:v", "libwebp_anim", "-quality", str(a.preview_quality), "-loop", "0", "-an", preview])
            for s in sides:
                os.remove(s)
        entries.append((clip, start, length, mp4, preview))
        print(f"{clip}: {start:.1f}-{start + length:.1f} s -> {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB), "
              f"{preview} ({os.path.getsize(preview) / 1e6:.1f} MB)", flush=True)
    if a.readme:
        rel = lambda path: os.path.relpath(path, os.path.dirname(os.path.abspath(a.readme)))  # noqa: E731
        lines = ["<table>"]
        for i in range(0, len(entries), a.cols):
            lines += ["<tr>"] + [f'<td><img src="{rel(preview)}" alt="{clip}"/></td>' for clip, _, _, _, preview in entries[i : i + a.cols]] + ["</tr>"]
        lines += ["</table>", f'<p align="center"><i>{a.caption}</i></p>']
        text = open(a.readme).read()
        begin, end = f"<!-- {a.marker}:begin -->", f"<!-- {a.marker}:end -->"
        i, j = text.index(begin) + len(begin), text.index(end)
        open(a.readme, "w").write(text[:i] + "\n" + "\n".join(lines) + "\n" + text[j:])
        print(f"gallery of {len(entries)} clips -> {a.readme}")


if __name__ == "__main__":
    main()
