"""Two rendered videos side by side as one animated WebP for the README, and optionally an MP4 of the full frames. Each half is
labelled in its top-left corner as tools/render.py labels its renders (a bold title, then a lighter subtitle, on a
translucent white pill, sized for the half's height), so that at the README's width the labels look like those of the LAFAN1
previews (tools/compare_videos.py). For renders without labels of their own, such as tools/render_hoi.py's and
tools/render_musclevae.py's (which carry the NDM Gym mark); without --labels the halves are joined as they are.

The inputs (MP4s, or anything ffmpeg reads at a constant frame rate; or animated WebPs, read with Pillow) must have the same
height and duration, and two ffmpeg inputs the same frame rate and frame count; with --hold_last the shorter one holds its
last frame until the longer one ends. The outputs run at --fps (default: the
inputs' rate), each frame showing the inputs' frames at its time. The WebP is --width pixels wide (Lanczos), lossy, with a
key frame at least every KMAX frames, as tools/hoi_gallery.py's previews (between key frames libwebp keeps the blocks within
its tolerance of the previous frame, so that slowly changing shading would drift); its quality starts at --quality and steps
down by 5 until the file is at most --max_mb, an error if it is still larger at --min_quality.

Usage: python tools/side_by_side.py st.mp4 sm.mp4 out.webp --labels SuperTrack SuperMimic
       [--subtitles "seed 0" "seed 0"] [--fps 25] [--mp4 side_by_side.mp4]
"""

import argparse
import json
import os
import subprocess
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw

from render import INK, MUTED, _font, _pill  # tools/render.py: its labels

KMAX = 10  # frames between WebP key frames at most (tools/hoi_gallery.py)


class Video:
    """A video's frames in order, each from the time it starts (s): an animated WebP through Pillow (its frames' durations),
    anything else through ffmpeg (its stream's constant frame rate; ``fps`` None for a WebP). ``at(t)`` returns the frame
    shown at time t, for t increasing from call to call; ``close()`` reads the rest and checks the frame count."""

    def __init__(self, path: str):
        self.path, self.k, self.frame, self._ff = path, -1, None, None
        if path.lower().endswith(".webp"):
            self.fps, self._im, ms = None, Image.open(path), []
            for i in range(getattr(self._im, "n_frames", 1)):
                self._im.seek(i)
                self._im.load()  # Pillow sets a WebP frame's duration on loading it
                ms.append(self._im.info["duration"])
            self.size, self.starts, self.end = self._im.size, np.cumsum([0] + ms[:-1]) / 1000.0, sum(ms) / 1000.0
            return
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                              "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets", "-of", "json", path],
                             capture_output=True, text=True, check=True).stdout  # fmt: skip
        streams = json.loads(out)["streams"]
        if not streams:
            raise ValueError(f"{path}: no video stream")
        s = streams[0]
        if s["r_frame_rate"] != s["avg_frame_rate"]:
            raise ValueError(f"{path}: frame rate {s['r_frame_rate']}, on average {s['avg_frame_rate']}: not constant")
        self.size, self.fps, n = (int(s["width"]), int(s["height"])), Fraction(s["r_frame_rate"]), int(s["nb_read_packets"])
        self.starts, self.end = np.arange(n) / float(self.fps), n / float(self.fps)

    def _next(self) -> np.ndarray:
        self.k += 1
        if self.fps is None:
            self._im.seek(self.k)
            return np.asarray(self._im.convert("RGB"))
        if self._ff is None:
            self._ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", self.path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                        stdout=subprocess.PIPE)  # fmt: skip
        w, h = self.size
        buf = self._ff.stdout.read(w * h * 3)
        if len(buf) != w * h * 3:
            raise RuntimeError(f"{self.path}: ffmpeg gave {self.k} frames, ffprobe counted {len(self.starts)}")
        return np.frombuffer(buf, np.uint8).reshape(h, w, 3)

    def at(self, t: float) -> np.ndarray:
        i = int(np.searchsorted(self.starts, t + 1e-9, side="right")) - 1
        if i < self.k:
            raise ValueError("Video.at: times must increase")
        while self.k < i:
            self.frame = self._next()
        return self.frame

    def close(self):
        if self.fps is None:
            self._im.close()
            return
        while self.k + 1 < len(self.starts):
            self._next()
        if self._ff is None:  # no frames
            return
        extra = len(self._ff.stdout.read())
        if self._ff.wait() != 0 or extra:
            raise RuntimeError(f"{self.path}: ffmpeg failed or gave more frames than ffprobe counted")


def label_overlay(size: tuple[int, int], title: str, subtitle: str) -> Image.Image:
    """A transparent frame with tools/render.py's label at its top left: its fonts, margin and pill for this height."""
    w, h = size
    s = h / 432
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    parts = [(title, _font(0, round(20 * s)), INK)] + ([(subtitle, _font(1, round(18 * s)), MUTED)] if subtitle else [])
    _pill(ImageDraw.Draw(over), (round(12 * s), round(12 * s)), parts)
    return over


def save_webp(frames: list[Image.Image], path: str, fps: Fraction, quality: int, min_quality: int, max_mb: float) -> int:
    """The frames as a looping lossy WebP at fps (durations in whole ms that add up to the frames' times), a key frame at
    least every KMAX frames, at the highest quality from ``quality`` down in steps of 5 whose file is at most max_mb;
    returns that quality."""
    ms = np.round(np.arange(len(frames) + 1) * 1000.0 / float(fps))
    tmp = path + ".tmp"
    for q in range(quality, min_quality - 1, -5):
        frames[0].save(tmp, format="WEBP", save_all=True, append_images=frames[1:], duration=np.diff(ms).astype(int).tolist(), loop=0,
                       lossless=False, quality=q, method=6, kmin=KMAX - 1, kmax=KMAX)  # fmt: skip
        if os.path.getsize(tmp) <= max_mb * 1e6:
            os.replace(tmp, path)
            return q
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    raise ValueError(f"{path}: {size:.2f} MB at quality {q}, above --max_mb {max_mb}: lower --fps or --width, or raise --max_mb")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("out", help="the animated WebP")
    p.add_argument("--labels", nargs=2, default=None, metavar=("LEFT", "RIGHT"), help="bold titles, e.g. the methods")
    p.add_argument("--subtitles", nargs=2, default=None, metavar=("LEFT", "RIGHT"), help="after the titles, e.g. the seeds")
    p.add_argument("--fps", type=float, default=None, help="frame rate of the outputs; default the inputs' (MP4s)")
    p.add_argument("--width", type=int, default=800, help="width of the WebP")
    p.add_argument("--quality", type=int, default=60, help="the WebP's highest quality, 0-100")
    p.add_argument("--min_quality", type=int, default=30)
    p.add_argument("--max_mb", type=float, default=3.0, help="the WebP's largest size")
    p.add_argument("--mp4", default=None, help="also an MP4 of the full frames (H.264, CRF 18) here")
    p.add_argument("--hold_last", action="store_true",
                   help="inputs of different lengths: the shorter one holds its last frame until the longer one ends (e.g. an episode that "
                        "ended earlier); without it they must match")
    a = p.parse_args()
    if a.subtitles and not a.labels:
        p.error("--subtitles need --labels")
    if not 0 <= a.min_quality <= a.quality <= 100:
        p.error("--min_quality and --quality must satisfy 0 <= min_quality <= quality <= 100")
    va, vb = Video(a.left), Video(a.right)
    if va.size[1] != vb.size[1]:
        raise ValueError(f"heights differ: {a.left} is {va.size[0]} x {va.size[1]}, {a.right} {vb.size[0]} x {vb.size[1]}")
    if va.fps is not None and vb.fps is not None and (va.fps != vb.fps or (len(va.starts) != len(vb.starts) and not a.hold_last)):
        raise ValueError(f"{a.left}: {len(va.starts)} frames at {va.fps} fps, {a.right}: {len(vb.starts)} at {vb.fps}; they must match")
    if a.fps is None and (va.fps is None or va.fps != vb.fps):
        p.error("--fps is needed for WebP inputs")
    fps = va.fps if a.fps is None else Fraction(a.fps).limit_denominator(1001)
    if abs(va.end - vb.end) > 0.5 / float(fps) and not a.hold_last:
        raise ValueError(f"{a.left} lasts {va.end:.3f} s, {a.right} {vb.end:.3f} s; they must last equally long (or --hold_last)")
    n = int(np.floor((max if a.hold_last else min)(va.end, vb.end) * float(fps) + 1e-6))
    W, H = va.size[0] + vb.size[0], va.size[1]
    h = int(round(H * a.width / W))
    labels = [label_overlay(v.size, t, s) for v, t, s in zip((va, vb), a.labels, a.subtitles or ("", ""))] if a.labels else None
    ff = None
    if a.mp4:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.mp4], stdin=subprocess.PIPE)  # fmt: skip
    previews = []
    for k in range(n):
        t = k / float(fps)
        halves = [Image.fromarray(v.at(t)) for v in (va, vb)]
        if labels:
            halves = [Image.alpha_composite(im.convert("RGBA"), over).convert("RGB") for im, over in zip(halves, labels)]
        frame = np.concatenate([np.asarray(im) for im in halves], axis=1)
        if ff is not None:
            ff.stdin.write(frame.tobytes())
        previews.append(Image.fromarray(frame).resize((a.width, h), Image.LANCZOS))
    va.close()
    vb.close()
    if ff is not None:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {a.mp4}")
    q = save_webp(previews, a.out, fps, a.quality, a.min_quality, a.max_mb)
    print(f"wrote {a.out}: {n} frames at {float(fps):g} fps ({n / float(fps):.2f} s), {a.width} x {h}, quality {q}, "
          f"{os.path.getsize(a.out) / 1e6:.2f} MB" + (f"; {a.mp4}: {W} x {H}" if a.mp4 else ""))


if __name__ == "__main__":
    main()
