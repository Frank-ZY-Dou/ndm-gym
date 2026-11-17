"""Animated WebP grids for the README: rendered clips as the tiles of one synchronized loop, with names in the margins only.

The --clip tiles fill rows of --columns in their order (a shorter last row centred). A clip is an animated WebP (read with
Pillow, by its frames' durations) or anything ffmpeg reads; it starts at START s (default 0), shows its whole frame or the box
X Y W H in its pixels, and is scaled to --tile (Lanczos; the frame or the box must have the tile's aspect). The tiles start
together at --fps. A tile that ends sooner holds its last frame until the longest one ends (or --seconds pass); then all
hold --hold s more and the loop restarts. Names stand in the margins only: --col_labels above the columns, --row_labels left
of the rows ("\\n" breaks a line), or --names above each tile (tiles of different things, e.g. tasks). Nothing is drawn on
the tiles, which keep the NDM Gym marks of their renders; --mark draws one mark at the right end of the top band instead,
for boxes that leave the renders' marks out (an error where it would touch a name).

Two looks: by default dark margins and small grey names, as the hand-object grid (benchmarks/xhand_tracking/scripts/hand_grid.py);
with --light, white margins and gaps and bold names that grow with the tile, as the teaser. The WebP is lossy at --quality,
with a key frame at least every KMAX frames (tools/hoi_gallery.py explains why); a file larger than --max_mb is an error.
Needs Pillow, and ffmpeg for clips other than WebP.

The README's grids, from the repository's root:
  M=docs/media
  C="dance1_subject2 fightAndSports1_subject1 jumps1_subject1 sprint1_subject2 fallAndGetUp2_subject2 fight1_subject2"
  python tools/make_grid.py $M/final/grid.webp --columns 2 --tile 320 187 --fps 12 --col_labels "MLP world model" "RigidFormer world model" \\
      --row_labels "dance1\\nsubject2" "fightAndSports1\\nsubject1" "jumps1\\nsubject1" \\
                   "sprint1\\nsubject2" "fallAndGetUp2\\nsubject2" "fight1\\nsubject2" \\
      $(for c in $C; do echo --clip $M/final/$c.webp --clip $M/rf/$c.webp; done)
  python tools/make_grid.py $M/musclevae/policies/grid.webp --columns 2 --tile 400 225 --fps 15 --col_labels walk run \\
      --clip $M/musclevae/policies/walk.webp --clip $M/musclevae/policies/run.webp
  T="move_suitcase carry_box_over_shoulder carry_and_place_bread_box move_foam move_stool-climb push_box push_door-hand open_door-feet
     open_foldchair-sit roll_ball-hand topple_wood_board_and_cross truman"
  python tools/make_grid.py $M/hoi/policies/grid.webp --columns 3 --tile 320 180 --quality 55 --names $T \\
      $(for t in $T; do echo --clip $M/hoi/policies/$t.webp; done)
  python tools/make_grid.py $M/robokit/grid.webp --columns 3 --tile 320 208 --hold 0.5 --col_labels "G1 box carrying" "G1 climbing" "Go2 jumping" \\
      --clip $M/robokit/g1_carry.webp --clip $M/robokit/g1_climb.webp --clip $M/robokit/go2_jump.webp
The teaser, from renders without labels (tools/render.py --no_labels and the other renderers; HDMI's task at 1.5 times the
speed, the hand-object clip at half speed), boxes that leave their marks out:
  python tools/make_grid.py $M/teaser.webp --light --mark --columns 3 --tile 512 384 --seconds 6 --quality 65 --max_mb 8 \\
      --names "LAFAN1 tracking" "Two G1s" "Muscles" "Hand-object" "Human-object interaction" "RoboKit planning" \\
      --clip lafan1.mp4 0 338 216 1152 864 --clip two_g1s.mp4 0 120 216 1440 1080 --clip muscles_walk.mp4 0.5 390 60 1216 912 \\
      --clip hand_object.mp4 0 176 24 1344 1008 --clip hdmi_task.mp4 0 410 170 1080 810 --clip $M/robokit/go2_jump.webp 1.0 150 80 640 480
"""

import argparse
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # the hand-object grid's names
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # tools/render.py's label font: the teaser's names and mark
DARK = {"back": (17, 18, 22), "ink": (214, 217, 224)}  # the hand-object grid's margins and names
LIGHT = {"back": (255, 255, 255), "ink": (28, 30, 34)}  # the teaser's
MARK_INK = (150, 152, 158)
KMAX = 10  # frames between WebP key frames at most
EPS = 1e-3  # s: WebP durations are whole milliseconds


def source_size(path: str) -> tuple[int, int]:
    """A clip's frame size in pixels."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".webp"):
        with Image.open(path) as im:
            return im.size
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.split(",")  # fmt: skip
    return int(out[0]), int(out[1])


def clip_frames(path: str, start: float, box: tuple[int, int, int, int] | None, size: tuple[int, int], fps: float, limit: int | None) -> np.ndarray:
    """(T, h, w, 3) uint8: the clip at the times start + k / fps up to its end (T at most ``limit``), the box (x, y, w, h) of each
    frame or the whole frame, scaled to ``size``. A WebP frame is the one on screen at that time."""
    if path.lower().endswith(".webp"):
        out, end = [], 0.0
        with Image.open(path) as im:
            for i in range(getattr(im, "n_frames", 1)):
                if limit is not None and len(out) == limit:
                    break
                im.seek(i)
                im.load()  # Pillow sets a WebP frame's duration on loading it
                end += im.info["duration"] / 1000.0
                tile = None
                while (limit is None or len(out) < limit) and start + len(out) / fps + EPS < end:
                    if tile is None:
                        frame = im.convert("RGB") if box is None else im.convert("RGB").crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                        tile = np.asarray(frame.resize(size, Image.LANCZOS))
                    out.append(tile)
        if not out:
            raise ValueError(f"{path}: no frame from {start} s (the clip lasts {end:.2f} s)")
        return np.stack(out)
    crop = "" if box is None else f",crop={box[2]}:{box[3]}:{box[0]}:{box[1]}"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-i", path, *(["-frames:v", str(limit)] if limit else []),
           "-vf", f"fps={fps:g}{crop},scale={size[0]}:{size[1]}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (size[0] * size[1] * 3)
    if not n:
        raise ValueError(f"{path}: no frame from {start} s")
    return np.frombuffer(raw, np.uint8)[: n * size[0] * size[1] * 3].reshape(n, size[1], size[0], 3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", help="the grid, an animated .webp")
    p.add_argument("--clip", nargs="+", action="append", required=True, metavar="ARG",
                   help="a tile, in row-major order: PATH, or PATH START, or PATH START X Y W H (the box in the clip's pixels)")  # fmt: skip
    p.add_argument("--columns", type=int, required=True, help="tiles per row")
    p.add_argument("--tile", type=int, nargs=2, default=(320, 180), metavar=("W", "H"), help="a tile's size in pixels")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=None, help="the loop's length before --hold; default: the longest clip from its start")
    p.add_argument("--hold", type=float, default=0.0, help="seconds all tiles hold their last frames before the loop restarts")
    p.add_argument("--col_labels", nargs="+", default=None, help="one per column, above the grid")
    p.add_argument("--row_labels", nargs="+", default=None, help='one per row, at its left ("\\n" breaks a line)')
    p.add_argument("--names", nargs="+", default=None, help="one per tile, above it")
    p.add_argument("--light", action="store_true", help="the teaser's look: white margins and gaps, bold names that grow with the tile")
    p.add_argument("--mark", action="store_true", help="the NDM Gym mark once, at the right end of the top band")
    p.add_argument("--gap", type=int, default=4, help="pixels between tiles, in the margins' colour")
    p.add_argument("--quality", type=int, default=70, help="WebP quality (lossy), 0-100")
    p.add_argument("--max_mb", type=float, default=5.0, help="the largest file allowed")
    a = p.parse_args()
    tiles = []
    for c in a.clip:
        if len(c) not in (1, 2, 6):
            p.error(f"--clip {' '.join(c)}: give PATH, PATH START or PATH START X Y W H")
        tiles.append((c[0], float(c[1]) if len(c) > 1 else 0.0, tuple(int(v) for v in c[2:]) if len(c) == 6 else None))
    n, C, (tw, th) = len(tiles), a.columns, a.tile
    R = math.ceil(n / C)
    if a.col_labels is not None and (len(a.col_labels) != C or n != R * C):
        p.error(f"--col_labels: {len(a.col_labels)} names for {C} columns, which {n} tiles must fill")
    if a.row_labels is not None and len(a.row_labels) != R:
        p.error(f"--row_labels: {len(a.row_labels)} names for {R} rows")
    if a.names is not None and (len(a.names) != n or a.col_labels is not None):
        p.error(f"--names: {len(a.names)} names for {n} tiles, and not with --col_labels")
    if a.mark and a.names is None and a.col_labels is None:
        p.error("--mark goes in the band of names above the tiles: give --names or --col_labels")
    for path, start, box in tiles:
        w, h = source_size(path)
        if box is not None and (box[0] < 0 or box[1] < 0 or box[0] + box[2] > w or box[1] + box[3] > h):
            raise ValueError(f"{path}: the box {box} leaves its {w} x {h} frame")
        bw, bh = (w, h) if box is None else box[2:]
        if abs(bw * th / (bh * tw) - 1) > 0.02:
            raise ValueError(f"{path}: {'the box' if box else 'the frame'} {bw} x {bh} is not of the tile's aspect {tw} x {th}")
    limit = None if a.seconds is None else round(a.seconds * a.fps)
    frames = [clip_frames(path, start, box, (tw, th), a.fps, limit) for path, start, box in tiles]
    length = max(len(f) for f in frames) if limit is None else limit
    for (path, start, _), f in zip(tiles, frames):
        held = f", its last frame held for {(length - len(f)) / a.fps:.2f} s" if len(f) < length else ""
        print(f"{path}: {len(f)} frames from {start:g} s{held}")
    look = LIGHT if a.light else DARK
    s = th / 225 if a.light else 1.0  # the teaser's names grow with its tiles
    font = ImageFont.truetype(BOLD, round(15 * s)) if a.light else ImageFont.truetype(FONT, 13)
    band = (round(30 * s) if a.light else 24) if (a.names or a.col_labels) else 0
    labels = [t.replace("\\n", "\n") for t in a.row_labels or ()]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left = max(probe.multiline_textbbox((0, 0), t, font=font, spacing=3)[2] for t in labels) + 24 if labels else 0
    step = th + a.gap + (band if a.names else 0)  # from one row's top to the next one's: each row has its band with --names
    row_y = [band + r * step for r in range(R)]
    counts = [min(C, n - r * C) for r in range(R)]
    row_x = [left + (C - k) * (tw + a.gap) // 2 for k in counts]  # a shorter row centred
    W, H = left + C * tw + (C - 1) * a.gap, row_y[-1] + th
    back = Image.new("RGB", (W, H), look["back"])
    draw = ImageDraw.Draw(back)
    top = [(t, (left + j * (tw + a.gap) + tw / 2, band / 2)) for j, t in enumerate(a.col_labels or ())]
    above = [(t, (row_x[i // C] + (i % C) * (tw + a.gap) + tw / 2, row_y[i // C] - band / 2)) for i, t in enumerate(a.names or ())]
    for t, xy in top + above:
        draw.text(xy, t, fill=look["ink"], font=font, anchor="mm")
    for r, t in enumerate(labels):
        draw.multiline_text((left / 2, row_y[r] + th / 2), t, fill=look["ink"], font=font, anchor="mm", align="center", spacing=3)
    if a.mark:  # tools/render.py's mark, in the teaser's grey, clear of the names in the top band
        small = ImageFont.truetype(BOLD, round(11 * s))
        x0 = draw.textbbox((W - 6, band / 2), "NDM Gym", font=small, anchor="rm")[0]
        for t, xy in (top + above)[:C]:
            if draw.textbbox(xy, t, font=font, anchor="mm")[2] + 8 > x0:
                raise ValueError(f"the mark would touch the name {t!r} in the top band: shorten it, or put a shorter name last in the top row")
        draw.text((W - 6, band / 2), "NDM Gym", fill=MARK_INK, font=small, anchor="rm")
    canvas, out = np.asarray(back).copy(), []
    places = [(row_x[i // C] + (i % C) * (tw + a.gap), row_y[i // C]) for i in range(n)]
    for k in range(length + round(a.hold * a.fps)):
        for (x, y), f in zip(places, frames):
            canvas[y : y + th, x : x + tw] = f[min(k, len(f) - 1)]
        out.append(Image.fromarray(canvas.copy()))
    ms = np.round(np.arange(len(out) + 1) * 1000.0 / a.fps)  # whole ms that add up to the frames' times
    tmp = a.out + ".tmp.webp"
    out[0].save(tmp, save_all=True, append_images=out[1:], duration=np.diff(ms).astype(int).tolist(), loop=0, lossless=False,
                quality=a.quality, method=6, kmin=KMAX - 1, kmax=KMAX)  # fmt: skip
    mb = os.path.getsize(tmp) / 1e6
    if mb > a.max_mb:
        os.remove(tmp)
        raise ValueError(f"{a.out}: {mb:.2f} MB at quality {a.quality}, above --max_mb {a.max_mb}: lower --quality, --fps or --tile")
    os.replace(tmp, a.out)
    print(f"{a.out}: {R} x {C} tiles of {tw} x {th}, {len(out)} frames at {a.fps:g} fps ({len(out) / a.fps:.2f} s), {W} x {H}, {mb:.2f} MB")


if __name__ == "__main__":
    main()
