"""Animated WebP grids for the README: rendered clips as the tiles of one synchronized loop, with names in the margins only.

The --clip tiles fill rows of --columns in their order (a shorter last row centred). With --block NAME COLS ROWS [W H], the
clips that follow fill that block's COLS x ROWS cells of --tile (or W x H) instead, each the first cells where it fits, row by
row (a clip may span ROWS x COLS cells, given after its box), with NAME above the block; the blocks, equally tall, fill rows of
--columns (or rows of the counts --row_blocks gives), --block_gap apart, each row centred.
A clip is an animated WebP (read with Pillow, by its frames' durations) or anything ffmpeg reads; it starts at START s
(default 0), shows its whole frame or the box X Y W H in its pixels, and is scaled to its tile (Lanczos; the frame or the box
must have the tile's aspect). The tiles start together at --fps. A tile that ends sooner holds its last frame until the
longest one ends (or --seconds pass); then all hold --hold s more and the loop restarts. Names stand in the margins only:
--col_labels above the columns, --row_labels left of the rows ("\\n" breaks a line), --names above each tile (tiles of
different things, e.g. tasks), or the blocks' names. Nothing is drawn on the tiles, which keep the NDM Gym marks of their
renders; --mark draws one mark at the right end of the top band instead, for boxes that leave the renders' marks out (an
error where it would touch a name).

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
  python tools/make_grid.py $M/mbrl/grid.webp --columns 2 --tile 400 300 --col_labels cart-pole pusher --clip cartpole.mp4 --clip pusher.mp4
      (tools/render_mbrl.py's renders)
  python tools/make_grid.py $M/quadrotor/grid.webp --columns 3 --tile 320 180 --quality 45 --hold 0.8 --max_mb 8 --col_labels ellipse lemniscate "race track" \\
      --row_labels "autonomous\\nSuperTrack" "autonomous\\nSuperMimic" "piloted\\nSuperTrack" "piloted\\nSuperMimic" \\
      $(for r in st sm stp smp; do for t in ellipse lemniscate race_track; do echo --clip ${r}_$t.webp; done; done)
  python tools/make_grid.py $M/go2/grid.webp --columns 3 --tile 320 180 --quality 50 --hold 0.8 --max_mb 6 --col_labels walk run play \\
      --row_labels SuperTrack SuperMimic $(for r in st sm; do for t in walk run play; do echo --clip $M/go2/${t}_$r.webp; done; done)
      (tools/render_quadruped.py's renders of seed 0's single-clip runs, 640 x 360)
      (tools/render_quadrotor.py's renders of 9 s, one lap each, at 20 fps; the piloted ones with --path_seconds 2)
The teaser: a block of 2 x 2 cells per task, from renders without labels (tools/render.py --no_labels and the other
renderers; HDMI's tasks at 1.5 times the speed, the hand-object clips and mbrl-lib's at half speed) in boxes that leave their
marks out (mbrl-lib's: tools/render_mbrl.py --no_mark); the racing quadrotor's a row of its own, four of its renders (the
piloted race track's from its grid) in cells as wide as the blocks above:
  python tools/make_grid.py $M/teaser.webp --light --mark --columns 4 --row_blocks 4 3 1 --tile 256 192 --seconds 6 --quality 60 --max_mb 11 \\
      --block "LAFAN1 tracking" 2 2 --clip $M/final/dance1_subject2.webp 2.33 40 0 281 211 \\
      --clip $M/final/fight1_subject2.webp 3.42 93 13 245 184 --clip $M/final/fallAndGetUp2_subject2.webp 1.17 55 34 267 200 \\
      --clip $M/final/sprint1_subject2.webp 1.5 61 26 260 195 \\
      --block "Two G1s" 2 2 --clip two_g1s.mp4 0 120 216 1440 1080 2 2 \\
      --block Muscles 2 2 --clip walk.mp4 0.5 0 195 1920 714 1 2 --clip run.mp4 0.5 0 195 1920 714 1 2 \\
      --block Hand-object 2 2 --clip xhand_spray_bottle.mp4 0 260 20 1333 1000 --clip inspire_can.mp4 0 330 140 1120 840 \\
      --clip allegro_spray_bottle.mp4 0 290 0 1040 780 --clip xhand_apple.mp4 0 450 180 1000 750 \\
      --block "Human-object interaction" 2 2 --clip carry_box_over_shoulder.mp4 0 410 170 1080 810 \\
      --clip open_foldchair-sit.mp4 0 344 98 1152 864 --clip roll_ball-hand.mp4 0 414 118 1152 864 \\
      --clip topple_wood_board_and_cross.mp4 0 400 130 1120 840 \\
      --block "RoboKit planning" 2 2 --clip $M/robokit/g1_carry.webp 0.2 375 100 343 520 2 1 \\
      --clip g1_climb_1440x936.webp 1 310 150 960 720 --clip $M/robokit/go2_jump.webp 1 150 80 640 480 \\
      --block "mbrl-lib planning" 2 1 522 388 --clip cartpole_nomark.mp4 --clip pusher_nomark.mp4 \\
      --block "Racing quadrotor" 4 1 525 388 --clip $M/quadrotor/race_track.webp 0 60 0 731 540 \\
      --clip $M/quadrotor/grid.webp 3 770 392 244 180 --clip $M/quadrotor/ellipse.webp 0 60 0 731 540 \\
      --clip $M/quadrotor/lemniscate.webp 0 60 0 731 540
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


class InOrder(argparse.Action):
    """Keeps --block and --clip in the order given: a block's tiles are the clips that follow it."""

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.items = [*(getattr(namespace, "items", None) or []), (self.dest, values)]


def place_blocks(blocks, tiles, gap, block_gap, per_row, band):
    """The block layout: (W, H, each tile's (x, y, w, h), each block's name and centre in its band). Blocks of COLS x ROWS cells
    of their size fill rows of ``per_row`` (a list: the blocks of each row), each row centred; every block must be equally tall.
    A block's tiles take the first cells where they fit, row by row, spanning ROWS x COLS cells each, and must cover its cells
    exactly."""
    sizes = [(c * cw + (c - 1) * gap, r * ch + (r - 1) * gap) for _, c, r, _, (cw, ch) in blocks]
    bh = sizes[0][1]
    if any(h != bh for _, h in sizes):
        raise ValueError(f"--block: every block must be equally tall (ROWS x H and the gaps), not {sorted({h for _, h in sizes})}")
    if isinstance(per_row, int):
        rows_of = [range(i, min(i + per_row, len(blocks))) for i in range(0, len(blocks), per_row)]
    else:  # --row_blocks
        ends = np.cumsum(per_row).tolist()
        rows_of = [range(e - k, e) for k, e in zip(per_row, ends)]
    widths = [sum(sizes[b][0] for b in row) + (len(row) - 1) * block_gap for row in rows_of]
    W, H = max(widths), len(rows_of) * (band + bh + block_gap) - block_gap
    places, names = [None] * len(tiles), []
    for r, row in enumerate(rows_of):
        x0, y0 = (W - widths[r]) // 2, r * (band + bh + block_gap) + band
        for b in row:
            name, cols, rows, members, (cw, ch) = blocks[b]
            names.append((name, (x0 + sizes[b][0] / 2, y0 - band / 2)))
            free = np.ones((rows, cols), bool)
            for i in members:
                rs, cs = tiles[i][3]
                spot = next(((y, x) for y in range(rows - rs + 1) for x in range(cols - cs + 1) if free[y : y + rs, x : x + cs].all()), None)
                if spot is None:
                    raise ValueError(f"--block {name!r}: no {rs} x {cs} cells left for {tiles[i][0]}")
                y, x = spot
                free[y : y + rs, x : x + cs] = False
                places[i] = (x0 + x * (cw + gap), y0 + y * (ch + gap), cs * cw + (cs - 1) * gap, rs * ch + (rs - 1) * gap)
            if free.any():
                raise ValueError(f"--block {name!r}: its tiles leave {int(free.sum())} of its {rows * cols} cells empty")
            x0 += sizes[b][0] + block_gap
    return W, H, places, names


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", help="the grid, an animated .webp")
    p.add_argument("--clip", nargs="+", action=InOrder, required=True, metavar="ARG",
                   help="a tile, in order: PATH, or PATH START, or PATH START X Y W H (the box in the clip's pixels); in a block also "
                        "PATH START X Y W H ROWS COLS, the cells it spans")  # fmt: skip
    p.add_argument("--block", nargs="+", action=InOrder, metavar="ARG",
                   help="NAME COLS ROWS [W H]: a block of COLS x ROWS cells of --tile (or W x H), its NAME above it; "
                        "the clips that follow fill it")  # fmt: skip
    p.add_argument("--columns", type=int, required=True, help="tiles per row, or with --block blocks per row")
    p.add_argument("--row_blocks", type=int, nargs="+", default=None, metavar="N",
                   help="with --block: the blocks of each row, in order, instead of rows of --columns")  # fmt: skip
    p.add_argument("--block_gap", type=int, default=16, help="pixels between blocks")
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
    tiles, blocks = [], []  # tiles: (path, start, box, (rows, cols) spanned); blocks: [name, cols, rows, its tiles]
    for kind, v in a.items:
        if kind == "block":
            if len(v) not in (3, 5):
                p.error(f"--block {' '.join(v)}: give NAME COLS ROWS, or NAME COLS ROWS W H")
            blocks.append((v[0], int(v[1]), int(v[2]), [], (int(v[3]), int(v[4])) if len(v) == 5 else tuple(a.tile)))
            continue
        if len(v) not in (1, 2, 6, 8) or (len(v) == 8 and not blocks):
            p.error(f"--clip {' '.join(v)}: give PATH, PATH START or PATH START X Y W H, and in a block ROWS COLS after the box")
        if any(k == "block" for k, _ in a.items) and not blocks:
            p.error(f"--clip {v[0]}: before the first --block")
        if blocks:
            blocks[-1][3].append(len(tiles))
        tiles.append((v[0], float(v[1]) if len(v) > 1 else 0.0, tuple(int(x) for x in v[2:6]) if len(v) >= 6 else None,
                      (int(v[6]), int(v[7])) if len(v) == 8 else (1, 1)))  # fmt: skip
    n, C, (tw, th) = len(tiles), a.columns, a.tile
    R = math.ceil(n / C)
    if blocks and (a.col_labels or a.row_labels or a.names):
        p.error("--col_labels, --row_labels and --names label a grid without blocks; a block carries its name")
    if a.col_labels is not None and (len(a.col_labels) != C or n != R * C):
        p.error(f"--col_labels: {len(a.col_labels)} names for {C} columns, which {n} tiles must fill")
    if a.row_labels is not None and len(a.row_labels) != R:
        p.error(f"--row_labels: {len(a.row_labels)} names for {R} rows")
    if a.names is not None and (len(a.names) != n or a.col_labels is not None):
        p.error(f"--names: {len(a.names)} names for {n} tiles, and not with --col_labels")
    if a.row_blocks is not None and (not blocks or min(a.row_blocks) < 1 or sum(a.row_blocks) != len(blocks)):
        p.error(f"--row_blocks {' '.join(map(str, a.row_blocks))}: blocks per row, with --block, adding up to its {len(blocks)} blocks")
    if a.mark and a.names is None and a.col_labels is None and not blocks:
        p.error("--mark goes in the band of names above the tiles: give --names, --col_labels or --block")
    look = LIGHT if a.light else DARK
    labels = [t.replace("\\n", "\n") for t in a.row_labels or ()]
    s = ((blocks[0][2] * blocks[0][4][1] + (blocks[0][2] - 1) * a.gap) if blocks else th) / 225 if a.light else 1.0  # grow with a block or tile
    font = ImageFont.truetype(BOLD, round(15 * s)) if a.light else ImageFont.truetype(FONT, 13)
    if blocks:
        band = round(30 * s) if a.light else 24
        W, H, places, top = place_blocks(blocks, tiles, a.gap, a.block_gap, a.row_blocks or C, band)
        left, row_y = 0, []
    else:
        band = (round(30 * s) if a.light else 24) if (a.names or a.col_labels) else 0
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        left = max(probe.multiline_textbbox((0, 0), t, font=font, spacing=3)[2] for t in labels) + 24 if labels else 0
        step = th + a.gap + (band if a.names else 0)  # from one row's top to the next one's: each row has its band with --names
        row_y = [band + r * step for r in range(R)]
        counts = [min(C, n - r * C) for r in range(R)]
        row_x = [left + (C - k) * (tw + a.gap) // 2 for k in counts]  # a shorter row centred
        W, H = left + C * tw + (C - 1) * a.gap, row_y[-1] + th
        places = [(row_x[i // C] + (i % C) * (tw + a.gap), row_y[i // C], tw, th) for i in range(n)]
        top = [(t, (left + j * (tw + a.gap) + tw / 2, band / 2)) for j, t in enumerate(a.col_labels or ())]
        top += [(t, (row_x[i // C] + (i % C) * (tw + a.gap) + tw / 2, row_y[i // C] - band / 2)) for i, t in enumerate(a.names or ())]
    for (path, start, box, _), (_, _, pw, ph) in zip(tiles, places):
        w, h = source_size(path)
        if box is not None and (box[0] < 0 or box[1] < 0 or box[0] + box[2] > w or box[1] + box[3] > h):
            raise ValueError(f"{path}: the box {box} leaves its {w} x {h} frame")
        bw, bh = (w, h) if box is None else box[2:]
        if abs(bw * ph / (bh * pw) - 1) > 0.02:
            raise ValueError(f"{path}: {'the box' if box else 'the frame'} {bw} x {bh} is not of the tile's aspect {pw} x {ph}")
    limit = None if a.seconds is None else round(a.seconds * a.fps)
    frames = [clip_frames(path, start, box, (pw, ph), a.fps, limit) for (path, start, box, _), (_, _, pw, ph) in zip(tiles, places)]
    length = max(len(f) for f in frames) if limit is None else limit
    for (path, start, _, _), f in zip(tiles, frames):
        held = f", its last frame held for {(length - len(f)) / a.fps:.2f} s" if len(f) < length else ""
        print(f"{path}: {len(f)} frames from {start:g} s{held}")
    back = Image.new("RGB", (W, H), look["back"])
    draw = ImageDraw.Draw(back)
    for t, xy in top:
        draw.text(xy, t, fill=look["ink"], font=font, anchor="mm")
    for r, t in enumerate(labels):
        draw.multiline_text((left / 2, row_y[r] + th / 2), t, fill=look["ink"], font=font, anchor="mm", align="center", spacing=3)
    if a.mark:  # tools/render.py's mark, in the teaser's grey, clear of the names in the top band
        small = ImageFont.truetype(BOLD, round(11 * s))
        x0 = draw.textbbox((W - 6, band / 2), "NDM Gym", font=small, anchor="rm")[0]
        for t, xy in top[: a.row_blocks[0] if a.row_blocks else C]:  # the top row's names
            if draw.textbbox(xy, t, font=font, anchor="mm")[2] + 8 > x0:
                raise ValueError(f"the mark would touch the name {t!r} in the top band: shorten it, or put a shorter name last in the top row")
        draw.text((W - 6, band / 2), "NDM Gym", fill=MARK_INK, font=small, anchor="rm")
    canvas, out = np.asarray(back).copy(), []
    for k in range(length + round(a.hold * a.fps)):
        for (x, y, pw, ph), f in zip(places, frames):
            canvas[y : y + ph, x : x + pw] = f[min(k, len(f) - 1)]
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
    shape = f"{len(blocks)} blocks of {blocks[0][1]} x {blocks[0][2]} cells" if blocks else f"{R} x {C} tiles"  # the first block's
    print(f"{a.out}: {shape} of {tw} x {th}, {len(out)} frames at {a.fps:g} fps ({len(out) / a.fps:.2f} s), {W} x {H}, {mb:.2f} MB")


if __name__ == "__main__":
    main()
