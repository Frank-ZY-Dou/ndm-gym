"""The README's hand grids: hand_grid.py <dir of the renders> <out.webp> [--fps 20] [--quality 70] [--hold 0.8] [--hands ...]
[--clips ...] [--algorithms ...]. The renders of tools/render_hand.py (`<hand>_<algorithm>_<clip>.mp4`, by default for xhand, allegro, inspire;
supertrack, supermimic; apple, oakink_a01002, can) as one synchronized loop: rows hand and algorithm, columns clip, tiles 320x180; a
tile holds its last frame until the longest clip has ended and the hold has passed, then all restart. Labels in the margins only.
Needs ffmpeg and Pillow.

The more tasks' grids, from the renders' directory:
  python hand_grid.py . more_tasks.webp --clips mustard_bottle="mustard bottle" mug=mug
  python hand_grid.py . tea.webp --clips tea="making tea"
  python hand_grid.py . tool_use.webp --hands xhand=XHand --clips stir_beaker="stir beaker" pour_tube="pour tube"
"""

import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HANDS = ("xhand=XHand", "allegro=Allegro", "inspire=Inspire")
ALGOS = ("supertrack=SuperTrack", "supermimic=SuperMimic")
CLIPS = ("apple=apple", "oakink_a01002=spray bottle", "can=can")
W, H, LEFT, TOP = 320, 180, 92, 24
BACK, TEXT = (17, 18, 22), (214, 217, 224)
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")


def tile_frames(path: str, fps: int) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--hold", type=float, default=0.8, help="seconds the last frames are held before the loop restarts")
    parser.add_argument("--hands", nargs="+", default=HANDS, metavar="NAME=LABEL", help="the rows' hands, top to bottom")
    parser.add_argument("--clips", nargs="+", default=CLIPS, metavar="NAME=LABEL", help="the columns' clips, left to right")
    parser.add_argument("--algorithms", nargs="+", default=ALGOS, metavar="NAME=LABEL", help="each hand's rows' algorithms")
    args = parser.parse_args()
    hands, clips, algos = (tuple(tuple(item.split("=", 1)) for item in items) for items in (args.hands, args.clips, args.algorithms))
    if any(len(item) != 2 for item in hands + clips + algos):
        parser.error("--hands, --clips and --algorithms take NAME=LABEL items")
    font_path = next((p for p in FONTS if os.path.exists(p)), None)
    if font_path is None:
        raise FileNotFoundError(f"no label font among {FONTS}")
    font = ImageFont.truetype(font_path, 13)
    rows = [(hand, alg, f"{hand_label}\n{alg_label}") for hand, hand_label in hands for alg, alg_label in algos]
    tiles = {(i, j): tile_frames(os.path.join(args.src, f"{hand}_{alg}_{clip}.mp4"), args.fps)
             for i, (hand, alg, _) in enumerate(rows) for j, (clip, _) in enumerate(clips)}  # fmt: skip
    length = max(len(t) for t in tiles.values()) + round(args.hold * args.fps)
    back = Image.new("RGB", (LEFT + len(clips) * W, TOP + len(rows) * H), BACK)
    draw = ImageDraw.Draw(back)
    for j, (_, label) in enumerate(clips):
        draw.text((LEFT + j * W + W // 2, TOP // 2), label, fill=TEXT, font=font, anchor="mm")
    for i, (_, _, label) in enumerate(rows):
        draw.multiline_text((LEFT // 2, TOP + i * H + H // 2), label, fill=TEXT, font=font, anchor="mm", align="center", spacing=3)
    canvas, frames = np.asarray(back).copy(), []
    for k in range(length):
        for (i, j), t in tiles.items():
            canvas[TOP + i * H : TOP + (i + 1) * H, LEFT + j * W : LEFT + (j + 1) * W] = t[min(k, len(t) - 1)]
        frames.append(Image.fromarray(canvas.copy()))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0, quality=args.quality, method=6)
    print(f"{args.out}: {length} frames at {args.fps} fps, {back.size[0]}x{back.size[1]}, {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
