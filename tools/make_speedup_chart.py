"""The README's speed chart: one row per measurement, our time per step as a bar against the baseline's at full width, and
the speedup with its baseline on the right. Drawn at twice the display size for sharp text.

Usage: python tools/make_speedup_chart.py docs/media/speedups.webp
"""

import sys

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
# (label, second line, speedup, baseline): 4,096 parallel worlds in every row (benchmarks/ndm_speed/results.md, benchmarks/g1_boxing/results.md)
# but the Adroit door's, at 16,384 worlds, SuperTrack's model as a bf16 copy against MuJoCo Warp (benchmarks/adroit_door/results_a100.md),
# the racing quadrotor's: the fused step in fp32 with the network as one kernel, on the acceleration and on the increment model, against MuJoCo Warp
# (benchmarks/quadrotor_racing_tracking/results_a100.md),
# RoboKit's, whose bars are its examples' whole planning time at the planner's own worlds, the median over five planner
# seeds, with the simulation step's speedup in the second line (benchmarks/robokit_ndm/results.md), and mbrl-lib's, whose bars
# are the planner's time per executed step on the acceleration and on the increment model (timed in one session), the median
# over three planner seeds, with the step's speedup at 4,096 worlds in the second line and the model in a third, after a
# newline (benchmarks/mbrl_lib/results.md)
ROWS = [
    ("NDM simulation", "HDMI's scenes", 45.3, "vs Isaac Lab + MuJoCo-Warp"),
    ("NDM simulation", "two G1s", 27.0, "vs Isaac Lab + MuJoCo-Warp"),
    ("NDM simulation", "Adroit door", 69.7, "vs MuJoCo Warp"),
    ("NDM simulation", "racing quadrotor\nacceleration model", 22.8, "vs MuJoCo Warp"),
    ("NDM simulation", "racing quadrotor\nincrement model", 22.6, "vs MuJoCo Warp"),
    ("RoboKit planning", "G1 carrying; step 9.8×", 3.0, "vs MuJoCo Warp"),
    ("RoboKit planning", "G1 climbing; step 3.8×", 1.8, "vs MuJoCo Warp"),
    ("RoboKit planning", "Go2 jumping; step 2.3×", 2.1, "vs MuJoCo Warp"),
    ("mbrl-lib planning", "cart-pole; step 17.7×\nacceleration model", 20.8, "vs MuJoCo Warp"),
    ("mbrl-lib planning", "cart-pole; step 17.7×\nincrement model", 20.9, "vs MuJoCo Warp"),
    ("mbrl-lib planning", "pusher; step 8.7×\nacceleration model", 9.9, "vs MuJoCo Warp"),
    ("mbrl-lib planning", "pusher; step 8.8×\nincrement model", 10.5, "vs MuJoCo Warp"),
    ("G1 forward step", "", 6.2, "vs NablaEngine"),
    ("G1 forward step", "", 23.2, "vs MuJoCo Warp"),
    ("G1 forward step", "", 98.0, "vs MJX"),
    ("G1 rollout gradient", "32 steps", 3.42, "vs NablaEngine"),
    ("G1 rollout gradient", "32 steps", 32.0, "vs MuJoCo Warp (PR #1535)"),
    ("G1 rollout gradient", "32 steps", 183.0, "vs MJX (solver as a fixed scan)"),
    ("RigidFormer NDM", "HDMI's scenes", 6.4, "vs Isaac Lab + MuJoCo-Warp"),
]
S = 2  # drawing scale
W, ROW_H, PAD = 860 * S, 86 * S, 18 * S
BLUE0, BLUE1, GREY, INK, MUTED, LINE = (13, 90, 158), (52, 150, 214), (226, 226, 226), (26, 28, 32), (90, 96, 104), (232, 232, 232)


def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, size * S)


def fmt(x):
    return f"{x:.1f}×" if x < 30 else f"{x:.0f}×"


def main(out):
    H = PAD * 2 + ROW_H * len(ROWS)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((S, S, W - 2 * S, H - 2 * S), radius=16 * S, outline=(218, 220, 224), width=S, fill=(255, 255, 255))
    lab, big, small = font("DejaVuSansMono.ttf", 13), font("DejaVuSans-Bold.ttf", 25), font("DejaVuSans.ttf", 12)
    x0, x1 = 236 * S, 626 * S  # the bars
    for i, (l1, l2, speed, base) in enumerate(ROWS):
        y = PAD + i * ROW_H
        if i:
            d.line((PAD, y, W - PAD, y), fill=LINE, width=S)
        lines = [l1, *l2.split("\n")] if l2 else [l1]  # the label, and one or two lines below it
        ty = y + (33, 22, 12)[len(lines) - 1] * S
        for k, text in enumerate(lines):
            d.text((PAD + 14 * S, ty + 21 * k * S), text, font=lab, fill=INK)
        by = y + 33 * S
        d.rounded_rectangle((x0, by + 16 * S, x1, by + 22 * S), radius=3 * S, fill=GREY)  # the baseline: full width
        bw, bh = max(round((x1 - x0) / speed), 12 * S), 8 * S
        bar = Image.new("RGB", (bw, bh))
        bd = ImageDraw.Draw(bar)
        for k in range(bw):  # a left-to-right gradient, pasted through a rounded mask
            t = k / max(bw - 1, 1)
            bd.line((k, 0, k, bh), fill=tuple(round(a + (b - a) * t) for a, b in zip(BLUE0, BLUE1)))
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, bh - 1), radius=bh // 2, fill=255)
        im.paste(bar, (x0, by - S), mask)
        sp = fmt(speed)
        d.text((W - PAD - 14 * S - d.textlength(sp, font=big), y + 12 * S), sp, font=big, fill=BLUE0)
        d.text((W - PAD - 14 * S - d.textlength(base, font=small), y + 50 * S), base, font=small, fill=MUTED)
    im.save(out, lossless=True) if out.endswith(".webp") else im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    main(sys.argv[1])
