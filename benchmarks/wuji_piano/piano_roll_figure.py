"""Plot target and measured keys for one saved physical episode at the control rate."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from note_events import intervals, TICK_MS


from ndm_gym.tasks.piano import KEY_DOWN_FRACTION


def draw_roll(goal, played, path):
    keys = np.flatnonzero((goal | played).any(0))[::-1]
    if not len(keys):
        raise ValueError("no notes in this episode")
    left, right, top, row = 150, 1460, 70, 29
    duration = len(goal) * TICK_MS / 1000
    height = top + len(keys) * row + 90
    image = Image.new("RGB", (1500, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    small = ImageFont.truetype("DejaVuSans.ttf", 17)
    scale = (right - left) / len(goal)
    draw.text((15, 13), "Ode to Joy", font=font, fill="#172635")
    for x, label, color in ((220, "Score", "#356697"), (420, "Correct press", "#21856c"), (680, "Extra press", "#b54c36")):
        draw.rectangle((x, 20, x + 24, 28), fill=color)
        draw.text((x + 33, 12), label, font=small, fill="#172635")
    for sec in range(int(duration) + 1):
        x = left + sec * 1000 / TICK_MS * scale
        draw.line((x, top - 6, x, top + row * len(keys)), fill="#d7dde0")
        draw.text((x, top + row * len(keys) + 8), str(sec), font=small, anchor="ma", fill="#334555")
    draw.text(((left + right) / 2, height - 38), "Episode time (s)", font=small, anchor="ma", fill="#334555")
    for i, k in enumerate(keys):
        midi = k + 21
        pitch = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")[midi % 12] + str(midi // 12 - 1)
        y = top + row * i
        draw.text((left - 12, y + 2), pitch, font=small, anchor="ra", fill="#334555")
        draw.line((left, y + row - 1, right, y + row - 1), fill="#e9edef")
        for mask, offset, color in (
            (goal[:, k], 2, "#356697"),
            (played[:, k] & goal[:, k], 14, "#21856c"),
            (played[:, k] & ~goal[:, k], 14, "#b54c36"),
        ):
            for _, a, b in intervals(mask[:, None]):
                draw.rectangle((left + a * scale, y + offset, left + b * scale - 1, y + offset + 7), fill=color)
    image.save(path, lossless=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    with np.load(args.trace) as trace:
        draw_roll(trace["goal"], trace["depth"] >= KEY_DOWN_FRACTION, args.out)


if __name__ == "__main__":
    main()
