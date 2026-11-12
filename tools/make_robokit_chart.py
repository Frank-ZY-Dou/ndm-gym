"""The RoboKit study's chart: per task, the world model's simulation step against MuJoCo Warp's at the planner's own worlds
(the NdmSim step, the costs' MuJoCo Warp stages included), and the example's median planning time over the planner's seeds on
the world models against as it is, with the outcomes over those seeds. From ``benchmarks/robokit_ndm/summarize.py``'s json:
every task in it, or only those of ``--tasks`` (the entries' ``task``, e.g. ``g1_climb go2_jump``), in the json's order. Drawn
at twice the display size.

Usage: python tools/make_robokit_chart.py results/summary.json docs/media/robokit_speedups.webp [--tasks g1_climb go2_jump]
"""

import argparse
import json

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
NAMES = {"g1_carry": "G1 box carrying", "g1_climb": "G1 climbing", "go2_jump": "Go2 jumping", "wuji_cube": "Wuji cube turning",
         "sharpa_pen": "Sharpa pen spinning"}  # fmt: skip
S = 2
W, ROW_H, PAD = 940 * S, 122 * S, 18 * S
BLUE0, BLUE1, GREY, INK, MUTED, LINE, RED = (13, 90, 158), (52, 150, 214), (226, 226, 226), (26, 28, 32), (90, 96, 104), (232, 232, 232), (176, 48, 48)


def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, size * S)


def fmt(x):
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


def outcome(plans: dict) -> tuple[list, bool]:
    """Two lines, the outcomes over the seeds on the world models and as it is, and whether the world models fall short."""
    ndm, base = plans.get("ndm"), plans.get("base")
    if not ndm or not base:
        return ["", ""], False
    if ndm["passes"] is not None:
        return ([f"world models: {ndm['passes']} of {len(ndm['seeds'])} seeds pass", f"as it is: {base['passes']} of {len(base['seeds'])}"],
                ndm["passes"] < base["passes"])  # fmt: skip
    drops = [sum("dropped" in r["outcome"] and "never dropped" not in r["outcome"] for r in k["runs"].values()) for k in (ndm, base)]
    return ([f"world models: the pen dropped with {drops[0]} of {len(ndm['seeds'])} seeds", f"as it is: with {drops[1]} of {len(base['seeds'])}"],
            drops[0] > drops[1])  # fmt: skip


def rows(summary: dict, only: list | None = None) -> list:
    """(task label, worlds, [(step ms, step speedup)] by step length, plan speedup or None, outcome lines, short of as it is) per task;
    ``only``: these tasks alone (``--tasks``), each of which the summary must hold."""
    tasks = {}
    for s in summary.values():
        tasks.setdefault(s["task"], []).append(s)
    if only is not None:
        missing = [t for t in only if t not in tasks]
        if missing:
            raise ValueError(f"make_robokit_chart: no task {missing} in the summary, whose tasks are {sorted(tasks)}")
        tasks = {t: stages for t, stages in tasks.items() if t in only}
    out = []
    for task, stages in tasks.items():
        stages.sort(key=lambda s: s["dt"])  # the planner's first stage first
        n = min(int(k) for k in stages[0]["speed"])  # the planner's own worlds (the other row is 8,192)
        steps = [(1e3 * s["dt"], s["speed"][str(n)]["speedup"]) for s in stages]
        plans = stages[0].get("plans", {})
        base, ndm = plans.get("base", {}).get("median_s"), plans.get("ndm", {}).get("median_s")
        out.append((NAMES.get(task, task), n, steps, base / ndm if base and ndm else None, *outcome(plans)))
    return out


def bar(im, x0, x1, y, speed, h):
    bw = max(round((x1 - x0) / speed), 12 * S)
    b = Image.new("RGB", (bw, h))
    bd = ImageDraw.Draw(b)
    for k in range(bw):
        t = k / max(bw - 1, 1)
        bd.line((k, 0, k, h), fill=tuple(round(a + (c - a) * t) for a, c in zip(BLUE0, BLUE1)))
    mask = Image.new("L", (bw, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, h - 1), radius=h // 2, fill=255)
    im.paste(b, (x0, y), mask)


def main(src, out, only=None):
    data = rows(json.load(open(src)), only)
    H = PAD * 2 + ROW_H * len(data)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((S, S, W - 2 * S, H - 2 * S), radius=16 * S, outline=(218, 220, 224), width=S, fill=(255, 255, 255))
    lab, big, small = font("DejaVuSansMono.ttf", 13), font("DejaVuSans-Bold.ttf", 21), font("DejaVuSans.ttf", 12)
    x0, x1 = 250 * S, 600 * S
    for i, (name, n, steps, plan, note, short) in enumerate(data):
        y = PAD + i * ROW_H
        if i:
            d.line((PAD, y, W - PAD, y), fill=LINE, width=S)
        d.text((PAD + 14 * S, y + 26 * S), name, font=lab, fill=INK)
        d.text((PAD + 14 * S, y + 50 * S), f"{n:,} worlds", font=small, fill=MUTED)
        step = steps[0][1]
        what = "simulation step" if len(steps) == 1 else f"simulation step at {steps[0][0]:g} ms ({fmt(steps[1][1])} at {steps[1][0]:g} ms)"
        for j, (speed, what) in enumerate(((step, what), (plan, "planning, median over the seeds"))):
            by = y + (22 + 38 * j) * S
            d.rounded_rectangle((x0, by + 12 * S, x1, by + 17 * S), radius=3 * S, fill=GREY)
            if speed is not None:
                bar(im, x0, x1, by, speed, 8 * S)
            label = fmt(speed) if speed is not None else "-"
            d.text((x1 + 24 * S, by - 6 * S), label, font=big, fill=BLUE0)
            d.text((x1 + 24 * S + d.textlength(label, font=big) + 10 * S, by + 2 * S), what, font=small, fill=MUTED)
        for k, line in enumerate(note):
            d.text((x1 + 24 * S, y + (82 + 16 * k) * S), line, font=small, fill=RED if short else INK)
    im.save(out, lossless=True) if out.endswith(".webp") else im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary", help="benchmarks/robokit_ndm/summarize.py's json")
    p.add_argument("out", help="the chart (.webp: lossless)")
    p.add_argument("--tasks", nargs="+", default=None, help="only these tasks, the entries' task (e.g. g1_climb go2_jump; default: every task)")
    a = p.parse_args()
    main(a.summary, a.out, a.tasks)
