"""tools/make_grid.py on small animated WebPs (read with Pillow, so no ffmpeg needed): tiles in rows of --columns with their names
in the margins, one loop as long as the longest clip in which a shorter one holds its last frame; wrong counts and aspects refused."""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "make_grid.py")


def _webp(path: str, rgb: tuple[int, int, int], n: int, size: tuple[int, int] = (64, 36), ms: int = 50):
    """n frames of ms each, shades of rgb (each frame differs, so that none is merged with the previous)."""
    frames = [Image.new("RGB", size, tuple(int(c * (0.5 + 0.5 * k / n)) for c in rgb)) for k in range(n)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, lossless=True)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def test_grid_layout_and_hold(tmp_path):
    clips = {"red": ((220, 40, 40), 8), "green": ((40, 220, 40), 4), "blue": ((40, 40, 220), 8)}
    for name, (rgb, n) in clips.items():
        _webp(str(tmp_path / f"{name}.webp"), rgb, n)
    out = str(tmp_path / "grid.webp")
    tiles = [a for name in clips for a in ("--clip", str(tmp_path / f"{name}.webp"))]
    r = _run(out, "--columns", "2", "--tile", "64", "36", "--names", "red", "green", "blue", "--quality", "90", *tiles)
    assert r.returncode == 0, r.stderr
    assert "last frame held for 0.20 s" in r.stdout  # green: 4 of the loop's 8 frames
    with Image.open(out) as im:
        assert im.size == (2 * 64 + 4, 2 * (24 + 36) + 4) and im.n_frames == 8
        x = []
        for k in (1, 7):
            im.seek(k)
            x.append(np.asarray(im.convert("RGB")).astype(float))
    for f in x:  # red | green in the first row under their band, blue centred in the second
        assert f[30:54, 8:56, 0].mean() > 2 * f[30:54, 8:56, 1].mean()
        assert f[30:54, 76:124, 1].mean() > 2 * f[30:54, 76:124, 0].mean()
        assert f[94:118, 42:90, 2].mean() > 2 * f[94:118, 42:90, 0].mean()
        for margin in (f[94:118, :20], f[94:118, -20:]):  # left and right of the centred tile: the dark margins
            assert margin.mean() < 40 and margin.std(axis=(0, 1)).max() < 2
    assert f[4:20, :64].std(axis=(0, 1)).max() > 10  # a name in the band
    assert abs(x[1][30:54, 76:124, 1].mean() - x[0][30:54, 76:124, 1].mean()) > 20  # green moved from frame 1 to its last
    _webp(str(tmp_path / "wide.webp"), (90, 90, 90), 4, size=(96, 36))
    for args, why in ((("--col_labels", "a"), "--col_labels"), (("--row_labels", "a", "b", "c"), "--row_labels")):
        r = _run(out, "--columns", "2", "--tile", "64", "36", *args, *tiles)
        assert r.returncode != 0 and why in r.stderr
    r = _run(out, "--columns", "1", "--tile", "64", "36", "--clip", str(tmp_path / "wide.webp"))
    assert r.returncode != 0 and "aspect" in r.stderr


def test_blocks_with_spans(tmp_path):
    clips = {
        "red": ((220, 40, 40), (132, 100)),
        "green": ((40, 220, 40), (64, 100)),
        "blue": ((40, 40, 220), (64, 48)),
        "grey": ((150, 150, 150), (64, 48)),
    }
    for name, (rgb, size) in clips.items():
        _webp(str(tmp_path / f"{name}.webp"), rgb, 4, size=size)
    c = {name: str(tmp_path / f"{name}.webp") for name in clips}
    out = str(tmp_path / "blocks.webp")
    tiles = ["--block", "A", "2", "2", "--clip", c["red"], "0", "0", "0", "132", "100", "2", "2",
             "--block", "B", "2", "2", "--clip", c["green"], "0", "0", "0", "64", "100", "2", "1",
             "--clip", c["blue"], "--clip", c["grey"]]  # fmt: skip
    r = _run(out, "--columns", "2", "--tile", "64", "48", "--quality", "90", *tiles)
    assert r.returncode == 0, r.stderr
    with Image.open(out) as im:
        assert im.size == (2 * 132 + 16, 24 + 100)  # two blocks of 2 x 2 cells, 16 px apart, under their band
        im.seek(2)
        f = np.asarray(im.convert("RGB")).astype(float)
    for (x0, x1, y0, y1), ch in (((10, 120, 35, 115), 0), ((152, 208, 35, 115), 1), ((220, 276, 30, 66), 2)):
        assert f[y0:y1, x0:x1, ch].mean() > 1.8 * np.delete(f[y0:y1, x0:x1], ch, axis=2).mean()
    assert abs(f[82:118, 220:276].std(axis=2).mean()) < 6 and f[82:118, 220:276].mean() > 100  # grey under blue, beside the tall green
    r = _run(out, "--columns", "2", "--tile", "64", "48", *tiles[:-2])
    assert r.returncode != 0 and "cells empty" in r.stderr
    r = _run(out, "--columns", "1", "--tile", "64", "48", "--clip", c["green"], "0", "0", "0", "64", "100", "2", "1")
    assert r.returncode != 0 and "in a block" in r.stderr
