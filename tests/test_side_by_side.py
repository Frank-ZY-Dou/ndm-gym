"""tools/side_by_side.py on two small animated WebPs (read with Pillow, so no ffmpeg needed): the two halves side by side at
the requested rate and width, each with its label at the top left; inputs of different heights or durations refused."""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "side_by_side.py")


def _webp(path: str, rgb: tuple[int, int, int], n: int = 6, size: tuple[int, int] = (96, 64), ms: int = 40):
    """n frames of ms each, shades of rgb (each frame differs, so that none is merged with the previous)."""
    frames = [Image.new("RGB", size, tuple(int(c * (0.6 + 0.4 * k / n)) for c in rgb)) for k in range(n)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, lossless=True)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def test_side_by_side_webps(tmp_path):
    a, b, out = str(tmp_path / "a.webp"), str(tmp_path / "b.webp"), str(tmp_path / "ab.webp")
    _webp(a, (200, 40, 40))
    _webp(b, (40, 40, 200))
    r = _run(a, b, out, "--labels", "SuperTrack", "SuperMimic", "--fps", "25", "--width", "192")
    assert r.returncode == 0, r.stderr
    with Image.open(out) as im:
        assert im.size == (192, 64) and im.n_frames == 6  # 0.24 s at 25 fps
        im.seek(3)
        x = np.asarray(im.convert("RGB")).astype(float)
    left, right = x[40:, :96].mean((0, 1)), x[40:, 96:].mean((0, 1))
    assert left[0] > 2 * left[2] and right[2] > 2 * right[0]  # the left input on the left, the right on the right
    for x0 in (0, 96):  # each half's label: a white pill at its top left
        assert x[2:12, x0 + 2 : x0 + 20].mean() > 1.5 * x[40:, x0 : x0 + 96].mean()
    _webp(str(tmp_path / "short.webp"), (40, 200, 40), n=3)
    _webp(str(tmp_path / "low.webp"), (40, 200, 40), size=(96, 48))
    for other, why in (("short.webp", "equally long"), ("low.webp", "heights differ")):
        r = _run(a, str(tmp_path / other), out, "--fps", "25")
        assert r.returncode != 0 and why in r.stderr
