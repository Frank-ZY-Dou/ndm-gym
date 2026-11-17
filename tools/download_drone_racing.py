"""Download the flights of Race Against the Machine (github.com/tii-racing/drone-racing-dataset, release v3.0.0) and convert
each to ``<out>/<flight>.npz``: the drone's motion-capture state, ``t`` (s), ``p`` (m) and ``quat`` (w, x, y, z), and the racing
gates' poses, ``gates`` (n, 12): each gate's centre (m) and rotation matrix (row-major), the gate's opening across its x axis.

    python tools/download_drone_racing.py [--set autonomous|piloted|both] [--out data/drone_racing] [--zip <archive>] [--keep_zip]

The release holds the 18 autonomous flights (``flight-<nn>a-<track>``) as one archive of 5 GB in three parts and the 18 piloted
ones (``flight-<nn>p-<track>``, flown by human pilots) as one of 11 GB in seven parts, camera frames included; only each flight's
``drone_state`` and ``mocap`` CSVs are read from them, and an archive is removed afterwards unless ``--keep_zip``. ``--zip``: an
archive already downloaded (converted alone), or its parts as the release serves them, in order, read as one without joining
them (``Chunks``). The data are CC BY 4.0 (Bosello et al., RA-L 2024) and stay out of Git.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import urllib.request
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = "https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0"
PARTS = {"autonomous": tuple(f"autonomous_zipchunk{i:02d}" for i in range(1, 4)), "piloted": tuple(f"piloted_zipchunk{i:02d}" for i in range(1, 8))}
COLUMNS = ("timestamp", "pose_position_x", "pose_position_y", "pose_position_z", "pose_orientation_x", "pose_orientation_y",
           "pose_orientation_z", "pose_orientation_w")  # fmt: skip


class Chunks(io.RawIOBase):
    """The release's parts of an archive, in order, as one seekable file (what ``cat`` of them would hold), for ``zipfile``."""

    def __init__(self, paths: list[str]):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(paths))]
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (offset, self.pos + offset, sum(self.sizes) + offset)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = 0
        while n < len(b):
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos) if self.pos < sum(self.sizes) else None
            if i is None:
                break
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            got = f.readinto(memoryview(b)[n : n + min(len(b) - n, self.starts[i] + self.sizes[i] - self.pos)])
            if not got:
                break
            n, self.pos = n + got, self.pos + got
        return n

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def convert(csv: bytes) -> dict[str, np.ndarray]:
    """A flight's ``drone_state`` CSV (timestamps in microseconds, the quaternion as x, y, z, w) as the arrays of the .npz."""
    lines = io.StringIO(csv.decode())
    header = lines.readline().strip().split(",")
    cols = [header.index(c) for c in COLUMNS]
    a = np.loadtxt(lines, delimiter=",", usecols=cols, dtype=np.float64)
    a = a[np.concatenate(([True], np.diff(a[:, 0]) > 0))]  # (increasing times)
    return {"t": (a[:, 0] - a[0, 0]) * 1e-6, "p": a[:, 1:4], "quat": a[:, [7, 4, 5, 6]]}


def convert_gates(csv: bytes) -> np.ndarray:
    """A flight's ``mocap`` CSV as the gates' poses (n, 12): per gate the median over the flight of its internal frame's
    position (``gate<i>_int_x``, ...) and rotation matrix (``gate<i>_int_rot[0..8]``); the gates stand still."""
    lines = io.StringIO(csv.decode())
    header = lines.readline().strip().split(",")
    names = sorted({h.split("_")[0] for h in header if h.startswith("gate")}, key=lambda g: int(g[4:]))
    cols = [header.index(f"{g}_int_{c}") for g in names for c in ("x", "y", "z", *(f"rot[{i}]" for i in range(9)))]
    a = np.genfromtxt(lines, delimiter=",", usecols=cols, dtype=np.float64)
    return np.nanmedian(a, 0).reshape(len(names), 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", type=str, default="autonomous", choices=["autonomous", "piloted", "both"], help="the flights to download")
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "drone_racing"))
    parser.add_argument("--zip", type=str, nargs="+", default=None, help="an archive of the release, if already downloaded, or its parts in order (converted alone)")
    parser.add_argument("--keep_zip", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sets = ("autonomous", "piloted") if args.set == "both" else (args.set,)
    total = 0
    for name in sets if args.zip is None else (None,):
        path = args.zip[0] if args.zip is not None and len(args.zip) == 1 else args.zip
        if path is None:
            path = os.path.join(args.out, f"{name}.zip")
            with open(path + ".part", "wb") as f:
                for part in PARTS[name]:
                    print(f"downloading {part}", flush=True)
                    with urllib.request.urlopen(f"{RELEASE}/{part}") as r:
                        shutil.copyfileobj(r, f, 1 << 24)
            os.replace(path + ".part", path)
        total += convert_archive(path, args.out)
        if args.zip is None and not args.keep_zip:
            os.remove(path)
    print(f"{total} flights in {args.out}")


def convert_archive(path, out: str) -> int:
    """Every flight of the archive at ``path`` (or of its parts, a list of paths) converted into ``out``; returns their number."""
    count = 0
    with zipfile.ZipFile(io.BufferedReader(Chunks(path), 1 << 20) if isinstance(path, list) else path) as z:
        names = set(z.namelist())
        for name in sorted(names):
            base = os.path.basename(name)
            if base.startswith("drone_state_") and base.endswith(".csv"):
                flight = base[len("drone_state_") : -len(".csv")]
                d = convert(z.read(name))
                mocap = os.path.join(os.path.dirname(os.path.dirname(name)), f"mocap_{flight}.csv")  # csv_raw/mocap_<flight>.csv
                d["gates"] = convert_gates(z.read(mocap)) if mocap in names else np.zeros((0, 12))
                np.savez(os.path.join(out, flight + ".npz"), **d)
                print(f"{flight}: {len(d['t'])} samples, {d['t'][-1]:.1f} s, {len(d['gates'])} gates", flush=True)
                count += 1
    return count


if __name__ == "__main__":
    main()
