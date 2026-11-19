"""Download the manipulation tracking task's demonstrations and scenes' assets at pinned revisions into ``<out>``: robomimic's
proficient-human demonstrations (huggingface.co/datasets/robomimic/robomimic_datasets, MIT), ``<out>/<task>/demo_v15.hdf5``,
and the meshes and textures of robosuite 1.5.0 (the v1.5.0 tag's commit) that their scenes name
(github.com/ARISE-Initiative/robosuite, MIT), ``<out>/robosuite_assets/``.

    python tools/download_robomimic.py [--out data/robomimic] [--tasks lift square transport]

Lift's file is 37 MB, Square's 69 MB, Transport's 226 MB; the assets 46 MB. Every file is checked against its SHA-256 in
``tools/download_robomimic.sha256.json`` (by its path under ``<out>``), also when it is already there; a file not listed
there, or one that differs, raises. The files keep their upstream terms and stay out of Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DATASET, HF_REVISION = "robomimic/robomimic_datasets", "74fa018461f479cd9fd15b924a16103012096203"
ROBOSUITE, ROBOSUITE_COMMIT = "ARISE-Initiative/robosuite", "1a8701b90c07c6595ace4af9935d7c5ebe1baed3"  # the v1.5.0 tag, as the data were recorded
TASKS = ("lift", "square", "transport")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_robomimic.sha256.json")) as _f:
    SHA256 = json.load(_f)  # each file's path under --out: its SHA-256 at the pinned revisions
TIMEOUT = 60  # s, a connection's and a read's


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, out: str, rel: str, timeout: float = TIMEOUT) -> str:
    """Download ``url`` to ``<out>/<rel>`` through a ``.part`` file, unless it exists; either way check its SHA-256 against
    ``SHA256[rel]``. A mismatching download is deleted. Returns the path."""
    if rel not in SHA256:
        raise ValueError(f"{rel}: no SHA-256 in download_robomimic.sha256.json")
    dest = os.path.join(out, rel)
    if os.path.exists(dest):
        got = sha256(dest)
        if got != SHA256[rel]:
            raise ValueError(f"{dest}: SHA-256 {got}, expected {SHA256[rel]}; delete it to download it again")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    got = sha256(dest + ".part")
    if got != SHA256[rel]:
        os.remove(dest + ".part")
        raise ValueError(f"{url}: SHA-256 {got}, expected {SHA256[rel]}")
    os.replace(dest + ".part", dest)
    return dest


def asset_paths(path: str) -> set[str]:
    """The robosuite asset files (relative to its ``models/assets``) that the scenes of a demonstration file name; a path
    that leaves that directory (absolute, or above it through ``..``) is an error."""
    import h5py

    out = set()
    with h5py.File(path, "r") as f:
        for g in f["data"].values():
            xml = g.attrs["model_file"]
            for p in re.findall(r'file="[^"]*?robosuite/models/assets/([^"]+)"', xml if isinstance(xml, str) else xml.decode()):
                q = os.path.normpath(p)
                if os.path.isabs(q) or q == os.pardir or q.startswith(os.pardir + os.sep):
                    raise ValueError(f"{path}: the asset path {p!r} leaves robosuite's models/assets")
                out.add(q.replace(os.sep, "/"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "robomimic"))
    parser.add_argument("--tasks", type=str, nargs="+", default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()
    assets = set()
    for task in args.tasks:
        dest = fetch(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/v1.5/{task}/ph/demo_v15.hdf5", args.out,
                     f"{task}/demo_v15.hdf5")
        assets |= asset_paths(dest)
        print(f"{task}: {dest}", flush=True)
    for p in sorted(assets):
        fetch(f"https://raw.githubusercontent.com/{ROBOSUITE}/{ROBOSUITE_COMMIT}/robosuite/models/assets/{urllib.parse.quote(p)}",
              args.out, f"robosuite_assets/{p}")
    print(f"{len(assets)} robosuite asset files in {os.path.join(args.out, 'robosuite_assets')}", flush=True)


if __name__ == "__main__":
    main()
