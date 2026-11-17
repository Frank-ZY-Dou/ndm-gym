"""Download the manipulation tracking task's demonstrations and scenes' assets at pinned revisions into ``<out>``: robomimic's
proficient-human demonstrations (huggingface.co/datasets/robomimic/robomimic_datasets, MIT), ``<out>/<task>/demo_v15.hdf5``,
and the meshes and textures of robosuite 1.5.0 that their scenes name (github.com/ARISE-Initiative/robosuite, MIT),
``<out>/robosuite_assets/``.

    python tools/download_robomimic.py [--out data/robomimic] [--tasks lift square transport]

Lift's file is 37 MB, Square's 69 MB, Transport's 226 MB; the assets 46 MB. The files keep their upstream terms and stay out
of Git.
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DATASET, HF_REVISION = "robomimic/robomimic_datasets", "74fa018461f479cd9fd15b924a16103012096203"
ROBOSUITE, ROBOSUITE_COMMIT = "ARISE-Initiative/robosuite", "dc7fcf9fa6cdf0796f79b4c873166a8fb9fe8c9e"  # v1.5.0, as the data were recorded
TASKS = ("lift", "square", "transport")


def fetch(url: str, dest: str):
    """Download ``url`` to ``dest`` through a ``.part`` file, unless ``dest`` exists."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(dest + ".part", dest)


def asset_paths(path: str) -> set[str]:
    """The robosuite asset files (relative to its ``models/assets``) that the scenes of a demonstration file name."""
    import h5py

    out = set()
    with h5py.File(path, "r") as f:
        for g in f["data"].values():
            xml = g.attrs["model_file"]
            for p in re.findall(r'file="[^"]*?robosuite/models/assets/([^"]+)"', xml if isinstance(xml, str) else xml.decode()):
                out.add(os.path.normpath(p).replace(os.sep, "/"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "robomimic"))
    parser.add_argument("--tasks", type=str, nargs="+", default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()
    assets = set()
    for task in args.tasks:
        dest = os.path.join(args.out, task, "demo_v15.hdf5")
        fetch(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/v1.5/{task}/ph/demo_v15.hdf5", dest)
        assets |= asset_paths(dest)
        print(f"{task}: {dest}", flush=True)
    for p in sorted(assets):
        fetch(f"https://raw.githubusercontent.com/{ROBOSUITE}/{ROBOSUITE_COMMIT}/robosuite/models/assets/{urllib.parse.quote(p)}",
              os.path.join(args.out, "robosuite_assets", p))
    print(f"{len(assets)} robosuite asset files in {os.path.join(args.out, 'robosuite_assets')}", flush=True)


if __name__ == "__main__":
    main()
