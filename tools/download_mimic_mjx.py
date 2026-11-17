"""Download the animal tracking task's reference clips and bodies at pinned revisions into ``<out>/<animal>/``:
MIMIC-MJX's reference clips (huggingface.co/datasets/talmolab/MIMIC-MJX, MIT) and the bodies' MJCF files with their meshes
(github.com/talmolab/vnl-playground, BSD-3-Clause; its fly is flybody's, github.com/TuragaLab/flybody, Apache-2.0).

    python tools/download_mimic_mjx.py [--out data/mimic_mjx] [--animals rat fly stick]

Per animal: ``reference.h5`` (stac-mjx's clips: ``qpos``, ``qvel``, ``xpos``, ``xquat`` with their names) and ``xmls/`` (the
body, its arena and meshes). The rat's clips are 0.5 GB, the fly's 2.3 GB, the stick insect's 35 MB. The files keep their
upstream terms and stay out of Git.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DATASET, HF_REVISION = "talmolab/MIMIC-MJX", "b4e53a72802f65c917d0d56737e699b1c53b34c8"
VNL_REPO, VNL_COMMIT = "talmolab/vnl-playground", "c1548057763637659307c242772bb02ec363b185"
# per animal: the clips in MIMIC-MJX and the body's directory in vnl-playground
ANIMALS = {
    "rat": ("data/rodent/rodent_reference_clips.h5", "vnl_playground/tasks/rodent/xmls"),
    "fly": ("data/fly/fly_reference_clip.h5", "vnl_playground/tasks/fruitfly/xmls"),
    "stick": ("data/stick/stick_mesh_reference.h5", "vnl_playground/tasks/stick/xmls"),
}


def fetch(url: str, dest: str):
    """Download ``url`` to ``dest`` through a ``.part`` file, unless ``dest`` exists."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(dest + ".part", dest)


def body_files(directory: str) -> list[str]:
    """Paths of the files under ``directory`` of vnl-playground at the pinned commit."""
    url = f"https://api.github.com/repos/{VNL_REPO}/git/trees/{VNL_COMMIT}?recursive=1"
    with urllib.request.urlopen(url) as r:
        tree = json.load(r)["tree"]
    return [x["path"] for x in tree if x["type"] == "blob" and x["path"].startswith(directory + "/")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "mimic_mjx"))
    parser.add_argument("--animals", type=str, nargs="+", default=list(ANIMALS), choices=list(ANIMALS))
    args = parser.parse_args()
    for animal in args.animals:
        clips, directory = ANIMALS[animal]
        for path in body_files(directory):
            fetch(f"https://raw.githubusercontent.com/{VNL_REPO}/{VNL_COMMIT}/{urllib.parse.quote(path)}",
                  os.path.join(args.out, animal, "xmls", os.path.relpath(path, directory)))
        fetch(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{urllib.parse.quote(clips)}",
              os.path.join(args.out, animal, "reference.h5"))
        print(f"{animal}: {os.path.join(args.out, animal)}", flush=True)


if __name__ == "__main__":
    main()
