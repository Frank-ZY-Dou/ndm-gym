"""Download a hand tracking task list's clips (default benchmarks/xhand_tracking/tasks.json) from the SPIDER dataset on
Hugging Face that the list names, at the revision it pins.

    python tools/download_hand_clips.py [--tasks benchmarks/xhand_tracking/tool_tasks.json] [--out data/spider] [--robots xhand allegro]

Per task and hand: the scene and its kinematic reference, the object's meshes (the folder of the task's name, or the
task's ``object``), and the hand's meshes of that source dataset. Every file is checked against the checksum the pinned
revision lists for it (the SHA-256 sum of a large file, the Git blob hash of a small one); a mismatch raises an error.
The clips keep their terms (the dataset card at the pinned revision, Apache-2.0, and their source datasets' and assets' own;
SPIDER's code, github.com/facebookresearch/spider, is CC BY-NC 4.0) and stay out of Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request

HUB = "https://huggingface.co"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def listed_checksum(entry: dict) -> str:
    """The checksum the dataset's file listing gives an entry: a large file's SHA-256 sum, a small one's Git blob hash."""
    return entry["lfs"]["oid"] if "lfs" in entry else entry["oid"]


def checksum(content: bytes, entry: dict) -> str:
    """``content``'s checksum of the kind ``listed_checksum`` gives the entry."""
    if "lfs" in entry:
        return hashlib.sha256(content).hexdigest()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", type=str, default=os.path.join(REPO, "benchmarks", "xhand_tracking", "tasks.json"))
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "spider"))
    parser.add_argument("--robots", type=str, nargs="+", default=None, help="the hands to download (default: all of the task list's)")
    args = parser.parse_args()
    with open(args.tasks) as f:
        suite = json.load(f)
    repo, rev, hand = suite["dataset_repository"], suite["revision"], suite["hand"]
    robots = args.robots or suite["robots"]
    if not set(robots) <= set(suite["robots"]):
        parser.error(f"--robots must be among {suite['robots']}")

    def tree(path: str, recursive: bool) -> list[dict]:
        url = f"{HUB}/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}?recursive={str(recursive).lower()}&limit=1000"
        with urllib.request.urlopen(url) as r:
            return json.load(r)

    def files(path: str) -> list[dict]:
        return [x for x in tree(path, True) if x["type"] == "file" and not x["path"].endswith((".mp4", ".viser"))]

    def fetch(entry: dict, dest: str) -> None:
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with urllib.request.urlopen(f"{HUB}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(entry['path'])}") as r:
                content = r.read()
            if checksum(content, entry) != listed_checksum(entry):
                raise ValueError(f"{entry['path']}: the download does not match the checksum listed at revision {rev}")
            with open(dest + ".part", "wb") as f:
                f.write(content)
            os.replace(dest + ".part", dest)

    def robots_folder(root: str) -> str:
        """The dataset's folder of hand meshes; a dataset may store it as a link to another dataset's."""
        folder = f"{root}/assets/robots"
        if any(x["path"] == folder and x["type"] == "directory" for x in tree(f"{root}/assets", False)):
            return folder
        with urllib.request.urlopen(f"{HUB}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(folder)}") as r:
            return os.path.normpath(os.path.join(root, "assets", r.read().decode().strip()))

    count = 0
    for t in suite["tasks"]:
        root = f"processed/{t['dataset']}"
        paths = [f"{root}/assets/objects/{t.get('object', t['task'])}"] + [f"{root}/{robot}/{hand}/{t['task']}" for robot in robots]
        for path in paths:
            for entry in files(path):
                fetch(entry, os.path.join(args.out, entry["path"]))
                count += 1
        meshes = robots_folder(root)
        for robot in robots:
            for entry in files(f"{meshes}/{robot}"):
                fetch(entry, os.path.join(args.out, root, "assets", "robots", os.path.relpath(entry["path"], meshes)))
                count += 1
        print(f"{t['id']}: {root}/{{{','.join(robots)}}}/{hand}/{t['task']}/{t['trial']}", flush=True)
    print(f"{count} files in {args.out}")


if __name__ == "__main__":
    main()
