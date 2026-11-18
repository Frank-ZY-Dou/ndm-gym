"""Download the hand tracking benchmark's clips (benchmarks/xhand_tracking/tasks.json) from SPIDER's
``retarget/retarget_full`` dataset on Hugging Face, at the revision the task list pins.

    python tools/download_hand_clips.py [--out data/spider] [--robots xhand allegro]

Per task and hand: the scene and its kinematic reference, the object's meshes, and the hand's meshes of that source dataset.
The clips keep their terms (the dataset card at the pinned revision, Apache-2.0, and their source datasets' and assets' own;
SPIDER's code, github.com/facebookresearch/spider, is CC BY-NC 4.0) and stay out of Git.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

HUB = "https://huggingface.co"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    def files(path: str) -> list[str]:
        url = f"{HUB}/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}?recursive=true&limit=1000"
        with urllib.request.urlopen(url) as r:
            return [x["path"] for x in json.load(r) if x["type"] == "file" and not x["path"].endswith(".mp4")]

    count = 0
    for t in suite["tasks"]:
        root = f"processed/{t['dataset']}"
        paths = [f"{root}/assets/objects/{t['task']}"]
        for robot in robots:
            paths += [f"{root}/{robot}/{hand}/{t['task']}", f"{root}/assets/robots/{robot}"]
        for path in paths:
            for name in files(path):
                dest = os.path.join(args.out, name)
                if not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with (
                        urllib.request.urlopen(f"{HUB}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(name)}") as r,
                        open(dest + ".part", "wb") as f,
                    ):
                        f.write(r.read())
                    os.replace(dest + ".part", dest)
                count += 1
        print(f"{t['id']}: {root}/{{{','.join(robots)}}}/{hand}/{t['task']}/{t['trial']}", flush=True)
    print(f"{count} files in {args.out}")


if __name__ == "__main__":
    main()
