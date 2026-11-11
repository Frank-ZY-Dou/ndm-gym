"""Download the XHand tracking benchmark's clips (benchmarks/xhand_tracking/tasks.json) from SPIDER's
``retarget/retarget_full`` dataset on Hugging Face, at the revision the task list pins.

    python tools/download_hand_clips.py [--out data/spider]

Per task: the scene and its kinematic reference, the object's meshes, and the hand's meshes of that source dataset.
The clips keep their upstream terms (github.com/facebookresearch/spider, CC BY-NC 4.0) and stay out of Git.
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
    args = parser.parse_args()
    with open(args.tasks) as f:
        suite = json.load(f)
    repo, rev, robot, hand = suite["dataset_repository"], suite["revision"], suite["robot"], suite["hand"]

    def files(path: str) -> list[str]:
        url = f"{HUB}/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}?recursive=true&limit=1000"
        with urllib.request.urlopen(url) as r:
            return [x["path"] for x in json.load(r) if x["type"] == "file" and not x["path"].endswith(".mp4")]

    count = 0
    for t in suite["tasks"]:
        root = f"processed/{t['dataset']}"
        for path in (f"{root}/{robot}/{hand}/{t['task']}", f"{root}/assets/objects/{t['task']}", f"{root}/assets/robots/{robot}"):
            for name in files(path):
                dest = os.path.join(args.out, name)
                if not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with urllib.request.urlopen(f"{HUB}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(name)}") as r, open(dest + ".part", "wb") as f:
                        f.write(r.read())
                    os.replace(dest + ".part", dest)
                count += 1
        print(f"{t['id']}: {root}/{robot}/{hand}/{t['task']}/{t['trial']}", flush=True)
    print(f"{count} files in {args.out}")


if __name__ == "__main__":
    main()
