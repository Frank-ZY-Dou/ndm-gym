"""tools/download_hand_clips.py against a stand-in for the dataset's site: the files of a task list are written at their
paths, a task's ``object`` names its mesh folder, and a file that differs from its listed checksum is refused."""

import hashlib
import io
import json
import os
import sys
import urllib.parse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import download_hand_clips  # noqa: E402

SUITE = {
    "dataset_repository": "owner/clips",
    "revision": "0123abc",
    "robots": ["xhand"],
    "hand": "right",
    "tasks": [{"id": "stir", "dataset": "source", "task": "stir", "object": "rod", "trial": 0}],
}
LARGE = "processed/source/xhand/right/stir/0/trajectory_kinematic.npz"
FILES = {
    "processed/source/assets/objects/rod/visual.obj": b"v 0 0 0\n",
    "processed/source/xhand/right/stir/0/scene.xml": b"<mujoco/>",
    LARGE: bytes(range(256)),
    "processed/source/assets/robots/xhand/palm.stl": b"solid palm",
}


def listing(path: str) -> dict:
    content = FILES[path]
    entry = {"type": "file", "path": path, "oid": hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()}
    if path == LARGE:
        entry["lfs"] = {"oid": hashlib.sha256(content).hexdigest()}
    return entry


def site(served: dict):
    """urlopen of the dataset's site: the file listings of FILES and the contents of ``served``."""

    def urlopen(url: str):
        path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        if path.startswith("/api/datasets/owner/clips/tree/0123abc/"):
            folder = path.split("/tree/0123abc/")[1]
            if folder == "processed/source/assets":
                entries = [{"type": "directory", "path": "processed/source/assets/robots"}]
            else:
                entries = [listing(p) for p in FILES if p.startswith(folder + "/")]
            return io.BytesIO(json.dumps(entries).encode())
        return io.BytesIO(served[path.split("/resolve/0123abc/")[1]])

    return urlopen


def run(tmp_path, monkeypatch, served: dict) -> str:
    tasks, out = tmp_path / "tasks.json", tmp_path / "out"
    tasks.write_text(json.dumps(SUITE))
    monkeypatch.setattr(download_hand_clips.urllib.request, "urlopen", site(served))
    monkeypatch.setattr(sys, "argv", ["download_hand_clips.py", "--tasks", str(tasks), "--out", str(out)])
    download_hand_clips.main()
    return str(out)


def test_files_are_written_at_their_paths(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, FILES)
    for path, content in FILES.items():
        with open(os.path.join(out, path), "rb") as f:
            assert f.read() == content


@pytest.mark.parametrize("path", [LARGE, "processed/source/xhand/right/stir/0/scene.xml"])
def test_a_file_that_differs_from_its_checksum_is_refused(tmp_path, monkeypatch, path):
    with pytest.raises(ValueError, match="does not match the checksum"):
        run(tmp_path, monkeypatch, {**FILES, path: FILES[path] + b" "})
    assert not os.path.exists(tmp_path / "out" / path)
