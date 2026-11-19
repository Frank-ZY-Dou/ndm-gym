"""Fetch and verify just the two hand models, their meshes, the piano and upstream notices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

HAND_REV = "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95"
KEYBOARD_REV = "f409faaf65d8a996425cc71a035c30845c7314d8"
MANIFEST = Path(__file__).with_name("assets.sha256")


def download(root, entry):
    digest, name = entry.split(maxsplit=1)
    dest = root / name
    if dest.exists():
        with dest.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == digest:
                return
        raise ValueError(f"Existing asset failed SHA-256 verification: {dest}")
    bundle, path = name.split("/", 1)
    if bundle == "keyboard":
        url = f"https://huggingface.co/datasets/Kashu7100/eden_assets/resolve/{KEYBOARD_REV}/objects/piano/{path}"
    else:
        url = f"https://raw.githubusercontent.com/wuji-technology/wuji-description/{HAND_REV}/{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with urlopen(url, timeout=90) as src, temp.open("wb") as f:
            shutil.copyfileobj(src, f)
        with temp.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch: {url}")
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/wuji_piano"))
    args = parser.parse_args()
    entries = MANIFEST.read_text().splitlines()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda line: download(args.out, line), entries))
    print(f"Verified {len(entries)} assets in {args.out}")


if __name__ == "__main__":
    main()
