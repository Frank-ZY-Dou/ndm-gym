"""gymnasium 0.26's hopper.xml and walker2d.xml in local coordinates, for the mbrl_lib study (benchmarks/mbrl_lib): MuJoCo >=
2.3.4 refuses their global coordinates. Each file is compiled by a MuJoCo that still reads it (2.3.3, mbrl-lib's own
environment) and saved by mj_saveLastXML as <name>_local.xml; that writer rounds the joint ranges to six digits (-150 degrees
become -2.61799), so they are written again with 17, the compiled original's values. The copy is then compiled and checked
against the original: its joint ranges bit for bit, every other numeric field within 1e-12, and ten steps under random
controls within 1e-12. A file whose compiler element does not ask for global coordinates is refused: later gymnasium versions
ship these models in local coordinates, which need no copy and are other models.

The copies are derived from gymnasium's files and are not committed: they go to data/mbrl_models (git-ignored), the directory
the study's tools take as --local_models (ndm_gym.sim.mbrl_envs.patch_envs).

Run in mbrl-lib's own environment (mujoco 2.3.3, gymnasium 0.26.3):
    python tools/convert_mbrl_models.py [--out data/mbrl_models] [--models hopper walker2d] [--assets <directory>]
"""

import argparse
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

MODELS = ("hopper", "walker2d")
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "mbrl_models"
SKIP = ("names", "names_map", "paths", "text_data")  # name tables, not physics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(OUT), help="the directory of the copies (default: data/mbrl_models in this repository)")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--assets", default=None, help="the directory of gymnasium's MuJoCo model files (default: the installed gymnasium's)")
    return p.parse_args(argv)


def assets_dir() -> pathlib.Path:
    import gymnasium.envs.mujoco as envs

    return pathlib.Path(envs.__file__).resolve().parent / "assets"


def in_global_coordinates(src: pathlib.Path) -> bool:
    """Whether the model file's compiler element asks for global coordinates (a comment that names them does not count)."""
    return any(c.get("coordinate") == "global" for c in ET.parse(src).getroot().iter("compiler"))


def numeric_fields(m) -> dict:
    out = {}
    for name in dir(m):
        x = getattr(m, name, None)
        if not name.startswith("_") and name not in SKIP and isinstance(x, np.ndarray) and x.dtype.kind in "biuf":
            out[name] = np.array(x, dtype=np.float64)
    return out


def convert(src: pathlib.Path, dst: pathlib.Path) -> str:
    """One model: ``src`` (global coordinates) to ``dst`` (local coordinates, exact joint ranges), checked; the line to print."""
    import mujoco

    if not in_global_coordinates(src):
        raise SystemExit(f"convert_mbrl_models: {src} is not in global coordinates: the tool converts gymnasium 0.26's files (--assets names "
                         "their directory), and a file in local coordinates needs no copy")  # fmt: skip
    try:
        m0 = mujoco.MjModel.from_xml_path(str(src))
    except ValueError as e:
        raise SystemExit(f"convert_mbrl_models: MuJoCo {mujoco.__version__} does not compile {src} ({str(e).strip()}). Run this in mbrl-lib's own "
                         "environment, whose MuJoCo 2.3.3 still reads global coordinates.") from e  # fmt: skip
    dst.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(dst), m0)
    text = dst.read_text()
    limited = [j for j in range(m0.njnt) if m0.jnt_limited[j]]
    if len(re.findall(r"<joint\b[^>]*\brange=", text)) != len(limited):
        raise SystemExit(f"convert_mbrl_models: {dst.name}'s joints with a range are not the original's {len(limited)} limited joints")
    for j in limited:
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = (float(x) for x in m0.jnt_range[j])
        text, n = re.subn(rf'(<joint name="{re.escape(name)}"[^>]*\brange=")[^"]*(")', rf"\g<1>{lo!r} {hi!r}\g<2>", text)
        if n != 1:
            raise SystemExit(f"convert_mbrl_models: {n} joints named {name} with a range in {dst.name}")
    head = f"<!-- gymnasium's {src.name} in local coordinates, joint ranges exact (tools/convert_mbrl_models.py, MuJoCo {mujoco.__version__}) -->\n"
    dst.write_text(head + text)
    m1 = mujoco.MjModel.from_xml_path(str(dst))
    if not np.array_equal(m0.jnt_range, m1.jnt_range):
        raise SystemExit(f"convert_mbrl_models: {dst.name}'s joint ranges are not the original's")
    a, b = numeric_fields(m0), numeric_fields(m1)
    worst = 0.0
    for name, x in a.items():
        y = b[name]
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise SystemExit(f"convert_mbrl_models: {dst.name}'s {name} is not the original's within 1e-12")
        if x.size:
            worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1.0))))
    rng = np.random.default_rng(0)
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
    apart = 0.0
    for _ in range(10):
        u = rng.uniform(m0.actuator_ctrlrange[:, 0], m0.actuator_ctrlrange[:, 1])
        d0.ctrl[:], d1.ctrl[:] = u, u
        mujoco.mj_step(m0, d0)
        mujoco.mj_step(m1, d1)
        apart = max(apart, float(np.abs(d0.qpos - d1.qpos).max()), float(np.abs(d0.qvel - d1.qvel).max()))
    if apart > 1e-12:
        raise SystemExit(f"convert_mbrl_models: {dst.name} is {apart:.3g} from the original after ten steps under random controls")
    return (f"{src.name} -> {dst}: MuJoCo {mujoco.__version__}, {len(limited)} joint ranges exact, {len(a)} fields within {worst:.1e} "
            f"of the original's, ten steps within {apart:.1e}")  # fmt: skip


def main(argv=None):
    args = parse_args(argv)
    assets = pathlib.Path(args.assets) if args.assets else assets_dir()
    for name in args.models:
        src = assets / f"{name}.xml"
        if not src.is_file():
            raise SystemExit(f"convert_mbrl_models: no {src} (gymnasium 0.26's MuJoCo model files; --assets names another directory)")
        print(convert(src, pathlib.Path(args.out) / f"{name}_local.xml"))


if __name__ == "__main__":
    main()
