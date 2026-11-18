"""tools/convert_mbrl_models.py: gymnasium 0.26's hopper.xml converted to local coordinates with exact joint ranges and checked
against the original, where the installed gymnasium ships that file and the installed MuJoCo still compiles global coordinates
(2.3.3, mbrl-lib's own environment); under a later MuJoCo, the tool's refusal of a file in global coordinates, which names that
environment; and its refusal of a file in local coordinates, whatever a comment in it says (a later gymnasium's hopper.xml).
The two refusals are read on files written here, so that they do not depend on the installed gymnasium. The tool needs mujoco,
gymnasium and numpy only, and so does this file."""

import importlib.util
import math
import pathlib

import pytest

mujoco = pytest.importorskip("mujoco")
gymnasium = pytest.importorskip("gymnasium")

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "convert_mbrl_models.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("convert_mbrl_models", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# a hinge above a floor: in global coordinates, and in local ones with a comment that names the global ones (as a later gymnasium's)
GLOBAL = """
<mujoco>
  <compiler coordinate="global"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body><joint type="hinge" pos="0 0 1" axis="0 1 0"/><geom type="capsule" fromto="0 0 1 0 0 0.5" size="0.05"/></body>
  </worldbody>
</mujoco>
"""
LOCAL = """
<!-- modified to not require `coordinate="global"` -->
<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def source(tool):
    src = tool.assets_dir() / "hopper.xml"
    if not src.is_file() or not tool.in_global_coordinates(src):  # gymnasium 1.x: local coordinates, the global ones named in a comment
        pytest.skip(f"no hopper.xml in global coordinates in {src.parent} (gymnasium {gymnasium.__version__}): the tool converts 0.26's file")
    return src


def _reads_global() -> bool:
    try:
        mujoco.MjModel.from_xml_string('<mujoco><compiler coordinate="global"/><worldbody/></mujoco>')
    except ValueError:
        return False
    return True


def test_hopper_converts_with_exact_ranges(tool, source, tmp_path):
    if not _reads_global():
        pytest.skip(f"MuJoCo {mujoco.__version__} refuses global coordinates: the conversion runs in mbrl-lib's own environment (MuJoCo 2.3.3)")
    dst = tmp_path / "models" / "hopper_local.xml"
    line = tool.convert(source, dst)
    assert "3 joint ranges exact" in line and "ten steps within" in line
    text = dst.read_text()
    assert 'coordinate="global"' not in text and 'range="-0.7853981633974483 0.7853981633974483"' in text
    assert 'range="-2.6179938779914944 0.0"' in text
    m0, m1 = mujoco.MjModel.from_xml_path(str(source)), mujoco.MjModel.from_xml_path(str(dst))
    assert (m0.jnt_range == m1.jnt_range).all() and m1.jnt_range[5].tolist() == [-math.radians(45), math.radians(45)]
    assert abs(m0.body_pos - m1.body_pos).max() < 1e-12 and abs(m0.body_mass - m1.body_mass).max() < 1e-12 and m1.nq == 6 and m1.nu == 3
    tool.main(["--out", str(tmp_path / "all"), "--models", "hopper"])  # the command line, into another directory
    assert (tmp_path / "all" / "hopper_local.xml").read_text() == text


def test_refusal_names_mbrl_libs_own_environment(tool, tmp_path):
    if _reads_global():
        pytest.skip(f"MuJoCo {mujoco.__version__} compiles global coordinates: there is nothing to refuse")
    (tmp_path / "hopper.xml").write_text(GLOBAL)
    assert tool.in_global_coordinates(tmp_path / "hopper.xml")
    with pytest.raises(SystemExit, match="mbrl-lib's own\\s+environment"):
        tool.convert(tmp_path / "hopper.xml", tmp_path / "hopper_local.xml")
    assert not (tmp_path / "hopper_local.xml").exists()


def test_a_file_in_local_coordinates_is_refused_whatever_its_comments_say(tool, tmp_path):
    (tmp_path / "hopper.xml").write_text(LOCAL)
    assert 'coordinate="global"' in LOCAL and not tool.in_global_coordinates(tmp_path / "hopper.xml")
    mujoco.MjModel.from_xml_path(str(tmp_path / "hopper.xml"))  # the installed MuJoCo compiles it: the refusal is the tool's own
    with pytest.raises(SystemExit, match="is not in global coordinates"):
        tool.convert(tmp_path / "hopper.xml", tmp_path / "out" / "hopper_local.xml")
    with pytest.raises(SystemExit, match="is not in global coordinates"):
        tool.main(["--out", str(tmp_path / "out"), "--models", "hopper", "--assets", str(tmp_path)])
    assert not (tmp_path / "out").exists()


def test_a_missing_source_is_named(tool, tmp_path):
    with pytest.raises(SystemExit, match="no .*hopper.xml"):
        tool.main(["--out", str(tmp_path), "--models", "hopper", "--assets", str(tmp_path)])
