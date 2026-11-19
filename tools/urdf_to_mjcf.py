"""Compile a Unitree humanoid's URDF to an MJCF with a free-floating pelvis (same kinematics and inertias).

    python tools/urdf_to_mjcf.py              # the G1 (tools/download_data.sh)
    python tools/urdf_to_mjcf.py --robot h1   # the H1 (tools/download_h1.sh)
"""

import argparse
import os
import re
import sys

import mujoco

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
# robot -> (its description's directory in data/, the URDF, the MJCF written beside it)
ROBOTS = {
    "g1": ("g1_description", "g1_29dof_rev_1_0.urdf", "g1_29dof.xml"),
    "h1": ("h1_description", "h1.urdf", "h1.xml"),
}

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--robot", choices=sorted(ROBOTS), default="g1")
args = parser.parse_args()
directory, urdf, mjcf = ROBOTS[args.robot]
ROOT = os.path.join(DATA, directory)
src = os.path.join(ROOT, urdf)
tmp = os.path.join(ROOT, f"_{args.robot}_mj.urdf")
out = os.path.join(ROOT, mjcf)

text = open(src).read()
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
# keep every link as its own body so the tracked link frames exist, and keep visuals for rendering
text = re.sub(
    r"<mujoco>.*?</mujoco>",
    '<mujoco><compiler meshdir="." balanceinertia="true" discardvisual="false" fusestatic="false" strippath="false"/></mujoco>',
    text,
    flags=re.S,
)
open(tmp, "w").write(text)
model = mujoco.MjModel.from_xml_path(tmp)
mujoco.mj_saveLastXML(out, model)
os.remove(tmp)

xml = open(out).read()
# free-floating pelvis
xml = re.sub(r'(<body name="pelvis"[^>]*>)', r'\1\n      <freejoint name="floating_base_joint"/>', xml, count=1)
if '<body name="pelvis"' not in xml:
    # URDF root link becomes worldbody children; wrap them in a pelvis body
    sys.exit("pelvis body not found; inspect " + out)
open(out, "w").write(xml)
m2 = mujoco.MjModel.from_xml_path(out)
print("nq", m2.nq, "nv", m2.nv, "nbody", m2.nbody, "ngeom", m2.ngeom, "mass", m2.body_mass.sum())
print([m2.joint(i).name for i in range(m2.njnt)])
