"""Render a humanoid tracking rollout (ndm_gym.evaluate_humanoid's rollout.npz, saved with --save_rollout) as
tools/render_quadruped.py renders the Go2's (its render_rollout): the simulated robot (``--robot``: the H1, or the G1)
solid and the retargeted reference a blue ghost laid over the frame (``--plain_ghost``: a faint ghost in the scene
instead), in the studio look, a camera that follows the robot's pelvis from the side, with the NDM Gym mark (``--no_mark``:
none). An ``--out`` ending in ``.webp`` is an animation in real time at ``--fps``; any other is a contact sheet: ``--frames``
frames at even steps in a grid, each captioned with its time. LAFAN1 clips last minutes: ``--start`` and ``--seconds``
draw a window of the clip (default: all of it).

    MUJOCO_GL=egl python tools/render_humanoid.py --rollout logs/<name>/evaluation/rollout.npz --clip walk1_subject1 \
        --start 20 --seconds 10 --out logs/<name>/evaluation/walk1_subject1.webp      # or .png

``--source reference_targets``: the rollout without the policy (the reference pose as the position targets) over the same
ghost. A rollout that terminated is drawn up to its end; frames after a termination are marked. The robot's scene is its
MJCF (robots.humanoid, tools/download_h1.sh) on a floor, the collision geoms hidden (geom group 3), the visual meshes shown.
"""

import argparse
import os
import sys
import tempfile

from render_quadruped import add_arguments, render_rollout  # tools/render_quadruped.py; importing it sets up EGL as tools/render.py does

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.robots.humanoid import ROBOTS, HumanoidSpec

PELVIS = "pelvis"  # the body the camera follows: the free root of both robots


def write_scene(spec: HumanoidSpec, path: str):
    """The robot's MJCF on a floor, written to ``path``: its collision geoms (those that collide) moved to geom group 3,
    hidden as the Go2's are, its meshes found from anywhere; its qpos is load_scene's (a free pelvis, then ``spec.joints``)."""
    if not os.path.exists(spec.mjcf):
        raise FileNotFoundError(f"{spec.mjcf}: run {spec.download}")
    s = mujoco.MjSpec.from_file(spec.mjcf)
    s.meshdir = os.path.join(os.path.dirname(spec.mjcf), s.meshdir)
    for g in s.geoms:
        if g.contype or g.conaffinity:
            g.group = 3
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    model = s.compile()
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or names != list(spec.joints) or model.body(PELVIS).id != model.jnt_bodyid[0]:
        raise ValueError(f"{spec.mjcf}: expected a free {PELVIS} and the joints {spec.joints} in that order, got {names}")
    with open(path, "w") as f:
        f.write(s.to_xml())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(parser)
    parser.add_argument("--robot", default="h1", choices=sorted(ROBOTS), help="the rollout's robot")
    parser.add_argument("--start", type=float, default=0.0, help="the window's start in the clip (s)")
    parser.add_argument("--seconds", type=float, default=None, help="the window's length (s; default: to the clip's end)")
    args = parser.parse_args()
    spec = ROBOTS[args.robot]
    with tempfile.TemporaryDirectory() as tmp:
        scene = os.path.join(tmp, "scene.xml")
        write_scene(spec, scene)
        render_rollout(args, scene, 7 + spec.num_joints, PELVIS, distance=3.4, look_height=0.85, start=args.start,
                       seconds=args.seconds, light_follows=True)  # fmt: skip


if __name__ == "__main__":
    main()
