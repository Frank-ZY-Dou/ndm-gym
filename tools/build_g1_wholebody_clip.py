"""A G1 hand task's walk-then-tea demo (tools/build_g1_hand_task.py --walk) as a human-object clip for the whole-body
pipeline (ndm_gym.train_hoi), in ndm_gym.motion.sources.hoi's format.

    python tools/build_g1_wholebody_clip.py --task <build_g1_hand_task.py's out dir> --name <task name> --out <dir>
           [--rollout <ndm_gym.evaluate_hand's rollout.npz of a policy on that task>]

With ``--rollout`` the arms, the fingers and the object from the reach on are the arms-and-hands policy's rollout (the
hands' prior: a reference the hands can follow in simulation), the walk the demo's. Writes <out>/robot.xml (the G1 with
its hands on a free joint, nothing else; jointless bodies fused into their parents, the palms into the wrists, as the
whole-body kinematics take only bodies with a joint), <out>/object.xml (the object on a free joint), <out>/table.xml (the
table, fixed), and <out>/<name>.npz with its <name>.json sidecar: the pelvis's pose, the hinges in the robot's model order,
the object's pose, the table's pose, the keypoints' poses and centre-of-mass velocities, the joint velocities, and per hand
whether it is within CONTACT_DISTANCE of the object. Prints the palms' mean place in the object's frame over the frames
each touches it, the task's ``contact_target_offsets`` (ndm_gym/tasks/hdmi_tasks.py). Then tools/g1_wholebody_usd.py
converts the three models to USD under data/hdmi/g1_inspire/<name>, and the clip goes to data/hdmi/clips.
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_g1_hands import policy_sequence  # noqa: E402

CONTACT_DISTANCE = 0.01  # m, between any geom of a hand and any geom of the object
COLLISION_GROUP = 3  # the hand task's collision geoms, which collide there through explicit pairs only
HDMI_KEYPOINTS = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)  # fmt: skip
HAND_KEYPOINTS = ("thumb_distal", "index_intermediate", "middle_intermediate", "ring_intermediate", "pinky_intermediate")
SIDES = ("left", "right")
KEYPOINTS = HDMI_KEYPOINTS + tuple(f"{s}_{k}" for s in SIDES for k in HAND_KEYPOINTS)
PALMS = tuple(f"{s}_hand_base" for s in SIDES)  # fused into the wrists in robot.xml: the end effectors are the wrists
CLIP_AFTER_REACH = 49  # the clip's first frame, a second after the reach starts (build_g1_hand_task.py's reach)
EEFS = tuple(f"{s}_wrist_yaw_link" for s in SIDES)


def subtree_spec(scene: str, keep: str | None, drop: tuple[str, ...]) -> mujoco.MjSpec:
    """The scene without the bodies ``drop`` and without the world's own geoms (the floor: the simulator has its own),
    its collision geoms colliding with any other; with ``keep``, the world holds only that body, on a free joint, and a
    fixed body (the table) keeps one collision geom, its top: the simulator's props are single shapes."""
    spec = mujoco.MjSpec.from_file(scene)
    for name in drop if keep is None else [b.name for b in spec.worldbody.bodies if b.name != keep]:
        spec.delete(spec.body(name))
    if keep is not None and not spec.body(keep).joints:  # a fixed body becomes a free one: the simulator holds it kinematic
        spec.body(keep).add_freejoint()
        colliding = [g for g in spec.body(keep).geoms if g.contype or g.conaffinity]
        top = max(colliding, key=lambda g: g.pos[2] + g.size[2])  # a prop collides through its top only: one shape per prop
        for g in colliding:
            if g is not top:
                g.contype, g.conaffinity = 0, 0
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for geom in spec.geoms:  # the simulator has no pairs: every collision geom collides with everything
        if geom.group == COLLISION_GROUP:
            geom.contype, geom.conaffinity = 1, 1
    return spec


def descendants(model: mujoco.MjModel, root: int) -> list[int]:
    return [b for b in range(model.nbody) if b == root or _ancestor(model, b, root)]


def _ancestor(model: mujoco.MjModel, b: int, root: int) -> bool:
    while b > 0:
        b = model.body_parentid[b]
        if b == root:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, required=True, help="build_g1_hand_task.py's out dir, with its walk (demo/)")
    parser.add_argument("--rollout", type=str, default="", help="an arms-and-hands policy's rollout.npz on that task")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    scene = os.path.join(args.task, "demo", "scene.xml")
    reference = np.load(os.path.join(args.task, "demo", "reference.npz"))
    qpos, dt, reach = reference["qpos"], float(reference["dt"]), int(reference["reach_starts"])
    model = mujoco.MjModel.from_xml_path(scene)
    if args.rollout:
        qpos = policy_sequence(args.task, model, qpos, reach, args.rollout)
    data = mujoco.MjData(model)
    objects = [model.body(model.jnt_bodyid[j]).name for j in range(1, model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(objects) != 1:
        raise ValueError(f"expected one free object besides the robot, found {objects}")
    obj_name = objects[0]
    os.makedirs(args.out, exist_ok=True)
    for path, keep, drop in (("robot.xml", None, (obj_name, "table")), ("object.xml", obj_name, ()), ("table.xml", "table", ())):
        spec = subtree_spec(scene, keep, drop)
        spec.compiler.fusestatic = keep is None
        with open(os.path.join(args.out, path), "w") as f:
            f.write(spec.to_xml())

    robot = mujoco.MjModel.from_xml_path(os.path.join(args.out, "robot.xml"))
    hinges = [robot.joint(j).name for j in range(robot.njnt) if robot.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_adr = [model.joint(n).qposadr[0] for n in hinges]
    hinge_dof = [model.joint(n).dofadr[0] for n in hinges]
    obj_adr = model.joint(model.body(obj_name).jntadr[0]).qposadr[0]
    keypoints = [model.body(n).id for n in KEYPOINTS]
    obj_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in descendants(model, model.body(obj_name).id)]
    hand_geoms = [[g for g in range(model.ngeom) if model.geom_bodyid[g] in descendants(model, model.body(p).id)] for p in PALMS]
    palm_offsets = [model.body(p).pos.round(4).tolist() for p in PALMS]  # in the wrist's frame, where the palm is welded
    table = model.body("table").id

    T = len(qpos)
    qvel = np.zeros((T, model.nv))
    for t in range(T):  # central differences, one-sided at the ends
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(model, qvel[t], (b - a) * dt, qpos[a], qpos[b])
    pos, quat, lin, ang = (np.zeros((T, len(keypoints), k)) for k in (3, 4, 3, 3))
    contact = np.zeros((T, len(EEFS)), dtype=bool)
    table_pos, table_quat = np.zeros((T, 1, 3)), np.zeros((T, 1, 4))
    fromto, vel = np.zeros(6), np.zeros(6)
    for t in range(T):
        data.qpos[:], data.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_forward(model, data)
        for k, b in enumerate(keypoints):
            pos[t, k], quat[t, k] = data.xpos[b], data.xquat[b]
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)  # the centre of mass's, world frame
            ang[t, k], lin[t, k] = vel[:3], vel[3:]
        for e, geoms in enumerate(hand_geoms):
            contact[t, e] = any(mujoco.mj_geomDistance(model, data, g, o, CONTACT_DISTANCE, fromto) < CONTACT_DISTANCE for g in geoms for o in obj_geoms)
        table_pos[t, 0], table_quat[t, 0] = data.xpos[table], data.xquat[table]

    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    fps = 1.0 / dt
    arrays = {
        "root_pos": f32(qpos[:, 0:3]), "root_quat_wxyz": f32(qpos[:, 3:7]), "joint_pos": f32(qpos[:, hinge_adr]),
        "object_pos": f32(qpos[:, obj_adr : obj_adr + 3]), "object_quat_wxyz": f32(qpos[:, obj_adr + 3 : obj_adr + 7]), "contact": contact,
        "extra_pos": f32(table_pos), "extra_quat_wxyz": f32(table_quat), "keypoint_pos": f32(pos), "keypoint_quat_wxyz": f32(quat),
        "keypoint_lin_vel": f32(lin), "keypoint_ang_vel": f32(ang), "joint_vel": f32(qvel[:, hinge_dof]), "fps": np.asarray(fps, dtype=np.float32),
    }  # fmt: skip
    np.savez(os.path.join(args.out, f"{args.name}.npz"), **arrays)
    side = {
        "source": "tools/build_g1_wholebody_clip.py", "task": args.name, "fps": fps, "frames": T, "reach_starts": reach,
        "clip_starts": reach + CLIP_AFTER_REACH, "robot_usd": f"g1_inspire/{args.name}/robot/robot.usda",
        "object_name": obj_name, "object_body_name": obj_name, "object_joint_name": None,
        "object_usd": f"g1_inspire/{args.name}/object/object.usda", "extra_objects": ["table"],
        "extra_object_usd": {"table": f"g1_inspire/{args.name}/table/table.usda"}, "usd_variants": {"Physics": "mujoco"},
        "contact_eef_bodies": list(EEFS),
        "contact_eef_offsets": palm_offsets, "contact_columns": len(EEFS), "contact_distance_m": CONTACT_DISTANCE, "root_body": "pelvis", "quat_convention": "wxyz",
        "joint_names": hinges, "keypoint_names": list(KEYPOINTS), "paths_note": "paths are relative to data/hdmi",
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
    }  # fmt: skip
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        f.write(json.dumps(side, indent=1) + "\n")
    starts = [int(np.argmax(c)) if c.any() else None for c in contact.T]
    print(f"{T} frames at {fps:g} fps, {len(hinges)} hinges, {len(KEYPOINTS)} keypoints; contact from frames {dict(zip(EEFS, starts))}, "
          f"{contact.mean(0).round(2).tolist()} of the frames")  # fmt: skip
    offsets = []
    for e, (eef, off) in enumerate(zip(EEFS, palm_offsets)):  # the palms' mean place in the object's frame while touching it
        k, R = KEYPOINTS.index(eef), np.zeros(9)
        places = []
        for t in np.nonzero(contact[:, e])[0]:
            mujoco.mju_quat2Mat(R, quat[t, k])
            palm = pos[t, k] + R.reshape(3, 3) @ off
            mujoco.mju_quat2Mat(R, qpos[t, obj_adr + 3 : obj_adr + 7])
            places.append(R.reshape(3, 3).T @ (palm - qpos[t, obj_adr : obj_adr + 3]))
        offsets.append(tuple(round(float(x), 3) for x in np.mean(places, 0)))
    print(f"contact_target_offsets={tuple(offsets)}")


if __name__ == "__main__":
    main()
