"""A SPIDER hand-object clip carried out by a Unitree G1 with the clip's hands on its wrists, as a task for
ndm_gym.train_hand (ndm_gym.robots.g1_hands).

    python tools/build_g1_hand_task.py --clip <clip dir> --out <dir>                    # the G1 stands at the table
    python tools/build_g1_hand_task.py --clip <clip dir> --walk <LAFAN1 G1 csv> --stand <frame> --out <dir>   # walks to it first

Standing, the pelvis is placed where the arms track the hands' bases best and the arms are solved for every frame. With
--walk the G1 first walks the LAFAN1 clip up to its frame --stand, the table and the object placed in front of that stand
where the standing placement puts them; while it walks its arms keep the walk's motion, moved only as far as the hands
need to stay clear of the legs and the table; at the stand the arms reach for the clip's first pose (--reach seconds) and
the hands then follow the clip while the legs and the pelvis hold the stand. Before the object leaves the table the arms
are smoothed, where the clip's hands jump toward it faster than the G1's joints turn.

Writes the task in the layout of SPIDER's clips, which train_hand reads as one: <out>/scene.xml, <out>/task_info.json and
<out>/0/trajectory_kinematic.npz; with --walk from the frame where the arms start to reach. In its scene the pelvis, the
legs and the waist are fixed as at the task's first frame, and the G1's own shapes take no contacts; per hand its arm's
seven hinges are actuated as SPIDER's floating palms' hinges (position servos of stiffness 1,000 on joints of armature 1;
a damping ratio of 1, as the fingers'), then its fingers as in the clip; the hands, the object, their contact pairs and the
clip's floor are the clip's. Also writes <out>/demo/scene.xml and <out>/demo/reference.npz, the whole sequence with the table
for tools/render_g1_hands.py, and <out>/summary.json.
"""

import argparse
import json
import os

import mujoco
import numpy as np

from ndm_gym.robots.g1 import MJCF_PATH
from ndm_gym.robots.g1_hands import (ARM, OBJECT, SIDES, STANCE, ArmSolver, build, freeze, hand_mounts, hand_targets, lift_frame, mesh_points,
                                     pelvis_quat, place, placement_frames, placement_guess, quat_yaw, rest_area, soften, yaw_quat)  # fmt: skip

WALK_FPS = 30.0  # LAFAN1's
ARM_PARTS = ("shoulder", "elbow", "wrist")
FOLLOW = 1e-4  # the weight of the walk's arm pose while the arms keep clear of the body, against 1e-5 while they follow the clip
ARM_STIFFNESS, ARM_ARMATURE, ARM_DAMPING_RATIO = 1000.0, 1.0, 1.0  # SPIDER's floating palms' hinges, the fingers' damping ratio
PHYSICS = {"timestep": 0.01, "iterations": 20, "ls_iterations": 50}  # SPIDER's, as motion.sources.spider.load_scene sets them


def resample(walk: np.ndarray, first: float, steps: int, dt: float) -> np.ndarray:
    """``steps`` frames of the walk (rows: position, quaternion (w, x, y, z), joints) every ``dt`` seconds from its frame
    ``first``, linearly interpolated, the quaternion normalized."""
    x = first + np.arange(steps) * dt * WALK_FPS
    low = np.minimum(np.floor(x).astype(int), len(walk) - 2)
    a, b, t = walk[low], walk[low + 1].copy(), (x - low)[:, None]
    b[:, 3:7] *= np.where((a[:, 3:7] * b[:, 3:7]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    out = a + (b - a) * t
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=-1, keepdims=True)
    return out


def standing(args, clip, hand_model, hand_data, sides, mounts, targets, rng):
    """The G1 standing at the table: the scene, the sequence's qpos, the frame where the task starts (0), the clip's floor
    in the scene (0: the clip's own frame) and a summary."""
    guess = placement_guess(targets, sides)
    model = build(args.g1, args.scene, sides, mounts, guess)[0].compile()
    solver = ArmSolver(model, sides, waist=False)
    pelvis = place(solver, model, [targets[frame] for frame in placement_frames(targets)], guess, rng, full_turn=len(sides) == 1)
    model.body_pos[model.body("pelvis").id], model.body_quat[model.body("pelvis").id] = pelvis[:3], yaw_quat(pelvis[3])
    solutions = soften(solver.track(targets, rng), lift_frame(clip[:, args.object_qpos + 2]))
    errors = np.array([solver.errors(q, target) for q, target in zip(solutions, targets, strict=True)])

    # the floor under the feet of the standing pose, the table under the resting object
    data = solver.data
    data.qpos[solver.qpos] = solutions[0]
    mujoco.mj_kinematics(model, data)
    floor = min(float(mesh_points(model, data, geom)[:, 2].min()) for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                if "ankle_roll" in model.body(model.geom_bodyid[geom]).name)  # the soles'  # fmt: skip
    center, half_size = rest_area(hand_model, hand_data, clip, pelvis[3])
    spec = build(args.g1, args.scene, sides, mounts, pelvis, {"floor": floor, "center": center.tolist(), "half_size": half_size.tolist()})[0]
    model = spec.compile()
    qpos = np.tile(model.qpos0, (len(clip), 1))
    for name, value in ((f"{side}_{joint}_joint", value) for side in SIDES for joint, value in STANCE.items()):
        qpos[:, model.joint(name).qposadr[0]] = value
    qpos[:, [model.joint(name).qposadr[0] for name in solver.names]] = solutions
    copy_clip(hand_model, model, clip, qpos)
    summary = {"pelvis_xyz": pelvis[:3].round(4).tolist(), "pelvis_yaw_deg": round(float(np.degrees(pelvis[3])), 1), "table_height_m": round(-floor, 3),
               **tracking_summary(errors, solutions, solver)}  # fmt: skip
    return spec, qpos, 0, 0.0, summary


def walking(args, clip, hand_model, hand_data, sides, mounts, targets, rng):
    """The G1 walking to the table, then reaching: the scene, the sequence's qpos, the frame where the reach starts (the
    task's first), the table's top (the clip's floor in the scene) and a summary."""
    g1_model = mujoco.MjModel.from_xml_path(args.g1)
    joints = [g1_model.joint(j).name for j in range(1, g1_model.njnt)]  # the csv's order: the legs, the waist, the arms
    raw = np.loadtxt(args.walk, delimiter=",")
    walk = np.concatenate((raw[:, :3], raw[:, 6:7], raw[:, 3:6], raw[:, 7:]), axis=1)  # the csv's quaternion is (x, y, z, w)
    steps = int(round(args.seconds / args.dt)) + 1
    walked = resample(walk, args.stand - args.seconds * WALK_FPS, steps, args.dt)
    stand_position, stand_quat, stand_joints = walked[-1, :3], walked[-1, 3:7], walked[-1, 7:]
    stand_yaw = quat_yaw(stand_quat)
    tilt = np.zeros(4)
    mujoco.mju_mulQuat(tilt, np.array(yaw_quat(-stand_yaw)), stand_quat)
    held = {name: float(value) for name, value in zip(joints, stand_joints, strict=True) if not any(part in name for part in ARM_PARTS)}

    # the clip with the pelvis held as at the stand: its tilt, the legs' and the waist's pose
    guess = placement_guess(targets, sides)
    model = build(args.g1, args.scene, sides, mounts, guess, tilt=tilt)[0].compile()
    solver = ArmSolver(model, sides, False, held)
    pelvis = place(solver, model, [targets[frame] for frame in placement_frames(targets)], guess, rng, full_turn=len(sides) == 1, tilt=tilt)
    model.body_pos[model.body("pelvis").id], model.body_quat[model.body("pelvis").id] = pelvis[:3], pelvis_quat(pelvis[3], tilt)
    solutions = soften(solver.track(targets, rng), lift_frame(clip[:, args.object_qpos + 2]))
    errors = np.array([solver.errors(q, target) for q, target in zip(solutions, targets, strict=True)])
    summary = tracking_summary(errors, solutions, solver)

    # the clip's frame in the walk's world: the placement's pelvis on the stand's
    turn = stand_yaw - pelvis[3]
    rotation = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
    shift = stand_position - rotation @ pelvis[:3]
    center, half_size = rest_area(hand_model, hand_data, clip, pelvis[3])
    table = {"floor": 0.0, "top": float(shift[2]), "center": (rotation[:2, :2] @ center + shift[:2]).tolist(), "half_size": half_size.tolist(),
             "yaw": stand_yaw}  # fmt: skip
    spec = build(args.g1, args.scene, sides, mounts, np.array([*stand_position, stand_yaw]), table, floating=True)[0]
    model = spec.compile()
    root = model.joint("floating_base_joint").qposadr[0]
    body = np.array([model.joint(name).qposadr[0] for name in joints])
    arms = np.array([model.joint(name).qposadr[0] for name in solver.names])
    in_walk = np.array([joints.index(name) for name in solver.names])
    poses = np.zeros((len(clip), 7))  # the object's in the walk's world
    poses[:, :3] = clip[:, args.object_qpos : args.object_qpos + 3] @ rotation.T + shift
    for frame in range(len(clip)):
        mujoco.mju_mulQuat(poses[frame, 3:], np.array(yaw_quat(turn)), clip[frame, args.object_qpos + 3 : args.object_qpos + 7])

    def frames(count: int, first: int = 0) -> np.ndarray:  # the stand with the clip's fingers and object from its frame ``first``
        out = np.tile(model.qpos0, (count, 1))
        out[:, root : root + 3], out[:, root + 3 : root + 7], out[:, body] = stand_position, stand_quat, stand_joints
        copy_clip(hand_model, model, clip[first : first + count] if first else np.repeat(clip[:1], count, axis=0), out)
        obj = model.joint(f"{OBJECT}_joint").qposadr[0]
        out[:, obj : obj + 7] = poses[first : first + count] if first else poses[0]
        return out

    # the walk: the arms as walked, moved only as far as the hands' and arms' clearance needs
    walker = ArmSolver(model, sides, False, {})
    walker.free_hands, walker.posture = True, FOLLOW
    walk_qpos = frames(steps)
    walk_qpos[:, root : root + 7], walk_qpos[:, body] = walked[:, :7], walked[:, 7:]
    for frame in range(steps):
        walker.data.qpos[:] = walk_qpos[frame]
        walker.rest = walked[frame, 7:][in_walk]
        walk_qpos[frame, arms] = walker.solve(walker.rest.copy(), None)

    # the reach: from the walk's last arm pose to the clip's first, clear of the table and the object until it arrives
    count = int(round(args.reach / args.dt))
    reach_qpos = frames(count)
    for frame in range(count):
        s = (frame + 1) / count
        blend = s * s * (3 - 2 * s)
        straight = (1 - blend) * walk_qpos[-1, arms] + blend * solutions[0]
        walker.data.qpos[:] = reach_qpos[frame]
        walker.rest = straight
        arrive = np.clip((s - 0.7) / 0.3, 0.0, 1.0)
        reach_qpos[frame, arms] = (1 - arrive) * walker.solve(straight.copy(), None) + arrive * straight

    clip_qpos = frames(len(clip) - 1, first=1)  # from the clip's second frame: the reach ends on its first
    clip_qpos[:, arms] = solutions[1:]
    summary = {"walk": os.path.basename(args.walk), "stand_frame": args.stand, "stand_yaw_deg": round(float(np.degrees(stand_yaw)), 1),
               "walked_m": round(float(np.linalg.norm(np.diff(walked[:, :2], axis=0), axis=1).sum()), 2), "table_height_m": round(float(shift[2]), 3),
               "reach_starts": steps, **summary}  # fmt: skip
    return spec, np.concatenate((walk_qpos, reach_qpos, clip_qpos)), steps, float(shift[2]), summary


def copy_clip(hand_model: mujoco.MjModel, model: mujoco.MjModel, clip: np.ndarray, qpos: np.ndarray) -> None:
    """The clip's joints that ``model`` has (the fingers and the object) into ``qpos``, the scenes' orders mapped."""
    for joint in range(hand_model.njnt):
        name = hand_model.joint(joint).name
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            width = 7 if hand_model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 1
            source, target = hand_model.jnt_qposadr[joint], model.joint(name).qposadr[0]
            qpos[:, target : target + width] = clip[:, source : source + width]


def tracking_summary(errors: np.ndarray, solutions: np.ndarray, solver: ArmSolver) -> dict:
    return {"base_position_error_mm": {"mean": round(float(errors[..., 0].mean() * 1000), 2), "max": round(float(errors[..., 0].max() * 1000), 2)},
            "base_rotation_error_deg": {"mean": round(float(np.degrees(errors[..., 1].mean())), 2), "max": round(float(np.degrees(errors[..., 1].max())), 2)},
            "arm_clearance_min_mm": round(min(solver.clearance(q) for q in solutions) * 1000, 1)}  # fmt: skip


def training_task(args, hand_spec: mujoco.MjSpec, hand_model: mujoco.MjModel, sides, mounts, demo_model: mujoco.MjModel, qpos: np.ndarray,
                  floor: float) -> tuple[mujoco.MjSpec, np.ndarray]:  # fmt: skip
    """train_hand's scene and reference for the sequence's qpos: the G1 fixed as at its first frame but for the arms."""
    if args.walk:
        root = demo_model.joint("floating_base_joint").qposadr[0]
        quat = qpos[0, root + 3 : root + 7]
        yaw = quat_yaw(quat)
        tilt = np.zeros(4)
        mujoco.mju_mulQuat(tilt, np.array(yaw_quat(-yaw)), quat)
        pelvis = np.array([*qpos[0, root : root + 3], yaw])
    else:
        pelvis_id = demo_model.body("pelvis").id
        tilt, pelvis = None, np.array([*demo_model.body_pos[pelvis_id], quat_yaw(demo_model.body_quat[pelvis_id])])
    spec = build(args.g1, args.scene, sides, mounts, pelvis, tilt=tilt)[0]
    arms = [f"{side}_{joint}_joint" for side in sides for joint in ARM]
    composite = spec.compile()
    for joint in range(composite.njnt):  # the legs and the waist, fixed as in the first frame
        name = composite.joint(joint).name
        if name in arms or name == "floating_base_joint" or mujoco.mj_name2id(hand_model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            continue
        freeze(spec, name, composite.body(int(composite.jnt_bodyid[joint])).name, float(qpos[0, demo_model.joint(name).qposadr[0]]))
    hand_geoms = {geom.name for geom in hand_spec.geoms if geom.name}
    for geom in spec.geoms:
        if geom.name not in hand_geoms:
            geom.contype, geom.conaffinity = 0, 0
    for joint in arms:
        spec.joint(joint).armature = ARM_ARMATURE
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), pos=(0, 0, floor), group=3, contype=0, conaffinity=0)
    names = {geom.name for geom in spec.geoms}
    for pair in hand_spec.pairs:
        if pair.geomname1 in names and pair.geomname2 in names:
            copy = spec.add_pair(name=pair.name, geomname1=pair.geomname1, geomname2=pair.geomname2, condim=pair.condim)
            copy.friction, copy.solref, copy.solimp, copy.solreffriction = pair.friction, pair.solref, pair.solimp, pair.solreffriction
            copy.margin, copy.gap = pair.margin, pair.gap

    palm = {f"{side}_{kind}_{axis}" for side in sides for kind in ("pos", "rot") for axis in "xyz"}
    for side in sides:  # its arm's hinges, then its fingers' actuators as the clip's
        for joint in (name for name in arms if name.startswith(side)):
            low, high = spec.joint(joint).range
            actuator = spec.add_actuator(name=f"{joint[: -len('_joint')]}_position", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.gaintype, actuator.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm, actuator.biasprm = [ARM_STIFFNESS] + [0.0] * 9, [0.0, -ARM_STIFFNESS, ARM_DAMPING_RATIO] + [0.0] * 7  # a positive third: the damping ratio
            actuator.ctrlrange, actuator.ctrllimited = (low, high), mujoco.mjtLimited.mjLIMITED_TRUE
        for a in range(hand_model.nu):
            joint = hand_model.joint(int(hand_model.actuator_trnid[a, 0])).name
            if joint.startswith(side) and joint not in palm:
                actuator = spec.add_actuator(name=hand_model.actuator(a).name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                actuator.gaintype = mujoco.mjtGain(int(hand_model.actuator_gaintype[a]))
                actuator.biastype = mujoco.mjtBias(int(hand_model.actuator_biastype[a]))
                actuator.gainprm, actuator.biasprm = hand_model.actuator_gainprm[a].tolist(), hand_model.actuator_biasprm[a].tolist()
                actuator.ctrlrange, actuator.ctrllimited = hand_model.actuator_ctrlrange[a], mujoco.mjtLimited.mjLIMITED_TRUE
    for name, value in PHYSICS.items():
        setattr(spec.option, name, value)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model = spec.compile()
    out = np.tile(model.qpos0, (len(qpos), 1))
    for joint in range(model.njnt):
        width = 7 if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 1
        source = demo_model.joint(model.joint(joint).name).qposadr[0]
        out[:, model.jnt_qposadr[joint] : model.jnt_qposadr[joint] + width] = qpos[:, source : source + width]
    return spec, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", type=str, required=True, help="a SPIDER clip's directory, e.g. data/spider/processed/gigahand/inspire/bimanual/p36-tea/0")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--g1", type=str, default=MJCF_PATH)
    parser.add_argument("--walk", type=str, default=None, help="a LAFAN1 clip retargeted to the G1 (csv: position, quaternion (x, y, z, w), 29 joints)")
    parser.add_argument("--stand", type=int, default=None, help="with --walk: the walk's frame at which the G1 stands at the table")
    parser.add_argument("--seconds", type=float, default=5.0, help="with --walk: how much of the walk before the stand")
    parser.add_argument("--reach", type=float, default=1.0, help="with --walk: seconds for the arms to go from the walk's pose to the clip's first")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.walk is None) != (args.stand is None):
        parser.error("--walk and --stand go together")
    rng = np.random.default_rng(args.seed)
    task_dir = os.path.dirname(os.path.normpath(args.clip))
    args.scene = os.path.join(task_dir, "scene.xml")
    with open(os.path.join(task_dir, "task_info.json")) as f:
        args.dt = float(json.load(f)["ref_dt"])
    clip = np.load(os.path.join(args.clip, "trajectory_kinematic.npz"))["qpos"]
    hand_spec = mujoco.MjSpec.from_file(args.scene)
    hand_model = hand_spec.compile()
    hand_data = mujoco.MjData(hand_model)
    args.object_qpos = hand_model.jnt_qposadr[hand_model.body_jntadr[hand_model.body(OBJECT).id]]
    sides = [side for side in SIDES if mujoco.mj_name2id(hand_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base") >= 0]
    targets = hand_targets(hand_model, hand_data, clip, sides)
    mounts = hand_mounts(hand_model, mujoco.MjModel.from_xml_path(args.g1), sides)

    make = walking if args.walk else standing
    demo_spec, qpos, reach_starts, floor, summary = make(args, clip, hand_model, hand_data, sides, mounts, targets, rng)
    demo_model = demo_spec.compile()
    spec, task_qpos = training_task(args, hand_spec, hand_model, sides, mounts, demo_model, qpos[reach_starts:], floor)

    os.makedirs(os.path.join(args.out, "0"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "demo"), exist_ok=True)
    with open(os.path.join(args.out, "scene.xml"), "w") as f:
        f.write(spec.to_xml())
    with open(os.path.join(args.out, "task_info.json"), "w") as f:
        json.dump({"task": os.path.basename(os.path.normpath(task_dir)), "robot_type": "g1_inspire", "hand_type": "bimanual" if len(sides) == 2 else sides[0],
                   "ref_dt": args.dt, "source_clip": os.path.abspath(args.clip)}, f, indent=2)  # fmt: skip
    np.savez(os.path.join(args.out, "0", "trajectory_kinematic.npz"), qpos=task_qpos)
    with open(os.path.join(args.out, "demo", "scene.xml"), "w") as f:
        f.write(demo_spec.to_xml())
    np.savez(os.path.join(args.out, "demo", "reference.npz"), qpos=qpos, dt=args.dt, reach_starts=reach_starts)
    summary = {"frames": len(qpos), "task_frames": len(task_qpos), "dt": args.dt, **summary}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
