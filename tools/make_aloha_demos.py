"""Generate the bimanual tracking task's demonstrations (benchmarks/aloha_bimanual_tracking) with ACT's scripted policies in
ACT's simulated ALOHA scenes (github.com/tonyzhaozh/act, MIT; Zhao et al., RSS 2023), as its ``record_sim_episodes.py``
records an episode, and keep the simulator's whole state, which ACT's recordings do not hold (they have the arms' joints,
the actions and a camera image, not the objects' poses):

1. ACT's scenes and meshes are downloaded at a pinned commit into ``<out>/act_assets`` and checked against their SHA-256
   in ``tools/make_aloha_demos.sha256.json``, also when they are already there.
2. In the end-effector scene (each gripper welded to a mocap body) the policy's waypoints are followed for 400 control
   steps of 20 ms; the arms' joint positions and the grippers' commands of every step are kept.
3. These are replayed as the position actuators' targets in the joint-controlled scene from the same object poses. The
   state after every control step (qpos, qvel), the targets and ACT's staged reward (0 to 4) are written.

An episode is kept if its replay reaches ACT's highest reward, 4 (Transfer Cube: the cube in the left gripper and off the
table; Insertion: the peg touching the pin in the socket). ``<out>/<task>/demos.npz`` holds the kept episodes, in the order
drawn: ``qpos`` (E, 401, nq), ``qvel`` (E, 401, nv), ``ctrl`` (E, 400, 16), ``reward`` (E, 400), ``objects`` (E, 7 per
object: the poses drawn) and ``meta`` (JSON). The objects start where ACT draws them, from ``--seed``'s generator.

    python tools/make_aloha_demos.py [--out data/aloha] [--tasks transfer_cube insertion] [--episodes 10] [--seed 0]

It runs with MuJoCo 2.3.7, the version ACT's instructions install, in an environment of its own (``pip install
mujoco==2.3.7 numpy``; the library's ``hand`` extra has another version), and refuses another: how the grippers follow
their mocap bodies in the end-effector scene depends on the version, and with MuJoCo 3.15 the scripted grasps miss. The
benchmark's set is the default one (10 kept episodes per task, seed 0): its arrays' SHA-256 is pinned beside the assets',
and a set that differs raises. Another seed or number of episodes needs ``--unpinned`` and is not the benchmark's. The
waypoints, the gripper's range, the start pose and the reward tests are ACT's (``scripted_policy.py``, ``constants.py``,
``ee_sim_env.py``, ``sim_env.py``); a control step is stepped in dm_control's order (``mj_step2``, the remaining physics
steps, ``mj_step1``), as ACT's environments step it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT, ACT_COMMIT = "tonyzhaozh/act", "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
MUJOCO = "2.3.7"  # the version ACT's instructions install: the end-effector scene's welds behave as ACT's policies expect
CONTROL_DT, EPISODE = 0.02, 400  # ACT's DT and its scripted episodes' length
START_ARM_POSE = (0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239)
GRIPPER_OPEN, GRIPPER_CLOSE = 0.05800, 0.01844  # a finger's position at command 1 and 0 (PUPPET_GRIPPER_POSITION_*)
MOCAP_START = (([-0.31718881, 0.5, 0.29525084], [1.0, 0.0, 0.0, 0.0]), ([0.31718881, 0.49999888, 0.29525084], [1.0, 0.0, 0.0, 0.0]))  # left, right
TASKS = {
    "transfer_cube": {"scene": "bimanual_viperx_transfer_cube.xml", "ee_scene": "bimanual_viperx_ee_transfer_cube.xml", "objects": 1},
    "insertion": {"scene": "bimanual_viperx_insertion.xml", "ee_scene": "bimanual_viperx_ee_insertion.xml", "objects": 2},
}
ASSETS = ("bimanual_viperx_ee_insertion.xml", "bimanual_viperx_ee_transfer_cube.xml", "bimanual_viperx_insertion.xml",
          "bimanual_viperx_transfer_cube.xml", "scene.xml", "tabletop.stl", "vx300s_10_custom_finger_left.stl",
          "vx300s_10_custom_finger_right.stl", "vx300s_10_gripper_finger.stl", "vx300s_11_ar_tag.stl", "vx300s_1_base.stl",
          "vx300s_2_shoulder.stl", "vx300s_3_upper_arm.stl", "vx300s_4_upper_forearm.stl", "vx300s_5_lower_forearm.stl",
          "vx300s_6_wrist.stl", "vx300s_7_gripper.stl", "vx300s_8_gripper_prop.stl", "vx300s_9_gripper_bar.stl",
          "vx300s_dependencies.xml", "vx300s_left.xml", "vx300s_right.xml")  # fmt: skip
SHA256_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_aloha_demos.sha256.json")
with open(SHA256_FILE) as _f:
    SHA256 = json.load(_f)  # each asset's path under --out, and the benchmark's set of each task: its SHA-256


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, out: str, rel: str) -> str:
    """Download ``url`` to ``<out>/<rel>`` through a ``.part`` file, unless it exists; either way check its SHA-256 against
    ``SHA256[rel]``. A mismatching download is deleted. Returns the path."""
    if rel not in SHA256:
        raise ValueError(f"{rel}: no SHA-256 in make_aloha_demos.sha256.json")
    dest = os.path.join(out, rel)
    if os.path.exists(dest):
        got = sha256(dest)
        if got != SHA256[rel]:
            raise ValueError(f"{dest}: SHA-256 {got}, expected {SHA256[rel]}; delete it to download it again")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    got = sha256(dest + ".part")
    if got != SHA256[rel]:
        os.remove(dest + ".part")
        raise ValueError(f"{url}: SHA-256 {got}, expected {SHA256[rel]}")
    os.replace(dest + ".part", dest)
    return dest


def quat_mul(a, b) -> np.ndarray:
    """The product of two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])  # fmt: skip


def axis_angle(axis, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2
    return np.concatenate([[np.cos(half)], np.sin(half) * np.asarray(axis, dtype=np.float64)])


def waypoints(task: str, mocap_left, mocap_right, objects: np.ndarray):
    """ACT's waypoints of both grippers (``PickAndTransferPolicy``, ``InsertionPolicy``): lists of (step, position, quaternion,
    gripper command: 1 open, 0 closed) for the left and the right gripper, from the mocap bodies' poses at the start
    (position, quaternion) and the objects' poses drawn."""
    (l_xyz, l_quat), (r_xyz, r_quat) = [(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)) for p, q in (mocap_left, mocap_right)]
    d = lambda *v: np.array(v, dtype=np.float64)  # noqa: E731
    if task == "transfer_cube":
        box = objects[:3]
        pick = quat_mul(r_quat, axis_angle([0, 1, 0], -60))
        meet_left = axis_angle([1, 0, 0], 90)
        rest = d(1, 0, 0, 0)
        meet = d(0, 0.5, 0.25)
        left = [(0, l_xyz, l_quat, 0), (100, meet + d(-0.1, 0, -0.02), meet_left, 1), (260, meet + d(0.02, 0, -0.02), meet_left, 1),
                (310, meet + d(0.02, 0, -0.02), meet_left, 0), (360, meet + d(-0.1, 0, -0.02), rest, 0), (400, meet + d(-0.1, 0, -0.02), rest, 0)]  # fmt: skip
        right = [(0, r_xyz, r_quat, 0), (90, box + d(0, 0, 0.08), pick, 1), (130, box + d(0, 0, -0.015), pick, 1), (170, box + d(0, 0, -0.015), pick, 0),
                 (200, meet + d(0.05, 0, 0), pick, 0), (220, meet, pick, 0), (310, meet, pick, 1), (360, meet + d(0.1, 0, 0), pick, 1),
                 (400, meet + d(0.1, 0, 0), pick, 1)]  # fmt: skip
    elif task == "insertion":
        peg, socket = objects[:3], objects[7:10]
        pick_right = quat_mul(r_quat, axis_angle([0, 1, 0], -60))
        pick_left = quat_mul(r_quat, axis_angle([0, 1, 0], 60))  # (ACT builds the left one from the right mocap's quaternion too)
        meet, lift = d(0, 0.5, 0.15), 0.00715
        left = [(0, l_xyz, l_quat, 0), (120, socket + d(0, 0, 0.08), pick_left, 1), (170, socket + d(0, 0, -0.03), pick_left, 1),
                (220, socket + d(0, 0, -0.03), pick_left, 0), (285, meet + d(-0.1, 0, 0), pick_left, 0), (340, meet + d(-0.05, 0, 0), pick_left, 0),
                (400, meet + d(-0.05, 0, 0), pick_left, 0)]  # fmt: skip
        right = [(0, r_xyz, r_quat, 0), (120, peg + d(0, 0, 0.08), pick_right, 1), (170, peg + d(0, 0, -0.03), pick_right, 1),
                 (220, peg + d(0, 0, -0.03), pick_right, 0), (285, meet + d(0.1, 0, lift), pick_right, 0), (340, meet + d(0.05, 0, lift), pick_right, 0),
                 (400, meet + d(0.05, 0, lift), pick_right, 0)]  # fmt: skip
    else:
        raise ValueError(f"unknown task {task!r}")
    return left, right


def pose_at(trajectory, t: int):
    """ACT's interpolation between the waypoints around step ``t``: linear in the position, the quaternion's components
    (not normalized, as ACT sets it; MuJoCo normalizes a mocap body's) and the gripper command."""
    i = max(k for k, w in enumerate(trajectory) if w[0] <= t)
    (t0, xyz0, quat0, grip0), (t1, xyz1, quat1, grip1) = trajectory[i], trajectory[i + 1]
    frac = (t - t0) / (t1 - t0)
    return xyz0 + (xyz1 - xyz0) * frac, quat0 + (quat1 - quat0) * frac, grip0 + (grip1 - grip0) * frac


def draw_objects(task: str, rng: np.random.Generator) -> np.ndarray:
    """The objects' poses at the start (position and unit quaternion each), in ACT's ranges (``sample_box_pose``,
    ``sample_insertion_pose``): the cube; the peg, then the socket."""
    ranges = {"transfer_cube": [((0.0, 0.2), (0.4, 0.6))], "insertion": [((0.1, 0.2), (0.4, 0.6)), ((-0.2, -0.1), (0.4, 0.6))]}[task]
    poses = [[rng.uniform(*x), rng.uniform(*y), 0.05, 1.0, 0.0, 0.0, 0.0] for x, y in ranges]
    return np.array(poses, dtype=np.float64).reshape(-1)


def control_step(mujoco, m, d, substeps: int):
    """One control step in dm_control's order, which ACT's environments use: positions and velocities are up to date after it."""
    mujoco.mj_step2(m, d)
    if substeps > 1:
        mujoco.mj_step(m, d, substeps - 1)
    mujoco.mj_step1(m, d)


def reward(mujoco, task: str, m, d) -> int:
    """ACT's staged reward (0 to 4) from the geoms in contact (``TransferCubeTask.get_reward``, ``InsertionTask.get_reward``);
    a pair counts in either order."""
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    pairs = {frozenset((name(c.geom1), name(c.geom2))) for c in d.contact[: d.ncon]}
    touch = lambda a, b: frozenset((a, b)) in pairs  # noqa: E731
    left, right = "vx300s_left/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger"
    if task == "transfer_cube":
        on_table = touch("red_box", "table")
        r = 1 if touch("red_box", right) else 0
        r = 2 if touch("red_box", right) and not on_table else r
        r = 3 if touch("red_box", left) else r
        return 4 if touch("red_box", left) and not on_table else r
    socket = ("socket-1", "socket-2", "socket-3", "socket-4")
    grasp_left, grasp_right = any(touch(s, left) for s in socket), touch("red_peg", right)
    peg_on_table, socket_on_table = touch("red_peg", "table"), any(touch(s, "table") for s in socket)
    r = 1 if grasp_left and grasp_right else 0
    r = 2 if grasp_left and grasp_right and not peg_on_table and not socket_on_table else r
    r = 3 if any(touch("red_peg", s) for s in socket) and not peg_on_table and not socket_on_table else r
    return 4 if touch("red_peg", "pin") else r


def joint_state(d) -> np.ndarray:
    """ACT's 14 joint values of the arms (``get_qpos``): six joints and the left finger's position per arm, unnormalized."""
    return np.concatenate([d.qpos[:7], d.qpos[8:15]])


def generate(task: str, assets: str, objects: np.ndarray) -> dict:
    """One episode: the waypoints followed in the end-effector scene, its joint positions and gripper commands replayed in the
    joint-controlled scene. Returns the replay's ``qpos`` (401, nq), ``qvel`` (401, nv), ``ctrl`` (400, 16), ``reward`` (400)."""
    import mujoco

    grip = lambda command: command * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE  # noqa: E731  (the finger's target)
    # 1. the end-effector scene: the grippers follow the waypoints
    m = mujoco.MjModel.from_xml_path(os.path.join(assets, TASKS[task]["ee_scene"]))
    d = mujoco.MjData(m)
    substeps = round(CONTROL_DT / m.opt.timestep)
    d.qpos[:16] = START_ARM_POSE
    d.qpos[16 : 16 + objects.size] = objects
    for k, (xyz, quat) in enumerate(MOCAP_START):
        d.mocap_pos[k], d.mocap_quat[k] = xyz, quat
    d.ctrl[:] = [GRIPPER_CLOSE, -GRIPPER_CLOSE, GRIPPER_CLOSE, -GRIPPER_CLOSE]
    mujoco.mj_forward(m, d)
    left, right = waypoints(task, MOCAP_START[0], MOCAP_START[1], objects)
    joints, commands = [joint_state(d)], [d.ctrl.copy()]
    for t in range(EPISODE):
        for k, trajectory in enumerate((left, right)):
            xyz, quat, command = pose_at(trajectory, t)
            d.mocap_pos[k], d.mocap_quat[k] = xyz, quat
            d.ctrl[2 * k], d.ctrl[2 * k + 1] = grip(command), -grip(command)
        control_step(mujoco, m, d, substeps)
        joints.append(joint_state(d))
        commands.append(d.ctrl.copy())
    # 2. the joint-controlled scene: the joint positions and the gripper commands as the actuators' targets
    m = mujoco.MjModel.from_xml_path(os.path.join(assets, TASKS[task]["scene"]))
    d = mujoco.MjData(m)
    d.qpos[:16] = START_ARM_POSE
    d.ctrl[:] = START_ARM_POSE
    d.qpos[16 : 16 + objects.size] = objects
    mujoco.mj_forward(m, d)
    out = {"qpos": [d.qpos.copy()], "qvel": [d.qvel.copy()], "ctrl": [], "reward": []}
    for t in range(EPISODE):
        (q, c) = joints[t], commands[t]
        d.ctrl[:] = np.concatenate([q[:6], [c[0], -c[0]], q[7:13], [c[2], -c[2]]])
        control_step(mujoco, m, d, substeps)
        out["qpos"].append(d.qpos.copy()), out["qvel"].append(d.qvel.copy()), out["ctrl"].append(d.ctrl.copy())
        out["reward"].append(reward(mujoco, task, m, d))
    return {k: np.array(v) for k, v in out.items()}


def digest(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in ("qpos", "qvel", "ctrl", "reward", "objects"):
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "data", "aloha"))
    parser.add_argument("--tasks", type=str, nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--episodes", type=int, default=10, help="the kept episodes per task")
    parser.add_argument("--seed", type=int, default=0, help="the generator of the objects' poses")
    parser.add_argument("--unpinned", action="store_true", help="a set other than the benchmark's (another --seed or --episodes): no SHA-256 to check")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes >= 1")
    import mujoco

    if mujoco.__version__ != MUJOCO:
        parser.error(f"MuJoCo {mujoco.__version__}: the demonstrations are generated with MuJoCo {MUJOCO} (pip install mujoco=={MUJOCO} "
                     "in an environment of its own); with another version the scripted grasps miss")  # fmt: skip
    assets = os.path.join(args.out, "act_assets")
    for name in ASSETS:
        fetch(f"https://raw.githubusercontent.com/{ACT}/{ACT_COMMIT}/assets/{name}", args.out, f"act_assets/{name}")
    for task in args.tasks:
        key = f"{task}/demos.npz:seed{args.seed}:episodes{args.episodes}"
        if key not in SHA256 and not args.unpinned:
            raise ValueError(f"{key}: not the benchmark's set (no SHA-256 in make_aloha_demos.sha256.json); --unpinned generates it")
        rng, kept, drawn = np.random.default_rng([args.seed, list(TASKS).index(task)]), [], 0
        while len(kept) < args.episodes:
            objects = draw_objects(task, rng)
            episode = generate(task, assets, objects)
            drawn += 1
            if episode["reward"].max() == 4:
                kept.append({**episode, "objects": objects})
            if drawn >= 20 * args.episodes:
                raise ValueError(f"{task}: {len(kept)} of {drawn} episodes reach ACT's reward of 4")
        arrays = {k: np.stack([e[k] for e in kept]) for k in ("qpos", "qvel", "ctrl", "reward", "objects")}
        got = digest(arrays)
        if not args.unpinned and got != SHA256[key]:
            raise ValueError(f"{key}: the generated set's SHA-256 is {got}, the benchmark's {SHA256[key]}")
        meta = {"task": task, "scene": TASKS[task]["scene"], "act_commit": ACT_COMMIT, "mujoco": mujoco.__version__, "seed": args.seed,
                "control_dt": CONTROL_DT, "drawn": drawn, "sha256": got, "benchmark_set": not args.unpinned}  # fmt: skip
        dest = os.path.join(args.out, task, "demos.npz")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        np.savez(dest, meta=json.dumps(meta), **arrays)
        last = [int(e["reward"][-1]) for e in kept]
        print(f"{task}: {len(kept)} kept of {drawn} drawn -> {dest} (SHA-256 {got}); reward at the last step {last}", flush=True)


if __name__ == "__main__":
    main()
