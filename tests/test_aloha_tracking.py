"""Bimanual demonstration tracking on ACT's simulated ALOHA tasks, on a small scene in ACT's naming (two arms of six hinges and
two finger slides each, position actuators, a free cube or a peg and a socket): the scene's layout for the task with its
mirrored fingers, the demonstration file and its targets, the action's spread over the actuators, the success tests, ACT's
staged reward, and the generator's waypoints, interpolation, draws and pins; with the demonstrations generated
(tools/make_aloha_demos.py), the real scenes' kinematics against MuJoCo's and the success tests against ACT's reward; on a
GPU, the simulator's control step against MuJoCo's and its contacts' reward against the demonstrations' recorded one."""

import importlib.util
import itertools
import json
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.motion.sources import aloha
from ndm_gym.robots.aloha_scenes import (CONTACTS, CONTROL_DT, FINGER_REACH, GRASP_REACH, HOLD, INSERTION_X, MAX_REWARD, TASKS, TRANSFER_OPENING,
                                         TRANSFER_Z, aloha_scene, load_scene)  # fmt: skip
from ndm_gym.tasks.robomimic_tracking import RobomimicTask, SceneKinematics, build_reference
from ndm_gym.worldmodels.mjcf import MjcfLayout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "aloha")
HAVE_DATA = all(os.path.exists(os.path.join(DATA, t["data"], "demos.npz")) for t in TASKS.values())
JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def scene_xml(task: str, actuated_fingers: bool = True) -> str:
    """Two short arms in ACT's names, the left at x = -0.4 and the right at x = 0.4 turned to face it, and the task's objects."""

    def arm(side: str, x: float, yaw: str) -> str:
        links = ""
        for k, j in enumerate(JOINTS):
            axis = ("0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0")[k]
            links += (f'<body name="{side}/l{k}" pos="0.03 0 0.02"><joint name="{side}/{j}" axis="{axis}" range="-3 3" damping="2" armature="0.02"/>'
                      f'<geom size="0.01" mass="0.2"/>')
        fingers = "".join(
            f'<body name="{side}/{f}_finger_link" pos="{FINGER_REACH} 0 0"><joint name="{side}/{f}_finger" type="slide" axis="0 1 0" range="{lo} {hi}" damping="2"/>'
            f'<geom name="{side}/10_{f}_gripper_finger" size="0.004" mass="0.02"/></body>'
            for f, lo, hi in (("left", 0.021, 0.057), ("right", -0.057, -0.021)))  # fmt: skip
        return (f'<body name="{side}" pos="{x} 0.5 0.1" euler="0 0 {yaw}">{links}<body name="{side}/gripper_link" pos="0.03 0 0">'
                f'<geom size="0.01" mass="0.1"/>{fingers}</body>{"</body>" * len(JOINTS)}</body>')  # fmt: skip

    def box(name: str, joint: str, x: float, geoms: tuple[str, ...]) -> str:
        parts = "".join(f'<geom name="{g}" type="box" size="0.02 0.02 0.02" mass="0.01"/>' for g in geoms)
        return f'<body name="{name}" pos="{x} 0.5 0.05"><joint name="{joint}" type="free"/>{parts}</body>'

    if task == "aloha_transfer_cube":
        objects = box("box", "red_box_joint", 0.1, ("red_box",))
    else:
        objects = box("peg", "red_peg_joint", 0.15, ("red_peg",)) + box("socket", "blue_socket_joint", -0.15, ("socket-1", "socket-2", "socket-3", "socket-4", "pin"))
    actuators = ""
    for side in ("vx300s_left", "vx300s_right"):
        actuators += "".join(f'<position joint="{side}/{j}" kp="50" ctrlrange="-3 3"/>' for j in JOINTS)
        if actuated_fingers:
            actuators += f'<position joint="{side}/left_finger" kp="20" ctrlrange="0.021 0.057"/><position joint="{side}/right_finger" kp="20" ctrlrange="-0.057 -0.021"/>'
    return (f'<mujoco><compiler angle="radian" autolimits="true"/><worldbody><geom name="table" type="plane" size="1 1 0.1"/>'
            f'{arm("vx300s_left", -0.4, "0")}{arm("vx300s_right", 0.4, "3.14159265")}{objects}</worldbody><actuator>{actuators}</actuator></mujoco>')  # fmt: skip


def small_task(name: str):
    model = mujoco.MjModel.from_xml_string(scene_xml(name))
    scene = aloha_scene(model, name)
    d = mujoco.MjData(model)
    d.qpos[[6, 14]], d.qpos[[7, 15]] = 0.04, -0.04  # the fingers within their ranges
    mujoco.mj_forward(model, d)
    qpos, qvel = np.tile(d.qpos, (4, 1)), np.zeros((4, model.nv))
    target = qpos[:, list(scene.act_q)] + 0.005  # (the demonstration's targets are not its joint positions)
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies)
    ref = build_reference(qpos, qvel, scene, kin, "cpu", target=target, hold=HOLD)
    return model, scene, kin, RobomimicTask(scene, kin, ref)


def test_scene_layout():
    model, scene, _, task = small_task("aloha_transfer_cube")
    assert (scene.nu, task.na, scene.control_dt, scene.num_arms) == (16, 14, CONTROL_DT, 2) and scene.mirror == ((6, 7), (14, 15))
    assert scene.arm == ((True,) * 6 + (False,) * 2) * 2 and len(scene.keypoint_bodies) == 6 and len(scene.objects) == 1
    assert task.obs_dim == 16 + 16 + 18 + (3 * 2 + 6 + 3 + 3) + 14 and task.ref.frames == 4 + HOLD
    assert len(aloha_scene(mujoco.MjModel.from_xml_string(scene_xml("aloha_insertion")), "aloha_insertion").objects) == 2
    with pytest.raises(ValueError, match="one of"):
        aloha_scene(model, "lift")
    with pytest.raises(ValueError, match="no free joint"):
        aloha_scene(model, "aloha_insertion")
    with pytest.raises(ValueError, match="no actuators on"):
        aloha_scene(mujoco.MjModel.from_xml_string(scene_xml("aloha_transfer_cube", actuated_fingers=False)), "aloha_transfer_cube")
    with pytest.raises(FileNotFoundError, match="make_aloha_demos"):
        load_scene("/nonexistent", "aloha_insertion")


def test_action_and_targets():
    _, scene, kin, task = small_task("aloha_transfer_cube")
    f = torch.tensor([0, 1])
    zero = task.setpoint(torch.zeros(2, task.na), f)
    assert torch.allclose(zero, task.ref.ctrl[f + 1])  # no action: the reference's joint positions, not the targets it recorded
    assert torch.allclose(task.recorded_setpoint(f), task.ref.target[f + 1]) and not torch.allclose(task.ref.target, task.ref.ctrl)
    a = torch.zeros(2, task.na)
    a[:, 6], a[:, 13] = 0.5, -0.25  # the left gripper's action, the right gripper's
    moved = task.setpoint(a, f) - zero
    assert moved[0].nonzero().flatten().tolist() == [6, 7, 14, 15]
    assert torch.allclose(moved[0, [6, 7, 14, 15]], torch.tensor([0.01, -0.01, -0.005, 0.005]), atol=1e-6)  # a gripper's fingers mirrored
    with pytest.raises(ValueError, match="targets"):
        build_reference(task.ref.qpos[:4].numpy(), task.ref.qvel[:4].numpy(), scene, kin, "cpu", target=np.zeros((3, 16)))
    bare = build_reference(task.ref.qpos[:4].numpy(), task.ref.qvel[:4].numpy(), scene, kin, "cpu")
    with pytest.raises(ValueError, match="recorded no position targets"):
        RobomimicTask(scene, kin, bare).recorded_setpoint(f)


def test_configurations():
    """The benchmark's configurations: the manipulation task's options with the benchmark's budget, checkpoints and episodes."""
    from ndm_gym.train_robomimic import parse_args

    for algo in ("supertrack", "supermimic"):
        a = parse_args(["--config", f"aloha-{algo}", "--task", "aloha_insertion"])
        b = parse_args(["--config", f"robomimic-{algo}", "--task", "lift"])
        assert (a.algo, a.iterations, a.lr_decay_iters, a.ckpt_every, a.ckpt_keep, a.solver_iterations, a.episode) == (algo, 12000, 12000, 200, True, 20, 500)
        assert a.data_dir.endswith(os.path.join("data", "aloha")) and b.data_dir.endswith(os.path.join("data", "robomimic"))
        changed = {k for k in vars(a) if getattr(a, k) != getattr(b, k)}
        assert changed == {"config", "task", "data_dir", "iterations", "lr_decay_iters", "ckpt_every", "episode"}


def act_reward(name: str, touch: dict) -> int:
    """ACT's get_reward (sim_env.py), from whether each named pair of geoms touches."""
    if name == "aloha_transfer_cube":
        reward = 0
        if touch["right"]:
            reward = 1
        if touch["right"] and not touch["table"]:  # lifted
            reward = 2
        if touch["left"]:  # attempted transfer
            reward = 3
        if touch["left"] and not touch["table"]:  # successful transfer
            reward = 4
        return reward
    reward = 0
    if touch["left"] and touch["right"]:  # touch both
        reward = 1
    if touch["left"] and touch["right"] and (not touch["peg_table"]) and (not touch["socket_table"]):  # grasp both
        reward = 2
    if touch["peg_socket"] and (not touch["peg_table"]) and (not touch["socket_table"]):  # peg and socket touching
        reward = 3
    if touch["pin"]:  # successful insertion
        reward = 4
    return reward


@pytest.mark.parametrize("name,groups", [("aloha_transfer_cube", ("left", "right", "table")),
                                         ("aloha_insertion", ("left", "right", "peg_table", "socket_table", "peg_socket", "pin"))])  # fmt: skip
def test_staged_reward(name, groups):
    model, scene, _, task = small_task(name)
    assert len(scene.contacts) == len(CONTACTS[name]) == len(groups)
    names = [[tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair) for pair in group] for group in scene.contacts]
    assert names == [list(group) for group in CONTACTS[name]]
    cases = list(itertools.product((False, True), repeat=len(groups)))  # every combination of the groups touching
    rewards = task.staged_reward(torch.tensor(cases))
    assert rewards.tolist() == [act_reward(name, dict(zip(groups, c, strict=True))) for c in cases] and int(rewards.max()) == MAX_REWARD


def test_success_tests():
    _, scene, kin, task = small_task("aloha_transfer_cube")
    qpos = task.ref.qpos[:1].clone()
    fk = kin(qpos)
    link, between = fk[0, 0], 0.5 * (fk[0, 1] + fk[0, 2])
    grasp = link + (between - link) * (GRASP_REACH / FINGER_REACH)  # the left gripper's grasp point
    (q, _), (lf, rf) = scene.objects[0], task.left_fingers

    def state(cube, opening):
        s = qpos.clone()
        s[0, q : q + 3], s[0, lf], s[0, rf] = cube, opening / 2, -opening / 2
        return s, task.ref.qvel[:1]

    assert float(grasp[2]) > TRANSFER_Z and task.success(state(grasp, 0.085)).item()  # held at the grasp point, off the table
    assert not task.success(state(grasp, TRANSFER_OPENING + 0.01)).item()  # the gripper open
    assert not task.success(state(grasp + torch.tensor([0.05, 0.0, 0.0]), 0.085)).item()  # not in the gripper
    assert not task.success(state(torch.tensor([float(grasp[0]), float(grasp[1]), 0.02]), 0.085)).item()  # on the table
    _, scene, kin, task = small_task("aloha_insertion")
    (peg, _), (socket, _) = scene.objects
    s = task.ref.qpos[:1].clone()
    s[0, socket : socket + 3] = torch.tensor([0.0, 0.5, 0.2])
    for offset, expected in (([0.1, 0.0, 0.0], True), ([-0.1005, 0.005, 0.0], True), ([INSERTION_X + 0.01, 0.0, 0.0], False), ([0.1, 0.02, 0.0], False)):
        s[0, peg : peg + 3] = s[0, socket : socket + 3] + torch.tensor(offset)
        assert task.success((s, task.ref.qvel[:1])).item() is expected


def test_demonstration_file(tmp_path):
    path = tmp_path / "transfer_cube" / "demos.npz"
    path.parent.mkdir()
    rng = np.random.default_rng(0)
    arrays = {"qpos": rng.normal(size=(2, 5, 23)), "qvel": rng.normal(size=(2, 5, 22)), "ctrl": rng.normal(size=(2, 4, 16)),
              "reward": np.array([[0, 1, 2, 4], [0, 0, 4, 4]]), "objects": np.zeros((2, 7))}  # fmt: skip
    np.savez(path, meta=json.dumps({"mujoco": "2.3.7", "benchmark_set": False}), **arrays)
    assert aloha.demo_path(str(tmp_path), "aloha_transfer_cube") == str(path)
    d = aloha.load_demo(str(path), 1)
    assert d["qpos"].shape == (5, 23) and d["target"].shape == (5, 16) and d["meta"]["mujoco"] == "2.3.7"
    assert np.array_equal(d["target"][1:], arrays["ctrl"][1]) and np.array_equal(d["target"][0], arrays["ctrl"][1, 0])  # a frame's: those that led to it
    with pytest.raises(ValueError, match="demonstrations 0 to 1"):
        aloha.load_demo(str(path), 2)
    with pytest.raises(FileNotFoundError, match="make_aloha_demos"):
        aloha.load_demo(str(tmp_path / "none.npz"), 0)
    with pytest.raises(ValueError, match="one of"):
        aloha.demo_path(str(tmp_path), "lift")


def test_generator(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("make_aloha_demos", os.path.join(ROOT, "tools", "make_aloha_demos.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    q = tool.quat_mul(tool.axis_angle([0, 1, 0], -60), tool.axis_angle([0, 1, 0], 60))
    assert np.allclose(q, [1, 0, 0, 0]) and np.allclose(tool.axis_angle([1, 0, 0], 90), [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    objects = tool.draw_objects("transfer_cube", np.random.default_rng(3))
    left, right = tool.waypoints("transfer_cube", tool.MOCAP_START[0], tool.MOCAP_START[1], objects)
    assert [w[0] for w in left] == [0, 100, 260, 310, 360, 400] and [w[0] for w in right] == [0, 90, 130, 170, 200, 220, 310, 360, 400]
    assert np.allclose(right[2][1], objects[:3] + [0, 0, -0.015]) and (right[2][3], right[3][3]) == (1, 0)  # down on the cube, then closed
    xyz, quat, grip = tool.pose_at(right, 110)  # halfway between the waypoints at 90 and 130
    assert np.allclose(xyz, 0.5 * (right[1][1] + right[2][1])) and np.allclose(quat, right[1][2]) and grip == 1
    assert np.allclose(tool.pose_at(right, 399)[0], right[-1][1])
    for task, n in (("transfer_cube", 1), ("insertion", 2)):
        draws = np.array([tool.draw_objects(task, np.random.default_rng(s)) for s in range(50)]).reshape(50, n, 7)
        assert np.allclose(draws[..., 2], 0.05) and np.allclose(draws[..., 3:], [1, 0, 0, 0]) and (draws[..., 1] >= 0.4).all() and (draws[..., 1] <= 0.6).all()
        assert (draws[:, 0, 0] >= (0.0 if n == 1 else 0.1)).all() and (draws[:, 0, 0] <= 0.2).all()
        assert f"{task}/demos.npz:seed0:episodes10" in tool.SHA256  # the benchmark's set is pinned
    assert all(f"act_assets/{name}" in tool.SHA256 for name in tool.ASSETS) and len(tool.ACT_COMMIT) == 40
    if mujoco.__version__ != tool.MUJOCO:  # the generation refuses another MuJoCo than ACT's
        monkeypatch.setattr(sys, "argv", ["make_aloha_demos.py", "--out", "unused"])
        with pytest.raises(SystemExit):
            tool.main()
        assert f"MuJoCo {tool.MUJOCO}" in capsys.readouterr().err


@pytest.mark.skipif(not HAVE_DATA, reason="the ALOHA demonstrations are not generated (tools/make_aloha_demos.py)")
@pytest.mark.parametrize("name", sorted(TASKS))
def test_demonstrations_and_scenes(name):
    from ndm_gym.train_robomimic import build

    model, task, _ = build(name, [0, 1], DATA, "cpu")
    ref, scene = task.ref, task.scene
    assert (task.nu, task.na, task.control_dt) == (16, 14, CONTROL_DT) and ref.frames == 2 * (401 + HOLD)
    d = mujoco.MjData(model)
    for f in (0, 200, 400, 401 + HOLD + 300):  # the keypoints against MuJoCo's body positions
        d.qpos[:] = ref.qpos[f].numpy().astype(np.float64)
        mujoco.mj_kinematics(model, d)
        assert np.abs(d.xpos[list(scene.keypoint_bodies)] - ref.keypoints[f].numpy()).max() < 1e-5
    ok = task.success((ref.qpos, ref.qvel), ref.keypoints).numpy()
    for k in range(2):  # the success test on a demonstration's frames against ACT's reward of the step that led to each
        demo = aloha.load_demo(aloha.demo_path(DATA, name), k)
        act = np.concatenate([[False], demo["reward"] == 4])
        mine = ok[k * (401 + HOLD) :][:401]
        assert not (act & ~mine).any() and np.mean(mine == act) > 0.97 and mine[-1] and demo["meta"]["mujoco"] == "2.3.7"


CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")


@GPU
def test_simulator_step_and_contacts():
    """RobomimicSim on the small scene at ALOHA's control step: MuJoCo's physics steps of it, a gripper's fingers mirrored
    under one action; ``touching`` reads the contacts of the worlds' present states."""
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim

    model, scene, _, _ = small_task("aloha_transfer_cube")
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies).to(CUDA)
    d = mujoco.MjData(model)
    d.qpos[[6, 14]], d.qpos[[7, 15]] = 0.04, -0.04
    qpos, qvel = np.tile(d.qpos, (4, 1)), np.zeros((4, model.nv))
    task = RobomimicTask(scene, kin, build_reference(qpos, qvel, scene, kin, CUDA, hold=HOLD))
    n = 3
    sim = RobomimicSim(model, task, n, CUDA, joint_noise=0.0)
    assert sim.substeps == round(CONTROL_DT / model.opt.timestep)
    frames = torch.zeros(n, dtype=torch.long, device=CUDA)
    sim.reset(torch.ones(n, dtype=torch.bool, device=CUDA), frames)
    (q, _) = scene.objects[0]
    sim.qpos[1, q + 2] = 0.0199  # world 1: the cube on the table; the others' in the air
    assert sim.touching().tolist() == [[False, False, False], [False, False, True], [False, False, False]]
    a = torch.zeros(n, task.na, device=CUDA)
    a[:, 0], a[:, 6] = 0.2, -0.5
    u = task.setpoint(a, frames)
    start = sim.state()[:, : model.nq].cpu().numpy().astype(np.float64)
    s1, _, _ = sim.step(u, reset=False)
    for i in range(n):
        d = mujoco.MjData(model)
        d.qpos[:], d.ctrl[:] = start[i], u[i].cpu().numpy()
        for _ in range(sim.substeps):
            mujoco.mj_step(model, d)
        diff = np.abs(s1[i, : model.nq].cpu().numpy() - d.qpos)
        assert diff[: scene.nu].max() < 1e-4 and diff.max() < 2e-3, diff.max()  # the arms' joints; the free cube (falling, or settling on the table)


@GPU
@pytest.mark.skipif(not HAVE_DATA, reason="the ALOHA demonstrations are not generated (tools/make_aloha_demos.py)")
@pytest.mark.parametrize("name", sorted(TASKS))
def test_contacts_reward_on_demonstrations(name):
    """ACT's reward from MuJoCo-Warp's contacts at a demonstration's frames against the one recorded with it (MuJoCo 2.3.7's
    contacts): the same stage at nearly every frame, and the last stage reached where the demonstration reaches it."""
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim
    from ndm_gym.train_robomimic import build

    model, task, (nconmax, njmax) = build(name, 0, DATA, CUDA)
    frames = torch.arange(1, 401, device=CUDA)  # frame t is the state after step t - 1, whose reward was recorded
    sim = RobomimicSim(model, task, len(frames), CUDA, nconmax=nconmax, njmax=njmax, joint_noise=0.0)
    sim.reset(torch.ones(len(frames), dtype=torch.bool, device=CUDA), frames)
    mine = task.staged_reward(sim.touching()).cpu().numpy()
    recorded = aloha.load_demo(aloha.demo_path(DATA, name), 0)["reward"]
    assert np.mean(mine == recorded) > 0.97, (np.mean(mine == recorded), np.nonzero(mine != recorded)[0])
    assert recorded.max() == MAX_REWARD and np.mean((mine == MAX_REWARD) == (recorded == MAX_REWARD)) > 0.98
