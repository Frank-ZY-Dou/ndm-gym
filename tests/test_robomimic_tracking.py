"""Manipulation demonstration tracking on small scenes in robosuite's naming (seven-joint arms with their torque actuators,
two-finger grippers with their grip sites, the task's free objects), their demonstrations made by MuJoCo itself: the servos
and the gravity compensation, the scene's layout for the task, the demonstration's file and assets and the downloader's
checks, the forward kinematics against MuJoCo's, the task at the reference, robosuite's success tests against their
transcription from robosuite, the world model's control step, the updates and training's arguments; on a GPU, the simulator
against MuJoCo and its success test against robosuite's."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.robomimic_algos import ROBOMIMIC_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.motion.sources.robomimic import load_demo, split_states
from ndm_gym.robots.robomimic_scenes import ARM_KP, RobomimicScene, build_spec, resolve_assets
from ndm_gym.tasks.robomimic_tracking import (CONTROL_DT, HOLD, Contacts, RewardCfg, RobomimicTask, SceneKinematics, build_reference,
                                              unflatten_state)  # fmt: skip
from ndm_gym.train import add_train_args
from ndm_gym.train_hand import UNSUPPORTED
from ndm_gym.train_robomimic import HAND_ONLY, ROBOMIMIC_DEFAULTS, parse_args
from ndm_gym.worldmodels.mjcf import MjcfLayout

# the arguments robomimic training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSTEPS = round(CONTROL_DT / 0.002)


def arm_xml(r: int, base: str, yaw: float = 0.0) -> str:
    """Robot ``r``: seven hinges (with the Panda's armature) on a base at ``base``, a gripper with its grip site at the end
    effector and two finger slides."""
    links, axes = "", ("0 0 1", "0 1 0", "0 0 1", "0 -1 0", "0 0 1", "0 -1 0", "0 0 1")
    for k, axis in enumerate(axes, start=1):
        links += (f'<body name="robot{r}_link{k}" pos="0 0 {0.0 if k == 1 else 0.12}">'
                  f'<joint name="robot{r}_joint{k}" axis="{axis}" range="-2.8 2.8" armature="1" damping="0.1"/>'
                  f'<geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.03" mass="1"/>')  # fmt: skip
    g = f"gripper{r}_right"
    gripper = (f'<body name="robot{r}_right_hand" pos="0 0 0.12"><body name="{g}_right_gripper">'
               f'<body name="{g}_eef" pos="0 0 0.097"><site name="{g}_grip_site" size="0.01"/></body>'
               f'<body name="{g}_leftfinger" pos="0 0 0.05"><joint name="{g}_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>'
               f'<geom type="box" pos="0 0.015 0" size="0.01 0.005 0.02" mass="0.05"/></body>'
               f'<body name="{g}_rightfinger" pos="0 0 0.05"><joint name="{g}_finger_joint2" type="slide" axis="0 1 0" range="-0.04 0"/>'
               f'<geom type="box" pos="0 -0.015 0" size="0.01 0.005 0.02" mass="0.05"/></body></body></body>')  # fmt: skip
    return f'<body name="robot{r}_base" pos="{base}" euler="0 0 {yaw}">{links}{gripper}{"</body>" * 7}</body>'


def actuators_xml(r: int) -> str:
    torques = "".join(f'<general name="robot{r}_torq_j{k}" joint="robot{r}_joint{k}" ctrlrange="{-t} {t}"/>' for k, t in
                      zip(range(1, 8), (80, 80, 80, 80, 80, 12, 12), strict=True))  # fmt: skip
    g = f"gripper{r}_right"
    return (f'{torques}<position name="{g}_gripper_finger_joint1" joint="{g}_finger_joint1" kp="1000" ctrlrange="0 0.04"/>'
            f'<position name="{g}_gripper_finger_joint2" joint="{g}_finger_joint2" kp="1000" ctrlrange="-0.04 0"/>')  # fmt: skip


def bin_xml(name: str, pos: str) -> str:
    """A bin of robosuite's TransportGroup in its names: a free body with its base and a wall (the base's top 1 cm above the
    root's bottom, on a table's top when the root is 0.075 m above it)."""
    return (f'<body name="{name}_root" pos="{pos}"><freejoint name="{name}_joint0"/>'
            f'<geom name="{name}_base" type="box" pos="0 0 -0.07" size="0.145 0.145 0.005" mass="2"/>'
            f'<geom name="{name}_wall0" type="box" pos="0.145 0 0" size="0.005 0.15 0.075" mass="0.2"/></body>')  # fmt: skip


# each task's objects in robosuite's names (Square's nut a slab beside its peg: on_peg reads the body's position, not a hole)
OBJECTS = {
    "lift": '<body name="cube_main" pos="0 0 0.82"><freejoint name="cube_joint0"/><geom name="cube_g0" type="box" size="0.02 0.02 0.02" mass="0.07"/></body>',
    "square": ('<body name="peg1" pos="0.23 0.1 0.85"><geom type="box" size="0.016 0.016 0.1"/></body>'
               '<body name="peg2" pos="0.23 -0.1 0.85"><geom type="cylinder" size="0.02 0.1"/></body>'
               '<body name="SquareNut_main" pos="0 0.2 0.8095"><freejoint name="SquareNut_joint0"/>'
               '<geom name="SquareNut_g0" type="box" size="0.005 0.02 0.01" mass="0.05"/></body>'
               '<body name="RoundNut_main" pos="0 -0.2 0.8095"><freejoint name="RoundNut_joint0"/>'
               '<geom name="RoundNut_g0" type="box" size="0.005 0.02 0.01" mass="0.05"/></body>'),
    "transport": (bin_xml("transport_target_bin", "0.2 0.25 0.875") + bin_xml("transport_trash_bin", "0.2 -0.25 0.875")
                  + '<body name="payload_root" pos="-0.2 0.25 0.8095"><freejoint name="payload_joint0"/>'
                  '<geom name="payload_handle" type="box" size="0.08 0.01 0.01" mass="0.1"/>'
                  '<geom name="payload_head" type="box" pos="0.08 0 0" size="0.01 0.03 0.01" mass="0.1"/>'
                  '<geom name="payload_handle_vis" type="box" size="0.08 0.01 0.01" group="1" contype="0" conaffinity="0" mass="0"/></body>'
                  '<body name="trash_main" pos="-0.2 -0.25 0.8195"><freejoint name="trash_joint0"/>'
                  '<geom name="trash_g0" type="box" size="0.02 0.02 0.02" mass="0.05"/></body>'
                  '<body name="transport_start_bin_lid_root" pos="-0.2 0 0.8045"><freejoint name="transport_start_bin_lid_joint0"/>'
                  '<geom name="transport_start_bin_lid_top" type="box" size="0.1 0.1 0.005" mass="0.2"/></body>'),
}  # fmt: skip


def scene_xml(texture: bool = False, task: str = "lift", arms: int = 1) -> str:
    """A Panda-like scene in robosuite's names: ``arms`` arms (seven hinges driven by torque actuators, a gripper's two finger
    slides on position servos), the task's free objects on a table; ``texture``: one in robosuite's assets' path, as recorded
    scenes name them."""
    bases = (("-0.5 0 0.8", 0.0), ("0.5 0 0.8", 3.14159))[:arms]
    asset = '<asset><texture name="t" type="2d" file="/home/someone/robosuite/models/assets/textures/wood.png"/></asset>' if texture else ""
    return f"""<mujoco><compiler angle="radian"/><option timestep="0.002"/>{asset}
      <worldbody><geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="table" pos="0 0 0.775"><geom type="box" size="0.4 0.4 0.025"/></body>
        {"".join(arm_xml(r, b, y) for r, (b, y) in enumerate(bases))}{OBJECTS[task]}
      </worldbody>
      <actuator>{"".join(actuators_xml(r) for r in range(arms))}</actuator></mujoco>"""


def scene_spec(task: str = "lift", arms: int = 1, **kw):
    return build_spec(scene_xml(task=task, arms=arms), "/unused", **kw)


def demonstration(model, frames: int, seed: int):
    """A demonstration made by the scene itself: the servos' targets wander smoothly about the start; qpos and qvel at every
    control step, MuJoCo's state as robosuite records it."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    target = np.array([d.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]] for a in range(model.nu)])
    qpos, qvel = [d.qpos.copy()], [d.qvel.copy()]
    for _ in range(frames - 1):
        target[:7] += 0.03 * rng.standard_normal(7)
        d.ctrl[:] = target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
        qpos.append(d.qpos.copy()), qvel.append(d.qvel.copy())
    return np.stack(qpos), np.stack(qvel)


@pytest.fixture(scope="module")
def setup():
    model = scene_spec().compile()
    scene = RobomimicScene.from_model(model, "lift")
    qpos, qvel = demonstration(model, 30, 0)
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies)
    return model, scene, qpos, qvel, kin, RobomimicTask(scene, kin, build_reference(qpos, qvel, scene, kin, "cpu"))


def test_servos_and_the_scene():
    """The arm's torque actuators become position servos within their torque ranges, the robot is compensated for gravity,
    the gripper keeps its servos; the scene's actuators, keypoints, objects and what the success tests read as the task
    reads them; an actuated joint without a range is an error."""
    raw = mujoco.MjModel.from_xml_string(scene_xml())
    model = scene_spec().compile()
    for k in range(7):
        assert model.actuator_gainprm[k, 0] == ARM_KP[k] and model.actuator_biasprm[k, 1] == -ARM_KP[k] and model.actuator_biasprm[k, 2] < 0
        assert model.actuator_forcelimited[k] and np.allclose(model.actuator_forcerange[k], raw.actuator_ctrlrange[k])
        assert not model.actuator_ctrllimited[k]
    assert np.allclose(model.actuator_gainprm[7:9, 0], raw.actuator_gainprm[7:9, 0])
    b = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    assert model.body_gravcomp[b("robot0_link3")] == 1 and model.body_gravcomp[b("gripper0_right_leftfinger")] == 1
    assert model.body_gravcomp[b("cube_main")] == 0 and scene_spec(gravcomp=False).compile().body_gravcomp.max() == 0
    scene = RobomimicScene.from_model(model, "lift")
    assert scene.arm == (True,) * 7 + (False,) * 2 and scene.num_arms == 1 and scene.act_q == tuple(range(9))
    assert scene.keypoint_bodies == (b("gripper0_right_eef"), b("gripper0_right_leftfinger"), b("gripper0_right_rightfinger"))
    assert scene.objects == ((9, 9),) and scene.others == () and scene.success_bodies == (b("cube_main"),)
    assert scene.grip_sites == (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site"),) and scene.ngeom == model.ngeom
    with pytest.raises(ValueError, match="SquareNut_joint0"):
        RobomimicScene.from_model(model, "square")
    with pytest.raises(ValueError, match="torque actuators"):
        build_spec(scene_xml().replace("robot0_torq_j7", "robot0_motor_j7"), "/unused")
    with pytest.raises(ValueError, match="no range"):  # (the targets' clip would be made up)
        RobomimicScene.from_model(build_spec(scene_xml().replace('axis="0 1 0" range="0 0.04"', 'axis="0 1 0"'), "/unused").compile(), "lift")
    with pytest.raises(ValueError, match="grip_site"):
        RobomimicScene.from_model(build_spec(scene_xml().replace("_grip_site", "_site"), "/unused").compile(), "lift")
    assert resolve_assets('<texture file="/x/y/robosuite/models/assets/textures/a.png"/>', "/data/assets") == '<texture file="/data/assets/textures/a.png"/>'
    two = scene_spec("square", arms=2).compile()
    sq = RobomimicScene.from_model(two, "square")
    name = lambda kind, i: mujoco.mj_id2name(two, kind, i)  # noqa: E731
    assert [name(mujoco.mjtObj.mjOBJ_BODY, i) for i in sq.success_bodies + sq.pegs] == ["SquareNut_main", "RoundNut_main", "peg1", "peg2"]
    assert [name(mujoco.mjtObj.mjOBJ_SITE, i) for i in sq.grip_sites] == ["gripper0_right_grip_site", "gripper1_right_grip_site"] and sq.num_arms == 2
    tr = scene_spec("transport").compile()
    pairs = RobomimicScene.from_model(tr, "transport").contact_pairs
    tr_name = lambda i: mujoco.mj_id2name(tr, mujoco.mjtObj.mjOBJ_GEOM, i)  # noqa: E731
    named = [([tr_name(g) for g in geoms], tr_name(base)) for geoms, base in pairs]
    assert named == [(["payload_handle", "payload_head"], "transport_target_bin_base"), (["trash_g0"], "transport_trash_bin_base")]


def test_downloader(tmp_path, monkeypatch):
    """The downloader keeps a file only at its SHA-256 in its list (each demonstration file and asset), checks one already
    there, refuses one not listed, and refuses an asset path that leaves robosuite's assets (a path through ``..`` inside
    them, as the recorded scenes name some, is kept)."""
    h5py = pytest.importorskip("h5py")
    import tools.download_robomimic as dl

    assert dl.ROBOSUITE_COMMIT.startswith("1a8701b")  # the v1.5.0 tag's commit
    assert all(f"{task}/demo_v15.hdf5" in dl.SHA256 for task in dl.TASKS)
    src = tmp_path / "src.bin"
    src.write_bytes(b"robomimic" * 1000)
    monkeypatch.setitem(dl.SHA256, "a/ok.bin", hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setitem(dl.SHA256, "a/bad.bin", "0" * 64)
    assert dl.fetch(src.as_uri(), str(tmp_path), "a/ok.bin") == str(tmp_path / "a" / "ok.bin")
    assert (tmp_path / "a" / "ok.bin").read_bytes() == src.read_bytes()
    dl.fetch(src.as_uri(), str(tmp_path), "a/ok.bin")  # there and checked
    with pytest.raises(ValueError, match="SHA-256"):
        dl.fetch(src.as_uri(), str(tmp_path), "a/bad.bin")
    assert not (tmp_path / "a" / "bad.bin").exists() and not (tmp_path / "a" / "bad.bin.part").exists()
    with pytest.raises(ValueError, match="no SHA-256"):
        dl.fetch(src.as_uri(), str(tmp_path), "a/other.bin")
    monkeypatch.setitem(dl.SHA256, "a/ok.bin", "0" * 64)
    with pytest.raises(ValueError, match="delete it"):
        dl.fetch(src.as_uri(), str(tmp_path), "a/ok.bin")

    def scene_file(name: str, files: list[str]) -> str:
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            g = f.create_group("data").create_group("demo_0")
            g.attrs["model_file"] = "<mujoco>" + "".join(f'<mesh file="/home/x/robosuite/models/assets/{q}"/>' for q in files) + "</mujoco>"
        return str(path)

    assert dl.asset_paths(scene_file("in.hdf5", ["arenas/../textures/a.png", "meshes/b.stl"])) == {"textures/a.png", "meshes/b.stl"}
    for bad in ("../../.ssh/id_ed25519", "textures/../../x.png", "/etc/passwd"):
        with pytest.raises(ValueError, match="leaves"):
            dl.asset_paths(scene_file("out.hdf5", [bad]))


def test_demonstration_file(tmp_path):
    """A demonstration file in robomimic's layout: the states' time, qpos and qvel; the scene; the control frequency."""
    h5py = pytest.importorskip("h5py")
    from tools.download_robomimic import asset_paths

    model = scene_spec().compile()
    qpos, qvel = demonstration(model, 5, 1)
    states = np.concatenate((np.arange(5)[:, None] * CONTROL_DT, qpos, qvel), -1)
    path = tmp_path / "demo_v15.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps({"env_name": "Lift", "env_kwargs": {"control_freq": 20}})
        g.create_group("demo_3").create_dataset("states", data=states)
        g["demo_3"].attrs["model_file"] = scene_xml(texture=True)
    d = load_demo(str(path), 3)
    assert d["env_name"] == "Lift" and d["control_freq"] == 20 and d["xml"] == scene_xml(texture=True)
    q, v = split_states(d["states"], model.nq, model.nv)
    assert np.array_equal(q, qpos) and np.array_equal(v, qvel)
    with pytest.raises(ValueError, match="width"):
        split_states(d["states"], model.nq + 1, model.nv)
    assert asset_paths(str(path)) == {"textures/wood.png"}


def test_kinematics_and_the_task_at_the_reference(setup):
    """The keypoints are MuJoCo's body positions; at the reference the task is at its best: the full reward, no error, no
    termination; the targets are offsets about the next frame's joints within their ranges; the reference ends at rest."""
    model, scene, qpos, qvel, kin, task = setup
    ref, d = task.ref, mujoco.MjData(model)
    assert ref.frames == len(qpos) + HOLD and torch.equal(ref.qvel[-HOLD:], torch.zeros(HOLD, model.nv))
    for t in (0, 10, 29):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, d)
        assert np.abs(d.xpos[list(scene.keypoint_bodies)] - ref.keypoints[t].numpy()).max() < 1e-5
    S, f = (ref.qpos, ref.qvel), torch.arange(ref.frames)
    fk, a0 = task.fk(ref.qpos), torch.zeros(ref.frames, task.nu)
    assert torch.allclose(task.reward(S, fk, f, a0, a0), torch.full((ref.frames,), task.max_step_reward), atol=1e-6)
    assert not task.termination(S, fk, f).any() and float(task.soft_termination(S, fk, f).max()) < 1e-3
    e = task.errors(S, fk, f)
    assert all(float(x.abs().max()) < 1e-5 for x in e.values())
    u = task.setpoint(torch.ones(ref.frames, task.nu), f)
    assert torch.all(u <= task.u_hi) and torch.all(u >= task.u_lo) and torch.allclose(task.setpoint(a0, f), ref.ctrl[ref.next_frame(f)].clamp(task.u_lo, task.u_hi))
    x = task.critic_features(S, fk, f, a0, 3)
    assert x.shape == (ref.frames, task.obs_dim + 3 * task.goal_frame_dim)
    moved = ref.qpos.clone()
    moved[:, 9 + 2] += 0.2  # the cube lifted, far from its reference: a termination
    assert task.termination((moved, ref.qvel), task.fk(moved), f).all()


def robosuite_check_success(m: mujoco.MjModel, d: mujoco.MjData, task: str) -> bool:
    """robosuite's ``_check_success`` of the task at the pinned commit, transcribed onto MuJoCo's data (the names robosuite
    gives these scenes' objects; Transport's object contact geoms listed as robosuite's models list them)."""
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    geom = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)  # noqa: E731
    if task == "lift":  # Lift: cube_height > table_height + 0.04
        return bool(d.xpos[body("cube_main")][2] > 0.8 + 0.04)
    if task == "square":  # NutAssembly, single_object_mode 2: a nut on its peg (on_peg) and r_reach < 0.6
        sites = [i for i in range(m.nsite) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i).endswith("_right_grip_site")]
        objects_on_pegs = []
        for nut, peg in (("SquareNut_main", "peg1"), ("RoundNut_main", "peg2")):
            obj_pos, peg_pos = d.xpos[body(nut)], d.xpos[body(peg)]
            dist = min(np.linalg.norm(d.site_xpos[i] - obj_pos) for i in sites)
            r_reach = 1 - np.tanh(10.0 * dist)
            on_peg = abs(obj_pos[0] - peg_pos[0]) < 0.03 and abs(obj_pos[1] - peg_pos[1]) < 0.03 and obj_pos[2] < 0.82 + 0.05
            objects_on_pegs.append(int(on_peg and r_reach < 0.6))
        return bool(np.sum(objects_on_pegs) > 0)

    def check_contact(geoms_1, geoms_2):  # sim_utils.check_contact
        for i in range(d.ncon):
            g1, g2 = geom(d.contact[i].geom1), geom(d.contact[i].geom2)
            if (g1 in geoms_1 and g2 in geoms_2) or (g1 in geoms_2 and g2 in geoms_1):
                return True
        return False

    return check_contact(["transport_target_bin_base"], ["payload_handle", "payload_head"]) and check_contact(["transport_trash_bin_base"], ["trash_g0"])


def task_of(name: str, arms: int = 1, device: str = "cpu"):
    """A scene of the task and the task on a short demonstration of it."""
    model = scene_spec(name, arms=arms).compile()
    scene = RobomimicScene.from_model(model, name)
    qpos, qvel = demonstration(model, 3, 0)
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies).to(device)
    return model, scene, qpos, RobomimicTask(scene, kin, build_reference(qpos, qvel, scene, kin, device))


def place(model: mujoco.MjModel, qpos: np.ndarray, where: dict) -> np.ndarray:
    """``qpos`` with free joints moved to positions (``where``: joint name -> x, y, z), upright."""
    q = qpos.copy()
    for joint, p in where.items():
        a = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        q[a : a + 7] = (*p, 1.0, 0.0, 0.0, 0.0)
    return q


def test_success_lift_and_square():
    """Lift's and Square's success tests read the simulator's body and grip-site positions as robosuite's do: the cube above
    the table by 4 cm; a nut on its own peg (within 3 cm in x and y, below the table's top + 5 cm) and released, more than
    atanh(0.4) / 10 = 4.24 cm from the nearer grip site, either nut."""
    model, _, qpos, task = task_of("lift")
    d = mujoco.MjData(model)
    rows = []
    for z in (0.82, 0.8399, 0.8401, 1.0):
        d.qpos[:] = place(model, qpos[0], {"cube_joint0": (0.0, 0.0, z)})
        mujoco.mj_kinematics(model, d)
        rows.append((d.xpos.copy(), d.site_xpos.copy(), robosuite_check_success(model, d, "lift")))
    got = task.success(torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32), torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32))
    assert got.tolist() == [r[2] for r in rows] == [False, False, True, True]

    model, scene, qpos, task = task_of("square", arms=2)
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[0]
    mujoco.mj_kinematics(model, d)
    sn, rn = scene.success_bodies
    far = np.array([[-0.5, 0.0, 1.9], [0.5, 0.0, 1.9]])
    on1, on2, away = np.array([0.23, 0.1, 0.83]), np.array([0.23, -0.1, 0.83]), np.array([0.0, 0.3, 0.81])
    cases = [  # the square nut, the round nut, the grip sites; robosuite's outcome
        (on1, away, far, True), (on1, away, [on1 + [0, 0, 0.03], far[1]], False), (on1, away, [far[0], on1 + [0.02, 0, 0]], False),
        (on1 + [0.035, 0, 0], away, far, False), (on1 + [0, 0, 0.045], away, far, False), (away, on2, far, True), (on2, away, far, False),
        (on1, away, [on1 + [0.042, 0, 0], far[1]], False), (on1, away, [on1 + [0.043, 0, 0], far[1]], True),
    ]  # fmt: skip
    xpos, site_xpos, want = [], [], []
    for nut, round_nut, sites, outcome in cases:
        x, sx = d.xpos.copy(), d.site_xpos.copy()
        x[sn], x[rn], sx[list(scene.grip_sites)] = nut, round_nut, sites
        dd = mujoco.MjData(model)
        dd.xpos[:], dd.site_xpos[:] = x, sx
        assert robosuite_check_success(model, dd, "square") == outcome
        xpos.append(x), site_xpos.append(sx), want.append(outcome)
    got = task.success(torch.tensor(np.stack(xpos), dtype=torch.float32), torch.tensor(np.stack(site_xpos), dtype=torch.float32))
    assert got.tolist() == want


def test_success_transport():
    """Transport's success test reads the contact list as robosuite's check_contact does: a contact between one of the
    payload's group-0 geoms and the target bin's base, and one between the trash's and the trash bin's base; touching a
    wall or the other bin's base, or no contact, is not one. The list's entries past its count are not read."""
    model, _, qpos, task = task_of("transport")
    rest_payload, rest_trash = 0.81 + 0.01 - 0.0005, 0.81 + 0.02 - 0.0005  # on a base's top, 0.5 mm into it
    cases = [  # where the payload and the trash are; robosuite's outcome
        ({"payload_joint0": (0.18, 0.25, rest_payload), "trash_joint0": (0.2, -0.25, rest_trash)}, True),
        ({"payload_joint0": (0.18, 0.25, 0.86), "trash_joint0": (0.2, -0.25, rest_trash)}, False),
        ({"payload_joint0": (0.18, 0.25, rest_payload), "trash_joint0": (-0.2, -0.25, 0.8 + 0.02 - 0.0005)}, False),
        ({"payload_joint0": (0.34 - 0.09 + 0.0005, 0.25, 0.86), "trash_joint0": (0.2, -0.25, rest_trash)}, False),  # the head on the wall
        ({"payload_joint0": (0.15, -0.25, rest_payload), "trash_joint0": (0.3, -0.3, rest_trash)}, False),  # both in the trash bin
    ]  # fmt: skip
    d = mujoco.MjData(model)
    xpos, site_xpos, geoms, worlds = [], [], [], []
    for k, (where, outcome) in enumerate(cases):
        d.qpos[:], d.qvel[:] = place(model, qpos[0], where), 0.0
        mujoco.mj_forward(model, d)
        assert d.ncon > 0 and robosuite_check_success(model, d, "transport") == outcome
        xpos.append(d.xpos.copy()), site_xpos.append(d.site_xpos.copy())
        geoms.append(np.array(d.contact.geom[: d.ncon])), worlds.append(np.full(d.ncon, k))
    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_handle")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "transport_target_bin_base")
    geoms.append(np.array([[payload, base]])), worlds.append(np.array([1]))  # an entry past the list's count: case 1 stays a failure
    valid = torch.ones(sum(len(w) for w in worlds), dtype=torch.bool)
    valid[-1] = False
    contacts = Contacts(torch.tensor(np.concatenate(geoms), dtype=torch.int32), torch.tensor(np.concatenate(worlds), dtype=torch.int32), valid)
    xpos, site_xpos = torch.tensor(np.stack(xpos), dtype=torch.float32), torch.tensor(np.stack(site_xpos), dtype=torch.float32)
    assert task.success(xpos, site_xpos, contacts).tolist() == [outcome for _, outcome in cases]
    with pytest.raises(ValueError, match="contacts"):
        task.success(xpos, site_xpos)


def test_several_demonstrations(setup):
    """Demonstrations one after the other: each holds its last state, a step and the goal's frames stay within their own
    demonstration, and the task is at its best on each."""
    model, scene, qpos, qvel, kin, _ = setup
    q2, v2 = demonstration(model, 12, 3)
    ref = build_reference([qpos, q2], [qvel, v2], scene, kin, "cpu")
    n1, n2 = len(qpos) + HOLD, len(q2) + HOLD
    assert ref.frames == n1 + n2 and torch.equal(torch.unique(ref.start), torch.tensor([0, n1]))
    assert torch.equal(ref.end[:n1], torch.full((n1,), n1 - 1)) and torch.equal(ref.end[n1:], torch.full((n2,), n1 + n2 - 1))
    f = torch.tensor([0, n1 - 2, n1 - 1, n1, n1 + n2 - 1])
    assert ref.next_frame(f).tolist() == [1, n1 - 1, n1 - 1, n1 + 1, n1 + n2 - 1]
    assert ref.ahead(f, 5).tolist() == [5, n1 - 1, n1 - 1, n1 + 5, n1 + n2 - 1]
    assert torch.allclose(ref.qpos[n1 : n1 + len(q2)], torch.tensor(q2, dtype=torch.float32))
    task = RobomimicTask(scene, kin, ref)
    S, g = (ref.qpos, ref.qvel), torch.arange(ref.frames)
    fk, a0 = task.fk(ref.qpos), torch.zeros(ref.frames, task.nu)
    assert torch.allclose(task.reward(S, fk, g, a0, a0), torch.full((ref.frames,), task.max_step_reward), atol=1e-6)
    goal = task.goal(S, fk, g, 3).reshape(ref.frames, 3, task.goal_frame_dim)
    assert torch.equal(goal[n1 - 1, 0], goal[n1 - 1, 2])  # at the first demonstration's last frame, every goal frame is that frame
    assert not torch.equal(goal[len(qpos) - 5, 0], goal[len(qpos) - 5, 2])  # (before the held last state)


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_world_model_steps_a_control_step(setup, kind):
    """The world model's step is of one control step: its targets are the control step's, and a model fit to them (the
    normalizers alone, from a zero network: the mean target) reproduces a step of constant acceleration or increment."""
    from ndm_gym.algos.robomimic_algos import world_model

    model, scene, qpos, qvel, _, _ = setup
    wm = world_model(model, argparse.Namespace(wm_hidden=[32], wm_act="silu", wm_no_ln=False), kind, SUBSTEPS)
    assert wm.lay.dt == CONTROL_DT and (wm.kind, hasattr(wm, "pos_lead")) == (kind, kind == "accel")
    q, v = torch.tensor(qpos, dtype=torch.float32), torch.tensor(qvel, dtype=torch.float32)
    u = q[1:, list(scene.act_q)]
    wm.update_norms(q[:-1], v[:-1], u, q[1:], v[1:])
    q1, v1 = wm.step(q[:-1], v[:-1], u)
    assert q1.shape == q[1:].shape and v1.shape == v[1:].shape and torch.isfinite(q1).all()
    e = wm.error(q1, v1, q[1:], v[1:])
    assert e.shape == (len(u), 2 * model.nv) and torch.isfinite(e).all()


def agent_and_replay(model, task, argv):
    args = parse_args(["--task", "lift", "--wm_hidden", "64", "64", "--pi_batch", "16", "--wm_batch", "8", "--wm_window", "4", "--horizon", "4",
                       "--goal_frames", "2", "--num_envs", "4", "--buffer_steps", "16", *argv])  # fmt: skip
    algo = ROBOMIMIC_ALGOS[args.algo](args, model, task, "cpu", SUBSTEPS)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, "cpu", fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    ref, rng = task.ref, torch.Generator().manual_seed(0)
    for t in range(12):  # transitions near the reference
        f = torch.full((args.num_envs,), t, dtype=torch.long)
        s = torch.cat((ref.qpos[f], ref.qvel[f]), -1) + 1e-3 * torch.randn(args.num_envs, task.state_dim, generator=rng)
        s1 = torch.cat((ref.qpos[f + 1], ref.qvel[f + 1]), -1)
        u = ref.ctrl[f + 1]
        algo.update_wm_norms(unflatten_state(s, task.nq), unflatten_state(s1, task.nq), u)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": torch.zeros(args.num_envs, task.nu), "f": f, "done": torch.zeros(args.num_envs, dtype=torch.bool)})
    return args, algo, buffer


@pytest.mark.parametrize("argv", [["--algo", "supermimic", "--critic_minibatches", "2"], ["--algo", "supertrack"], ["--algo", "supertrack", "--st_task_weights", "--st_fail_w", "2"]])
def test_updates(setup, monkeypatch, argv):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    model, _, _, _, _, task = setup
    args, algo, buffer = agent_and_replay(model, task, argv)
    assert algo.wm.kind == ("accel" if args.algo == "supertrack" else "increment")
    before = [p.clone() for p in algo.pi.parameters()]
    assert torch.isfinite(algo.wm_update(buffer))
    loss, v_loss, imag_r, grad_norm, alive = algo.policy_update(buffer)
    assert all(torch.isfinite(x) for x in (loss, imag_r, grad_norm, alive)) and 0 < float(imag_r) <= 1 and 0 < float(alive) <= 1
    assert torch.isfinite(v_loss) == (args.algo == "supermimic")
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    if args.algo == "supertrack":
        c = task.cfg
        assert algo.bal.calibrated and (algo.step_worth == c.w_keypoint + c.w_joint + c.w_object_pos + c.w_object_ori) == args.st_task_weights
        algo2 = ROBOMIMIC_ALGOS[args.algo](args, model, task, "cpu", SUBSTEPS)
        algo2.load_state_dict(algo.state_dict())
        assert torch.equal(algo2.bal.w, algo.bal.w)


def test_training_arguments():
    """Every argument of ndm_gym.train is implemented or refused; the defaults are ROBOMIMIC_DEFAULTS over LAFAN1's."""
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    assert set(ROBOMIMIC_DEFAULTS) <= set(SUPPORTED)
    args = parse_args(["--task", "square"])
    assert all(getattr(args, name) == value for name, value in ROBOMIMIC_DEFAULTS.items()) and (args.task, args.demo, args.episode) == ("square", [0], 200)
    for refused in (["--algo", "blinkmimic"], ["--wm_backbone", "rigidformer"], ["--st_fail_w", "2"], ["--st_task_weights"],
                    ["--algo", "supertrack", "--gamma", "0.995"], ["--episode", "1"], ["--no_sync"], ["--collect_graph"],
                    ["--solver_ls_iterations", "10"], ["--solver_iterations", "-1"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--task", "lift"] + refused)
    with pytest.raises(SystemExit):
        parse_args(["--task", "stack"])
    st = parse_args(["--task", "lift", "--algo", "supertrack", "--st_fail_w", "2", "--st_task_weights", "--demo", "7", "--solver_iterations", "20",
                     "--solver_ls_iterations", "10"])  # fmt: skip
    assert (st.algo, st.st_fail_w, st.st_task_weights, st.demo, st.solver_iterations, st.solver_ls_iterations) == ("supertrack", 2, True, [7], 20, 10)


def test_evaluation_refuses_no_rollouts(monkeypatch, capsys):
    from ndm_gym import evaluate_robomimic

    monkeypatch.setattr(sys, "argv", ["evaluate_robomimic", "--ckpt", "last.pt", "--out", "eval", "--num_envs", "0"])
    with pytest.raises(SystemExit):
        evaluate_robomimic.main()
    assert "--num_envs >= 1" in capsys.readouterr().err


def test_time_to_the_target(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("time_to", os.path.join(ROOT, "benchmarks", "robomimic_manipulation_tracking", "time_to.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(sys, "argv", ["time_to.py", str(run)])
    with pytest.raises(SystemExit, match="no ckpt_"):  # a run without kept checkpoints is not "target not reached"
        tool.main()

    def checkpoint(it, success, object_mm, num_envs=256, own=False):
        (run / f"ckpt_{it:06d}.pt").write_bytes(b"")
        out = run / "eval" if own else run / "eval_ckpts" / f"{it:06d}"
        out.mkdir(parents=True, exist_ok=True)
        metrics = {"iteration": it, "wall_time_s": 6.0 * it, "num_envs": num_envs, "policy": {"success_rate": success, "object_mm": object_mm}}
        (out / "metrics.json").write_text(json.dumps(metrics))

    for it, success, object_mm in ((100, 1.0, 5.0), (200, 0.9, 5.0), (300, 0.99, 12.0), (400, 1.0, 9.0)):  # the target: 0.99, 10 mm
        checkpoint(it, success, object_mm)
    checkpoint(500, 0.995, 8.0, own=True)  # the last checkpoint's evaluation is the run's own
    tool.main()  # the evaluations are there: none is run
    res = json.loads((run / "time_to.json").read_text())
    assert res["iteration_to_target"] == 400 and res["minutes_to_target"] == 40.0  # held from 400 on, not from 100
    assert [r["target"] for r in res["checkpoints"]] == [True, False, False, True, True]
    checkpoint(450, 1.0, 5.0, num_envs=64)  # another evaluation's metrics
    with pytest.raises(SystemExit, match="not this script's evaluation"):
        tool.main()
# hand_fast's options that robomimic training implements: the systems options, the optimizers, the policy's average and
# learning rate, and the inductor's settings; the rest are HAND_ONLY
HAND_FAST_IMPLEMENTED = ("fused_adam", "reset_masked", "no_sync", "collect_graph", "compile_eager", "update_graphs", "pi_opt", "wm_opt", "pi_ema",
                         "pi_lr_mult", "lr_decay_ret", "lr_decay_len", "lr_decay_by", "muon_batched", "elementwise_3x3", "combo_kernels",
                         "mix_order_reductions", "autotune_gemm")  # fmt: skip


def test_hand_options(capsys):
    """Every option of hand_fast is implemented here or refused (HAND_ONLY: what only hand training's code reads), so none is
    silently ignored; the inductor's settings are accepted."""
    from ndm_gym.algos.hand_fast import add_hand_fast_args

    parser = argparse.ArgumentParser()
    add_hand_fast_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(HAND_FAST_IMPLEMENTED) | set(HAND_ONLY) and not set(HAND_FAST_IMPLEMENTED) & set(HAND_ONLY)
    systems = ["--no_sync", "--fused_adam", "--reset_masked"]
    for name in HAND_ONLY:
        with pytest.raises(SystemExit):
            parse_args(["--task", "lift", f"--{name}", *systems])
        assert "hand training" in capsys.readouterr().err
    args = parse_args(["--task", "lift", "--elementwise_3x3", "--combo_kernels", "--mix_order_reductions", "--autotune_gemm", *systems])
    assert args.elementwise_3x3 and args.combo_kernels and args.mix_order_reductions and args.autotune_gemm


CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")


@GPU
def test_simulator_matches_mujoco_and_resets():
    """RobomimicSim's control step is MuJoCo's physics steps of it; episodes start at reference frames with the arms'
    noise asked for; ``step_nosync`` is ``step`` with the ends reset by mask."""
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim

    model = scene_spec().compile()
    scene = RobomimicScene.from_model(model, "lift")
    qpos, qvel = demonstration(model, 20, 2)
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies).to(CUDA)
    task = RobomimicTask(scene, kin, build_reference(qpos, qvel, scene, kin, CUDA), RewardCfg())
    n = 8
    sim = RobomimicSim(model, task, n, CUDA, joint_noise=0.0)
    frames = torch.arange(n, device=CUDA)
    sim.reset(torch.ones(n, dtype=torch.bool, device=CUDA), frames)
    assert torch.allclose(sim.qpos, task.ref.qpos[frames]) and torch.equal(sim.last, torch.full_like(frames, task.ref.frames - 1))
    u = task.setpoint(torch.zeros(n, task.nu, device=CUDA), frames)
    s1, term, done = sim.step(u, reset=False)
    d = mujoco.MjData(model)
    for i in (0, 5):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], u[i].cpu().numpy()
        for _ in range(sim.substeps):
            mujoco.mj_step(model, d)
        diff = np.abs(s1[i, : model.nq].cpu().numpy() - d.qpos)
        assert diff[: scene.nu].max() < 1e-5 and diff.max() < 1e-3  # the robot's joints; a resting cube's contact settles to 0.2 mm
    assert torch.equal(sim.f, frames + 1) and not term.any()
    s1n, _, _ = sim.step_nosync(task.setpoint(torch.zeros(n, task.nu, device=CUDA), sim.f))
    assert torch.isfinite(s1n).all()


# each task's starts for the success test after a control step: where the objects are, robosuite's outcome
REST = {"payload": 0.81 + 0.01 - 0.0005, "trash": 0.81 + 0.02 - 0.0005, "nut": 0.8 + 0.01 - 0.0005}
SUCCESS_STARTS = {
    "square": [({"SquareNut_joint0": (0.255, 0.1, REST["nut"])}, True), ({"SquareNut_joint0": (0.28, 0.1, REST["nut"])}, False),
               ({"RoundNut_joint0": (0.257, -0.1, REST["nut"])}, True)],
    "transport": [({"payload_joint0": (0.18, 0.25, REST["payload"]), "trash_joint0": (0.2, -0.25, REST["trash"])}, True),
                  ({"payload_joint0": (0.18, 0.25, 0.84), "trash_joint0": (0.2, -0.25, REST["trash"])}, False),
                  ({"payload_joint0": (0.18, 0.25, REST["payload"]), "trash_joint0": (-0.2, -0.25, 0.8 + 0.02 - 0.0005)}, False)],
}  # fmt: skip


@GPU
@pytest.mark.parametrize("name", ["square", "transport"])
def test_simulator_success_matches_robosuite(name):
    """RobomimicSim.success after a control step is robosuite's test on MuJoCo's data after the same physics steps: the
    bodies' and grip sites' positions and the contact list the last step left (also after the step's resets)."""
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim

    model, _, qpos, task = task_of(name, device=CUDA)
    starts = [place(model, qpos[0], where) for where, _ in SUCCESS_STARTS[name]]
    n = len(starts)
    sim = RobomimicSim(model, task, n, CUDA, joint_noise=0.0)
    frames = torch.zeros(n, dtype=torch.long, device=CUDA)
    sim.reset(torch.ones(n, dtype=torch.bool, device=CUDA), frames)
    sim.qpos.copy_(torch.tensor(np.stack(starts), dtype=torch.float32, device=CUDA)), sim.qvel.zero_()
    u = task.setpoint(torch.zeros(n, task.nu, device=CUDA), frames)
    sim.step(u, reset=False)
    got = sim.success().tolist()
    sim.reset(torch.ones(n, dtype=torch.bool, device=CUDA), frames)  # (a reset writes the state, not what the test reads)
    assert sim.success().tolist() == got and not sim.contact_overflow()
    d = mujoco.MjData(model)
    want = []
    for q, uu in zip(starts, u.cpu().numpy(), strict=True):
        d.qpos[:], d.qvel[:], d.ctrl[:] = q, 0.0, uu
        for _ in range(sim.substeps):
            mujoco.mj_step(model, d)
        want.append(robosuite_check_success(model, d, name))
    assert got == want == [outcome for _, outcome in SUCCESS_STARTS[name]]
