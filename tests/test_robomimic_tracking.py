"""Manipulation demonstration tracking on small scenes in robosuite's naming (seven-joint arms with their torque actuators,
two-finger grippers with their grip sites, the task's free objects), their demonstrations made by MuJoCo itself: the servos
and the gravity compensation, the scene's layout for the task, the demonstration's file and assets and the downloader's
checks, the forward kinematics against MuJoCo's, the task at the reference, robosuite's success tests against their
transcription from robosuite, the world model's control step, the updates, training's arguments, the evaluation's rollout
and summary, and the time to the target; on a GPU, the simulator against MuJoCo and its success test against robosuite's."""

import argparse
import hashlib
import importlib.util
import json
import math
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the arguments robomimic training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip
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
    "lift": ('<body name="cube_main" pos="0 0 0.82"><freejoint name="cube_joint0"/>'
             '<geom name="cube_g0" type="box" size="0.02 0.02 0.02" mass="0.07"/></body>'),
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
    resolved = resolve_assets('<texture file="/x/y/robosuite/models/assets/textures/a.png"/>', "/data/assets")
    assert resolved == '<texture file="/data/assets/textures/a.png"/>'
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
    there, refuses one not listed, starts a failed transfer again up to its attempts, and refuses an asset path that leaves
    robosuite's assets (a path through ``..`` inside them, as the recorded scenes name some, is kept)."""
    h5py = pytest.importorskip("h5py")
    import tools.download_robomimic as dl

    assert dl.ROBOSUITE_COMMIT.startswith("1a8701b")  # the v1.5.0 tag's commit
    assert all(f"{task}/demo_v15.hdf5" in dl.SHA256 for task in dl.TASKS)
    src = tmp_path / "src.bin"
    src.write_bytes(b"robomimic" * 1000)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    monkeypatch.setitem(dl.SHA256, "a/ok.bin", digest)
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
    import urllib.request

    calls, real = [], urllib.request.urlopen

    def flaky(url, timeout):  # a reset connection twice, then the file
        calls.append(url)
        if len(calls) <= 2:
            raise ConnectionResetError(104, "Connection reset by peer")
        return real(url, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setitem(dl.SHA256, "a/retried.bin", digest)
    monkeypatch.setitem(dl.SHA256, "a/never.bin", digest)
    dl.fetch(src.as_uri(), str(tmp_path), "a/retried.bin", wait=0.0)
    assert len(calls) == 3 and (tmp_path / "a" / "retried.bin").read_bytes() == src.read_bytes()
    calls.clear()
    with pytest.raises(ConnectionResetError):
        dl.fetch(src.as_uri(), str(tmp_path), "a/never.bin", attempts=2, wait=0.0)
    assert len(calls) == 2 and not (tmp_path / "a" / "never.bin").exists()

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
    assert torch.all(u <= task.u_hi) and torch.all(u >= task.u_lo)
    assert torch.allclose(task.setpoint(a0, f), ref.ctrl[ref.next_frame(f)].clamp(task.u_lo, task.u_hi))
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

    payload_in_target_bin = check_contact(["transport_target_bin_base"], ["payload_handle", "payload_head"])
    return payload_in_target_bin and check_contact(["transport_trash_bin_base"], ["trash_g0"])


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
    f32 = lambda xs: torch.tensor(np.stack(xs), dtype=torch.float32)  # noqa: E731
    got = task.success(f32([r[0] for r in rows]), f32([r[1] for r in rows]))
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
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": torch.zeros(args.num_envs, task.nu), "f": f,
                    "done": torch.zeros(args.num_envs, dtype=torch.bool)})  # fmt: skip
    return args, algo, buffer


@pytest.mark.parametrize("argv", [["--algo", "supermimic", "--critic_minibatches", "2"], ["--algo", "supertrack"],
                                  ["--algo", "supertrack", "--st_task_weights", "--st_fail_w", "2"]])  # fmt: skip
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


def cost_by_hand(a, a_prev, w: float, scale, width) -> np.ndarray:
    """--target_rate_w's cost written out: w times the mean over the actuators of (scale (a - a_prev) / width)^2."""
    return w * ((np.asarray(scale) * (np.asarray(a) - np.asarray(a_prev)) / np.asarray(width)) ** 2).mean(-1)


SCALE, WIDTH = [0.25] * 7 + [0.02] * 2, [0.4] * 7 + [0.04] * 2  # an arm joint's offset scale and joint_std, a finger's and keypoint_std


def test_target_rate_cost(setup):
    """--target_rate_w's cost in the task: off by default (None, the reward as it was); on, an action after another worked out
    by hand, its gradient, the reward less it, a change of the kernel's width on every actuator costing the weight, and the
    widths following --reward_std_scale."""
    _, scene, _, _, kin, task0 = setup
    w = 4.0
    task = RobomimicTask(scene, kin, task0.ref, RewardCfg(w_target_rate=w))
    assert RewardCfg().w_target_rate == 0.0 and task0.smooth_cost(torch.ones(2, 9), torch.zeros(2, 9)) is None
    assert task.rate_k.tolist() == pytest.approx([0.625] * 7 + [0.5] * 2)  # 0.25 / 0.4 and 0.02 / 0.04
    a_prev, a = torch.zeros(1, 9), torch.zeros(1, 9)
    a[0, 0], a[0, 7] = 0.4, -0.2  # an arm joint's target 0.1 rad on, a finger's 4 mm back
    a.requires_grad_()
    cost = task.smooth_cost(a, a_prev)
    assert float(cost.detach()) == pytest.approx(4 * ((0.625 * 0.4) ** 2 + (0.5 * 0.2) ** 2) / 9, rel=1e-6)  # 4 * 0.0725 / 9
    assert float(cost.detach()) == pytest.approx(float(cost_by_hand(a.detach(), a_prev, w, SCALE, WIDTH)[0]), rel=1e-6)
    (grad,) = torch.autograd.grad(cost.sum(), a)
    by_hand = torch.zeros(9)
    by_hand[0], by_hand[7] = 2 * w * 0.625**2 * 0.4 / 9, 2 * w * 0.5**2 * -0.2 / 9  # 2 w k^2 (a - a_prev) / actuators
    assert torch.allclose(grad[0], by_hand, atol=1e-7)
    f = torch.tensor([3])
    S = (task.ref.qpos[f], task.ref.qvel[f])
    fk = task.fk(S[0])
    with torch.no_grad():
        assert torch.allclose(task.reward(S, fk, f, a, a_prev), task0.reward(S, fk, f, a, a_prev) - cost * CONTROL_DT, atol=1e-7)
        assert torch.equal(task0.reward(S, fk, f, a, a), task.reward(S, fk, f, a, a))  # no change of the target, no cost
        one_width = torch.tensor(WIDTH) / torch.tensor(SCALE)  # every actuator's target changed by its kernel's width
        assert float(task.smooth_cost(one_width[None], a_prev)) == pytest.approx(w, rel=1e-6)
        wide = RobomimicTask(scene, kin, task0.ref, RewardCfg(w_target_rate=w).scaled(2.0))
        assert float(wide.smooth_cost(a, a_prev)) == pytest.approx(float(cost) / 4, rel=1e-6)


def test_target_rate_on_both_algorithms(setup, monkeypatch):
    """--target_rate_w on both algorithms' own paths, each against the same update with the option off (the same networks, starts
    and noise). The per-step quantity is the same for both, the cost written out by hand from the actions each recorded:
    SuperTrack's loss rises by its mean over the imagined steps and the batch, SuperMimic's policy loss by its discounted,
    survival-weighted sum times the control step (the reward's place). In each algorithm's own step the cost's gradient on the
    action is the one worked out by hand."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    model, scene, _, _, kin, task0 = setup
    w = 4.0
    task_on = RobomimicTask(scene, kin, task0.ref, RewardCfg(w_target_rate=w))
    per_step = {}
    for argv in (["--algo", "supertrack", "--st_task_weights", "--st_fail_w", "2"], ["--algo", "supermimic", "--critic_minibatches", "2"]):
        st = argv[1] == "supertrack"
        runs = {}
        for name, task in (("off", task0), ("on", task_on)):
            torch.manual_seed(0)  # the same networks in both
            args, algo, buffer = agent_and_replay(model, task, argv + (["--target_rate_w", str(w)] if name == "on" else []))
            assert args.target_rate_w == (w if name == "on" else 0.0)
            with torch.no_grad():  # the output layer at its plain initialization (the policy starts it 100 times smaller): actions
                algo.pi.net[-1].weight.mul_(100.0)  # that change from step to step
            seen = []
            step = algo.st_step if st else algo.imag_step

            def recorded(*x, _step=step, _seen=seen):
                out = _step(*x)
                _seen.append((x[3], out[3], out[5], out[6] if st else out[4], out[7] if st else None))  # a_prev, a, d, reward, cost
                return out

            setattr(algo, "st_step" if st else "imag_step", recorded)
            torch.manual_seed(1)  # the same starts and imagination noise in both
            loss = algo.policy_update(buffer)[0]
            runs[name] = (float(loss), seen, args, algo, buffer)
        (loss0, seen0, args, _, _), (loss1, seen1, _, _, _) = runs["off"], runs["on"]
        assert len(seen0) == len(seen1) == args.horizon and all(torch.equal(x[1], y[1]) for x, y in zip(seen0, seen1, strict=True))
        costs = np.stack([cost_by_hand(a.detach(), a_prev.detach(), w, SCALE, WIDTH) for a_prev, a, _, _, _ in seen0])  # (H, B)
        assert costs.shape == (args.horizon, args.pi_batch) and costs.mean() > 1e-4
        for (_, _, _, r0, c0), (_, _, _, r1, c1), by_hand in zip(seen0, seen1, costs, strict=True):
            assert c0 is None and np.allclose((r0 - r1).detach().numpy() / CONTROL_DT, by_hand, atol=1e-6)  # the reward less the cost
            if st:
                assert np.allclose(c1.detach().numpy(), by_hand, atol=1e-7)
        per_step[argv[1]] = costs
        if st:  # the loss: plus the cost's mean over the steps and the batch
            assert loss1 - loss0 == pytest.approx(costs.mean(), rel=1e-4, abs=1e-6)
        else:  # the policy loss, -J: plus sum_t gamma^t c_t cost_t dt, c_t the survival before step t
            d = torch.stack([x[2] for x in seen0]).detach().numpy()
            c_t = np.concatenate((np.ones((1, d.shape[1])), np.cumprod(1 - d, 0)[:-1]), 0)
            disc = args.gamma ** np.arange(args.horizon)[:, None]
            assert loss1 - loss0 == pytest.approx((disc * c_t * costs * CONTROL_DT).sum(0).mean(), rel=1e-4, abs=1e-7)
        # the gradient on the action, in the algorithm's own step
        _, _, _, algo0, buffer = runs["off"]
        algo1 = runs["on"][3]
        algo1.load_state_dict(algo0.state_dict())  # (both updated above: the same networks again)
        torch.manual_seed(2)
        S, f, _ = algo0._starts(buffer)
        a_prev = 0.3 * torch.randn(args.pi_batch, task0.nu)
        grads = {}
        for name, algo in (("off", algo0), ("on", algo1)):
            if st:
                out = algo._st_step(S, algo.task.fk(S[0]), f, a_prev, torch.zeros_like(a_prev))
            else:
                fk, x = algo._start_features(S, f, a_prev)
                out = algo._imag_step(S, fk, f, a_prev, x)
            a, reward = out[3], out[6] if st else out[4]
            grads[name] = (a.detach(), torch.autograd.grad(reward.sum(), a, retain_graph=True)[0])
            if st and name == "on":
                grads["cost"] = torch.autograd.grad(out[7].sum(), a)[0]
        (a, g0), (a1, g1) = grads["off"], grads["on"]
        assert torch.equal(a, a1)
        by_hand = 2 * w * (torch.tensor(SCALE) / torch.tensor(WIDTH)) ** 2 * (a - a_prev) / task0.nu  # d cost / d a
        assert by_hand.abs().max() > 1e-3 and torch.allclose(g0 - g1, by_hand * CONTROL_DT, atol=1e-6)  # the reward's gradient less it
        if st:
            assert torch.allclose(grads["cost"], by_hand, atol=1e-6)
    # the same per-step formula on both paths (each on its own actions), in SuperTrack's loss and in SuperMimic's reward
    assert set(per_step) == {"supertrack", "supermimic"}


def test_target_rate_wiring(setup, tmp_path, monkeypatch):
    """--target_rate_w from the command line to the task: the parser (off by default, refused below 0 and for SuperTrack without
    --st_task_weights), main's call of build, build's task (on a demonstration file written here), and the agent's refusal."""
    h5py = pytest.importorskip("h5py")
    from ndm_gym import train_robomimic

    assert parse_args(["--task", "lift"]).target_rate_w == 0.0
    assert parse_args(["--task", "lift", "--algo", "supermimic", "--target_rate_w", "4"]).target_rate_w == 4.0
    assert parse_args(["--task", "lift", "--algo", "supertrack", "--st_task_weights", "--target_rate_w", "2"]).target_rate_w == 2.0
    for refused in (["--target_rate_w", "-1"], ["--algo", "supertrack", "--target_rate_w", "4"]):
        with pytest.raises(SystemExit):
            parse_args(["--task", "lift"] + refused)
    model, scene, _, _, kin, task0 = setup
    task_on = RobomimicTask(scene, kin, task0.ref, RewardCfg(w_target_rate=4.0))
    plain = argparse.Namespace(**{**vars(parse_args(["--task", "lift", "--algo", "supertrack"])), "wm_hidden": [64, 64]})
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    with pytest.raises(ValueError, match="st_task_weights"):  # an agent built without the parser
        ROBOMIMIC_ALGOS["supertrack"](plain, model, task_on, "cpu", SUBSTEPS)
    # build: a demonstration file in robomimic's layout, the scene of these tests
    qpos, qvel = demonstration(model, 6, 1)
    (tmp_path / "lift").mkdir()
    with h5py.File(tmp_path / "lift" / "demo_v15.hdf5", "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps({"env_name": "Lift", "env_kwargs": {"control_freq": 20}})
        g.create_group("demo_0").create_dataset("states", data=np.concatenate((np.arange(6)[:, None] * CONTROL_DT, qpos, qvel), -1))
        g["demo_0"].attrs["model_file"] = scene_xml()
    a, a_prev = torch.full((1, 9), 0.2), torch.zeros(1, 9)
    _, built, _ = train_robomimic.build("lift", 0, str(tmp_path), "cpu", target_rate_w=4.0)
    assert built.cfg.w_target_rate == 4.0
    assert float(built.smooth_cost(a, a_prev)) == pytest.approx(float(cost_by_hand(a, a_prev, 4.0, SCALE, WIDTH)[0]), rel=1e-6)
    assert train_robomimic.build("lift", 0, str(tmp_path), "cpu")[1].smooth_cost(a, a_prev) is None
    # main: the option reaches build
    pytest.importorskip("mujoco_warp")  # main imports the simulator
    calls = []

    class Stop(Exception):
        pass

    def fake_build(*args, **kw):
        calls.append((args, kw))
        raise Stop

    monkeypatch.setattr(train_robomimic, "build", fake_build)
    monkeypatch.setattr(sys, "argv", ["train_robomimic", "--task", "lift", "--algo", "supermimic", "--target_rate_w", "4", "--log_dir",
                                      str(tmp_path / "run"), "--data_dir", str(tmp_path)])  # fmt: skip
    with pytest.raises(Stop):
        train_robomimic.main()
    (args, kw), = calls
    assert args[:3] == ("lift", [0], str(tmp_path)) and args[-1] == 4.0 and len(args) == 8 and not kw
    assert json.load(open(tmp_path / "run" / "args.json"))["target_rate_w"] == 4.0


def test_evaluation_saves_targets(setup):
    """evaluate_robomimic's rollout hands out each step's position targets (--save_targets): the task's setpoints of the
    actions at the frames; without the list it runs as before."""
    from ndm_gym.evaluate_robomimic import rollout

    _, _, _, _, _, task = setup
    starts = torch.zeros(2, dtype=torch.long)

    class Offsets:  # the policy's place: an action that grows with the step
        def __init__(self):
            self.k = 0

        def act(self, S, a_prev, f):
            self.k += 1
            return torch.full((len(f), task.nu), 0.01 * self.k)

    targets = []
    states, lens, _, _, _ = rollout(StubSim(task, 2), Offsets(), task, starts, targets=targets)
    n = int(lens[0])
    assert len(targets) == n == states.shape[0] - 1 and all(u.shape == (2, task.nu) for u in targets)
    for k, u in enumerate(targets):
        f = torch.full((2,), k, dtype=torch.long)
        assert torch.equal(u, task.setpoint(torch.full((2, task.nu), 0.01 * (k + 1)), f))
    zero = task.setpoint(torch.zeros(n, task.nu), torch.arange(n))  # the reference's targets, as --save_targets writes them
    assert torch.equal(zero, task.ref.ctrl[1 : n + 1].clamp(task.u_lo, task.u_hi))
    again = rollout(StubSim(task, 2), Offsets(), task, starts)
    assert torch.equal(again[0], states)


def test_training_arguments():
    """Every argument of ndm_gym.train is implemented or refused; the defaults are ROBOMIMIC_DEFAULTS over LAFAN1's."""
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    assert set(ROBOMIMIC_DEFAULTS) <= set(SUPPORTED)
    args = parse_args(["--task", "square"])
    assert all(getattr(args, name) == value for name, value in ROBOMIMIC_DEFAULTS.items())
    assert (args.task, args.demo, args.episode) == ("square", [0], 200)
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


def test_time_to_script_and_its_errors(tmp_path, monkeypatch):
    """time_to.py as a script: a run without kept checkpoints and an evaluation that is not its own are errors."""
    spec = importlib.util.spec_from_file_location("time_to", os.path.join(REPO, "benchmarks", "robomimic_manipulation_tracking", "time_to.py"))
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


class StubSim:
    """A simulator that follows the reference exactly: ``term_at`` (world -> step) terminates a world, ``success_frames``
    (world -> frame offsets from its demonstration's end) says at which frames robosuite's test holds, ``overflow_at`` a
    step whose contact list is full."""

    def __init__(self, task, n: int, term_at=None, success_frames=None, overflow_at=None):
        self.task, self.N, self.device, self.ref = task, n, "cpu", task.ref
        self.term_at, self.success_frames, self.overflow_at, self.t = term_at or {}, success_frames or {}, overflow_at, 0
        self.contact_geom = torch.zeros(4, 2)

    def reset(self, mask, frame):
        self.f, self.t = frame.clone(), 0

    def state(self):
        return torch.cat((self.ref.qpos[self.f], self.ref.qvel[self.f]), -1)

    def step(self, u, reset=True):
        assert not reset and u.shape == (self.N, self.task.nu)
        self.f, self.t = self.ref.next_frame(self.f), self.t + 1
        term = torch.tensor([self.term_at.get(i) == self.t for i in range(self.N)])
        return self.state(), term, term

    def contact_overflow(self):
        return self.t == self.overflow_at

    def success(self):
        return torch.tensor([int(self.ref.end[self.f[i]] - self.f[i]) in self.success_frames.get(i, (0,)) for i in range(self.N)])


def test_evaluation_rollout_and_summary(setup):
    """evaluate_robomimic's rollout: every world to its own demonstration's last frame (a shorter one holds it), its
    termination and robosuite's test read after the step that reaches the last frame; a full contact list is an error. The
    summary's rates and times per demonstration, its errors over the rollouts that do not terminate."""
    from ndm_gym.evaluate_robomimic import rollout, summarize

    model, scene, qpos, qvel, kin, _ = setup
    q2, v2 = demonstration(model, 12, 3)
    ref = build_reference([qpos, q2], [qvel, v2], scene, kin, "cpu")
    task = RobomimicTask(scene, kin, ref)
    firsts = torch.unique(ref.start)
    starts = firsts[torch.tensor([0, 0, 0, 1])]  # three worlds on the first demonstration, one on the second
    n1 = int(ref.end[firsts[0]] - firsts[0])

    class Hold:  # the policy's place: no action
        def act(self, S, a_prev, f):
            return torch.zeros(len(f), task.nu)

    # world 1 terminates at step 6, world 2 meets the test one frame before its last only
    sim = StubSim(task, 4, term_at={1: 6}, success_frames={2: (1,)})
    states, lens, failed, until, success = out = rollout(sim, Hold(), task, starts)
    assert states.shape == (n1 + 1, 4, task.state_dim) and lens.tolist() == [n1, n1, n1, int(ref.end[firsts[1]] - firsts[1])]
    assert failed.tolist() == [False, True, False, False] and until.tolist() == [n1, 6, n1, int(lens[3])]
    assert success.tolist() == [True, False, False, True]
    m = summarize(task, out, torch.tensor([0, 1, 2]), int(firsts[0]))
    assert m["success_rate"] == pytest.approx(1 / 3) and m["terminated"] == pytest.approx(1 / 3)
    assert m["tracked_s"] == pytest.approx((2 * n1 + 6) / 3 * CONTROL_DT) and m["object_mm"] < 1e-3 and m["keypoint_mm"] < 1e-2
    assert math.isnan(summarize(task, out, torch.tensor([1]), int(firsts[0]))["object_mm"])  # (every rollout terminated)
    with pytest.raises(RuntimeError, match="contact list is full"):
        rollout(StubSim(task, 4, overflow_at=2), Hold(), task, starts)


def test_time_to_the_target(tmp_path):
    """time_to.py: the wall time of the first kept checkpoint from which every later one meets the target (a success of
    0.99 with the objects within 10 mm), none if the last does not; the run's own evaluation of its last checkpoint."""
    path = os.path.join(REPO, "benchmarks", "robomimic_manipulation_tracking", "time_to.py")
    spec = importlib.util.spec_from_file_location("robomimic_time_to", path)
    time_to = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(time_to)

    def run(name: str, rows: dict, own: int | None = None):
        path = tmp_path / name
        path.mkdir()
        for it in rows:
            (path / f"ckpt_{it:06d}.pt").write_bytes(b"")
        metrics = {it: {"iteration": it, "wall_time_s": 60.0 * it / 100, "num_envs": time_to.NUM_ENVS,
                        "policy": {"success_rate": r[0], "object_mm": r[1]}} for it, r in rows.items()}  # fmt: skip
        if own is not None:
            (path / "eval").mkdir()
            (path / "eval" / "metrics.json").write_text(json.dumps(metrics[own]))
        calls = []

        def evaluate(ckpt, out):
            calls.append(int(os.path.basename(ckpt)[5:11]))
            return metrics[calls[-1]]

        return time_to.time_to(str(path), evaluate), calls

    res, calls = run("a", {100: (1.0, 5.0), 200: (0.98, 5.0), 300: (1.0, 8.0), 400: (0.995, 9.9), 500: (1.0, 4.0)}, own=500)
    assert (res["iteration_to_target"], res["minutes_to_target"]) == (300, 3.0) and calls == [100, 200, 300, 400]
    assert [r["target"] for r in res["checkpoints"]] == [True, False, True, True, True] and (tmp_path / "a" / "time_to.json").exists()
    res, _ = run("b", {100: (1.0, 5.0), 200: (1.0, float("nan"))})
    assert res["iteration_to_target"] is None and res["minutes_to_target"] is None


def test_stop_rule(tmp_path):
    """stop_rule.py: the first evaluation where the three-evaluation means of the object error improved by less than 3 %
    over 1,000 iterations and the success holds (0.99) or stopped improving; a run's checkpoints read in order until it
    holds, its last one otherwise."""
    path = os.path.join(REPO, "benchmarks", "robomimic_manipulation_tracking", "stop_rule.py")
    spec = importlib.util.spec_from_file_location("robomimic_stop_rule", path)
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    falling = {i: (1.0, max(20 - i / 250, 8.0)) for i in range(250, 6001, 250)}  # 8 mm from 3,000 on
    assert rule.stop(falling) == 4500  # E(4500) = E(3500) = 8 mm; at 4,000 E(3000) is 9 mm
    learning = {i: (min(1.0, 0.3 + i / 5000), 30.0) for i in range(250, 6001, 250)}  # the error flat, the success rising
    assert rule.stop(learning) == 4000  # S(4000) = 1.0: the success holds
    noisy = {i: (0.5 + (0.08 if (i // 250) % 2 else -0.08), 10.0) for i in range(250, 3001, 250)}  # flat around 0.5
    assert rule.stop(noisy) == 1750
    ended = dict(falling)
    ended[3000] = (0.0, math.nan)  # every rollout terminated: no error, no stop in the windows that hold it
    assert rule.stop(ended) == 4750 and rule.stop({i: (1.0, 30.0 - i / 100) for i in range(250, 6001, 250)}) is None
    run = tmp_path / "run"
    run.mkdir()
    for it in range(250, 6001, 250):
        (run / f"ckpt_{it:06d}.pt").write_bytes(b"")
    seen = []

    def evaluate(ckpt, out):
        seen.append(int(os.path.basename(ckpt)[5:11]))
        s, e = falling[seen[-1]]
        return {"policy": {"success_rate": s, "object_mm": e}}

    res = rule.stop_of(str(run), evaluate)
    assert res["stop_iteration"] == 4500 and res["by_rule"] and seen == list(range(250, 4501, 250)) and res["stop_ckpt"].endswith("ckpt_004500.pt")
    assert json.loads((run / "stop.json").read_text())["stop_iteration"] == 4500


def test_motion_measures(tmp_path, monkeypatch):
    """motion.py: the joints' acceleration, jerk, high-frequency share and limit hits of sampled sines against their closed forms;
    the slip over the pairs of held steps; in a scene, a grasp is both fingers of a gripper touching a tracked object, and the
    slip is the object's displacement in the end effector's frame; a rollout is read up to its termination, beside the
    demonstration's same frames."""
    path = os.path.join(REPO, "benchmarks", "robomimic_manipulation_tracking", "motion.py")
    spec = importlib.util.spec_from_file_location("robomimic_motion", path)
    motion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motion)
    dt, n = 0.05, 201  # 200 velocity samples over 10 s: 1 Hz and 8 Hz fall on the spectrum's bins
    t = dt * np.arange(n)
    q = np.stack((np.sin(2 * np.pi * t), np.sin(2 * np.pi * t) + 0.05 * np.sin(2 * np.pi * 8 * t)), 1)
    lo, hi = np.array([-2.0, -1.0]), np.array([2.0, 3.0])
    m = motion.joint_measures(q, dt, lo, hi)
    gain = lambda f: 2 * np.sin(np.pi * f * dt) / dt  # noqa: E731: a difference's gain on a sine of f Hz
    share = (0.05 * gain(8)) ** 2 / (gain(1) ** 2 + (0.05 * gain(8)) ** 2)  # the velocity's power above 5 Hz, second joint
    assert m["hf_share"] == pytest.approx(share / 2, abs=1e-9) and 0.08 < share < 0.09
    assert m["jerk"] == pytest.approx((gain(1) ** 3 + np.hypot(gain(1) ** 3, 0.05 * gain(8) ** 3)) / 2 / np.sqrt(2), rel=0.02)
    assert m["acc"] == pytest.approx((gain(1) ** 2 + np.hypot(gain(1) ** 2, 0.05 * gain(8) ** 2)) / 2 / np.sqrt(2), rel=0.02)
    assert m["limit_hits"] == pytest.approx((q[:, 1] < -0.96).mean() / 2) and m["limit_hits"] > 0  # within 1 % of the second joint's range
    assert motion.joint_measures(np.zeros((20, 2)), dt, lo, hi)["hf_share"] == 0.0  # a joint at rest has no share

    hold = np.array([0, 1, 1, 1, 0, 1, 1], bool).reshape(-1, 1, 1)
    rel = np.zeros((7, 1, 1, 3))
    rel[:, 0, 0, 0] = 0.001 * np.arange(7) ** 2  # 1, 3, 5, ... mm a step
    assert motion.slip(hold, rel) == (pytest.approx((3 + 5 + 11) / 3), 3)  # the pairs (1, 2), (2, 3) and (5, 6)
    assert motion.slip(np.zeros_like(hold), rel) == (None, 0)

    model = scene_spec().compile()
    sc = motion.Scene(model, "lift")
    assert len(sc.arm_q) == 7 and len(sc.fin_q) == 2 and len(sc.eef) == 1 and len(sc.objects) == 1
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    left, right = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gripper0_right_{k}finger") for k in ("left", "right"))
    assert sc.finger[left] == 0 and sc.finger[right] == 1 and sc.object[sc.objects[0]] == 0 and (sc.finger >= 0).sum() == 2
    grip = 0.5 * (d.xpos[left] + d.xpos[right])  # between the fingers' boxes, whose inner faces are 1 cm from it at rest
    cube = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint0")]
    closed, opened = d.qpos.copy(), d.qpos.copy()
    closed[cube : cube + 3], opened[cube : cube + 3] = grip, grip
    closed[sc.fin_q], opened[sc.fin_q] = (0.009, -0.009), (0.04, -0.04)  # on the 4 cm cube with 1 mm to spare; open
    one = closed.copy()
    one[sc.fin_q[1]] = -0.04  # one finger alone
    lifted = closed.copy()
    lifted[cube + 2] += 0.002
    hold, rel = sc.grasps(np.stack((closed, lifted, opened, one, closed)))
    assert hold[:, 0, 0].tolist() == [True, True, False, False, True]
    assert rel[0, 0, 0] == pytest.approx(d.xmat[sc.eef[0]].reshape(3, 3).T @ (grip - d.xpos[sc.eef[0]]), abs=1e-9)
    assert motion.slip(hold, rel) == (pytest.approx(2.0), 1)  # the cube 2 mm up in the gripper between the two held steps

    # a rollout file: read up to its termination, the demonstration over the same frames and over all of them
    monkeypatch.setattr(motion, "scene_of", lambda data_dir, task, demo: motion.Scene(model, task))
    T = 40
    ref = np.repeat(closed[None], T, 0)
    ref[:, sc.arm_q[0]] = 0.3 * np.sin(2 * np.pi * 0.5 * dt * np.arange(T))
    state = np.concatenate((ref, np.zeros((T, model.nv))), 1)
    state[:, sc.arm_q[0]] += 0.01 * (-1.0) ** np.arange(T)  # a 10 Hz tremor on the first joint (the gripper spins about the cube)
    state[:, cube + 2] += 0.001 * (-1.0) ** np.arange(T)  # and the held cube 1 mm up and down
    for name, term in (("whole", -1), ("ended", 19), ("short", 5)):
        np.savez(tmp_path / f"{name}.npz", state=state, reference_qpos=ref, reference_targets_state=state, nq=model.nq, control_dt=dt,
                 task="lift", demo=0, terminated_at=term, success=term < 0)  # fmt: skip
    scenes = {}
    rows = [motion.measure(name, str(tmp_path / f"{name}.npz"), "unused", scenes) for name in ("whole", "ended", "short")]
    assert [r["frames"] for r in rows] == [40, 20, 6] and len(scenes) == 1
    whole, ended, short = rows
    p, d = whole["policy"], whole["demonstration"]
    assert 0.03 < p["hf_share"] < 0.05 and d["hf_share"] < 0.002 and p["jerk"] > 50 * d["jerk"]  # one of the seven joints trembles
    assert d == whole["demonstration_whole"] == ended["demonstration_whole"] and ended["demonstration"] != ended["demonstration_whole"]
    assert p["held_pairs"] == 39 and ended["policy"]["held_pairs"] == 19
    assert p["slip_mm"] == pytest.approx(2.0) and d["slip_mm"] == pytest.approx(0.0, abs=1e-9)
    assert short["policy"] is None and short["demonstration"] is None  # fewer than 8 frames
    text = motion.table(rows).splitlines()
    assert len(text) == 6 and text[2].startswith("| lift | whole | 40 |") and text[3].startswith("| lift | ended | 20 of 40 |")
    assert text[4].startswith("| lift | short | 6 of 40 | - | - |") and text[5].startswith("| lift | the demonstration | 40 |")
    assert "target change" not in text[0] and p["target_change"] is None  # no saved targets: no such column
    # a rollout saved with --save_targets: the targets' change per step, the reference's beside it
    ref_tg = ref[1:][:, sc.act_q]
    tg = ref_tg.copy()
    tg[:, 0] += 0.01 * (-1.0) ** np.arange(T - 1)  # the first arm joint's target 0.02 rad up and down, step by step
    np.savez(tmp_path / "targets.npz", state=state, reference_qpos=ref, reference_targets_state=state, nq=model.nq, control_dt=dt, task="lift",
             demo=0, terminated_at=-1, success=True, targets=tg, reference_targets=ref_tg)  # fmt: skip
    row = motion.measure("saved", str(tmp_path / "targets.npz"), "unused", scenes)
    own = np.sqrt((np.diff(ref_tg[:, sc.arm], axis=0) ** 2).mean(0))  # the reference's own change, per arm joint
    assert row["demonstration"]["target_change"] == pytest.approx(own.mean()) == row["demonstration_whole"]["target_change"]
    assert own[1:].max() == 0 and row["policy"]["target_change"] == pytest.approx((np.hypot(0.02, own[0]) + 0) / 7, rel=0.02)
    text = motion.table([row]).splitlines()
    assert text[0].endswith("| target change (rad per step) |") and text[2].split("|")[-2].strip() == f"{row['policy']['target_change']:.4f}"


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


@GPU
@pytest.mark.parametrize("name", ["lift", "square", "transport"])
def test_success_inside_a_cuda_graph(name):
    """RobomimicSim.success reads nothing on the host, so the collection's CUDA graph (--collect_graph) holds it: captured
    and replayed, it gives what it gives outside the graph."""
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim

    model, _, _, task = task_of(name, device=CUDA)
    sim = RobomimicSim(model, task, 4, CUDA, joint_noise=0.0)
    sim.step(task.setpoint(torch.zeros(4, task.nu, device=CUDA), sim.f), reset=False)
    want = sim.success().clone()
    out = torch.zeros(4, dtype=torch.bool, device=CUDA)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out.copy_(sim.success())  # (the first call outside the capture)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(sim.success())
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)


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
