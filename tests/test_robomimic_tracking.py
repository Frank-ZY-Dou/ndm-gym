"""Manipulation demonstration tracking on a small scene in robosuite's naming (a seven-joint arm with its torque actuators, a
two-finger gripper, a free cube), its demonstration made by MuJoCo itself: the servos and the gravity compensation, the scene's
layout for the task, the demonstration's file and assets, the forward kinematics against MuJoCo's, the task at the reference,
the world model's control step, the updates and training's arguments; on a GPU, the simulator against MuJoCo."""

import argparse
import importlib.util
import json
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.hand_fast import add_hand_fast_args
from ndm_gym.algos.robomimic_algos import ROBOMIMIC_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.motion.sources.robomimic import load_demo, split_states
from ndm_gym.robots.robomimic_scenes import ARM_KP, LIFT_HEIGHT, LIFT_TABLE_TOP, RobomimicScene, build_spec, resolve_assets
from ndm_gym.tasks.robomimic_tracking import CONTROL_DT, HOLD, RewardCfg, RobomimicTask, SceneKinematics, build_reference, unflatten_state
from ndm_gym.train import add_train_args
from ndm_gym.train_hand import UNSUPPORTED
from ndm_gym.train_robomimic import ROBOMIMIC_DEFAULTS, UNSUPPORTED_SYSTEMS, parse_args
from ndm_gym.worldmodels.mjcf import MjcfLayout

# the arguments robomimic training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip
# hand training's systems options that robomimic training implements
SYSTEMS = ("fused_adam", "reset_masked", "no_sync", "collect_graph", "compile_eager", "update_graphs", "pi_opt", "wm_opt", "pi_ema", "pi_lr_mult",
           "lr_decay_ret", "lr_decay_len", "lr_decay_by", "muon_batched")  # fmt: skip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSTEPS = round(CONTROL_DT / 0.002)


def scene_xml(texture: bool = False) -> str:
    """A Panda-like scene in robosuite's names: seven hinges (with the Panda's armature) driven by torque actuators, a gripper's two finger slides on
    position servos, a free cube on a table; ``texture``: one in robosuite's assets' path, as recorded scenes name them."""
    links, axes = "", ("0 0 1", "0 1 0", "0 0 1", "0 -1 0", "0 0 1", "0 -1 0", "0 0 1")
    for k, axis in enumerate(axes, start=1):
        links += (f'<body name="robot0_link{k}" pos="0 0 {0.0 if k == 1 else 0.12}"><joint name="robot0_joint{k}" axis="{axis}" range="-2.8 2.8" armature="1" damping="0.1"/>'
                  f'<geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.03" mass="1"/>')  # fmt: skip
    gripper = ('<body name="robot0_right_hand" pos="0 0 0.12"><body name="gripper0_right_right_gripper"><body name="gripper0_right_eef" pos="0 0 0.097"/>'
               '<body name="gripper0_right_leftfinger" pos="0 0 0.05"><joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>'
               '<geom type="box" pos="0 0.015 0" size="0.01 0.005 0.02" mass="0.05"/></body>'
               '<body name="gripper0_right_rightfinger" pos="0 0 0.05"><joint name="gripper0_right_finger_joint2" type="slide" axis="0 1 0" range="-0.04 0"/>'
               '<geom type="box" pos="0 -0.015 0" size="0.01 0.005 0.02" mass="0.05"/></body></body></body>')  # fmt: skip
    torques = "".join(f'<general name="robot0_torq_j{k}" joint="robot0_joint{k}" ctrlrange="{-r} {r}"/>' for k, r in
                      zip(range(1, 8), (80, 80, 80, 80, 80, 12, 12), strict=True))  # fmt: skip
    asset = '<asset><texture name="t" type="2d" file="/home/someone/robosuite/models/assets/textures/wood.png"/></asset>' if texture else ""
    return f"""<mujoco><compiler angle="radian"/><option timestep="0.002"/>{asset}
      <worldbody><geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="table" pos="0 0 0.775"><geom type="box" size="0.4 0.4 0.025"/></body>
        <body name="robot0_base" pos="-0.5 0 0.8">{links}{gripper}{"</body>" * 7}</body>
        <body name="cube_main" pos="0 0 0.82"><freejoint name="cube_joint0"/><geom type="box" size="0.02 0.02 0.02" mass="0.07"/></body>
      </worldbody>
      <actuator>{torques}
        <position name="gripper0_right_gripper_finger_joint1" joint="gripper0_right_finger_joint1" kp="1000" ctrlrange="0 0.04"/>
        <position name="gripper0_right_gripper_finger_joint2" joint="gripper0_right_finger_joint2" kp="1000" ctrlrange="-0.04 0"/>
      </actuator></mujoco>"""


def scene_spec(**kw):
    return build_spec(scene_xml(), "/unused", **kw)


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
    the gripper keeps its servos; the scene's actuators, keypoints and objects as the task reads them."""
    raw = mujoco.MjModel.from_xml_string(scene_xml())
    model = scene_spec().compile()
    for k in range(7):
        assert model.actuator_gainprm[k, 0] == ARM_KP[k] and model.actuator_biasprm[k, 1] == -ARM_KP[k] and model.actuator_biasprm[k, 2] < 0
        assert model.actuator_forcelimited[k] and np.allclose(model.actuator_forcerange[k], raw.actuator_ctrlrange[k]) and not model.actuator_ctrllimited[k]
    assert np.allclose(model.actuator_gainprm[7:9, 0], raw.actuator_gainprm[7:9, 0])
    b = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    assert model.body_gravcomp[b("robot0_link3")] == 1 and model.body_gravcomp[b("gripper0_right_leftfinger")] == 1 and model.body_gravcomp[b("cube_main")] == 0
    assert scene_spec(gravcomp=False).compile().body_gravcomp.max() == 0
    scene = RobomimicScene.from_model(model, "lift")
    assert scene.arm == (True,) * 7 + (False,) * 2 and scene.num_arms == 1 and scene.act_q == tuple(range(9))
    assert scene.keypoint_bodies == (b("gripper0_right_eef"), b("gripper0_right_leftfinger"), b("gripper0_right_rightfinger"))
    assert scene.objects == ((9, 9),) and scene.others == ()
    with pytest.raises(ValueError, match="SquareNut_joint0"):
        RobomimicScene.from_model(model, "square")
    with pytest.raises(ValueError, match="torque actuators"):
        build_spec(scene_xml().replace("robot0_torq_j7", "robot0_motor_j7"), "/unused")
    assert resolve_assets('<texture file="/x/y/robosuite/models/assets/textures/a.png"/>', "/data/assets") == '<texture file="/data/assets/textures/a.png"/>'


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
    moved[:, 9 + 2] = LIFT_TABLE_TOP + LIFT_HEIGHT + 0.2  # the cube lifted: robosuite's success; and far from its reference: a termination
    assert task.success((moved, ref.qvel)).all() and not task.success(S).any()
    assert task.termination((moved, ref.qvel), task.fk(moved), f).all()


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


def test_systems_options_implemented_or_refused(capsys):
    parser = argparse.ArgumentParser()
    add_hand_fast_args(parser)
    assert {a.dest for a in parser._actions} - {"help"} == set(SYSTEMS) | set(UNSUPPORTED_SYSTEMS)
    for name in UNSUPPORTED_SYSTEMS:  # each is a flag
        with pytest.raises(SystemExit):
            parse_args(["--task", "lift", f"--{name}"])
        assert f"--{name} is an option of hand training" in capsys.readouterr().err


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
