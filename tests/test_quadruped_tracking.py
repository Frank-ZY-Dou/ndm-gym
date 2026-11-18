"""Quadruped motion tracking on the Go2 (MuJoCo Menagerie's model, data/menagerie) and the retargeted dog clips
(data/lifelike_go2, tools/convert_lifelike.py): the scene's position servos, the forward kinematics against MuJoCo's,
the clips' loading and resampling, the reference's velocities in MuJoCo's convention and its clip boundaries, the task
at the reference and its invariance to the heading and the horizontal position, the world model's state layout, one
SuperMimic and one SuperTrack update on the CPU and training's arguments."""

import argparse
import importlib.util
import math
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos import hand_fast
from ndm_gym.algos.common import set_lr
from ndm_gym.algos.quadruped_algos import QUADRUPED_ALGOS, WINDOW_KEYS, QuadrupedCollector, replay_fields, world_model
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.muon import MuonAdam
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.evaluate_quadruped import length_buckets, max_step_reward, minutes_to_success, rollout, summarize
from ndm_gym.motion.sources.lifelike import DATA_DIR, clip_paths, load_clip
from ndm_gym.robots.go2 import FEET, JOINT_NAMES, KP, NQ, NUM_J, NUM_KP, SCENE_PATH, Go2Kinematics, load_scene
from ndm_gym.state import CONTROL_DT
from ndm_gym.tasks.quadruped_tracking import GOAL_FRAME_DIM, OBS_DIM, STATE_DIM, QuadrupedTask, build_reference, flatten_state, unflatten_state
from ndm_gym.train import add_train_args
from ndm_gym.train_quadruped import QUADRUPED_DEFAULTS, UNSUPPORTED, parse_args
from ndm_gym.worldmodels.mjcf import differentiate_pos

# the arguments quadruped training implements; the rest of ndm_gym.train's are train_quadruped.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w", "clips")  # fmt: skip
CLIPS = ["dog_quad_walk_002_1", "dog_zig_walk_001_1"]
pytestmark = pytest.mark.skipif(not (os.path.exists(SCENE_PATH) and all(os.path.exists(os.path.join(DATA_DIR, c + ".npz")) for c in CLIPS)),
                                reason="needs data/menagerie/unitree_go2 and the converted clips (tools/convert_lifelike.py)")  # fmt: skip


@pytest.fixture(scope="module")
def setup():
    model = load_scene()
    kin = Go2Kinematics(model, "cpu")
    clips = [load_clip(p, CONTROL_DT) for p in clip_paths(CLIPS)]
    for c in clips:  # short clips keep the tests fast
        c["qpos"], c["contacts"] = c["qpos"][:40], c["contacts"][:40]
    return model, kin, QuadrupedTask(kin, build_reference(clips, kin, "cpu"))


def test_scene_has_position_servos():
    model = load_scene()
    assert model.nu == NUM_J and float(model.geom_margin.max()) == 0 and model.opt.timestep == 0.002
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    data.ctrl[:] = data.qpos[7:] + 0.01
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, KP * 0.01, rtol=1e-6)  # at rest: kp times the offset
    data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(model, data)
    assert np.all(np.abs(data.actuator_force) <= model.actuator_forcerange[:, 1] + 1e-9)  # the motors' torque limits
    assert [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)] == JOINT_NAMES


def test_kinematics_match_mujoco(setup):
    model, kin, task = setup
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FEET]
    qpos = task.ref.qpos[::7].clone()
    qpos[:, 7:] = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(qpos.shape[0], NUM_J)  # any pose, not only the reference's
    kp, R = kin.forward(qpos)
    assert kp.shape == (qpos.shape[0], NUM_KP, 3)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i].numpy()
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(kp[i, 0].numpy(), data.xpos[1], atol=1e-5)
        np.testing.assert_allclose(kp[i, 1:].numpy(), data.geom_xpos[feet], atol=1e-5)
        np.testing.assert_allclose(R[i].numpy(), data.xmat[1].reshape(3, 3), atol=1e-5)


def test_clips_load_and_resample():
    assert clip_paths("dog_quad_walk_002_1") == clip_paths(os.path.join(DATA_DIR, "dog_quad_walk_002_1.npz"))
    assert len(clip_paths("dog_zig_walk_*")) >= 2 and len(clip_paths(None)) == len(clip_paths(DATA_DIR))
    with pytest.raises(FileNotFoundError):
        clip_paths("no_such_clip")
    with np.load(clip_paths(CLIPS[0])[0]) as d:
        raw = d["qpos"]
    clip = load_clip(clip_paths(CLIPS[0])[0], CONTROL_DT)
    q = clip["qpos"]
    assert q.shape == raw.shape and clip["contacts"].shape == (raw.shape[0], 4)  # the clips are at the control rate
    assert torch.allclose(q[:, 7:], torch.tensor(raw[:, 7:])) and torch.allclose(q[:, 3:7].norm(dim=-1), torch.ones(q.shape[0]))
    assert float((q[1:, 3:7] * q[:-1, 3:7]).sum(-1).min()) > 0  # the quaternion's sign is continuous
    half = load_clip(clip_paths(CLIPS[0])[0], 2 * CONTROL_DT)["qpos"]
    assert half.shape[0] == (raw.shape[0] - 1) // 2 + 1 and torch.allclose(half[:, 7:], q[::2, 7:], atol=1e-6)


def test_reference_velocities_and_clips(setup):
    model, kin, task = setup
    ref = task.ref
    assert ref.num_clips == 2 and ref.frames == 80 and ref.qpos.shape == (80, NQ)
    assert ref.last.tolist() == [39] * 40 + [79] * 40 and ref.clip.tolist() == [0] * 40 + [1] * 40
    f = torch.tensor([0, 38, 39, 40, 79])
    assert ref.next_frame(f).tolist() == [1, 39, 39, 41, 79] and ref.ahead(f, 3).tolist() == [3, 39, 39, 43, 79]
    # central differences: the velocity at f is the tangent from f - 1 to f + 1 over two control steps, in MuJoCo's convention
    from ndm_gym.worldmodels.mjcf import MjcfLayout

    lay = MjcfLayout.from_model(model)
    mid = torch.arange(1, 39)
    tangent = differentiate_pos(lay, ref.qpos[mid - 1].double(), ref.qpos[mid + 1].double()) / (2 * CONTROL_DT)
    assert torch.allclose(ref.qvel[mid, :3].double(), tangent[:, :3], atol=1e-4)
    assert torch.allclose(ref.qvel[mid, 6:].double(), tangent[:, 6:], atol=1e-4)
    # the angular velocity in the trunk's frame: the tangent's, up to the difference of turning about f - 1's frame and f's
    assert float((ref.qvel[mid, 3:6].double() - tangent[:, 3:6]).abs().max()) < 0.05 * float(tangent[:, 3:6].abs().max()) + 1e-3


def test_task_at_the_reference(setup):
    _, kin, task = setup
    ref = task.ref
    f = torch.arange(ref.frames)
    S = ref.state(f)
    fk, a = kin.forward(S[0]), torch.zeros(ref.frames, NUM_J)
    assert task.obs(S, fk, a).shape == (ref.frames, OBS_DIM) and task.goal(S, fk, f, 3).shape == (ref.frames, 3 * GOAL_FRAME_DIM)
    assert torch.allclose(task.reward(S, fk, f, a, a), torch.full((ref.frames,), task.max_step_reward), atol=1e-5)
    assert not task.termination(S, fk, f).any() and float(task.soft_termination(S, fk, f).max()) < 1e-3
    assert torch.allclose(task.setpoint(a, f), ref.qpos[ref.next_frame(f), 7:])
    for shift in ((0.0, 0.0, -0.2), (0.7, 0.0, 0.0)):  # fallen, or far from the reference
        far = S[0].clone()
        far[:, :3] += torch.tensor(shift)
        fk = kin.forward(far)
        assert task.termination((far, S[1]), fk, f).all() and float(task.soft_termination((far, S[1]), fk, f).min()) > 0.99


def test_observation_is_heading_invariant(setup):
    _, kin, task = setup
    f = torch.arange(0, task.ref.frames, 5)
    qpos, qvel = (x.clone() for x in task.ref.state(f))
    a = torch.rand(f.shape[0], NUM_J)
    yaw = 1.1
    q_yaw = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    Rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    moved, mvel = qpos.clone(), qvel.clone()
    moved[:, :3] = qpos[:, :3] @ Rz.T + torch.tensor([1.0, -2.0, 0.0])
    w, v = q_yaw[0], q_yaw[1:]
    qv, qw = qpos[:, 4:7], qpos[:, 3:4]  # q_yaw * q
    moved[:, 3:4] = w * qw - (v * qv).sum(-1, keepdim=True)
    moved[:, 4:7] = w * qv + qw * v + torch.linalg.cross(v.expand_as(qv), qv)
    mvel[:, :3] = qvel[:, :3] @ Rz.T  # the world linear velocity turns; the body-frame angular velocity does not
    S, M = (qpos, qvel), (moved, mvel)
    assert torch.allclose(task.obs(S, kin.forward(qpos), a), task.obs(M, kin.forward(moved), a), atol=1e-4)
    wm = world_model(load_scene(), "accel", hidden=(16,))
    u = qpos[:, 7:] + 0.1
    assert torch.allclose(wm.inputs(qpos, qvel, u), wm.inputs(moved, mvel, u), atol=1e-4)


def test_state_layout_and_world_model(setup):
    model, _, task = setup
    f = torch.arange(task.ref.frames - 1)
    S, S1 = task.ref.state(f), task.ref.state(task.ref.next_frame(f))
    x = flatten_state(S)
    assert x.shape == (len(f), STATE_DIM) and all(torch.equal(a, b) for a, b in zip(unflatten_state(x), S, strict=True))
    for kind in ("accel", "increment"):
        wm = world_model(model, kind, hidden=(32,))
        assert wm.lay.dt == CONTROL_DT and wm.lay.anchor == (0, 0) and wm.lay.nu == NUM_J
        wm.update_norms(*S, S[0][:, 7:], *S1)
        assert float(wm.error(*S1, *S1).abs().max()) < 1e-4
        q1, v1 = wm.step(*S, S[0][:, 7:])
        assert q1.shape == S[0].shape and v1.shape == S[1].shape and torch.isfinite(q1).all()


def small(extra: list) -> list:
    argv = ["--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch", "16", "--wm_window", "4",
            "--pi_batch", "16"] + extra  # fmt: skip
    return argv + ([] if "supertrack" in extra else ["--critic_epochs", "1", "--critic_minibatches", "2"])


@pytest.mark.parametrize("extra", [[], ["--algo", "supertrack", "--st_fail_w", "2"], ["--algo", "supertrack", "--st_task_weights"]],
                         ids=["supermimic", "supertrack", "supertrack_task_weights"])  # fmt: skip
def test_updates(setup, monkeypatch, extra):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    torch.manual_seed(0)
    model, kin, task = setup
    args = parse_args(small(extra))
    algo = QUADRUPED_ALGOS[args.algo](args, model, kin, task, "cpu")
    buffer = ReplayBuffer(16, 8, "cpu", fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.randint(0, task.ref.frames - 1, (8,))
    S, a_prev = task.ref.state(f), torch.zeros(8, NUM_J)
    for _ in range(16):
        a = algo.act(S, a_prev, f, noise=0.1)
        u, f1 = task.setpoint(a, f), task.ref.next_frame(f)
        S1 = task.ref.state(f1)
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": flatten_state(S), "s1": flatten_state(S1), "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(8, dtype=torch.bool)})
        S, f, a_prev = S1, f1, a
    before = [p.clone() for p in algo.pi.parameters()]
    assert torch.isfinite(algo.wm_update(buffer))
    out = algo.policy_update(buffer)
    assert all(torch.isfinite(x) for x in out[:1] + out[2:]) and 0 < float(out[2]) <= 1 and 0 < float(out[4]) <= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    assert set(algo.state_dict()) >= {"wm", "pi", "opt_wm", "opt_pi"}


def test_training_arguments():
    """Every argument of ndm_gym.train is implemented or refused; the defaults are QUADRUPED_DEFAULTS over LAFAN1's."""
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    assert set(QUADRUPED_DEFAULTS) <= set(SUPPORTED)
    args = parse_args([])
    assert all(getattr(args, name) == value for name, value in QUADRUPED_DEFAULTS.items()) and args.clips is None
    for refused in (["--algo", "blinkmimic"], ["--wm_backbone", "rigidformer"], ["--hist", "4"], ["--st_fail_w", "2"], ["--st_task_weights"],
                    ["--algo", "supertrack", "--gamma", "0.995"], ["--algo", "supertrack", "--bptt_decay", "0.95"], ["--motion_dir", "x"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(refused)
    st = parse_args(["--algo", "supertrack", "--st_fail_w", "2", "--st_task_weights", "--clips", "a,b"])
    assert (st.algo, st.st_fail_w, st.st_task_weights, st.clips) == ("supertrack", 2, True, "a,b")


# ------------------------------------------------- the systems options (the hand-object task's, algos.hand_fast)
SYSTEMS = ["--fused_adam", "--reset_masked", "--no_sync"]
ST_ARM = ["--algo", "supertrack", "--st_fail_w", "2", "--st_imag_noise", "0.2", "--st_task_weights"]
CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")


def test_systems_arguments():
    """The systems options are off by default; the combinations that cannot run are refused; a resumed run keeps its
    checkpoint's algorithm, optimizers and policy average."""
    args = parse_args([])
    assert not (args.fused_adam or args.reset_masked or args.no_sync or args.collect_graph or args.update_graphs or args.compile_eager)
    assert (args.pi_opt, args.wm_opt, args.pi_ema, args.pi_lr_mult, args.lr_decay_ret, args.lr_decay_by) == ("adam", "adam", 0.0, 1.0, 0.0, 0)
    assert not args.keep_ckpts
    for refused in (["--no_sync"], ["--no_sync", "--fused_adam"], ["--no_sync", "--reset_masked"], ["--collect_graph"], ["--compile_eager"],
                    ["--update_graphs", "--fused_adam", "--reset_masked"], ["--pi_ema", "1"], ["--lr_decay_ret", "0.5"],
                    ["--pi_lr_mult", "2", "--lr_decay_by", "100"], ["--pi_opt", "sgd"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(refused)
    on = parse_args([*ST_ARM, *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_opt", "muon", "--wm_opt", "muon", "--pi_ema",
                     "0.995", "--pi_lr_mult", "2", "--lr_decay_ret", "0.95", "--lr_decay_by", "1400", "--keep_ckpts"])  # fmt: skip
    assert on.no_sync and on.collect_graph and on.compile_eager and on.update_graphs and (on.pi_opt, on.wm_opt, on.pi_ema) == ("muon", "muon", 0.995)
    hand_fast.check_resume({"algo": "supermimic"}, args)  # a checkpoint from before the options, resumed without them
    hand_fast.check_resume(vars(on), on)
    for k in ("algo", "st_task_weights", "fused_adam", "update_graphs", "pi_opt", "wm_opt", "pi_ema"):  # each option's other value in the checkpoint
        other = (not getattr(on, k)) if isinstance(getattr(on, k), bool) else getattr(args, k)
        with pytest.raises(ValueError, match=k):
            hand_fast.check_resume({**vars(on), k: other}, on)


def test_agents_take_the_systems_options(setup, monkeypatch):
    """Without the options an agent's optimizers are Adam's default kernel and its updates read the host; --no_sync and
    Muon reach the agent, whose updates stay finite with the policy's step gated on the device."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    model, kin, task = setup
    plain = QUADRUPED_ALGOS["supermimic"](parse_args(small([])), model, kin, task, "cpu")
    assert not plain.no_sync and all(type(o) is torch.optim.Adam and not o.defaults.get("fused") for o in plain.optimizers)
    assert plain.fk == kin.forward
    for extra in ([], ST_ARM):
        monkeypatch.setattr(RunningNorm, "no_sync", True)
        args, algo, buffer = agent_and_replay(model, kin, task, extra + SYSTEMS + ["--pi_opt", "muon", "--wm_opt", "muon"])
        assert algo.no_sync and isinstance(algo.opt_pi, MuonAdam) and isinstance(algo.opt_wm, MuonAdam)
        before = [p.detach().clone() for p in algo.pi.parameters()]
        assert torch.isfinite(algo.wm_update(buffer))
        out = algo.policy_update(buffer)
        assert all(bool(torch.isfinite(x)) for x in out[:1] + out[2:])
        assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
        other = QUADRUPED_ALGOS[args.algo](args, model, kin, task, "cpu")
        other.load_state_dict(algo.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(trained(algo), trained(other), strict=True))


def test_minutes_to_success():
    """The training time to a success: the first checkpoint from which every later one reaches it."""
    rows = [{"wall_min": float(t), "success_rate": r} for t, r in enumerate([0.0, 1.0, 0.5, 0.99, 1.0])]
    assert minutes_to_success(rows, 0.99) == 3.0 and minutes_to_success(rows, 0.4) == 1.0
    assert minutes_to_success(rows[:3], 0.99) is None and minutes_to_success([], 0.99) is None
    for r, mm in zip(rows, [5.0, 30.0, 5.0, 30.0, 5.0], strict=True):  # with the trunk's threshold: both asked of every later checkpoint
        r["trunk_mm"] = mm
    assert minutes_to_success(rows, 0.99, 10.0) == 4.0 and minutes_to_success(rows, 0.99, 50.0) == 3.0


def test_evaluation_summary(setup):
    """evaluate_quadruped.summarize: at the reference's reward every step the normalized return is 1 and no clip fails; a
    rollout terminated halfway returns half and its later steps count for nothing; the failed clips are counted as
    ndm_gym.evaluate's failed motions (one rollout per clip) and listed below ``clip_success``."""
    task = setup[2]
    T, C = 39, 2  # two 40-frame clips, two rollouts each: world i on clip i % 2
    zeros = torch.zeros(T, 2 * C)
    errs = {"trunk_pos": zeros + 0.01, "trunk_ori_sq": zeros, "feet_world": torch.zeros(T, 2 * C, 4), "feet": torch.zeros(T, 2 * C, 4),
            "joint": zeros + math.pi / 180}  # fmt: skip
    rew = zeros + max_step_reward(task.cfg)
    m = summarize(task, errs, torch.full((2 * C,), -1), CLIPS, rew)
    assert m["norm_return"] == pytest.approx(1.0) and m["norm_return_framew"] == pytest.approx(1.0)
    assert m["failed_clips"] == 0 and m["n_failed"] == 0 and m["failed"] == [] and m["success_rate"] == 1.0
    assert m["trunk_mm"] == pytest.approx(10.0) and m["joint_deg"] == pytest.approx(1.0) and m["joint_rad"] == pytest.approx(math.pi / 180)
    term_at = torch.tensor([-1, -1, 13, -1])  # the first clip's second rollout ends after 13 of its 39 steps
    m = summarize(task, errs, term_at, CLIPS, rew * 5)  # the reward after the termination is not counted, whatever it is
    first, second = m["clips"][CLIPS[0]], m["clips"][CLIPS[1]]
    assert first["norm_return"] == pytest.approx(5 * (1 + 13 / 39) / 2) and second["norm_return"] == pytest.approx(5.0)
    assert m["norm_return"] == pytest.approx((first["norm_return"] + second["norm_return"]) / 2)
    assert m["norm_return_framew"] == pytest.approx(5 * (3 * 39 + 13) / (4 * 39))
    assert first["success_rate"] == 0.5 and not first["success"] and second["success"]
    assert m["failed_clips"] == pytest.approx(0.5) and m["failed"] == [CLIPS[0]] and m["n_failed"] == 1
    assert summarize(task, errs, term_at, CLIPS, rew, clip_success=0.5)["failed"] == []
    assert "norm_return" not in summarize(task, errs, term_at, CLIPS)  # without the rewards: the errors and the failures only
    w = task.cfg
    assert max_step_reward(w) == pytest.approx((w.w_feet + w.w_joint + w.w_trunk_pos + w.w_trunk_ori + w.w_lin_vel + w.w_ang_vel) * CONTROL_DT)


def agent_and_replay(model, kin, task, argv, n=8, steps=16, dev="cpu"):
    """An agent with the arguments ``argv`` and a replay of ``steps`` transitions of ``n`` worlds along the reference, from seed 0."""
    torch.manual_seed(0)
    args = parse_args(small(argv))
    algo = QUADRUPED_ALGOS[args.algo](args, model, kin, task, dev)
    buffer = ReplayBuffer(steps, n, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.randint(0, task.ref.frames - 1, (n,), device=dev)
    S, a_prev, done = task.ref.state(f), torch.zeros(n, NUM_J, device=dev), torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(steps):
        a = algo.act(S, a_prev, f, noise=0.1)
        u, f1 = task.setpoint(a, f), task.ref.next_frame(f)
        S1 = task.ref.state(f1)
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": flatten_state(S), "s1": flatten_state(S1), "u": u, "a_prev": a_prev, "f": f, "done": done})
        S, f, a_prev = S1, f1, a
    return args, algo, buffer


def trained(algo) -> list[torch.Tensor]:
    """The agent's parameters and buffers (normalizers, SuperTrack's weights), detached."""
    mods = [m for m in (algo.wm, algo.pi, algo.critic, getattr(algo, "bal", None)) if m is not None]
    return [t.detach() for m in mods for t in (*m.parameters(), *m.buffers())]


# ------------------------------------------------- on a GPU: the simulator's steps and the CUDA graphs
@pytest.fixture(scope="module")
def gpu_setup():
    """``setup`` on the GPU: the Go2 scene, its kinematics and a task of two 40-frame clips."""
    model = load_scene()
    kin = Go2Kinematics(model, CUDA)
    clips = [load_clip(p, CONTROL_DT) for p in clip_paths(CLIPS)]
    for c in clips:
        c["qpos"], c["contacts"] = c["qpos"][:40], c["contacts"][:40]
    return model, kin, QuadrupedTask(kin, build_reference(clips, kin, CUDA))


def on_stream(stream, fn):
    """``fn()`` on ``stream`` between the caller's work before and after it (hand_fast.Collector's eager call)."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return out


@GPU
def test_step_without_sync_and_masked_reset(gpu_setup):
    """QuadrupedSim.step_nosync on Warp's stream is QuadrupedSim.step with the ended worlds reset by mask: from the same
    state and targets, the same next state, terminations and episode ends; the ended worlds restart at the reference's
    state of a start frame (resets without perturbation), the others are untouched. With the perturbation, the reset
    worlds' joints stay within its bounds and the joints' limits."""
    from ndm_gym.sim.rigid.quadruped_sim import QuadrupedSim

    model, kin, task = gpu_setup
    ref, n = task.ref, 64
    torch.manual_seed(0)
    a, b = (QuadrupedSim(model, task, n, CUDA, joint_noise=0.0, vel_noise=0.0) for _ in range(2))
    stream, ends, worst = b.torch_stream(), torch.zeros(2, device=CUDA), 0.0
    for k in range(60):
        if k % 5 == 0:  # some worlds lifted beyond the termination's height
            a.qpos[torch.randint(0, n, (4,), device=CUDA), 2] += 0.3
        for x, y in ((a.qpos, b.qpos), (a.qvel, b.qvel), (a.ctrl, b.ctrl), (a.warmstart, b.warmstart)):
            y.copy_(x)  # each step from the same state: two simulations drift apart by rounding
        b.f.copy_(a.f)
        u = task.setpoint(2 * torch.rand(n, NUM_J, device=CUDA) - 1, a.f)
        before = a.qpos.clone()
        (s1a, ta, da), (s1b, tb, db) = a.step(u), on_stream(stream, lambda: b.step_nosync(u))  # noqa: B023
        worst = max(worst, float((s1a - s1b).abs().max()))
        assert torch.equal(ta, tb) and torch.equal(da, db)
        assert torch.equal(a.f[~da], b.f[~da]) and torch.allclose(a.qpos[~da], b.qpos[~da], atol=1e-3)
        for sim in (a, b):  # the ended worlds: at the reference's state of a frame an episode may start at
            f = sim.f[da]
            assert bool(torch.isin(f, sim.starts).all()) and torch.equal(sim.qpos[da], ref.qpos[f]) and torch.equal(sim.qvel[da], ref.qvel[f])
            assert torch.equal(sim.ctrl[da], ref.qpos[f, 7:]) and bool((sim.warmstart[da] == 0).all())
        assert not torch.equal(a.qpos[~da], before[~da])
        ends += torch.stack((ta.sum(), (da & ~ta).sum()))
    assert worst < 1e-3, worst
    assert bool((ends > 0).all())  # terminations and episodes that reached their clip's last frame
    c = QuadrupedSim(model, task, n, CUDA)  # the perturbed resets
    kept, mask = (c.qpos.clone(), c.qvel.clone(), c.ctrl.clone(), c.f.clone()), torch.arange(n, device=CUDA) % 3 == 0
    c.reset_masked(mask)
    for x, y in zip((c.qpos, c.qvel, c.ctrl, c.f), kept, strict=True):
        assert torch.equal(x[~mask], y[~mask])
    q, f = c.qpos[mask], c.f[mask]
    assert bool(torch.isin(f, c.starts).all()) and torch.equal(q[:, :7], ref.qpos[f, :7]) and torch.equal(q[:, 7:], c.ctrl[mask])
    assert 0 < float((q[:, 7:] - ref.qpos[f, 7:]).abs().max()) < 6 * c.joint_noise and bool(((q[:, 7:] >= kin.q_lo) & (q[:, 7:] <= kin.q_hi)).all())
    assert 0 < float((c.qvel[mask] - ref.qvel[f]).abs().max()) < 6 * c.vel_noise


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_collect_graph_replays_the_eager_step(gpu_setup, monkeypatch, extra):
    """--collect_graph: the captured collection step, replayed, computes what the step run eagerly on Warp's stream does
    (--no_sync) when both draw the same random numbers: the same episode ends and frames, and the same replay, log sums,
    actions and normalizers up to the rounding by which two simulations differ; with --compile_eager's compiled forward
    kinematics too."""
    from ndm_gym.sim.rigid.quadruped_sim import QuadrupedSim

    monkeypatch.setattr(RunningNorm, "no_sync", True)
    model, kin, task = gpu_setup
    n, steps, runs = 64, 50, []
    for flags in ([], ["--collect_graph"], ["--collect_graph", "--compile_eager"]):
        graphed = bool(flags)
        torch.manual_seed(0)
        args = parse_args(small(extra + SYSTEMS + flags))
        algo = QUADRUPED_ALGOS[args.algo](args, model, kin, task, CUDA)
        sim = QuadrupedSim(model, task, n, CUDA)
        buffer = ReplayBuffer(32, n, CUDA, fields=replay_fields(), priorities=False)
        if graphed:
            buffer.use_device_ptr(device_size=True)
        c = QuadrupedCollector(sim, task, algo, buffer, args)
        for k in range(steps):
            torch.manual_seed(100 + k)
            c()
        torch.cuda.synchronize()
        assert (c.graph is not None) == graphed and (buffer.ptr, buffer.size) == (steps % 32, 32)
        assert algo.fk == c.fk and (c.fk == kin.forward) == ("--compile_eager" not in flags)
        runs.append((buffer, c, sim, trained(algo)))
    b0, c0, sim0, t0 = runs[0]
    for b1, c1, sim1, t1 in runs[1:]:
        assert (int(b1.ptr_t), int(b1.size_t)) == (b1.ptr, b1.size)
        for k in ("f", "done"):
            assert torch.equal(b0.data[k], b1.data[k]), k
        for k in ("s", "s1", "u", "a_prev"):
            assert torch.allclose(b0.data[k], b1.data[k], atol=2e-3), (k, float((b0.data[k] - b1.data[k]).abs().max()))
        for k in ("steps", "done", "term", "ep_len"):
            assert torch.equal(c0.acc[k], c1.acc[k]), k
        for k in ("r", "trunk_err", "feet_err"):
            assert torch.allclose(c0.acc[k], c1.acc[k], rtol=1e-3), k
        assert torch.equal(c0.ep_len, c1.ep_len) and torch.allclose(c0.a_prev, c1.a_prev, atol=1e-3)
        assert torch.equal(sim0.f, sim1.f) and torch.allclose(sim0.state(), sim1.state(), atol=2e-3)
        assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-4) for a, b in zip(t0, t1, strict=True))
    assert float(c0.acc["done"]) > 0 and float(c0.acc["steps"]) == n * steps
    host_head = ReplayBuffer(32, n, CUDA, fields=replay_fields(), priorities=False)
    with pytest.raises(ValueError, match="device-side head"):
        QuadrupedCollector(sim, task, algo, host_head, parse_args(small(SYSTEMS + ["--collect_graph"])))


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_update_graphs_replay_the_eager_updates(gpu_setup, monkeypatch, extra):
    """--update_graphs: the world-model updates and the policy update (with the policy average's step) captured as CUDA
    graphs and replayed return the values and leave the parameters, normalizers, optimizer states and average that the
    same updates run eagerly do, when both draw the same random numbers; the replays follow the learning rate's schedule."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    model, kin, task = gpu_setup
    runs = []
    for graphs in (False, True):
        args, algo, buffer = agent_and_replay(model, kin, task, extra + SYSTEMS + ["--update_graphs", "--pi_ema", "0.9"], n=16, dev=CUDA)
        buffer.use_device_ptr(device_size=True)
        avg = hand_fast.PolicyAverage(algo.pi, 0.9)
        wm_update, policy_update = hand_fast.update_functions(algo, buffer, args if graphs else argparse.Namespace(update_graphs=False), avg)
        outs = []
        for k in range(7):  # three eager calls, the capture, three more replays
            torch.manual_seed(10 + k)
            for o in algo.optimizers:
                set_lr(o, args.lr * (1 - 0.1 * k))
            outs += [wm_update(), *policy_update()]
        torch.cuda.synchronize()
        state = [v for o in algo.optimizers for v in o.state_dict()["state"].values()]
        runs.append((torch.stack(outs), trained(algo) + list(avg.avg.state_dict().values()), [x for s in state for x in s.values()]))
    (out0, t0, s0), (out1, t1, s1) = runs
    finite = torch.isfinite(out0)
    assert torch.equal(finite, torch.isfinite(out1)) and int(finite.sum()) == 7 * (6 if not extra else 5)
    assert torch.equal(out0[finite], out1[finite]), float((out0[finite] - out1[finite]).abs().max())
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert len(s0) == len(s1) and all(torch.equal(a, b) for a, b in zip(s0, s1, strict=True))
    assert int(s0[0]) == 7 * args.wm_batches  # the world model's Adam step count


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_training_loop_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, extra):
    """train_quadruped's loop with every systems option on two short clips: the graphs are captured, the log's rows are
    finite, the checkpoint holds the policy average and the ramp's state, --keep_ckpts keeps each checkpoint's policy
    with the run's time, and a resumed run continues the log."""
    import ndm_gym.train_quadruped as train_quadruped

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_quadruped, "build", lambda *a, **k: (*gpu_setup, CLIPS))
    run = tmp_path / "run"
    argv = [*small(extra), *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_ema", "0.9", "--pi_lr_mult", "2",
            "--lr_decay_ret", "0.99", "--lr_decay_by", "8", "--lr_decay_len", "4", "--num_envs", "32", "--buffer_steps", "16", "--warmup_steps", "8",
            "--wm_only_iters", "2", "--log_every", "4", "--ckpt_every", "4", "--keep_ckpts", "--log_dir", str(run)]  # fmt: skip
    for iterations, resume in ((12, []), (20, ["--resume", str(run / "last.pt")])):
        monkeypatch.setattr(sys, "argv", ["train_quadruped", *argv, "--iterations", str(iterations), *resume])
        train_quadruped.main()
        rows = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
        assert rows[:, 0].tolist() == list(range(4, iterations + 1, 4))
        cols = [train_quadruped.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "ep_len")]
        assert np.isfinite(rows[1:, cols]).all() and (rows[:, train_quadruped.LOG_COLUMNS.index("sim_r")] > 0).all()
        ck = torch.load(run / "last.pt", map_location=CUDA, weights_only=False)
        assert ck["iteration"] == iterations and {"pi", "pi_live", "pi_avg"} <= set(ck) and ck["pi_lr_decay_at"] == 8
        assert int(ck["pi_avg"]["n_averaged"]) == iterations - 2
        assert any(not torch.equal(ck["pi"][k], ck["pi_live"][k]) for k in ck["pi"])
        kept = sorted(p.name for p in run.glob("ckpt_*.pt"))
        assert kept == [f"ckpt_{i:06d}.pt" for i in range(4, iterations + 1, 4)]
        light = torch.load(run / kept[-1], map_location=CUDA, weights_only=False)
        assert set(light) == {"pi", "iteration", "wall_time", "sim_steps", "args"} and all(torch.equal(light["pi"][k], ck["pi"][k]) for k in ck["pi"])
    monkeypatch.setattr(sys, "argv", ["train_quadruped", *argv, "--iterations", "24", "--resume", str(run / "last.pt"), "--pi_opt", "muon"])
    with pytest.raises(ValueError, match="pi_opt"):
        train_quadruped.main()


@GPU
def test_evaluation_rollout_and_pushes(gpu_setup):
    """evaluate_quadruped.rollout: per step a reward within its maximum and the errors; a push adds its velocity to the
    trunk, horizontally, at its period only and with the same directions for the same seed; pushes hard enough end every
    rollout."""
    from ndm_gym.sim.rigid.quadruped_sim import QuadrupedSim

    model, _, task = gpu_setup
    sim = QuadrupedSim(model, task, 8, CUDA)
    qpos, errs, term_at, rew = rollout(sim, task)
    assert qpos.shape == (40, 8, NQ) and rew.shape == (39, 8) and errs["trunk_pos"].shape == (39, 8)
    assert bool((rew <= max_step_reward(task.cfg) + 1e-6).all()) and bool(torch.isfinite(rew).all())
    m = summarize(task, errs, term_at, CLIPS, rew)
    assert 0 < m["norm_return"] <= 1 and m["success_rate"] <= 1

    def pushed(mag, seed):
        return rollout(sim, task, push=(mag, 10, torch.Generator(device=CUDA).manual_seed(seed)))

    q0, q1, q2 = pushed(0.0, 0)[0], pushed(1.0, 0)[0], pushed(1.0, 0)[0]

    def close(a, b):  # the simulator's steps repeat to rounding only
        return torch.allclose(a, b, atol=1e-3)

    assert close(q0, qpos) and close(q1, q2)  # no push: the plain rollout; the same seed: the same pushes
    assert close(q1[:11], qpos[:11]) and not close(q1[11], qpos[11])  # the first push is before step 10
    assert not close(pushed(1.0, 1)[0][11], q1[11])
    dz = (q1[11, :, 2] - qpos[11, :, 2]).abs().max()
    dxy = (q1[11, :, :2] - qpos[11, :, :2]).norm(dim=-1)
    assert float(dz) < 5e-3 and bool((dxy > 5e-3).all()) and bool((dxy < 1.0 * CONTROL_DT + 5e-3).all())  # horizontal, about 1 m/s for a step
    assert bool((pushed(20.0, 0)[2] >= 0).all())


def test_length_buckets():
    """evaluate_quadruped.length_buckets: every clip in one group, the groups of consecutive lengths; one long clip
    among many short ones gets a group of its own when the worlds are many, and none when a step's fixed cost
    outweighs them."""
    lengths = [100] * 30 + [4000]
    groups = length_buckets(lengths, per_clip=256)
    assert sorted(sum(groups, [])) == list(range(31)) and groups[-1] == [30] and len(groups) == 2
    assert length_buckets(lengths, per_clip=1) == [list(range(31))]
    assert length_buckets([7], per_clip=3) == [[0]]
