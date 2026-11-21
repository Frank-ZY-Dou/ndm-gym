"""Animal motion tracking on a small body written here (a free root, two hinges, a filtered torque actuator and a position
servo): the activation filters' closed form against MuJoCo's step, the clips' resampling, the world model's control input
with activations, the task at the reference (its errors, terminations and the actions' mapping), the updates of both
algorithms, training's arguments and the download's SHA-256 checks; with MIMIC-MJX's data downloaded (tools/download_mimic_mjx.py), its bodies against their
reference clips."""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.animal_algos import ANIMAL_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.algos.hand_fast import add_hand_fast_args
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.rotations import quat_to_mat
from ndm_gym.motion.sources.mimic_mjx import resample
from ndm_gym.robots.animals import activation_filter
from ndm_gym.tasks.animal_tracking import AnimalKinematics, AnimalTask, RewardCfg, build_reference
from ndm_gym.train import add_train_args
from ndm_gym.train_animal import ANIMAL_DEFAULTS, UNSUPPORTED, UNSUPPORTED_SYSTEMS, parse_args, parse_clips
from ndm_gym.worldmodels.animal import AnimalWorldModel
from ndm_gym.worldmodels.mjcf import MjcfLayout

SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w", "wm_lr")  # fmt: skip
# hand training's systems options that animal training implements
SYSTEMS = ("fused_adam", "reset_masked", "no_sync", "collect_graph", "compile_eager", "update_graphs", "pi_opt", "wm_opt", "pi_ema", "pi_lr_mult",
           "lr_decay_ret", "lr_decay_len", "lr_decay_by", "muon_batched")  # fmt: skip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "mimic_mjx")
CTRL_DT, SUBSTEPS = 0.01, 5

BODY = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="torso" pos="0 0 .2">
      <freejoint name="root"/>
      <geom type="capsule" size=".03" fromto="-.1 0 0 .1 0 0" mass="1"/>
      <body name="leg" pos=".1 0 0">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.15" mass=".2"/>
        <body name="foot" pos="0 0 -.15">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size=".015" fromto="0 0 0 0 0 -.1" mass=".1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="hip" joint="hip" dyntype="filter" dynprm=".04" gainprm="5" ctrlrange="-1 1" ctrllimited="true"/>
    <position name="knee" joint="knee" kp="20" ctrlrange="-2 0" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def body(activation: bool = True) -> mujoco.MjModel:
    xml = BODY if activation else BODY.replace('dyntype="filter" dynprm=".04" ', "")
    return mujoco.MjModel.from_xml_string(xml)


def clip(model, n: int, seed: int):
    """qpos and qvel of the body under smooth random controls, every control step."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    q, v = [], []
    for k in range(n):
        d.ctrl[:] = (0.5 * np.sin(0.1 * k + seed), -1.0 + 0.5 * np.sin(0.07 * k))
        d.ctrl[0] += 0.1 * rng.standard_normal()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
        q.append(d.qpos.copy()), v.append(d.qvel.copy())
    return torch.tensor(np.array(q), dtype=torch.float32), torch.tensor(np.array(v), dtype=torch.float32)


def make_task(cfg: RewardCfg = RewardCfg()):
    model = body()
    lay = MjcfLayout.from_model(body(activation=False))
    bodies = [1, 2, 3]
    kin = AnimalKinematics(lay, bodies)
    ref = build_reference([clip(model, 40, 0), clip(model, 30, 1)], kin, "cpu")
    r, rbar = activation_filter(model, SUBSTEPS)
    return model, lay, AnimalTask("rat", model, kin, ref, [2], CTRL_DT, r, rbar, cfg)


def test_activation_filter_matches_mujoco():
    model = body()
    r, rbar = activation_filter(model, SUBSTEPS)
    assert r[1] == 0 and rbar[1] == 0  # the position servo has no filter
    d = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(5):
        mujoco.mj_resetData(model, d)
        a0, u = rng.uniform(-1, 1), rng.uniform(-1, 1)
        d.act[0], d.ctrl[0] = a0, u
        forces = []
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
            forces.append(d.actuator_force[0])
        assert abs(d.act[0] - (u + (a0 - u) * r[0])) < 1e-6  # r is float32
        # the forces of the physics steps: the activations at their starts (actearly off), whose mean the closed form gives
        acts = [u + (a0 - u) * (1 - model.opt.timestep / 0.04) ** k for k in range(SUBSTEPS)]
        assert np.allclose(forces, [5 * a for a in acts], atol=1e-9)
        assert abs(np.mean(acts) - (u + (a0 - u) * rbar[0])) < 1e-6


def test_resample():
    T, hz = 11, 50.0
    q = torch.zeros(T, 7)
    q[:, 0] = torch.arange(T, dtype=torch.float32)
    ang = torch.linspace(0, 1.0, T)
    q[:, 3], q[:, 6] = torch.cos(ang / 2), torch.sin(ang / 2)  # a turn about z
    v = torch.arange(T, dtype=torch.float32).unsqueeze(-1).expand(T, 6).clone()
    q2, v2 = resample(q, v, hz, 0.01, [3])
    assert q2.shape[0] == 2 * (T - 1) + 1
    assert torch.allclose(q2[::2], q, atol=1e-6) and torch.allclose(v2[1::2, 0], torch.arange(T - 1) + 0.5)
    assert torch.allclose(q2[:, 3:7].norm(dim=-1), torch.ones(q2.shape[0]), atol=1e-6)
    assert torch.allclose(2 * torch.atan2(q2[:, 6], q2[:, 3]), torch.linspace(0, 1.0, q2.shape[0]), atol=1e-5)
    q3, v3 = resample(q, v, 100.0, 0.01, [3])  # the control rate: unchanged
    assert torch.allclose(q3, q) and torch.allclose(v3, v)


def test_world_model_control_input():
    lay = dataclasses.replace(MjcfLayout.from_model(body(activation=False)), dt=CTRL_DT)
    plain, filt = AnimalWorldModel(lay), AnimalWorldModel(lay, n_act=1)
    assert filt.n_in == plain.n_in + 1  # the start activation beside the controls
    q = torch.zeros(4, lay.nq)
    q[:, 3] = 1
    u = torch.randn(4, lay.nu + 1)
    assert filt.inputs(q, torch.zeros(4, lay.nv), u).shape[-1] == filt.n_in
    with pytest.raises(ValueError):
        AnimalWorldModel(lay, n_act=1, backbone="rigidformer")


def test_task_at_the_reference():
    model, _, task = make_task()
    ref = task.ref
    assert task.state_dim == model.nq + model.nv + model.na and (task.nj, task.nu, task.na) == (2, 2, 1)
    f = torch.arange(ref.frames)
    S = (ref.qpos[f], ref.qvel[f], torch.zeros(ref.frames, 1))
    fk = task.kin(S[0])
    e = task.errors(S, fk, f)
    for k in ("root_pos", "joint", "joint_vel"):
        assert float(e[k].abs().max()) < 1e-5
    assert float(e["body"].max()) < 1e-5 and float(e["root_rot_sq"].max()) < 1e-5
    assert not bool(task.termination(S, f).any()) and float(task.soft_termination(S, f).max()) < 1e-3
    a = torch.zeros(ref.frames, task.nu)
    r = task.reward(S, fk, f, a, a)
    assert torch.allclose(r, torch.full_like(r, task.max_step_reward), rtol=1e-4)
    # the clips: the second starts at the first's end + 1, and the reference stays on a clip's last frame
    assert int(ref.end[0]) == 39 and int(ref.start[40]) == 40 and int(ref.next_frame(torch.tensor(39))) == 39
    assert task.obs(S, fk, a).shape[-1] == task.obs_dim
    assert task.goal(S, f, 5).shape[-1] == 5 * task.goal_frame_dim
    # the actions: the filtered torque actuator over its range, the servo about the reference's next joint position
    u = task.setpoint(torch.tensor([[0.5, 0.2]]), torch.tensor([3]))
    assert abs(float(u[0, 0]) - 0.5) < 1e-6
    knee = int(model.jnt_qposadr[model.joint("knee").id])
    assert abs(float(u[0, 1]) - float(ref.qpos[4, knee] + 0.5 * 0.2)) < 1e-6 or float(u[0, 1]) in (-2.0, 0.0)
    # a root moved away terminates
    far = ref.qpos[f].clone()
    far[:, 0] += 1.0
    assert bool(task.termination((far, S[1], S[2]), f).all())


def test_task_invariant_to_heading_and_position():
    _, _, task = make_task()
    ref = task.ref
    f = torch.tensor([5, 10, 50])
    S = (ref.qpos[f].clone(), ref.qvel[f].clone(), torch.zeros(3, 1))
    obs = task.obs(S, task.kin(S[0]), torch.zeros(3, 2))
    # the same state shifted: obs unchanged (height kept)
    q = S[0].clone()
    q[:, :2] += 3.0
    obs2 = task.obs((q, S[1], S[2]), task.kin(q), torch.zeros(3, 2))
    assert torch.allclose(obs, obs2, atol=1e-5)
    assert quat_to_mat(S[0][:, 3:7]).shape == (3, 3, 3)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_updates(algo, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    torch.manual_seed(0)
    model, lay, task = make_task()
    args = argparse.Namespace(**{**vars(parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo])), "horizon": 4, "pi_batch": 16,
                                 "wm_batch": 8, "wm_window": 3, "wm_batches": 1, "wm_hidden": [32, 32]})  # fmt: skip
    agent = ANIMAL_ALGOS[algo](args, task, dataclasses.replace(lay, dt=CTRL_DT), "cpu")
    buffer = ReplayBuffer(8, 4, "cpu", fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.tensor([0, 5, 41, 50])
    for _ in range(8):
        s = torch.cat((task.ref.qpos[f], task.ref.qvel[f], torch.zeros(4, 1)), -1)
        s1 = torch.cat((task.ref.qpos[f + 1], task.ref.qvel[f + 1], torch.zeros(4, 1)), -1)
        u = task.setpoint(torch.zeros(4, task.nu), f)
        agent.update_wm_norms(task.unflatten(s), task.unflatten(s1), u)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": torch.zeros(4, task.nu), "f": f, "done": torch.zeros(4, dtype=torch.bool)})
        f = f + 1
    assert torch.isfinite(agent.wm_update(buffer))
    out = agent.policy_update(buffer)
    assert torch.isfinite(out[0]) and torch.isfinite(out[2])


def test_training_arguments():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {a.dest for a in parser._actions} - {"help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED)  # every argument implemented or refused
    args = parse_args(["--animal", "fly", "--clip_ids", "0-3", "7"])
    assert args.num_envs == ANIMAL_DEFAULTS["num_envs"] and parse_clips(args.clip_ids) == [0, 1, 2, 3, 7]
    with pytest.raises(SystemExit):
        parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", "supermimic", "--st_fail_w", "2"])
    with pytest.raises(SystemExit):
        parse_args(["--animal", "rat", "--clip_ids", "0", "--wm_backbone", "rigidformer"])
    assert parse_args(["--animal", "rat", "--clip_ids", "0", "--wm_lr", "2e-3"]).wm_lr == 2e-3


def test_systems_options_implemented_or_refused(capsys):
    parser = argparse.ArgumentParser()
    add_hand_fast_args(parser)
    assert {a.dest for a in parser._actions} - {"help"} == set(SYSTEMS) | set(UNSUPPORTED_SYSTEMS)
    for name in UNSUPPORTED_SYSTEMS:  # each is a flag
        with pytest.raises(SystemExit):
            parse_args(["--animal", "rat", "--clip_ids", "0", f"--{name}"])
        assert f"--{name} is an option of hand training" in capsys.readouterr().err


def test_evaluation_refuses_unused_options(monkeypatch, capsys):
    from ndm_gym import evaluate_animal

    for argv, message in ((["--ckpt", "last.pt", "--joint_noise", "0.05"], "--joint_noise perturbs the repeated starts"),
                          (["--ckpt", "last.pt", "--repeats", "-1"], "--repeats >= 0")):  # fmt: skip
        monkeypatch.setattr(sys, "argv", ["evaluate_animal", *argv])
        with pytest.raises(SystemExit):
            evaluate_animal.main()
        assert message in capsys.readouterr().err


def test_time_to_the_target(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("time_to", os.path.join(ROOT, "benchmarks", "mimic_mjx_animal_tracking", "time_to.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(sys, "argv", ["time_to.py", str(run)])
    with pytest.raises(SystemExit, match="no ckpt_"):  # a run without kept checkpoints is not "target not reached"
        tool.main()

    def checkpoint(it, ended, body_mm, episodes=50):
        (run / f"ckpt_{it:06d}.pt").write_bytes(b"")
        (run / "eval_ckpts" / f"{it:06d}").mkdir(parents=True, exist_ok=True)
        metrics = {"iteration": it, "minutes": it / 100, "animal": "stick", "clips": 10, "episodes": episodes, "ended_share": ended, "body_mm": body_mm}
        (run / "eval_ckpts" / f"{it:06d}" / "metrics.json").write_text(json.dumps(metrics))

    for it, ended, body_mm in ((250, 0.0, 0.5), (500, 0.1, 0.4), (750, 0.02, 0.5), (1000, 0.0, 0.4)):  # the target: 2 %, 0.53 mm
        checkpoint(it, ended, body_mm)
    tool.main()  # the evaluations are there: none is run
    res = json.loads((run / "time_to.json").read_text())
    assert res["iteration_to_target"] == 750 and res["minutes_to_target"] == 7.5  # held from 750 on, not from 250
    assert [r["target"] for r in res["checkpoints"]] == [True, False, True, True]
    checkpoint(1250, 0.0, 0.4, episodes=10)  # another evaluation's metrics (no perturbed repeats)
    with pytest.raises(SystemExit, match="not this script's evaluation"):
        tool.main()


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "rat", "reference.h5")), reason="MIMIC-MJX's data not downloaded")
@pytest.mark.parametrize("animal", ["rat", "fly", "stick"])
def test_bodies_match_their_clips(animal):
    h5py = pytest.importorskip("h5py")
    from ndm_gym.robots.animals import build_model

    if not os.path.exists(os.path.join(DATA, animal, "reference.h5")):
        pytest.skip(f"{animal}'s data not downloaded")
    m = build_model(animal, DATA)
    with h5py.File(os.path.join(DATA, animal, "reference.h5"), "r") as f:
        assert f["qpos"].shape[1] == m.nq and f["qvel"].shape[1] == m.nv
    lay = MjcfLayout.from_model(build_model(animal, DATA, activation=False))
    assert lay.anchor == (0, 0) and m.opt.noslip_iterations == 0


def test_fused_kinematics_match():
    model, lay, task = make_task()
    fused = AnimalKinematics(lay, [1, 2, 3])
    fused.fused = True
    q = task.ref.qpos[::7]
    assert torch.allclose(task.kin(q), fused(q), atol=1e-5)


def test_download_checks_sha256(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_mimic_mjx", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "download_mimic_mjx.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    (tmp_path / "src").write_bytes(b"clip")
    good = hashlib.sha256(b"clip").hexdigest()
    url = (tmp_path / "src").as_uri()
    tool.fetch(url, str(tmp_path / "a" / "ok"), good)  # downloaded and checked
    tool.fetch(url, str(tmp_path / "a" / "ok"), good)  # already there, checked again
    with pytest.raises(ValueError, match="SHA-256"):
        tool.fetch(url, str(tmp_path / "a" / "bad"), "0" * 64)
    assert not (tmp_path / "a" / "bad").exists() and not (tmp_path / "a" / "bad.part").exists()
    with pytest.raises(ValueError, match="SHA-256"):
        tool.fetch(url, str(tmp_path / "a" / "ok"), "0" * 64)
    with open(tool.SHA256_FILE) as fh:
        pinned = json.load(fh)
    assert {"rat/reference.h5", "fly/reference.h5", "stick/reference.h5"} <= set(pinned) and all(len(v) == 64 for v in pinned.values())
