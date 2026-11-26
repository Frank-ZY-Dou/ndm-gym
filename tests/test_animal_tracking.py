"""Animal motion tracking on a small body written here (a free root, two hinges, a filtered torque actuator and a position
servo): the activation filters' closed form against MuJoCo's step, the clips' resampling, the world model's control input
with activations, the task at the reference (its errors, terminations and the actions' mapping), the updates of both
algorithms and training's arguments; the evaluation's summary, the benchmark's stop rule and time to the target, the motion
check, and the download's SHA-256 checks; with MIMIC-MJX's data downloaded (tools/download_mimic_mjx.py), its bodies against
their reference clips."""

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.animal_algos import ANIMAL_ALGOS, WINDOW_KEYS, cap_rollout_grads, clip_rollout_grads, live_median, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.rotations import quat_to_mat
from ndm_gym.motion.sources.mimic_mjx import resample, velocities_from_positions
from ndm_gym.robots.animals import activation_filter
from ndm_gym.tasks.animal_tracking import AnimalKinematics, AnimalTask, RewardCfg, build_reference
from ndm_gym.algos.hand_fast import add_hand_fast_args
from ndm_gym.evaluate_animal import ERRORS, summarize
from ndm_gym.train import add_train_args
from ndm_gym.train_animal import (ANIMAL_DEFAULTS, REF_QVEL, UNSUPPORTED, UNSUPPORTED_SYSTEMS, build, parse_args, parse_clips, resume_mismatch,
                                  world_model_lead)  # fmt: skip
from ndm_gym.worldmodels.animal import AnimalWorldModel
from ndm_gym.worldmodels.mjcf import MjcfLayout

SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "bptt_clip_k", "pi_rollout_clip", "st_reg", "st_imag_noise", "st_fail_w", "wm_lr")  # fmt: skip
# hand training's systems options that animal training implements (the others are UNSUPPORTED_SYSTEMS)
SYSTEMS = ("fused_adam", "reset_masked", "no_sync", "collect_graph", "compile_eager", "update_graphs", "pi_opt", "wm_opt", "pi_ema",
                  "pi_lr_mult", "lr_decay_ret", "lr_decay_len", "lr_decay_by", "muon_batched")  # fmt: skip
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "mimic_mjx")
BENCH = os.path.join(REPO, "benchmarks", "mimic_mjx_animal_tracking")


def load(path: str, name: str):
    """A script of the repository (tools/, benchmarks/) as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def make_task(cfg: RewardCfg = RewardCfg(), animal: str = "rat"):
    """The small body as ``animal``'s task (its terminations)."""
    model = body()
    lay = MjcfLayout.from_model(body(activation=False))
    bodies = [1, 2, 3]
    kin = AnimalKinematics(lay, bodies)
    ref = build_reference([clip(model, 40, 0), clip(model, 30, 1)], kin, "cpu")
    r, rbar = activation_filter(model, SUBSTEPS)
    return model, lay, AnimalTask(animal, model, kin, ref, [2], CTRL_DT, r, rbar, cfg)


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


def test_velocities_from_positions():
    """The velocities of a clip's positions: each frame's takes it to the next (MuJoCo's integration), a constant velocity is
    recovered (the root's angular velocity in the body's frame), the last frame repeats the one before, and a file's
    velocities scaled as MIMIC-MJX's are (10 times the root's) are not used."""
    model, dt, T = body(), CTRL_DT, 12
    v0 = np.array([0.3, -0.2, 0.1, 0.5, -1.0, 2.0, 0.7, -0.4])  # root linear (world), root angular (body), two hinges
    q = np.zeros((T, model.nq))
    q[0, 2], q[0, 3:7] = 0.2, (0.9, 0.1, -0.3, 0.3)
    q[0, 3:7] /= np.linalg.norm(q[0, 3:7])
    for t in range(T - 1):
        q[t + 1] = q[t]
        mujoco.mj_integratePos(model, q[t + 1], v0, dt)
    v = velocities_from_positions(model, torch.tensor(q, dtype=torch.float32), dt)
    assert v.shape == (T, model.nv) and v.dtype == torch.float32
    assert torch.allclose(v, torch.tensor(v0, dtype=torch.float32).expand(T, -1), atol=2e-3)
    assert torch.equal(v[-1], v[-2])
    # a simulated clip: its positions' velocities take every frame to the next
    qs, vs = clip(model, 30, 0)
    vp = velocities_from_positions(model, qs, dt).double().numpy()
    for t in range(qs.shape[0] - 1):
        q1 = qs[t].double().numpy().copy()
        mujoco.mj_integratePos(model, q1, vp[t], dt)
        assert np.allclose(q1, qs[t + 1].double().numpy(), atol=1e-5)
    # the simulator's own velocities are close to them, a file's with the root's linear velocity 10 times are not
    assert float((torch.tensor(vp[1:-1, :3]) - vs[1:-1, :3].double()).abs().max()) < 0.25
    assert float((torch.tensor(vp[1:-1, :3]) - 10 * vs[1:-1, :3].double()).abs().max()) > 1.0
    with pytest.raises(ValueError):
        velocities_from_positions(model, qs[:, :-1], dt)


def test_ref_qvel_option(capsys):
    """--ref_qvel: the file's velocities by default, the positions' in the named configurations; another value is refused."""
    base = ["--animal", "rat", "--clip_ids", "0"]
    assert parse_args(base).ref_qvel == "stored"
    assert parse_args([*base, "--ref_qvel", "positions"]).ref_qvel == "positions"
    for name in ("stick-supertrack", "fly-supermimic", "rat-supertrack", "rat-supermimic"):
        assert parse_args(["--config", name]).ref_qvel == "positions"
    assert parse_args(["--config", "rat-supertrack", "--ref_qvel", "stored"]).ref_qvel == "stored"
    with pytest.raises(SystemExit):
        parse_args([*base, "--ref_qvel", "fitted"])
    assert "--ref_qvel" in capsys.readouterr().err


def test_velocities_from_positions_by_hand():
    """A clip integrated from known velocities gives them back (the root's and the hinges'); its last frame takes the one
    before's; resampled positions' velocities are their source interval's."""
    model = body()
    rng = np.random.default_rng(0)
    T = 12
    v = rng.normal(size=(T - 1, model.nv))
    q = np.zeros((T, model.nq))
    q[0] = model.qpos0
    q[0, 3:7] = (0.8, 0.2, -0.4, 0.4)
    for k in range(T - 1):
        q[k + 1] = q[k]
        mujoco.mj_integratePos(model, q[k + 1], v[k], CTRL_DT)
    out = velocities_from_positions(model, torch.tensor(q), CTRL_DT)
    assert out.shape == (T, model.nv) and out.dtype == torch.float64
    assert torch.allclose(out[:-1], torch.tensor(v), atol=1e-9) and torch.equal(out[-1], out[-2])
    out32 = velocities_from_positions(model, torch.tensor(q, dtype=torch.float32), CTRL_DT)
    assert out32.dtype == torch.float32 and torch.allclose(out32[:-1], torch.tensor(v, dtype=torch.float32), atol=1e-3)
    # a clip recorded at 50 Hz played at 100 Hz: both control steps of a source interval move at the interval's velocity
    q2, _ = resample(torch.tensor(q), torch.zeros(T, model.nv, dtype=torch.float64), 1 / (2 * CTRL_DT), CTRL_DT, [3])
    out2 = velocities_from_positions(model, q2, CTRL_DT)
    assert torch.allclose(out2[:-1:2], torch.tensor(v) / 2, atol=1e-6) and torch.allclose(out2[1:-1:2], torch.tensor(v) / 2, atol=1e-6)
    with pytest.raises(ValueError):
        velocities_from_positions(model, torch.tensor(q[:1]), CTRL_DT)


def test_world_model_control_input():
    lay = dataclasses.replace(MjcfLayout.from_model(body(activation=False)), dt=CTRL_DT)
    plain, filt = AnimalWorldModel(lay), AnimalWorldModel(lay, n_act=1)
    assert filt.n_in == plain.n_in + 1  # the start activation beside the controls
    q = torch.zeros(4, lay.nq)
    q[:, 3] = 1
    u = torch.randn(4, lay.nu + 1)
    assert filt.inputs(q, torch.zeros(4, lay.nv), u).shape[-1] == filt.n_in
    with pytest.raises(ValueError):  # the controls without the start activations: refused, not padded
        filt.inputs(q, torch.zeros(4, lay.nv), u[:, : lay.nu])
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


def test_root_distance_horizontal_for_the_stick_insect():
    """The stick insect's root distance leaves its height out (vnl-playground's stick environment); the rat's is in 3D."""
    for animal, horizontal in (("stick", True), ("rat", False), ("fly", False)):
        _, _, task = make_task(animal=animal)
        f = torch.arange(5)
        S = (task.ref.qpos[f].clone(), task.ref.qvel[f], torch.zeros(5, 1))
        up, side = S[0].clone(), S[0].clone()
        up[:, 2] += 2 * task.term_dist  # straight up, twice the distance
        side[:, 0] += 2 * task.term_dist
        assert bool(task.termination((side, S[1], S[2]), f).all())
        assert bool(task.termination((up, S[1], S[2]), f).any()) != horizontal
        assert (float(task.soft_termination((up, S[1], S[2]), f).max()) < 0.5) == horizontal


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
    # the speed script's world model: the algorithm's, its imagined step the same
    from ndm_gym.wmbench.speed_animal import world_model, wm_step

    wm = world_model(args, task, dataclasses.replace(lay, dt=CTRL_DT), 1.0)
    assert type(wm) is type(agent.wm) and wm.kind == agent.wm.kind
    wm.load_state_dict(agent.wm.state_dict())
    S = task.unflatten(buffer.at("s", torch.zeros(4, dtype=torch.long), torch.arange(4)))
    u = task.setpoint(torch.rand(4, task.nu) * 2 - 1, f - 8)
    with torch.no_grad():
        for x, y in zip(wm_step(wm, task, *S, u), agent.wm_step(S, u), strict=True):
            assert torch.allclose(x, y)


def small_agent(algo: str, **over):
    """An agent of ``algo`` on the small body with a filled replay (test_updates'), torch.compile patched out by the caller."""
    torch.manual_seed(0)
    model, lay, task = make_task()
    args = argparse.Namespace(**{**vars(parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo])), "horizon": 4, "pi_batch": 16,
                                 "wm_batch": 8, "wm_window": 3, "wm_batches": 1, "wm_hidden": [32, 32], **over})  # fmt: skip
    agent = ANIMAL_ALGOS[algo](args, task, dataclasses.replace(lay, dt=CTRL_DT), "cpu")
    buffer = ReplayBuffer(8, 4, "cpu", fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.tensor([0, 5, 41, 50])
    for k in range(8):
        s = torch.cat((task.ref.qpos[f], task.ref.qvel[f], torch.zeros(4, 1)), -1)
        s1 = torch.cat((task.ref.qpos[f + 1], task.ref.qvel[f + 1], torch.zeros(4, 1)), -1)
        u = task.setpoint(torch.zeros(4, task.nu), f)
        agent.update_wm_norms(task.unflatten(s), task.unflatten(s1), u)
        a_prev = 0.3 * torch.sin(torch.arange(4 * task.nu, dtype=torch.float32).reshape(4, task.nu) + k)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(4, dtype=torch.bool)})
        f = f + 1
    with torch.no_grad():  # a policy away from its zero start, so that its actions and their gradients are not degenerate
        for q in agent.pi.net[-1].parameters():
            q.add_(0.2 * torch.randn_like(q))
    return agent, buffer, task


def policy_after_update(algo: str, **over) -> torch.Tensor:
    agent, buffer, _ = small_agent(algo, **over)
    torch.manual_seed(1)
    out = agent.policy_update(buffer)
    assert torch.isfinite(out[0]) and torch.isfinite(out[3])
    return torch.cat([q.detach().reshape(-1) for q in agent.pi.parameters()])


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_bptt_options(algo, monkeypatch):
    """The main trainer's options on the policy's backward pass, for both algorithms: accepted; off by default; a rollout clip
    that caps nothing leaves the update as it is (the tapped steps compute the same rollout); each option on changes it."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    a = parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo])
    assert (a.bptt_clip_k, a.pi_rollout_clip) == (0.0, 0.0) and a.bptt_decay == 1.0
    a = parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo, "--bptt_decay", "0.95", "--bptt_clip_k", "10", "--pi_rollout_clip", "10"])
    assert (a.bptt_decay, a.bptt_clip_k, a.pi_rollout_clip) == (0.95, 10.0, 10.0)
    for bad in (["--bptt_decay", "0"], ["--bptt_decay", "1.5"], ["--bptt_clip_k", "-1"], ["--pi_rollout_clip", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo, *bad])
    base = policy_after_update(algo)
    assert torch.equal(policy_after_update(algo), base)  # the same seeds, the same update
    assert torch.equal(policy_after_update(algo, pi_rollout_clip=1e9), base)  # nothing capped: bit for bit the default's
    both = {"bptt_decay": 0.9, "bptt_clip_k": 2.0, "pi_rollout_clip": 2.0}
    for over in ({"bptt_decay": 0.5}, {"bptt_clip_k": 0.5}, {"pi_rollout_clip": 0.5}, both):
        assert not torch.equal(policy_after_update(algo, **over), base), over


def test_bptt_decay_weighs_a_later_loss(monkeypatch):
    """--bptt_decay lam in SuperTrack's update: over two imagined steps the policy gradient is affine in lam (the second
    step's loss reaches the first action through one state), and with lam = 1 it is the default's."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)

    def grad(lam):
        agent, buffer, _ = small_agent("supertrack", horizon=2, st_imag_noise=0.0, bptt_decay=lam)
        agent.opt_pi.step = lambda *a, **k: None  # the gradient as the update computes it, the weights left
        nn_clip = torch.nn.utils.clip_grad_norm_
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda params, m: nn_clip(params, 1e30))
        monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", lambda params, m: None)
        torch.manual_seed(1)
        agent.policy_update(buffer)
        return torch.cat([q.grad.reshape(-1) for q in agent.pi.parameters()])

    g3, g6, g9 = grad(0.3), grad(0.6), grad(0.9)
    assert (g9 - g6).norm() > 1e-6 * g6.norm()  # the decay reaches the gradient
    assert torch.allclose(g9 - g6, g6 - g3, rtol=1e-3, atol=1e-6 * float(g6.norm()))


def test_rollout_clip_as_the_main_trainer():
    """clip_rollout_grads (fixed shapes, no host read) gives the gradients of common.clip_rollout_output_grads: the rollouts
    over k times the median scaled down to it, the others untouched."""
    from ndm_gym.algos.common import clip_rollout_output_grads
    from ndm_gym.policies.hand_policy import HandPolicy

    torch.manual_seed(0)
    H, B, D, J, k = 3, 12, 7, 4, 2.0
    pi = HandPolicy(5, 2, J, hidden=(16,))
    with torch.no_grad():
        pi.net[-1].weight.add_(0.5 * torch.randn_like(pi.net[-1].weight))
    X = [torch.randn(B, D) for _ in range(H)]
    C = torch.randn(H, B, J)
    C[:, 3] *= 200.0  # one rollout far over the median
    C[:, 7] = 0.0  # one without a gradient: not in the median

    def backward():
        pi.zero_grad(set_to_none=True)
        taps = [torch.zeros(B, J, requires_grad=True) for _ in range(H)]
        sum((pi.from_input(X[t]) + taps[t]).mul(C[t]).sum() for t in range(H)).backward()
        return taps

    def grads():
        return torch.cat([q.grad.reshape(-1) for q in pi.parameters()])

    taps = backward()
    plain = grads().clone()
    n = clip_rollout_output_grads(lambda idx: pi.from_input(torch.stack([x[idx] for x in X]).flatten(0, 1)), taps, k)
    main = grads().clone()
    taps = backward()
    clip_rollout_grads(pi, X, taps, k)
    ours = grads().clone()
    assert n >= 1 and (main - plain).norm() > 0.5 * plain.norm()  # the large rollout's part is cut down
    # (both subtract most of the large rollout's part from the same sum: equal up to that subtraction's rounding)
    assert torch.allclose(ours, main, rtol=1e-5, atol=1e-6 * float(plain.abs().max()))
    taps = backward()
    clip_rollout_grads(pi, X, taps, 1e9)  # nothing over the cap: the gradients as they were
    assert torch.equal(grads(), plain)


def test_state_cap_as_the_main_trainer():
    """cap_rollout_grads here (its median without a host read) scales the rows' gradients as common.cap_rollout_grads does;
    live_median is torch.median over the positive entries, inf without any."""
    from ndm_gym.algos import common

    torch.manual_seed(0)
    for n_zero in (0, 1, 5):
        size = torch.rand(9) + 0.1
        size[:n_zero] = 0.0
        assert live_median(size) == size[size > 0].median()
    assert live_median(torch.zeros(4)) == torch.inf
    x = [torch.randn(10, 6, requires_grad=True), torch.randn(10, 3, 2, requires_grad=True)]
    c = [torch.randn(10, 6), torch.randn(10, 3, 2)]
    c[0][2] *= 300.0
    c[1][4] = 0.0
    out = []
    for cap in (cap_rollout_grads, common.cap_rollout_grads):
        for t in x:
            t.grad = None
        sum((y * w).sum() for y, w in zip(cap(tuple(x), 2.0), c, strict=True)).backward()
        out.append([t.grad.clone() for t in x])
    assert all(torch.equal(a, b) for a, b in zip(*out, strict=True))
    assert out[0][0][2].norm() < 0.05 * c[0][2].norm() and torch.equal(out[0][0][0], c[0][0])  # the large row cut, another untouched
    x[0].grad = None
    cap_rollout_grads((x[0],), 2.0)[0].mul(0.0).sum().backward()  # every row's gradient zero: nothing scaled, nothing non-finite
    assert torch.equal(x[0].grad, torch.zeros_like(x[0]))


def test_st_rate_w(monkeypatch):
    """--st_rate_w: SuperTrack's only; off by default; on, it changes the update."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    assert parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", "supertrack"]).st_rate_w == 0.0
    with pytest.raises(SystemExit):
        parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", "supermimic", "--st_rate_w", "0.05"])
    with pytest.raises(SystemExit):
        parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", "supertrack", "--st_rate_w", "-1"])
    base = policy_after_update("supertrack")
    assert not torch.equal(policy_after_update("supertrack", st_rate_w=5.0), base)


def _agent_with(algo: str, cfg: RewardCfg, **over):
    """``small_agent``'s agent on the small body's task with the reward settings ``cfg``, its replay the same draws."""
    torch.manual_seed(0)
    _, lay, task = make_task(cfg)
    args = argparse.Namespace(**{**vars(parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", algo])), "horizon": 4, "pi_batch": 16,
                                 "wm_batch": 8, "wm_window": 3, "wm_batches": 1, "wm_hidden": [32, 32], **over})  # fmt: skip
    agent = ANIMAL_ALGOS[algo](args, task, dataclasses.replace(lay, dt=CTRL_DT), "cpu")
    buffer = ReplayBuffer(8, 4, "cpu", fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.tensor([0, 5, 41, 50])
    for k in range(8):
        s = torch.cat((task.ref.qpos[f], task.ref.qvel[f], torch.zeros(4, 1)), -1)
        s1 = torch.cat((task.ref.qpos[f + 1], task.ref.qvel[f + 1], torch.zeros(4, 1)), -1)
        u = task.setpoint(torch.zeros(4, task.nu), f)
        agent.update_wm_norms(task.unflatten(s), task.unflatten(s1), u)
        a_prev = 0.3 * torch.sin(torch.arange(4 * task.nu, dtype=torch.float32).reshape(4, task.nu) + k)
        buffer.add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(4, dtype=torch.bool)})
        f = f + 1
    with torch.no_grad():
        for q in agent.pi.net[-1].parameters():
            q.add_(0.2 * torch.randn_like(q))
    return agent, buffer, task


def _policy_after_update_with(algo: str, cfg: RewardCfg, **over) -> torch.Tensor:
    agent, buffer, _ = _agent_with(algo, cfg, **over)
    torch.manual_seed(1)
    out = agent.policy_update(buffer)
    assert torch.isfinite(out[0]) and torch.isfinite(out[3])
    return torch.cat([q.detach().reshape(-1) for q in agent.pi.parameters()])


def test_target_rate_w_arguments(capsys):
    """--target_rate_w: off by default, both algorithms; it stands in the place of the action-rate terms (not with them), and
    SuperTrack's is in the units of its task-weighted loss; a resumed run shares it with its checkpoint."""
    st = ["--animal", "rat", "--clip_ids", "0", "--algo", "supertrack"]
    sm = ["--animal", "rat", "--clip_ids", "0", "--algo", "supermimic"]
    assert parse_args(st).target_rate_w == 0.0 and parse_args(sm).target_rate_w == 0.0
    assert parse_args(st + ["--st_task_weights", "--target_rate_w", "4"]).target_rate_w == 4.0
    assert parse_args(sm + ["--target_rate_w", "4"]).target_rate_w == 4.0
    weights, place = "a finite number >= 0", "stands in the place of the action-rate terms"
    for bad, base, why in ((["--target_rate_w", "-1"], sm, weights), (["--target_rate_w", "-0.5"], st + ["--st_task_weights"], weights),
                           (["--target_rate_w", "nan"], sm, weights), (["--target_rate_w", "inf"], sm, weights),
                           (["--target_rate_w", "nan"], st + ["--st_task_weights"], weights), (["--target_rate_w=-inf"], sm, weights),
                           (["--target_rate_w", "1"], st, "give --st_task_weights too"),
                           (["--target_rate_w", "1", "--st_rate_w", "0.05"], st + ["--st_task_weights"], place)):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(base + bad)
        err = capsys.readouterr().err
        assert "--target_rate_w" in err and why in err, (bad, err)
    # --st_rate_w given as 0 is accepted: dropping SuperTrack's term is explicit
    assert parse_args(st + ["--st_task_weights", "--target_rate_w", "2", "--st_rate_w", "0"]).st_rate_w == 0.0
    assert inspect.signature(build).parameters["target_rate_w"].default == 0.0
    # the resume rule: a checkpoint of before the option trained without it
    saved = vars(parse_args(sm))
    old = {k: v for k, v in saved.items() if k != "target_rate_w"}
    assert resume_mismatch(old, parse_args(sm)) is None
    why = resume_mismatch(old, parse_args(sm + ["--target_rate_w", "4"]))
    assert why is not None and "target_rate_w" in why
    assert resume_mismatch({**saved, "target_rate_w": 4.0}, parse_args(sm + ["--target_rate_w", "4"])) is None


def test_target_rate_w_reaches_the_task_from_the_trainer(monkeypatch, tmp_path):
    """The trainer's main gives build the run's --target_rate_w (0 without it), as it gives it the other task options."""
    pytest.importorskip("mujoco_warp")
    import ndm_gym.train_animal as ta

    seen, sig = {}, inspect.signature(ta.build)

    def recorder(*a, **k):
        seen["call"] = sig.bind(*a, **k)
        seen["call"].apply_defaults()
        raise _Stop

    monkeypatch.setattr(ta, "build", recorder)
    base = ["train_animal", "--animal", "rat", "--clip_ids", "0", "--algo", "supermimic", "--data_dir", "unread"]
    for k, (extra, want) in enumerate((([], 0.0), (["--target_rate_w", "4"], 4.0))):
        monkeypatch.setattr(sys, "argv", [*base, *extra, "--log_dir", str(tmp_path / str(k))])
        with pytest.raises(_Stop):
            ta.main()
        assert seen["call"].arguments["target_rate_w"] == want


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "stick", "reference.h5")), reason="MIMIC-MJX's data not downloaded")
def test_target_rate_w_in_a_bodys_task():
    """build puts the weight in the task's reward settings, scaled by nothing else; the stick insect's actuators are all torques (their change
    over half their control range: the action's change), the fly's all position servos (over the joints' kernel width)."""
    for animal, servos in (("stick", False), ("fly", True)):
        plain = build(animal, [0], DATA, "cpu")[3]
        on = build(animal, [0], DATA, "cpu", target_rate_w=4.0, reward_std_scale=0.7, action_scale=0.3)[3]
        assert plain.cfg.w_target_rate == 0.0 and on.cfg.w_target_rate == 4.0 and bool(on.servo.all()) is servos and bool(on.servo.any()) is servos
        k = 0.3 / (0.7 * RewardCfg().joint_std) if servos else 1.0
        assert torch.allclose(on.rate_k, torch.full((on.nu,), k)), animal
        # a change of the action by 1 on one actuator moves its target by its width's share: the cost is that actuator's weight over nu
        a, ap = torch.zeros(1, on.nu), torch.zeros(1, on.nu)
        a[0, 0] = 1.0
        assert float(on.smooth_cost(a, ap)) == pytest.approx(4.0 * k**2 / on.nu, rel=1e-5)
        f = torch.zeros(1, dtype=torch.long)
        du = (on.setpoint(a, f) - on.setpoint(ap, f))[0, 0]  # the target's change in the actuator's own unit, by the mapping
        width = 0.7 * RewardCfg().joint_std if servos else float((on.u_hi - on.u_lo)[0]) / 2
        assert float(du) / width == pytest.approx(k, rel=1e-5)


def test_target_rate_w_cost_and_reward():
    """The cost by hand: the mean over the actuators of (the target's change over its width), squared, times the weight: a torque
    actuator's width half its control range (the action's unit), a position servo's the joints' kernel width over the action
    scale; in the reward with a minus sign in the place of the action-rate term; nothing else of the reward changes, and weight 0
    leaves the reward bit for bit as it was."""
    w = 4.0
    _, _, task0 = make_task()
    _, _, task = make_task(dataclasses.replace(RewardCfg(), w_target_rate=w))
    # the hip is a torque actuator (not a servo): width 1; the knee a servo: 0.5 rad of target per unit of action over 0.4 rad
    assert task.servo.tolist() == [False, True]
    assert torch.allclose(task.rate_k, torch.tensor([1.0, 0.5 / 0.4]))
    _, _, wide = make_task(dataclasses.replace(RewardCfg(), w_target_rate=w, joint_std=0.8))
    assert torch.allclose(wide.rate_k, torch.tensor([1.0, 0.5 / 0.8]))  # the servo's width follows the kernel (--reward_std_scale)
    filt = activation_filter(body(), SUBSTEPS)
    small = AnimalTask("rat", body(), task.kin, task.ref, [2], CTRL_DT, *filt, dataclasses.replace(RewardCfg(), w_target_rate=w), 0.3)
    assert torch.allclose(small.rate_k, torch.tensor([1.0, 0.3 / 0.4]))  # and the action scale
    # a torque range of [-3, 1]: the action still costs per unit of it (a control change of half the range, 2, over the width 2)
    xml = BODY.replace('gainprm="5" ctrlrange="-1 1"', 'gainprm="5" ctrlrange="-3 1"')
    assert xml != BODY
    skew = AnimalTask("rat", mujoco.MjModel.from_xml_string(xml), task.kin, task.ref, [2], CTRL_DT, *filt,
                      dataclasses.replace(RewardCfg(), w_target_rate=w), 0.5)  # fmt: skip
    f3 = torch.zeros(1, dtype=torch.long)
    du = (skew.setpoint(torch.tensor([[0.5, 0.0]]), f3) - skew.setpoint(torch.tensor([[0.0, 0.0]]), f3))[0, 0]
    assert float(du) == pytest.approx(1.0) and float(skew.rate_k[0]) == 1.0  # half of the range's half: 0.5 action units, 1 control unit
    assert float(skew.smooth_cost(torch.tensor([[0.5, 0.0]]), torch.zeros(1, 2))) == pytest.approx(w * 0.5**2 / 2)
    a, ap = torch.tensor([[0.5, -0.2], [0.0, 0.3], [1.0, 1.0]]), torch.tensor([[0.1, 0.4], [0.0, 0.3], [0.0, 1.0]])
    d = a - ap
    hand = w * ((1.0 * d[:, 0]).pow(2) + (1.25 * d[:, 1]).pow(2)) / 2
    assert torch.allclose(task.smooth_cost(a, ap), hand, atol=1e-6)
    assert float(task.smooth_cost(a[1:2], ap[1:2])) == 0.0  # no change costs nothing
    # the gradient of the cost in the action: 2 w k_j^2 (a - a')_j / nu
    x = a.clone().requires_grad_(True)
    g = torch.autograd.grad(task.smooth_cost(x, ap).sum(), x)[0]
    assert torch.allclose(g, 2 * w * torch.tensor([1.0, 1.25]).pow(2) * d / 2, atol=1e-6)
    # the reward: minus the cost, per second, in the place of the task's action-rate term (0.05 times the mean squared change)
    S = (task.ref.qpos[:3], task.ref.qvel[:3], torch.zeros(3, 1))
    f, fk = torch.arange(3), task.kin(S[0])
    r, r0 = task.reward(S, fk, f, a, ap), task0.reward(S, fk, f, a, ap)
    old = task0.cfg.w_action_rate * d.pow(2).mean(-1)
    assert torch.allclose(r, r0 - old * CTRL_DT - hand * CTRL_DT, atol=1e-7)
    # the same numbers worked out: at the reference every kernel is 1 and their weights sum to 5.25; the terms left are the action's
    # change and the control's: the task's own -0.05 and -0.02 (its settings, pinned), or minus 4 per squared width of the change
    assert torch.allclose(r0, torch.tensor([0.0523410, 0.0524910, 0.0520500]), atol=1e-6)  # (5.25 - 0.05 d^2 - 0.02 a^2) / 100: d^2 .26, 0, .5
    assert torch.allclose(r, torch.tensor([0.0380210, 0.0524910, 0.0323000]), atol=1e-6)  # (5.25 - 4 (0.16 + 0.5625) / 2 - ...) / 100
    assert (RewardCfg().w_action_rate, RewardCfg().w_ctrl, RewardCfg().w_target_rate) == (-0.05, -0.02, 0.0)
    # its gradient in the action, the state held: minus the cost's and the control term's (the old action-rate term is gone)
    x = a.clone().requires_grad_(True)
    g = torch.autograd.grad(task.reward(S, fk, f, x, ap).sum(), x)[0]
    want = (-2 * w * torch.tensor([1.0, 1.25]).pow(2) * d / 2 + 2 * task.cfg.w_ctrl * a / 2) * CTRL_DT
    assert torch.allclose(g, want, atol=1e-7)
    # and in the action before (an imagined window's earlier action): the cost's, 2 w k^2 (a - a') / nu with the other sign, per second
    x = ap.clone().requires_grad_(True)
    g = torch.autograd.grad(task.reward(S, fk, f, a, x).sum(), x)[0]
    assert torch.allclose(g, 2 * w * torch.tensor([1.0, 1.25]).pow(2) * d / 2 * CTRL_DT, atol=1e-7)
    # weight 0: the reward bit for bit, whatever the new field
    assert torch.equal(task0.reward(S, fk, f, a, ap), make_task(dataclasses.replace(RewardCfg(), w_target_rate=0.0))[2].reward(S, fk, f, a, ap))
    k = RewardCfg().scaled(0.5)
    assert k.w_target_rate == 0.0 and dataclasses.replace(RewardCfg(), w_target_rate=2.0).scaled(0.5).w_target_rate == 2.0


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_target_rate_w_update(algo, monkeypatch):
    """On, the policy's update changes, off it is the same bit for bit, for both algorithms; SuperTrack's enters its loss as its mean over
    the imagined steps and the batch (at actuators of width 1, what --st_rate_w at that weight adds), and it needs the
    task-weighted loss and no other term on the action's change."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    over = {"st_task_weights": True} if algo == "supertrack" else {}
    base = _policy_after_update_with(algo, RewardCfg(), **over)
    assert torch.equal(base, policy_after_update(algo, **over))
    assert torch.equal(_policy_after_update_with(algo, dataclasses.replace(RewardCfg(), w_target_rate=0.0), **over), base)
    on = _policy_after_update_with(algo, dataclasses.replace(RewardCfg(), w_target_rate=8.0), **over)
    assert not torch.equal(on, base)
    if algo == "supermimic":
        return
    # every actuator of width 1 (task.rate_k = 1): the cost of weight w is --st_rate_w's of weight w, up to rounding
    agent, buffer, task = _agent_with("supertrack", dataclasses.replace(RewardCfg(), w_target_rate=3.0), st_task_weights=True)
    task.rate_k = torch.ones_like(task.rate_k)
    torch.manual_seed(1)
    agent.policy_update(buffer)
    mine = torch.cat([q.detach().reshape(-1) for q in agent.pi.parameters()])
    theirs = _policy_after_update_with("supertrack", RewardCfg(), st_task_weights=True, st_rate_w=3.0)
    assert torch.allclose(mine, theirs, atol=1e-5) and not torch.allclose(mine, base, atol=1e-5)
    # at the actuators' own widths it is another update
    own = _policy_after_update_with("supertrack", dataclasses.replace(RewardCfg(), w_target_rate=3.0), st_task_weights=True)
    assert not torch.allclose(own, theirs, atol=1e-5)
    for bad in ({"st_task_weights": False}, {"st_task_weights": True, "st_rate_w": 0.05}):
        with pytest.raises(ValueError):
            _agent_with("supertrack", dataclasses.replace(RewardCfg(), w_target_rate=3.0), **bad)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_target_rate_w_under_the_rollout_clip(algo, monkeypatch):
    """--pi_rollout_clip with the term: a clip that caps nothing leaves the update as it is, one that caps changes it, and the term is in the
    capped gradient too (the capped update with it is not the one without)."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    over = {"st_task_weights": True} if algo == "supertrack" else {}
    cfg = dataclasses.replace(RewardCfg(), w_target_rate=8.0)
    on = _policy_after_update_with(algo, cfg, **over)
    assert torch.equal(_policy_after_update_with(algo, cfg, pi_rollout_clip=1e9, **over), on)
    capped = _policy_after_update_with(algo, cfg, pi_rollout_clip=0.5, **over)
    assert not torch.equal(capped, on)
    assert not torch.equal(capped, policy_after_update(algo, pi_rollout_clip=0.5, **over))


@pytest.mark.parametrize("tap", [False, True])
def test_target_rate_w_in_the_imagined_step(tap, monkeypatch):
    """SuperMimic's imagined step carries the cost in its reward, and the cost reaches the action before it (an earlier action of the window),
    in the plain step and in --pi_rollout_clip's: with a policy whose output is zero whatever its input, the reward's gradient in
    the action before is the cost's alone, 2 w k^2 (a - a') / nu per second with the sign of the change."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    agent, buffer, task = _agent_with("supermimic", dataclasses.replace(RewardCfg(), w_target_rate=4.0))
    with torch.no_grad():
        for q in agent.pi.net[-1].parameters():
            q.zero_()
    S, f, ap = agent._starts(buffer)
    ap = ap.clone().requires_grad_(True)
    fk = task.kin(S[0])
    out = agent._imag_step_tap(S, fk, f, ap, torch.zeros_like(ap)) if tap else agent._imag_step(S, fk, f, ap)
    a, r = out[3], out[4]
    assert torch.equal(a, torch.zeros_like(a)) and float(ap.detach().abs().max()) > 0
    want = 2 * 4.0 * task.rate_k.pow(2) * (a.detach() - ap.detach()) / task.nu * CTRL_DT
    assert torch.allclose(torch.autograd.grad(r.sum(), ap)[0], want, atol=1e-7)


def test_target_rate_w_cost_of_a_window():
    """SuperTrack's term by hand: over an imagined window (H, B, nu) of actions after the action before it, the mean over the steps and
    the batch of each step's cost; the window's first change is from the action before it."""
    _, _, task = make_task(dataclasses.replace(RewardCfg(), w_target_rate=2.0))
    A = torch.tensor([[[0.5, 0.0], [0.0, 0.2]], [[0.5, 0.4], [1.0, 0.2]]])  # (H 2, B 2, nu 2)
    start = torch.tensor([[0.1, 0.0], [0.0, -0.2]])
    got = task.smooth_cost(A, torch.cat((start.unsqueeze(0), A[:-1]))).mean()
    k2 = torch.tensor([1.0, 1.25**2])
    want = 0.0
    for h in range(2):
        for b in range(2):
            before = start[b] if h == 0 else A[0, b]
            want += float(2.0 * (k2 * (A[h, b] - before).pow(2)).mean())
    assert abs(float(got) - want / 4) < 1e-6


def _replay_with_ends():
    """A small replay (T 12, 3 worlds, one-number states): world 0 terminates at ages 4 and 10, world 1 only times out (age 6),
    world 2 terminates at age 8 and again at age 10, after an episode of two stored steps."""
    buf = ReplayBuffer(12, 3, "cpu", fields={"s": (1, torch.float32), "s1": (1, torch.float32), "u": (1, torch.float32),
                                              "done": ((), torch.bool), "term": ((), torch.bool)}, priorities=False)  # fmt: skip
    buf.window_keys = ("s", "s1", "u", "done")
    ends = {0: {4: True, 10: True}, 1: {6: False}, 2: {8: True, 10: True}}  # age: a termination (True) or a time-out
    for t in range(12):
        done = torch.tensor([t in ends[w] for w in range(3)])
        term = torch.tensor([ends[w].get(t, False) for w in range(3)])
        x = torch.tensor([[100.0 * w + t] for w in range(3)])
        buf.add({"s": x, "s1": x + 1, "u": x, "done": done, "term": term})
    return buf, ends


def test_wm_pos_lead(monkeypatch):
    """--wm_pos_lead: SuperTrack's acceleration model's position lead; by default build's value (the constant acceleration's
    over the body's physics steps), refused for SuperMimic (its model predicts increments) and outside (0, 1]; the model
    keeps the run's lead (MjcfWorldModel's integration with it: test_mjcf_pos_lead)."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    st = ["--animal", "rat", "--clip_ids", "0", "--algo", "supertrack"]
    assert parse_args(st).wm_pos_lead is None
    assert parse_args(st + ["--wm_pos_lead", "1.0"]).wm_pos_lead == 1.0
    for bad in (["--wm_pos_lead", "0"], ["--wm_pos_lead", "-0.5"], ["--wm_pos_lead", "1.5"]):
        with pytest.raises(SystemExit):
            parse_args(st + bad)
    with pytest.raises(SystemExit):  # SuperTrack's option
        parse_args(["--animal", "rat", "--clip_ids", "0", "--algo", "supermimic", "--wm_pos_lead", "1.0"])
    assert world_model_lead(parse_args(st), 0.6) == 0.6 and world_model_lead(parse_args(st + ["--wm_pos_lead", "1"]), 0.6) == 1.0
    assert world_model_lead(argparse.Namespace(), 0.55) == 0.55  # a run of before the option: build's value
    _, lay, task = make_task()
    args = argparse.Namespace(**{**vars(parse_args(st)), "wm_hidden": [32, 32]})
    agents = {}
    for lead in (0.6, 1.0):
        torch.manual_seed(0)
        agents[lead] = ANIMAL_ALGOS["supertrack"](args, task, dataclasses.replace(lay, dt=CTRL_DT), "cpu", lead)
        wm = agents[lead].wm
        assert wm._lead == (lead != 1.0) and (not wm._lead or float(wm.pos_lead) == pytest.approx(lead))
    # the step the lead changes, by hand: the same network gives the same next velocity v'; the joints then advance by
    # dt (v + L (v' - v)): by dt v' at 1.0 (the velocity at the step's end), and by dt 0.4 (v' - v) less at 0.6
    with torch.no_grad():
        for q_ in agents[0.6].wm.parameters():  # a network away from its zero start: accelerations that are not zero
            q_.add_(0.05 * torch.randn_like(q_))
        agents[1.0].wm.load_state_dict({k: v for k, v in agents[0.6].wm.state_dict().items() if k != "pos_lead"})
        f = torch.tensor([3, 17, 45])
        S = (task.ref.qpos[f], task.ref.qvel[f], torch.zeros(3, task.na))
        u = task.setpoint(0.3 * torch.ones(3, task.nu), f)
        (q6, v6, _), (q1, v1, _) = agents[0.6].wm_step(S, u), agents[1.0].wm_step(S, u)
    v0 = S[1]
    assert torch.equal(v6, v1) and float((v1 - v0).abs().max()) > 1e-4  # the same next velocity, and a change of velocity to lead
    jq, jd = task.jq, task.jd
    assert torch.allclose(q1[:, jq], S[0][:, jq] + CTRL_DT * v1[:, jd], atol=1e-6)  # lead 1.0: one semi-implicit Euler step
    assert torch.allclose(q6[:, jq], S[0][:, jq] + CTRL_DT * (v0[:, jd] + 0.6 * (v1[:, jd] - v0[:, jd])), atol=1e-6)
    assert torch.allclose(q1[:, jq] - q6[:, jq], CTRL_DT * 0.4 * (v1[:, jd] - v0[:, jd]), atol=1e-6)


HOPPER = """<mujoco><worldbody>
  <geom name="floor" type="plane" pos="0 0 -0.005" size="5 5 0.01"/>
  <body name="b" pos="0 0 0"><freejoint/>
    <geom name="s" type="sphere" size="0.01" pos="0.02 0 0"/>
    <geom name="c" type="capsule" size="0.004 0.015" pos="-0.02 0.01 0" euler="35 20 0"/>
    <geom name="e" type="ellipsoid" size="0.012 0.006 0.003" pos="0 -0.02 0.004" euler="10 50 30"/>
    <geom name="y" type="cylinder" size="0.006 0.01" pos="0.01 0.02 0.002" euler="60 15 0"/>
    <geom name="x" type="box" size="0.008 0.005 0.003" pos="-0.01 -0.01 -0.002" euler="25 40 70"/>
    <geom name="ghost" type="sphere" size="0.05" pos="0 0 -0.03" contype="0" conaffinity="0"/>
  </body></worldbody></mujoco>"""


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
    args = parse_args(["--animal", "rat", "--clip_ids", "0"])  # the reference's velocities: the dataset's by default
    assert args.ref_qvel == "stored" and REF_QVEL == ("stored", "positions")
    assert parse_args(["--animal", "rat", "--clip_ids", "0", "--ref_qvel", "positions"]).ref_qvel == "positions"
    with pytest.raises(SystemExit):
        parse_args(["--animal", "rat", "--clip_ids", "0", "--ref_qvel", "zeros"])


def test_named_configurations():
    """The benchmark's configurations: a run's cap, its learning rate decayed over it, a kept checkpoint at the rule's spacing."""
    stop_rule = load(os.path.join(BENCH, "watch.py"), "animal_watch")
    for animal in ("stick", "fly", "rat"):
        for algo in ("supertrack", "supermimic"):
            a = parse_args(["--config", f"{animal}-{algo}"])
            assert (a.animal, a.algo) == (animal, algo) and a.lr_decay_iters == a.iterations and a.ckpt_keep
            assert a.ckpt_every == stop_rule.RULE[animal]["every"] and a.iterations % a.ckpt_every == 0
            assert a.log_every == parse_args(["--animal", animal, "--clip_ids", "0"]).log_every  # the trainer's (the deadline's rows)


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


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "rat", "reference.h5")), reason="MIMIC-MJX's data not downloaded")
@pytest.mark.parametrize("animal,factor", [("rat", 10.0), ("fly", 0.1), ("stick", 10.0)])
def test_reference_velocities(animal, factor):
    """At the pinned revision the files' velocities are not those of their positions (the root's linear velocity ``factor``
    times); with --ref_qvel positions the task's reference velocities take each of its frames to the next."""
    pytest.importorskip("h5py")
    from ndm_gym.motion.sources.mimic_mjx import load_clips
    from ndm_gym.robots.animals import ANIMALS
    from ndm_gym.train_animal import build

    if not os.path.exists(os.path.join(DATA, animal, "reference.h5")):
        pytest.skip(f"{animal}'s data not downloaded")
    data = load_clips(DATA, animal, [0, 1])
    fd = np.diff(data["qpos"][:, :, :3].astype(np.float64), axis=1) * ANIMALS[animal].mocap_hz
    stored = data["qvel"][:, :-1, :3].astype(np.float64)
    assert abs((stored * fd).sum() / (fd * fd).sum() / factor - 1) < 0.01
    model, _, _, task = build(animal, [0, 1], DATA, "cpu", ref_qvel="positions")
    ref, dt = task.ref, ANIMALS[animal].ctrl_dt
    q, v = ref.qpos.double().numpy(), ref.qvel.double().numpy()
    inner = np.flatnonzero((torch.arange(ref.frames) < ref.end).numpy())  # every frame but each clip's last
    for t in inner[:: max(1, len(inner) // 200)]:
        q1 = q[t].copy()
        mujoco.mj_integratePos(model, q1, v[t], dt)
        assert np.allclose(q1, q[t + 1], atol=1e-4)
    from_file = build(animal, [0, 1], DATA, "cpu")[3]
    assert torch.equal(from_file.ref.qpos, ref.qpos) and not torch.allclose(from_file.ref.qvel, ref.qvel, atol=1e-2)
    assert from_file.joint_speed != task.joint_speed  # the joint-velocity reward's scale follows the reference's velocities


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "rat", "reference.h5")), reason="MIMIC-MJX's data not downloaded")
@pytest.mark.parametrize("animal,factor", [("rat", 10.0), ("fly", 0.1), ("stick", 10.0)])
def test_reference_velocities_by_least_squares(animal, factor):
    """--ref_qvel on the dataset's clips: positions gives the velocities with which a frame integrated over a control step is
    the next, in the same positions and frames as stored; the dataset's stored root velocity is ``factor`` times them (it was
    inferred with the physics step in the frame interval's place), which is why the option exists."""
    pytest.importorskip("h5py")
    model, _, _, stored = build(animal, [0, 1], DATA, "cpu")
    own = build(animal, [0, 1], DATA, "cpu", ref_qvel="positions")[3]
    assert torch.equal(own.ref.qpos, stored.ref.qpos) and torch.equal(own.ref.end, stored.ref.end) and own.ref.qvel.shape == stored.ref.qvel.shape
    q, v = own.ref.qpos.double().numpy(), own.ref.qvel.double().numpy()
    inner = (torch.arange(own.ref.frames) < own.ref.end).numpy()  # every frame but a clip's last
    worst = 0.0
    for i in np.flatnonzero(inner)[::7]:
        q1 = q[i].copy()
        mujoco.mj_integratePos(model, q1, v[i], own.ctrl_dt)
        worst = max(worst, float(np.abs(q1 - q[i + 1]).max()))
    assert worst < 1e-5  # float32 clips
    rd = own.rd
    a, b = own.ref.qvel[inner, rd : rd + 3].flatten().double(), stored.ref.qvel[inner, rd : rd + 3].flatten().double()
    k = float((a * b).sum() / (a * a).sum())  # least squares: stored = k x the positions' own
    assert 0.85 * factor < k < 1.05 * factor and float(torch.corrcoef(torch.stack((a, b)))[0, 1]) > 0.95
    with pytest.raises(ValueError):
        build(animal, [0], DATA, "cpu", ref_qvel="zeros")
REFERENCE_BEFORE = {("rat", "stored"): "2a4f14ba9b133b47", ("rat", "positions"): "8153182b571a49f4", ("fly", "stored"): "1f2f1c39f4b975be",
                    ("fly", "positions"): "bbc3527441cff863", ("stick", "stored"): "3279c94e0931c8fe",
                    ("stick", "positions"): "e939786dbfaf905f"}  # fmt: skip


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "rat", "reference.h5")), reason="MIMIC-MJX's data not downloaded")
@pytest.mark.parametrize("animal", ["rat", "fly", "stick"])
def test_reference_default_is_unchanged(animal):
    """The reference of every clip, with each kind of velocities, is byte for byte the one its digest records."""
    pytest.importorskip("h5py")
    for ref_qvel in REF_QVEL:
        task = build(animal, list(range(20)), DATA, "cpu", ref_qvel=ref_qvel)[3]
        h = hashlib.sha256()
        for name in ("qpos", "qvel", "bodies", "start", "end"):
            h.update(getattr(task.ref, name).numpy().tobytes())
        assert h.hexdigest()[:16] == REFERENCE_BEFORE[animal, ref_qvel]


def test_checkpoints_of_one_run_and_resume():
    """evaluate_animal --ckpts refuses checkpoints of another run's task (its reference's velocities, clips, body or the task's
    settings), a checkpoint of before --ref_qvel counting as its default; train_animal --resume refuses a checkpoint of other
    reference velocities the same way."""
    from ndm_gym.evaluate_animal import same_run
    from ndm_gym.train_animal import resume_mismatch

    st = ["--animal", "rat", "--clip_ids", "0", "--algo", "supertrack"]
    a = vars(parse_args(st))
    assert same_run(a, dict(a)) is None
    for k, v in (("ref_qvel", "positions"), ("clip_ids", ["1"]), ("animal", "fly"), ("action_scale", 0.25), ("goal_stride", 2),
                 ("reward_std_scale", 0.7), ("fused_fk", True)):  # fmt: skip
        assert k in same_run(a, {**a, k: v})
    old = {k: v for k, v in a.items() if k != "ref_qvel"}
    assert same_run(a, old) is None and same_run(old, a) is None
    p = parse_args(st + ["--ref_qvel", "positions"])
    assert resume_mismatch(vars(p), p) is None and "--ref_qvel" in resume_mismatch(vars(p), parse_args(st))
    assert "--ref_qvel" in resume_mismatch({k: v for k, v in vars(p).items() if k != "ref_qvel"}, p)  # before the option: stored


class _Stop(Exception):
    """Raised by a stand-in where a GPU path would begin: the test has what it needs."""


def test_benchmark_scripts_build_their_evaluators(monkeypatch, tmp_path):
    """watch.py's stop rule and final.py's readings build their evaluators from the run's own arguments, on its own clips: both
    run up to their first evaluator, with the checkpoint read a stand-in and the Evaluator a recorder."""
    import ndm_gym.evaluate_animal as ea

    run = {**vars(parse_args(["--animal", "rat", "--clip_ids", "1-2", "--algo", "supertrack", "--ref_qvel", "positions"])), "ckpt_every": 500,
           "ckpt_keep": True, "iterations": 40000}  # fmt: skip
    seen = {}

    class Recorder:
        def __init__(self, r, clip_ids, repeats, joint_noise, seed, data_dir=None, device="cuda:0"):
            seen["run"], seen["clips"] = r, clip_ids
            raise _Stop

    monkeypatch.setattr(ea, "Evaluator", Recorder)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"args": run, "iteration": 500, "wall_time": 1.0})
    d = tmp_path / "rat_supertrack_s0"
    d.mkdir()
    (d / "args.json").write_text(json.dumps(run))
    (d / "ckpt_000500.pt").write_text("")
    for script, argv in (("watch.py", [str(d), "--poll", "0.1", "--max_hours", "0.01"]), ("final.py", [str(d), "--stop", "500", "--reason", "rule"])):
        seen.clear()
        m = load(os.path.join(BENCH, script), "animal_" + script[:-3])
        monkeypatch.setattr(sys, "argv", [script, *argv])
        with pytest.raises(_Stop):
            m.main()
        assert seen["run"]["ref_qvel"] == "positions" and seen["run"]["clip_ids"] == ["1-2"] and seen["clips"] is None, script


def test_fused_kinematics_match():
    model, lay, task = make_task()
    fused = AnimalKinematics(lay, [1, 2, 3])
    fused.fused = True
    q = task.ref.qpos[::7]
    assert torch.allclose(task.kin(q), fused(q), atol=1e-5)


def test_evaluation_summary():
    """The errors of the completed episodes apart from the ended ones'; none when every episode ended."""
    steps = torch.tensor([10.0, 4.0, 10.0])
    sums = {k: torch.tensor([1.0, 4.0, 3.0]) for k in ERRORS}
    scale = dict.fromkeys(ERRORS, 1000.0)
    m = summarize(sums, steps, torch.tensor([False, True, False]), scale)
    assert (m["episodes"], m["ended"], m["completed"]) == (3, 1, 2) and abs(m["ended_share"] - 1 / 3) < 1e-6
    assert abs(m["body_mm"] - 200.0) < 1e-3 and abs(m["lived"]["body_mm"] - 1000 * (0.1 + 1.0 + 0.3) / 3) < 1e-3
    assert summarize(sums, steps, torch.ones(3, dtype=torch.bool), scale)["body_mm"] is None


def test_stop_rule():
    """The rule on three-evaluation means over the body's window, from its first iteration on; the cap; no error, no stop."""
    w = load(os.path.join(BENCH, "watch.py"), "animal_watch")
    r = w.RULE["stick"]
    flat = {i: (0.0, 1.0) for i in range(r["every"], 8001, r["every"])}
    assert w.decide(flat, "stick", 8000) == ("STOP", r["first"])
    falling = {i: (0.0, 1.0 * 0.95 ** (i / 250)) for i in range(250, 8001, 250)}  # 5 % better every evaluation
    assert w.decide(falling, "stick", 8000) == ("CAP", 8000)
    assert w.decide({i: v for i, v in falling.items() if i < 8000}, "stick", 8000) == ("CONTINUE", None)
    # the error flat from 4,000 on, the ended share falling by 3 points per 1,000 iterations until 5,000: the first
    # three-evaluation mean less than a point below the one 1,000 iterations before is 6,000's
    q = {i: (max(0.05, 0.2 - 0.03 * i / 1000), 4000 / min(i, 4000)) for i in range(250, 8001, 250)}
    assert w.decide(q, "stick", 8000) == ("STOP", 6000)
    nan = {i: (1.0, math.nan) for i in range(250, 8001, 250)}  # every episode ended
    assert w.decide(nan, "stick", 8000) == ("CAP", 8000)
    rat = {i: (0.2, 10.0) for i in range(500, 40001, 500)}
    assert w.decide(rat, "rat", 40000) == ("STOP", 10000)
    # the quick evaluations: 100 episodes on ten clips, the rat's 200; their seed other than the final evaluations'
    assert {b: 10 * (1 + r) for b, r in w.QUICK_REPEATS.items()} == {"stick": 100, "fly": 100, "rat": 200}
    # stop.json's reasons are final.py's (the watcher's verdicts, and "end" for a training that ended without either)
    f = load(os.path.join(BENCH, "final.py"), "animal_final")
    assert set(w.REASON) == {"STOP", "CAP"} and sorted({*w.REASON.values(), "end"}) == sorted(f.REASONS)
    assert w.QUICK_SEED != f.FINAL["seed"]


def test_time_to_target():
    """Read backward from the stop: the first checkpoint of the run of checkpoints meeting the target that ends at the stop."""
    f = load(os.path.join(BENCH, "final.py"), "animal_final")

    def m(it, ended, body):
        return {"iteration": it, "minutes": it / 100, "ended_share": ended, "body_mm": body, "size_mm": 10.0}

    evals = {250: m(250, 0.0, 0.4), 500: m(500, 0.02, 0.4), 750: m(750, 0.0, 0.45), 1000: m(1000, 0.005, 0.5)}
    seen = []

    def evaluate(it):
        seen.append(it)
        return evals[it]

    t = f.time_to_target(evaluate, 1000, 250, evals[1000])
    assert (t["iteration"], seen) == (750, [750, 500]) and [c["target"] for c in t["checkpoints"]] == [False, True, True]
    assert f.time_to_target(evaluate, 1000, 250, m(1000, 0.0, 0.6))["iteration"] is None  # the stop misses: not reached
    assert f.time_to_target(lambda it: evals[it], 250, 250, evals[250])["iteration"] == 250
    assert f.meets(m(0, 0.01, 0.5)) and not f.meets(m(0, 0.0, None))


def test_download_checks_sha256(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_mimic_mjx", os.path.join(REPO, "tools", "download_mimic_mjx.py"))
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


def test_motion_check_by_hand():
    """motion.py: the differences' RMS (a cubic's third difference over dt^3 is 6, a square's second 2, an impulse's third
    the RMS of 1, -3, 3, -1: the square root of 5), the share of a pure 30 Hz velocity above 27 Hz (all of it) and of a 27 Hz
    one (none: above, strictly), the limit shares of a hinge at its low limit for 3 of 10 steps (new where the reference is
    not there), an ended episode left out, the ratios and the check, which a limit alone fails."""
    m = load(os.path.join(BENCH, "motion.py"), "animal_motion")
    t = np.arange(12, dtype=np.float64)[:, None]
    assert m.diff_rms(t**3, 3, 1.0) == pytest.approx(6.0) and m.diff_rms((0.5 * t) ** 3, 3, 0.5) == pytest.approx(6.0)
    assert m.diff_rms(t**2, 2, 1.0) == pytest.approx(2.0) and m.diff_rms(t, 3, 1.0) == 0.0
    assert m.diff_rms(np.array([[0.0], [0], [0], [1], [0], [0], [0]]), 3, 1.0) == pytest.approx(math.sqrt(5))
    dt = 0.01

    def pure(hz):  # positions whose first difference over dt is sin at hz: 100 steps, whole cycles in 1 s
        v = np.sin(2 * np.pi * hz * dt * np.arange(100))[:, None]
        return np.concatenate((np.zeros((1, 1)), np.cumsum(v * dt, 0)))

    high, total = m.high_power(pure(30), dt, 27)
    assert total > 0 and high == pytest.approx(total)
    high, total = m.high_power(pure(27), dt, 27)
    assert total > 0 and high == pytest.approx(0.0, abs=1e-9 * total)
    # limits: one limited hinge in [-1, 1]; the policy at its low limit (within 1 % of the range: 0.02 rad) for steps 0-2 of 10
    rq = np.zeros((10, 1))
    pq = rq.copy()
    pq[:3] = -0.995
    assert m.limit_shares(pq, rq, np.array([-1.0]), np.array([1.0])) == (0.3, 0.0, 0.3)
    assert m.limit_shares(pq, pq, np.array([-1.0]), np.array([1.0])) == (0.3, 0.3, 0.0)  # the reference there too: none new
    # a record of three worlds over 10 steps of a 10-frame reference: two hinges, the second limited and at its low limit in
    # the reference's frame 0; hinge 1 moves at 40 Hz (its jerk far above a step's at the limit); world 1 ended
    T = 10
    ref = np.zeros((T, 3), np.float32)
    ref[:, 1] = 0.2 * np.sin(2 * np.pi * 40 * dt * np.arange(T))
    ref[0, 2] = -0.995
    pol = np.repeat(ref[None], 3, 0).copy()
    pol[0, :3, 2] = -0.995  # world 0 at hinge 2's low limit for steps 0-2, the reference only at 0
    pol[1] += 5.0  # the ended world: far off, must not count
    record = {"qpos": pol, "frame": np.tile(np.arange(T), (3, 1)), "lived": np.array([T, 4, T]), "ended": np.array([False, True, False]),
              "ref_qpos": ref}
    args = (np.array([1, 2]), np.array([False, True]), np.array([-1.0]), np.array([1.0]), dt, 27, 3)
    out = m.measure(record, *args)
    assert out["episodes"] == 2 and out["reference"]["limit"] == pytest.approx(0.1)
    assert out["policy"]["limit"] == pytest.approx(0.2) and out["policy"]["new_limit"] == pytest.approx(0.1)  # (0.3 + 0.1) / 2; (0.2 + 0) / 2
    assert 1.0 < out["ratio"]["jerk"] <= 1.5 and out["ratio"]["high_share"] <= 2.0 and not out["passes"]  # the limit alone fails it
    record["qpos"][0, 1:3, 2] = 0.0  # world 0 now the reference itself
    out = m.measure(record, *args)
    assert out["ratio"]["jerk"] == pytest.approx(1.0) and out["ratio"]["high_share"] == pytest.approx(1.0) and out["passes"]
    # world 0's hinge 1 at three times the reference's amplitude: the jerk alone fails the check (the share stays the
    # reference's, every hinge-1 frequency being 40 Hz); the run's share is power over power, not the episodes' mean share
    record["qpos"][0, :, 1] = 3 * ref[:, 1]
    out = m.measure(record, *args)
    assert out["ratio"]["jerk"] > 1.5 and out["ratio"]["high_share"] <= 2.0 and not out["passes"]
    assert out["policy"]["limit"] <= out["reference"]["limit"]
    eps = [m.high_power(record["qpos"][w].astype(np.float64)[:, [1, 2]], dt, 27) for w in (0, 2)]
    pooled = sum(h for h, _ in eps) / sum(t for _, t in eps)
    assert out["policy"]["high_share"] == pytest.approx(pooled) and abs(pooled - np.mean([h / t for h, t in eps])) > 1e-3
    record["ended"][:] = True
    with pytest.raises(ValueError, match="no episode"):
        m.measure(record, *args)


def test_summary_tables(tmp_path):
    """summarize.py: every seed and the mean of a body's runs (a run without completed episodes counted apart), SuperTrack's
    rows first, each with the options that change its result (the smoothness term's weight; empty without it); the speed rows
    with their ratio and open-loop errors."""
    s = load(os.path.join(BENCH, "summarize.py"), "animal_summarize")
    m = {"ended_share": 0.0, "body_mm": 0.5, "end_eff_mm": 0.6, "joint_deg": 4.0}
    for algo, body in (("supertrack", 0.5), ("supermimic", None)):
        for seed in (0, 1, 2):
            run = tmp_path / "logs" / f"stick_{algo}_s{seed}"
            run.mkdir(parents=True)
            train = {**m, "body_mm": body}
            res = {"stop_iteration": 2000 + 250 * seed, "stop_reason": "rule", "final": {"train": train, "heldout": m},
                   "time_to_target": {"minutes": 1.0 if body else None}, "it_per_s_301_to_stop": 10.0}  # fmt: skip
            (run / "result.json").write_text(json.dumps(res))
    rows = s.table(str(tmp_path / "logs"), [0, 1, 2]).splitlines()
    stick = [r for r in rows if r.startswith("| stick insect")]
    assert stick[0].startswith("| stick insect | SuperTrack | 0 | 2,000 (rule)") and "SuperMimic" in stick[4]
    assert "0.5 ± 0 mm" in stick[3] and "| - mm |" in stick[7] and "not reached" in stick[4]
    assert stick[0].endswith("| 10.00 |  |") and stick[4].endswith("| 10.00 |  |") and stick[3].endswith("| 10.00 ± 0.00 | |")
    runs = [tmp_path / "logs" / f"stick_supermimic_s{seed}" for seed in (0, 1, 2)]

    # the smoothness weight (--target_rate_w) the runs trained with: recorded in the result, else from args.json, else 0 (the runs of
    # before); seeds of different weights are refused, not averaged; a row of one weight above 0 names it
    def weights(*ws, args=None):
        """The three SuperMimic runs' weights in their results (None: not recorded), and the runs' arguments where given."""
        for run, x in zip(runs, ws, strict=True):
            r = {k: v for k, v in json.loads((run / "result.json").read_text()).items() if k != "target_rate_w"}
            (run / "result.json").write_text(json.dumps({**r, **({} if x is None else {"target_rate_w": x})}))
            (run / "args.json").unlink(missing_ok=True)
        for run, a in zip(runs, args or (), strict=False):
            (run / "args.json").write_text(json.dumps(a))

    with_w = [{"target_rate_w": w} for w in (2.0, 0.0, 2.0)]
    for ws, args in (((2.0, 4.0, 2.0), None), ((None, 2.0, None), None), ((0.0, 2.0, 0.0), None), ((None, None, None), with_w)):
        weights(*ws, args=args)
        with pytest.raises(SystemExit, match="different weights of the smoothness term"):
            s.table(str(tmp_path / "logs"), [0, 1, 2])
    for ws, args, want in (((None,) * 3, None, ""), ((0.0,) * 3, None, ""), ((2.0,) * 3, None, "target_rate_w 2"),
                           ((0.5,) * 3, None, "target_rate_w 0.5"), ((None,) * 3, [{"target_rate_w": 4.0}] * 3, "target_rate_w 4"),
                           ((None,) * 3, [{"seed": 0}] * 3, "")):  # fmt: skip
        weights(*ws, args=args)
        stick = [r for r in s.table(str(tmp_path / "logs"), [0, 1, 2]).splitlines() if r.startswith("| stick insect")]
        assert stick[4].endswith(f"| 10.00 | {want} |") and stick[0].endswith("| 10.00 |  |"), (ws, args)
    assert all("missing" in r for r in rows if r.startswith("| rat") and "mean" not in r)
    speed = tmp_path / "speed"
    speed.mkdir()
    (speed / "stick_supertrack.json").write_text(json.dumps({"wm_kind": "accel", "open_loop": {"bodies_mm": {"1": 0.01, "8": 0.1, "16": 0.2}},
                                                             "rows": [{"what": "physics", "nworld": 4096, "world_steps_per_s": 1e5},
                                                                      {"what": "ndm_fp32", "nworld": 4096, "world_steps_per_s": 1e6},
                                                                      {"what": "ndm_bf16", "nworld": 4096, "world_steps_per_s": 2e6}]}))  # fmt: skip
    line = s.speed_table(str(speed)).splitlines()[2]
    assert "SuperTrack's acceleration model | 4,096 | 100,000 | 1,000,000 | 2,000,000 | 20.0x | 0.01 / 0.1 / 0.2 mm" in line
