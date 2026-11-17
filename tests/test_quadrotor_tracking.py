"""Quadrotor trajectory tracking on flights made by the drone's own model: the model against its specification, the data's
conversion and resampling, the reference's feedforward command, the state layout, the task at the reference and its
invariance to position and heading, the world model's two parameterizations against MuJoCo's step, the updates and
training's arguments; on a GPU, the simulator against MuJoCo, its step without host synchronization and training's loop
with the systems options."""

import argparse
import importlib.util
import math
import sys

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.quadrotor_algos import QUAD_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.core.rotations import quat_to_mat, so3_exp
from ndm_gym.motion.sources.drone_racing import load_flight, load_gates
from ndm_gym.robots.quadrotor import CONTROL_DT, GRAVITY, NUM_U, QuadrotorSpec, load_model
from ndm_gym.tasks.quadrotor_tracking import GOAL_FRAME_DIM, OBS_DIM, STATE_DIM, QuadTask, build_reference, flatten_state, unflatten_state
from ndm_gym.train import add_train_args
from ndm_gym.train_quadrotor import QUAD_DEFAULTS, UNSUPPORTED, parse_args
from ndm_gym.worldmodels.quadrotor import QuadWorldModel

# the arguments quadrotor training implements; the rest of ndm_gym.train's are train_quadrotor.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip
SPEC = QuadrotorSpec()
SUBSTEPS = round(CONTROL_DT / SPEC.timestep)


def fly(model, seconds: float, seed: int, yaw: float = 0.0, offset=(0.0, 0.0, 1.0)):
    """A flight of the model itself under smooth commands about the hover: the poses every control step and the commands."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    d.qpos[:3], d.qpos[3:7] = offset, (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    n = round(seconds / CONTROL_DT)
    phase, freq = rng.uniform(0, 6.28, 4), rng.uniform(0.5, 1.5, 4)
    qpos, ctrl = [d.qpos.copy()], []
    for i in range(n):
        s = np.sin(2 * math.pi * freq * i * CONTROL_DT + phase)
        d.ctrl[:] = (SPEC.weight * (1.05 + 0.25 * s[0]), 1.5 * s[1], 1.5 * s[2], 1.0 * s[3])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
        qpos.append(d.qpos.copy()), ctrl.append(d.ctrl.copy())
    qpos = torch.tensor(np.array(qpos), dtype=torch.float32)
    return {"p": qpos[:, :3], "quat": qpos[:, 3:]}, torch.tensor(np.array(ctrl), dtype=torch.float32)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = load_model(SPEC)
    (a, ctrl), (b, _) = fly(model, 1.2, 0), fly(model, 0.8, 1)
    return model, QuadTask(build_reference([a, b], SPEC, "cpu"), SPEC), ctrl


def test_model_follows_its_specification():
    model = load_model(SPEC)
    assert (model.nq, model.nv, model.nu) == (7, 6, 4) and abs(model.body_mass[1] - SPEC.mass) < 1e-9
    assert abs(SUBSTEPS * model.opt.timestep - CONTROL_DT) < 1e-12
    d = mujoco.MjData(model)
    d.ctrl[:] = (SPEC.weight, 2.0, -1.0, 0.5)  # the hover's thrust and a body-rate command
    for _ in range(round(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, d)
    assert np.allclose(d.qvel[3:6], (2.0, -1.0, 0.5), atol=0.02)  # the rate servos' time constants are 20 to 50 ms
    d = mujoco.MjData(model)
    d.ctrl[0], d.qvel[0] = SPEC.weight, 10.0
    mujoco.mj_step(model, d)
    assert abs((d.qvel[0] - 10.0) / model.opt.timestep + SPEC.drag * 10.0 / SPEC.mass) < 0.02  # the linear drag
    assert abs(d.qvel[2]) < 1e-9 and SPEC.max_thrust == pytest.approx(7.5 * SPEC.mass * GRAVITY)
    d.ctrl[0] = 2 * SPEC.max_thrust  # commands are clamped to the actuators' ranges
    mujoco.mj_forward(model, d)
    assert d.actuator_force[0] == pytest.approx(SPEC.max_thrust)


def test_conversion_and_resampling(tmp_path):
    """tools/download_drone_racing.convert reads a drone_state CSV; load_flight keeps the part in the air, every control step."""
    sys.path.insert(0, "tools")
    try:
        tool = importlib.import_module("download_drone_racing")
        convert = tool.convert
    finally:
        sys.path.remove("tools")
    t = np.arange(0, 6, 1 / 275)
    z = np.clip(np.minimum(t, 6 - t), 0, 1.5)  # up to 1.5 m, a hover, down again
    yaw = 0.3 * t
    rows = ["timestamp,pose_position_x,pose_position_y,pose_position_z,pose_orientation_x,pose_orientation_y,pose_orientation_z,"
            "pose_orientation_w,velocity_linear_x"]  # fmt: skip
    rows += [f"{1e9 + 1e6 * ti:.0f},{2 * ti:.6f},0,{zi:.6f},0,0,{math.sin(y / 2):.8f},{math.cos(y / 2):.8f},0" for ti, zi, y in zip(t, z, yaw, strict=True)]
    d = convert("\n".join(rows).encode())
    assert d["t"][0] == 0 and np.allclose(np.diff(d["t"]), 1 / 275, atol=2e-6) and np.allclose(d["quat"][:, 0], np.cos(yaw / 2), atol=1e-6)
    mocap = "timestamp,drone_x,gate2_int_x,gate2_int_y,gate2_int_z," + ",".join(f"gate2_int_rot[{i}]" for i in range(9))
    mocap += ",gate1_int_x,gate1_int_y,gate1_int_z," + ",".join(f"gate1_int_rot[{i}]" for i in range(9))
    row = "0,0,4,-1,1.2,1,0,0,0,1,0,0,0,1,-4,1,1.1,0,-1,0,1,0,0,0,0,1"
    gates = tool.convert_gates("\n".join([mocap, row, row.replace("-4,1,1.1", "nan,nan,nan"), row]).encode())  # (a frame without the gate)
    assert gates.shape == (2, 12) and gates[0, :3].tolist() == [-4, 1, 1.1] and gates[1, :3].tolist() == [4, -1, 1.2]  # gate 1 first
    assert gates[0, 3:].tolist() == [0, -1, 0, 1, 0, 0, 0, 0, 1]
    np.savez(tmp_path / "f.npz", **d, gates=gates)
    assert np.array_equal(load_gates(str(tmp_path / "f.npz")), gates)
    fl = load_flight(str(tmp_path / "f.npz"), CONTROL_DT, min_height=0.5, smooth=0.0)
    n = fl["p"].shape[0]
    assert abs(n * CONTROL_DT - 5.0) < 0.05 and float(fl["p"][:, 2].min()) > 0.49
    assert torch.allclose(fl["p"][1:, 0] - fl["p"][:-1, 0], torch.full((n - 1,), 2 * CONTROL_DT), atol=1e-4)
    assert torch.allclose(fl["quat"].norm(dim=-1), torch.ones(n), atol=1e-6)
    smooth = load_flight(str(tmp_path / "f.npz"), CONTROL_DT, min_height=0.5)
    assert float((smooth["p"][5:-5] - fl["p"][5 : 5 + smooth["p"].shape[0] - 10]).abs().max()) < 0.02
    with pytest.raises(ValueError, match="never rises"):
        load_flight(str(tmp_path / "f.npz"), CONTROL_DT, min_height=3.0)
    # frames the capture dropped are filled in before the smoothing: the flight with gaps is the flight without them
    keep = np.ones(len(d["t"]), dtype=bool)
    keep[600:615], keep[900:903] = False, False
    np.savez(tmp_path / "g.npz", **{k: v[keep] for k, v in d.items()})
    gapped = load_flight(str(tmp_path / "g.npz"), CONTROL_DT, min_height=0.5)
    assert gapped["p"].shape == smooth["p"].shape and float((gapped["p"] - smooth["p"]).abs().max()) < 2e-3
    speed = (gapped["p"][1:] - gapped["p"][:-1]).norm(dim=-1) / CONTROL_DT
    assert float(speed.max()) < 2.3  # (the flight moves at 2 m/s and climbs at up to 1 m/s; a filter over the gap's samples would not hold it)


def test_reference_and_its_feedforward(setup):
    """The reference's feedforward command is the command that flew the flight; frames stay within their flights."""
    _, task, ctrl = setup
    ref, n = task.ref, ctrl.shape[0] + 1
    assert ref.frames == n + 81 and bool((ref.end[:n] == n - 1).all()) and bool((ref.end[n:] == ref.frames - 1).all())
    assert bool((ref.start[:n] == 0).all()) and bool((ref.start[n:] == n).all())
    f = torch.tensor([0, n - 2, n - 1, n, ref.frames - 1])
    assert ref.next_frame(f).tolist() == [1, n - 1, n - 1, n + 1, ref.frames - 1]
    assert ref.ahead(f, 7).tolist() == [7, n - 1, n - 1, n + 7, ref.frames - 1]
    mid = slice(3, n - 3)  # (central differences; the ends are one-sided)
    held = 0.5 * (ctrl[2 : n - 4, 0] + ctrl[3 : n - 3, 0])  # the thrust about frame f: the steps before and after it
    assert float((ref.thrust[mid] - held).abs().max()) < 0.03 * SPEC.weight
    # the body rates follow their commands with the servos' lag (20 to 50 ms of commands that change by up to 14 rad/s^2)
    assert float((ref.w[mid] - 0.5 * (ctrl[2 : n - 4, 1:] + ctrl[3 : n - 3, 1:])).abs().max()) < 0.6
    u = task.setpoint(torch.zeros(5, NUM_U), f)
    assert torch.equal(u[:, 0], ref.thrust[f]) and torch.equal(u[:, 1:], ref.w[f])
    hi = task.setpoint(torch.full((1, NUM_U), 100.0), f[:1])[0]
    assert hi.tolist() == pytest.approx([SPEC.max_thrust, SPEC.rate_limit, SPEC.rate_limit, SPEC.rate_limit])


def test_state_layout_and_task_at_the_reference(setup):
    _, task, _ = setup
    f = torch.tensor([3, 40, 150])
    S, zero = task.ref.state(f), torch.zeros(3, NUM_U)
    s = flatten_state(S)
    assert s.shape == (3, STATE_DIM) and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(unflatten_state(s), S, strict=True))
    assert task.obs(S, zero).shape == (3, OBS_DIM) and task.goal(S, f, 4).shape == (3, 4 * GOAL_FRAME_DIM)
    assert torch.allclose(task.reward(S, f, zero, zero), torch.full((3,), task.max_step_reward), atol=1e-6)
    assert not bool(task.termination(S, f).any()) and float(task.soft_termination(S, f).max()) < 1e-3
    far = (S[0] + torch.tensor([0.0, 2 * task.cfg.term_pos, 0.0]), *S[1:])
    assert bool(task.termination(far, f).all()) and float(task.soft_termination(far, f).min()) > 0.999
    assert float(task.reward(far, f, zero, zero).max()) < task.max_step_reward * 0.71  # the position's share is lost
    e = task.errors(far, f)
    assert torch.allclose(e["pos"], torch.full((3,), 2 * task.cfg.term_pos)) and float(e["rot_sq"].max()) < 1e-5


def test_inputs_do_not_depend_on_position_or_heading():
    """The policy's observation and goal and the world model's inputs are the same for a flight moved and turned about z."""
    torch.manual_seed(0)
    model = load_model(SPEC)
    yaw, shift = 1.1, (3.0, -2.0, 1.0)
    a, b = fly(model, 1.0, 3)[0], fly(model, 1.0, 3, yaw=yaw, offset=shift)[0]
    Rz = quat_to_mat(torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    assert torch.allclose(b["p"], (a["p"] - torch.tensor([0.0, 0.0, 1.0])) @ Rz.T + torch.tensor(shift), atol=2e-3)  # (the same flight)
    ta, tb = (QuadTask(build_reference([x], SPEC, "cpu"), SPEC) for x in (a, b))
    f = torch.tensor([10, 50, 80])
    dp, dth, dv = 0.2 * torch.randn(3, 3), 0.2 * torch.randn(3, 3), torch.randn(3, 3)  # offsets from the reference, in the body's frame
    states = []
    for task in (ta, tb):
        p, R, v, w = task.ref.state(f)
        states.append((p + (R @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth), v + (R @ dv.unsqueeze(-1)).squeeze(-1), w + 0.3))
    a_prev, u = torch.rand(3, NUM_U), torch.rand(3, NUM_U)
    wm = QuadWorldModel(hidden=(8,))
    assert torch.allclose(ta.obs(states[0], a_prev), tb.obs(states[1], a_prev), atol=3e-3)
    assert torch.allclose(ta.goal(states[0], f, 4), tb.goal(states[1], f, 4), atol=3e-3)
    assert torch.allclose(wm.inputs(states[0], u), wm.inputs(states[1], u), atol=3e-3)


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_world_model_reproduces_a_step_from_its_targets(setup, kind):
    """A network that outputs a transition's targets steps to the transition's next state (the acceleration model with the
    simulator's substeps in closed form: MuJoCo's own positions), and the error of the true next state is zero."""
    model, task, _ = setup
    d = mujoco.MjData(model)
    f = torch.tensor([20, 60, 130])
    S, u = task.ref.state(f), task.setpoint(torch.tensor([[0.5, -0.4, 0.3, 0.2]]).expand(3, -1), f)
    nxt = []
    for i in range(3):
        s = flatten_state(S)[i].double().numpy()
        d.qpos[:], d.qvel[:], d.ctrl[:] = s[:7], s[7:], u[i].numpy()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
        nxt.append(np.concatenate((d.qpos, d.qvel)))
    S1 = unflatten_state(torch.tensor(np.array(nxt), dtype=torch.float32))
    wm = QuadWorldModel(hidden=(8,), kind=kind, substeps=SUBSTEPS)
    assert wm.inputs(S, u).shape == (3, wm.in_norm.mean.shape[0]) and wm.targets(S, S1).shape == (3, wm.out_norm.mean.shape[0])
    target = wm.targets(S, S1)

    class Exact(torch.nn.Module):
        def forward(self, x):
            return (target - wm.out_norm.mean) / wm.out_norm.std()

    wm.net = Exact()
    S_hat = wm.step(S, u)
    # (accel: one rotation by the step's mean body rates for the simulator's four, whose rates the servos change within the step)
    for a, b, tol in zip(S_hat, S1, (2e-5 if kind == "accel" else 1e-6, 1e-3 if kind == "accel" else 1e-6, 1e-5, 1e-5), strict=True):
        assert torch.allclose(a, b, atol=tol), (a - b).abs().max()
    assert float(wm.error(S1, S1).abs().max()) < 1e-5 and float(wm.error(S, S1).abs().max()) > 0.1


def agent_and_replay(task, argv, n=8, steps=16, dev="cpu"):
    """An agent with the arguments ``argv`` and a replay of ``steps`` transitions of ``n`` worlds along the reference."""
    torch.manual_seed(0)
    args = parse_args(["--flight", "f", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch", "16",
                       "--wm_window", "4", "--pi_batch", "16", *argv])  # fmt: skip
    algo = QUAD_ALGOS[args.algo](args, task, dev, substeps=SUBSTEPS)
    buffer = ReplayBuffer(steps, n, dev, fields=replay_fields(), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.randint(0, 80, (n,), device=dev)
    S, a_prev = task.ref.state(f), torch.zeros(n, NUM_U, device=dev)
    for _ in range(steps):
        a = algo.act(S, a_prev, f, noise=0.1)
        u, f1 = task.setpoint(a, f), task.ref.next_frame(f)
        S1 = task.ref.state(f1)
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": flatten_state(S), "s1": flatten_state(S1), "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(n, dtype=torch.bool, device=dev)})
        S, f, a_prev = S1, f1, a
    return args, algo, buffer


SM = ["--critic_epochs", "1", "--critic_minibatches", "2"]
ST = ["--algo", "supertrack", "--st_fail_w", "2", "--st_imag_noise", "0.2"]


@pytest.mark.parametrize("argv", [SM, [*SM, "--bptt_decay", "0.9"], ST, [*ST, "--st_task_weights"]], ids=["supermimic", "decay", "supertrack", "task_weights"])
def test_updates(setup, monkeypatch, argv):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, task, _ = setup
    args, algo, buffer = agent_and_replay(task, argv)
    assert algo.wm.kind == ("accel" if args.algo == "supertrack" else "increment")
    before = [p.clone() for p in algo.pi.parameters()]
    assert torch.isfinite(algo.wm_update(buffer))
    loss, v_loss, imag_r, grad_norm, alive = algo.policy_update(buffer)
    assert all(torch.isfinite(x) for x in (loss, imag_r, grad_norm, alive)) and 0 < float(imag_r) <= 1 and 0 < float(alive) <= 1
    assert torch.isfinite(v_loss) == (args.algo == "supermimic")
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    if args.algo == "supertrack":
        c = task.cfg
        assert algo.bal.calibrated and (algo.step_worth == c.w_pos + c.w_rot + c.w_vel + c.w_ang) == args.st_task_weights
        algo2 = QUAD_ALGOS[args.algo](args, task, "cpu")
        algo2.load_state_dict(algo.state_dict())
        assert torch.equal(algo2.bal.w, algo.bal.w)


def test_training_arguments():
    """Every argument of ndm_gym.train is implemented or refused; the defaults are QUAD_DEFAULTS over LAFAN1's."""
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    assert set(QUAD_DEFAULTS) <= set(SUPPORTED)
    args = parse_args(["--flight", "a", "b"])
    assert all(getattr(args, name) == value for name, value in QUAD_DEFAULTS.items()) and args.flight == ["a", "b"]
    assert (args.goal_stride, args.episode, args.ckpt_keep) == (5, 500, False)
    for refused in (["--algo", "blinkmimic"], ["--wm_backbone", "rigidformer"], ["--hist", "4"], ["--st_fail_w", "2"], ["--st_task_weights"],
                    ["--algo", "supertrack", "--gamma", "0.995"], ["--algo", "supertrack", "--bptt_decay", "0.95"], ["--goal_stride", "0"],
                    ["--fused_adam", "--reset_masked", "--no_sync", "--compile_eager"], ["--no_sync"], ["--collect_graph"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--flight", "a"] + refused)
    st = parse_args(["--flight", "a", "--algo", "supertrack", "--st_fail_w", "2", "--st_task_weights", "--goal_stride", "3", "--episode", "200"])
    assert (st.algo, st.st_fail_w, st.st_task_weights, st.goal_stride, st.episode) == ("supertrack", 2, True, 3, 200)


CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")
SYSTEMS = ["--fused_adam", "--reset_masked", "--no_sync"]


@pytest.fixture(scope="module")
def gpu_setup():
    model = load_model(SPEC)
    flights = [fly(model, 1.2, 0)[0], fly(model, 0.8, 1)[0]]
    return model, QuadTask(build_reference(flights, SPEC, CUDA), SPEC)


@GPU
def test_simulator_matches_mujoco_and_resets(gpu_setup):
    """QuadSim's control step is MuJoCo's physics steps of it; episodes start at the reference with the noise asked for and
    end at their flights' last frames or after ``episode`` steps; ``step_nosync`` is ``step`` with the ends reset by mask."""
    from ndm_gym.sim.rigid.quadrotor_sim import QuadSim

    model, task = gpu_setup
    ref, n = task.ref, 32
    torch.manual_seed(0)
    sim = QuadSim(model, task, n, CUDA, pos_noise=0.0, rot_noise=0.0, vel_noise=0.0, ang_noise=0.0)
    assert torch.allclose(sim.state(), flatten_state(ref.state(sim.f)), atol=1e-5) and bool((sim.f < ref.end[sim.f]).all())
    assert bool((sim.last == torch.minimum(sim.f + sim.episode, ref.end[sim.f])).all())
    s, f = sim.state(), sim.f.clone()
    u = task.setpoint(2 * torch.rand(n, NUM_U, device=CUDA) - 1, f)
    s1, term, done = sim.step(u, reset=False)
    d = mujoco.MjData(model)
    for i in range(4):
        x = s[i].double().cpu().numpy()
        d.qpos[:], d.qvel[:], d.ctrl[:] = x[:7], x[7:], u[i].cpu().numpy()
        for _ in range(sim.substeps):
            mujoco.mj_step(model, d)
        assert np.allclose(s1[i].cpu().numpy(), np.concatenate((d.qpos, d.qvel)), atol=2e-4)
    assert bool((sim.f == ref.next_frame(f)).all()) and not bool(term.any())
    start = torch.full((n,), int(ref.start[-1]), device=CUDA)
    sim.reset(torch.ones(n, dtype=torch.bool, device=CUDA), start)  # evaluation's start: a flight's first frame to its last
    assert bool((sim.f == start).all()) and bool((sim.last == ref.frames - 1).all())
    noisy = QuadSim(model, task, 256, CUDA, episode=20, stagger=True)
    e = task.errors(unflatten_state(noisy.state()), noisy.f)
    assert 0.1 < float(e["pos"].mean()) < 0.25 and 0.3 < float(e["vel"].mean()) < 0.7 and bool((noisy.last - noisy.f <= 20).all())
    a, b = (QuadSim(model, task, n, CUDA, episode=6, pos_noise=0.0, rot_noise=0.0, vel_noise=0.0, ang_noise=0.0) for _ in range(2))
    stream, ends = b.torch_stream(), 0
    for _ in range(30):
        for x, y in ((a.qpos, b.qpos), (a.qvel, b.qvel), (a.f, b.f), (a.last, b.last)):
            y.copy_(x)
        u = task.setpoint(2 * torch.rand(n, NUM_U, device=CUDA) - 1, a.f)
        sa, ta, da = a.step(u, reset=False)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sb, tb, db = b.step_nosync(u)
        torch.cuda.current_stream().wait_stream(stream)
        assert float((sa - sb).abs().max()) < 1e-3 and torch.equal(ta, tb) and torch.equal(da, db)
        keep = ~db
        assert torch.equal(b.qpos[keep], a.qpos[keep]) and torch.equal(b.f[keep], a.f[keep])
        if bool(db.any()):  # the ended worlds restart on the reference
            assert torch.allclose(b.state()[db], flatten_state(ref.state(b.f[db])), atol=1e-5)
        ends += int(db.sum())
        a.reset(da)
    assert ends > 0


@GPU
@pytest.mark.parametrize("extra", [SM, [*ST, "--st_task_weights"]], ids=["supermimic", "supertrack"])
def test_training_loop_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, extra):
    """train_quadrotor's loop with every systems option on short flights: the graphs are captured, the log's rows are finite,
    the checkpoint holds the policy average, a resumed run continues the log, and evaluate_quadrotor reads a kept checkpoint."""
    import json

    import ndm_gym.evaluate_quadrotor as evaluate_quadrotor
    import ndm_gym.train_quadrotor as train_quadrotor

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_quadrotor, "build", lambda flights, data_dir, device, reward_std_scale=1.0, goal_stride=5, model_spec=None: (gpu_setup[0], SPEC, gpu_setup[1]))
    run = tmp_path / "run"
    argv = ["--flight", "a", "b", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch", "16",
            "--wm_window", "4", "--pi_batch", "16", *extra, *SYSTEMS, "--collect_graph", "--update_graphs", "--pi_ema", "0.9", "--pi_lr_mult", "2",
            "--lr_decay_ret", "0.99", "--lr_decay_by", "8", "--lr_decay_len", "4", "--num_envs", "32", "--buffer_steps", "16", "--warmup_steps", "8",
            "--wm_only_iters", "2", "--log_every", "4", "--ckpt_every", "4", "--ckpt_keep", "--episode", "30", "--log_dir", str(run)]  # fmt: skip
    for iterations, resume in ((12, []), (20, ["--resume", str(run / "last.pt")])):
        monkeypatch.setattr(sys, "argv", ["train_quadrotor", *argv, "--iterations", str(iterations), *resume])
        train_quadrotor.main()
        rows = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
        assert rows[:, 0].tolist() == list(range(4, iterations + 1, 4))
        cols = [train_quadrotor.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "pos_mm")]
        assert np.isfinite(rows[1:, cols]).all() and (rows[:, train_quadrotor.LOG_COLUMNS.index("sim_r")] > 0).all()
        ck = torch.load(run / "last.pt", map_location=CUDA, weights_only=False)
        assert ck["iteration"] == iterations and {"pi", "pi_live", "pi_avg"} <= set(ck) and ck["pi_lr_decay_at"] == 8
        assert int(ck["pi_avg"]["n_averaged"]) == iterations - 2
    kept = sorted(p.name for p in run.glob("ckpt_*.pt"))
    assert kept == [f"ckpt_{i:06d}.pt" for i in range(4, 21, 4)]
    monkeypatch.setattr(sys, "argv", ["train_quadrotor", *argv, "--iterations", "24", "--resume", str(run / "last.pt"), "--pi_opt", "muon"])
    with pytest.raises(ValueError, match="pi_opt"):
        train_quadrotor.main()
    monkeypatch.setattr(evaluate_quadrotor, "__name__", "ndm_gym.evaluate_quadrotor")
    monkeypatch.setattr(sys, "argv", ["evaluate_quadrotor", "--ckpt", str(run / kept[-1]), "--out", str(run / "evaluation"), "--num_envs", "16"])
    evaluate_quadrotor.main()
    m = json.load(open(run / "evaluation" / "metrics.json"))
    assert list(m["flights"]) == ["a", "b"] and m["iteration"] == 20
    for name, frames in (("a", 121), ("b", 81)):
        fl = m["flights"][name]
        assert fl["frames"] == frames and set(fl) >= {"nominal", "policy", "reference_command"}
        assert 0 <= fl["policy"]["success_rate"] <= 1 and 0 < fl["policy"]["flown_s"] <= fl["seconds"] + 1e-6
        roll = np.load(run / "evaluation" / f"rollout_{name}.npz")
        assert roll["state"].shape == (frames, STATE_DIM) and roll["reference_p"].shape == (frames, 3)
        assert roll["reference_command_state"].shape == (frames, STATE_DIM) and roll["gates"].shape == (0, 12)
        # the flights' own commands, open loop from noisy starts: most rollouts stay within the termination's metre for the second
        assert fl["reference_command"]["success_rate"] > 0.3 and fl["reference_command"]["flown_s"] > 0.5 * fl["seconds"]
