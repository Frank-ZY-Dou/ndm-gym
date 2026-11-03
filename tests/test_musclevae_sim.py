"""MuscleVAE's simulator (ndm_gym.sim.rigid.muscle_sim) on the CPU, with Newton's MuJoCo-Warp kernels on warp's CPU device
(skipped without Newton): the layout and frames of Newton's per-body wrenches on a small model; with the converted assets,
two environments of the simulator: its reference tables, the reset (poses, the velocity conventions, fatigue, reference
frame, masks), MuscleVAE's termination and reference rules, a step and its transition; a contracting muscle pulling two
free bodies together through Newton; on a GPU, the CUDA graphs against eager on discrete outcomes (the GPU physics is
not run-to-run deterministic)."""

import os

import numpy as np
import pytest
import torch

newton = pytest.importorskip("newton")
pytest.importorskip("mujoco_warp")

from ndm_gym.core.rotations import quat_to_mat, so3_exp, so3_log  # noqa: E402
from ndm_gym.robots import musclevae_char as mc  # noqa: E402
from ndm_gym.robots.muscles import fatigue_init_curve  # noqa: E402

HAVE_DATA = all(os.path.isfile(p) for p in (mc.MJCF_PATH, mc.CHARACTER_PATH, mc.MUSCLES_PATH)) and os.path.isdir(mc.CLIPS_DIR)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="MuscleVAE's assets are optional data: run tools/download_musclevae.sh, then "
                                                      "python tools/convert_musclevae.py")  # fmt: skip
BOX_QUAT = np.array([np.cos(0.35), *(np.sin(0.35) * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0))])  # wxyz, 0.7 rad about (1, 2, 3)
BOX = (f'<mujoco><option gravity="0 0 0"/><worldbody><body name="box" pos="0.1 -0.2 1.0" quat="{" ".join(map(repr, BOX_QUAT.tolist()))}"><freejoint/>'
       '<inertial pos="0 0 0" mass="2.0" diaginertia="0.1 0.15 0.2"/><geom type="box" size="0.1 0.2 0.3" contype="0" conaffinity="0"/>'
       "</body></worldbody></mujoco>")  # fmt: skip
TWO_BODIES = ('<mujoco><option gravity="0 0 0"/><worldbody>' + "".join(
    f'<body name="{name}" pos="{x} 0 1"><freejoint/><inertial pos="0 0 0" mass="{m}" diaginertia="0.01 0.01 0.01"/>'
    '<geom type="sphere" size="0.05" contype="0" conaffinity="0"/></body>' for name, x, m in (("a", 0.0, 1.0), ("b", 0.3, 2.0))) +
    "</worldbody></mujoco>")  # fmt: skip


def test_body_wrench_frames():
    """Newton's State.body_f on MuJoCo-Warp, one row per body of every world: a force at the body's center of mass and a
    torque, both in the world frame. A force f at a point p of a turned free body, written as (f, (p - c) x f), gives
    a = f / m and I_w alpha = (p - c) x f in its own world only, and the body moves by h v (semi-implicit Euler)."""
    from newton.solvers import SolverMuJoCo

    def builder():
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(b)
        return b

    one = builder()
    one.add_mjcf(BOX)
    scene = builder()
    scene.replicate(one, 2)
    model = scene.finalize(device="cpu")
    solver = SolverMuJoCo(model, solver="newton", integrator="implicitfast", cone="pyramidal")
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    body_q = s0.body_q.numpy().astype(np.float64)
    c, R = body_q[1, :3], quat_to_mat(torch.tensor(body_q[1, [6, 3, 4, 5]])).numpy()
    assert np.allclose(R, quat_to_mat(torch.tensor(BOX_QUAT)).numpy(), atol=1e-6) and np.allclose(c, [0.1, -0.2, 1.0], atol=1e-6)
    f, p = np.array([3.0, -1.0, 2.0]), c + R @ np.array([0.05, 0.1, -0.2])
    wrench = np.zeros((2, 6))
    wrench[1, :3], wrench[1, 3:] = f, np.cross(p - c, f)
    s0.body_f.assign(wrench.astype(np.float32))
    h = 0.01
    solver.step(s0, s1, None, None, h)
    qd = s1.body_qd.numpy().astype(np.float64)
    inertia_w = R @ np.diag([0.1, 0.15, 0.2]) @ R.T
    np.testing.assert_allclose(qd[1, :3] / h, f / 2.0, rtol=1e-4)
    np.testing.assert_allclose(inertia_w @ (qd[1, 3:] / h), np.cross(p - c, f), rtol=1e-4, atol=1e-5)
    assert np.abs(qd[0]).max() == 0.0 and np.abs(s1.body_q.numpy()[0] - body_q[0]).max() < 1e-7
    np.testing.assert_allclose(s1.body_q.numpy()[1, :3] - c, h * qd[1, :3], atol=1e-6)


def test_contracting_muscle_pulls_bodies_together():
    """Two free bodies 0.3 m apart (1 and 2 kg) joined by one muscle along the line between their centers (each other
    body part's muscle within one body, where its forces cancel), the muscle model's wrenches in Newton's body_f, one
    physics step from rest: a PD target shorter than the muscle makes it pull, and the bodies accelerate toward each
    other by its force over their masses; a longer target does not push them apart (muscles only pull)."""
    from newton.solvers import SolverMuJoCo

    from ndm_gym.robots.muscles import MUSCLE_MODELS, NUM_PARTS, MuscleModel, MuscleTable

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(TWO_BODIES)
    model = b.finalize(device="cpu")
    solver = SolverMuJoCo(model, solver="newton", integrator="implicitfast", cone="pyramidal")
    M, h, mass = NUM_PARTS, 1.0 / 120.0, np.array([1.0, 2.0])
    body0 = np.array([0, 1] + [0, 0] * (M - 1))
    local = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]] + [[0.0, 0.0, 0.02], [0.0, 0.0, -0.02]] * (M - 1))
    table = MuscleTable(names=tuple(f"m{i}" for i in range(M)), num_bodies=2, anchor_count=np.full(M, 2), anchor_body0=body0, anchor_local0=local,
                        anchor_body1=body0, anchor_local1=local, anchor_weight0=np.ones(2 * M), rest_length=np.r_[0.2, np.full(M - 1, 0.04)],
                        lm=np.full(M, 0.8), lt=np.full(M, 0.2), f0=np.full(M, 500.0), f0max=np.full(M, 750.0), kp=np.full(M, 1000.0),
                        part=np.arange(M))  # fmt: skip
    for params in MUSCLE_MODELS.values():
        muscles = MuscleModel(table, params)
        for scale, pulls in ((0.8, True), (1.2, False)):
            s0, s1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            q = torch.tensor(s0.body_q.numpy())
            pos, rot = q[:, :3], quat_to_mat(q[:, [6, 3, 4, 5]])
            length = muscles.lengths(muscles.anchor_positions(pos, rot))
            fresh = torch.zeros(M), torch.ones(M), torch.zeros(M)  # (MA, MR, MF)
            out = muscles.substep(pos, rot, length, scale * length, *fresh)
            s0.body_f.assign(torch.cat((out.body_force, out.body_torque), -1).numpy())
            solver.step(s0, s1, None, None, h)
            v, w = s1.body_qd.numpy()[:, :3].astype(np.float64), s1.body_qd.numpy()[:, 3:]
            toward = (pos[1] - pos[0]).numpy() / np.linalg.norm((pos[1] - pos[0]).numpy())
            force = float(out.force[0])
            if pulls:
                assert force > 0 and v[0] @ toward > 0 and v[1] @ toward < 0
                np.testing.assert_allclose(v, h * force / mass[:, None] * np.stack((toward, -toward)), rtol=1e-4, atol=1e-6)
            else:
                assert force == 0.0 and np.abs(v).max() < 1e-9
            assert np.abs(w).max() < 1e-6


@pytest.fixture(scope="module")
def sim():
    from ndm_gym.sim.rigid.muscle_sim import MuscleSim

    return MuscleSim(2, "cpu", clips=("lafan_walk",), seed=0)


def clip_poses(ref_q: torch.Tensor):
    """Our forward kinematics (double precision) of Newton's coordinates (..., 71)."""
    q = ref_q.double()
    joints = []
    for j, a in enumerate(mc.JOINT_QPOS_ADR):
        a += 7
        joints.append(q[..., [a + 3, a, a + 1, a + 2]] if mc.JOINT_TYPES[j] == "ball" else q[..., a : a + 1])
    return mc.forward_kinematics(q[..., :3], q[..., [6, 3, 4, 5]], torch.cat(joints, -1))


@needs_data
def test_reference_tables(sim):
    """The clip in Newton's coordinates (quaternions xyzw), raised above the floor; velocities by backward differences;
    the reset frames (neither the first nor the last 25 of a clip); muscle lengths in each frame's pose and 1/120 s
    before; their placement's head heights."""
    from ndm_gym.motion.sources import musclevae as src
    from ndm_gym.sim.rigid.muscle_sim import RESET_LIFT, geom_bottom, interpolate_previous

    ref, path = sim.ref, os.path.join(mc.CLIPS_DIR, "lafan_walk.npz")
    clip = src.load_clip(path)
    T = clip.root_p.shape[0]
    assert ref.names == ("lafan_walk",) and ref.joint_q.shape == (T, 71) and ref.joint_qd.shape == (T, 56) and ref.length.shape == (T, 284)
    assert torch.equal(ref.reset_frames, torch.arange(1, T - 25)) and ref.is_last.nonzero().flatten().tolist() == [T - 1]
    pos, rot = mc.forward_kinematics(clip.root_p.double(), clip.root_quat.double(), clip.joints.double())
    assert torch.allclose(ref.joint_q[:, :2].double(), pos[:, 0, :2], atol=1e-6)
    assert torch.allclose(ref.joint_q[:, 2].double() - ref.lift.double(), pos[:, 0, 2], atol=1e-6)
    bottom = geom_bottom(pos, rot, sim.char) + ref.lift.double()
    raised = ref.lift > 0
    assert raised.any() and bottom.min() > -1e-6 and torch.allclose(bottom[raised], torch.full_like(bottom[raised], RESET_LIFT), atol=1e-6)
    p2, r2 = clip_poses(ref.joint_q)  # Newton's layout back to ours: the same poses
    assert torch.allclose(p2[..., :2], pos[..., :2], atol=1e-5) and torch.allclose(r2, rot, atol=1e-5)
    torch.testing.assert_close(ref.joint_qd[1:, :3], (clip.root_p[1:] - clip.root_p[:-1]) * 20.0, rtol=1e-5, atol=1e-5)
    assert torch.equal(ref.joint_qd[0], ref.joint_qd[1])
    w = so3_log(quat_to_mat(clip.root_quat[1:].double()) @ quat_to_mat(clip.root_quat[:-1].double()).transpose(-1, -2)) * 20.0
    torch.testing.assert_close(ref.joint_qd[1:, 3:6].double(), w, rtol=1e-5, atol=1e-4)
    m = sim.muscles
    f = torch.tensor([0, 1, 500, T - 1])
    torch.testing.assert_close(ref.length[f], m.lengths(m.anchor_positions(pos[f].float(), rot[f].float())), rtol=1e-5, atol=1e-6)
    # 1/120 s before a frame: the joint positions 5/6 of the way from the previous frame, close to the lengths' own interpolation
    pb, rb = mc.forward_kinematics(*interpolate_previous(clip.root_p.double(), clip.root_quat.double(), clip.joints.double()))
    torch.testing.assert_close(ref.length_prev[f], m.lengths(m.anchor_positions(pb[f].float(), rb[f].float())), rtol=1e-5, atol=1e-6)
    dev = (ref.length_prev[1:] - (ref.length[:-1] + 5.0 / 6.0 * (ref.length[1:] - ref.length[:-1]))).abs()
    assert dev.max() < 1e-2 and dev.amax(-1).median() < 1e-3 and torch.equal(ref.length_prev[0], ref.length[0])
    body_pos, _ = src.load_placement(path)
    assert torch.equal(ref.head_z, body_pos[:, mc.HEAD, 2])


@needs_data
def test_reset(sim):
    """The reset pose is the frame's (forward kinematics, raised); the fatigue state is one of the initial curve's per
    body part; the reference is the next frame; a masked reset leaves the other environments alone."""
    frames = torch.tensor([100, 2000])
    sim.reset(frames=frames)
    state = sim.body_state()
    pos, rot = clip_poses(sim.ref.joint_q[frames])
    torch.testing.assert_close(state[..., :3].double(), pos, rtol=0, atol=2e-5)
    torch.testing.assert_close(quat_to_mat(state[..., 3:7].double()), rot, rtol=0, atol=2e-5)
    assert torch.equal(sim.ref_frame, frames + 1) and torch.equal(sim.ep_step, torch.zeros(2, dtype=torch.long))
    assert torch.equal(sim.length_prev, sim.ref.length_prev[frames])
    curve = torch.tensor(fatigue_init_curve(sim.muscles.params), dtype=torch.float32)[:1200]
    fat = sim.fatigue_state()
    part = torch.tensor(sim.muscles.table.part)
    for k in range(5):
        x = fat[:, part == k]
        assert (x == x[:, :1]).all() and ((x[:, 0, None, :] == curve[None]).all(-1).sum(-1) >= 1).all()
    q1 = sim.views[0].joint_q[1].clone()
    sim.reset(torch.tensor([True, False]), torch.tensor([300, 400]))
    assert torch.equal(sim.views[0].joint_q[1], q1) and torch.equal(sim.ref_frame, torch.tensor([301, 2001]))
    with pytest.raises(ValueError):
        sim.reset(frames=torch.tensor([0, 100]))  # a clip's first frame has no previous one
    with pytest.raises(ValueError):
        sim.reset(frames=torch.tensor([100, len(sim.ref.clip) - 1]))


@needs_data
def test_velocity_conventions(sim):
    """Newton's velocities as the reset writes them: the root's linear and angular velocity in the world frame, each
    ball joint's angular velocity relative to its parent in the parent's frame, hinge rates. Newton's body velocities
    of a short motion with constant joint velocities match finite differences of our forward kinematics."""
    g = torch.Generator().manual_seed(1)
    q0 = sim.ref.joint_q[torch.tensor([700, 3000])].double()
    qd = 0.8 * torch.randn(2, 56, generator=g, dtype=torch.float64)
    d = 1e-5
    q1 = q0.clone()
    q1[:, :3] += d * qd[:, :3]
    R = quat_to_mat(q0[:, [6, 3, 4, 5]])
    q1[:, 3:7] = _xyzw(so3_exp(d * qd[:, 3:6]) @ R)
    for j, (a, v) in enumerate(zip(mc.JOINT_QPOS_ADR, mc.JOINT_QVEL_ADR)):
        a, v = a + 7, v + 6
        if mc.JOINT_TYPES[j] == "ball":
            q1[:, a : a + 4] = _xyzw(so3_exp(d * qd[:, v : v + 3]) @ quat_to_mat(q0[:, [a + 3, a, a + 1, a + 2]]))
        else:
            q1[:, a] += d * qd[:, v]
    (p0, r0), (p1, r1) = clip_poses(q0), clip_poses(q1)
    v_fd, w_fd = (p1 - p0) / d, so3_log(r1 @ r0.transpose(-1, -2)) / d
    sim.views[0].joint_q.copy_(q0.float())
    sim.views[0].joint_qd.copy_(qd.float())
    newton.eval_fk(sim.model, sim.states[0].joint_q, sim.states[0].joint_qd, sim.states[0])
    body_qd = sim.views[0].body_qd.double()
    assert (body_qd[..., 3:] - w_fd).abs().max() < 2e-3 * w_fd.abs().max() and (body_qd[..., :3] - v_fd).abs().max() < 2e-3 * v_fd.abs().max()
    sim.reset()


def _xyzw(R: torch.Tensor) -> torch.Tensor:
    from ndm_gym.core.rotations import mat_to_quat

    return mat_to_quat(R)[..., [1, 2, 3, 0]]


@needs_data
def test_done_and_reference_rules(sim):
    """MuscleVAE's rules (done_flags, next_frames): the head-height counter (+1 off track, -1 on, at least 0), termination
    once it reaches 20 after 26 steps or when the state is not finite, truncation at 512 steps; the reference's next
    frame, a random one at a clip's end and every 96 steps."""
    from ndm_gym.sim.rigid.muscle_sim import ERR_LENGTH, MAX_LENGTH, MIN_LENGTH, done_flags, next_frames

    sim.reset(frames=torch.tensor([100, 200]))
    q, qd = sim.views[0].body_q, sim.views[0].body_qd
    head = q[:, mc.HEAD, 2]
    off_track, on_track = head + 1.0, head.clone()
    flags = lambda before, head_ref, off_count, ep_step: done_flags(q, qd, before, head_ref, torch.tensor(off_count), torch.tensor(ep_step),  # noqa: E731
                                                                   sim.xyzw_to_wxyz)  # fmt: skip
    terminated, truncated, off = flags(q, off_track, [ERR_LENGTH - 1, ERR_LENGTH + 5], [MIN_LENGTH, MIN_LENGTH - 1])
    assert terminated.tolist() == [True, False] and not truncated.any() and off.tolist() == [ERR_LENGTH, ERR_LENGTH + 6]
    terminated, truncated, off = flags(q, on_track, [0, 3], [MAX_LENGTH, MAX_LENGTH - 1])
    assert not terminated.any() and truncated.tolist() == [True, False] and off.tolist() == [0, 2]
    bad = q.clone()
    bad[1, 5, 0] = float("nan")  # the state's velocities (backward differences) are not finite
    assert flags(bad, on_track, [0, 0], [3, 3])[0].tolist() == [False, True]
    fast = q.clone()
    fast[0, 5, 0] -= 6.0  # 120 m/s over the control step: some component of the observation exceeds 50
    assert flags(fast, on_track, [0, 0], [3, 3])[0].tolist() == [True, False]
    last = len(sim.ref.clip) - 1
    random_frames = torch.tensor([7, 8])
    assert next_frames(torch.tensor([last - 1, 500]), torch.tensor([3, 96]), sim.ref.is_last, 96, random_frames).tolist() == [7, 8]
    assert next_frames(torch.tensor([500, 600]), torch.tensor([95, 0]), sim.ref.is_last, 96, random_frames).tolist() == [501, 601]
    assert next_frames(torch.tensor([500, 600]), torch.tensor([96, 192]), sim.ref.is_last, 0, random_frames).tolist() == [501, 601]
    sim.reset()


@needs_data
def test_step_and_transition(sim):
    """A control step with the data policy: finite states, the episode step and reference advance, fatigue changes;
    the transition matches transition_fields and fits the replay buffer."""
    from ndm_gym.buffer import ReplayBuffer

    sim.reset(frames=torch.tensor([1000, 4000]))
    before = sim.before_step()
    torch.testing.assert_close(sim.reference_action(), sim.ref.length[sim.ref_frame] / sim.muscles.rest_length - 1.0)
    a = sim.data_policy(0.05)
    terminated, truncated = sim.step(a)
    assert terminated.shape == truncated.shape == (2,) and terminated.dtype == torch.bool and not (terminated | truncated).any()
    assert torch.equal(sim.ep_step, torch.ones(2, dtype=torch.long)) and torch.equal(sim.ref_frame, torch.tensor([1002, 4002]))
    tr = sim.transition(before, a, terminated, truncated)
    fields = sim.transition_fields()
    assert set(tr) == set(fields)
    for k, (shape, dtype) in fields.items():
        assert tuple(tr[k].shape) == (2,) + ((shape,) if isinstance(shape, int) else shape) and tr[k].dtype == dtype, k
    assert torch.isfinite(tr["s1"]).all() and (tr["s1"] - tr["s"]).abs().max() > 0 and (tr["fatigue1"] - tr["fatigue"]).abs().max() > 0
    assert torch.allclose(tr["fatigue1"].sum(-1), torch.ones(2, 284), atol=1e-5)
    assert torch.equal(tr["t"], before["t"]) and tr["ctx"].shape == (2, 0)
    buf = ReplayBuffer(3, 2, "cpu", fields=fields)
    buf.add(tr)
    assert torch.equal(buf.data["a"][0], a)
    # the same transition written into a row of contiguous buffers, then a second step recorded in place
    rec = sim.transition_buffers(3)
    assert all(v.shape == buf.data[k].shape and v.dtype == buf.data[k].dtype and v[1].is_contiguous() for k, v in rec.items())
    row0, row1 = ({k: v[t] for k, v in rec.items()} for t in (0, 1))
    sim.transition(before, a, terminated, truncated, out=row0)
    assert all(torch.equal(row0[k], tr[k]) for k in fields)
    sim.before_step(out=row1)
    a1 = sim.data_policy(0.05)
    t1, tr1 = sim.step(a1)
    sim.transition(row1, a1, t1, tr1, out=row1)
    assert torch.equal(row1["s"], tr["s1"]) and torch.equal(row1["fatigue"], tr["fatigue1"]) and torch.equal(row1["a"], a1)
    assert torch.equal(row1["s1"], sim.body_state()) and torch.equal(row1["fatigue1"], sim.fatigue_state())
    assert row1["t"].tolist() == [1002, 4002] and row1["ep_step"].tolist() == [1, 1] and torch.equal(row1["done"], t1 | tr1)
    excess = sim.ball_limit_excess()
    assert excess.shape == (2, 14, 3) and (excess >= 0).all()
    assert sim.part_fatigue().shape == (2, 15) and sim.check_solver() == {"iterations": 0, "ls_iterations": 0}


def test_euler_coords():
    """The ball joints' three hinges, about our x, z, y (their z, y, x) in that order, compose to the joint's rotation."""
    from ndm_gym.core.rotations import axis_angle_to_mat, mat_to_quat
    from ndm_gym.sim.rigid.muscle_sim import EULER_HINGES, euler_coords

    g = torch.Generator().manual_seed(3)
    L = so3_exp(0.9 * torch.randn(64, 3, generator=g, dtype=torch.float64))
    qpos = torch.zeros(64, mc.JOINT_QPOS_DIM, dtype=torch.float64)
    j = mc.BALL_JOINTS[4]
    qpos[:, mc.JOINT_QPOS_ADR[j] : mc.JOINT_QPOS_ADR[j] + 4] = mat_to_quat(L)
    c = euler_coords(qpos, j)
    R = torch.eye(3, dtype=torch.float64).expand(64, 3, 3)
    for i, (_, _, axis) in enumerate(EULER_HINGES):
        R = R @ axis_angle_to_mat(torch.tensor([float(x) for x in axis.split()], dtype=torch.float64).expand(64, 3), c[:, i])
    torch.testing.assert_close(R, L, rtol=0, atol=1e-10)


@needs_data
def test_character_mjcf_modes():
    """Ball joints as three hinges with their Euler limits; ODE's damping as armature h D without MuJoCo's damping."""
    import re

    from ndm_gym.sim.rigid.muscle_sim import SIM_DT, character_mjcf

    char = mc.load_character()
    xml = character_mjcf(ball_joints="euler", joint_damping="ode")
    assert 'type="ball"' not in xml and f'damping="0" armature="{mc.JOINT_DAMPING * SIM_DT!r}"' in xml
    for j in mc.BALL_JOINTS:
        name, lim = mc.JOINT_NAMES[j], char["joints"][j]["euler_limits"]
        hinges = re.findall(rf'<joint name="{name}_([zyx])" type="hinge" pos="[^"]+" axis="([^"]+)" range="([^"]+)"/>', xml)
        assert [(ax, axis) for ax, axis, _ in hinges] == [("z", "1 0 0"), ("y", "0 0 1"), ("x", "0 1 0")]
        for (_, _, rng), k in zip(hinges, (2, 1, 0)):
            assert [float(x) for x in rng.split()] == [lim["lo"][k], lim["hi"][k]]
    assert character_mjcf().count('type="ball"') == len(mc.BALL_JOINTS)
    with pytest.raises(ValueError):
        character_mjcf(ball_joints="cone")


@needs_data
def test_euler_ode_sim():
    """MuscleSim with Euler hinges and ODE's damping: the reset pose is the frame's (as with ball joints), the hinge
    coordinates the Euler angles and their rates; the explicit damping force of the last substep's input state, -D qd
    for the hinges, for each Euler hinge -D axis_i . (w_child - w_parent) = -D (G qd)_i with G = J^T J the Gram matrix of
    the three hinges' axes (our x; our z turned by the first angle about x; our y), whose only off-diagonal entries are
    G_02 = G_20 = -sin(q_1); the reset frames near the Euler singularity and beyond the Euler limits, and their
    exclusion."""
    from ndm_gym.sim.rigid.muscle_sim import NQ_EULER, MuscleSim, euler_coords

    sim = MuscleSim(2, "cpu", clips=("lafan_walk",), seed=0, ball_joints="euler", joint_damping="ode")
    frames = torch.tensor([100, 2000])
    sim.reset(frames=frames)
    from ndm_gym.motion.sources import musclevae as src

    clip = src.load_clip(os.path.join(mc.CLIPS_DIR, "lafan_walk.npz"))
    pos, rot = mc.forward_kinematics(clip.root_p.double(), clip.root_quat.double(), clip.joints.double())
    state = sim.body_state()
    torch.testing.assert_close(state[..., :3].double(), pos[frames] + torch.stack((torch.zeros_like(sim.ref.lift[frames]),) * 2 +
                                                                                  (sim.ref.lift[frames],), -1)[:, None].double(), rtol=0, atol=2e-5)  # fmt: skip
    torch.testing.assert_close(quat_to_mat(state[..., 3:7].double()), rot[frames], rtol=0, atol=2e-5)
    q, qd = sim.joint_state()
    assert q.shape == (2, NQ_EULER) and qd.shape == (2, 56)
    j = mc.BALL_JOINTS[0]
    a = 7 + mc.JOINT_QPOS_ADR[j]
    torch.testing.assert_close(q[:, a : a + 3], euler_coords(clip.joints[frames].double(), j).float(), rtol=0, atol=1e-5)
    rate = (euler_coords(clip.joints[frames].double(), j) - euler_coords(clip.joints[frames - 1].double(), j)) * mc.CONTROL_HZ
    v = 6 + mc.JOINT_QVEL_ADR[j]
    torch.testing.assert_close(qd[:, v : v + 3], rate.float(), rtol=1e-4, atol=1e-4)
    terminated, _ = sim.step(sim.reference_action())
    assert not terminated.any() and torch.isfinite(sim.body_state()).all()
    state = sim.views[1]
    f, q, qd = sim.joint_f.double(), state.joint_q.double(), state.joint_qd.double()
    euler = np.zeros(56, dtype=bool)
    for j in mc.BALL_JOINTS:
        a, d = 7 + sum(1 if t == "hinge" else 3 for t in mc.JOINT_TYPES[:j]), 6 + mc.JOINT_QVEL_ADR[j]
        euler[d : d + 3] = True
        s1, r = torch.sin(q[:, a + 1]), qd[:, d : d + 3]
        want = -mc.JOINT_DAMPING * torch.stack((r[:, 0] - s1 * r[:, 2], r[:, 1], r[:, 2] - s1 * r[:, 0]), -1)
        torch.testing.assert_close(f[:, d : d + 3], want, rtol=1e-5, atol=1e-4)
    hinge = torch.tensor(~euler)
    assert torch.equal(sim.joint_f[:, hinge], (state.joint_qd * sim.damping_gain)[:, hinge])
    assert (sim.damping_gain[:6] == 0).all() and (sim.damping_gain[6:][hinge[6:]] == -mc.JOINT_DAMPING).all()
    assert (sim.joint_f[:, :6] == 0).all() and sim.joint_f[:, torch.tensor(euler)].abs().max() > 1.0
    # the reset frames' Euler statistics, from the reference's Euler angles; the exclusions drop exactly those frames
    stats = sim.reset_frame_stats()
    angles = sim.ref.ball_euler[sim.ref.reset_frames]
    lo, hi = (torch.tensor(x, dtype=torch.float32) for x in mc.ball_euler_limits())
    near, beyond = angles[..., 1].abs() > np.pi / 2 - 0.2, ((angles < lo) | (angles > hi)).any(-1)
    assert stats["reset_frames"] == stats["kept"] == len(sim.ref.reset_frames) and stats["near_gimbal"] == int(near.any(-1).sum())
    assert stats["beyond_limits"] == int(beyond.any(-1).sum()) > 0
    assert [v["beyond_limits"] for v in stats["joints"].values()] == beyond.sum(0).tolist()
    kept = MuscleSim(1, "cpu", clips=("lafan_walk",), ball_joints="euler", joint_damping="ode", reset_exclude=("limits",))
    assert torch.equal(kept.ref.reset_frames, sim.ref.reset_frames[~beyond.any(-1)]) and kept.reset_frame_stats()["kept"] == len(kept.ref.reset_frames)
    with pytest.raises(ValueError):
        MuscleSim(1, "cpu", clips=("lafan_walk",), reset_exclude=("hips",))


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA graphs need a GPU")
@pytest.mark.parametrize("joints", [("ball", "implicit"), ("euler", "ode")])
def test_graph_and_eager_agree(joints):
    """MuscleSim's CUDA graphs (compiled) against eager (not compiled) on a GPU over 40 steps of the data policy with
    masked resets (every other environment every 8 steps, staggered, and every environment that ends): the same reset
    frames and fatigue draws, reference frames (random switches every 3 steps), episode steps, off-track counts, done
    flags and just-reset marks after every step and reset. Discrete outcomes only: the GPU physics is not run-to-run
    deterministic (ndm_gym.sim.rigid.muscle_sim), the head heights of the two runs part by up to a few cm."""
    from ndm_gym.sim.rigid.muscle_sim import MuscleSim

    N, T = 64, 40
    sims = [MuscleSim(N, "cuda:0", random_count=3, graph=g, compiled=g, ball_joints=joints[0], joint_damping=joints[1]) for g in (True, False)]
    assert sims[0].reset_graph is not None and sims[0].step_graph is not None and sims[1].reset_graph is None
    for s in sims:  # the capture's warm-up drew numbers: the same generator state in both, then every environment reset
        s.gen.manual_seed(11)
        s.reset()
    env = torch.arange(N, device="cuda:0")

    def outcomes(s):
        return torch.stack((s.ref_frame, s.ep_step, s.off_count, s.terminated.long(), s.truncated.long(), s.just_reset.long()))

    assert torch.equal(outcomes(sims[0]), outcomes(sims[1])) and torch.equal(sims[0].fatigue_state(), sims[1].fatigue_state())
    ended = off = switched = resets = 0
    for t in range(T):
        before = sims[0].ref_frame.clone()
        for s in sims:
            s.step(s.data_policy(0.05))
        assert torch.equal(outcomes(sims[0]), outcomes(sims[1])), f"step {t}"
        done = sims[0].terminated | sims[0].truncated
        ended, off = ended + int(done.sum()), off + int((sims[0].off_count > 0).sum())
        switched += int((sims[0].ref_frame != before + 1).sum())
        mask = (((env + t) % 8 == 0) & (env % 2 == 0)) | done
        resets += int(mask.sum())
        for s in sims:
            s.reset(mask)
        assert torch.equal(outcomes(sims[0]), outcomes(sims[1])), f"reset after step {t}"
        for k in ("joint_q", "joint_qd"):  # the reset's writes (compiled in the graph) bit for bit
            assert torch.equal(getattr(sims[0].views[0], k)[mask], getattr(sims[1].views[0], k)[mask])
        assert torch.equal(sims[0].fatigue_state()[mask], sims[1].fatigue_state()[mask]) and torch.equal(sims[0].length_prev[mask], sims[1].length_prev[mask])
    assert ended > 0 and off > 0 and switched > 0 and resets > ended
    # without a step graph of its own (a caller's graph runs step_body), stepping the simulator alone is refused
    lone = MuscleSim(4, "cuda:0", clips=("lafan_walk",), capture_step=False, ball_joints=joints[0], joint_damping=joints[1])
    assert lone.step_graph is None and lone.reset_graph is not None
    with pytest.raises(RuntimeError, match="capture_step"):
        lone.step(lone.reference_action())


@needs_data
def test_collect_muscles_packing(sim):
    """Transitions recorded into contiguous rows across a reset pack with collect_muscles' rules and unpack bit for bit
    (collect.pack checks it); the next state and fatigue are stored only for the rows the next row does not repeat."""
    pytest.importorskip("isaaclab")
    from ndm_gym.wmbench.collect import pack, unpack
    from ndm_gym.wmbench.collect_muscles import MUSCLE_RULES

    sim.reset(frames=torch.tensor([1000, 4000]))
    buf = sim.transition_buffers(4)
    for t in range(4):
        row = {k: v[t] for k, v in buf.items()}
        sim.before_step(out=row)
        a = sim.data_policy(0.05)
        terminated, truncated = sim.step(a)
        sim.transition(row, a, terminated, truncated, out=row)
        if t == 1:
            sim.reset(torch.tensor([True, False]), torch.tensor([2000, 4002]))
    packed = pack(buf, MUSCLE_RULES)
    assert packed["s1"]["idx"].tolist() == [2, 6, 7] and packed["fatigue1"]["idx"].tolist() == [2, 6, 7]  # env 0 at row 1, the last row
    back = unpack(packed, "cpu")
    assert all(torch.equal(back[k], buf[k]) for k in buf)
    sim.reset()
