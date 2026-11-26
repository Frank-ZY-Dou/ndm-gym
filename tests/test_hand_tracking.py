"""Hand-object tracking on a small scene with the XHand's joints and keypoints: the forward kinematics against
MuJoCo's, the state layout, the task at the reference, the world model's increments, one SuperMimic update and
training's arguments; the systems options (algos.hand_fast) against the default loop, on the CPU and, where they need
the simulator and CUDA graphs, on a GPU."""

import argparse
import importlib.util
import sys
import types

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos import hand_fast
from ndm_gym.algos.blinkmimic import policy_average
from ndm_gym.algos.common import set_lr
from ndm_gym.algos.hand_algos import HAND_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.muon import MuonAdam, hidden_matrices
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.robots.hands import HandKinematics, scene_indices
from ndm_gym.train import add_train_args
from ndm_gym.train_hand import HAND_DEFAULTS, UNSUPPORTED, parse_args
from ndm_gym.tasks.hand_tracking import HOLD, HandTask, build_reference, flatten_state, unflatten_state
from ndm_gym.worldmodels.hand import HandWorldModel

# the arguments hand training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip


XHAND_FINGERS = {"thumb": 3, "index": 3, "middle": 2, "ring": 2, "pinky": 2}  # the XHand's joints per finger
ALLEGRO_FINGERS = {"thumb": 4, "index": 4, "middle": 4, "ring": 4}  # the Allegro hand's: no little finger
NUM_J = 6 + sum(XHAND_FINGERS.values())


def scene_xml(fingers: dict = XHAND_FINGERS, palm_bodies: bool = True, sides: tuple = ("right",)) -> str:
    """Per side a hand: the palm's slides and hinges (a chain of bodies, or the joints of one body), each finger a chain
    of its hinges off the palm with its fingertip's site; and a free box."""
    axes = ["1 0 0", "0 1 0", "0 0 1", "0 0 1", "1 0 0", "0 -1 0"]
    geom = '<geom size="0.01" mass="0.1"/>'
    hands, actuated = "", []
    for side in sides:
        palm_joints = [f"{side}_palm_joint_{i}" for i in range(6)]
        joint = lambda i, names=palm_joints: f'<joint name="{names[i]}" type="{"slide" if i < 3 else "hinge"}" axis="{axes[i]}" range="-3 3"/>'  # noqa: E731
        palm = "".join(f"<body>{joint(i)}{geom}" for i in range(6)) if palm_bodies else "<body>" + "".join(joint(i) for i in range(6)) + geom
        chains, finger_joints = "", []
        for k, (finger, count) in enumerate(fingers.items()):
            joints = [f"{side}_{finger}_joint_{i}" for i in range(count)]
            finger_joints += joints
            body = "".join(f'<body pos="0.01 {0.02 * k - 0.04} -0.03" quat="0.96 0.1 0.2 0.1"><joint name="{j}" axis="1 0.2 0" pos="0 0.005 0" range="0 1.5"/>'
                           f'<geom size="0.008" mass="0.02"/>' for j in joints)  # fmt: skip
            chains += body + f'<site name="{side}_{finger}_tip" pos="0 0.004 -0.027"/>' + "</body>" * count
        hands += f'{palm}<site name="{side}_palm"/>{chains}{"</body>" * (6 if palm_bodies else 1)}'
        actuated += palm_joints + finger_joints
    actuators = "".join(f'<position joint="{j}" kp="50"/>' for j in actuated)
    return (f'<mujoco><compiler angle="radian"/><worldbody><geom type="plane" size="1 1 0.1"/>{hands}<body name="right_object" pos="0.1 0 0.05">'
            f'<freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body></worldbody><actuator>{actuators}</actuator></mujoco>')  # fmt: skip


def hand_task(model: mujoco.MjModel, device: str = "cpu"):
    kin = HandKinematics(model, device)
    n = 12
    quat = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.3 * torch.rand(n, 4), dim=-1)
    clip = {"q": 0.5 * torch.rand(n, kin.num_joints).cumsum(0) / n, "p": torch.tensor([0.1, 0, 0.05]) + 0.01 * torch.rand(n, 3).cumsum(0),
            "quat": quat}  # fmt: skip
    return kin, HandTask(kin, build_reference(clip, kin, device))


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_string(scene_xml())
    return (model, *hand_task(model))


@pytest.mark.parametrize("fingers,palm_bodies,sides", [(XHAND_FINGERS, True, ("right",)), (ALLEGRO_FINGERS, False, ("right",)),
                                                       (XHAND_FINGERS, True, ("right", "left"))])  # fmt: skip
def test_kinematics_match_mujoco(fingers, palm_bodies, sides):
    """The palm as a chain of bodies or as one body's joints, five fingers or four, one hand or two: the keypoints are
    per hand the palm and the fingertips it has, and follow MuJoCo's."""
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_string(scene_xml(fingers, palm_bodies, sides))
    kin = HandKinematics(model, "cpu")
    hands = len(sides)
    assert (kin.num_joints, kin.num_keypoints, kin.num_hands) == (hands * (6 + sum(fingers.values())), hands * (1 + len(fingers)), hands)
    assert kin.palm_keypoints.tolist() == [h * (1 + len(fingers)) for h in range(hands)]
    idx, data = scene_indices(model), mujoco.MjData(model)
    names = [f"{side}_{part}" for side in sides for part in ("palm", *(f"{finger}_tip" for finger in fingers))]
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(4, kin.num_joints)
    keypoints, palm_R = kin.forward(q)
    for i in range(4):
        data.qpos[idx["qpos"]] = q[i].numpy()
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(keypoints[i].numpy(), data.site_xpos[sites], atol=1e-5)
        palms = palm_R[i].reshape(-1, 3, 3)  # one hand's (3, 3), two hands' (2, 3, 3)
        for hand, k in enumerate(kin.palm_keypoints.tolist()):
            np.testing.assert_allclose(palms[hand].numpy(), data.site_xmat[sites[k]].reshape(3, 3), atol=1e-5)


def test_one_hand_keeps_the_plain_slices(setup):
    """With one hand the kinematics, the observation, the errors, the termination and the world model's inputs are the
    single-hand expressions (plain slices of the palm, its keypoint 0 and its joints 3: and 6:) bit for bit."""
    from ndm_gym.core.rotations import mat_to_6d, rot_angle_sq

    model, kin, task = setup
    torch.manual_seed(1)
    n = 8
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(n, kin.num_joints)
    keypoints, palm_R = fk = kin.forward(q)
    assert palm_R.shape == (n, 3, 3) and kin.palm(q)[1].shape == (n, 3, 3)
    R = torch.linalg.qr(torch.randn(n, 3, 3))[0]
    S = (q, torch.randn(n, kin.num_joints), torch.randn(n, 3) * 0.05, R, torch.randn(n, 3), torch.randn(n, 3))
    a_prev, f = torch.rand(n, kin.num_joints), torch.randint(0, task.ref.frames, (n,))
    _, qd, p, _, v, w = S
    Rt = palm_R.transpose(-1, -2)
    local = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    tips = (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ palm_R
    obs = torch.cat((q, 0.1 * qd, tips.flatten(-2), mat_to_6d(palm_R), local(p - keypoints[..., 0, :]), mat_to_6d(Rt @ R), local(v), 0.1 * local(w), a_prev), -1)
    assert torch.equal(task.obs(S, fk, a_prev), obs) and torch.equal(kin.fingertips_about_palms(keypoints, palm_R), tips)
    e = task.errors(S, fk, f)
    assert torch.equal(e["finger"], (q[..., 6:] - task.ref.q[f][..., 6:]).pow(2).mean(-1).sqrt())
    assert torch.equal(e["palm_ori_sq"], rot_angle_sq(palm_R, task.ref.palm_R[f]))
    assert torch.equal(task.termination(S, fk, f), (e["object_pos"] > task.cfg.term_object) | (e["keypoint"][..., 0] > task.cfg.term_palm))
    wm = HandWorldModel(kin.num_joints)
    u = torch.rand(n, kin.num_joints)
    inputs = torch.cat((q[..., 3:], qd, local(p - keypoints[..., 0, :]), mat_to_6d(Rt @ R), v, w, p[..., 2:3], mat_to_6d(R), u - q), -1)
    assert torch.equal(wm.inputs(S, fk, u), inputs) and torch.equal(wm.inputs(S, kin.palm(q), u), inputs)
    assert all(b.dtype == torch.float32 for b in wm.buffers())  # none but the normalizers' and the error's floors
    two = HandWorldModel(2 * NUM_J, [0, 1, 2, NUM_J, NUM_J + 1, NUM_J + 2], [0, 6])
    assert two.posed.tolist() == [*range(3, NUM_J), *range(NUM_J + 3, 2 * NUM_J)] and "posed" not in two.state_dict()


@pytest.mark.parametrize("fingers,sides,sizes", [(ALLEGRO_FINGERS, ("right",), (22, 57, 3 * 22 + 3 * 4 + 21)),
                                                 (XHAND_FINGERS, ("right", "left"), (36, 85, 3 * 36 + 3 * 10 + 2 * 21))])  # fmt: skip
def test_sizes_follow_the_hands(monkeypatch, fingers, sides, sizes):
    """A four-fingered hand with sixteen finger joints, and two XHands: the state, the observation and the networks take
    their sizes, and an update of each algorithm runs, also with the world model on the palms' kinematics alone."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    torch.manual_seed(0)
    kin, task = hand_task(mujoco.MjModel.from_xml_string(scene_xml(fingers, palm_bodies=False, sides=sides)))
    assert (kin.num_joints, task.state_dim, task.obs_dim) == sizes
    for argv in (["--algo", "supermimic"], ["--algo", "supertrack", "--st_task_weights"], ["--algo", "supermimic", "--wm_palm_fk"]):
        args, algo, buffer = agent_and_replay(kin, task, argv)
        assert algo.pi.net[-1].out_features == kin.num_joints
        assert torch.isfinite(algo.wm_update(buffer))
        out = algo.policy_update(buffer)  # (loss, value loss: none for SuperTrack, ...)
        assert torch.isfinite(out[0]) and torch.isfinite(out[3])


def test_reference_is_optimal_and_held(setup):
    _, kin, task = setup
    ref = task.ref
    assert ref.frames == 12 + HOLD and torch.equal(ref.q[-1], ref.q[-HOLD - 1]) and float(ref.qd[-1].abs().max()) == 0
    f = torch.arange(ref.frames)
    S = ref.state(f)
    fk, a = kin.forward(S[0]), torch.zeros(ref.frames, NUM_J)
    assert task.obs(S, fk, a).shape == (ref.frames, task.obs_dim)
    assert torch.allclose(task.reward(S, fk, f, a, a), torch.full((ref.frames,), task.max_step_reward), atol=1e-5)
    assert not task.termination(S, fk, f).any() and float(task.soft_termination(S, fk, f).max()) < 1e-3
    assert torch.allclose(task.setpoint(a, f), ref.q[ref.next_frame(f)])
    far = (S[0], S[1], S[2] + 0.5, *S[3:])
    assert task.termination(far, fk, f).all() and float(task.soft_termination(far, fk, f).min()) > 0.99


def test_state_and_increments_round_trip(setup):
    _, kin, task = setup
    f = torch.arange(task.ref.frames - 1)
    S, S1 = task.ref.state(f), task.ref.state(f + 1)
    x = flatten_state(S)
    assert x.shape == (len(f), task.state_dim)
    for a, b in zip(unflatten_state(x), S, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    wm = HandWorldModel(NUM_J, hidden=(32,))
    wm.out_norm.update(wm.targets(S, S1))
    assert float(wm.error(S1, S1).abs().max()) < 1e-4
    increments = (wm.targets(S, S1) - wm.out_norm.mean) / wm.out_norm.std()

    class Exact(torch.nn.Module):  # a network that outputs the normalized increments reproduces the next state
        def forward(self, x):
            return increments

    wm.net = Exact()
    for a, b in zip(wm.step(S, kin.forward(S[0]), S[0]), S1, strict=True):
        assert torch.allclose(a, b, atol=1e-4)


def test_supermimic_update(setup, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    args = parse_args(["--clip", "clip", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch",
                       "16", "--wm_window", "4", "--pi_batch", "16", "--critic_epochs", "1", "--critic_minibatches", "2"])  # fmt: skip
    algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    buffer = ReplayBuffer(16, 8, "cpu", fields=replay_fields(task), priorities=False)
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
    assert all(torch.isfinite(x) for x in out) and 0 < float(out[2]) <= 1 and 0 < float(out[4]) <= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))


def test_training_arguments():
    """Every argument of ndm_gym.train is implemented or refused; the defaults are HAND_DEFAULTS over LAFAN1's."""
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    names = {action.dest for action in parser._actions if action.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    assert set(HAND_DEFAULTS) <= set(SUPPORTED)
    args = parse_args(["--clip", "clip"])
    assert all(getattr(args, name) == value for name, value in HAND_DEFAULTS.items())
    assert args.lr == parser.get_default("lr") and args.wm_window == parser.get_default("wm_window")
    for refused in (["--algo", "blinkmimic"], ["--wm_backbone", "rigidformer"], ["--hist", "4"], ["--utd", "1"], ["--st_fail_w", "2"],
                    ["--st_vel_terms"], ["--st_obj_w", "2"], ["--st_unit_scales"], ["--st_task_weights"], ["--st_imag_noise", "0.2"],
                    ["--algo", "supertrack", "--st_task_weights", "--st_vel_terms"], ["--algo", "supertrack", "--st_task_weights", "--st_unit_scales"],
                    ["--algo", "supertrack", "--gamma", "0.995"], ["--algo", "supertrack", "--lam", "0.9"],
                    ["--algo", "supertrack", "--critic_epochs", "3"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--clip", "clip"] + refused)
    st = parse_args(["--clip", "clip", "--algo", "supertrack", "--st_fail_w", "2", "--st_vel_terms", "--st_obj_w", "2", "--st_imag_noise", "0.2"])
    assert (st.algo, st.st_fail_w, st.st_vel_terms, st.st_obj_w, st.st_imag_noise) == ("supertrack", 2, True, 2, 0.2)
    assert not st.st_unit_scales and parse_args(["--clip", "clip", "--algo", "supertrack", "--st_unit_scales"]).st_unit_scales
    for algo in ("supermimic", "supertrack"):  # both algorithms' option, 1 (off) unless given
        assert parse_args(["--clip", "clip", "--algo", algo, "--bptt_decay", "0.95"]).bptt_decay == 0.95
        assert parse_args(["--clip", "clip", "--algo", algo]).bptt_decay == 1.0


def test_accel_world_model_round_trip(setup):
    """The acceleration model: a network that outputs the normalized accelerations reproduces the next velocities and, by
    semi-implicit Euler, the positions they lead to; its error is in units of the increments' spread, as the other's."""
    _, kin, task = setup
    f = torch.arange(task.ref.frames - 1)
    S, S1 = task.ref.state(f), task.ref.state(f + 1)
    wm = HandWorldModel(NUM_J, hidden=(32,), kind="accel")
    wm.out_norm.update(wm.targets(S, S1)), wm.err_norm.update(wm.increments(S, S1))
    assert wm.targets(S, S1).shape[-1] == NUM_J + 6 and float(wm.error(S1, S1).abs().max()) < 1e-4
    accel = (wm.targets(S, S1) - wm.out_norm.mean) / wm.out_norm.std()

    class Exact(torch.nn.Module):
        def forward(self, x):
            return accel

    wm.net = Exact()
    q1, qd1, p1, R1, v1, w1 = wm.step(S, kin.forward(S[0]), S[0])
    for a, b in ((qd1, S1[1]), (v1, S1[4]), (w1, S1[5])):
        assert torch.allclose(a, b, atol=1e-4)
    dt = 0.02
    assert torch.allclose(q1, S[0] + dt * S1[1], atol=1e-5) and torch.allclose(p1, S[2] + dt * S1[4], atol=1e-5)
    from ndm_gym.core.rotations import so3_exp

    assert torch.allclose(R1, so3_exp(dt * S1[5]) @ S[3], atol=1e-5)
    with pytest.raises(ValueError, match="kind"):
        HandWorldModel(NUM_J, hidden=(32,), kind="other")


@pytest.mark.parametrize("extra", [[], ["--st_vel_terms", "--st_obj_w", "2", "--st_fail_w", "2", "--st_imag_noise", "0.2"],
                                   ["--st_vel_terms", "--st_obj_w", "2", "--st_unit_scales"]])  # fmt: skip
def test_supertrack_update(setup, monkeypatch, extra):
    """SuperTrack on the hand: the acceleration model, no critic; the first policy update calibrates the terms' weights to
    1 / their summed error, the object's then weighed --st_obj_w times; the update moves the policy; the reference's own
    states have no error. --st_unit_scales: the fingertips' positions about the palm and the object's position take the
    keypoints' weight, the object's rotation the palm's."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    args = parse_args(["--clip", "clip", "--algo", "supertrack", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches",
                       "2", "--wm_batch", "16", "--wm_window", "4", "--pi_batch", "16"] + extra)  # fmt: skip
    algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    assert algo.wm.kind == "accel" and algo.critic is None and len(algo.optimizers) == 2
    assert len(algo.terms) == (9 if extra else 6)
    buffer = ReplayBuffer(16, 8, "cpu", fields=replay_fields(task), priorities=False)
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
    fr = torch.arange(task.ref.frames)
    Sr = task.ref.state(fr)
    assert float(algo._errors(Sr, kin.forward(Sr[0]), fr).abs().max()) < 1e-3  # the reference tracks itself
    before = [p.clone() for p in algo.pi.parameters()]
    assert torch.isfinite(algo.wm_update(buffer)) and not algo.bal.calibrated
    out = algo.policy_update(buffer)
    assert algo.bal.calibrated and bool((algo.bal.w > 0).all())
    scale = algo.term_scale.tolist()
    assert scale == ([1, 1, 1, 1, 2, 2, 1, 1, 1] if extra else [1] * 6)
    w = algo.bal.w.tolist()
    if "--st_unit_scales" in extra:
        assert algo.cal_idx.tolist() == [0, 0, 2, 3, 0, 2, 6, 7, 8] and w[1] == w[4] == w[0] and w[5] == w[2]
    else:
        assert algo.cal_idx.tolist() == list(range(len(w))) and len(set(w)) == len(w)  # every term by its own error
    assert torch.isfinite(out[0]) and torch.isnan(out[1]) and 0 < float(out[2]) <= 1 and 0 < float(out[4]) <= 1 and torch.isfinite(out[3])
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    sd = algo.state_dict()
    assert "critic" not in sd and "bal" in sd
    other = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    other.load_state_dict(sd)
    assert other.bal.calibrated and torch.equal(other.bal.w, algo.bal.w)


# ---------------------------------------------------------------------- the systems options (algos.hand_fast)
SYSTEMS = ["--fused_adam", "--reset_masked", "--no_sync"]
SMALL = ["--clip", "clip", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch", "16", "--wm_window",
         "4", "--pi_batch", "16"]  # fmt: skip
SM_CRITIC = ["--critic_epochs", "1", "--critic_minibatches", "2"]  # SuperMimic's critic, small (SuperTrack refuses these options)


def small(extra: list) -> list:
    """SMALL with ``extra``, and SuperMimic's small critic unless ``extra`` is SuperTrack's."""
    return SMALL + extra + ([] if "supertrack" in extra else SM_CRITIC)


ST_ARM = ["--algo", "supertrack", "--st_fail_w", "2", "--st_imag_noise", "0.2", "--st_vel_terms", "--st_obj_w", "2"]
CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")


def agent_and_replay(kin, task, argv, n=8, steps=16, dev="cpu"):
    """An agent with the arguments ``argv`` and a replay of ``steps`` transitions of ``n`` worlds along the reference, from seed 0."""
    torch.manual_seed(0)
    args = parse_args(small(argv))
    algo = HAND_ALGOS[args.algo](args, kin, task, dev)
    buffer = ReplayBuffer(steps, n, dev, fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.randint(0, task.ref.frames - 1, (n,), device=dev)
    S, a_prev, done = task.ref.state(f), torch.zeros(n, kin.num_joints, device=dev), torch.zeros(n, dtype=torch.bool, device=dev)
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


def test_supertrack_task_weights(setup, monkeypatch):
    """--st_task_weights: no calibration; a term weighs, per unit of error and imagined step, the task's reward weight over its
    kernel's width, the fingertips about the palm nothing; the failure penalty's unit is the reward weights' sum; the update
    leaves the weights and moves the policy; the checkpoint carries them."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    args = parse_args(["--clip", "clip", "--algo", "supertrack", "--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches",
                       "2", "--wm_batch", "16", "--wm_window", "4", "--pi_batch", "16", "--st_task_weights", "--st_obj_w", "2", "--st_fail_w", "2"])  # fmt: skip
    algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    assert algo.terms == algo.TERMS and algo.bal.calibrated and algo.step_worth == 5.0
    expect = torch.tensor([1 / 0.04, 0, 0.5 / 0.4, 0.5 / 0.4, 2 / 0.04, 1 / 0.5]) / 4
    assert torch.allclose(algo.bal.w, expect) and algo.term_scale.tolist() == [1, 1, 1, 1, 2, 2]
    buffer = ReplayBuffer(16, 8, "cpu", fields=replay_fields(task), priorities=False)
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
    assert torch.allclose(algo.bal.w, expect) and torch.isfinite(out[0]) and torch.isnan(out[1]) and torch.isfinite(out[3])
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    other = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    other.load_state_dict(algo.state_dict())
    assert other.bal.calibrated and torch.equal(other.bal.w, algo.bal.w)


def test_checkpoint_to_host():
    """hand_fast.to_host: a checkpoint's nested structure with equal tensors of their own storage, the other values kept."""
    from collections import OrderedDict

    from ndm_gym.algos.hand_fast import to_host

    w = torch.randn(3, 4)
    tree = {"net": OrderedDict(w=w, b=torch.randn(4), n=torch.tensor(7), flag=torch.tensor(True), empty=torch.zeros(0, 2)),
            "opt": {"state": {0: {"step": torch.tensor(5.0), "m": torch.randn(3, 4)}}, "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.95), "params": [0]}]},
            "iteration": 12, "args": {"wm_hidden": [8, 8], "clip": "c"}}  # fmt: skip
    out = to_host(tree)
    assert type(out["net"]) is OrderedDict and list(out["net"]) == list(tree["net"]) and out["iteration"] == 12 and out["args"] == tree["args"]
    assert out["opt"]["param_groups"] == tree["opt"]["param_groups"] and type(out["opt"]["param_groups"][0]["betas"]) is tuple
    for a, b in ((out["net"][k], tree["net"][k]) for k in tree["net"]):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) and a.data_ptr() != b.data_ptr() or a.numel() == 0
    assert torch.equal(out["opt"]["state"][0]["m"], tree["opt"]["state"][0]["m"]) and float(out["opt"]["state"][0]["step"]) == 5.0
    out["net"]["w"].zero_()
    assert bool((w != 0).any()) and float(out["net"]["b"].abs().sum()) > 0  # a copy, and no storage shared between the outputs


def test_resume_requires_the_checkpoints_algorithm_and_terms():
    """hand_fast.check_resume: a resumed run with another algorithm or other SuperTrack terms is refused; a checkpoint from
    before the options (no such keys) is the default SuperMimic's."""
    from ndm_gym.algos.hand_fast import check_resume

    st = parse_args(["--clip", "clip", "--algo", "supertrack", "--st_task_weights"])
    check_resume(vars(st), st)
    for other in (parse_args(["--clip", "clip"]), parse_args(["--clip", "clip", "--algo", "supertrack"])):
        with pytest.raises(ValueError, match="--resume"):
            check_resume(vars(st), other)
    check_resume({}, parse_args(["--clip", "clip"]))
    with pytest.raises(ValueError, match="--algo"):
        check_resume({}, st)
    # the task's objective too: a run resumed with another target-rate weight is refused
    run = parse_args(["--clip", "clip", "--target_rate_w", "4"])
    check_resume(vars(run), run)
    with pytest.raises(ValueError, match="--resume: --target_rate_w must match"):
        check_resume(vars(parse_args(["--clip", "clip"])), run)
    # and the collection's noise-free worlds, and the gradient's decay (both algorithms'), each a run's own from before the option
    for name, value in (("noise_free_frac", "0.25"), ("bptt_decay", "0.95")):
        for algo in ("supermimic", "supertrack"):
            base = ["--clip", "clip", "--algo", algo]
            run = parse_args(base + [f"--{name}", value])
            check_resume(vars(run), run)
            for a, b in ((parse_args(base), run), (run, parse_args(base))):
                with pytest.raises(ValueError, match=f"--resume: --{name} must match"):
                    check_resume(vars(a), b)
            check_resume({k: v for k, v in vars(parse_args(base)).items() if k != name}, parse_args(base))


def test_systems_arguments():
    """The systems options are off by default; the combinations that cannot run are refused; a resumed run keeps its
    checkpoint's optimizers and policy average."""
    args = parse_args(["--clip", "clip"])
    assert not (args.fused_adam or args.reset_masked or args.no_sync or args.collect_graph or args.update_graphs)
    assert (args.pi_opt, args.wm_opt, args.pi_ema, args.pi_lr_mult, args.lr_decay_ret, args.lr_decay_by) == ("adam", "adam", 0.0, 1.0, 0.0, 0)
    assert not args.compile_eager
    for refused in (["--no_sync"], ["--no_sync", "--fused_adam"], ["--no_sync", "--reset_masked"], ["--collect_graph"], ["--compile_eager"],
                    ["--update_graphs", "--fused_adam", "--reset_masked"], ["--pi_ema", "1"], ["--lr_decay_ret", "0.5"],
                    ["--pi_lr_mult", "2", "--lr_decay_by", "100"], ["--pi_opt", "sgd"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--clip", "clip"] + refused)
    on = parse_args(["--clip", "clip", *ST_ARM, *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_opt", "muon", "--wm_opt",
                     "muon", "--pi_ema", "0.995", "--pi_lr_mult", "4", "--lr_decay_ret", "0.9", "--lr_decay_by", "1400"])  # fmt: skip
    assert on.no_sync and on.collect_graph and on.compile_eager and on.update_graphs and (on.pi_opt, on.wm_opt, on.pi_ema) == ("muon", "muon", 0.995)
    hand_fast.check_resume({}, args)  # a checkpoint from before the options, resumed without them
    hand_fast.check_resume(vars(on), on)
    for k in hand_fast.RESUME_MATCH:  # each option's other value in the checkpoint
        other = (not getattr(on, k)) if isinstance(getattr(on, k), bool) else getattr(args, k)
        if other == getattr(on, k):  # a number ``on`` leaves at its default (the target-rate weight): another value
            other = getattr(on, k) + 1.0
        with pytest.raises(ValueError, match=k):
            hand_fast.check_resume({**vars(on), k: other}, on)


def test_optimizers(setup, monkeypatch):
    """Without the options the agents' optimizers are Adam as before them, also for a checkpoint's arguments that lack the
    options; --fused_adam, --update_graphs and Muon change them."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    args = parse_args(small([]))
    options = ("fused_adam", "reset_masked", "no_sync", "collect_graph", "update_graphs", "pi_opt", "wm_opt", "pi_ema")
    for a in (args, argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in options})):
        algo = HAND_ALGOS["supermimic"](a, kin, task, "cpu")
        for opt, betas in ((algo.opt_wm, (0.9, 0.999)), (algo.opt_pi, (0.9, 0.95)), (algo.opt_v, (0.9, 0.999))):
            assert type(opt) is torch.optim.Adam and opt.defaults["lr"] == args.lr and opt.defaults["betas"] == betas
            assert not opt.defaults["fused"] and not opt.defaults["capturable"]
        assert not algo.no_sync
    fused = HAND_ALGOS["supermimic"](parse_args(small(["--fused_adam"])), kin, task, "cpu")
    assert all(o.defaults["fused"] and not o.defaults["capturable"] for o in fused.optimizers)
    graphs = HAND_ALGOS["supertrack"](parse_args(small(ST_ARM + SYSTEMS + ["--update_graphs"])), kin, task, "cpu")
    assert graphs.no_sync and len(graphs.optimizers) == 2
    assert all(o.defaults["fused"] and o.defaults["capturable"] and torch.is_tensor(o.param_groups[0]["lr"]) for o in graphs.optimizers)
    assert graphs.opt_wm.param_groups[0]["lr"] is not graphs.opt_pi.param_groups[0]["lr"]  # each optimizer its own learning rate
    muon = HAND_ALGOS["supermimic"](parse_args(small(["--pi_opt", "muon", "--wm_opt", "muon"])), kin, task, "cpu")
    assert isinstance(muon.opt_wm, MuonAdam) and isinstance(muon.opt_pi, MuonAdam) and type(muon.opt_v) is torch.optim.Adam


def test_survive_and_finite_step():
    """``survive`` is cumprod, with --no_sync as a loop of products; ``finite_step`` takes the optimizer's step unless the
    gradient's norm is non-finite, tested on the host or (--no_sync, the fused Adam's gate) on the device: a skipped step
    leaves the parameters and the step count."""
    torch.manual_seed(0)
    d = 0.3 * torch.rand(6, 5)
    assert torch.equal(hand_fast.survive(d, False), torch.cumprod(1 - d, 0))
    assert torch.allclose(hand_fast.survive(d, True), torch.cumprod(1 - d, 0), rtol=1e-6)
    for no_sync in (False, True):
        lin = torch.nn.Linear(4, 3)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2, fused=no_sync)
        x = torch.randn(8, 4)
        for finite in (True, False, True):
            before = [p.detach().clone() for p in lin.parameters()]
            opt.zero_grad(set_to_none=True)
            lin(x).pow(2).mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(lin.parameters(), 1.0)
            if not finite:
                norm = norm * float("nan")
                lin.weight.grad.fill_(float("nan"))
            hand_fast.finite_step(opt, lin.parameters(), norm, no_sync)
            moved = any(not torch.equal(a, b) for a, b in zip(before, lin.parameters(), strict=True))
            assert moved == finite and all(bool(torch.isfinite(p).all()) for p in lin.parameters())
        assert [int(s["step"]) for s in opt.state.values()] == [2, 2]
    with pytest.raises(ValueError, match="fused"):
        hand_fast.finite_step(torch.optim.Adam(lin.parameters()), lin.parameters(), norm, True)


@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_no_sync_updates_match_default(setup, monkeypatch, extra):
    """--fused_adam --reset_masked --no_sync: from the same state and with the same random draws, the world-model and
    policy updates return the default's values and move the parameters as the default's do, up to rounding (the fused
    kernel, the survival as a loop of products); the normalizers written in place hold the default's statistics."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    runs = []
    for flags in ([], SYSTEMS):
        monkeypatch.setattr(RunningNorm, "no_sync", bool(flags))
        args, algo, buffer = agent_and_replay(kin, task, extra + flags)
        params = [p for m in (algo.wm, algo.pi) for p in m.parameters()]
        start = [p.detach().clone() for p in params]
        torch.manual_seed(1)
        out = torch.stack([algo.wm_update(buffer), *algo.policy_update(buffer), *algo.policy_update(buffer)])
        move = torch.cat([(p.detach() - p0).flatten() for p, p0 in zip(params, start, strict=True)])
        runs.append((out, move, torch.cat([b.flatten() for m in (algo.wm, algo.pi) for b in m.buffers()])))
    (out0, move0, norms0), (out1, move1, norms1) = runs
    finite = torch.isfinite(out0)  # SuperTrack's value loss is NaN
    assert int(finite.sum()) == (11 if not extra else 9) and torch.equal(finite, torch.isfinite(out1))
    assert torch.allclose(out0[finite], out1[finite], rtol=1e-5)
    assert torch.equal(norms0, norms1)
    assert float(move0.abs().max()) > 0.5 * args.lr and float((move0 - move1).abs().max()) < 0.005 * args.lr


class KinematicSim:
    """HandSim's interface without physics, on the CPU: a control step puts the joints at their targets and the object at
    the reference's next frame, worlds 0 and 1 losing it after a few frames (a termination); an episode ends there or at
    ``last``, and its world restarts at the first frame. ``step`` resets the ended worlds as HandSim.step does (by index,
    the frame counter rebound), ``step_nosync`` as HandSim.step_nosync does (by mask, in place)."""

    def __init__(self, task, n: int):
        self.task, self.ref, self.N, self.device = task, task.ref, n, "cpu"
        self.f = torch.zeros(n, dtype=torch.long)
        self.last = torch.randint(3, task.ref.frames, (n,))
        self.s = flatten_state(task.ref.state(self.f)).clone()

    def state(self) -> torch.Tensor:
        return self.s.clone()

    def _move(self, u):
        f1 = self.ref.next_frame(self.f)
        _, qd, p, R, v, w = self.ref.state(f1)
        lost = (torch.arange(self.N) < 2) & (f1 > 4)
        self.s = flatten_state((u, qd, p - 0.3 * lost.unsqueeze(-1), R, v, w))
        return f1

    def _ends(self):
        S1 = unflatten_state(self.s)
        term = self.task.termination(S1, self.task.kin.forward(S1[0]), self.f)
        return self.s.clone(), term, term | (self.f >= self.last)

    def step(self, u, reset: bool = True):
        self.f = self._move(u)
        s1, term, done = self._ends()
        if reset:
            self.s[done] = flatten_state(self.ref.state(torch.zeros(int(done.sum()), dtype=torch.long)))
            self.f[done] = 0
            self.last[done] = self.ref.frames - 1
        return s1, term, done

    def step_nosync(self, u, physics=None, fk=None):
        self.f.copy_(self._move(u))
        s1, term, done = self._ends()
        self.s.copy_(torch.where(done.unsqueeze(-1), flatten_state(self.ref.state(torch.zeros_like(self.f))), self.s))
        self.f.masked_fill_(done, 0)
        self.last.masked_fill_(done, self.ref.frames - 1)
        return s1, term, done

    # step_nosync in HandSim's three parts (--compile_collect); its resets draw nothing
    def reset_draws(self):
        return None

    def set_targets(self, u):
        self.u = u

    def advance(self, physics=None):
        self._move(self.u)

    def finish_nosync(self, s, fk=None, draws=None):
        assert draws is None
        self.f.copy_(self.ref.next_frame(self.f))
        s1, term, done = self._ends()
        self.s.copy_(torch.where(done.unsqueeze(-1), flatten_state(self.ref.state(torch.zeros_like(self.f))), self.s))
        self.f.masked_fill_(done, 0)
        self.last.masked_fill_(done, self.ref.frames - 1)
        return s1, term, done


def default_collect(sim, task, algo, buffer, st: dict, noise: float, noise_free=None):
    """train_hand's default collection step (its closure ``collect``) on the loop's state ``st``; ``noise_free``: --noise_free_frac's
    multiplier of the noise's draws (hand_fast.NoiseFree)."""
    s, f = sim.state(), sim.f.clone()
    S = unflatten_state(s)
    if noise_free is None:
        a = algo.act(S, st["a_prev"], f, noise=noise)
    else:
        a = algo.act(S, st["a_prev"], f, noise=noise, z=noise_free.noise(st["a_prev"]))
    u = task.setpoint(a, f)
    s1, term, done = sim.step(u)
    S1, f1 = unflatten_state(s1), task.ref.next_frame(f)
    fk1 = task.kin.forward(S1[0])
    algo.update_wm_norms(S, S1, u)
    buffer.add({"s": s, "s1": s1, "u": u, "a_prev": st["a_prev"], "f": f, "done": done})
    st["ep_len"].add_(1)
    acc = st["acc"]
    acc["r"] += task.reward(S1, fk1, f1, a, st["a_prev"]).sum() / task.max_step_reward
    acc["object_err"] += task.errors(S1, fk1, f1)["object_pos"].sum()
    acc["steps"] += sim.N
    acc["done"] += done.sum()
    acc["term"] += term.sum()
    acc["ep_len"] += st["ep_len"][done].sum()
    st["ep_len"][done] = 0
    st["a_prev"] = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)


@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
@pytest.mark.parametrize("graphed,compiled", [(False, False), (True, False), (True, True), (False, True)],
                         ids=["eager", "device_head", "device_head_compile_collect", "compile_collect"])  # fmt: skip
def test_collector_matches_default_step(setup, monkeypatch, extra, graphed, compiled):
    """hand_fast.Collector's step on a simulator without physics computes the default loop's: the same replay, log sums,
    episode lengths, previous actions and normalizers, bit for bit; with the replay's device-side write head
    (--collect_graph's) and the forward kinematics through the agent's ``fk`` (--compile_eager's, not compiled here) too;
    and with --compile_collect's step (its parts before and after the physics, not compiled here, the random draws made
    first) too."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    n, steps = 8, 90
    runs = []
    for collector in (False, True):
        monkeypatch.setattr(RunningNorm, "no_sync", collector)
        torch.manual_seed(0)
        flags = (["--collect_graph", "--compile_eager"] if graphed else []) + (["--compile_collect"] if compiled else [])
        args = parse_args(small(extra + ((SYSTEMS + flags) if collector else [])))
        algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
        sim = KinematicSim(task, n)
        buffer = ReplayBuffer(32, n, "cpu", fields=replay_fields(task), priorities=False)
        if collector and graphed:
            buffer.use_device_ptr(device_size=True)
        st = {"a_prev": torch.zeros(n, NUM_J), "ep_len": torch.zeros(n), "acc": {k: torch.zeros(()) for k in hand_fast.ACC_KEYS}}
        c = hand_fast.Collector(sim, task, algo, buffer, args) if collector else None
        for k in range(steps):
            torch.manual_seed(k)
            if c is None:
                default_collect(sim, task, algo, buffer, st, args.action_noise)
                continue
            c.core()
            if graphed:
                buffer.advance_host()
        if c is not None:
            st = {"a_prev": c.a_prev, "ep_len": c.ep_len, "acc": c.acc}
            assert (algo.fk is c.fk) == (graphed or compiled) and c.compiled == compiled
            assert graphed == (buffer.ptr_t is not None) and (not graphed or (int(buffer.ptr_t), int(buffer.size_t)) == (buffer.ptr, buffer.size))
        runs.append((buffer, st, sim, trained(algo)))
    (b0, st0, sim0, t0), (b1, st1, sim1, t1) = runs
    assert (b0.ptr, b0.size) == (b1.ptr, b1.size) == (steps % 32, 32)
    for k in b0.data:
        assert torch.equal(b0.data[k], b1.data[k]), k
    for k in hand_fast.ACC_KEYS:
        assert torch.equal(st0["acc"][k], st1["acc"][k]), k
    assert torch.equal(st0["ep_len"], st1["ep_len"]) and torch.equal(st0["a_prev"], st1["a_prev"])
    assert torch.equal(sim0.s, sim1.s) and torch.equal(sim0.f, sim1.f) and torch.equal(sim0.last, sim1.last)
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert 0 < float(st0["acc"]["term"]) < float(st0["acc"]["done"])  # episodes ended by termination and at their last frame
    host_head = ReplayBuffer(32, n, "cpu", fields=replay_fields(task), priorities=False)
    with pytest.raises(ValueError, match="device-side head"):
        hand_fast.Collector(sim, task, algo, host_head, parse_args(small(SYSTEMS + ["--collect_graph"])))


def test_policy_average(setup, monkeypatch):
    """--pi_ema: the average's first step copies the policy, the later ones are AveragedModel's on its weights and
    normalizer; the checkpoint holds the average as 'pi' and the trained policy as 'pi_live', and a resumed run has both."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    args, algo, buffer = agent_and_replay(kin, task, ["--pi_ema", "0.9"])
    avg, ref = hand_fast.PolicyAverage(algo.pi, 0.9), policy_average(algo.pi, 0.9)
    wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, avg)
    assert torch.isfinite(wm_update())
    for i in range(3):
        algo.pi.norm.update(torch.randn(8, algo.pi.norm.mean.shape[0]))  # the policy's normalizer moves with the collection
        assert len(policy_update()) == 5 and int(avg.avg.n_averaged) == i + 1
        ref.update_parameters(algo.pi)
        assert all(torch.equal(a, b) for a, b in zip(avg.avg.state_dict().values(), ref.state_dict().values(), strict=True))
    live, mean = algo.pi.state_dict(), avg.avg.module.state_dict()
    assert all(not torch.equal(mean[k], live[k]) for k in ("net.0.weight", "norm.mean"))  # the average lags the trained policy
    ck = avg.checkpoint(algo.state_dict())
    assert all(torch.equal(ck["pi"][k], mean[k]) and torch.equal(ck["pi_live"][k], live[k]) for k in live)
    other = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    resumed = hand_fast.PolicyAverage(other.pi, 0.9)
    other.load_state_dict(resumed.resume(ck))
    assert int(resumed.avg.n_averaged) == 3 and all(torch.equal(other.pi.state_dict()[k], live[k]) for k in live)
    assert all(torch.equal(a, b) for a, b in zip(resumed.avg.state_dict().values(), avg.avg.state_dict().values(), strict=True))
    evaluated = HAND_ALGOS[args.algo](args, kin, task, "cpu")  # evaluate_hand loads the checkpoint as it is: the average
    evaluated.load_state_dict(ck)
    assert all(torch.equal(evaluated.pi.state_dict()[k], mean[k]) for k in mean)
    with pytest.raises(ValueError, match="pi_avg"):
        resumed.resume(algo.state_dict())


@pytest.mark.parametrize("flags", [[], SYSTEMS], ids=["host", "no_sync"])
def test_muon_updates(setup, monkeypatch, flags):
    """--pi_opt muon --wm_opt muon: the updates move the hidden matrices of the world model's and the policy's MLP, also
    with the policy's step gated on the device (--no_sync); the optimizers' states load into another agent."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", bool(flags))
    _, kin, task = setup
    args, algo, buffer = agent_and_replay(kin, task, ["--pi_opt", "muon", "--wm_opt", "muon"] + flags)
    hidden = hidden_matrices([algo.wm.net]) + hidden_matrices([algo.pi.net])
    before = [p.detach().clone() for p in hidden]
    assert len(hidden) == 3 and torch.isfinite(algo.wm_update(buffer))
    assert all(bool(torch.isfinite(x)) for x in algo.policy_update(buffer))
    assert all(not torch.equal(a, b) and bool(torch.isfinite(b).all()) for a, b in zip(before, hidden, strict=True))
    other = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    other.load_state_dict(algo.state_dict())
    for a, b in zip(hidden_matrices([algo.pi.net]), hidden_matrices([other.pi.net]), strict=True):
        assert torch.equal(algo.opt_pi.muon.state[a]["momentum_buffer"], other.opt_pi.muon.state[b]["momentum_buffer"])


# ---------------------------------------------------------------------- --phase_times (hand_fast.PhaseTimes)
PHASE_COLUMNS = ("iteration", "collect_ms", "wm_ms", "policy_ms", "critic_ms", "log_ms", "ckpt_ms", "host_ms", "it_per_s")  # phase_times.csv


class Clock:
    """A clock the test advances, in time.perf_counter's place; ``takes(seconds)``: a function that takes so long."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def takes(self, seconds: float):
        def fn():
            self.t += seconds

        return fn


def phase_rows(run) -> np.ndarray:
    """The rows of phase_times.csv in the directory ``run``, under its one header."""
    lines = (run / "phase_times.csv").read_text().splitlines()
    assert lines[0] == ",".join(PHASE_COLUMNS) and lines.count(lines[0]) == 1
    return np.array([[float(x) for x in line.split(",")] for line in lines[1:]])


def phase_summary(text: str) -> dict:
    """PhaseTimes.finish's summary: the ms per phase (None: not timed), "total", "it_per_s" and the first line as "head"."""
    head, *lines = text.splitlines()
    ms = {line.split()[0]: (None if line.split()[1] == "-" else float(line.split()[1])) for line in lines}
    return {**ms, "it_per_s": float(lines[-1].split()[3]), "head": head}


def check_loop_rows(rows: np.ndarray, iterations: list, saved: list):
    """Rows of train_hand's loop: one per log row, with time in the collection, the world-model updates, the policy update,
    the log row and the host's rest, and in the checkpoints where ``saved``; the phases add up to the iteration."""
    timed = [PHASE_COLUMNS.index(k) for k in ("collect_ms", "wm_ms", "policy_ms", "log_ms", "host_ms")]
    assert rows[:, 0].tolist() == iterations and (rows[:, timed] > 0).all()
    assert (rows[:, PHASE_COLUMNS.index("ckpt_ms")] > 0).tolist() == saved
    assert np.allclose(np.nansum(rows[:, 1:-1], 1) * rows[:, -1], 1000, rtol=1e-4)


def test_phase_times_accounting(tmp_path):
    """hand_fast.PhaseTimes on a clock the test advances. A phase's time is the time between its boundaries and the host's
    is the rest, so the phases add up to the iteration; a row per --log_every iterations holds the means since the last
    row; SuperMimic's critic update has its column where the agent calls ``lap`` (without --update_graphs) and NaN with
    the graphs, as an agent without a critic; the summary is over iterations 1,001 to 3,000 of a run from its first
    iteration and over the second half of a shorter or a resumed run, which continues the file; a new run refuses it."""
    assert hand_fast.PhaseTimes.COLUMNS == PHASE_COLUMNS and not parse_args(["--clip", "clip"]).phase_times

    def run(log_dir, argv, critic, first, last, resumed=False):
        """A loop from iteration ``first`` to ``last``: 1 ms of collection (10 ms in the first 1,000 iterations), 4 ms of
        world-model updates, from iteration 301 a policy update of 3 ms, 2 ms for the critic and 0.5 ms for the
        average, 0.25 ms of host work, 5 ms a log row and 20 ms a checkpoint. Returns the file's rows and the summary."""
        log_dir.mkdir(exist_ok=True)
        args = parse_args(small(argv) + ["--log_dir", str(log_dir), "--log_every", "50", "--ckpt_every", "200", "--phase_times"])
        clock, algo = Clock(), types.SimpleNamespace(critic=critic, lap=None)
        phases = hand_fast.PhaseTimes(args, algo, "cpu", resumed, clock=clock)
        assert (algo.lap is not None) == (critic is not None and not args.update_graphs)

        def policy():
            clock.t += 3e-3
            if algo.lap is not None:  # as HandSuperMimic.policy_update: its step ends, then its critic's update
                algo.lap("policy")
            clock.t += 2e-3
            if algo.lap is not None:
                algo.lap("critic")
            clock.t += 0.5e-3

        slow, collect, wm_update = (phases.timed(k, clock.takes(t)) for k, t in (("collect", 10e-3), ("collect", 1e-3), ("wm", 4e-3)))
        policy_update = phases.timed("policy", policy)
        for it in range(first, last + 1):
            clock.t += 0.25e-3
            (slow if it <= 1000 else collect)()
            wm_update()
            if it > 300:
                policy_update()
            if it % 50 == 0:
                clock.t += 5e-3
                phases.lap("log")
            if it % 200 == 0:
                clock.t += 20e-3
                phases.lap("ckpt")
            phases.end(it)
        return phase_rows(log_dir), phase_summary(phases.finish())

    rows, mean = run(tmp_path / "a", [], object(), 1, 3200)
    assert rows[:, 0].tolist() == list(range(50, 3201, 50))
    assert np.allclose(rows[0], [50, 10, 4, 0, 0, 0.1, 0, 0.25, 1000 / 14.35], rtol=1e-4)  # iterations 1 to 50: no policy update, no checkpoint
    assert np.allclose(rows[23], [1200, 1, 4, 3.5, 2, 0.1, 0.4, 0.25, 1000 / 11.25], rtol=1e-4)  # 1,151 to 1,200, with a checkpoint
    assert np.allclose(rows[:, 1:-1].sum(1) * rows[:, -1], 1000, rtol=1e-4)  # the phases add up to the iteration
    want = {"collect": 1, "wm": 4, "policy": 3.5, "critic": 2, "log": 0.1, "ckpt": 0.1, "host": 0.25, "total": 10.95}
    assert "iterations 1001-3000" in mean["head"] and {k: mean[k] for k in want} == pytest.approx(want, abs=1e-3)
    assert mean["it_per_s"] == pytest.approx(1000 / 10.95, abs=0.05)
    # a shorter run: its second half; continued by a resumed run, whose summary is over the second half of its own iterations
    rows, mean = run(tmp_path / "b", [], object(), 1, 600)
    assert "iterations 301-600" in mean["head"] and (mean["collect"], mean["policy"]) == pytest.approx((10, 3.5), abs=1e-3)
    rows, mean = run(tmp_path / "b", [], object(), 601, 3200, resumed=True)
    assert rows[:, 0].tolist() == list(range(50, 3201, 50))
    assert "iterations 1901-3200" in mean["head"] and mean["collect"] == pytest.approx(1, abs=1e-3)
    with pytest.raises(ValueError, match="phase_times.csv exists"):
        run(tmp_path / "b", [], object(), 1, 50)
    (tmp_path / "b" / "phase_times.csv").write_text("iteration,collect_ms\n")
    with pytest.raises(ValueError, match="other columns"):
        run(tmp_path / "b", [], object(), 3201, 3250, resumed=True)
    # the critic's update in the policy update's graph, and an agent without a critic: in the policy's time, or absent
    for name, argv, critic in (("c", SYSTEMS + ["--update_graphs"], object()), ("d", ST_ARM, None)):
        rows, mean = run(tmp_path / name, argv, critic, 1, 400)
        assert np.isnan(rows[:, 4]).all() and mean["critic"] is None and "iterations 201-400" in mean["head"]
        assert np.allclose(rows[-1, [1, 2, 3, 5, 6, 7]], [10, 4, 5.5, 0.1, 0.4, 0.25], rtol=1e-4)
        assert np.allclose(np.nansum(rows[:, 1:-1], 1) * rows[:, -1], 1000, rtol=1e-4)


def same_tree(a, b) -> bool:
    """Whether two checkpoints' trees hold the same values, the tensors bit for bit."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_phase_times_in_the_training_loop(setup, monkeypatch, tmp_path, capsys, extra):
    """train_hand's loop on the CPU, a simulator without physics in HandSim's place, with and without --phase_times: the
    same checkpoint (networks, normalizers, optimizers, policy average) and the same log bit for bit, the wall-clock
    columns aside; with it phase_times.csv has a row per log row whose phases add up to the iteration, a checkpoint's
    time in the row of its iteration, SuperMimic's critic update timed apart and no critic for SuperTrack; the summary
    is over the second half of the short run."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "DEVICE", "cpu")
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: setup)
    hand_sim = types.ModuleType("ndm_gym.sim.rigid.hand_sim")  # HandSim's module, which needs the simulator and a GPU
    hand_sim.HandSim = lambda model, task, num_envs, device, stagger=False: KinematicSim(task, num_envs)
    monkeypatch.setitem(sys.modules, "ndm_gym.sim.rigid.hand_sim", hand_sim)
    argv = [*small(extra), "--pi_opt", "muon", "--wm_opt", "muon", "--pi_ema", "0.9", "--pi_lr_mult", "2", "--lr_decay_ret", "0.99", "--lr_decay_by",
            "4", "--lr_decay_len", "2", "--num_envs", "8", "--buffer_steps", "16", "--warmup_steps", "8", "--wm_only_iters", "1", "--log_every", "2",
            "--ckpt_every", "4", "--iterations", "6"]  # fmt: skip
    runs = {}
    for name, flag in (("off", []), ("on", ["--phase_times"])):
        monkeypatch.setattr(sys, "argv", ["train_hand", *argv, *flag, "--log_dir", str(tmp_path / name)])
        train_hand.main()
        ck = torch.load(tmp_path / name / "last.pt", map_location="cpu", weights_only=False)
        assert ck["iteration"] == 6 and ck["args"]["phase_times"] == bool(flag) and int(ck["pi_avg"]["n_averaged"]) == 5
        log = np.loadtxt(tmp_path / name / "train_log.csv", delimiter=",", skiprows=1)
        runs[name] = ({k: v for k, v in ck.items() if k not in ("wall_time", "args")}, log)
    (ck0, log0), (ck1, log1) = runs["off"], runs["on"]
    assert {"wm", "pi", "pi_live", "pi_avg", "opt_wm", "opt_pi"} <= set(ck0) and same_tree(ck0, ck1)
    assert not same_tree(ck0["pi"], ck0["pi_live"])  # the comparison tells trained weights apart
    same = [i for i, k in enumerate(train_hand.LOG_COLUMNS) if k not in ("wall_h", "it_per_s")]
    assert log0.shape == (3, len(train_hand.LOG_COLUMNS)) and np.array_equal(log0[:, same], log1[:, same], equal_nan=True)
    assert not (tmp_path / "off" / "phase_times.csv").exists()
    rows = phase_rows(tmp_path / "on")
    check_loop_rows(rows, [2, 4, 6], [False, True, False])
    critic = rows[:, PHASE_COLUMNS.index("critic_ms")]
    assert np.isnan(critic).all() if extra else (critic > 0).all()
    mean = phase_summary(capsys.readouterr().out.split("--phase_times: ")[-1])
    assert "iterations 4-6" in mean["head"] and (mean["critic"] is None) == bool(extra) and mean["policy"] > 0 and mean["total"] > 0


# ---------------------------------------------------------------------- the updates' engineering options (algos.hand_fast)
def test_engineering_arguments():
    """--compile_updates, --muon_batched, --elementwise_3x3, --combo_kernels, --wm_palm_fk, --compile_collect,
    --mix_order_reductions, --autotune_gemm and --compile_rollout are off by default; the combinations that cannot run are
    refused."""
    args = parse_args(["--clip", "clip"])
    assert not (args.compile_updates or args.muon_batched or args.elementwise_3x3 or args.combo_kernels or args.wm_palm_fk)
    assert not (args.compile_collect or args.mix_order_reductions or args.autotune_gemm or args.compile_rollout)
    for refused in (["--compile_updates"], ["--compile_updates", "--fused_adam", "--reset_masked"], ["--muon_batched"], ["--compile_collect"],
                    ["--compile_collect", "--fused_adam", "--reset_masked"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--clip", "clip"] + refused)
    on = parse_args(["--clip", "clip", *SYSTEMS, "--compile_updates", "--muon_batched", "--wm_opt", "muon", "--elementwise_3x3", "--combo_kernels",
                     "--wm_palm_fk", "--compile_collect", "--mix_order_reductions", "--autotune_gemm", "--compile_rollout"])  # fmt: skip
    assert on.compile_updates and on.muon_batched and on.elementwise_3x3 and on.combo_kernels and on.wm_palm_fk
    assert on.compile_collect and on.mix_order_reductions and on.autotune_gemm and on.compile_rollout


def test_inductor_settings(monkeypatch):
    """--mix_order_reductions and --autotune_gemm set inductor's configuration (hand_fast.mix_order_reductions,
    autotune_gemm), not its CUDA graphs."""
    import torch._inductor.config as inductor

    for k in ("mix_order_reduction", "mix_order_reduction_non_strict_mode"):
        monkeypatch.setattr(inductor.triton, k, False)
    monkeypatch.setattr(inductor, "max_autotune_gemm", False)
    cudagraphs = inductor.triton.cudagraphs
    hand_fast.mix_order_reductions()
    assert inductor.triton.mix_order_reduction and inductor.triton.mix_order_reduction_non_strict_mode and not inductor.max_autotune_gemm
    hand_fast.autotune_gemm()
    assert inductor.max_autotune_gemm and inductor.triton.cudagraphs == cudagraphs


@pytest.mark.parametrize("fingers,palm_bodies,sides", [(XHAND_FINGERS, True, ("right",)), (ALLEGRO_FINGERS, False, ("right",)),
                                                       (XHAND_FINGERS, True, ("right", "left"))])  # fmt: skip
def test_palm_kinematics(fingers, palm_bodies, sides):
    """HandKinematics.palm (--wm_palm_fk) is forward's palm keypoints and rotations bit for bit, from the palms' chains
    alone, for one hand or two."""
    kin = HandKinematics(mujoco.MjModel.from_xml_string(scene_xml(fingers, palm_bodies, sides)), "cpu")
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(64, kin.num_joints, generator=torch.Generator().manual_seed(0))
    (keypoints, palm_R), (palm, palm_R1) = kin.forward(q), kin.palm(q)
    assert palm.shape == (64, len(sides), 3) and torch.equal(palm, keypoints[:, kin.palm_keypoints]) and torch.equal(palm_R1, palm_R)
    assert len(kin.palm_chains) == len(sides) * (6 if palm_bodies else 1) < len(kin.steps)


@pytest.mark.parametrize("extra", [[], ST_ARM, ST_ARM[:6] + ["--st_task_weights"]], ids=["supermimic", "supertrack", "supertrack_task_weights"])
def test_engineering_options_match_default(setup, monkeypatch, extra):
    """With the compilation left out (torch.compile as the identity, as on the CPU here), --compile_updates and
    --wm_palm_fk compute what the updates compute without them: from the same state and random draws the same values,
    parameters, normalizers and optimizer states bit for bit, SuperMimic's critic's too (on a GPU the compiled parts may
    round differently). --muon_batched builds the optimizers with Muon's batched step, which batches the world model's two
    hidden matrices."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    runs = []
    for flags in ([], ["--compile_updates", "--wm_palm_fk"]):
        args, algo, buffer = agent_and_replay(kin, task, extra + SYSTEMS + ["--pi_opt", "muon", "--wm_opt", "muon"] + flags)
        assert algo.compile_updates == bool(flags) and (algo.wm_fk == kin.palm) == bool(flags)
        torch.manual_seed(1)
        out = torch.stack([algo.wm_update(buffer), *algo.policy_update(buffer), algo.wm_update(buffer), *algo.policy_update(buffer)])
        opts = [part for o in algo.optimizers for part in ((o.muon, o.adam) if isinstance(o, MuonAdam) else (o,))]
        state = [v for o in opts for st in o.state.values() for v in st.values()]
        runs.append((out, trained(algo), state))
    (out0, t0, s0), (out1, t1, s1) = runs
    finite = torch.isfinite(out0)  # SuperTrack's value loss is NaN
    assert int(finite.sum()) == (12 if not extra else 10) and torch.equal(out0[finite], out1[finite]) and torch.equal(finite, torch.isfinite(out1))
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert len(s0) == len(s1) > 0 and all(torch.equal(a, b) for a, b in zip(s0, s1, strict=True))
    args, algo, _ = agent_and_replay(kin, task, extra + ["--pi_opt", "muon", "--wm_opt", "muon", "--muon_batched"])
    assert all(o.batched for o in algo.optimizers if isinstance(o, MuonAdam)) and len(algo.optimizers) == (3 if not extra else 2)
    assert [tuple(p.shape) for p in algo.opt_wm.muon.param_groups[0]["params"]] == [(32, 32)]  # the small test model's one hidden matrix


@pytest.mark.parametrize("extra", [[], ["--sm_reuse_input"], ST_ARM], ids=["supermimic", "supermimic_reuse_input", "supertrack"])
def test_compile_rollout_matches_default(setup, monkeypatch, extra):
    """--compile_rollout (torch.compile as the identity, as on the CPU here): the policy update's rollout as one function
    computes what the steps' functions compute, from the same state and random draws (SuperTrack's imagination noise
    drawn before the rollout, in the steps' order): the same values, parameters, normalizers and optimizer states bit for
    bit, SuperMimic's critic's too."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    runs = []
    for flags in ([], ["--compile_rollout"]):
        args, algo, buffer = agent_and_replay(kin, task, extra + SYSTEMS + ["--compile_updates"] + flags)
        assert (algo.rollout is not None) == bool(flags)
        torch.manual_seed(1)
        out = torch.stack([algo.wm_update(buffer), *algo.policy_update(buffer), algo.wm_update(buffer), *algo.policy_update(buffer)])
        state = [v for o in algo.optimizers for st in o.state.values() for v in st.values()]
        runs.append((out, trained(algo), state))
    (out0, t0, s0), (out1, t1, s1) = runs
    finite = torch.isfinite(out0)  # SuperTrack's value loss is NaN
    assert int(finite.sum()) == (12 if "supertrack" not in extra else 10) and torch.equal(out0[finite], out1[finite])
    assert torch.equal(finite, torch.isfinite(out1)) and all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert len(s0) == len(s1) > 0 and all(torch.equal(a, b) for a, b in zip(s0, s1, strict=True))


def test_sm_reuse_input(setup, monkeypatch):
    """--sm_reuse_input (SuperMimic's; refused for SuperTrack): the imagined steps take the policy's input from the critic's
    input of the step before; from the same state and random draws the policy update returns the same values and moves
    the parameters as without it (eagerly on the CPU, bit for bit: the same operations on the same values)."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    with pytest.raises(SystemExit):
        parse_args(small(ST_ARM) + ["--sm_reuse_input"])
    runs, inputs = [], []
    for flags in ([], ["--sm_reuse_input"]):
        args, algo, buffer = agent_and_replay(kin, task, SYSTEMS + flags)
        assert algo.reuse_input == bool(flags)
        step = algo.imag_step
        monkeypatch.setattr(algo, "imag_step", lambda *a: inputs.append(len(a)) or step(*a))  # noqa: B023
        torch.manual_seed(1)
        out = torch.stack([*algo.policy_update(buffer), *algo.policy_update(buffer)])
        runs.append((out, trained(algo)))
    assert inputs == [4] * (2 * args.horizon) + [5] * (2 * args.horizon)
    (out0, t0), (out1, t1) = runs
    assert torch.equal(out0, out1) and all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))


# ------------------------------------------------- on a GPU: the simulator's steps and the CUDA graphs
@pytest.fixture(scope="module")
def gpu_setup():
    """The scene of ``setup`` on the GPU, made to simulate (rotor inertia and damping on the hand's joints, stiffer
    actuators), with its kinematics and a task: the hand above the floor and an object whose reference leaves it by 4 cm
    a frame, so that the worlds terminate at the fourth frame, those whose first episode ends earlier (HandSim's stagger)
    at its end."""
    torch.manual_seed(0)
    xml = scene_xml().replace('<compiler angle="radian"/>', '<compiler angle="radian"/><default><joint armature="0.01" damping="1"/></default>')
    model = mujoco.MjModel.from_xml_string(xml.replace('kp="50"', 'kp="500"'))
    kin = HandKinematics(model, CUDA)
    n = 12
    quat = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.3 * torch.rand(n, 4), dim=-1)
    q = 0.1 * torch.rand(n, NUM_J).cumsum(0) / n
    q[:, 2] += 0.3  # the palm's height
    p = torch.tensor([0.1, 0, 0.05]) + 0.04 * torch.arange(n).unsqueeze(-1) * torch.tensor([1.0, 0, 0])
    return model, kin, HandTask(kin, build_reference({"q": q, "p": p, "quat": quat}, kin, CUDA))


def on_stream(stream, fn):
    """``fn()`` on ``stream`` between the caller's work before and after it (hand_fast.Collector's eager call)."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return out


@GPU
def test_step_without_sync_and_masked_reset(gpu_setup):
    """HandSim.step_nosync on Warp's stream is HandSim.step with the ended worlds reset by mask: from the same state and
    targets, the same next state, terminations and episode ends, and the same worlds reset to the same values (resets
    without perturbation; with it, the reset worlds stay within its bounds and the others are untouched)."""
    from ndm_gym.sim.rigid.hand_sim import HandSim

    model, kin, task = gpu_setup
    n = 64
    torch.manual_seed(0)
    a, b = (HandSim(model, task, n, CUDA, joint_noise=0.0, object_noise=0.0) for _ in range(2))
    a.last = torch.randint(1, task.ref.frames, (n,), device=CUDA)
    b.last.copy_(a.last)
    stream, ends, worst = b.torch_stream(), torch.zeros(2, device=CUDA), 0.0
    for _ in range(60):
        for x, y in ((a.qpos, b.qpos), (a.qvel, b.qvel), (a.ctrl, b.ctrl), (a.warmstart, b.warmstart)):
            y.copy_(x)  # each step from the same state: two simulations drift apart by rounding
        u = task.setpoint(2 * torch.rand(n, NUM_J, device=CUDA) - 1, a.f)
        (s1a, ta, da), (s1b, tb, db) = a.step(u), on_stream(stream, lambda: b.step_nosync(u))  # noqa: B023
        worst = max(worst, float((s1a - s1b).abs().max()))
        assert torch.equal(ta, tb) and torch.equal(da, db) and torch.equal(a.f, b.f) and torch.equal(a.last, b.last)
        reset = da.unsqueeze(-1)
        assert torch.equal(torch.where(reset, a.qpos, 0), torch.where(reset, b.qpos, 0)) and bool((a.qvel[da] == 0).all())
        assert torch.equal(torch.where(reset, a.ctrl, 0), torch.where(reset, b.ctrl, 0)) and bool((b.qvel[da] == 0).all())
        ends += torch.stack((ta.sum(), (da & ~ta).sum()))
    assert worst < 1e-3, worst  # one step of two default simulations from the same state differs by up to 1e-4 here
    assert bool((ends > 0).all())  # terminations and episodes that reached their last frame
    c = HandSim(model, task, n, CUDA)  # the perturbed resets
    before, mask = (c.qpos.clone(), c.ctrl.clone(), c.qvel.clone()), torch.arange(n, device=CUDA) % 3 == 0
    c.qvel.fill_(1.0), c.f.fill_(5), c.last.fill_(7)
    c.reset_masked(mask)
    o, ref = c.object_qpos, task.ref
    assert bool((c.f == torch.where(mask, 0, 5)).all()) and bool((c.last == torch.where(mask, ref.frames - 1, 7)).all())
    assert bool((c.qvel == (~mask).float().unsqueeze(-1)).all())
    assert torch.equal(c.qpos[~mask], before[0][~mask]) and torch.equal(c.ctrl[~mask], before[1][~mask])
    q = c.qpos[mask][:, c.qpos_idx]
    assert torch.equal(q, c.ctrl[mask][:, c.ctrl_idx]) and not torch.equal(q, before[0][mask][:, c.qpos_idx])
    assert float((q - ref.q[0]).abs().max()) < 6 * c.joint_noise and bool(((q >= kin.q_lo) & (q <= kin.q_hi)).all())
    obj = c.qpos[mask][:, o : o + 3] - ref.p[0]
    assert float(obj[:, :2].abs().max()) <= c.object_noise and float(obj[:, 2].abs().max()) == 0


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_collect_graph_replays_the_eager_step(gpu_setup, monkeypatch, extra):
    """--collect_graph: the captured collection step, replayed, computes what the step run eagerly on Warp's stream does
    (--no_sync) when both draw the same random numbers: the same episode ends and frames, and the same replay, log sums,
    actions and normalizers up to the rounding by which two simulations differ; with --compile_eager's compiled forward
    kinematics too, and with --compile_collect's compiled step, run eagerly and captured."""
    from ndm_gym.sim.rigid.hand_sim import HandSim

    monkeypatch.setattr(RunningNorm, "no_sync", True)
    model, kin, task = gpu_setup
    n, steps, runs = 64, 40, []
    for flags in ([], ["--collect_graph"], ["--collect_graph", "--compile_eager"], ["--compile_collect"], ["--collect_graph", "--compile_collect"]):
        graphed = "--collect_graph" in flags
        torch.manual_seed(0)
        args = parse_args(small(extra + SYSTEMS + flags))
        algo = HAND_ALGOS[args.algo](args, kin, task, CUDA)
        sim = HandSim(model, task, n, CUDA, stagger=True)
        buffer = ReplayBuffer(32, n, CUDA, fields=replay_fields(task), priorities=False)
        if graphed:
            buffer.use_device_ptr(device_size=True)
        c = hand_fast.Collector(sim, task, algo, buffer, args)
        for k in range(steps):
            torch.manual_seed(100 + k)
            c()
        torch.cuda.synchronize()
        assert (c.graph is not None) == graphed and (buffer.ptr, buffer.size) == (steps % 32, 32)
        runs.append((buffer, c, sim, trained(algo)))
    b0, c0, sim0, t0 = runs[0]
    for b1, c1, sim1, t1 in runs[1:]:
        assert b1.ptr_t is None or (int(b1.ptr_t), int(b1.size_t)) == (b1.ptr, b1.size)
        for k in ("f", "done"):
            assert torch.equal(b0.data[k], b1.data[k]), k
        for k in ("s", "s1", "u", "a_prev"):
            assert torch.allclose(b0.data[k], b1.data[k], atol=1e-3), (k, float((b0.data[k] - b1.data[k]).abs().max()))
        for k in ("steps", "done", "term", "ep_len"):
            assert torch.equal(c0.acc[k], c1.acc[k]), k
        assert torch.allclose(c0.acc["r"], c1.acc["r"], rtol=1e-4) and torch.allclose(c0.acc["object_err"], c1.acc["object_err"], rtol=1e-4)
        assert torch.equal(c0.ep_len, c1.ep_len) and torch.allclose(c0.a_prev, c1.a_prev, atol=1e-3)
        assert torch.equal(sim0.f, sim1.f) and torch.equal(sim0.last, sim1.last) and torch.allclose(sim0.state(), sim1.state(), atol=1e-3)
        assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-4) for a, b in zip(t0, t1, strict=True))
    assert 0 < float(c0.acc["term"]) < float(c0.acc["done"])
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(n, NUM_J, device=CUDA)  # the compiled forward kinematics against the eager
    for a, b in zip(runs[2][1].fk(q), kin.forward(q), strict=True):
        assert torch.allclose(a, b, atol=1e-5)


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM, ["--bptt_decay", "0.9"], [*ST_ARM, "--bptt_decay", "0.9"]],
                         ids=["supermimic", "supertrack", "supermimic_decay", "supertrack_decay"])  # fmt: skip
def test_update_graphs_replay_the_eager_updates(gpu_setup, monkeypatch, extra):
    """--update_graphs: the world-model updates and the policy update (with the policy average's step) captured as CUDA
    graphs and replayed return the values and leave the parameters, normalizers, optimizer states and average that the
    same updates run eagerly do, when both draw the same random numbers; the replays follow the learning rate's schedule."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = gpu_setup
    runs = []
    for graphs in (False, True):
        args, algo, buffer = agent_and_replay(kin, task, extra + SYSTEMS + ["--update_graphs", "--pi_ema", "0.9"], n=16, dev=CUDA)
        buffer.use_device_ptr(device_size=True)
        avg = hand_fast.PolicyAverage(algo.pi, 0.9)
        if graphs:
            wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, avg)
        else:
            wm_update, policy_update = hand_fast.update_functions(algo, buffer, argparse.Namespace(update_graphs=False), avg)
        outs = []
        for k in range(7):  # three eager calls, the capture, three more replays
            torch.manual_seed(10 + k)
            for o in algo.optimizers:
                set_lr(o, args.lr * (1 - 0.1 * k))
            outs += [wm_update(), *policy_update()]
        torch.cuda.synchronize()
        state = [v for o in algo.optimizers for v in ((o.adam if isinstance(o, MuonAdam) else o).state_dict()["state"]).values()]
        runs.append((torch.stack(outs), trained(algo) + list(avg.avg.state_dict().values()), [x for s in state for x in s.values()]))
    (out0, t0, s0), (out1, t1, s1) = runs
    finite = torch.isfinite(out0)
    assert torch.equal(finite, torch.isfinite(out1)) and int(finite.sum()) == 7 * (6 if args.algo == "supermimic" else 5)
    assert torch.equal(out0[finite], out1[finite]), float((out0[finite] - out1[finite]).abs().max())
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert len(s0) == len(s1) and all(torch.equal(a, b) for a, b in zip(s0, s1, strict=True))
    assert int(s0[0]) == 7 * args.wm_batches  # the world model's Adam step count


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_training_loop_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, extra):
    """train_hand's loop with every systems option on the small scene: the graphs are captured, the log's rows are finite,
    the checkpoint holds the policy average and the ramp's state, and a resumed run continues the log."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: gpu_setup)
    run = tmp_path / "run"
    argv = [*small(extra), *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_ema", "0.9", "--pi_lr_mult", "2",
            "--lr_decay_ret", "0.99", "--lr_decay_by", "8", "--lr_decay_len", "4", "--num_envs", "32", "--buffer_steps", "16", "--warmup_steps", "8",
            "--wm_only_iters", "2", "--log_every", "4", "--log_dir", str(run)]  # fmt: skip
    for iterations, resume in ((12, []), (20, ["--resume", str(run / "last.pt")])):
        monkeypatch.setattr(sys, "argv", ["train_hand", *argv, "--iterations", str(iterations), *resume])
        train_hand.main()
        rows = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
        assert rows[:, 0].tolist() == list(range(4, iterations + 1, 4))
        cols = [train_hand.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "ep_len")]
        assert np.isfinite(rows[1:, cols]).all() and (rows[:, train_hand.LOG_COLUMNS.index("sim_r")] > 0).all()
        ck = torch.load(run / "last.pt", map_location=CUDA, weights_only=False)
        assert ck["iteration"] == iterations and {"pi", "pi_live", "pi_avg"} <= set(ck) and ck["pi_lr_decay_at"] == 8
        assert int(ck["pi_avg"]["n_averaged"]) == iterations - 2
        assert any(not torch.equal(ck["pi"][k], ck["pi_live"][k]) for k in ck["pi"])
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv, "--iterations", "24", "--resume", str(run / "last.pt"), "--pi_opt", "muon"])
    with pytest.raises(ValueError, match="pi_opt"):
        train_hand.main()


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_phase_times_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, capsys, extra):
    """--phase_times in train_hand's loop with every systems option on the small scene (the collection step and the
    updates as CUDA graphs, Muon, the policy average): the graphs are captured and replayed between the boundaries, the
    log's rows are finite, phase_times.csv has a row per log row whose phases add up to the iteration, and the critic's
    update, in the policy update's graph, has no time of its own for either algorithm."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: gpu_setup)
    run = tmp_path / "run"
    argv = [*small(extra), *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_opt", "muon", "--wm_opt", "muon", "--pi_ema",
            "0.9", "--pi_lr_mult", "2", "--lr_decay_ret", "0.99", "--lr_decay_by", "8", "--lr_decay_len", "4", "--num_envs", "32", "--buffer_steps",
            "16", "--warmup_steps", "8", "--wm_only_iters", "2", "--log_every", "4", "--ckpt_every", "8", "--iterations", "12", "--log_dir", str(run),
            "--phase_times"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv])
    train_hand.main()
    out = capsys.readouterr().out
    assert "captured one collection step" in out and "captured the world-model updates" in out and "captured the policy update" in out
    log = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
    cols = [train_hand.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "ep_len")]
    assert log[:, 0].tolist() == [4, 8, 12] and np.isfinite(log[1:, cols]).all()
    rows = phase_rows(run)
    check_loop_rows(rows, [4, 8, 12], [False, True, False])
    assert np.isnan(rows[:, PHASE_COLUMNS.index("critic_ms")]).all()
    text = out.split("--phase_times: ")[-1]
    mean = phase_summary(text)
    assert "iterations 7-12" in mean["head"] and mean["critic"] is None and text.count("one CUDA graph") == (4 if not extra else 3)


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_engineering_options_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, capsys, extra):
    """--compile_updates, --muon_batched, --wm_palm_fk, --elementwise_3x3 and --combo_kernels in train_hand's loop with
    every systems option on the small scene (compilation left out, inductor's settings recorded instead of made): the
    collection step and the updates are captured as CUDA graphs and replayed, and the log's rows are finite."""
    import ndm_gym.train as train
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: gpu_setup)
    made = []
    monkeypatch.setattr(train, "elementwise_3x3_products", lambda: made.append("elementwise_3x3"))
    monkeypatch.setattr(train, "combo_kernels", lambda: made.append("combo_kernels"))
    run = tmp_path / "run"
    argv = [*small(extra), *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_opt", "muon", "--wm_opt", "muon", "--pi_ema",
            "0.9", "--compile_updates", "--muon_batched", "--wm_palm_fk", "--elementwise_3x3", "--combo_kernels", "--num_envs", "32",
            "--buffer_steps", "16", "--warmup_steps", "8", "--wm_only_iters", "2", "--log_every", "4", "--iterations", "12", "--log_dir", str(run),
            *([] if extra else ["--sm_reuse_input"])]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv])
    train_hand.main()
    out = capsys.readouterr().out
    assert "captured one collection step" in out and "captured the world-model updates" in out and "captured the policy update" in out
    assert made == ["elementwise_3x3", "combo_kernels"]
    log = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
    cols = [train_hand.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "ep_len")]
    assert log[:, 0].tolist() == [4, 8, 12] and np.isfinite(log[1:, cols]).all()


# ---------------------------------------------- the target-rate penalty (--target_rate_w; off by default, the same for both algorithms)
ST_TASK = ["--algo", "supertrack", "--st_fail_w", "2", "--st_imag_noise", "0.2", "--st_task_weights"]  # the task's weights


def test_target_rate_flag_and_cost(setup):
    """--target_rate_w sets the task's target-rate weight (none by default); ``smooth_cost`` is None when it is off, else the rate
    term (each joint's target change over its kernel width, squared, mean), as written out here; the reward is the same less that
    cost times the control step."""
    import dataclasses

    from ndm_gym.robots.hands import NUM_PALM
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.hand_tracking import RewardCfg
    from ndm_gym.train_hand import smoothing

    _, kin, task = setup
    assert smoothing(parse_args(small([]))) == {} and task.smooth_cost(None, torch.zeros(2, NUM_J), torch.zeros(2, NUM_J)) is None
    assert smoothing(parse_args(small(["--target_rate_w", "0.5"]))) == {"w_target_rate": 0.5}
    with pytest.raises(ValueError):
        smoothing(parse_args(small(["--target_rate_w", "-1"])))
    torch.manual_seed(0)
    f = torch.arange(3)
    S = task.ref.state(f)
    a, a_prev = 0.3 * torch.randn(3, NUM_J), 0.3 * torch.randn(3, NUM_J)
    fk = kin.forward(S[0])
    r0 = task.reward(S, fk, f, a, a_prev)
    rate = HandTask(kin, task.ref, dataclasses.replace(RewardCfg(), w_target_rate=0.5))
    da = a - a_prev
    width = torch.tensor([0.04] * 3 + [0.4] * 3 + [0.4] * (NUM_J - NUM_PALM))
    want_rate = 0.5 * (kin.action_scale * da / width).pow(2).mean(-1)
    assert torch.allclose(rate.smooth_cost(S, a, a_prev), want_rate)
    assert torch.allclose(rate.reward(S, fk, f, a, a_prev), r0 - want_rate * CONTROL_DT)


def test_st_objective_adds_the_costs(setup, monkeypatch):
    """SuperTrack's objective: with the task's smoothness costs, the loss is the same plus their sum over the imagined steps,
    mean over the batch; without them it is unchanged."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    _, algo, _ = agent_and_replay(kin, task, ST_ARM)
    H, B = algo.horizon, 5
    terms, d = torch.rand(len(algo.terms)), 0.1 * torch.rand(H, B)
    acts, costs = [torch.randn(B, NUM_J) for _ in range(H)], [torch.rand(B) for _ in range(H)]
    base, alive = algo._st_objective(terms, d, acts)
    same, _ = algo._st_objective(terms, d, acts, [None] * H)
    more, _ = algo._st_objective(terms, d, acts, costs)
    assert torch.equal(base, same) and torch.allclose(more, base + torch.stack(costs).sum(0).mean())


@pytest.mark.parametrize("algo_flags", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_target_rate_updates(setup, monkeypatch, algo_flags):
    """The target-rate penalty in both algorithms' updates: finite losses and gradients, and a policy update that differs from the
    one without it (the same state and draws)."""
    flags = ["--target_rate_w", "1.0"]
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    import dataclasses

    from ndm_gym.train_hand import smoothing

    outs = []
    for extra in ([], flags):
        args = parse_args(small(algo_flags + extra))
        t = HandTask(kin, task.ref, dataclasses.replace(task.cfg, **smoothing(args)))
        _, algo, buffer = agent_and_replay(kin, t, algo_flags + extra)
        torch.manual_seed(1)
        algo.wm_update(buffer)
        out = torch.stack(list(algo.policy_update(buffer)))
        outs.append((out, [p.detach().clone() for p in algo.pi.parameters()]))
    (o0, p0), (o1, p1) = outs
    assert torch.isfinite(o1[[0, 3, 4]]).all()  # pi_loss, gradient norm, survival
    assert any(not torch.equal(a, b) for a, b in zip(p0, p1, strict=True))


def test_st_cost_mean_flag(setup, monkeypatch):
    """--st_cost_mean: SuperTrack's objective adds the costs' mean over the imagined steps (the sum over the horizon) and is otherwise
    unchanged; refused for SuperMimic and without --st_task_weights."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    _, algo, _ = agent_and_replay(kin, task, ST_TASK + ["--st_cost_mean"])
    assert algo.cost_mean
    H, B = algo.horizon, 5
    terms, d = torch.rand(len(algo.terms)), 0.1 * torch.rand(H, B)
    acts, costs = [torch.randn(B, NUM_J) for _ in range(H)], [torch.rand(B) for _ in range(H)]
    base, _ = algo._st_objective(terms, d, acts)
    more, _ = algo._st_objective(terms, d, acts, costs)
    assert torch.allclose(more, base + torch.stack(costs).mean(0).mean())
    algo.cost_mean = False  # the default path: the sum
    summed, _ = algo._st_objective(terms, d, acts, costs)
    assert torch.allclose(summed - base, H * (more - base))
    for argv in (["--st_cost_mean"], ST_ARM + ["--st_cost_mean"]):
        with pytest.raises(SystemExit):
            parse_args(small(argv))


def test_st_cost_mean_relative_weight(setup, monkeypatch):
    """The penalty's scale: a target change of one kernel width at one imagined step costs the target-rate weight, as a
    tracking error of one width costs the term's weight. In SuperMimic's reward both enter per step at the control step's scale. In
    SuperTrack's loss with --st_task_weights the tracking error costs its weight over the horizon; with --st_cost_mean the change costs
    its weight over the horizon too, so the ratio is the reward's; without it the change costs the horizon times more."""
    import dataclasses

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    w = 1.0
    t = HandTask(kin, task.ref, dataclasses.replace(task.cfg, w_target_rate=w))
    a_prev = torch.zeros(1, NUM_J)
    a = a_prev + 1.0 / t.rate_k  # every joint's target moved by one kernel width
    S = t.ref.state(torch.zeros(1, dtype=torch.long))
    one_width = t.smooth_cost(S, a, a_prev)
    assert torch.allclose(one_width, torch.tensor([w]))  # SuperMimic's reward per step: -w (times the control step, as its terms)
    k = "finger_pos"
    for flag, factor in (([], None), (["--st_cost_mean"], 1.0)):
        _, algo, _ = agent_and_replay(kin, t, ST_TASK + flag)
        H, i = algo.horizon, algo.terms.index(k)
        d, acts = torch.zeros(H, 1), [torch.zeros(1, NUM_J) for _ in range(H)]
        algo.reg = 0.0
        zero = torch.zeros(len(algo.terms))
        base, _ = algo._st_objective(zero, d, acts)
        err = zero.clone()
        err[i] = float(algo.bal.scale[i])  # one width of the term's error at one imagined step (summed over the window)
        track = algo._st_objective(err, d, acts)[0] - base
        costs = [one_width] + [torch.zeros(1)] * (H - 1)
        cost = algo._st_objective(zero, d, acts, costs)[0] - base
        ratio = float(cost / track) / (w / t.cfg.w_finger)  # against the reward's ratio
        assert abs(ratio - (H if factor is None else factor)) < 1e-4


@pytest.mark.parametrize("extra", [[], ["--compile_rollout"]], ids=["steps", "compile_rollout"])
def test_st_no_imag_r(setup, monkeypatch, extra):
    """--st_no_imag_r (SuperTrack's; off by default, refused for SuperMimic): the imagined steps compute no reward, which
    SuperTrack's loss does not read. From the same state and random draws the updates return the same values but the
    imagined reward (NaN) and move the parameters, normalizers and optimizer states as without it (eagerly on the CPU, bit
    for bit), per imagined step or with the rollout as one function."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    assert not parse_args(small(ST_ARM)).st_no_imag_r and parse_args(small(ST_ARM) + ["--st_no_imag_r"]).st_no_imag_r
    with pytest.raises(SystemExit):
        parse_args(small([]) + ["--st_no_imag_r"])  # SuperMimic's agent computes its reward for its objective
    calls, reward = [], task.reward
    monkeypatch.setattr(task, "reward", lambda *a: calls.append(1) or reward(*a))
    runs = []
    for flags in ([], ["--st_no_imag_r"]):
        args, algo, buffer = agent_and_replay(kin, task, ST_ARM + SYSTEMS + ["--compile_updates"] + extra + flags)
        assert algo.imag_r != bool(flags)
        calls.clear()
        torch.manual_seed(1)
        out = torch.stack([algo.wm_update(buffer), *algo.policy_update(buffer), algo.wm_update(buffer), *algo.policy_update(buffer)])
        assert len(calls) == (0 if flags else 2 * args.horizon)  # the task's reward at every imagined step without the option
        state = [v for o in algo.optimizers for st in o.state.values() for v in st.values()]
        runs.append((out, trained(algo), state))
    (out0, t0, s0), (out1, t1, s1) = runs
    imag_r, same = [3, 9], [0, 1, 4, 5, 6, 7, 10, 11]  # (wm_loss, pi_loss, v_loss, imag_r, pi_gradnorm, imag_survival) twice; v_loss NaN
    assert torch.isfinite(out0[imag_r]).all() and torch.isnan(out1[imag_r]).all()
    assert torch.isnan(out0[[2, 8]]).all() and torch.isnan(out1[[2, 8]]).all()
    assert torch.equal(out0[same], out1[same]) and torch.isfinite(out0[same]).all()
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))
    assert len(s0) == len(s1) > 0 and all(torch.equal(a, b) for a, b in zip(s0, s1, strict=True))


def test_speed_options_keep_target_rate(setup, monkeypatch):
    """Each algorithm's own speed option on a task with the target-rate penalty (the timed configurations at the adopted setting):
    from the same state and random draws the policy update returns the same values and moves the parameters as without the option,
    bit for bit; and the penalty is in: against the unpenalized task SuperTrack's loss is higher by the imagined costs (--st_cost_mean:
    their mean over the window) and SuperMimic's imagined reward is lower."""
    import dataclasses

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    pen = HandTask(kin, task.ref, dataclasses.replace(task.cfg, w_target_rate=4.0))
    st, sm = ST_TASK + ["--st_cost_mean"] + SYSTEMS + ["--compile_updates"], SYSTEMS + ["--compile_updates"]
    for base, flag, same in ((st, "--st_no_imag_r", [0, 3, 4]), (sm, "--sm_reuse_input", [0, 1, 2, 3, 4])):
        runs, costs = {}, []
        for name, t, flags in (("plain", task, []), ("pen", pen, []), ("option", pen, [flag])):
            _, algo, buffer = agent_and_replay(kin, t, base + flags)
            if name == "pen" and flag == "--st_no_imag_r":  # the imagined steps' costs of the first update, as the loss takes them
                step = algo.st_step
                monkeypatch.setattr(algo, "st_step", lambda *a, _s=step: (lambda o: costs.append(o[7].detach()) or o)(_s(*a)))
            torch.manual_seed(1)
            first = torch.stack([*algo.policy_update(buffer)])
            n_first = len(costs)
            second = torch.stack([*algo.policy_update(buffer)])
            runs[name] = (first, second, trained(algo), n_first)
        (p0, _, _, _), (q0, q1, tq, n), (o0, o1, to, _) = runs["plain"], runs["pen"], runs["option"]
        assert torch.equal(q0[same], o0[same]) and torch.equal(q1[same], o1[same]) and torch.isfinite(q0[same]).all()
        assert all(torch.equal(a, b) for a, b in zip(tq, to, strict=True))
        if flag == "--st_no_imag_r":
            assert torch.isnan(o0[2]) and n == 4  # no imagined reward for the log; one cost per imagined step (horizon 4)
            mean_cost = torch.stack(costs[:n]).mean(0).mean()  # --st_cost_mean: the window's mean, then the batch's
            assert float(mean_cost) > 0 and torch.allclose(q0[0] - p0[0], mean_cost, rtol=1e-4, atol=1e-6)
        else:
            assert float(q0[2]) < float(p0[2]) and not torch.equal(q0[0], p0[0])  # the imagined reward lower with the penalty


class _Fixed(torch.nn.Module):
    """A policy that returns a given action (the worked tests below)."""

    def __init__(self, a: torch.Tensor):
        super().__init__()
        self.a = a

    def forward(self, obs, goal):
        return self.a


def test_target_rate_worked_by_hand(setup, monkeypatch):
    """--target_rate_w 4 and --st_cost_mean on one pair of consecutive targets worked out by hand, through the task's function and
    through each algorithm's imagined step. The pair: the second target is 12 mm further than the first on a palm slide and 0.2 rad
    further on a finger joint, beyond the reference's own step (the policy's offsets differ by 0.4 of 30 mm and by 0.5 of 0.4
    rad). Over the kernels' widths, 40 mm and 0.4 rad, that is 0.3 and 0.5; squared, 0.09 and 0.25; the mean over the 18 joints,
    0.34 / 18; times the weight 4: 0.0755556. Its gradient on the second offset is 4 x 2 x 0.75^2 x 0.4 / 18 = 0.1 on the slide and
    4 x 2 x 1^2 x 0.5 / 18 = 0.2222 on the finger, zero on the other joints, and the opposite on the first offset. SuperMimic: the
    imagined step's reward is lower by the cost times the control step, 0.02 s, and so is its gradient. SuperTrack: the imagined
    step returns the cost; with --st_cost_mean the loss is higher by the costs' mean over the horizon (one such step of four:
    0.0755556 / 4), without it by their sum. At the default weight 0 the cost is None and the loss is the same bit for bit."""
    import dataclasses

    from ndm_gym.robots.hands import NUM_PALM
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.hand_tracking import RewardCfg

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    assert (NUM_J, CONTROL_DT, task.cfg.w_target_rate) == (18, 0.02, 0.0)
    scales = (float(kin.action_scale[0]), float(kin.action_scale[NUM_PALM]), task.cfg.keypoint_std, task.cfg.finger_std)
    assert scales == pytest.approx((0.03, 0.4, 0.04, 0.4))  # the largest offsets (m, rad) and the kernels' widths
    pen = HandTask(kin, task.ref, dataclasses.replace(RewardCfg(), w_target_rate=4.0))
    f = torch.tensor([3])
    a_prev = torch.zeros(1, NUM_J, requires_grad=True)
    a = torch.zeros(1, NUM_J)
    a[0, 0], a[0, NUM_PALM] = 0.4, 0.5
    a.requires_grad_(True)
    worked, grad = 0.0755556, torch.zeros(1, NUM_J)  # 4 x (0.3^2 + 0.5^2) / 18
    grad[0, 0], grad[0, NUM_PALM] = 0.1, 0.2222222  # 4 x 2 x 0.75^2 x 0.4 / 18 and 4 x 2 x 1 x 0.5 / 18

    step = (pen.setpoint(a, f) - pen.setpoint(a_prev, f)).detach()  # the pair of targets: 12 mm and 0.2 rad apart
    assert abs(float(step[0, 0]) - 0.012) < 1e-7 and abs(float(step[0, NUM_PALM]) - 0.2) < 1e-6 and int((step != 0).sum()) == 2
    S = task.ref.state(f)
    cost = pen.smooth_cost(S, a, a_prev)
    ga, gp = torch.autograd.grad(cost.sum(), (a, a_prev))
    assert abs(float(cost.detach()) - worked) < 1e-7 and torch.allclose(ga, grad, atol=1e-7) and torch.allclose(gp, -grad, atol=1e-7)
    assert task.smooth_cost(S, a, a_prev) is None  # the default weight

    fk = kin.forward(S[0])
    _, sm, _ = agent_and_replay(kin, task, [])  # SuperMimic: the cost in the imagined step's reward
    sm.pi = _Fixed(a)
    rewards = {}
    for name, t in (("plain", task), ("penalized", pen)):
        sm.task = t
        out = sm._imag_step(S, fk, f, a_prev)
        assert out[3] is a
        rewards[name] = out[4]
    diff = rewards["penalized"] - rewards["plain"]
    ga, gp = torch.autograd.grad(diff.sum(), (a, a_prev))
    assert abs(float(diff.detach()) + worked * 0.02) < 2e-7
    assert torch.allclose(ga, -0.02 * grad, atol=1e-7) and torch.allclose(gp, 0.02 * grad, atol=1e-7)

    _, st, _ = agent_and_replay(kin, pen, ST_TASK + ["--st_cost_mean"])  # SuperTrack: the cost in the loss
    st.pi = _Fixed(a)
    H = st.horizon
    assert H == 4 and st.cost_mean
    out = st._st_step(S, fk, f, a_prev, torch.zeros(1, NUM_J))
    assert out[3] is a and abs(float(out[-1].detach()) - worked) < 1e-7
    terms, d = torch.zeros(len(st.terms)), torch.zeros(H, 1)
    acts, costs = [a] + [torch.zeros(1, NUM_J)] * (H - 1), [out[-1]] + [torch.zeros(1)] * (H - 1)
    base, _ = st._st_objective(terms, d, acts)
    more, _ = st._st_objective(terms, d, acts, costs)
    ga, gp = torch.autograd.grad(more - base, (a, a_prev), retain_graph=True)
    assert abs(float((more - base).detach()) - worked / 4) < 1e-7
    assert torch.allclose(ga, grad / 4, atol=1e-7) and torch.allclose(gp, -grad / 4, atol=1e-7)
    st.cost_mean = False  # without --st_cost_mean: the sum over the horizon
    summed, _ = st._st_objective(terms, d, acts, costs)
    assert abs(float((summed - base).detach()) - worked) < 1e-7
    st.task = task  # the default weight: no cost, the same loss bit for bit
    assert st._st_step(S, fk, f, a_prev, torch.zeros(1, NUM_J))[-1] is None
    assert torch.equal(st._st_objective(terms, d, acts, [None] * H)[0], base)


@GPU
@pytest.mark.parametrize("extra", [[], ["--compile_rollout"]], ids=["steps", "compile_rollout"])
def test_st_no_imag_r_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, capsys, extra):
    """--st_no_imag_r in train_hand's loop with every systems option on the small scene (compilation left out): the
    collection step and the updates are captured as CUDA graphs and replayed, the log's imag_r is NaN and its other
    columns finite."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: gpu_setup)
    run = tmp_path / "run"
    argv = [*small(ST_ARM), *SYSTEMS, "--collect_graph", "--compile_eager", "--update_graphs", "--pi_opt", "muon", "--wm_opt", "muon", "--pi_ema",
            "0.9", "--compile_updates", "--wm_palm_fk", "--st_no_imag_r", *extra, "--num_envs", "32", "--buffer_steps", "16", "--warmup_steps", "8",
            "--wm_only_iters", "2", "--log_every", "4", "--iterations", "12", "--log_dir", str(run)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv])
    train_hand.main()
    out = capsys.readouterr().out
    assert "captured one collection step" in out and "captured the world-model updates" in out and "captured the policy update" in out
    log = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
    cols = [train_hand.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "pi_gradnorm", "imag_survival", "sim_r", "ep_len")]
    assert log[:, 0].tolist() == [4, 8, 12] and np.isfinite(log[1:, cols]).all() and np.isnan(log[:, train_hand.LOG_COLUMNS.index("imag_r")]).all()


# ------------------------------------------------- --noise_free_frac (both algorithms' option, off by default)
def test_noise_free_values():
    """--noise_free_frac f, worked by hand: the multiplier of the noise's draws is zero in the first round(f N) worlds and one in the
    others; the draws are the default step's own (torch.randn_like of the previous action). Off by default; refused without the noise
    or with f outside [0, 1)."""
    args = parse_args(small(["--noise_free_frac", "0.25"]))
    nf = hand_fast.NoiseFree(args, 8, "cpu")
    assert torch.equal(nf.mult[:, 0], torch.tensor([0.0, 0.0] + [1.0] * 6))
    torch.manual_seed(1)
    z = torch.randn(8, NUM_J)
    torch.manual_seed(1)
    assert torch.equal(nf.noise(torch.zeros(8, NUM_J)), z * nf.mult) and torch.equal(nf.noise(torch.zeros(8, NUM_J))[:2], torch.zeros(2, NUM_J))
    for n, k in ((10, 3), (9, 3)):  # round(3.0) and round(2.7), not the truncated 2
        assert torch.equal(hand_fast.NoiseFree(parse_args(small(["--noise_free_frac", "0.3"])), n, "cpu").mult[:, 0],
                           torch.tensor([0.0] * k + [1.0] * (n - k)))  # fmt: skip
    for f in ("0.01", "0.99"):  # round(0.08) = 0 worlds and round(7.92) = all of them: refused, not silently off
        with pytest.raises(ValueError, match="at least 1, not all"):
            hand_fast.NoiseFree(parse_args(small(["--noise_free_frac", f])), 8, "cpu")
    default = parse_args(small([]))
    assert default.noise_free_frac == 0.0 and not hand_fast.noise_free_on(default)
    for bad in (["--noise_free_frac", "1.0"], ["--noise_free_frac", "-0.1"], ["--noise_free_frac", "0.25", "--action_noise", "0"]):
        with pytest.raises(SystemExit):
            parse_args(small(bad))


@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
@pytest.mark.parametrize("graphed,compiled", [(False, False), (True, False), (True, True), (False, True)],
                         ids=["collector", "graphed", "graphed_compiled", "compiled"])  # fmt: skip
def test_noise_free_collector_paths(setup, monkeypatch, extra, graphed, compiled):
    """With --noise_free_frac each collector path (hand_fast.Collector's step, its --collect_graph bookkeeping, its --compile_collect
    form) collects what the default step does: the same replay, log sums, previous actions and normalizers, bit for bit."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    n, steps, opts, runs = 8, 60, ["--noise_free_frac", "0.25"], []
    for collector in (False, True):
        monkeypatch.setattr(RunningNorm, "no_sync", collector)
        torch.manual_seed(0)
        flags = (["--collect_graph", "--compile_eager"] if graphed else []) + (["--compile_collect"] if compiled else [])
        args = parse_args(small(extra + opts + ((SYSTEMS + flags) if collector else [])))
        algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
        sim = KinematicSim(task, n)
        buffer = ReplayBuffer(32, n, "cpu", fields=replay_fields(task), priorities=False)
        if collector and graphed:
            buffer.use_device_ptr(device_size=True)
        nf = hand_fast.NoiseFree(args, n, "cpu")
        st = {"a_prev": torch.zeros(n, NUM_J), "ep_len": torch.zeros(n), "acc": {k: torch.zeros(()) for k in hand_fast.ACC_KEYS}}
        c = hand_fast.Collector(sim, task, algo, buffer, args, noise_free=nf) if collector else None
        for k in range(steps):
            torch.manual_seed(k)
            if c is None:
                default_collect(sim, task, algo, buffer, st, args.action_noise, noise_free=nf)
                continue
            c.core()
            if graphed:
                buffer.advance_host()
        if c is not None:
            st = {"a_prev": c.a_prev, "ep_len": c.ep_len, "acc": c.acc}
        runs.append((buffer, st, sim, trained(algo)))
    (b0, st0, sim0, t0), (b1, st1, sim1, t1) = runs
    for k in b0.data:
        assert torch.equal(b0.data[k], b1.data[k]), k
    for k in hand_fast.ACC_KEYS:
        assert torch.equal(st0["acc"][k], st1["acc"][k]), k
    assert torch.equal(st0["a_prev"], st1["a_prev"]) and torch.equal(sim0.s, sim1.s)
    assert all(torch.equal(a, b) for a, b in zip(t0, t1, strict=True))


def test_noise_free_worlds_act_as_the_policy(setup, monkeypatch):
    """--noise_free_frac 0.25 on 8 worlds: in a collection step the first two worlds' targets are the policy's own action's (its output
    on the step's inputs, clamped), the others' are not."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    torch.manual_seed(0)
    args = parse_args(small(["--noise_free_frac", "0.25"]))
    algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
    sim = KinematicSim(task, 8)
    buffer = ReplayBuffer(8, 8, "cpu", fields=replay_fields(task), priorities=False)
    nf = hand_fast.NoiseFree(args, 8, "cpu")
    st = {"a_prev": 0.1 * torch.ones(8, NUM_J), "ep_len": torch.zeros(8), "acc": {k: torch.zeros(()) for k in hand_fast.ACC_KEYS}}
    s, f, a_prev = sim.state(), sim.f.clone(), st["a_prev"].clone()
    default_collect(sim, task, algo, buffer, st, args.action_noise, noise_free=nf)
    S = unflatten_state(s)
    with torch.no_grad():  # the policy's output on the step's inputs (its normalizer as the step left it)
        own = algo.pi(task.obs(S, algo.fk(S[0]), a_prev), task.goal(S, f, algo.goal_frames)).clamp(-1, 1)
    u, want = buffer.data["u"][0], task.setpoint(own, f)
    assert torch.equal(u[:2], want[:2]) and not torch.allclose(u[2:], want[2:])


class RecordingSim(KinematicSim):
    """KinematicSim recording every position target it is given (train_hand's step, hand_fast.Collector's step_nosync and, with
    --compile_collect, its set_targets)."""

    def __init__(self, task, n: int, record: list):
        super().__init__(task, n)
        self.record = record

    def step(self, u, reset: bool = True):
        self.record.append(u.clone())
        return super().step(u, reset)

    def step_nosync(self, u, physics=None, fk=None):
        self.record.append(u.clone())
        return super().step_nosync(u, physics, fk)

    def set_targets(self, u):
        self.record.append(u.clone())
        super().set_targets(u)


@pytest.mark.parametrize("path", [[], SYSTEMS, [*SYSTEMS, "--compile_collect"]], ids=["collect", "collector", "collector_compiled"])
def test_noise_free_in_training(setup, monkeypatch, tmp_path, path):
    """train_hand on the CPU, a simulator without physics in HandSim's place, on each collection path: with --noise_free_frac the first
    step's targets of the other worlds are the default run's and the first two worlds' are not, and every step of the warm-up and of
    the iterations draws its noise through the multiplier, zero in those two worlds; the option reaches the checkpoint's arguments,
    and a resume without it is refused."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)  # main() sets it; restored after the test
    monkeypatch.setattr(train_hand, "DEVICE", "cpu")
    monkeypatch.setattr(hand_fast.Collector, "__call__", lambda self: self.core())  # the step itself: its stream is CUDA's
    record: list = []
    model, kin, task = setup
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0, smooth=None: (model, kin, task))
    hand_sim = types.ModuleType("ndm_gym.sim.rigid.hand_sim")  # HandSim's module, which needs the simulator and a GPU
    hand_sim.HandSim = lambda model, task, num_envs, device, stagger=False, **_: RecordingSim(task, num_envs, record)
    monkeypatch.setitem(sys.modules, "ndm_gym.sim.rigid.hand_sim", hand_sim)
    argv = [*small(path), "--num_envs", "8", "--buffer_steps", "16", "--warmup_steps", "4", "--wm_only_iters", "1", "--log_every", "2",
            "--ckpt_every", "2", "--iterations", "3"]  # fmt: skip
    targets, drawn, noise = {}, [], hand_fast.NoiseFree.noise
    monkeypatch.setattr(hand_fast.NoiseFree, "noise", lambda self, like: (drawn.append(noise(self, like)), drawn[-1])[1])
    for name, flag in (("plain", []), ("free", ["--noise_free_frac", "0.25"])):
        record.clear()
        monkeypatch.setattr(sys, "argv", ["train_hand", *argv, *flag, "--log_dir", str(tmp_path / name)])
        train_hand.main()
        targets[name] = [u.clone() for u in record]
    assert len(drawn) == 7 and all(not z[:2].any() and (z[2:] != 0).all() for z in drawn)
    assert len(targets["plain"]) == len(targets["free"]) == 7  # four warm-up steps and three iterations
    a, b = targets["plain"][0], targets["free"][0]  # the first step: the same policy, state and draws; later the normalizers differ
    assert torch.equal(a[2:], b[2:]) and not torch.equal(a[:2], b[:2])
    ck = torch.load(tmp_path / "free" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["noise_free_frac"] == 0.25
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv, "--log_dir", str(tmp_path / "free"), "--iterations", "4", "--resume",
                                      str(tmp_path / "free" / "last.pt")])  # fmt: skip
    with pytest.raises(ValueError, match="noise_free_frac"):
        train_hand.main()


@GPU
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
def test_collect_graph_with_noise_free_worlds(gpu_setup, monkeypatch, extra):
    """--noise_free_frac with --collect_graph on the GPU: the captured step, replayed, computes what the step run eagerly on Warp's stream
    does under the same draws, on each collector path, up to the rounding by which two simulations differ; and the option acts in the
    graph (the targets differ from a run without it)."""
    from ndm_gym.sim.rigid.hand_sim import HandSim

    monkeypatch.setattr(RunningNorm, "no_sync", True)
    model, kin, task = gpu_setup
    n, steps, runs = 64, 40, []
    for flags in ([], ["--collect_graph"], ["--collect_graph", "--compile_eager"], ["--compile_collect"], ["--collect_graph", "--compile_collect"],
                  ["--collect_graph", "NONE"]):  # fmt: skip
        plain = "NONE" in flags
        flags = [x for x in flags if x != "NONE"]
        torch.manual_seed(0)
        args = parse_args(small(extra + SYSTEMS + flags + ([] if plain else ["--noise_free_frac", "0.25"])))
        algo = HAND_ALGOS[args.algo](args, kin, task, CUDA)
        sim = HandSim(model, task, n, CUDA, stagger=True)
        buffer = ReplayBuffer(32, n, CUDA, fields=replay_fields(task), priorities=False)
        if "--collect_graph" in flags:
            buffer.use_device_ptr(device_size=True)
        c = hand_fast.Collector(sim, task, algo, buffer, args, noise_free=None if plain else hand_fast.NoiseFree(args, n, CUDA))
        for k in range(steps):
            torch.manual_seed(100 + k)
            c()
        torch.cuda.synchronize()
        runs.append((buffer, c, sim, trained(algo)))
    b0, c0, sim0, t0 = runs[0]
    for b1, c1, sim1, t1 in runs[1:-1]:
        for k in ("f", "done"):
            assert torch.equal(b0.data[k], b1.data[k]), k
        for k in ("s", "s1", "u", "a_prev"):
            assert torch.allclose(b0.data[k], b1.data[k], atol=1e-3), (k, float((b0.data[k] - b1.data[k]).abs().max()))
        assert torch.equal(c0.ep_len, c1.ep_len) and torch.allclose(c0.a_prev, c1.a_prev, atol=1e-3)
        assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-4) for a, b in zip(t0, t1, strict=True))
    assert not torch.allclose(runs[-1][0].data["u"], b0.data["u"], atol=1e-3)  # without the option: other targets


# ------------------------------------------------- --bptt_decay (both algorithms' option; SuperTrack's is SuperMimic's operation)
RATE = ["--target_rate_w", "4.0"]
DECAY_ARMS = {
    "supermimic": [], "supermimic_reuse_input": ["--sm_reuse_input"], "supermimic_costs": RATE,
    "supertrack": ["--algo", "supertrack"], "supertrack_vel_fail": ["--algo", "supertrack", "--st_fail_w", "2", "--st_vel_terms"],
    "supertrack_weights_noise": ["--algo", "supertrack", "--st_task_weights", "--st_imag_noise", "0.2", "--st_fail_w", "2"],
    "supertrack_costs": ["--algo", "supertrack", "--st_task_weights", "--st_cost_mean", *RATE],
}  # fmt: skip


def decay_task(kin, task, extra):
    """The task with the target-rate cost of ``extra``'s options (as train_hand makes it)."""
    import dataclasses

    from ndm_gym.train_hand import smoothing

    return HandTask(kin, task.ref, dataclasses.replace(task.cfg, **smoothing(parse_args(small(extra)))))


@pytest.mark.parametrize("arm,lam,exact,rollout", [(arm, lam, exact, rollout) for arm in DECAY_ARMS for lam, exact, rollout in
                                                   ((0.5, True, False), (0.95, False, False), (0.5, True, True))])  # fmt: skip
def test_bptt_decay(setup, monkeypatch, arm, lam, exact, rollout):
    """--bptt_decay, the same operation for both algorithms (the target-rate cost on too): 1 (the default) is the
    unchanged update, bit for bit, the rollout as one function too; lam < 1 leaves the imagined forward pass but for a float32 rounding
    (``lam x + (1 - lam) x.detach()`` is the identity only at some lam: exactly at 0.5), and the whole policy update is then the one whose
    world model's step is wrapped, in the policy update alone, in algos.common.decay_rollout_grads (its own tests): bit for bit at 0.5,
    within rounding at 0.95."""
    from ndm_gym.algos.common import decay_rollout_grads

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)
    _, kin, task = setup
    task = decay_task(kin, task, DECAY_ARMS[arm])
    extra = DECAY_ARMS[arm] + SYSTEMS + (["--compile_updates", "--compile_rollout"] if rollout else [])

    def update(flags, before_policy=None):
        args, algo, buffer = agent_and_replay(kin, task, extra + flags)
        torch.manual_seed(2)
        for _ in range(3):  # an untrained world model's last layer is zero: its step would not depend on the actions
            algo.wm_update(buffer)
        if before_policy is not None:
            before_policy(algo)
        torch.manual_seed(3)
        out = algo.policy_update(buffer)
        return [torch.nan_to_num(x) for x in out], [p.detach().clone() for p in algo.pi.parameters()], algo

    plain, one, half = update([]), update(["--bptt_decay", "1"]), update(["--bptt_decay", str(lam)])
    for a, b in zip(plain[0] + plain[1], one[0] + one[1], strict=True):
        assert torch.equal(a, b)  # the default is bit for bit the update with the option at 1

    def wrap(algo):  # the world model's step in the policy update wrapped in the generic operation (not in its own fit)
        step = algo.wm.step
        monkeypatch.setattr(algo.wm, "step", lambda *a: decay_rollout_grads(step(*a), lam), raising=False)

    same = update([], wrap)
    pairs = list(zip(half[0] + half[1], same[0] + same[1], strict=True))
    if exact:
        assert all(torch.equal(a, b) for a, b in pairs)  # the class's own decay is the generic operation, bit for bit
        assert torch.equal(half[0][0], plain[0][0])  # the loss: the same forward pass, so the policy's change is the decay's alone
        assert any(not torch.equal(a, b) for a, b in zip(half[1], plain[1], strict=True))  # the decay acts
    else:
        assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-5) for a, b in pairs)
        assert torch.allclose(half[0][0], plain[0][0], rtol=1e-5)  # the forward pass but for the rounding of 1 - lam
        if "supertrack" in arm:  # the policy gradient's norm (SuperMimic's moves too little to tell it from the rounding)
            assert abs(float(half[0][3]) / float(plain[0][3]) - 1) > 1e-4


@pytest.mark.parametrize("algo_name", ["supermimic", "supertrack"])
def test_bptt_decay_arguments(setup, algo_name):
    """A run's arguments from before the option build the agent with it off; outside (0, 1] the agent refuses it."""
    _, kin, task = setup
    extra = ["--algo", algo_name] + SYSTEMS
    args, _, _ = agent_and_replay(kin, task, extra)
    args_before = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "bptt_decay"})
    torch.manual_seed(0)
    assert HAND_ALGOS[algo_name](args_before, kin, task, "cpu").bptt_decay == 1.0
    for bad in ("0", "-0.5", "1.5"):
        with pytest.raises(ValueError, match="bptt_decay"):
            agent_and_replay(kin, task, extra + ["--bptt_decay", bad])


@pytest.mark.parametrize("algo_name", ["supermimic", "supertrack"])
def test_bptt_decay_by_hand(setup, monkeypatch, algo_name):
    """The gradient's weights by hand on a chain of known maps (a constant policy; the joints, the velocities and the object's position
    times 0.9 at every step, the object's rotation kept): what the d-th imagined step reads (SuperTrack's errors in every term, SuperMimic's
    reward) reaches the start state with lam**d through every path (weights 1 without the option), and the same gradient comes from the
    step's values weighted lam**d without it. The decay multiplies the state's gradient, not the actions'."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    horizon, lam, B = 4, 0.6, 5

    def contract(S, fk, u):  # the world model's step stand-in
        q, qd, p, R, v, w = S
        return 0.9 * q, 0.9 * qd, 0.9 * p, R, 0.9 * v, 0.9 * w

    torch.manual_seed(1)
    f0 = torch.randint(0, task.ref.frames - horizon - 1, (B,))
    start = [x.clone() for x in task.ref.state(f0)]
    for k, s in ((0, 0.1), (1, 0.3), (2, 0.05), (4, 0.3), (5, 0.3)):
        start[k] = start[k] + s * torch.randn_like(start[k])
    paths = (0, 1, 2, 4, 5)  # the joints, their velocities, the object's position, its velocities: not its rotation

    def chain(extra, weights):
        torch.manual_seed(0)
        arm = ["--algo", "supertrack", "--st_vel_terms", "--st_no_imag_r"] if algo_name == "supertrack" else []
        algo = HAND_ALGOS[algo_name](parse_args(small(arm + extra)), kin, task, "cpu")
        algo.pi = lambda obs, goal: torch.zeros(B, kin.num_joints)
        monkeypatch.setattr(algo.wm, "step", contract, raising=False)
        x = [s.clone().requires_grad_(k in paths) for k, s in enumerate(start)]
        S, fk, f, a_prev, parts = tuple(x), kin.forward(x[0]), f0, torch.zeros(B, kin.num_joints), []
        for d in range(1, horizon + 1):
            if algo_name == "supertrack":
                S, fk, f, a_prev, e, _, _, _ = algo._st_step(S, fk, f, a_prev, torch.zeros(B, kin.num_joints))
            else:
                S, fk, f, a_prev, e, _, _ = algo._imag_step(S, fk, f, a_prev)
            parts.append((weights(d) * e).sum())
        return parts, [x[k] for k in paths]

    parts, x = chain(["--bptt_decay", str(lam)], lambda d: 1.0)
    total = torch.autograd.grad(sum(parts), x, allow_unused=True)
    plain, x1 = chain([], lambda d: 1.0)
    ref = [torch.zeros_like(y) for y in x1]
    for d, e in enumerate(plain, 1):
        ref = [r + lam**d * (g if g is not None else 0) for r, g in zip(ref, torch.autograd.grad(e, x1, retain_graph=True, allow_unused=True), strict=True)]
    weighted, x2 = chain([], lambda d: lam**d)
    again = torch.autograd.grad(sum(weighted), x2, allow_unused=True)
    total, again = ([torch.zeros_like(y) if g is None else g for g, y in zip(gs, x1, strict=True)] for gs in (total, again))
    for name, g, r, w in zip(("joints", "joint velocities", "object position", "object velocity", "object angular velocity"), total, ref, again, strict=True):
        assert torch.allclose(g, r, atol=1e-5, rtol=1e-5) and torch.allclose(g, w, atol=1e-5, rtol=1e-5), name
    assert float(total[0].abs().max()) > 1e-4 and float(total[2].abs().max()) > 1e-4  # the joints' and the object's paths carry the gradient
    if algo_name == "supertrack":  # the velocities' too, in the velocity terms
        assert all(float(g.abs().max()) > 1e-4 for g in total)
