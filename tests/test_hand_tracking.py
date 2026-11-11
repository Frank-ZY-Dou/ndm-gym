"""Hand-object tracking on a small scene with the XHand's joints and keypoints: the forward kinematics against
MuJoCo's, the state layout, the task at the reference, the world model's increments, one SuperMimic update and
training's arguments."""

import argparse

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.hand_algos import HAND_ALGOS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.robots.xhand import FINGER_JOINTS, KEYPOINTS, NUM_J, PALM_JOINTS, HandKinematics, scene_indices
from ndm_gym.train import add_train_args
from ndm_gym.train_hand import HAND_DEFAULTS, UNSUPPORTED, parse_args
from ndm_gym.tasks.hand_tracking import HOLD, OBS_DIM, STATE_DIM, HandTask, build_reference, flatten_state, unflatten_state
from ndm_gym.worldmodels.hand import HandWorldModel

# the arguments hand training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp")  # fmt: skip


def scene_xml() -> str:
    """The palm's slides and hinges in a chain, each finger a chain of its hinges off the palm, and a free box."""
    axes = ["1 0 0", "0 1 0", "0 0 1", "0 0 1", "1 0 0", "0 -1 0"]
    palm = "".join(f'<body name="p{i}"><joint name="{n}" type="{"slide" if i < 3 else "hinge"}" axis="{a}" range="-3 3"/>'
                   f'<geom size="0.01" mass="0.1"/>' for i, (n, a) in enumerate(zip(PALM_JOINTS, axes, strict=True)))  # fmt: skip
    fingers = ""
    for k, name in enumerate(("thumb", "index", "mid", "ring", "pinky")):
        joints = [j for j in FINGER_JOINTS if f"_{name}_" in j]
        body = "".join(f'<body pos="0.01 {0.02 * k - 0.04} -0.03" quat="0.96 0.1 0.2 0.1"><joint name="{j}" axis="1 0.2 0" pos="0 0.005 0" range="0 1.5"/>'
                       f'<geom size="0.008" mass="0.02"/>' for j in joints)  # fmt: skip
        fingers += body + f'<site name="{KEYPOINTS[k + 1]}" pos="0 0.004 -0.027"/>' + "</body>" * len(joints)
    actuators = "".join(f'<position joint="{j}" kp="50"/>' for j in PALM_JOINTS + FINGER_JOINTS)
    return (f'<mujoco><compiler angle="radian"/><worldbody><geom type="plane" size="1 1 0.1"/>{palm}<site name="right_palm"/>{fingers}'
            f'{"</body>" * 6}<body name="right_object" pos="0.1 0 0.05"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>'
            f"</worldbody><actuator>{actuators}</actuator></mujoco>")  # fmt: skip


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_string(scene_xml())
    kin = HandKinematics(model, "cpu")
    n = 12
    quat = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.3 * torch.rand(n, 4), dim=-1)
    clip = {"q": 0.5 * torch.rand(n, NUM_J).cumsum(0) / n, "p": torch.tensor([0.1, 0, 0.05]) + 0.01 * torch.rand(n, 3).cumsum(0), "quat": quat}
    return model, kin, HandTask(kin, build_reference(clip, kin, "cpu"))


def test_kinematics_match_mujoco(setup):
    model, kin, _ = setup
    idx, data = scene_indices(model), mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in KEYPOINTS]
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(4, NUM_J)
    kp, palm_R = kin.forward(q)
    for i in range(4):
        data.qpos[idx["qpos"]] = q[i].numpy()
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(kp[i].numpy(), data.site_xpos[sites], atol=1e-5)
        np.testing.assert_allclose(palm_R[i].numpy(), data.site_xmat[sites[0]].reshape(3, 3), atol=1e-5)


def test_reference_is_optimal_and_held(setup):
    _, kin, task = setup
    ref = task.ref
    assert ref.frames == 12 + HOLD and torch.equal(ref.q[-1], ref.q[-HOLD - 1]) and float(ref.qd[-1].abs().max()) == 0
    f = torch.arange(ref.frames)
    S = ref.state(f)
    fk, a = kin.forward(S[0]), torch.zeros(ref.frames, NUM_J)
    assert task.obs(S, fk, a).shape == (ref.frames, OBS_DIM)
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
    assert x.shape == (len(f), STATE_DIM)
    for a, b in zip(unflatten_state(x), S, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    wm = HandWorldModel(hidden=(32,))
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
    for refused in (["--algo", "supertrack"], ["--wm_backbone", "rigidformer"], ["--hist", "4"], ["--utd", "1"]):
        with pytest.raises(SystemExit):
            parse_args(["--clip", "clip"] + refused)
