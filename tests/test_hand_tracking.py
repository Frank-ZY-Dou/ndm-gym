"""Hand-object tracking on a small scene with the XHand's joints and keypoints: the forward kinematics against
MuJoCo's, the state layout, the task at the reference, the world model's increments, one SuperMimic update and
training's arguments; the systems options (algos.hand_fast) against the default loop, on the CPU and, where they need
the simulator and CUDA graphs, on a GPU."""

import argparse
import importlib.util
import sys

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
from ndm_gym.robots.xhand import FINGER_JOINTS, KEYPOINTS, NUM_J, PALM_JOINTS, HandKinematics, scene_indices
from ndm_gym.train import add_train_args
from ndm_gym.train_hand import HAND_DEFAULTS, UNSUPPORTED, parse_args
from ndm_gym.tasks.hand_tracking import HOLD, OBS_DIM, STATE_DIM, HandTask, build_reference, flatten_state, unflatten_state
from ndm_gym.worldmodels.hand import HandWorldModel

# the arguments hand training implements; the rest of ndm_gym.train's are train_hand.UNSUPPORTED
SUPPORTED = ("algo", "num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches",
             "wm_batch", "pi_batch", "lr", "gamma", "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
             "wm_only_iters", "lr_decay_iters", "ckpt_every", "max_hours", "log_every", "log_dir", "seed", "resume", "amp", "bptt_decay",
             "st_reg", "st_imag_noise", "st_fail_w")  # fmt: skip


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
    for refused in (["--algo", "blinkmimic"], ["--wm_backbone", "rigidformer"], ["--hist", "4"], ["--utd", "1"], ["--st_fail_w", "2"],
                    ["--st_vel_terms"], ["--st_obj_w", "2"], ["--st_unit_scales"], ["--st_task_weights"], ["--st_imag_noise", "0.2"],
                    ["--algo", "supertrack", "--st_task_weights", "--st_vel_terms"], ["--algo", "supertrack", "--st_task_weights", "--st_unit_scales"],
                    ["--algo", "supertrack", "--gamma", "0.995"], ["--algo", "supertrack", "--lam", "0.9"], ["--algo", "supertrack", "--critic_epochs", "3"],
                    ["--algo", "supertrack", "--bptt_decay", "0.95"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--clip", "clip"] + refused)
    st = parse_args(["--clip", "clip", "--algo", "supertrack", "--st_fail_w", "2", "--st_vel_terms", "--st_obj_w", "2", "--st_imag_noise", "0.2"])
    assert (st.algo, st.st_fail_w, st.st_vel_terms, st.st_obj_w, st.st_imag_noise) == ("supertrack", 2, True, 2, 0.2)
    assert not st.st_unit_scales and parse_args(["--clip", "clip", "--algo", "supertrack", "--st_unit_scales"]).st_unit_scales
    assert parse_args(["--clip", "clip", "--bptt_decay", "0.95"]).bptt_decay == 0.95  # SuperMimic's


def test_accel_world_model_round_trip(setup):
    """The acceleration model: a network that outputs the normalized accelerations reproduces the next velocities and, by
    semi-implicit Euler, the positions they lead to; its error is in units of the increments' spread, as the other's."""
    _, kin, task = setup
    f = torch.arange(task.ref.frames - 1)
    S, S1 = task.ref.state(f), task.ref.state(f + 1)
    wm = HandWorldModel(hidden=(32,), kind="accel")
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
        HandWorldModel(hidden=(32,), kind="other")


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


def default_collect(sim, task, algo, buffer, st: dict, noise: float):
    """train_hand's default collection step (its closure ``collect``) on the loop's state ``st``."""
    s, f = sim.state(), sim.f.clone()
    S = unflatten_state(s)
    a = algo.act(S, st["a_prev"], f, noise=noise)
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
@pytest.mark.parametrize("graphed", [False, True], ids=["eager", "device_head"])
def test_collector_matches_default_step(setup, monkeypatch, extra, graphed):
    """hand_fast.Collector's step on a simulator without physics computes the default loop's: the same replay, log sums,
    episode lengths, previous actions and normalizers, bit for bit; with the replay's device-side write head
    (--collect_graph's) and the forward kinematics through the agent's ``fk`` (--compile_eager's, not compiled here) too."""
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    _, kin, task = setup
    n, steps = 8, 90
    runs = []
    for collector in (False, True):
        monkeypatch.setattr(RunningNorm, "no_sync", collector)
        torch.manual_seed(0)
        args = parse_args(small(extra + ((SYSTEMS + (["--collect_graph", "--compile_eager"] if graphed else [])) if collector else [])))
        algo = HAND_ALGOS[args.algo](args, kin, task, "cpu")
        sim = KinematicSim(task, n)
        buffer = ReplayBuffer(32, n, "cpu", fields=replay_fields(), priorities=False)
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
            assert (algo.fk is c.fk) == graphed
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
    host_head = ReplayBuffer(32, n, "cpu", fields=replay_fields(), priorities=False)
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
    kinematics too."""
    from ndm_gym.sim.rigid.hand_sim import HandSim

    monkeypatch.setattr(RunningNorm, "no_sync", True)
    model, kin, task = gpu_setup
    n, steps, runs = 64, 40, []
    for flags in ([], ["--collect_graph"], ["--collect_graph", "--compile_eager"]):
        graphed = bool(flags)
        torch.manual_seed(0)
        args = parse_args(small(extra + SYSTEMS + flags))
        algo = HAND_ALGOS[args.algo](args, kin, task, CUDA)
        sim = HandSim(model, task, n, CUDA, stagger=True)
        buffer = ReplayBuffer(32, n, CUDA, fields=replay_fields(), priorities=False)
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
        assert (int(b1.ptr_t), int(b1.size_t)) == (b1.ptr, b1.size)
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
@pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])
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
    assert torch.equal(finite, torch.isfinite(out1)) and int(finite.sum()) == 7 * (6 if not extra else 5)
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
