"""Two hands in one state against numbers worked out from MuJoCo's own frames: a right hand of five fingers and a left hand of
four, each at its own place and the left one mirrored, the right hand's actuators first or the left hand's. The hands' columns,
the observation, the errors, the reward, the terminations, the world model's inputs and SuperTrack's errors."""

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.hand_algos import HAND_ALGOS
from ndm_gym.core.rotations import mat_to_6d
from ndm_gym.robots.hands import HandKinematics, scene_indices
from ndm_gym.tasks.hand_tracking import HandTask, RewardCfg, build_reference
from ndm_gym.train_hand import parse_args
from ndm_gym.worldmodels.hand import KINDS, HandWorldModel

FINGERS = {"right": {"thumb": 3, "index": 3, "middle": 2, "ring": 2, "pinky": 2}, "left": {"thumb": 4, "index": 4, "middle": 4, "ring": 4}}
JOINTS = {side: 6 + sum(fingers.values()) for side, fingers in FINGERS.items()}
PLACE = {"right": 'pos="0.02 -0.11 0.2" quat="0.9 0.3 -0.2 0.1"', "left": 'pos="-0.04 0.13 0.25" quat="0.7 -0.1 0.4 0.5"'}


def scene(order: tuple) -> mujoco.MjModel:
    """The two hands and a free box; the actuators of ``order``'s first hand, then the other's."""
    hands, actuators = "", {}
    for side, s in (("right", 1), ("left", -1)):
        axes = ("1 0 0", "0 1 0", "0 0 1", "0 0 1", f"{s} 0 0", f"0 {-s} 0")
        names = [f"{side}_palm_{i}" for i in range(6)]
        body = f"<body {PLACE[side]}>" + "".join(f'<joint name="{n}" type="{"slide" if i < 3 else "hinge"}" axis="{axes[i]}" range="-3 3"/>'
                                                  for i, n in enumerate(names))  # fmt: skip
        body += f'<geom size="0.01" mass="0.1"/><site name="{side}_palm" pos="0.01 {0.02 * s} 0.005"/>'
        for k, (finger, count) in enumerate(FINGERS[side].items()):
            for i in range(count):
                names.append(f"{side}_{finger}_{i}")
                body += (f'<body pos="0.01 {s * (0.02 * k - 0.04)} -0.03" quat="0.96 {0.1 * s} 0.2 0.1"><joint name="{names[-1]}" axis="1 {0.2 * s} 0.1" '
                         f'pos="0 0.005 0.002" range="0 1.5"/><geom size="0.008" mass="0.02"/>')  # fmt: skip
            body += f'<site name="{side}_{finger}_tip" pos="0.001 {0.004 * s} -0.027"/>' + "</body>" * count
        hands += body + "</body>"
        actuators[side] = "".join(f'<position joint="{n}" kp="50"/>' for n in names)
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{hands}<body name="right_object" pos="0.1 0 0.05"><freejoint/>'
        f'<geom type="box" size="0.02 0.03 0.02" mass="0.1"/></body></worldbody><actuator>{"".join(actuators[s] for s in order)}</actuator></mujoco>'
    )


def frames(model: mujoco.MjModel, order: tuple, q: np.ndarray) -> list:
    """MuJoCo's palm position, palm rotation and fingertip positions of each hand of ``order`` at the joints ``q``."""
    data, hands = mujoco.MjData(model), []
    data.qpos[scene_indices(model)["qpos"]] = q
    mujoco.mj_kinematics(model, data)
    for side in order:
        palm = model.site(f"{side}_palm").id
        tips = [model.site(f"{side}_{finger}_tip").id for finger in FINGERS[side]]
        hands.append((data.site_xpos[palm].copy(), data.site_xmat[palm].reshape(3, 3).copy(), data.site_xpos[tips].copy()))
    return hands


def angle(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


@pytest.fixture(params=[("right", "left"), ("left", "right")], ids=["right_first", "left_first"])
def two(request):
    torch.manual_seed(0)
    model, n = scene(request.param), 9
    kin = HandKinematics(model, "cpu")
    quat = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.3 * torch.rand(n, 4), dim=-1)
    clip = {"q": 0.4 * torch.rand(n, kin.num_joints).cumsum(0) / n, "p": torch.tensor([0.1, 0, 0.05]) + 0.01 * torch.rand(n, 3).cumsum(0), "quat": quat}
    return request.param, model, kin, build_reference(clip, kin, "cpu")


def test_the_columns_of_the_two_hands(two):
    order, _, kin, _ = two
    first, second = (JOINTS[side] for side in order)
    tips = [len(FINGERS[side]) for side in order]
    assert kin.num_joints == first + second and kin.palm_keypoints.tolist() == [0, 1 + tips[0]]
    assert kin.fingertip_keypoints.tolist() == [*range(1, 1 + tips[0]), *range(2 + tips[0], 2 + sum(tips))]
    assert kin.palm_translations == [0, 1, 2, first, first + 1, first + 2]
    assert kin.finger_joints.tolist() == [*range(6, first), *range(first + 6, first + second)]
    scale = [s for count in (first, second) for s in [0.03] * 3 + [0.3] * 3 + [0.4] * (count - 6)]
    assert torch.allclose(kin.action_scale, torch.tensor(scale))
    width = [w for count in (first, second) for w in [0.04] * 3 + [0.3] * 3 + [0.5] * (count - 6)]
    rate_k = HandTask(kin, two[3], RewardCfg(palm_ori_std=0.3, finger_std=0.5)).rate_k
    assert torch.allclose(rate_k, torch.tensor(scale) / torch.tensor(width))


def test_the_task_against_mujocos_frames(two):
    """Every quantity of the task at states off the reference, per hand in its own palm's frame."""
    order, model, kin, ref = two
    n, J, cfg = 6, kin.num_joints, RewardCfg()
    task = HandTask(kin, ref, cfg)
    f = torch.randint(0, ref.frames - 1, (n,))
    S = [x + 0.03 * torch.randn(x.shape) for x in ref.state(f)]
    R = torch.linalg.qr(S[3])[0]
    S[3] = R * torch.sign(torch.linalg.det(R))[:, None, None]
    fk, a, a_prev = kin.forward(S[0]), 2 * torch.rand(n, J) - 1, 2 * torch.rand(n, J) - 1
    u = task.setpoint(a, f)
    sixd = lambda M: mat_to_6d(torch.as_tensor(M)).numpy()  # noqa: E731
    keep = [c for c in range(J) if c not in kin.palm_translations]
    want = {k: [] for k in ("obs", "inputs", "keypoint", "finger", "palm_ori_sq", "reward", "soft", "supertrack")}
    for i in range(n):
        q, qd, p, R, v, w = (x[i].double().numpy() for x in S)
        rq, rp, rR = (x[f[i]].double().numpy() for x in (ref.q, ref.p, ref.R))
        now, goal = frames(model, order, q), frames(model, order, rq)
        obs, about, points, local, palms, fingers, start = [q, 0.1 * qd], [], [], [], [], [], 0
        for (x, Rp, tips), (gx, gR, gtips), side in zip(now, goal, order, strict=True):
            about += [Rp.T @ (p - x), sixd(Rp.T @ R)]
            obs += [((tips - x) @ Rp).reshape(-1), sixd(Rp), Rp.T @ (p - x), sixd(Rp.T @ R), Rp.T @ v, 0.1 * Rp.T @ w]
            points.append(np.concatenate((x[None], tips)) - np.concatenate((gx[None], gtips)))
            local.append(np.abs((tips - x) @ Rp - (gtips - gx) @ gR).sum(-1))
            palms.append(angle(gR, Rp))
            fingers.append((q - rq)[start + 6 : start + JOINTS[side]])
            start += JOINTS[side]
        keypoint, fingers, palms = np.linalg.norm(np.concatenate(points), axis=-1), np.concatenate(fingers), np.array(palms)
        obj, ori = np.linalg.norm(p - rp), angle(rR, R)
        alive = np.prod([1 / (1 + np.exp((x - 0.15) / 0.015)) for x in (obj, *(np.linalg.norm(d[0]) for d in points))])
        want["obs"].append(np.concatenate((*obs, a_prev[i].numpy())))
        want["inputs"].append(np.concatenate((q[keep], qd, *about, v, w, p[2:], sixd(R), u[i].numpy() - q)))
        want["keypoint"].append(keypoint), want["finger"].append(np.sqrt((fingers**2).mean())), want["palm_ori_sq"].append((palms**2).mean())
        want["reward"].append(0.02 * (np.exp(-(keypoint**2).mean() / 0.04**2) + 0.5 * np.exp(-(fingers**2).mean() / 0.4**2)
                                      + 0.5 * np.exp(-(palms**2).mean() / 0.4**2) + 2 * np.exp(-(obj**2) / 0.04**2) + np.exp(-(ori**2) / 0.5**2)
                                      - 0.05 * ((a[i] - a_prev[i]).numpy() ** 2).mean()))  # fmt: skip
        want["soft"].append(1 - alive)
        want["supertrack"].append([np.abs(np.concatenate(points)).sum(-1).mean(), np.concatenate(local).mean(), palms.mean(), np.abs(fingers).mean(),
                                   np.abs(p - rp).sum(), ori])  # fmt: skip
    e = task.errors(S, fk, f)
    got = {"obs": task.obs(S, fk, a_prev), **{k: e[k] for k in ("keypoint", "finger", "palm_ori_sq")}, "reward": task.reward(S, fk, f, a, a_prev),
           "soft": task.soft_termination(S, fk, f)}  # fmt: skip
    agent = HAND_ALGOS["supertrack"](parse_args("--clip clip --algo supertrack --st_task_weights --wm_hidden 8".split()), kin, task, "cpu")
    got["supertrack"] = agent._errors(S, fk, f)
    for kind in KINDS:
        wm = HandWorldModel(J, kin.palm_translations, kin.palm_keypoints.tolist(), hidden=(8,), kind=kind)
        assert wm.net[0].in_features == len(want["inputs"][0])
        got["inputs"] = wm.inputs(S, fk, u)  # on the forward kinematics, then on the palms' alone (--wm_palm_fk)
        np.testing.assert_allclose(wm.inputs(S, kin.palm(S[0]), u).numpy(), want["inputs"], atol=1e-5)
        np.testing.assert_allclose(got["inputs"].numpy(), want["inputs"], atol=1e-5)
    for k, x in want.items():
        np.testing.assert_allclose(got[k].detach().numpy(), np.array(x), atol=1e-4, err_msg=k)
    assert task.obs_dim == len(want["obs"][0])


def test_either_palm_ends_the_episode(two):
    order, _, kin, ref = two
    task, f, start = HandTask(kin, ref), torch.tensor([3]), 0
    for side in order:
        S = [x.clone() for x in ref.state(f)]
        S[0][:, start + 6 : start + JOINTS[side]] += 0.5  # its fingers off their reference: no end
        assert not task.termination(S, kin.forward(S[0]), f) and float(task.soft_termination(S, kin.forward(S[0]), f)) < 0.01
        S[0][:, start] += 0.2  # its palm too
        assert task.termination(S, kin.forward(S[0]), f) and float(task.soft_termination(S, kin.forward(S[0]), f)) > 0.9
        start += JOINTS[side]
