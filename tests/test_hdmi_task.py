"""HDMI's task heads (ndm_gym.tasks.hdmi) on synthetic tensors, without a simulator: maxima at the reference, the
kernels' values for known errors, the contact reward, consecutive-step terminations, the feet's contact timers, the
reference and observation timing, the observation layout, gradients, and inputs left untouched."""

import dataclasses
import math

import pytest
import torch

from ndm_gym.core.rotations import axis_angle_to_mat, so3_exp
from ndm_gym.tasks import hdmi, hdmi_tasks
from ndm_gym.tasks.hdmi import DECIMATION, KEYPOINT_NAMES, FeetContact, HDMIClip, HDMIInput, HDMIRef, HDMIState, HDMITask

N, F, K, J, A = 4, 5, 16, 29, 23
DT = 0.02
MASS = 33.341
# hard joint limits of HDMI's G1 in its simulator order (spec §3.3)
LIMITS = torch.tensor([
    (-2.5307, 2.8798), (-2.5307, 2.8798), (-2.618, 2.618), (-0.5236, 2.9671), (-2.9671, 0.5236), (-0.52, 0.52), (-2.7576, 2.7576),
    (-2.7576, 2.7576), (-0.52, 0.52), (-0.0873, 2.8798), (-0.0873, 2.8798), (-3.0892, 2.6704), (-3.0892, 2.6704), (-0.8727, 0.5236),
    (-0.8727, 0.5236), (-1.5882, 2.2515), (-2.2515, 1.5882), (-0.2618, 0.2618), (-0.2618, 0.2618), (-2.618, 2.618), (-2.618, 2.618),
    (-1.0472, 2.0944), (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.9722, 1.9722), (-1.6144, 1.6144), (-1.6144, 1.6144), (-1.6144, 1.6144),
    (-1.6144, 1.6144),
], dtype=torch.float64)  # fmt: skip
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
UPPER = [KP[f"{s}_{b}_link"] for s in ("left", "right") for b in ("shoulder_pitch", "elbow", "wrist_yaw")]
LOWER = [KP[f"{s}_{b}_link"] for s in ("left", "right") for b in ("hip_pitch", "knee", "ankle_roll")]
VEL = [i for i, n in enumerate(KEYPOINT_NAMES) if "hip_pitch" not in n]  # tracking_body_linvel / angvel's 14 bodies
JOINT_REWARD = [i for i, n in enumerate(hdmi.JOINT_NAMES[:A]) if "ankle" not in n]  # joint_pos/vel_tracking's 19 joints
SOFT_LO = (LIMITS[:, 0] + LIMITS[:, 1]) / 2 - 0.45 * (LIMITS[:, 1] - LIMITS[:, 0])
SOFT_HI = (LIMITS[:, 0] + LIMITS[:, 1]) / 2 + 0.45 * (LIMITS[:, 1] - LIMITS[:, 0])


def make_task(name, noise=False, dtype=torch.float64, n=N):
    return HDMITask(hdmi_tasks.get(name), n, LIMITS, MASS, noise=noise, dtype=dtype)


def rot_about(axis, angle, dtype=torch.float64):
    a = torch.tensor(axis, dtype=dtype)
    return axis_angle_to_mat(a / a.norm(), torch.tensor(angle, dtype=dtype))


def random_ref(task, g, dtype):
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    mid, half = (LIMITS[:A, 0] + LIMITS[:A, 1]) / 2, 0.45 * (LIMITS[:A, 1] - LIMITS[:A, 0])
    q = (mid + 0.5 * half * (2 * torch.rand(N, F, A, generator=g, dtype=torch.float64) - 1)).to(dtype)  # inside the soft limits
    yaw = axis_angle_to_mat(torch.tensor((0.0, 0.0, 1.0), dtype=dtype).expand(N, F, 3), 3 * r(N, F))
    root = yaw @ so3_exp(0.2 * r(N, F, 3))  # a root near upright, any heading
    body_rot = torch.cat((root.unsqueeze(2), so3_exp(r(N, F, K - 1, 3))), 2)
    return HDMIRef(r(N, F, K, 3), body_rot, r(N, F, K, 3), r(N, F, K, 3), q, r(N, F, A), r(N, F, 3), so3_exp(r(N, F, 3)),
                   torch.rand(N, F, task.num_contact_cols, generator=g) < 0.5, torch.rand(N, generator=g, dtype=dtype),
                   r(N, F) if task.hinged else None)  # fmt: skip


def at_reference(task, seed=0, dtype=torch.float64) -> HDMIInput:
    """A step whose state is what HDMI compares with frame f_{n-1}: the keypoints and joints after the step, the root,
    object and contact snapshots of the previous step, all on the reference; the contacts pressed with 20 N on every
    axis (above both force branches' 10 N); the feet in the air, constant actions, no joint velocity or torque."""
    g = torch.Generator().manual_seed(seed)
    ref, ref_next = random_ref(task, g, dtype), random_ref(task, g, dtype)
    E = task.num_eefs
    mid = ((LIMITS[:, 0] + LIMITS[:, 1]) / 2).to(dtype)
    q = torch.cat((ref.joint_pos[:, 0], mid[A:].expand(N, J - A)), -1)
    qd = torch.cat((ref.joint_vel[:, 0], torch.zeros(N, J - A, dtype=dtype)), -1)
    target = torch.randn(N, E, 3, generator=g, dtype=dtype)
    force = torch.full((N, E, 3), 20.0, dtype=dtype)
    hinge = (ref.obj_joint_pos[:, 0], torch.randn(N, generator=g, dtype=dtype)) if task.hinged else (None, None)
    s = HDMIState(ref.body_pos[:, 0], ref.body_rot[:, 0], ref.body_lin_vel[:, 0], ref.body_ang_vel[:, 0], q, qd, ref.obj_pos[:, 0], ref.obj_rot[:, 0],
                  target, target, force, *hinge)  # fmt: skip
    actions = torch.randn(N, A, 1, generator=g, dtype=dtype).expand(N, A, 3)
    offset = 0.01 * (2 * torch.rand(N, J, generator=g, dtype=dtype) - 1)
    zeros = lambda *s: torch.zeros(*s, dtype=dtype)  # noqa: E731
    return HDMIInput(s, s, ref, ref_next, q.unsqueeze(1).expand(N, 2, J), zeros(N, 2, J), offset, torch.randn(N, A, generator=g, dtype=dtype),
                     zeros(N, J), actions, zeros(N, DECIMATION, 2, 3), torch.randn(N, generator=g, dtype=dtype) if task.hinged else None)  # fmt: skip


def no_contact(dtype=torch.float64):
    return FeetContact(torch.zeros(N, 2, dtype=torch.bool), torch.zeros(N, 2, dtype=dtype), torch.zeros(N, 2, dtype=dtype))


def terms(task, x, feet=None):
    return task.rewards(x, no_contact(task.dtype) if feet is None else feet)[1]


def replace(x, **kw):
    """x with fields of ``now`` / ``prev`` / ``ref`` replaced: replace(x, now={"body_pos": t})."""
    return x._replace(**{k: getattr(x, k)._replace(**v) if isinstance(v, dict) else v for k, v in kw.items()})


def close(a, b, tol=1e-9):
    torch.testing.assert_close(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64).expand_as(torch.as_tensor(a)),
                               atol=tol, rtol=0)  # fmt: skip


# ---------------- configurations ----------------
def test_task_table():
    T = hdmi_tasks.TASKS
    assert len(T) == 13
    hinged = {n for n, c in T.items() if c.object_joint}
    assert hinged == {"open_door-feet", "open_foldchair-sit", "push_door-hand", "traverse_door"}
    for n, c in T.items():
        ot = c.rewards["object_tracking"]
        assert ot["object_pos_tracking"].enabled == ot["object_ori_tracking"].enabled == (n not in hinged)
        assert ("object_joint_pos_tracking" in ot) == (n in hinged)
        assert c.terminations["cum_object_pos_error"].min_steps == (5000 if n in hinged else 25)
    rb = T["roll_ball-hand"]
    assert not rb.rewards["object_tracking"]["eef_contact_exp"].enabled
    mx = rb.rewards["object_tracking"]["eef_contact_exp_max"]
    assert mx.enabled and mx.params["per_axis"] and mx.params["pos_tolerance"] == 0.4
    lc = rb.terminations["cum_lost_contact_steps"]
    assert (lc.min_steps, lc.params["pos_thres"], lc.params["frc_thres"]) == (500, 0.4, 2.0)
    assert rb.terminations["cum_object_ori_error"].threshold == 5.0
    assert T["open_foldchair-sit"].terminations["cum_lost_contact_steps"].min_steps == 200
    tr = T["truman"]
    assert list(tr.rewards["loco"])[-1] == "feet_stumble" and tr.max_episode_length == 4000 and tr.contact_cols == 2
    assert tr.terminations["cum_lost_contact_steps"].min_steps == 1000 and tr.terminations["cum_body_z_error"].threshold == 0.2
    assert T["move_stool-climb"].terminations["cum_body_z_error"].threshold == 0.3 and T["move_suitcase"].max_episode_length == 500
    kp4 = {"carry_and_place_bread_box", "carry_box_over_shoulder", "move_foam", "topple_wood_board_and_cross"}
    assert {n for n, c in T.items() if c.wrist_kp == 4.0} == kp4
    assert {n for n, c in T.items() if c.wrist_yaw_default == (-0.4, 0.4)} == kp4 | {"move_stool-climb", "move_suitcase"}
    assert T["open_door-feet"].contact_eef_bodies == ("left_ankle_roll_link",) and T["push_door-hand"].contact_eef_bodies == ("right_wrist_yaw_link",)
    assert T["traverse_door"].contact_eef_bodies == ("right_wrist_yaw_link", "left_ankle_roll_link") and T["traverse_door"].contact_cols == 2
    with pytest.raises(KeyError):
        hdmi_tasks.get("box_lift")


def test_supertrack_selections_of_every_task():
    """What SuperTrack's options read of each of the 13 tasks: one enabled reward of each velocity class, on HDMI's 14
    velocity keypoints and 19 tracked joints (reward_selection); the loco and feet penalties without survival, truman's
    feet_stumble included (penalty_terms); the full per-step tracking score (full_tracking_score), 5 from the tracking
    group plus the object group's largest values: 2 for a free object's pose or 1 for a hinge angle, and 5 for the contact
    reward with every flag on (1 for roll_ball's). A term whose largest value is not known, or a loco or feet term that is
    not a penalty <= 0, is an error."""
    for name, cfg in hdmi_tasks.TASKS.items():
        assert hdmi.reward_selection(cfg, "keypoint_lin_vel_tracking_product") == VEL
        assert hdmi.reward_selection(cfg, "keypoint_ang_vel_tracking_product") == VEL
        assert hdmi.reward_selection(cfg, "joint_vel_tracking_product") == JOINT_REWARD
        loco = ["action_rate_l2", "joint_vel_l2", "joint_pos_limits", "joint_torque_limits"] + (["feet_stumble"] if name == "truman" else [])
        assert hdmi.penalty_terms(cfg) == [f"loco/{n}" for n in loco] + ["feet/impact_force_l2", "feet/feet_slip", "feet/feet_air_time"], name
        assert hdmi.full_tracking_score(cfg) == 5.0 + (1.0 if cfg.object_joint else 2.0) + (1.0 if name == "roll_ball-hand" else 5.0), name
    cfg, term = hdmi_tasks.get("move_suitcase"), hdmi_tasks.term
    with pytest.raises(ValueError, match="largest value"):
        hdmi.full_tracking_score(dataclasses.replace(cfg, rewards=hdmi_tasks.reward_table(add={"tracking": {"rate": term("action_rate_l2", 0.1)}})))
    for add in ({"loco": {"pos": term("keypoint_pos_tracking_product", 0.5, body_names=("pelvis",), sigma=0.5)}},
                {"feet": {"slip": term("feet_slip", -0.5, tolerance=0.0, body_names=(hdmi_tasks.FEET,))}}):  # fmt: skip
        with pytest.raises(ValueError, match="not a penalty"):
            hdmi.penalty_terms(dataclasses.replace(cfg, rewards=hdmi_tasks.reward_table(add=add)))


# ---------------- rewards ----------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_is_the_maximum_and_never_terminates(dtype):
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    for name, cfg in hdmi_tasks.TASKS.items():
        task = make_task(name, dtype=dtype)
        x = at_reference(task, dtype=dtype)
        task.reset(torch.arange(N), x.now)
        for _ in range(30):
            out = task.step(x)
            assert not out.terminated.any(), name
        assert not task.term_count.any()
        t = out.terms
        for k in cfg.rewards["tracking"]:
            close(t[f"tracking/{k}"], 0.5, tol)
        g = x.ref.contact[:, 0].double()  # (N, C); the contact reward is 5 where flagged, exactly 1 elsewhere
        close(t["object_tracking/eef_contact_exp"], (4 * g + 1).expand(N, cfg.num_eefs).mean(-1), tol)
        obj = (1.0 if cfg.object_joint else 2.0) + (g.any(-1).double() if name == "roll_ball-hand" else (4 * g + 1).expand(N, cfg.num_eefs).mean(-1))
        for k in ("object_pos_tracking", "object_ori_tracking") + (("object_joint_pos_tracking",) if cfg.object_joint else ()):
            close(t[f"object_tracking/{k}"], 1.0, tol)
        one = torch.ones(N, dtype=torch.float64)
        close(out.reward, torch.stack((5 * one, obj, one, one), -1) * DT, tol)


def test_tracking_kernels():
    task = make_task("move_suitcase")
    x = at_reference(task)
    now = x.now
    bp, br, v, w, q, qd = (t.clone() for t in (now.body_pos, now.body_rot, now.body_lin_vel, now.body_ang_vel, now.joint_pos, now.joint_vel))
    d = torch.tensor((0.1, -0.05, 0.2), dtype=torch.float64)
    p2 = bp.clone()
    p2[:, UPPER] += d  # every upper keypoint off by |d| in any frame
    t = terms(task, replace(x, now={"body_pos": p2}))
    close(t["tracking/tracking_upper_body_pos"], 0.5 * math.exp(-d.norm().item() / 0.5))
    for k in ("tracking_lower_body_pos", "tracking_upper_body_ori", "tracking_root_pos"):
        close(t[f"tracking/{k}"], 0.5)
    r2 = br.clone()
    r2[:, LOWER] = rot_about((1, 2, 3), 0.7) @ br[:, LOWER]
    close(terms(task, replace(x, now={"body_rot": r2}))["tracking/tracking_lower_body_ori"], 0.5 * math.exp(-0.7 / 1.0))
    r2 = br.clone()
    r2[:, LOWER[0]] = rot_about((0, 1, 0), 0.6) @ br[:, LOWER[0]]  # one of six bodies: the mean error is 0.1
    close(terms(task, replace(x, now={"body_rot": r2}))["tracking/tracking_lower_body_ori"], 0.5 * math.exp(-0.1 / 1.0))
    p2 = bp.clone()
    p2[:, 0] += torch.tensor((0.3, 0.0, 0.4), dtype=torch.float64)  # the pelvis: world error 0.5; the local terms use the root snapshot
    t = terms(task, replace(x, now={"body_pos": p2}))
    close(t["tracking/tracking_root_pos"], 0.5 * math.exp(-1.0))
    close(t["tracking/tracking_upper_body_pos"], 0.5)
    r2 = br.clone()
    r2[:, 0] = rot_about((1, 1, 0), 0.4) @ br[:, 0]
    close(terms(task, replace(x, now={"body_rot": r2}))["tracking/tracking_root_ori"], 0.5 * math.exp(-0.4 / 0.5))
    v2 = v.clone()
    v2[:, VEL] += torch.tensor((0.3, 0.4, 0.0), dtype=torch.float64)
    close(terms(task, replace(x, now={"body_lin_vel": v2}))["tracking/tracking_body_linvel"], 0.5 * math.exp(-0.5 / 0.5))
    v2 = v.clone()
    v2[:, [KP["left_hip_pitch_link"], KP["right_hip_pitch_link"]]] += 1.0  # not among the 14 velocity bodies
    close(terms(task, replace(x, now={"body_lin_vel": v2}))["tracking/tracking_body_linvel"], 0.5)
    w2 = w.clone()
    w2[:, KP["torso_link"]] += torch.tensor((0.0, 0.0, 1.4), dtype=torch.float64)  # mean over 14 bodies: 0.1
    close(terms(task, replace(x, now={"body_ang_vel": w2}))["tracking/tracking_body_angvel"], 0.5 * math.exp(-0.1 / 2.5))
    q2 = q.clone()
    q2[:, [2, 9]] += 0.19  # waist yaw, left knee: mean over 19 joints 0.02
    q2[:, [13, 17, 23, 28]] += 1.0  # ankles and wrists are not tracked
    close(terms(task, replace(x, now={"joint_pos": q2}))["tracking/joint_pos_tracking_product"], 0.5 * math.exp(-0.02 / 0.25))
    qd2 = qd.clone()
    qd2[:, JOINT_REWARD] += 0.5
    close(terms(task, replace(x, now={"joint_vel": qd2}))["tracking/joint_vel_tracking_product"], 0.5 * math.exp(-0.5 / 2.5))


def test_heading_frames():
    """The local terms compare keypoints in heading frames: a yaw and a horizontal shift of the robot leave them at
    their maximum, a vertical shift moves every keypoint by dz (the root's height is zeroed, the keypoints' not)."""
    task = make_task("carry_box_over_shoulder")
    x = at_reference(task)
    Rz = rot_about((0, 0, 1), 0.8)
    for dz, expected in ((0.0, 0.5), (0.15, 0.5 * math.exp(-0.15 / 0.5))):
        shift = torch.tensor((0.3, -0.2, dz), dtype=torch.float64)
        moved = {"body_pos": x.now.body_pos @ Rz.T + shift, "body_rot": Rz @ x.now.body_rot}
        t = terms(task, replace(x, now=moved, prev=moved))
        for k in ("upper", "lower"):
            close(t[f"tracking/tracking_{k}_body_pos"], expected)
            close(t[f"tracking/tracking_{k}_body_ori"], 0.5)
        assert (t["tracking/tracking_root_pos"] < 0.5).all() and (t["tracking/tracking_root_ori"] < 0.5).all()


def test_object_kernels_read_the_snapshot():
    """Object rewards compare CU(n-1)'s snapshot (time n-1) with frame f_{n-1}; the object after the step is unused (§0)."""
    task = make_task("move_suitcase")
    x = at_reference(task)
    d = torch.tensor((0.15, 0.0, -0.2), dtype=torch.float64)
    t = terms(task, replace(x, prev={"obj_pos": x.prev.obj_pos + d, "obj_rot": rot_about((1, 0, 1), 0.3) @ x.prev.obj_rot}))
    close(t["object_tracking/object_pos_tracking"], math.exp(-0.25 / 0.5))
    close(t["object_tracking/object_ori_tracking"], math.exp(-0.3 / 0.5))
    t = terms(task, replace(x, now={"obj_pos": x.now.obj_pos + 1.0, "obj_rot": rot_about((1, 0, 0), 1.0) @ x.now.obj_rot}))
    close(t["object_tracking/object_pos_tracking"], 1.0)
    close(t["object_tracking/object_ori_tracking"], 1.0)
    door = make_task("open_door-feet")
    x = at_reference(door)
    t = terms(door, replace(x, prev={"obj_joint_pos": x.prev.obj_joint_pos + 0.3}))
    close(t["object_tracking/object_joint_pos_tracking"], math.exp(-0.3 / 0.5))
    assert (t["object_tracking/object_pos_tracking"] == 1).all()  # evaluated, but disabled for a hinged object
    close(terms(door, replace(x, now={"obj_joint_pos": x.now.obj_joint_pos + 1.0}))["object_tracking/object_joint_pos_tracking"], 1.0)


def test_contact_reward():
    task = make_task("move_suitcase")
    x = at_reference(task)
    g = torch.Generator().manual_seed(1)
    far = x.prev.target_pos + torch.randn(N, 2, 3, generator=g, dtype=torch.float64)
    weak = torch.randn(N, 2, 3, generator=g, dtype=torch.float64)  # below 10 N
    off = torch.zeros(N, F, 1, dtype=torch.bool)
    t = terms(task, replace(x, prev={"eef_pos": far, "eef_force": weak}, ref={"contact": off}))
    close(t["object_tracking/eef_contact_exp"], 1.0)  # outside the annotated contact: exactly 1
    on = torch.ones(N, F, 1, dtype=torch.bool)
    t = terms(task, replace(x, prev={"eef_pos": far, "eef_force": weak}, ref={"contact": on}))
    d, f = (far - x.prev.target_pos).norm(dim=-1), weak.norm(dim=-1)
    r = torch.exp(-d / 0.3) * torch.exp((f - 10.0).clamp_max(0.0) / 40.0)
    close(t["object_tracking/eef_contact_exp"], (5.0 * r).mean(-1))
    # the flag of frame f_n is index 0 of CU(n-1)'s window; CU(n)'s flags and the end effectors after the step are unused
    t2 = terms(task, replace(x, prev={"eef_pos": far, "eef_force": weak}, ref={"contact": on}, ref_next={"contact": off},
                             now={"eef_pos": x.now.eef_pos + 5.0}))  # fmt: skip
    close(t2["object_tracking/eef_contact_exp"], (5.0 * r).mean(-1))
    # one flag column per end effector (traverse_door): the unflagged one counts 1
    door = make_task("traverse_door")
    y = at_reference(door)
    flags = torch.zeros(N, F, 2, dtype=torch.bool)
    flags[:, 0, 0] = True
    t = terms(door, replace(y, prev={"eef_pos": far, "eef_force": weak}, ref={"contact": flags}))
    d = (far - y.prev.target_pos).norm(dim=-1)
    r = torch.exp(-d / 0.3) * torch.exp((weak.norm(dim=-1) - 10.0).clamp_max(0.0) / 40.0)
    close(t["object_tracking/eef_contact_exp"], (5.0 * r[:, 0] + 1.0) / 2)


def test_roll_ball_contact_variant():
    """roll_ball: max over the hands, 0.4 m tolerance, the per-axis force branch (an int threshold in HDMI's YAML), 0
    without a flag; the base contact reward is evaluated but not added."""
    task = make_task("roll_ball-hand")
    x = at_reference(task)
    g = torch.Generator().manual_seed(2)
    eef = x.prev.target_pos + 0.5 * torch.randn(N, 2, 3, generator=g, dtype=torch.float64)
    frc = 12.0 * torch.randn(N, 2, 3, generator=g, dtype=torch.float64)
    on = torch.ones(N, F, 1, dtype=torch.bool)
    y = replace(x, prev={"eef_pos": eef, "eef_force": frc}, ref={"contact": on})
    reward, t = task.rewards(y, no_contact())
    d = ((eef - x.prev.target_pos).norm(dim=-1) - 0.4).clamp_min(0.0)
    r = torch.exp(-d / 0.3) * torch.exp((frc.abs() - 10.0).clamp_max(0.0).mean(-1) / 40.0)
    close(t["object_tracking/eef_contact_exp_max"], r.amax(-1))
    close(reward[:, 1], (2.0 + r.amax(-1)) * DT)  # object pos, ori and the max variant; eef_contact_exp is left out
    assert not torch.allclose(t["object_tracking/eef_contact_exp"], torch.ones(N, dtype=torch.float64))
    close(terms(task, replace(y, ref={"contact": torch.zeros(N, F, 1, dtype=torch.bool)}))["object_tracking/eef_contact_exp_max"], 0.0)


@pytest.mark.parametrize("name", ["move_suitcase", "traverse_door", "roll_ball-hand"])
def test_contact_error_is_the_contact_reward_as_an_error(name):
    """contact_error (SuperTrack's --st_contact_force) is -log of the task's contact factor r_e: the mean over the end
    effectors of g_e (-log r_e) for eef_contact_exp (one flag column broadcasts, traverse_door has one per end effector),
    -log max_e r_e while any flag is on for roll_ball's max variant; a force over the threshold adds nothing."""
    from types import SimpleNamespace

    task = make_task(name)
    kind, kw = hdmi.contact_reward(task.cfg)
    assert kind == ("eef_contact_exp_max" if name == "roll_ball-hand" else "eef_contact_exp") and "gain" not in kw
    x = at_reference(task)
    g = torch.Generator().manual_seed(3)
    E, C = x.prev.eef_pos.shape[1], x.ref.contact.shape[-1]
    s = x.prev._replace(eef_pos=x.prev.target_pos + 0.5 * torch.randn(N, E, 3, generator=g, dtype=torch.float64),
                        eef_force=12.0 * torch.randn(N, E, 3, generator=g, dtype=torch.float64))  # fmt: skip
    flag = torch.rand(N, C, generator=g) < 0.5
    flag[0] = True
    r = hdmi._contact_factor(SimpleNamespace(prev=s), **kw)  # (N, E)
    want = (-torch.log(r)).amin(-1) * flag.any(-1) if kind == "eef_contact_exp_max" else (-torch.log(r) * flag.double()).mean(-1)
    close(hdmi.contact_error(s, flag, kind, **kw), want)
    strong = s._replace(eef_force=torch.full((N, E, 3), 15.0, dtype=torch.float64))  # over 10 N in norm and on every axis
    d = ((s.eef_pos - s.target_pos).norm(dim=-1) - kw["pos_tolerance"]).clamp_min(0.0) / kw["pos_sigma"]
    want = d.amin(-1) * flag.any(-1) if kind == "eef_contact_exp_max" else (d * flag.double()).mean(-1)
    close(hdmi.contact_error(strong, flag, kind, **kw), want)


def test_loco_kernels():
    task = make_task("truman")
    x = at_reference(task)
    a = x.action_buf.clone()
    a[..., 0] += 0.1
    close(terms(task, x._replace(action_buf=a))["loco/action_rate_l2"], 0.1 * -23 * 0.01)
    qd = torch.zeros(N, 2, J, dtype=torch.float64)
    qd[:, :, 0], qd[:, 0, 1], qd[:, 1, 1] = 3.0, 0.5, 1.5  # 9 clamps to 5; joint 1 averages to 1
    close(terms(task, x._replace(joint_vel_sub=qd))["loco/joint_vel_l2"], 5e-4 * -(5.0 + 1.0))
    q = x.now.joint_pos.clone()
    q[:, 0], q[:, 3] = SOFT_HI[0] + 0.1, SOFT_LO[3] - 0.2
    close(terms(task, replace(x, now={"joint_pos": q}))["loco/joint_pos_limits"], 10.0 * -0.3)
    tau = torch.zeros(N, J, dtype=torch.float64)
    tau[:, 0], tau[:, 25] = 0.6 * 88 * 1.25, -0.6 * 5 * 2
    close(terms(task, x._replace(applied_torque=tau))["loco/joint_torque_limits"], 0.01 * -(0.25 + 1.0))
    ff = torch.zeros(N, DECIMATION, 2, 3, dtype=torch.float64)
    ff[:, -1, 0] = torch.tensor((0.6, 0.0, 100.0), dtype=torch.float64)  # the left foot stumbles at the latest substep
    ff[:, 1, 1] = torch.tensor((5.0, 0.0, 100.0), dtype=torch.float64)  # the right one did earlier
    close(terms(task, x._replace(feet_force=ff))["loco/feet_stumble"], -0.5)
    close(terms(task, x)["loco/survival"], 1.0)


def _forces(left, right, f=100.0):
    """(N, 4, 2, 3) substep forces: f newtons up on a foot where its pattern has a 1."""
    out = torch.zeros(N, DECIMATION, 2, 3, dtype=torch.float64)
    out[:, :, 0, 2] = f * torch.tensor(left, dtype=torch.float64)
    out[:, :, 1, 2] = f * torch.tensor(right, dtype=torch.float64)
    return out


def test_feet_terms_and_timers():
    task = make_task("truman")
    x = at_reference(task)
    task.reset(torch.arange(N), x.now)
    step = lambda left, right, **kw: task.step(replace(x, feet_force=_forces(left, right), **kw)).terms  # noqa: E731
    W = MASS * 9.81
    step((0, 0, 0, 0), (0, 0, 0, 0))
    t = step((0, 0, 1, 1), (0, 0, 0, 0))  # left touchdown after 6 substeps in the air: last air time 7 x 5 ms
    close(t["feet/feet_air_time"], 5.0 * (0.035 - 0.5))
    close(t["feet/impact_force_l2"], -((200.0 / 3) / W) ** 2)  # mean force over the sensor's last 3 substeps
    close(t["feet/feet_slip"], 0.0)  # in contact for only 10 ms
    v = x.now.body_lin_vel.clone()
    v[:, KP["left_ankle_roll_link"]] = torch.tensor((0.3, 0.4, 0.2), dtype=torch.float64)
    t = step((1, 1, 1, 1), (0, 0, 0, 0), now={"body_lin_vel": v})  # no longer a first contact; sliding at 0.5 m/s
    close(t["feet/feet_air_time"], 0.0)
    close(t["feet/impact_force_l2"], 0.0)
    close(t["feet/feet_slip"], 0.5 * -0.5)
    step((0, 0, 0, 0), (0, 0, 0, 0))
    t = step((0, 0, 0, 1), (0, 0, 0, 0), now={"body_lin_vel": v})  # touchdown in the last substep after 7 in the air
    close(t["feet/feet_air_time"], 5.0 * (0.04 - 0.5))
    t = step((1, 1, 1, 1), (0, 0, 0, 0), now={"body_lin_vel": v})  # 5 substeps in contact: not first, slipping (> 0.02 s)
    close(t["feet/feet_air_time"], 0.0)
    close(t["feet/feet_slip"], 0.5 * -0.5)
    step((0, 0, 0, 0), (0, 0, 0, 0))
    t = step((1, 1, 1, 1), (0, 0, 0, 0), now={"body_lin_vel": v})  # contact for exactly 0.02 s: a first contact, no slip yet
    close(t["feet/feet_air_time"], 5.0 * (0.025 - 0.5))
    close(t["feet/feet_slip"], 0.0)
    for _ in range(25):  # a 0.5 s swing: touchdown 101 x 5 ms after liftoff, no penalty
        step((0, 0, 0, 0), (0, 0, 0, 0))
    close(step((1, 1, 1, 1), (0, 0, 0, 0))["feet/feet_air_time"], 0.0)
    task.reset(torch.arange(N), x.now)  # after a reset a foot on the ground makes a first contact after 0 s in the air
    close(step((1, 1, 1, 1), (1, 1, 1, 1))["feet/feet_air_time"], 5.0 * 2 * -0.5)


# ---------------- terminations ----------------
def test_terminations_count_consecutive_steps():
    task = make_task("move_suitcase", dtype=torch.float32)
    x = at_reference(task, dtype=torch.float32)
    p = x.now.body_pos.clone()
    p[:, KP["torso_link"], 0] += 0.6
    bad = replace(x, now={"body_pos": p})
    task.reset(torch.arange(N), x.now)
    for _ in range(24):
        assert not task.step(bad).terminated.any()
    task.step(x)  # one good step clears the counters
    for _ in range(24):
        assert not task.step(bad).terminated.any()
    out = task.step(bad)
    assert out.terminated.all() and out.term_flags["cum_body_pos_error"].all() and out.term_flags["cum_body_pos_error_local"].all()
    assert not out.term_flags["cum_body_ori_error"].any()
    task.reset(torch.tensor([0]), x.now)
    out = task.step(bad)
    assert out.terminated.tolist() == [False, True, True, True]


def test_lost_contact():
    """Lost contact: a flagged end effector farther than pos_thres or pressing less than frc_thres in CU(n-1)'s
    snapshot; base 0.2 m, 1 N and 25 steps, roll_ball 0.4 m, 2 N and 500, the chair 200, truman 1000."""
    names = ("move_suitcase", "roll_ball-hand", "open_foldchair-sit", "truman")
    for name in names:
        task = make_task(name, dtype=torch.float32)
        x = at_reference(task, dtype=torch.float32)
        i = task.term_names.index("cum_lost_contact_steps")
        assert task.term_min_steps[i] == {"move_suitcase": 25, "roll_ball-hand": 500, "open_foldchair-sit": 200, "truman": 1000}[name]
        on = torch.ones(N, F, task.num_contact_cols, dtype=torch.bool)
        near = x.prev.target_pos + torch.tensor((0.3, 0.0, 0.0))
        err = task.termination_errors(replace(x, prev={"eef_pos": near}, ref={"contact": on}))[:, i]
        assert (err == (0.0 if name == "roll_ball-hand" else 2.0)).all(), name
        weak = torch.full_like(x.prev.eef_force, 0.5)  # |F| = 0.87 N
        assert (task.termination_errors(replace(x, prev={"eef_force": weak}, ref={"contact": on}))[:, i] == 2.0).all()
        assert (task.termination_errors(replace(x, prev={"eef_force": weak}, ref={"contact": on & False}))[:, i] == 0.0).all()
        assert (task.termination_errors(replace(x, now={"eef_pos": near + 9.0}, ref={"contact": on}))[:, i] == 0.0).all()  # snapshot only
    task = make_task("open_foldchair-sit", dtype=torch.float32)
    x = at_reference(task, dtype=torch.float32)
    lost = replace(x, prev={"eef_pos": x.prev.target_pos + 1.0}, ref={"contact": torch.ones(N, F, 1, dtype=torch.bool)})
    task.reset(torch.arange(N), x.now)
    for _ in range(199):
        assert not task.step(lost).terminated.any()
    assert task.step(lost).term_flags["cum_lost_contact_steps"].all()


def test_object_terminations_off_for_hinged_objects():
    for name, steps in (("move_suitcase", 25), ("open_door-feet", None)):
        task = make_task(name, dtype=torch.float32)
        x = at_reference(task, dtype=torch.float32)
        far = replace(x, now={"obj_pos": x.now.obj_pos + 1.0})  # the object after the step, against frame f_{n-1}
        task.reset(torch.arange(N), x.now)
        fired = [task.step(far).term_flags["cum_object_pos_error"].all().item() for _ in range(30)]
        assert fired == [False] * 24 + [True] * 6 if steps else not any(fired)


# ---------------- reference and observations ----------------
def test_reference_window_timing():
    T, C = 40, 2
    fr = torch.arange(T, dtype=torch.float64)
    contact = (torch.arange(T) % 3 == 0).unsqueeze(-1).expand(T, C)
    clip = HDMIClip(fr[:, None, None].expand(T, K, 3), torch.eye(3, dtype=torch.float64).expand(T, K, 3, 3), fr[:, None, None].expand(T, K, 3),
                    fr[:, None, None].expand(T, K, 3), fr[:, None].expand(T, A), fr[:, None].expand(T, A), fr[:, None].expand(T, 3),
                    torch.eye(3, dtype=torch.float64).expand(T, 3, 3), contact, fr)  # fmt: skip
    t = torch.tensor([0, 5, 30, 39])
    ref = hdmi.reference_window(clip, t)
    steps = torch.tensor(hdmi.FUTURE_STEPS)
    frames = (t[:, None] + steps).clamp(max=T - 1)  # held at the clip's last frame
    assert torch.equal(ref.body_pos[..., 0, 0], frames.double()) and torch.equal(ref.joint_pos[..., 0], frames.double())
    assert torch.equal(ref.obj_joint_pos, frames.double())
    assert torch.equal(ref.contact[..., 0], (t[:, None] + 1 + steps).clamp(max=T - 1) % 3 == 0)  # one frame ahead of the poses
    close(ref.phase, (t + 1).double() / T)


def test_observation_dims():
    priv = {"open_foldchair-sit": 1717, "truman": 1719, "traverse_door": 1722}  # 1714 for the others (spec §4.4)
    for name, cfg in hdmi_tasks.TASKS.items():
        task = make_task(name, noise=True, dtype=torch.float32)
        x = at_reference(task, dtype=torch.float32)
        task.reset(torch.arange(N), x.now)
        obs = task.step(x).obs
        dims = {k: v.shape for k, v in obs.items()}
        assert dims == {"command": (N, 356), "policy": (N, 249), "object": (N, 4 + 3 * cfg.num_eefs), "priv": (N, priv.get(name, 1714))}, name
        assert {k: v[1] for k, v in dims.items()} == task.obs_dims
        assert all(torch.isfinite(v).all() for v in obs.values())


def test_observation_content():
    task = make_task("traverse_door")
    x = at_reference(task)
    x = replace(x, now={"body_pos": x.now.body_pos + 0.1})  # after the step: differs from the snapshot
    task.reset(torch.arange(N), x.prev)
    obs = task.step(x).obs
    ref, nxt = x.ref, x.ref_next
    # command: CU(n-1)'s keypoints in the heading frame of its frame-f_{n-1} root; CU(n)'s future joints and phase
    i, k, b = 1, 3, KP["left_elbow_link"]
    yaw = math.atan2(ref.body_rot[i, 0, 0, 1, 0].item(), ref.body_rot[i, 0, 0, 0, 0].item())
    rel = ref.body_pos[i, k, b] - ref.body_pos[i, 0, 0] * torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64)
    close(obs["command"][i].view(-1)[:240].view(F, K, 3)[k, b], rot_about((0, 0, 1), yaw).T @ rel, 1e-10)
    close(obs["command"][:, 240:355], nxt.joint_pos.flatten(1))
    close(obs["command"][:, 355], nxt.phase)
    # policy: newest angular velocity and gravity in the root frame, joint history minus the offsets, actions j * 3 + k
    R0 = x.now.root_rot
    close(obs["policy"][:, 0:3], (R0.transpose(-1, -2) @ x.now.body_ang_vel[:, 0, :, None])[..., 0])
    close(obs["policy"][:, 3:6], -R0[:, 2, :])
    hist = obs["policy"][:, 6:180].view(N, 6, J)
    close(hist[:, 0], x.joint_pos_sub.mean(1) - x.joint_offset)
    close(hist[:, 1:], (x.prev.joint_pos - x.joint_offset).unsqueeze(1).expand(N, 5, J))  # filled at the reset
    close(obs["policy"][:, 180:249].view(N, A, 3), x.action_buf)
    # object: the object after the step in the heading frame of the root snapshot; contact targets of the snapshot
    rs, Rs = x.prev.root_pos, x.prev.root_rot
    yaw_s = torch.atan2(Rs[:, 1, 0], Rs[:, 0, 0])
    c, s = yaw_s.cos(), yaw_s.sin()
    dp = x.now.obj_pos - rs
    close(obs["object"][:, 0], c * dp[:, 0] + s * dp[:, 1], 1e-8)  # ndm_gym's heading frame normalizes with a 1e-9 regularizer
    close(obs["object"][:, 1], -s * dp[:, 0] + c * dp[:, 1], 1e-8)
    phi = torch.atan2(x.now.obj_rot[:, 1, 0], x.now.obj_rot[:, 0, 0]) - yaw_s
    close(obs["object"][:, 2], phi.cos(), 1e-10)
    close(obs["object"][:, 3], phi.sin(), 1e-10)
    # priv at the reference: zero differences at index 0 of the window, identity orientations, CU(n)'s contact flags
    pv = obs["priv"]
    close(pv[:, 315:318], 0.0, 1e-10)  # ref_root_pos_future_b
    close(pv[:, 330:336], torch.tensor((1.0, 0, 0, 0, 1, 0), dtype=torch.float64), 1e-10)  # ref_root_ori_future_b
    close(pv[:, 360:600].view(N, F, K, 3)[:, 0], 0.0, 1e-10)  # diff_body_pos_future_local against the snapshot
    close(pv[:, 1575:1579], x.now.body_pos[:, [0, 5, 10, 11], 2])  # body_height: pelvis, torso, ankles
    close(pv[:, 1579:1602], x.applied_action)
    close(pv[:, 1643:1658].view(N, F, 3)[:, 0], -0.0, 1e-10)  # diff_object_pos_future: the object is on frame f_{n-1}
    close(pv[:, 1703:1713], nxt.contact.flatten(1).double())
    close(pv[:, 1713:1719], 0.0)  # diff_contact_pos_b
    close(pv[:, 1719:1722], torch.stack((x.now.obj_joint_pos, x.now.obj_joint_vel, x.obj_joint_torque), -1))


def test_observation_noise_shapes_and_bounds():
    task = make_task("move_suitcase", noise=True)
    quiet = make_task("move_suitcase")
    x = at_reference(task)
    for t in (task, quiet):
        t.reset(torch.arange(N), x.now)
    o, o0 = task.step(x).obs, quiet.step(x).obs
    torch.testing.assert_close(o["command"], o0["command"], atol=0, rtol=0)  # the command and priv groups carry no noise
    torch.testing.assert_close(o["priv"], o0["priv"], atol=0, rtol=0)
    dc = (o["object"][:, 4:] - o0["object"][:, 4:]).abs()
    assert (dc <= 0.03 + 0.01 + 0.02 + 1e-9).all() and (dc > 0).any()  # 3 sigma, 1 sigma and the episodic uniform noise
    assert ((o["object"][:, :2] - o0["object"][:, :2]).abs() <= 0.03 + 1e-9).all()
    assert ((o["policy"][:, 6:35] - o0["policy"][:, 6:35]).abs() <= 3 * 0.015 + 1e-9).all()
    close(o["policy"][:, 3:6].norm(dim=-1), 1.0)  # projected gravity is renormalized after its noise


# ---------------- differentiability, inputs, geometry ----------------
def test_gradients_flow():
    task = make_task("move_suitcase")
    x = at_reference(task)
    g = torch.Generator().manual_seed(3)
    r = lambda *s: 0.05 * torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    leaves = {
        "body_pos": (x.now.body_pos + r(N, K, 3)).requires_grad_(), "rotvec": r(N, K, 3).requires_grad_(),
        "lin_vel": (x.now.body_lin_vel + r(N, K, 3)).requires_grad_(), "ang_vel": (x.now.body_ang_vel + r(N, K, 3)).requires_grad_(),
        "joint_pos": (x.now.joint_pos + r(N, J)).requires_grad_(), "joint_vel": (x.now.joint_vel + r(N, J)).requires_grad_(),
        "root_pos": (x.prev.body_pos + r(N, K, 3)).requires_grad_(), "obj_pos": (x.prev.obj_pos + r(N, 3)).requires_grad_(),
        "eef_pos": (x.prev.eef_pos + r(N, 2, 3)).requires_grad_(), "eef_force": torch.full((N, 2, 3), 2.0, dtype=torch.float64).requires_grad_(),
        "actions": (x.action_buf + r(N, A, 3)).requires_grad_(), "joint_vel_sub": (1.0 + r(N, 2, J)).requires_grad_(),
        "torque": torch.full((N, J), 80.0, dtype=torch.float64).requires_grad_(),
    }  # fmt: skip
    L = leaves
    build = lambda: replace(  # noqa: E731
        x, now={"body_pos": L["body_pos"], "body_rot": so3_exp(L["rotvec"]) @ x.now.body_rot, "body_lin_vel": L["lin_vel"],
                "body_ang_vel": L["ang_vel"], "joint_pos": L["joint_pos"], "joint_vel": L["joint_vel"]},
        prev={"body_pos": L["root_pos"], "obj_pos": L["obj_pos"], "eef_pos": L["eef_pos"], "eef_force": L["eef_force"]},
        ref={"contact": torch.ones(N, F, 1, dtype=torch.bool)}, action_buf=L["actions"], joint_vel_sub=L["joint_vel_sub"], applied_torque=L["torque"],
    )  # fmt: skip
    reward, _ = task.rewards(build(), no_contact())
    reward.sum().backward()
    for name, t in leaves.items():
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0, name
    for t in leaves.values():
        t.grad = None
    task.reset(torch.arange(N), x.now)
    y = build()
    out = task.step(y)
    (sum(o.sum() for o in out.obs.values()) + task.termination_errors(y).sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in leaves.values() if t.grad is not None)
    assert leaves["body_pos"].grad.abs().sum() > 0 and leaves["root_pos"].grad.abs().sum() > 0


def test_gradients_at_the_reference_are_finite():
    task = make_task("open_door-feet")
    x = at_reference(task)
    p = x.now.body_pos.clone().requires_grad_()
    w = torch.zeros(N, K, 3, dtype=torch.float64, requires_grad=True)
    y = replace(x, now={"body_pos": p, "body_rot": so3_exp(w) @ x.now.body_rot})
    task.rewards(y, no_contact())[0].sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(w.grad).all()


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, tuple):
        return [t for o in obj for t in _tensors(o)]
    return []


def test_inputs_are_not_modified():
    task = make_task("traverse_door", noise=True)
    x = at_reference(task)
    ts = _tensors(x)
    versions, values = [t._version for t in ts], [t.clone() for t in ts]
    task.reset(torch.arange(N), x.now)
    for _ in range(3):
        task.step(x)
    assert [t._version for t in ts] == versions
    assert all(torch.equal(a, b) for a, b in zip(ts, values))


def test_contact_points_and_object_frame_force():
    task = make_task("move_suitcase")
    body_pos = torch.randn(N, K, 3, dtype=torch.float64)
    body_rot = torch.eye(3, dtype=torch.float64).expand(N, K, 3, 3)
    Rz = rot_about((0, 0, 1), math.pi / 2).expand(N, 3, 3)
    p_ob = torch.randn(N, 3, dtype=torch.float64)
    scale = torch.tensor((2.0, 1.0, 1.0), dtype=torch.float64).expand(N, 3)
    eef, target = task.contact_points(body_pos, body_rot, p_ob, Rz, scale)
    close(eef, body_pos[:, [KP["left_wrist_yaw_link"], KP["right_wrist_yaw_link"]]] + torch.tensor((0.05, 0.0, 0.0), dtype=torch.float64))
    # suitcase targets (-0.1, +-0.18, 0.25), x scaled by 2, then turned 90 degrees about z
    close(target, p_ob.unsqueeze(1) + torch.tensor(((-0.18, -0.2, 0.25), (0.18, -0.2, 0.25)), dtype=torch.float64), 1e-12)
    close(hdmi.object_frame_force(Rz, torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64).expand(N, 2, 3)), torch.tensor((0.0, -1.0, 0.0)), 1e-12)


def test_inconsistent_inputs_fail():
    task = make_task("open_door-feet")
    x = at_reference(task)
    with pytest.raises(ValueError):
        task.step(x._replace(obj_joint_torque=None))
    with pytest.raises(ValueError):
        task.step(replace(x, ref={"contact": x.ref.contact.double()}))
    with pytest.raises(ValueError):
        make_task("move_suitcase").step(x)  # one end effector and a hinge for a two-handed rigid-object task


def test_action_pipeline():
    """The delay and filter equal the substep loop they replace in the simulator, bit for bit, and with alpha = 1 the
    action at substep i is a_{k - ceil((d - i) / 4)} (§3.1); gradients reach the new action."""
    torch.manual_seed(0)
    N = 64
    buf, filt = torch.randn(N, hdmi.NUM_ACTIONS, 3), torch.randn(N, hdmi.NUM_ACTIONS)
    delay, alpha = torch.randint(2, 7, (N,)), torch.empty(N, 1).uniform_(0.8, 1.0)
    a = torch.randn(N, hdmi.NUM_ACTIONS, requires_grad=True)
    sub, buf1, filt1 = hdmi.action_pipeline(a, buf, filt, delay, alpha)
    ref_buf = torch.cat((a.unsqueeze(-1), buf[..., :-1]), -1)  # the simulator's former loop, verbatim
    ref_filt, envs = filt.clone(), torch.arange(N)
    for i in range(DECIMATION):
        idx = torch.div(delay - i + DECIMATION - 1, DECIMATION, rounding_mode="floor").clamp(0, ref_buf.shape[-1] - 1)
        ref_filt = ref_filt + alpha * (ref_buf[envs, :, idx] - ref_filt)
        torch.testing.assert_close(sub[:, i], ref_filt, rtol=0, atol=0)
    torch.testing.assert_close(buf1, ref_buf, rtol=0, atol=0)
    torch.testing.assert_close(filt1, ref_filt, rtol=0, atol=0)
    sub1, _, _ = hdmi.action_pipeline(a, buf, filt, delay, torch.ones(N, 1))
    for i in range(DECIMATION):
        lag = torch.clamp(torch.ceil((delay - i) / DECIMATION).long(), 0, 2)
        torch.testing.assert_close(sub1[:, i], ref_buf[envs, :, lag])  # x + (y - x) rounds, so up to float precision
    sub.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
