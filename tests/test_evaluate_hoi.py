"""The HOI evaluator's statistics (ndm_gym.evaluate_hoi) on synthetic tensors, without a simulator: HDMI's success flag
at the clip's end, first episodes only, the episode count, the termination breakdown, HDMI's per-step means and stds,
the debug group's terms against hand-computed values and against the reward kernels on the same errors, the feet's
contact state after a step, the run's algorithm and checkpoint checks, and the CSV row. A short GPU run of both
trainers and of the evaluator on their checkpoints is marked slow and needs CUDA and Isaac Lab."""

import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from ndm_gym.core.rotations import axis_angle_to_mat, so3_exp
from ndm_gym.evaluate_hoi import (DEBUG_TERMS, EvalTerms, accumulate, check_ckpt, csv_columns, csv_row, feet_after_step, load_run_args, new_stats,
                                  reward_keys, run_algo, summarize)  # fmt: skip
from ndm_gym.tasks import hdmi_tasks
from ndm_gym.tasks.hdmi import DECIMATION, GROUPS, JOINT_NAMES, KEYPOINT_NAMES, NUM_ACTIONS, FeetContact, HDMIInput, HDMIRef, HDMIState, HDMITask

N, F, K, J = 4, 5, 16, 29
LIMITS = torch.tensor([(-2.0, 2.0)] * J, dtype=torch.float64)
MASS = 33.341


def make_task(name="move_suitcase", n=N):
    return HDMITask(hdmi_tasks.get(name), n, LIMITS, MASS, noise=False, dtype=torch.float64)


def at_reference(task, seed=0) -> HDMIInput:
    """A step whose state after it and whose previous root snapshot are on the reference of the previous command update
    (index 0): keypoints, joints and object; every end effector on its target, pressed with 20 N; contact flags on."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    E, C = task.num_eefs, task.num_contact_cols
    yaw = axis_angle_to_mat(torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64).expand(N, F, 3), 3 * r(N, F))
    rot = torch.cat(((yaw @ so3_exp(0.2 * r(N, F, 3))).unsqueeze(2), so3_exp(r(N, F, K - 1, 3))), 2)
    hinge = r(N, F) if task.hinged else None
    ref = HDMIRef(r(N, F, K, 3), rot, r(N, F, K, 3), r(N, F, K, 3), 0.5 * r(N, F, NUM_ACTIONS), r(N, F, NUM_ACTIONS), r(N, F, 3), so3_exp(r(N, F, 3)),
                  torch.ones(N, F, C, dtype=torch.bool), torch.rand(N, generator=g, dtype=torch.float64), hinge)  # fmt: skip
    q = torch.cat((ref.joint_pos[:, 0], torch.zeros(N, J - NUM_ACTIONS, dtype=torch.float64)), -1)
    target = r(N, E, 3)
    s = HDMIState(ref.body_pos[:, 0], ref.body_rot[:, 0], ref.body_lin_vel[:, 0], ref.body_ang_vel[:, 0], q, torch.zeros_like(q), ref.obj_pos[:, 0],
                  ref.obj_rot[:, 0], target, target.clone(), torch.full((N, E, 3), 20.0, dtype=torch.float64),
                  hinge[:, 0] if task.hinged else None, torch.zeros(N, dtype=torch.float64) if task.hinged else None)  # fmt: skip
    z = lambda *sh: torch.zeros(*sh, dtype=torch.float64)  # noqa: E731
    return HDMIInput(s, s, ref, ref, q.unsqueeze(1).expand(N, 2, J), z(N, 2, J), z(N, J), z(N, NUM_ACTIONS), z(N, J), z(N, NUM_ACTIONS, 3),
                     z(N, DECIMATION, 2, 3), z(N) if task.hinged else None)  # fmt: skip


def replace(x, **kw):
    return x._replace(**{k: getattr(x, k)._replace(**v) for k, v in kw.items()})


def no_feet():
    return FeetContact(torch.zeros(N, 2, dtype=torch.bool), torch.zeros(N, 2, dtype=torch.float64), torch.zeros(N, 2, dtype=torch.float64))


def debug(terms, x, feet=None):
    v = terms(x, no_feet() if feet is None else feet)
    return {k: v[:, i] for i, k in enumerate(terms.names)}


def close(a, b, tol=1e-9):
    torch.testing.assert_close(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64).expand_as(torch.as_tensor(a)),
                               atol=tol, rtol=0)  # fmt: skip


# ---------------- HDMI's episode statistics ----------------
def _episodes(L=6, steps=8):
    """Four scored envs and the spare under HDMI's loop (HLP:227-235): s = 0, t and the episode's step count go up by one
    per step, an episode ends at a termination (scheduled by episode step), at t >= L or at a time limit, and the env
    restarts at once. env 0 reaches the clip's end; env 1 terminates at its 3rd step, then twice more; env 2 terminates
    at its 6th step, exactly when it reaches the clip's end; env 3 is truncated by a time limit at its 5th step. The
    spare env (index 4) is done at every step. Value column 0 is the global step k, column 1 is 1."""
    term_at = {1: (3, 2, 2), 2: (6,)}
    st = new_stats(N + 1, 2, 2, "cpu")
    t, ep, n_term = torch.zeros(N + 1, dtype=torch.long), torch.zeros(N + 1, dtype=torch.long), [0] * (N + 1)
    for k in range(1, steps + 1):
        t, ep = t + 1, ep + 1
        terminated = torch.tensor([i in term_at and n_term[i] < len(term_at[i]) and int(ep[i]) == term_at[i][n_term[i]] for i in range(N + 1)])
        time_limit = torch.tensor([i == 3 and int(ep[i]) == 5 for i in range(N + 1)])
        done = terminated | time_limit | (t >= L)
        done[N] = True
        flags = torch.stack((terminated, terminated & (torch.arange(N + 1) == 2)), -1)  # term 1 fires only in env 2
        values = torch.stack((torch.full((N + 1,), float(k)), torch.ones(N + 1)), -1)
        values[N] = float("nan")  # the spare env's values are never read
        st = accumulate(st, values, done, terminated, flags, t, ep, L, N)
        for i in range(N + 1):
            n_term[i] += int(terminated[i])
        t, ep = torch.where(done, 0, t), torch.where(done, 0, ep)
    return st


def test_first_episode_statistics():
    st = _episodes()
    assert st["length"][:N].tolist() == [6, 3, 6, 5]
    # HDMI's success: its frame counter before the step's command update at L - 1 (CMD:225-227), i.e. t >= L after the
    # step; env 2 succeeds although a termination fires at that same step
    assert st["success"][:N].tolist() == [True, False, True, False]
    assert st["terminated"][:N].tolist() == [False, True, True, False]
    assert st["flags"][:N].tolist() == [[False, False], [True, False], [True, True], [False, False]]
    # sums over the first episode only: steps 1..len
    close(st["sums"][:N, 0], [21.0, 6.0, 21.0, 15.0])
    close(st["sums"][:N, 1], [6.0, 3.0, 6.0, 5.0])
    # episode ends of the scored envs in the 8 steps: env 0 at 6, env 1 at 3, 5, 7, env 2 at 6, env 3 at 5
    assert int(st["done_count"]) == 6
    assert bool(st["ended"][:N].all())
    per_step = st["sums"][:N] / st["length"][:N].unsqueeze(-1).float()
    close(per_step[:, 0], [3.5, 2.0, 3.5, 3.0])
    close(per_step[:, 1], 1.0)  # a constant averages to itself over any episode



def test_flagged_env_ends_as_terminated_without_its_values():
    """An env the guard flags (hoi_guard.nonfinite) ends its first episode as terminated: its non-finite values of that
    step are left out of the sums, the flag is kept, and it is not a success."""
    st = new_stats(3, 2, 1, "cpu")
    ones = torch.ones(3, 2)
    no, t, ep = torch.zeros(3, dtype=torch.bool), torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])
    st = accumulate(st, ones, no, no, torch.zeros(3, 1, dtype=torch.bool), t, ep, 6, 3)
    bad = torch.tensor([False, True, False])
    values = torch.ones(3, 2)
    values[1] = float("nan")
    st = accumulate(st, values, bad, bad, torch.zeros(3, 1, dtype=torch.bool), t + 1, ep + 1, 6, 3, bad)
    assert torch.isfinite(st["sums"]).all()
    close(st["sums"][1], [1.0, 1.0])  # the flagged step left out
    assert st["nonfinite"].tolist() == [False, True, False] and st["terminated"].tolist() == [False, True, False]
    assert st["length"][1] == 2 and not bool(st["success"][1])
    assert "nonfinite" in csv_columns(["cum_a"], ("e_hinge",))

def _names():
    return [f"{g}/return" for g in GROUPS] + [f"debug/{d}" for d in DEBUG_TERMS] + ["e_kp", "e_obj_pos", "e_obj_ori"]


def test_summary_is_hdmis():
    """HLP:252-266: per key the mean and (unbiased) std over envs of the first episode's value at its last step, divided
    by the episode length except the length and the success flag; the derived metrics on the same episodes."""
    names = _names()
    L, M = 6, len(names)
    st = new_stats(N, M, 2, "cpu")
    length = torch.tensor([6, 3, 6, 5])
    sums = torch.arange(N * M, dtype=torch.float32).view(N, M) + 1.0
    st.update({"sums": sums, "length": length, "success": torch.tensor([True, False, True, False]), "terminated": torch.tensor([False, True, True, False]),
               "flags": torch.tensor([[False, False], [True, False], [True, True], [False, False]]), "ended": torch.ones(N, dtype=torch.bool),
               "done_count": torch.tensor(7)})  # fmt: skip
    res = summarize(st, names, ["cum_a", "cum_b"], N, L)
    h, m = res["hdmi"], res["metrics"]
    per_step = sums / length.unsqueeze(-1).float()
    lf = length.float()
    assert h["eval/episode_len"] == pytest.approx(lf.mean().item()) and h["eval/episode_len_std"] == pytest.approx(lf.std().item())
    assert h["eval/success"] == pytest.approx(0.5) and h["eval/success_std"] == pytest.approx(torch.tensor([1.0, 0, 1, 0]).std().item())
    for i, k in enumerate(names):
        if not k.startswith("e_"):
            assert h[f"eval/{k}"] == pytest.approx(per_step[:, i].mean().item(), rel=1e-6)
            assert h[f"eval/{k}_std"] == pytest.approx(per_step[:, i].std().item(), rel=1e-6)
    # a termination flag of the last step divided by the episode length (HLP:258-261), as HDMI computes it
    assert h["eval/termination/cum_a"] == pytest.approx(((torch.tensor([0, 1, 1, 0]) / lf).mean()).item())
    assert h["eval/termination/cum_b"] == pytest.approx(((torch.tensor([0, 0, 1, 0]) / lf).mean()).item())
    assert h["eval/debug/return"] == 0.0 and h["episode_cnt"] == 7 and not any(k.startswith("eval/e_") for k in h)
    assert list(h) == sorted(h)  # HLP:282
    assert m["n_success"] == 2 and m["success"] == pytest.approx(0.5) and m["ep_len"] == pytest.approx(5.0)
    assert m["clip_frac"] == pytest.approx((lf / L).mean().item())
    assert m["terminated"] == pytest.approx(0.5) and m["time_limit"] == pytest.approx(0.25)
    assert m["term_cum_a"] == pytest.approx(0.5) and m["term_cum_b"] == pytest.approx(0.25)
    col = {k: per_step[:, i] for i, k in enumerate(names)}
    assert m["r_tracking"] == pytest.approx(col["tracking/return"].mean().item(), rel=1e-6)
    assert m["e_kp"] == pytest.approx(col["e_kp"].mean().item(), rel=1e-6)
    assert m["e_kp_p90"] == pytest.approx(torch.quantile(col["e_kp"], 0.9).item(), rel=1e-6)
    assert m["e_joint"] == pytest.approx(col["debug/joint_pos_error"].mean().item(), rel=1e-6)
    assert m["e_obj_ori"] == pytest.approx(col["e_obj_ori"].mean().item(), rel=1e-6)
    assert m["episode_cnt"] == 7 and "e_hinge" not in m


def test_csv_row_matches_columns():
    cols = csv_columns(["cum_a", "cum_b"], ("e_hinge",))
    assert cols[:3] == ["iteration", "train_wall_h", "eval_at"] and cols[-1] == "eval_s" and "e_hinge" in cols and "e_obj_pos" not in cols
    assert "term_cum_a" in cols and all(f"r_{g}" in cols for g in GROUPS) and len(set(cols)) == len(cols)
    metrics = {k: 0.5 for k in cols if k not in ("iteration", "train_wall_h", "eval_at", "n_envs", "eval_s")}
    metrics.update({"n_success": 3, "episode_cnt": 9})
    row = csv_row(cols, {"metrics": metrics, "iteration": 600, "n_envs": 16, "eval_s": 12.25, "wall_time": 7200.0, "eval_at": "2026-10-30 01:02:03"})
    fields = row.rstrip("\n").split(",")
    assert len(fields) == len(cols) and fields[0] == "600" and fields[1] == "2.00000" and fields[2] == "2026-10-30 01:02:03"
    assert fields[cols.index("n_success")] == "3" and fields[cols.index("success")] == "0.50000" and fields[cols.index("eval_s")] == "12.25000"


# ---------------- the debug group and the errors ----------------
def test_terms_at_the_reference():
    for name in ("move_suitcase", "traverse_door", "open_door-feet"):
        task = make_task(name)
        terms = EvalTerms(task)
        v = debug(terms, at_reference(task))
        for k in ("root_pos_error", "root_ori_error", "body_pos_error_local", "body_ori_error_local", "joint_pos_error"):
            close(v[f"debug/{k}"], 0.0, tol=1e-6)
        close(v["e_kp"], 0.0)
        close(v["debug/eef_contact_all"], 1.0)  # every flagged end effector near its target and pressed
        close(v["debug/feet_air_time"], 0.0)
        close(v["debug/feet_contact_count"], 0.0)
        if task.hinged:
            close(v["e_hinge"], 0.0)
            assert "e_obj_pos" not in v
        else:
            close(v["e_obj_pos"], 0.0)
            close(v["e_obj_ori"], 0.0, tol=1e-6)


def test_errors_for_known_offsets():
    task = make_task()
    terms = EvalTerms(task)
    x = at_reference(task)
    d = torch.tensor([0.03, -0.04, 0.12], dtype=torch.float64)
    # every keypoint moved by d (the root snapshot of the previous step unchanged): all position errors are |d|
    v = debug(terms, replace(x, now={"body_pos": x.now.body_pos + d}))
    for k in ("debug/root_pos_error", "debug/body_pos_error_local", "e_kp"):
        close(v[k], d.norm())
    close(v["debug/body_ori_error_local"], 0.0, tol=1e-6)
    # every keypoint turned by 0.3 rad about the world's z axis: the orientation errors are 0.3
    Rz = axis_angle_to_mat(torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64))
    v = debug(terms, replace(x, now={"body_rot": Rz @ x.now.body_rot}))
    close(v["debug/root_ori_error"], 0.3, tol=1e-6)
    close(v["debug/body_ori_error_local"], 0.3, tol=1e-6)
    # joints: the 19 tracked ones (no ankles, no wrists, Y:241)
    tracked = [i for i, n in enumerate(JOINT_NAMES[:NUM_ACTIONS]) if "ankle" not in n]
    assert len(tracked) == 19 and terms.joints.tolist() == tracked
    q = x.now.joint_pos.clone()
    q[:, tracked] += 0.05
    close(debug(terms, replace(x, now={"joint_pos": q}))["debug/joint_pos_error"], 0.05)
    q = x.now.joint_pos.clone()
    q[:, [JOINT_NAMES.index("left_ankle_pitch_joint"), JOINT_NAMES.index("left_wrist_yaw_joint")]] += 0.5
    close(debug(terms, replace(x, now={"joint_pos": q}))["debug/joint_pos_error"], 0.0)
    # the 12 bodies of the local terms: the six upper and six lower bodies (Y:239-240), not the pelvis, torso or hip yaw
    assert sorted(KEYPOINT_NAMES[i] for i in terms.bodies.tolist()) == sorted(
        f"{s}_{b}_link" for s in ("left", "right") for b in ("hip_pitch", "knee", "ankle_roll", "shoulder_pitch", "elbow", "wrist_yaw"))
    # the object: moved by d and turned by 0.3 rad
    v = debug(terms, replace(x, now={"obj_pos": x.now.obj_pos + d, "obj_rot": Rz @ x.now.obj_rot}))
    close(v["e_obj_pos"], d.norm())
    close(v["e_obj_ori"], 0.3, tol=1e-6)
    hinge = make_task("push_door-hand")
    xh = at_reference(hinge)
    close(debug(EvalTerms(hinge), replace(xh, now={"obj_joint_pos": xh.now.obj_joint_pos - 0.2}))["e_hinge"], 0.2)


def test_errors_are_the_reward_kernels_errors():
    """HDMI's debug errors are the errors inside its tracking rewards (REW:52-322): on the same step,
    tracking_root_pos = 0.5 exp(-root_pos_error / 0.5), tracking_root_ori = 0.5 exp(-root_ori_error / 0.5) and
    joint_pos_tracking_product = 0.5 exp(-joint_pos_error / 0.25); loco's joint_pos_limits is 10 x the debug one."""
    task = make_task()
    terms = EvalTerms(task)
    x = at_reference(task, seed=3)
    g = torch.Generator().manual_seed(4)
    q = x.now.joint_pos + 0.3 * torch.randn(N, J, generator=g, dtype=torch.float64)
    q[:, 0] = 2.5  # past the soft limit
    x = replace(x, now={"body_pos": x.now.body_pos + 0.05 * torch.randn(N, K, 3, generator=g, dtype=torch.float64),
                        "body_rot": so3_exp(0.2 * torch.randn(N, K, 3, generator=g, dtype=torch.float64)) @ x.now.body_rot, "joint_pos": q})
    v = debug(terms, x)
    r = task.rewards(x, no_feet())[1]
    close(r["tracking/tracking_root_pos"], 0.5 * torch.exp(-v["debug/root_pos_error"] / 0.5))
    close(r["tracking/tracking_root_ori"], 0.5 * torch.exp(-v["debug/root_ori_error"] / 0.5))
    close(r["tracking/joint_pos_tracking_product"], 0.5 * torch.exp(-v["debug/joint_pos_error"] / 0.25))
    close(r["loco/joint_pos_limits"], 10.0 * v["debug/joint_pos_limits"])
    assert (v["debug/joint_pos_limits"] < 0).all()


def test_contact_and_feet_terms():
    """eef_contact_all (REW:452-490): mean over the end effectors of [near and pressed] where flagged, 1 elsewhere; the
    position threshold is 0.1 m, 0.4 m on roll_ball (T:roll_ball-hand:43-44); the force threshold 1 N."""
    for name, thres in (("move_suitcase", 0.1), ("roll_ball-hand", 0.4)):
        task = make_task(name)
        terms = EvalTerms(task)
        assert terms.contact_pos_thres == thres
        x = at_reference(task)
        away = x.prev.eef_pos.clone()
        away[:, 0, 0] += 0.15
        v = debug(terms, replace(x, prev={"eef_pos": away}))
        close(v["debug/eef_contact_all"], 0.5 if thres == 0.1 else 1.0)
        weak = x.prev.eef_force.clone()
        weak[:, 1] = 0.5
        close(debug(terms, replace(x, prev={"eef_force": weak}))["debug/eef_contact_all"], 0.5)
        off = x.ref._replace(contact=torch.zeros_like(x.ref.contact))
        close(debug(terms, x._replace(ref=off, prev=x.prev._replace(eef_pos=away)))["debug/eef_contact_all"], 1.0)
    task = make_task()
    terms = EvalTerms(task)
    feet = FeetContact(torch.tensor([[True, False]] * N), torch.zeros(N, 2, dtype=torch.float64), torch.tensor([[0.1, 0.3]] * N, dtype=torch.float64))
    v = debug(terms, at_reference(task), feet)
    close(v["debug/feet_air_time"], 5.0 * (0.1 - 0.2))  # Y:233: thres 0.2, weight 5; a penalty on the touchdown after 0.1 s
    close(v["debug/feet_contact_count"], 1.0)


def test_feet_after_step_is_advance_feet():
    task = make_task()
    g = torch.Generator().manual_seed(0)
    for _ in range(12):
        force = 2.0 * torch.rand(N, DECIMATION, 2, 3, generator=g, dtype=torch.float64)  # norms around the 1 N threshold
        feet = task.advance_feet(force)
        after = feet_after_step(task)
        for a, b in zip(feet, after):
            assert torch.equal(a, b)


def test_value_columns():
    task = make_task("truman")
    keys = reward_keys(task)
    assert "loco/feet_stumble" in keys and keys[0] == "tracking/tracking_upper_body_pos"
    x = at_reference(task)
    out = task.rewards(x, no_feet())[1]
    assert sorted(keys) == sorted(out)  # every term HDMITask computes, disabled ones included
    assert len(EvalTerms(task).names) == len(DEBUG_TERMS) + 3


# ---------------- runs and checkpoints ----------------
def test_run_algo_and_checks(tmp_path):
    assert run_algo({"task": "move_suitcase", "algo": "supertrack"}) == "supertrack"
    assert run_algo({"task": "move_suitcase", "algo": "supermimic"}) == "supermimic"
    assert run_algo({"task": "move_suitcase", "phase": "train"}) == "ppo_roa_train"
    assert run_algo({"task": "move_suitcase", "phase": "finetune"}) == "ppo_roa_finetune"
    with pytest.raises(ValueError):
        run_algo({"algo": "supertrack"})  # a LAFAN1 run: no task
    with pytest.raises(ValueError):
        run_algo({"task": "move_suitcase", "algo": "ppo"})
    with pytest.raises(ValueError):
        run_algo({"task": "move_suitcase", "phase": "adapt"})
    with pytest.raises(ValueError):
        run_algo({"task": "move_suitcase"})
    with pytest.raises(FileNotFoundError):
        load_run_args(str(tmp_path))
    a = {"task": "move_suitcase", "algo": "supertrack"}
    (tmp_path / "args.json").write_text(json.dumps(a))
    assert load_run_args(str(tmp_path)) == a
    check_ckpt({"args": dict(a), "pi": {}}, a, "ckpt")
    with pytest.raises(ValueError):
        check_ckpt({"args": {**a, "task": "push_box"}, "pi": {}}, a, "ckpt")
    with pytest.raises(ValueError):
        check_ckpt({"args": {**a, "algo": "supermimic"}, "pi": {}}, a, "ckpt")
    with pytest.raises(KeyError):
        check_ckpt({"args": dict(a), "agent": {}}, a, "ckpt")
    p = {"task": "move_suitcase", "phase": "finetune"}
    check_ckpt({"args": dict(p), "agent": {}}, p, "ckpt")
    with pytest.raises(ValueError):
        check_ckpt({"args": {**p, "phase": "train"}, "agent": {}}, p, "ckpt")


HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainers and the evaluator need a GPU and Isaac Lab")
def test_evaluator_gpu_smoke(tmp_path):
    """Tiny runs of ndm_gym.train_hoi (SuperTrack, 2 iterations) and ndm_gym.train_ppo_hoi (the teacher, 2 iterations,
    then the student, 1) on move_suitcase, each evaluated on 16 envs: the run's checkpoints into eval_hoi.csv, and one
    of them again with --ckpt and --out_json."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], cwd=repo, capture_output=True, text=True, timeout=1800)  # noqa: E731
    ok = lambda p: p.returncode == 0 or pytest.fail(p.stdout[-3000:] + p.stderr[-3000:])  # noqa: E731
    st, te, stu = tmp_path / "st", tmp_path / "teacher", tmp_path / "student"
    ok(run("ndm_gym.train_hoi", "--task", "move_suitcase", "--algo", "supertrack", "--num_envs", "64", "--iterations", "2", "--ckpt_every", "1",
           "--warmup_steps", "8", "--buffer_steps", "64", "--wm_batches", "1", "--wm_batch", "32", "--wm_hidden", "64", "64", "--log_every", "1",
           "--log_dir", str(st)))  # fmt: skip
    ppo = ["ndm_gym.train_ppo_hoi", "--task", "move_suitcase", "--num_envs", "64", "--train_every", "8", "--log_every", "1"]
    ok(run(*ppo, "--iterations", "2", "--ckpt_every", "1", "--log_dir", str(te)))
    ok(run(*ppo, "--phase", "finetune", "--teacher", str(te / "ckpt_000002.pt"), "--iterations", "1", "--log_dir", str(stu)))
    for d, n in ((st, 2), (te, 2), (stu, 1)):
        ok(run("ndm_gym.evaluate_hoi", "--run_dir", str(d), "--num_envs", "16"))
        rows = (d / "eval_hoi.csv").read_text().strip().splitlines()
        cols = rows[0].split(",")
        assert len(rows) == n + 1 and all(len(r.split(",")) == len(cols) for r in rows[1:])
        assert [int(r.split(",")[0]) for r in rows[1:]] == list(range(1, n + 1))
        for r in rows[1:]:
            v = dict(zip(cols, r.split(",")))
            assert 0.0 <= float(v["success"]) <= 1.0 and 1.0 <= float(v["ep_len"]) <= 500 and int(v["episode_cnt"]) >= 16
            assert all(math.isfinite(float(v[k])) for k in cols if k not in ("eval_at",))
        res = json.loads((d / f"ckpt_{n:06d}_eval_hoi.json").read_text())
        assert res["n_envs"] == 16 and res["steps"] == 500 and res["hdmi"]["episode_cnt"] == res["metrics"]["episode_cnt"]
    out = tmp_path / "one.json"  # one checkpoint, uncompiled and without the simulator's CUDA graph (the same computations)
    ok(run("ndm_gym.evaluate_hoi", "--run_dir", str(te), "--ckpt", str(te / "ckpt_000001.pt"), "--num_envs", "16", "--out_json", str(out),
           "--no_compile", "--eager_sim"))  # fmt: skip
    res = json.loads(out.read_text())
    assert res["algo"] == "ppo_roa_train" and res["iteration"] == 1 and "eval/success" in res["hdmi"] and "eval/debug/joint_pos_error" in res["hdmi"]
