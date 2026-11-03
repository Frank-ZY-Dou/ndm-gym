"""HOI training's guard against non-finite simulator states and its bookkeeping (ndm_gym.algos.hoi_guard, the replay,
normalizer and update handling of ndm_gym.algos.hoi_algos, ndm_gym.train_hoi's options) on CPU: the guard flags each
checked quantity per env, NaN included; a flagged transition is finite in the replay and left out of the model's windows
and normalizers; the masked normalizer update is RunningNorm's on the kept rows; the monitor records the run's first
event; the checks stop training; updates with a non-finite gradient are skipped and counted; train_hoi refuses the
LAFAN1 options it does not implement. A short GPU run of train_hoi (marked slow, needs CUDA and Isaac Lab) checks the
log's columns, the checkpoints, --ckpt_hours, --lr_decay_hours and --resume."""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys

import pytest
import torch
from test_hdmi_task import at_reference, make_task
from test_hoi_imagination import collect, fake_env, small_agent

from ndm_gym.algos.hoi_guard import FORCE_BOUND, VEL_BOUND, NonfiniteMonitor, SkippedUpdates, check, guard_terms, nonfinite, record_layout
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.tasks.hdmi import NUM_ACTIONS

WINDOW_KEYS = ("s", "s1", "o", "o1", "u", "ctx", "f_eef", "f_feet", "done", "bad")  # train_hoi's


def step_input(name="move_suitcase"):
    task = make_task(name, dtype=torch.float32)
    x = at_reference(task, dtype=torch.float32)
    return task, x, task.observations(x)


def with_now(x, **kw):
    return x._replace(now=x.now._replace(**kw))


def test_guard_flags_each_quantity_per_env():
    """Every checked quantity of every env: NaN, an infinity and a value beyond its bound fail, values at the bound pass;
    the record holds the quantities, the action and the frame counter, as record_layout splits it."""
    task, x, obs = step_input()
    a, t = torch.zeros(4, NUM_ACTIONS), torch.arange(4)
    failed, record = nonfinite(x, obs, a, t)
    names = [n for n, _, _ in guard_terms(x, obs)]
    assert failed.shape == (4, len(names)) and not failed.any()
    assert {"body_pos", "joint_vel", "eef_force", "feet_force", "applied_torque", "obs/command", "obs/priv"} <= set(names)
    jv, ff, bp = x.now.joint_vel.clone(), x.feet_force.clone(), x.now.body_pos.clone()
    jv[1, 3], ff[2, 1, 0, 2], bp[3, 7, 0] = float("nan"), float("inf"), 1.5e3
    jv[0, 0], ff[0, 0, 0, 0] = VEL_BOUND, -FORCE_BOUND  # at the bounds
    x2 = with_now(x, joint_vel=jv, body_pos=bp)._replace(feet_force=ff)
    obs2 = dict(obs, object=obs["object"].clone())
    obs2["object"][0, 1] = float("nan")
    failed, record = nonfinite(x2, obs2, a, t)
    want = torch.zeros_like(failed)
    for env, name in ((1, "joint_vel"), (2, "feet_force"), (3, "body_pos"), (0, "obs/object")):
        want[env, names.index(name)] = True
    assert torch.equal(failed, want)
    layout = record_layout(x, obs, a)
    widths = [torch.Size(s).numel() for _, s in layout]
    assert sum(widths) == record.shape[1] and layout[-2:] == [("action", (NUM_ACTIONS,)), ("t", ())]
    assert torch.equal(record[:, -1], t.float()) and torch.isnan(record[1]).any() and torch.isfinite(record[3]).all()
    hinged_task, hx, hobs = step_input("open_door-feet")
    assert {"hinge_pos", "hinge_vel", "hinge_torque"} <= {n for n, _, _ in guard_terms(hx, hobs)}


def test_sanitize_transition():
    """A flagged row: the state and object after the step become those before it, the forces zero, the pushed joint
    observation the previous one; the other rows are untouched, and every value is finite."""
    from ndm_gym.algos.hoi_algos import sanitize_transition

    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    tr = {"s": r(4, 71), "s1": r(4, 71), "o": r(4, 13), "o1": r(4, 13), "f_eef": r(4, 2, 3), "f_feet": r(4, 4, 2, 3), "q_obs": r(4, 29), "a": r(4, 23)}
    bad = torch.tensor([False, True, False, True])
    tr["s1"][1], tr["o1"][3], tr["f_eef"][1, 0, 0], tr["f_feet"][3, 2, 1, 0], tr["q_obs"][1, 4] = float("nan"), float("inf"), float("nan"), 1e9, float("nan")
    q_prev = r(4, 29)
    out = sanitize_transition(tr, bad, q_prev)
    assert all(torch.isfinite(v).all() for k, v in out.items() if k != "bad") and torch.equal(out["bad"], bad)
    for k, src in (("s1", "s"), ("o1", "o")):
        assert torch.equal(out[k][bad], tr[src][bad]) and torch.equal(out[k][~bad], tr[k][~bad])
    assert (out["f_eef"][bad] == 0).all() and (out["f_feet"][bad] == 0).all() and torch.equal(out["f_feet"][~bad], tr["f_feet"][~bad])
    assert torch.equal(out["q_obs"][bad], q_prev[bad]) and torch.equal(out["q_obs"][~bad], tr["q_obs"][~bad]) and out["a"] is tr["a"]


def test_masked_normalizer_update():
    """RunningNorm.update on the kept rows (NaN in the others), in place; without kept rows or once frozen, unchanged."""
    from ndm_gym.algos.hoi_algos import masked_update

    torch.manual_seed(0)
    a, b = RunningNorm(5), RunningNorm(5)
    ptrs = [t.data_ptr() for t in a.buffers()]
    for _ in range(3):
        x, keep = 3 * torch.randn(64, 5) + 1, torch.rand(64) > 0.3
        x[~keep] = float("nan")
        masked_update(a, x, keep)
        b.update(x[keep])
    for p, q in zip(a.buffers(), b.buffers()):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    assert ptrs == [t.data_ptr() for t in a.buffers()]
    before = [t.clone() for t in a.buffers()]
    masked_update(a, x, torch.zeros(64, dtype=torch.bool))
    assert all(torch.equal(p, q) for p, q in zip(before, a.buffers()))
    a.count.fill_(a.max_count)
    before = [t.clone() for t in a.buffers()]
    masked_update(a, torch.randn(64, 5), torch.ones(64, dtype=torch.bool))
    assert all(torch.equal(p, q) for p, q in zip(before, a.buffers()))


def test_monitor_records_the_first_event(tmp_path):
    """The flagged envs are counted per window; the first event's env, step, failed quantities and its last five
    records are written once, and a later monitor of the same run directory does not overwrite them."""
    task, x, obs = step_input()
    a = torch.zeros(4, NUM_ACTIONS)
    layout, fields = record_layout(x, obs, a), [n for n, _, _ in guard_terms(x, obs)]
    mon = NonfiniteMonitor(str(tmp_path), layout, fields, 4, "cpu", compile=False)
    for step in range(9):
        jv = x.now.joint_vel.clone()
        if step == 6:
            jv[2, 0] = float("nan")
        if step == 8:
            jv[3, 0], jv[1, 0] = float("nan"), float("nan")  # later events do not replace the first
        failed, record = nonfinite(with_now(x, joint_vel=jv), obs, a + step, torch.full((4,), 100 + step))
        mon.step(failed.any(-1), failed, record)
    assert mon.take() == 3 and mon.take() == 0 and bool(mon.captured) and mon.event_step() == 6
    mon.dump(iteration=11)
    ev = torch.load(tmp_path / "nonfinite_event.pt")
    assert (ev["env"], ev["iteration"], ev["step"], ev["t"], ev["failed"]) == (2, 11, 6, 106, ["joint_vel"])
    h = ev["history"]
    assert h["t"].tolist() == [102, 103, 104, 105, 106] and h["action"][:, 0].tolist() == [2, 3, 4, 5, 6]
    assert torch.isnan(h["joint_vel"][-1, 0]) and torch.isfinite(h["joint_vel"][:-1]).all() and h["joint_vel"].shape == (5, 29)
    assert NonfiniteMonitor(str(tmp_path), layout, fields, 4, "cpu", compile=False).dumped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host synchronizations are a CUDA notion")
def test_guard_and_monitor_compiled_without_host_sync(tmp_path):
    """The compiled guard and the monitor's capture, as the trainers run them every step, read nothing on the host (CUDA's
    synchronization debug mode raises on any synchronization)."""
    task = make_task("move_suitcase", dtype=torch.float32)
    cpu = at_reference(task, dtype=torch.float32)
    cuda = lambda v: v.cuda() if torch.is_tensor(v) else type(v)(*map(cuda, v)) if isinstance(v, tuple) else v  # noqa: E731
    x, obs = cuda(cpu), {k: v.cuda() for k, v in task.observations(cpu).items()}
    a, t = torch.zeros(4, NUM_ACTIONS, device="cuda"), torch.arange(4, device="cuda")
    guard = torch.compile(nonfinite)
    mon = NonfiniteMonitor(str(tmp_path), record_layout(x, obs, a), [n for n, _, _ in guard_terms(x, obs)], 4, "cuda")
    jv = x.now.joint_vel.clone()
    jv[2, 0] = float("nan")
    for step in range(4):  # compiled on the first calls, with and without an event
        failed, record = guard(with_now(x, joint_vel=jv) if step % 2 else x, obs, a, t)
        mon.step(failed.any(-1), failed, record)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for step in range(4):
            failed, record = guard(with_now(x, joint_vel=jv) if step % 2 else x, obs, a, t)
            mon.step(failed.any(-1), failed, record)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert mon.take() == 4 and mon.event_step() == 1


def test_checks_stop_training():
    """Non-finite parameters or normalizer buffers, an event rate above MAX_EVENT_RATE and runs of more than MAX_SKIPPED
    skipped updates raise; the counter of skipped updates counts and carries a run over windows."""
    m = torch.nn.ModuleDict({"lin": torch.nn.Linear(3, 2), "norm": RunningNorm(3)})
    check(1, {"m": m}, n_flagged=10, env_steps=10**5)
    check(1, {"m": m}, skipped={"model": 20, "policy": 0})
    with pytest.raises(RuntimeError, match="flagged"):
        check(1, {"m": m}, n_flagged=11, env_steps=10**5)
    with pytest.raises(RuntimeError, match="21 consecutive policy"):
        check(1, {"m": m}, skipped={"model": 0, "policy": 21})
    m["norm"].mean[1] = float("nan")
    with pytest.raises(RuntimeError, match="m.norm.mean"):
        check(1, {"m": m, "none": None})
    s = SkippedUpdates()
    for skipped in (True, True, False, True, True, True):
        s(skipped)
    assert s.take() == (5, 3)
    s(True), s(False)
    assert s.take() == (1, 4)  # the run of 3 continued into the new window


def test_window_mask_leaves_out_flagged_steps():
    """A flagged step and the window's steps after it are out of the model's loss; its sanitized values keep the loss
    finite, and so does the mask on its own (window_loss leaves masked steps out even when their error is NaN)."""
    env = fake_env(4)
    agent = small_agent(env)
    buf, _ = collect(env, agent, 12)
    buf.window_keys = WINDOW_KEYS
    w = buf.windows_at(torch.tensor([1, 2]), torch.tensor([0, 3]), 4)
    w["bad"] = w["bad"].clone()
    w["bad"][0, 2] = True
    loss = agent._wm_window(w)
    preds = agent.wm.rollout(*[f(w[k][:, 0]) for f, k in ((agent_state, "s"), (agent.obj, "o"))], w["u"], w["ctx"])
    trues = [(agent_state(w["s1"][:, j]), agent.obj(w["o1"][:, j]), (w["f_eef"][:, j], w["f_feet"][:, j])) for j in range(4)]
    mask = w["mask"].clone()
    mask[0, 2:] = 0
    torch.testing.assert_close(loss, agent.wm.window_loss(preds, trues, mask)[0])
    w["s1"] = w["s1"].clone()
    w["s1"][0, 2] = float("nan")
    torch.testing.assert_close(agent._wm_window(w), loss)


def agent_state(s):
    from ndm_gym.state import unflatten_state

    return unflatten_state(s)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_updates_with_a_nonfinite_gradient_are_skipped(algo):
    """A model update whose loss is not finite leaves the model and its optimizer unchanged and is counted; so does a
    policy update whose imagined rollout is not finite (SuperMimic: the critic's too). Finite updates step."""
    torch.manual_seed(0)
    env = fake_env(4)
    agent = small_agent(env, algo, B=16)
    agent.wm_window, agent.imag_step, agent.imag_start = agent._wm_window, agent._imag_step, agent._imag_start  # eager on CPU
    if algo == "supermimic":
        agent.critic_targets, agent.critic_loss = agent._critic_targets, agent._critic_loss
    buf, _ = collect(env, agent, 12)
    buf.window_keys = WINDOW_KEYS
    assert agent.wm_update(buf) is not None and agent.wm_skips.run == 0
    out = agent.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values())
    snap = lambda m: [p.detach().clone() for p in m.parameters()]  # noqa: E731
    same = lambda m, ps: all(torch.equal(p, q) for p, q in zip(m.parameters(), ps))  # noqa: E731
    wm0, pi0, critic0 = snap(agent.wm), snap(agent.pi), snap(agent.critic) if algo == "supermimic" else None
    adam0 = {k: v.clone() for k, v in agent.opt_wm.state[next(agent.wm.parameters())].items()}
    good = buf.data["s1"].clone()
    buf.data["s1"][:] = float("nan")  # every window's targets
    assert agent.wm_update(buf) is None and agent.wm_skips.n == 1 and agent.wm_skips.run == 1
    assert same(agent.wm, wm0) and all(torch.equal(v, adam0[k]) for k, v in agent.opt_wm.state[next(agent.wm.parameters())].items())
    buf.data["s1"].copy_(good)
    with torch.no_grad():
        next(agent.wm.parameters()).view(-1)[0] = float("nan")  # the imagined rollout's states
    assert agent.policy_update(buf) == {"skipped": True} and agent.pi_skips.take() == (1, 1) and same(agent.pi, pi0)
    if critic0 is not None:
        assert same(agent.critic, critic0)
    with torch.no_grad():
        next(agent.wm.parameters()).view(-1)[0] = wm0[0].view(-1)[0]
    assert agent.wm_update(buf) is not None and agent.wm_skips.run == 0 and not same(agent.wm, wm0)


def test_supertrack_contact_force_column():
    """--st_contact_force changes only SuperTrack's contact column, to contact_error of the state the step ends at with the
    task's contact reward; the default keeps the end effectors' distance. A policy update with it is finite."""
    from ndm_gym.tasks.hdmi import contact_error, contact_reward

    torch.manual_seed(0)
    env = fake_env(4)
    plain, force = small_agent(env), small_agent(env, contact_force=True)
    x = at_reference(make_task("move_suitcase"))
    flag = x.ref.contact[:, 0].double()
    e0, e1 = plain._errors(x), force._errors(x)
    assert torch.equal(e0[:, :-1], e1[:, :-1])
    dist = (x.now.eef_pos - x.now.target_pos).norm(dim=-1)
    torch.testing.assert_close(e0[:, -1], (dist * flag.expand_as(dist)).mean(-1))
    kind, kw = contact_reward(env.cfg)
    torch.testing.assert_close(e1[:, -1], contact_error(x.now, flag, kind, **kw))
    force.wm_window, force.imag_step, force.imag_start = force._wm_window, force._imag_step, force._imag_start  # eager on CPU
    buf, _ = collect(env, force, 12)
    buf.window_keys = WINDOW_KEYS
    assert force.wm_update(buf) is not None
    out = force.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values())


def test_supertrack_object_weight_and_noise():
    """--st_obj_w scales the object's three terms in SuperTrack's loss after calibration (1: the default code path, the same
    loss bit for bit); --st_obj_imag_noise perturbs the imagined object's velocities (0: no noise drawn, the same rollout)."""
    torch.manual_seed(0)
    env = fake_env(4)
    agents = {}
    for key, kw in {"plain": {}, "w1": {"obj_w": 1.0}, "w3": {"obj_w": 3.0}, "noise": {"obj_imag_noise": 0.5}}.items():
        torch.manual_seed(0)
        agents[key] = small_agent(env, **kw)
        agents[key].wm_window, agents[key].imag_step, agents[key].imag_start = agents[key]._wm_window, agents[key]._imag_step, agents[key]._imag_start
    assert agents["plain"].term_scale is None and agents["w1"].term_scale is None
    assert torch.equal(agents["w3"].term_scale, torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]))
    buf, _ = collect(env, agents["plain"], 12)
    buf.window_keys = WINDOW_KEYS
    losses = {}
    for key, a in agents.items():
        a.bal.calibrate(torch.ones(len(a.TERMS)), torch.ones(len(a.TERMS)))
        torch.manual_seed(1)
        st = a.starts(buf)
        torch.manual_seed(2)
        _, steps = a.imagine(st)
        losses[key] = torch.stack([x[0] for x in steps]).sum(0).mean(0)  # (terms,)
    assert torch.equal(losses["plain"], losses["w1"]) and torch.equal(losses["plain"], losses["w3"])  # the terms themselves are unchanged
    assert not torch.equal(losses["plain"], losses["noise"])  # the object's velocities were perturbed in imagination
    w = agents["w3"].bal.w * agents["w3"].term_scale
    torch.testing.assert_close(w[4:], 3 * agents["plain"].bal.w[4:])


def test_supertrack_velocity_terms_and_task_penalties():
    """--st_vel_terms appends to SuperTrack's errors the L1 errors of the velocities HDMI's tracking group tracks, against
    CU(n-1)'s reference: its 14 keypoints' linear and angular velocities, its 19 joints' velocities. --st_task_penalties
    appends to the record minus HDMI's loco and feet terms but survival, which the loss adds as their mean over imagined
    steps and rollouts times n / full score, n the tracking terms (LAFAN1's SuperTrack rule; move_suitcase's full score is
    12). The first seven columns stay the default's, and with both off the agent and its record are the default ones. With
    both: --st_obj_w's scale covers the longer list, the new quantities' gradients reach the policy, and a policy update's
    loss is the calibrated tracking terms plus the weighted penalty."""
    from test_hdmi_task import JOINT_REWARD, VEL

    from ndm_gym.algos.hoi_algos import HOISuperTrack
    from ndm_gym.state import CONTROL_DT

    torch.manual_seed(0)
    env = fake_env(4)
    plain, both = small_agent(env), small_agent(env, vel_terms=True, task_penalties=True)
    assert plain.terms == HOISuperTrack.TERMS and plain.vel_idx is None and plain.penalties == [] and plain.bal.w.shape == (7,)
    assert both.terms == HOISuperTrack.TERMS + HOISuperTrack.VEL_TERMS and both.bal.w.shape == (10,) and both.w_pen == 10 / 12
    assert small_agent(env, task_penalties=True).w_pen == 7 / 12
    assert [v.tolist() for v in both.vel_idx] == [VEL, VEL, JOINT_REWARD]
    assert torch.equal(small_agent(env, obj_w=3.0, vel_terms=True).term_scale, torch.tensor([1.0] * 4 + [3.0] * 3 + [1.0] * 3))
    task = make_task("move_suitcase")
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    x = with_now(at_reference(task), body_lin_vel=r(4, 16, 3), body_ang_vel=r(4, 16, 3), joint_vel=r(4, 29))
    feet = torch.zeros(4, 4, 2, 3, dtype=torch.float64)
    feet[..., 2] = 300.0  # a touchdown: the impact force term
    x = x._replace(action_buf=r(4, NUM_ACTIONS, 3), joint_vel_sub=r(4, 2, 29), applied_torque=100 * r(4, 29), feet_force=feet)
    o = task.step(x)
    e0, e1 = plain._errors(x), both._errors(x)
    assert e1.shape == (4, 10) and torch.equal(e1[:, :7], e0)
    now, ref = x.now, x.ref
    torch.testing.assert_close(e1[:, 7], (now.body_lin_vel[:, VEL] - ref.body_lin_vel[:, 0, VEL]).abs().sum(-1).mean(-1))
    torch.testing.assert_close(e1[:, 8], (now.body_ang_vel[:, VEL] - ref.body_ang_vel[:, 0, VEL]).abs().sum(-1).mean(-1))
    torch.testing.assert_close(e1[:, 9], (now.joint_vel[:, JOINT_REWARD] - ref.joint_vel[:, 0, JOINT_REWARD]).abs().mean(-1))
    survival = sum(t.weight for grp in ("loco", "feet") for t in env.cfg.rewards[grp].values() if t.enabled and t.kind == "survival")
    pen = survival - (o.reward[:, 2] + o.reward[:, 3]) / CONTROL_DT
    assert survival == 2.0 and (pen > 0).all()
    torch.testing.assert_close(both._penalties(o), pen)
    args = (torch.zeros(4, NUM_ACTIONS, dtype=torch.float64), torch.zeros(4, dtype=torch.long), torch.zeros(4))
    rec, rec0 = both._record(o, x, *args), plain._record(o, x, *args)
    assert len(rec0) == 3 and len(rec) == 4 and torch.equal(rec[0], e1) and torch.equal(rec[3], both._penalties(o))
    # the new quantities' gradients through the imagined rollout, and a policy update's loss
    torch.nn.init.normal_(both.wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    both.wm_window, both.imag_step, both.imag_start = both._wm_window, both._imag_step, both._imag_start  # eager on CPU
    buf, _ = collect(env, both, 12)
    buf.window_keys = WINDOW_KEYS
    _, steps = both.imagine(both.starts(buf))
    for q in (torch.stack([s[0] for s in steps])[..., 7:], torch.stack([s[3] for s in steps])):
        grads = torch.autograd.grad(q.sum(), list(both.pi.parameters()), retain_graph=True, allow_unused=True)
        assert any(gr is not None and gr.abs().max() > 0 for gr in grads)
    assert both.wm_update(buf) is not None
    both.bal.calibrate(torch.ones(10), torch.ones(10))
    torch.manual_seed(5)
    _, steps = both.imagine(both.starts(buf))
    expected = torch.stack([s[0] for s in steps]).sum(0).mean(0).sum() + 10 / 12 * torch.stack([s[3] for s in steps]).mean()
    torch.manual_seed(5)
    out = both.policy_update(buf)  # st_fail_w and st_reg are 0
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values())
    torch.testing.assert_close(out["pi_loss"], expected.detach())


def eager(agent):
    """The agent's compiled functions as their eager originals (CPU tests)."""
    agent.wm_window, agent.imag_step, agent.imag_start = agent._wm_window, agent._imag_step, agent._imag_start
    agent.wm_smoothness = agent._wm_smoothness
    if hasattr(agent, "critic"):
        agent.critic_targets, agent.critic_loss = agent._critic_targets, agent._critic_loss
    return agent


def one_step(agent, st, seed: int = 5):
    """One imagined step (``_imag_step``) from the starts ``st``, with its random draws from ``seed``."""
    kp, sc = agent.kp_com_b[st["env"]], agent.obj_scale[st["env"]]
    torch.manual_seed(seed)
    prev, obs0 = agent._imag_start(st, kp, sc, agent.task.draw_noise(push=False))
    obs = {k: obs0[k] for k in agent.groups}
    return agent._imag_step(st["S"], st["So"], obs, st["t"], st["abuf"], st["afilt"], prev, st["count0"], st["ctx"], kp, sc, st["delay"], st["alpha"])


def test_supertrack_reference_replay():
    """--st_ref_replay tau (DiffMimic's demonstration replay): after an imagined step, exactly the rollouts whose keypoints'
    mean distance to CU(n-1)'s reference exceeds tau continue from the reference's state (reference_state: the pelvis's
    pose, its link velocity from the reference's centre-of-mass velocity, every joint clamped as HDMI's reset clamps it,
    the object at rest), with no gradient through it, their soft run lengths and the task's counters restarted and their
    histories filled with it; the others' next inputs are the default step's. The step returns the rollout's HDMIState
    and the reference's (without the contact forces), which imagine mixes into the next prev (mix_states). The record is
    the default step's, then the mask; a policy update reports the replaced fraction. 0 (and a run from before the
    option) is the default code path."""
    from ndm_gym.algos.hoi_algos import mix_states
    from ndm_gym.tasks.hdmi import HDMIState

    torch.manual_seed(0)
    env = fake_env(4)
    plain = eager(small_agent(env, B=16))
    assert plain.replay == 0 and eager(small_agent(env, B=16, st_ref_replay=0.0)).replay == 0
    torch.nn.init.normal_(plain.wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    buf, _ = collect(env, plain, 12)
    buf.window_keys = WINDOW_KEYS
    torch.manual_seed(3)
    st = plain.starts(buf)
    out0 = one_step(plain, st)
    f = st["t"].clamp(max=env.L - 1)
    dist = (out0[4].body_pos - env.clip.body_pos[f]).norm(dim=-1).mean(-1).detach()
    tau = float(dist.median())
    rep = eager(small_agent(env, B=16, st_ref_replay=tau))
    rep.wm.load_state_dict(plain.wm.state_dict()), rep.pi.load_state_dict(plain.pi.state_dict())
    torch.manual_seed(3)
    st = rep.starts(buf)
    out1 = one_step(rep, st)
    hit = out1[7][-1]
    assert torch.equal(hit, dist > tau) and 0 < int(hit.sum()) < 16
    assert len(out1[7]) == len(out0[7]) + 1 and all(torch.equal(a, b) for a, b in zip(out1[7][:-1], out0[7]))  # the step's losses
    kp = env.kp_com_b[st["env"]]
    Sr, Sor = rep.reference_state(st["t"], kp)
    keep = ~hit
    for x, y, r in zip(out1[0] + out1[1], out0[0] + out0[1], Sr + Sor):
        assert torch.equal(x[keep], y[keep]) and torch.equal(x[hit], r[hit])
    for g in rep.groups:  # the command group is the reference's alone; the others observe the replaced state
        assert torch.equal(out1[5][g][keep], out0[5][g][keep]) and torch.equal(out1[5][g][hit], out0[5][g][hit]) == (g == "command")
    assert torch.equal(out1[6][keep], out0[6][keep]) and (out1[6][hit] == 0).all()
    assert (rep.task.term_count[hit] == 0).all()
    # the reference state: the pelvis's centre-of-mass velocity is the reference's, the joints clamped, the object at rest
    rollout, ref = out1[4]
    assert all(torch.equal(a, b) for a, b in zip(rollout, out0[4]) if a is not None) and ref.eef_force is None
    now_r = rep.imag.state(Sr, Sor, rollout.eef_force, st["ctx"], kp, env.obj_scale[st["env"]])
    torch.testing.assert_close(now_r.body_lin_vel[:, 0], env.clip.body_lin_vel[f, 0])
    assert torch.equal(Sr[4], env.ref_joint_pos_all[f].clamp(-2.5, 2.0)) and torch.equal(Sr[5], env.ref_joint_vel_all[f].clamp(-30.0, 30.0))
    assert (Sr[4] == 2.0).any() and (Sr[5] == -30.0).any() and torch.equal(Sor[2], torch.zeros(16, 3))
    nxt = mix_states(hit, rollout, ref)  # the next step's prev
    assert isinstance(nxt, HDMIState) and nxt.eef_force is rollout.eef_force
    assert torch.equal(nxt.body_pos[hit], now_r.body_pos[hit]) and torch.equal(nxt.body_pos[keep], rollout.body_pos[keep])
    ang = rep.task.ang_vel_priv[hit]  # the histories filled with the replaced state, as HDMI's reset fills them
    assert torch.equal(ang, ang[:, :1].expand_as(ang))
    # no gradient through a replaced state; the others' reach the policy
    params = list(rep.pi.parameters())
    for rows, live in ((hit, False), (keep, True)):
        grads = torch.autograd.grad(out1[0][0][rows].sum(), params, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.abs().max() > 0 for g in grads) == live
    assert rep.wm_update(buf) is not None
    out = rep.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values()) and 0 <= float(out["replay_frac"]) <= 1


def replay_finite(agent, buf, frac: float):
    """With the agent's --st_ref_replay: more than ``frac`` of the rollouts are replaced after the imagined steps, the
    losses and their gradients to the policy are finite, and a policy update is not skipped."""
    torch.manual_seed(3)
    _, steps = agent.imagine(agent.starts(buf))
    hit = torch.stack([s[-1] for s in steps]).float()
    assert float(hit.mean()) > frac, float(hit.mean())
    E, p = torch.stack([s[0] for s in steps]), torch.stack([s[1] for s in steps])
    assert bool(torch.isfinite(E).all()) and bool(torch.isfinite(p).all())
    grads = torch.autograd.grad(E.sum() + p.sum(), list(agent.pi.parameters()), allow_unused=True)  # one backward: a compiled one may not be retained
    assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    out = agent.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values()) and float(out["replay_frac"]) > frac


@pytest.mark.parametrize("priv_obs", [False, True])
def test_supertrack_reference_replay_most_rollouts(priv_obs):
    """--st_ref_replay with a threshold that replaces nearly every rollout after every imagined step, with and without
    --priv_obs (whose privileged group observes the replaced state and the step before it): finite losses, gradients and
    policy update (replay_finite)."""
    torch.manual_seed(0)
    env = fake_env(4)
    agent = eager(small_agent(env, B=16, horizon=4, contact_force=True, priv_obs=priv_obs, st_ref_replay=1e-3))
    agent.args.st_fail_w = 5.0  # imag02's failure penalty
    torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    buf, _ = collect(env, agent, 12)
    buf.window_keys = WINDOW_KEYS
    assert agent.wm_update(buf) is not None
    replay_finite(agent, buf, 0.9)


@pytest.mark.slow
@pytest.mark.parametrize("priv_obs", [False, True])
def test_supertrack_reference_replay_compiled(priv_obs):
    """The imagination step compiled as in training (Inductor, on the CPU here) with --st_ref_replay replacing most
    rollouts, with and without --priv_obs: finite losses, gradients and policy update (replay_finite), and without random
    draws the compiled step's policy gradient is the eager step's. With the mixed next prev an output of the step's graph
    (before mix_states), Inductor's backward was non-finite here from the first update, as in training on GPUs."""
    from ndm_gym.algos.common import set_requires_grad

    torch.manual_seed(0)
    env = fake_env(4, task="move_stool-climb")
    agent = small_agent(env, B=16, horizon=4, contact_force=True, priv_obs=priv_obs, st_ref_replay=0.2)
    agent.args.st_fail_w = 5.0  # imag02's failure penalty
    agent.wm_window, agent.imag_start = agent._wm_window, agent._imag_start  # the imagination step stays compiled
    torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)
    buf, _ = collect(env, agent, 12)
    buf.window_keys = WINDOW_KEYS
    assert agent.wm_update(buf) is not None
    replay_finite(agent, buf, 0.5)
    agent.imag_noise, agent.task.noise = 0.0, False  # no random draws: the compiled and the eager step compute the same
    grads, compiled = [], agent.imag_step
    for step in (compiled, agent._imag_step):
        agent.imag_step = step
        torch.manual_seed(3)
        st = agent.starts(buf)
        set_requires_grad(agent.wm, False)
        _, steps = agent.imagine(st)
        loss = torch.stack([s[0] for s in steps]).sum() + torch.stack([s[1] for s in steps]).sum()
        grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss, list(agent.pi.parameters()))]))
        set_requires_grad(agent.wm, True)
    assert bool(torch.isfinite(grads[0]).all()) and float(grads[1].abs().max()) > 0
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-3, atol=1e-4 * float(grads[1].abs().max()))


def test_supertrack_late_weights():
    """--st_late_w w (LAFAN1's): the tracking terms of imagined step h weigh 1 + w h / (H - 1) before the calibration; 0
    (and a run from before the option) is the default."""
    torch.manual_seed(0)
    env = fake_env(4)
    plain, late = eager(small_agent(env, horizon=3)), eager(small_agent(env, horizon=3, st_late_w=2.0))
    assert plain.late is None and torch.equal(late.late, torch.tensor([1.0, 2.0, 3.0]))
    torch.nn.init.normal_(late.wm.net[-1].weight, std=0.05)
    buf, _ = collect(env, late, 12)
    buf.window_keys = WINDOW_KEYS
    late.bal.calibrate(torch.ones(7), torch.ones(7))
    torch.manual_seed(5)
    _, steps = late.imagine(late.starts(buf))
    expected = (torch.stack([s[0] for s in steps]) * late.late[:, None, None]).sum(0).mean(0).sum()
    torch.manual_seed(5)
    out = late.policy_update(buf)  # st_fail_w and st_reg are 0
    assert out.pop("skipped") is False
    torch.testing.assert_close(out["pi_loss"], expected.detach())


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_pi_rollout_clip(algo, monkeypatch):
    """--pi_rollout_clip k (LAFAN1's, either algorithm): 0 (and a run from before the option) and a cap no rollout reaches
    are the default update bit for bit; with k = 1 the policy gradient is sum_b w_b G_b, every rollout's part G_b its own
    vector-Jacobian product through the policy and w_b = min(1, k median / size_b), size_b the norm over the horizon of
    the gradient that reached the policy's output; the loss is unchanged; k < 0 is refused."""
    from ndm_gym.algos import hoi_algos
    from ndm_gym.algos.common import set_lr

    clip, seen = hoi_algos.clip_rollout_output_grads, []

    def spy(rerun, outs, k):
        seen.append((rerun, torch.stack([t.grad.clone() for t in outs])))
        return clip(rerun, outs, k)

    monkeypatch.setattr(hoi_algos, "clip_rollout_output_grads", spy)
    env = fake_env(4)
    out, grads = {}, {}
    for k in (None, 0.0, 1e9, 1.0):
        torch.manual_seed(0)
        agent = eager(small_agent(env, algo, B=16, **({} if k is None else {"pi_rollout_clip": k})))
        torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)
        buf, _ = collect(env, agent, 12)
        buf.window_keys = WINDOW_KEYS
        for o in agent.optimizers:
            set_lr(o, 0.0)
        params = list(agent.pi.parameters())
        raw = [None] * len(params)
        for i, p in enumerate(params):  # the gradient before the update's norm clipping
            p.register_post_accumulate_grad_hook(lambda p, i=i: raw.__setitem__(i, p.grad.detach().clone()))
        torch.manual_seed(2)
        out[k] = agent.policy_update(buf)
        grads[k] = torch.cat([g.flatten() for g in raw])
    assert len(seen) == 2  # only k = 1e9 and k = 1 tap the policy's output
    for k in (0.0, 1e9):
        assert torch.equal(grads[k], grads[None]) and all(torch.equal(out[k][n], out[None][n]) for n in out[None] if n != "skipped")
    assert torch.equal(out[1.0]["pi_loss"], out[None]["pi_loss"])  # the same forward pass
    rerun, A = seen[1]  # (H, B, 23)
    size = A.square().sum((0, 2)).sqrt()
    w = torch.clamp(size[size > 0].median() / size.clamp_min(1e-30), max=1.0)
    assert 0 < int((w < 1).sum()) < len(w)
    expect = 0
    for b in range(A.shape[1]):
        with torch.enable_grad():
            parts = torch.autograd.grad(rerun(torch.tensor([b])), params, A[:, b])
        expect = expect + w[b] * torch.cat([x.flatten() for x in parts])
    torch.testing.assert_close(grads[1.0], expect, rtol=1e-4, atol=1e-6 * float(expect.abs().max()))
    with pytest.raises(ValueError, match="pi_rollout_clip"):
        small_agent(env, algo, pi_rollout_clip=-1.0)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_world_model_input_noise_and_smoothness(algo):
    """LAFAN1's --wm_input_noise (noise on the model's normalized inputs while fitting) and --wm_smooth_w (the mean
    squared directional derivative of its network at the windows' first inputs, those the window loss keeps, against
    central differences in float64) for either algorithm; 0 (and a run from before them) is the default fit bit for bit
    without a random draw; the penalty changes the parameters' update but not the reported fit loss; a step below 0.01 is
    refused."""
    import copy

    from ndm_gym.state import unflatten_state

    torch.manual_seed(0)
    env = fake_env(4)
    plain = eager(small_agent(env, algo, B=16))
    torch.nn.init.normal_(plain.wm.net[-1].weight, std=0.05)
    buf, _ = collect(env, plain, 12)
    buf.window_keys = WINDOW_KEYS
    agents = {"plain": plain}
    for key, kw in {"zero": dict(wm_input_noise=0.0, wm_smooth_w=0.0, wm_smooth_eps=0.1, wm_smooth_dirs=1), "noise": dict(wm_input_noise=0.3),
                    "smooth": dict(wm_smooth_w=1.0, wm_smooth_eps=1e-2, wm_smooth_dirs=3)}.items():  # fmt: skip
        agents[key] = eager(small_agent(env, algo, B=16, **kw))
        agents[key].wm.load_state_dict(plain.wm.state_dict())
    torch.manual_seed(1)
    w = buf.sample_windows(16, 4)
    w["bad"][:3, 0] = True  # windows whose first step the guard flagged: left out of the penalty as of the loss
    losses, rng = {}, {}
    for key, a in agents.items():
        torch.manual_seed(2)
        losses[key], rng[key] = a._wm_window(w), torch.get_rng_state()
    assert torch.equal(losses["zero"], losses["plain"]) and torch.equal(rng["zero"], rng["plain"])
    assert not torch.equal(losses["noise"], losses["plain"]) and torch.equal(losses["smooth"], losses["plain"])
    sm = agents["smooth"]
    torch.manual_seed(5)
    pen = sm._wm_smoothness(w)
    torch.manual_seed(5)  # the same directions
    x = sm.wm.in_norm(sm.wm.inputs(unflatten_state(w["s"][:, 0]), sm.obj(w["o"][:, 0]), w["u"][:, 0], w["ctx"][:, 0])).detach().double()
    u = torch.randn((3,) + x.shape, dtype=torch.float32).double()
    u = u / u.norm(dim=-1, keepdim=True)
    net, h = copy.deepcopy(sm.wm.net).double(), 1e-4
    ju = torch.stack([(net(x + h * ui) - net(x - h * ui)) / (2 * h) for ui in u])[:, 3:]
    torch.testing.assert_close(pen.double(), (ju**2).sum(-1).mean(), rtol=2e-2, atol=0)
    fits, params = {}, {}
    for key in ("plain", "smooth"):
        a = agents[key]
        torch.manual_seed(7)
        fits[key] = a.wm_update(buf)
        params[key] = [p.detach().clone() for p in a.wm.parameters()]
    assert torch.equal(fits["smooth"], fits["plain"]) and any(not torch.equal(p, q) for p, q in zip(params["smooth"], params["plain"]))
    with pytest.raises(ValueError, match="wm_smooth_eps"):
        small_agent(env, algo, wm_smooth_w=1.0, wm_smooth_eps=1e-3, wm_smooth_dirs=1)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_privileged_observations(algo):
    """--priv_obs: the policy also reads HDMI's privileged group (either algorithm), through an encoder of it and the
    object group as HDMI's PPO teacher's, in imagination as in collection; its normalizer is fit with the others; a
    policy update's gradient reaches the encoder. Without the option the policy has no such parts."""
    from ndm_gym.policies.hoi_policy import PRIV_FEATURES

    torch.manual_seed(0)
    env = fake_env(4)
    plain = small_agent(env, algo, B=16)
    assert plain.groups == ("command", "policy", "object") and not hasattr(plain.pi, "priv_enc")
    agent = small_agent(env, algo, B=16, priv_obs=True)
    assert agent.groups == ("command", "policy", "object", "priv") and agent.pi.priv_enc[-1].out_features == PRIV_FEATURES
    from ndm_gym.algos.ppo_hoi import make_mlp

    dims = agent.task.obs_dims
    teacher = torch.nn.Sequential(*make_mlp(dims["priv"] + dims["object"], (PRIV_FEATURES,)), torch.nn.Linear(PRIV_FEATURES, PRIV_FEATURES))
    shape = lambda net: [(type(m), tuple(p.shape for p in m.parameters())) for m in net]  # noqa: E731
    assert shape(agent.pi.priv_enc) == shape(teacher)  # HDMI's teacher encoder, layer by layer
    torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    agent.wm_window, agent.imag_step, agent.imag_start = agent._wm_window, agent._imag_step, agent._imag_start  # eager on CPU
    agent.pi_norm_update = agent._pi_norm_update
    if algo == "supermimic":
        agent.critic_targets, agent.critic_loss = agent._critic_targets, agent._critic_loss
    buf, seen = collect(env, agent, 12)
    buf.window_keys = WINDOW_KEYS
    obs = {k: seen[3][k] for k in agent.groups}
    agent.update_pi_norms(obs)
    assert float(agent.pi.priv_norm.count) == 4 and float(agent.pi.command_norm.count) == 4
    assert agent.act(obs, env.t).shape == (4, NUM_ACTIONS)
    st = agent.starts(buf)
    obs0, steps = agent.imagine(st)
    assert "priv" in obs0 and obs0["priv"].shape == (16, agent.task.obs_dims["priv"])
    out = agent.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values())
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in agent.pi.priv_enc.parameters())


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="train_hoi imports Isaac Lab's launcher")
def test_train_hoi_refuses_the_options_it_does_not_implement():
    """Every argument of ndm_gym.train is either implemented by train_hoi or refused unless at its default."""
    from ndm_gym import train_hoi
    from ndm_gym.train import add_train_args

    p = argparse.ArgumentParser()
    add_train_args(p)
    actions = {a.dest: a for a in p._actions if a.dest != "help"}
    implemented = {"algo", "st_reg", "st_imag_noise", "st_fail_w", "num_envs", "iterations", "buffer_steps", "utd", "horizon", "wm_window",
                   "wm_hidden", "wm_act", "wm_no_ln", "wm_backbone", "rf_dim", "rf_layers", "rf_heads", "rf_registers", "wm_batches", "wm_batch",
                   "pi_batch", "lr", "wm_lr", "gamma", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
                   "wm_only_iters", "lr_decay_iters", "lr_decay_hours", "ckpt_every", "max_hours", "ckpt_hours", "log_every", "log_dir", "seed",
                   "resume", "amp", "st_late_w", "pi_rollout_clip", "wm_input_noise", "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs"}  # fmt: skip
    refused = set(train_hoi.UNSUPPORTED) | {"clips"}
    assert len(train_hoi.UNSUPPORTED) == 34 and not implemented & refused and implemented | refused == set(actions)
    base = ["--algo", "supertrack", "--task", "move_suitcase"]
    assert not train_hoi.parse_args(base).st_contact_force and train_hoi.parse_args(base + ["--st_contact_force"]).st_contact_force
    for extra in (["--st_contact_force"], ["--st_obj_w", "3"], ["--st_obj_imag_noise", "0.1"], ["--st_vel_terms"], ["--st_task_penalties"],
                  ["--st_late_w", "1"], ["--st_ref_replay", "0.4"]):  # fmt: skip
        with pytest.raises(SystemExit):
            train_hoi.parse_args(["--algo", "supermimic", "--task", "move_suitcase"] + extra)
    a = train_hoi.parse_args(base + ["--st_obj_w", "3", "--st_obj_imag_noise", "0.1"])
    assert a.st_obj_w == 3 and a.st_obj_imag_noise == 0.1 and not a.st_vel_terms and not a.st_task_penalties and a.st_ref_replay == 0
    a = train_hoi.parse_args(base + ["--st_vel_terms", "--st_task_penalties", "--st_ref_replay", "0.4", "--st_late_w", "1"])
    assert a.st_vel_terms and a.st_task_penalties and a.st_ref_replay == 0.4 and a.st_late_w == 1
    assert not a.priv_obs and train_hoi.parse_args(base + ["--priv_obs"]).priv_obs
    assert train_hoi.parse_args(["--algo", "supermimic", "--task", "move_suitcase", "--priv_obs"]).priv_obs
    sm = ["--algo", "supermimic", "--task", "move_suitcase"]
    a = train_hoi.parse_args(sm + ["--pi_rollout_clip", "3", "--wm_input_noise", "0.1", "--wm_smooth_w", "1"])  # both algorithms' options
    assert a.pi_rollout_clip == 3 and a.wm_input_noise == 0.1 and a.wm_smooth_w == 1
    for bad in (["--st_obj_w", "0"], ["--st_obj_imag_noise", "-1"], ["--st_ref_replay", "-0.1"]):
        with pytest.raises(SystemExit):
            train_hoi.parse_args(base + bad)
    for name in refused:
        a = actions[name]
        if a.nargs == 0:
            value = []
        elif a.choices:
            value = [next(c for c in a.choices if c != a.default)]
        elif a.type in (int, float):
            value = [str(a.default + 1)]
        else:
            value = ["elsewhere"]
        with pytest.raises(SystemExit):
            train_hoi.parse_args(base + [f"--{name}"] + value)


HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
def test_train_hoi_gpu_smoke(tmp_path):
    """SuperTrack on move_suitcase with 64 envs: four iterations with a checkpoint at a training time (--ckpt_hours) and
    the learning rate decayed over a fraction of a second (--lr_decay_hours); a new run in the same directory is refused;
    the resumed run continues the log without a second header, from last.pt (optimizers and loss weights included)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = tmp_path / "run"
    cmd = [sys.executable, "-m", "ndm_gym.train_hoi", "--task", "move_suitcase", "--algo", "supertrack", "--num_envs", "64", "--pi_batch", "256",
           "--wm_batch", "256", "--warmup_steps", "16", "--wm_only_iters", "1", "--log_every", "2", "--ckpt_every", "4", "--lr_decay_hours", "1e-4",
           "--log_dir", str(log)]  # fmt: skip
    run = lambda *a: subprocess.run(cmd + list(a), cwd=repo, capture_output=True, text=True, timeout=1800)  # noqa: E731
    r = run("--iterations", "4", "--ckpt_hours", "1e-6")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "checkpoint at 1e-06 h of training: it 1" in r.stdout
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == [2, 4] and rows[0]["nonfinite_envs"] == "0" and rows[1]["pi_skipped"] == "0"
    assert rows[1]["v_loss"] == "" and rows[1]["imag_r"] == "" and float(rows[1]["lr"]) == pytest.approx(0.03 * 2e-4)  # decayed to its floor
    ck = torch.load(log / "last.pt", map_location="cpu", weights_only=False)
    assert {"wm", "pi", "opt_wm", "opt_pi", "bal", "iteration"} <= set(ck) and ck["iteration"] == 4 and bool(ck["bal"]["ready"])
    assert (log / "ckpt_000001.pt").exists() and (log / "ckpt_000004.pt").exists()
    r = run("--iterations", "6")
    assert r.returncode != 0 and "--resume" in r.stderr
    r = run("--iterations", "6", "--resume", str(log / "last.pt"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lines = (log / "train_log.csv").read_text().strip().splitlines()
    assert sum(x.startswith("iteration,") for x in lines) == 1 and [x.split(",")[0] for x in lines[1:]] == ["2", "4", "6"]
    assert torch.load(log / "last.pt", map_location="cpu", weights_only=False)["iteration"] == 6
