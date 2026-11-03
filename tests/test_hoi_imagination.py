"""HOI imagination (ndm_gym/algos/hoi.py, and the starts and terminations of ndm_gym/algos/hoi_algos.py) on CPU: a
hinged object's contact body placed by its joint frames, against poses the simulator reported (HDMISim with Newton at
hinge angle 0.5 rad; the frames are Newton's joint_X_p, joint_X_c and joint_axis of the door and the folding chair); the
terminations against HDMI's counter, near the threshold too; imagination starts against the observations of a replay
collected as train_hoi collects it. The full step input is checked against the simulator on GPU."""

import pytest
import torch

from ndm_gym.robots.g1 import NUM_J
from ndm_gym.tasks.hdmi import HDMIClip, NUM_ACTIONS
from ndm_gym.tasks.hdmi_tasks import TASKS
from ndm_gym.worldmodels.hoi import ContextLayout

S2 = 0.5**0.5
HINGES = {  # frames as HDMISim.hinge_frames returns them; the simulator's contact body relative to the base at q = 0.5
    "open_door-feet": ({"p_p": [0.0, 0.5, 0.0], "R_p": torch.eye(3).tolist(), "axis": [0.0, 0.0, 1.0], "p_c": [0.0] * 3, "R_c": torch.eye(3).tolist()},
                       [0.0, 0.5, 0.0], [0.8776, -0.4794, 0.0]),
    "open_foldchair-sit": ({"p_p": [0.0] * 3, "R_p": [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], "axis": [0.0, 1.0, 0.0],
                            "p_c": [0.0] * 3, "R_c": torch.eye(3).tolist()}, [0.0, 0.0, 0.0], [0.4794, 0.0, -0.8776]),
}  # fmt: skip


def _clip(T=10):
    z = lambda *s: torch.zeros(T, *s)  # noqa: E731
    return HDMIClip(z(16, 3), torch.eye(3).expand(T, 16, 3, 3), z(16, 3), z(16, 3), z(NUM_ACTIONS), z(NUM_ACTIONS), z(3), torch.eye(3).expand(T, 3, 3),
                    torch.zeros(T, 1, dtype=torch.bool), torch.zeros(T))  # fmt: skip


@pytest.mark.parametrize("task", sorted(HINGES))
def test_hinged_contact_body(task):
    from ndm_gym.algos.hoi import HOIImagination
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.tasks.hdmi import JOINT_NAMES

    cfg = TASKS[task]
    frames, p_rel, R_row0 = HINGES[task]
    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    consts = {"eef_bodies": list(cfg.contact_eef_bodies), "q0": [0.0] * NUM_J, "act_scale": [0.5] * NUM_ACTIONS,
              "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip
    lay = ContextLayout(3, True, 0)
    imag = HOIImagination(cfg, _clip(), consts, frames, lay, "cpu")
    yaw = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # a base turned by 90 degrees and shifted
    base_p = torch.tensor([1.0, 2.0, 0.3])
    ctx = torch.cat((torch.zeros(lay.scalars), base_p, yaw.flatten())).unsqueeze(0)
    p, R = imag.contact_body((torch.tensor([[0.5]]), torch.zeros(1, 1)), ctx)
    torch.testing.assert_close(p[0], base_p + yaw @ torch.tensor(p_rel), atol=1e-4, rtol=0)
    torch.testing.assert_close((yaw.T @ R[0])[0], torch.tensor(R_row0), atol=1e-4, rtol=0)
    with pytest.raises(ValueError):
        HOIImagination(cfg, _clip(), consts, None, lay, "cpu")  # a hinged task needs its frames


def hdmi_counter(errors: torch.Tensor, thr: torch.Tensor, min_steps: torch.Tensor, count0: torch.Tensor):
    """HDMI's termination rule (TRM:12-32) over a sequence of errors (steps, N, terms): the counts and whether the env
    terminates, per step."""
    count, counts, fired = count0.clone(), [], []
    for e in errors:
        count = torch.where(e >= thr, count + 1, 0)
        counts.append(count), fired.append((count >= min_steps).any(-1))
    return torch.stack(counts), torch.stack(fired)


def test_soft_terminations_follow_hdmis_consecutive_rule():
    """The rule is HDMI's: a term fires once its error has stayed at or over the threshold for min_steps consecutive
    steps, counting on from the recorded run length; a step under the threshold resets it. The probability is within
    5e-5 of HDMI's 0 or 1 at every step, and the error's gradient is the exceedance sigmoid's."""
    from ndm_gym.algos.hoi_algos import soft_terminations

    thr, min_steps = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 1.0])
    over, under = torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 0.0]])
    count, fired = torch.zeros(1, 2), []
    for e in (over, over, over, under, over):
        d, count = soft_terminations(e, thr, min_steps, count)
        fired.append(float(d))
    assert all(abs(f - g) < 1e-4 for f, g in zip(fired, [0.0, 0.0, 1.0, 0.0, 0.0])), fired
    d, _ = soft_terminations(over, thr, min_steps, torch.tensor([[2.0, 0.0]]))  # recorded run of 2: the third step fires
    assert float(d) > 1 - 1e-4
    d, _ = soft_terminations(torch.tensor([[0.0, 5.0]]), thr, min_steps, torch.zeros(1, 2))  # min_steps 1 fires at once
    assert float(d) > 1 - 1e-4
    d, _ = soft_terminations(torch.tensor([[1.0, 0.0]]), thr, min_steps, torch.tensor([[2.0, 0.0]]))  # at the threshold: >= (TRM:24)
    assert float(d) > 1 - 1e-4
    e = torch.tensor([[1.02, 0.0]], requires_grad=True)  # the exceedance's gradient reaches the error through the count
    d, c = soft_terminations(e, thr, min_steps, torch.tensor([[1.0, 0.0]]))
    (c[0, 0] + d).backward()
    assert float(c.detach()[0, 0]) == 2.0 and float(d.detach()) < 1e-4 and e.grad[0, 0] > 0


@pytest.mark.parametrize("factor", [1.1, 1.3, 1.5, 2.0])
def test_soft_terminations_near_the_threshold(factor):
    """Errors sustained just over the threshold terminate after min_steps (25) steps, as HDMI's counter does: the old
    soft exceedance's geometric decay (a sigmoid times count + 1) never fired up to 1.3 times the threshold and needed
    27 steps at 1.5 times. A run of 20 steps at 1.1 times, a step under the threshold, then 25 more: HDMI's rule fires at
    the last step only."""
    from ndm_gym.algos.hoi_algos import soft_terminations

    thr, min_steps = torch.tensor([0.5, 1.2]), torch.tensor([25.0, 25.0])
    sustained = torch.tensor([factor * 0.5, 0.0]).expand(40, 1, 2)
    run = torch.cat((torch.tensor([0.55, 0.0]).expand(20, 1, 2), torch.tensor([0.4, 0.0]).expand(1, 1, 2), torch.tensor([0.55, 0.0]).expand(25, 1, 2)))
    for errors in (sustained, run):
        counts, fired = hdmi_counter(errors, thr, min_steps, torch.zeros(1, 2))
        count, d = torch.zeros(1, 2), []
        for e, c_ref in zip(errors, counts):
            p, count = soft_terminations(e, thr, min_steps, count)
            assert torch.equal(count, c_ref.float())  # HDMI's count exactly
            d.append(p)
        d = torch.cat(d)
        torch.testing.assert_close(d, fired.flatten().float(), atol=1e-4, rtol=0)
    assert int(hdmi_counter(sustained, thr, min_steps, torch.zeros(1, 2))[1].flatten().nonzero()[0]) == 24  # the 25th step
    assert hdmi_counter(run, thr, min_steps, torch.zeros(1, 2))[1].flatten().nonzero().flatten().tolist() == [45]


def fake_env(n: int, T: int = 400, task: str = "move_suitcase", seed: int = 0):
    """What HOIAgent reads of HDMISim, for a random clip of T frames; the simulator's joint order is G1's here."""
    from types import SimpleNamespace

    from test_hdmi_task import LIMITS

    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.sim.rigid.hdmi_sim import action_scale, default_pose
    from ndm_gym.tasks.hdmi import JOINT_NAMES

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    cfg = TASKS[task]
    clip = HDMIClip(r(T, 16, 3), so3_exp(r(T, 16, 3)), r(T, 16, 3), r(T, 16, 3), 0.3 * r(T, NUM_ACTIONS), r(T, NUM_ACTIONS), r(T, 3), so3_exp(r(T, 3)),
                    torch.rand(T, cfg.contact_cols, generator=g) < 0.5, None)  # fmt: skip
    j_hdmi = torch.tensor([G1_JOINT_NAMES.index(n) for n in JOINT_NAMES])
    # the reset's tables (--st_ref_replay): every joint's reference in G1 order (some beyond the soft limits), the limits
    wave = torch.sin(0.1 * torch.arange(T)[:, None] + torch.arange(NUM_J)[None])
    return SimpleNamespace(cfg=cfg, clip=clip, L=T, hinged=False, context_layout=(5, False, len(cfg.extra_objects)), joint_limits=LIMITS.float(),
                           robot_mass=33.341, q0=torch.tensor(default_pose(cfg)), j_act=j_hdmi[:NUM_ACTIONS], j_hdmi=j_hdmi,
                           scale=torch.tensor([action_scale(n) for n in JOINT_NAMES[:NUM_ACTIONS]]), kp_com_b=0.05 * r(n, 16, 3),
                           obj_scale=torch.ones(n, 3), N=n, device="cpu", future_steps=torch.tensor((1, 2, 8, 16, 32)),
                           ref_joint_pos_all=3.0 * wave, ref_joint_vel_all=40.0 * wave.roll(1, 0), soft_lo=torch.full((NUM_J,), -2.5),
                           soft_hi=torch.full((NUM_J,), 2.0), vel_limit=torch.full((NUM_J,), 30.0), j_g1=torch.arange(NUM_J), hinge_lim=None,
                           ref_obj_joint_vel=None)  # fmt: skip


def small_agent(env, algo: str = "supertrack", B: int = 64, horizon: int = 2, contact_force: bool = False, obj_w: float = 1.0,
                obj_imag_noise: float = 0.0, priv_obs: bool = False, vel_terms: bool = False, task_penalties: bool = False, **lafan1):  # fmt: skip
    """``lafan1``: options of ndm_gym.train's that HOI training implements (st_late_w, st_ref_replay, pi_rollout_clip,
    wm_input_noise, wm_smooth_w with wm_smooth_eps and wm_smooth_dirs), set only if given, as a run from before them."""
    from types import SimpleNamespace

    from ndm_gym.algos.hoi_algos import HOI_ALGOS

    args = SimpleNamespace(wm_backbone="mlp", wm_hidden=[32], wm_act="silu", wm_no_ln=False, pi_batch=B, lr=1e-3, horizon=horizon, st_imag_noise=0.0,
                           wm_batch=16, wm_window=4, st_fail_w=0.0, st_reg=0.0, gamma=0.99, lam=0.95, critic_epochs=1, critic_minibatches=1,
                           st_contact_force=contact_force, st_obj_w=obj_w, st_obj_imag_noise=obj_imag_noise, priv_obs=priv_obs,
                           st_vel_terms=vel_terms, st_task_penalties=task_penalties, **lafan1)  # fmt: skip
    return HOI_ALGOS[algo](args, env, "cpu")


def collect(env, agent, steps: int, seed: int = 0):
    """A replay of ``steps`` control steps as train_hoi collects it, from a stand-in simulator that follows HDMISim's
    conventions (random states and forces; resets through HDMISim.reset_input) and whose step inputs between resets are
    the imagination's (``HOIImagination.step_input``); HDMITask without noise. Returns the replay and, per row and env,
    the observation groups the policy acted on."""
    from ndm_gym.algos.hoi_algos import GUARD_FIELDS, task_fields, task_record
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim
    from ndm_gym.state import OBJ_DIM, STATE_DIM, flatten_object, flatten_state, unflatten_object, unflatten_state
    from ndm_gym.tasks.hdmi import DECIMATION, HDMITask, action_pipeline
    from ndm_gym.wmbench.collect_hoi import merge_states

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    n, imag, lay = env.N, agent.imag, agent.layout
    task = HDMITask(env.cfg, n, env.joint_limits, env.robot_mass, noise=False)
    f32 = torch.float32
    fields = {"s": (STATE_DIM, f32), "s1": (STATE_DIM, f32), "o": (OBJ_DIM, f32), "o1": (OBJ_DIM, f32), "u": ((DECIMATION, NUM_ACTIONS), f32),
              "a": (NUM_ACTIONS, f32), "ctx": (lay.dim, f32), "f_eef": ((env.cfg.num_eefs, 3), f32), "f_feet": ((DECIMATION, 2, 3), f32),
              "q_obs": (29, f32), "t": ((), torch.long), "ep_step": ((), torch.int32), "done": ((), torch.bool), "term": ((), torch.bool),
              **task_fields(len(task.term_names)), **GUARD_FIELDS}  # fmt: skip
    buf = ReplayBuffer(steps, n, "cpu", fields=fields)
    rt = lambda S: unflatten_state(flatten_state(S))  # noqa: E731  what the replay stores
    rto = lambda So: unflatten_object(flatten_object(So))  # noqa: E731

    def draw(k):  # a random state of k envs: robot (G1 order), object, contexts' reset part
        S = (r(k, 3), so3_exp(0.3 * r(k, 3)), r(k, 3), r(k, 3), 0.3 * r(k, NUM_J), r(k, NUM_J))
        return S, (r(k, 3), so3_exp(r(k, 3)), r(k, 3), r(k, 3)), 0.01 * r(k, NUM_J), torch.randint(2, 7, (k,), generator=g), 0.8 + 0.2 * torch.rand(k, 1, generator=g)

    def context(off, delay, alpha):
        return torch.cat((torch.ones(n, lay.startup), off, delay.unsqueeze(-1).float(), alpha, torch.zeros(n, 2)), -1)

    S, So, off, delay, alpha = draw(n)
    S, So = rt(S), rto(So)
    env.offset, env.t = off, torch.randint(0, env.L - 40, (n,), generator=g)
    ep_step, abuf, afilt, f_eef = torch.zeros(n, dtype=torch.int32), torch.zeros(n, NUM_ACTIONS, 3), torch.zeros(n, NUM_ACTIONS), torch.zeros(n, 2, 3)
    ctx = context(off, delay, alpha)
    state = imag.state(S, So, f_eef, ctx, env.kp_com_b, env.obj_scale)
    task.reset(torch.arange(n), state)
    obs = task.observations(HDMISim.reset_input(env, state))
    seen = []
    for k in range(steps):
        seen.append(obs)
        before, rec = (flatten_state(S), flatten_object(So), ctx, env.t.clone(), ep_step.clone()), task_record(task)
        a = r(n, NUM_ACTIONS)
        u, abuf, afilt = action_pipeline(a, abuf, afilt, delay, alpha)
        S1, So1 = rt((S[0] + 0.02 * r(n, 3), so3_exp(0.05 * r(n, 3)) @ S[1], r(n, 3), r(n, 3), S[4] + 0.05 * r(n, NUM_J), r(n, NUM_J))), rto(So)
        f_eef, f_feet = r(n, 2, 3), r(n, DECIMATION, 2, 3)
        now = imag.state(S1, So1, f_eef, ctx, env.kp_com_b, env.obj_scale)
        x = imag.step_input(now, state, env.t, u, abuf, ctx, f_feet)
        o = task.step(x)
        done = (torch.arange(n) + k) % 7 == 6  # resets at different steps of different envs
        buf.add({"s": before[0], "s1": flatten_state(S1), "o": before[1], "o1": flatten_object(So1), "u": u, "a": a, "ctx": before[2],
                 "f_eef": f_eef, "f_feet": f_feet, "q_obs": x.joint_pos_sub.mean(1), "t": before[3], "ep_step": before[4], "done": done,
                 "term": torch.zeros(n, dtype=torch.bool), "bad": torch.zeros(n, dtype=torch.bool), **rec})  # fmt: skip
        S, So, state, obs = S1, So1, now, o.obs
        env.t, ep_step = env.t + 1, ep_step + 1
        ids = done.nonzero().flatten()
        S0, So0, off0, delay0, alpha0 = draw(len(ids))
        S0, So0 = rt(S0), rto(So0)
        m = lambda new, old: old.index_put((ids,), new)  # noqa: E731
        S, So = tuple(m(v, w) for v, w in zip(S0, S)), tuple(m(v, w) for v, w in zip(So0, So))
        off, delay, alpha = m(off0, off), m(delay0, delay), m(alpha0, alpha)
        env.offset, env.t = off, m(torch.randint(0, env.L - 40, (len(ids),), generator=g), env.t)
        ep_step, abuf, afilt = m(torch.zeros((), dtype=torch.int32), ep_step), m(torch.zeros(()), abuf), m(torch.zeros(()), afilt)
        f_eef = m(torch.zeros(()), f_eef)
        ctx = context(off, delay, alpha)
        state = merge_states(done, state, imag.state(S, So, f_eef, ctx, env.kp_com_b, env.obj_scale))
        task.reset(ids, state)
        fresh = task.observations(HDMISim.reset_input(env, state))
        obs = {k: torch.where(done.unsqueeze(-1), fresh[k], v) for k, v in obs.items()}
    return buf, seen


def test_imagination_starts_match_collection():
    """A start's observation groups are those the policy acted on at its replay row, without noise: mid-episode, from
    the step that ended at the row (reference windows of its frame counter t - 1); at an episode's first row, from the
    reset (HDMISim.reset_input: windows of t, no actuator torque); the histories, actions, counters and the task's
    noise setting rebuilt from the replay."""
    torch.manual_seed(0)
    env = fake_env(8)
    agent = small_agent(env, B=256)
    buf, seen = collect(env, agent, 30)
    agent.task.noise = False
    st = agent.starts(buf)
    _, obs0 = agent._imag_start(st, env.kp_com_b[st["env"]], env.obj_scale[st["env"]], None)
    first = ~st["v1"]
    assert first.sum() > 10 and (~first).sum() > 10  # both kinds of rows among the starts
    for g in ("command", "policy", "object", "priv"):
        want = torch.stack([seen[int(r)][g][int(e)] for r, e in zip(st["row"], st["env"])])
        torch.testing.assert_close(obs0[g], want, atol=2e-4, rtol=1e-4, msg=lambda m, g=g: f"{g}: {m}")
