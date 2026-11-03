"""Training on MuscleVAE's muscle-actuated character (ndm_gym.train_muscles): SuperTrack's local terms (algos.muscle_algos)
against algos.supertrack's on random states; every LAFAN1 training argument is either implemented or refused; with the converted assets and Newton, on two CPU environments of MuscleSim, the reference tables
against the simulator's reset, the reference action and MuscleVAE's termination rule against the simulator's own after a
step; on a GPU, a few iterations of each algorithm, a resumed run and an evaluation of its checkpoint."""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from ndm_gym.robots import musclevae_char as mc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_DATA = all(os.path.isfile(p) for p in (mc.MJCF_PATH, mc.CHARACTER_PATH, mc.MUSCLES_PATH)) and os.path.isdir(mc.CLIPS_DIR)
HAVE_SIM = all(importlib.util.find_spec(m) is not None for m in ("newton", "mujoco_warp"))
HAVE_LAUNCHER = importlib.util.find_spec("isaaclab") is not None  # ndm_gym.train's arguments come with Isaac Lab's launcher
# the arguments muscle training implements; the rest of ndm_gym.train's are train_muscles.UNSUPPORTED
SUPPORTED = ("algo", "st_reg", "st_imag_noise", "st_balance_w", "imag_start_perturb", "st_fail_w", "num_envs", "iterations", "buffer_steps", "utd",
             "horizon", "wm_window", "wm_hidden", "wm_act", "wm_no_ln", "wm_batches", "wm_batch", "pi_batch", "lr", "wm_lr", "gamma",
             "reward_std_scale", "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps", "wm_only_iters", "lr_decay_iters",
             "lr_decay_hours", "ckpt_every", "max_hours", "ckpt_hours", "log_every", "log_dir", "seed", "resume", "amp", "wm_backbone", "rf_dim",
             "rf_layers", "rf_heads", "rf_registers", "rf_no_tree_bias", "rf_no_rope", "st_late_w", "pi_rollout_clip", "wm_input_noise",
             "wm_smooth_w", "wm_smooth_eps", "wm_smooth_dirs")  # fmt: skip


def test_local_terms_match_supertrack():
    """SuperTrack's Local(X) with the products elementwise equals algos.supertrack.local_features, and is invariant to a
    turn and shift of the scene; the world-model terms vanish at the true state."""
    from ndm_gym.algos.muscle_algos import local_terms, wm_terms
    from ndm_gym.algos.supertrack import local_features

    from test_muscle_worldmodels import random_state, turn

    S, F = random_state(4, 5)
    for x, y in zip(local_terms(S), local_features(*S)):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)
    for x, y in zip(local_terms(turn(S, -0.7, (1.0, 5.0))), local_terms(S)):  # root-relative, heights and up: invariant
        torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)
    assert wm_terms(S, F, S, F).abs().max() < 1e-12
    S2, F2 = random_state(4, 6)
    assert (wm_terms(S2, F2, S, F) > 0).all()


def synthetic_reference(F: int = 60, seed: int = 0):
    """A MuscleReference of F frames of random bodies (one clip), for tests without MuscleVAE's assets."""
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.tasks import muscle_tracking as mt

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    p = 0.3 * r(F, mt.NB, 3) + torch.tensor([0.0, 0.0, 1.0])
    return mt.MuscleReference(p, so3_exp(r(F, mt.NB, 3)), r(F, mt.NB, 3), r(F, mt.NB, 3), 1.5 + 0.1 * r(F), 0.1 * r(F, mc.NUM_MUSCLES),
                              torch.full((F,), F - 1))  # fmt: skip


def small_muscle_agent(algo: str, ref, B: int = 8, horizon: int = 3, **opts):
    """A CPU learner with a small world model and its compiled functions eager; ``opts``: further training arguments."""
    from types import SimpleNamespace

    from ndm_gym.algos.muscle_algos import MUSCLE_ALGOS

    args = SimpleNamespace(wm_hidden=[32], wm_act="silu", wm_no_ln=False, wm_backbone="mlp", lr=1e-3, pi_batch=B, horizon=horizon, action_noise=0.1,
                           imag_start_perturb=0.0, wm_batch=16, wm_window=4, st_imag_noise=0.0, st_balance_w=1.0, st_reg=0.0, st_fail_w=0.0,
                           reward_std_scale=1.0, gamma=0.99, lam=0.95, critic_epochs=1, critic_minibatches=1, **opts)  # fmt: skip
    agent = MUSCLE_ALGOS[algo](args, ref, "cpu")
    agent.wm_window, agent.imag_step, agent.wm_smoothness = agent._wm_window, agent._imag_step, agent._wm_smoothness
    agent.pi_norm_update, agent.wm_norm_update = agent._pi_norm_update, agent._wm_norm_update
    if algo == "supermimic":
        agent.critic_input_c = agent.critic_input
    return agent


def muscle_replay(ref, n: int = 4, steps: int = 12, seed: int = 0):
    """A replay of random transitions about the reference's frames (no flags), as train_muscles stores them."""
    from ndm_gym.algos.muscle_algos import WINDOW_KEYS, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.tasks import muscle_tracking as mt

    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731

    def near(f):  # bodies about the reference's frames f
        return (ref.body_p[f] + 0.05 * r(n, mt.NB, 3), so3_exp(0.1 * r(n, mt.NB, 3)) @ ref.body_R[f], ref.body_v[f] + 0.1 * r(n, mt.NB, 3),
                ref.body_w[f] + 0.1 * r(n, mt.NB, 3))  # fmt: skip

    buf = ReplayBuffer(steps, n, "cpu", fields=replay_fields(), priorities=False)
    buf.window_keys = WINDOW_KEYS
    t0, fat, no = torch.randint(0, 20, (n,), generator=g), torch.rand(n, mt.FATIGUE_DIM, generator=g), torch.zeros(n, dtype=torch.bool)
    for k in range(steps):
        f, fat1 = t0 + k, torch.rand(n, mt.FATIGUE_DIM, generator=g)
        buf.add({"s": mt.flatten_bodies(near(f)), "s1": mt.flatten_bodies(near(f + 1)), "fat": fat, "fat1": fat1, "a": 0.1 * r(n, mc.NUM_MUSCLES),
                 "t": f, "ep_step": torch.full((n,), k, dtype=torch.int32), "off": torch.zeros(n, dtype=torch.int32), "done": no, "term": no,
                 "bad": no})  # fmt: skip
        fat = fat1
    return buf


def test_muscle_reference_replay():
    """--st_ref_replay tau (SuperTrack): after an imagined step, exactly the rollouts whose bodies' mean distance to the
    reference frame re-anchored to their roots exceeds tau continue from that anchored reference state, with no gradient
    through it and their off counts restarted; the muscle state, the termination probability and the record are the
    default step's (the record ends with the mask); a policy update reports the replaced fraction. 0 is the default."""
    from ndm_gym.tasks import muscle_tracking as mt

    torch.manual_seed(0)
    ref = synthetic_reference()
    plain = small_muscle_agent("supertrack", ref)
    assert plain.replay == 0 and small_muscle_agent("supertrack", ref, st_ref_replay=0.0).replay == 0
    torch.nn.init.normal_(plain.wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    buf = muscle_replay(ref)
    step = lambda a, st: a._imag_step(st["S"], st["F"], st["t"], st["count"] + 3, st["ep"], st["a_prev"], st["has_prev"])  # noqa: E731
    torch.manual_seed(3)
    st = plain.starts(buf)
    out0 = step(plain, st)
    anchored = mt.anchored_reference(tuple(x.detach() for x in out0[0]), ref, st["t"])
    dist = (out0[0][0].detach() - anchored[0]).norm(dim=-1).mean(-1)
    tau = float(dist.median())
    rep = small_muscle_agent("supertrack", ref, st_ref_replay=tau)
    rep.wm.load_state_dict(plain.wm.state_dict()), rep.pi.load_state_dict(plain.pi.state_dict())
    torch.manual_seed(3)
    out1 = step(rep, rep.starts(buf))
    hit, keep = out1[4][-1], ~out1[4][-1]
    assert torch.equal(hit, dist > tau) and 0 < int(hit.sum()) < 8
    assert len(out1[4]) == len(out0[4]) + 1 and all(torch.equal(a, b) for a, b in zip(out1[4][:-1], out0[4]))
    for x, y, r in zip(out1[0], out0[0], anchored):
        assert torch.equal(x[keep], y[keep]) and torch.equal(x[hit], r[hit])
    assert torch.equal(out1[1], out0[1]) and torch.equal(out1[3], out0[3]) and torch.equal(out1[5], out0[5])
    assert torch.equal(out1[2][keep], out0[2][keep]) and (out1[2][hit] == 0).all() and (out0[2][hit] > 0).all()
    params = list(rep.pi.parameters())
    for rows, live in ((hit, False), (keep, True)):
        grads = torch.autograd.grad(out1[0][0][rows].sum(), params, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.abs().max() > 0 for g in grads) == live
    assert rep.wm_update(buf) is not None
    out = rep.policy_update(buf)
    assert out.pop("skipped") is False and all(torch.isfinite(v) for v in out.values()) and 0 <= float(out["replay_frac"]) <= 1


def test_muscle_late_weights():
    """--st_late_w w (SuperTrack): the tracking terms of imagined step h weigh 1 + w h / (H - 1) before the calibration."""
    torch.manual_seed(0)
    ref = synthetic_reference()
    assert small_muscle_agent("supertrack", ref).late is None
    agent = small_muscle_agent("supertrack", ref, st_late_w=2.0)
    assert torch.equal(agent.late, torch.tensor([1.0, 2.0, 3.0]))
    torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)
    buf = muscle_replay(ref)
    agent.bal_pi.calibrate(torch.ones(6), torch.ones(6))
    torch.manual_seed(5)
    steps, _ = agent.imagine(agent.starts(buf))
    E, rate = torch.stack([s[4][0] for s in steps]), torch.stack([s[4][2] for s in steps])
    expected = (E * agent.late[:, None, None]).sum(0).mean(0).sum() + agent.w_rate * rate.mean()
    torch.manual_seed(5)
    out = agent.policy_update(buf)  # st_fail_w and st_reg are 0, st_balance_w 1
    assert out.pop("skipped") is False
    torch.testing.assert_close(out["pi_loss"], expected.detach())


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_muscle_pi_rollout_clip(algo, monkeypatch):
    """--pi_rollout_clip k (either algorithm, LAFAN1's): 0 and a cap no rollout reaches are the default update bit for
    bit; with k = 1 the policy gradient is sum_b w_b G_b (brute force, as on LAFAN1); k < 0 is refused."""
    from ndm_gym.algos import muscle_algos
    from ndm_gym.algos.common import set_lr

    clip, seen = muscle_algos.clip_rollout_policy_grads, []

    def spy(pi, taps, k):
        seen.append([(o.detach().clone(), g.detach().clone(), t.grad.clone()) for o, g, t in taps])
        return clip(pi, taps, k)

    monkeypatch.setattr(muscle_algos, "clip_rollout_policy_grads", spy)
    ref = synthetic_reference()
    out, grads = {}, {}
    for k in (None, 0.0, 1e9, 1.0):
        torch.manual_seed(0)
        agent = small_muscle_agent(algo, ref, **({} if k is None else {"pi_rollout_clip": k}))
        torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)
        buf = muscle_replay(ref)
        for o in agent.optimizers:
            set_lr(o, 0.0)
        params = list(agent.pi.parameters())
        raw = [None] * len(params)
        for i, p in enumerate(params):
            p.register_post_accumulate_grad_hook(lambda p, i=i: raw.__setitem__(i, p.grad.detach().clone()))
        torch.manual_seed(2)
        out[k] = agent.policy_update(buf)
        grads[k] = torch.cat([g.flatten() for g in raw])
    assert len(seen) == 2
    for k in (0.0, 1e9):
        assert torch.equal(grads[k], grads[None]) and all(torch.equal(out[k][n], out[None][n]) for n in out[None] if n != "skipped")
    assert torch.equal(out[1.0]["pi_loss"], out[None]["pi_loss"])
    obs, goal, A = (torch.stack(x) for x in zip(*seen[1]))  # (H, B, ...)
    size = A.square().sum((0, 2)).sqrt()
    w = torch.clamp(size[size > 0].median() / size.clamp_min(1e-30), max=1.0)
    assert 0 < int((w < 1).sum()) < len(w)
    expect = 0
    for b in range(A.shape[1]):
        with torch.enable_grad():
            parts = torch.autograd.grad(agent.pi(obs[:, b], goal[:, b]), params, A[:, b])
        expect = expect + w[b] * torch.cat([x.flatten() for x in parts])
    torch.testing.assert_close(grads[1.0], expect, rtol=1e-4, atol=1e-6 * float(expect.abs().max()))
    with pytest.raises(ValueError, match="pi_rollout_clip"):
        small_muscle_agent(algo, ref, pi_rollout_clip=-1.0)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_muscle_world_model_input_noise_and_smoothness(algo):
    """LAFAN1's --wm_input_noise and --wm_smooth_w for either algorithm: 0 is the default fit bit for bit without a random
    draw; noise changes the window loss; the penalty is the mean squared directional derivative at the windows' first
    inputs that the loss keeps (central differences in float64), and changes the update but not the reported fit loss."""
    import copy

    from ndm_gym.tasks import muscle_tracking as mt

    torch.manual_seed(0)
    ref = synthetic_reference()
    plain = small_muscle_agent(algo, ref)
    torch.nn.init.normal_(plain.wm.net[-1].weight, std=0.05)
    buf = muscle_replay(ref)
    agents = {"plain": plain}
    for key, kw in {"zero": dict(wm_input_noise=0.0, wm_smooth_w=0.0, wm_smooth_eps=0.1, wm_smooth_dirs=1), "noise": dict(wm_input_noise=0.3),
                    "smooth": dict(wm_smooth_w=1.0, wm_smooth_eps=1e-2, wm_smooth_dirs=3)}.items():  # fmt: skip
        agents[key] = small_muscle_agent(algo, ref, **kw)
        agents[key].wm.load_state_dict(plain.wm.state_dict())
    for a in agents.values():  # the same normalizers, fit on the replay
        a.update_wm_norms({k: v.flatten(0, 1) for k, v in buf.data.items()})
    torch.manual_seed(1)
    w = buf.sample_windows(16, 4)
    w["bad"][:3, 0] = True
    losses, rng = {}, {}
    for key, a in agents.items():
        torch.manual_seed(2)
        losses[key], rng[key] = a._wm_window(w), torch.get_rng_state()
    assert torch.equal(losses["zero"], losses["plain"]) and torch.equal(rng["zero"], rng["plain"])
    assert not torch.equal(losses["noise"], losses["plain"]) and torch.equal(losses["smooth"], losses["plain"])
    sm = agents["smooth"]
    torch.manual_seed(5)
    pen = sm._wm_smoothness(w)
    torch.manual_seed(5)
    x = sm.wm.in_norm(sm.wm.inputs(mt.unflatten_bodies(w["s"][:, 0]), w["fat"][:, 0], w["a"][:, 0])).detach().double()
    u = torch.randn((3,) + x.shape, dtype=torch.float32).double()
    u = u / u.norm(dim=-1, keepdim=True)
    net, h = copy.deepcopy(sm.wm.net).double(), 1e-4
    ju = torch.stack([(net(x + h * ui) - net(x - h * ui)) / (2 * h) for ui in u])[:, 3:]
    torch.testing.assert_close(pen.double(), (ju**2).sum(-1).mean(), rtol=2e-2, atol=0)
    fits, params = {}, {}
    for key in ("plain", "smooth"):
        torch.manual_seed(7)
        fits[key] = agents[key].wm_update(buf)
        params[key] = [p.detach().clone() for p in agents[key].wm.parameters()]
    assert torch.equal(fits["smooth"], fits["plain"]) and any(not torch.equal(p, q) for p, q in zip(params["smooth"], params["plain"]))


@pytest.mark.skipif(not HAVE_LAUNCHER, reason="ndm_gym.train imports Isaac Lab's launcher")
def test_arguments():
    import argparse

    from ndm_gym.train import add_train_args
    from ndm_gym.train_muscles import UNSUPPORTED, parse_args

    p = argparse.ArgumentParser()
    add_train_args(p)
    names = {a.dest for a in p._actions if a.dest != "help"}
    assert names == set(SUPPORTED) | set(UNSUPPORTED) and not set(SUPPORTED) & set(UNSUPPORTED)
    a = parse_args(["--algo", "supertrack"])
    assert a.ball_joints == "euler" and a.joint_damping == "ode" and a.mv_clips == ["lafan_jump", "lafan_run", "lafan_walk"]
    for bad in (["--st_com"], ["--hist", "4"], ["--rf_rope_detach"], ["--clips", "x.txt"], ["--rf_dim", "64"],
                ["--wm_backbone", "rigidformer", "--wm_hidden", "512"], ["--rf_muscle_tokens", "body"],
                ["--wm_backbone", "rigidformer", "--rf_muscle_dim", "16"], ["--wm_backbone", "rigidformer", "--rf_fused_features"],
                ["--wm_backbone", "rigidformer", "--rf_muscle_tokens", "body", "--rf_fused_train"], ["--rf_fused_features"],
                ["--st_late_w", "1"], ["--st_ref_replay", "0.3"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(["--algo", "supermimic"] + bad)
    with pytest.raises(SystemExit):
        parse_args(["--algo", "supertrack", "--st_ref_replay", "-0.1"])
    a = parse_args(["--algo", "supertrack", "--st_ref_replay", "0.3", "--st_late_w", "1", "--pi_rollout_clip", "3", "--wm_smooth_w", "1"])
    assert a.st_ref_replay == 0.3 and a.st_late_w == 1 and a.pi_rollout_clip == 3 and a.wm_smooth_w == 1
    assert parse_args(["--algo", "supertrack"]).st_ref_replay == 0
    assert parse_args(["--algo", "supermimic", "--pi_rollout_clip", "3", "--wm_input_noise", "0.1"]).wm_input_noise == 0.1  # both algorithms' options
    a = parse_args(["--algo", "supertrack", "--wm_backbone", "rigidformer", "--rf_dim", "64", "--muscle_model", "paper"])
    from ndm_gym.worldmodels.muscle import rigidformer_kwargs

    assert rigidformer_kwargs(a) == dict(muscle_model="paper", dim=64, layers=3, heads=4, registers=2, tree_bias=True, rope=True)
    assert rigidformer_kwargs(parse_args(["--algo", "supertrack"])) is None
    a = parse_args(["--algo", "supermimic", "--wm_backbone", "rigidformer", "--rf_muscle_tokens", "body", "--rf_muscle_dim", "16"])
    assert rigidformer_kwargs(a) == dict(muscle_model="released", dim=128, layers=3, heads=4, registers=2, tree_bias=True, rope=True,
                                         tokens="body", muscle_dim=16)  # fmt: skip
    del a.rf_muscle_tokens, a.rf_muscle_dim  # a run from before the option: muscle tokens
    assert "tokens" not in rigidformer_kwargs(a)
    a = parse_args(["--algo", "supertrack", "--wm_backbone", "rigidformer", "--rf_muscle_tokens", "body", "--rf_registers", "9", "--rf_fused_train",
                    "--rf_fused_features"])  # fmt: skip
    kw = rigidformer_kwargs(a)
    assert kw["registers"] == 9 and kw["fused_train"] and kw["fused_features"]


needs_cpu_sim = pytest.mark.skipif(not (HAVE_DATA and HAVE_SIM), reason="needs MuscleVAE's assets (tools/download_musclevae.sh, "
                                                                        "tools/convert_musclevae.py) and Newton")  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    from ndm_gym.sim.rigid.muscle_sim import MuscleSim

    return MuscleSim(2, "cpu", clips=("lafan_walk",), seed=0, ball_joints="euler", joint_damping="ode")


@needs_cpu_sim
def test_reference_and_rule_against_the_simulator(sim):
    """A reset starts on the reference tables' bodies (forward kinematics, raised) with their velocities; the reference
    action is the simulator's; after steps of the data policy, MuscleVAE's rule on our state (tasks.muscle_tracking)
    gives the simulator's off counts and flags."""
    from ndm_gym.sim.rigid import muscle_sim as ms
    from ndm_gym.tasks import muscle_tracking as mt

    assert (ms.ERR_THRESHOLD, ms.ERR_LENGTH, ms.MIN_LENGTH, ms.OBS_LIMIT) == (mt.ERR_THRESHOLD, mt.ERR_LENGTH, mt.MIN_LENGTH, mt.OBS_LIMIT)
    ref = mt.reference_from_sim(sim)
    frames = torch.tensor([300, 2500])
    sim.reset(frames=frames)
    v = sim.views[0]
    S = mt.unflatten_bodies(mt.body_states(v.body_q, v.body_qd, sim.before, sim.just_reset))
    torch.testing.assert_close(S[0], ref.body_p[frames], rtol=0, atol=3e-5)
    torch.testing.assert_close(S[1], ref.body_R[frames], rtol=0, atol=3e-5)
    # the reset's joint velocities (backward differences) give body velocities close to the bodies' backward differences
    dv = (S[2] - ref.body_v[frames]).norm(dim=-1)
    assert dv[:, 0].max() < 1e-4 and dv.mean() < 0.1 and dv.max() < 0.6
    torch.testing.assert_close(ref.action[sim.ref_frame], sim.reference_action())
    assert torch.equal(ref.clip_last, torch.full_like(ref.clip_last, len(sim.ref.clip) - 1))
    count = sim.off_count.clone()
    for _ in range(40):
        f, ep = sim.ref_frame.clone(), sim.ep_step.clone()
        terminated, _ = sim.step(sim.data_policy(0.05))
        S1 = mt.unflatten_bodies(mt.body_states(v.body_q, v.body_qd, sim.before, sim.just_reset))
        term, count = mt.termination(S1, ref.head_z[f], count, ep)
        assert torch.equal(term, terminated) and torch.equal(count, sim.off_count)
        if terminated.any():
            break
    sim.reset()


@pytest.mark.slow
@pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_DATA and HAVE_SIM and HAVE_LAUNCHER), reason="needs a GPU, MuscleVAE's assets, Newton")
def test_training_smoke(tmp_path):
    """A few iterations of each algorithm (world-model-only first), its log and checkpoints; SuperMimic resumed from its
    last.pt for two more; an evaluation of the last checkpoint."""
    small = ["--num_envs", "64", "--warmup_steps", "16", "--buffer_steps", "64", "--wm_batch", "64", "--pi_batch", "64", "--horizon", "8",
             "--wm_batches", "2", "--wm_only_iters", "2", "--log_every", "2", "--ckpt_every", "2"]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": REPO}

    def run(*args):
        out = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        return out.stdout

    for algo, extra in (("supertrack", ["--st_fail_w", "5", "--st_balance_w", "4"]),
                        ("supermimic", ["--reward_std_scale", "0.7", "--action_noise", "0.3", "--imag_start_perturb", "0.3"])):  # fmt: skip
        d = str(tmp_path / algo)
        run("-m", "ndm_gym.train_muscles", "--algo", algo, *extra, *small, "--iterations", "4", "--log_dir", d)
        lines = open(os.path.join(d, "train_log.csv")).read().splitlines()
        assert len(lines) == 3 and lines[0].startswith("iteration,wall_h") and lines[2].split(",")[0] == "4"
        assert all(os.path.isfile(os.path.join(d, f)) for f in ("last.pt", "ckpt_000002.pt", "ckpt_000004.pt", "args.json"))
    d = str(tmp_path / "supermimic")
    run("-m", "ndm_gym.train_muscles", "--algo", "supermimic", *small, "--iterations", "6", "--log_dir", d, "--resume", os.path.join(d, "last.pt"))
    lines = open(os.path.join(d, "train_log.csv")).read().splitlines()
    assert len(lines) == 4 and lines[3].split(",")[0] == "6" and lines.count(lines[0]) == 1
    out = run("tools/evaluate_muscles.py", "--ckpt", os.path.join(d, "ckpt_000006.pt"), "--num_envs", "12", "--steps", "16")
    assert "lafan_walk" in out and os.path.isfile(os.path.join(d, "eval_ckpt_000006.json"))
