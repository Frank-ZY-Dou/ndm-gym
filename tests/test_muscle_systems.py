"""Muscle training's speed options (ndm_gym.train_muscles; algos.muscle_algos), on CPU in eager mode: the updates written
without host synchronization (--no_sync, captured by --update_graphs) are the default updates bit for bit and skip a
non-finite update on the device; --update_graphs' device learning rates follow the schedule; --graph_early's draws have
randint's distribution; the policy average and Muon; train_muscles' handling of ndm_gym.train's speed options and
BlinkMimic's. With Newton, MuJoCo-Warp and MuscleVAE's converted assets (marked slow): on the CPU, train_muscles' iterations
with the systems options' code are the default ones bit for bit, the collection graph's body included; on a GPU, the
collection step as one CUDA graph plays the episodes of its eager run, and a short run with all the systems options
captures and replays its graphs."""

import argparse
import copy
import csv
import os
import subprocess
import sys

import pytest
import torch
from test_algo_equivalence import EXACT
from test_muscle_training import HAVE_DATA, HAVE_LAUNCHER, HAVE_SIM, REPO, muscle_replay, small_muscle_agent, synthetic_reference
from test_systems import LR_ARGS, _cumprod_loop, check_separate_learning_rates

from ndm_gym.core.normalizer import RunningNorm

SYSTEMS = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked", "--ckpt_light"]


@pytest.fixture()
def no_sync():
    """RunningNorm's device-side freeze test, as train_muscles --no_sync sets it."""
    RunningNorm.no_sync = True
    yield
    RunningNorm.no_sync = False


def tensors(*mods):
    return [t.detach().clone() for m in mods if m is not None for t in list(m.parameters()) + list(m.buffers())]


def opt_states(opt):
    return [v.clone() for st in opt.state.values() for v in st.values() if torch.is_tensor(v)]


def pair(algo: str, **kw):
    """Two agents with equal parameters, the second with --no_sync, on one synthetic reference, and a replay each."""
    ref = synthetic_reference()
    agents = []
    for no_sync in (False, True):
        torch.manual_seed(0)
        agents.append(small_muscle_agent(algo, ref, fused_adam=True, no_sync=no_sync, **kw))
    torch.nn.init.normal_(agents[0].wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    agents[1].wm.load_state_dict(agents[0].wm.state_dict())
    buf = muscle_replay(ref)
    return agents, (buf, copy.deepcopy(buf))


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_graphable_muscle_updates_are_the_default_ones(no_sync, monkeypatch, algo):
    """wm_update_graphable and policy_update_graphable give the default updates' values and parameters bit for bit (the
    same windows and starts, the optimizer steps gated on the device, the policy average's lerp); the default updates'
    cumprod is the graphable ones' loop here (equal up to rounding: test_systems)."""
    monkeypatch.setattr(torch, "cumprod", _cumprod_loop)
    (a0, a1), (b0, b1) = pair(algo, pi_ema=0.995)
    a0.args.st_fail_w = a1.args.st_fail_w = 5.0 if algo == "supertrack" else 0.0  # SuperTrack's survival products
    critic = lambda a: getattr(a, "critic", None)  # noqa: E731
    for step in range(2):  # the second policy update steps the policy average (the first copies it)
        torch.manual_seed(10 + step)
        r0 = a0.wm_update(b0)
        torch.manual_seed(10 + step)
        r1, ok = a1.wm_update_graphable(b1)
        assert bool(ok)
        torch.testing.assert_close(r1, r0, **EXACT)
        torch.manual_seed(20 + step)
        p0 = a0.policy_update(b0)
        a0.update_pi_avg()
        torch.manual_seed(20 + step)
        n_avg = int(a1.pi_avg.n_averaged)
        *p1, ok = a1.policy_update_graphable(b1)
        if int(a1.pi_avg.n_averaged) == n_avg:
            a1.pi_avg.n_averaged += 1  # as train_muscles does
        assert bool(ok) and p0.pop("skipped") is False and set(p0) == set(a1.graph_keys)
        for k, v in zip(a1.graph_keys, p1):
            torch.testing.assert_close(v, p0[k], **EXACT, msg=k)
    for x, y in zip(tensors(a0.wm, a0.pi, critic(a0), a0.pi_avg.module), tensors(a1.wm, a1.pi, critic(a1), a1.pi_avg.module)):
        torch.testing.assert_close(y, x, **EXACT)
    for o0, o1 in zip(a0.optimizers, a1.optimizers):
        for x, y in zip(opt_states(o0), opt_states(o1)):
            torch.testing.assert_close(y, x, **EXACT)
    assert a1.wm_skips.take() == (0, 0) and a1.pi_skips.take() == (0, 0) and int(a1.short_starts) == 0


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_nonfinite_muscle_updates_are_skipped_on_the_device(no_sync, algo):
    """--no_sync: a non-finite model update and a policy update whose imagined rollout is not finite leave the networks
    (SuperMimic: the critic and its normalizer too) and the optimizers' states unchanged, and are counted on the device."""
    (_, agent), (_, buf) = pair(algo)
    assert bool(agent.wm_update_graphable(buf)[1]) and bool(agent.policy_update_graphable(buf)[-1])
    snap = lambda: tensors(agent.wm, agent.pi, getattr(agent, "critic", None)) + [t for o in agent.optimizers for t in opt_states(o)]  # noqa: E731
    s0 = snap()
    good = buf.data["s1"].clone()
    buf.data["s1"][:] = float("nan")
    assert not bool(agent.wm_update_graphable(buf)[1]) and (agent.wm_skips.n, agent.wm_skips.run) == (1, 1)
    assert all(torch.equal(x, y) for x, y in zip(s0, snap()))
    buf.data["s1"].copy_(good)
    with torch.no_grad():
        agent.wm.net[0].weight.view(-1)[0] = float("nan")  # the imagined rollout's states
    s0 = snap()
    assert not bool(agent.policy_update_graphable(buf)[-1]) and agent.pi_skips.take() == (1, 1)
    for x, y in zip(s0, snap()):
        torch.testing.assert_close(y, x, rtol=0, atol=0, equal_nan=True)


def test_muscle_policy_average_and_muon():
    """--pi_ema with --pi_opt muon: Muon on the hidden matrices of the policy's encoder and decoder (the world model's,
    --wm_opt muon, as LAFAN1's: test_blinkmimic); checkpoints hold the average as "pi" and the trained policy as "pi_live",
    and a resumed agent continues both."""
    ref = synthetic_reference()
    torch.manual_seed(0)
    agent = small_muscle_agent("supertrack", ref, pi_ema=0.9, pi_opt="muon")
    torch.nn.init.normal_(agent.wm.net[-1].weight, std=0.05)
    buf = muscle_replay(ref)
    hidden = {id(lin.weight) for m in (agent.pi.enc, agent.pi.dec) for lin in [x for x in m.modules() if isinstance(x, torch.nn.Linear)][1:-1]}
    assert {id(p) for g in agent.opt_pi.muon.param_groups for p in g["params"]} == hidden
    for _ in range(2):
        assert agent.wm_update(buf) is not None
        assert agent.policy_update(buf).pop("skipped") is False
        agent.update_pi_avg()
    sd = agent.state_dict()
    live = agent.pi.state_dict()
    assert all(torch.equal(sd["pi_live"][k], live[k]) for k in live) and not all(torch.equal(sd["pi"][k], live[k]) for k in live)
    torch.manual_seed(1)
    other = small_muscle_agent("supertrack", ref, pi_ema=0.9, pi_opt="muon")
    other.load_state_dict(sd)
    assert all(torch.equal(other.pi.state_dict()[k], live[k]) for k in live) and int(other.pi_avg.n_averaged) == 2


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_update_graphs_learning_rates_follow_the_schedule(no_sync, algo):
    """--update_graphs: every optimizer's learning rate is a device tensor, which the captured updates read and
    train_muscles' schedule (algos.common.set_lr) overwrites in place: the tensors stay those built with the optimizers,
    and the graphable updates step as with plain learning rates set to the same values (powers of two, which the
    tensors hold exactly), bit for bit."""
    from ndm_gym.algos.common import set_lr

    ref = synthetic_reference()
    agents = []
    for graphs in (False, True):
        torch.manual_seed(0)
        agents.append(small_muscle_agent(algo, ref, fused_adam=True, no_sync=True, update_graphs=graphs))
    plain, graphed = agents
    torch.nn.init.normal_(plain.wm.net[-1].weight, std=0.05)
    graphed.wm.load_state_dict(plain.wm.state_dict())
    groups = [g for o in graphed.optimizers for g in o.param_groups]
    lrs = [g["lr"] for g in groups]
    assert all(torch.is_tensor(x) for x in lrs)
    bufs = (muscle_replay(ref), muscle_replay(ref))
    critic = lambda a: getattr(a, "critic", None)  # noqa: E731
    for step, lr in enumerate((2.0**-10, 2.0**-11, 2.0**-13)):
        for agent, buf in zip(agents, bufs):
            for o in agent.optimizers:
                set_lr(o, lr)
            torch.manual_seed(30 + step)
            assert bool(agent.wm_update_graphable(buf)[1]) and bool(agent.policy_update_graphable(buf)[-1])
        assert all(g["lr"] is x and float(x) == lr for g, x in zip(groups, lrs))
        for x, y in zip(tensors(plain.wm, plain.pi, critic(plain)), tensors(graphed.wm, graphed.pi, critic(graphed))):
            torch.testing.assert_close(y, x, **EXACT)


@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_update_graphs_learning_rates_are_each_optimizers_own(no_sync, algo):
    """--update_graphs with Adam for the world model and the policy (and SuperMimic's critic): each optimizer has its own
    device learning rate, so that train_muscles' --wm_lr sets the world model's alone and --pi_lr_mult the policy's alone,
    as without the option (test_systems.check_separate_learning_rates)."""
    ref = synthetic_reference()
    agents = []
    for graphs in (False, True):
        torch.manual_seed(0)
        agents.append(small_muscle_agent(algo, ref, fused_adam=True, no_sync=True, update_graphs=graphs, **LR_ARGS))
    torch.nn.init.normal_(agents[0].wm.net[-1].weight, std=0.05)  # zero at initialization: the imagined states would not depend on the actions
    agents[1].wm.load_state_dict(agents[0].wm.state_dict())
    check_separate_learning_rates(*agents, (muscle_replay(ref), muscle_replay(ref)))


def test_device_skip_counter_is_the_host_one():
    """DeviceSkippedUpdates counts what SkippedUpdates counts: the window's skipped updates and longest run, a run carrying
    over windows."""
    from ndm_gym.algos.hoi_guard import DeviceSkippedUpdates
    from ndm_gym.algos.muscle_guard import SkippedUpdates

    host, dev = SkippedUpdates(), DeviceSkippedUpdates("cpu")
    for window in ([1, 1, 0, 1, 1, 1, 0], [1, 1], [0, 0], [1, 0, 1, 1]):
        for s in window:
            host(bool(s)), dev(torch.tensor(bool(s)))
        assert dev.run == host.run and dev.take() == host.take()


def test_gated_running_norm_update():
    """RunningNorm.update's gate (--no_sync, the critic's normalizer): the update where it holds, nothing where it does not;
    refused without --no_sync."""
    torch.manual_seed(0)
    xs = [torch.randn(50, 7) * 3 + 1 for _ in range(3)]
    a, b = RunningNorm(7), RunningNorm(7)
    with pytest.raises(ValueError):
        b.update(xs[0], gate=torch.tensor(True))
    RunningNorm.no_sync = True
    try:
        for x in xs:
            a.update(x)
            b.update(x, gate=torch.tensor(True))
            b.update(torch.full_like(x, float("nan")), gate=torch.tensor(False))
    finally:
        RunningNorm.no_sync = False
    for k in ("mean", "var", "count"):
        torch.testing.assert_close(getattr(b, k), getattr(a, k), **EXACT)


def test_graph_early_draws():
    """--graph_early's draws of the window starts and the imagination starts' ages (ReplayBuffer._randint with the fill
    level on the device: min(floor(U n), n - 1), U uniform in [0, 1)) take the integers of [0, n) with randint's
    probabilities (Pearson's chi-square within six standard deviations of its mean) and nothing else."""
    from ndm_gym.buffer import ReplayBuffer

    buf = ReplayBuffer(4, 2, "cpu", fields={"x": ((), torch.float32)}, priorities=False)
    buf.use_device_ptr(device_size=True)
    torch.manual_seed(0)
    draws = 400_000
    for n in (1, 2, 7, 993):
        x = buf._randint(torch.tensor(n), draws)
        assert x.dtype == torch.long and int(x.min()) >= 0 and int(x.max()) < n
        if n > 1:
            expected = draws / n
            chi2 = float(((torch.bincount(x, minlength=n) - expected) ** 2).sum() / expected)
            assert chi2 < (n - 1) + 6 * (2 * (n - 1)) ** 0.5, (n, chi2)


@pytest.mark.skipif(not HAVE_LAUNCHER, reason="ndm_gym.train's arguments come with Isaac Lab's launcher")
def test_train_muscles_speed_options():
    """Every speed option of ndm_gym.train without the G1 RigidFormer's (add_speed_args) and BlinkMimic option is either
    implemented by train_muscles or refused unless at its default; the implemented ones' constraints."""
    from ndm_gym.algos.blinkmimic import add_blinkmimic_args
    from ndm_gym.train import add_speed_args
    from ndm_gym.train_muscles import SPEED_UNSUPPORTED, parse_args

    p = argparse.ArgumentParser()
    add_speed_args(p, rigidformer=False)
    add_blinkmimic_args(p)
    actions = {a.dest: a for a in p._actions if a.dest != "help"}
    implemented = {"fused_adam", "elementwise_3x3", "combo_kernels", "pi_ema", "pi_opt", "wm_opt", "pi_lr_mult", "lr_decay_len", "lr_decay_by",
                   "no_sync", "update_graphs", "graph_early", "collect_graph", "reset_masked", "ckpt_light", "last_every_min"}  # fmt: skip
    assert not implemented & set(SPEED_UNSUPPORTED) and implemented | set(SPEED_UNSUPPORTED) == set(actions)
    base = ["--algo", "supertrack"]
    a = parse_args(base + SYSTEMS + ["--pi_ema", "0.995", "--pi_opt", "muon", "--wm_opt", "muon", "--pi_lr_mult", "4", "--lr_decay_by", "1400"])
    assert a.update_graphs and a.graph_early and a.collect_graph and a.reset_masked and a.ckpt_light and a.pi_ema == 0.995 and a.lr_decay_by == 1400
    early = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early"]
    for bad in (["--no_sync"], ["--update_graphs", "--fused_adam"], ["--graph_early"], ["--no_sync", "--fused_adam", "--pi_rollout_clip", "3"],
                ["--lr_decay_by", "100"], ["--pi_ema", "1"], early + ["--collect_graph"], early[:3] + ["--collect_graph", "--reset_masked"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(base + bad)
    for name in SPEED_UNSUPPORTED:
        act = actions[name]
        value = [] if act.nargs == 0 else [next(c for c in act.choices if c != act.default)] if act.choices else (
            [str(act.default + 1)] if act.type in (int, float) else ["elsewhere"])  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(base + [f"--{name}"] + value)


@pytest.mark.skipif(not HAVE_LAUNCHER, reason="ndm_gym.train's arguments come with Isaac Lab's launcher")
def test_train_muscles_last_every_min(capsys):
    """train_muscles' --last_every_min: off (None) by default; minutes with --ckpt_light, for either algorithm; refused
    without --ckpt_light (every periodic checkpoint then writes the full last.pt) and at M <= 0. When a periodic checkpoint
    writes last.pt too: ndm_gym.train.last_due (test_systems)."""
    from ndm_gym.train_muscles import parse_args

    for algo in ("supertrack", "supermimic"):
        base = ["--algo", algo]
        assert parse_args(base + SYSTEMS).last_every_min is None
        assert parse_args(base + SYSTEMS + ["--last_every_min", "30"]).last_every_min == 30.0
        for bad, why in ((["--last_every_min", "30"], "needs --ckpt_light"), (["--ckpt_light", "--last_every_min", "0"], "must be > 0"),
                         (["--ckpt_light", "--last_every_min", "-1"], "must be > 0")):  # fmt: skip
            with pytest.raises(SystemExit):
                parse_args(base + bad)
            assert f"--last_every_min {why}" in capsys.readouterr().err


ACC_KEYS = ("r", "mpjpe", "n_ok", "rate", "n_rate", "ep_len", "done", "term", "bad")
SIM_BUFFERS = ("action", "length_target", "length_prev", "MA", "MR", "MF", "before", "terminated", "truncated", "reset_mask", "just_reset",
               "ref_frame", "ep_step", "off_count", "world_mask")  # fmt: skip


def loop_state(sim, agent, buf, acc: dict, a_prev) -> dict:
    """What train_muscles' iterations carry: the simulator's buffers and both Newton states, the replay, the networks with
    their normalizers, the optimizers' states, the statistics and the previous action."""
    out = {f"sim.{k}": getattr(sim, k) for k in SIM_BUFFERS}
    for i, v in enumerate(sim.views):
        out.update({f"sim.state{i}.{k}": x for k, x in v._asdict().items()})
    out.update({f"replay.{k}": x for k, x in buf.data.items()})
    for name in ("wm", "pi", "critic"):
        if getattr(agent, name, None) is not None:
            out.update({f"{name}.{k}": x for k, x in getattr(agent, name).state_dict().items()})
    for i, o in enumerate(agent.optimizers):
        out.update({f"opt{i}.{j}.{k}": x for j, st in enumerate(o.state.values()) for k, x in st.items() if torch.is_tensor(x)})
    return {**out, **{f"acc.{k}": x for k, x in acc.items()}, "a_prev": a_prev}


@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_SIM and HAVE_DATA), reason="needs Newton, MuJoCo-Warp and MuscleVAE's assets")
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_systems_iterations_are_the_default_ones_on_cpu(monkeypatch, algo):
    """train_muscles' iterations on the CPU, where MuscleSim is deterministic (8 environments, Euler ball joints, ODE
    damping, MuscleVAE's reference switches), with the default code and with the systems options' (--collect_graph's
    collection body: the control step's and the reset's bodies, the replay written at its device-side head; --no_sync's
    graphable updates and RunningNorm's device-side freeze test): 14 iterations of two collection steps, two model updates
    and, from the fourth, a policy update leave the same simulator states and buffers, replay, networks with their
    normalizers, optimizer states, statistics and random streams, and return the same update values, bit for bit. With
    SuperMimic's settings (action noise 0.3, reward widths x0.7, start kicks 0.3, two critic epochs of two minibatches;
    SuperTrack: --st_fail_w 5), episode ends (two truncated after 5 steps, terminations), a replay that wraps around and
    normalizers that reach their freeze count. Both learners use the fused Adam, the default updates' cumprod is the
    graphable ones' loop (equal up to rounding: test_systems), and the systems replay draws as the default's
    (--graph_early's draws: test_graph_early_draws). The CUDA graphs: test_graphed_muscle_collection_is_the_eager_one."""
    from ndm_gym.algos.muscle_algos import ACTION_UNIT, WINDOW_KEYS, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.robots.musclevae_char import NUM_MUSCLES
    from ndm_gym.sim.rigid import muscle_sim as ms
    from ndm_gym.tasks import muscle_tracking as mt
    from ndm_gym.train_muscles import collection_step, step_stats_fn

    monkeypatch.setattr(torch, "cumprod", _cumprod_loop)
    n, T, B, H = 8, 10, 8, 3
    sims = [ms.MuscleSim(n, "cpu", random_count=ms.RANDOM_COUNT, seed=0, ball_joints="euler", joint_damping="ode") for _ in range(2)]
    for sim in sims:  # two episodes truncate after 5 steps; two end at their first step off track
        sim.ep_step[:2].fill_(ms.MAX_LENGTH - 5)
        sim.ep_step[2:4].fill_(ms.MIN_LENGTH)
        sim.off_count[2:4].fill_(ms.ERR_LENGTH - 1)
    ref = mt.reference_from_sim(sims[0])
    agents = []
    for no_sync in (False, True):
        torch.manual_seed(0)
        agent = small_muscle_agent(algo, ref, B=B, horizon=H, fused_adam=True, no_sync=no_sync)
        agent.action_noise = ACTION_UNIT * 0.3
        if algo == "supermimic":
            agent.args.imag_start_perturb, agent.args.critic_epochs, agent.args.critic_minibatches = 0.3, 2, 2
            agent.reward_cfg = mt.MuscleRewardCfg().scaled(0.7)
            agent.critic.norm.max_count = 2.5 * (H + 1) * B  # reached at the third policy update
        else:
            agent.args.st_fail_w = 5.0
        agent.pi.obs_norm.max_count = agent.wm.in_norm.max_count = 100.0  # reached at the 13th collection step
        agents.append(agent)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # the world model's output layer starts at zero, the policy's near it: both active here
        w_wm = 0.05 * torch.randn(agents[0].wm.net[-1].weight.shape, generator=g)
        w_pi = 0.01 * torch.randn(agents[0].pi.dec[-1].weight.shape, generator=g)
        for agent in agents:
            agent.wm.net[-1].weight.copy_(w_wm), agent.pi.dec[-1].weight.copy_(w_pi)
    bufs = []
    for systems in (False, True):
        buf = ReplayBuffer(T, n, "cpu", fields=replay_fields(), priorities=False)
        buf.window_keys = WINDOW_KEYS
        if systems:
            buf.use_device_ptr(device_size=True)
            monkeypatch.setattr(buf, "_randint", lambda m, k: torch.randint(0, int(m), (k,)))
        bufs.append(buf)
    stats = step_stats_fn(ref)
    accs = [{k: torch.zeros(()) for k in ACC_KEYS} for _ in range(2)]
    a_prevs = [torch.zeros(n, NUM_MUSCLES) for _ in range(2)]
    torch.manual_seed(1)
    rng, outs = [torch.get_rng_state()] * 2, ([], [])
    for it in range(14):
        for p, (sim, agent, buf, acc, a_prev) in enumerate(zip(sims, agents, bufs, accs, a_prevs)):
            torch.set_rng_state(rng[p])  # each path continues its own stream of the default generator
            RunningNorm.no_sync = p == 1
            try:
                with torch.no_grad():
                    for _ in range(2):
                        collection_step(sim, agent, buf, stats, acc, a_prev, captured=p == 1)
                        if p == 1:
                            buf.advance_host()
                if buf.size >= agent.args.wm_window:
                    for _ in range(2):
                        outs[p].append(agent.wm_update(buf) if p == 0 else agent.wm_update_graphable(buf)[0])
                if it >= 3 and p == 0:
                    out = agent.policy_update(buf)
                    agent.update_pi_avg()
                    assert out.pop("skipped") is False
                    outs[p].extend(out[k] for k in agent.graph_keys)
                elif it >= 3:
                    *vals, ok = agent.policy_update_graphable(buf)
                    assert bool(ok)
                    outs[p].extend(vals)
            finally:
                RunningNorm.no_sync = False
            rng[p] = torch.get_rng_state()
    assert float(accs[0]["done"]) >= 4 and float(accs[0]["term"]) >= 1, "too few episode ends"
    assert float(agents[0].pi.obs_norm.count) >= 100.0
    assert algo == "supertrack" or float(agents[0].critic.norm.count) >= agents[0].critic.norm.max_count
    assert len(outs[0]) == len(outs[1]) > 0
    for i, (x, y) in enumerate(zip(*outs)):
        torch.testing.assert_close(y, x, **EXACT, msg=f"update value {i}")
    s0, s1 = loop_state(sims[0], agents[0], bufs[0], accs[0], a_prevs[0]), loop_state(sims[1], agents[1], bufs[1], accs[1], a_prevs[1])
    assert s0.keys() == s1.keys()
    for k in s0:
        torch.testing.assert_close(s1[k], s0[k], **EXACT, equal_nan=True, msg=k)
    assert torch.equal(rng[0], rng[1]) and torch.equal(sims[0].gen.get_state(), sims[1].gen.get_state())
    assert (bufs[1].ptr, bufs[1].size) == (bufs[0].ptr, bufs[0].size) == (int(bufs[1].ptr_t), int(bufs[1].size_t))
    assert agents[1].wm_skips.take() == (0, 0) and agents[1].pi_skips.take() == (0, 0) and int(agents[1].short_starts) == 0


@pytest.mark.slow
@pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_SIM and HAVE_DATA and HAVE_LAUNCHER), reason="needs a GPU, Newton and the assets")
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_train_muscles_systems_options(tmp_path, algo):
    """A short train_muscles run with the systems options and the policy average: the collection step, the model and policy
    updates are captured and replayed, none is skipped, the periodic checkpoints hold the policy average only
    (--last_every_min, here at each of them: the full last.pt too) and the last one is complete."""
    log = tmp_path / "run"
    cmd = [sys.executable, "-m", "ndm_gym.train_muscles", "--algo", algo, "--num_envs", "64", "--pi_batch", "256", "--wm_batch", "256",
           "--warmup_steps", "16", "--wm_only_iters", "1", "--log_every", "2", "--ckpt_every", "4", "--iterations", "8", "--pi_ema", "0.995",
           *SYSTEMS, "--last_every_min", "1e-6", "--log_dir", str(log)]  # fmt: skip
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for what in ("one collection step", "the world-model update", "the policy update"):
        assert f"captured {what}" in r.stdout, what
    assert "last.pt at it 4 (" in r.stdout
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == [2, 4, 6, 8] and all(x["wm_skipped"] == "0" and x["pi_skipped"] == "0" for x in rows)
    assert rows[-1]["ep_len"] != "", "no episode ended"  # MuscleVAE's first episodes end after about 32 steps
    light = torch.load(log / "ckpt_000004.pt", map_location="cpu", weights_only=False)
    assert set(light) == {"pi", "iteration", "wall_time", "sim_steps", "args", "saved_at"} and light["iteration"] == 4
    last = torch.load(log / "last.pt", map_location="cpu", weights_only=False)
    assert last["iteration"] == 8 and {"wm", "opt_pi", "pi_live", "pi_avg"} <= set(last)


def worker_collect(out: str, mode: str):
    """MuscleSim (16 envs, the training loop's settings) without its own step graph, fixed random actions: 24 collection
    steps of train_muscles.collection_step's captured body (the control step's and the reset's bodies, the replay's
    device-side head), run eagerly (``body``) or as one CUDA graph (``graph``: MuscleSim.capture, its warm-up the first
    step, as train_muscles --collect_graph runs it). The resets' random frames and fatigue come from the simulator's
    generator, registered with the graph. Four episodes start 5 steps before MuscleVAE's length limit, so that they end
    within the 24 steps (none terminates before MIN_LENGTH steps) and the captured reset runs."""
    from types import SimpleNamespace

    sys.path.insert(0, REPO)
    from ndm_gym.algos.muscle_algos import WINDOW_KEYS, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.robots.musclevae_char import NUM_MUSCLES
    from ndm_gym.sim.rigid import muscle_sim as ms
    from ndm_gym.tasks import muscle_tracking as mt
    from ndm_gym.train_muscles import collection_step, step_stats_fn

    torch.manual_seed(0)
    dev, n, steps = "cuda:0", 16, 24
    sim = ms.MuscleSim(n, dev, random_count=ms.RANDOM_COUNT, seed=0, ball_joints="euler", joint_damping="ode", capture_step=False)
    sim.ep_step[:4].fill_(ms.MAX_LENGTH - 5)
    ref = mt.reference_from_sim(sim)
    g = torch.Generator().manual_seed(1)
    acts = (0.3 * torch.randn(steps, n, NUM_MUSCLES, generator=g)).to(dev)
    a = torch.zeros(n, NUM_MUSCLES, device=dev)
    algo = SimpleNamespace(update_pi_norms=lambda o, f: None, act=lambda o, f: (a, a), action_noise=0.0, update_wm_norms=lambda tr: None)
    buf = ReplayBuffer(steps, n, dev, fields=replay_fields(), priorities=False)
    buf.window_keys = WINDOW_KEYS
    buf.use_device_ptr(device_size=True)
    stats = step_stats_fn(ref)
    acc = {k: torch.zeros((), device=dev) for k in ("r", "mpjpe", "n_ok", "rate", "n_rate", "ep_len", "done", "term", "bad")}
    a_prev = torch.zeros(n, NUM_MUSCLES, device=dev)
    rec = {k: [] for k in ("ref_frame", "ep_step", "off_count", "acc")}
    graph = None
    with torch.no_grad():
        for t in range(steps):
            a.copy_(acts[t])
            if mode == "body":
                with sim._on_torch_stream():
                    collection_step(sim, algo, buf, stats, acc, a_prev, captured=True)
            elif graph is None:
                graph = sim.capture(lambda: collection_step(sim, algo, buf, stats, acc, a_prev, captured=True))
            else:
                graph.replay()
            buf.advance_host()
            rec["ref_frame"].append(sim.ref_frame.cpu()), rec["ep_step"].append(sim.ep_step.cpu()), rec["off_count"].append(sim.off_count.cpu())
            rec["acc"].append(torch.stack(list(acc.values())).cpu())
    rec = {k: torch.stack(v) for k, v in rec.items()}
    rec.update({f"buf_{k}": v.cpu() for k, v in buf.data.items()})
    torch.save(rec, out)


@pytest.mark.slow
@pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_SIM and HAVE_DATA), reason="needs a GPU, Newton and the assets")
def test_graphed_muscle_collection_is_the_eager_one(tmp_path):
    """--collect_graph: train_muscles.collection_step's body as one CUDA graph plays the episodes of its eager run: the same
    reference frames, episode steps, off counts, flags and replay rows' discrete fields, the states within the simulator's
    own run-to-run differences across processes (MuJoCo-Warp's atomics; measured by a second eager run). A contact's
    activation can turn a rounding-sized difference into a large one in a single episode (a change of 1e-6 in the actions
    alone moved one of the 16 episodes by 0.35 in 3 of 6 eager runs), so the states are compared by the median episode and
    by the episodes that reset."""
    recs = {}
    for mode in ("body", "body2", "graph"):
        out = tmp_path / f"{mode}.pt"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", mode.rstrip("2"), "--out", str(out)], cwd=REPO,
                           capture_output=True, text=True, timeout=1800)  # fmt: skip
        err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr
        assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
        recs[mode] = torch.load(out)
    a = recs["body"]
    assert bool(a["buf_done"].any()), "no episode ended"
    own = max(float((recs["body2"][k] - a[k]).abs().max()) for k in ("buf_s", "buf_s1"))
    b = recs["graph"]
    for k in ("ref_frame", "ep_step", "off_count", "buf_done", "buf_term", "buf_bad", "buf_t", "buf_ep_step", "buf_off"):
        assert torch.equal(b[k], a[k]), k
    for k in ("buf_s", "buf_s1"):
        diff = (b[k] - a[k]).abs().flatten(2).max(2).values.max(0).values  # each episode's largest
        own_ep = (recs["body2"][k] - a[k]).abs().flatten(2).max(2).values.max(0).values
        assert diff.median() <= 4 * own_ep.median() + 1e-5, (k, diff, own_ep)
        assert diff[:4].max() <= 4 * own + 1e-5, (k, diff[:4], own)  # worker_collect's four episodes that end and reset
    for k in ("buf_fat", "buf_fat1", "buf_a"):
        diff = float((b[k] - a[k]).abs().max())
        assert diff <= 4 * own + 1e-5, (k, diff, own)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["body", "graph"], required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    worker_collect(cli.out, cli.worker)
