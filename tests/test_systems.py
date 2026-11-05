"""BlinkMimic's systems options (train.py's shared loop; docs/blinkmimic.md): the updates written for CUDA graphs
(--update_graphs) are the default updates, the device-side replay head (--graph_early, --collect_graph), the normalizers'
device-side freeze test (--no_sync), the weighted bin statistics of the masked resets (--reset_masked) and the gated Muon
step, on CPU in eager mode; when --ckpt_light's periodic checkpoints also write the full last.pt (--last_every_min). The
simulator's tests (marked slow; a GPU and Isaac Lab, in processes of this file or of train.py): the masked training step
equals the indexed one, and a training run with all the options captures and replays its graphs."""

import argparse
import csv
import importlib.util
import os
import sys

import pytest
import torch

from conftest import DEV, REPO, synthetic_clips
from test_algo_equivalence import EXACT, _fill
from test_bptt_tricks import _algo
from test_lafan1_training import TRAIN, run, write_motions

HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None
SYSTEMS = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked", "--ckpt_light"]

CLOSE = dict(rtol=1e-5, atol=1e-7)


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _cumprod_loop(x, dim):
    """cumprod as the graphable updates compute the survival products: a loop of products."""
    acc, out = torch.ones_like(x.select(dim, 0)), []
    for t in range(x.shape[dim]):
        acc = acc * x.select(dim, t)
        out.append(acc)
    return torch.stack(out, dim)


def test_survival_products_loop_is_cumprod():
    """The graphable updates' survival products (a loop, since cumprod's backward tests for zeros on the host) and their
    gradient are cumprod's up to rounding (the CPU's cumprod accumulates in double precision)."""
    torch.manual_seed(0)
    d, w = (0.2 * torch.rand(32, 512)).requires_grad_(), torch.rand(32, 512)
    grads = []
    for prod in (torch.cumprod, _cumprod_loop):
        c = prod(1 - d, 0)
        (w * c).sum().backward()
        grads.append((c.detach(), d.grad))
        d.grad = None
    torch.testing.assert_close(grads[1][0], grads[0][0], **CLOSE)
    torch.testing.assert_close(grads[1][1], grads[0][1], **CLOSE)


def _tensors(*mods):
    return [t.detach().clone() for m in mods for t in list(m.parameters()) + list(m.buffers())]


def _params(algo):
    return _tensors(algo.wm, algo.pi, *([algo.critic] if hasattr(algo, "critic") else []))


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_graphable_updates_are_the_default_ones(eager, monkeypatch, kin, synthetic_motion, synthetic_mdp, name):
    """wm_update_graphable and policy_update_graphable, run eagerly, give the default updates' parameters bit for bit: the
    same starts (the stable sort keeps the first valid ones, as the default selection does), the same imagination, losses
    and critic steps, the optimizer step gated on the device instead of skipped on the host, the policy average's lerp. The
    default updates' cumprod is the graphable ones' loop here (equal up to rounding, test_survival_products_loop_is_cumprod),
    as a rounding difference in the policy is amplified by the critic's normalization of near-constant features."""
    monkeypatch.setattr(torch, "cumprod", _cumprod_loop)
    kw = dict(fused_adam=True, no_sync=True, pi_ema=0.995)
    if name == "supertrack":
        kw["st_fail_w"] = 5.0  # its survival products
    algos = []
    for _ in range(2):
        a, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **kw)
        algos.append((a, buf))
    (a0, b0), (a1, b1) = algos
    assert all(torch.equal(x, y) for x, y in zip(_params(a0), _params(a1)))
    for step in range(2):  # the second policy update steps the policy average (the first copies it)
        torch.manual_seed(10 + step)
        r0 = a0.wm_update(b0)
        torch.manual_seed(10 + step)
        r1 = a1.wm_update_graphable(b1)
        torch.testing.assert_close(r1[0], r0[0], **EXACT)
        torch.manual_seed(20 + step)
        p0 = a0.policy_update(b0)
        a0.update_pi_avg()
        torch.manual_seed(20 + step)
        n_avg0 = int(a1.pi_avg.n_averaged)
        p1 = a1.policy_update_graphable(b1)
        if int(a1.pi_avg.n_averaged) == n_avg0:
            a1.pi_avg.n_averaged += 1  # as train.py does: the captured average's step does not advance the host-side counter
        torch.testing.assert_close(p1[0], p0[0], **EXACT)
        assert int(a1.nonfinite) == 0
    for x, y in zip(_params(a0), _params(a1)):
        torch.testing.assert_close(y, x, **EXACT)
    for x, y in zip(_tensors(a0.pi_avg.module), _tensors(a1.pi_avg.module)):
        torch.testing.assert_close(y, x, **EXACT)
    assert int(a0.pi_avg.n_averaged) == int(a1.pi_avg.n_averaged) == 2


# the learning rates of check_separate_learning_rates: powers of two at iteration 0, where the two runs train bit for bit
# (the world model at --wm_lr = 2 x --lr, the policy at 4 x --lr); the schedule decays to its floor by iteration 3 and the
# policy's multiplier, triggered by --lr_decay_by at iteration 2, ramps to 1 by iteration 4
LR_ARGS = dict(lr=2.0**-10, wm_lr=2.0**-9, pi_lr_mult=4.0, lr_decay_iters=3, lr_decay_hours=None, lr_decay_ret=0.0, lr_decay_len=2, lr_decay_by=2)


def trainer_learning_rates(algo, args, pi_lr, it: int) -> float:
    """The training loops' learning rates at iteration ``it`` (train.py, train_muscles.py): the schedule's for
    every optimizer, then --wm_lr's for the world model's and --pi_lr_mult's for the policy's; returns the schedule's."""
    from ndm_gym.algos.common import lr_at, set_lr

    lr = lr_at(it, args.lr, args.lr_decay_iters)
    for o in algo.optimizers:
        set_lr(o, lr)
    if args.wm_lr is not None:
        set_lr(algo.opt_wm, lr * args.wm_lr / args.lr)
    if pi_lr.on:
        set_lr(algo.opt_pi, lr * pi_lr.at(it))
    return lr


def check_separate_learning_rates(plain, graphed, bufs, its: int = 5):
    """Two equal agents with Adam for the world model and the policy (and the critic), ``plain`` without --update_graphs and
    ``graphed`` with it (device learning rates), both with --no_sync, through ``its`` training iterations: the loops'
    learning rates (trainer_learning_rates; LR_ARGS), the graphable world-model and policy updates (what --update_graphs
    captures) on the agents' replays ``bufs``, and the multiplier's trigger at each log row. At every iteration the world
    model's rate is the schedule's x --wm_lr / --lr, the policy's the schedule's x the multiplier, the critic's the
    schedule's: in the plain run as numbers, in the graphed run in its device tensors, each optimizer's own (built with
    it and kept, the tensors the captured updates read), the same values in float32. After the first iteration the two
    runs' parameters are equal bit for bit (powers of two: --wm_lr 2 x --lr, --pi_lr_mult 4)."""
    from ndm_gym.algos.blinkmimic import PolicyLR

    args = plain.args
    pi_lr = PolicyLR(args)
    lrs = [g["lr"] for o in graphed.optimizers for g in o.param_groups]
    assert all(torch.is_tensor(x) for x in lrs) and len({id(x) for x in lrs}) == len(lrs) == len(graphed.optimizers) >= 2
    assert all(not torch.is_tensor(g["lr"]) for o in plain.optimizers for g in o.param_groups)
    params = lambda a: [p for o in a.optimizers for g in o.param_groups for p in g["params"]]  # noqa: E731
    seen = set()
    for it in range(its):
        lr = trainer_learning_rates(plain, args, pi_lr, it)
        assert trainer_learning_rates(graphed, args, pi_lr, it) == lr
        want = [lr * args.wm_lr / args.lr, lr * pi_lr.at(it)] + [lr] * (len(plain.optimizers) - 2)
        if it == 0:
            assert want[:2] == [args.wm_lr, args.lr * args.pi_lr_mult]
        assert [g["lr"] for o in plain.optimizers for g in o.param_groups] == want
        assert [float(g["lr"]) for o in graphed.optimizers for g in o.param_groups] == [float(torch.tensor(x)) for x in want]
        assert all(g["lr"] is x for g, x in zip([g for o in graphed.optimizers for g in o.param_groups], lrs))
        seen.add((lr == args.lr, pi_lr.at(it)))
        for agent, buf in zip((plain, graphed), bufs):
            torch.manual_seed(100 + it)
            agent.wm_update_graphable(buf)
            torch.manual_seed(200 + it)
            agent.policy_update_graphable(buf)
        if it == 0:
            assert all(torch.equal(p, q) for p, q in zip(params(plain), params(graphed)))
        pi_lr.observe(it + 1, None)
    assert {(True, 4.0), (False, 4.0), (False, 1.0)} <= seen  # before and after the decay and the multiplier's ramp


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_update_graphs_learning_rates_are_each_optimizers_own(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--update_graphs with Adam for the world model and the policy (and SuperMimic's critic): each optimizer has its own
    device learning rate, so that train.py's --wm_lr sets the world model's alone and --pi_lr_mult the policy's alone, as
    without the option (check_separate_learning_rates)."""
    runs = [_algo(name, kin, synthetic_motion, synthetic_mdp, fused_adam=True, no_sync=True, update_graphs=graphs, **LR_ARGS) for graphs in (False, True)]
    (plain, buf0), (graphed, buf1) = runs
    check_separate_learning_rates(plain, graphed, (buf0, buf1))


def test_nonfinite_policy_step_is_skipped_on_the_device(eager, kin, synthetic_motion, synthetic_mdp):
    """--no_sync: a non-finite policy gradient leaves the policy and its optimizer's state unchanged and is counted."""
    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, fused_adam=True, no_sync=True)
    algo.policy_update_graphable(buf)  # Adam's state exists
    pi0 = [p.detach().clone() for p in algo.pi.parameters()]
    st0 = {k: v.clone() for k, v in algo.opt_pi.state[next(algo.pi.parameters())].items()}
    with torch.no_grad():
        algo.critic.net[-1].bias.fill_(float("nan"))  # the bootstrap, so the policy gradient, is non-finite
    algo.policy_update_graphable(buf)
    assert int(algo.nonfinite) == 1
    assert all(torch.equal(p, q) for p, q in zip(pi0, algo.pi.parameters()))
    st1 = algo.opt_pi.state[next(algo.pi.parameters())]
    assert all(torch.equal(st0[k], st1[k]) for k in st0)


def test_running_norm_device_freeze_test():
    """--no_sync: the normalizer's update with the freeze test on the device equals the default update, before and after
    the count limit."""
    from ndm_gym.core.normalizer import RunningNorm

    torch.manual_seed(0)
    xs = [torch.randn(50, 7) * 3 + 1 for _ in range(4)]
    for limit in (1e9, 120):
        a, b = RunningNorm(7, max_count=limit), RunningNorm(7, max_count=limit)
        try:
            for x in xs:
                RunningNorm.no_sync = False
                a.update(x)
                RunningNorm.no_sync = True
                b.update(x)
        finally:
            RunningNorm.no_sync = False
        for k in ("mean", "var", "count"):
            torch.testing.assert_close(getattr(b, k), getattr(a, k), **EXACT)


def test_device_side_replay_head(synthetic_motion):
    """--graph_early, --collect_graph: add_device writes what add writes; the device head and fill level follow the host's;
    the device-bound draws stay in range."""
    from ndm_gym.buffer import ReplayBuffer

    host = _fill(ReplayBuffer, synthetic_motion, seed=7)
    dev = ReplayBuffer(host.T, host.N, DEV, priorities=False)
    dev.use_device_ptr(device_size=True)
    for t in range(host.T):
        dev.add_device({k: host.data[k][t] for k in host.fields})
        dev.advance_host()
    for k in host.fields:
        torch.testing.assert_close(dev.data[k], host.data[k], **EXACT)
    assert (dev.ptr, dev.size) == (host.ptr % host.T, host.size) and int(dev.ptr_t) == dev.ptr and int(dev.size_t) == dev.size
    ages = torch.arange(dev.size)
    torch.testing.assert_close(dev._row(ages), (dev.ptr - dev.size + ages) % dev.T, **EXACT)
    w = dev.sample_windows(4096, 8)
    assert w["s"].shape[:2] == (4096, 8)
    age, env = dev.sample_start_points(4096)
    assert int(age.min()) >= 0 and int(age.max()) < dev.size and int(env.max()) < dev.N


def test_weighted_bin_statistics_are_the_ended_episodes():
    """--reset_masked with --bin_fix: the update weighted by done over all environments equals the update over the ended
    ones."""
    from ndm_gym.sim.rigid.reset_stats import bin_ema, bin_ema_w

    torch.manual_seed(0)
    stat, bins, x = torch.rand(20), torch.randint(0, 20, (300,)), torch.randn(300)
    done = torch.rand(300) < 0.2
    torch.testing.assert_close(bin_ema_w(stat, bins, x, done.float(), 0.8), bin_ema(stat, bins[done], x[done], 0.8), **CLOSE)


def test_gated_muon_step():
    """MuonAdam.step(ok): skipped entirely where ok is false, the plain step where it is true."""
    import torch.nn as nn

    from ndm_gym.core.muon import MuonAdam

    torch.manual_seed(0)
    nets = [nn.Sequential(nn.Linear(6, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU(), nn.Linear(16, 3)) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [MuonAdam(n, [n], lr=1e-2, fused=True) for n in nets]
    x = torch.randn(32, 6)
    for i, ok in enumerate((True, False, True)):
        for n, o in zip(nets, opts):
            o.zero_grad()
            n(x).square().mean().backward()
        before = [p.detach().clone() for p in nets[1].parameters()]
        if ok:
            opts[0].step()
        opts[1].step(ok=torch.tensor(ok))
        if not ok:
            assert all(torch.equal(p, q) for p, q in zip(before, nets[1].parameters()))
        else:
            for p, q in zip(nets[0].parameters(), nets[1].parameters()):
                torch.testing.assert_close(q, p, **EXACT)


def test_last_due():
    """--last_every_min M with --ckpt_light (train.py and train_muscles ask last_due at each periodic light
    checkpoint): the checkpoint also writes the full last.pt once M minutes of training have passed since the last one,
    whose time the trainer then keeps (before the first: the run's start, or its resume point, the resumed last.pt's
    wall_time); without the option, never."""
    from ndm_gym.train import last_due

    def full(start: float, walls: list, every_min):  # the training times of the periodic checkpoints that write last.pt too
        last, out = start, []
        for wall in walls:
            if last_due(wall, last, every_min):
                out.append(wall)
                last = wall
        return out

    assert not last_due(119.9, 0.0, 2.0) and last_due(120.0, 0.0, 2.0) and last_due(45.0, 30.0, 0.25)
    walls = [50.0 * k for k in range(1, 13)]  # every 50 s of training
    assert full(0.0, walls, None) == [] and full(0.0, walls, 2.0) == [150.0, 300.0, 450.0, 600.0]
    assert full(330.0, [330.0 + 50.0 * k for k in range(1, 6)], 2.0) == [480.0]  # resumed at 330 s of training


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="ndm_gym.train imports Isaac Lab's launcher")
def test_train_takes_last_every_min(monkeypatch, capsys):
    """train.py --last_every_min: off (None) by default; minutes with --ckpt_light for every algorithm (the loop is shared)
    and with --algo blinkmimic (its recipe has --ckpt_light); refused without --ckpt_light (algos.blinkmimic.check_args,
    whose other refusals are in test_blinkmimic)."""
    from ndm_gym import train

    def parse(*argv):
        monkeypatch.setattr(sys, "argv", ["ndm_gym.train", *argv])
        return train.parse_args()

    assert parse("--algo", "supertrack", "--ckpt_light").last_every_min is None
    for algo in ("supertrack", "supermimic"):
        assert parse("--algo", algo, *SYSTEMS, "--last_every_min", "30").last_every_min == 30.0
    assert parse("--algo", "blinkmimic", "--last_every_min", "7.5").last_every_min == 7.5
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parse("--algo", "supermimic", "--last_every_min", "30")
    assert "--last_every_min needs --ckpt_light" in capsys.readouterr().err



# ---------------------------------------------------------------------------------------------------- GPU, simulator
def worker_steps(out: str, mode: str):
    """TrackingSim on the synthetic clips, 8 environments, episodes of at most 5 steps, bin statistics and failure resets
    on (--bin_fix, --fail_reset), no reset noise: 24 steps with fixed random actions, by step() with its resets by index
    (``indexed``), by step_core and step_tail_masked with reset_masked (``masked``), or by those as one CUDA graph
    (``graph``: GraphedStep, as train.py --collect_graph --reset_masked). Each environment's k-th reset goes to its k-th
    entry of fixed tables of frames and joint offsets, so that all runs play the same episodes."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import GraphedStep, TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, n, steps = "cuda:0", 8, 24
    kin = g1.kinematics(dev)
    motion = MotionLib([Clip(**c) for c in synthetic_clips(sim=True)], kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    g = torch.Generator().manual_seed(1)
    acts = 0.3 * torch.randn(steps, n, 29, generator=g)
    clip = torch.randint(0, len(motion.names), (n, steps + 1), generator=g)
    start, end = motion.clip_start.cpu()[clip], motion.clip_end.cpu()[clip]
    frames = (start + (torch.rand(n, steps + 1, generator=g) * (end - 1 - start).float()).long()).to(dev)  # >= 1 step before the end
    offsets = (0.02 * torch.rand(n, steps + 1, 29, generator=g) - 0.01).to(dev)
    sim_cfg = make_sim(dev, launcher)
    rec = {k: [] for k in ("s", "s1", "r", "term", "done", "f", "ep_step", "a_prev", "xi", "start_bin")}
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = TrackingSim(sim_cfg, n, motion, mdp, max_episode_steps=5, reset_noise=False, fail_reset=0.3, bin_fix=True)
        count, ar = torch.zeros(n, dtype=torch.long, device=dev), torch.arange(n, device=dev)
        reset, reset_masked = env.reset, env.reset_masked

        def reset_(ids, frames_=None):
            reset(ids, frames[ids, count[ids]])
            env.xi[ids, 1:30] = offsets[ids, count[ids]]
            count[ids] += 1

        def reset_masked_(mask, frames_=None):
            reset_masked(mask, frames[ar, count])
            env.xi[:, 1:30] = torch.where(mask.unsqueeze(-1), offsets[ar, count], env.xi[:, 1:30])
            count.add_(mask.long())

        env.reset, env.reset_masked = reset_, reset_masked_
        a = torch.zeros(n, 29, device=dev)

        def masked():
            tr = env.step_tail_masked(env.step_core(a, env.state(env.robot.data._root_link_vel_w.data)))
            return {k: tr[k] for k in ("s", "s1", "r", "term", "done")}

        graphed = GraphedStep(env, masked, resets=True, name="the masked step")
        for t in range(steps):
            a.copy_(acts[t])
            if mode == "indexed":
                tr = env.step(a, env.state())
            else:
                env.robot.data.root_link_vel_w  # current, as GraphedStep makes it before each call
                tr = graphed() if mode == "graph" else masked()
            for k in ("s", "s1", "r", "term", "done"):
                rec[k].append(tr[k].cpu())
            for k in ("f", "ep_step", "a_prev", "xi", "start_bin"):
                rec[k].append(getattr(env, k).cpu())
        rec = {k: torch.stack(v) for k, v in rec.items()}
        rec.update(count=count.cpu(), bin_G=env.bin_G.cpu(), bin_fail=env.bin_fail.cpu())
    torch.save(rec, out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_masked_steps_are_the_indexed_ones(tmp_path):
    """--collect_graph, --reset_masked: step_core with step_tail_masked (reset_masked's fixed-size mask writes), eagerly and
    as one CUDA graph, gives the transitions, episode bookkeeping and bin statistics of step() with its resets by index,
    for the same actions and reset targets: the same episodes, the reset states bit for bit, and the states and rewards
    within the simulator's own run-to-run differences (MuJoCo-Warp is not bitwise reproducible across processes: about
    1e-5 within these 5-step episodes, from the first step on)."""
    recs = {}
    for mode in ("indexed", "masked", "graph"):
        out = tmp_path / f"{mode}.pt"
        run(os.path.abspath(__file__), "--worker", mode, "--out", str(out))
        recs[mode] = torch.load(out)
    a = recs["indexed"]
    assert int(a["count"].min()) >= 3, "too few resets"
    starts = torch.cat((torch.ones_like(a["done"][:1]), a["done"][:-1]))  # the steps from a reset state
    for mode in ("masked", "graph"):
        b = recs[mode]
        for k in ("term", "done", "f", "ep_step", "start_bin", "count", "a_prev", "xi"):
            assert torch.equal(b[k], a[k]), (mode, k)
        assert torch.equal(b["s"][starts], a["s"][starts]), (mode, "reset states")
        for k in ("s", "s1", "r", "bin_G", "bin_fail"):
            torch.testing.assert_close(b[k], a[k], rtol=1e-4, atol=1e-4, msg=lambda m, k=k, mode=mode: f"{mode} {k}: {m}")


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_systems_options_train(tmp_path, name):
    """A training run with all the systems options (and --pi_ema): the collection step, the world-model update and the
    policy update are captured and replayed, episodes end and reset inside the collection graph, no update is skipped,
    the periodic checkpoints hold the policy average only (--last_every_min, here at each of them: the full last.pt too)
    and the last one is complete."""
    motions, log = write_motions(tmp_path / "motions"), tmp_path / "run"
    args = [*TRAIN[2:], "--algo", name, "--pi_ema", "0.995", *SYSTEMS, "--last_every_min", "1e-6", "--iterations", "8", "--motion_dir", motions,
            "--log_dir", str(log)]  # fmt: skip
    r = run("-m", "ndm_gym.train", *args)
    for what in ("one collection step", "the world-model update", "the policy update"):
        assert f"captured {what}" in r.stdout, what
    assert "last.pt at it 7 (" in r.stdout
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == list(range(1, 9)) and all(x["nonfinite"] == "0" for x in rows)
    assert any(x["ep_len"] not in ("nan", "") for x in rows), "no episode ended"
    light = torch.load(log / "ckpt_000005.pt", map_location="cpu", weights_only=False)
    assert set(light) == {"pi", "iteration", "wall_time", "sim_steps", "args", "saved_at"} and light["iteration"] == 5
    last = torch.load(log / "last.pt", map_location="cpu", weights_only=False)
    assert last["iteration"] == 8 and "wm" in last and "opt_pi" in last


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["indexed", "masked", "graph"], required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    worker_steps(cli.out, cli.worker)
