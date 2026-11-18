"""train_hand --layout (algos.hand_layout, algos.layouts.Split) on the CPU, every "device" the CPU, with a stand-in for
the simulator's collection: the split loop gives the networks of a plain loop over one agent that applies the staleness
by hand, bit for bit, for both agents and both layouts (the first iteration makes no policy update, AWL's replica is the
replay after every exchange, also when the write head wraps), and not those of the one-GPU order; the policy learner's update on the batch it was sent is the
update on its own draw; train_hand's loop runs under a layout (the checkpoint, the log, --phase_times with the exchange
timed apart); the layout's refused arguments raise. The CUDA graphs on two or three GPUs are the GPU runs'."""

import sys
import types

import mujoco
import numpy as np
import pytest
import torch
from test_hand_tracking import NUM_J, ST_ARM, SYSTEMS, KinematicSim, hand_task, same_tree, scene_xml, small

from ndm_gym.algos import hand_fast, hand_layout
from ndm_gym.algos.hand_algos import HAND_ALGOS, START_KEYS, WINDOW_KEYS, replay_fields
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.tasks.hand_tracking import flatten_state
from ndm_gym.train_hand import parse_args

LAYOUT = [*SYSTEMS, "--update_graphs", "--pi_ema", "0.9"]  # what a layout needs, and the policy average on the learner
N, T, WARM, ITERS = 8, 16, 12, 6  # worlds, replay steps, warm-up steps, iterations: the head wraps in the fourth iteration
AGENTS = pytest.mark.parametrize("extra", [[], ST_ARM], ids=["supermimic", "supertrack"])


@pytest.fixture(scope="module")
def scene():
    """test_hand_tracking's small scene with the XHand's joints, its kinematics and a task (its fixture ``setup``)."""
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_string(scene_xml())
    return (model, *hand_task(model))


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", True)  # --no_sync: the normalizers written in place, which the copies rely on


def _replay(task) -> ReplayBuffer:
    buf = ReplayBuffer(T, N, "cpu", fields=replay_fields(task), priorities=False)
    buf.window_keys = WINDOW_KEYS
    buf.use_device_ptr(device_size=True)
    buf.ptr = T - WARM - 3
    buf.ptr_t.fill_(buf.ptr)
    return buf


def _agent(kin, task, extra, layout="AL"):
    args = parse_args(small([*extra, *LAYOUT, "--layout", layout]))
    torch.manual_seed(0)
    return HAND_ALGOS[args.algo](args, kin, task, "cpu")


def _collect(algo, buf, task, k: int):
    """A collection step's stand-in (``k``: the iteration, negative in the warm-up): the reference's states at random frames
    with random previous actions, the agent's noisy actions (which fit its policy's normalizer) as the rows' targets, the
    reference's next states; the world model's normalizers fit on the transition, as hand_fast.Collector fits them."""
    g = torch.Generator().manual_seed(1000 + k)
    f = torch.randint(0, task.ref.frames - 1, (N,), generator=g)
    S, S1, a_prev = task.ref.state(f), task.ref.state(task.ref.next_frame(f)), 0.1 * torch.randn(N, NUM_J, generator=g)
    with torch.no_grad():
        u = task.setpoint(algo.act(S, a_prev, f, noise=0.1), f)
        algo.update_wm_norms(S, S1, u)
    buf.add({"s": flatten_state(S), "s1": flatten_state(S1), "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(N, dtype=torch.bool)})


def _snap(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def _snap_buffers(module):
    return {k: v.clone() for k, v in module.named_buffers()}


@torch.no_grad()
def _load_buffers(module, snap):
    for k, v in module.named_buffers():
        v.copy_(snap[k])


def _draw(ref, buf):
    s, h = ref.sample_valid_starts_nosync(buf, ref.args.pi_batch)
    assert not h and set(s) == set(START_KEYS)
    return {k: v.clone() for k, v in s.items()}, {}


def _given_update(ref, msg, average):
    """The policy update of one agent on a batch, a world model and a policy normalizer of an earlier iteration (``msg``),
    with the policy average's step and count; the policy's normalizer is the current one again afterwards."""
    now = _snap_buffers(ref.pi)
    ref.wm.load_state_dict(msg["wm"])
    _load_buffers(ref.pi, msg["pi_norms"])
    ref.sample_valid_starts_nosync = lambda buffer, B: msg["batch"]
    ref.policy_update_graphable(None)
    average.count()
    del ref.sample_valid_starts_nosync
    _load_buffers(ref.pi, now)


@torch.no_grad()
def _copy_replay(dst, src):
    for k in src.data:
        dst.data[k].copy_(src.data[k])
    dst.ptr, dst.size = src.ptr, src.size
    dst.ptr_t.copy_(src.ptr_t), dst.size_t.copy_(src.size_t)


def _run_stale(ref, buf, task, average, layout: str):
    """The layout's staleness by hand on one agent. AL: iteration i's policy update takes the batch drawn at the end of
    iteration i - 1, the world model as iteration i - 1's updates left it and the policy's normalizer of then; the
    collection and the world-model updates are the current ones. AWL: the world-model updates also see the replay up to
    iteration i - 1's collection (a second buffer, brought up to date at each iteration's end) with the normalizers that
    collection left, and the policy update takes the world model of iteration i - 1's updates with the normalizers they
    used."""
    for j in range(WARM):
        _collect(ref, buf, task, -1 - j)
    lag = _replay(task)
    _copy_replay(lag, buf)
    lag_norms, msg = _snap_buffers(ref.wm), None
    for k in range(ITERS):
        _collect(ref, buf, task, k)
        live = _snap_buffers(ref.wm)
        if layout == "AWL":
            _load_buffers(ref.wm, lag_norms)
        ref.wm_update_graphable(lag if layout == "AWL" else buf)
        wm_now = _snap(ref.wm)
        if msg is not None:
            _given_update(ref, msg, average)
        ref.wm.load_state_dict(wm_now)
        _load_buffers(ref.wm, live)
        msg = {"batch": _draw(ref, buf), "wm": wm_now, "pi_norms": _snap_buffers(ref.pi)}
        _copy_replay(lag, buf)
        lag_norms = live


def _run_split(split, buf, task, average, after=None):
    """train_hand's loop under a layout, its hooks in its order: the warm-up collection, the updates; then per iteration
    the collection, the world-model updates, the policy update once the learner holds a batch, the exchange."""
    for j in range(WARM):
        _collect(split.actor, buf, task, -1 - j)
    wm_update, policy_update = hand_layout.update_functions(split, average)
    for k in range(ITERS):
        _collect(split.actor, buf, task, k)
        wm_update()
        if split.ready:
            policy_update()
        split.exchange()
        if after is not None:
            after(k)


def _networks(wm, pi, average, agent) -> dict:
    mods = {"wm": wm, "pi": pi, "avg": average.avg, **{k: getattr(agent, k) for k in ("critic", "bal") if getattr(agent, k, None) is not None}}
    return {f"{m}.{k}": v.detach().clone() for m, mod in mods.items() for k, v in mod.state_dict().items()}


@AGENTS
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_split_loop_is_the_one_agent_loop_with_the_staleness_by_hand(eager, scene, extra, layout):
    """Six iterations (five policy updates) of the split loop, one agent per device, against the plain loop over one agent
    that swaps in the previous iteration's batch, world model and normalizers by hand: the world model, the policy, the
    policy average with its count, SuperMimic's critic and SuperTrack's weights are equal bit for bit (the same operations
    in the same order, the copies exact), and so are the replays (the collections acted with the same policies). After
    every exchange the actor's policy has the learner's weights, the learner holds the batch the actor drew, and AWL's
    replica is the replay; the write head wraps during the run."""
    _, kin, task = scene
    actor, ref = _agent(kin, task, extra, layout), _agent(kin, task, extra, layout)
    buf, rbuf = _replay(task), _replay(task)
    split, average = hand_layout.build(actor.args, actor, buf, lambda d: (kin, task))
    ref.average = ref_average = hand_fast.PolicyAverage(ref.pi, actor.args.pi_ema)
    assert (split.wmer is actor) == (layout == "AL") and split.learner.average is average and split.algo.pi_avg is average.avg
    start = _networks(ref.wm, ref.pi, ref_average, ref)
    assert same_tree(_networks(split.wmer.wm, split.learner.pi, average, split.learner), start)  # every device's agent starts from the actor's
    seen = []

    def after(k):
        assert all(torch.equal(p, q) for p, q in zip(split.learner.pi.parameters(), actor.pi.parameters(), strict=True)), k
        assert all(torch.equal(split.out.s[f], split.given.s[f]) for f in START_KEYS) and split.given.h == {}, k
        if split.replica is not None:
            rep = split.replica
            assert all(torch.equal(rep.data[f], buf.data[f]) for f in buf.data) and (rep.ptr, rep.size) == (buf.ptr, buf.size), k
            assert (int(rep.ptr_t), int(rep.size_t)) == (buf.ptr, buf.size) and rep.window_keys == WINDOW_KEYS, k
        seen.append(int(average.avg.n_averaged))

    torch.manual_seed(123)
    _run_split(split, buf, task, average, after)
    torch.manual_seed(123)
    _run_stale(ref, rbuf, task, ref_average, layout)
    assert seen == list(range(ITERS)) and buf.ptr == rbuf.ptr == ITERS - 3 and buf.size == T  # no update in the first iteration; the head wrapped
    want, got = _networks(ref.wm, ref.pi, ref_average, ref), _networks(split.wmer.wm, split.learner.pi, average, split.learner)
    assert want.keys() == got.keys()
    for k in want:
        assert torch.equal(got[k], want[k]), k
    assert all(any(not torch.equal(want[k], start[k]) for k in want if k.startswith(m)) for m in ("wm.", "pi.", "avg."))
    assert all(torch.equal(buf.data[f], rbuf.data[f]) for f in buf.data)
    sd = split.algo.state_dict()  # the checkpoint's: the learner's, the world model and its optimizer from their owner
    assert sd.keys() == ref.state_dict().keys() and same_tree(sd["wm"], split.wmer.wm.state_dict()) and same_tree(sd["pi"], split.learner.pi.state_dict())


def test_split_loop_is_not_the_one_gpu_loop(eager, scene):
    """The staleness is real: the one-GPU order (every policy update on a batch drawn after the iteration's collection, in
    the current world model) gives another policy than AL's loop, and makes one policy update more."""
    _, kin, task = scene
    actor, ref = _agent(kin, task, []), _agent(kin, task, [])
    buf, rbuf = _replay(task), _replay(task)
    split, average = hand_layout.build(actor.args, actor, buf, lambda d: (kin, task))
    ref.average = ref_average = hand_fast.PolicyAverage(ref.pi, actor.args.pi_ema)
    torch.manual_seed(123)
    _run_split(split, buf, task, average)
    torch.manual_seed(123)
    for j in range(WARM):
        _collect(ref, rbuf, task, -1 - j)
    for k in range(ITERS):
        _collect(ref, rbuf, task, k)
        ref.wm_update_graphable(rbuf)
        ref.policy_update_graphable(rbuf)
        ref_average.count()
    assert int(ref_average.avg.n_averaged) == ITERS == int(average.avg.n_averaged) + 1
    assert any(not torch.equal(p, q) for p, q in zip(ref.pi.parameters(), split.learner.pi.parameters(), strict=True))


@AGENTS
def test_policy_update_on_a_given_batch_is_the_update_on_its_draw(eager, scene, extra):
    """The learner's update takes its batch from layouts.GivenStarts in place of its own draw from the replay: on the batch
    the draw would give, and from the random stream's position after that draw, it is the update on the replay bit for
    bit. Given anything else as its replay, or another batch size, it raises."""
    from ndm_gym.algos.layouts import GivenStarts

    _, kin, task = scene
    (a0, a1), (b0, b1) = [_agent(kin, task, extra) for _ in range(2)], [_replay(task) for _ in range(2)]
    for algo, buf in ((a0, b0), (a1, b1)):
        for j in range(WARM):
            _collect(algo, buf, task, -1 - j)
    B = a0.args.pi_batch
    torch.manual_seed(5)
    out0 = a0.policy_update_graphable(b0)
    torch.manual_seed(5)
    given = GivenStarts(b1, B, 0, "cpu", START_KEYS)
    given.fill(*a1.sample_valid_starts_nosync(b1, B))
    a1.sample_valid_starts_nosync = given.starts
    out1 = a1.policy_update_graphable(given)
    assert all(torch.equal(x, y) or bool(x.isnan() & y.isnan()) for x, y in zip(out0, out1, strict=True))  # SuperTrack's value loss: NaN
    assert all(torch.equal(p, q) for p, q in zip(a0.pi.parameters(), a1.pi.parameters(), strict=True))
    with pytest.raises(ValueError):
        a1.policy_update_graphable(b1)
    with pytest.raises(ValueError):
        given.starts(given, B + 1)
    with pytest.raises(ValueError, match="draw of"):  # a draw with a history, or with other fields, does not fit the batch
        given.fill({k: torch.zeros(1) for k in START_KEYS}, {"s": torch.zeros(1)})


@AGENTS
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_training_loop_under_a_layout(eager, scene, monkeypatch, tmp_path, capsys, extra, layout):
    """train_hand's loop on the CPU under a layout, a simulator without physics in HandSim's place and the collection step
    without its CUDA stream: the split is described, the policy updates start with the learner's first batch (after
    --wm_only_iters), the log's rows are finite, the checkpoint holds the learner's policy, its average and the world model
    of its owner; --phase_times times the exchange apart."""
    import ndm_gym.train_hand as train_hand

    monkeypatch.setattr(train_hand, "DEVICE", "cpu")
    monkeypatch.setattr(train_hand, "build", lambda clip, device, reward_std_scale=1.0: scene)
    hand_sim = types.ModuleType("ndm_gym.sim.rigid.hand_sim")  # HandSim's module, which needs the simulator and a GPU
    hand_sim.HandSim = lambda model, task, num_envs, device, stagger=False: KinematicSim(task, num_envs)
    monkeypatch.setitem(sys.modules, "ndm_gym.sim.rigid.hand_sim", hand_sim)
    monkeypatch.setattr(hand_fast.Collector, "__call__", lambda self: self.core())  # the step itself: its stream is CUDA's
    run = tmp_path / layout
    argv = [*small(extra), *LAYOUT, "--layout", layout, "--phase_times", "--num_envs", str(N), "--buffer_steps", str(T), "--warmup_steps", "8",
            "--wm_only_iters", "1", "--log_every", "2", "--ckpt_every", "4", "--iterations", "6", "--log_dir", str(run)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_hand", *argv])
    train_hand.main()
    out = capsys.readouterr().out
    assert f"--layout {layout}: actor cpu" in out and ("world-model learner" in out) == (layout == "AWL")
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert ck["iteration"] == 6 and ck["args"]["layout"] == layout and {"wm", "pi", "pi_live", "pi_avg", "opt_wm", "opt_pi"} <= set(ck)
    assert int(ck["pi_avg"]["n_averaged"]) == 5  # iterations 2-6
    log = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
    cols = [train_hand.LOG_COLUMNS.index(k) for k in ("wm_loss", "pi_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r")]
    assert log.shape == (3, len(train_hand.LOG_COLUMNS)) and np.isfinite(log[:, cols]).all()
    lines = (run / "phase_times.csv").read_text().splitlines()
    assert lines[0] == ",".join(("iteration", *(f"{p}_ms" for p in hand_fast.LAYOUT_PHASES), "it_per_s")) and len(lines) == 4
    rows = np.array([[float(x) for x in line.split(",")] for line in lines[1:]])
    assert (rows[:, 1 + hand_fast.LAYOUT_PHASES.index("exchange")] > 0).all() and np.isnan(rows[:, 1 + hand_fast.LAYOUT_PHASES.index("critic")]).all()
    assert np.allclose(np.nansum(rows[:, 1:-1], 1) * rows[:, -1], 1000, rtol=1e-4)
    assert f"--layout {layout}: every device of the layout" in out and "exchange" in out.split("--phase_times: ")[-1]


def test_layout_arguments(monkeypatch):
    """--layout is off by default (1GPU); a layout needs --no_sync and --update_graphs, takes no --resume and no HAND_PROFILE;
    --layout_devices places its steps, one index per step, the actor's 0 first, the policy learner's not on the actor's GPU."""
    from ndm_gym.train_hand import parse_args as parse

    assert parse(["--clip", "clip"]).layout == "1GPU"
    on = parse(["--clip", "clip", *LAYOUT, "--layout", "AWL"])
    assert on.layout == "AWL"
    assert on.layout_devices is None and parse(["--clip", "clip", *LAYOUT, "--layout", "AWL", "--layout_devices", "0", "0", "1"]).layout_devices == [0, 0, 1]
    for refused in (["--layout", "AL"], [*SYSTEMS, "--layout", "AL"], [*LAYOUT, "--layout", "AL", "--resume", "last.pt"], [*LAYOUT, "--layout", "2GPU"],
                    ["--layout_devices", "0", "1"], [*LAYOUT, "--layout", "AL", "--layout_devices", "0", "0", "1"],
                    [*LAYOUT, "--layout", "AWL", "--layout_devices", "1", "0", "1"], [*LAYOUT, "--layout", "AWL", "--layout_devices", "0", "1", "0"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse(["--clip", "clip", *refused])
    monkeypatch.setenv("HAND_PROFILE", "1")
    with pytest.raises(ValueError, match="HAND_PROFILE"):
        hand_layout.build(on, None, None, None)
