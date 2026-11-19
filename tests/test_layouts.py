"""XMimic's iteration split across GPUs (train.py --layout; algos/layouts.py, docs/algorithms/xmimic.md) with a stand-in for the
simulator's collection. On the CPU, every "device" the CPU: the split loop gives the parameters of a plain loop over one
object that applies the staleness by hand, bit for bit, and not those of the one-GPU order; every exchange leaves each
object with what its next step reads; what crosses the devices is every tensor of the policy and the world model, the
batch and the replay rows, and nothing else; the training loop reads each quantity from the object that holds it; the
graphs' and the run-ahead limit's device handling; refused arguments raise. With two or three GPUs (marked slow, so that
the fast suite takes no GPU; skipped without them): the split's updates as CUDA graphs run on their devices, and give the
eager split's parameters."""

import argparse
import contextlib
import importlib.util

import pytest
import torch
import torch.nn as nn

from conftest import random_states, synthetic_clips
from test_algo_equivalence import EXACT, N, T, _args, _fill
from test_bptt_tricks import ST_ARGS

KW = dict(fused_adam=True, no_sync=True, update_graphs=True, pi_ema=0.995, utd=0.4, pi_batch=32, horizon=16)
WM_BATCHES = 2
ITERS = 5  # four policy updates: the first copies the policy average, the others step it
GPUS = torch.cuda.device_count()
needs_isaaclab = pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="ndm_gym.train imports Isaac Lab's launcher")


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


@pytest.fixture()
def in_place_norms(monkeypatch):
    """--no_sync: the normalizers' statistics written in place, which the layouts' copies between devices rely on."""
    from ndm_gym.core.normalizer import RunningNorm

    monkeypatch.setattr(RunningNorm, "no_sync", True)


@pytest.fixture()
def task(kin, synthetic_motion, synthetic_mdp):
    """device -> the task's tables there (kinematics, motion library, MDP), as train.py's algo_on builds them: the session's
    on the CPU, built once per test on a GPU."""
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP

    made = {"cpu": (kin, synthetic_motion, synthetic_mdp)}

    def on(d):
        d = str(d)
        if d not in made:
            k = g1.kinematics(d)
            m = MotionLib([Clip(**c) for c in synthetic_clips()], k, d)
            made[d] = (k, m, MDP(k, m, torch.tensor(g1.default_joint_pos_list(), device=d)))
        return made[d]

    return on


def _agent(name, task, dev="cpu", **kw):
    """test_bptt_tricks' ``_algo`` on ``dev`` with the split's options: a small-MLP algorithm whose world model's last layer
    is random, the replay data of ``_fill`` in a buffer with its write head and fill level as tensors (--graph_early), the
    normalizers fit on it."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.state import normalize_xi, unflatten_state

    kin, motion, mdp = task(dev)
    args = _args()
    vars(args).update({"algo": name, "wm_hidden": [64, 64], **ST_ARGS, **KW, **kw})
    torch.manual_seed(0)
    algo = ALGOS[name](args, kin, motion, mdp, dev)
    nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
    host = _fill(ReplayBuffer, task("cpu")[1], seed=7)
    buf = ReplayBuffer(host.T, host.N, dev, priorities=False)
    for k in host.fields:
        buf.data[k].copy_(host.data[k])
    buf.ptr, buf.size = host.ptr, host.size
    buf.use_device_ptr(device_size=True)
    w = buf.data
    S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
    algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1))))
    algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    algo.pi.obs_norm.update(mdp.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten()))
    algo.pi.goal_norm.update(motion.goal(w["f"].flatten(), args.goal_frames))
    return algo, buf


def _turn(buf, ptr: int, size: int | None = None):
    """The replay's write head at ``ptr`` (and its fill level ``size``), on the host and the device: a full replay may have
    its head anywhere."""
    buf.ptr, buf.size = ptr, buf.size if size is None else size
    buf.ptr_t.fill_(buf.ptr), buf.size_t.fill_(buf.size)


def _split(name, layout, task, devices=None, **kw):
    """The split's objects: the actor's on the first device, the others built per device as train.py's algo_on builds them.
    ``devices`` None: every one the CPU. The replay's head starts six rows before its end, so that a run's rows wrap."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.layouts import Split
    from ndm_gym.algos.xmimic import LAYOUTS

    devices = devices or ("cpu",) * LAYOUTS[layout]
    actor, buf = _agent(name, task, devices[0], layout=layout, **kw)
    _turn(buf, T - 6)
    made = []

    def algo_on(d):
        made.append(ALGOS[name](actor.args, *task(d), d))
        return made[-1]

    split = Split(actor.args, actor, algo_on, buf, devices=devices)
    assert len(made) == LAYOUTS[layout] - 1 and [str(torch.device(m.dev)) for m in made[::-1]] == [str(torch.device(d)) for d in devices[1:]]
    return split, buf


def _collect(algo, buf, motion, k: int):
    """A collection's stand-in: two or three steps (--utd 0.4's schedule) of noisy reference states (drawn on the CPU from
    ``motion``, the CPU's library) with the setpoints of the algorithm's policy, so that the rows depend on the policy that
    acts, written at the replay's device-side head, the normalizers updated as train.py's collection updates them. ``k``
    -1: the warm-up collection."""
    from ndm_gym.state import flatten_state

    K, Hh, dev, mdp = algo.args.goal_frames, algo.args.hist, algo.dev, algo.mdp
    with torch.no_grad():
        for j in range(2 + k % 2):
            f, S, xi = random_states(motion, N, seed=5000 + 10 * k + j)
            f, S, xi = f.to(dev), tuple(x.to(dev) for x in S), xi.to(dev)
            o = mdp.obs(S, torch.zeros(N, 29, device=dev), xi, f)
            g = algo.motion.goal(f, K)
            algo.update_pi_norms(o, g)
            a = algo.setpoint(algo.pi(o.unsqueeze(1).expand(N, Hh, -1), g), f)
            S1 = algo.motion.ref_state(f + 1)
            buf.add_device({"s": flatten_state(S), "s1": flatten_state(S1), "a": a, "a_prev": a, "xi": xi, "r": torch.zeros(N, device=dev), "f": f,
                            "ep_step": torch.full((N,), 3 + j, dtype=torch.int32, device=dev),
                            "done": torch.zeros(N, dtype=torch.bool, device=dev)})  # fmt: skip
            buf.advance_host()
            algo.update_wm_norms(S, S1, a, xi)


def _policy(algo, update):
    """A policy update as train.py's loop calls it: the captured average's step does not advance the host-side counter."""
    n0 = int(algo.pi_avg.n_averaged)
    out = update()
    if int(algo.pi_avg.n_averaged) == n0:
        algo.pi_avg.n_averaged += 1
    return out


def _run_split(split, buf, motion, iters=ITERS, after=None, graphs=False, seed=None):
    """train.py's loop under a layout, its hooks in its order: the warm-up collection (the split is built before it), the
    graphs; then per iteration the collection, the world-model updates, the policy update once the learner holds a batch,
    the exchange. ``seed``: the random streams set anew at every iteration (a captured update against its eager twin).
    Returns the two updates."""
    _collect(split.actor, buf, motion, -1)
    wm_update, policy_update = split.update_graphs(graphs=graphs)
    for k in range(iters):
        if seed is not None:
            torch.manual_seed(seed + k)
        _collect(split.actor, buf, motion, k)
        for _ in range(WM_BATCHES):
            wm_update()
        if split.ready:
            _policy(split.algo, policy_update)
        split.exchange()
        if after is not None:
            after(k)
    return wm_update, policy_update


def _snap(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def _snap_buffers(module):
    return {k: v.clone() for k, v in module.named_buffers()}


@torch.no_grad()
def _load_buffers(module, snap):
    for k, v in module.named_buffers():
        v.copy_(snap[k])


def _given_update(ref, msg):
    """One object's policy update on a batch, a world model and policy normalizers of an earlier iteration (``msg``); the
    policy's normalizers are the current ones again afterwards (the world model is left to the caller)."""
    now = _snap_buffers(ref.pi)
    ref.wm.load_state_dict(msg["wm"])
    _load_buffers(ref.pi, msg["pi_norms"])
    ref.sample_valid_starts_nosync = lambda buffer, B: msg["batch"]
    _policy(ref, lambda: ref.policy_update_graphable(None))
    del ref.sample_valid_starts_nosync
    _load_buffers(ref.pi, now)


def _draw(ref, buf):
    s, h = ref.sample_valid_starts_nosync(buf, ref.args.pi_batch)
    return {k: v.clone() for k, v in s.items()}, {k: v.clone() for k, v in h.items()}


def _run_stale_al(ref, buf, motion, iters=ITERS):
    """AL's staleness by hand on one object: iteration i's policy update takes the batch drawn at the end of iteration
    i - 1, the world model as iteration i - 1's updates left it and the policy's normalizers of then; the collection and
    the world-model updates are the current ones."""
    _collect(ref, buf, motion, -1)
    msg = None
    for k in range(iters):
        _collect(ref, buf, motion, k)
        for _ in range(WM_BATCHES):
            ref.wm_update_graphable(buf)
        if msg is not None:
            wm_now = _snap(ref.wm)
            _given_update(ref, msg)
            ref.wm.load_state_dict(wm_now)
        msg = {"batch": _draw(ref, buf), "wm": _snap(ref.wm), "pi_norms": _snap_buffers(ref.pi)}


@torch.no_grad()
def _copy_replay(dst, src):
    for k in src.data:
        dst.data[k].copy_(src.data[k])
    dst.ptr, dst.size = src.ptr, src.size
    dst.ptr_t.copy_(src.ptr_t), dst.size_t.copy_(src.size_t)


def _run_stale_awl(ref, buf, motion, iters=ITERS):
    """AWL's staleness by hand on one object: iteration i's world-model updates see the replay up to iteration i - 1's
    collection (a second buffer, brought up to date at each iteration's end) with the normalizers that collection left;
    its policy update takes the batch and the policy's normalizers of iteration i - 1 and the world model as iteration
    i - 1's updates left it, with the normalizers those used."""
    from ndm_gym.buffer import ReplayBuffer

    _collect(ref, buf, motion, -1)
    lag = ReplayBuffer(buf.T, buf.N, buf.device, fields=buf.fields, priorities=False)
    lag.use_device_ptr(device_size=True)
    _copy_replay(lag, buf)
    lag_norms, msg = _snap_buffers(ref.wm), None
    for k in range(iters):
        _collect(ref, buf, motion, k)
        live = _snap_buffers(ref.wm)
        _load_buffers(ref.wm, lag_norms)
        for _ in range(WM_BATCHES):
            ref.wm_update_graphable(lag)
        wm_now = _snap(ref.wm)
        if msg is not None:
            _given_update(ref, msg)
        ref.wm.load_state_dict(wm_now)
        _load_buffers(ref.wm, live)
        msg = {"batch": _draw(ref, buf), "wm": wm_now, "pi_norms": _snap_buffers(ref.pi)}
        _copy_replay(lag, buf)
        lag_norms = live


def _run_one_gpu(ref, buf, motion, iters=ITERS):
    """The one-GPU order: every policy update on a batch drawn after the iteration's collection, in the current world model."""
    _collect(ref, buf, motion, -1)
    for k in range(iters):
        _collect(ref, buf, motion, k)
        for _ in range(WM_BATCHES):
            ref.wm_update_graphable(buf)
        _policy(ref, lambda: ref.policy_update_graphable(buf))


def _state(wm, pi, avg, critic=None):
    mods = {"wm": wm, "pi": pi, "avg": avg.module, **({"critic": critic} if critic is not None else {})}
    return {f"{m}.{k}": v.detach().cpu().clone() for m, mod in mods.items() for k, v in mod.state_dict().items()}


def _split_state(split):
    """The run's networks under a layout, each from its owner (on the CPU); the actor's policy is the learner's (weights)
    with its own normalizers."""
    learner = split.learner
    return _state(split.wmer.wm, learner.pi, learner.pi_avg, getattr(learner, "critic", None))


def _ref_state(ref):
    return _state(ref.wm, ref.pi, ref.pi_avg, getattr(ref, "critic", None))


def _same(a, b) -> bool:
    return torch.equal(a.detach().cpu(), b.detach().cpu())


def _replica_is_the_replay(split, buf):
    rep = split.replica
    assert all(_same(rep.data[f], buf.data[f]) for f in buf.data)
    assert (rep.ptr, rep.size) == (buf.ptr, buf.size)
    assert int(rep.ptr_t) == buf.ptr == int(buf.ptr_t) and int(rep.size_t) == buf.size == int(buf.size_t)


def _check_split_run(split, buf, motion, graphs: bool, iters: int = 8):
    """The split loop with a look at every device after each exchange (which reads them, so waits for them). After every
    exchange: the actor's policy weights are the learner's and the learner's policy normalizers the actor's; the learner
    holds the batch the actor drew; the learner's world model is its owner's (AL: all of it; AWL: the weights, and the
    owner's normalizers are the actor's); AWL's replica is the actor's replay. From one iteration to the next the world
    model changed, the batch is another draw and, from the second iteration on, the policy and its average changed: an
    update that did not run on its device, or a copy that did not wait for it, breaks one of these. No policy update was
    skipped. Returns the two updates and the networks after every iteration."""
    from ndm_gym.algos.layouts import START_KEYS

    actor, wmer, learner = split.actor, split.wmer, split.learner
    seen, states = [], []

    def after(k):
        for (name, p), q in zip(learner.pi.named_parameters(), actor.pi.parameters()):
            assert _same(p, q), f"iteration {k}: the actor's policy weight {name}"
        for (name, b), c in zip(actor.pi.named_buffers(), learner.pi.buffers()):
            assert _same(b, c), f"iteration {k}: the learner's policy normalizer {name}"
        for f in START_KEYS:
            assert _same(split.out.s[f], split.given.s[f]) and _same(split.out.h[f], split.given.h[f]), f"iteration {k}: the batch's {f}"
        if split.replica is None:
            for (name, v), w in zip(actor.wm.state_dict().items(), learner.wm.state_dict().values()):
                assert _same(v, w), f"iteration {k}: the learner's world model, {name}"
        else:
            for (name, p), q in zip(wmer.wm.named_parameters(), learner.wm.parameters()):
                assert _same(p, q), f"iteration {k}: the learner's world-model weight {name}"
            for (name, b), c in zip(actor.wm.named_buffers(), wmer.wm.buffers()):
                assert _same(b, c), f"iteration {k}: the world-model learner's normalizer {name}"
            _replica_is_the_replay(split, buf)
        cpu = lambda ts: [t.detach().cpu().clone() for t in ts]  # noqa: E731
        seen.append({"wm": cpu(wmer.wm.parameters()), "pi": cpu(learner.pi.parameters()), "avg": cpu(learner.pi_avg.module.parameters()),
                     "batch": split.given.s["s"].cpu().clone()})  # fmt: skip
        states.append(_split_state(split))

    wm_update, policy_update = _run_split(split, buf, motion, iters=iters, after=after, graphs=graphs, seed=1000)
    differ = lambda a, b: any(not torch.equal(x, y) for x, y in zip(a, b))  # noqa: E731
    for k in range(1, iters):
        assert differ(seen[k]["wm"], seen[k - 1]["wm"]), f"iteration {k}: the world-model updates left the world model as it was"
        assert not torch.equal(seen[k]["batch"], seen[k - 1]["batch"]), f"iteration {k}: the batch is the one before"
        assert differ(seen[k]["pi"], seen[k - 1]["pi"]), f"iteration {k}: the policy update left the policy as it was"
        assert differ(seen[k]["avg"], seen[k - 1]["avg"]), f"iteration {k}: the policy update left the policy average as it was"
    assert int(learner.nonfinite) == 0 and int(learner.pi_avg.n_averaged) == iters - 1
    assert all(bool(torch.isfinite(v).all()) for v in states[-1].values() if v.is_floating_point())
    return wm_update, policy_update, states


# ---------------------------------------------------------------- the split loop is the stale loop
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_split_loop_is_the_one_object_loop_with_the_staleness_by_hand(eager, in_place_norms, task, synthetic_motion, name, layout):
    """Five iterations (four policy updates) of the split loop, one object per device, against the plain loop over one
    object that swaps in the previous iteration's batch, world model and normalizers by hand: the world model, the policy,
    the policy average and the critic are equal bit for bit (the same operations in the same order; the copies between the
    objects are exact). At the end the actor's policy has the learner's weights, the learner's world model is the owner's
    as it was sent, and after every iteration AWL's replica is the actor's replay (the head wraps past the replay's end
    during the run). The first iteration makes no policy update."""
    split, buf = _split(name, layout, task)
    ref, rbuf = _agent(name, task)
    _turn(rbuf, buf.ptr)
    start = _ref_state(ref)
    assert all(torch.equal(v, start[k]) for k, v in _split_state(split).items())  # every device's object starts from the actor's networks
    updates = []

    def after(k):
        updates.append(int(split.learner.pi_avg.n_averaged))
        if split.replica is not None:
            _replica_is_the_replay(split, buf)

    torch.manual_seed(123)
    _run_split(split, buf, synthetic_motion, after=after)
    torch.manual_seed(123)
    (_run_stale_al if layout == "AL" else _run_stale_awl)(ref, rbuf, synthetic_motion)
    assert updates == [0, 1, 2, 3, 4] and buf.ptr == rbuf.ptr < 12  # the head wrapped
    want, got = _ref_state(ref), _split_state(split)
    assert want.keys() == got.keys()
    for k in want:
        torch.testing.assert_close(got[k], want[k], **EXACT, msg=k)
    assert all(any(not torch.equal(want[k], start[k]) for k in want if k.startswith(m)) for m in ("wm.", "pi.", "avg."))
    assert int(split.learner.nonfinite) == 0
    assert all(torch.equal(buf.data[f], rbuf.data[f]) for f in buf.data)  # the collections acted with the same policies
    # the exchange left the actor with the learner's policy weights (its normalizers its own), the learner with the world model sent
    for (k, p), q in zip(split.learner.pi.named_parameters(), split.actor.pi.parameters()):
        assert torch.equal(p, q), k
    for (k, b), c in zip(split.actor.pi.named_buffers(), ref.pi.buffers()):
        assert torch.equal(b, c), k
    for (k, v), w in zip(split.wmer.wm.state_dict().items(), split.learner.wm.state_dict().values()):
        if layout == "AL" or not k.split(".")[0].endswith("_norm"):  # AWL: the learner's normalizers are those the weights were fit with
            assert torch.equal(v, w), k


def test_split_loop_is_not_the_one_gpu_loop(eager, in_place_norms, task, synthetic_motion):
    """The staleness is real: the one-GPU order (every policy update on a fresh batch in the current world model) gives
    another policy than AL's loop, and makes one more policy update."""
    split, buf = _split("supermimic", "AL", task)
    ref, rbuf = _agent("supermimic", task)
    _turn(rbuf, buf.ptr)
    torch.manual_seed(123)
    _run_split(split, buf, synthetic_motion)
    torch.manual_seed(123)
    _run_one_gpu(ref, rbuf, synthetic_motion)
    assert int(ref.pi_avg.n_averaged) == ITERS == int(split.learner.pi_avg.n_averaged) + 1
    assert any(not torch.equal(p, q) for p, q in zip(ref.pi.parameters(), split.learner.pi.parameters()))


@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_policy_update_on_a_given_batch_is_the_update_on_its_draw(eager, in_place_norms, task, name):
    """The learner's update takes its batch from GivenStarts in place of its own draw from a replay buffer: on the batch
    the draw would give, and from the random stream's position after that draw, it is the update on the buffer bit for bit.
    Given anything else as its buffer, or another batch size, it raises."""
    from ndm_gym.algos.layouts import GivenStarts

    (a0, b0), (a1, b1) = (_agent(name, task) for _ in range(2))
    B, Hh = a0.args.pi_batch, a0.args.hist
    torch.manual_seed(5)
    a0.policy_update_graphable(b0)
    torch.manual_seed(5)
    given = GivenStarts(b1, B, Hh, "cpu")
    given.fill(*a1.sample_valid_starts_nosync(b1, B))
    a1.sample_valid_starts_nosync = given.starts
    a1.policy_update_graphable(given)
    for p, q in zip(a0.pi.parameters(), a1.pi.parameters()):
        torch.testing.assert_close(q, p, **EXACT)
    for p, q in zip(a0.pi_avg.module.parameters(), a1.pi_avg.module.parameters()):
        torch.testing.assert_close(q, p, **EXACT)
    with pytest.raises(ValueError):
        a1.policy_update_graphable(b1)
    with pytest.raises(ValueError):
        given.starts(given, B + 1)


@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_every_exchange_leaves_each_object_with_what_its_next_step_reads(eager, in_place_norms, task, synthetic_motion, name, layout):
    """_check_split_run on the CPU, the updates as plain functions: the checks the GPUs' run makes (below)."""
    split, buf = _split(name, layout, task)
    _check_split_run(split, buf, synthetic_motion, graphs=False, iters=4)


@needs_isaaclab
def test_train_calls_the_split_in_the_order_these_tests_run_it():
    """ndm_gym.train's loop under a layout is the one _run_split mirrors (its source, since the loop itself needs the
    simulator): the split is built before the warm-up collection and its graphs after it; an iteration collects, updates
    the world model, updates the policy once the learner holds a batch, then exchanges, before the log reads the devices."""
    import inspect

    from ndm_gym import train

    src = inspect.getsource(train.main)
    marks = ["split = Split(args, algo, algo_on, buffer)", "collect(args.warmup_steps, stats)", "split.update_graphs()", "collect(n, stats",
             "split is None or split.ready", "= policy_update_g()", "split.exchange()", "if log_now:"]  # fmt: skip
    assert all(src.count(m) == 1 for m in marks), [m for m in marks if src.count(m) != 1]
    at = [src.index(m) for m in marks]
    assert at == sorted(at), marks
    wm = [i for i in range(len(src)) if src.startswith("res = wm_update_g()", i)]  # one update at a time and --wm_presample's (refused with a layout)
    assert len(wm) == 2 and all(at[3] < i < at[4] for i in wm), wm
    assert src.index("GraphedCollector(env, algo,") < src.index("algo = split.algo")  # the collection's graph keeps the actor's object


# ---------------------------------------------------------------- what crosses the devices
def test_parcel_round_trip():
    """A parcel copies every source into its destination, in place and exactly, whatever the dtypes and shapes; later
    changes of a source do not reach the destination before the next send; mismatched entries are refused."""
    from ndm_gym.algos.layouts import Parcel

    g = torch.Generator().manual_seed(0)
    src = {"w": torch.randn(7, 5, generator=g), "b": torch.randn(5, generator=g), "count": torch.tensor(3.0),
           "f": torch.randint(0, 99, (4, 3), generator=g), "step": torch.tensor(7, dtype=torch.int32), "done": torch.rand(6, generator=g) < 0.5,
           "half": torch.randn(2, 2, generator=g).to(torch.bfloat16)}  # fmt: skip
    dst = {k: torch.zeros_like(v) for k, v in src.items()}
    ids = {k: id(v) for k, v in dst.items()}
    parcel = Parcel({k: (src[k], dst[k]) for k in src})
    assert parcel.names == tuple(src) == tuple(parcel.pairs) and len(parcel.groups) == 5  # one message per dtype
    assert sorted(len(items) for items, *_ in parcel.groups) == [1, 1, 1, 1, 3]  # the three float32 tensors travel together
    parcel.send()
    assert all(torch.equal(dst[k], src[k]) and dst[k].dtype == src[k].dtype and id(dst[k]) == ids[k] for k in src)
    kept = {k: v.clone() for k, v in dst.items()}
    src["w"].mul_(2), src["f"].add_(1), src["done"].logical_not_()
    assert all(torch.equal(dst[k], kept[k]) for k in src)
    parcel.send()
    assert all(torch.equal(dst[k], src[k]) for k in src) and not torch.equal(dst["w"], kept["w"])
    p = torch.nn.Parameter(torch.zeros(7, 5))  # into a parameter, as the weights go
    Parcel({"w": (src["w"], p)}).send()
    assert torch.equal(p, src["w"]) and p.requires_grad
    q = torch.nn.Parameter(torch.randn(7, 5, generator=g))  # and out of one
    Parcel({"w": (q, dst["w"])}).send()
    assert torch.equal(dst["w"], q) and not dst["w"].requires_grad
    bad = ({"w": (src["w"], torch.zeros(5, 7))}, {"w": (src["w"], torch.zeros(7, 5, dtype=torch.float64))}, {"w": (src["w"], src["w"])}, {})
    for pairs in bad:
        with pytest.raises(ValueError):
            Parcel(pairs)


@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_parcels_carry_the_policy_the_world_model_and_the_batch(eager, in_place_norms, task, name, layout):
    """Every entry of the policy's state dict crosses in exactly one direction (weights learner -> actor, normalizers actor
    -> learner), every entry of the world model's reaches the learner, the batch is ReplayBuffer.starts_at's fields, and
    AWL's replica gets every replay field with the head and the fill level; no parcel carries anything else. The parcels
    hold the modules' own tensors, which the updates write in place."""
    from ndm_gym.algos.layouts import START_KEYS

    split, buf = _split(name, layout, task)
    actor, wmer, learner = split.actor, split.wmer, split.learner
    pi, wm = actor.pi, actor.wm
    weights, norms = {f"pi.{k}" for k, _ in pi.named_parameters()}, {f"pi.{k}" for k, _ in pi.named_buffers()}
    assert weights | norms == {f"pi.{k}" for k in pi.state_dict()} and not weights & norms and len(norms) == 6
    wm_all = {f"wm.{k}" for k in wm.state_dict()}
    wm_norms = {f"wm.{k}" for k, _ in wm.named_buffers()}
    assert wm_norms and wm_norms < wm_all
    batch = {f"batch.{part}.{k}" for part in ("s", "h") for k in START_KEYS}
    assert set(split.to_actor.names) == weights
    if layout == "AL":
        assert set(split.to_learner.names) == batch | norms | wm_all
        assert split.replica is None and wmer is actor
    else:
        assert set(split.to_learner.names) == batch | norms
        assert set(split.wm_to_learner.names) == wm_all and set(split.wm_norms.names) == wm_norms
        assert set(split.rows.names) == {f"rows.{k}" for k in buf.fields} and set(split.heads.names) == {"ptr", "size"}
        assert wmer is not actor and wmer is not learner
    s, h = actor.sample_valid_starts_nosync(buf, split.args.pi_batch)
    assert set(s) == set(h) == set(START_KEYS)
    assert all(split.given.s[k].shape == s[k].shape and split.given.s[k].dtype == s[k].dtype for k in s)
    assert all(split.given.h[k].shape == h[k].shape and split.given.h[k].dtype == h[k].dtype for k in h)

    def ends(parcel, prefix):  # the ids of a parcel's sources and destinations under a name prefix
        sent = {(id(src), id(dst)) for items, *_ in parcel.groups for src, dst in items}  # what send() copies: every pair, once
        assert sent == {(id(src), id(dst)) for src, dst in parcel.pairs.values()} and len(sent) == len(parcel.pairs)
        return ({id(src) for k, (src, _) in parcel.pairs.items() if k.startswith(prefix)},
                {id(dst) for k, (_, dst) in parcel.pairs.items() if k.startswith(prefix)})  # fmt: skip

    ids = lambda ts: {id(t) for t in ts}  # noqa: E731
    assert ends(split.to_actor, "pi.") == (ids(learner.pi.parameters()), ids(actor.pi.parameters()))
    assert ends(split.to_learner, "pi.") == (ids(actor.pi.buffers()), ids(learner.pi.buffers()))
    both = lambda m: ids(list(m.parameters()) + list(m.buffers()))  # noqa: E731
    if layout == "AL":
        assert ends(split.to_learner, "wm.") == (both(actor.wm), both(learner.wm))
    else:
        assert ends(split.wm_to_learner, "wm.") == (both(wmer.wm), both(learner.wm))
        assert ends(split.wm_norms, "wm.") == (ids(actor.wm.buffers()), ids(wmer.wm.buffers()))
        assert ends(split.heads, "") == ({id(buf.ptr_t), id(buf.size_t)}, {id(split.replica.ptr_t), id(split.replica.size_t)})


@pytest.mark.parametrize("ptr, size", [(T - 4, T), (5, 5), (0, T)])
def test_replica_follows_the_actors_replay(eager, in_place_norms, task, ptr, size):
    """AWL's replica through forty iterations of zero to three new rows (--utd 0.4: at most three), from a full replay with
    its head near the end and from a replay that is still filling: after every hand-over it is the actor's replay, head
    and fill level included, and windows drawn from it are the actor's. More rows than an iteration can bring are refused,
    as is a replay shorter than an iteration's rows."""
    split, buf = _split("supermimic", "AWL", task)
    _turn(buf, ptr, size)
    split.update_graphs(graphs=False)  # the replica starts as the actor's replay
    _replica_is_the_replay(split, buf)
    g = torch.Generator().manual_seed(3)

    def add():
        row = {}
        for k, (shape, dt) in buf.fields.items():
            shp = (N,) + ((shape,) if isinstance(shape, int) else shape)
            if dt.is_floating_point:
                row[k] = torch.randn(shp, generator=g)
            elif dt == torch.bool:
                row[k] = torch.rand(shp, generator=g) < 0.3
            else:
                row[k] = torch.randint(0, 50, shp, generator=g).to(dt)
        buf.add_device(row)
        buf.advance_host()

    assert split.n_max == 3
    for k in range(40):
        for _ in range((2, 3, 0, 1)[k % 4]):
            add()
        split._send_rows()
        _replica_is_the_replay(split, buf)
    assert buf.size == T and buf.ptr != ptr  # full by now, the head has gone around
    torch.manual_seed(11)
    w0 = buf.sample_windows(16, 8)
    torch.manual_seed(11)
    w1 = split.replica.sample_windows(16, 8)
    assert w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0)
    for _ in range(4):
        add()
    with pytest.raises(RuntimeError, match="more than 3 collection steps"):
        split._send_rows()
    with pytest.raises(ValueError, match="collection steps per iteration"):
        _split("supermimic", "AWL", task, utd=0.02)  # fifty steps per iteration, a replay of forty-eight


@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
def test_every_module_starts_from_the_actors(eager, in_place_norms, task, name):
    """copy_modules brings every module state of the actor's object (networks, normalizers, policy average, SuperTrack's
    balancers) into another device's, whose own initialization differs."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.layouts import copy_modules

    actor, _ = _agent(name, task)
    torch.manual_seed(99)
    other = ALGOS[name](actor.args, *task("cpu"), "cpu")
    assert any(not torch.equal(p, q) for p, q in zip(actor.pi.parameters(), other.pi.parameters()))
    names = copy_modules(actor, other)
    assert {"wm", "pi", "pi_avg"} <= set(names) and ("critic" in names) == (name == "supermimic") and ("bal_pi" in names) == (name == "supertrack")
    mods = [k for k, v in vars(actor).items() if isinstance(v, torch.nn.Module)]
    assert sorted(names) == sorted(mods)
    for k in mods:
        a, b = getattr(actor, k).state_dict(), getattr(other, k).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[x], b[x]) for x in a), k


# ---------------------------------------------------------------- what the training loop reads
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_the_loop_reads_each_quantity_from_its_owner(eager, in_place_norms, task, name, layout):
    """SplitAlgo: the optimizers train.py sets the learning rates of are the live ones (the world model's from its owner,
    the policy's and the critic's from the learner), each with its own device learning rate; the checkpoint state has an
    algorithm's keys, the world model's parts from its owner and the rest from the learner; anything else is refused."""
    from test_systems import trainer_learning_rates

    from ndm_gym.algos.blinkmimic import PolicyLR

    split, buf = _split(name, layout, task)
    algo, actor, wmer, learner = split.algo, split.actor, split.wmer, split.learner
    assert algo.optimizers[0] is wmer.opt_wm is algo.opt_wm and algo.optimizers[1] is learner.opt_pi is algo.opt_pi
    assert len(algo.optimizers) == len(actor.optimizers) and all(o is p for o, p in zip(algo.optimizers[1:], learner.optimizers[1:]))
    assert algo.pi is learner.pi and algo.pi_avg is learner.pi_avg and algo.wm is wmer.wm and algo.nonfinite is learner.nonfinite
    assert algo.residual == actor.residual and algo.log_lr(1e-3) == actor.log_lr(1e-3) and algo.fail_starts == 0
    # train.py's learning rates reach the live optimizers' device tensors, and only those
    lrs = dict(lr=2.0**-10, wm_lr=2.0**-9, pi_lr_mult=4.0, lr_decay_ret=0.0, lr_decay_len=2, lr_decay_by=2)
    args = argparse.Namespace(**{**vars(actor.args), **lrs})
    every = {id(o): o for obj in (actor, wmer, learner) for o in obj.optimizers}  # each object's own, once
    unused = [o for o in every.values() if not any(o is live for live in algo.optimizers)]
    assert len(unused) == len(actor.optimizers) * (1 if layout == "AL" else 2)
    stale = [float(g["lr"]) for o in unused for g in o.param_groups]
    trainer_learning_rates(algo, args, PolicyLR(args), 0)
    assert {float(g["lr"]) for g in wmer.opt_wm.param_groups} == {2.0**-9} and {float(g["lr"]) for g in learner.opt_pi.param_groups} == {2.0**-8}
    assert all({float(g["lr"]) for g in o.param_groups} == {2.0**-10} for o in algo.optimizers[2:])
    assert [float(g["lr"]) for o in unused for g in o.param_groups] == stale
    # the checkpoint state
    with torch.no_grad():
        for p in wmer.wm.parameters():
            p.add_(1.0)
        for p in learner.pi.parameters():
            p.sub_(1.0)
        if name == "supertrack":
            wmer.bal_wm.w.fill_(3.0), learner.bal_pi.w.fill_(5.0)
    sd = algo.state_dict()
    assert sd.keys() == actor.state_dict().keys()
    assert all(torch.equal(v, wmer.wm.state_dict()[k]) for k, v in sd["wm"].items())
    assert all(torch.equal(v, learner.pi.state_dict()[k]) for k, v in sd["pi_live"].items())
    assert all(torch.equal(v, learner.pi_avg.module.state_dict()[k]) for k, v in sd["pi"].items())
    assert not all(torch.equal(v, actor.pi.state_dict()[k]) for k, v in sd["pi_live"].items())
    if name == "supermimic":
        assert all(torch.equal(v, learner.critic.state_dict()[k]) for k, v in sd["critic"].items())
    else:
        assert bool((sd["bal_wm"]["w"] == 3.0).all()) and bool((sd["bal_pi"]["w"] == 5.0).all())
    with pytest.raises(AttributeError, match="--layout"):
        algo.wm_update
    with pytest.raises(AttributeError, match="--layout"):
        algo.policy_update


def test_split_refuses_what_it_cannot_run(eager, in_place_norms, task, monkeypatch):
    """Too few GPUs, devices that are not the actor's first, a replay without its head on the device, normalizers that are
    not written in place, a one-GPU layout, another algorithm on a device, a policy update before the first exchange, an
    exchange before the graphs: each raises."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.layouts import Split
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm

    actor, buf = _agent("supermimic", task, layout="AL")
    algo_on = lambda d: ALGOS["supermimic"](actor.args, *task(d), d)  # noqa: E731
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="needs 2 GPUs, the trainer sees 1"):
        Split(actor.args, actor, algo_on, buf)
    for devices in (("cpu",), ("cpu", "cpu", "cpu"), ("cuda:1", "cpu")):
        with pytest.raises(ValueError, match="devices"):
            Split(actor.args, actor, algo_on, buf, devices=devices)
    with pytest.raises(ValueError, match="graph_early"):
        Split(actor.args, actor, algo_on, ReplayBuffer(4, 2, "cpu", priorities=False), devices=("cpu", "cpu"))
    with monkeypatch.context() as m:  # the default normalizer update replaces its tensors
        m.setattr(RunningNorm, "no_sync", False)
        with pytest.raises(ValueError, match="written in place"):
            Split(actor.args, actor, algo_on, buf, devices=("cpu", "cpu"))
    one = argparse.Namespace(**{**vars(actor.args), "layout": "1GPU"})
    with pytest.raises(ValueError, match="not a split layout"):
        Split(one, actor, algo_on, buf, devices=("cpu",))
    with pytest.raises(ValueError, match="algo_on built"):
        Split(actor.args, actor, lambda d: ALGOS["supertrack"](actor.args, *task(d), d), buf, devices=("cpu", "cpu"))
    split = Split(actor.args, actor, algo_on, buf, devices=("cpu", "cpu"))
    with pytest.raises(RuntimeError, match="before Split.update_graphs"):
        split.exchange()
    _, policy_update = split.update_graphs(graphs=False)
    with pytest.raises(RuntimeError, match="first batch"):
        policy_update()


# ---------------------------------------------------------------- devices: the graphs' and the run-ahead limit's handling
def test_updates_run_with_their_device_current_and_capture_on_its_stream(monkeypatch):
    """OnDevice makes its device torch's current CUDA device for the call (a CUDA graph's warm-up stream and capture are the
    current device's) and is transparent on the CPU; Graphed hands its capture stream to torch.cuda.graph, torch's default
    one (None) unless given."""
    from ndm_gym.algos import blinkmimic
    from ndm_gym.algos.layouts import OnDevice

    entered = []

    @contextlib.contextmanager
    def device(idx):
        entered.append(idx)
        yield
        entered.append(-1)

    monkeypatch.setattr(torch.cuda, "device", device)
    assert OnDevice(lambda: "x", "cpu")() == "x" and entered == []
    assert OnDevice(lambda: list(entered), "cuda:2")() == [2] and entered == [2, -1]
    assert OnDevice(lambda: 1, torch.device("cuda"))() == 1 and entered[-2:] == [0, -1]
    seen = []

    class Stream:
        def wait_stream(self, other):
            pass

    @contextlib.contextmanager
    def graph(g, stream=None):
        seen.append(stream)
        yield

    @contextlib.contextmanager
    def on_stream(s):
        yield

    fake = dict(Stream=Stream, current_stream=lambda *a: Stream(), stream=on_stream, synchronize=lambda *a: None, empty_cache=lambda: None,
                CUDAGraph=lambda: type("G", (), {"replay": lambda self: None})(), graph=graph, memory_reserved=lambda *a: 0)  # fmt: skip
    for k, v in fake.items():
        monkeypatch.setattr(torch.cuda, k, v)
    mine = Stream()
    for stream in (None, mine):
        g = blinkmimic.Graphed(lambda: (torch.ones(2),), lambda: True, "a test update", stream=stream)
        for _ in range(5):  # three eager calls, the capture and its replay, a replay
            assert torch.equal(g()[0], torch.ones(2))
    assert seen == [None, mine]
    assert blinkmimic.Graphed(lambda: (), lambda: True, "the default").stream is None


@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_each_update_is_a_graph_of_its_own_device(eager, in_place_norms, task, monkeypatch, layout):
    """Split.update_graphs wraps the draw, the world-model update and the policy update each for its object's device, with
    a capture stream of that device, one per device (the objects are told here that they are on GPUs 0, 1 and 2; the
    streams are stand-ins, and nothing is called)."""
    from ndm_gym.algos.blinkmimic import Graphed
    from ndm_gym.algos.layouts import OnDevice

    class Stream:
        def __init__(self, device=None):
            self.device = device

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    split, _ = _split("supermimic", layout, task)
    objects = (split.actor, split.learner) if layout == "AL" else (split.actor, split.wmer, split.learner)
    for i, o in enumerate(objects):
        o.dev = f"cuda:{i}"
    wm_update, policy_update = split.update_graphs()
    calls = {"draw": split.draw, "wm": wm_update, "pi": policy_update}
    assert all(isinstance(c, OnDevice) and isinstance(c.fn, Graphed) for c in calls.values())
    want = {"draw": 0, "wm": 0 if layout == "AL" else 1, "pi": len(objects) - 1}
    assert {k: c.index for k, c in calls.items()} == want
    assert all(c.fn.stream.device == torch.device(f"cuda:{want[k]}") for k, c in calls.items())
    assert (calls["draw"].fn.stream is calls["wm"].fn.stream) == (layout == "AL") and calls["pi"].fn.stream is not calls["draw"].fn.stream
    assert [c.fn.name for c in calls.values()] == ["the draw of the imagination batch", "the world-model update", "the policy update"]
    assert calls["pi"].fn.ready() is False and calls["wm"].fn.ready() and calls["draw"].fn.ready()  # the average's first step is eager
    if layout == "AL":
        assert split.describe() == ("--layout AL: actor cuda:0 (simulator, collection, replay, world-model updates) | "
                                    "policy learner cuda:1 (policy update, average, critic)")  # fmt: skip
    else:
        assert split.describe() == ("--layout AWL: actor cuda:0 (simulator, collection, replay) | world-model learner cuda:1 (world-model updates "
                                    "on a replica of the replay) | policy learner cuda:2 (policy update, average, critic)")  # fmt: skip


def test_run_ahead_limit_waits_for_the_previous_iteration(monkeypatch):
    """RunAhead.step waits for the events of the call before, then records one at the end of each CUDA device's queue: the
    host is at most one iteration ahead of the devices. Devices are counted once; nothing on the CPU."""
    from ndm_gym.algos.layouts import RunAhead

    log = []

    class Event:
        def record(self, stream):
            self.at = stream
            log.append(("record", stream))

        def synchronize(self):
            log.append(("wait", self.at))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d: str(d))
    cpu = RunAhead(("cpu", "cpu"))
    cpu.step(), cpu.step()
    assert log == [] and cpu.devices == []
    limit = RunAhead(("cuda:0", "cuda:1", "cuda:1"))
    assert [str(d) for d in limit.devices] == ["cuda:0", "cuda:1"]
    limit.step()
    assert log == [("record", "cuda:0"), ("record", "cuda:1")]
    limit.step()
    assert log[2:] == [("wait", "cuda:0"), ("wait", "cuda:1"), ("record", "cuda:0"), ("record", "cuda:1")]


# ---------------------------------------------------------------- GPUs: the updates as CUDA graphs on their devices
def _gpus(layout):
    from ndm_gym.algos.xmimic import LAYOUTS

    if GPUS < LAYOUTS[layout]:
        pytest.skip(f"--layout {layout} needs {LAYOUTS[layout]} GPUs, {GPUS} visible")
    return tuple(f"cuda:{i}" for i in range(LAYOUTS[layout]))


@pytest.mark.slow
@pytest.mark.skipif(GPUS < 2, reason="CUDA graphs on several devices and copies between them need two GPUs (AWL: three)")
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_split_on_gpus_runs_its_graphs_on_their_devices(in_place_norms, task, synthetic_motion, name, layout):
    """On the layout's GPUs, eight iterations with every update a CUDA graph on its device (Graphed: three calls on a side
    stream, then captured on a stream of that device and replayed), the step functions compiled as the trainer runs them
    (a capture takes no copy from the host's memory, and SuperTrack's tracking terms build a constant there that only the
    compiled function folds): _check_split_run's checks hold after every exchange, so each graph ran on its device and
    each copy between devices waited for the work before it. The graphs were captured, each on its own device's stream;
    every object's tensors are on its device; the current device is the actor's again."""
    devices = _gpus(layout)
    split, buf = _split(name, layout, task, devices)
    where = lambda m: next(m.parameters()).device  # noqa: E731
    assert where(split.actor.pi) == torch.device(devices[0]) == buf.data["s"].device == split.out.s["s"].device
    assert where(split.learner.pi) == torch.device(devices[-1]) == split.given.s["s"].device == where(split.learner.wm)
    assert where(split.wmer.wm) == torch.device(devices[1] if layout == "AWL" else devices[0])
    assert split.replica is None or split.replica.data["s"].device == torch.device(devices[1])
    wm_update, policy_update, _ = _check_split_run(split, buf, synthetic_motion, graphs=True)
    calls = {"draw": (split.draw, devices[0]), "wm": (wm_update, devices[1] if layout == "AWL" else devices[0]), "pi": (policy_update, devices[-1])}
    for k, (call, dev) in calls.items():
        assert call.fn.graph is not None and call.fn.stream.device == torch.device(dev), k
    assert torch.cuda.current_device() == 0


@pytest.mark.slow
@pytest.mark.skipif(GPUS < 2, reason="CUDA graphs on several devices and copies between them need two GPUs (AWL: three)")
@pytest.mark.parametrize("name", ["supermimic", "supertrack"])
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_split_on_gpus_with_graphs_is_the_eager_split(in_place_norms, task, synthetic_motion, name, layout):
    """On the layout's GPUs, the split loop with its updates as CUDA graphs against the same loop with the same (compiled)
    functions called outside a graph, the random streams set anew at every iteration: the networks after every iteration
    are equal bit for bit, as a captured update is its eager twin on one GPU (test_wm_muon_rf's world-model update). This
    rests on the updates being deterministic on these GPUs; the first iteration that differs is named."""
    devices = _gpus(layout)
    runs = {}
    for graphs in (False, True):
        split, buf = _split(name, layout, task, devices)
        runs[graphs] = _check_split_run(split, buf, synthetic_motion, graphs=graphs)[2]
    for k, (plain, graphed) in enumerate(zip(runs[False], runs[True])):
        assert plain.keys() == graphed.keys()
        for key, v in plain.items():
            torch.testing.assert_close(graphed[key], v, **EXACT, msg=lambda m, k=k, key=key: f"iteration {k}, {key}: {m}")


# ---------------------------------------------------------------- arguments
def _layout_args(**kw):
    a = dict(eprobe_gpu=-1, eprobe_share=0.0, njmax=0, gc_freeze=False, layout="AL", algo="supermimic", wm_backbone="mlp", paired=False, resume=None,
             pi_updates=1, no_sync=True, update_graphs=True, graph_early=True, collect_graph=True, reset_masked=True,
             pi_encode_once=False, imag_hoist=False, pi_deferred_grads=False, wm_deferred_grads=False, deferred_bf16=False, muon_batched=False,
             wm_presample=False, pi_cudagraphs=False, wm_cudagraphs=False, pi_rollout_clip=0.0, amp="bf16", pi_opt="muon", wm_opt="muon",
             wm_sample="uniform", wm_fail_windows=0.0)  # fmt: skip
    a.update(kw)
    return argparse.Namespace(**a)


def test_layout_is_off_by_default_and_parses():
    from ndm_gym.algos.xmimic import LAYOUTS, add_xmimic_args, check_args

    p = argparse.ArgumentParser()
    add_xmimic_args(p)
    assert p.parse_args([]).layout == "1GPU" and LAYOUTS == {"1GPU": 1, "AL": 2, "AWL": 3}
    assert p.parse_args(["--layout", "AWL"]).layout == "AWL"
    with pytest.raises(SystemExit):
        p.parse_args(["--layout", "AWLP"])  # XMimic's prober GPU is --eprobe_gpu
    for layout in ("AL", "AWL"):
        for algo in ("supermimic", "supertrack"):
            check_args(_layout_args(layout=layout, algo=algo))
            check_args(_layout_args(layout=layout, algo=algo, paired=True))  # the two-robot scene: the split does not see the pairing
    check_args(_layout_args(layout="1GPU", no_sync=False, update_graphs=False, graph_early=False, collect_graph=False, reset_masked=False,
                            algo="other", wm_backbone="rigidformer", paired=True, resume="x.pt", pi_updates=2))  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [dict(no_sync=False), dict(update_graphs=False), dict(graph_early=False), dict(collect_graph=False), dict(reset_masked=False), dict(algo="other"),
     dict(wm_backbone="rigidformer"), dict(resume="last.pt"), dict(pi_updates=2), dict(wm_presample=True),
     dict(wm_deferred_grads=True), dict(muon_batched=True), dict(pi_encode_once=True, imag_hoist=True),
     dict(pi_encode_once=True, imag_hoist=True, pi_deferred_grads=True, deferred_bf16=True)],
)  # fmt: skip
@pytest.mark.parametrize("layout", ["AL", "AWL"])
def test_layouts_refuse_what_they_do_not_cover(kw, layout):
    from ndm_gym.algos.xmimic import check_args

    with pytest.raises(ValueError, match=f"--layout {layout}"):
        check_args(_layout_args(layout=layout, **kw))


@needs_isaaclab
def test_train_takes_a_layout_with_the_recipe():
    """ndm_gym.train's arguments: --algo blinkmimic (whose recipe has the systems options) and SuperTrack with them take
    --layout AL and AWL, with the probe worker on a GPU of its own (--eprobe_gpu: XMimic's ALP and AWLP); without the
    systems options, with the RigidFormer or with --resume the layout is refused."""
    from ndm_gym.algos import blinkmimic, xmimic
    from ndm_gym.train import add_speed_args, add_train_args

    parser = argparse.ArgumentParser()
    add_train_args(parser), add_speed_args(parser), blinkmimic.add_blinkmimic_args(parser), xmimic.add_xmimic_args(parser)

    def check(argv):
        a = parser.parse_args(blinkmimic.expand_preset(argv))
        blinkmimic.check_args(a), xmimic.check_args(a)
        return a

    assert check(["--algo", "blinkmimic"]).layout == "1GPU"
    systems = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked"]
    for layout, gpu in (("AL", 2), ("AWL", 3)):
        a = check(["--algo", "blinkmimic", "--layout", layout, "--eprobe_gpu", str(gpu)])
        assert (a.layout, a.algo, a.eprobe_gpu, a.pi_updates, a.utd) == (layout, "supermimic", gpu, 1, 0.4)
        assert check(["--algo", "supertrack", "--layout", layout, *systems]).layout == layout
        for bad in (["--algo", "supermimic", "--layout", layout], ["--algo", "supertrack", "--layout", layout, *systems[:-1]],
                    ["--algo", "blinkmimic", "--layout", layout, "--resume", "last.pt"],
                    ["--algo", "supertrack", "--layout", layout, *systems, "--wm_backbone", "rigidformer"],
                    ["--algo", "blinkmimic", "--layout", layout, "--pi_updates", "2"],
                    ["--algo", "blinkmimic", "--layout", layout, "--wm_presample"],
                    ["--algo", "blinkmimic", "--layout", layout, "--pi_encode_once", "--imag_hoist"]):  # fmt: skip
            with pytest.raises(ValueError, match="--layout|--reset_masked needs|--collect_graph"):
                check(bad)
        assert check(["--algo", "blinkmimic", "--layout", layout, "--pi_encode_once"]).pi_encode_once
