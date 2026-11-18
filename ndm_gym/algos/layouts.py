"""XMimic's iteration split across GPUs for ndm_gym.train (``--layout``; docs/algorithms/xmimic.md, section 2).

On one GPU an iteration takes the sum of its steps: the collection, the world-model updates, the policy update. XMimic
(N. Bohlinger and J. Peters, 2026; xmimic/multi_gpu.py) runs them at the same time on separate GPUs, each on the others'
results of the previous iteration, so that an iteration takes about its longest step. Here, in one process and one host
thread:

- ``AL``, two GPUs. Actor (the first GPU the trainer sees): the simulator, the collection, the replay buffer, the world
  model with its updates, the draw of the next imagination batch. Policy learner (the second): the policy update with the
  policy average and the critic.
- ``AWL``, three GPUs. The world-model updates move to the second GPU, on a replica of the replay buffer that receives
  the rows the actor wrote; the policy learner is the third.

Every device has an object of the run's algorithm (``Split``): the actor's is train.py's own, the others are built from
the same arguments and start from its networks. The collection, the world-model update and the policy update stay the
functions the one-GPU loop captures as CUDA graphs (``--collect_graph``, ``--update_graphs``); a graph's replay returns
at once, so the host queues the three devices' work back to back and they run side by side.

What iteration i computes (the loop's order: collection, world-model updates, policy update, then ``exchange``):

- the collection acts with the policy as iteration i - 1's policy update left it, as on one GPU (iteration i - 1's
  exchange brought it to the actor);
- AL's world-model updates follow the collection on the actor, as on one GPU; AWL's run on the replica, which holds the
  rows up to iteration i - 1's collection, with the normalizers as that collection left them;
- the policy update imagines from the batch the actor drew at the end of iteration i - 1, in the world model as iteration
  i - 1's updates left it (AWL: iteration i - 1's updates, which saw the rows up to iteration i - 2), with the policy's
  normalizers of iteration i - 1. On one GPU all three would be iteration i's. The run's first iteration has no batch
  yet and makes no policy update, so every later one has made one update fewer than on one GPU.

This is XMimic's staleness (its layouts AL and AWL): the actor collects with the policy of the learner's previous
iteration, and the learner updates on the actor's batch and world model of the previous iteration.

``exchange`` ends the iteration. The actor draws the next imagination batch; then, as in-place copies into the fixed
tensors the other device's captured updates read (``Parcel``: packed on the source, one device-to-device copy per dtype,
unpacked on the destination; PyTorch orders a copy between two GPUs after the work queued on both, so every copy carries
what the iteration computed and no step of the next iteration starts before its inputs have arrived):

- actor -> learner: the batch (start states and their observation histories' states), the policy's normalizers, and in
  AL the world model (weights and normalizers);
- AWL, world-model learner -> learner: the world model (weights, and the normalizers its updates used); then actor ->
  world-model learner: the replay rows of the iteration, the write head and fill level, the world model's normalizers;
- learner -> actor: the policy's weights.

The copies are exact (float32 as it is, where XMimic sends the weights as bf16), so the actor's policy is the learner's
bit for bit. The exchange also runs in the iterations --wm_only_iters keeps for the world model alone.

The split itself waits for the devices once per iteration, and only for the work of the iteration before
(``RunAhead``), so that the host queues an iteration while the previous one runs. The collection schedule, the
learning-rate schedule with its --lr_decay_ret trigger, the logging and the checkpoints stay train.py's loop on the
host; the failure-focused resets stay in the simulator on the actor; the probe sweeps read the policy average, which
lives on the learner (``SplitAlgo`` gives the loop each quantity from the object that holds it).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..core.normalizer import RunningNorm
from .xmimic import LAYOUTS

START_KEYS = ("s", "a_prev", "xi", "f")  # ReplayBuffer.starts_at's fields
WM_STATE = ("wm", "opt_wm", "bal_wm")  # an algorithm's state_dict entries that live with the world model's owner


class Parcel:
    """Tensors of one device copied into their counterparts on another, in place: packed into one flat vector per dtype
    on the source, one device-to-device copy each, unpacked on the destination. ``pairs``: name -> (source, destination),
    of equal shape and dtype; all sources on one device, all destinations on one. The tensors are held, not looked up
    again: they must be written in place (--no_sync's normalizers, the optimizers' steps)."""

    def __init__(self, pairs: dict):
        if not pairs:
            raise ValueError("an empty parcel")
        self.pairs, self.names = dict(pairs), tuple(pairs)
        groups: dict = {}
        for name, (src, dst) in pairs.items():
            if src.shape != dst.shape or src.dtype != dst.dtype:
                raise ValueError(f"parcel entry {name}: {tuple(src.shape)} {src.dtype} cannot go into {tuple(dst.shape)} {dst.dtype}")
            if src is dst:
                raise ValueError(f"parcel entry {name}: source and destination are one tensor")
            groups.setdefault(src.dtype, []).append((src, dst))
        first = next(iter(pairs.values()))
        self.src_device, self.dst_device = first[0].device, first[1].device
        if any(s.device != self.src_device or d.device != self.dst_device for s, d in pairs.values()):
            raise ValueError("a parcel goes from one device to one device")
        self.groups = []
        for dtype, items in groups.items():
            sizes = [s.numel() for s, _ in items]
            out = torch.zeros(sum(sizes), dtype=dtype, device=self.src_device)
            self.groups.append((items, sizes, out, torch.zeros(sum(sizes), dtype=dtype, device=self.dst_device)))

    @torch.no_grad()
    def send(self):
        for items, sizes, out, box in self.groups:
            torch.cat([s.reshape(-1) for s, _ in items], out=out)
            box.copy_(out, non_blocking=True)
            for (_, d), piece in zip(items, box.split(sizes)):
                d.copy_(piece.view_as(d))


class OnDevice:
    """``fn`` called with ``device`` as torch's current CUDA device: a CUDA graph's warm-up stream, capture and memory
    report (algos.blinkmimic.Graphed) are the current device's. As is on the CPU."""

    def __init__(self, fn, device):
        d = torch.device(device)
        self.fn, self.index = fn, (d.index or 0) if d.type == "cuda" else None

    def __call__(self):
        if self.index is None:
            return self.fn()
        with torch.cuda.device(self.index):
            return self.fn()


class RunAhead:
    """The loop's one synchronization with the devices per iteration: ``step`` waits for the events the previous call
    recorded at the end of every CUDA device's queue, then records new ones. The host so queues an iteration's work while
    the previous one runs, and no more (XMimic's run-ahead limit). Nothing to wait for on the CPU."""

    def __init__(self, devices):
        self.devices = [d for d in dict.fromkeys(torch.device(x) for x in devices) if d.type == "cuda"]
        self.events: list = []

    def step(self):
        for e in self.events:
            e.synchronize()
        self.events = []
        for d in self.devices:
            e = torch.cuda.Event()
            e.record(torch.cuda.current_stream(d))
            self.events.append(e)


class LayoutTimer:
    """``LAYOUT_PROFILE=1`` (train.py with --layout): every ``every``-th iteration, CUDA events on each device's current stream
    around the iteration's steps (the collection on the actor, the world-model updates on their owner, the policy update on
    the learner, the exchange on every device) and at the iteration's start; after the exchange the devices are synchronized
    and one line is printed: per device, the milliseconds of its steps, its span (from the iteration's start to the end of its
    part of the exchange, as its stream reached them) and its idle time, the span less its steps. The events add no work; the
    synchronization stalls the host's run-ahead once every ``every`` iterations."""

    def __init__(self, split, every: int = 20):
        self.devs = [torch.device(d) for d in split.devices]
        self.role = {"collect": self.devs[0], "wm": self.devs[1] if split.layout == "AWL" else self.devs[0], "policy": self.devs[-1]}
        self.every, self.ev = every, {}

    def due(self, it: int, after: int) -> bool:
        return it % self.every == 0 and it > after

    def mark(self, name: str, role: str | None = None):
        """An event named ``name`` on the device of ``role`` (None: on every device)."""
        for d in [self.role[role]] if role is not None else self.devs:
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(d))
            self.ev[(name, d)] = e

    def report(self, it: int) -> str:
        """The timed iteration's line (after the exchange's end marks); the events are then dropped."""
        for d in self.devs:
            torch.cuda.synchronize(d)
        ms = lambda a, b, d: self.ev[(a, d)].elapsed_time(self.ev[(b, d)]) if (a, d) in self.ev and (b, d) in self.ev else 0.0  # noqa: E731
        parts = []
        for i, d in enumerate(self.devs):
            steps = {k: ms(a, b, d) for k, a, b in (("collect", "c0", "c1"), ("wm", "w0", "w1"), ("policy", "p0", "p1"), ("exchange", "x0", "x1"))}
            span = ms("start", "x1", d)
            roles = "+".join(k for k in ("collect", "wm", "policy") if self.role[k] == d)
            parts.append(f"GPU{i} ({roles}): " + ", ".join(f"{k} {v:.2f}" for k, v in steps.items()) +
                         f", idle {span - sum(steps.values()):.2f} of {span:.2f} ms")  # fmt: skip
        self.ev = {}
        return f"[prof-layout] it {it} | " + " | ".join(parts)


class GivenStarts:
    """An imagination batch as fixed tensors on ``device``: the start states and their observation histories' states, as
    ``ReplayBuffer.starts_at`` returns them for ``B`` starts. ``starts`` stands in for the learner's own draw
    (``Algo.sample_valid_starts_nosync``), which the policy update calls with what it was given as its replay buffer.
    ``keys``: the replay's fields a start holds; ``hist`` 0: starts without a history (train_hand: the hand agents' draw,
    ``hand_algos.START_KEYS``)."""

    def __init__(self, buffer: ReplayBuffer, B: int, hist: int, device, keys: tuple = START_KEYS):
        def like(lead):
            return {k: torch.zeros(lead + buffer.data[k].shape[2:], dtype=buffer.data[k].dtype, device=device) for k in keys}

        self.B, self.s, self.h = B, like((B,)), like((B, hist)) if hist > 0 else {}

    def starts(self, buffer, B: int):
        if buffer is not self or B != self.B:
            raise ValueError("--layout: the policy learner's update takes the batch the actor drew (Split.update_graphs)")
        return self.s, self.h

    @torch.no_grad()
    def fill(self, s: dict, h: dict):
        if s.keys() != self.s.keys() or h.keys() != self.h.keys():
            raise ValueError(f"--layout: a draw of {sorted(s)} with a history of {sorted(h)}, the batch holds {sorted(self.s)} and {sorted(self.h)}")
        for part, got in ((self.s, s), (self.h, h)):
            for k in part:
                if got[k].shape != part[k].shape or got[k].dtype != part[k].dtype:  # copy_ would broadcast or cast it silently
                    raise ValueError(f"--layout: the draw's {k} is {tuple(got[k].shape)} {got[k].dtype}, the batch's {tuple(part[k].shape)} "
                                     f"{part[k].dtype}")  # fmt: skip
                part[k].copy_(got[k])

    def pairs(self, other: GivenStarts) -> dict:
        """The parcel entries that bring this batch into ``other``."""
        return {f"batch.{part}.{k}": (getattr(self, part)[k], getattr(other, part)[k]) for part in ("s", "h") for k in getattr(self, part)}


def copy_modules(src, dst) -> list[str]:
    """The state of every module of algorithm ``src`` into ``dst``'s, an object of the same class and arguments on another
    device: the networks with their normalizers, the policy average, SuperTrack's balancers. The optimizers keep their
    own state. Returns the modules' names."""
    names = [k for k, v in vars(src).items() if isinstance(v, nn.Module)]
    for k in names:
        getattr(dst, k).load_state_dict(getattr(src, k).state_dict())
    return names


class SplitAlgo:
    """What ndm_gym.train's loop reads of its algorithm, each from the object that holds it under a layout: the world
    model, its optimizer and SuperTrack's world-model balancer from the world model's owner (AL: the actor's object; AWL:
    the world-model learner's); the policy, the critic, their optimizers, the policy average, SuperTrack's policy
    balancer and the count of skipped policy updates from the policy learner's."""

    def __init__(self, actor, wmer, learner):
        self.actor, self.wmer, self.learner = actor, wmer, learner

    wm = property(lambda self: self.wmer.wm)
    opt_wm = property(lambda self: self.wmer.opt_wm)
    pi = property(lambda self: self.learner.pi)
    opt_pi = property(lambda self: self.learner.opt_pi)
    pi_avg = property(lambda self: self.learner.pi_avg)
    nonfinite = property(lambda self: self.learner.nonfinite)
    fail_starts = property(lambda self: self.learner.fail_starts)
    residual = property(lambda self: self.actor.residual)

    @property
    def optimizers(self) -> tuple:
        """The live optimizers, in an algorithm's order: the world model's, then the learner's others."""
        return (self.wmer.opt_wm, *self.learner.optimizers[1:])

    def log_lr(self, lr: float) -> float:
        return self.actor.log_lr(lr)

    def state_dict(self) -> dict:
        """The run's state with an algorithm's keys: the policy learner's, the world model's parts from its owner."""
        sd, own = self.learner.state_dict(), self.wmer.state_dict()
        sd.update({k: own[k] for k in WM_STATE if k in own})
        return sd

    def __getattr__(self, name):
        raise AttributeError(f"--layout: the training loop reads {name!r} of the algorithm, which SplitAlgo does not route to a device's object")


class Split:
    """``--layout AL | AWL``: the objects of the run's algorithm on the layout's devices and what crosses between them
    (the module's docstring). ``actor``: train.py's algorithm object on the simulator's device; ``algo_on(device)``: a new
    object of the same arguments on another device, with the task's tables there; ``buffer``: the replay buffer, on the
    actor's device with its write head and fill level on the device (--graph_early); ``devices``: the layout's, the
    actor's first (the first GPUs the trainer sees by default). ``start_keys``, ``hist``: the replay's fields of an
    imagination start and its history's length (GivenStarts; None: --hist);
    ``steps``: the collection steps of an iteration at most (None: those of --utd), the rows AWL's replica receives per
    iteration. train_hand passes the hand agents' (algos.hand_layout).

    The training loop: ``update_graphs()`` once after the warm-up collection; per iteration its collection, the returned
    world-model update and, once ``ready``, the returned policy update; then ``exchange()``. ``algo`` (``SplitAlgo``) is
    what the loop reads otherwise."""

    def __init__(self, args, actor, algo_on, buffer: ReplayBuffer, devices: tuple | None = None, start_keys: tuple = START_KEYS,
                 hist: int | None = None, steps: int | None = None):
        layout = args.layout
        if layout not in LAYOUTS or LAYOUTS[layout] < 2:
            raise ValueError(f"--layout {layout}: not a split layout")
        n = LAYOUTS[layout]
        if devices is None:
            have = torch.cuda.device_count()
            if have < n:
                raise ValueError(f"--layout {layout} needs {n} GPUs, the trainer sees {have} (CUDA_VISIBLE_DEVICES)")
            devices = tuple(f"cuda:{i}" for i in range(n))
        if len(devices) != n or torch.device(devices[0]) != torch.device(actor.dev) or torch.device(devices[0]) != torch.device(buffer.device):
            raise ValueError(f"--layout {layout}: {n} devices, the first the actor's and its replay buffer's, not {devices}")
        if buffer.ptr_t is None or buffer.size_t is None:
            raise ValueError(f"--layout {layout}: the replay's write head and fill level on the device (--graph_early)")
        if not RunningNorm.no_sync:  # a normalizer's default update replaces its tensors: the copies would keep the old ones
            raise ValueError(f"--layout {layout}: the normalizers' statistics written in place (--no_sync)")
        for name in ("wm_update_graphable", "policy_update_graphable", "sample_valid_starts_nosync"):
            if getattr(actor, name, None) is None:
                raise ValueError(f"--layout {layout}: the algorithm has no {name} (SuperMimic and SuperTrack do)")
        self.args, self.layout, self.devices, self.buffer, self.actor = args, layout, tuple(devices), buffer, actor
        self.steps = steps
        self.learner = algo_on(devices[-1])
        self.wmer = algo_on(devices[1]) if layout == "AWL" else actor  # the world model's owner
        for other in (self.learner, self.wmer):
            if other is not actor:
                if type(other) is not type(actor):
                    raise ValueError(f"--layout {layout}: algo_on built a {type(other).__name__}, the actor's is a {type(actor).__name__}")
                copy_modules(actor, other)
        self.algo = SplitAlgo(actor, self.wmer, self.learner)
        # the imagination batch: the actor's draw (out) and the learner's copy (given), which its policy update reads
        hist = args.hist if hist is None else hist
        self.out = GivenStarts(buffer, args.pi_batch, hist, actor.dev, start_keys)
        self.given = GivenStarts(buffer, args.pi_batch, hist, self.learner.dev, start_keys)
        self.learner.sample_valid_starts_nosync = self.given.starts  # the learner's own draw: the batch it was sent
        tensors = lambda m, kind: dict(getattr(m, f"named_{kind}")())  # noqa: E731
        pair = lambda prefix, src, dst: {f"{prefix}.{k}": (v, dst[k]) for k, v in src.items()}  # noqa: E731
        wm_state = lambda m: {**tensors(m, "parameters"), **tensors(m, "buffers")}  # noqa: E731
        to_learner = {**self.out.pairs(self.given), **pair("pi", tensors(actor.pi, "buffers"), tensors(self.learner.pi, "buffers"))}
        self.replica = None
        if layout == "AWL":
            self.wm_to_learner = Parcel(pair("wm", wm_state(self.wmer.wm), wm_state(self.learner.wm)))
            self.wm_norms = Parcel(pair("wm", tensors(actor.wm, "buffers"), tensors(self.wmer.wm, "buffers")))
            self._make_replica()
        else:
            to_learner.update(pair("wm", wm_state(actor.wm), wm_state(self.learner.wm)))
        self.to_learner = Parcel(to_learner)
        self.to_actor = Parcel(pair("pi", tensors(self.learner.pi, "parameters"), tensors(actor.pi, "parameters")))
        self.limit = RunAhead(self.devices)
        self.ready = False  # the learner holds a batch: from the first exchange on
        self.draw = None  # update_graphs: the actor's draw of the next batch

    def describe(self) -> str:
        """The layout's devices and what each does, for the run's output."""
        holds = ["policy update"] + ["average"] * (self.learner.pi_avg is not None) + ["critic"] * hasattr(self.learner, "critic")
        actor, learner = f"actor {self.actor.dev} (simulator, collection, replay", f"policy learner {self.learner.dev} ({', '.join(holds)})"
        if self.replica is None:
            return f"--layout {self.layout}: {actor}, world-model updates) | {learner}"
        return f"--layout {self.layout}: {actor}) | world-model learner {self.wmer.dev} (world-model updates on a replica of the replay) | {learner}"

    # ------------------------------------------------------------------ AWL's replica of the replay buffer
    def _make_replica(self):
        buf, dev = self.buffer, self.wmer.dev
        self.replica = ReplayBuffer(buf.T, buf.N, dev, fields=buf.fields, priorities=False)
        self.replica.window_keys = buf.window_keys  # the fields a world-model window reads
        self.replica.use_device_ptr(device_size=True)
        # collection steps per iteration: floor(acc) <= ceil(1 / utd), or the trainer's
        self.n_max = math.ceil(1.0 / self.args.utd - 1e-9) if self.steps is None else self.steps
        if not 1 <= self.n_max <= buf.T:
            why = f"--utd {self.args.utd}" if self.steps is None else "the trainer's"
            raise ValueError(f"--layout AWL: {self.n_max} collection steps per iteration ({why}) and a replay of {buf.T} steps")
        rows = lambda d: {k: torch.zeros((self.n_max,) + v.shape[1:], dtype=v.dtype, device=d) for k, v in buf.data.items()}  # noqa: E731
        self.rows_out, self.rows_in = rows(buf.device), rows(dev)
        self.rows = Parcel({f"rows.{k}": (self.rows_out[k], self.rows_in[k]) for k in buf.data})
        self.heads = Parcel({"ptr": (buf.ptr_t, self.replica.ptr_t), "size": (buf.size_t, self.replica.size_t)})
        self.sent = buf.ptr_t.clone()  # on the actor: the write head up to which the replica has the rows
        self.ar_out, self.ar_in = torch.arange(self.n_max, device=buf.device), torch.arange(self.n_max, device=dev)

    @torch.no_grad()
    def _start_replica(self):
        """The replica as the actor's buffer is now (after the warm-up collection)."""
        buf, rep = self.buffer, self.replica
        for k in buf.data:
            rep.data[k].copy_(buf.data[k], non_blocking=True)
        self.heads.send()
        self.sent.copy_(buf.ptr_t)
        rep.ptr, rep.size = buf.ptr, buf.size

    @torch.no_grad()
    def _send_rows(self):
        """The rows the actor wrote since the last call into the replica at its own write head, then the heads: always
        ``n_max`` rows from the head on, the new ones and then rows both buffers hold already (a fixed size, as XMimic)."""
        buf, rep = self.buffer, self.replica
        if (buf.ptr - rep.ptr) % buf.T > self.n_max or buf.size - rep.size > self.n_max:
            raise RuntimeError(f"--layout AWL: more than {self.n_max} collection steps since the replica's last rows")
        idx = (self.sent + self.ar_out) % buf.T
        for k in buf.data:
            torch.index_select(buf.data[k], 0, idx, out=self.rows_out[k])
        self.rows.send()
        idx = (rep.ptr_t + self.ar_in) % rep.T
        for k in rep.data:
            rep.data[k].index_copy_(0, idx, self.rows_in[k])
        self.heads.send()
        self.sent.copy_(buf.ptr_t)
        rep.ptr, rep.size = buf.ptr, buf.size

    # ------------------------------------------------------------------ the loop's calls
    def _draw(self) -> tuple:
        with torch.no_grad():
            self.out.fill(*self.actor.sample_valid_starts_nosync(self.buffer, self.args.pi_batch))
        return ()

    def _policy_update(self):
        if not self.ready:
            raise RuntimeError("--layout: the policy update needs the actor's first batch (after the first Split.exchange)")
        return self.learner.policy_update_graphable(self.given)

    def update_graphs(self, graphs: bool = True) -> tuple:
        """train.py's ``update_graphs`` under the layout, called once after the warm-up collection (AWL's replica starts
        as the actor's buffer then): the world-model update on its owner's device and replay, and the policy update on the
        learner's device, on the batch it was sent. Each is a CUDA graph on its device (algos.blinkmimic.Graphed, with a
        capture stream of that device: torch's default one belongs to the device that captured first, and work of another
        device would not be recorded on it); ``graphs`` False: the plain functions, as the tests run them on the CPU."""
        from .blinkmimic import Graphed

        if self.replica is not None:  # the world-model learner starts from the replay and the normalizers the warm-up left
            self._start_replica()
            self.wm_norms.send()
        wm_buffer = self.buffer if self.replica is None else self.replica
        avg = self.learner.pi_avg

        streams: dict = {}  # one capture stream per device, shared by its graphs (as torch's default one is on one GPU)

        def on(fn, ready, name, dev):
            if not graphs:
                return fn
            d = torch.device(dev)
            if d not in streams:
                streams[d] = torch.cuda.Stream(device=d)
            return OnDevice(Graphed(fn, ready, name, stream=streams[d]), d)

        self.draw = on(self._draw, lambda: True, "the draw of the imagination batch", self.actor.dev)
        wm = on(lambda: self.wmer.wm_update_graphable(wm_buffer), lambda: True, "the world-model update", self.wmer.dev)
        pi = on(self._policy_update, lambda: avg is None or int(avg.n_averaged) > 0, "the policy update", self.learner.dev)
        return wm, pi

    def exchange(self):
        """The end of an iteration: the actor draws the next imagination batch, and what the next iteration's steps read
        crosses the devices (the module's docstring), in this order: AWL's world model to the learner before the actor's
        normalizers reach the world-model learner, so that the learner gets the normalizers the weights were fit with."""
        if self.draw is None:
            raise RuntimeError("Split.exchange before Split.update_graphs")
        self.draw()
        if self.replica is not None:
            self.wm_to_learner.send()
            self._send_rows()
            self.wm_norms.send()
        self.to_learner.send()
        self.to_actor.send()
        self.ready = True
        self.limit.step()
