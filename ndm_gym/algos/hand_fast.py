"""Systems options of hand-object tracking's training (ndm_gym.train_hand), for both of its agents (algos.hand_algos). Each
is off by default; with all of them off the loop and the agents compute what they do without this module.

--fused_adam     Adam's fused CUDA kernel for every optimizer (``optimizer``).
--reset_masked   the ended worlds are reset by masked writes of fixed shapes, the random draws made for every world
                 (HandSim.reset_masked): no host read, another random stream.
--no_sync        an iteration without host-device synchronization: the control step on Warp's stream, where torch's
                 writes, the physics graph and torch's reads follow each other (HandSim.step_nosync, ``Collector``); the
                 normalizers' freeze test on the device and their statistics written in place (RunningNorm.no_sync); the
                 policy's step skipped on the device for a non-finite gradient (``finite_step``); the survival products
                 as a loop (``survive``: cumprod's backward reads the device). The device is read at the log's rows, and
                 a checkpoint's tensors are moved to the host in one transfer per dtype (``to_host``).
--collect_graph  a collection step (the policy's action and its noise, the control step with MuJoCo-Warp's physics as
                 kernels, the terminations and masked resets, the normalizers, the replay write, the log's sums) as one
                 CUDA graph, captured at the third step and replayed (``Collector``).
--compile_eager  the collection step's forward kinematics (of the state for the policy and for the world model's
                 normalizer, of the next state for the termination and for the reward; each hundreds of small kernels
                 when run eagerly) as one compiled function (equal up to rounding).
--update_graphs  the iteration's world-model updates as one CUDA graph and its policy update (SuperMimic's with the
                 critic's) as another (``update_functions``; blinkmimic.Graphed), with capturable optimizers. With either
                 graph the replay's write head and fill level are read on the device (ReplayBuffer.use_device_ptr:
                 another random stream for the sampled starts).
--pi_opt muon, --wm_opt muon   Muon on the hidden matrices of the policy's or the world model's MLP, Adam on the rest
                 (core.muon.MuonAdam).
--pi_ema         an exponential moving average of the policy, which checkpoints store as 'pi' (``PolicyAverage``).
--pi_lr_mult     the policy's learning rate is --lr times this, ramped to 1 from the log row where sim_r reaches
                 --lr_decay_ret (at the latest from --lr_decay_by), over --lr_decay_len iterations (blinkmimic.PolicyLR).

An agent takes part by building its optimizers with ``optimizer`` and by taking its policy step with ``finite_step`` on
the survival of ``survive``; the collector and the graphs call its ``act``, ``update_wm_norms``, ``wm_update`` and
``policy_update`` as the loop does. docs/tasks/hand.md lists what each option changes in a run's numbers.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from ..core.muon import MuonAdam
from ..tasks.hand_tracking import unflatten_state
from .blinkmimic import Graphed, policy_average
from .common import adam_kwargs, ema_step, gated_step, survival, zero_nonfinite_grads

# the options a resumed run must share with its checkpoint (the algorithm and SuperTrack's terms, the optimizers' layouts and
# states, the policy average); the defaults are those of a checkpoint written before the option existed
RESUME_MATCH = {"algo": "supermimic", "st_vel_terms": False, "st_unit_scales": False, "st_task_weights": False, "fused_adam": False,
                "update_graphs": False, "pi_opt": "adam", "wm_opt": "adam", "pi_ema": 0.0}  # fmt: skip
ACC_KEYS = ("r", "steps", "done", "term", "ep_len", "object_err")  # the log's sums over a row's collection steps (train_hand)


def add_hand_fast_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("systems", "hand training's systems options (ndm_gym.algos.hand_fast), off by default")
    g.add_argument("--fused_adam", action="store_true", help="Adam's fused CUDA kernel for every optimizer, one launch per step (changes results)")
    g.add_argument("--reset_masked", action="store_true",
                   help="the ended worlds reset by masked writes of fixed shapes, without a host read; the reset worlds get the same "
                   "values, the random draws are made for all worlds (another random stream)")  # fmt: skip
    g.add_argument("--no_sync", action="store_true",
                   help="no host-device synchronization in an iteration: the control step on Warp's stream, the normalizers' freeze "
                   "test, the non-finite policy step skipped on the device (needs --fused_adam --reset_masked; changes results by "
                   "rounding)")  # fmt: skip
    g.add_argument("--collect_graph", action="store_true",
                   help="with --no_sync: a collection step (action, control step with its physics, terminations and resets, normalizers, "
                   "replay write) replayed as one CUDA graph")  # fmt: skip
    g.add_argument("--compile_eager", action="store_true",
                   help="with --no_sync: the collection step's forward kinematics, which otherwise run eagerly, as a compiled function "
                   "(changes results by rounding)")  # fmt: skip
    g.add_argument("--update_graphs", action="store_true",
                   help="with --no_sync: the world-model updates as one CUDA graph, the policy (and critic) update as another")  # fmt: skip
    g.add_argument("--pi_opt", type=str, default="adam", choices=["adam", "muon"],
                   help="the policy's optimizer; muon: Muon on the hidden weight matrices of its MLP, Adam on the rest")  # fmt: skip
    g.add_argument("--wm_opt", type=str, default="adam", choices=["adam", "muon"],
                   help="the world model's optimizer; muon: Muon on the hidden weight matrices of its MLP, Adam on the rest")  # fmt: skip
    g.add_argument("--pi_ema", type=float, default=0.0,
                   help="an exponential moving average of the policy, this decay per policy update; checkpoints store it as 'pi' (the "
                   "policy evaluated) and the trained policy as 'pi_live' (0: off)")  # fmt: skip
    g.add_argument("--pi_lr_mult", type=float, default=1.0, help="the policy's learning rate is --lr times this")
    g.add_argument("--lr_decay_ret", type=float, default=0.0,
                   help="with --pi_lr_mult: once the log's sim_r reaches this, the multiplier ramps linearly to 1 over --lr_decay_len "
                   "iterations (0: never)")  # fmt: skip
    g.add_argument("--lr_decay_len", type=int, default=300)
    g.add_argument("--lr_decay_by", type=int, default=0, help="with --lr_decay_ret: the ramp starts at this iteration at the latest (0: no deadline)")


def check_hand_fast_args(args: argparse.Namespace):
    """The options' constraints on a run's arguments; raises ValueError."""
    if args.no_sync and not (args.fused_adam and args.reset_masked):
        raise ValueError("--no_sync skips a non-finite policy step on the device, with the fused Adam, and resets by mask: "
                         "needs --fused_adam --reset_masked")  # fmt: skip
    if (args.collect_graph or args.update_graphs) and not args.no_sync:
        raise ValueError("--collect_graph and --update_graphs capture an iteration written without host synchronization: they need --no_sync")
    if args.compile_eager and not args.no_sync:
        raise ValueError("--compile_eager compiles the forward kinematics of the collection step without host synchronization: needs --no_sync")
    if not 0 <= args.pi_ema < 1:
        raise ValueError(f"--pi_ema must be in [0, 1), got {args.pi_ema}")
    if args.pi_lr_mult <= 0 or args.lr_decay_ret < 0 or args.lr_decay_len < 1 or args.lr_decay_by < 0:
        raise ValueError("--pi_lr_mult > 0, --lr_decay_ret >= 0, --lr_decay_len >= 1, --lr_decay_by >= 0")
    if args.lr_decay_ret > 0 and args.pi_lr_mult == 1:
        raise ValueError("--lr_decay_ret decays --pi_lr_mult, which is 1")
    if args.lr_decay_by and not args.lr_decay_ret:
        raise ValueError("--lr_decay_by needs --lr_decay_ret")


def check_resume(ck_args: dict, args: argparse.Namespace):
    """A resumed run must have its checkpoint's algorithm, SuperTrack's terms, optimizers and policy average; raises ValueError."""
    for k, default in RESUME_MATCH.items():
        if ck_args.get(k, default) != getattr(args, k):
            raise ValueError(f"--resume: --{k} must match the checkpoint's run ({ck_args.get(k, default)!r})")


# ---------------------------------------------------------------------- the agents' part
def optimizer(args, dev, module: nn.Module, kind: str):
    """The optimizer of an agent's ``kind`` ("wm", "pi" or "critic") network: Adam at --lr, the policy's with betas
    (0.9, 0.95). --fused_adam: its fused kernel; --update_graphs: the capturable fused kernel with a device learning
    rate (common.adam_kwargs, set_lr); --pi_opt muon, --wm_opt muon: MuonAdam on the network's MLP. ``args`` without
    these options (a checkpoint's from before them) builds Adam as before."""
    betas = (0.9, 0.95) if kind == "pi" else (0.9, 0.999)
    graphs = bool(getattr(args, "update_graphs", False))
    adam_kw = {"fused": True} if getattr(args, "fused_adam", False) else {}
    if getattr(args, f"{kind}_opt", "adam") == "muon":
        return MuonAdam(module, [module.net], lr=args.lr, betas=betas, capturable=graphs, **adam_kw)
    return torch.optim.Adam(module.parameters(), betas=betas, **adam_kwargs(args.lr, dev, graphs, adam_kw))


def survive(d: torch.Tensor, no_sync: bool) -> torch.Tensor:
    """The survival after each imagined step from the termination probabilities ``d`` (H, B): cumprod(1 - d, 0), with
    --no_sync as a loop of products (common.survival; equal up to rounding)."""
    return survival(d) if no_sync else torch.cumprod(1 - d, 0)


def finite_step(opt, params, grad_norm: torch.Tensor, no_sync: bool):
    """``opt``'s step unless the gradient's norm is non-finite: tested on the host, or with --no_sync on the device (the
    gradients of ``params`` zeroed and the step skipped there, parameters and optimizer state unchanged: common.gated_step)."""
    if not no_sync:
        if torch.isfinite(grad_norm):
            opt.step()
        return
    finite = torch.isfinite(grad_norm)
    zero_nonfinite_grads(params, finite)
    gated_step(opt, finite)


# ---------------------------------------------------------------------- the loop's part
class PolicyAverage:
    """--pi_ema: an exponential moving average of the policy's weights and normalizer (blinkmimic.policy_average), stepped
    after every policy update as AveragedModel.update_parameters steps it: the first step copies the policy, the later
    ones are its lerp (common.ema_step). A step reads only the average's count, which AveragedModel keeps on the host, so
    it can be captured with the update; the loop counts the steps (``count``), a graph's replays too. Checkpoints hold the
    average as 'pi', the policy that evaluate_hand loads, the trained policy as 'pi_live' and the average's own state as
    'pi_avg'."""

    def __init__(self, pi: nn.Module, decay: float):
        self.pi, self.decay, self.avg = pi, decay, policy_average(pi, decay)

    @torch.no_grad()
    def step(self):
        if int(self.avg.n_averaged) > 0:
            ema_step(self.avg, self.pi, self.decay)
            return
        for a, x in zip((*self.avg.module.parameters(), *self.avg.module.buffers()), (*self.pi.parameters(), *self.pi.buffers()), strict=True):
            a.copy_(x)

    def count(self):
        self.avg.n_averaged += 1

    def checkpoint(self, ck: dict) -> dict:
        """The agent's checkpoint ``ck`` with the average in the policy's place."""
        return {**ck, "pi": self.avg.module.state_dict(), "pi_live": ck["pi"], "pi_avg": self.avg.state_dict()}

    def resume(self, ck: dict) -> dict:
        """Loads the average from the checkpoint ``ck``; returns ``ck`` with the trained policy as 'pi', for the agent."""
        if "pi_avg" not in ck:
            raise ValueError("--pi_ema: the checkpoint holds no policy average (pi_avg)")
        self.avg.load_state_dict(ck["pi_avg"])
        return {**ck, "pi": ck["pi_live"]}


def update_functions(algo, buffer, args, average: PolicyAverage | None = None):
    """The loop's two updates as functions without arguments: the iteration's world-model updates (returns the last
    loss) and its policy update with the policy average's step (returns the agent's values for the log). With
    --update_graphs each is a CUDA graph (blinkmimic.Graphed: three calls run eagerly on a side stream, the fourth is
    captured, later ones replay it), its values copied out of the graph's memory."""

    def policy():
        out = algo.policy_update(buffer)
        if average is not None:
            average.step()
        return out

    wm, pi = (lambda: algo.wm_update(buffer)), policy
    if args.update_graphs:
        wm_graph = Graphed(lambda: (algo.wm_update(buffer),), lambda: True, "the world-model updates")
        wm, pi = (lambda: wm_graph()[0]), Graphed(policy, lambda: True, "the policy update")
    if average is None:
        return wm, pi

    def policy_counted():
        out = pi()
        average.count()  # on the host: a replay does not run the step's Python side
        return out

    return wm, policy_counted


def to_host(tree):
    """``tree`` (a checkpoint: nested dicts, lists and tuples) with its tensors on the host, moved in one transfer per dtype
    and device: the tensors are flattened into one tensor, copied, and split again into tensors of their own. On some GPUs
    a small transfer takes as long as a large one, and a checkpoint is a hundred of them (train_hand --no_sync)."""
    groups = {}

    def collect(x):
        if torch.is_tensor(x):
            groups.setdefault((x.dtype, x.device), []).append(x)
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)

    collect(tree)
    host = {}
    for ts in groups.values():
        flat = torch.cat([t.detach().reshape(-1) for t in ts]).cpu()
        for t, part in zip(ts, flat.split([t.numel() for t in ts]), strict=True):
            host[id(t)] = part.reshape(t.shape).clone()

    def rebuild(x):
        if torch.is_tensor(x):
            return host[id(x)]
        if isinstance(x, dict):
            return type(x)((k, rebuild(v)) for k, v in x.items())
        if isinstance(x, (list, tuple)):
            return type(x)(rebuild(v) for v in x)
        return x

    return rebuild(tree)


class Timed:
    """HAND_PROFILE=1 (train_hand): ``fn`` with the device synchronized before and after each call, so that the call's
    time is its own; ``take`` returns the mean time of the calls since the last one, in ms."""

    def __init__(self, fn):
        self.fn, self.seconds, self.calls = fn, 0.0, 0

    def __call__(self):
        torch.cuda.synchronize()
        t0 = time.time()
        out = self.fn()
        torch.cuda.synchronize()
        self.seconds, self.calls = self.seconds + time.time() - t0, self.calls + 1
        return out

    def take(self) -> float:
        ms = 1e3 * self.seconds / max(self.calls, 1)
        self.seconds, self.calls = 0.0, 0
        return ms


class Collector:
    """train_hand's collection step for --no_sync: the default step's computations (train_hand.collect) on the loop's
    state kept in fixed tensors (the previous action, the episode lengths, the log's sums ``acc``), without a host read.
    The control step is HandSim.step_nosync, the ended episodes' lengths are summed and cleared by mask; with
    --compile_eager the step's forward kinematics, the agent's too (its ``fk``), are one compiled function. A call runs
    the step on Warp's stream, where torch's work and the physics graph are ordered, and has the caller's stream wait
    for it. With --collect_graph the first two calls run so (kernels loaded, workspaces allocated), the third captures
    the step with the physics as kernels in one CUDA graph (torch's capture on Warp's stream, Warp's registered as
    external: sim.rigid.isaaclab_mjwarp.capture_step's way), and it is replayed from then on; the transition is written
    at the replay's device-side head, the host's advanced after each call."""

    def __init__(self, sim, task, algo, buffer, args):
        self.sim, self.task, self.algo, self.buffer, self.args = sim, task, algo, buffer, args
        self.graphed = bool(args.collect_graph)
        if self.graphed and buffer.size_t is None:
            raise ValueError("--collect_graph writes at the replay's device-side head (ReplayBuffer.use_device_ptr(device_size=True))")
        self.a_prev = torch.zeros(sim.N, task.kin.q_lo.shape[0], device=sim.device)
        self.ep_len = torch.zeros(sim.N, device=sim.device)
        self.acc = {k: torch.zeros((), device=sim.device) for k in ACC_KEYS}
        self.fk = task.kin.forward
        if args.compile_eager:  # one compiled function for the step's forward kinematics: the agent's, the termination's, the reward's
            self.fk = algo.fk = torch.compile(task.kin.forward)
        self.calls, self.stream, self.graph, self.record = 0, None, None, None

    @torch.no_grad()
    def core(self, physics=None):
        """One collection step on the fixed tensors; ``physics``: the simulator's physics as kernels, inside the capture."""
        sim, task, algo, acc, a_prev = self.sim, self.task, self.algo, self.acc, self.a_prev
        s, f = sim.state(), sim.f.clone()
        S = unflatten_state(s)
        a = algo.act(S, a_prev, f, noise=self.args.action_noise)
        u = task.setpoint(a, f)
        s1, term, done = sim.step_nosync(u, physics, self.fk)
        S1, f1 = unflatten_state(s1), task.ref.next_frame(f)
        fk1 = self.fk(S1[0])
        algo.update_wm_norms(S, S1, u)
        tr = {"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done}
        if self.graphed:
            self.buffer.add_device(tr)
        else:
            self.buffer.add(tr)
        self.ep_len.add_(1)
        acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
        acc["object_err"] += task.errors(S1, fk1, f1)["object_pos"].sum()
        acc["steps"] += sim.N
        acc["done"] += done.sum()
        acc["term"] += term.sum()
        acc["ep_len"] += torch.where(done, self.ep_len, 0.0).sum()
        self.ep_len.masked_fill_(done, 0.0)
        a_prev.copy_(torch.where(done.unsqueeze(-1), 0.0, a))

    def capture(self):
        import warp as wp

        t0 = time.time()
        stream = wp.get_stream(self.sim.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream, capture_error_mode="thread_local"):
            wp.capture_begin(stream=stream, external=True, capture_mode=wp.CaptureMode.THREAD_LOCAL)
            try:
                self.core(self.sim.physics)
            finally:
                self.record = wp.capture_end(stream=stream)  # Warp's record of the capture, kept with the graph
        print(f"--collect_graph: captured one collection step in {time.time() - t0:.1f} s", flush=True)

    def __call__(self):
        self.calls += 1
        if self.stream is None:
            self.stream = self.sim.torch_stream()
        if self.graphed and self.calls > 2:
            if self.graph is None:
                self.capture()
            self.graph.replay()
        else:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.core()
            torch.cuda.current_stream().wait_stream(self.stream)
        if self.graphed:
            self.buffer.advance_host()
