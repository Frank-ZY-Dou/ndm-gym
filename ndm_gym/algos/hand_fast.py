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
--phase_times    a measurement, which changes no result: the wall time of an iteration's phases (collection, world-model
                 updates, policy update, critic update, log rows, checkpoints, the host's rest), the device synchronized
                 at every boundary, in ``phase_times.csv`` with a summary at the end (``PhaseTimes``).
--compile_updates  with --no_sync: the updates' parts that otherwise run eagerly between the compiled ones, compiled (equal
                 up to rounding): the policy's objective from the imagined window (the survival, SuperTrack's weighted
                 terms, failure penalty and action regularizer, SuperMimic's return with the critic's value at the end),
                 SuperTrack's kinematics of the start states (SuperMimic's are compiled already), SuperMimic's critic
                 targets (its normalizer's update, the values and the lambda-returns) and its minibatch loss; the backward
                 passes of these parts with them.
--muon_batched   with --pi_opt muon or --wm_opt muon: Muon's Newton-Schulz iterations of equally shaped matrices as
                 batched products (core.muon.MuonAdam's ``batched``, train.py's option; the world model's two hidden
                 matrices; equal up to rounding).
--elementwise_3x3, --combo_kernels   train.py's options: inside compiled functions the products of 3 x 3 matrices and
                 3-vectors as fused fp32 elementwise products, and independent small kernels launched together
                 (train.elementwise_3x3_products, train.combo_kernels; equal up to rounding).
--wm_palm_fk     the world model's fit computes the kinematics of the palm alone (HandKinematics.palm), the only part of
                 them the world model reads, instead of every keypoint's (the same values).
--sm_reuse_input  SuperMimic: an imagined step's policy input is the critic's input the step before computed (the same
                 observation and goal of the same state; the start's from start_features), not computed again (equal up
                 to rounding).
--compile_collect  with --no_sync: the collection step's torch work before the physics (the state, the policy's action,
                 the position targets written) and after it (the frame, the terminations, the masked resets, the
                 normalizers, the episode lengths, the log's sums) as two compiled functions, which --collect_graph
                 captures with the physics' kernels (``Collector.core_compiled``); the forward kinematics are compiled in
                 them (--compile_eager's are then not used). The random draws are made before them in the default
                 step's order and the replay write stays outside (equal up to rounding).
--compile_rollout  the policy update's imagined rollout (the start's kinematics and the --horizon steps) as one compiled
                 function (the agent's ``_rollout``), whose backward is one graph, instead of a compiled function per step:
                 the steps' kernels fuse across their boundaries and the policy's gradients are summed inside the graph.
                 SuperTrack's imagination noise is drawn before it, in the steps' order (equal up to rounding).
--mix_order_reductions  inductor's mix-order reductions without its size checks (``mix_order_reductions``): a row
                 reduction and a column reduction that read the same activations fused into one kernel, as in
                 LayerNorm's backward in the world model's fit (the input's gradient per row, the weight's and bias's
                 per column), which inductor fuses by default only from 4,096 rows (the fit has --wm_batch rows; equal up
                 to rounding).
--autotune_gemm  the compiled functions' matrix products from the fastest of cuBLAS and inductor's Triton templates,
                 timed per shape when they are compiled (``autotune_gemm``: inductor's max_autotune_gemm, without its CUDA
                 graphs; longer compilations; equal up to rounding).

An agent takes part by building its optimizers with ``optimizer`` and by taking its policy step with ``finite_step`` on
the survival of ``survive``; the collector and the graphs call its ``act``, ``update_wm_norms``, ``wm_update`` and
``policy_update`` as the loop does. docs/tasks/hand.md lists what each option changes in a run's numbers.
"""

from __future__ import annotations

import argparse
import os
import time
from array import array

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
PHASES = ("collect", "wm", "policy", "critic", "log", "ckpt", "host")  # --phase_times: an iteration's phases (PhaseTimes)
LAYOUT_PHASES = (*PHASES[:4], "exchange", *PHASES[4:])  # with --layout: the exchange between the devices too (algos.hand_layout)


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
    g.add_argument("--compile_updates", action="store_true",
                   help="with --no_sync: the updates' parts that otherwise run eagerly between the compiled ones (the policy's objective, "
                   "SuperTrack's start kinematics, SuperMimic's critic targets and minibatch loss) compiled, with their backward passes "
                   "(changes results by rounding)")  # fmt: skip
    g.add_argument("--muon_batched", action="store_true",
                   help="with --pi_opt muon or --wm_opt muon: Muon's Newton-Schulz iterations of equally shaped matrices as batched products "
                   "(changes results by rounding)")  # fmt: skip
    g.add_argument("--elementwise_3x3", action="store_true",
                   help="3x3 products in compiled functions as fused fp32 elementwise products (train.elementwise_3x3_products; changes "
                   "results by rounding)")  # fmt: skip
    g.add_argument("--combo_kernels", action="store_true",
                   help="independent small kernels of compiled functions launched together (train.combo_kernels; may change results by "
                   "rounding)")  # fmt: skip
    g.add_argument("--wm_palm_fk", action="store_true",
                   help="the world model's fit computes the kinematics of the palm alone, the only part the world model reads (the same "
                   "values)")  # fmt: skip
    g.add_argument("--sm_reuse_input", action="store_true",
                   help="supermimic: an imagined step's policy input is the critic's input of the step before (the same observation and goal "
                   "of the same state), not computed again (changes results by rounding)")  # fmt: skip
    g.add_argument("--compile_collect", action="store_true",
                   help="with --no_sync: the collection step's torch work before and after the physics as two compiled functions, the "
                   "random draws made before them in the default's order (changes results by rounding)")  # fmt: skip
    g.add_argument("--compile_rollout", action="store_true",
                   help="the policy update's imagined rollout (start kinematics and the horizon's steps) as one compiled function, its "
                   "backward one graph (changes results by rounding)")  # fmt: skip
    g.add_argument("--mix_order_reductions", action="store_true",
                   help="inductor's mix-order reductions without its size checks: LayerNorm's backward in the world model's fit, its "
                   "row and column reductions, as one kernel (may change results by rounding)")  # fmt: skip
    g.add_argument("--autotune_gemm", action="store_true",
                   help="the compiled functions' matrix products from the fastest of cuBLAS and Triton's templates, timed per shape "
                   "at compilation (inductor's max_autotune_gemm; changes results by rounding)")  # fmt: skip
    g.add_argument("--phase_times", action="store_true",
                   help="time an iteration's phases (collection, world-model updates, policy update, critic update where it is not in "
                   "the policy update's CUDA graph, log rows, checkpoints, the host's rest), the device synchronized at every boundary: "
                   "a row per log row in <log_dir>/phase_times.csv and a summary at the end (a slower run, the same results)")  # fmt: skip


def check_hand_fast_args(args: argparse.Namespace):
    """The options' constraints on a run's arguments; raises ValueError."""
    if args.no_sync and not (args.fused_adam and args.reset_masked):
        raise ValueError("--no_sync skips a non-finite policy step on the device, with the fused Adam, and resets by mask: "
                         "needs --fused_adam --reset_masked")  # fmt: skip
    if (args.collect_graph or args.update_graphs) and not args.no_sync:
        raise ValueError("--collect_graph and --update_graphs capture an iteration written without host synchronization: they need --no_sync")
    if args.compile_eager and not args.no_sync:
        raise ValueError("--compile_eager compiles the forward kinematics of the collection step without host synchronization: needs --no_sync")
    if args.compile_updates and not args.no_sync:
        raise ValueError("--compile_updates compiles the updates' parts written without host synchronization: needs --no_sync")
    if args.compile_collect and not args.no_sync:
        raise ValueError("--compile_collect compiles the collection step written without host synchronization: needs --no_sync")
    if args.muon_batched and "muon" not in (args.pi_opt, args.wm_opt):
        raise ValueError("--muon_batched needs --pi_opt muon or --wm_opt muon")
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
        if ck_args.get(k, default) != getattr(args, k, default):  # (an option the task's training does not have: its default)
            raise ValueError(f"--resume: --{k} must match the checkpoint's run ({ck_args.get(k, default)!r})")


def mix_order_reductions():
    """``--mix_order_reductions``: inductor fuses a reduction over rows and one over columns that read the same tensor into
    one kernel (``torch._inductor.config.triton.mix_order_reduction``, on by default), here without its checks of size
    (``mix_order_reduction_non_strict_mode``), which leave it out below 4,096 rows: the world model's fit at --wm_batch
    2,048 runs LayerNorm's backward as one kernel per layer that reads the activations once, instead of a row kernel for
    the input's gradient and a split column kernel for the weight's and bias's that reads them again."""
    import torch._inductor.config

    torch._inductor.config.triton.mix_order_reduction = True
    torch._inductor.config.triton.mix_order_reduction_non_strict_mode = True


def autotune_gemm():
    """``--autotune_gemm``: inductor times each matrix product of a compiled function, per shape, between cuBLAS and its
    Triton templates and takes the fastest (``torch._inductor.config.max_autotune_gemm``; the products' tiles, so their
    rounding, may change). Not its max-autotune mode: no CUDA graphs of its own, the loop's graphs capture the kernels."""
    import torch._inductor.config

    torch._inductor.config.max_autotune_gemm = True


# ---------------------------------------------------------------------- the agents' part
def optimizer(args, dev, module: nn.Module, kind: str):
    """The optimizer of an agent's ``kind`` ("wm", "pi" or "critic") network: Adam at --lr, the policy's with betas
    (0.9, 0.95). --fused_adam: its fused kernel; --update_graphs: the capturable fused kernel with a device learning
    rate (common.adam_kwargs, set_lr); --pi_opt muon, --wm_opt muon: MuonAdam on the network's MLP (--muon_batched: its
    batched Newton-Schulz iterations). ``args`` without these options (a checkpoint's from before them) builds Adam as before."""
    betas = (0.9, 0.95) if kind == "pi" else (0.9, 0.999)
    graphs = bool(getattr(args, "update_graphs", False))
    adam_kw = {"fused": True} if getattr(args, "fused_adam", False) else {}
    if getattr(args, f"{kind}_opt", "adam") == "muon":
        batched = bool(getattr(args, "muon_batched", False))
        return MuonAdam(module, [module.net], lr=args.lr, betas=betas, capturable=graphs, batched=batched, **adam_kw)
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


class PhaseTimes:
    """--phase_times (train_hand): the wall time of each phase of an iteration, in ``<log_dir>/phase_times.csv``. The device
    is synchronized at every phase's boundary, so that a phase's time holds the work it queued there; an iteration then
    takes at least as long as without the option, where the host queues a phase while the device runs the one before.
    The phases (``PHASES``) add up to the iteration:

    collect  the collection step: the policy's action, the control step with its physics, the terminations and resets,
             the normalizers, the replay write, the log's sums; with --collect_graph one CUDA graph
    wm       the iteration's --wm_batches world-model updates; with --update_graphs one CUDA graph
    policy   the policy update with the policy average's step (--pi_ema); with --update_graphs one CUDA graph, which
             holds SuperMimic's critic update too
    critic   SuperMimic's critic update, timed apart only without --update_graphs: the agent then calls ``lap`` where its
             policy step ends and where its critic's update ends (NaN where it is in ``policy`` and without a critic)
    log      a log row: its read of the device, the file's line, the output
    ckpt     a checkpoint
    host     the rest: the loop's host work between the phases (the learning rates, the log's lists, the timing itself)
             and what it queues on the device

    The loop times its three functions through ``timed``, calls ``lap`` after a log row and after a checkpoint and ``end``
    after every iteration; every --log_every-th writes a row (``COLUMNS``: the iteration, each phase's mean ms per
    iteration since the last row, the iterations per second; a resumed run continues the file). ``finish`` returns the
    summary: the means over iterations 1,001 to 3,000 of a run from its first iteration, past its compilations and its
    graphs' captures (the policy update's follows --wm_only_iters); over the second half of the iterations run for a
    shorter run and for a resumed one, which compiles and captures again. ``clock``: the tests' in time.perf_counter's
    place. ``devices`` (--layout, algos.hand_layout): the layout's, all synchronized at every boundary, and the phases
    ``LAYOUT_PHASES``: the exchange between them (the draw of the next imagination batch, the copies) is timed apart;
    ``algo`` is then the policy learner's agent."""

    COLUMNS = ("iteration", *(f"{p}_ms" for p in PHASES), "it_per_s")  # phase_times.csv
    WINDOW = (1000, 3000)  # the summary's iterations, of a run from its first: after the first of the two, up to the second

    def __init__(self, args, algo, device, resumed: bool, clock=time.perf_counter, devices: tuple = ()):
        self.args, self.device, self.clock = args, device, clock
        self.devices = [d for d in dict.fromkeys(torch.device(x) for x in (device, *devices)) if d.type == "cuda"]  # synchronized
        self.phases = LAYOUT_PHASES if devices else PHASES
        self.columns = ("iteration", *(f"{p}_ms" for p in self.phases), "it_per_s") if devices else self.COLUMNS
        self.has_critic = getattr(algo, "critic", None) is not None
        self.critic_apart = self.has_critic and not args.update_graphs  # with the graphs the critic's update is in the policy update's
        if self.critic_apart:
            algo.lap = self.lap
        self.path, header = os.path.join(args.log_dir, "phase_times.csv"), ",".join(self.columns)
        if os.path.exists(self.path):
            if not resumed:
                raise ValueError(f"{self.path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
            with open(self.path) as f:
                if f.readline().strip() != header:
                    raise ValueError(f"{self.path} has other columns than {header!r}; continue in another --log_dir")
        self.file = open(self.path, "a")
        if os.path.getsize(self.path) == 0:
            self.file.write(header + "\n")
        self.cur = dict.fromkeys(self.phases, 0.0)  # the running iteration's seconds per phase
        self.row, self.row_n = [0.0] * len(self.phases), 0  # their sums over the iterations since the last row
        self.first, self.secs = None, array("d")  # the first iteration timed; every iteration's seconds per phase, flat
        self.last = self._now()  # the last boundary: the loop's start, the warm-up collection done

    def _now(self) -> float:
        for d in self.devices:  # the device's work queued until now is the time's
            torch.cuda.synchronize(d)
        return self.clock()

    def lap(self, phase: str):
        """A boundary: the time since the last one is ``phase``'s."""
        now = self._now()
        self.cur[phase] += now - self.last
        self.last = now

    def timed(self, phase: str, fn):
        """``fn`` with each call's time as ``phase``'s and the time from the last boundary to the call as the host's."""

        def call():
            self.lap("host")
            out = fn()
            self.lap(phase)
            return out

        return call

    def _ms(self, sums: list, n: int) -> tuple[list, float]:
        """The phases' mean ms per iteration from their ``sums`` of seconds over ``n`` iterations, and the iterations per second."""
        ms = [1e3 * s / n for s in sums]
        if not self.critic_apart:
            ms[self.phases.index("critic")] = float("nan")
        return ms, n / sum(sums)

    def end(self, it: int):
        """The end of iteration ``it``: what is left of its time is the host's; a row with every --log_every-th."""
        self.lap("host")
        secs = list(self.cur.values())
        self.first = it if self.first is None else self.first
        self.secs.extend(secs)
        self.row, self.row_n = [a + b for a, b in zip(self.row, secs, strict=True)], self.row_n + 1
        self.cur = dict.fromkeys(self.phases, 0.0)
        if it % self.args.log_every == 0:
            ms, ips = self._ms(self.row, self.row_n)
            self.file.write(",".join([str(it), *(f"{x:.6g}" for x in (*ms, ips))]) + "\n")
            self.file.flush()
            self.row, self.row_n = [0.0] * len(self.phases), 0
            self.last = self.clock()  # the row's write is no iteration's

    def finish(self) -> str:
        """Closes the file; returns the summary, a line per phase with what it holds."""
        self.file.close()
        k, args = len(self.phases), self.args
        n = len(self.secs) // k
        if n == 0:
            return "--phase_times: no iteration was run"
        a, b = self.WINDOW
        if self.first != 1 or n < b:  # a resumed run or a shorter one: the second half of its iterations
            a, b = n // 2, n
        ms, ips = self._ms([sum(self.secs[i::k][a:b]) for i in range(k)], b - a)
        graph = lambda on: "one CUDA graph" if on else "eager"  # noqa: E731
        policy = ["the policy update"] + ["the critic's update"] * (self.has_critic and not self.critic_apart)
        policy += ["the policy average's step"] * (args.pi_ema > 0)
        if self.critic_apart:
            critic = f"the critic's update ({args.critic_epochs * args.critic_minibatches} steps): eager"
        else:
            critic = "in policy: one CUDA graph with the policy update" if self.has_critic else "no critic"
        what = {"collect": f"the collection step of {args.num_envs} worlds (action, control step, resets, normalizers, replay write): "
                           + ("one CUDA graph" if args.collect_graph else "eager, its physics a CUDA graph"),
                "wm": f"{args.wm_batches} world-model updates: {graph(args.update_graphs)}",
                "policy": f"{', '.join(policy)}: {graph(args.update_graphs)}", "critic": critic,
                "log": f"a log row every {args.log_every} iterations", "ckpt": f"a checkpoint every {args.ckpt_every} iterations",
                "host": "the rest: the loop's host work and what it queues",
                "exchange": "the exchange between the devices: the draw of the next imagination batch, the copies"}  # fmt: skip
        total = sum(x for x in ms if x == x)
        lines = [f"--phase_times: mean over iterations {self.first + a}-{self.first + b - 1}, the device synchronized at every phase's boundary"]
        if self.phases is LAYOUT_PHASES:
            lines[0] += (f" (--layout {args.layout}: every device of the layout, its steps one after another; the layout's speed is "
                         "train_log.csv's it_per_s without --phase_times)")  # fmt: skip
        for p, x in zip(self.phases, ms, strict=True):
            lines.append(f"  {p:<8}{x:9.3f} ms {100 * x / total:5.1f} %  {what[p]}" if x == x else f"  {p:<8}{'-':>9}             {what[p]}")
        lines.append(f"  {'total':<8}{total:9.3f} ms  {ips:.1f} iterations/s")
        return "\n".join(lines)


class Collector:
    """train_hand's collection step for --no_sync: the default step's computations (train_hand.collect) on the loop's
    state kept in fixed tensors (the previous action, the episode lengths, the log's sums ``acc``), without a host read.
    The control step is HandSim.step_nosync, the ended episodes' lengths are summed and cleared by mask; with
    --compile_eager the step's forward kinematics, the agent's too (its ``fk``), are one compiled function. A call runs
    the step on Warp's stream, where torch's work and the physics graph are ordered, and has the caller's stream wait
    for it. With --collect_graph the first two calls run so (kernels loaded, workspaces allocated), the third captures
    the step with the physics as kernels in one CUDA graph (torch's capture on Warp's stream, Warp's registered as
    external: sim.rigid.isaaclab_mjwarp.capture_step's way), and it is replayed from then on; the transition is written
    at the replay's device-side head, the host's advanced after each call.

    Another task's training (train_quadruped) subclasses it with its own log sums (``acc_keys``, ``log_errors``) and its
    state's layout (``unflatten``); its simulator has HandSim's ``step_nosync``, ``physics`` and ``torch_stream``."""

    acc_keys = ACC_KEYS
    unflatten = staticmethod(unflatten_state)

    def log_errors(self, errors: dict):
        """The task's errors of a step's next states, into the log's sums."""
        self.acc["object_err"] += errors["object_pos"].sum()

    def __init__(self, sim, task, algo, buffer, args):
        self.sim, self.task, self.algo, self.buffer, self.args = sim, task, algo, buffer, args
        self.graphed = bool(args.collect_graph)
        if self.graphed and buffer.size_t is None:
            raise ValueError("--collect_graph writes at the replay's device-side head (ReplayBuffer.use_device_ptr(device_size=True))")
        self.a_prev = torch.zeros(sim.N, task.kin.num_joints, device=sim.device)
        self.ep_len = torch.zeros(sim.N, device=sim.device)
        self.acc = {k: torch.zeros((), device=sim.device) for k in self.acc_keys}
        self.fk = task.kin.forward
        if args.compile_eager:  # one compiled function for the step's forward kinematics: the agent's, the termination's, the reward's
            self.fk = algo.fk = torch.compile(task.kin.forward)
        self.compiled = bool(getattr(args, "compile_collect", False))
        if self.compiled:  # the torch work before and after the physics as two compiled functions, the forward kinematics in them
            self.fk = algo.fk = task.kin.forward
            self.before, self.after = torch.compile(self._before), torch.compile(self._after)
        self.calls, self.stream, self.graph, self.record = 0, None, None, None

    @torch.no_grad()
    def core(self, physics=None):
        """One collection step on the fixed tensors; ``physics``: the simulator's physics as kernels, inside the capture."""
        if self.compiled:
            return self.core_compiled(physics)
        sim, task, algo, acc, a_prev = self.sim, self.task, self.algo, self.acc, self.a_prev
        s, f = sim.state(), sim.f.clone()
        S = self.unflatten(s)
        a = algo.act(S, a_prev, f, noise=self.args.action_noise)
        u = task.setpoint(a, f)
        s1, term, done = sim.step_nosync(u, physics, self.fk)
        S1, f1 = self.unflatten(s1), task.ref.next_frame(f)
        fk1 = self.fk(S1[0])
        algo.update_wm_norms(S, S1, u)
        tr = {"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done}
        if self.graphed:
            self.buffer.add_device(tr)
        else:
            self.buffer.add(tr)
        self.ep_len.add_(1)
        acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
        self.log_errors(task.errors(S1, fk1, f1))
        acc["steps"] += sim.N
        acc["done"] += done.sum()
        acc["term"] += term.sum()
        acc["ep_len"] += torch.where(done, self.ep_len, 0.0).sum()
        self.ep_len.masked_fill_(done, 0.0)
        a_prev.copy_(torch.where(done.unsqueeze(-1), 0.0, a))

    @torch.no_grad()
    def core_compiled(self, physics=None):
        """--compile_collect: ``core``'s step with its torch work before the physics (``_before``) and after it (``_after``)
        as compiled functions. The random draws are made first, in the default step's order (the action's noise, then
        the resets'), so that the step draws the same numbers; the replay write and the previous action's update stay
        outside (the replay's fields are written in place)."""
        sim = self.sim
        z = torch.randn_like(self.a_prev) if self.args.action_noise > 0 else None
        draws = sim.reset_draws()
        s, f, a, u = self.before(z)
        sim.advance(physics)
        s1, done, a_next = self.after(s, f, a, u, draws)
        tr = {"s": s, "s1": s1, "u": u, "a_prev": self.a_prev, "f": f, "done": done}
        if self.graphed:
            self.buffer.add_device(tr)
        else:
            self.buffer.add(tr)
        self.a_prev.copy_(a_next)

    def _before(self, z: torch.Tensor | None):
        """--compile_collect: the state, the policy's action (its noise ``z``) and the position targets written."""
        sim, task = self.sim, self.task
        s, f = sim.state(), sim.f.clone()
        a = self.algo.act(unflatten_state(s), self.a_prev, f, noise=self.args.action_noise, z=z)
        u = task.setpoint(a, f)
        sim.set_targets(u)
        return s, f, a, u

    def _after(self, s: torch.Tensor, f: torch.Tensor, a: torch.Tensor, u: torch.Tensor, draws):
        """--compile_collect: after the physics, the simulator's step to its end (the resets with ``draws``), the normalizers,
        the episode lengths and the log's sums; returns the next state, the ends and the next previous action."""
        sim, task, acc = self.sim, self.task, self.acc
        s1, term, done = sim.finish_nosync(s, self.fk, draws)
        S, S1, f1 = unflatten_state(s), unflatten_state(s1), task.ref.next_frame(f)
        fk1 = self.fk(S1[0])
        self.algo.update_wm_norms(S, S1, u)
        self.ep_len.add_(1)
        acc["r"] += task.reward(S1, fk1, f1, a, self.a_prev).sum() / task.max_step_reward
        acc["object_err"] += task.errors(S1, fk1, f1)["object_pos"].sum()
        acc["steps"] += sim.N
        acc["done"] += done.sum()
        acc["term"] += term.sum()
        acc["ep_len"] += torch.where(done, self.ep_len, 0.0).sum()
        self.ep_len.masked_fill_(done, 0.0)
        return s1, done, torch.where(done.unsqueeze(-1), 0.0, a)

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
