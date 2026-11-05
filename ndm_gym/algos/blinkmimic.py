# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658: scripts/presets.sh, src/blinkmimic/mirror.py,
# src/blinkmimic/train.py) under its MIT License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""BlinkMimic (F. Chen, 2026; https://github.com/fuyichen0725/blinkmimic): SuperMimic's objective with InstantTrack's and its
own training changes.

``--algo blinkmimic`` is ``--algo supermimic`` with BlinkMimic's released recipe (``PRESET``, their ``PRESET[main]`` in this
library's options); it is not a separate algorithm. Every change is an option of its own, off by default. The policy
average, Muon, mirror augmentation, failure-focused and evaluation-guided resets (algos/eprobe.py) and the policy's
learning-rate multiplier apply to any algorithm; the semi-implicit world model and the contact reward to SuperMimic. docs/blinkmimic.md maps BlinkMimic's
changes to the options and lists what is not ported.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from ..robots.g1 import G1_JOINT_NAMES, NUM_J
from .eprobe import add_eprobe_args, check_eprobe_args
from ..state import flatten_state, unflatten_state

# BlinkMimic's PRESET[main] (scripts/presets.sh @ d264658) in this library's options, in its order; NOT_PORTED lists the rest
PRESET = (
    "--pi_ema 0.995 --no_sync --fused_adam --elementwise_3x3 --compile_eager --lr 4e-4 --wm_batches 5 --wm_hidden 1024 1024 1024 "
    "--fail_reset 0.3 --contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --sm_residual --wm_only_iters 0 "
    "--bin_fix --ref_contact_fix --update_graphs --eprobe_share 0.15 --graph_early --collect_graph --reset_masked --pi_batch 2048 "
    "--pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --iterations 3500 --ckpt_every 100 --ckpt_light --eprobe_graph "
    "--eprobe_async 20 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7 --lr_decay_by 1400"
).split()
NOT_PORTED = "--ckpt_minutes 0, which needs no counterpart (train.py checkpoints by time only at --ckpt_hours)"
# options whose value a resumed run must share with its checkpoint (the optimizers' and networks' layouts)
RESUME_MATCH = {"pi_ema": 0.0, "pi_opt": "adam", "wm_opt": "adam", "wm_structured": False}


def add_blinkmimic_args(parser: argparse.ArgumentParser):
    """BlinkMimic's options (off by default). Tools that rebuild a run's arguments add them after ``add_speed_args``, as
    ``ndm_gym.train.parse_args`` does."""
    g = parser.add_argument_group("BlinkMimic", "--algo blinkmimic: --algo supermimic with BlinkMimic's released recipe "
                                  "(algos.blinkmimic.PRESET; options given on the command line take precedence)")  # fmt: skip
    g.add_argument("--preset", type=str, default=None, help=argparse.SUPPRESS)  # set by --algo blinkmimic
    g.add_argument("--pi_ema", type=float, default=0.0,
                   help="any algorithm: an exponential moving average of the policy, this decay per policy update; checkpoints store it "
                   "as 'pi' (the policy evaluated) and the trained policy as 'pi_live' (0: off)")  # fmt: skip
    g.add_argument("--pi_opt", type=str, default="adam", choices=["adam", "muon"],
                   help="any algorithm: the policy's optimizer; muon: Muon on the hidden weight matrices of its encoder and decoder, Adam on the rest")
    g.add_argument("--wm_opt", type=str, default="adam", choices=["adam", "muon"],
                   help="any algorithm: the world model's optimizer; muon: Muon on its hidden weight matrices (the RigidFormer's: its blocks' "
                   "attention and SwiGLU matrices), Adam on the rest")  # fmt: skip
    g.add_argument("--wm_mirror", type=float, default=0.0,
                   help="any algorithm: this share of every world-model batch replaced by its left-right mirror image (0: off)")
    g.add_argument("--wm_structured", action="store_true",
                   help="supermimic, MLP world model: semi-implicit world model, velocity increments and position corrections on top of "
                   "the drift that the velocities imply (worldmodels/semi_implicit.py)")  # fmt: skip
    g.add_argument("--contact_reward", type=float, default=0.0,
                   help="supermimic: weight of a foot-contact term in the imagined reward (tasks/contact.py; the evaluation's reward is "
                   "unchanged; 0: off)")  # fmt: skip
    g.add_argument("--ref_contact_fix", action="store_true",
                   help="supermimic, with --contact_reward: the reference's foot contact from each clip's own floor instead of a fixed 4.5 cm")
    g.add_argument("--fail_reset", type=float, default=0.0,
                   help="any algorithm: share of the resets one bin before recent failure points, by each bin's termination rate (0: off)")
    g.add_argument("--bin_fix", action="store_true",
                   help="any algorithm: a step's episodes that share a bin averaged in the bin's reset statistics (instead of one of them kept)")
    g.add_argument("--pi_lr_mult", type=float, default=1.0, help="any algorithm: the policy's learning rate is --lr times this")
    g.add_argument("--lr_decay_ret", type=float, default=0.0,
                   help="with --pi_lr_mult: once the logged training return reaches this, the multiplier ramps linearly to 1 over "
                   "--lr_decay_len iterations (0: never)")  # fmt: skip
    g.add_argument("--lr_decay_len", type=int, default=300)
    g.add_argument("--lr_decay_by", type=int, default=0,
                   help="with --lr_decay_ret: the ramp starts at this iteration at the latest (0: no deadline)")
    g.add_argument("--no_sync", action="store_true",
                   help="systems: no host-device synchronization in the update loop (normalizer freeze test, the valid starts' "
                   "selection, the non-finite policy step skipped on the device)")  # fmt: skip
    g.add_argument("--update_graphs", action="store_true",
                   help="systems (BlinkMimic's --cuda_graph): one world-model update and one policy and critic update (forward, "
                   "backward, optimizer steps, policy average) as CUDA graphs, captured once the replay buffer is full (needs "
                   "--no_sync --fused_adam)")  # fmt: skip
    g.add_argument("--graph_early", action="store_true",
                   help="systems, with --update_graphs: the replay buffer's fill level on the device, the updates captured at once")
    g.add_argument("--collect_graph", action="store_true",
                   help="systems, with --graph_early: one collection step up to the episode ends (observation, policy and action noise, "
                   "simulator step, reward, replay write, normalizer updates) replayed as a CUDA graph")  # fmt: skip
    g.add_argument("--reset_masked", action="store_true",
                   help="systems, with --collect_graph: the episode ends too (bin statistics, resets) as fixed-size masked "
                   "updates inside the collection graph; the reset worlds get the same values as with ids, random draws are made for "
                   "all worlds")  # fmt: skip
    g.add_argument("--ckpt_light", action="store_true",
                   help="systems: periodic checkpoints hold only the deployed policy, written by a background thread; last.pt is "
                   "written at the end only (and every --last_every_min)")  # fmt: skip
    g.add_argument("--last_every_min", type=float, default=None,
                   help="systems, with --ckpt_light: the full last.pt (what --resume continues from) also every this many minutes of "
                   "training, at the next periodic checkpoint, so that a run stopped before its end can be resumed (default: at the "
                   "end only)")  # fmt: skip
    add_eprobe_args(g)


def expand_preset(argv: list[str]) -> list[str]:
    """``argv`` with ``--algo blinkmimic`` (or ``--algo=blinkmimic``) replaced by ``--algo supermimic --preset blinkmimic`` and
    ``PRESET``, before the other arguments, which take precedence (the preset's switches cannot be turned off); other
    command lines are returned unchanged."""
    rest, found, i = [], False, 0
    while i < len(argv):
        if argv[i] == "--algo" and i + 1 < len(argv) and argv[i + 1] == "blinkmimic":
            found, i = True, i + 2
        elif argv[i] == "--algo=blinkmimic":
            found, i = True, i + 1
        else:
            rest.append(argv[i])
            i += 1
    if any(x == "--preset" or x.startswith("--preset=") for x in rest):
        raise ValueError("--preset is set by --algo blinkmimic")
    if not found:
        return argv
    if any(x == "--algo" or x.startswith("--algo=") for x in rest):
        raise ValueError("--algo blinkmimic is SuperMimic with a preset; give --algo once")
    return ["--algo", "supermimic", "--preset", "blinkmimic", *PRESET, *rest]


def check_args(args: argparse.Namespace):
    """The BlinkMimic options' constraints on a run's arguments (``ndm_gym.train``); raises ValueError."""
    if not 0 <= args.pi_ema < 1:
        raise ValueError(f"--pi_ema must be in [0, 1), got {args.pi_ema}")
    if not 0 <= args.wm_mirror <= 1:
        raise ValueError(f"--wm_mirror must be in [0, 1], got {args.wm_mirror}")
    if args.wm_mirror > 0 and args.wm_sample == "prioritized":
        raise ValueError("--wm_mirror and --wm_sample prioritized cannot be combined (a mirrored window's error would set its original's priority)")
    if args.wm_structured and (args.algo != "supermimic" or args.wm_backbone != "mlp"):
        raise ValueError("--wm_structured: SuperMimic's MLP world model only (SuperTrack's acceleration world model integrates semi-implicitly already)")
    if args.contact_reward < 0 or (args.contact_reward > 0 and args.algo != "supermimic"):
        raise ValueError("--contact_reward: a weight >= 0, and a term of SuperMimic's imagined reward (SuperTrack has no reward)")
    if args.ref_contact_fix and args.contact_reward == 0:
        raise ValueError("--ref_contact_fix needs --contact_reward")
    if not 0 <= args.fail_reset < 1:
        raise ValueError(f"--fail_reset must be in [0, 1), got {args.fail_reset}")
    if (args.fail_reset > 0 or args.bin_fix) and args.paired:
        raise ValueError("--fail_reset and --bin_fix: not implemented for the two-robot scene (--paired)")
    if args.pi_lr_mult <= 0 or args.lr_decay_ret < 0 or args.lr_decay_len < 1 or args.lr_decay_by < 0:
        raise ValueError("--pi_lr_mult > 0, --lr_decay_ret >= 0, --lr_decay_len >= 1, --lr_decay_by >= 0")
    if args.lr_decay_ret > 0 and args.pi_lr_mult == 1:
        raise ValueError("--lr_decay_ret decays --pi_lr_mult, which is 1")
    if args.lr_decay_by and not args.lr_decay_ret:
        raise ValueError("--lr_decay_by needs --lr_decay_ret")
    # BlinkMimic's systems options and evaluation-guided resets (train.py; algos/eprobe.py)
    if args.update_graphs and not (args.no_sync and args.fused_adam):
        raise ValueError("--update_graphs needs --no_sync --fused_adam")
    if args.update_graphs and (args.wm_cudagraphs or args.pi_cudagraphs or args.critic_cudagraphs):
        raise ValueError("--update_graphs captures whole updates: not with --wm_cudagraphs, --pi_cudagraphs or --critic_cudagraphs")
    if args.no_sync and (args.imag_fail_starts > 0 or args.wm_fail_windows > 0 or args.wm_sample == "prioritized" or args.pi_rollout_clip > 0):
        raise ValueError("--no_sync: not with --imag_fail_starts, --wm_fail_windows, --wm_sample prioritized or --pi_rollout_clip "
                         "(they read the device on the host)")  # fmt: skip
    if args.graph_early and not args.update_graphs:
        raise ValueError("--graph_early needs --update_graphs")
    if args.collect_graph and not args.graph_early:
        raise ValueError("--collect_graph needs --graph_early")
    if args.collect_graph and args.motions != "lafan1" and not args.paired:
        raise ValueError("--collect_graph: the one-robot LAFAN1 scene and the two-robot scene (--paired) only")
    if args.collect_graph and args.paired and not args.reset_masked:
        raise ValueError("--collect_graph with --paired needs --reset_masked (the two-robot graph captures the episode ends too)")
    if args.reset_masked and not args.collect_graph:
        raise ValueError("--reset_masked needs --collect_graph")
    if args.last_every_min is not None and not args.ckpt_light:
        raise ValueError("--last_every_min needs --ckpt_light (without it, every periodic checkpoint writes the full last.pt)")
    if args.last_every_min is not None and not args.last_every_min > 0:
        raise ValueError(f"--last_every_min must be > 0, got {args.last_every_min}")
    check_eprobe_args(args)


def check_resume(ck_args: dict, args: argparse.Namespace):
    """A resumed run must have the checkpoint's optimizers, policy average and world model; raises ValueError."""
    for k, default in RESUME_MATCH.items():
        if ck_args.get(k, default) != getattr(args, k):
            raise ValueError(f"--{k} must match the checkpoint's run ({ck_args.get(k, default)!r})")


def reset_kwargs(args: argparse.Namespace) -> dict:
    """The simulator's arguments for --fail_reset and --bin_fix (none while both are off)."""
    return {"fail_reset": args.fail_reset, "bin_fix": args.bin_fix} if args.fail_reset > 0 or args.bin_fix else {}


def policy_average(pi: nn.Module, decay: float) -> torch.optim.swa_utils.AveragedModel:
    """``--pi_ema`` (InstantTrack, BlinkMimic): an exponential moving average of the policy's weights and normalizers."""
    return torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay), use_buffers=True)


class Mirror:
    """``--wm_mirror`` (BlinkMimic): left-right mirror images of the G1's states, setpoints and sim context. The reflection
    M = diag(1, -1, 1) of the world's y axis maps positions and linear velocities to M x, rotations to M R M, angular
    velocities (pseudo-vectors) to -M w, and every joint to its left/right partner, the sign flipped for roll and yaw
    joints (and the waist's roll and yaw). In xi the joint offsets are mirrored as joints and the torso's lateral
    center-of-mass offset changes sign. Setpoints mirror as joints since the default pose and the action scales are
    symmetric."""

    def __init__(self, device: str):
        perm, sign = [], []
        for n in G1_JOINT_NAMES:
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(G1_JOINT_NAMES.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in ("roll", "yaw")) else 1.0)
        self.perm = torch.tensor(perm, device=device)
        self.sign = torch.tensor(sign, device=device)
        self.M = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return (p @ M, M @ R @ M, v @ M, -(w @ M), self.joints(q), self.joints(qd))

    def xi(self, xi: torch.Tensor) -> torch.Tensor:
        out = xi.clone()
        out[..., 1 : 1 + NUM_J] = self.joints(xi[..., 1 : 1 + NUM_J])
        out[..., 2 + NUM_J] = -xi[..., 2 + NUM_J]  # the lateral center-of-mass offset
        return out

    def windows(self, w: dict, n: int) -> dict:
        """Replay windows (``ReplayBuffer.sample_windows``) with the first ``n`` replaced by their mirror images."""
        out = dict(w)
        ms = lambda x: flatten_state(self.state(unflatten_state(x)))  # noqa: E731
        for k, fn in (("s", ms), ("s1", ms), ("a", self.joints), ("xi", self.xi)):
            out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
        return out


class Graphed:
    """``--update_graphs`` (BlinkMimic's ``--cuda_graph``): ``fn`` runs eagerly until ``ready()``; then eagerly on a side
    stream for ``warmup`` calls, is captured as a CUDA graph on the next call and replayed from then on. Each call is one
    update; its outputs are cloned out of the graph's memory."""

    def __init__(self, fn, ready, name: str, warmup: int = 3):
        self.fn, self.ready, self.name, self.warmup, self.calls, self.graph, self.out = fn, ready, name, warmup, 0, None, None
        self.side = None  # the warm-up calls' stream: one, so that each reuses the blocks the caching allocator keeps for it

    def __call__(self):
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            if self.side is None:
                self.side = torch.cuda.Stream()
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(self.side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # the eager phase's cached blocks back before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.fn()
            torch.cuda.synchronize()
            print(f"--update_graphs: captured {self.name} in {time.time() - t0:.1f} s, {torch.cuda.memory_reserved() / 2**30:.1f} GB reserved",
                  flush=True)  # fmt: skip
        self.graph.replay()
        return tuple(x.clone() for x in self.out)


class PolicyLR:
    """``--pi_lr_mult`` with ``--lr_decay_ret``, ``--lr_decay_len``, ``--lr_decay_by`` (BlinkMimic): the policy's learning
    rate is the schedule's times a multiplier. From the log row where the logged training return first reaches
    --lr_decay_ret (or at --lr_decay_by), the multiplier ramps linearly to 1 over --lr_decay_len iterations. train_muscles,
    whose log has no normalized return, ramps at the deadline --lr_decay_by alone."""

    def __init__(self, args: argparse.Namespace):
        self.mult, self.ret, self.len, self.by = args.pi_lr_mult, args.lr_decay_ret, args.lr_decay_len, args.lr_decay_by
        self.decay_at = None  # the trigger's iteration

    @property
    def on(self) -> bool:
        return self.mult != 1.0

    def at(self, it: int) -> float:
        """The multiplier at iteration ``it``."""
        if self.decay_at is None:
            return self.mult
        return self.mult + (1.0 - self.mult) * min(1.0, (it - self.decay_at) / self.len)

    def observe(self, it: int, ret: float | None) -> str | None:
        """At a log row, ``it`` iterations done and the logged training return ``ret`` (None: no return, the deadline
        alone): the trigger's message when it fires (once), else None."""
        if (self.ret <= 0 and self.by <= 0) or self.decay_at is not None:
            return None
        reached = self.ret > 0 and ret is not None and ret >= self.ret
        if not reached and not (self.by > 0 and it >= self.by):
            return None
        self.decay_at = it
        why = f"training return {ret:.3f} >= {self.ret}" if reached else f"deadline --lr_decay_by {self.by}"
        return f"--lr_decay_ret: {why} at it {it}: policy LR x{self.mult:g} -> x1 over {self.len} its"

    def state_dict(self) -> dict:
        """The trigger's iteration for a checkpoint (nothing without a trigger)."""
        return {"pi_lr_decay_at": self.decay_at} if self.ret > 0 or self.by > 0 else {}

    def load_state_dict(self, ck: dict):
        if self.ret > 0 or self.by > 0:
            if "pi_lr_decay_at" not in ck:
                raise ValueError("--lr_decay_ret, --lr_decay_by: the checkpoint has no trigger state (pi_lr_decay_at)")
            self.decay_at = ck["pi_lr_decay_at"]
