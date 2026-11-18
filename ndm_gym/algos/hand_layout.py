"""train_hand --layout AL | AWL: hand-object training's iteration on two or three GPUs at the same time, each step on the
others' results of the previous iteration. LAFAN1's split (algos.layouts.Split; docs/xmimic.md, section 4) with the hand
agents (algos.hand_algos), off by default: --layout 1GPU is the loop as without this module.

- AL, two GPUs. Actor (the first GPU the trainer sees): the simulator, the collection, the replay, the world model with its
  updates, the draw of the next imagination batch. Policy learner (the second): the policy update with SuperMimic's
  critic and the policy average (--pi_ema).
- AWL, three GPUs. The world-model updates move to the second GPU, on a replica of the replay that receives the row the
  actor wrote; the policy learner is the third.

--layout_devices places the steps otherwise, as indices of the GPUs the trainer sees: AWL on two GPUs is 0 0 1, the
world-model learner beside the actor (its updates then share the actor's GPU and its queue).

What crosses the devices, and when, is layouts.Split's: at the end of every iteration the actor draws the next batch of
imagination starts (the hand agents' ``sample_valid_starts_nosync``: the state, the frame and the action before it, no
history); the batch, the policy's normalizer and in AL the world model go to the learner, the policy's weights to the
actor; in AWL the world model goes from its learner to the policy learner, the iteration's replay row (an iteration
collects one step) and the world model's normalizers from the actor to the world-model learner. The staleness is
XMimic's: the policy update of iteration i imagines from the batch drawn at the end of iteration i - 1, in the world model
as iteration i - 1's updates left it (AWL: updates on the rows up to iteration i - 2), with the policy's normalizer of then;
the collection acts with the policy of iteration i - 1's update. A layout so changes results.

The loop (train_hand): ``build`` after the collection step's object (hand_fast.Collector keeps the actor's agent) and
before the warm-up collection; ``update_functions`` after the warm-up (AWL's replica starts from it); per iteration the
collection, the world-model updates, the policy update from the learner's first batch on, then ``Split.exchange``, which
waits for the devices' work of the iteration before (the run-ahead limit). --pi_ema's average lives on the learner and
steps in its graph (hand_algos ``policy_update_graphable``); the loop counts its steps as without a layout. --phase_times
synchronizes every device of the layout at every phase's boundary and times the exchange apart (hand_fast.PhaseTimes):
each phase's own time, the steps one after another; the layout's speed is train_log.csv's it_per_s without it.
"""

from __future__ import annotations

import argparse
import os

import torch

from .xmimic import LAYOUTS


def add_hand_layout_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("layout", "the iteration split across GPUs (ndm_gym.algos.hand_layout), off by default")
    g.add_argument("--layout", type=str, default="1GPU", choices=list(LAYOUTS),
                   help="the iteration's steps at the same time on the first GPUs the trainer sees, each on the others' results of the "
                   "previous iteration (changes results). AL: actor (simulator, collection, world model) | policy learner; AWL: actor | "
                   "world-model learner | policy learner (algos/hand_layout.py; needs --no_sync --update_graphs)")  # fmt: skip
    g.add_argument("--layout_devices", type=int, nargs="+", default=None,
                   help="with --layout: its devices, actor first, as indices of the GPUs the trainer sees (the actor's 0, the simulator's); "
                   "default: the first GPUs, one each. AWL on two GPUs: 0 0 1 (the world-model learner beside the actor, on its own "
                   "replica of the replay)")  # fmt: skip


def check_hand_layout_args(args: argparse.Namespace):
    """The layout's constraints on a run's arguments; raises ValueError."""
    if args.layout == "1GPU":
        if args.layout_devices is not None:
            raise ValueError("--layout_devices places a layout's steps: needs --layout AL or AWL")
        return
    if not (args.no_sync and args.update_graphs):
        raise ValueError(f"--layout {args.layout} needs --no_sync --update_graphs (the normalizers written in place, the replay's write head "
                         "on the device, the updates as CUDA graphs on their devices)")  # fmt: skip
    if args.resume:
        raise ValueError(f"--layout {args.layout}: --resume is not implemented (a checkpoint's state goes to one device's object)")
    d = args.layout_devices
    if d is not None and (len(d) != LAYOUTS[args.layout] or d[0] != 0 or min(d) < 0 or d[-1] == 0):
        raise ValueError(f"--layout_devices {' '.join(map(str, d))}: {LAYOUTS[args.layout]} indices for --layout {args.layout}, the actor's 0 first "
                         "and the policy learner's on another GPU")  # fmt: skip


def build(args, actor, buffer, tables):
    """The layout's split and its policy average: ``actor`` the agent on the simulator's device, ``buffer`` the replay with
    its write head on the device, ``tables(device)`` the clip's kinematics and task on ``device``. The devices are the
    first GPUs the trainer sees, or those of --layout_devices; on the CPU (the tests) every one the CPU. Returns
    (layouts.Split, the policy average on the learner or None)."""
    from .hand_algos import HAND_ALGOS, START_KEYS
    from .hand_fast import PolicyAverage
    from .layouts import Split

    if os.environ.get("HAND_PROFILE", "0") == "1":
        raise ValueError(f"--layout {args.layout}: HAND_PROFILE times the steps on one device; use --phase_times")
    n = LAYOUTS[args.layout]
    if torch.device(actor.dev).type != "cuda":
        devices = (actor.dev,) * n
    elif args.layout_devices is None:
        devices = None
    else:
        if max(args.layout_devices) >= torch.cuda.device_count():
            raise ValueError(f"--layout_devices {args.layout_devices}: the trainer sees {torch.cuda.device_count()} GPUs (CUDA_VISIBLE_DEVICES)")
        devices = tuple(f"cuda:{i}" for i in args.layout_devices)

    def algo_on(device):  # the agent for another device, with the clip's tables there
        return HAND_ALGOS[args.algo](args, *tables(device), device)

    split = Split(args, actor, algo_on, buffer, devices=devices, start_keys=START_KEYS, hist=0, steps=1)
    average = None
    if args.pi_ema > 0:  # the average of the learner's policy, stepped in its policy update's graph
        average = split.learner.average = PolicyAverage(split.learner.pi, args.pi_ema)
    return split, average


def update_functions(split, average):
    """hand_fast.update_functions under the layout, after the warm-up collection: the iteration's world-model updates on
    their owner's device (returns the last loss) and the policy update with the average's step on the learner's (returns
    the agent's values for the log), each a CUDA graph on its device; on the CPU (the tests) the plain functions."""
    wm_graph, pi = split.update_graphs(graphs=torch.device(split.actor.dev).type == "cuda")

    def wm():
        return wm_graph()[0]

    if average is None:
        return wm, pi

    def policy_counted():
        out = pi()
        average.count()  # on the host: a replay does not run the step's Python side
        return out

    return wm, policy_counted
