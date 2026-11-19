"""XMimic's machinery as options of ndm_gym.train (docs/algorithms/xmimic.md).

XMimic (N. Bohlinger and J. Peters, 2026; https://github.com/nico-bohlinger/xmimic, MIT) is BlinkMimic's trainer rebuilt in
JAX: the same algorithm and hyperparameters (here ``--algo blinkmimic``), with an iteration turned into a shorter chain of
larger GPU operations and its steps run at the same time on up to four GPUs. The changes that carry over to this library's
PyTorch pipeline are options here, off by default:

- ``--eprobe_gpu i``: the probe worker of the evaluation-guided resets (algos.eprobe) on the i-th GPU the trainer sees
  instead of the trainer's own, as XMimic's prober GPU (its layouts FP, ALP and AWLP). The sweeps and what they set are the
  same; they no longer take the trainer's GPU while they run.
- ``--njmax n``: MuJoCo Warp's constraint rows per world in the training simulator (XMimic: 256, against the scene's 500).
  A world that needs more loses the extra rows for that step, so this option changes results. The evaluation and the probe
  worker keep the scene's.
- ``--gc_freeze``: Python's garbage collector frozen once the run is set up (``gc.freeze``): a full collection then scans
  only what the loop allocates. Host time only.

and its learner's restructuring (``xmimic/fast.py``), for SuperTrack's and SuperMimic's updates with the MLP world model on
the one-robot scene; the values are the default's up to rounding unless stated:

- ``--pi_encode_once`` (train.py's option, SuperTrack's before) now also in SuperMimic's imagination: the policy's goal
  encoder once over every imagined step's goal, the steps its decoder (the policy's alone: with either world model).
- ``--imag_hoist`` (with --pi_encode_once): the imagined steps compute only what depends on the imagined state. Before
  the rollout: the reference's terms of all its frames, the noise of all its steps (one draw per kind instead of one per
  step: another random stream) and the start history's normalization; each new observation is normalized once, when it
  is produced, instead of with its history at every step.
- ``--pi_deferred_grads`` (with --imag_hoist) and ``--wm_deferred_grads``: the weight and bias gradients of the policy
  decoder's Linear layers in the BPTT update, and of the world model's in its k-step fit, as one matrix product per layer
  over all steps after the backward pass (``core.nets.run_mlp_deferred``, ``deferred_grads``), instead of one per step
  inside it. ``--deferred_bf16``: those products in bf16 under --amp bf16 (their results rounded to bf16, as each step's is
  by default) instead of fp32 products of the bf16 operands.
- ``--muon_batched``: Muon's Newton-Schulz iterations of equally shaped matrices as batched products (``core.muon``).
- ``--wm_presample``: the windows of an iteration's --wm_batches world-model updates drawn, read and mirrored in one
  pass before the updates (``ReplayBuffer.sample_windows_multi``); with --update_graphs the updates are one CUDA graph.
  The same windows as one update at a time unless the fit draws random numbers itself (--wm_input_noise, --wm_smooth_w).

and its iteration split across GPUs:

- ``--layout AL | AWL``: the iteration's steps on two or three GPUs at the same time, each on the others' results of the
  previous iteration (algos.layouts; XMimic's layouts of those names without their prober, which is ``--eprobe_gpu``).
  This one changes results: the policy update's batch and world model are one iteration older.
"""

from __future__ import annotations

import argparse

LAYOUTS = {"1GPU": 1, "AL": 2, "AWL": 3}  # --layout: the GPUs it takes
LAYOUT_NEEDS = ("no_sync", "update_graphs", "graph_early", "collect_graph", "reset_masked")  # the systems options a split layout relies on


def add_xmimic_args(parser: argparse.ArgumentParser):
    """XMimic's options (off by default), in a group of their own."""
    g = parser.add_argument_group("XMimic", "machinery of XMimic that carries over to this pipeline (algos.xmimic, docs/algorithms/xmimic.md)")
    g.add_argument("--eprobe_gpu", type=int, default=-1,
                   help="with --eprobe_share: the probe worker on this one of the GPUs the trainer sees (an index into "
                   "CUDA_VISIBLE_DEVICES; the same sweeps; -1: the trainer's GPU)")  # fmt: skip
    g.add_argument("--njmax", type=int, default=0,
                   help="MuJoCo Warp's constraint rows per world in the training simulator (XMimic: 256; changes results: a world "
                   "that needs more loses the rest for that step; 0: the scene's 500, 1000 with --paired)")  # fmt: skip
    g.add_argument("--gc_freeze", action="store_true", help="freeze Python's garbage collector once the run is set up (gc.freeze; host time only)")
    g.add_argument("--imag_hoist", action="store_true",
                   help="with --pi_encode_once: the imagined steps compute only what depends on the imagined state; the reference's terms and "
                   "the noise of all steps before the rollout (one draw per kind: another random stream), each observation normalized once "
                   "(equal up to rounding given the same noise)")  # fmt: skip
    g.add_argument("--pi_deferred_grads", action="store_true",
                   help="with --imag_hoist: the policy decoder's weight gradients as one matrix product per layer over all imagined steps, "
                   "after the backward pass (equal up to rounding)")  # fmt: skip
    g.add_argument("--wm_deferred_grads", action="store_true",
                   help="the MLP world model's weight gradients as one matrix product per layer over the fit's window, after the backward "
                   "pass (equal up to rounding)")  # fmt: skip
    g.add_argument("--deferred_bf16", action="store_true",
                   help="with --pi_deferred_grads or --wm_deferred_grads under --amp bf16: their products in bf16, the results rounded to "
                   "bf16 as each step's is by default, instead of fp32 products of the bf16 operands")  # fmt: skip
    g.add_argument("--muon_batched", action="store_true",
                   help="with --pi_opt muon or --wm_opt muon: Muon's Newton-Schulz iterations of equally shaped matrices as batched "
                   "products (equal up to the products' rounding)")  # fmt: skip
    g.add_argument("--wm_presample", action="store_true",
                   help="the windows of the iteration's --wm_batches world-model updates drawn, read and mirrored in one pass before the "
                   "updates (the same windows unless the fit draws random numbers: --wm_input_noise, --wm_smooth_w)")  # fmt: skip

    g.add_argument("--layout", type=str, default="1GPU", choices=list(LAYOUTS),
                   help="SuperMimic and SuperTrack, MLP world model: the iteration's steps at the same time on the first GPUs the "
                   "trainer sees, each on the others' results of the previous iteration (changes results). AL: actor (simulator, "
                   "collection, world model) | policy learner; AWL: actor | world-model learner | policy learner (algos/layouts.py)")  # fmt: skip

# the learner's options (xmimic/fast.py): SuperTrack's and SuperMimic's updates, the MLP world model, the one-robot scene
LEARNER = ("imag_hoist", "pi_deferred_grads", "wm_deferred_grads", "deferred_bf16", "muon_batched", "wm_presample")


def check_args(args: argparse.Namespace):
    """The options' constraints; raises ValueError."""
    if args.eprobe_gpu < -1:
        raise ValueError(f"--eprobe_gpu must be a GPU index or -1, got {args.eprobe_gpu}")
    if args.eprobe_gpu >= 0 and args.eprobe_share == 0:
        raise ValueError("--eprobe_gpu places the probe worker: needs --eprobe_share")
    if args.njmax < 0:
        raise ValueError(f"--njmax must be >= 0, got {args.njmax}")
    on = [n for n in LEARNER if getattr(args, n)]
    if on and args.algo not in ("supertrack", "supermimic"):
        raise ValueError(f"--{on[0]}: SuperTrack's and SuperMimic's updates only, not --algo {args.algo}")
    if on and args.wm_backbone != "mlp":
        raise ValueError(f"--{on[0]}: the MLP world model only, not --wm_backbone {args.wm_backbone}")
    if args.paired and on:  # (--pi_encode_once is checked there: a robot's imagined goals are its own clip's frames)
        raise ValueError(f"--{on[0]}: the one-robot scene only, not checked with --paired")
    if args.pi_cudagraphs and (args.imag_hoist or args.pi_deferred_grads):
        raise ValueError("--imag_hoist, --pi_deferred_grads: not with --pi_cudagraphs (not checked under the compiled parts' CUDA graphs; "
                         "--update_graphs captures the whole update)")  # fmt: skip
    if args.wm_cudagraphs and (args.wm_deferred_grads or args.wm_presample):
        raise ValueError("--wm_deferred_grads, --wm_presample: not with --wm_cudagraphs (not checked under the compiled parts' CUDA graphs; "
                         "--update_graphs captures the whole updates)")  # fmt: skip
    if args.pi_encode_once and args.pi_rollout_clip > 0:
        raise ValueError("--pi_encode_once: not with --pi_rollout_clip (its rerun of the policy reads every step's goal)")
    if args.imag_hoist and not args.pi_encode_once:
        raise ValueError("--imag_hoist needs --pi_encode_once")
    if args.pi_deferred_grads and not args.imag_hoist:
        raise ValueError("--pi_deferred_grads needs --imag_hoist")
    if args.deferred_bf16 and (not (args.pi_deferred_grads or args.wm_deferred_grads) or args.amp != "bf16"):
        raise ValueError("--deferred_bf16: with --pi_deferred_grads or --wm_deferred_grads, under --amp bf16")
    if args.muon_batched and args.pi_opt != "muon" and args.wm_opt != "muon":
        raise ValueError("--muon_batched needs --pi_opt muon or --wm_opt muon")
    if args.wm_presample and (args.wm_sample == "prioritized" or args.wm_fail_windows > 0):
        raise ValueError("--wm_presample: not with --wm_sample prioritized (the priorities change between the updates) or --wm_fail_windows")
    if args.layout != "1GPU":  # the iteration split across GPUs (algos.layouts): what it relies on and what it does not cover
        if not all(getattr(args, k) for k in LAYOUT_NEEDS):
            raise ValueError(f"--layout {args.layout} needs {' '.join('--' + k for k in LAYOUT_NEEDS)} (the steps as CUDA graphs, "
                             "nothing read on the host)")  # fmt: skip
        if args.algo not in ("supermimic", "supertrack"):
            raise ValueError(f"--layout {args.layout}: SuperMimic and SuperTrack (--algo blinkmimic included), not --algo {args.algo}")
        if args.wm_backbone != "mlp":
            raise ValueError(f"--layout {args.layout}: the MLP world model only")
        if args.resume:
            raise ValueError(f"--layout {args.layout}: --resume is not implemented (a checkpoint's state goes to one device's object)")
        if args.pi_updates != 1:
            raise ValueError(f"--layout {args.layout}: one policy update per iteration (the learner holds one batch)")
        if on:  # Split replays wm_update_graphable and policy_update_graphable on a batch it was sent
            raise ValueError(f"--layout {args.layout}: not with the learner's options (--{on[0]}): the two are not checked together, "
                             "and --wm_presample's one graph of the iteration's updates is not the layouts'")  # fmt: skip


def probe_device(index: int, visible: str | None) -> str:
    """``--eprobe_gpu index``: the probe worker's CUDA_VISIBLE_DEVICES, entry ``index`` of the trainer's ``visible`` (the
    device ``index`` itself where the trainer's is unset)."""
    if not visible:
        return str(index)
    ids = [v.strip() for v in visible.split(",") if v.strip()]
    if index >= len(ids):
        raise ValueError(f"--eprobe_gpu {index}: the trainer sees {len(ids)} GPU(s) (CUDA_VISIBLE_DEVICES={visible})")
    return ids[index]
