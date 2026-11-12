"""Training loop (SuperMimic Algorithm A.1): collect -> fit world model -> BPTT policy update.

``--algo`` selects the algorithm (:mod:`ndm_gym.algos`); the simulator, observations, networks, replay buffer and
schedule are shared. Run: ``python -m ndm_gym.train --algo supertrack|supermimic --log_dir logs/<run>``. ``--resume``
continues from a run's ``last.pt`` (the networks with their normalizers, the optimizers' states, the algorithm's own
state); the replay buffer is not saved, so a resumed run refills it with ``--warmup_steps`` of collection.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ndm_gym.sim.launcher import add_launcher_args, launch_simulation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_train_args(parser: argparse.ArgumentParser):
    """Training arguments (also used by the offline world-model benchmark, ``ndm_gym.wmbench.fit``)."""
    parser.add_argument("--algo", type=str, required=True, choices=["supermimic", "supertrack"])
    parser.add_argument("--st_reg", type=float, default=0.01, help="supertrack: weight of the L2 and L1 PD-offset regularizers")
    parser.add_argument("--st_imag_noise", type=float, default=0.1, help="supertrack: Gaussian noise on PD offsets in imagination")
    parser.add_argument("--st_no_root_terms", action="store_true", help="supertrack: drop the world-space root terms (paper base loss)")
    parser.add_argument("--st_no_task_regs", action="store_true", help="supertrack: drop the task's action-rate and joint-limit penalties")
    parser.add_argument("--st_substeps", type=int, default=1, help="supertrack: world-model integration substeps (1 = paper, 4 = simulator)")
    parser.add_argument("--st_balance_w", type=float, default=1.0, help="supertrack: extra weight on the height and up-vector terms")
    parser.add_argument("--imag_start_perturb", type=float, default=0.0, help="both: std of horizontal root-velocity kicks on imagination starts (m/s)")
    parser.add_argument("--bptt_clip_k", type=float, default=0.0,
                        help="both: in the policy's BPTT, cap each rollout's state gradient after every imagined step at k x the batch median (0: off)")
    parser.add_argument("--bptt_decay", type=float, default=1.0,
                        help="both: in the policy's BPTT, the gradient through every imagined state times this (a loss d steps after an action "
                        "weighs lam^d; 1: off)")  # fmt: skip
    parser.add_argument("--pi_rollout_clip", type=float, default=0.0,
                        help="both: each rollout's part of the policy gradient at most k x the median over the rollouts, sized by the gradient "
                        "that reaches the policy's output (0: off)")  # fmt: skip
    parser.add_argument("--wm_bptt_fp32", action="store_true",
                        help="both: the world model without bf16 autocast in the policy's BPTT (its matmuls still run in TF32 under allow_tf32)")
    parser.add_argument("--imag_fail_starts", type=float, default=0.0,
                        help="both: fraction of imagination starts drawn from the last steps before the replay's terminations")
    parser.add_argument("--imag_fail_window", type=int, default=50, help="both: steps before a termination that count as pre-failure (50 = 1 s)")
    parser.add_argument("--sm_residual", action="store_true",
                        help="supermimic: the policy outputs offsets about the reference pose (SuperTrack's action parameterization)")  # fmt: skip
    parser.add_argument("--st_com", action="store_true", help="supertrack: add a center-of-mass (position, velocity) tracking term")
    parser.add_argument("--st_quad_w", type=float, default=0.0, help="supertrack: weight of e^2/s next to |e| (stronger pull on large errors)")
    parser.add_argument("--st_late_w", type=float, default=0.0, help="supertrack: extra weight on the last imagined steps (linear ramp)")
    parser.add_argument("--st_fail_w", type=float, default=0.0,
                        help="supertrack: cost of every imagined step past the termination thresholds, in calibrated per-step tracking losses")
    parser.add_argument("--st_fail_margin", type=float, default=1.0,
                        help="supertrack: the failure penalty's thresholds as a fraction of the task's termination thresholds (< 1: earlier)")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=320_000)
    parser.add_argument("--buffer_steps", type=int, default=1000)
    parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--goal_frames", type=int, default=10)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm_window", type=int, default=8)
    parser.add_argument("--wm_hidden", type=int, nargs="+", default=[2048, 2048, 2048], help="world-model MLP widths")
    parser.add_argument("--wm_act", type=str, default="silu", choices=["silu", "gelu", "elu", "mish", "relu"])
    parser.add_argument("--wm_no_ln", action="store_true", help="world model without LayerNorm")
    parser.add_argument("--wm_input_noise", type=float, default=0.0, help="std of noise on normalized world-model inputs while fitting")
    parser.add_argument("--wm_smooth_w", type=float, default=0.0,
                        help="both: weight of a finite-difference Jacobian penalty on the world model's network while fitting (0: off)")  # fmt: skip
    parser.add_argument("--wm_smooth_eps", type=float, default=0.1, help="both: the smoothness penalty's input step (normalized units, >= 0.01)")
    parser.add_argument("--wm_smooth_dirs", type=int, default=1, help="both: the smoothness penalty's random input directions per window")
    parser.add_argument("--wm_backbone", type=str, default="mlp", choices=["mlp", "rigidformer"], help="world-model network")
    parser.add_argument("--rf_dim", type=int, default=128, help="rigidformer: token width")
    parser.add_argument("--rf_layers", type=int, default=3, help="rigidformer: Transformer blocks")
    parser.add_argument("--rf_heads", type=int, default=4, help="rigidformer: attention heads")
    parser.add_argument("--rf_registers", type=int, default=2, help="rigidformer: extra learned register tokens")
    parser.add_argument("--rf_no_tree_bias", action="store_true", help="rigidformer: no kinematic-tree attention bias")
    parser.add_argument("--rf_no_rope", action="store_true", help="rigidformer: no rotary body positions")
    parser.add_argument("--rf_rope_detach", action="store_true", help="rigidformer: no gradient through the rotary positions in the policy's BPTT")
    parser.add_argument("--rf_fused_rope", action="store_true",
                        help="rigidformer: the rotary embedding fused into one elementwise expression (faster when compiled, equal up to rounding)")
    parser.add_argument("--rf_anchors", type=int, default=1,
                        help="rigidformer: anchor points per body for the rotary attention (RigidFormer's ARoPE; 1: the link origin)")
    parser.add_argument("--rf_anchor_mode", type=str, default="mean", choices=["mean", "concat"],
                        help="rigidformer, --rf_anchors > 1: the anchors' rotary angles mean-pooled (ARoPE, their centroid's) or concatenated")
    parser.add_argument("--rf_rope_center", action="store_true",
                        help="rigidformer: rotary positions relative to their centroid over the bodies (and anchors), where the registers sit")
    parser.add_argument("--rf_merge", type=str, default="none", choices=["none", "distal", "coincident", "joints"],
                        help="rigidformer: bodies that share a token (worldmodels.rigidformer.token_groups): distal (the G1's wrists and ankles, "
                        "24 tokens), coincident (also its waist with the torso and its shoulders' pitch and roll links, 20 tokens) or joints "
                        "(one token per anatomical joint, 14 tokens)")  # fmt: skip
    parser.add_argument("--wm_sample", type=str, default="uniform", choices=["uniform", "prioritized"],
                        help="world-model window sampling: uniform, or proportional to the window's last error^alpha")
    parser.add_argument("--wm_prio_alpha", type=float, default=0.6)
    parser.add_argument("--wm_fail_windows", type=float, default=0.0,
                        help="both: fraction of world-model windows that end shortly before one of the replay's terminations")
    parser.add_argument("--wm_fail_window", type=int, default=50, help="both: steps before a termination such windows may end in")
    parser.add_argument("--st_wm_loss", type=str, default="l1", choices=["l1", "l2"], help="supertrack: world-model window loss")
    parser.add_argument("--wm_batches", type=int, default=10)
    parser.add_argument("--wm_batch", type=int, default=2048)
    parser.add_argument("--pi_batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wm_lr", type=float, default=None, help="world-model learning rate, on the same schedule (default: --lr)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_std_scale", type=float, default=1.0,
                        help="scale of the reward's tracking-kernel widths during training (the policy objective of supermimic, "
                        "and reset sampling); evaluation keeps the task's")
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--critic_minibatches", type=int, default=4)
    parser.add_argument("--no_critic", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
    parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
    parser.add_argument("--pi_updates", type=int, default=1, help="both: policy updates per iteration (1: one, as before)")
    parser.add_argument("--lr_decay_iters", type=int, default=250_000)
    parser.add_argument("--lr_decay_hours", type=float, default=None,
                        help="decay the learning rate over this much training wall-clock time instead of --lr_decay_iters")  # fmt: skip
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--max_hours", type=float, default=None, help="stop with a checkpoint after this much training wall-clock time")
    parser.add_argument("--ckpt_hours", type=float, nargs="+", default=[],
                        help="also checkpoint when the training wall-clock time first reaches each of these hours")  # fmt: skip
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
    parser.add_argument("--motions", type=str, default="lafan1", choices=["lafan1", "interhuman"],
                        help="the clips' source: LAFAN1's G1 CSVs, or two-person clips retargeted to two G1s "
                        "(tools/convert_interhuman.py)")  # fmt: skip
    parser.add_argument("--paired", action="store_true",
                        help="the two-robot scene: each env's two G1s track the two sides of a two-person clip "
                        "(--num_envs counts robots)")  # fmt: skip
    parser.add_argument("--clips", type=str, default=None,
                        help="train on the clips listed in this file (one name per line) instead of every clip in --motion_dir")  # fmt: skip
    parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")


def training_reward_cfg(reward_std_scale: float):
    """The task's reward configuration with sharper (< 1) or wider tracking kernels, for training only; evaluation
    keeps the task's (``--reward_std_scale``)."""
    from ndm_gym.tasks.tracking import RewardCfg

    cfg = RewardCfg()
    if reward_std_scale != 1.0:
        for name in ("body_pos_std", "body_ori_std", "body_linvel_std", "body_angvel_std", "root_pos_std", "root_ori_std"):
            setattr(cfg, name, reward_std_scale * getattr(cfg, name))
    return cfg


def elementwise_3x3_products():
    """``--elementwise_3x3``: inside compiled functions, inductor decomposes the batched matrix products whose dimensions
    are all at most 3 (rotation matrices, 3-vectors) into fp32 elementwise products and sums that it fuses with their
    neighbours, instead of one cuBLAS call each; under allow_tf32, cuBLAS runs the products with a transposed factor
    (R^T R') as TF32 tensor-core GEMMs on 64 x 64 tiles (relative error 6e-4), the others in fp32."""
    import torch
    import torch.compiler.config
    from torch._inductor import decomposition

    default = decomposition.decompositions[torch.ops.aten.bmm.default]

    def bmm(a, b, out_dtype=None):
        if out_dtype is None and a.dtype == torch.float32 and max(a.shape[1], a.shape[2], b.shape[2]) <= 3:
            return (a.unsqueeze(-1) * b.unsqueeze(1)).sum(2)
        return default(a, b, out_dtype)

    decomposition.decompositions[torch.ops.aten.bmm.default] = bmm
    decomposition.fast_random_decomps.cache_clear()
    # the compile caches do not key on the decomposition table (a graph cached without this one would keep the cuBLAS
    # products): their key's tag tells the two apart
    torch.compiler.config.cache_key_tag = "ndm_gym:elementwise_3x3"


def combo_kernels():
    """``--combo_kernels``: inside compiled functions, inductor runs independent kernels of a graph as one launch
    (``torch._inductor.config.combo_kernels``): the world models' kinematics, features and losses are many small kernels
    (an A100, RigidFormer's world-model step at 2048 windows: 223 forward and 498 backward kernels without, 93 and 307
    with). The kernels' code is generated as without; the combined kernels' launch configurations may differ."""
    import torch._inductor.config

    torch._inductor.config.combo_kernels = True


def add_speed_args(parser: argparse.ArgumentParser, rigidformer: bool = True):
    """Speed options of the training run (off by default). The tools that rebuild a run's arguments (tools/grad_stats.py)
    add them after ``add_train_args``, as ``parse_args`` does. ``rigidformer``: also the G1 RigidFormer's kernel options
    (a trainer with a RigidFormer of its own leaves them out: train_muscles)."""
    parser.add_argument("--fused_adam", action="store_true",
                        help="Adam's fused CUDA kernel for every optimizer, one launch per step (changes results)")
    parser.add_argument("--elementwise_3x3", action="store_true",
                        help="3x3 products in compiled functions as fused fp32 elementwise products (elementwise_3x3_products; changes results)")
    parser.add_argument("--wm_cudagraphs", action="store_true",
                        help="the world-model update's compiled rollout (RigidFormer: step) and loss as CUDA graphs (same kernels)")
    parser.add_argument("--pi_cudagraphs", action="store_true",
                        help="the policy update's compiled imagination step and losses as CUDA graphs (same kernels)")
    parser.add_argument("--critic_cudagraphs", action="store_true",
                        help="supermimic: the critic's minibatch step as a CUDA graph, with capturable Adam (changes results)")
    parser.add_argument("--combo_kernels", action="store_true",
                        help="independent small kernels of compiled functions launched together (combo_kernels; may change results by rounding)")
    parser.add_argument("--compile_eager", action="store_true",
                        help="compile the forward kinematics that otherwise run eagerly: the collection's reward and world-model normalizer "
                             "features, supermimic's critic input in the policy loss (hundreds of small launches each; changes results by rounding)")
    parser.add_argument("--goal_table", action="store_true",
                        help="the policy's goal of every reference frame computed once (MotionLib.cache_goals) and read in collection and in "
                             "every imagined step (equal up to rounding; 1.2 GB for LAFAN1)")
    parser.add_argument("--pi_encode_once", action="store_true",
                        help="supertrack: the policy's goal encoder runs once per policy update, over every imagined step's goal (they depend on the "
                             "reference frames only), and the steps its decoder (equal up to rounding: the encoder's products over all steps at once)")
    if not rigidformer:
        return
    parser.add_argument("--rf_fused_train", action="store_true",
                        help="rigidformer: the blocks' bf16 forward and backward passes as fused Triton kernels (kernels/rf_train.py; changes "
                             "results; fp32 passes as without)")
    parser.add_argument("--rf_fit_keep_ops", action="store_true",
                        help="rigidformer, with --rf_fused_train: the world-model fit's fused forward passes keep the bf16 operands their "
                             "backward passes recompute (equal up to rounding; about 2.4 GB more during the fit at 2048 windows)")
    parser.add_argument("--rf_bptt_keep_blocks", action="store_true",
                        help="rigidformer, with --rf_fused_train: the policy's BPTT keeps the fused blocks' tensors, recomputing only the "
                             "read-out (same results, about 12 GB more at 4096 rollouts x 32 steps)")


def parse_args():
    from ndm_gym import configs
    from ndm_gym.algos.blinkmimic import NOT_PORTED, add_blinkmimic_args, check_args, expand_preset

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_speed_args(parser)
    add_blinkmimic_args(parser)
    add_launcher_args(parser)
    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first; --algo blinkmimic: SuperMimic with BlinkMimic's recipe
        args = parser.parse_args(expand_preset(configs.expand(sys.argv[1:], "train")))
        check_args(args)
    except ValueError as e:
        parser.error(str(e))
    if args.pi_updates < 1:
        parser.error(f"--pi_updates must be at least 1, got {args.pi_updates}")
    if args.preset:
        print(f"--algo blinkmimic: SuperMimic with BlinkMimic's recipe; not ported: {NOT_PORTED}", flush=True)
    if args.critic_cudagraphs and args.algo != "supermimic":
        parser.error("--critic_cudagraphs: only SuperMimic's critic step has a CUDA graph")
    if args.pi_encode_once and args.algo != "supertrack":
        parser.error("--pi_encode_once: SuperTrack's imagination only")
    if args.paired and (args.motions != "interhuman" or args.num_envs % 2):
        parser.error("--paired: two-person clips (--motions interhuman) and an even --num_envs (two robots per env)")
    return args


class GraphedCollector:
    """train.py --collect_graph (BlinkMimic): one collection step up to the episode ends (observation, policy and action noise,
    simulator step, reward, replay write, normalizer updates; with --reset_masked the episode ends too) as one CUDA graph
    (sim/rigid/isaaclab_mjwarp.py's GraphedStep). The eager loop's computations on fixed tensors: the observation history
    ``obs_hist``, the simulator's counters (TrackingSim.step_core) and the replay's device-side write head."""

    def __init__(self, env, algo, buffer, mdp, motion, obs_hist, args):
        from ndm_gym.sim.rigid.isaaclab_mjwarp import GraphedStep

        self.env, self.algo, self.buffer, self.mdp, self.motion, self.obs_hist, self.args = env, algo, buffer, mdp, motion, obs_hist, args
        self.step = GraphedStep(env, self.core, args.reset_masked, "one collection step (--collect_graph)")

    def core(self):
        """One collection step (the graph's body); returns the step's outputs the training log reads."""
        import torch

        from ndm_gym.state import unflatten_state

        env, algo, mdp, args = self.env, self.algo, self.mdp, self.args
        S = env.capture_state()  # the root velocities' buffers, made current before the step
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        fresh = env.ep_step == 0
        oh = self.obs_hist
        oh_new = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(oh), torch.cat((oh[:, 1:], o.unsqueeze(1)), 1))
        g = self.motion.goal(env.f, args.goal_frames)
        algo.update_pi_norms(o, g)
        a = algo.setpoint(algo.pi(oh_new, g), env.f)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step_core(a, S)
        self.buffer.add_device(out)
        algo.update_wm_norms(unflatten_state(out["s"]), unflatten_state(out["s1"]), out["a"], out["xi"])
        oh.copy_(oh_new)
        core = {"done": out["done"], "term": out["term"], "f": out["f"], "r": out["r"].mean() / 0.1, "rate": out["terms"]["action_rate"].mean()}
        if args.reset_masked:
            out = env.step_tail_masked(out)
            n = out["n_done"]
            nan = torch.full((), float("nan"), device=n.device)
            core.update(ep_len=torch.where(n > 0, out["ep_len_sum"] / n.clamp(min=1), nan),
                        ep_ret=torch.where(n > 0, out["ep_ret_norm_sum"] / n.clamp(min=1), nan),
                        term_sum=out["term"].float().sum(), done_sum=n)  # fmt: skip
        return core

    def __call__(self):
        import torch

        with torch.no_grad():
            core = self.step()
            self.buffer.advance_host()
        return core


def update_graphs(algo, buffer, args):
    """train.py --update_graphs (BlinkMimic's --cuda_graph): the world-model update and the policy update (with the policy
    average's step) as CUDA graphs, captured once the replay buffer is full (--graph_early: at once) and, for the policy
    update, once the policy average has its first value. An algorithm without a capturable policy update keeps its eager
    one (None)."""
    from ndm_gym.algos.blinkmimic import Graphed

    full = (lambda: True) if args.graph_early else (lambda: buffer.size == buffer.T)  # noqa: E731
    wm = Graphed(lambda: algo.wm_update_graphable(buffer), full, "the world-model update")
    if getattr(algo, "policy_update_graphable", None) is None:
        return wm, None
    pi = Graphed(lambda: algo.policy_update_graphable(buffer), lambda: full() and (algo.pi_avg is None or int(algo.pi_avg.n_averaged) > 0),
                 "the policy update")  # fmt: skip
    return wm, pi


class LightCheckpoints:
    """train.py --ckpt_light (BlinkMimic): periodic checkpoints hold only the deployed policy (the policy average with --pi_ema)
    and the run's counters, written by one background thread (at most two queued); a failed write raises in the training
    loop or at the end. last.pt and the full checkpoint are written at the end only; with --last_every_min, last.pt also
    every that many minutes of training (``last_due``), in the training loop."""

    def __init__(self):
        self.pool, self.pending = ThreadPoolExecutor(max_workers=1), []

    @staticmethod
    def _save(obj, path):
        import torch

        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _wait(self, drain: bool = False):
        while self.pending and (drain or self.pending[0].done() or len(self.pending) > 2):
            self.pending.pop(0).result()

    def save(self, algo, iteration: int, wall_time: float, sim_steps: int, args):
        sd = algo.pi_avg.module.state_dict() if algo.pi_avg is not None else algo.pi.state_dict()
        small = {"pi": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}, "iteration": iteration, "wall_time": wall_time,
                 "sim_steps": sim_steps, "args": vars(args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        self._wait()
        self.pending.append(self.pool.submit(self._save, small, os.path.join(args.log_dir, f"ckpt_{iteration:06d}.pt")))

    def close(self):
        self._wait(drain=True)
        self.pool.shutdown()


def last_due(wall_time: float, last_time: float, every_min: float | None) -> bool:
    """--last_every_min with --ckpt_light (train.py, train_muscles): whether the periodic checkpoint at
    ``wall_time`` seconds of training also writes the full last.pt, which --resume continues from: once ``every_min``
    minutes of training have passed since the last one, written at ``last_time`` (before the first: the run's start or
    resume point). None: last.pt at the end only."""
    return every_min is not None and wall_time - last_time >= 60.0 * every_min


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.blinkmimic import PolicyLR, check_resume, reset_kwargs
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.eprobe import EvalProbe
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1, read_clip_list, select
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, TrackingSim, make_sim
    from ndm_gym.state import unflatten_state
    from ndm_gym.tasks.tracking import MDP, OBS_DIM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    RunningNorm.no_sync = args.no_sync  # --no_sync (BlinkMimic): the normalizers' freeze test on the device
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    if args.elementwise_3x3:
        elementwise_3x3_products()
    if args.combo_kernels:
        combo_kernels()
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

    kin = g1.kinematics(dev)
    if args.motions == "interhuman":
        from ndm_gym.motion.sources import interhuman

        clips = interhuman.load(args.motion_dir)
    else:
        clips = lafan1.load_dir(args.motion_dir)
    if args.clips:
        clips = select(clips, read_clip_list(args.clips))
        with open(os.path.join(args.log_dir, "clips.txt"), "w") as f:  # the clips trained on, whatever the list file says later
            f.write("".join(f"{c.name}\n" for c in clips))
    motion = MotionLib(clips, kin, dev)
    if args.goal_table:
        motion.cache_goals(args.goal_frames)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), training_reward_cfg(args.reward_std_scale))
    K, Hh = args.goal_frames, args.hist
    algo = ALGOS[args.algo](args, kin, motion, mdp, dev)
    pi_lr = PolicyLR(args)  # --pi_lr_mult
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        assert ck["args"]["algo"] == args.algo, (ck["args"]["algo"], args.algo)
        # the optimizers' settings are loaded with their state: another Adam implementation would be replaced by the saved one
        assert ck["args"].get("fused_adam", False) == args.fused_adam, "--fused_adam must match the checkpoint's run"
        assert ck["args"].get("critic_cudagraphs", False) == args.critic_cudagraphs, "--critic_cudagraphs must match the checkpoint's run"
        check_resume(ck["args"], args)
        algo.load_state_dict(ck)
        pi_lr.load_state_dict(ck)
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        resume_ck = ck
    profile = bool(int(os.environ.get("SM_PROFILE", "0")))

    sim_cfg = make_sim(dev, args, paired=args.paired)
    with launch_simulation(sim_cfg, args):
        env = (PairedTrackingSim if args.paired else TrackingSim)(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, **reset_kwargs(args))
        if args.compile_eager:
            env.reward_fn = torch.compile(mdp.reward)
        assert torch.allclose(env.default_q, mdp.default_q), (env.default_q, mdp.default_q)
        probe = EvalProbe(args, motion, env, args.log_dir) if args.eprobe_share > 0 else None  # --eprobe_share: the worker process
        if probe is not None:
            atexit.register(probe.close)
            if args.reset_masked:  # the captured collection step resets from the focus's fixed tensors
                env.use_static_focus()
            if args.resume:
                probe.load_state_dict(resume_ck)
        if algo.residual:
            assert torch.equal(env.alpha, algo.alpha), (env.alpha, algo.alpha)
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, priorities=args.wm_sample == "prioritized")
        if args.update_graphs:  # BlinkMimic's --cuda_graph: sampling bounds on the device (--graph_early: the fill level too)
            buffer.use_device_ptr(device_size=args.graph_early)
        obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
        graphed_step = GraphedCollector(env, algo, buffer, mdp, motion, obs_hist, args) if args.collect_graph else None

        @torch.no_grad()
        def collect(n_steps: int, stats: dict):
            nonlocal obs_hist, sim_steps
            if args.collect_graph:
                for _ in range(n_steps):
                    core = graphed_step()
                    sim_steps += args.num_envs
                    stats["r"].append(core["r"].clone())
                    stats["rate"].append(core["rate"].clone())
                    if args.reset_masked:  # every step appends; a step without an ended episode carries NaN means
                        for k_s, k_c in (("ep_len", "ep_len"), ("ep_ret", "ep_ret"), ("term", "term_sum"), ("done", "done_sum")):
                            stats[k_s].append(core[k_c].clone())
                        continue
                    out = env.step_tail({"done": core["done"], "term": core["term"], "f": core["f"]})
                    if "ep_len" in out:
                        stats["ep_len"].append(out["ep_len"].mean())
                        stats["ep_ret"].append(out["ep_ret_norm"].mean())
                        stats["term"].append(core["term"].float().sum())
                        stats["done"].append(core["done"].float().sum())
                return
            for _ in range(n_steps):
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
                fresh = env.ep_step == 0
                obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
                g = motion.goal(env.f, K)
                algo.update_pi_norms(o, g)
                a = algo.setpoint(algo.pi(obs_hist, g), env.f)
                a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
                out = env.step(a, S)  # S: the state just read, nothing written since
                buffer.add(out)
                algo.update_wm_norms(unflatten_state(out["s"]), unflatten_state(out["s1"]), out["a"], out["xi"])
                sim_steps += args.num_envs
                stats["r"].append(out["r"].mean() / 0.1)
                stats["rate"].append(out["terms"]["action_rate"].mean())
                if "ep_len" in out:
                    stats["ep_len"].append(out["ep_len"].mean())
                    stats["ep_ret"].append(out["ep_ret_norm"].mean())
                    stats["term"].append(out["term"].float().sum())
                    stats["done"].append(out["done"].float().sum())

        stats = {"r": [], "rate": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        # the replay buffer is not checkpointed: a resumed run refills it with --warmup_steps of collection, as a new run starts
        collect(args.warmup_steps, stats)
        if args.update_graphs:
            wm_update_g, policy_update_g = update_graphs(algo, buffer, args)
        log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
        if start_it == 0:
            log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,"
                        "ep_ret_norm,term_frac,it_per_s,lr,wm_body8_m,wm_q8_rad,sim_action_rate,fail_starts"
                        + (",nonfinite" if args.no_sync else "") + "\n")  # fmt: skip
        light = LightCheckpoints() if args.ckpt_light else None

        t_start = time.time() - wall_offset
        t_log = time.time()
        hours_left = sorted(h for h in args.ckpt_hours if h > wall_offset / 3600)
        last_time = wall_offset  # --last_every_min: the training time (s) of the last full last.pt (or of the start or resume point)
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "fs": []}

        def save_last(iteration: int) -> dict:
            """The full checkpoint, which --resume continues from, as last.pt (written whole, then renamed); returns it."""
            ck = {**algo.state_dict(), **pi_lr.state_dict(), **(probe.state_dict() if probe is not None else {}), "iteration": iteration,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            return ck

        for it in range(start_it, args.iterations):
            if args.lr_decay_hours is not None:
                lr = lr_at((time.time() - t_start) / 3600, args.lr, args.lr_decay_hours)
            else:
                lr = lr_at(it, args.lr, args.lr_decay_iters)
            for o in algo.optimizers:
                set_lr(o, lr)
            if args.wm_lr is not None:
                for gparam in algo.opt_wm.param_groups:
                    if torch.is_tensor(gparam["lr"]):
                        gparam["lr"].fill_(lr * args.wm_lr / args.lr)
                    else:
                        gparam["lr"] = lr * args.wm_lr / args.lr
            if pi_lr.on:
                set_lr(algo.opt_pi, lr * pi_lr.at(it))
            step_acc += 1.0 / args.utd
            n = int(step_acc)
            step_acc -= n
            prof = profile and it % 20 == 0 and it > args.wm_only_iters + 5
            if probe is not None:  # --eprobe_share: a sweep's failures applied when due, a sweep started when due
                probe.tick(it, algo.pi_avg.module)
            if prof:
                torch.cuda.synchronize()
                tp0 = time.time()
            collect(n, stats)
            if prof:
                torch.cuda.synchronize()
                tp1 = time.time()
            log_now = (it + 1) % args.log_every == 0
            for b in range(args.wm_batches):
                if args.update_graphs:  # no probe of the k-step body and joint errors (the log's columns are NaN)
                    res = wm_update_g()
                else:
                    res = algo.wm_update(buffer, probe=log_now and b == args.wm_batches - 1)
            wm_l, e8 = res[:2]
            if prof:
                torch.cuda.synchronize()
                tp2 = time.time()
            agg["wm"].append(wm_l), agg["wm8"].append(e8)
            if it >= args.wm_only_iters:
                for _ in range(args.pi_updates):  # --pi_updates: more policy updates per iteration, each on its own batch
                    if args.update_graphs and policy_update_g is not None:
                        n_avg0 = int(algo.pi_avg.n_averaged) if algo.pi_avg is not None else 0
                        pl, vl, ir, gn, surv = policy_update_g()  # with the policy average's step
                        if algo.pi_avg is not None and int(algo.pi_avg.n_averaged) == n_avg0:
                            algo.pi_avg.n_averaged += 1  # the captured average's step does not advance the host-side counter
                    else:
                        pl, vl, ir, gn, surv = algo.policy_update(buffer)
                        algo.update_pi_avg()  # --pi_ema
                if prof:
                    torch.cuda.synchronize()
                    tp3 = time.time()
                    print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy {1e3*(tp3-tp2):.0f} ms", flush=True)
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                agg["fs"].append(algo.fail_starts)
            if log_now:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                mean_nn = lambda x: mean([v for v in (float(v) for v in x) if v == v])  # noqa: E731  (--reset_masked: NaN = no end)
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                probes = (float(res[2]), float(res[3])) if len(res) > 2 else (float("nan"), float("nan"))
                row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean_nn(stats["ep_len"]), mean_nn(stats["ep_ret"]), term_frac, ips,
                       algo.log_lr(lr), *probes, mean(stats["rate"]), mean(agg["fs"])]  # fmt: skip
                if args.no_sync:
                    row.append(int(algo.nonfinite))
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
                msg = pi_lr.observe(it + 1, row[12])  # --lr_decay_ret's trigger
                if msg:
                    print(msg, flush=True)
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            trained_h = (time.time() - t_start) / 3600
            out_of_time = args.max_hours is not None and trained_h >= args.max_hours
            marks = [h for h in hours_left if trained_h >= h]
            final_ck = it + 1 == args.iterations or out_of_time
            if light is not None and not final_ck and ((it + 1) % args.ckpt_every == 0 or marks):
                wall = time.time() - t_start
                light.save(algo, it + 1, wall, sim_steps, args)
                if last_due(wall, last_time, args.last_every_min):  # --last_every_min: the full last.pt too
                    save_last(it + 1)
                    last_time = wall
                    print(f"last.pt at it {it + 1} ({wall / 60:.1f} min of training, --last_every_min)", flush=True)
            elif (it + 1) % args.ckpt_every == 0 or final_ck or marks:
                ck = save_last(it + 1)
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)
            for h in marks:
                hours_left.remove(h)
                print(f"checkpoint at {h:g} h of training: it {it + 1}", flush=True)
            if out_of_time:
                print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
                break
        if light is not None:
            light.close()
        if probe is not None:
            probe.close()


if __name__ == "__main__":
    main()
