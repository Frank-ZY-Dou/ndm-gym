"""Training loop (SuperMimic Algorithm A.1): collect -> fit world model -> BPTT policy update.

``--algo`` selects the algorithm (:mod:`ndm_gym.algos`); the simulator, observations, networks, replay buffer and
schedule are shared. Run: ``python -m ndm_gym.train --algo supertrack|supermimic --log_dir logs/<run>``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

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
    parser.add_argument("--wm_bptt_fp32", action="store_true", help="both: the world model without bf16 autocast in the policy's BPTT")
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
    parser.add_argument("--wm_backbone", type=str, default="mlp", choices=["mlp", "rigidformer"], help="world-model network")
    parser.add_argument("--rf_dim", type=int, default=128, help="rigidformer: token width")
    parser.add_argument("--rf_layers", type=int, default=3, help="rigidformer: Transformer blocks")
    parser.add_argument("--rf_heads", type=int, default=4, help="rigidformer: attention heads")
    parser.add_argument("--rf_registers", type=int, default=2, help="rigidformer: extra learned register tokens")
    parser.add_argument("--rf_no_tree_bias", action="store_true", help="rigidformer: no kinematic-tree attention bias")
    parser.add_argument("--rf_no_rope", action="store_true", help="rigidformer: no rotary body positions")
    parser.add_argument("--rf_rope_detach", action="store_true", help="rigidformer: no gradient through the rotary positions in the policy's BPTT")
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
    parser.add_argument("--lr_decay_iters", type=int, default=250_000)
    parser.add_argument("--lr_decay_hours", type=float, default=None,
                        help="decay the learning rate over this much training wall-clock time instead of --lr_decay_iters")  # fmt: skip
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--max_hours", type=float, default=None, help="stop with a checkpoint after this much training wall-clock time")
    parser.add_argument("--ckpt_hours", type=float, nargs="+", default=[],
                        help="also checkpoint when the training wall-clock time first reaches each of these hours")  # fmt: skip
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
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


def parse_args():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import lr_at
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1, read_clip_list, select
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.state import normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP, OBS_DIM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

    kin = g1.kinematics(dev)
    clips = lafan1.load_dir(args.motion_dir)
    if args.clips:
        clips = select(clips, read_clip_list(args.clips))
        with open(os.path.join(args.log_dir, "clips.txt"), "w") as f:  # the clips trained on, whatever the list file says later
            f.write("".join(f"{c.name}\n" for c in clips))
    motion = MotionLib(clips, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), training_reward_cfg(args.reward_std_scale))
    K, Hh = args.goal_frames, args.hist
    algo = ALGOS[args.algo](args, kin, motion, mdp, dev)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        assert ck["args"]["algo"] == args.algo, (ck["args"]["algo"], args.algo)
        algo.load_state_dict(ck)
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    profile = bool(int(os.environ.get("SM_PROFILE", "0")))

    sim_cfg = make_sim(dev, args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma)
        assert torch.allclose(env.default_q, mdp.default_q), (env.default_q, mdp.default_q)
        if algo.residual:
            assert torch.equal(env.alpha, algo.alpha), (env.alpha, algo.alpha)
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
        obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

        @torch.no_grad()
        def collect(n_steps: int, stats: dict):
            nonlocal obs_hist, sim_steps
            for _ in range(n_steps):
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
                fresh = env.ep_step == 0
                obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
                g = motion.goal(env.f, K)
                algo.pi.obs_norm.update(o)
                algo.pi.goal_norm.update(g)
                a = algo.setpoint(algo.pi(obs_hist, g), env.f)
                a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
                out = env.step(a)
                buffer.add(out)
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                algo.wm.in_norm.update(algo.wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                algo.wm.out_norm.update(algo.wm.targets(S0, S1))
                sim_steps += args.num_envs
                stats["r"].append(out["r"].mean() / 0.1)
                stats["rate"].append(out["terms"]["action_rate"].mean())
                if "ep_len" in out:
                    stats["ep_len"].append(out["ep_len"].mean())
                    stats["ep_ret"].append(out["ep_ret_norm"].mean())
                    stats["term"].append(out["term"].float().sum())
                    stats["done"].append(out["done"].float().sum())

        stats = {"r": [], "rate": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        if start_it == 0:
            collect(args.warmup_steps, stats)
        log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
        if start_it == 0:
            log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,"
                        "ep_ret_norm,term_frac,it_per_s,lr,wm_body8_m,wm_q8_rad,sim_action_rate,fail_starts\n")  # fmt: skip
        t_start = time.time() - wall_offset
        t_log = time.time()
        hours_left = sorted(h for h in args.ckpt_hours if h > wall_offset / 3600)
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "fs": []}
        for it in range(start_it, args.iterations):
            if args.lr_decay_hours is not None:
                lr = lr_at((time.time() - t_start) / 3600, args.lr, args.lr_decay_hours)
            else:
                lr = lr_at(it, args.lr, args.lr_decay_iters)
            for o in algo.optimizers:
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            if args.wm_lr is not None:
                for gparam in algo.opt_wm.param_groups:
                    gparam["lr"] = lr * args.wm_lr / args.lr
            step_acc += 1.0 / args.utd
            n = int(step_acc)
            step_acc -= n
            prof = profile and it % 20 == 0 and it > args.wm_only_iters + 5
            if prof:
                torch.cuda.synchronize()
                tp0 = time.time()
            collect(n, stats)
            if prof:
                torch.cuda.synchronize()
                tp1 = time.time()
            log_now = (it + 1) % args.log_every == 0
            for b in range(args.wm_batches):
                res = algo.wm_update(buffer, probe=log_now and b == args.wm_batches - 1)
            wm_l, e8 = res[:2]
            if prof:
                torch.cuda.synchronize()
                tp2 = time.time()
            agg["wm"].append(wm_l), agg["wm8"].append(e8)
            if it >= args.wm_only_iters:
                pl, vl, ir, gn, surv = algo.policy_update(buffer)
                if prof:
                    torch.cuda.synchronize()
                    tp3 = time.time()
                    print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy {1e3*(tp3-tp2):.0f} ms", flush=True)
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                agg["fs"].append(algo.fail_starts)
            if log_now:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                       float(res[2]), float(res[3]), mean(stats["rate"]), mean(agg["fs"])]  # fmt: skip
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            trained_h = (time.time() - t_start) / 3600
            out_of_time = args.max_hours is not None and trained_h >= args.max_hours
            marks = [h for h in hours_left if trained_h >= h]
            if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or out_of_time or marks:
                ck = {**algo.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start, "sim_steps": sim_steps,
                      "args": vars(args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
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


if __name__ == "__main__":
    main()
