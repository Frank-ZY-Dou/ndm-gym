"""Training loop for manipulation demonstration tracking on robomimic demonstrations of one scene (docs/tasks/robomimic.md): collect in
RobomimicSim -> fit the world model on replay windows -> policy update through imagined rollouts
(ndm_gym.algos.robomimic_algos). The loop, the schedule and the systems options are hand-object tracking's (ndm_gym.train_hand,
ndm_gym.algos.hand_fast) with the defaults of ROBOMIMIC_DEFAULTS; ndm_gym.train's arguments and hand training's systems
options that this training does not implement are refused unless left at their defaults (train_hand.UNSUPPORTED,
UNSUPPORTED_SYSTEMS).

An iteration is one control step of every world, ``--wm_batches`` world-model steps and, after ``--wm_only_iters``, one policy
(and critic) update. Writes ``train_log.csv``, ``args.json`` and ``last.pt`` to ``--log_dir`` (``--ckpt_keep``: also
``ckpt_<iteration>.pt`` at every ``--ckpt_every``). ``--resume`` continues from a run's ``last.pt``; the replay is not saved.
``ROBOMIMIC_PROFILE=1`` prints each part's time with every log row.

Run: ``python -m ndm_gym.train_robomimic --task lift --demo 0 --log_dir logs/<run>``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ndm_gym.train import add_train_args

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival", "sim_r", "object_mm",
               "success", "ep_len", "term_frac", "it_per_s", "lr")  # fmt: skip
# ndm_gym.train's defaults that this training replaces (hand-object tracking's)
ROBOMIMIC_DEFAULTS = {"algo": "supermimic", "num_envs": 2048, "iterations": 20_000, "lr_decay_iters": 20_000, "buffer_steps": 256, "warmup_steps": 32,
                      "wm_only_iters": 300, "horizon": 16, "goal_frames": 5, "wm_hidden": [1024, 1024, 1024], "wm_batches": 4, "amp": "none"}  # fmt: skip
# the options a resumed run must share with its checkpoint: the algorithm and SuperTrack's weights, the optimizers' layouts
# and states, the policy average
RESUME_MATCH = ("algo", "st_task_weights", "fused_adam", "update_graphs", "pi_opt", "wm_opt", "pi_ema")

# hand training's systems options (algos.hand_fast) that this training does not implement; each is refused unless at its
# default: train_hand's own (the kernels' options, its phase times), HandSuperMimic's, and hand_fast.Collector's
# --compile_collect (RobomimicCollector compiles the forward kinematics only: --compile_eager)
UNSUPPORTED_SYSTEMS = ("compile_updates", "compile_rollout", "compile_collect", "wm_palm_fk", "sm_reuse_input", "elementwise_3x3",
                       "combo_kernels", "mix_order_reductions", "autotune_gemm", "phase_times")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    from ndm_gym.algos.hand_fast import add_hand_fast_args, check_hand_fast_args
    from ndm_gym.train_hand import UNSUPPORTED

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_hand_fast_args(parser)  # the systems options
    parser.add_argument("--task", type=str, required=True, choices=["lift", "square", "transport", "aloha_transfer_cube", "aloha_insertion"],
                        help="robomimic's tasks, or ACT's simulated ALOHA tasks (docs/tasks/aloha.md)")
    parser.add_argument("--demo", type=int, nargs="+", default=[0],
                        help="the demonstrations (demo_<i>) of the task's file; several: one policy on all of them, whose scenes must be one")  # fmt: skip
    parser.add_argument("--data_dir", type=str, default=None,
                        help="tools/download_robomimic.py's output (default data/robomimic); the ALOHA tasks: tools/make_aloha_demos.py's "
                        "(default data/aloha)")  # fmt: skip
    parser.add_argument("--episode", type=int, default=200, help="an episode's longest length in control steps (20 Hz)")
    parser.add_argument("--solver_iterations", type=int, default=0,
                        help="the constraint solver's iterations, in place of the scene's (robosuite's: MuJoCo's 100); a GPU step lasts as "
                        "long as its slowest world, so a cap shortens the collection (changes the physics; 0: the scene's)")  # fmt: skip
    parser.add_argument("--solver_ls_iterations", type=int, default=0, help="the solver's line-search iterations with --solver_iterations (0: the scene's)")
    parser.add_argument("--st_task_weights", action="store_true",
                        help="supertrack: no calibration; each tracked quantity weighs, per unit of error, the task's reward weight over its "
                        "kernel's width")  # fmt: skip
    parser.add_argument("--ckpt_keep", action="store_true", help="also keep ckpt_<iteration>.pt at every --ckpt_every (the policy and its arguments)")
    for action in parser._actions:
        if action.dest == "algo":
            action.required = False
    parser.set_defaults(**ROBOMIMIC_DEFAULTS)
    from ndm_gym import configs

    configs.add_config_arg(parser)
    try:  # --config: a named configuration's options first
        argv = configs.expand(sys.argv[1:] if argv is None else list(argv), "train_robomimic")
    except ValueError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    if args.algo not in ("supertrack", "supermimic"):
        parser.error("robomimic training implements --algo supertrack and --algo supermimic")
    st_only = {"st_task_weights": False, **{k: parser.get_default(k) for k in ("st_reg", "st_imag_noise", "st_fail_w")}}
    sm_only = {k: parser.get_default(k) for k in ("bptt_decay", "gamma", "lam", "critic_epochs", "critic_minibatches")}  # its critic and return
    if args.algo == "supermimic":
        for name, default in st_only.items():
            if getattr(args, name) != default:
                parser.error(f"--{name} is SuperTrack's option")
    else:
        for name, default in sm_only.items():
            if getattr(args, name) != default:
                parser.error(f"--{name} is SuperMimic's option here")
    for name in UNSUPPORTED:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of LAFAN1 training (ndm_gym.train) that robomimic training does not implement")
    for name in UNSUPPORTED_SYSTEMS:
        if getattr(args, name) != parser.get_default(name):
            parser.error(f"--{name} is an option of hand training (ndm_gym.train_hand) that robomimic training does not implement")
    if args.episode < 2:
        parser.error("--episode >= 2")
    if args.data_dir is None:
        args.data_dir = os.path.join(REPO, "data", "aloha" if args.task.startswith("aloha_") else "robomimic")
    if args.solver_iterations < 0 or args.solver_ls_iterations < 0 or (args.solver_ls_iterations and not args.solver_iterations):
        parser.error("--solver_iterations >= 0, --solver_ls_iterations >= 0 and only with --solver_iterations")
    try:
        check_hand_fast_args(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def build(task_name: str, demo, data_dir: str, device: str, reward_std_scale: float = 1.0, solver_iterations: int = 0,
          solver_ls_iterations: int = 0):  # fmt: skip
    """The scene with its servos, its layout for the task and the task of demonstrations (``demo``: an index or a list of them,
    whose recorded scenes must be the same: robosuite draws Lift's cube and Transport's hammer per demonstration), and
    MuJoCo-Warp's per-world contact and constraint capacities for them. ``solver_iterations``, ``solver_ls_iterations``: the constraint solver's caps (0: the
    scene's)."""
    from ndm_gym.motion.sources.robomimic import demo_path, load_demo, split_states
    from ndm_gym.robots.robomimic_scenes import RobomimicScene, contact_capacity, load_scene
    from ndm_gym.tasks.robomimic_tracking import HOLD, RewardCfg, RobomimicTask, SceneKinematics, build_reference
    from ndm_gym.worldmodels.mjcf import MjcfLayout

    ids = [demo] if isinstance(demo, int) else demo
    if task_name.startswith("aloha_"):  # ACT's scene, one for every demonstration, with the demonstrations' own targets
        from ndm_gym.motion.sources import aloha
        from ndm_gym.robots import aloha_scenes

        demos = [aloha.load_demo(aloha.demo_path(data_dir, task_name), i) for i in ids]
        model = aloha_scenes.load_scene(os.path.abspath(os.path.join(data_dir, "act_assets")), task_name)
        scene, states = aloha_scenes.aloha_scene(model, task_name), [(x["qpos"], x["qvel"]) for x in demos]
        targets, hold = [x["target"] for x in demos], aloha_scenes.HOLD
    else:
        demos = [load_demo(demo_path(data_dir, task_name), i) for i in ids]
        if any(x["xml"] != demos[0]["xml"] for x in demos[1:]):
            raise ValueError(f"{task_name}: the demonstrations {demo} were recorded in different scenes (one policy on several needs one scene)")
        model = load_scene(demos[0]["xml"], os.path.abspath(os.path.join(data_dir, "robosuite_assets")))
        scene, states = RobomimicScene.from_model(model, task_name), [split_states(x["states"], model.nq, model.nv) for x in demos]
        targets, hold = None, HOLD
    if solver_iterations:
        model.opt.iterations = solver_iterations
        model.opt.ls_iterations = solver_ls_iterations or model.opt.ls_iterations
    kin = SceneKinematics(MjcfLayout.from_model(model), scene.keypoint_bodies).to(device)
    ref = build_reference([q for q, _ in states], [v for _, v in states], scene, kin, device, target=targets, hold=hold)
    task = RobomimicTask(scene, kin, ref, RewardCfg().scaled(reward_std_scale))
    caps = [contact_capacity(model, q, v) for q, v in states]
    return model, task, (max(c[0] for c in caps), max(c[1] for c in caps))


def main():
    args = parse_args()

    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import hand_fast
    from ndm_gym.algos.blinkmimic import PolicyLR
    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.algos.robomimic_algos import ACC_KEYS, ROBOMIMIC_ALGOS, WINDOW_KEYS, RobomimicCollector, collect_step, replay_fields
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.core.normalizer import RunningNorm
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    nets.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    RunningNorm.no_sync = args.no_sync  # --no_sync: the normalizers' freeze test on the device, their statistics written in place
    dev = "cuda:0"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path, header = os.path.join(args.log_dir, "train_log.csv"), ",".join(LOG_COLUMNS)
    resume_ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
    if resume_ck is not None:
        for k in RESUME_MATCH:
            if resume_ck["args"][k] != getattr(args, k):
                raise ValueError(f"--resume: --{k} must match the checkpoint's run ({resume_ck['args'][k]!r})")
    if os.path.exists(log_path):
        if resume_ck is None:
            raise ValueError(f"{log_path} exists: continue that run with --resume (its last.pt) or use another --log_dir")
        with open(log_path) as f:
            if f.readline().strip() != header:
                raise ValueError(f"{log_path} has other columns than {header!r}; continue in another --log_dir")
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, task, (nconmax, njmax) = build(args.task, args.demo, args.data_dir, dev, args.reward_std_scale, args.solver_iterations,
                                          args.solver_ls_iterations)  # fmt: skip
    sim = RobomimicSim(model, task, args.num_envs, dev, nconmax=nconmax, njmax=njmax, episode=args.episode, stagger=True)
    algo = ROBOMIMIC_ALGOS[args.algo](args, model, task, dev, substeps=sim.substeps)
    average = hand_fast.PolicyAverage(algo.pi, args.pi_ema) if args.pi_ema > 0 else None  # --pi_ema
    pi_lr = PolicyLR(args)  # --pi_lr_mult
    start_it, wall_offset = 0, 0.0
    if resume_ck is not None:
        algo.load_state_dict(resume_ck if average is None else average.resume(resume_ck))
        pi_lr.load_state_dict(resume_ck)
        start_it, wall_offset = resume_ck["iteration"], resume_ck["wall_time"]
        del resume_ck
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    if args.collect_graph or args.update_graphs:  # the graphs read the replay's write head and fill level on the device
        buffer.use_device_ptr(device_size=True)
    N = args.num_envs
    if args.no_sync:  # the step without a host read; --collect_graph: as one CUDA graph
        collect = RobomimicCollector(sim, task, algo, buffer, args)
        acc = collect.acc
    else:
        a_prev, ep_len = torch.zeros(N, task.na, device=dev), torch.zeros(N, device=dev)
        acc = {k: torch.zeros((), device=dev) for k in ACC_KEYS}

        def host_step(u):
            out = sim.step(u, reset=not args.reset_masked)
            if args.reset_masked:  # the ended worlds reset by masked writes, the draws made for all worlds
                sim.reset_masked(out[2])
            return out

        @torch.no_grad()
        def collect():
            collect_step(sim, task, algo, buffer, args, a_prev, ep_len, acc, host_step, buffer.add, task.fk)

    # the iteration's updates; --update_graphs: each as a CUDA graph; --pi_ema: the average's step with the policy's
    wm_update, policy_update = hand_fast.update_functions(algo, buffer, args, average)
    timers = None
    if os.environ.get("ROBOMIMIC_PROFILE", "0") == "1":  # each part's time per call, the device synchronized around it, with every log row
        timers = collect, wm_update, policy_update = [hand_fast.Timed(f) for f in (collect, wm_update, policy_update)]

    def save(iteration: int, keep: bool = False):
        ck = {**algo.state_dict(), **pi_lr.state_dict(), "iteration": iteration, "wall_time": time.time() - t_start, "sim_steps": iteration * N,
              "args": vars(args)}  # fmt: skip
        if average is not None:  # --pi_ema: 'pi' is the average (the policy evaluated), 'pi_live' the trained policy
            ck = average.checkpoint(ck)
        if args.no_sync:  # the checkpoint's tensors to the host in one transfer per dtype, not one each
            ck = hand_fast.to_host(ck)
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        if keep:  # what evaluate_robomimic reads of a checkpoint
            torch.save({k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "args")}, os.path.join(args.log_dir, f"ckpt_{iteration:06d}.pt"))

    for _ in range(args.warmup_steps):
        collect()
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:
        log_f.write(header + "\n")
    t_log = time.time()
    t_start = t_log - wall_offset
    agg = {k: [] for k in ("wm_loss", "pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival")}
    it = start_it
    for it in range(start_it + 1, args.iterations + 1):
        lr = lr_at(it, args.lr, args.lr_decay_iters)
        for opt in algo.optimizers:
            set_lr(opt, lr)
        if pi_lr.on:  # --pi_lr_mult
            set_lr(algo.opt_pi, lr * pi_lr.at(it - 1))
        collect()
        agg["wm_loss"].append(wm_update())
        if it > args.wm_only_iters:
            for k, x in zip(("pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival"), policy_update(), strict=True):
                agg[k].append(x)
        if it % args.log_every == 0:
            nan = torch.full((), float("nan"), device=dev)
            vals = torch.stack([*(torch.stack(agg[k]).mean() if agg[k] else nan for k in agg), acc["done"], acc["r"] / acc["steps"],
                                acc["object_err"] / acc["steps"], acc["success"], acc["at_end"], acc["ep_len"], acc["term"]]).tolist()  # fmt: skip
            # (the row's values above in one read of the device)
            means, (n_done, sim_r, object_err, n_success, n_end, ep_sum, n_term) = vals[: len(agg)], vals[len(agg) :]
            done, now = max(n_done, 1.0), time.time()
            # success: of the episodes that reached their demonstration's last frame in the row, the share at robosuite's success there
            row = [it, (now - t_start) / 3600, it * N, *means, sim_r, 1000 * object_err, n_success / max(n_end, 1.0), ep_sum / done,
                   n_term / done, args.log_every / (now - t_log), lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            log_f.flush()
            print(" ".join(f"{k}={x:.4g}" for k, x in zip(LOG_COLUMNS, row, strict=True)), flush=True)
            if timers is not None:
                print("[prof] collect {:.1f} ms | model {:.1f} ms | policy {:.1f} ms".format(*(t.take() for t in timers)), flush=True)
            msg = pi_lr.observe(it, row[LOG_COLUMNS.index("sim_r")])  # --lr_decay_ret's trigger
            if msg:
                print(msg, flush=True)
            t_log = now
            for xs in agg.values():
                xs.clear()
            for x in acc.values():
                x.zero_()
        if it % args.ckpt_every == 0:
            save(it, keep=args.ckpt_keep)
        if args.max_hours is not None and time.time() - t_start > args.max_hours * 3600:
            break
    save(it)


if __name__ == "__main__":
    main()
