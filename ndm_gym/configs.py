"""Named run configurations: each task's best settings for SuperTrack and SuperMimic as this repository measured them (the
README's tables), so that ``ndm-train --config lafan1-supertrack-fast`` stands for the option list. A configuration expands
into its options before the command line's own, which take precedence where an option takes a value (a switch it sets
cannot be turned off). Each trainer reads only its own configurations; an unknown name is an error, and ``--config list``
prints a trainer's names with what each measured."""

import shlex
import sys

# option groups, as the README's commands name them
RECIPE = ("--pi_ema 0.995 --lr 4e-4 --wm_batches 5 --wm_mirror 0.5 --pi_opt muon --wm_opt muon --wm_only_iters 0 --pi_batch 2048 "
          "--pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --lr_decay_by 1400 --wm_hidden 1024 1024 1024")  # BlinkMimic's  # fmt: skip
RF_RECIPE = RECIPE.replace(" --wm_hidden 1024 1024 1024", "")  # the recipe without the MLP's size
RESETS = "--fail_reset 0.3 --bin_fix --eprobe_share 0.15 --eprobe_graph --eprobe_async 20 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7"
SYSTEMS = "--fused_adam --elementwise_3x3 --combo_kernels --compile_eager --no_sync --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light"
# speed options of the 3 h runs: SPEED for both backbones (SuperMimic's runs also --critic_cudagraphs), RF_SPEED the
# RigidFormer's own
SPEED = "--wm_cudagraphs --pi_cudagraphs --fused_adam --elementwise_3x3 --combo_kernels --compile_eager"
RF_SPEED = "--rf_fused_train --rf_bptt_keep_blocks --rf_fit_keep_ops"
RF_SMALL = "--rf_merge joints --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 2"
TIMED = "--iterations 6000 --ckpt_every 100"  # the timed runs' length and checkpoints, each evaluated (ndm-eval)
LEAN = "--eprobe_lean --imag_hoist --wm_presample --muon_batched --gc_freeze"  # leaner probe sweeps and the learner's set (one GPU)
AWL4 = "--layout AWL --eprobe_gpu 3 --eprobe_every 250 --eprobe_async 4 --eprobe_lean --gc_freeze"  # four GPUs: actor | WM | policy, probes
ST_FAST = f"--algo supertrack --st_fail_w 1 --st_balance_w 2 {RECIPE} --st_quad_w 0.5 --wm_lr 2e-3 --goal_table --pi_encode_once {RESETS} {SYSTEMS} {TIMED}"
SM_FAST = (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.15 {RECIPE} --pi_ema 0.99 --pi_batch 1024 --pi_lr_mult 2 "
           f"--pi_updates 1 --wm_lr 1.5e-3 --goal_table --contact_reward 0.15 --ref_contact_fix --sm_residual --wm_structured {RESETS} {SYSTEMS} {TIMED}")  # fmt: skip
PAIRED = "--motions interhuman --paired"  # with --motion_dir: a two-person clip (tools/convert_interhuman.py)
MUSCLE_RECIPE = ("--pi_ema 0.995 --lr 4e-4 --pi_opt muon --wm_opt muon --wm_only_iters 0 --pi_batch 2048 --lr_decay_len 300 --lr_decay_by 1400 "
                 "--wm_hidden 1024 1024 1024")  # fmt: skip
MUSCLE_SYS = "--no_sync --fused_adam --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light"
MUSCLE_RF = ("--wm_backbone rigidformer --rf_muscle_tokens body --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 9 --rf_fused_train "
             "--rf_fused_features --wm_lr 5e-3")  # fmt: skip  # the body-token RigidFormer of the muscle 3 h runs
HAND_SYSTEMS = "--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs"
HOI_OPT = "--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300"  # HOI's optimizer options
QUAD_SYSTEMS = "--fused_adam --reset_masked --no_sync --collect_graph --update_graphs"  # the hand task's, without the forward kinematics' option
# the quadrotor's selected settings: an imagined window of 0.64 s, a world model of 3 x 256
QUAD_ST = f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights --horizon 64 {HOI_OPT} --wm_hidden 256 256 256 {QUAD_SYSTEMS}"
QUAD_SM = f"--algo supermimic --lr 4e-4 --horizon 64 --wm_hidden 256 256 256 {QUAD_SYSTEMS}"
# the Go2's selected settings (benchmarks/go2_lifelike_tracking): SuperTrack with the hand task's options, SuperMimic with HOI's
# optimizer options and a discount of 0.95
GO2_ST = f"--algo supertrack {HAND_SYSTEMS} --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights"
GO2_SM = f"--algo supermimic {HAND_SYSTEMS} --lr 4e-4 --action_noise 0.3 --bptt_decay 0.95 {HOI_OPT} --gamma 0.95"
GO2_TIMED = "--ckpt_every 500 --keep_ckpts"  # the checkpoints evaluate_quadruped --ckpts times
# manipulation demonstration tracking: the hand task's settings and systems options as the starting point
ROBOMIMIC_ST = f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights {HAND_SYSTEMS}"
ROBOMIMIC_SM = f"--algo supermimic {HAND_SYSTEMS}"
# the benchmark's runs (benchmarks/robomimic_manipulation_tracking): 1,500 iterations on the schedule of 4,000, the solver at 20
# iterations, a checkpoint kept every 500
ROBOMIMIC_RUN = "--solver_iterations 20 --iterations 1500 --lr_decay_iters 4000 --ckpt_every 500 --ckpt_keep"

# trainer -> name -> (options, what the README measured with them)
CONFIGS = {
    "train": {
        "lafan1-supertrack-accurate": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 {SPEED} --max_hours 3",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB SXM4 with the speed options (about 36,200 iterations): return "
                                       "0.857 +- 0.004, MPJPE 15.2 +- 1.0 mm (seeds 7-9)"),
        "lafan1-supermimic-accurate": (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.1 --imag_start_perturb 0.3 --sm_residual {SPEED} "
                                       "--critic_cudagraphs --max_hours 3",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB SXM4 with the speed options (about 33,800 iterations): return "
                                       "0.872 +- 0.003, MPJPE 21.2 +- 0.3 mm (seeds 7-9)"),
        "lafan1-supertrack-fast": (ST_FAST,
                                   "LAFAN1: return 0.82 after 4.19 +- 0.18 min on an A100 40GB SXM4 (seeds 0-2, one run at a time)"),
        "lafan1-supermimic-fast": (f"{SM_FAST} --pi_encode_once {LEAN}",
                                   "LAFAN1: return 0.82 after 3.53 +- 0.30 min on an A100 40GB SXM4 (seeds 0-2, one run at a time)"),
        "lafan1-supertrack-fast-4gpu": (f"{ST_FAST} {AWL4}",
                                        "LAFAN1 on four A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of four GPUs): return 0.82 after 1.48 +- 0.13 min, 5.9 GPU-minutes "
                                        "(seeds 0-2)"),
        "lafan1-supermimic-fast-4gpu": (f"{SM_FAST} --pi_encode_once {AWL4}",
                                        "LAFAN1 on four A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of four GPUs): return 0.82 after 1.37 +- 0.05 min, 5.5 GPU-minutes "
                                        "(seeds 0-2)"),
        "lafan1-supertrack-rf-accurate": (f"--algo supertrack --wm_backbone rigidformer --wm_lr 4e-4 --st_fail_w 5 --st_balance_w 2 --wm_batches 5 "
                                          f"{SPEED} {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.832 +- 0.003, MPJPE 21.0 +- 1.3 mm (seeds 7-9)"),
        "lafan1-supermimic-rf-accurate": (f"--algo supermimic --wm_backbone rigidformer --wm_lr 4e-4 --reward_std_scale 0.7 --action_noise 0.3 "
                                          f"--imag_start_perturb 0.4 --wm_batches 7 {SPEED} --critic_cudagraphs {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.804 +- 0.007, MPJPE 33.1 +- 0.5 mm (seeds 7-9)"),
        "two-g1-supertrack": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 --st_com {RECIPE} --pi_batch 4096 {SYSTEMS} {PAIRED} "
                              "--ckpt_every 1000 --iterations 9396",
                              "two G1s, InterHuman clip 1566 (add --motion_dir): return 0.8925 +- 0.0011, MPJPE 46.2 / 14.1 mm global / "
                              "root-relative (seeds 0-2 to 9,396 iterations: seed 0 on an A100 80GB, in 0.5 h, seeds 1-2 on an A100 40GB SXM4)"),
        "two-g1-supermimic": (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 {RECIPE} --pi_lr_mult 2 "
                              f"--contact_reward 0.25 --ref_contact_fix --sm_residual --wm_structured {SYSTEMS} {PAIRED} --ckpt_every 1000 "
                              "--iterations 9396",
                              "two G1s, InterHuman clip 1566 (add --motion_dir): return 0.9055 +- 0.0011, MPJPE 45.3 / 16.0 mm global / "
                              "root-relative (seeds 0-2 to two-g1-supertrack's 9,396 iterations, A100 40GB SXM4; for 0.3 h, 6,264 iterations: "
                              "0.9042 +- 0.0015, 47.5 / 16.1 mm)"),
        "two-g1-supertrack-rf": (f"--algo supertrack --st_fail_w 2 --st_balance_w 2 --st_com {RF_RECIPE} --wm_backbone rigidformer --wm_lr 3.5e-3 "
                                 f"{RF_SMALL} {RF_SPEED} {SYSTEMS} {PAIRED} --ckpt_every 1000 --max_hours 0.5",
                                 "two G1s, small RigidFormer (add --motion_dir): return 0.894 +- 0.002, MPJPE 47.2 / 14.7 mm global / root-relative "
                                 "(seeds 0-2: seed 0 on an A100 80GB, seeds 1-2 on an A100 40GB to its iteration count)"),
        "two-g1-supermimic-rf": (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 {RF_RECIPE} --wm_batches 2 "
                                 f"--contact_reward 0.25 --ref_contact_fix --sm_residual --wm_backbone rigidformer --wm_lr 3e-3 {RF_SMALL} {RF_SPEED} "
                                 f"{SYSTEMS} {PAIRED} --ckpt_every 1000 --max_hours 0.5",
                                 "two G1s, small RigidFormer (add --motion_dir): return 0.904 +- 0.001, MPJPE 45.8 / 17.0 mm global / root-relative "
                                 "(seeds 0-2: seed 0 on an A100 80GB, seeds 1-2 on an A100 40GB to its iteration count)"),
    },
    "train_hand": {
        "hand-supertrack-accurate": (f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights {HAND_SYSTEMS}",
                                     "XHand, 20,000 iterations: object 2.4 / 1.6 / 2.3 mm on the apple, the spray bottle and the can (A100 40GB, seeds 0-2)"),
        "hand-supermimic-accurate": (f"--algo supermimic {HAND_SYSTEMS}",
                                     "XHand, 20,000 iterations: object 3.5 / 2.9 / 3.2 mm on the apple, the spray bottle and the can; Inspire hand "
                                     "5.2 / 2.5 / 11.2 mm (A100 40GB, seeds 0-2)"),
        "hand-supertrack-fast": (f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights {HOI_OPT} {HAND_SYSTEMS} "
                                 "--iterations 6000 --ckpt_every 200",
                                 "XHand: a success of 0.99 with the object within 10 mm after 0.7 / 0.9 / 1.8 min with the speed options (README); "
                                 "Inspire hand 1.0 / 0.8 / 2.0 min (A100 40GB, mean of seeds 0-2)"),
        "hand-supermimic-fast": (f"--algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95 {HOI_OPT} {HAND_SYSTEMS} "
                                 "--iterations 6000 --ckpt_every 200",
                                 "XHand: a success of 0.99 with the object within 10 mm after 0.8 / 1.1 / 1.2 min with the speed options (README); "
                                 "Allegro hand 1.0 / 3.7 / 1.1 min (A100 40GB, mean of seeds 0-2)"),
    },
    "train_muscles": {
        "muscles-supertrack": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 {MUSCLE_RECIPE} {MUSCLE_SYS} --wm_batches 8 --pi_lr_mult 4 --wm_lr 2.5e-3 "
                               "--iterations 1000000 --last_every_min 10 --max_hours 3",
                               "MuscleVAE's character, up to 3 h on an A100 40GB PCIe: MPJPE 38.1 +- 1.0 mm (root frame), episodes ended 0.011 (seeds 1-3)"),
        "muscles-supermimic": (f"--algo supermimic --reward_std_scale 0.5 --action_noise 0.3 --imag_start_perturb 0.3 {MUSCLE_RECIPE} {MUSCLE_SYS} "
                               "--wm_batches 5 --pi_lr_mult 2 --wm_lr 2.5e-3 --iterations 1000000 --last_every_min 10 --max_hours 3",
                               "MuscleVAE's character, up to 3 h on an A100 40GB PCIe: MPJPE 65.6 +- 8.1 mm (root frame), episodes ended 0.053 (seeds 1-3)"),
        "muscles-supertrack-rf": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 {MUSCLE_RECIPE.replace(' --wm_hidden 1024 1024 1024', '')} --wm_batches 5 "
                                  f"--pi_lr_mult 4 {MUSCLE_RF} {MUSCLE_SYS} --max_hours 3 --iterations 1000000 --ckpt_every 2000 --last_every_min 10",
                                  "MuscleVAE's character, RigidFormer world model, 3 h on an A100 40GB PCIe: MPJPE 34.9 +- 0.7 mm, episodes ended 0.0004 "
                                  "(seeds 1-3)"),
        "muscles-supermimic-rf": ("--algo supermimic --reward_std_scale 0.5 --action_noise 0.3 --imag_start_perturb 0.3 --pi_ema 0.995 --lr 4e-4 --wm_batches 8 "
                                  "--pi_opt muon --wm_only_iters 0 --pi_batch 2048 --pi_lr_mult 2 --lr_decay_len 300 --lr_decay_by 1400 "
                                  f"{MUSCLE_RF} {MUSCLE_SYS} --max_hours 3 --iterations 1000000 --ckpt_every 2000 --last_every_min 10",
                                  "MuscleVAE's character, RigidFormer world model (Adam), 3 h on an A100 40GB PCIe: MPJPE 60.2 +- 2.0 mm, episodes ended "
                                  "0.009 (seeds 1-3)"),
    },
    "train_quadrotor": {
        "quadrotor-supertrack-accurate": (f"{QUAD_ST} --iterations 6000 --lr_decay_iters 6000",
                                          "racing quadrotor (add --flight), 6,000 iterations, 5.6-5.9 min on an H200: position 3.38 / 2.02 / 2.19 mm on "
                                          "the ellipse, the lemniscate and the race track at a success of 100 % (seeds 0-2)"),
        "quadrotor-supermimic-accurate": (f"{QUAD_SM} --iterations 6000 --lr_decay_iters 6000",
                                          "racing quadrotor (add --flight), 6,000 iterations, 7.0-7.3 min on an H200: position 6.55 / 4.96 / 5.08 mm on "
                                          "the ellipse, the lemniscate and the race track at a success of 100 % (seeds 0-2)"),
        "quadrotor-supertrack-fast": (f"{QUAD_ST} --iterations 1000 --lr_decay_iters 6000 --ckpt_every 250 --ckpt_keep",
                                      "racing quadrotor (add --flight): a success of 0.99 with the position within 20 mm after 0.79 / 0.68 / 0.75 min "
                                      "(H200, mean of seeds 0-2, checkpoints every 250 iterations; not engineered)"),
        "quadrotor-supermimic-fast": (f"{QUAD_SM} --iterations 1000 --lr_decay_iters 6000 --ckpt_every 250 --ckpt_keep",
                                      "racing quadrotor (add --flight): a success of 0.99 with the position within 20 mm after 0.93 / 0.78 / 0.89 min "
                                      "(H200, mean of seeds 0-2, checkpoints every 250 iterations; not engineered)"),
    },
    "train_quadruped": {
        "go2-supertrack": (f"{GO2_ST} --iterations 10000 --lr_decay_iters 10000 {GO2_TIMED}",
                           "Go2, all 65 Lifelike clips, 10,000 iterations, 11.1 min on an RTX 6000 Ada: success 0.9987 +- 0.0012, return 0.977, "
                           "feet 7.3 mm, trunk 12.8 mm; a success of 0.99 with the trunk within 20 mm after 4.43 +- 0.58 min (seeds 0-2)"),
        "go2-supermimic": (f"{GO2_SM} --iterations 10000 --lr_decay_iters 10000 {GO2_TIMED}",
                           "Go2, all 65 Lifelike clips, 10,000 iterations, 14.0 min on an RTX 6000 Ada: success 1.000, return 0.986, feet 5.3 mm, "
                           "trunk 12.1 mm; a success of 0.99 with the trunk within 20 mm after 1.31 +- 0.03 min (seeds 0-2)"),
        "go2-supertrack-clip": (f"{GO2_ST} --iterations 6000 --lr_decay_iters 6000 {GO2_TIMED}",
                                "Go2, one Lifelike clip (add --clips <name>), 6,000 iterations on an RTX 6000 Ada: return 0.992 / 0.973 / 0.985 on "
                                "the walk, the run and the play clip at a success of 1.000 (seeds 0-2)"),
        "go2-supermimic-clip": (f"{GO2_SM} --iterations 6000 --lr_decay_iters 6000 {GO2_TIMED}",
                                "Go2, one Lifelike clip (add --clips <name>), 6,000 iterations on an RTX 6000 Ada: return 0.995 / 0.977 / 0.989 on "
                                "the walk, the run and the play clip at a success of 1.000 (seeds 0-2)"),
    },
    "train_robomimic": {
        "robomimic-supertrack": (f"{ROBOMIMIC_ST} {ROBOMIMIC_RUN}",
                                 "one demonstration (add --task; --demo 0 by default), 1,500 iterations on an RTX 4090: success 100 / 100 / 89.8 %, "
                                 "object 0.60 / 1.96 / 7.13 mm on Lift / Square / Transport (seeds 0-2, 256 perturbed rollouts)"),
        "robomimic-supermimic": (f"{ROBOMIMIC_SM} {ROBOMIMIC_RUN}",
                                 "one demonstration (add --task; --demo 0 by default), 1,500 iterations on an RTX 4090: success 100 / 100 / 37.1 %, "
                                 "object 0.80 / 4.08 / 12.3 mm on Lift / Square / Transport (seeds 0-2, 256 perturbed rollouts)"),
    },
}


def expand(argv: list[str], trainer: str) -> list[str]:
    """``argv`` with ``--config NAME`` (or ``--config=NAME``) followed by NAME's options, the rest after them; ``--config list``
    prints the trainer's configurations and exits. The ``--config`` option stays in argv, so that the trainer records it."""
    names = CONFIGS[trainer]
    out, name, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--config" or a.startswith("--config="):
            if name is not None:
                raise ValueError("--config: give one configuration")
            if a == "--config":
                if i + 1 >= len(argv):
                    raise ValueError("--config needs a name (--config list: the names)")
                name, i = argv[i + 1], i + 2
            else:
                name, i = a.split("=", 1)[1], i + 1
        else:
            out.append(a)
            i += 1
    if name is None:
        return argv
    if name == "list":
        print(describe(trainer))
        sys.exit(0)
    if name not in names:
        raise ValueError(f"--config {name}: not a configuration of this trainer ({', '.join(names)}; --config list)")
    return shlex.split(names[name][0]) + out + ["--config", name]


def describe(trainer: str) -> str:
    return "\n".join(f"{name}\n    {measured}\n    {opts}" for name, (opts, measured) in CONFIGS[trainer].items())


def add_config_arg(parser):
    """The ``--config`` option itself (expanded before parsing by ``expand``), so that --help lists it and the run records it."""
    parser.add_argument("--config", type=str, default=None,
                        help="a named configuration, this repository's measured best settings for a task (--config list: the names); "
                        "options given beside it take precedence")  # fmt: skip
