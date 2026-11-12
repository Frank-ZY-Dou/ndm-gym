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
RESETS = "--fail_reset 0.3 --bin_fix --eprobe_share 0.15 --eprobe_graph --eprobe_async 20 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7"
SYSTEMS = "--fused_adam --elementwise_3x3 --combo_kernels --compile_eager --no_sync --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light"
# speed options of the 3 h confirmation runs: SPEED for both backbones (SuperMimic's RigidFormer runs also --critic_cudagraphs), RF_SPEED
# RigidFormer's own
SPEED = "--wm_cudagraphs --pi_cudagraphs --fused_adam --elementwise_3x3 --combo_kernels --compile_eager"
RF_SPEED = "--rf_fused_train --rf_bptt_keep_blocks --rf_fit_keep_ops"
TIMED = "--iterations 6000 --ckpt_every 100"  # the timed runs' length and checkpoints, each evaluated while training (ndm-eval --watch)
PAIRED = "--motions interhuman --paired"  # with --motion_dir: a two-person clip (tools/convert_interhuman.py)
MUSCLE_RECIPE = ("--pi_ema 0.995 --lr 4e-4 --pi_opt muon --wm_opt muon --wm_only_iters 0 --pi_batch 2048 --lr_decay_len 300 --lr_decay_by 1400 "
                 "--wm_hidden 1024 1024 1024")  # fmt: skip
MUSCLE_SYS = "--no_sync --fused_adam --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light"
HAND_SYSTEMS = "--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs"
HOI_OPT = "--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300"  # HOI's optimizer options

# trainer -> name -> (options, what the README measured with them)
CONFIGS = {
    "train": {
        "lafan1-supertrack-accurate": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 {SPEED} --max_hours 3",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB with the speed options (about 35,700 iterations): return 0.861 +- 0.003, "
                                       "MPJPE 14.5 +- 0.2 mm (seeds 7-9)"),
        "lafan1-supermimic-accurate": ("--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 --max_hours 3",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB: return 0.824 +- 0.013, MPJPE 28.4 +- 0.5 mm (seeds 1-3)"),
        "lafan1-supertrack-fast": (f"--algo supertrack --st_fail_w 1 --st_balance_w 2 {RECIPE} --st_quad_w 0.5 --wm_lr 2e-3 --goal_table "
                                   f"--pi_encode_once {RESETS} {SYSTEMS} {TIMED}",
                                   "LAFAN1: return 0.82 after 4.19 +- 0.16 min on an A100 40GB SXM4, 2.54 +- 0.11 min on an H100 (seeds 0-5)"),
        "lafan1-supermimic-fast": (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.15 {RECIPE} --pi_ema 0.99 "
                                   f"--pi_batch 1024 --pi_lr_mult 2 --pi_updates 1 --wm_lr 1.5e-3 --goal_table --contact_reward 0.15 --ref_contact_fix "
                                   f"--sm_residual --wm_structured {RESETS} {SYSTEMS} {TIMED}",
                                   "LAFAN1: return 0.82 after 4.38 +- 0.57 min on an A100 40GB SXM4, 2.68 +- 0.19 min on an H100 (seeds 0-5)"),
        "lafan1-supertrack-rf-accurate": (f"--algo supertrack --wm_backbone rigidformer --wm_lr 4e-4 --st_fail_w 5 --st_balance_w 2 --wm_batches 5 "
                                          f"{SPEED} {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.832 +- 0.003, MPJPE 21.0 +- 1.3 mm (seeds 7-9)"),
        "lafan1-supermimic-rf-accurate": (f"--algo supermimic --wm_backbone rigidformer --wm_lr 4e-4 --reward_std_scale 0.7 --action_noise 0.3 "
                                          f"--imag_start_perturb 0.4 --wm_batches 7 {SPEED} --critic_cudagraphs {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.804 +- 0.007, MPJPE 33.1 +- 0.5 mm (seeds 7-9)"),
        "two-g1-supertrack": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 --st_com {RECIPE} --pi_batch 4096 {SYSTEMS} {PAIRED} "
                              "--ckpt_every 1000 --max_hours 0.5",
                              "two G1s, InterHuman clip 1566 (add --motion_dir): return 0.893, MPJPE 39 / 14.2 mm global / local (A100 80GB, seed 0)"),
        "two-g1-supermimic": (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 {RECIPE} --pi_lr_mult 2 "
                              f"--contact_reward 0.25 --ref_contact_fix --sm_residual --wm_structured {SYSTEMS} {PAIRED} --ckpt_every 1000 "
                              "--max_hours 0.3",
                              "two G1s, InterHuman clip 1566 (add --motion_dir): return 0.905, MPJPE 40 / 16.4 mm global / local (A100 80GB, seed 0)"),
    },
    "train_hand": {
        "hand-supertrack-accurate": (f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights {HAND_SYSTEMS}",
                                     "XHand, 20,000 iterations: object 2.4 / 1.6 / 2.3 mm on the apple, the spray bottle and the can (seeds 0-2)"),
        "hand-supermimic-accurate": (f"--algo supermimic {HAND_SYSTEMS}",
                                     "XHand, 20,000 iterations: object 3.5 / 2.9 / 3.2 mm on the apple, the spray bottle and the can (seeds 0-2)"),
        "hand-supertrack-fast": (f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights {HOI_OPT} {HAND_SYSTEMS} "
                                 "--iterations 6000 --ckpt_every 200",
                                 "XHand: a success of 0.99 with the object within 10 mm after 0.9 / 1.5 / 2.4 min (A100 40GB, mean of seeds 0-2)"),
        "hand-supermimic-fast": (f"--algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95 {HOI_OPT} {HAND_SYSTEMS} "
                                 "--iterations 6000 --ckpt_every 200",
                                 "XHand: a success of 0.99 with the object within 10 mm after 1.3 / 1.6 / 1.6 min (A100 40GB, mean of seeds 0-2)"),
    },
    "train_muscles": {
        "muscles-supertrack": (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 {MUSCLE_RECIPE} {MUSCLE_SYS} --wm_batches 8 --pi_lr_mult 4 --wm_lr 2.5e-3 "
                               "--iterations 1000000 --last_every_min 10 --max_hours 3",
                               "MuscleVAE's character, up to 3 h on an A100 40GB: MPJPE 38.1 +- 1.0 mm, episodes ended 0.011 (seeds 1-3)"),
        "muscles-supermimic": (f"--algo supermimic --reward_std_scale 0.5 --action_noise 0.3 --imag_start_perturb 0.3 {MUSCLE_RECIPE} {MUSCLE_SYS} "
                               "--wm_batches 5 --pi_lr_mult 2 --wm_lr 2.5e-3 --iterations 1000000 --last_every_min 10 --max_hours 3",
                               "MuscleVAE's character, up to 3 h on an A100 40GB: MPJPE 65.6 +- 8.1 mm, episodes ended 0.053 (seeds 1-3)"),
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
