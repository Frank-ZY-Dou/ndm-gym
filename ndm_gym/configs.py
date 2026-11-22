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
# speed options of the 3 h runs: SPEED for the RigidFormer's and the MLP's earlier ones (SuperMimic's runs also
# --critic_cudagraphs), RF_SPEED the RigidFormer's own
SPEED = "--wm_cudagraphs --pi_cudagraphs --fused_adam --elementwise_3x3 --combo_kernels --compile_eager"
RF_SPEED = "--rf_fused_train --rf_bptt_keep_blocks --rf_fit_keep_ops"
RF_SMALL = "--rf_merge joints --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 2"
TIMED = "--iterations 6000 --ckpt_every 100"  # the timed runs' length and checkpoints, each evaluated (ndm-eval)
LEAN = "--eprobe_lean --imag_hoist --wm_presample --muon_batched --gc_freeze"  # leaner probe sweeps and the learner's set (one GPU)
AWL4 = "--layout AWL --eprobe_gpu 3 --eprobe_every 250 --eprobe_async 4 --eprobe_lean --gc_freeze"  # four GPUs: actor | WM | policy, probes
ST_FAST = f"--algo supertrack --st_fail_w 1 --st_balance_w 2 {RECIPE} --st_quad_w 0.5 --wm_lr 2e-3 --goal_table --pi_encode_once {RESETS} {SYSTEMS} {TIMED}"
SM_FAST = (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.15 {RECIPE} --pi_ema 0.99 --pi_batch 1024 --pi_lr_mult 2 "
           f"--pi_updates 1 --wm_lr 1.5e-3 --goal_table --contact_reward 0.15 --ref_contact_fix --sm_residual --wm_structured {RESETS} {SYSTEMS} {TIMED}")  # fmt: skip
# the most accurate configurations: the fastest ones trained 3 h with the learning rate's cosine over the 3 h, as their runs ran
# (the arguments the runs recorded: tests/test_configs.py); SuperMimic's with action noise 0.1 in place of 0.3
CONFIRMED_3H = "--max_hours 3 --iterations 1000000 --ckpt_every 2000 --lr_decay_hours 3"
SM_CONFIRMED = f"{SM_FAST} --pi_encode_once {LEAN}".replace("--action_noise 0.3", "--action_noise 0.1")
PAIRED = "--motions interhuman --paired"  # with --motion_dir: a two-person clip (tools/convert_interhuman.py)
ENCODE_ONCE = "--goal_table --pi_encode_once"  # the policy's goal encoded once per update from every frame's goal (equal up to rounding)
TWO_G1_ST = (f"--algo supertrack --st_fail_w 5 --st_balance_w 4 --st_com {RECIPE} --pi_batch 4096 {ENCODE_ONCE} {SYSTEMS} {PAIRED} "
             "--ckpt_every 1000 --iterations 9396")  # fmt: skip
TWO_G1_SM = (f"--algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 {RECIPE} --pi_lr_mult 2 --contact_reward 0.25 "
             f"--ref_contact_fix --sm_residual --wm_structured {ENCODE_ONCE} {SYSTEMS} {PAIRED} --ckpt_every 1000 --iterations 9396")  # fmt: skip
AWL3 = "--layout AWL"  # three GPUs: actor | world-model learner | policy learner (the two-G1 configurations run no probe sweeps)
# each algorithm's speed options on the two G1s and its policy batch halved (4,096 to 2,048 and 2,048 to 1,024): shorter iterations, the
# same accuracy at the published budget within the seeds' spread
TWO_G1_ST_FAST = f"{TWO_G1_ST.replace('--pi_batch 4096', '--pi_batch 2048')} --st_wm_fused"
TWO_G1_SM_FAST = f"{TWO_G1_SM.replace('--pi_batch 2048', '--pi_batch 1024')} --critic_compile --compile_norms --critic_fused"
MUSCLE_RECIPE = ("--pi_ema 0.995 --lr 4e-4 --pi_opt muon --wm_opt muon --wm_only_iters 0 --pi_batch 2048 --lr_decay_len 300 --lr_decay_by 1400 "
                 "--wm_hidden 1024 1024 1024")  # fmt: skip
MUSCLE_SYS = "--no_sync --fused_adam --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light"
MUSCLE_RF = ("--wm_backbone rigidformer --rf_muscle_tokens body --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 9 --rf_fused_train "
             "--rf_fused_features --wm_lr 5e-3")  # fmt: skip  # the body-token RigidFormer of the muscle 3 h runs
HAND_SYSTEMS = "--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs"
HOI_OPT = "--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300"  # HOI's optimizer options
QUAD_SYSTEMS = "--fused_adam --reset_masked --no_sync --collect_graph --update_graphs"  # the hand task's, without the forward kinematics' option
# the quadrotor's settings: a world model of 3 x 256; the most accurate configurations imagine 100 steps (1 s) with HOI's optimizer
# options, the fast ones 64 steps (0.64 s) with Muon, no policy average and smaller batches
QUAD_ST = f"--algo supertrack --st_fail_w 2 --st_task_weights --wm_hidden 256 256 256 {QUAD_SYSTEMS}"
QUAD_SM = f"--algo supermimic --wm_hidden 256 256 256 {QUAD_SYSTEMS}"
QUAD_ST_OPT = HOI_OPT.replace("--pi_ema 0.995", "--pi_ema 0")  # HOI's optimizer options without the policy average
# the 6,000-iteration runs' cosine; a checkpoint every 50 iterations for the time to the target (time_to.sh <run> 1500 500)
QUAD_TIMED = "--iterations 1500 --lr_decay_iters 6000 --ckpt_every 50 --ckpt_keep"
# the Go2's selected settings (benchmarks/go2_lifelike_tracking): SuperTrack with the hand task's options, SuperMimic with HOI's
# optimizer options and a discount of 0.95
GO2_ST = f"--algo supertrack {HAND_SYSTEMS} --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights"
GO2_SM = f"--algo supermimic {HAND_SYSTEMS} --lr 4e-4 --action_noise 0.3 --bptt_decay 0.95 {HOI_OPT} --gamma 0.95"
GO2_TIMED = "--ckpt_every 500 --keep_ckpts"  # the checkpoints evaluate_quadruped --ckpts times
# manipulation demonstration tracking: the hand task's settings and systems options as the starting point
ROBOMIMIC_ST = f"--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights {HAND_SYSTEMS}"
ROBOMIMIC_SM = f"--algo supermimic {HAND_SYSTEMS}"
# the benchmark's runs (benchmarks/robomimic_manipulation_tracking): 1,500 iterations on the schedule of 4,000, the solver at 20
# iterations, a checkpoint kept every 100
ROBOMIMIC_RUN = "--solver_iterations 20 --iterations 1500 --lr_decay_iters 4000 --ckpt_every 100 --ckpt_keep"

# trainer -> name -> (options, what the README measured with them)
CONFIGS = {
    "train": {
        "lafan1-supertrack-accurate": (f"{ST_FAST} {LEAN} {CONFIRMED_3H}",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB SXM4 with the cosine over the 3 h (about 63,310 iterations): "
                                       "return 0.898 +- 0.002, MPJPE 10.8 +- 0.0 mm (seeds 7-9)"),
        "lafan1-supermimic-accurate": (f"{SM_CONFIRMED} {CONFIRMED_3H}",
                                       "LAFAN1, 40 clips, 3 h on an A100 40GB SXM4 with the cosine over the 3 h (about 67,030 iterations): "
                                       "return 0.915 +- 0.005, MPJPE 16.7 +- 0.1 mm (seeds 7-9)"),
        "lafan1-supertrack-fast": (f"{ST_FAST} {LEAN}",
                                   "LAFAN1: return 0.82 after 4.13 +- 0.18 min on an A100 40GB SXM4 (seeds 0-2, one run at a time; 3.96 +- 0.03 min "
                                   "without the lean options in the same set of runs, level within the seeds' spread)"),
        "lafan1-supermimic-fast": (f"{SM_FAST} --pi_encode_once {LEAN}",
                                   "LAFAN1: return 0.82 after 3.52 +- 0.30 min on an A100 40GB SXM4 (seeds 0-2, one run at a time)"),
        "lafan1-supertrack-fast-4gpu": (f"{ST_FAST} {AWL4} --fk_fused",
                                        "LAFAN1 on four A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of four GPUs): return 0.82 after 1.47 +- 0.05 min, 5.9 GPU-minutes "
                                        "(seeds 3-5; 1.52 +- 0.08 without --fk_fused in the same session)"),
        "lafan1-supermimic-fast-4gpu": (f"{SM_FAST} --pi_encode_once {AWL4} --fk_fused",
                                        "LAFAN1 on four A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of four GPUs): return 0.82 after 1.42 +- 0.09 min, 5.7 GPU-minutes "
                                        "(seeds 3-5; 1.49 +- 0.05 without --fk_fused in the same session)"),
        "lafan1-supertrack-rf-accurate": (f"--algo supertrack --wm_backbone rigidformer --wm_lr 4e-4 --st_fail_w 5 --st_balance_w 2 --wm_batches 5 "
                                          f"{SPEED} {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.832 +- 0.003, MPJPE 21.0 +- 1.3 mm (seeds 7-9)"),
        "lafan1-supermimic-rf-accurate": (f"--algo supermimic --wm_backbone rigidformer --wm_lr 4e-4 --reward_std_scale 0.7 --action_noise 0.3 "
                                          f"--imag_start_perturb 0.4 --wm_batches 7 {SPEED} --critic_cudagraphs {RF_SPEED} --max_hours 3",
                                          "LAFAN1, RigidFormer world model, 3 h on an A100 40GB: return 0.804 +- 0.007, MPJPE 33.1 +- 0.5 mm (seeds 7-9)"),
        "two-g1-supertrack": (TWO_G1_ST_FAST,
                              "two G1s, InterHuman clip 1566 (add --motion_dir); its world-model loss as fused kernels (--st_wm_fused) and its "
                              "policy batch halved (--pi_batch 2048): return 0.88 after 3.48 +- 0.49 min, root-relative MPJPE there 16.8 mm, "
                              "143.3 ms per iteration on an A100 40GB SXM4 (seeds 0-3); at its 9,396 iterations return 0.8925 +- 0.0011, MPJPE "
                              "46.2 / 14.1 mm global / root-relative with the policy batch of 4,096 (seeds 0-2: seed 0 on an A100 80GB, in 0.5 h, "
                              "seeds 1-2 on an A100 40GB SXM4; before the goal was encoded once, which computes the same values up to float "
                              "rounding), 0.8929 +- 0.0011, 44.7 / 14.3 mm with 2,048 (seeds 0-2, A100 40GB SXM4)"),
        "two-g1-supermimic": (TWO_G1_SM_FAST,
                              "two G1s, InterHuman clip 1566 (add --motion_dir); its critic's update and its world-model normalizer updates "
                              "compiled (--critic_compile --compile_norms), the critic's steps as fused kernels (--critic_fused) and its policy "
                              "batch halved (--pi_batch 1024): return 0.88 after 2.70 +- 0.27 min, root-relative MPJPE there 20.5 mm, 127.2 ms "
                              "per iteration on an A100 40GB SXM4 (seeds 0-3); at two-g1-supertrack's 9,396 iterations return 0.9055 +- 0.0011, "
                              "MPJPE 45.3 / 16.0 mm global / root-relative with the policy batch of 2,048 (seeds 0-2, A100 40GB SXM4; for 0.3 h, "
                              "6,264 iterations: 0.9042 +- 0.0015, 47.5 / 16.1 mm; before the goal was encoded once, which computes the same "
                              "values up to float rounding), 0.9053 +- 0.0007, 45.6 / 16.2 mm with 1,024 (seeds 0-2)"),
        "two-g1-supertrack-3gpu": (f"{TWO_G1_ST_FAST} {AWL3}",
                                   "two G1s on three A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of three GPUs; add --motion_dir): actor, world-model "
                                   "learner and policy learner one GPU each, two-g1-supertrack's options; return 0.88 after 1.63 +- 0.15 min, "
                                   "root-relative MPJPE there 16.1 mm, 63.7 ms per iteration (seeds 0-2)"),
        "two-g1-supermimic-3gpu": (f"{TWO_G1_SM_FAST} {AWL3}",
                                   "two G1s on three A100 40GB SXM4 (CUDA_VISIBLE_DEVICES of three GPUs; add --motion_dir): actor, world-model "
                                   "learner and policy learner one GPU each, two-g1-supermimic's options; return 0.88 after 1.48 +- 0.16 min, "
                                   "root-relative MPJPE there 20.6 mm, 64.4 ms per iteration (seeds 0-2)"),
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
                                 "XHand: a success of 0.99 with the object within 10 mm after 0.7 / 0.9 / 1.8 min with the speed options, without "
                                 "--target_rate_w (results page); Inspire hand 1.0 / 0.8 / 2.0 min (A100 40GB, mean of seeds 0-2)"),
        "hand-supermimic-fast": (f"--algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95 {HOI_OPT} {HAND_SYSTEMS} "
                                 "--iterations 6000 --ckpt_every 200",
                                 "XHand: a success of 0.99 with the object within 10 mm after 0.8 / 1.1 / 1.2 min with the speed options, without "
                                 "--target_rate_w (results page); Allegro hand 1.0 / 3.7 / 1.1 min (A100 40GB, mean of seeds 0-2)"),
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
        "quadrotor-supertrack-accurate": (f"{QUAD_ST} --lr 8e-4 --horizon 100 {HOI_OPT} --iterations 6000 --lr_decay_iters 6000",
                                          "racing quadrotor (add --flight), 6,000 iterations, 13.1-13.6 min on an A100 40GB SXM4: position 3.06 / "
                                          "1.89 / 1.98 mm on the ellipse, the lemniscate and the race track, 1.11 / 0.99 / 2.75 mm on their piloted "
                                          "flights, 2.31 mm with one policy on all 18 autonomous flights, at a success of 100 % (seeds 0-2)"),
        "quadrotor-supermimic-accurate": (f"{QUAD_SM} --lr 4e-4 --horizon 100 {HOI_OPT} --iterations 6000 --lr_decay_iters 6000",
                                          "racing quadrotor (add --flight), 6,000 iterations, 17.6-18.1 min on an A100 40GB SXM4: position 5.71 / "
                                          "3.95 / 4.08 mm on the ellipse, the lemniscate and the race track, 4.48 / 4.00 / 6.63 mm on their piloted "
                                          "flights, 4.94 mm with one policy on all 18 autonomous flights, at a success of 100 % (seeds 0-2)"),
        "quadrotor-supertrack-fast": (f"{QUAD_ST} --lr 4e-4 --horizon 64 {QUAD_ST_OPT} --pi_batch 1024 --wm_batch 1024 {QUAD_TIMED}",
                                      "racing quadrotor (add --flight): a success of 0.99 with the position within 20 mm, held for 500 iterations, "
                                      "from 0.38 / 0.32 / 0.36 min on the ellipse, the lemniscate and the race track (A100 40GB SXM4, mean of seeds "
                                      "0-2, every run; a checkpoint every 50 iterations, 64 perturbed starts, warm compile caches; not engineered)"),
        "quadrotor-supermimic-fast": (f"{QUAD_SM} --lr 4e-4 --horizon 64 --pi_opt muon --wm_opt muon --pi_batch 2048 {QUAD_TIMED}",
                                      "racing quadrotor (add --flight): a success of 0.99 with the position within 20 mm, held for 500 iterations, "
                                      "from 0.71 / 0.69 / 0.67 min on the ellipse, the lemniscate and the race track (A100 40GB SXM4, mean of seeds "
                                      "0-2, every run; a checkpoint every 50 iterations, 64 perturbed starts, warm compile caches; not engineered)"),
    },
    "train_quadruped": {
        "go2-supertrack": (f"{GO2_ST} --iterations 10000 --lr_decay_iters 10000 {GO2_TIMED}",
                           "Go2, all 65 Lifelike clips, 10,000 iterations"),
        "go2-supermimic": (f"{GO2_SM} --iterations 10000 --lr_decay_iters 10000 {GO2_TIMED}",
                           "Go2, all 65 Lifelike clips, 10,000 iterations"),
        "go2-supertrack-clip": (f"{GO2_ST} --iterations 6000 --lr_decay_iters 6000 {GO2_TIMED}",
                                "Go2, one Lifelike clip (add --clips <name>), 6,000 iterations"),
        "go2-supermimic-clip": (f"{GO2_SM} --iterations 6000 --lr_decay_iters 6000 {GO2_TIMED}",
                                "Go2, one Lifelike clip (add --clips <name>), 6,000 iterations"),
    },
    "train_robomimic": {
        "robomimic-supertrack": (f"{ROBOMIMIC_ST} {ROBOMIMIC_RUN}",
                                 "one demonstration (add --task; --demo 0 by default), 1,500 iterations: success 100 / 100 / 70.4 %, object 0.62 / 1.57 / "
                                 "5.69 mm on Lift / Square / Transport (seeds 0-2, 256 perturbed rollouts, A100 80GB SXM4)"),
        "robomimic-supermimic": (f"{ROBOMIMIC_SM} {ROBOMIMIC_RUN}",
                                 "one demonstration (add --task; --demo 0 by default), 1,500 iterations: success 100 / 84.9 / 29.7 %, object 0.87 / 5.34 / "
                                 "10.6 mm on Lift / Square / Transport (seeds 0-2, Transport's object error 2 of them; 256 perturbed rollouts, A100 80GB "
                                 "SXM4)"),
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
