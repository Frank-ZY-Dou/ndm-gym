# Muscle-actuated character: results

SuperTrack and SuperMimic, with MLP world models, track MuscleVAE's reference clips on MuscleVAE's character.

## Setup

- **Character:** MuscleVAE's ([Feng et al. 2023](https://arxiv.org/abs/2312.07340)): 23 bodies, 22 joints and 284 Hill-type
  muscles with fatigue.
- **Simulator:** our Newton / MuJoCo-Warp simulator ([`muscle_sim.py`](../../ndm_gym/sim/rigid/muscle_sim.py)): physics at
  120 Hz, control at 20 Hz, muscle forces at every substep, with `--ball_joints euler --joint_damping ode`, the setting in
  which MuscleVAE's released policy tracks ([design](../../docs/musclevae_design.md)).
- **Task:** MuscleVAE's three clips (LAFAN1 walk, run and jump, retargeted to its skeleton), resets, reference switches and
  termination.
- **Training:** `ndm_gym.train_muscles`, 4096 environments, 32 imagined steps.
- **Data:** MuscleVAE's repository has no license and LAFAN1 is CC BY-NC-ND 4.0, so `tools/download_musclevae.sh` and
  `tools/convert_musclevae.py` download and convert them locally. Neither they nor weights of policies trained on them are in
  this repository; the README's animations are renders for non-commercial illustration.

## Results

Selected configurations, seeds 1-3, mean ± std over seeds. Commands:
[README](../../README.md#muscle-actuated-character).

| | SuperTrack | SuperMimic |
|---|---|---|
| MPJPE, root frame (mm) | 36.4 ± 1.9 | 63.6 ± 4.1 |
| episodes ended (share) | 0.014 ± 0.011 | 0.059 ± 0.027 |
| joint error (deg) | 10.9 ± 0.9 | 10.1 ± 0.6 |

Per seed:

| | seed | MPJPE, root frame (mm) | episodes ended | joint error (deg) | best checkpoint, of the run's iterations |
|---|---|---|---|---|---|
| SuperTrack | 1 | 35.58 | 0.0013 | 10.47 | 20,000 of 22,650 |
| | 2 | 34.98 | 0.0182 | 10.39 | 24,000 of 26,600 |
| | 3 | 38.54 | 0.0221 | 11.98 | 24,000 of 24,800 |
| SuperMimic | 1 | 66.10 | 0.0430 | 10.59 | 6,000 of 16,600 |
| | 2 | 65.84 | 0.0443 | 10.10 | 8,000 of 24,650 |
| | 3 | 58.81 | 0.0911 | 9.47 | 10,000 of 16,600 |

SuperMimic's seeds 1 and 2 later reach 57.0 and 59.3 mm with 0.11 and 0.43 of the episodes ended, which the rule below excludes.

## How the numbers were obtained

- **Budget:** at most 3 h of training per run, on one A100 40GB PCIe, seeds 1-3, a checkpoint every 2,000 iterations.
- **Stopping:** stop when, over the last three checkpoints, both the best MPJPE among checkpoints with at most 0.10 of
  episodes ended and the last checkpoint's MPJPE fall by less than 1 mm. All six runs stopped this way before 3 h.
  Stopping is external, as evaluations arrive; `train_muscles` itself runs to `--max_hours`.
- **Evaluation:** `tools/evaluate_muscles.py`, 768 environments started at frames spread evenly over the clips, 512 control
  steps, no resets, no action noise.
- **Best checkpoint:** evaluate every checkpoint with seed 0 as it is written. Select the lowest root-frame MPJPE among
  checkpoints with at most 0.10 of episodes ended, then reevaluate with seed 4, which no run trained with. Tables report
  this evaluation.
- **Measures:** root-frame MPJPE averages position errors of the 23 bodies over the episode, each side in its own root's
  frame. Episodes ended is the share of 768 episodes that MuscleVAE's termination rule ends within 512 steps. Joint error
  averages rotation errors of the 22 joints over the episode.

## Tuning

Both algorithms had the same three rounds, all with seed 0 and a budget of 1 h on an A100 40GB PCIe, ranked by the same rule.

- **Rule:** in each run, the best checkpoint as above (a checkpoint every 1,000 iterations) is evaluated with seed 1. Per
  algorithm the lowest MPJPE is selected; within 2 mm, the lower share of episodes ended. A run with no checkpoint at or
  below 0.10 ended cannot be selected.
- **Round 1:** the starting configuration, LAFAN1's selection mapped to the character, and five changes to it. The recipe is
  BlinkMimic's, the parts muscle training implements: `--pi_ema 0.995 --lr 4e-4 --wm_batches 5 --pi_opt muon --wm_opt muon
  --wm_only_iters 0 --pi_batch 2048 --pi_lr_mult 4 --lr_decay_len 300 --lr_decay_by 1400 --wm_hidden 1024 1024 1024`.
- **Round 2:** six single changes to round 1's selection.
- **Round 3:** round 2's selection with BlinkMimic's systems options, alone and with `--wm_lr 1.6e-3`, on LAFAN1 the fastest
  single change for both algorithms; stopped as above.

Root-frame MPJPE (mm) / episodes ended, in bold the run each round selects:

| round | SuperTrack | SuperMimic |
|---|---|---|
| 1 | starting configuration (`--st_fail_w 5 --st_balance_w 4`): 92.0 / 0.049 | starting configuration (`--reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3`): 117.9 / 0.034 |
| | the recipe: **50.0 / 0.007** | the recipe: 151.9 / 0.065 |
| | the recipe with `--pi_lr_mult 2`: 124.6 / 0.115 | the recipe with `--pi_lr_mult 2`: **92.5 / 0.043** |
| | the recipe with `--pi_batch 4096 --st_balance_w 2`: 65.5 / 0.010 | the recipe with `--pi_batch 4096`: 106.4 / 0.113 |
| | `--st_balance_w 2`: 88.3 / 0.129 | `--action_noise 0.1`: 129.1 / 0.122 |
| | `--st_fail_w 10`: 51.9 / 0.020 | `--reward_std_scale 0.5`: 96.1 / 0.009 |
| 2 | `--imag_start_perturb 0.15`: 71.5 / 0.025 | `--imag_start_perturb 0.15`: 114.6 / 0.082 |
| | `--pi_ema 0.99`: 47.0 / 0.007 | `--pi_ema 0.99`: 155.3 / 0.042 |
| | `--wm_batches 8`: **47.3 / 0.003** | `--wm_batches 8`: 74.9 / 0.100 |
| | `--lr_decay_by 2000`: 84.8 / 0.100 | `--lr_decay_by 2000`: 85.8 / 0.064 |
| | `--st_fail_w 10`: 51.4 / 0.008 | `--reward_std_scale 0.5`: **66.8 / 0.057** |
| | `--st_fail_w 3`: 89.0 / 0.004 | `--pi_lr_mult 1`: 100.6 / 0.046 |
| 3 | round 2's selection: 99.0 / 0.021 | round 2's selection: **60.5 / 0.059** |
| | with `--wm_lr 1.6e-3`: **37.6 / 0.001** | with `--wm_lr 1.6e-3`: 62.3 / 0.074 |

Each entry is one run. Round 2's selections scored 47.3 mm (SuperTrack) and 66.8 mm (SuperMimic) in round 2 and, with the
systems options, which also change the random stream, 99.0 and 60.5 mm in round 3.

Before round 1, each starting configuration trained once for 12 h with seed 0 as a development run. Before round 3, the
selections of rounds 1 and 2 trained for up to 3 h with the default updates. Round 2's, seeds 1-3: SuperTrack 38.6, 105.4 and
78.6 mm (74.2 ± 33.7), ended 0.022 ± 0.019, joint error 16.1 ± 4.4 deg; SuperMimic 69.2, 64.1 and 70.9 mm (68.0 ± 3.5), ended
0.079 ± 0.026, joint error 10.0 ± 0.3 deg. Round 1's, seed 1: SuperTrack 79.2 mm, ended 0.096; SuperMimic 116.3 mm, ended
0.112. Round 3 was declared after SuperTrack's runs had ended, while SuperMimic's were ending.

**SuperTrack's world-model rate.** Published learning rates stay on a grid of 0.5 steps (1.5e-3, 2e-3, 2.5e-3). SuperTrack's
world-model rate was therefore set to 1.5e-3, the grid value next to the selected 1.6e-3, and its runs repeated with seeds
1-3: 36.4 ± 1.9 mm, ended 0.014 ± 0.011, joint error 10.9 ± 0.9 deg, against 36.7 ± 1.0 mm, 0.008 ± 0.002 and 11.5 ± 0.8 deg at
1.6e-3. The rule, fixed before these runs, was to report 1.5e-3 if its mean MPJPE lay within 2 mm of 36.7 mm or below.
SuperMimic's selection keeps the recipe's world-model rate (`--lr 4e-4`), so it had no rerun.

## The systems options

Round 3 and the reported runs use BlinkMimic's systems options (`--no_sync --fused_adam --update_graphs --graph_early
--collect_graph --reset_masked --ckpt_light`; [details](../../docs/blinkmimic.md)). On the same random numbers their code
gives the default updates bit for bit (`tests/test_muscle_systems.py`); the random stream differs. Their effect on the results
was checked on SuperMimic's round-2 selection, with and without them, seeds 1-3, 4,000 iterations, one A100 40GB PCIe per
run, evaluated with seed 0:

| | seed 1 | seed 2 | seed 3 | Welch test over the seeds |
|---|---|---|---|---|
| MPJPE, root frame (mm), with minus without | +4.5 | +7.2 | +36.9 | p = 0.19 |
| episodes ended, with minus without | +0.020 | +0.083 | +0.048 | p = 0.20 |
| joint error (deg), with minus without | +0.24 | +1.22 | +3.84 | p = 0.20 |
| training time to 4,000 iterations (h), with / without | 0.253 / 0.466 | 0.252 / 0.477 | 0.250 / 0.417 | |

The rule, fixed before these runs: the options change the results only if all three seeds show a joint error at least 2 deg
higher with them and the Welch test gives p < 0.05. Neither holds.

An earlier development check on the starting configurations (training and evaluation seed 0, at 4,000 iterations): SuperTrack
101.0 mm without the options, 107.4 and 103.7 mm with them (two runs); SuperMimic 114.7 mm without, 139.4 and 121.2 mm with.
