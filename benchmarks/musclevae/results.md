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

Round 4's selections, seeds 1-3, mean ± std over seeds. Commands:
[README](../../README.md#muscle-actuated-character).

| | SuperTrack | SuperMimic |
|---|---|---|
| MPJPE, root frame (mm) | 38.1 ± 1.0 | 65.6 ± 8.1 |
| MPJPE, heading frame (mm) | 38.5 ± 1.1 | 45.7 ± 2.5 |
| episodes ended (share) | 0.011 ± 0.005 | 0.053 ± 0.014 |
| joint error (deg) | 10.6 ± 1.1 | 9.7 ± 0.4 |

Per seed:

| | seed | MPJPE, root frame (mm) | MPJPE, heading frame (mm) | episodes ended | joint error (deg) | best checkpoint, of the run's iterations |
|---|---|---|---|---|---|---|
| SuperTrack | 1 | 37.19 | 37.46 | 0.0052 | 10.16 | 26,000 of 28,400 |
| | 2 | 39.13 | 39.69 | 0.0117 | 11.91 | 22,000 of 24,400 |
| | 3 | 38.04 | 38.46 | 0.0156 | 9.83 | 30,000 of 32,500 |
| SuperMimic | 1 | 68.71 | 46.75 | 0.0690 | 9.58 | 6,000 of 18,400 |
| | 2 | 71.75 | 47.61 | 0.0417 | 10.22 | 6,000 of 16,450 |
| | 3 | 56.48 | 42.86 | 0.0495 | 9.37 | 8,000 of 14,450 |

SuperMimic's later checkpoints reach 55.8-67.6 mm in the root frame with 0.13-0.46 of the episodes ended, which the rule below excludes.

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
  frame, so a root's tilt moves every body; heading-frame MPJPE removes only each root's position and yaw. Selection uses the root frame. Episodes ended is the share of 768 episodes that MuscleVAE's termination rule ends within 512 steps. Joint error
  averages rotation errors of the 22 joints over the episode. Neither objective scores drift in the world on this character: world-frame MPJPE over
  the same evaluations is 2.24 ± 0.05 m (SuperTrack) and 2.31 ± 0.06 m (SuperMimic).

## Tuning

Both algorithms had the same four rounds, all with seed 0 and a budget of 1 h on an A100 40GB PCIe, ranked by the same rule.

- **Rule:** in each run, the best checkpoint as above (a checkpoint every 1,000 iterations) is evaluated with seed 1. Per
  algorithm the lowest MPJPE is selected; within 2 mm, the lower share of episodes ended. A run with no checkpoint at or
  below 0.10 ended cannot be selected.
- **Round 1:** the starting configuration, LAFAN1's selection mapped to the character, and five changes to it. The recipe is
  BlinkMimic's, the parts muscle training implements: `--pi_ema 0.995 --lr 4e-4 --wm_batches 5 --pi_opt muon --wm_opt muon
  --wm_only_iters 0 --pi_batch 2048 --pi_lr_mult 4 --lr_decay_len 300 --lr_decay_by 1400 --wm_hidden 1024 1024 1024`.
- **Round 2:** six single changes to round 1's selection.
- **Round 3:** round 2's selection with BlinkMimic's systems options, alone and with `--wm_lr 1.6e-3`, on LAFAN1 the fastest
  single change for both algorithms; stopped as above.
- **Round 4:** the same selections at `--wm_lr` 1e-3 and 2.5e-3; stopped as above. Both algorithms select 2.5e-3, the runs reported above.

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
| 4 | with `--wm_lr 1e-3`: 42.6 / 0.003 | with `--wm_lr 1e-3`: 69.8 / 0.087 |
| | with `--wm_lr 2.5e-3`: **37.2 / 0.000** | with `--wm_lr 2.5e-3`: **57.0 / 0.092** |

Each entry is one run. Round 2's selections scored 47.3 mm (SuperTrack) and 66.8 mm (SuperMimic) in round 2 and, with the
systems options, which also change the random stream, 99.0 and 60.5 mm in round 3.

**Runs outside the rounds.** Before round 1, each starting configuration trained once for 12 h with seed 0 as a development run; at about 3 h,
SuperTrack 32.8 mm, ended 0.008, joint error 9.8 deg; SuperMimic 73.6 mm, ended 0.056, joint error 9.8 deg. Round 1's selections, up to 3 h with the
default updates, seed 1: SuperTrack 79.2 mm, ended 0.096; SuperMimic 116.3 mm, ended 0.112. Round 2's, the same, seeds 1-3: SuperTrack 38.6, 105.4 and
78.6 mm (74.2 ± 33.7), ended 0.022 ± 0.019, joint error 16.1 ± 4.4 deg; SuperMimic 69.2, 64.1 and 70.9 mm (68.0 ± 3.5), ended 0.079 ± 0.026, joint
error 10.0 ± 0.3 deg.

**Round 3 was declared after those three SuperTrack runs had ended, while SuperMimic's three were ending.**

**Round 3's selections at 3 h.** Up to 3 h, seeds 1-3, evaluated as above: SuperTrack at `--wm_lr` 1.6e-3 36.7 ± 1.0 mm, ended 0.008 ± 0.002, joint error 11.5 ± 0.8 deg; at 1.5e-3, the next value
on the grid of published rates, 36.4 ± 1.9 mm, 0.014 ± 0.011, 10.9 ± 0.9 deg; SuperMimic at the recipe's 4e-4 63.6 ± 4.1 mm, 0.059 ± 0.027, 10.1 ± 0.6 deg. Round 4 selects by the same 1 h rule,
so its selections are the ones reported; at 3 h their means are about 2 mm above these.

## The systems options

Rounds 3 and 4 and the reported runs use BlinkMimic's systems options (`--no_sync --fused_adam --update_graphs --graph_early
--collect_graph --reset_masked --ckpt_light`; [details](../../docs/blinkmimic.md)). On a CPU in eager mode, with fused Adam on both sides, their code gives the default updates bit for bit on the same random numbers
(`tests/test_muscle_systems.py`). The runs without them use the non-fused Adam, and the random stream differs. Their effect on the results
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
