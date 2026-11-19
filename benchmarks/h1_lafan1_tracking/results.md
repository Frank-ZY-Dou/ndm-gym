# Humanoid motion tracking on MuJoCo-Warp (H1): results

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make a Unitree H1 track LAFAN1
clips retargeted to it ([the task](../../docs/tasks/humanoid.md), [the rules](README.md)).

| goal | configuration (`train_humanoid`) | result |
|---|---|---|
| one policy on 34 LAFAN1 clips, a fixed budget of 8,000 iterations | `--config h1-supertrack` | return 0.786 ± 0.011, success 0.820 ± 0.037, MPJPE about the pelvis 22.7 mm; 14.5 min of training (seeds 0-2, RTX 6000 Ada) |
| | `--config h1-supermimic` | return 0.827 ± 0.008, success 0.771 ± 0.016, MPJPE about the pelvis 30.7 mm; 12.0 min of training (seeds 0-2, RTX 6000 Ada) |

**In short:** on the 34 clips SuperTrack / SuperMimic reach a return of 0.786 / 0.827 and complete 82 % / 77 % of the
rollouts of whole clips (2.2 to 4.6 minutes each) after 14.5 / 12.0 min of training on an RTX 6000 Ada; on the 27 clips
whose reference stays upright they complete 96 % / 95 %.

## Setup

- **Reference:** the H1 retargets of LAFAN1 in the
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (`h1/`), 40 clips
  at 30 fps, played at 50 Hz. The selected configurations train on 34 of them: all but the six `fallAndGetUp` clips
  (131 minutes of motion; the clips are 2.2 to 4.6 minutes long). The data are downloaded by `tools/download_h1.sh` and
  are not in this repository.
- **Training:** `ndm_gym.train_humanoid`, 4,096 worlds, 8,000 iterations with the learning rate's decay over the run, a
  checkpoint every 500 iterations (`--keep_ckpts`), on an RTX 6000 Ada (48 GB). The budget is fixed and the same for
  both algorithms; "Checkpoints" below gives the return over the run.
- **Evaluation:** `ndm_gym.evaluate_humanoid` on the last checkpoint: 64 rollouts per clip of the whole clip from
  perturbed starts (evaluation seed 0), without exploration noise and without resets. A rollout succeeds if it never
  terminates: the termination is the library G1's (the pelvis 0.25 m off the reference's height, or tilted by 0.8 from
  it). The library's G1 benchmark plays each clip once from its unperturbed first frame; the success rate here is over
  perturbed starts. The return counts nothing after a termination; errors are means over the frames before one. Clips
  with a failed rollout: the clips in which any of the 64 rollouts terminates. Pushes: the same rollouts with a
  horizontal velocity of 1 or 2 m/s added to the pelvis every 2 s in a random direction (no pushes in training).
- **Training time:** the wall time of the 8,000 iterations, from runs that train only (no evaluation), each alone on
  its GPU. The clock starts at the first training iteration, after the simulator's setup and the warm-up collection.
  Not engineered: this path has no per-phase time table, and the times are an RTX 6000 Ada's, not an A100's or H100's.
- Mean ± sample SD over three training seeds (0, 1, 2).

Configurations (`ndm_gym/configs.py`):

- **Both:** the Go2 task's selected options with the episodes of the library's G1 training (`--max_episode_steps 500
  --bin_starts --fail_reset 0.3`), the position targets' offsets bounded at 4 times the action scale in place of 1
  (`--action_bound 4`) and position targets outside the joints' ranges accepted (`--target_range free`), as the library
  G1's policy, which is unbounded. One shared option differs in its effect: the start sampling ranks its bins by
  returns discounted with the trainer's `--gamma` (as the library's G1 training), 0.99 with SuperTrack and 0.97 with
  SuperMimic.
- **SuperTrack** (`--config h1-supertrack`): the Go2's SuperTrack options; its tuning started from `--horizon 32`
  (the library G1's SuperTrack's imagined window; the Go2's is 16) and `--pi_batch 2048` (the Go2's is 4096) and
  selected five changes: `--horizon 48`, `--lr 8e-4` (from 4e-4), `--st_fail_w 1` (2), `--st_imag_noise 0.4` (0.2)
  and `--action_noise 0.2` (0.1).
- **SuperMimic** (`--config h1-supermimic`): the Go2's SuperMimic options with the one change its tuning round
  selected: `--gamma 0.97` (0.95).
- **Reference's targets, open loop:** the same rollouts with the reference's joint positions as the position targets
  and no policy.

```bash
benchmarks/h1_lafan1_tracking/run.sh selected <seed> logs/h1/h1-supertrack --config h1-supertrack   # the 34 clips
benchmarks/h1_lafan1_tracking/run.sh train <seed> logs/h1/h1-supermimic --config h1-supermimic      # subjects 1 to 4
benchmarks/h1_lafan1_tracking/run.sh all 0 logs/h1/h1-supermimic --config h1-supermimic             # all 40 clips
python -m ndm_gym.train_humanoid --config h1-supertrack --seed <seed> --log_dir logs/h1/time/h1-supertrack/seed_<seed>   # training only: the time
python benchmarks/h1_lafan1_tracking/summarize.py SuperTrack=logs/h1/h1-supertrack SuperMimic=logs/h1/h1-supermimic \
    --each selected --kinds selected --tipped selected --heldout --curve selected \
    --time SuperTrack=logs/h1/time/h1-supertrack SuperMimic=logs/h1/time/h1-supermimic
```

## 34 clips

|  | seeds | return | success | clips with a failed rollout | MPJPE (world) | MPJPE (pelvis) | pelvis | joints | success, push 1 / 2 m/s |
|---|---|---|---|---|---|---|---|---|---|
| reference's targets, open loop | | 0.004 | 0.000 | 34.0 | 177 mm | 90.2 mm | 195 mm | 3.36° | |
| SuperTrack | 0, 1, 2 | 0.786 ± 0.011 | 0.820 ± 0.037 | 9.0 ± 2.6 | 136 ± 8 mm | 22.7 ± 0.7 mm | 130 ± 8 mm | 2.38 ± 0.04° | 0.562 ± 0.035 / 0.096 ± 0.021 |
| SuperMimic | 0, 1, 2 | 0.827 ± 0.008 | 0.771 ± 0.016 | 12.3 ± 0.6 | 266 ± 16 mm | 30.7 ± 0.6 mm | 260 ± 16 mm | 4.01 ± 0.09° | 0.236 ± 0.086 / 0.008 ± 0.005 |

Every seed:

|  | seed | return | success | clips with a failed rollout | clips with no successful rollout |
|---|---|---|---|---|---|
| SuperTrack | 0 | 0.774 | 0.778 | 11 | `dance1_subject1`, `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject3`, `walk3_subject4` |
| SuperTrack | 1 | 0.789 | 0.839 | 10 | `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk3_subject3`, `walk3_subject4` |
| SuperTrack | 2 | 0.796 | 0.844 | 6 | `dance2_subject4`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject4` |
| SuperMimic | 0 | 0.831 | 0.783 | 13 | `dance1_subject1`, `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject4` |
| SuperMimic | 1 | 0.832 | 0.778 | 12 | `dance1_subject1`, `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject4` |
| SuperMimic | 2 | 0.818 | 0.753 | 12 | `dance1_subject1`, `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject3`, `walk3_subject4` |

Training time (training only, each run alone on an RTX 6000 Ada):

|  | seeds | training | iterations/s |
|---|---|---|---|
| SuperTrack | 0, 1, 2 | 14.5 ± 0.1 min | 9.04 ± 0.05 |
| SuperMimic | 0, 1, 2 | 12.0 ± 0.0 min | 10.97 ± 0.00 |

By the kind of motion:

| kind | clips | SuperTrack: return | SuperMimic: return | SuperTrack: success | SuperMimic: success |
|---|---|---|---|---|---|
| `dance` | 8 | 0.731 ± 0.015 | 0.776 ± 0.000 | 0.833 ± 0.072 | 0.742 ± 0.005 |
| `fight` | 3 | 0.751 ± 0.011 | 0.817 ± 0.006 | 0.698 ± 0.063 | 0.653 ± 0.024 |
| `fightAndSports` | 2 | 0.799 ± 0.009 | 0.767 ± 0.027 | 0.997 ± 0.005 | 0.818 ± 0.083 |
| `jumps` | 3 | 0.761 ± 0.032 | 0.831 ± 0.010 | 0.641 ± 0.045 | 0.667 ± 0.000 |
| `run` | 4 | 0.787 ± 0.001 | 0.854 ± 0.004 | 1.000 ± 0.000 | 0.993 ± 0.011 |
| `sprint` | 2 | 0.768 ± 0.009 | 0.748 ± 0.131 | 0.966 ± 0.052 | 0.865 ± 0.215 |
| `walk` | 12 | 0.839 ± 0.009 | 0.877 ± 0.006 | 0.774 ± 0.031 | 0.749 ± 0.032 |

### Checkpoints

Every kept checkpoint, 4 rollouts per clip, mean and sample SD over the three seeds:

| iteration | SuperTrack: return | SuperMimic: return | SuperTrack: success | SuperMimic: success |
|---|---|---|---|---|
| 500 | 0.022 ± 0.026 | 0.004 ± 0.000 | 0.010 ± 0.017 | 0.000 ± 0.000 |
| 1000 | 0.228 ± 0.241 | 0.368 ± 0.032 | 0.262 ± 0.236 | 0.199 ± 0.060 |
| 1500 | 0.512 ± 0.103 | 0.694 ± 0.010 | 0.559 ± 0.045 | 0.654 ± 0.019 |
| 2000 | 0.625 ± 0.018 | 0.734 ± 0.011 | 0.647 ± 0.007 | 0.684 ± 0.019 |
| 2500 | 0.694 ± 0.012 | 0.761 ± 0.009 | 0.703 ± 0.037 | 0.699 ± 0.013 |
| 3000 | 0.692 ± 0.042 | 0.780 ± 0.021 | 0.701 ± 0.036 | 0.721 ± 0.019 |
| 3500 | 0.703 ± 0.012 | 0.787 ± 0.013 | 0.699 ± 0.019 | 0.706 ± 0.041 |
| 4000 | 0.736 ± 0.008 | 0.812 ± 0.005 | 0.723 ± 0.015 | 0.743 ± 0.013 |
| 4500 | 0.733 ± 0.006 | 0.807 ± 0.015 | 0.725 ± 0.034 | 0.735 ± 0.022 |
| 5000 | 0.750 ± 0.018 | 0.815 ± 0.003 | 0.752 ± 0.011 | 0.735 ± 0.007 |
| 5500 | 0.756 ± 0.012 | 0.815 ± 0.010 | 0.748 ± 0.015 | 0.740 ± 0.008 |
| 6000 | 0.761 ± 0.030 | 0.814 ± 0.016 | 0.762 ± 0.052 | 0.733 ± 0.026 |
| 6500 | 0.773 ± 0.017 | 0.821 ± 0.014 | 0.782 ± 0.040 | 0.757 ± 0.022 |
| 7000 | 0.779 ± 0.010 | 0.816 ± 0.008 | 0.797 ± 0.030 | 0.748 ± 0.004 |
| 7500 | 0.784 ± 0.015 | 0.819 ± 0.007 | 0.814 ± 0.054 | 0.750 ± 0.007 |
| 8000 | 0.785 ± 0.013 | 0.827 ± 0.009 | 0.814 ± 0.030 | 0.772 ± 0.013 |

## Clips not seen in training

LAFAN1's split: one policy trains on the clips of subjects 1 to 4 (28 of the 34) and is evaluated on subject 5's
(`dance2_subject5`, `fight1_subject5`, `jumps1_subject5`, `run1_subject5`, `walk1_subject5`, `walk3_subject5`), 64
rollouts per clip. Beside it, the same six clips by the policies that trained on all 34.

|  | clips | trained on | SuperTrack: return | SuperMimic: return | SuperTrack: success | SuperMimic: success |
|---|---|---|---|---|---|---|
| subject 5's | 6 | subjects 1 to 4 (not these clips) | 0.724 ± 0.022 | 0.511 ± 0.046 | 0.839 ± 0.124 | 0.350 ± 0.099 |
| subject 5's | 6 | all five subjects | 0.815 ± 0.016 | 0.883 ± 0.003 | 0.987 ± 0.023 | 0.993 ± 0.012 |
| subjects 1 to 4's | 28 | subjects 1 to 4 | 0.811 ± 0.003 | 0.810 ± 0.009 | 0.851 ± 0.022 | 0.722 ± 0.027 |

## All 40 clips

Seed 0, the selected configurations with `--clips '*'`, 64 rollouts per clip.

|  | seeds | return | success | clips with a failed rollout | MPJPE (world) | MPJPE (pelvis) | pelvis | joints | success, push 1 / 2 m/s |
|---|---|---|---|---|---|---|---|---|---|
| reference's targets, open loop | | 0.004 | 0.000 | 40.0 | 177 mm | 90.6 mm | 195 mm | 3.36° | |
| SuperTrack | 0 | 0.695 | 0.676 | 16.0 | 175 mm | 26.6 mm | 169 mm | 2.55° | 0.519 / 0.122 |
| SuperMimic | 0 | 0.695 | 0.625 | 20.0 | 410 mm | 33.7 mm | 404 mm | 4.35° | 0.298 / 0.041 |

| kind | clips | SuperTrack: return | SuperMimic: return | SuperTrack: success | SuperMimic: success |
|---|---|---|---|---|---|
| `dance` | 8 | 0.711 | 0.757 | 0.748 | 0.738 |
| `fallAndGetUp` | 6 | 0.223 | 0.095 | 0.148 | 0.000 |
| `fight` | 3 | 0.743 | 0.796 | 0.708 | 0.661 |
| `fightAndSports` | 2 | 0.789 | 0.620 | 1.000 | 0.664 |
| `jumps` | 3 | 0.782 | 0.788 | 0.667 | 0.599 |
| `run` | 4 | 0.783 | 0.838 | 1.000 | 1.000 |
| `sprint` | 2 | 0.766 | 0.817 | 1.000 | 1.000 |
| `walk` | 12 | 0.831 | 0.850 | 0.669 | 0.665 |

SuperTrack / SuperMimic complete 15 % / 0 % of the rollouts of the six `fallAndGetUp` clips (`fallAndGetUp3_subject1`:
69 % / 0 %, `fallAndGetUp1_subject5`: 20 % / 0 %, the other four clips: 0 % / 0 %). On the other 34 clips
these policies reach a return of 0.779 / 0.801 and a success of 0.769 / 0.735; seed 0 trained on the 34 clips reaches
0.774 / 0.831 and 0.778 / 0.783.

The rule that leaves the six clips out of the selected configurations was fixed after earlier seed 0 runs on all 40
clips, with the starting configurations, in which neither algorithm completed a rollout of them ("Beside the starting
point"). Other humanoid tracking work removes motions its robot cannot perform: H2O (He et al., 2024) trains a
privileged policy on its motions and removes those it fails; GMT (Chen et al., 2025) filters motions by the root's roll,
pitch and height (crawling, fallen states), then by a first policy's failures.

## The clips that fail

Looked at after the runs, with no change to the 34 clips: in 7 of them the reference's pelvis tips past the horizontal
in some frame (the z component of its z axis below 0: cartwheels, rolls, lying on the floor), as it does in all six
`fallAndGetUp` clips and in no other clip.

| clips |  | SuperTrack: return | SuperMimic: return | SuperTrack: success | SuperMimic: success |
|---|---|---|---|---|---|
| 7 | the pelvis tips past the horizontal: `dance1_subject1`, `dance2_subject4`, `fight1_subject3`, `jumps1_subject2`, `walk2_subject3`, `walk3_subject1`, `walk3_subject4` | 0.659 ± 0.023 | 0.676 ± 0.006 | 0.275 ± 0.132 | 0.097 ± 0.012 |
| 27 | the others | 0.819 ± 0.009 | 0.866 ± 0.010 | 0.962 ± 0.015 | 0.946 ± 0.017 |

## Findings

SuperTrack / SuperMimic, the selected configurations, seeds 0-2 (mean ± sample SD), an RTX 6000 Ada.

- **Tracking, 34 clips:** return 0.786 ± 0.011 / 0.827 ± 0.008, success 0.820 ± 0.037 / 0.771 ± 0.016, with 9.0 ± 2.6 /
  12.3 ± 0.6 clips of 34 in which a rollout fails. Before a termination the tracked bodies are within 22.7 / 30.7 mm of
  the reference's about the pelvis and the joints within 2.38 / 4.01°; the pelvis is 130 / 260 mm from the reference's.
  The reference's own joint positions as the targets, open loop, complete no rollout.
- **Where the rollouts end:** in the 7 clips whose reference tips past the horizontal the success is 0.275 ± 0.132 /
  0.097 ± 0.012; in the other 27 clips it is 0.962 ± 0.015 / 0.946 ± 0.017, with a return of 0.819 / 0.866. By kind, the
  run clips are completed at 1.000 / 0.993 and the sprints at 0.966 / 0.865; the jumps (0.641 / 0.667) and the fights
  (0.698 / 0.653) each have one such clip among their three.
- **Clips not seen in training:** on subject 5's six clips the policies trained on subjects 1 to 4 reach a return of
  0.724 ± 0.022 / 0.511 ± 0.046 and a success of 0.839 ± 0.124 / 0.350 ± 0.099; the policies trained on all five
  subjects reach 0.815 / 0.883 and 0.987 / 0.993 on the same clips.
- **All 40 clips, seed 0:** return 0.695 / 0.695, success 0.676 / 0.625; on the six `fallAndGetUp` clips the success is
  0.148 / 0.000.
- **Pushes,** none in training: at 1 m/s every 2 s the success is 0.562 ± 0.035 / 0.236 ± 0.086, at 2 m/s 0.096 ± 0.021
  / 0.008 ± 0.005.
- **Training time:** the 8,000 iterations take 14.5 ± 0.1 / 12.0 ± 0.0 min (9.04 / 10.97 iterations/s; the imagined
  window is 48 / 16 steps). Half the budget gives a return of 0.736 / 0.812.
- **The budget:** over the last 1,000 iterations, as the learning rate decays to its end, the return rises by 0.006 /
  0.011 and the success by 0.017 / 0.024: the numbers are the fixed budget's, not a plateau's. With twice the budget
  (16,000 iterations and the decay over them, seed 0) the return is 0.830 / 0.854 and the success 0.909 / 0.818,
  against 0.774 / 0.831 and 0.778 / 0.783 for seed 0 at 8,000; over the last 2,000 of the 16,000 iterations the return
  changes by +0.001 / -0.006 (4 rollouts per clip).
- **Tuning:** each algorithm's round of eleven runs on seed 0 selected five changes for SuperTrack and one for
  SuperMimic ("Beside the starting point"). On seeds 0-2 SuperTrack's starting configuration reaches a return of
  0.721 ± 0.002 and a success of 0.722 ± 0.002; SuperMimic's starting configuration ran on seed 0 only (4 rollouts per
  clip: return 0.780, success 0.757, against 0.820 and 0.757 with its change).

## Beside the starting point

All runs of this section: seed 0, an RTX 6000 Ada, the last checkpoint with 4 rollouts per clip unless stated;
SuperTrack / SuperMimic.

**From the Go2's options to the selected ones** (all 40 clips):

| configuration | iterations | return | success |
|---|---|---|---|
| the Go2's selected options (SuperTrack with `--horizon 32 --pi_batch 2048`): actions bounded at 1, episodes to the clip's end from uniform starts, the termination also at 1 m from the reference; 256 rollouts per clip | 10,000 | 0.199 / 0.178 | 0.074 / 0.067 |
| the library G1's episodes and termination, `--action_bound 4` | 8,000 | 0.623 / 0.651 | 0.631 / 0.625 |
| the same with `--target_range free` | 8,000 | 0.615 / 0.652 | 0.606 / 0.619 |
| the same, trained and evaluated on the 34 clips | 8,000 | 0.712 / 0.780 | 0.721 / 0.757 |
| the selected configurations (each with its round's changes, below), 34 clips | 8,000 | 0.804 / 0.820 | 0.831 / 0.757 |

- The options of the second row were identified with `--robot g1` on this path, against the library's G1 training; with
  the bound of 1 an offset's corrective torque is at most a quarter of the motor's effort. Those runs are not reported
  here.
- `--target_range free` changes the H1's numbers by less than the difference between two seeds; it is selected because
  the library G1's servos accept such targets.
- In every run on all 40 clips of this section (both tables) no rollout of a `fallAndGetUp` clip completes, for either
  algorithm; the rule that leaves these clips out was fixed after them. The policies of the third row, evaluated on the
  34 other clips: 0.714 / 0.756 (success 0.71 / 0.73), against 0.712 / 0.780 trained on the 34 only.

**Changes of the robot's model and of the episodes, none selected** (all 40 clips, 8,000 iterations; against the second
row above, 0.623 / 0.651 and 0.631 / 0.625, or the third where stated):

| change | return | success |
|---|---|---|
| `--foot menagerie`: MuJoCo Menagerie's foot capsules | 0.632 / 0.619 | 0.637 / 0.481 |
| `--fail_reset 0`: no starts before failures | 0.633 / 0.646 | 0.619 / 0.619 |
| `--kd_scale 3`: three times the servos' damping | 0.604 / 0.637 | 0.600 / 0.625 |
| `--kd_scale 8` | 0.535 / 0.449 | 0.588 / 0.306 |
| `--target_range effort` | 0.632 / 0.660 | 0.644 / 0.631 |
| `--gains tracking` (with `--target_range free`): the gains of HumanoidVerse and ProtoMotions | 0.610 / 0.646 | 0.619 / 0.606 |
| `--ref_ground clip` (with `--target_range free`), scored against the retarget's own heights | 0.632 / 0.661 | 0.631 / 0.625 |
| `--ref_ground stance` (with `--target_range free`), scored against the retarget's own heights | 0.606 / 0.671 | 0.588 / 0.650 |
| `--ankle_effort_scale 2`: twice the ankle motors' 40 N m, for diagnosis, not the robot's | 0.644 / 0.655 | 0.637 / 0.619 |
| `--ankle_effort_scale 2 --target_range effort`, for diagnosis | 0.655 / 0.661 | 0.637 / 0.550 |

**Each algorithm's own options: one round of eleven runs each** (34 clips, 8,000 iterations, 4 rollouts per clip). The
rule, fixed before each round: a change is adopted if it raises the return by at least 0.02 over the starting
configuration's (the fourth row above) and does not lower the success by more than 0.02; with several such changes,
one run of all of them, adopted if its return is at least the best single change's.

| SuperTrack: one change | return | success | adopted |
|---|---|---|---|
| the starting configuration (`--horizon 32 --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --action_noise 0.1 --pi_batch 2048`) | 0.712 | 0.721 | |
| `--horizon 16` | 0.280 | 0.176 | |
| `--horizon 48` | 0.750 | 0.765 | yes |
| `--st_fail_w 1` | 0.745 | 0.735 | yes |
| `--st_fail_w 5` | 0.692 | 0.684 | |
| `--st_imag_noise 0.1` | 0.717 | 0.743 | |
| `--st_imag_noise 0.4` | 0.736 | 0.750 | yes |
| `--action_noise 0.05` | 0.727 | 0.750 | |
| `--action_noise 0.2` | 0.735 | 0.750 | yes |
| `--lr 2e-4` | 0.672 | 0.676 | |
| `--lr 8e-4` | 0.761 | 0.779 | yes |
| `--pi_batch 4096` | 0.729 | 0.743 | |
| the five adopted changes together | 0.804 | 0.831 | selected |

| SuperMimic: one change | return | success | adopted |
|---|---|---|---|
| the starting configuration (`--horizon 16 --gamma 0.95 --bptt_decay 0.95 --action_noise 0.3 --lr 4e-4 --pi_batch 4096`) | 0.780 | 0.757 | |
| `--horizon 8` | 0.737 | 0.699 | |
| `--horizon 24` | 0.778 | 0.757 | |
| `--gamma 0.97` | 0.820 | 0.757 | yes, selected |
| `--gamma 0.99` | 0.777 | 0.779 | |
| `--bptt_decay 0.9` | 0.728 | 0.706 | |
| `--bptt_decay 1.0` | 0.831 | 0.735 | no: the success is lower by 0.022 |
| `--action_noise 0.15` | 0.721 | 0.603 | |
| `--action_noise 0.5` | 0.775 | 0.765 | |
| `--lr 2e-4` | 0.712 | 0.640 | |
| `--lr 8e-4` | 0.779 | 0.713 | |
| `--pi_batch 2048` | 0.780 | 0.765 | |

- SuperMimic's round ran before the three-seed runs. SuperTrack's ran after three seeds of its starting configuration,
  which are listed here: return 0.721 ± 0.002 (0.723, 0.720, 0.719), success 0.722 ± 0.002
  (0.724, 0.723, 0.720), 17.7 ± 1.2 clips with a failed rollout, MPJPE about the pelvis 24.2 ± 0.5 mm (64 rollouts per
  clip), 11.4 min of training.
- Two runs of one configuration and seed differ by up to 0.005 in the return and 0.04 in the success at 4 rollouts per
  clip (SuperTrack's starting configuration: 0.712 and 0.717, 0.721 and 0.684); the changes adopted with a return
  between 0.735 and 0.750 are within a few such differences of the rule's bound.

## Notes

- **Repeatability:** the simulator on the GPU is not bit-reproducible. Two evaluations of one checkpoint with the same
  evaluation seed differ by up to 0.004 in the return and 0.011 in the success (SuperMimic's seven runs, 64 rollouts
  per clip, evaluated twice); the 4-rollout and the 64-rollout evaluation of a last checkpoint differ by up to 0.006 and
  0.013 (the six runs of "34 clips").
- **GPUs:** every run used one RTX 6000 Ada. Some of the runs behind the tracking numbers shared their GPU with other
  jobs for part of their training or evaluation, which changes how long they took and not what they computed. The
  training times are from separate runs that train only, each alone on its GPU from start to end.
- **Not measured:** the times on an A100- or H100-class GPU; a time to a target (no target was fixed before the runs);
  more than three seeds, and more than seed 0 at 16,000 iterations; a budget beyond 16,000 iterations; domain
  randomization, observation noise and anything on a robot. `--robot g1` on this path served as a check against the
  library's G1 training and has no results here.
- **The 34 clips:** the rule is by clip name and was fixed before the three-seed runs. The split by the reference's
  pelvis ("The clips that fail") was made after them and changes no number of the other sections.
- **Seed 0** is the tuning rounds' seed and one of the three reported seeds; the rounds used 4 rollouts per clip.
- **The 1 m/s and 2 m/s pushes** are in the same directions for both algorithms and all seeds (push seed 0).
