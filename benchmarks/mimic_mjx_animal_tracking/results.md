# Animal motion tracking: results

Contributed by [Jingdong Zhang](https://github.com/Evergreen0929) ([#8](https://github.com/Frank-ZY-Dou/ndm-gym/pull/8)).

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make MIMIC-MJX's stick insect
track ten clips of its recorded kinematics on MuJoCo-Warp ([the task](../../docs/tasks/animals.md), [the rules](README.md)).
The rat and the fruit fly run with the same code; their results are in progress.

## Setup

- **Reference:** clips 0-9 of the stick insect's file (MIMIC-MJX at revision `b4e53a7`), 225 frames each at 50 Hz (4.5 s),
  resampled to the 10 ms control step, with the velocities of the clips' own positions (`--ref_qvel positions`: at this
  revision the file's are not theirs). Held out: clips 10-19, never trained on.
- **Configurations** (`ndm_gym/configs.py`), each chosen by the same rules for both algorithms ([below](#configurations)):
  - `stick-supertrack`: `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights --action_noise 0.3
    --bptt_decay 0.95 --pi_rollout_clip 10 --wm_pos_lead 0.8 --target_rate_w 0.5`;
  - `stick-supermimic`: `--algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.95 --horizon 32 --bptt_decay 0.9
    --pi_rollout_clip 10 --wm_batches 8 --target_rate_w 0.5`;
  - both: the hand task's optimizer and systems options (Muon, the policy average, the learning-rate ramp, CUDA-graphed
    collection and updates), `--pi_batch 2048 --wm_lr 2e-3 --fused_fk`, 2,048 worlds, a cap of 8,000 iterations with the
    learning rate's decay over it, a checkpoint kept every 250; training seeds 13, 14, 15.
- **Training:** one run at a time on one A100 40GB SXM4; each kept checkpoint was evaluated for the stop rule as it
  appeared, on the same GPU, and the training ended at the stop. Not timed: other work shared the machine.
- **Stop rule, the same for both algorithms** (`watch.py`): each kept checkpoint on the training clips, each from its first
  frame and from 9 perturbed starts (evaluation seed 1). With S(i) and E(i) the means over the evaluations at i, i - 250 and
  i - 500 of the share of episodes ended and of the bodies' error, a run stops at the first i of at least 2,000 where
  E(i) > 0.97 E(i - 1,000) and either S(i) <= 0.01 or S(i) > S(i - 1,000) - 0.01; otherwise at 8,000. The checkpoint at the
  stop, not the best one, is the run's result.
- **Evaluation** (`final.py`): the stop checkpoint plays the training clips and the held-out clips, each from its first
  frame and from 19 perturbed starts (0.02 rad on the joints, evaluation seed 0: other starts than the stop rule's), 200
  episodes per set, without exploration noise. Ended: the share of episodes ended by MIMIC-MJX's terminations before their
  clip's end. Errors are the completed episodes' means over their steps: the tracked bodies' positions (mm), the six claws'
  (the end effectors), the root's, and the joints' RMS angle (deg). The body is 17.6 mm long (the bodies' median largest
  distance from the root).
- **Motion check** (`motion.py`, the same for both algorithms): on the recorded rollouts of the ten unperturbed starts of the
  training clips, against the reference's same frames: the joints' jerk at most 1.5 times the reference's, their share of
  velocity power above 27 Hz (power over power) at most 2 times, and the steps at a joint limit at most the reference's. 27 Hz
  is the lowest whole frequency above which the reference's joint velocity holds at most 5 % of its power.
- Mean ± sample SD over the three training seeds.

```bash
for algo in supertrack supermimic; do for seed in 13 14 15; do
  benchmarks/mimic_mjx_animal_tracking/run.sh stick $algo $seed logs/animals
  python benchmarks/mimic_mjx_animal_tracking/motion.py logs/animals/stick_${algo}_s$seed
done; done
python benchmarks/mimic_mjx_animal_tracking/summarize.py logs/animals --seeds 13 14 15
```

## Results

| | stick insect, SuperTrack / SuperMimic |
|---|---|
| episodes ended ↓ | 0 / 0 % |
| bodies ↓ | 0.2564 ± 0.0014 / 0.2831 ± 0.0011 mm |
| end effectors ↓ | 0.491 ± 0.002 / 0.490 ± 0.005 mm |
| root ↓ | 0.084 ± 0.001 / 0.141 ± 0.001 mm |
| joints ↓ | 7.98 ± 0.05 / 4.84 ± 0.04° |
| held-out clips: episodes ended ↓ | 0.0 ± 0.0 / 3.3 ± 3.3 % |
| held-out clips: bodies ↓ | 0.552 ± 0.009 / 0.644 ± 0.029 mm |
| held-out clips: joints ↓ | 10.28 ± 0.08 / 6.56 ± 0.16° |
| motion: jerk, the reference's = 1 | 1.25 / 0.99 |
| motion: share above 27 Hz, the reference's = 1 | 1.13 / 0.79 |
| motion: steps at a joint limit (the reference's 25.8 %) | 18.8 / 16.9 % |
| stop (iteration) | 3,333 ± 144 / 3,417 ± 144 |

SuperTrack's configuration has the lower bodies' and root errors on the training clips and the held-out ones, and ends no
held-out episode; SuperMimic's has the lower joint-angle error, and its end effectors' error is level.

### Every seed

| algorithm | seed | stop | ended | bodies | end effectors | joints | held-out: ended | held-out: bodies | jerk | share above 27 Hz | at a limit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SuperTrack | 13 | 3,250 | 0 % | 0.2568 mm | 0.493 mm | 7.95° | 0 % | 0.557 mm | 1.23 | 1.10 | 18.9 % |
| | 14 | 3,500 | 0 % | 0.2549 mm | 0.489 mm | 7.94° | 0 % | 0.556 mm | 1.25 | 1.14 | 18.7 % |
| | 15 | 3,250 | 0 % | 0.2577 mm | 0.492 mm | 8.04° | 0 % | 0.542 mm | 1.26 | 1.14 | 18.9 % |
| SuperMimic | 13 | 3,500 | 0 % | 0.2843 mm | 0.496 mm | 4.87° | 0.5 % | 0.658 mm | 0.97 | 0.75 | 17.0 % |
| | 14 | 3,500 | 0 % | 0.2824 mm | 0.487 mm | 4.80° | 2.5 % | 0.664 mm | 0.99 | 0.81 | 17.1 % |
| | 15 | 3,250 | 0 % | 0.2825 mm | 0.487 mm | 4.85° | 7.0 % | 0.611 mm | 1.02 | 0.81 | 16.5 % |

Jerk and share: the policy's over the reference's same frames. Every run passes the motion check.

## Smooth motion

A policy's action sets the stick insect's 41 torque actuators each control step (10 ms). The task's only cost on how the
action changes from one step to the next was small: 0.05 times the action's mean squared change, in SuperMimic's reward and,
with `--st_rate_w 0.05`, in SuperTrack's loss. At that weight the torques could alternate from step to step at little cost,
and the joints' motion showed it: their jerk was 3.06 and 1.59 times the reference's (SuperTrack / SuperMimic) and their
share of velocity power above 27 Hz 4.37 and 1.97 times. The smoothness term (`--target_rate_w 0.5`, in both
configurations) charges each step's change of the policy's part of each actuator's command, over its width (half the
actuator's control range), squared, in the place of the action-rate cost: a jump costs, following the reference does not. On
this body, whose actuators are all torques, it is the same cost at ten times the weight. With it the jerk is 1.25 and 0.99
times the reference's and the share 1.13 and 0.79 times, and the tracking is unchanged: the bodies' error is 0.99 and 1.01
times the runs' without it.

The same configurations without the term, on the same seeds (SuperTrack with `--st_rate_w 0.05` in its place):

| | weight | bodies | held-out: ended | held-out: bodies | jerk | share above 27 Hz | at a limit | at a limit the reference is not at | motion check |
|---|---|---|---|---|---|---|---|---|---|
| SuperTrack | 0.5 | 0.2564 ± 0.0014 mm | 0.0 ± 0.0 % | 0.552 ± 0.009 mm | 1.25 | 1.13 | 18.8 % | 1.89 % | 3 of 3 runs |
| | 0 | 0.2598 ± 0.0020 mm | 1.0 ± 1.3 % | 0.540 ± 0.006 mm | 3.06 | 4.37 | 18.6 % | 1.90 % | 0 of 3 runs |
| SuperMimic | 0.5 | 0.2831 ± 0.0011 mm | 3.3 ± 3.3 % | 0.644 ± 0.029 mm | 0.99 | 0.79 | 16.9 % | 0.88 % | 3 of 3 runs |
| | 0 | 0.2792 ± 0.0024 mm | 1.8 ± 1.3 % | 0.654 ± 0.031 mm | 1.59 | 1.97 | 16.8 % | 0.88 % | 0 of 3 runs |

No episode on the training clips ended in any of these runs. Without the term, every seed's jerk: SuperTrack 3.03 / 3.12 /
3.03, SuperMimic 1.60 / 1.54 / 1.61 times the reference's; its share above 27 Hz: SuperTrack 4.27 / 4.43 / 4.40, SuperMimic
2.12 / 1.87 / 1.91 times (seeds 13 / 14 / 15). The reference's share above 27 Hz is 4.7 %.

## Configurations

Each algorithm's configuration came from the task's presets (the hand task's fast configurations, with `--ref_qvel
positions`) through screens of the same kinds of change for both algorithms, on training seeds 5-12, never the results':

- **Training options,** in three rounds, each on the previous round's pick:
  1. one change at a time: exploration noise, the learning rate, and one option of each algorithm's own (SuperTrack's
     failure weight, SuperMimic's discount);
  2. the decay of the policy's gradient through the imagined states at its two other values of 1, 0.95 and 0.9, a cap on
     each rollout's part of the policy gradient, a cap on each imagined state's gradient, the horizon doubled, two of these
     together, and one more option of each algorithm's own (SuperTrack: a cost on the action's change; SuperMimic: reward
     kernels 0.7 times as wide);
  3. the world model's training: a 16-step window, 8 updates per iteration, and SuperTrack's acceleration model's position
     lead at 0.8 (SuperMimic's increment model has none: the window and the updates together in its place).

  Each round's rule, the same for both: the lowest share of the selection episodes ended, ties within one point broken by
  the lower bodies' error, on one to four seeds. SuperTrack's picks: action noise 0.3, `--bptt_decay 0.95`,
  `--pi_rollout_clip 10` with `--st_rate_w 0.05`, and the position lead 0.8; SuperMimic's: discount 0.95, horizon 32,
  `--bptt_decay 0.9` with `--pi_rollout_clip 10`, and 8 world-model updates per iteration.
- **The smoothness term:** weights 0, 0.5, 1, 2, 4, 8 and 16, the same grid for both, at seeds 5-8. The rule: the lowest
  weight at which every run passes the motion check and the accuracy holds (the bodies' error at most 1.10 times weight 0's,
  the ended share at most one point above it). Both algorithms: 0.5, confirmed at seeds 9-12.

## The contributed runs

The contributor's runs ([Jingdong Zhang](https://github.com/Evergreen0929)): seeds 0-2, each run alone on an A100 80GB
SXM4. They differ from the runs above in four ways:

- the configurations before the screens, the hand task's fast configurations with `--ref_qvel positions`: SuperTrack at the
  trainer's action noise of 0.1, without the decay, the cap and the action-rate cost, its position lead the constant
  acceleration's 0.6; SuperMimic at discount 0.99, horizon 16 and `--bptt_decay 0.95`, without the cap, with 4 world-model
  updates per iteration;
- no smoothness term on either algorithm;
- a fixed budget of 4,000 iterations, the last checkpoint evaluated;
- an evaluation of each clip from its first frame and four perturbed starts (50 episodes per set), errors over the steps the
  episodes lived.

| | stick insect, SuperTrack / SuperMimic |
|---|---|
| episodes ended | 0 / 0 % |
| bodies | 0.35 ± 0.01 / 0.33 ± 0.00 mm |
| end effectors | 0.60 ± 0.02 / 0.56 ± 0.00 mm |
| joints | 8.4 ± 0.2 / 4.2 ± 0.1° |
| held-out clips: ended | 10.7 ± 1.2 / 0.0 ± 0.0 % |
| held-out clips: bodies | 0.64 ± 0.01 / 0.74 ± 0.03 mm |

Every run's metrics, time to the target and training summary, as JSON records: [`a100_runs/`](a100_runs/README.md).

## Renders

`tools/render_animal.py` renders a rollout recorded by `final.py` (`evaluate_animal --record`), the reference laid over it
as a translucent ghost, in MIMIC-MJX's arena. `docs/media/animals/` holds, per algorithm, the run with the lowest bodies'
error (seed 14 for both; every run passes the motion check), clip 3 from its first frame (`--world 3 --distance 2`), and the
grid of the two (`tools/make_grid.py`). Clip 3's own episode passes the check for both: jerk 1.18 and 0.93 times the
reference's, the share above 27 Hz 1.01 and 0.74 times. The teaser's tile is SuperTrack's (`--distance 2.5`, so that the
insect stays inside the tile's 4:3 frame).
