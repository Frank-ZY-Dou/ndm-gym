# Hand-object tracking: results

Contributed by [Dechen Gao](https://github.com/gaodechen): the task with SuperMimic on it
([#2](https://github.com/Frank-ZY-Dou/ndm-gym/pull/2)), the Allegro and the Inspire hand ([#6](https://github.com/Frank-ZY-Dou/ndm-gym/pull/6)).

SuperTrack and SuperMimic, with MLP world models, track human grasp-and-lift clips with a floating right XHand
([the task](../../docs/tasks/hand.md)); both also with the Allegro hand and the Inspire hand ([more hands](#more-hands)).
The README's XHand rows train with smooth position targets ([the target-rate penalty](#the-xhand-with-the-target-rate-penalty)).

## Setup

- **Reference:** the kinematic retarget of the captured human hand and of the object's pose
  ([SPIDER](https://huggingface.co/datasets/retarget/retarget_full)'s `trajectory_kinematic.npz`, at the revision in
  `tasks.json`); no physics-optimized trajectory is used.
- **Training:** `ndm_gym.train_hand`, 20,000 iterations in 2,048 worlds, 16 imagined steps, on one A100 40GB, with the
  systems options (`--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs`).
- **Evaluation:** `ndm_gym.evaluate_hand` on the last checkpoint: 256 rollouts of the whole clip from perturbed starts
  (evaluation seed 0 for every run, whatever its training seed), without exploration noise. A rollout succeeds if it never
  terminates (the object or the palm
  15 cm from the reference) and the object ends within 5 cm of the reference. Errors are means over the clip's frames
  and the rollouts; mean ± SD over three training seeds. Reference targets: the same rollouts with the reference's joint
  positions as the position targets (the values of SuperTrack's seed-0 evaluation).
- **Data:** SPIDER's clips are downloaded by `tools/download_hand_clips.py` and are not in this repository.

| task | clip | frames (50 Hz) | object lift |
|---|---|---|---|
| `apple` | HRDexDB `apple-0` | 102 | 26.2 cm |
| `oakink_a01002` | OakInk `oakink_0000_right_A01002_00` (a spray bottle) | 86 | 24.3 cm |
| `can` | DexYCB `dexycb_01_right_002_master_chef_can_20200709_141754_00` | 106 | 32.7 cm |

```bash
python tools/download_hand_clips.py                    # the clips of tasks.json into data/spider
SYSTEMS="--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs"
# train, evaluate and render a task and seed: SuperTrack, then SuperMimic's defaults
benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/xhand_st --algo supertrack --lr 4e-4 --st_fail_w 2 \
    --st_imag_noise 0.2 --st_task_weights $SYSTEMS
benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/xhand_sm $SYSTEMS
```

## Results

| task | | seeds | success | object position | object position, last frame | fingertips | object lift |
|---|---|---|---|---|---|---|---|
| `apple` | reference targets | | 100 % | 20.0 mm | 15.3 mm | 26.6 mm | 24.8 cm |
| | SuperTrack | 0, 1, 2 | 100.0 ± 0.0 % | 2.4 ± 0.0 mm | 0.2 ± 0.0 mm | 4.8 ± 0.1 mm | 26.2 ± 0.0 cm |
| | SuperTrack, object terms at half weight | 0, 1, 2 | 100.0 ± 0.0 % | 2.9 ± 0.1 mm | 0.3 ± 0.1 mm | 3.9 ± 0.1 mm | 26.2 ± 0.0 cm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 100.0 ± 0.0 % | 3.3 ± 0.1 mm | 0.7 ± 0.2 mm | 6.0 ± 0.1 mm | 26.2 ± 0.0 cm |
| | SuperMimic, defaults | 0, 1, 2 | 100.0 ± 0.0 % | 3.5 ± 0.1 mm | 1.1 ± 1.0 mm | 6.8 ± 1.7 mm | 26.2 ± 0.1 cm |
| `oakink_a01002` | reference targets | | 0 % | 198.3 mm | 356.5 mm | 29.7 mm | -8.4 cm |
| | SuperTrack | 0, 1, 2 | 100.0 ± 0.0 % | 1.6 ± 0.1 mm | 0.8 ± 0.0 mm | 5.8 ± 0.1 mm | 24.3 ± 0.0 cm |
| | SuperTrack, object terms at half weight | 0, 1, 2 | 99.7 ± 0.2 % | 2.9 ± 0.5 mm | 3.2 ± 0.8 mm | 4.6 ± 0.1 mm | 24.2 ± 0.1 cm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 100.0 ± 0.0 % | 2.7 ± 0.1 mm | 2.1 ± 1.1 mm | 6.1 ± 0.2 mm | 24.3 ± 0.1 cm |
| | SuperMimic, defaults | 0, 1, 2 | 100.0 ± 0.0 % | 2.9 ± 0.4 mm | 3.1 ± 0.9 mm | 6.6 ± 0.7 mm | 24.1 ± 0.2 cm |
| `can` | reference targets | | 0 % | 113.3 mm | 335.6 mm | 32.6 mm | -0.5 cm |
| | SuperTrack | 0, 1, 2 | 99.9 ± 0.2 % | 2.3 ± 0.3 mm | 1.0 ± 0.7 mm | 6.4 ± 0.2 mm | 32.7 ± 0.1 cm |
| | SuperTrack, object terms at half weight | 0, 1, 2 | 100.0 ± 0.0 % | 2.8 ± 0.1 mm | 1.1 ± 0.1 mm | 5.5 ± 0.2 mm | 32.7 ± 0.0 cm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 99.9 ± 0.2 % | 4.3 ± 0.4 mm | 3.7 ± 1.6 mm | 7.6 ± 0.2 mm | 32.7 ± 0.1 cm |
| | SuperMimic, defaults | 0, 1, 2 | 100.0 ± 0.0 % | 3.2 ± 0.1 mm | 1.8 ± 0.0 mm | 6.6 ± 0.6 mm | 32.7 ± 0.1 cm |

| configuration | options beside the systems options |
|---|---|
| SuperTrack | `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights` |
| SuperTrack, object terms at half weight | the same with `--st_obj_w 0.5` |
| SuperMimic, HDMI's settings at γ 0.99 | `--lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95` |
| SuperMimic, defaults | none |

## The XHand with the target-rate penalty

A policy's action is an offset about the next reference pose. Nothing in SuperTrack's loss, and only a weak term in
SuperMimic's reward (0.05 times the actions' mean squared change), tied one control step's offset to the next, so the
fingers' position targets could jump back and forth at every step. The target-rate penalty (`--target_rate_w`) is a cost per
step on the change of the policy's offsets, each joint's over its tracking kernel's width (the palm's translation over 40 mm,
its rotation and the fingers over 0.4 rad), squared, averaged over the joints: in SuperMimic's reward with a minus sign, in
SuperTrack's loss as it is. A jump costs; following the reference does not. SuperTrack's loss sums its tracking terms over
the imagined window with weights divided by the window's length (`--st_task_weights`, in its presets); with `--st_cost_mean`
it takes the cost's mean over the window too, so that the cost weighs against those terms as in the reward.

**Configurations** (each with `--target_rate_w 4`; the systems options of the presets):

| goal | SuperTrack | SuperMimic |
|---|---|---|
| most accurate | `hand-supertrack-fast --st_cost_mean --iterations 20000` | `hand-supermimic-fast --action_noise 0.2 --iterations 20000` |
| time | `hand-supertrack-fast --st_cost_mean`, the speed options with `--st_no_imag_r` | `hand-supermimic-fast --wm_batches 2`, the speed options with `--sm_reuse_input` |

The speed options: `--compile_updates --elementwise_3x3 --combo_kernels --wm_palm_fk --compile_collect --autotune_gemm`.

**Motion against the reference** (the most accurate configurations; nine runs a row on A100 40GB, training seeds 0-2 by the
three clips; the unperturbed rollout at each run's stop checkpoint; in brackets, the same configuration without the penalty:
its motion from training seed 0, its error and success from its nine runs; `tools/hand_motion_check.py`, finger joints, the
share above 5 Hz per joint then the mean over the joints):

| | share of joint-velocity power above 5 Hz | jerk (rad/s³) | target change per step (rad) | times the kinematic reference (share / jerk / change) | object error | success |
|---|---|---|---|---|---|---|
| SuperTrack | 0.032 (0.350) | 183 (1,687) | 0.0242 (0.184) | 1.336 / 0.426 / 1.817 | 2.27 mm (2.01) | 100.0 % (100.0) |
| SuperMimic | 0.014 (0.373) | 119 (1,282) | 0.0201 (0.128) | 0.589 / 0.277 / 1.505 | 3.20 mm (3.21) | 100.0 % (100.0) |

A row passes the motion check if its finger joints' share of velocity power above 5 Hz, their jerk and the targets' change
per step are at most 2, 1.5 and 2 times its clips' kinematic references' (the mean over its runs against the mean over their
references). Every row of this section passes.

**What the penalty leaves.**

- *Slower swings, for both algorithms.* Every run's fingers reverse direction (by more than 0.02 rad) more often than the
  reference's: 8-24 more reversals over the 12 finger joints for SuperTrack, 9-36 for SuperMimic, on 5-10 of the joints. A
  swing of 0.1 rad at 2 Hz moves the target by at most 0.025 rad a step, about the reference's own step (0.012-0.016 rad), so
  the penalty charges it little. The chatter it removed moved the target by 0.13-0.18 rad a step.
- *The palm on `apple`'s lift.* The reference lifts the apple at up to 0.75 m/s. The palm's position actuators follow a moving
  target about 0.1 s behind, and the policy's offset of the palm is at most 30 mm. So the palm ends the lift 45 mm below the
  reference with both algorithms, every seed, with the penalty and without; the fingers keep the apple within 18.4 mm of its
  path. The slower lifts of `oakink_a01002` and `can` (up to 0.32 and 0.34 m/s) leave 13-17 mm.

**Most accurate** (mean ± SD over training seeds 0-2). Each run stops once converged: at the first checkpoint from 6,000
iterations on (every 1,000) with a success of at least 0.99 and an object error less than 3 % better than 2,000 iterations
earlier, both read on 64 perturbed starts of another evaluation seed (6); at most 20,000 iterations. That checkpoint is the
run's result, evaluated as in [Setup](#setup).

| task | | stop (seeds 0 / 1 / 2) | success | object position | object position, last frame | fingertips | object lift |
|---|---|---|---|---|---|---|---|
| `apple` | SuperTrack | 6,000 / 6,000 / 7,000 | 100.0 ± 0.0 % | 2.9 ± 0.0 mm | 0.4 ± 0.0 mm | 5.2 ± 0.4 mm | 26.2 ± 0.0 cm |
| | SuperMimic | 7,000 / 8,000 / 6,000 | 100.0 ± 0.0 % | 3.6 ± 0.2 mm | 1.3 ± 0.3 mm | 6.3 ± 0.6 mm | 26.3 ± 0.0 cm |
| `oakink_a01002` | SuperTrack | 6,000 / 6,000 / 6,000 | 100.0 ± 0.0 % | 1.6 ± 0.0 mm | 0.8 ± 0.0 mm | 6.0 ± 0.1 mm | 24.3 ± 0.0 cm |
| | SuperMimic | 6,000 / 9,000 / 11,000 | 100.0 ± 0.0 % | 2.2 ± 0.1 mm | 1.8 ± 0.3 mm | 5.8 ± 0.1 mm | 24.3 ± 0.0 cm |
| `can` | SuperTrack | 6,000 / 6,000 / 6,000 | 100.0 ± 0.0 % | 2.3 ± 0.1 mm | 0.8 ± 0.2 mm | 7.7 ± 1.0 mm | 32.7 ± 0.0 cm |
| | SuperMimic | 6,000 / 9,000 / 7,000 | 100.0 ± 0.0 % | 3.8 ± 0.2 mm | 1.8 ± 0.2 mm | 7.2 ± 0.4 mm | 32.7 ± 0.1 cm |

**Time to the target.** Minutes: the training wall clock at the first checkpoint (every 200 iterations) from which every
later one has a success of 0.99 or more with the object within 10 mm; the iteration there in brackets. 6,000 iterations a
run, after an untimed warm-up of each configuration on each clip, so every compiled kernel was in the cache. One GPU: one
run per A100 40GB, four at a time on four GPUs of a node, its other GPUs idle. Three GPUs: one run at a time on three A100
40GB of a node, its other GPUs idle. Object: position error at 6,000 iterations.

| GPUs | | `apple`, seeds 0 / 1 / 2 | `oakink_a01002` | `can` | mean of nine | it/s | object |
|---|---|---|---|---|---|---|---|
| one | SuperTrack | 0.68 (600), 0.68 (600), 0.68 (600) | 0.85 (800), 0.85 (800), 0.67 (600) | 1.39 (1,400), 1.22 (1,200), 1.39 (1,400) | 0.93 min | 18.4 | 2.51 mm |
| | SuperMimic | 0.81 (800), 0.63 (600), 0.81 (800) | 0.78 (800), 0.94 (1,000), 0.77 (800) | 0.94 (1,000), 1.11 (1,200), 0.94 (1,000) | 0.86 min | 19.5 | 3.46 mm |
| three | SuperTrack | 0.51 (600), 0.51 (600), 0.51 (600) | 0.51 (600), 0.60 (800), 0.60 (800) | 0.77 (1,200), 0.77 (1,200), 0.77 (1,200) | 0.62 min | 39.4 | 2.23 mm |
| | SuperMimic | 0.61 (800), 0.50 (600), 0.73 (1,000) | 0.72 (1,000), 0.61 (800), 0.72 (1,000) | 0.72 (1,000), 0.72 (1,000), 0.84 (1,200) | 0.69 min | 30.4 | 3.37 mm |

- One A100: SuperMimic's mean time is lower by 0.07 min (0.86 against 0.93); SuperTrack has the lower object error (2.51
  against 3.46 mm).
- Three A100s (`--layout AWL`): SuperTrack's mean time is lower by 0.07 min (0.62 against 0.69), and so is its object error
  (2.23 against 3.37 mm).
- Every timed configuration passes the motion check at 6,000 iterations (share / jerk / change, times the references'): one
  A100, SuperTrack 1.241 / 0.430 / 1.836 and SuperMimic 0.516 / 0.300 / 1.589; three A100s, 1.285 / 0.440 / 1.861 and 0.549
  / 0.305 / 1.605.
- On three A100s each algorithm also timed another configuration with the penalty: SuperTrack with `--action_noise 0.2`
  2.15 min (4 of 9 runs at the target; none on `oakink_a01002`), SuperMimic without `--wm_batches 2` 0.71 min (9 of 9).

**Per phase** of the timed paths (ms per iteration over iterations 1,001-3,000 of a 3,000-iteration run, `apple`, training
seed 0, on the timed runs' node; every device synchronized at each phase's boundary, so on three GPUs, where the phases
otherwise run at once, the total is not the layout's running speed):

| GPUs | | collection | world model | policy | exchange | total | iterations/s |
|---|---|---|---|---|---|---|---|
| one | SuperTrack | 5.63 | 24.89 | 24.35 | | 55.79 | 17.9 |
| | SuperMimic | 7.57 | 12.71 | 32.64 | | 53.76 | 18.6 |
| three | SuperTrack | 7.26 | 25.03 | 24.06 | 1.52 | 58.73 | 17.0 |
| | SuperMimic | 9.24 | 15.92 | 32.51 | 1.43 | 60.18 | 16.6 |

## More hands

The Allegro hand (16 finger joints, four fingers) and the Inspire hand (12 finger joints) track the same three clips,
from SPIDER's retargets of the same demonstrations to them. A hand is read from its scene (`ndm_gym/robots/hands.py`: the
actuated joints in the actuators' order, the palm's site and the fingertip sites the hand has), so the task, the world
models, the policies and the options are the XHand's. The two hands and this reading of a scene were contributed in
[pull request 6](https://github.com/Frank-ZY-Dou/ndm-gym/pull/6); on the XHand's three scenes the code before and after
it gives the same indices, kinematics, task outputs and training updates bit for bit on the CPU.

- **Training:** both algorithms with the XHand's named configurations, `hand-supertrack-fast` and `hand-supermimic-fast`
  (6,000 iterations, with HOI's optimizer options) and `hand-supertrack-accurate` and `hand-supermimic-accurate` (20,000
  iterations); 2,048 worlds, training seeds 0-2, on A100 40GB GPUs. Then, per hand and algorithm, four single changes
  of its fast configuration on training seed 0, the same kinds for both, and three-seed runs of the changes chosen by the
  same rules ([below](#how-the-two-hands-configurations-were-chosen)): timed at 6,000 iterations, and trained to 20,000
  (untimed).
- **Evaluation:** as above, on the last checkpoint. The 6,000-iteration runs trained alone on a GPU; their checkpoints,
  every 200 iterations, were evaluated after training, as in [training time](#training-time).

Each algorithm's configurations of record, on `apple` / `oakink_a01002` / `can` (mean of training seeds 0-2; most
accurate: object position at 20,000 iterations; fastest: minutes to a success of 0.99 with the object within 10 mm, over
the seeds that reach it, with the [speed options](#training-time-with-the-speed-options) on one A100 40GB and on three
with [`--layout AWL`](#training-time-on-three-gpus)):

| hand | | SuperTrack | SuperMimic |
|---|---|---|---|
| Inspire | most accurate | `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000`: object 2.4 / 1.3 / 2.0 mm at a success of 99.7 / 100.0 / 99.9 % | `hand-supermimic-accurate`: 5.2 / 2.5 / 11.2 mm at 99.9 / 100.0 / 93.8 % |
| | fastest, one GPU | `hand-supertrack-fast`: 1.0 / 0.8 / 2.0 min, 9 of its 9 runs at the target, mean 1.30 min | `hand-supermimic-fast --lr 2e-4`: 1.0 / 0.8 / 4.0 min, 8 of 9, mean 2.19 min |
| | fastest, three GPUs | `hand-supertrack-fast` with `--layout AWL`: 0.6 / 0.6 / 0.9 min, 9 of its 9 runs at the target, mean 0.70 min | `hand-supermimic-fast --lr 2e-4` with `--layout AWL`: 1.0 / 0.6 / 1.0 min, 7 of 9, mean 1.46 min |
| Allegro | most accurate | `hand-supertrack-fast --st_fail_w 5 --iterations 20000`: object 2.6 / 1.3 / 2.0 mm at a success of 100.0 / 100.0 / 99.7 % | `hand-supermimic-fast --action_noise 0.2 --iterations 20000`: 3.7 / 11.7 / 3.4 mm at 100.0 / 82.7 / 99.6 % |
| | fastest, one GPU | `hand-supertrack-fast --st_fail_w 5`: 1.1 / 0.8 / 3.9 min, 9 of its 9 runs at the target, mean 1.96 min | `hand-supermimic-fast`: 1.0 / 3.7 / 1.1 min, 9 of 9, mean 1.96 min |
| | fastest, three GPUs | `hand-supertrack-fast --st_fail_w 5` with `--layout AWL`: 0.7 / 0.7 / 2.7 min, 8 of its 9 runs at the target, mean 1.40 min | `hand-supermimic-fast` with `--layout AWL`: 0.7 / 0.9 / 0.8 min, 7 of 9, mean 1.44 min |

```bash
python tools/download_hand_clips.py --robots allegro inspire
ROBOT=allegro benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/allegro_st --config hand-supertrack-fast --st_fail_w 5 --iterations 20000
ROBOT=allegro benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/allegro_sm --config hand-supermimic-fast --action_noise 0.2 --iterations 20000
ROBOT=inspire benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/inspire_st --config hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000
ROBOT=inspire benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/inspire_sm --config hand-supermimic-accurate
```

### Accuracy, 20,000 iterations

The last checkpoint over 256 perturbed starts, mean ± SD over training seeds 0-2. Per hand and algorithm two
configurations trained for 20,000 iterations: its accurate preset and the arm its screen chose (its fast preset or one of
its four changes). The one with the
lower object error over the three clips is its most accurate configuration: Allegro, SuperTrack 2.24 mm (preset) and
1.97 mm, SuperMimic 7.89 and 6.28 mm; Inspire, SuperTrack 2.56 and 1.92 mm, SuperMimic 6.31 and 46.88 mm. Trained on to
20,000 iterations, SuperMimic's fast configuration loses the Inspire hand's can in all three seeds (a success of 2.0,
43.4 and 0.0 %; 100.0, 81.2 and 98.8 % at 6,000 iterations), its return in the simulator falling through training (seed
0: 0.90 at iteration 3,000, 0.78 at 20,000).

**The Allegro hand**

| task | | seeds | success | object position | object position, last frame | fingertips | object lift |
|---|---|---|---|---|---|---|---|
| `apple` | reference targets | | 0 % | 98.0 mm | 268.0 mm | 20.2 mm | 0.0 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 2.4 ± 0.1 mm | 0.3 ± 0.1 mm | 6.2 ± 0.0 mm | 26.2 ± 0.0 cm |
| | SuperTrack, `hand-supertrack-fast --st_fail_w 5 --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 2.6 ± 0.1 mm | 0.3 ± 0.1 mm | 6.1 ± 0.2 mm | 26.2 ± 0.0 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 97.7 ± 4.1 % | 6.0 ± 3.9 mm | 8.3 ± 12.4 mm | 9.8 ± 1.7 mm | 25.7 ± 1.0 cm |
| | SuperMimic, `hand-supermimic-fast --action_noise 0.2 --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 3.7 ± 0.2 mm | 2.0 ± 0.6 mm | 7.6 ± 0.3 mm | 26.3 ± 0.1 cm |
| `oakink_a01002` | reference targets | | 0 % | 134.5 mm | 279.2 mm | 19.6 mm | -2.7 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 1.4 ± 0.0 mm | 0.4 ± 0.1 mm | 7.4 ± 0.2 mm | 24.3 ± 0.0 cm |
| | SuperTrack, `hand-supertrack-fast --st_fail_w 5 --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 1.3 ± 0.0 mm | 0.3 ± 0.0 mm | 7.2 ± 0.1 mm | 24.3 ± 0.0 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 84.5 ± 17.4 % | 13.5 ± 12.4 mm | 33.7 ± 35.9 mm | 13.1 ± 6.4 mm | 21.7 ± 3.4 cm |
| | SuperMimic, `hand-supermimic-fast --action_noise 0.2 --iterations 20000` | 0, 1, 2 | 82.7 ± 16.2 % | 11.7 ± 5.1 mm | 29.6 ± 18.9 mm | 12.0 ± 0.3 mm | 22.2 ± 1.7 cm |
| `can` | reference targets | | 0 % | 175.3 mm | 341.4 mm | 21.5 mm | -1.0 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 99.5 ± 0.2 % | 2.8 ± 0.6 mm | 2.2 ± 0.8 mm | 7.9 ± 0.6 mm | 32.5 ± 0.1 cm |
| | SuperTrack, `hand-supertrack-fast --st_fail_w 5 --iterations 20000` | 0, 1, 2 | 99.7 ± 0.2 % | 2.0 ± 0.7 mm | 1.3 ± 0.9 mm | 7.5 ± 0.2 mm | 32.6 ± 0.1 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 99.6 ± 0.0 % | 4.1 ± 0.2 mm | 4.5 ± 0.5 mm | 7.9 ± 0.1 mm | 32.5 ± 0.0 cm |
| | SuperMimic, `hand-supermimic-fast --action_noise 0.2 --iterations 20000` | 0, 1, 2 | 99.6 ± 0.0 % | 3.4 ± 0.2 mm | 3.5 ± 0.5 mm | 7.2 ± 0.4 mm | 32.6 ± 0.1 cm |

**The Inspire hand**

| task | | seeds | success | object position | object position, last frame | fingertips | object lift |
|---|---|---|---|---|---|---|---|
| `apple` | reference targets | | 0 % | 97.1 mm | 264.3 mm | 13.9 mm | -0.1 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 2.8 ± 0.0 mm | 1.7 ± 0.1 mm | 8.2 ± 0.6 mm | 26.1 ± 0.0 cm |
| | SuperTrack, `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000` | 0, 1, 2 | 99.7 ± 0.2 % | 2.4 ± 0.1 mm | 1.5 ± 0.5 mm | 7.9 ± 0.2 mm | 26.1 ± 0.1 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 99.9 ± 0.2 % | 5.2 ± 0.7 mm | 5.4 ± 2.4 mm | 10.2 ± 1.7 mm | 25.8 ± 0.2 cm |
| | SuperMimic, `hand-supermimic-fast --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 4.5 ± 0.1 mm | 4.7 ± 1.1 mm | 8.0 ± 0.3 mm | 25.8 ± 0.1 cm |
| `oakink_a01002` | reference targets | | 99 % | 29.9 mm | 31.2 mm | 17.2 mm | 21.5 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 1.7 ± 0.0 mm | 1.0 ± 0.1 mm | 5.2 ± 0.1 mm | 24.3 ± 0.0 cm |
| | SuperTrack, `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 1.3 ± 0.0 mm | 0.7 ± 0.0 mm | 5.6 ± 0.1 mm | 24.2 ± 0.0 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 2.5 ± 0.1 mm | 1.8 ± 0.5 mm | 5.0 ± 0.1 mm | 24.3 ± 0.0 cm |
| | SuperMimic, `hand-supermimic-fast --iterations 20000` | 0, 1, 2 | 100.0 ± 0.0 % | 2.5 ± 0.1 mm | 1.8 ± 0.3 mm | 4.8 ± 0.1 mm | 24.3 ± 0.0 cm |
| `can` | reference targets | | 0 % | 161.4 mm | 334.7 mm | 18.2 mm | -0.7 cm |
| | SuperTrack, `hand-supertrack-accurate` | 0, 1, 2 | 99.7 ± 0.2 % | 3.1 ± 0.6 mm | 2.1 ± 0.9 mm | 5.7 ± 0.7 mm | 32.5 ± 0.1 cm |
| | SuperTrack, `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000` | 0, 1, 2 | 99.9 ± 0.2 % | 2.0 ± 0.5 mm | 1.2 ± 0.9 mm | 4.1 ± 0.1 mm | 32.6 ± 0.1 cm |
| | SuperMimic, `hand-supermimic-accurate` | 0, 1, 2 | 93.8 ± 10.8 % | 11.2 ± 13.7 mm | 25.7 ± 40.9 mm | 6.5 ± 1.4 mm | 30.4 ± 3.9 cm |
| | SuperMimic, `hand-supermimic-fast --iterations 20000` | 0, 1, 2 | 15.1 ± 24.5 % | 133.7 ± 34.9 mm | 324.6 ± 91.8 mm | 22.4 ± 6.9 mm | 2.9 ± 8.6 cm |

### Time to the target, 6,000 iterations

These times are before the speed options; with them, see [below](#training-time-with-the-speed-options).

Each run trained alone on a GPU. Minutes: the training wall clock at the first of the run's checkpoints (every 200
iterations) from which every later one has a success of 0.99 or more with the object within 10 mm (-: none within 6,000
iterations); iterations/s are with policy updates (the log's rows 1,000-3,000). Per hand and algorithm two
configurations: its fast preset and its change with the fewest mean iterations to the target among those reaching it on
all three clips (a tie: the lower object error at 6,000). The one with the
lower mean time over its nine runs (a run without the target counted at its whole training time) is its fast
configuration: Allegro, SuperTrack 4.21 min (preset) and 3.90 min, SuperMimic 5.31 and 5.32 min; Inspire, SuperTrack
2.34 and 3.26 min, SuperMimic 4.13 and 3.81 min.

**The Allegro hand**

| task | | seeds | minutes to a success of 0.99 with the object within 10 mm ↓ | iterations/s ↑ | success at 6,000 | object position at 6,000 ↓ | fingertips at 6,000 ↓ |
|---|---|---|---|---|---|---|---|
| `apple` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 1.9, 3.2, 1.3 | 10.6 | 100.0 ± 0.0 % | 2.6 ± 0.1 mm | 6.3 ± 0.2 mm |
|  | SuperTrack, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 2.2, 1.3, 1.3 | 10.6 | 100.0 ± 0.0 % | 2.6 ± 0.0 mm | 6.1 ± 0.1 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | 1.5, 1.2, 1.5 | 8.7 | 100.0 ± 0.0 % | 3.8 ± 0.1 mm | 7.9 ± 0.2 mm |
|  | SuperMimic, `hand-supermimic-fast --gamma 0.995` | 0, 1, 2 | 1.5, 1.5, 1.9 | 8.7 | 100.0 ± 0.0 % | 4.0 ± 0.1 mm | 8.7 ± 0.4 mm |
| `oakink_a01002` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 2.7, 1.6, 1.3 | 10.9 | 100.0 ± 0.0 % | 1.4 ± 0.1 mm | 8.0 ± 0.4 mm |
|  | SuperTrack, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 1.6, 1.3, 1.0 | 10.9 | 100.0 ± 0.0 % | 1.4 ± 0.1 mm | 7.6 ± 0.3 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | -, -, - | 8.9 | 91.4 ± 4.7 % | 8.0 ± 2.0 mm | 8.6 ± 1.1 mm |
|  | SuperMimic, `hand-supermimic-fast --gamma 0.995` | 0, 1, 2 | 11.3, -, 10.1 | 8.9 | 98.4 ± 2.4 % | 5.1 ± 2.1 mm | 8.6 ± 1.0 mm |
| `can` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 10.0, 6.8, 9.0 | 10.9 | 99.6 ± 0.0 % | 2.7 ± 0.1 mm | 7.8 ± 0.1 mm |
|  | SuperTrack, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 9.0, 8.1, 9.3 | 10.9 | 99.5 ± 0.2 % | 3.2 ± 0.3 mm | 7.6 ± 0.3 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | 3.9, 2.3, 1.9 | 8.9 | 99.6 ± 0.0 % | 3.3 ± 0.0 mm | 7.6 ± 0.2 mm |
|  | SuperMimic, `hand-supermimic-fast --gamma 0.995` | 0, 1, 2 | 5.7, 2.3, 2.3 | 8.9 | 99.6 ± 0.0 % | 4.5 ± 0.6 mm | 8.7 ± 0.7 mm |

**The Inspire hand**

| task | | seeds | minutes to a success of 0.99 with the object within 10 mm ↓ | iterations/s ↑ | success at 6,000 | object position at 6,000 ↓ | fingertips at 6,000 ↓ |
|---|---|---|---|---|---|---|---|
| `apple` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 1.5, 0.9, 4.8 | 11.2 | 99.9 ± 0.2 % | 3.0 ± 0.2 mm | 8.2 ± 0.4 mm |
|  | SuperTrack, `hand-supertrack-fast --lr 8e-4` | 0, 1, 2 | 6.3, 2.4, 1.8 | 11.3 | 100.0 ± 0.0 % | 2.9 ± 0.0 mm | 8.3 ± 0.1 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | 3.2, 1.8, 1.8 | 9.2 | 100.0 ± 0.0 % | 4.7 ± 0.2 mm | 8.4 ± 0.9 mm |
|  | SuperMimic, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 1.4, 1.8, 4.3 | 9.2 | 100.0 ± 0.0 % | 4.4 ± 0.3 mm | 8.7 ± 0.8 mm |
| `oakink_a01002` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 2.1, 1.2, 1.2 | 11.7 | 100.0 ± 0.0 % | 1.7 ± 0.0 mm | 5.1 ± 0.0 mm |
|  | SuperTrack, `hand-supertrack-fast --lr 8e-4` | 0, 1, 2 | 1.2, 0.9, 0.9 | 11.6 | 100.0 ± 0.0 % | 1.6 ± 0.0 mm | 5.3 ± 0.0 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | 3.7, 1.0, 1.0 | 9.4 | 100.0 ± 0.0 % | 2.6 ± 0.1 mm | 5.0 ± 0.3 mm |
|  | SuperMimic, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 1.4, 3.5, 1.4 | 9.4 | 100.0 ± 0.0 % | 2.9 ± 0.3 mm | 5.4 ± 0.5 mm |
| `can` | SuperTrack, `hand-supertrack-fast` | 0, 1, 2 | 2.7, 5.5, 1.2 | 11.6 | 100.0 ± 0.0 % | 2.4 ± 0.2 mm | 4.6 ± 0.1 mm |
|  | SuperTrack, `hand-supertrack-fast --lr 8e-4` | 0, 1, 2 | 1.5, 6.9, 7.5 | 11.6 | 99.9 ± 0.2 % | 3.1 ± 0.5 mm | 5.4 ± 0.7 mm |
|  | SuperMimic, `hand-supermimic-fast` | 0, 1, 2 | 3.3, -, - | 9.4 | 93.4 ± 10.5 % | 14.3 ± 16.0 mm | 7.3 ± 1.0 mm |
|  | SuperMimic, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 7.4, -, 2.5 | 9.4 | 97.0 ± 5.2 % | 8.5 ± 6.7 mm | 8.2 ± 1.2 mm |

### The README's animations

Per hand, algorithm and clip the README's grid shows one run, from the unperturbed start.

- *The XHand* (smooth targets, [above](#the-xhand-with-the-target-rate-penalty)): of the three seeds of each algorithm's most
  accurate configuration on the clip, the run at a success of 0.99 or more whose fingers reverse direction least often against
  the reference (a tie broken by the lower power of the joints' velocity between 1 and 5 Hz). Each is within the motion check's
  bounds on its own rollout (share / jerk / change, times the reference's; `apple`, `oakink_a01002`, `can`): SuperTrack 1.10 /
  0.49 / 1.97, 1.68 / 0.38 / 1.74 and 1.25 / 0.38 / 1.63; SuperMimic 0.37 / 0.33 / 1.56, 0.44 / 0.18 / 1.25 and 0.82 / 0.34 /
  1.61.
- *The others:* the run with the lowest object error over the perturbed starts at its last checkpoint among that algorithm's
  configurations of record on the hand.

Configuration and training seed:

| hand | | `apple` | `oakink_a01002` | `can` |
|---|---|---|---|---|
| XHand | SuperTrack | `hand-supertrack-fast --target_rate_w 4 --st_cost_mean --iterations 20000`, 2 | the same, 1 | the same, 2 |
| | SuperMimic | `hand-supermimic-fast --action_noise 0.2 --target_rate_w 4 --iterations 20000`, 2 | the same, 1 | the same, 2 |
| Allegro | SuperTrack | `hand-supertrack-fast --st_fail_w 5 --iterations 20000`, 2 | the same, 1 | the same, 0 |
| | SuperMimic | `hand-supermimic-fast --action_noise 0.2 --iterations 20000`, 0 | `hand-supermimic-fast`, 2 | as on `apple`, 2 |
| Inspire | SuperTrack | `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000`, 1 | the same, 1 | the same, 2 |
| | SuperMimic | `hand-supermimic-fast --lr 2e-4`, 0 | `hand-supermimic-accurate`, 0 | `hand-supermimic-accurate`, 0 |

### Pull request 6's table, and its configuration here

The pull request reported SuperMimic with `hand-supermimic-fast` from runs with two speed changes that are not part of
its code (the palm's kinematics alone in the world model's rollout, and an imagined state's observation and goal
computed once). Here its code trained as it is, on the same GPU type, beside SuperTrack's fast configuration (6,000
iterations, the last checkpoint, training seeds 0-2):

| hand | task | | seeds | success % | object mm | object, last frame mm | fingertips mm | lift cm |
|---|---|---|---|---|---|---|---|---|
| Allegro | `apple` | SuperTrack here | 0, 1, 2 | 100.0 ± 0.0 | 2.6 ± 0.1 | 0.3 ± 0.1 | 6.3 ± 0.2 | 26.2 ± 0.0 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 100.0 ± 0.0 | 3.8 ± 0.1 | 2.1 ± 0.6 | 7.9 ± 0.2 | 26.3 ± 0.1 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 100.0 ± 0.0 | 3.8 ± 0.2 | 2.1 ± 0.7 | 8.1 ± 0.6 | 26.3 ± 0.2 |
| Allegro | `oakink_a01002` | SuperTrack here | 0, 1, 2 | 100.0 ± 0.0 | 1.4 ± 0.1 | 0.4 ± 0.0 | 8.0 ± 0.4 | 24.3 ± 0.0 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 91.4 ± 4.7 | 8.0 ± 2.0 | 18.2 ± 7.8 | 8.6 ± 1.1 | 23.0 ± 0.6 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 98.8 ± 2.0 | 4.1 ± 0.7 | 5.5 ± 2.3 | 7.6 ± 0.6 | 24.1 ± 0.2 |
| Allegro | `can` | SuperTrack here | 0, 1, 2 | 99.6 ± 0.0 | 2.7 ± 0.1 | 1.9 ± 0.0 | 7.8 ± 0.1 | 32.6 ± 0.0 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 99.6 ± 0.0 | 3.3 ± 0.0 | 3.8 ± 0.3 | 7.6 ± 0.2 | 32.6 ± 0.0 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 99.6 ± 0.0 | 3.5 ± 0.6 | 3.6 ± 0.7 | 7.5 ± 0.1 | 32.6 ± 0.1 |
| Inspire | `apple` | SuperTrack here | 0, 1, 2 | 99.9 ± 0.2 | 3.0 ± 0.2 | 2.1 ± 0.6 | 8.2 ± 0.4 | 26.1 ± 0.1 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 100.0 ± 0.0 | 4.7 ± 0.2 | 5.0 ± 1.3 | 8.4 ± 0.9 | 25.8 ± 0.1 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 100.0 ± 0.0 | 4.6 ± 0.3 | 5.3 ± 0.6 | 8.2 ± 0.4 | 25.8 ± 0.0 |
| Inspire | `oakink_a01002` | SuperTrack here | 0, 1, 2 | 100.0 ± 0.0 | 1.7 ± 0.0 | 1.1 ± 0.1 | 5.1 ± 0.0 | 24.3 ± 0.0 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 100.0 ± 0.0 | 2.6 ± 0.1 | 2.0 ± 0.1 | 5.0 ± 0.3 | 24.2 ± 0.0 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 100.0 ± 0.0 | 2.9 ± 0.5 | 2.5 ± 0.9 | 5.2 ± 0.3 | 24.3 ± 0.0 |
| Inspire | `can` | SuperTrack here | 0, 1, 2 | 100.0 ± 0.0 | 2.4 ± 0.2 | 1.0 ± 0.0 | 4.6 ± 0.1 | 32.7 ± 0.0 |
|  |  | SuperMimic here, the pull request's code | 0, 1, 2 | 93.4 ± 10.5 | 14.3 ± 16.0 | 27.3 ± 38.5 | 7.3 ± 1.0 | 30.4 ± 3.7 |
|  |  | SuperMimic, the pull request's table | 0, 1, 2 | 97.7 ± 3.7 | 7.8 ± 6.3 | 12.0 ± 13.1 | 6.2 ± 0.9 | 31.9 ± 1.3 |

Four of the six pairs of hand and clip read as the pull request's table, the object error within 0.3 mm of it. On the
Allegro hand's `oakink_a01002` the seeds read lower here: success / object position 88.7 / 88.7 / 96.9 % and 7.9 / 10.1
/ 6.0 mm (the pull request: 100.0 / 96.5 / 100.0 % and 4.2 / 4.7 / 3.3 mm). On the Inspire hand's `can` one seed of
three loses the object in part of the rollouts in both: 100.0 / 81.2 / 98.8 % and 3.5 / 32.8 / 6.8 mm here (the pull
request: 93.4 / 100.0 / 99.6 % and 15.1 / 3.7 / 4.7 mm). On the GPU a run does not repeat itself bit for bit (two runs
of the same code and seed differ from the first iterations on), so a seed's outcome is not reproduced exactly.

### How the two hands' configurations were chosen

Per hand and algorithm, four single changes of the fast configuration, the same kinds for both (training seed 0, 6,000
iterations, the three clips): the learning rate at 2e-4 and 8e-4 (4e-4 in both presets), the exploration (SuperTrack's
imagined noise 0.1 for 0.2, SuperMimic's action noise 0.2 for 0.3) and the failure's weight or the return's horizon
(SuperTrack `--st_fail_w 5` for 2, SuperMimic `--gamma 0.995` for 0.99). Iterations to the target are the first
checkpoint from which every later one has a success of 0.99 or more with the object within 10 mm (-: none by 6,000).
The screen's runs trained on an A100 40GB, but for the Inspire hand's four SuperTrack arms and SuperMimic's `--lr 2e-4`
arm, which trained on an RTX 2080 Ti (fp32, as all): a development reading of iterations and errors, not of time; every
configuration the rules chose then trained on A100 40GB with three seeds.

| hand | | arm | iterations to the target, apple / spray bottle / can ↓ | object position at 6,000, mean of the clips ↓ |
|---|---|---|---|---|
| Allegro | SuperTrack | `hand-supertrack-fast` | 1,200 / 1,000 / 5,800 | 2.26 mm |
| | | `--lr 2e-4` | 1,000 / 800 / 5,800 | (can 98.8 %) |
| | | `--lr 8e-4` | 1,200 / 800 / 3,200 | (can 98.8 %) |
| | | `--st_imag_noise 0.1` | 1,400 / 1,000 / 3,200 | (can 98.4 %) |
| | | `--st_fail_w 5` | 800 / 1,000 / 3,400 | 2.00 mm |
| | SuperMimic | `hand-supermimic-fast` | 800 / - / 1,200 | (spray bottle 88.7 %) |
| | | `--lr 2e-4` | 800 / - / 1,600 | (spray bottle 98.0 %) |
| | | `--lr 8e-4` | - / - / 1,000 | (apple 23.8 %) |
| | | `--action_noise 0.2` | 1,000 / 2,000 / 1,200 | 3.52 mm |
| | | `--gamma 0.995` | 800 / 1,200 / 1,600 | 4.21 mm |
| Inspire | SuperTrack | `hand-supertrack-fast` | 1,000 / 800 / 1,200 | 2.36 mm |
| | | `--lr 2e-4` | 1,000 / 800 / 3,000 | 2.26 mm |
| | | `--lr 8e-4` | 1,000 / 800 / 2,200 | 2.37 mm |
| | | `--st_imag_noise 0.1` | 2,200 / 600 / 4,400 | 2.02 mm |
| | | `--st_fail_w 5` | 800 / 800 / 2,600 | 2.34 mm |
| | SuperMimic | `hand-supermimic-fast` | 1,800 / 1,200 / 1,000 | 3.64 mm |
| | | `--lr 2e-4` | 1,000 / 800 / 1,200 | 3.90 mm |
| | | `--lr 8e-4` | - / 800 / - | (can 98.4 %) |
| | | `--action_noise 0.2` | 800 / 600 / 4,400 | 3.65 mm |
| | | `--gamma 0.995` | 1,000 / 1,200 / - | (can 96.1 %) |

Object errors in brackets: a success below 0.99 on that clip, so the arm was not a candidate for the most accurate
configuration. Each algorithm's fast configuration on a hand is the faster, by the mean minutes over nine timed runs,
of its preset and its change with the fewest mean iterations to the target among those reaching it on all three clips (a
tie: the lower object error at 6,000); its most accurate one is the lower, by the object error over the three clips and
three seeds at 20,000 iterations, of its accurate preset and the arm (its fast preset or one of its four changes) with the
lowest object error at 6,000, trained for 20,000 iterations.

## Training time with the speed options

The six fast configurations with the speed options (`--compile_updates --elementwise_3x3 --combo_kernels --wm_palm_fk
--compile_collect --autotune_gemm`, SuperMimic also `--sm_reuse_input`), each run alone on an A100 40GB for 6,000
iterations, three training seeds per configuration, the other GPUs of the node idle. Each configuration first ran untimed
on each clip past its first policy updates, so the timed runs found every compiled kernel in the cache; a configuration's
first run on a machine compiles for about 3 minutes more. Minutes: the training wall clock at the first checkpoint (every
200 iterations) from which every later one has a success of 0.99 or more with the object within 10 mm (-: none within
6,000 iterations); iterations/s with policy updates (the log's rows 1,000-3,000).

Mean minutes over the nine runs of a configuration (a run without the target counted at its whole training time), before
and with the speed options: Inspire 2.34 and 1.30 (SuperTrack), 3.81 and 2.19 (SuperMimic); Allegro 3.90 and 1.96, 5.31 and
1.96; XHand 1.60 and 1.16, 1.49 and 1.03. On the Allegro hand `hand-supertrack-fast --st_fail_w 5` stays the faster of
SuperTrack's two candidates (`--lr_decay_iters 6000 --st_fail_w 5`: 2.29 min).

| task | | seeds | minutes to a success of 0.99 with the object within 10 mm ↓ | iterations/s ↑ | object position at 6,000 ↓ |
|---|---|---|---|---|---|
| `apple` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 1.2, 0.9, 1.0 | 17.7 | 2.8 ± 0.0 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 1.0, 1.0, 1.0 | 15.4 | 4.7 ± 0.1 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 1.3, 1.1, 0.9 | 17.5 | 2.4 ± 0.1 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 1.0, 1.0, 1.0 | 14.7 | 3.7 ± 0.1 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 0.7, 0.7, 0.7 | 18.1 | 2.5 ± 0.1 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 0.8, 0.8, 0.8 | 15.2 | 3.4 ± 0.0 mm |
| `oakink_a01002` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 0.8, 0.8, 0.8 | 18.6 | 1.7 ± 0.0 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 0.9, 0.7, 0.7 | 15.9 | 2.9 ± 0.2 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 0.9, 0.9, 0.7 | 18.0 | 1.4 ± 0.0 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 5.6, 4.2, 1.4 | 15.4 | 3.7 ± 0.3 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 1.0, 0.9, 0.9 | 18.5 | 1.5 ± 0.0 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 1.2, 1.0, 1.2 | 15.9 | 2.4 ± 0.1 mm |
| `can` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 1.0, 2.8, 2.3 | 18.6 | 2.2 ± 0.1 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 6.5, 1.6, - | 15.9 | 7.1 ± 5.3 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 3.5, 3.9, 4.4 | 18.0 | 2.8 ± 0.4 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 1.2, 1.2, 1.0 | 15.4 | 2.7 ± 0.4 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 1.8, 1.9, 1.8 | 18.5 | 2.3 ± 0.3 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 1.2, 1.2, 1.2 | 15.9 | 3.3 ± 0.2 mm |

Per iteration, the phases on the Allegro hand's apple (training seed 0, 3,000 iterations, ms, means over iterations
1,001-3,000, the device synchronized at every boundary):

| phase | SuperTrack | SuperTrack, speed options | SuperMimic | SuperMimic, speed options |
|---|---|---|---|---|
| collection | 9.0 | 7.8 | 10.8 | 8.3 |
| world model's updates | 42.7 | 24.7 | 42.9 | 25.2 |
| policy's update (SuperMimic's with the critic's) | 41.2 | 24.8 | 61.8 | 33.9 |
| iterations/s | 10.6 | 17.2 | 8.6 | 14.7 |

The two update graphs are about 90 % of an iteration. Their kernels' number, not their arithmetic, makes most of their
time outside the matrix products; compiling the remaining eager parts and the collection, launching small kernels
together and choosing each matrix product's kernel by timing removes most of it.

## Training time on three GPUs

The same six configurations and speed options with `--layout AWL`: the collection with the simulator and the replay, the
world model's updates and the policy's update each on its own A100 40GB, every step on the others' results of the
previous iteration. One run at a time on three GPUs of a node, the other GPUs idle, after the same untimed warm-up;
minutes, iterations/s and the object error as [with the speed options](#training-time-with-the-speed-options).

Mean minutes over the nine runs, one GPU and three: Inspire 1.30 and 0.70 (SuperTrack), 2.19 and 1.46 (SuperMimic); Allegro
1.96 and 1.40, 1.96 and 1.44; XHand 1.16 and 0.78, 1.03 and 0.72. Runs at the target on three GPUs: SuperTrack 9, 8 and 9
of 9, SuperMimic 7, 7 and 9 (Inspire, Allegro, XHand).

| task | | seeds | minutes to a success of 0.99 with the object within 10 mm ↓ | iterations/s ↑ | object position at 6,000 ↓ |
|---|---|---|---|---|---|
| `apple` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 0.7, 0.7, 0.5 | 39.8 | 2.8 ± 0.0 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 0.9, 0.7, 1.5 | 30.5 | 4.7 ± 0.1 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 0.8, 0.7, 0.7 | 39.6 | 2.4 ± 0.0 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 0.7, 0.7, 0.7 | 29.6 | 4.0 ± 0.2 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 0.6, 0.6, 0.6 | 39.4 | 2.5 ± 0.2 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 0.6, 0.6, 0.7 | 30.3 | 3.3 ± 0.1 mm |
| `oakink_a01002` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 0.6, 0.6, 0.6 | 39.8 | 1.6 ± 0.0 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | 0.8, 0.6, 0.6 | 30.5 | 3.1 ± 0.3 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | 0.6, 0.7, 0.7 | 39.6 | 1.3 ± 0.0 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 0.9, -, - | 29.6 | 6.4 ± 2.7 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 0.6, 0.6, 0.6 | 39.4 | 1.5 ± 0.1 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 0.7, 0.8, 0.7 | 30.3 | 2.6 ± 0.1 mm |
| `can` | Inspire, `hand-supertrack-fast` | 0, 1, 2 | 0.9, 0.7, 1.0 | 39.7 | 2.3 ± 0.2 mm |
|  | Inspire, `hand-supermimic-fast --lr 2e-4` | 0, 1, 2 | -, -, 1.0 | 30.5 | 8.6 ± 2.6 mm |
|  | Allegro, `hand-supertrack-fast --st_fail_w 5` | 0, 1, 2 | -, 2.9, 2.6 | 39.7 | 3.2 ± 1.0 mm |
|  | Allegro, `hand-supermimic-fast` | 0, 1, 2 | 0.8, 0.8, 0.8 | 29.6 | 3.2 ± 0.6 mm |
|  | XHand, `hand-supertrack-fast` | 0, 1, 2 | 1.0, 1.1, 1.4 | 39.4 | 2.1 ± 0.1 mm |
|  | XHand, `hand-supermimic-fast` | 0, 1, 2 | 0.8, 0.8, 0.8 | 30.3 | 3.4 ± 0.0 mm |

Per iteration on the Allegro hand's apple (training seed 0, 3,000 iterations, ms, means over iterations 1,001-3,000),
with the code and GPUs of the timed runs. With three GPUs the phases are timed one after another, every device
synchronized at each boundary, so their sum is not the running iteration:

| phase | SuperTrack, one GPU | SuperTrack, three GPUs | SuperMimic, one GPU | SuperMimic, three GPUs |
|---|---|---|---|---|
| collection | 8.0 | 12.3 | 10.0 | 13.2 |
| world model's updates | 24.8 | 24.9 | 25.1 | 26.8 |
| policy's update (SuperMimic's with the critic's) | 24.8 | 24.6 | 33.4 | 33.3 |
| exchange between the GPUs | - | 1.5 | - | 1.4 |
| iterations/s of the timed runs on this clip | 17.5 | 39.6 | 14.7 | 29.6 |

With three GPUs the steps run at once, so an iteration takes about as long as the slowest: 25.3 ms for SuperTrack (its
two updates, 24.9 and 24.6 ms) and 33.8 ms for SuperMimic (its policy update with the critic's, 33.3 ms).

## A G1 with Inspire hands

GigaHands' making tea carried out by a Unitree G1 with the two Inspire hands on its wrists
([`build_g1_hand_task.py`](../../tools/build_g1_hand_task.py), [`g1_hands.py`](../../ndm_gym/robots/g1_hands.py)). The
hands' bases replace the G1's rubber hands; the G1 stands at a small table whose top is the clip's floor, and its arms are
solved by inverse kinematics so that each base keeps the clip's pose, clear of the robot's body. The policy drives each
arm's seven hinges and each hand's twelve finger joints. The arms' model: position servos of stiffness 1,000 on joints of
armature 1 (a damping ratio of 1), as SPIDER's floating palms' hinges; the G1's own shapes without contacts; the pelvis,
the legs and the waist fixed; after a walk, the walk replayed. Two tasks:

- **standing:** the G1 at the table for the whole clip (338 frames);
- **walk, then tea:** the G1 walks 4.35 m to the table (LAFAN1's `walk1_subject1` retargeted to the G1, up to its frame
  3,600), its arms clear of its legs and the table, then reaches for the clip's first pose in 1 s and makes tea. The walk
  is the reference's, replayed; the task is the reach and the tea (387 frames).

```bash
TEA=data/spider/processed/gigahand/inspire/bimanual/p36-tea/0
python tools/build_g1_hand_task.py --clip $TEA --out data/g1_hands/tea
python tools/build_g1_hand_task.py --clip $TEA --walk data/lafan1_g1/walk1_subject1.csv --stand 3600 --out data/g1_hands/walk_tea
python -m ndm_gym.train_hand --config hand-supermimic-fast --iterations 12000 --clip data/g1_hands/walk_tea/0 --seed 0 --log_dir logs/g1_walk_tea_s0
python -m ndm_gym.evaluate_hand --ckpt logs/g1_walk_tea_s0/last.pt --out logs/g1_walk_tea_s0/evaluation
MUJOCO_GL=egl python tools/render_g1_hands.py --task data/g1_hands/walk_tea --rollout logs/g1_walk_tea_s0/evaluation/rollout.npz \
       --out g1_walk_tea.mp4 --webp g1_walk_tea.webp
```

**In short:** with the same options, seeds and tuning, SuperMimic holds 98.4-100 % on every seed standing and
77.3-98.4 % after the walk; SuperTrack 39.8-81.6 % and 22.3-69.1 %. Three of the four pairs' runs pass the motion check;
SuperMimic's after the walk does not (the targets' change per step 2.3 times the reference's) and is not shown.

| task | | configuration | success | object position | object position, last frame | fingertips | motion check (seed 0) |
|---|---|---|---|---|---|---|---|
| standing | reference targets | | 0 % | 142.8 mm | 93.9 mm | 117.9 mm | |
| | SuperTrack | `--config hand-supertrack-fast --iterations 12000 --lr 2e-4 --target_rate_w 4 --st_cost_mean` | 73.4 / 81.6 / 39.8 % | 19.0 / 24.1 / 33.2 mm | 36.3 / 27.6 / 18.8 mm | 27.6 / 37.0 / 31.7 mm | passes |
| | SuperMimic | `--config hand-supermimic-fast --iterations 12000 --action_noise 0.5 --target_rate_w 4` | 100.0 / 98.4 / 100.0 % | 7.8 / 9.3 / 8.0 mm | 9.9 / 9.6 / 9.5 mm | 18.5 / 19.5 / 18.3 mm | passes |
| walk, then tea | reference targets | | 0 % | 127.7 mm | 95.7 mm | 100.8 mm | |
| | SuperTrack | `--config hand-supertrack-fast --iterations 12000 --lr 2e-4 --target_rate_w 4 --st_cost_mean` | 22.3 / 69.1 / 62.1 % | 45.7 / 26.9 / 26.1 mm | 57.0 / 46.3 / 43.2 mm | 31.8 / 27.4 / 30.4 mm | passes |
| | SuperMimic | `--config hand-supermimic-fast --iterations 12000 --lr 2e-4 --target_rate_w 4` | 77.3 / 92.2 / 98.4 % | 25.1 / 16.1 / 8.0 mm | 17.6 / 23.4 / 8.6 mm | 21.0 / 18.8 / 16.3 mm | fails: target change 2.3x |

- **Configurations:** each algorithm's fast configuration at 12,000 iterations with the target-rate penalty
  (`--target_rate_w 4`, SuperTrack also `--st_cost_mean`, as the XHand's [above](#the-xhand-with-the-target-rate-penalty)),
  and one single change, the same four tried for both ([below](#the-g1s-tuning)).
- **Training:** 2,048 worlds, training seeds 0-2, on L40S GPUs, four runs to a GPU, so no times are given.
- **Evaluation:** the last checkpoint, 256 rollouts from perturbed starts, evaluation seed 0.
- **Motion check:** `tools/hand_motion_check.py` on seed 0's unperturbed rollout against the kinematic reference, the arms'
  hinges counted as the palms': the finger joints' jerk at most 1.5 times, their share of velocity power above 5 Hz and the
  position targets' change per step at most 2 times the reference's. A run that fails is reported and not shown.

### The G1's tuning

Both algorithms got the same single changes (`--lr 2e-4`, `--action_noise 0.5`, `--reward_std_scale 2`,
`--reward_std_scale 3`) on training seeds 3-5 and evaluation seed 1, which the reported runs do not use; each pair keeps
the one with the highest mean success, then the lowest mean object error (in bold). Success on seeds 3 / 4 / 5, without
the penalty:

| task | algorithm | no change | `--lr 2e-4` | `--action_noise 0.5` | `--reward_std_scale 2` | `--reward_std_scale 3` |
|---|---|---|---|---|---|---|
| standing | SuperTrack | 0.4 / 27.7 / 23.8 % | **10.2 / 21.5 / 35.5 %** | 0.0 / 27.3 / 34.8 % | 0.0 / 13.7 / 0.0 % | 2.3 / 27.3 / 17.6 % |
| standing | SuperMimic | 99.2 / 98.8 / 100.0 % | 100.0 / 87.9 / 98.0 % | **100.0 / 100.0 / 100.0 %** | 98.4 / 99.6 / 98.0 % | 98.8 / 99.6 / 94.9 % |
| walk, then tea | SuperTrack | 9.4 / 3.9 / 0.0 % | **40.2 / 43.4 / 14.8 %** | 22.7 / 1.2 / 0.8 % | 5.9 / 0.0 / 16.0 % | 14.5 / 0.4 / 19.1 % |
| walk, then tea | SuperMimic | 98.8 / 97.3 / 98.0 % | **96.9 / 98.8 / 100.0 %** | 98.8 / 86.3 / 91.8 % | 100.0 / 96.1 / 99.6 % | 96.5 / 100.0 / 99.2 % |

Without the target-rate penalty every pair's reported run failed the motion check; with it, success over seeds 0 / 1 / 2:

| task | algorithm | without the penalty: success | object position | with it: success | object position | motion check without / with |
|---|---|---|---|---|---|---|
| standing | SuperTrack | 43.8 / 2.3 / 70.7 % | 55.6 / 91.6 / 46.9 mm | 73.4 / 81.6 / 39.8 % | 19.0 / 24.1 / 33.2 mm | fails / passes |
| standing | SuperMimic | 99.2 / 100.0 / 100.0 % | 14.7 / 13.2 / 19.6 mm | 100.0 / 98.4 / 100.0 % | 7.8 / 9.3 / 8.0 mm | fails / passes |
| walk, then tea | SuperTrack | 28.1 / 10.2 / 39.1 % | 58.5 / 60.3 / 45.2 mm | 22.3 / 69.1 / 62.1 % | 45.7 / 26.9 / 26.1 mm | fails / passes |
| walk, then tea | SuperMimic | 100.0 / 98.0 / 100.0 % | 7.6 / 10.5 / 7.1 mm | 77.3 / 92.2 / 98.4 % | 25.1 / 16.1 / 8.0 mm | fails / fails |

### The whole body

The same making tea with the G1's whole body in the human-object pipeline (`ndm_gym.train_hoi`, HDMI's task on Isaac
Lab with MuJoCo-Warp): one policy drives all 53 joints, the legs, the waist, the arms, the wrists and the fingers; it walks
the 4.35 m, keeps its balance, reaches and makes tea. HDMI's rewards with the fingertips' and the finger joints' tracking
added, each hand's contact force summed over its palm and its fingers, HDMI's terminations. The reference is the walk and,
from the reach on, the rollout of an arms-and-hands SuperMimic policy on SPIDER's physics-optimised retarget
(`trajectory_mjwp.npz`), in which the fingers stay out of the teapot: the hands' prior (`g1_inspire_walk_tea_prior`,
[`build_g1_wholebody_clip.py`](../../tools/build_g1_wholebody_clip.py)). Training is staged:

1. **The body:** the fingers follow the reference (`--pi_fixed_joints`: the policy's residual on them is zero), the rest
   learns.
2. **The whole body:** from the body's networks (`--init`), the policy's output rows of the fingers zeroed
   (`--init_zero_joints`), so that it starts as the body's policy; then every joint learns.

| | seeds | success | object position | keypoints |
|---|---|---|---|---|
| the body (the fingers following the reference) | 0 / 1 / 2 | 90.1 / 78.9 / 79.9 % | 107 / 75 / 114 mm | 182 / 187 / 204 mm |
| the whole body, from the body | 0 / 1 / 2 | 93.1 / 96.2 / 86.6 % | 82 / 76 / 115 mm | 133 / 182 / 205 mm |

- **Training:** SuperMimic with HDMI's settings (`--lr 4e-4 --action_noise 0.3 --gamma 0.995`), 3 h per stage, 2,048
  worlds from the clip's first frame, seeds 0-2, on L40S GPUs.
- **Evaluation:** `ndm_gym.evaluate_hoi` on the last checkpoint, 2,048 worlds, 1,000 steps; a success never terminates.
  Object position and keypoints: the mean distance to the reference.

```bash
# the hands' prior: the arms-and-hands task on the physics-optimised retarget, SuperMimic, then its rollout as the clip
python -m ndm_gym.train_hand --config hand-supermimic-fast --iterations 12000 --clip data/g1_hands/walk_tea_mjwp/0 --seed 0 --log_dir logs/g1_prior
python -m ndm_gym.evaluate_hand --ckpt logs/g1_prior/last.pt --out logs/g1_prior/evaluation
python tools/build_g1_wholebody_clip.py --task data/g1_hands/walk_tea_mjwp --rollout logs/g1_prior/evaluation/rollout.npz \
       --name g1_inspire_walk_tea_prior --out data/g1_hands/wholebody_prior
python tools/g1_wholebody_usd.py data/g1_hands/wholebody_prior data/hdmi/g1_inspire/g1_inspire_walk_tea_prior
cp data/g1_hands/wholebody_prior/g1_inspire_walk_tea_prior.{npz,json} data/hdmi/clips/
# the two stages
FINGERS='.*_(thumb|index|middle|ring|pinky)_.*_joint'
SM="--task g1_inspire_walk_tea_prior --algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.995 --max_hours 3"
python -m ndm_gym.train_hoi $SM --seed 0 --pi_fixed_joints "$FINGERS" --log_dir logs/wb/body_s0
python -m ndm_gym.train_hoi $SM --seed 0 --init logs/wb/body_s0/last.pt --init_zero_joints "$FINGERS" --log_dir logs/wb/whole_s0
```

`data/g1_hands/walk_tea_mjwp` is `build_g1_hand_task.py --walk` on the tea clip with SPIDER's `trajectory_mjwp.npz`
(the physics-optimised retarget, 100 Hz) taken every second step as its `trajectory_kinematic.npz`.

## Training time

These times are before the speed options ([with them](#training-time-with-the-speed-options)).

Each run trains alone on an A100 40GB for 6,000 iterations with the systems options, three training seeds per
configuration; its checkpoints, every 200 iterations, are evaluated after training as above. A run's time is its
training wall clock (compilation, the CUDA graphs' captures and the checkpoints' writes included) at its first
checkpoint from which every later one has a success of 0.99 or more with the object within 10 mm (-: none; SuperTrack's
seed 1 on `can` ends at a success of 0.988). Iterations/s are with policy updates (the log's rows 1,000-3,000).

| task | | seeds | minutes to a success of 0.99 with the object within 10 mm ↓ | iterations/s ↑ | object position at 6,000 ↓ |
|---|---|---|---|---|---|
| `apple` | SuperTrack | 0, 1, 2 | 0.9, 1.7, 0.9 | 11.7 | 2.6 ± 0.1 mm |
| | SuperMimic, defaults | 0, 1, 2 | 4.9, 4.2, 6.3 | 9.5 | 3.7 ± 0.2 mm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 5.3, 2.1, 1.4 | 9.4 | 3.5 ± 0.2 mm |
| | SuperTrack, with Muon, the policy average and the rate ramp | 0, 1, 2 | 0.9, 0.9, 0.9 | 11.4 | 2.5 ± 0.0 mm |
| | SuperMimic, HDMI's settings at γ 0.99, with them | 0, 1, 2 | 1.4, 1.0, 1.4 | 9.2 | 3.2 ± 0.0 mm |
| `oakink_a01002` | SuperTrack | 0, 1, 2 | 2.3, 4.5, 4.8 | 12.0 | 2.4 ± 1.0 mm |
| | SuperMimic, defaults | 0, 1, 2 | 2.7, 3.4, 5.9 | 9.6 | 3.6 ± 0.4 mm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 1.7, 1.3, 1.3 | 9.7 | 3.3 ± 0.3 mm |
| | SuperTrack, with Muon, the policy average and the rate ramp | 0, 1, 2 | 1.2, 1.8, 1.5 | 11.6 | 1.5 ± 0.0 mm |
| | SuperMimic, HDMI's settings at γ 0.99, with them | 0, 1, 2 | 1.7, 1.4, 1.7 | 9.4 | 2.5 ± 0.1 mm |
| `can` | SuperTrack | 0, 1, 2 | 7.5, -, 8.4 | 12.0 | 3.1 ± 0.6 mm |
| | SuperMimic, defaults | 0, 1, 2 | 8.3, 2.7, 6.5 | 9.7 | 4.1 ± 0.2 mm |
| | SuperMimic, HDMI's settings at γ 0.99 | 0, 1, 2 | 3.1, 3.4, 2.0 | 9.7 | 4.5 ± 0.6 mm |
| | SuperTrack, with Muon, the policy average and the rate ramp | 0, 1, 2 | 2.0, 3.2, 2.0 | 11.6 | 2.1 ± 0.1 mm |
| | SuperMimic, HDMI's settings at γ 0.99, with them | 0, 1, 2 | 1.7, 1.4, 1.7 | 9.4 | 3.6 ± 0.3 mm |

"With them": HOI's optimizer options, `--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95
--lr_decay_by 1400 --lr_decay_len 300`. On `apple` (seed 0; SuperMimic with HDMI's settings at γ 0.99) the training loop
without the systems options runs 3.7 (SuperTrack) and 2.7 iterations/s, with them and `--amp bf16` 14.0 and 11.0.

## How the configurations were chosen

Both algorithms started from the settings selected on HDMI's tasks and were then tuned by the same rule.

- With HDMI's settings as they are (SuperTrack `--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_vel_terms --st_obj_w 2`,
  SuperMimic `--lr 4e-4 --action_noise 0.3 --gamma 0.995 --bptt_decay 0.95`; training seed 0, 20,000 iterations),
  SuperTrack read 19.5 and 25.8 mm on the object at a success of 100 and 96.5 % on `can` and `oakink_a01002`, and
  SuperMimic 3.4 and 246.2 mm at 100 and 0 % (on `apple`, not run for SuperTrack, 3.4 mm at 100 %).
- SuperTrack's terms were weighted to 1 at its first policy update. On clips whose reference pose alone does not lift
  the object, the object's first error is large and the finger joints' is small, so the object weighed little and the
  fingers much. `--st_task_weights` takes the task's reward weight over its kernel's width for each term instead.
- Nine arms of each algorithm (6,000 iterations, seed 0) on `oakink_a01002` and `can`, and one more of SuperTrack at
  64 imagined steps, chose one arm of each algorithm by the same rule (a success of 0.99 on both clips, then the lower
  object error). A second arm of each followed (SuperTrack with the object's terms at half weight; SuperMimic's
  defaults), and all four trained with three seeds for the table above.
- Means over the three clips, object / fingertips: SuperTrack 2.1 / 5.7 mm, with the object's terms at half weight
  2.9 / 4.7 mm; SuperMimic with HDMI's settings at γ 0.99 3.4 / 6.6 mm, with its defaults 3.2 / 6.7 mm.

## Acknowledgment

The task, its simulator, SuperMimic on it and the evaluation were contributed by Dechen Gao
([pull request 2](https://github.com/Frank-ZY-Dou/ndm-gym/pull/2)), the Allegro and the Inspire hand and the reading of a
hand from its scene by the same author ([pull request 6](https://github.com/Frank-ZY-Dou/ndm-gym/pull/6)).
