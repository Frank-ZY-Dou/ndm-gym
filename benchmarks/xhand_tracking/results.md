# Hand-object tracking on the XHand: results

SuperTrack and SuperMimic, with MLP world models, track human grasp-and-lift clips with a floating right XHand
([the task](../../docs/tasks/hand.md)).

## Setup

- **Reference:** the kinematic retarget of the captured human hand and of the object's pose
  ([SPIDER](https://huggingface.co/datasets/retarget/retarget_full)'s `trajectory_kinematic.npz`, at the revision in
  `tasks.json`); no physics-optimized trajectory is used.
- **Training:** `ndm_gym.train_hand`, 20,000 iterations in 2,048 worlds, 16 imagined steps, on one A100 40GB, with the
  systems options (`--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs`).
- **Evaluation:** `ndm_gym.evaluate_hand` on the last checkpoint: 256 rollouts of the whole clip from perturbed starts
  (evaluation seed 0), without exploration noise. A rollout succeeds if it never terminates (the object or the palm
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

## Training time

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
([pull request 2](https://github.com/Frank-ZY-Dou/ndm-gym/pull/2)).
