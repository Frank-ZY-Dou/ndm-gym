# Hand-object tracking: results

SuperTrack and SuperMimic, with MLP world models, track human grasp-and-lift clips with a floating right XHand
([the task](../../docs/tasks/hand.md)); both also with the Allegro hand and the Inspire hand ([more hands](#more-hands)).

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

## More hands

The Allegro hand (16 finger joints, four fingers) and the Inspire hand (12 finger joints) track the same three clips,
from SPIDER's retargets of the same demonstrations to them. A hand is read from its scene (`ndm_gym/robots/hands.py`: the
actuated joints in the actuators' order, the palm's site and the fingertip sites the hand has), so the task, the world
models, the policies and the options are the XHand's. The two hands and this reading of a scene were contributed in
[pull request 6](https://github.com/Frank-ZY-Dou/ndm-gym/pull/6); on the XHand's three scenes the code before and after
it gives the same indices, kinematics, task outputs and training updates bit for bit on the CPU.

- **Training:** both algorithms with the XHand's named configurations, `hand-supertrack-fast` and `hand-supermimic-fast`
  (6,000 iterations, with HOI's optimizer options) and `hand-supertrack-accurate` and `hand-supermimic-accurate` (20,000
  iterations); 2,048 worlds, training seeds 0-2, one A100 40GB per run. Then, per hand and algorithm, four single changes
  of its fast configuration on training seed 0, the same kinds for both, and three-seed runs of the changes chosen by the
  same rules ([below](#how-the-two-hands-configurations-were-chosen)): timed at 6,000 iterations, and at 20,000.
- **Evaluation:** as above, on the last checkpoint. The 6,000-iteration runs trained alone on a GPU; their checkpoints,
  every 200 iterations, were evaluated after training, as in [training time](#training-time).

Each algorithm's configurations of record, on `apple` / `oakink_a01002` / `can` (mean of training seeds 0-2; most
accurate: object position at 20,000 iterations; time: minutes to a success of 0.99 with the object within 10 mm, over the
seeds that reach it, not engineered):

| hand | | SuperTrack | SuperMimic |
|---|---|---|---|
| Allegro | most accurate | `hand-supertrack-fast --st_fail_w 5 --iterations 20000`: object 2.6 / 1.3 / 2.0 mm at a success of 100.0 / 100.0 / 99.7 % | `hand-supermimic-fast --action_noise 0.2 --iterations 20000`: 3.7 / 11.7 / 3.4 mm at 100.0 / 82.7 / 99.6 % |
| | time to the target, not engineered | `hand-supertrack-fast --st_fail_w 5`: 1.6 / 1.3 / 8.8 min, 9 of its 9 runs at the target | `hand-supermimic-fast`: 1.4 / - / 2.7 min, 6 of 9 |
| Inspire | most accurate | `hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000`: object 2.4 / 1.3 / 2.0 mm at a success of 99.7 / 100.0 / 99.9 % | `hand-supermimic-accurate`: 5.2 / 2.5 / 11.2 mm at 99.9 / 100.0 / 93.8 % |
| | time to the target, not engineered | `hand-supertrack-fast`: 2.4 / 1.5 / 3.1 min, 9 of its 9 runs at the target | `hand-supermimic-fast --lr 2e-4`: 2.5 / 2.1 / 4.9 min, 8 of 9 |

```bash
python tools/download_hand_clips.py --robots allegro inspire
ROBOT=allegro benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/allegro_st --config hand-supertrack-fast --st_fail_w 5 --iterations 20000
ROBOT=allegro benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/allegro_sm --config hand-supermimic-fast --action_noise 0.2 --iterations 20000
ROBOT=inspire benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/inspire_st --config hand-supertrack-fast --st_imag_noise 0.1 --iterations 20000
ROBOT=inspire benchmarks/xhand_tracking/run.sh <task> <seed> data/spider logs/inspire_sm --config hand-supermimic-accurate
```

### Accuracy, 20,000 iterations

The last checkpoint over 256 perturbed starts, mean ± SD over training seeds 0-2. Per hand and algorithm two
configurations trained for 20,000 iterations: its accurate preset and the change its screen chose. The one with the
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

These times are not engineered: the hand trainer has no per-phase time table yet (collection, the world model's and the
policy's updates, evaluation, host time), so no configuration here is called the fastest.

Each run trained alone on a GPU. Minutes: the training wall clock at the first of the run's checkpoints (every 200
iterations) from which every later one has a success of 0.99 or more with the object within 10 mm (-: none within 6,000
iterations); iterations/s are with policy updates (the log's rows 1,000-3,000). Per hand and algorithm two
configurations: its fast preset and its change with the fewest iterations to the target in the screen. The one with the
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

Per hand, algorithm and clip the README's grid shows the run with the lowest object error over the perturbed starts at
its last checkpoint among that algorithm's configurations of record on the hand (for the XHand its accurate preset: the
fast presets' timed runs kept no rollout), from the unperturbed start. Configuration and training seed:

| hand | | `apple` | `oakink_a01002` | `can` |
|---|---|---|---|---|
| XHand | SuperTrack | `hand-supertrack-accurate`, 1 | the same, 1 | the same, 2 |
| | SuperMimic | `hand-supermimic-accurate`, 2 | the same, 1 | the same, 2 |
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
of its preset and its change with the fewest mean iterations to the target; its most accurate one is the lower, by the
object error over the three clips and three seeds at 20,000 iterations, of its accurate preset and its change with the
lowest object error at 6,000 trained for 20,000 iterations.

## Training time

These times are not engineered: the hand trainer has no per-phase time table yet (collection, the world model's and the
policy's updates, evaluation, host time), so no configuration here is called the fastest.

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
