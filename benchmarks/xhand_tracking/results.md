# Hand-object tracking on the XHand: results

A floating right XHand tracks human grasp-and-lift clips with SuperMimic ([the task](../../docs/tasks/hand.md)). The
reference is the kinematic retarget of the captured human hand and of the object's pose
([SPIDER](https://huggingface.co/datasets/retarget/retarget_full)'s `trajectory_kinematic.npz`, at the revision in
`tasks.json`); no physics-optimized trajectory is used.

| task | clip | frames (50 Hz) | object lift |
|---|---|---|---|
| `apple` | HRDexDB `apple-0` | 102 | 26.2 cm |
| `oakink_a01002` | OakInk `oakink_0000_right_A01002_00` (a spray bottle) | 86 | 24.3 cm |
| `can` | DexYCB `dexycb_01_right_002_master_chef_can_20200709_141754_00` | 106 | 32.7 cm |

```bash
python tools/download_hand_clips.py                    # the clips of tasks.json into data/spider
benchmarks/xhand_tracking/run.sh <task> <seed>         # train, evaluate and render
```

## Results

`ndm_gym.train_hand`'s defaults (20,000 iterations in 2048 worlds; about 1 h 35 min on one A40), then
`ndm_gym.evaluate_hand` on the last checkpoint: 256 rollouts of the whole clip from perturbed starts, without exploration
noise. A rollout succeeds if it never terminates (the object or the palm 15 cm from the reference) and the object ends
within 5 cm of the reference. Errors are means over the clip's frames and the rollouts; mean ± SD over seeds. The can's
seeds 1 and 2 trained in three parts, each continued with `--resume` (one on an A100).

Reference targets: the same rollouts with the reference's joint positions as the position targets.

| task | | seeds | success | object position | object position, last frame | fingertips | object lift |
|---|---|---|---|---|---|---|---|
| `apple` | reference targets | | | 19.9 mm | 15.2 mm | 26.6 mm | 24.8 cm |
| | SuperMimic | 0, 1, 2 | 100 ± 0 % | 3.4 ± 0.1 mm | 0.6 ± 0.2 mm | 6.3 ± 0.2 mm | 26.2 ± 0.0 cm |
| `oakink_a01002` | reference targets | | | 198.4 mm | 357.1 mm | 29.7 mm | -8.4 cm |
| | SuperMimic | 0, 1, 2 | 100 ± 0 % | 3.0 ± 0.5 mm | 2.4 ± 0.5 mm | 6.9 ± 0.2 mm | 24.3 ± 0.1 cm |
| `can` | reference targets | | | 106.7 mm | 332.7 mm | 32.8 mm | -0.3 cm |
| | SuperMimic | 0, 1, 2 | 100 ± 0 % | 3.7 ± 0.5 mm | 2.4 ± 0.7 mm | 7.3 ± 0.3 mm | 32.7 ± 0.1 cm |
