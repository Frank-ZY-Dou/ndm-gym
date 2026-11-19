# Humanoid motion tracking: a Unitree H1 on LAFAN1

A Unitree H1 tracks LAFAN1 clips retargeted to it with SuperTrack and SuperMimic, on MuJoCo-Warp
([the task](../../docs/tasks/humanoid.md), [results](results.md)). It adds a humanoid other than the G1 to the library:
the H1 is taller and heavier (51.6 kg in its description against the G1's 33.3 kg), has one ankle joint per leg (pitch, no
roll) and no wrists.

- **Motions:** the H1 retargets of LAFAN1 (Ubisoft La Forge) in the dataset the library already reads for the G1,
  [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (`h1/`, 40
  clips at 30 fps, at the revision `tools/download_data.sh` pins; CC BY-NC-ND 4.0), downloaded by `tools/download_h1.sh`,
  checked against hashes and not redistributed here. The retargets are kinematic (the dataset's README: no dynamics, no
  actuator limits).
- **Robot:** the dataset's H1 description, Unitree's ([unitree_ros](https://github.com/unitreerobotics/unitree_ros),
  `robots/h1_description`, BSD-3-Clause), converted to MJCF as the G1's is; its motors as joint position servos within
  the description's torque limits, with the gains of Unitree's H1 training configuration
  ([unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), BSD-3-Clause) and the joints' armature and
  damping of Unitree's MuJoCo model of the H1 ([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco),
  BSD-3-Clause).
- **Training options:** the Go2 task's selected options as the starting point; the termination, the 10 s episodes, the
  episode starts, the wider bound on the position targets' offsets and the targets not clamped to the joints' ranges
  follow the library's G1 training (`ndm_gym/train.py`, whose policy is unbounded).

| file | |
|---|---|
| `run.sh` | train and evaluate one task and seed: the last checkpoint (64 rollouts per clip, with pushes), every kept checkpoint, and for `train` the held-out clips |
| `summarize.py` | the tables of `results.md` from the runs |
| `results.md` | setup, results, findings |

## Rules

Fixed before the three-seed runs of the reported configurations.

- **Clips:** one policy on 34 clips: LAFAN1's 40 without the six whose names start with `fallAndGetUp`. The rule is by
  name and was fixed after the seed 0 runs on all 40 clips, in which neither algorithm completed any rollout of these six
  (results, "All 40 clips"); the 40-clip numbers of seed 0 are reported beside the main result.
- **Clips not seen in training:** LAFAN1's own split (`benchmarks/g1_lafan1_tracking/splits`): training on the clips of
  subjects 1 to 4 (28 of the 34), subject 5's (6) held out.
- **Algorithms:** SuperTrack and SuperMimic, each from the library's selected options of the Go2 task; SuperTrack's
  starting configuration has the library G1's imagined window (`--horizon 32`; the Go2's is 16) and `--pi_batch 2048`
  (the Go2's is 4096). The options the two share (the episodes, the action bound, the accepted position targets) are
  the same for both, with one difference: the start sampling's bins rank by returns discounted with the trainer's
  `--gamma`, as in the library's G1 training, which is 0.99 for SuperTrack (it has no discount of its own) and
  SuperMimic's own discount (0.97 in its selected configuration).
- **Tuning:** one round per algorithm of the same size on seed 0 (34 clips, the fixed budget, 4 rollouts per clip):
  eleven runs, each one change of one of the algorithm's own options. The rule, fixed before each round: a change is
  adopted if it raises the return by at least 0.02 and does not lower the success by more than 0.02; with several such
  changes, one run of all of them together, adopted if its return is at least the best single change's (otherwise the
  best single change). The adopted configuration is then run on the three seeds. Every run of both rounds is reported
  (results, "Beside the starting point").
- **Budget:** 8,000 iterations for both algorithms, fixed; the last checkpoint is evaluated.
- **Seeds:** three training seeds (0, 1, 2) per reported configuration; mean and sample SD, every seed listed.
- **Evaluation:** `ndm_gym.evaluate_humanoid` on the last checkpoint: 64 rollouts per clip of the whole clip from
  perturbed starts, without exploration noise and without resets; a rollout succeeds if it never terminates (the
  library G1's termination: the pelvis 0.25 m off the reference's height, or tilted by 0.8). Reported: the success rate,
  the normalized return (nothing counted after a termination), the clips with a failed rollout, MPJPE in the world and
  about the pelvis, the pelvis' and the joints' errors, and the success under pelvis pushes of 1 and 2 m/s every 2 s.
  Beside them, the reference's joint positions as the targets, open loop.
- **Training time:** the wall time of the fixed budget, from runs that train only, each alone on its GPU, and the return
  of every kept checkpoint (every 500 iterations); not a time to a target.
- **Length:** a run is expected to take minutes; a configuration that needs more than an hour is treated as a defect.
