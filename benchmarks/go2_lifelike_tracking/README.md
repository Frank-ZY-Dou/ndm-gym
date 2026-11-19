# Quadruped motion tracking: a Unitree Go2 on a Labrador's motion capture

A Unitree Go2 tracks the motion capture of a Labrador retriever from "Lifelike agility and play in quadrupedal robots"
(Han et al., Nature Machine Intelligence 2024) with SuperTrack and SuperMimic, on MuJoCo-Warp
([the task](../../docs/tasks/quadruped.md), [results](results.md)). It adds a legged floating base other than the G1 to
the library, with flight phases and fast gaits.

- **Motion capture:** the 33 BVH files (16.3 min at 120 fps) of
  [Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play)
  (`data/raw_mocap_data` at `cc9eaa9`; its licence file is MIT, its README says the code and data are for research
  purposes only), downloaded by `tools/download_lifelike.sh` and not redistributed here.
- **Robot:** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_go2` (at `370c439`,
  BSD-3-Clause), its motors as joint position servos within their torque limits, the joints' damping that of Unitree's own
  model ([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)).
- **Retargeting:** [supermimic-quadruped](https://github.com/Houserz/supermimic-quadruped)'s `quadmotion` package (MIT;
  the results used `46a81c2`), called from the user's own checkout by `tools/convert_lifelike.py`; none of its code is
  copied here. 65 clips, 488 s at the robot's time scale.

| file | |
|---|---|
| `tasks.json` | the benchmark's tasks: a walk, a run and a play clip (`walk`, `run`, `play`) and one policy on all 65 clips (`all`) |
| `run.sh` | train and evaluate one task and seed (with pushes), then time the run |
| `time_to.sh` | the training time from which a run holds the target (its kept checkpoints, evaluated together) |
| `summarize.py` | the tables of `results.md` from the runs |
| `results.md` | setup, results, findings |

## Rules

Fixed before the three-seed runs.

- **Tasks:** one policy on all 65 clips; and one clip of each kind, chosen for their gaits before their runs: a walk
  (`dog_quad_walk_002_1`, 6.0 s), a run (`dog_quad_run_002_0`, 5.5 s, up to 2.95 m/s) and a play clip
  (`dog_play_002_0`, 10.1 s). The two long clips (70 s and 79 s) stay whole.
- **Algorithms:** SuperTrack and SuperMimic, each from the library's selected options (the hand task's systems and
  SuperTrack options; HOI's optimizer options for SuperMimic), then any change that improves on them, reported beside the
  starting point.
- **Seeds:** three training seeds (0, 1, 2) per reported configuration; mean and sample SD; one run per GPU, no other job on
  it (a run that shared its GPU was run again).
- **Evaluation:** `ndm_gym.evaluate_quadruped` on the last checkpoint: rollouts of the whole clip from perturbed starts,
  without exploration noise and without resets; a rollout succeeds if it never terminates. Reported: the success rate, the
  normalized return, the failed clips, the feet's, the trunk's and the joints' errors, and the failed clips under trunk
  pushes of 1 and 2 m/s every 2 s. Beside them, the reference's joint positions as the targets, open loop.
- **Target of the training time:** a success of at least 0.99 with a mean trunk error below 20 mm, held by every later
  checkpoint (every 500 iterations), as the hand task's study times it.
- **Length:** a run is expected to take minutes; a configuration that needs more than an hour is treated as a defect.
