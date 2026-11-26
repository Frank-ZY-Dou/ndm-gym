# Animal motion tracking: a rat, a fruit fly and a stick insect on MIMIC-MJX's data

Anatomically detailed bodies of three animals track their animals' recorded kinematics in physics: a rat (walking, rearing,
grooming), a fruit fly (walking) and a stick insect (walking), with SuperTrack and SuperMimic on MuJoCo-Warp
([the task](../../docs/tasks/animals.md), [results](results.md)). It adds the library's first non-human animal bodies,
torque-driven skeletons with many passive degrees of freedom; MIMIC-MJX's PPO trains them in 6 to 20 hours on one GPU, as
published.

- **Kinematics:** the reference clips of MIMIC-MJX (Zhang, Yang et al., "MIMIC-MJX: Neuromechanical Emulation of Animal
  Behavior", arXiv 2511.20532), [talmolab/MIMIC-MJX](https://huggingface.co/datasets/talmolab/MIMIC-MJX) on Hugging Face
  (MIT), at a pinned revision: the rat's `rodent_reference_clips.h5` (842 clips of 250 frames, 5 s at 50 Hz, from DANNCE
  keypoints), the fly's `fly_reference_clip.h5` (1,730 clips of 600 frames, 1.2 s of treadmill walking at 500 Hz) and the
  stick insect's `stick_mesh_reference.h5` (75 clips of 225 frames at 50 Hz, the clips of its mesh model). Each holds
  stac-mjx's qpos and qvel of the body per frame (none are retargeted here), resampled to the control rate. At this revision
  the files' qvel is not the velocity of their qpos (the root's linear velocity is 10 times the positions' for the rat and
  the stick insect and 0.1 times for the fly; joint speeds are clipped at 20 rad/s), so the benchmark takes the reference's
  velocities from its positions ([the task](../../docs/tasks/animals.md)). 2.9 GB of the dataset's 101 GB; downloaded by a
  script, not redistributed.
- **Bodies:** the rat's, the fly's and the stick insect's models of
  [talmolab/vnl-playground](https://github.com/talmolab/vnl-playground) (BSD-3-Clause; its fly is
  [flybody](https://github.com/TuragaLab/flybody)'s, Apache-2.0), built as MIMIC-MJX's environments build them
  (`ndm_gym/robots/animals.py`): the rat (74 qpos, 38 actuators) with its motors as torque actuators, scaled by 0.9, its
  actuators' first-order activation filters kept (40 ms); the fly (43 qpos, 36 position actuators on its legs); the stick
  insect (48 qpos, 41 torque actuators) with its feet's contact pairs and its actuators' scaling. Their qpos and qvel match the
  clips' layouts exactly, and one control step on MuJoCo-Warp matches MuJoCo's to 2e-8 (median) from the clips' frames.
- **Simulator:** MuJoCo-Warp at MIMIC-MJX's rates (the rat and the stick insect: 100 Hz control, 2 ms steps; the fly:
  500 Hz control, 0.2 ms steps), the control step as one CUDA graph. No Isaac Lab, no MJX.
- **Baseline:** [talmolab/track-mjx](https://github.com/talmolab/track-mjx) (BSD-3-Clause), MIMIC-MJX's PPO: its published
  numbers are quoted; it was not run here.

## Rules

Fixed before the three-seed runs.

- **Clips:** one policy per body on clips 0-9 of its reference; clips 10-19 held out, never trained on.
- **Configurations:** one per algorithm, the same for the three bodies apart from the budget (`--config <body>-<algorithm>`,
  [results](results.md#how-the-numbers-were-obtained)).
- **Reference velocities:** those of the reference's positions at the control rate (`--ref_qvel positions`), for both
  algorithms.
- **Budget:** fixed and the same for both algorithms: 4,000 iterations (stick insect), 6,000 (fruit fly), 10,000 (rat); a
  checkpoint kept every 250.
- **Seeds:** three training seeds per configuration (0, 1, 2); mean and sample SD; the GPU stated.
- **Evaluation:** the last checkpoint plays every clip from its first frame and four more times from perturbed starts
  (0.02 rad on the joints), without exploration noise, to the clip's end: the share of episodes ended by MIMIC-MJX's
  terminations, the tracked bodies', the end effectors' and the root's position errors and the joints' error, over the steps
  the episodes lived; the same on the held-out clips.

The target of the time to the target (at most 2 % of the episodes ended, the bodies within 3 % of the body's size) was set
after earlier runs of these configurations, on their kept checkpoints. MIMIC-MJX's median keypoint error against stac-mjx's keypoints is not computed
here: the bodies' error is a mean over the body frames' origins.
