# Animal motion tracking: a rat, a fruit fly and a stick insect on MIMIC-MJX's data (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a plan:
no results yet.

## Task

Anatomically detailed bodies of three animals track their animals' recorded kinematics in physics: a rat (walking, rearing,
grooming), a fruit fly (walking) and a stick insect (walking). SuperTrack and SuperMimic train the policy through an MLP
world model of the simulator, as in the library's other tracking tasks. It adds the library's first non-human animal bodies,
torque-driven skeletons with many passive degrees of freedom, and a task whose published training is slow: MIMIC-MJX's PPO
converges in 6 to 20 hours on one GPU (over 30 hours on CPUs).

- **Kinematics:** the reference clips of MIMIC-MJX (Zhang, Yang et al., "MIMIC-MJX: Neuromechanical Emulation of Animal
  Behavior", arXiv 2511.20532), [talmolab/MIMIC-MJX](https://huggingface.co/datasets/talmolab/MIMIC-MJX) on Hugging Face
  (MIT), at a pinned revision: the rat's `rodent_reference_clips.h5` (842 clips of 5 s at 50 Hz, from DANNCE keypoints),
  the fly's `fly_reference_clip.h5` (treadmill walking, 500 Hz) and the stick insect's `stick_insect_mocap.h5` and model
  references (treadmill walking, 75 Hz). The joint trajectories are theirs (stac-mjx's inverse kinematics); none are
  retargeted here. About 3 GB of the dataset's 101 GB; downloaded by a script, not redistributed.
- **Bodies:** the rat's and the stick insect's models of [talmolab/vnl-playground](https://github.com/talmolab/vnl-playground)
  (BSD-3-Clause) and the fly of [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (Apache-2.0), as MIMIC-MJX uses
  them (the rat: 74 DoF, 38 torque actuators; the fly: 102 DoF, 61 torque actuators with the wings, proboscis and antennae
  off; the stick insect: 42 DoF), at pinned commits, downloaded by the same script. Contacts between the end effectors and
  the floor, as in MIMIC-MJX.
- **Simulator:** MuJoCo-Warp at MIMIC-MJX's rates (the rat and the stick insect: 100 Hz control, 2 ms steps; the fly:
  500 Hz control, 0.2 ms steps), the control step as one CUDA graph. No Isaac Lab, no MJX.
- **Baseline:** [talmolab/track-mjx](https://github.com/talmolab/track-mjx) (BSD-3-Clause), MIMIC-MJX's PPO, run unmodified
  with its published configuration on the same GPU, and its published numbers.

## Deliverables

- `tools/download_mimic_mjx.py` (the clips and the bodies at pinned revisions) and a loader of the clips
  (`ndm_gym/motion/sources/mimic_mjx.py`).
- The task, behind the library's interfaces: the bodies, the simulator, the tracking task (observation, goal, reward,
  termination), policies, both algorithms with the existing generalized-coordinate world model (`worldmodels/mjcf.py`),
  training and evaluation scripts, the hand task's systems options, and a per-phase time table of the training loop.
- Speed options of the G1 path that the MuJoCo-Warp trainers do not take yet (among them a world-model learning rate of its
  own, the goal table and the goal encoder run once per update), each behind a flag whose default keeps the behaviour, each
  timed against the path it replaces on the same GPU.
- Tests in `tests/` that run without a simulator where the library's do, and GPU tests for the simulator paths.
- A task page in `docs/`, `configs.py` entries, renders with the reference as a blue ghost, and the entries in `NOTICE` and
  the README's "Papers and repositories" for the sources above.

## What will be measured

Rules to be fixed here before the runs they apply to; the items marked "to fix" are open until the data audit.

- **Tasks:** one policy per body: the rat on its training clips (MIMIC-MJX's split; to fix: how the split is defined, which
  the paper does not state), the fly and the stick insect on their walking clips (to fix: which clips, after the audit).
- **Seeds:** three training seeds per configuration (0, 1, 2); mean and sample SD; the GPU and settings stated.
- **Run lengths:** every run capped at 1 h of training; a configuration that needs longer is treated as a bug to find.
- **Evaluation:** rollouts of the whole clip from perturbed starts, without exploration noise. Per body and configuration:
  the median keypoint error (mm) against the stac-mjx keypoints, as MIMIC-MJX reports it (the rat: about 10 mm on held-out
  clips), the joint-angle error, the root position and orientation errors, and the share of episodes ended early; for the
  rat, the same on the held-out clips.
- **Training time:** the training wall clock at the first checkpoint from which every later one has a median keypoint error
  below a threshold and no more than 5 % of episodes ended (to fix: the threshold per body), and iterations/s.
- **Against the baseline:** track-mjx's median keypoint error at the same wall clock (1 h) on the same GPU, beside its
  published time to convergence; not ranked with the library's table where its evaluator differs.
- **Reference:** the same rollouts with the reference's joint positions as PD targets, without a policy.

## Progress

- The data, the bodies and the baseline are public under the licences above and pinned (MIMIC-MJX revision `b4e53a72802f`,
  vnl-playground `c15480577636`, flybody `d015e9bfe441`, track-mjx `f4791c0b4d4e`).
- Next: the download script, the bodies loaded in MuJoCo-Warp and checked step for step against MuJoCo, the data audit
  that fixes the open items above.
