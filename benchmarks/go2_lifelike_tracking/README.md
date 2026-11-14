# Quadruped motion tracking: a Unitree Go2 on a Labrador's motion capture (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a plan:
the numbers under "Progress" are preliminary, from one training seed, and are not results.

## Task

A Unitree Go2 tracks the motion capture of a Labrador retriever: walking, trotting, running, turning, backing, jumping and
playing. SuperTrack and SuperMimic train the policy through an MLP world model of the simulator, as in the library's other
tracking tasks. It adds a legged floating base other than the G1 to the library, with flight phases and fast gaits.

- **Motion capture:** the 33 BVH files (16.3 min at 120 fps) of "Lifelike agility and play in quadrupedal robots using
  reinforcement learning and generative pre-trained models" (Han et al., Nature Machine Intelligence 2024),
  [Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play),
  `data/raw_mocap_data`, at a pinned commit. Its licence file is MIT; its README says the code and data are for research
  purposes only. The files are downloaded by a script and are not redistributed here.
- **Robot:** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_go2` (BSD-3-Clause), at a
  pinned commit, downloaded by the same script. Its motors become joint position servos within the motors' torque limits.
- **Retargeting:** [supermimic-quadruped](https://github.com/Houserz/supermimic-quadruped)'s `quadmotion` package (MIT),
  called from the user's own checkout at a pinned commit; none of its code is copied here. Frames that fail its checks
  (an unreachable foot, a joint at its limit, the trunk or a knee in the floor, ...) are cut; each run of valid frames of
  at least 2 s is a clip. With its default checks: 65 clips, 488 s at the robot's time scale.
- **Simulator:** MuJoCo-Warp, 50 Hz control, 10 substeps, the control step as one CUDA graph. No Isaac Lab.

## Deliverables

- `tools/download_lifelike.sh` and `tools/convert_lifelike.py` (download at pinned commits; retarget).
- The task, behind the library's interfaces: the robot and its kinematics, the clip loader, the simulator, the tracking task
  (observation, goal, reward, termination), policies, both algorithms with the existing generalized-coordinate world model
  (`worldmodels/mjcf.py`), training and evaluation scripts, and the systems options of the hand task (`--no_sync`,
  the graphed collection and updates, ...), off by default.
- Tests in `tests/` that run without a simulator where the library's do, and GPU tests for the simulator paths.
- A task page in `docs/`, `configs.py` entries, renders with the reference as a blue ghost, and the entries in `NOTICE`,
  `THIRD_PARTY_NOTICES.md` and the README's "Papers and repositories" for the three sources above.

## What will be measured

Rules to be fixed here before the runs they apply to; the items marked "to fix" are open.

- **Clips:** three single-clip tasks of different gaits (a walk, a run, a play clip; to fix: which), and one policy on all
  clips.
- **Seeds:** three training seeds per configuration; mean and sample SD; the GPU and settings stated.
- **Evaluation:** rollouts of the whole clip from perturbed starts, without exploration noise, as `evaluate_hand` does. Per
  configuration: the normalized return, the failed clips, the feet's and the trunk's position errors (mm), the joint-angle
  error, and the failed clips under trunk pushes, following the G1 LAFAN1 study's table.
- **Training time:** the training wall clock at the first checkpoint from which every later one has no failed clip and a
  position error below a threshold (to fix: the threshold), and iterations/s, as in the XHand study.
- **Reference:** the same rollouts with the reference's joint positions as the position targets, without a policy.

## Progress (preliminary, seed 0, one RTX 6000 Ada)

- The download and the conversion run; the retargeted clips have no joint-limit violation and no penetration above 5 mm.
- The task trains with both algorithms. On one walk clip (6 s), with the systems options, 6,000 iterations take 6.6 min
  (SuperTrack, 15.1 iterations/s) and 8.0 min (SuperMimic, 12.4 iterations/s); 256 perturbed rollouts all complete, with
  the trunk within 11.5 and 9.2 mm on average. Without the options the loop runs 3.9 and 3.1 iterations/s.
- One SuperTrack policy on all 65 clips (without the systems options, 28 min) completes 63 clips from their first frame;
  the reference's joint positions as targets complete 3.
- Open: the fastest run clip is not kept up with (under study: the model's joint damping against the motors' torque
  limits at the reference's joint speeds); the evaluation under pushes; multi-clip runs with the systems options.
