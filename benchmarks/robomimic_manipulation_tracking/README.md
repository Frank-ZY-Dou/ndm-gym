# Manipulation demonstration tracking: a Panda arm on robomimic's human demonstrations (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a plan:
no results yet.

## Task

A Franka Panda arm with its parallel gripper tracks human teleoperated demonstrations in physics: the arm's joints, the
gripper and the objects it handles, from the demonstration's own simulator states. SuperTrack and SuperMimic train the policy
through an MLP world model of the simulator, as in the library's other tracking tasks. It adds the library's first fixed-base
arm with a gripper, contact-rich manipulation with millimetre tolerances, and two arms handing an object over.

- **Demonstrations:** robomimic's proficient-human (PH) datasets (Mandlekar et al., CoRL 2021),
  [robomimic/robomimic_datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets) on Hugging Face (MIT),
  `v1.5/<task>/ph/demo_v15.hdf5`, at a pinned revision: 200 demonstrations per task, recorded at 20 Hz with robosuite 1.5.0.
  Each demonstration holds MuJoCo's flattened state at every control step (`states`: time, qpos, qvel) and its scene
  (`model_file`), so the objects' trajectories are the demonstration's own, not estimated. About 330 MB for the three
  tasks; downloaded by a script, not redistributed.
- **Scenes and robot:** each demonstration's `model_file`, with the meshes and textures of
  [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT) at the release the data were recorded with (v1.5.0),
  downloaded by the same script. The arm's torque actuators (80 N m on joints 1-5, 12 N m on 6-7), which robosuite drives
  by operational-space control, become joint position servos within the same torque limits, with gravity compensation on
  the robot's bodies (robosuite's controller compensates gravity); the gripper keeps its position servos. robosuite itself
  is not a dependency: no controller, no environment code.
- **Simulator:** MuJoCo-Warp at the demonstrations' rates (2 ms physics steps, 20 Hz control, 25 steps a control step),
  the control step as one CUDA graph. No Isaac Lab.

## Deliverables

- `tools/download_robomimic.py` (the demonstrations and robosuite's assets at pinned revisions) and a loader of a
  demonstration's scene and reference (`ndm_gym/motion/sources/robomimic.py`).
- The task, behind the library's interfaces: the scene with the servos, the simulator, the tracking task (observation,
  goal, reward, termination, success), policies, both algorithms with the existing generalized-coordinate world model
  (`worldmodels/mjcf.py`, which takes these scenes as they are: a fixed base, the objects as free joints), training and
  evaluation scripts, the hand task's systems options, and a per-phase time table of the training loop.
- Speed options of the G1 path that the MuJoCo-Warp trainers do not take yet, each behind a flag whose default keeps the
  behaviour, each timed against the path it replaces on the same GPU.
- Tests in `tests/` that run without a simulator where the library's do, and GPU tests for the simulator paths.
- A task page in `docs/`, `configs.py` entries, renders with the reference as a blue ghost, and the entries in `NOTICE` and
  the README's "Papers and repositories" for the sources above.

## What will be measured

Rules to be fixed here before the runs they apply to; the items marked "to fix" are open.

- **Tasks:** three tasks of rising difficulty, one policy per task:
  - **Lift**: one arm lifts a cube (demonstrations of 1.8-3.2 s);
  - **Square**: one arm puts a square nut onto its peg, millimetre clearances (5.4-11.8 s);
  - **Transport**: two arms hand a hammer over and move a piece of trash (18.7-35.7 s).
  To fix: which demonstrations (one per task, and a set for one policy on many). Lift draws the cube's size and mass per
  demonstration (about ±1 mm, ±5 g), so a policy on many of them needs per-world geometry or one scene.
- **Seeds:** three training seeds per configuration (0, 1, 2); mean and sample SD; the GPU and settings stated.
- **Run lengths:** every run capped at 1 h of training; a configuration that needs longer is treated as a bug to find.
- **Evaluation:** rollouts of the whole demonstration from perturbed starts, without exploration noise, as `evaluate_hand`
  does. Per task and configuration:
  - the objects' position (mm) and rotation errors against the demonstration;
  - the arm's joint error, the gripper's opening error;
  - the task's success, robosuite's criterion written as a function of the state (Lift: the cube above the table by 4 cm;
    Square: the nut on its peg; Transport: the hammer in its bin and the trash in the trash bin);
  - the share of episodes ended early.
- **Training time:** the training wall clock at the first checkpoint from which every later one has the task's success on
  every rollout and an object error below a threshold (to fix: the threshold per task), and iterations/s, as in the XHand
  study.
- **Reference:** the same rollouts with the demonstration's joint positions as the servos' targets, without a policy.

## Progress

- The data and the assets are public under the licences above and pinned (robomimic_datasets revision
  `74fa018461f4`, robosuite v1.5.0 `dc7fcf9fa6cd`).
- Each demonstration's `states` are MuJoCo's [time, qpos, qvel] of its `model_file` (Lift 1 + 16 + 15, Square 1 + 23 + 21,
  Transport 1 + 60 + 54), the arm's 7 joints and the gripper's 2 first, then the objects' free joints. The scenes load in
  MuJoCo 3.15 with robosuite's 73 asset files, and `worldmodels/mjcf.py`'s layout accepts them with and without the
  servos (no anchor, the objects as free joints).
- The demonstration's joint positions as the servos' targets, open loop, on MuJoCo (10 demonstrations a task): the arm
  follows them to 0.024 rad (RMS) on Lift and 0.040 rad on Square, but the objects are left behind: the cube ends 49 mm
  (median) from its demonstration on Lift, the square nut 302 mm on Square. The tracking policy has the task to do.
- On Transport (both arms, the hammer, the trash and four free bins) the same open-loop targets leave the hammer 935 mm
  (median) from its demonstration.
- MuJoCo-Warp 3.15 runs the three scenes with the servos: one control step (25 physics steps) from a demonstration's state
  agrees with MuJoCo's to 3e-7 (Lift), 3e-6 (Square) and 6e-7 (Transport) in qpos. A control step at 2048 worlds takes
  20 ms (Lift), 35 ms (Square) and 79 ms (Transport) on one RTX 4090, 0.10, 0.06 and 0.03 M env-steps/s (development
  readings, the GPU shared; at 512 worlds 21, 25 and 29 ms: the small scenes are bound by the steps' latency, not the worlds).
  The physics' steps are the iteration's largest part to be replaced by the world model's.
- Next: the download script, the loader and the data audit that fixes the open items above.
