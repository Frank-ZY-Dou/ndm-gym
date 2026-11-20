# Manipulation demonstration tracking: a Panda arm on robomimic's human demonstrations

The task and the rules of its benchmark ([the task](../../docs/tasks/robomimic.md), [results](results.md)).

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

## What the task adds

- `tools/download_robomimic.py` (the demonstrations and robosuite's assets at pinned revisions, each file checked against
  its SHA-256) and a loader of a demonstration's scene and reference (`ndm_gym/motion/sources/robomimic.py`).
- The scene with the servos (`ndm_gym/robots/robomimic_scenes.py`), the simulator (`ndm_gym/sim/rigid/robomimic_sim.py`),
  the tracking task (observation, goal, reward, termination, success; `ndm_gym/tasks/robomimic_tracking.py`), the policy,
  both algorithms with the library's generalized-coordinate world model (`worldmodels/mjcf.py`, which takes these scenes as
  they are: a fixed base, the objects as free joints), training and evaluation with the hand task's systems options.
- Tests in `tests/` that run on the CPU, and a GPU test of the simulator against MuJoCo.
- A task page in `docs/`, `configs.py` entries, renders with the reference as a blue ghost, and the entries in `NOTICE` and
  the README's "Papers and repositories" for the sources above.

## Rules

- **Tasks:** three tasks of rising difficulty, one policy per task, on each task's demonstration 0:
  - **Lift**: one arm lifts a cube (demonstrations of 1.8-3.2 s);
  - **Square**: one arm puts a square nut onto its peg, millimetre clearances (5.4-11.8 s);
  - **Transport**: two arms hand a hammer over and move a piece of trash (18.7-35.7 s).
- **Algorithms:** SuperTrack and SuperMimic from the library's hand-task options (`configs.py`: `robomimic-supertrack`,
  `robomimic-supermimic`), the same task defaults, budget and evaluation for both; no tuning on these tasks.
- **Seeds:** three training seeds per configuration (0, 1, 2); mean and sample SD; the GPU and settings stated.
- **Budget:** 1,500 iterations for both algorithms; every run is expected to take minutes, and one that needs more than an
  hour is treated as a defect.
- **Evaluation:** `evaluate_robomimic` on the last checkpoint: 256 rollouts of the whole demonstration from perturbed
  starts, without exploration noise. Per task and configuration:
  - the objects' position (mm) and rotation errors against the demonstration;
  - the arm's joint error, the gripper's opening error;
  - the success: the rollout never terminates and meets robosuite's success test at the demonstration's last frame,
    written as a function of the state (Lift: the cube above the table by 4 cm; Square: the nut on its peg; Transport:
    the hammer in its bin and the trash in the trash bin), a tracking proxy for robomimic's success;
  - the share of episodes ended early.
- **Time to the target:** the training wall clock at the first kept checkpoint (every 500 iterations) from which every
  later one has a success of at least 0.99 with the objects within 10 mm (the hand task's target), and iterations/s.
- **Reference:** the same rollouts with the demonstration's joint positions as the servos' targets, without a policy.

## Data audit

- The data and the assets are public under the licences above and pinned (robomimic_datasets revision
  `74fa018461f4`, robosuite v1.5.0 `dc7fcf9fa6cd`).
- Each demonstration's `states` are MuJoCo's [time, qpos, qvel] of its `model_file` (Lift 1 + 16 + 15, Square 1 + 23 + 21,
  Transport 1 + 60 + 54), the arm's 7 joints and the gripper's 2 first, then the objects' free joints. The scenes load in
  MuJoCo 3.15 with robosuite's 73 asset files, and `worldmodels/mjcf.py`'s layout accepts them with and without the
  servos (no anchor, the objects as free joints).
- Lift draws the cube's size and mass per demonstration (about ±1 mm, ±5 g), so a policy on several of its demonstrations
  needs per-world geometry or one scene.
- The demonstration's joint positions as the servos' targets, open loop, on MuJoCo (10 demonstrations a task): the arm
  follows them to 0.024 rad (RMS) on Lift and 0.040 rad on Square, but the objects are left behind: the cube ends 49 mm
  (median) from its demonstration on Lift, the square nut 302 mm on Square, the hammer 935 mm on Transport.
- MuJoCo-Warp 3.15 runs the three scenes with the servos: one control step (25 physics steps) from 32 states along
  demonstration 0 agrees with MuJoCo's in qpos to 4e-7 (Lift), 5e-6 (Square) and 6e-5 (Transport) (the largest entry's
  difference, median over the states; the largest over all of them 2e-4, 1e-3 and 8e-4).
