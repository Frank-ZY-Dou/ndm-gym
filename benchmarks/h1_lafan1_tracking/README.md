# Humanoid motion tracking: the Unitree H1 and H1-2 on LAFAN1 (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a
plan; results will follow in this directory.

## Task

The Unitree H1, and then the H1-2, track the 40 LAFAN1 clips that the library's G1 tracks, with SuperTrack and SuperMimic
through an MLP world model, as in the G1 / LAFAN1 benchmark. The same motions on other bodies: the H1 is taller and heavier
than the G1 (51.6 kg in its description against the G1's 33.3 kg) and has one ankle joint per leg (pitch, no roll);
the H1-2 has two, and wrists. It adds humanoid embodiments other than the G1 to the library.

| | joints | legs | arms | mass (description) |
|---|---|---|---|---|
| G1 (in the library) | 29 | 6 per leg (two at the ankle) | shoulder 3, elbow, wrist 3 | 33.3 kg |
| H1 | 19 | 5 per leg (one at the ankle) | shoulder 3, elbow | 51.6 kg |
| H1-2 | 27 | 6 per leg (two at the ankle) | shoulder 3, elbow, wrist 3 | 67.0 kg |

- **Motions:** the H1 and H1-2 retargets of LAFAN1 in the dataset the library already reads,
  [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (`h1/`,
  `h1_2/`, 40 clips each at 30 fps, at the revision `tools/download_data.sh` pins). LAFAN1 (Ubisoft La Forge) and its
  retargets are CC BY-NC-ND 4.0; they are downloaded by a script, checked against hashes and not redistributed here. The
  retargets are kinematic (the dataset's README: no dynamics, no actuator limits).
- **Robots:** the dataset's H1 and H1-2 descriptions (`robot_description/h1`, `robot_description/h1_2`), Unitree's
  ([unitree_ros](https://github.com/unitreerobotics/unitree_ros), `robots/h1_description`, `robots/h1_2_description`,
  BSD-3-Clause), converted to MJCF as the G1's is; the motors as joint position servos within the descriptions' torque
  limits.
- **Simulator:** MuJoCo-Warp, 50 Hz control, the control step as one CUDA graph; no Isaac Lab. The floating-base path of
  the quadruped tracking task (#4: simulator, MJCF world model in generalized coordinates, batched evaluation, the hand
  task's systems options) with a humanoid's task.

## Deliverables

- The download (pinned revision, hashes) and the conversion of the H1's and the H1-2's clips and descriptions.
- The task behind the library's interfaces: the robots and their kinematics, the clip loader, the tracking task
  (observation, goal, reward, termination following the G1 / LAFAN1 task's terms), training and evaluation scripts, both
  algorithms with the existing generalized-coordinate world model (`worldmodels/mjcf.py`), the systems options off by
  default.
- Tests in `tests/` that run without a simulator where the library's do, and GPU tests for the simulator paths.
- A task page in `docs/`, `configs.py` entries, renders with the reference laid over the robot, and the entries in
  `NOTICE`, `THIRD_PARTY_NOTICES.md` and the README's "Papers and repositories".

## What will be measured

Rules to be fixed here before the runs they apply to; the items marked "to fix" are open.

- **Tasks:** one policy on all 40 clips per robot; LAFAN1's own split held out (`benchmarks/g1_lafan1_tracking/splits`)
  for clips not seen in training.
- **Algorithms:** SuperTrack and SuperMimic, each from the library's selected options (the G1's and the quadruped task's
  as the starting points), then any change that improves on them, reported beside the starting point.
- **Seeds:** three training seeds per configuration; mean and sample SD; the GPU and settings stated; one run per GPU.
- **Evaluation:** rollouts of every whole clip from perturbed starts, without exploration noise, as the G1's: the
  normalized return, the failed clips, MPJPE in the world and the root frame, the joint-angle error, and the failed clips
  under pushes. Beside them, the reference's joint positions as the position targets, open loop.
- **Comparison with the G1:** the G1 on the same MuJoCo-Warp path, the same clips and evaluation, so that the three bodies
  compare under one simulator (the library's G1 numbers come from Isaac Lab and are not compared directly).
- **Training time:** the wall clock from which every later checkpoint holds a target (to fix: the normalized return and
  the failed clips of the target), and iterations/s.
- **Length:** a run is expected to take minutes; a configuration that needs more than an hour is treated as a defect.
