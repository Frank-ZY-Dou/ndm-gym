# Human motion tracking: a scaled OpenSim skeleton on force-plate motion capture (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a
plan; results will follow in this directory.

## Task

A human skeleton tracks one person's motion capture, with SuperTrack and SuperMimic through an MLP world model, and the
ground reaction forces of the simulated feet are compared with the force plates' measurements. The body is the person's
own: an OpenSim full-body model scaled to the subject, without muscles, each joint a position servo. The library's
muscle-actuated character (MuscleVAE's, 284 muscles) tracks LAFAN1 retargeted to a generic skeleton; this task adds a
subject-specific human body with measured kinematics and measured forces, so that a tracking result can be checked
against a physical quantity that the reward does not contain.

- **Motion capture:** subject `s001` of GroundLink (Han et al., SIGGRAPH Asia 2023) as the AddBiomechanics dataset
  (Werling et al., ECCV 2024) processes it (`Han2023`, with arms): marker-based motion capture at 250 Hz on level ground
  with five force plates, with the per-foot ground reaction force and centre of pressure of every frame. No treadmill.
  17 clips, 113 s: tennis ground strokes and a serve, walks, two warrior poses, soccer kicks, small ballet jumps, a
  jumping jack and tai chi. The frames in which a foot is off the plates are flagged by the dataset and are left out of
  every force comparison.
- **Body:** the Rajagopal full-body model (Rajagopal et al., IEEE TBME 2016) as AddBiomechanics scaled it to the subject
  and stores it in each of its files, converted to MJCF by a converter of this library: 20 bodies, 23 actuated joints
  (hip 3, knee, ankle per leg; lumbar 3; shoulder 3, elbow, forearm rotation per arm) on a free pelvis, the knee's
  coupled translations and rotations as equality constraints, the coordinates that the model locks (subtalar, toes,
  wrist) welded, no muscles. The dataset scaled the model in each of the subject's files separately, so the same person
  has four slightly different skeletons here (thigh 40.5 to 46.0 cm, 71.3 to 73.6 kg) of one topology; a clip keeps its
  file's skeleton.
- **Feet:** the OpenSim model has no contact geometry: five spheres per foot (two at the heel, two at the metatarsal
  heads, one at the hallux), their heights fitted to the measured forces and centres of pressure of the subject (in
  stance the lowest sphere at the floor, in swing above it), on clips other than those that check the fit.
- **Actuation (to fix before the runs):** position servos with the gains, damping and force ranges of dm_control's
  position-controlled CMU humanoid (`cmu_humanoid.py`, `_POSITION_ACTUATORS_V2020`, a 70 kg human) as the starting
  point; the torque limits no lower than the subject's inverse-dynamics peaks on these clips; a physics step of 2 ms and
  50 Hz control. Any departure from the cited values is reported beside them.
- **Simulator:** MuJoCo-Warp alone, 50 Hz control, the control step as one CUDA graph; no Isaac Lab. The floating-base
  path of the humanoid tracking task (#12) with the world model of generalized coordinates (`worldmodels/mjcf.py`).

## Deliverables

- The download (the chosen files of the dataset's archive by range requests, checked against hashes) and the conversion
  of the clips and of the scaled models, run on the user's machine; no data and no model in this repository.
- The calibration of the feet and its check against the measured forces, as a tool with its report.
- The task behind the library's interfaces: the skeleton and its kinematics (joints of several coordinates, coupled
  coordinates), the clip loader, the tracking task (observation, goal, reward and termination following the humanoid
  task's terms), training and evaluation scripts, both algorithms with the existing world model, the four skeletons in
  one batch.
- The force evaluation: the simulated per-foot ground reaction force against the measured one.
- Tests in `tests/` that run without a simulator where the library's do, and GPU tests for the simulator paths.
- A task page in `docs/`, `configs.py` entries, renders with the reference laid over the body, and the entries in
  `NOTICE`, `THIRD_PARTY_NOTICES.md` and the README's "Papers and repositories".

## What will be measured

Rules to be fixed here before the runs they apply to; the items marked "to fix" are open.

- **Tasks:** one clip (a walk, a tennis ground stroke, a ballet jump); one policy on all 17 clips. Clips held out of
  training: to fix (other takes of the same motions, and one motion kind left out).
- **Algorithms:** SuperTrack and SuperMimic, each from the humanoid task's selected options as the starting point, then
  any change that improves on them, reported beside the starting point.
- **Seeds:** three training seeds per configuration; mean and sample SD; the GPU and settings stated; one run per GPU.
- **Tracking:** rollouts of every whole clip from perturbed starts, without exploration noise, as the humanoid task's:
  the success (a rollout that never terminates), the normalized return, MPJPE in the world and about the pelvis, the
  joint-angle error, and the same under pushes. Beside them, the reference's joint positions as the position targets,
  open loop.
- **Forces:** on the frames with valid force-plate data, per foot: the ratio of the simulated to the measured vertical
  ground reaction force and their correlation over a clip, the horizontal force's, and the distance between the simulated
  and the measured centre of pressure. Reported for the policies and for the open-loop targets; not a term of the reward.
  No external force or torque acts on the pelvis in training or in evaluation.
- **Training time:** the wall clock from which every later checkpoint holds a target (to fix: the success and the error
  of the target), and iterations/s.
- **Length:** a run is expected to take minutes; a configuration that needs more than an hour is treated as a defect.

Not in the first version: muscles; other subjects and one policy over several people; treadmill recordings; the clips
with the hands on the floor; gains as a part of the action (variable impedance).

## Sources and licences

- **AddBiomechanics dataset** ([download page](https://addbiomechanics.org/download_data.html)): "available for free under
  a Creative Commons BY 4.0 license"; Werling et al., "AddBiomechanics Dataset: Capturing the Physics of Human Motion at
  Scale", ECCV 2024. Its `Han2023` part is processed from **GroundLink** (Han et al., "GroundLink: A Dataset Unifying
  Human Body Movement and Ground Reaction Dynamics", SIGGRAPH Asia 2023); GroundLink's own data page states no licence
  for the data, and its repository's MIT licence is worded for the software. The data are used under the terms that
  AddBiomechanics states, downloaded by the user and not redistributed here.
- **Rajagopal full-body model** ([SimTK](https://simtk.org/projects/full_body)): MIT, "Copyright (c) 2015, Stanford
  University"; Rajagopal et al., "Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait", IEEE TBME
  2016. The scaled copies inside the dataset's files are converted on the user's machine.
- **dm_control** ([google-deepmind/dm_control](https://github.com/google-deepmind/dm_control), Apache-2.0): the values of
  its position-controlled CMU humanoid's actuators; no file is copied.
- **nimblephysics** (BSD-2-Clause) reads the dataset's files in the conversion tool, on the user's machine.
