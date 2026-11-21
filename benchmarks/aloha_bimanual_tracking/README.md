# Bimanual manipulation tracking: ALOHA's two arms on ACT's simulated tasks (plan)

This file claims the task (CONTRIBUTING.md, "Claiming a task") and fixes what will be measured before the runs. It is a
plan; results will follow in this directory.

## Task

The two arms of ALOHA (Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023,
arXiv 2304.13705) track demonstrations of ACT's two simulated tasks, with SuperTrack and SuperMimic through the
library's MLP world model of generalized coordinates. Manipulation demonstration tracking (#10) has one Panda arm on two
tasks and two on a hand-over; this task adds two six-joint arms that work on one object at the same time: a hand-over in
the air, and a peg that one arm inserts into a socket that the other arm holds.

- **Tasks:**
  - Transfer Cube: the right arm picks a 4 cm cube up from the table and hands it to the left arm in the air.
  - Insertion: the left arm picks up a socket and the right arm a peg (2 cm across, the socket's opening 3.6 cm), and
    the peg goes into the socket in the air until it touches the pin inside.
- **Demonstrations:** generated on the user's machine with ACT's scripted policies, as ACT records them
  (`record_sim_episodes.py`): the policy's gripper waypoints are followed in the end-effector scene, and the resulting
  joint positions with the gripper commands are replayed as position targets in the joint-controlled scene. Here the
  full MuJoCo state of every control step of that replay is kept: the arms, the fingers and the objects' poses. 400
  control steps at 50 Hz (8 s). A demonstration is kept only if ACT's reward reaches its maximum of 4 in the replay. The
  objects start where ACT draws them (the cube in 20 x 20 cm of the table; the peg and the socket in 10 x 20 cm each),
  from a seeded generator.
- **The public recordings are not used.** ACT's recordings of these tasks (`sim_transfer_cube_*`, `sim_insertion_*`; on
  the Hugging Face Hub as `lerobot/aloha_sim_*`, 50 episodes each) hold the arms' joint positions, the actions and a
  camera image, and not the objects' poses, so a rollout cannot be compared with them object by object.
- **Scenes:** ACT's MJCF files (`bimanual_viperx_transfer_cube.xml`, `bimanual_viperx_insertion.xml`, and their
  end-effector versions for the generation) and meshes, downloaded at a pinned revision and checked against SHA-256
  sums. Two ViperX 300 arms: six joints and two finger slides each, 16 position actuators with ACT's gains and force
  ranges; a free cube of 50 g; a free peg and a free socket of 50 g each.
- **Action:** 14 values, as ACT's: six joint position targets and one gripper command per arm, the two fingers mirrored
  (to fix: as offsets from the demonstration's targets, as the manipulation task has them).
- **Simulator:** MuJoCo-Warp alone at ACT's 50 Hz control with the scenes' own physics step, the control step as one
  CUDA graph; the path of the manipulation tracking task (#10) with the world model of generalized coordinates
  (`worldmodels/mjcf.py`). A test compares a control step with MuJoCo's.

## Deliverables

- The tool that fetches ACT's scenes (pinned, checked against hashes) and generates the demonstrations; no data and no
  scene in this repository.
- The task behind the library's interfaces: the scenes, the kinematics, the tracking task (observation, goal, reward and
  termination following the manipulation task's terms), training and evaluation scripts, both algorithms with the
  existing world model.
- Tests in `tests/` that run on the CPU, and a GPU test of the simulator against MuJoCo.
- A task page in `docs/`, `configs.py` entries, renders with the demonstration laid over the scene, and the entries in
  `NOTICE`, `THIRD_PARTY_NOTICES.md` and the README's "Papers and repositories".

## What will be measured

Rules fixed here before the runs they apply to. The items marked "to fix" are open; each is fixed in this file by a
commit before the reported runs start.

- **Tasks:** Transfer Cube and Insertion, one policy per demonstration (each task's demonstration 0, the generator's
  first kept episode). One policy on several demonstrations, and initial poses held out of training: to fix.
- **Algorithms:** SuperTrack and SuperMimic, treated alike: the same starting point (the manipulation task's
  configurations), the same seeds, budget, checkpoints and evaluation starts. If the starting point is changed, both
  algorithms get a tuning round of the same number of runs and the same budget, on demonstrations that the reported runs
  do not use, and the round is reported with its runs.
- **Stopping:** one rule for both algorithms, with a cap in iterations (to fix: the target held over consecutive
  checkpoints, or a plateau of the objects' error). The last checkpoint is the one evaluated.
- **Seeds:** training seeds 0, 1, 2 per configuration, every seed reported; mean and sample SD.
- **Evaluation:** 256 rollouts of the whole demonstration from perturbed starts, without exploration noise. Beside them,
  the demonstration's own position targets, open loop, from the same starts.
- **Measures:** the success, the objects' position and rotation error, the joints' error and the share of the rollouts
  that end early; SuperTrack first, both numbers side by side. The success is ACT's test of a task's final stage (the
  cube in the left gripper and off the table; the peg touching the pin), read at the demonstration's last frame or at
  any step (to fix), and named a proxy where it is not ACT's success of a learned policy.
- **Time:** each run alone on one A100 (80 GB). The clock starts at the first training iteration, and the text says what
  it includes. A checkpoint is kept every 100 iterations; the time to the target is that of the first checkpoint from
  which every later one holds the target (to fix: the target's success and error). Without a per-phase time table of
  the path: time to the target, not engineered.
- **Length:** a run is expected to take minutes.

Not in the first version: ALOHA's real recordings; the public simulated recordings (no object poses); image
observations; ACT or another imitation policy trained here (a published success rate is quoted at most, as another
protocol's number).

## Sources and licences

- **ACT** ([tonyzhaozh/act](https://github.com/tonyzhaozh/act), MIT, "Copyright (c) 2023 Tony Z. Zhao"): its scenes and
  meshes, its scripted policies and its reward tests; Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware", RSS 2023. The scenes and meshes are downloaded by the user at a pinned revision and are not
  redistributed here.
