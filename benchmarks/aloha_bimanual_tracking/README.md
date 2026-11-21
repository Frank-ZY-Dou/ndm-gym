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
  from a seeded generator. `tools/make_aloha_demos.py` writes ten demonstrations per task and checks their SHA-256.
- **MuJoCo 2.3.7 for the generation.** The tool runs with the version that ACT's instructions install, in an
  environment of its own. With the library's MuJoCo 3.15 the grippers follow their mocap bodies differently in the
  end-effector scene and the scripted grasps miss (Transfer Cube: 1 of 12 episodes reaches the reward of 4, Insertion
  none; with 2.3.7, 12 of 12 each). The joint-controlled scene, which the task simulates, differs less: 2.3.7's targets
  replayed open loop in 3.15 reach the reward of 4 in 12 of 12 Transfer Cube episodes and 4 of 12 Insertion episodes.
- **The public recordings are not used.** ACT's recordings of these tasks (`sim_transfer_cube_*`, `sim_insertion_*`; on
  the Hugging Face Hub as `lerobot/aloha_sim_*`, 50 episodes each) hold the arms' joint positions, the actions and a
  camera image, and not the objects' poses, so a rollout cannot be compared with them object by object.
- **Scenes:** ACT's MJCF files (`bimanual_viperx_transfer_cube.xml`, `bimanual_viperx_insertion.xml`, and their
  end-effector versions for the generation) and meshes, downloaded at a pinned revision and checked against SHA-256
  sums. Two ViperX 300 arms: six joints and two finger slides each, 16 position actuators with ACT's gains and force
  ranges; a free cube of 50 g; a free peg and a free socket of 50 g each.
- **Action:** 14 values, as ACT's: six joint position targets and one gripper command per arm, the two fingers mirrored;
  each an offset about the reference's joint position at the next frame, as the manipulation task has them (at most
  0.25 rad on a joint, 2 cm on a finger). The policy does not get the position targets that a demonstration recorded
  (ACT's actions); the evaluation replays them open loop beside the policy.
- **Simulator:** MuJoCo-Warp alone at ACT's 50 Hz control with the scenes' own physics step (2 ms, 10 steps a control
  step), the control step as one CUDA graph, the solver at 20 iterations as in the manipulation task; the path of the
  manipulation tracking task (#10) with the world model of generalized coordinates (`worldmodels/mjcf.py`). A test
  compares a control step with MuJoCo's.

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

Rules fixed here before the runs they apply to.

- **Tasks:** Transfer Cube and Insertion, one policy per demonstration: each task's demonstration 0 (the generator's
  first kept episode). One policy on several demonstrations, and start poses held out of training, are not in this
  version.
- **Algorithms:** SuperTrack and SuperMimic, treated alike: the same starting point (the manipulation task's
  configurations, `robomimic-supertrack` and `robomimic-supermimic`), the same seeds, budget, checkpoints and
  evaluation starts.
- **Tuning round:** the starting point is changed (its budget of 1,500 iterations is the manipulation task's, at 20 Hz),
  so both algorithms get one round of the same runs at the same budget, on demonstration 1 of each task with seed 0 (the
  reported runs use demonstration 0). Four variants per algorithm and task, each 6,000 iterations with the learning
  rate's cosine over them:
  1. nothing else changed;
  2. the imagined window of 40 control steps (0.8 s: the manipulation task's 16 steps at its 20 Hz);
  3. episodes of up to 500 control steps (10 s: the manipulation task's 200 steps at its 20 Hz);
  4. both.

  Per algorithm the variant with the highest success at its last checkpoint, averaged over the two tasks, is taken;
  among those within 0.02 of it the one with the lowest objects' position error (averaged over the tasks, over the
  rollouts that do not terminate), then the one with the fewest changes. The round is reported with every run.
- **Budget and stopping:** one rule for both algorithms. Every reported run trains for the round's 6,000 iterations
  (the cap) and its last checkpoint is the one evaluated. A run has settled if its last five kept checkpoints (the last
  1,000 iterations) agree: their successes within 0.05 of one another and their objects' errors within 1 mm or a
  quarter of their mean, whichever is larger. The results say per run whether it settled; where runs of a case did
  not, its numbers are named the cap's and no comparison is drawn from them.
- **Seeds:** training seeds 0, 1, 2 per configuration, every seed reported; mean and sample SD.
- **Evaluation:** 256 rollouts of the whole demonstration from perturbed starts, without exploration noise. Beside them
  the same rollouts without a policy: under the reference's joint positions as the targets, and under the position
  targets the demonstration recorded, open loop.
- **Success:** ACT scores a state by the geoms in contact (`get_reward` of its `TransferCubeTask` and `InsertionTask`,
  stages 0 to 4; the last: the cube touches the left gripper's finger and not the table; the peg touches the pin) and
  counts an episode a success if its highest reward is 4. The evaluation reads MuJoCo-Warp's contacts after every
  control step for the same pairs of geoms; a rollout succeeds if it never terminates and its highest reward is 4. The
  share of the rollouts that reach each stage is reported with it. These rollouts track one demonstration from
  perturbed starts: the number is not ACT's success rate of an imitation policy from images.
- **Final state:** beside ACT's test, the last stage as a function of the state at the demonstration's last frame, as
  the manipulation task reads its success: the cube off the table at the left gripper's grasp point with the gripper
  closed on it; the peg inside the socket's opening with its end within 2 mm of the pin's. ACT's test asks for the
  touch itself, which the demonstrations' recorded targets, replayed in this simulator, reach in 5 of the 10 Insertion
  demonstrations; the state test says how many rollouts end inserted.
- **Measures:** the success, the stages and the final state, the objects' position and rotation error, the joints'
  error and the share of the rollouts that end early (a tracked object 15 cm or a gripper keypoint 20 cm from the
  reference); SuperTrack first, both numbers side by side.
- **Time:** each run alone on one A100 (80 GB). The clock starts at the first training iteration, and the text says what
  it includes. A checkpoint is kept every 200 iterations; the time to the target is that of the first checkpoint from
  which every later one has a success of at least 0.99 with the objects within 10 mm (the manipulation task's target).
  Without a per-phase time table of the path: time to the target, not engineered.
- **Length:** a run is expected to take minutes.

Not in the first version: ALOHA's real recordings; the public simulated recordings (no object poses); image
observations; ACT or another imitation policy trained here (a published success rate is quoted at most, as another
protocol's number).

## Sources and licences

- **ACT** ([tonyzhaozh/act](https://github.com/tonyzhaozh/act), MIT, "Copyright (c) 2023 Tony Z. Zhao"): its scenes and
  meshes, its scripted policies and its reward tests; Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware", RSS 2023. The scenes and meshes are downloaded by the user at a pinned revision and are not
  redistributed here.
