# Bimanual demonstration tracking: ALOHA's two arms on ACT's simulated tasks

The task and the rules of its benchmark ([the task](../../docs/tasks/aloha.md), [results](results.md)).

## Task

The two arms of ALOHA (Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023,
arXiv 2304.13705) track demonstrations of ACT's two simulated tasks, with SuperTrack and SuperMimic through the
library's MLP world model of generalized coordinates. Manipulation demonstration tracking has one Panda arm on two
tasks and two on a hand-over; this task adds two six-joint arms that work on one object at the same time: a hand-over in
the air, and a peg that one arm inserts into a socket that the other arm holds.

- **Tasks:**
  - Transfer Cube: the right arm picks a 4 cm cube up from the table and hands it to the left arm in the air.
  - Insertion: the left arm picks up a socket and the right arm a peg (2 cm across, the socket's opening 3.6 cm), and
    the peg goes into the socket in the air until it touches the pin inside.
- **Demonstrations:** generated on the user's machine with ACT's scripted policies, as ACT records them
  (`record_sim_episodes.py`): the policy's gripper waypoints are followed in the end-effector scene, and the resulting
  joint positions with the gripper commands are replayed as position targets in the joint-controlled scene. The full
  MuJoCo state of every control step of that replay is kept: the arms, the fingers and the objects' poses. 400 control
  steps at 50 Hz (8 s). A demonstration is kept only if ACT's reward reaches its maximum of 4 in the replay. The objects
  start where ACT draws them (the cube in 20 x 20 cm of the table; the peg and the socket in 10 x 20 cm each), from a
  seeded generator. `tools/make_aloha_demos.py` writes ten demonstrations per task and checks their SHA-256.
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
  manipulation tracking task with the world model of generalized coordinates (`worldmodels/mjcf.py`).

## What the task adds

- `tools/make_aloha_demos.py`: ACT's scenes at a pinned commit, each file checked against its SHA-256, and the
  demonstrations, whose arrays' SHA-256 is pinned too; a loader (`ndm_gym/motion/sources/aloha.py`). No data and no scene
  in this repository.
- ACT's scenes for the tracking task (`ndm_gym/robots/aloha_scenes.py`). The manipulation task's simulator, task,
  algorithms, training and evaluation take them with the scene's own control step, a gripper's two fingers under one
  action, the position targets a demonstration recorded as an open-loop reference, and ACT's reward read off the
  simulator's contacts (`RobomimicSim.touching`, `RobomimicTask.staged_reward`); the manipulation task's own runs are
  unchanged.
- Tests in `tests/test_aloha_tracking.py` that run on the CPU, and GPU tests of the simulator against MuJoCo and of the
  contacts' reward against the demonstrations' recorded one.
- A task page in `docs/`, `configs.py` entries, renders with the demonstration laid over ACT's scene, and the entries in
  `NOTICE`, `THIRD_PARTY_NOTICES.md` and the README's "Papers and repositories".

## Rules

Fixed before the runs they apply to.

- **Tasks:** Transfer Cube and Insertion, one policy per demonstration. From the third round on, a task's demonstration
  is the first whose recorded targets, replayed open loop on MuJoCo-Warp from its own start, reach ACT's reward of 4
  (the data audit): Transfer Cube's demonstration 0 and Insertion's demonstration 4. The runs before it tracked
  demonstration 0 of both tasks (the generator's first kept episode); Insertion's demonstration 0 replays to the reward
  of 3 there, so a success on it partly measures the difference between MuJoCo 2.3.7 and MuJoCo-Warp (the pull
  request's review). One policy on several demonstrations, and start poses held out of training, are not measured.
- **Algorithms:** SuperTrack and SuperMimic, treated alike: the same starting point (the manipulation task's
  configurations, `robomimic-supertrack` and `robomimic-supermimic`), the same seeds, budget, checkpoints and
  evaluation starts.
- **Tuning round:** the starting point's budget of 1,500 iterations is the manipulation task's, at 20 Hz, so both
  algorithms get one round of the same runs at the same budget, on demonstration 1 of each task with seed 0 (the
  reported runs track demonstration 0). Four variants per algorithm and task, each 6,000 iterations with the learning
  rate's cosine over them:
  1. nothing else changed;
  2. the imagined window of 40 control steps (0.8 s: the manipulation task's 16 steps at its 20 Hz);
  3. episodes of up to 500 control steps (10 s: the manipulation task's 200 steps at its 20 Hz);
  4. both.

  Per algorithm the variant with the highest success at its last checkpoint, averaged over the two tasks, is taken;
  among those within 0.02 of it the one with the lowest objects' position error (averaged over the tasks, over the
  rollouts that do not terminate), then the one with the fewest changes. The round is reported with every run.
- **Second round:** at the first round's 6,000 iterations some run of every case had not settled (below;
  [results](results.md)). One more round follows, as the first: on demonstration 1 of each task with seed 0, the same
  runs for both algorithms, from the first round's choice (episodes of up to 500 control steps). Four variants per
  algorithm and task, the learning rate's cosine over each one's iterations, a checkpoint every 200:
  1. 12,000 iterations;
  2. 12,000 iterations with the policy's average (`--pi_ema 0.995`, the value of the hand task's fast configurations:
     the checkpoints hold the average and the evaluation plays it);
  3. 24,000 iterations;
  4. 24,000 iterations with the policy's average.

  One variant is taken for both algorithms, so that their budgets stay the same: the one with the most settled runs
  among its four (two algorithms, two tasks); among those, the one with the highest success at its last checkpoint
  averaged over the four; among those within 0.02 of it the one with the lowest objects' position error (as in the
  first round), then the one with fewer iterations, then the one without the average. The round is reported with every
  run. There is no third round.

  Its outcome: variant 1, 12,000 iterations without the average. Two of every variant's four runs settled, and variant 1
  has the highest success at its last checkpoint averaged over the four (54.1 % against 41.0, 44.7 and 41.3 % for
  variants 2 to 4); so the reported runs train 12,000 iterations.
- **Third round** (asked for in the pull request's review; the second round's text ruled a third out): the first two
  rounds vary options that the two algorithms share, so neither algorithm's own options were searched. One more round
  follows, as the first two: seed 0, the same number of runs for both algorithms, from the second round's choice
  (12,000 iterations), on each task's tuning demonstration. That is the next demonstration after the reported one whose
  replay reaches the reward of 4: Transfer Cube's demonstration 1 and Insertion's demonstration 5 (the first two rounds
  used demonstration 1 of both; Insertion's replays to the reward of 2). Four variants per algorithm and task:
  1. nothing else changed;
  2. the algorithm's own option: for SuperTrack the noise on the imagined actions at 0.2 (`--st_imag_noise 0.2`; 0.1
     otherwise), for SuperMimic the gradient through the imagined state scaled by 0.95 at every step
     (`--bptt_decay 0.95`; 1 otherwise);
  3. the exploration noise on the actions at 0.3 (`--action_noise 0.3`; 0.1 otherwise);
  4. both.

  Per algorithm one variant is taken, by the first round's rule: the highest success at its last checkpoint averaged
  over the two tasks; among those within 0.02 of it the lowest objects' position error (averaged over the tasks, over
  the rollouts that do not terminate); then the fewest changes (the order above). The round is reported with every run.
  The reported runs then train until the settled test holds (also asked for in review); their budgets and stopping rule
  are fixed here after this round and before those runs.

  Its outcome: variant 1 for both algorithms, so the configurations stay as they are. Averaged over the two tasks its
  success at the last checkpoint is 49.4 % for SuperTrack (against 1.4, 0.2 and 0.0 % for variants 2 to 4) and 40.0 %
  for SuperMimic (against 0.0 % for each of variants 2 to 4); no other variant is within 0.02 of it.
- **Budget and stopping:** one rule for both algorithms, fixed after the third round and before the reported runs (the
  pull request's review: train until the settled test holds).
  - A run has settled at a checkpoint if that checkpoint and the four kept before it (1,000 iterations) agree: their
    successes within 0.05 of one another and their objects' errors within 1 mm or a quarter of their mean, whichever is
    larger. A run's checkpoints are evaluated from the same perturbed starts.
  - A reported run trains the configuration's 12,000 iterations (the second round's choice, the learning rate's cosine
    over them) and goes on at the learning rate the cosine ends on (0.03 of the first). It stops at the first
    checkpoint, from 12,000 iterations on, at which it has settled; the cap is 48,000 iterations. The checkpoint it
    stops at is its last: the one evaluated, and its wall time the run's training time.
  - A run that has not settled at the cap is reported so, with the cap's checkpoint; no comparison is drawn from a case
    with such a run.
  - The runs are trained to the cap with every checkpoint kept, and the stop is found from the checkpoints' evaluations
    afterwards (`time_to.py`): the checkpoint is the one that stopping the run there leaves. The checkpoints after the
    stop are evaluated too and counted beside it (how many have a success within 0.05 of the stop's); they are not the
    run's numbers.
  - The runs of the first two rounds' budgets (6,000 and 12,000 iterations, demonstration 0 of both tasks, the last
    checkpoint evaluated) stay in the results beside them, as they were.
- **Seeds:** training seeds 0, 1, 2 per configuration, every seed reported; mean and sample SD; the GPU and settings
  stated.
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
  closed on it; the peg inside the socket's opening with its end within 2 mm of the pin's.
- **Measures:** the success, the stages and the final state, the objects' position and rotation error, the joints'
  error and the share of the rollouts that end early (a tracked object 15 cm or a gripper keypoint 20 cm from the
  reference); SuperTrack first, both numbers side by side. Insertion's main rows have the share of the rollouts that
  reach stage 3 (the peg touches the socket) and the objects' error beside the success (the pull request's review).
- **Time:** each run alone on one A100 (80 GB). The clock starts at the first training iteration. A checkpoint is kept
  every 200 iterations; the time to the target is that of the first checkpoint from which every later one, up to the
  run's stop, has a success of at least 0.99 with the objects within 10 mm (the manipulation task's target). Without a
  per-phase time table of the path: time to the target, not engineered.

Not measured: ALOHA's real recordings; the public simulated recordings (no object poses); image observations; ACT or
another imitation policy trained here.

## Data audit

- ACT ([tonyzhaozh/act](https://github.com/tonyzhaozh/act), MIT, "Copyright (c) 2023 Tony Z. Zhao") is pinned at commit
  `742c753`; its 22 scene and mesh files are checked against SHA-256 sums and are not redistributed.
- The demonstrations: 10 kept of the first 10 episodes drawn per task (seed 0), every one reaching ACT's reward of 4.
  A state is MuJoCo's qpos and qvel of the joint-controlled scene (Transfer Cube 23 + 22, Insertion 30 + 28): the arms'
  16 joints first, then the objects' free joints.
- **MuJoCo 2.3.7 for the generation.** The tool runs with the version that ACT's instructions install, in an
  environment of its own. With the library's MuJoCo 3.15 the grippers follow their mocap bodies differently in the
  end-effector scene and the scripted grasps miss (Transfer Cube: 1 of 12 episodes reaches the reward of 4, Insertion
  none; with 2.3.7, 12 of 12 each).
- The demonstration's position targets replayed open loop on MuJoCo-Warp, from the unperturbed start: ACT's reward
  reaches 4 in 10 of the 10 Transfer Cube demonstrations and in 5 of the 10 Insertion ones (with MuJoCo 3.15 itself:
  10 and 3); in the other Insertion replays the peg stops short of the pin, the objects ending 5 to 18 mm from the
  demonstration's. The task's own demonstration 0 of Insertion is one of them (its replay reaches the reward of 3).
- The reference's joint positions as the targets, open loop: no demonstration of either task reaches the reward of 4,
  and every rollout terminates (a position target at the finger's own position does not squeeze; the cube ends 250 to
  346 mm from its demonstration, the peg or the socket 155 to 202 mm).
- MuJoCo-Warp 3.15 runs the two scenes as they are: one control step (10 physics steps) from 32 states along
  demonstration 0 under its recorded targets agrees with MuJoCo's in qpos to 2.5e-7 (Transfer Cube) and 1.2e-5
  (Insertion) (the largest entry's difference, median over the states; the largest over all of them 3.7e-4 and 5.0e-4,
  on an arm's joint), with the solver at 20 iterations and at the scene's own 100 alike.
- ACT's reward from MuJoCo-Warp's contacts at the frames of demonstration 0 agrees with the reward recorded with the
  demonstration (MuJoCo 2.3.7's contacts) on more than 97 % of the frames of each task (a test).

## Sources and licences

- **ACT** ([tonyzhaozh/act](https://github.com/tonyzhaozh/act), MIT, "Copyright (c) 2023 Tony Z. Zhao"): its scenes and
  meshes, its scripted policies and its reward tests; Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware", RSS 2023. The scenes and meshes are downloaded by the user at a pinned revision and are not
  redistributed here; `THIRD_PARTY_NOTICES.md` has the licence's text.
