# Manipulation demonstration tracking: Panda arms on robomimic's human demonstrations

The task and the rules of its benchmark ([the task](../../docs/tasks/robomimic.md), [results](results.md)).

## Task

A Franka Panda arm with its parallel gripper, two for Transport, tracks a proficient human's teleoperated demonstration in
physics: the arms' joints, the grippers and the objects they handle, from the demonstration's own simulator states.
SuperTrack and SuperMimic train the policy through an MLP world model of the scene, as in the library's other tracking
tasks. It adds the library's first fixed-base arm with a gripper, contact-rich manipulation with millimetre tolerances, and
two arms handing an object over.

- **Demonstrations:** robomimic's proficient-human (PH) datasets (Mandlekar et al., CoRL 2021),
  [robomimic/robomimic_datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets) on Hugging Face (MIT),
  `v1.5/<task>/ph/demo_v15.hdf5` at revision `74fa018`: 200 demonstrations per task, recorded at 20 Hz with robosuite
  1.5.0. Each holds MuJoCo's state at every control step (`states`: time, qpos, qvel) and its scene (`model_file`), so the
  objects' trajectories are the demonstration's own, not estimated. About 330 MB for the three tasks; downloaded by a
  script, each file checked against its SHA-256, not redistributed.
- **Scenes and robot:** each demonstration's `model_file`, with the meshes and textures of
  [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT) at the v1.5.0 tag (`1a8701b`), downloaded by the same
  script. The arms' torque actuators (80 N m on joints 1-5, 12 N m on 6-7), which robosuite drives by operational-space
  control, become joint position servos within the same torque limits, with gravity compensation on the robots' bodies
  (robosuite's controller compensates gravity); the grippers keep their position servos. robosuite itself is not a
  dependency: no controller, no environment code.
- **Simulator:** MuJoCo-Warp at the demonstrations' rates: 2 ms physics steps, 25 to a 50 ms control step as one CUDA
  graph, the constraint solver at 20 iterations.
- **Success:** robosuite's `_check_success` at that commit, read from the simulator after the control step that reaches
  the demonstration's last frame, as robosuite reads it after its own step:
  - **Lift:** the cube more than 4 cm above the table;
  - **Square:** a nut within 3 cm of its own peg in x and y, below the table's top plus 5 cm, and released (more than
    4.24 cm from the nearer grip site);
  - **Transport:** the hammer touching the target bin's base and the trash touching the trash bin's base, in the
    simulator's contact list.

  A rollout succeeds if it never terminates and passes the test there: a tracking proxy, not robomimic's success rate of a
  learned policy (the test met at any step within a horizon). Each of the 200 demonstrations of each task passes it at its
  last state.

## What the task adds

- `tools/download_robomimic.py` (the demonstrations and robosuite's assets at pinned revisions, each file checked against
  its SHA-256 in `download_robomimic.sha256.json`) and a loader of a demonstration's scene and reference
  (`ndm_gym/motion/sources/robomimic.py`).
- The scene with the servos (`ndm_gym/robots/robomimic_scenes.py`), the simulator with robosuite's success tests
  (`ndm_gym/sim/rigid/robomimic_sim.py`), the tracking task (observation, goal, reward, termination, success;
  `ndm_gym/tasks/robomimic_tracking.py`), the policy, both algorithms with the library's generalized-coordinate world model
  (`worldmodels/mjcf.py`, which takes these scenes as they are: a fixed base, the objects as free joints), training and
  evaluation with the hand task's systems options, and the world model's throughput beside MuJoCo-Warp
  (`ndm_gym/wmbench/speed_robomimic.py`).
- Tests in `tests/` that run on the CPU, and GPU tests of the simulator against MuJoCo and of robosuite's success tests on
  MuJoCo-Warp.
- A task page in `docs/`, `configs.py` entries, renders with the reference as a blue ghost, and the entries in `NOTICE` and
  the README's "Papers and repositories" for the sources above.

## Rules

- **Tasks:** three tasks of rising difficulty, one policy per task, on each task's demonstration 0 (its last state held for
  0.5 s: Lift 3.45 s, Square 6.85 s, Transport 36.2 s):
  - **Lift**: one arm lifts a cube (demonstrations of 1.8-3.2 s);
  - **Square**: one arm puts a square nut onto its peg, millimetre clearances (5.4-11.8 s);
  - **Transport**: two arms hand a hammer over and move a piece of trash (18.7-35.7 s).
- **Configurations:** for each algorithm, one of four, all chosen by one rule on training seed 5 and evaluation seed 2:
  the hand task's accurate preset and three single changes of the same kinds for both (SuperTrack: imagination noise 0
  instead of 0.2, learning rate 2e-4 instead of 4e-4, failure weight 5 instead of 2; SuperMimic: action noise 0.3 instead
  of 0.1, learning rate 4e-4 instead of 2e-4, discount 0.995 instead of 0.99), each on the three tasks with the stop rule
  below. The rule: the highest success averaged over the three tasks; within one point of it, the lower mean object error.
  The chosen ones are `configs.py`'s `robomimic-supertrack` and `robomimic-supermimic`.
- **Seeds:** three training seeds per configuration (0, 1, 2); mean and sample SD, and every seed; the GPU stated.
- **Training:** 2,048 worlds, a cap of 6,000 iterations with the learning rate's cosine decay over it, a checkpoint kept
  every 250 iterations.
- **Stop rule, the same for both algorithms:** each kept checkpoint is evaluated on 256 perturbed starts of evaluation
  seed 1 (other starts than the results'). With S(i) and E(i) the mean success and object error of the evaluations at i,
  i - 250 and i - 500, a run stops at the first i where E(i) > 0.97 E(i - 1,000) and either S(i) >= 0.99 or
  S(i) < S(i - 1,000) + 0.01: the error improved by less than 3 % over 1,000 iterations, and the success holds or stopped
  improving. Otherwise it runs to the cap. The checkpoint at the stop, not the best one, is the run's result.
- **Evaluation:** `evaluate_robomimic` on the stop checkpoint: 256 rollouts of the whole demonstration from perturbed
  starts (0.01 rad on the arms' joints, evaluation seed 0), without exploration noise. Per task and configuration: the
  success; the share of episodes ended early; the tracked objects' position (mm) and rotation errors against the
  demonstration (means over the frames of the rollouts that do not terminate) and at the last frame; the grippers'
  keypoints. MuJoCo-Warp's contact order is not deterministic, so two evaluations of one checkpoint on the same starts
  differ a little; each number is one evaluation.
- **Time to the target:** a success of at least 0.99 with the objects within 10 mm (the hand task's target): the training
  clock of the first kept checkpoint from which every later one up to the stop meets it on the evaluation's starts.
- **Reference:** the same rollouts with the demonstration's joint positions as the servos' targets, without a policy.
- **Motion:** each run's rollout from the unperturbed start beside the demonstration's same frames (`motion.py`): the
  arms' joint acceleration and jerk, the share of their velocity above 5 Hz, limit hits, and a held object's
  displacement in its gripper's frame per control step.

```bash
python tools/download_robomimic.py                                                          # the data and assets, pinned and checked
benchmarks/robomimic_manipulation_tracking/run_formal.sh                                    # 18 runs, each evaluated at its stop
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?     # the time to the target
python benchmarks/robomimic_manipulation_tracking/motion.py SuperTrack=<run>/eval/rollout_0.npz   # its motion beside the demonstration's
```

## Data audit

- The data and the assets are public under the licences above and pinned (robomimic_datasets revision `74fa018461f4`,
  robosuite's v1.5.0 tag `1a8701b90c07`).
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
