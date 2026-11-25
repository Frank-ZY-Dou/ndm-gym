# Manipulation demonstration tracking

Contributed by [Jingdong Zhang](https://github.com/Evergreen0929) ([#10](https://github.com/Frank-ZY-Dou/ndm-gym/pull/10)).

A Franka Panda arm with its parallel gripper (two of them for Transport) tracks a human teleoperated demonstration in
physics: the arm's joints, the gripper and the objects it handles, from the demonstration's own simulator states. The scenes
run on MuJoCo-Warp directly (no Isaac Lab; `ndm_gym/sim/rigid/robomimic_sim.py`). One policy tracks one demonstration, or
several of one scene.

The demonstrations are robomimic's proficient-human ones (Mandlekar et al., "What Matters in Learning from Offline Human
Demonstrations for Robot Manipulation", CoRL 2021;
[robomimic/robomimic_datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets), v1.5, MIT), recorded at 20 Hz
with [robosuite](https://github.com/ARISE-Initiative/robosuite) 1.5.0 (MIT). Each holds MuJoCo's flattened state at every
control step and the scene it was recorded in, so the objects' trajectories are the demonstration's own. Both are downloaded
at pinned revisions by a script and are not redistributed here; robosuite itself is not a dependency.

```bash
pip install -e '.[hand]'                             # MuJoCo-Warp, h5py
python tools/download_robomimic.py                   # the demonstrations and robosuite 1.5.0's assets, pinned and checked

# one demonstration: train (SuperTrack, then SuperMimic), the stop checkpoint by the benchmark's rule, evaluated from
# perturbed starts beside the reference's targets, rendered
python -m ndm_gym.train_robomimic --config robomimic-supertrack --task square --log_dir logs/square_st
python -m ndm_gym.train_robomimic --config robomimic-supermimic --task square --log_dir logs/square_sm
python benchmarks/robomimic_manipulation_tracking/stop_rule.py logs/square_st     # its stop_ckpt in logs/square_st/stop.json
python -m ndm_gym.evaluate_robomimic --ckpt logs/square_st/ckpt_<stop>.pt --out logs/square_st/eval
MUJOCO_GL=egl python tools/render_robomimic.py --rollout logs/square_st/eval/rollout_0.npz --out square.webp --title SuperTrack
```

## Scenes

`ndm_gym/robots/robomimic_scenes.py` compiles a demonstration's scene (its `model_file`) with robosuite's meshes and
textures resolved under the downloaded assets.

| | Lift | Square | Transport |
|---|---|---|---|
| arms | one Panda | one Panda | two Pandas |
| objects tracked | the cube | the square nut | the hammer, the trash, the start bin's lid |
| demonstrations | 200, 1.8-3.2 s | 200, 5.4-11.8 s | 200, 18.7-35.7 s |
| success (robosuite's test) | the cube more than 4 cm above the table | a nut on its peg, released (more than 4.24 cm from the grip sites) | the hammer touching the target bin's base, the trash the trash bin's |

- **Actuators:** robosuite drives each arm's torque actuators by operational-space control with gravity compensation. Here
  each becomes a joint position servo, force = kp (target - q) - kd q̇ with kp of 600, 600, 600, 600, 250, 150 and
  50 N m / rad, clipped to the torque actuator's own range (80 N m on joints 1-5, 12 N m on 6 and 7), and the robots' bodies
  are compensated for gravity (`gravcomp`). The grippers keep their position servos.
- **Physics:** MuJoCo-Warp at the demonstrations' rates: 2 ms physics steps, 20 Hz control (25 physics steps a control
  step, replayed as one CUDA graph). `--solver_iterations` sets the solver's iterations (the results: 20).

## Task

`ndm_gym/tasks/robomimic_tracking.py`; the reference is one or more demonstrations of one scene (`--demo`), each with its
last state held for 0.5 s.

| | |
|---|---|
| state | the scene's qpos and qvel: the arms', the grippers' and every free object's |
| action | one per actuator: an offset of its position target about the reference's joint position at the next frame |
| observation | the actuated joints and their velocities, the grippers' keypoints (end effector and fingers), each tracked object about each end effector, its rotation and velocities, the previous action |
| goal | 5 reference frames ahead about the state: the joints' offsets, the tracked objects' offsets and relative rotations, the keypoints' offsets |
| reward | kernels on the keypoints (4 cm), the arms' joints (0.4 rad), the objects' positions (4 cm) and rotations (0.5 rad); an action-rate cost, or with `--target_rate_w` a target-rate cost in its place ([smooth targets](#smooth-targets)) |
| termination | a tracked object 15 cm or a keypoint 20 cm from its reference |
| episode | from a random frame of the reference with noise on the arms' joints (the objects and the grippers as recorded, so that a grasp stays a grasp), to its demonstration's end |

## Method

Both algorithms fit MjcfWorldModel (worldmodels.mjcf, the scene in generalized coordinates: fixed bases, the objects as
free joints) over one control step on windows of simulator transitions, and train the policy through imagined rollouts of
the frozen model (`ndm_gym/algos/robomimic_algos.py`): SuperTrack with the acceleration model and L1 errors of the
keypoints, the joints, the fingers and the objects' poses; SuperMimic with the increment model, the imagined reward and a
critic. The systems options are the hand task's.

## Smooth targets

A policy's action is an offset about the demonstration's joint position at the next frame. Nothing in SuperTrack's loss, and
only a small term in SuperMimic's reward (`w_action_rate`, 0.05 times the actions' mean squared change), ties one step's
offset to the next, so the servos' targets can jump back and forth at every 50 ms step. The servos pass part of that on to
the arms as jerk and tremor.

`--target_rate_w w` (both algorithms; 0, the default, is off) adds a cost per step (`RobomimicTask.smooth_cost`): `w` times
the mean over the actuators of the change of the policy's part of the targets since the previous step, each actuator's over
its tracking kernel's width (an arm joint 0.4 rad, a finger 0.04 m), squared. A jump costs; following the demonstration does
not. The reward carries it in the place of `w_action_rate`'s term, and SuperTrack's loss adds its mean over the imagined
steps, at the scale of the task-weighted terms (`--st_task_weights`, required with it), so that both algorithms have the one
same term. The benchmark's configurations use `--target_rate_w 64`.

`benchmarks/robomimic_manipulation_tracking/motion.py` measures an evaluation's unperturbed rollout against its
demonstration's same frames: the arms' joint acceleration and jerk, the share of their velocity's spectral power above 5 Hz, the
change of their position targets per step (with `evaluate_robomimic --save_targets`), limit hits, and a held object's slip.

```bash
python -m ndm_gym.evaluate_robomimic --ckpt logs/square_st/ckpt_<stop>.pt --out logs/square_st/eval --save_targets
python benchmarks/robomimic_manipulation_tracking/motion.py st=logs/square_st/eval/rollout_0.npz --json motion.jsonl
```

## Configurations

`--config list` prints the configurations with what each measured. Each is one of five candidates with the same kinds of
change from the hand task's accurate presets, chosen by the same rules on other seeds: `robomimic-supertrack` is
`hand-supertrack-accurate`'s options with imagination noise 0 (`--st_imag_noise 0`), `robomimic-supermimic`
`hand-supermimic-accurate`'s with `--bptt_decay 0.95`; both add the smoothness term at one weight for both
(`--target_rate_w 64`), `--solver_iterations 20`, a cap of 6,000 iterations with the learning rate's decay over it and a
checkpoint kept every 250, from which the benchmark's stop rule
(`benchmarks/robomimic_manipulation_tracking/stop_rule.py`) takes the run's result.

Results: [benchmarks/robomimic_manipulation_tracking](../../benchmarks/robomimic_manipulation_tracking/results.md). In
short (seeds 10-12, one A100 40GB SXM4 per run, each run to its stop): SuperTrack a success of 100 / 100 / 99.9 % with
the object within 0.26 / 1.48 / 4.24 mm on Lift / Square / Transport, the target after 1.03 / 2.12 / 15.76 min;
SuperMimic 100 / 100 / 89.2 %, 1.01 / 2.55 / 4.50 mm, 1.11 / 4.53 min / not reached (time to the target, not engineered).
Both configurations' motion passes the check of the
[results' smooth motion](../../benchmarks/robomimic_manipulation_tracking/results.md#smooth-motion). Success is
robosuite's test at the demonstration's last frame on a rollout that never terminated, a tracking proxy; MuJoCo-Warp's
contact order is not deterministic, and two evaluations of one checkpoint differ by about 4 points on Transport. As the
simulator, the world models' bf16 copies run 765 and 757 times MuJoCo-Warp's control step on Transport at 4,096 worlds
(11.63 M and 11.06 M world-steps per second on one A100 40GB SXM4; not engineered).

![SuperTrack and SuperMimic on Lift, Square and Transport](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/grid.webp)

Per case the run with the lowest object error among those whose own rollout from the unperturbed start passes the success
test and the motion check (Transport at four times real time), the demonstration's arms and objects in translucent blue. The animations: [Lift](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/lift_st.webp), [Square](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/square_st.webp), [Transport](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/transport_st.webp)
(SuperTrack); [Lift](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/lift_sm.webp), [Square](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/square_sm.webp), [Transport](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/robomimic/transport_sm.webp) (SuperMimic).
