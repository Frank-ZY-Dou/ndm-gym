# Hand-object tracking

A dexterous hand tracks a human hand-object demonstration: a right XHand (12 finger joints) on a position-controlled
floating palm reaches an object, grasps it and lifts it, on MuJoCo-Warp directly (no Isaac Lab;
`ndm_gym/sim/rigid/hand_sim.py`). The clips and scenes are [SPIDER](https://github.com/facebookresearch/spider)'s
(`retarget/retarget_full`: DexYCB, HRDexDB and OakInk demonstrations retargeted to robot hands).

The reference is SPIDER's **kinematic** retarget of the demonstration (inverse kinematics of the human hand's motion, `trajectory_kinematic.npz`), resampled to 50 Hz, with the last pose held for 0.5 s. No physics shaped it: the fingers may float off or cut into the object, as a retargeted humanoid motion's feet do with the floor. The policy has to find
the contacts that make the object follow.

```bash
pip install -e '.[hand]'                             # MuJoCo-Warp
python tools/download_hand_clips.py                  # the benchmark's clips into data/spider
benchmarks/xhand_tracking/run.sh apple 0             # train, evaluate and render one task and seed (SuperMimic's defaults)

# or step by step: SuperTrack, then SuperMimic
CLIP=data/spider/processed/hrdexdb_24f/xhand/right/apple-0/0
python -m ndm_gym.train_hand --clip $CLIP --algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights --log_dir logs/apple_st
python -m ndm_gym.train_hand --clip $CLIP --log_dir logs/apple_sm
python -m ndm_gym.evaluate_hand --ckpt logs/apple_st/last.pt --out logs/apple_st/evaluation
python tools/render_hand.py --clip $CLIP --rollout logs/apple_st/evaluation/rollout.npz --out apple_st.mp4
```

## Task

| | |
|---|---|
| state | the hand's 18 joint positions and velocities; the object's position, rotation, linear and angular velocity |
| action | one offset per joint in [-1, 1] about the next reference frame's pose (3 cm palm translation, 0.3 rad palm rotation, 0.4 rad fingers), as position targets |
| observation | the joints, the fingertips and the object in the palm's frame, the previous action |
| goal | the next 5 reference frames about the state: joint offsets, the object's offset and relative rotation |
| reward | tracking kernels on the keypoints (palm and fingertips), the finger joints, the palm's rotation and the object's position and rotation; an action-rate penalty |
| termination | the object or the palm more than 15 cm from its reference |
| episode | from the reference's first frame (joints and the object's position perturbed) to its last |

Everything is an analytic function of the state and the reference frame (`ndm_gym/tasks/hand_tracking.py`), so the
simulator's transitions and the world model's imagined rollouts share it.

## Method

Both algorithms fit an MLP world model of the hand-object state (`ndm_gym/worldmodels/hand.py`) on 8-step windows of
simulator transitions and train the policy through imagined rollouts of the frozen world model from replayed states,
over `--horizon` steps (`ndm_gym/algos/hand_algos.py`). Training takes `ndm_gym.train`'s arguments with its own
defaults (`ndm_gym.train_hand.HAND_DEFAULTS`); those it does not implement are refused.

[SuperTrack](../algorithms/supertrack.md) (`--algo supertrack`): the world model predicts the accelerations of the
joints and of the object, integrated over a control step. The policy minimizes the L1 errors of what the task tracks
over the imagined window (the keypoints, the palm's rotation, the finger joints, the object's position and rotation;
the fingertips about the palm with the calibrated weights), plus a failure penalty on the task's soft termination
(`--st_fail_w`) and a regularizer on the actions; no reward, no critic.

| option | the terms' weights |
|---|---|
| default | each term weighs 1 at the first policy update (calibration) |
| `--st_unit_scales` | calibration, terms of one unit sharing a weight: the object's position and the fingertips about the palm per metre as the keypoints, the object's rotation per radian as the palm's |
| `--st_task_weights` | no calibration: each of the task's five quantities weighs the reward's weight over its kernel's width |
| `--st_obj_w k` | the object's two terms weigh `k` times that |
| `--st_vel_terms` | adds the velocities of the joints and of the object (calibrated weights only) |

A clip whose reference pose alone does not lift the object starts with a large error of the object; calibrated by
itself the object then weighs little, and the finger joints, whose first error is small, weigh much. The task's weights
trade the hand's pose against the object as the reward does.

[SuperMimic](../algorithms/supermimic.md) (the default): the world model predicts the increments of the state; the
policy maximizes the imagined tracking return with a soft survival weight and a critic beyond the horizon.
`--bptt_decay` scales the gradient through the imagined state at every step.

## Systems options

Off by default (`ndm_gym/algos/hand_fast.py`); with all of them off the loop and the agents compute what they compute
without them. Each applies to both algorithms.

| option | what it does | what it changes in a run's numbers |
|---|---|---|
| `--fused_adam` | Adam's fused CUDA kernel for every optimizer | rounding |
| `--reset_masked` | ended worlds are reset by masked writes of fixed shapes, the draws made for every world | the random stream |
| `--no_sync` | no host-device synchronization in an iteration: the control step on Warp's stream, the normalizers' freeze test and the skip of a non-finite policy step on the device (needs the two above) | rounding |
| `--collect_graph` | a collection step with its physics, resets and replay write as one CUDA graph (needs `--no_sync`) | the random stream of the replay's sampling |
| `--compile_eager` | the collection step's forward kinematics as a compiled function (needs `--no_sync`) | rounding |
| `--update_graphs` | the iteration's world-model updates as one CUDA graph, its policy (and critic) update as another (needs `--no_sync`) | the random stream of the replay's sampling |
| `--pi_opt muon`, `--wm_opt muon` | Muon on the hidden matrices of the policy's or the world model's MLP, Adam on the rest | the optimizer |
| `--pi_ema d` | a moving average of the policy with decay `d`, stored as the checkpoint's `pi` (the trained policy as `pi_live`) | the policy that is evaluated |
| `--pi_lr_mult m` | the policy's learning rate is `--lr` times `m`, ramped to 1 over `--lr_decay_len` iterations from the log row where `sim_r` reaches `--lr_decay_ret`, at the latest from `--lr_decay_by` | the policy's schedule |

`--amp bf16` (the MLPs' products in bf16) combines with all of them. `HAND_PROFILE=1` prints, with every log row, the
time of the collection step, the world-model updates and the policy update, the device synchronized around each.

## Evaluation and videos

`ndm_gym.evaluate_hand` rolls the policy out from the first frame to the last without resets: once from the
unperturbed start (saved for rendering), and from `--num_envs` perturbed starts, each also with the reference pose as
the position targets in the policy's place, for comparison. A rollout succeeds if it never terminates and the object ends
within 5 cm of its last reference position.

`tools/render_hand.py` renders a saved rollout in the look of `tools/render.py`, at 1920x1080, 60 fps and half speed:
the simulated hand and object solid, the kinematic reference a translucent ghost, a camera that follows them, the NDM
Gym mark (`--no_mark`: none). `--source reference_targets` renders the rollout with
the reference pose as the targets, `--source reference` the reference alone; `tools/side_by_side.py` joins two renders.
`--look noir` renders on a dark stage, and `--ghost_overlay` draws the reference over the frame as a blue silhouette; the
README's animations use `--look noir --orbit 30 --zoom 0.72 --follow_z --elevation -12 --ghost_overlay`, each render titled
with ffmpeg's `drawtext` and the two joined by `tools/side_by_side.py`.

Tasks and results: [`benchmarks/xhand_tracking`](../../benchmarks/xhand_tracking/results.md).
