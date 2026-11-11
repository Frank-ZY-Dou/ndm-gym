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
benchmarks/xhand_tracking/run.sh apple 0             # train, evaluate and render one task and seed

# or step by step
CLIP=data/spider/processed/hrdexdb_24f/xhand/right/apple-0/0
python -m ndm_gym.train_hand --clip $CLIP --log_dir logs/apple
python -m ndm_gym.evaluate_hand --ckpt logs/apple/last.pt --out logs/apple/evaluation
python tools/render_hand.py --clip $CLIP --rollout logs/apple/evaluation/rollout.npz --out apple.mp4
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

[SuperMimic](../algorithms/supermimic.md) (`ndm_gym/algos/hand_algos.py`): an MLP world model predicts the
increments of the hand-object state (`ndm_gym/worldmodels/hand.py`) and is fit on 8-step windows of simulator
transitions; the policy is trained by backpropagating the imagined tracking return through the frozen world model
over `--horizon` steps, with a soft survival weight and a critic beyond the horizon. Training takes `ndm_gym.train`'s
arguments with its own defaults (`ndm_gym.train_hand.HAND_DEFAULTS`); those it does not implement are refused.

## Evaluation and videos

`ndm_gym.evaluate_hand` rolls the policy out from the first frame to the last without resets: once from the
unperturbed start (saved for rendering), and from `--num_envs` perturbed starts, each also with the reference pose as
the position targets in the policy's place, for comparison. A rollout succeeds if it never terminates and the object ends
within 5 cm of its last reference position.

`tools/render_hand.py` renders a saved rollout in the look of `tools/render.py`, at 1920x1080, 60 fps and half speed:
the simulated hand and object solid, the kinematic reference a translucent ghost, a camera that follows them, the NDM
Gym mark (`--no_mark`: none). `--source reference_targets` renders the rollout with
the reference pose as the targets, `--source reference` the reference alone; `tools/side_by_side.py` joins two renders.

Tasks and results: [`benchmarks/xhand_tracking`](../../benchmarks/xhand_tracking/results.md).
