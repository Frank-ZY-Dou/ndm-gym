# Quadruped motion tracking

Contributed by [Jionghao Wang](https://github.com/shanemankiw) ([#4](https://github.com/Frank-ZY-Dou/ndm-gym/pull/4),
[#14](https://github.com/Frank-ZY-Dou/ndm-gym/pull/14)).

A Unitree Go2 tracks the motion capture of a Labrador retriever: walking, trotting, running, turning, backing, jumping
and playing, on MuJoCo-Warp directly (no Isaac Lab; `ndm_gym/sim/rigid/quadruped_sim.py`). One policy tracks one clip
or all of them.

The motion capture is the 33 BVH files (16.3 min at 120 fps) of "Lifelike agility and play in quadrupedal robots using
reinforcement learning and generative pre-trained models" (Han et al., Nature Machine Intelligence 2024;
[Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play),
`data/raw_mocap_data`; its licence file is MIT, its README says the code and data are for research purposes only). The
robot is [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_go2` (BSD-3-Clause). Both
are downloaded at pinned commits by a script and are not redistributed here.

The reference is a **kinematic** retarget of the dog to the robot, by
[supermimic-quadruped](https://github.com/Houserz/supermimic-quadruped)'s `quadmotion` package (MIT), called from the
user's own checkout; none of its code is copied here. No physics shaped it: the policy has to find the contacts that
make the trunk follow.

```bash
pip install -e '.[hand]'                             # MuJoCo-Warp
tools/download_lifelike.sh                           # the motion capture and the Go2, at pinned commits, into data/
git clone https://github.com/Houserz/supermimic-quadruped data/third_party/supermimic-quadruped
git -C data/third_party/supermimic-quadruped checkout 46a81c2                          # the pinned commit
python tools/convert_lifelike.py --quadmotion data/third_party/supermimic-quadruped/src    # 65 clips into data/lifelike_go2

# SuperTrack, then SuperMimic, one policy on all clips, with their named configurations (ndm_gym/configs.py);
# one clip: --config go2-supertrack-clip --clips <name> (--clips <name>,<name>: on those)
python -m ndm_gym.train_quadruped --config go2-supertrack --log_dir logs/go2_st
python -m ndm_gym.train_quadruped --config go2-supermimic --log_dir logs/go2_sm
python -m ndm_gym.evaluate_quadruped --ckpt logs/go2_st/last.pt --out logs/go2_st/evaluation --push 1 2 --save_rollout
python -m ndm_gym.evaluate_quadruped --ckpts logs/go2_st --out logs/go2_st/evaluation --success_trunk_mm 20   # the time to the target
MUJOCO_GL=egl python tools/render_quadruped.py --rollout logs/go2_st/evaluation/rollout.npz --clip dog_play_002_0 --out play.webp
```

## Clips

`tools/convert_lifelike.py` retargets every file and cuts it into clips:

- The dog's shoulder and hip joints give the trunk's frame. Each paw's offset from its shoulder or hip, scaled by the
  robot's standing height over the dog's (0.528), gives a foot target below the robot's own thigh joint; planted paws
  are held in place on the floor; the legs are solved in closed form.
- Time is stretched by the square root of the scale (0.727), so that flight phases keep gravity.
- Frames that fail one of `quadmotion`'s checks (an unreachable foot, a joint at its limit, the trunk or a knee in the
  floor, a capture glitch, ...) are cut out. Each run of valid frames of at least 2 s is a clip, named
  `<file>_<index>`, its start at the origin facing +x.

With the default checks: 65 clips, 488 s at the robot's time scale (the shortest 2 s, the median 3.8 s, the longest
79 s), resampled to the control rate of 50 Hz. `data/lifelike_go2/manifest.json` holds the calibration and, per source
file, the frames kept and what cut the others.

## Task

| | |
|---|---|
| state | MuJoCo's generalized coordinates of the Go2: the trunk's position and quaternion and the 12 joints (19), the trunk's world linear and body-frame angular velocity and the joints' (18) |
| action | one offset per joint in [-1, 1] about the next reference frame's pose (0.3 rad at the hips, 0.5 rad at the thighs and the calves), as position targets |
| control | 50 Hz, 10 simulator steps of 2 ms; position servos of stiffness 40 N m / rad and damping 1 N m s / rad within the motors' torque limits (23.7 N m at the hips and the thighs, 45.43 N m at the calves) |
| observation | the trunk's height, tilt and body-frame velocities, the joints' positions and velocities, the feet about the trunk in its frame, the previous action (61) |
| goal | the next 5 reference frames in the trunk's heading frame: the trunk's offset, rotation and velocities, the joints' offsets |
| reward | tracking kernels on the feet about the trunk, the joints, the trunk's position and rotation in the world, and its linear and angular velocity; an action-rate penalty |
| termination | the trunk's height more than 12 cm from the reference's, its rotation more than 1 rad, or its position more than 50 cm |
| episode | from a random frame of a clip (the joints and the velocities perturbed) to the clip's last frame |

Everything is an analytic function of the state and the reference frame (`ndm_gym/tasks/quadruped_tracking.py`), so the
simulator's transitions and the world model's imagined rollouts share it. The observation does not depend on the
trunk's heading and horizontal position; the reward and the termination compare the trunk with the reference in the
world, so that drift costs.

**The joints' damping** is 0.1 N m s / rad, the value of Unitree's own MuJoCo model of the Go2
([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), `unitree_robots/go2/go2.xml`), in place of
Menagerie's 2 (`ndm_gym/robots/go2.py`; `--joint_damping -1`: the model's own). At 2, the joint speeds of the fast clips
(up to 16 rad/s) take more than the hip and thigh motors' torque to overcome, and the robot falls behind them.

## Method

Both algorithms fit the library's MLP world model of generalized coordinates (`ndm_gym/worldmodels/mjcf.py`) on windows
of simulator transitions and train the policy through imagined rollouts of the frozen world model from replayed
states, over `--horizon` steps (`ndm_gym/algos/quadruped_algos.py`). Training takes `ndm_gym.train`'s arguments with its
own defaults (`ndm_gym.train_quadruped.QUADRUPED_DEFAULTS`); those it does not implement are refused.

- [SuperTrack](../algorithms/supertrack.md) (`--algo supertrack`): the policy minimizes the L1 errors of what the task
  tracks over the imagined window, plus a failure penalty on the task's soft termination and a regularizer on the
  actions; no reward, no critic. `--st_task_weights`: the terms weighted as the task's reward weighs them.
- [SuperMimic](../algorithms/supermimic.md) (`--algo supermimic`): the policy maximizes the imagined window's
  discounted reward plus a critic's value at its end.

The systems and optimizer options are the hand-object task's ([Hand-object tracking](hand.md)), off by default:
`--fused_adam`, `--reset_masked`, `--no_sync`, `--compile_eager`, `--collect_graph`, `--update_graphs`, `--pi_opt muon`,
`--wm_opt muon`, `--pi_ema`, `--pi_lr_mult`. `QUADRUPED_PROFILE=1` prints, with every log row, the time of the
collection step, the world-model updates and the policy update.

## Evaluation and renders

`ndm_gym.evaluate_quadruped` rolls the policy out on every clip from its first frame to its last without resets:
`--num_envs` times per clip from perturbed starts (256; with many clips at most `--max_worlds` worlds in all), each also
with the reference pose as the position targets in the policy's place, for comparison.

| metric | |
|---|---|
| success rate | a rollout succeeds if it never terminates; a clip's rate is its rollouts' share, the summary's the mean of the clips' |
| normalized return | a rollout's reward sum over the most its clip's steps can give, the steps after a termination counting nothing (as `ndm_gym.evaluate`'s) |
| failed clips | the mean number of clips failed by a pass of one rollout per clip |
| errors | means over the frames before a termination: the trunk's position (mm) and rotation (rad), the feet in the world and about the trunk (mm), the joints (degrees, RMS) |
| pushes | `--push 1 2`: the same rollouts with a horizontal velocity of that size (m/s) added to the trunk in a random direction every 2 s |
| training time | `--ckpts <run>`: every checkpoint of a run trained with `--keep_ckpts`; the wall time of the first one from which every later one has a success rate of `--success` (0.99) and, with `--success_trunk_mm`, the trunk within that distance |

`tools/render_quadruped.py` (`pip install -e '.[render]'`) renders one clip of a rollout saved with `--save_rollout`, the
simulated robot solid and the retargeted reference laid over it as a translucent blue silhouette with an outline, so that
it shows where the two overlap (`--plain_ghost`: a faint ghost in the scene instead): as an animation in real time (`--out` ending in
`.webp`) or as a contact sheet of frames at even steps. `--source reference_targets` draws the rollout with the
reference pose as the targets.

Tasks and rules: [`benchmarks/go2_lifelike_tracking`](../../benchmarks/go2_lifelike_tracking/README.md). Tracking results
on this task are not reported here.
