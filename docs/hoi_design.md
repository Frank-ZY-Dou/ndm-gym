# Human-object interaction (HOI): aligned with HDMI

The task follows HDMI (LeCAR-Lab/HDMI); [`docs/hdmi_task_spec.md`](hdmi_task_spec.md) records its source line numbers.
BeyondMimic is not used. LAFAN1's default paths remain unchanged bit for bit.

## 1. What follows HDMI

| Aspect | Content (spec section) |
|---|---|
| Tasks | HDMI's 13 tasks (§8) |
| Robot | HDMI's G1 collision variant: spherical feet, box end effectors, capsule bodies (§3.4); actuators as ours (10 Hz, damping ratio 2) |
| Actions | residual: reference joint angles plus the policy's output; the 23 non-wrist joints, scaled per joint by 0.55 / 0.35 / 0.44; wrists held at q0; per 5 ms physics substep a 2-6 step delay and smoothing with α ~ U[0.8, 1] (§3.1) |
| Observations | phase, proprioception (with history), object (object pose, reference contact points); future reference frames 1/2/8/16/32; noise as HDMI's (§4) |
| Rewards | robot tracking, object tracking, interaction (contact reward), regularization and feet terms; each group summed and multiplied by dt (§5) |
| Terminations | 25 consecutive steps over a threshold: torso, keypoints, object pose, lost contact; doors and chairs without the object terms (§6) |
| Resets and randomization | start frame, robot and object noise, mass, friction, center of mass, joint offsets, object scale, hinge damping (§2, §7) |
| Objects | rigid bodies (boxes, suitcase, ball, board, stool), hinged bodies (doors, folding chair) and static props (§9) |

**Engine difference: non-finite states.** HDMI does not handle NaN: PhysX uses TGS (8 position, 4 velocity iterations)
and enforces joint velocity limits (§0, §3.2). MuJoCo-Warp has no reset of bad states, and Newton's MuJoCo solver does not
support joint velocity limits (solver_mujoco.py:552-553; `HDMISim`'s substitute below). In recorded transitions joint
velocities reach 3.1 times HDMI's limit. At every control step, `train_hoi` and `train_ppo_hoi` check each environment's
post-step state, substep joints, feet and end-effector contact forces, actuator and hinge torques, and observations
(`ndm_gym/algos/hoi_guard.py`). NaN, infinity or a value beyond a loose bound (position 1e3 m, velocity 1e3 m/s or rad/s,
force 1e6 N) flags the environment:

- it terminates (no bootstrap) and resets normally. Isaac Lab's reset write makes NewtonManager.forward
  call SolverMuJoCo.reset, which clears that world's solver state (`qacc_warmstart` and others); `tests/test_hoi_nonfinite.py`
  checks this on a GPU;
- the replay stores finite substitutes (the post-step state taken from before the step, forces 0, joint observations
  from the previous step) marked `bad`, kept out of the world model's window loss and normalizers; PPO zeroes its reward
  and observations before its normalizers;
- every log row records the flagged environments and the updates skipped for non-finite gradients. Training stops with
  an error, without a checkpoint, if parameters or normalizers are non-finite, if more than 1e-4 of a window's env-steps
  are flagged, or after more than 20 consecutive skipped updates. The first flag of a run writes
  `<log_dir>/nonfinite_event.pt` with that environment's last 5 states, actions, observations and forces. `collect_hoi`
  never saves data with non-finite values.

**Engine difference: joint velocity limits.** HDMI's PhysX enforces per-motor joint velocity limits (32/20/37/22 rad/s,
G1:95-99, 122, 130, 138, 163-170). `HDMISim` clamps joint velocities in the simulation state after every physics substep,
so the next substep and joint reads use them. It recomputes link velocities by forward kinematics after the last substep.
PhysX clamps inside the solver and integrates positions with the clamped velocity. Here, the substep that hits a limit
integrates positions and contacts with the unclamped velocity; the next starts from the limit.
PhysX's link velocity limits (1000 m/s, 1000 deg/s) and its 1 m/s penetration recovery velocity (G1:58-60) are not
reproduced. HDMI does not clamp the policy's actions (ACT:97-119; the action distribution is unbounded,
learning/modules/distributions.py:139-158), and neither do we.

## 2. What world-model methods need in addition (HDMI uses PPO)

World models add object dynamics and contact-force predictions for imagined rollouts
([section 3](#3-world-models-and-imagination)). RigidFormer uses one token per object.

## 3. World models and imagination

**State.** The robot keeps the G1 tracking state (root pose and velocity, 29 joints in Unitree order); conversion to
HDMI's order happens once, at the simulator's boundary. A free object's state is its root link's pose and velocity; a door
or folding chair has a fixed base, and its state is the hinge's angle and rate, with the base pose (constant over an
episode) in the context. Static props' poses are in the context too.

**Control.** Given (a_k, the action buffer, ã, d, α), HDMI's delay and smoothing are deterministic, so every substep's PD
target can be computed directly. The world model's input is these 4 substeps' targets (4×23) rather than the raw action;
imagination advances ã and the buffer with the same formulas.

**Context.** The world model sees every per-environment, per-episode random quantity; the policy does not: the 29 joint
offsets, d and α, the pelvis's and torso's center-of-mass offsets, the feet's and wrists' friction,
the links' mass factors, the object's mass, friction and scale, the hinge's f, c and armature, and the props' poses.

**Outputs.** The robot keeps the two parameterizations (SuperTrack predicts accelerations and integrates them, SuperMimic
predicts increments). Objects are predicted the same way; a hinge only predicts its angular acceleration or increment.
Two groups of contact forces are added: each end effector's force from the object (E×3) and the two feet's net contact
forces over the 4 substeps (4×2×3), both supervised by the simulator's sensors. The
simulator records feet forces in the world frame. Because model inputs are invariant to scene rotation about z, the model
predicts these forces in the start state's heading frame. Targets and increments use Hᵀ·F (H the yaw rotation of the start
state's root); errors use the true state's heading frame (one step of rotation from the start); `step` turns them back
with H. End-effector forces use the object's contact body frame, which turns with the scene. The state dict carries
`format_version` (2); loading rejects older HOI models that predicted world-frame feet forces.

**Imagination.** Predicted states and forces form an `HDMIInput` for the simulator's reward, observation and termination
functions (`ndm_gym/tasks/hdmi.py`):

- keypoints: poses from our forward kinematics; linear velocities, by HDMI's definition, are center-of-mass velocities
  (the center-of-mass offsets from the simulator, randomization included);
- joint quantities of the two substep slots: the 4th substep is the step's end state, the 3rd is taken 3/4 of the way
  from the previous step to this one (an approximation);
- actuator torques: by the PD law and its limits, from the interpolated 3rd-substep state (an approximation);
- the feet's contact timers: advanced with the predicted feet forces and a 1 N threshold (not differentiable; a gate only);
- terminations: HDMI needs 25 consecutive steps over a threshold. The forward pass keeps HDMI's count (plus one over the
  threshold, else zero); the gradient is that of a sigmoid of each step's excess (straight-through); the trigger
  probability is a sigmoid of the count, within 5e-5 of HDMI's 0/1 at integer counts. The count at the start comes from
  the replay.

HDMITask's history quantities (angular velocity, gravity, joint angles) are assembled from consecutive replay rows and
filled at an episode's start by HDMI's reset rule. When an episode's first row is the start, the observations are built
from the post-reset input (reference window frames t−1 and t, actuator and hinge torques 0); without task noise, the start
has none either.

**Algorithms.** Actions keep HDMI's teacher's residual form: the policy outputs (q_ref − q0)/scale plus the network's output.

- SuperTrack: the loss is the L1 error of HDMI's tracked quantities (keypoints, joints, object pose or hinge angle,
  contact points). By default the contact term is the end effectors' distance to their contact points where the reference
  is in contact; `--st_contact_force` replaces it by the negative log of HDMI's contact reward factor, the distance / 0.3
  plus the predicted contact force's shortfall below 10 N / 40 (roll_ball: HDMI's version with the better of the two hands,
  0.4 m tolerance and per-axis forces), since HDMI's lost-contact termination needs a force of at least 1 N.
  Options, all off by default:

  - `--st_obj_w k` multiplies the object's three terms (position, orientation or hinge angle, contact) by k after the
    calibration; `--st_obj_imag_noise s` adds Gaussian noise of std s (m/s, rad/s) to the object's velocities (a hinge's
    rate) after every imagined step.
  - `--st_vel_terms` adds three terms, the L1 errors of the velocities HDMI's tracking group tracks: its 14 keypoints'
    linear and angular velocities (world frame) and its 19 tracked joints' velocities, each calibrated to 1 like the
    other terms (the failure penalty, measured in calibrated per-step loss, grows with the term count).
  - `--st_task_penalties` adds HDMI's own loco and feet penalties (action rate, joint velocity, joint position and torque
    limits, landing impact, foot slip, air time, and on truman stumbling; each ≤ 0, air time being min(t_air − 0.5, 0)
    at touchdown), summed with HDMI's weights and negated, survival (a constant) left out; the tracking and object groups'
    exponential kernels never enter SuperTrack. They are weighted at the reward's ratio to its full tracking score, as on
    LAFAN1: the mean penalty per imagined step and rollout times n / full score, n the number of tracking terms (7, 10
    with `--st_vel_terms`), the full score the sum of each enabled tracking and object term's weight times its maximum
    (1 for the exponential kernels, the gain 5 for eef_contact_exp with every end effector in contact): 12 for free
    objects, 11 for the doors and the folding chair, 8 for roll_ball.
  - `--st_late_w w` (LAFAN1's) weighs the tracking terms of the window's step h by 1 + w·h/(H−1), before the calibration.
  - `--st_ref_replay τ`, DiffMimic's demonstration replay: after an imagined step's losses, a rollout whose keypoints are
    on average more than τ m from the step's reference frame continues from that frame's reference state, set as HDMI's
    reset sets it without noise (the pelvis's pose and angular velocity, the root link's velocity from the reference
    center-of-mass velocity, the joints clamped as the reset clamps them; a free object at its reference pose and at
    rest, a hinge at its reference angle and rate), with no gradient through it. Its observation histories are filled
    as at a reset and its termination counts are cleared; the action buffer and filter, the predicted contact forces,
    the feet's timers and the contact-point noise stay the rollout's. The failure penalty is unchanged: the step's
    termination probability is the rollout's, and a failure before a replacement still counts for every later step.
    The replaced rollouts' next state is mixed between steps, outside the compiled step.

  SuperTrack keeps LAFAN1's failure penalty (`--st_fail_w 5`). LAFAN1's balance weights (`--st_balance_w 4`) act on the
  height and up-direction terms, which the HOI loss does not have, so train_hoi refuses that option.
- SuperMimic: the imagined return is the sum of HDMI's four reward groups (times dt); critic and GAE unchanged.

Both algorithms start from their LAFAN1 configurations where the options mean the same here: SuperTrack `--st_fail_w 5`,
SuperMimic `--action_noise 0.3`. SuperMimic's reward-kernel scale (`--reward_std_scale 0.7`) would change HDMI's task, and
the imagination-start perturbation (`--imag_start_perturb 0.3`) is not implemented for HOI, so train_hoi refuses both.
LAFAN1's `--wm_input_noise`, `--wm_smooth_w` (with `--wm_smooth_eps`, `--wm_smooth_dirs`) and `--pi_rollout_clip` are
available to both algorithms with LAFAN1's meaning, off by default; the smoothness penalty only uses the starts of the
windows the window loss keeps.

**Observations.** By default both algorithms' policies read HDMI's student groups (command, policy, object), which do not
contain the robot's global offset from the reference or the object's offset from its future trajectory; HDMI's PPO
teacher (our PPO baseline) also reads the privileged group. `--priv_obs` (both algorithms, off by default) gives the
policy the privileged group too, encoded as the teacher encodes it: the privileged and object groups go through an MLP
to 256 features, which join the other inputs into the decoder.

**Numerics and memory.** The 3×3 products in targets, increments and errors are computed elementwise, so they keep fp32
precision when training enables TF32 (A100: rotation error 4.5e-4 for batched products, 9e-8 elementwise). The
RigidFormer recomputes activations when its parameters are frozen (the policy's BPTT), by default as on LAFAN1
(`checkpoint="frozen"`): move_suitcase, pi_batch 512, horizon 32, peak memory per policy update on an A100 from 11.53 to
2.39 GiB (SuperTrack) and from 11.68 to 2.53 GiB (SuperMimic), time from 745 to 771 ms and from 725 to 836 ms.

**Checks.** The world models' open-loop errors on held-out data, the imagined `HDMIInput` against the simulator's in the
same state term by term, training curves, and a replay video per task.
