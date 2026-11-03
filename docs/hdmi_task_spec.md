# HDMI human–object-interaction (HOI) task: reimplementation spec

Source: LeCAR-Lab/HDMI, commit `32282f6`, read from that commit without running any of its code.
References are `ALIAS:line`, relative to the repo root (aliases are in the table below). Binary USD assets were decoded with a
standalone pure-python USDC reader; they are cited as `USD:file⟨prim path⟩`. A sample of those values (door, foldchair,
stool, bread box, robot foot/hand/elbow shapes, masses, joint limits) was re-read with the official `pxr` library and matched
exactly. `[IL]` marks behavior that lives inside
IsaacLab v2.2 / IsaacSim 4.5 (README.md:26-51), not in this repo: it is stated from IsaacLab's source and not verified here.

| alias | file | alias | file |
|---|---|---|---|
| Y | cfg/task/base/hdmi-base.yaml | T:x | cfg/task/G1/hdmi/x.yaml |
| CMD | active_adaptation/envs/mdp/commands/hdmi/command.py | OBS | …/commands/hdmi/observations.py |
| REW | …/commands/hdmi/rewards.py | TRM | …/commands/hdmi/terminations.py |
| HRN | …/commands/hdmi/randomizations.py | ACT | active_adaptation/envs/mdp/action.py |
| RND | active_adaptation/envs/mdp/randomizations.py | MB | active_adaptation/envs/mdp/base.py |
| ENV | active_adaptation/envs/base.py | LOC | active_adaptation/envs/locomotion.py |
| MOT | active_adaptation/utils/motion.py | G1 | active_adaptation/assets/g1.py |
| OBJ | active_adaptation/assets/objects.py | SPN | active_adaptation/assets/spawn.py |
| OC | active_adaptation/envs/mdp/observations/common.py | OP | active_adaptation/envs/mdp/observations/priv_body.py |
| RC | active_adaptation/envs/mdp/rewards/common.py | RF | active_adaptation/envs/mdp/rewards/feet.py |
| UM | active_adaptation/utils/math.py | UI | active_adaptation/utils/\_\_init\_\_.py |
| PPO | active_adaptation/learning/ppo/ppo_roa.py | PC | active_adaptation/learning/ppo/common.py |
| HLP | scripts/helpers.py | TRN | scripts/train.py |
| TER | active_adaptation/envs/terrain/regular.py | REC | record_motion/x65r1823/meta.json (simulator joint/body order, written by CMD:205-212) |
| USD:o/x | active_adaptation/assets/objects/x | USD:g1/x | active_adaptation/assets/g1/x |

`envs/humanoid.py` is dead code for HDMI: it imports `Env, LocomotionEnv` (humanoid.py:15), which `locomotion.py` does not define.
The env class actually used is `SimpleEnv` (LOC:9, envs/\_\_init\_\_.py:2, HLP:115-130).

**Notation.**
* Quaternions are (w,x,y,z). R(q) is the rotation matrix with v_world = R(q)·v_body. q* is the conjugate and ⊗ the Hamilton product.
* yaw(q) = (cos ψ/2, 0, 0, sin ψ/2) with ψ = atan2(2(wz+xy), 1−2(y²+z²)) [IL yaw_quat].
* ang(q) = ‖axis_angle(q)‖ ∈ [0,π], after canonicalizing w ≥ 0 [IL].
* p̄ is p with z set to 0.
* n₃ = clamp(N(0,1),−3,3) and n₁ = clamp(N(0,1),−1,1).
* U[a,b] is the uniform distribution.
* F = (1,2,8,16,32) are the future steps (Y:93).
* K = 16 tracking keypoints and J = 23 tracking/action joints.
* E = number of contact end-effectors and C = number of columns of `object_contact`.
* dt = 0.02 s.

---------------------------------------------------------------------------------------------------------------------

## 0. Simulation constants, step order, data timing

| item | value | ref |
|---|---|---|
| control step | 0.02 s | Y:17 |
| physics step | 0.005 s (`isaac_physics_dt`) | Y:18, LOC:174-175 |
| decimation | int(0.02/0.005) = 4 | ENV:137-139 |
| `mujoco_physics_dt` | 0.002, MuJoCo sim2sim backend only, unused in training | Y:19 |
| num_envs | 4096 | Y:13 |
| max episode length | 1000 steps (move_suitcase 500, truman 4000) | Y:14, T:move_suitcase:21, T:truman:16 |
| ground | plane; static=dynamic friction 1.0, restitution 1.0, combine `multiply` | Y:11, TER:28-38, TER:519 |
| PhysX | `enable_stabilization=False` | LOC:191 |
| robot articulation | solver iterations pos 8 / vel 4, self-collision off, max depenetration velocity 1.0, no damping | G1:53-64 |
| door/foldchair articulation | solver iterations pos 4 / vel 0, self-collision off | OBJ:47-51, OBJ:309-314 |

**One env step** (ENV:470-519):
1. For each substep i = 0..3 (ENV:472-485):
   * The action manager writes PD targets (ACT:97-119).
   * `scene.write_data_to_sim`, which includes the hinge torque of the door/foldchair (OBJ:33-40).
   * One physics step, then `scene.update`.
   * post_step hooks: joint_pos_history (OC:101-102), joint_vel_l2 (RC:131-132), root_linvel_b EMA (OP:140-141).
2. `_update()` (ENV:444-468) runs the update hooks in registration order: randomizations (no-ops here), every observation term's `update()`, reward `update()`, termination `update()` (ENV:219-244, 259-315). Then episode_length += 1 (ENV:465).
3. Rewards (ENV:406-427).
4. Command update **CU** (ENV:494-499). The code comment says it must run after the rewards.
5. Observations (ENV:501). A term returns the value it stored in `update()` (stage 2) or computes it now in `compute()`.
6. Termination and truncation:
   * terminated = OR of all termination terms (ENV:429-442, 502).
   * truncated = (episode_length ≥ max_episode_length) OR (t ≥ L) (ENV:503-505, CMD:229-233).
   * done = terminated OR truncated.

**Reset** (ENV:371-389, LOC:243-250):
1. `CMD.sample_init` (§2).
2. `scene.reset` [IL: clears contact-sensor force history and air/contact times].
3. episode_length = 0.
4. Reset hooks in this order: command (no-op, MB:76-77), action (ACT:86-95), randomizations (RND:742-744, HRN:107-112), observations, rewards, terminations (TRM:28-29).

CU is **not** run at reset. The reset returns **all-zero tensors for every observation group** (ENV:385-386), so the first action of each episode is computed from zeros (after observation normalization).

**CU** (CMD:235-290, 536-584), with t = counter before the increment:
1. Pose/joint/object references are loaded for frames t+F (CMD:247, MOT:336), so reference index 0 is frame t+1.
2. t ← t+1 (CMD:290).
3. Contact flags are loaded for frames t+F using the **incremented** t (CMD:565-568). Contact flags therefore lead the pose references by one frame.
4. Sim state is snapshotted into command buffers:
   * robot keypoint pose/velocity and joints (CMD:263-268, copies);
   * robot root pose (CMD:269-270);
   * object root pose (CMD:554-555) and object joint (CMD:562-563);
   * contact target points, end-effector points and filtered contact forces (CMD:571-584).

**Timing table.** s = start frame, n = steps since reset (n ≥ 1), f_n = s+n. f_n is the reference frame the robot should match after n steps; CU(n) sets index-0 reference = f_n.

| consumer at step n | reference used | state used |
|---|---|---|
| rewards | CU(n−1): pose f_{n−1}, contact flag f_n | direct sim reads = time n; command snapshots (root pose in `*_local`, object, contact points/forces) = time n−1 |
| terminations | CU(n−1) | body and object terms read the sim at time n (`*_local` also reads the root fresh, TRM:94-95); lost-contact uses snapshots (n−1) |
| observation terms computed in `update()` | CU(n−1): frames f_{n−1}+(0,1,7,15,31) | direct sim reads = n; command snapshots = n−1 |
| observation terms computed in `compute()` | CU(n): frames f_n+(0,1,7,15,31); contact flags f_n+F | n |
| n = 1 | CU buffers still hold the previous episode's last CU (no CU at reset) | — |

PPO drops the first two transitions of every episode from the policy and value losses (`valid = step_count > 1`, PPO:670-674, 685, 691). That covers both the zero observation and the stale n = 1 values.

---------------------------------------------------------------------------------------------------------------------

## 1. Motion data

### 1.1 File format (`data/motion/<clip>/{motion.npz, meta.json}`; README.md:67-72)

| key | shape / type | content |
|---|---|---|
| `meta.json: body_names` | list (28 robot links + object bodies) | robot links in unitree order (same as MOT:46-75), then the main object, then extra objects. Object bodies are named by the **scene asset name** (`door`, `foldchair`, `suitcase`…), not by the asset body (`door_panel`) |
| `meta.json: joint_names` | list | G1 joints in unitree order (MOT:14-44); some clips omit `*_wrist_pitch/yaw`; articulated-object joint appended (`door_joint` / `foldchair_joint`) |
| `meta.json: fps` | 50.0 for all 13 HOI clips | |
| `body_pos_w` | [T,B,3] float64 | world-frame link positions (m) |
| `body_quat_w` | [T,B,4] float64 | world-frame orientations (w,x,y,z) (MOT:97-98 converts wxyz⇄xyzw for slerp) |
| `body_lin_vel_w`, `body_ang_vel_w` | [T,B,3] | world-frame. Verified: consistent with finite differences of pos/quat (mean abs deviation ≈0.00–0.07 m/s); angular velocity is expressed in the world frame |
| `joint_pos`, `joint_vel` | [T,Jm] | |
| `object_contact` | [T,C] bool (float64 0/1 in push_box) | reference "end-effector should be in contact" flags; C=1 is shared by all end-effectors, C=2 is one column per end-effector in `contact_eef_body_name` order (broadcast REW:405-407, TRM:187-189) |

### 1.2 Loading, resampling, indexing

* Target frame rate = int(1/step_dt) = 50 (CMD:54-58).
* `data_path` is a regex, fullmatched against the parent directory of each `motion.npz`, relative to the repo root (MOT:206-217).
* All matched `meta.json` files must be identical (MOT:224-235). `RobotObjectTracking` requires exactly 1 clip (CMD:442-443).
* Resampling, only when fps ≠ 50 (MOT:103-121):
  * The target grid is arange(0, T/fps, 1/50); a last sample beyond the source end is dropped.
  * `np.interp` linear interpolation for positions, velocities and joints; scipy slerp for quaternions.
  * Any extra npz key (e.g. `object_contact`) raises NotImplementedError (MOT:106-108). HOI clips therefore must already be 50 fps (all 13 are).
* Joint remap (MOT:247-267):
  * Output joints = the 29 simulator joints in simulator order, followed by clip-only joints (`door_joint`, `foldchair_joint`).
  * Simulator joints absent from the clip get pos = vel = 0.
* Bodies are kept in meta order and looked up by name (CMD:62, 75, 370, 383).
* Clips are concatenated (MOT:280-300). `get_slice(id, t, steps)` uses idx = start + t + steps, clamped to end−1 (MOT:333-338), so references past the last frame **hold the last frame**.
* `object_contact` is read separately from the same npz, not resampled, and cast to bool (CMD:441-446). It is indexed with motion_start + t + F and the same clamp (CMD:565-567).
* All reference positions are shifted by the env origin (CMD:171, 251, 257, 470, 485, 550).

### 1.3 Start frame, advance, end

| item | rule | ref |
|---|---|---|
| start frame (training) | s = floor(u·(L−32)), u ~ U[0,1), so s ∈ {0,…,L−33}; `reset_range` unset | CMD:137-140, 151; CMD:93 |
| start frame (eval or `env.training=False`) | s = 0 | CMD:144-145 |
| clip / motion id | sampled once at the first reset; single clip | CMD:126-133 |
| advance | +1 motion frame per control step (t+=1 in every CU) | CMD:290 |
| clip end | truncate when t ≥ L, so the episode has ≤ L−s steps; references past the end are clamped | CMD:229-233, MOT:337 |
| success statistic | t ≥ L−1 (statistics only) | CMD:225-227, ENV:423-424 |

---------------------------------------------------------------------------------------------------------------------

## 2. Reset / initialization (training; m = motion frame s)

| quantity | value | ref |
|---|---|---|
| robot root position | p_ref,pelvis(m) + origin + δp, δp_i ~ U(pose_range x,y,z) | CMD:162, 168, 171 |
| robot root orientation | q_ref,pelvis(m) ⊗ quat_from_euler_xyz(δroll,δpitch,δyaw) [IL: R = Rz·Ry·Rx], δ ~ U(pose_range) | CMD:163, 172-173 |
| robot root velocity (COM, world) | [v_ref,pelvis(m), ω_ref,pelvis(m)] + δ, δ ~ U(velocity_range) per component | CMD:164-165, 176-182 |
| robot joints (all 29) | q = q_ref(m) + 0.1·U[−1,1]; q̇ = q̇_ref(m) + 0.1·U[−1,1]; then clamp q to soft limits and \|q̇\| to the velocity limit | CMD:184-198, Y:90-91 |
| soft joint limits | mid ± 0.5·0.9·(upper−lower) [IL formula; factor 0.9 from G1:80] | G1:80 |
| object root pose | p_ref,obj(m) + δp + origin; q_ref,obj(m) ⊗ quat_from_euler_xyz(δ); δ ~ U(object_pose_range) (independent draw) | CMD:460-471, 474 |
| object velocity | **0** (not the reference velocity) | CMD:472, 475 |
| extra objects | own reference body pose(m) + **the same** δp and δrot as the main object; velocity 0 | CMD:477-490 |
| object joint (door, chair) | q = q_ref(m) + U[−1,1]·0.1, q̇ = q̇_ref(m) + U[−1,1]·0.1 (CMD:356-357 defaults; foldchair 0/0), clamped to the object's soft limits (factor 1.0 [IL default]) | CMD:499-514 |
| eval | robot pose/velocity noise set to 0 (CMD:169-170, 177-178); object and joint noise not zeroed | |

Defaults:
* `pose_range` = x,y ±0.05 m; z ±0.01; roll,pitch ±0.1 rad; yaw ±0.2 (Y:74-80).
* `velocity_range` = lin ±0.2 m/s, ang ±0.5 rad/s (Y:82-88).
* `object_pose_range` = x,y ±0.05; z ±0.01; roll,pitch,yaw ±0.1 (Y:100-106).
* Missing keys default to (0,0) (CMD:95-98, 385-386).
* Per-task overrides are in §8.

---------------------------------------------------------------------------------------------------------------------

## 3. Action, actuation, robot model

### 3.1 `JointPosition` (ACT:37-119, Y:25-46)

* **Actuated joints**: the keys of `action_scaling` = 23 joints, i.e. all joints except the 6 wrist joints (Y:27-43).
* **Action order**: action index i = i-th matched joint in simulator order (ACT:48-52 [IL `resolve_matching_names_values` keeps target order]). In the simulator order of REC:3-31 these are exactly simulator joints 0–22 (table §3.3).
* **PD target**, per physics substep (ACT:117-119):
  * action joints: q*_j = q0_j + o_j + scale_j·ã_j;
  * wrist joints: q*_j = q0_j + o_j.
  * q0 = default joint pose (§3.3), o = random joint offset (§7).
* **No clipping** of the policy output anywhere (ACT:97-119). The policy samples an unbounded Gaussian (learning/modules/distributions.py:139-158).
* **Buffer**: history length H = max((6−1)//4+1, 3) = 3 (ACT:71-74). At substep 0 of step k: buf[:,:,1:] ← buf[:,:,:−1], buf[:,:,0] ← a_k (ACT:98-102).
* **Delay**: d ~ randint{2,…,6} per env per reset (ACT:90-91, Y:44-45). Units are **physics substeps (5 ms)**, so 10–30 ms.
  * At substep i: idx = (d − i + 4 − 1)//4 = ⌈(d−i)/4⌉ and a_used = buf[:,:,idx] = a_{k−idx} (ACT:113-114).
  * This is exactly a pure transport delay of d substeps with zero-order hold.
* **Filter** (first-order low-pass, applied **every substep**, 4×/step): ã ← ã + α(a_used − ã) (ACT:115). α ~ U[0.8,1.0] per env per reset (ACT:92-95, Y:46).
* **Reset**: buf = 0, ã = 0 (ACT:87-88).
* **Teacher residual**: the ROA teacher's action mean is loc = (q_ref(f_n) − q0)/scale + MLP(·) (PPO:223-236, OBS:37-50). See §11.

### 3.2 Actuator model (G1:10-199)

* Implicit PD solved by PhysX: τ = kp(q* − q) + kd(0 − q̇), with \|τ\| ≤ effort_limit_sim and \|q̇\| ≤ velocity_limit_sim [IL ImplicitActuator]. The velocity target is never set, so it stays 0.
* Gains: ω = 2π·10 rad/s (G1:15), ζ = 2 (G1:16), kp = I_a·ω², kd = 2ζ·I_a·ω (G1:18-26), where I_a is the motor armature (also applied as joint armature).
* Motor armatures: 5020 = 0.003609725, 7520-14 = 0.010177520, 7520-22 = 0.025101925, 4010 = 0.00425 (G1:10-13).
* Feet and waist roll/pitch use 2× the 5020 values (G1:120-135).

| motor class | kp | kd | armature |
|---|---|---|---|
| 7520-14 | 40.179238 | 2.557890 | 0.010177520 |
| 7520-22 | 99.098428 | 6.308802 | 0.025101925 |
| 2×5020 (ankles, waist r/p) | 28.501246 | 1.814446 | 0.007219450 |
| 5020 | 14.250623 | 0.907223 | 0.003609725 |
| 4010 | 16.778327 | 1.068142 | 0.004250000 |

Robot config overrides are merged recursively into the defaults; they do not replace the dicts (UI:24-31, 94-106; LOC:62-64).

### 3.3 Per-joint table (simulator order, REC:3-31; limits from `USD:g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd⟨*_joint.physics:lower/upperLimit⟩`)

| sim idx | joint | action scale (Y) | q0 (G1:66-79) | kp / kd / armature (G1) | τ_max N·m | q̇_max rad/s | limits rad |
|---|---|---|---|---|---|---|---|
| 0,1 | L/R hip_pitch | 0.55 (Y:30) | −0.312 | 7520-14 (G1:102,108,114) | 88 | 32 | [−2.5307, 2.8798] |
| 2 | waist_yaw | 0.55 (Y:36) | 0 | 7520-14 (G1:136-143) | 88 | 32 | [−2.6180, 2.6180] |
| 3,4 | L/R hip_roll | 0.35 (Y:29) | 0 | 7520-22 | 139 | 20 | L [−0.5236, 2.9671]; R [−2.9671, 0.5236] |
| 5 | waist_roll | 0.44 (Y:34) | 0 | 2×5020 (G1:128-135) | 50 | 37 | [−0.52, 0.52] |
| 6,7 | L/R hip_yaw | 0.55 (Y:28) | 0 | 7520-14 | 88 | 32 | [−2.7576, 2.7576] |
| 8 | waist_pitch | 0.44 (Y:35) | 0 | 2×5020 | 50 | 37 | [−0.52, 0.52] |
| 9,10 | L/R knee | 0.35 (Y:31) | 0.669 | 7520-22 | 139 | 20 | [−0.0873, 2.8798] |
| 11,12 | L/R shoulder_pitch | 0.44 (Y:37) | 0.2 / 0.2 | 5020 (G1:173) | 25 | 37 | [−3.0892, 2.6704] |
| 13,14 | L/R ankle_pitch | 0.44 (Y:32) | −0.363 | 2×5020 (G1:120-127) | 50 | 37 | [−0.8727, 0.5236] |
| 15,16 | L/R shoulder_roll | 0.44 (Y:38) | +0.2 / −0.2 | 5020 | 25 | 37 | L [−1.5882, 2.2515]; R [−2.2515, 1.5882] |
| 17,18 | L/R ankle_roll | 0.44 (Y:33) | 0 | 2×5020 | 50 | 37 | [−0.2618, 0.2618] |
| 19,20 | L/R shoulder_yaw | 0.44 (Y:39) | 0 | 5020 | 25 | 37 | [−2.6180, 2.6180] |
| 21,22 | L/R elbow | 0.44 (Y:40) | 0.6 | 5020 | 25 | 37 | [−1.0472, 2.0944] |
| 23,24 | L/R wrist_roll | — | 0 | 5020 (kp 4.0 in 4 tasks*) | 25 | 37 | [−1.9722, 1.9722] |
| 25,26 | L/R wrist_pitch | — | 0 | 4010 (kp 4.0*) | 5 | 22 | [−1.6144, 1.6144] |
| 27,28 | L/R wrist_yaw | — | 0 (−0.4 / +0.4 in 6 tasks†) | 4010 (kp 4.0*) | 5 | 22 | [−1.6144, 1.6144] |

Table notes:
* q0 comes from G1:69-77; joints not listed there default to 0 [IL].
* (*) Wrist kp = 4.0 for roll, pitch and yaw (kd and armature unchanged) in carry_and_place_bread_box, carry_box_over_shoulder, move_foam and topple_wood_board_and_cross (e.g. T:carry_and_place_bread_box:31-36).
* (†) Default wrist yaw L −0.4 / R +0.4 in those four tasks plus move_stool-climb and move_suitcase (e.g. T:move_suitcase:27-30).

Tracking joints (Y:63-70) are the same 23 joints in the same order. Their reference arrays are 23-wide.

### 3.4 Robot bodies, mass, collision

* **Simulator body order**, 39 bodies (REC:34-72):

  pelvis, imu_in_pelvis, left_hip_pitch_link, pelvis_contour_link, right_hip_pitch_link, waist_yaw_link, left_hip_roll_link,
  right_hip_roll_link, waist_roll_link, left_hip_yaw_link, right_hip_yaw_link, torso_link, left_knee_link, right_knee_link,
  d435_link, head_link, imu_in_torso, left_shoulder_pitch_link, logo_link, mid360_link, right_shoulder_pitch_link,
  left_ankle_pitch_link, right_ankle_pitch_link, left_shoulder_roll_link, right_shoulder_roll_link, left_ankle_roll_link,
  right_ankle_roll_link, left_shoulder_yaw_link, right_shoulder_yaw_link, left_elbow_link, right_elbow_link,
  left_wrist_roll_link, right_wrist_roll_link, left_wrist_pitch_link, right_wrist_pitch_link, left_wrist_yaw_link,
  right_wrist_yaw_link, left_rubber_hand, right_rubber_hand.

  The `nohand` USD has the same bodies minus the two rubber hands (37 bodies).
* **Tracking keypoints** K = 16 (Y:52-61), in simulator order: pelvis, L/R hip_pitch, L/R hip_yaw, torso, L/R knee, L/R shoulder_pitch, L/R ankle_roll, L/R elbow, L/R wrist_yaw (CMD:61-63).
* **Mass** (USD `physics:mass`): total 33.341 kg.
  * pelvis 3.813, torso 6.78, head 1.036.
  * hip_pitch 1.35, hip_roll 1.52, hip_yaw 1.702, knee 1.932, ankle_pitch 0.074, ankle_roll 0.608.
  * shoulder pitch 0.718 / roll 0.643 / yaw 0.734, elbow 0.600.
  * wrist roll 0.0854 / pitch 0.484 / yaw 0.0846, rubber_hand 0.17.
  * waist_yaw 0.214, waist_roll 0.086.
  * sensors and logo ≈0.
  * The rubber hand is fixed to wrist_yaw_link at (0.0415, 0.003, 0) (⟨left_hand_palm_joint⟩).
* **Collision shapes** (USD `*/collisions*` prims, link frame). Every link not listed has none. Self-collisions are off.

| link | shape (USD) |
|---|---|
| ankle_roll, `feet_sphere` | 4 spheres r=0.005 at (−0.05, ±0.025, −0.03) and (0.12, ±0.03, −0.03) |
| ankle_roll, `feet_box` | box 0.17×0.05×0.01 centered (0.035, 0, −0.03) |
| wrist_yaw, `eef_box` | box 0.10×0.05×0.05 @ (0.02,0,0); box 0.10×0.02×0.10 @ (0.09,0,0); box 0.05×0.02×0.10 @ (0.15, ∓0.01, 0) rotated ∓0.4 rad about z (left: −0.01 and −0.4; right: +0.01 and +0.4) |
| wrist_yaw, `nohand…eef_L` | plate 0.06×0.08×0.004 @ (0.13,0,0); plate 0.004×0.08×0.06 @ (0.10,0,−0.03) |
| pelvis_contour_link | box 0.1³ @ (0,0,−0.08) |
| torso_link | 2 capsules r=0.05, cylinder length 0.15, axis z, @ (0, ±0.05, 0.15) |
| hip_roll_link | capsule r=0.05, length 0.06, axis z @ (0.02,0,−0.05) |
| hip_yaw_link | capsule r=0.04, length 0.08 @ (−0.02,0,−0.10), axis z rotated 0.4 rad about y |
| knee_link | capsule r=0.03, length 0.20, axis z @ (0,0,−0.16) |
| elbow_link (rubberhand USDs only) | capsule r=0.03, length 0.10 @ (0.05,0,0), axis rotated 1.570 rad about y (along x) |

Robot types: `g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule` (spheres) and `…-feet_box-…` (box feet) differ **only** in the ankle collision.

Other robot details:
* Contact sensor `contact_forces` on `.*(ankle_roll|wrist_.*)_link` (8 bodies), history 3 physics steps, air time tracked (LOC:113-117). Contact threshold [IL] 1.0 N.
* No physics material is authored on the robot. Non-randomized shapes use the scene default material [IL default μs=μd=0.5, restitution 0].

---------------------------------------------------------------------------------------------------------------------

## 4. Observations

Group order = YAML order. Per-task additions are appended at the end of their group.

Groups not used by the algorithm are deleted, except names ending in `_` (HLP:120-128). For `ppo_roa_*` the groups kept are `command`, `policy`, `object`, `priv` and `ref_joint_pos_`. The `depth` group is dropped (PPO:86, Y:134-135; cameras off Y:115).

All float groups except `ref_joint_pos_` go through a running-mean/std normalizer: torchrl VecNorm, decay 0.9999 (HLP:138-146).

Each term is a separate instance with its own buffers, noise and history.

**Timing column.** `U` = computed in `update()` from CU(n−1) command buffers; `C` = computed at observation time from CU(n); `S` = direct read of the simulator (time n). See §0.

**Symbols.**
* r0 = reference pelvis at index 0 (CMD:280-281).
* `snap` = robot root pose from the command snapshot (CMD:269-270).
* p_obj, q_obj = object **root link** pose read from the sim (OBS:344, 374, 396, 414, 469, 472, 488). This is door_frame for the door and base_link for the foldchair.
* c_e = contact target point and e_e = end-effector point (§5.3), both from CU(n−1).

### 4.1 `command` group (actor input) — 356 dims

| term | dim | definition | timing | noise | ref |
|---|---|---|---|---|---|
| ref_body_pos_future_local | 5·16·3=240 | R(yaw q_r0)ᵀ (p_ref,b(k) − p̄_r0), k∈F, b∈K | U | none | OBS:95-115, Y:121 |
| ref_joint_pos_future | 5·23=115 | q_ref,j(k), k∈F, tracking joints | C | none | OBS:18-20, CMD:255, Y:122 |
| ref_motion_phase | 1 | t/L (t after CU) | C | none | OBS:255-257, Y:123 |

### 4.2 `policy` group (proprioception) — 249 dims

| term | dim | definition | timing | noise | ref |
|---|---|---|---|---|---|
| root_ang_vel_history | 3 | root angular velocity in the root frame [IL root_ang_vel_b]; history steps [0] | S | +n₃·0.05 when stored | OC:14-39, Y:125 |
| projected_gravity_history | 3 | R(q_root)ᵀ(0,0,−1) [IL]; [0] | S | +n₃·0.05 then renormalized to unit length | OC:41-68, Y:126 |
| joint_pos_history | 6·29=174 | q̂_j − o_j for all 29 joints; q̂ = mean of q at physics substeps 2 and 3 (`substep%2` slots) + n₃·0.015; history [0,1,2,3,4,8] control steps; slot 0 = newest | S | n₃·0.015 | OC:70-121, Y:127 |
| prev_actions | 3·23=69 | raw policy actions (a_n, a_{n−1}, a_{n−2}), flattened as index j·3+k | C | none | OC:123-138, Y:128 |

History buffers are filled with the post-reset value at reset (OC:24-29, 51-57, 104-108). Projected gravity is stored without noise at reset (OC:57).

### 4.3 `object` group — 4 + 3E dims (10 for E=2, 7 for E=1)

| term | dim | definition | timing | noise | ref |
|---|---|---|---|---|---|
| object_xy_b | 2 | [R(yaw q_snap)ᵀ (p_obj − p_snap)]_xy | U (object S, root snapshot) | +n₃·0.01 per step; episodic 0 | OBS:324-352, Y:131 |
| object_heading_b | 2 | (cos φ, sin φ), φ = wrap_to_pi(ψ(q_obj) − ψ(q_snap)) + noise | U | n₃·0.01 rad; episodic 0 | OBS:354-384, OBS:260-263, Y:132 |
| ref_contact_pos_b | 3E | R(yaw q_snap)ᵀ (c_e − p_snap) (root z not zeroed) | U | +0.01·n₃ + 0.01·n₁ per step, + ε_e with ε_e ~ 0.02·U[−1,1]³ drawn per env, per end-effector, per reset | OBS:267-303, Y:133 |

### 4.4 `priv` group (teacher encoder and critic) — 1703 + 5C + 3E (+3 for articulated objects)

| term | dim | definition | timing | ref |
|---|---|---|---|---|
| root_ang_vel_history | 27 | as policy, history [0..8], no noise | S | Y:138 |
| projected_gravity_history | 27 | history [0..8], no noise | S | Y:139 |
| joint_pos_history | 261 | 29 joints × history [0..8], no noise, minus offset | S | Y:140 |
| ref_root_pos_future_b | 15 | R(q_snap)ᵀ (p_ref,pelvis(k) − p_snap), full 3-D rotation | U | OBS:52-70, Y:142 |
| ref_root_ori_future_b | 30 | rows 0–1 of R(q_snap* ⊗ q_ref,pelvis(k)) | U | OBS:72-93, Y:143 |
| diff_body_pos_future_local | 240 | R(yaw q_r0)ᵀ(p_ref,b(k) − p̄_r0) − R(yaw q_snap)ᵀ(p_b,snap − p̄_snap) | U | OBS:140-168, Y:145 |
| diff_body_ori_future_local | 480 | rows 0–1 of R((yaw(q_snap)*⊗q_b,snap)* ⊗ (yaw(q_r0)*⊗q_ref,b(k))) | U | OBS:196-228, Y:146 |
| diff_body_lin_vel_future_local | 240 | R(yaw q_r0)ᵀ v_ref,b(k) − R(yaw q_snap)ᵀ v_b,snap (robot COM velocity) | U | OBS:170-193, Y:147 |
| diff_body_ang_vel_future_local | 240 | same with ω | U | OBS:230-253, Y:148 |
| root_linvel_b | 3 | R(q_root)ᵀ v_root; v = root COM linear velocity at the last substep (EMA with γ=0) | S | OP:128-152, UM:148-176, Y:150 |
| body_pos_b | 6 | ankle_roll links: R(yaw q_root)ᵀ(p_f − p̄_root) | S | OP:15-34, Y:151 |
| body_vel_b | 6 | ankle_roll links: R(q_root)ᵀ v_f (COM linear velocity; full root rotation) | S | OP:45-63, Y:152 |
| body_height | 4 | z − ground height (plane: z), for [pelvis, torso, L ankle_roll, R ankle_roll] (simulator order) | S | OP:170-181, ENV:530-548, Y:153 |
| applied_action | 23 | ã after the last substep | S | OC:146-152, Y:155 |
| applied_torque | 29 | [IL implicit estimate] clip(kp(q*−q) − kd q̇, ±τ_max) | S | OC:159-167, Y:156 |
| object_pos_b | 3 | R(q_snap)ᵀ(p_obj − p_snap) | U | OBS:387-403, Y:158 |
| object_ori_b | 9 | R(q_snap* ⊗ q_obj), 3×3 row-major | U | OBS:405-424, Y:159 |
| diff_object_pos_future | 15 | R(q_obj)ᵀ(p_ref,obj(k) − p_obj) | U | OBS:459-476, Y:160 |
| diff_object_ori_future | 45 | R(q_obj* ⊗ q_ref,obj(k)), 3×3 | U | OBS:478-497, Y:161 |
| ref_object_contact_future | 5C | contact flags at frames t+F (bool, cast to 0/1 by concatenation) | C | OBS:509-511, CMD:565-567, Y:163 |
| diff_contact_pos_b | 3E | R(q_snap)ᵀ (c_e − e_e) | U | OBS:305-322, Y:164 |
| object_joint_pos, object_joint_vel, object_joint_torque (door ×3 tasks, foldchair) | 1+1+1 | hinge q, q̇ (CU(n)), hinge torque actually applied (§9) | C | OBS:426-445, T:open_door-feet:42-46 |

Priv dimensions per task:
* 1714: all tasks with E=2, C=1; also open_door-feet and push_door-hand (E=1, C=1, +3 joint terms).
* 1717: open_foldchair-sit.
* 1719: truman (C=2).
* 1722: traverse_door (C=2, +3).

### 4.5 `ref_joint_pos_` group (not normalized) — 23 dims

ref_joint_pos_action_policy = (q_ref(f_n) − q0)/scale over the action joints (C). It is used only as the teacher residual base (OBS:37-50, Y:166-167, PPO:223-236).

---------------------------------------------------------------------------------------------------------------------

## 5. Rewards

### 5.1 Combination

* Each term returns weight × value (MB:212-220).
* Group reward = Σ over **enabled** terms (ENV:617-647). `_multiplicative: true` would take the product instead (ENV:268-271, 644-645); it is commented out in Y:217, so every group is a **sum**.
* Every group is multiplied by dt = 0.02 (`_mult_dt_` defaults to True, ENV:252, 412-416).
* The env emits a **5-dim reward vector** [tracking, object_tracking, loco, feet, debug] (ENV:259-289, Y:170-241).
* The critic predicts one value per group (PPO:268-272). GAE is computed per group (PC:195-225, γ 0.99, λ 0.95). Advantages are scaled by 1/5, summed and normalized (PPO:169-172, 633-639). Net effect: the policy optimizes the **sum** of the five groups.
* Disabled terms are still evaluated (for logging) but not added (ENV:625-638).

### 5.2 Tracking group (REW:14-322, Y:171-215)

Every tracking term is r = exp(−mean_b e_b / σ). The error e is an **unsquared** norm or absolute value. Tolerance is 0 (REW:15, 39-47, 58).

Robot data: positions are link frames and velocities are COM velocities, read from the sim at time n (REW:54, 64, 177, 188, 247, 256, 300, 317). The reference is CU(n−1) index 0 = frame f_{n−1}.

| term (Y line) | weight | σ | bodies / joints | e | ref |
|---|---|---|---|---|---|
| tracking_upper_body_pos (172-175) | 0.5 | 0.5 | L/R shoulder_pitch, elbow, wrist_yaw (6) | ‖R(yaw q_ref,root)ᵀ(p_ref,b − p̄_ref,root) − R(yaw q_snap)ᵀ(p_b − p̄_snap)‖ | REW:62-89 |
| tracking_upper_body_ori (176-179) | 0.5 | 1.0 | same 6 | ang((yaw(q_ref,root)*⊗q_ref,b)* ⊗ (yaw(q_snap)*⊗q_b)) | REW:186-208 |
| tracking_lower_body_pos (181-184) | 0.5 | 0.5 | L/R hip_pitch, knee, ankle_roll (6) | as upper_body_pos | REW:62-89 |
| tracking_lower_body_ori (185-188) | 0.5 | 1.0 | same 6 | as upper_body_ori | REW:186-208 |
| tracking_root_pos (190-193) | 0.5 | 0.5 | pelvis | ‖p_ref − p‖ (world) | REW:52-60 |
| tracking_root_ori (194-197) | 0.5 | 0.5 | pelvis | ang(q_ref* ⊗ q) | REW:175-184 |
| tracking_body_linvel (199-202) | 0.5 | 0.5 | pelvis, L/R hip_yaw, knee, ankle_roll, torso, L/R shoulder_pitch, elbow, wrist_yaw (14) | ‖v_ref − v‖ (world) | REW:245-253 |
| tracking_body_angvel (203-206) | 0.5 | 2.5 | same 14 | ‖ω_ref − ω‖ (world) | REW:254-262 |
| joint_pos_tracking_product (208-211) | 0.5 | 0.25 | waist (3), hip (6), knee (2), shoulder (6), elbow (2) = 19 (no ankles) | mean_j \|q_ref − q\| | REW:298-305 |
| joint_vel_tracking_product (212-215) | 0.5 | 2.5 | same 19 | mean_j \|q̇_ref − q̇\| | REW:315-322 |

The `*_product` suffix is only a name: these terms are summed (5.1). Group range per step is [0, 5] before ×dt.

### 5.3 Object-tracking group (Y:216-220)

Contact geometry, computed in CU (CMD:571-584):
* **Contact target** c_e = p_ob + R(q_ob)·(o_e ⊙ s) (CMD:418, 439, 571-573).
  * p_ob, q_ob = object **contact body** link (`object_body_name`, e.g. door_panel).
  * o_e = `contact_target_pos_offset[e]`.
  * s = per-env per-axis object scale from `body_scale` (§7); env 0 has s = 1.
* **End-effector point** e_e = p_eef + R(q_eef)·`contact_eef_pos_offset[e]` (CMD:419, 575-577). The eef link comes from `contact_eef_body_name` (default L/R wrist_yaw_link, Y:109).
* **Contact force** F_e:
  * One ContactSensor per eef link, filtered on the object contact body prim `{env}/{asset}/{object_body_name}` (LOC:89-100).
  * F_e,w = force_matrix_w[eef, filter 0] [IL: contact force on the eef link from that body, world frame, latest physics substep] (CMD:401-415, 579-582).
  * Expressed in the object contact-body frame: F_e = R(q_ob)ᵀ F_e,w (CMD:584).
* **Reference flag** g_e = object_contact at index 0 (CMD:568), broadcast as in §1.1.

All quantities in this group are CU(n−1) snapshots: object/eef at time n−1, flag at frame f_n (REW:390-397).

| term | weight | formula | ref |
|---|---|---|---|
| object_pos_tracking | 1.0, σ=0.5 | exp(−‖p_ref,obj − p_obj,root‖/σ) | REW:326-337, Y:218 |
| object_ori_tracking | 1.0, σ=0.5 | exp(−ang(q_ref,obj* ⊗ q_obj,root)/σ) | REW:339-351, Y:219 |
| eef_contact_exp | 1.0; pos_σ=0.3, frc_σ=40, frc_thres=10.0, gain=5, pos_tol=0 | see below | REW:367-408, Y:220 |
| object_joint_pos_tracking (doors, foldchair) | 1.0, σ=0.5 | exp(−\|q_ref,joint − q_joint\|/σ) | REW:353-365, e.g. T:open_door-feet:69 |
| eef_contact_exp_max (roll_ball only) | 1.0; pos_σ=0.3, pos_tol=0.4, frc_σ=40, frc_thres=10 (**int**) | see below | REW:410-449, T:roll_ball-hand:42 |

**eef_contact_exp** (REW:396-408):
* d_e = max(‖e_e − c_e‖ − pos_tol, 0).
* φ_e = min(‖F_e‖ − 10, 0) when frc_thres is a float; otherwise the mean over axes of min(\|F_e,axis\| − thres_axis, 0).
* r_e = exp(−d_e/0.3)·exp(φ_e/40).
* R = mean_e [ g_e·5·r_e + (1 − g_e) ].
* Range: (0, 5] while the reference flag is on, exactly 1 while it is off. The force factor stays in (0.7788, 1].

**eef_contact_exp_max** (REW:442-449): R = max_e r_e · any_e(g_e), with d_e = max(‖e_e − c_e‖ − 0.4, 0). Because the YAML value `frc_thres: 10` is an **int**, `isinstance(…, float)` is false and the **per-axis** branch is used: φ_e = mean_axis min(\|F_e,axis\| − 10, 0) in the object frame (REW:442-445). It gives 0 when no flag is on.

### 5.4 Loco group (Y:221-226)

| term | weight | formula | ref |
|---|---|---|---|
| action_rate_l2 | 0.1 | −‖a_n − a_{n−1}‖² over raw actions (23). After reset the previous action is 0 | RC:95-105 |
| joint_vel_l2 | 5e-4 | −Σ_{29 joints} min(q̄̇_j², 5); q̄̇ = mean of q̇ at substeps 2 and 3 | RC:122-136 |
| joint_pos_limits | 10.0 | −Σ_{29} [max(l_j − q_j, 0) + max(q_j − u_j, 0)]; [l,u] = mid ± 0.5·0.9·range of the hard USD limits | RC:64-80 |
| joint_torque_limits | 0.01 | −Σ_{29} [max(τ_j/τ̄_j − 1, 0) + max(−τ_j/τ̄_j − 1, 0)], τ̄ = 0.6·effort limit, τ = applied torque estimate [IL] | RC:82-93 |
| survival | 1.0 | +1 | RC:12-14 |
| feet_stumble (truman only, loco group) | 1.0 | −mean_feet 1[‖F_xy,net‖ > 0.5 N] | RF:71-80, T:truman:62 |

### 5.5 Feet group (Y:227-231; feet = `.*ankle_roll_link`, `contact_forces` sensor)

| term | weight | formula | ref |
|---|---|---|---|
| impact_force_l2 | 1.0 | −Σ_f min((F̄_f/W)²·fc_f, 10). F̄ = mean over the 3 stored physics substeps of the net contact force magnitude. W = 9.81·Σ link masses of **env 0** after startup mass randomization (rewards are built after startup hooks, ENV:246-247 vs 259). fc = first contact within the last 0.02 s [IL `compute_first_contact`] | RF:83-103 |
| feet_slip | 0.5 | −Σ_f 1[current contact time > 0.02 s]·clamp(‖v_f,xy‖ − 0, 0, 1) (COM linear velocity) | RF:11-32 |
| feet_air_time | 5.0, thres 0.5 | Σ_f min(t_air,last,f − 0.5, 0)·fc_f, i.e. a penalty (≤0) on touchdowns after short swings; `soft_discount` 1.0 means the discount is unchanged | RF:106-154 |
| survival | 1.0 | +1 | RC:12-14 |

### 5.6 Debug group (Y:232-241)

All terms are `enabled: false`, so this channel is constantly 0 (ENV:617-618, 641-647). It is still a separate reward and value channel.

---------------------------------------------------------------------------------------------------------------------

## 6. Terminations (TRM; Y:259-269)

**Counter rule** (TRM:12-32):
* exceeded = (error ≥ threshold); cnt ← cnt+1 if exceeded, else cnt ← 0. The counter counts **consecutive** steps over threshold.
* terminate when cnt ≥ min_steps; cnt = 0 at reset.
* Counters update in `_update()` (before CU) and are read at stage 6.
* min_steps 25 = 0.5 s.
* Terminations set `terminated` (no bootstrap); clip end and time limit set `truncated` (§0).

| term | error definition (max over the listed bodies) | bodies | thr | min_steps | ref |
|---|---|---|---|---|---|
| cum_body_pos_error | ‖p_ref,b(f_{n−1}) − p_b(n)‖ world | torso_link | 0.5 m | 25 | TRM:35-48, Y:260 |
| cum_body_ori_error | ang(q_ref,b* ⊗ q_b) | torso_link | 1.2 rad | 25 | TRM:65-79, Y:261 |
| cum_body_pos_error_local | ‖R(yaw q_ref,root)ᵀ(p_ref,b − p̄_ref,root) − R(yaw q_root)ᵀ(p_b − p̄_root)‖; robot root read **fresh** | all 16 keypoints (`.*`) | 0.5 m | 25 | TRM:81-108, Y:263 |
| cum_body_ori_error_local | ang((yaw(q_ref,root)*⊗q_ref,b)* ⊗ (yaw(q_root)*⊗q_b)) | all 16 | 1.2 rad | 25 | TRM:110-134, Y:264 |
| cum_object_pos_error | ‖p_ref,obj − p_obj,root(n)‖ | — | 0.5 m | 25 | TRM:153-159, Y:266 |
| cum_object_ori_error | ang(q_obj,root* ⊗ q_ref,obj) | — | 0.8 rad | 25 | TRM:161-167, Y:267 |
| cum_lost_contact_steps | 2·1[∃e: g_e ∧ ¬(‖e_e − c_e‖ < pos_thres ∧ ‖F_e‖ ≥ frc_thres)] (CU(n−1) snapshots); threshold default 1.0, so it fires iff contact is lost | E eefs | pos 0.2, frc 1.0 | 25 | TRM:169-190, Y:269 |
| cum_body_z_error (stool, truman) | \|z_ref,b − z_b\| | torso_link | 0.3 (stool) / 0.2 (truman) | 25 | TRM:50-63 |

Per-task overrides (merged into the base):
* object pos/ori min_steps 5000 (never reached) for open_door-feet, push_door-hand, traverse_door, open_foldchair-sit;
* lost-contact min_steps 200 (foldchair), 500 with pos 0.4 / frc 2.0 (roll_ball), 1000 (truman);
* object_ori threshold 5.0 (roll_ball; never, since ang ≤ π).

---------------------------------------------------------------------------------------------------------------------

## 7. Randomization

No interval randomization: no pushes and no external forces. `step` and `update` are no-ops for all terms used.

| term | what | when | distribution | ref |
|---|---|---|---|---|
| perturb_body_mass `.*: [0.9,1.1]` | every robot link (39) mass × u; inertia × the same u | startup | u ~ U[0.9,1.1] per env per body | RND:342-369, Y:245-246 |
| perturb_body_materials | all collision shapes of `.*ankle_roll_link`, `.*_wrist_yaw_link`: static μ, dynamic μ, restitution | startup | 64 buckets each, drawn once from U[0.3,1.6] / U[0.3,1.2] / U[0,0.2]; each env×shape picks independent bucket indices (static and dynamic independent) | RND:242-293, Y:247-251 |
| perturb_body_com | pelvis, torso_link COM offset (link frame) | startup | U[−0.02,0.02]³ per env per body | RND:371-383, Y:252-254 |
| random_joint_offset `.*: [-0.01,0.01]` | o_j for all 29 joints: added to the PD target (ACT:117) **and** subtracted from observed joint positions (OC:86, 118) | reset | U[−0.01,0.01] per env per joint | RND:731-744, Y:255-256 |
| object_body_randomization | main object only, every body of it: mass (inertia × m_new/m_old), all shapes: dynamic μ, static μ = dynamic·ratio, restitution | startup | mass ~ U(mass_range) per env per body; μ_d, ratio, e from 64 buckets each, one draw per env (same for all shapes) | HRN:12-83 |
| object_joint_randomization (doors, chair) | hinge armature | startup | U(armature_range) | HRN:103-105 |
| object_joint_randomization | hinge Coulomb friction f and viscous damping c (OBJ:33-40) | reset | U(friction_range), U(damping_range) | HRN:107-112 |
| body_scale | per-env **per-axis** scale of the whole object asset prim (xformOp:scale); env 0 = 1 (unscaled); ball: one scalar for all axes; contact target offsets × the same scale (CMD:427-439); reference trajectory **not** scaled | spawn (once) | U(scale_range) | SPN:292-311, LOC:102-110, ENV:220-221 |

Object default (pre-randomization) masses are listed in §9. They only set the inertia scaling ratio, because every task defines `object_body_randomization`.

---------------------------------------------------------------------------------------------------------------------

## 8. Per-task differences (cfg/task/G1/hdmi/*.yaml; base = Y)

### 8.1 Data, objects, robot

Column notes:
* robot: `sph` = rubberhand-feet_sphere, `box` = rubberhand-feet_box, `L` = nohand-feet_sphere-eef_L.
* L = clip length in frames at 50 fps; C and contact segments are [start,end) in frames, from motion.npz.
* `wrist*` = wrist_yaw default ∓0.4 / wrist kp 4.0.

| task (name) | data_path | L | object asset ⟨body⟩, joint | extra objects | robot | eef bodies | C, contact segments | wrist* |
|---|---|---|---|---|---|---|---|---|
| carry_and_place_bread_box (G1CarryAndPlaceBreadBox) | data/motion/data_for_sim/carry_and_place_bread_box-0829 | 840 | bread_box⟨bread_box⟩ | support0, support1 | sph | L/R wrist_yaw | 1: [260,550) | yes/yes |
| carry_box_over_shoulder (G1LiftBoxOverShoulder) | data/motion/data_for_sim/lift_box_over_shoulder | 590 | suitcase⟨suitcase⟩ | — | sph | L/R wrist | 1: [180,460) | yes/yes |
| move_foam (G1TrackFoam) | data/motion/data_for_sim/move_foam-climb-0825-2-0_580 | 680 | foam⟨foam⟩ | stool_support | box | L/R wrist | 1: [240,480) | yes/yes |
| move_stool-climb (G1TrackStool) | data/motion/data_for_sim/move_stool-climb-0814 | 833 | stool⟨stool⟩ | — | box | L/R wrist | 1: [240,530) | yes/no |
| move_suitcase (G1TrackSuitcase) | data/motion/g1/omomo/sub1_suitcase_011 | 472 | suitcase⟨suitcase⟩ | — | sph | L/R wrist | 1: [170,340) | yes/no |
| open_door-feet (G1OpenDoorFeet) | data/motion/data_for_sim/open_door-feet-0828 | 928 | door⟨door_panel⟩, door_joint | — | box | left_ankle_roll_link | 1: [470,540) | no/no |
| open_foldchair-sit (G1OpenFoldchairSit) | data/motion/g1/multi_motion_0726/open_chair_and_sit_amass | 585 | foldchair⟨seat_link⟩, foldchair_joint | — | box | L/R wrist | 1: [120,255) | no/no |
| push_box (G1PushBox) | data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height | 792 | box⟨box⟩ | — | L | L/R wrist | 1: [200,542) (float flags) | no/no |
| push_door-hand (G1PushDoorHand) | data/motion/data_for_sim/push_door-hand-0828 | 573 | door⟨door_panel⟩, door_joint | — | sph | right_wrist_yaw_link | 1: [230,300) | no/no |
| roll_ball-hand (G1RollBall) | data/motion/g1/multi_motion_0726/roll_ball_amass | 602 | ball⟨ball⟩ | — | sph | L/R wrist | 1: all frames | no/no |
| topple_wood_board_and_cross (G1ToppleWoodBoardAndCross) | data/motion/data_for_sim/topple_wood_board_and_cross | 582 | wood_board⟨wood_board⟩ | platform0, platform1, wall0 | sph | L/R wrist | 1: [150,270) | yes/yes |
| traverse_door (G1TraverseDoor) | data/motion/data_for_sim/traverse_door | 1463 | door⟨door_panel⟩, door_joint | — | box | right_wrist_yaw, left_ankle_roll | 2: [270,370) / [870,920) | no/no |
| truman (G1Truman) | data/motion/data_for_sim/combo-2 | 3352 | stair⟨stair⟩ | — | box | L/R wrist | 2: [850,1130) / [800,900)∪[1100,1130) | no/no |

Notes:
* YAML refs: name/robot/data/object lines are T:<task>:6-44 (exact lines per file: e.g. carry_and_place_bread_box 6, 25, 39, 42-44; truman 6, 20, 26, 34-35).
* Clips lacking wrist pitch/yaw joints (zero-filled): open_chair_and_sit_amass, roll_ball_amass, push_box.
* Door clips have 30 joints (+door_joint); foldchair has 26 (25 + foldchair_joint).

### 8.2 Contact offsets (target in object-body frame before scale; eef in eef-link frame)

| task | contact_target_pos_offset | contact_eef_pos_offset | ref |
|---|---|---|---|
| carry_and_place_bread_box | L (0, 0.17, 0.14); R (0, −0.17, 0.14) | (0.05,0,0) ×2 | T:carry_and_place_bread_box:50-55 |
| carry_box_over_shoulder | (−0.1, ±0.18, 0.15) | (0.05,0,0) ×2 | T:carry_box_over_shoulder:34-39 |
| move_foam | (0, ±0.21, −0.1) | (0,0,0) ×2 | T:move_foam:35-40 |
| move_stool-climb | (0, ±0.27, −0.05) | (0.05,0,0) ×2 | T:move_stool-climb:28-33 |
| move_suitcase | (−0.1, ±0.18, 0.25) | (0.05,0,0) ×2 | T:move_suitcase:45-50 |
| open_door-feet | (0, −0.75, 0.1) [left ankle] | (0.05,0,0) | T:open_door-feet:36-40 |
| open_foldchair-sit | L (0.28, −0.1, 0.02); R (0.28, 0.1, 0.02) | base (0,0,0) ×2 (Y:110-112) | T:open_foldchair-sit:30-32 |
| push_box | L (0, −0.2, 0.8); R (0, 0.2, 0.8) | (0.1,0,0) ×2 | T:push_box:24-30 |
| push_door-hand | (0, −0.6, 1.0) [right wrist] | (0.05,0,0) | T:push_door-hand:37-41 |
| roll_ball-hand | (0.1, 0, 0) ×2 | base (0,0,0) ×2 | T:roll_ball-hand:23-25 |
| topple_wood_board_and_cross | L (1, −0.17, 0.04); R (1, −0.17, −0.04) | (0.05,0,0) ×2 | T:topple_wood_board_and_cross:41-46 |
| traverse_door | right wrist (0, −0.6, 1.0); left ankle (0, −0.75, 0.1) | (0.05,0,0) ×2 | T:traverse_door:45-51 |
| truman | L (0, −0.2, 0.65); R (0.3, −0.2, 0.65) | (0.05,0,0) ×2 | T:truman:42-47 |

### 8.3 Overrides of reset ranges, rewards, terminations, randomization

Unlisted = base. `0` means the range is set to [−0, 0].

| task | max ep. | pose_range | object_pose_range | reward changes | termination changes | object_body_randomization (ratio / μ_d / e / mass kg) | body_scale | hinge rand. (armature / friction / damping) |
|---|---|---|---|---|---|---|---|---|
| carry_and_place_bread_box | 1000 | base | x,y 0 | — | — | [1,2]/[0.3,0.8]/[0,0.2]/[1.2,1.8] | bread_box [0.9,1.1] | — |
| carry_box_over_shoulder | 1000 | base | base | — | — | same / [1.2,1.8] | suitcase [0.9,1.1] | — |
| move_foam | 1000 | base | base | — | — | [1,2]/[0.3,0.8]/[0,0.2]/[0.9,1.2] | foam [0.9,1.1] | — |
| move_stool-climb | 1000 | base | base | — (alt. eef_contact commented, :46-48) | + cum_body_z_error torso 0.3 m / 25 (:50-51) | …/[1.5,3.0] | stool [0.9,1.1] | — |
| move_suitcase | **500** (:21) | base | base | — | — | …/[1.2,1.8] | suitcase [0.9,1.1] | — |
| open_door-feet | 1000 | x,y ±0.2, yaw ±0.1 (:22-25) | z,roll,pitch 0 (:31-34) | object_pos/ori_tracking off; + object_joint_pos_tracking w1 σ0.5 (:65-69) | object pos/ori min_steps 5000 (:71-73) | …/[5,20] | door [0.9,1.1] | [0,0.01]/[5,10]/[1,5] (:49-53) |
| open_foldchair-sit | 1000 | base | roll,pitch 0; object joint noise 0/0 (:24-28) | same as door (:57-61) | object 5000; lost_contact min 200 (:63-67) | …/[2,5] | foldchair [0.9,1.1] | [0,0.01]/[5,10]/[1,5] (:41-45) |
| push_box | 1000 | base | base | — | — | **[1.2,2.0]/[0.4,0.6]**/[0,0.2]/[6,12] (:33-37) | box [0.9,1.1] | — |
| push_door-hand | 1000 | x,y ±0.2, yaw ±0.1 (:23-26) | z,roll,pitch 0 | as door (:66-70) | object 5000 (:72-74) | …/[5,20] | door [0.9,1.1] | [0,0.01]/**[0.1,0.5]/[0.1,1.0]** (:50-54) |
| roll_ball-hand | 1000 | base | base | eef_contact_exp off; + eef_contact_exp_max (:39-44) | lost_contact min 500, pos 0.4, frc 2.0; object_ori thr 5.0 (:46-48) | **[1.2,2.0]**/[0.3,0.8]/[0,0.2]/[2,4] | ball [0.9,1.1], homogeneous (:34-37) | — |
| topple_wood_board_and_cross | 1000 | base | roll,pitch,yaw 0 (:36-39) | — | — | …/[1.2,1.8] | wood_board [0.9,1.1] | — |
| traverse_door | 1000 | x,y ±0.2, yaw ±0.1 (:31-34) | z,roll,pitch 0 | as door (:76-80) | object 5000 (:83-85) | …/[5,20] | door [0.9,1.1] | [0,0.01]/[0.1,0.5]/[0.1,1.0] (:60-64) |
| truman | **4000** (:16) | x,y ±0.1, yaw ±0.1 (:29-32) | z,roll,pitch 0 (:37-40) | + loco.feet_stumble w1 (:60-62) | + cum_body_z_error torso 0.2 m / 25; lost_contact min 1000 (:65-67) | [1,2]/**[0.8,1.2]**/[0,0.2]/[20,40] | stair **[0.98,1.02]** | — |

* Door and foldchair tasks add priv terms object_joint_pos/vel/torque (e.g. T:open_door-feet:42-46).
* truman sets `robot.override_params.spawn.enabled_self_collisions: true` (T:truman:21-23). The merge adds a new attribute to the UsdFileCfg (UI:94-106). The self-collision flag lives in `articulation_props` (G1:62-64), so this most likely has **no effect** (UNKNOWN; depends on [IL] spawner).

### 8.4 Reference-data facts relevant to physics (motion.npz)

| task | fact |
|---|---|
| bread_box | box starts on support0 (box z 0.7005, support0 z 0.6605) and is placed on support1 (1.3813,−0.2844,0.6251). Supports are static |
| move_foam | foam on stool_support (foam z 0.715, support z 0.515) |
| move_stool-climb | stool reference z = 0.463 for frames 0–~300. The collision legs reach 0.35 below the origin (§9) and no support object exists, so the stool drops ≈0.11 m after reset. Final z = 0.30 |
| topple_wood_board_and_cross | wood_board reference **position** is constant for all 582 frames, pos (0.4,0.9,1.0); its **orientation** turns 90.6° about y between about frames 119 and 300 (corrected after converting the clip; it starts at quat (0.7106,0,−0.7036,0), long axis vertical). Object pos/ori tracking and terminations keep base settings |
| doors | door frame reference static at z=0; door_joint ref: open_door [−1.4,−0.7] (starts −0.7); push_door and traverse_door [−1.5,0] (start 0) |
| truman, open_foldchair-sit | stair / foldchair reference pose static |
| roll_ball | ball center z = 0.32 (= radius) throughout |

---------------------------------------------------------------------------------------------------------------------

## 9. Objects (OBJ; USD decoded)

**Collision.** Every collision shape below is a convex hull of a unit-cube mesh scaled to the listed size (`physics:approximation = convexHull`), unless it says Cube or Sphere.

**Contact sensors.** Object contact sensors filter on the contact body prim (LOC:89-100).

**Mass pipeline.** USD mass → `mass_props.mass` override in OBJ → randomized at startup to the per-task range (§8.3).

| asset (cfg) | type | bodies / joints | collision geometry (body frame, m) | USD mass → OBJ mass | COM (USD) | ref |
|---|---|---|---|---|---|---|
| box | RigidObject, contact reporting on | /box/box | box 1.0×0.8×0.8 centered (−0.5,0,0.4): x∈[−1,0], y∈[−0.4,0.4], z∈[0,0.8] | 5.0 → 5.0 | (−0.5,0,0.4) | OBJ:66-87; USD:o/box/box.usd |
| suitcase | RigidObject | /suitcase/suitcase | box 0.2×0.3×0.4 at (−0.1,0,0.2): x∈[−0.2,0], z∈[0,0.4] | 2.0 → 0.5 | not authored | OBJ:109-118 |
| foam | RigidObject | /foam/foam | box 0.32×0.32×0.18 at (0,0,−0.09) (origin = top face) | 0.5 → 0.5 | (0,0,−0.09) | OBJ:119-128 |
| stool | RigidObject | /stool/stool | seat 0.35×0.35×0.04 at (0,0,−0.02); 4 legs 0.04×0.04×0.35 at (±0.155,±0.155,−0.175) (origin = seat top, height 0.35) | 0.0 → 0.5 | not authored | OBJ:129-138 |
| bread_box | RigidObject | /bread_box/bread_box | bottom 0.18×0.34×0.02 @0; walls 0.02×0.34×0.26 @(±0.08,0,0.13); ends 0.18×0.02×0.26 @(0,±0.16,0.13); handles 0.06×0.04×0.02 @(0,±0.17,0.21) | 2.0 → 3.0; inertia diag (0.0334,0.02167,0.04167) | (0,0,−0.2) | OBJ:272-281 |
| wood_board | RigidObject | /wood_board/wood_board | box 1.6×0.6×0.01 at (0.8,0,0) | 3.0 → 3.0 | not authored | OBJ:282-291 |
| ball | RigidObject | /ball/ball | sphere mesh, radius 0.32 (0.5·0.64), convex hull | 4.0 → 3.0; inertia 0.6554 | (0,0,0) | OBJ:292-301 |
| stair | RigidObject (dynamic) | /stair/stair | steps 0.98×0.6×0.21 @(0.49,0,0.105); 0.74×0.6×0.20 @(0.37,0,0.31); 0.50×0.6×0.20 @(0.25,0,0.51) | 30 → 20 | not authored | OBJ:159-168 |
| support0 / support1 | RigidObject, **kinematic**, collision on | /supportX/supportX_link | Cube 0.2×0.2×0.04 | 1.0 → 0.5 | — | OBJ:169-202 |
| stool_support | kinematic | | Cube 0.15×0.15×0.04 | 1.0 → 0.5 | — | OBJ:203-219 |
| platform0 / platform1 | kinematic | | Cube 2.0×2.0×0.04 | ~0 → 0.5 | — | OBJ:238-271 |
| wall0 | kinematic | | Cube 0.04×0.4×1.0 | ~0 → 0.5 | — | OBJ:221-237 |
| door | Articulation (`CustomArticulation`), **fixed base**: door_frame fixed to world by `⟨/door/root_joint⟩` (FixedJoint + ArticulationRootAPI) | door_frame (root, 10 kg, COM (0,0,1)); door_panel (20 kg, COM (0,−0.5,1.0) in panel frame, inertia (8.333,6.667,1.667)); revolute `door_joint`, axis z, joint frame at (0,0.5,0) of the frame (localPos0) = panel origin, limits [−150°, 0°] = [−2.618, 0] rad | frame: 2 posts 0.1×1.0×2.0 at (0,±1,1); panel: 0.05×0.9×1.95 at (0,−0.5,1.0) in panel frame, i.e. y∈[−0.45,0.45] in door coordinates at q=0 | no override | | OBJ:42-64; USD:o/door/door.usd |
| foldchair | Articulation (`CustomArticulation`), fixed base (`fix_root_link=True`, OBJ:313, plus USD FixedJoint) | base_link (0.01 kg, root); backrest_panel_link (1.5 kg, fixed joint, link rotated (0.99499,0,−0.09998,0) ≈ −11.5° about y); seat_link (1.0 kg); revolute `foldchair_joint` axis y at base origin, localRot0 = −90° about y, limits [−5.73°, 89.95°] = [−0.1, 1.570] rad | backrest 0.04×0.3×0.6 at (−0.05,0,0.3); seat 0.3×0.3×0.04 at (0.13,0,0) | no override | | OBJ:303-327; USD:o/foldchair/foldchair.usd |

**Hinge model (door, foldchair).**
* USD drives (door kp 1e7, kd 2; chair kd 0.1, joint friction 0.01) are **not** used.
* `IdealPDActuatorCfg` has stiffness 0, damping 0/1, effort limit 100, velocity limit 20 (OBJ:53-63, 316-326).
* `_initialize_impl` writes 0 stiffness, damping and friction to the sim and zeroes the actuator gains (OBJ:19-31).
* Every physics substep the effort target is τ = −sign(q̇)·1[\|q̇\|>0.01]·f − c·q̇ (OBJ:33-40), with f, c from `object_joint_randomization` (§7). [IL] IdealPD clips it to ±100 N·m.
* `object_joint_torque` observes this applied torque.

**Object pose used by observations and rewards.**
* Object state for tracking and observations = the **root link**. For the door and foldchair the root link is fixed, which is why object pos/ori tracking is disabled for them.
* Contact targets use the `object_body_name` link (door_panel, seat_link).
* Reference object pose = motion body named after the asset (CMD:383, 550-551).

**Materials.**
* Only box.usd authors a physics material: μs = μd = 1.0, restitution 0.1, combine multiply (`⟨/box/PhysicsMaterial⟩`). Whether it is bound is UNKNOWN.
* All main-object materials are overwritten at startup (HRN:70-83).
* Kinematic extra objects keep the default material [IL default 0.5/0.5/0].

**MJCF copies.** The files in `assets_mjcf/objects/*.xml` (MuJoCo sim2sim) are **not identical** to the training USDs. Examples: MJCF box is 0.6×0.8×0.8 with 8 kg; MJCF stool seat is 0.3 with 0.3 m legs; MJCF ball is 8 kg. Use the USD values above.

---------------------------------------------------------------------------------------------------------------------

## 10. Scene placement

* Each env has a grid origin (spacing 3.0, or 5.0 in bread_box, open_door, push_door, topple, traverse_door, truman; Y:9, LOC:51-52). All references are shifted by it. Envs do not interact [IL].
* Object prim path is `{ENV}/<asset name>` (LOC:71-86). Extra objects are spawned with their asset names (LOC:72-76).

---------------------------------------------------------------------------------------------------------------------

## 11. Learner-side facts (only what the task interface needs)

| item | value | ref |
|---|---|---|
| algo configs used | `ppo_roa_train` (teacher, phase train, VecNorm updated) → `ppo_roa_finetune` (student PPO, VecNorm frozen) | PPO:89, 91; README.md:98-112; cfg/train_sequential.yaml:31-38 |
| groups consumed | in_keys = (command, policy, object, priv) | PPO:86 |
| teacher actor input | cat[command, policy, encoder(cat[priv, object])]; encoder = MLP[256]+Linear(256) | PPO:190-205, 259-260 |
| teacher action mean | ref_joint_pos_ + MLP[512,256,256] output (residual) | PPO:223-236, 241-242 |
| student actor input | cat[command, policy, adapt(cat[policy, command, object])]; adapt = MLP[256,256]+Linear(256); no priv; no residual | PPO:191-197, 214-218, 264-265 |
| critic input | priv, policy, command, object → MLP[512,256,128] → 5 values (one per reward group) | PPO:192, 198, 268-272 |
| action distribution | diagonal Gaussian, state-independent std (init 1.0), no squashing/clipping | PC:145-164; PPO:66; learning/modules/distributions.py:139-146 |
| loss masks | teacher: step_count > 1; student: > 5 | PPO:670-674 |
| truncation bootstrap | on done steps, next-state value = current value (approximation); terminated → no bootstrap | TRN:155-158; PC:214-223 |

---------------------------------------------------------------------------------------------------------------------

## 12. UNKNOWN / not determinable from the repo

* All `[IL]` items:
  * implicit-PD integration inside PhysX;
  * the `applied_torque` estimate for implicit actuators;
  * contact-sensor force definitions (`force_matrix_w`, `net_forces_w_history`, contact threshold, air/contact time bookkeeping, `compute_first_contact`);
  * the default PhysX material;
  * soft-limit formula;
  * `quat_from_euler_xyz` convention;
  * whether command snapshots of the root pose (views of IsaacLab buffers) are overwritten in place by reset writes.

  These live in IsaacLab v2.2 source, which is not vendored here.
* Simulator joint/body order for the `nohand` robot type (push_box). REC was recorded with a rubberhand robot. The nohand USD has the same kinematic tree minus the rubber hands, so the same 29-joint order is likely but not verified.
* Effect of truman's `spawn.enabled_self_collisions` (§8.3): most likely none.
* How the reference velocities were produced (smoothed? which differencing?). They match finite differences only approximately.
* Robot link inertias and COMs: present in the USD but not tabulated here (masses are).
* Whether `/box/PhysicsMaterial` is bound to the box collider; irrelevant after startup randomization.
* Exact PhysX behavior of per-env non-uniform xformOp:scale on the door articulation (joint anchor scaling).

---------------------------------------------------------------------------------------------------------------------

## 13. Ten facts a reimplementation must get right

1. **Reference timing.**
   * Each control step advances the reference by exactly one 50-fps frame (CMD:290).
   * Rewards and terminations at step n compare the post-physics state (time n) against the reference loaded by the *previous* command update (frame s+n−1). The command update (t+=1, new snapshots) runs after the rewards and before observations (ENV:489-502).
   * Observation terms computed in `update()` use the previous command buffers. `ref_joint_pos_future`, `ref_motion_phase`, contact-flag and object-joint observations use the new ones.
   * Contact flags are indexed one frame ahead of pose references (CMD:565-567).
2. **Episode start and end.**
   * Start frame s ~ uniform over [0, L−33] (CMD:137-140).
   * Robot reset to reference frame s plus noise: pose ±5 cm / ±0.1 rad / yaw ±0.2 (doors ±0.2 m), velocity ±0.2 m/s / ±0.5 rad/s, joints ±0.1 (CMD:154-198).
   * Object reset to its reference pose plus noise with **zero velocity**; extra objects share the object's noise draw (CMD:460-490).
   * Truncation at the clip end (t ≥ L) or the time limit; references past the end are clamped (CMD:233, MOT:337).
   * The first observation of every episode is all zeros (ENV:385-386).
3. **Action.**
   * 23 non-wrist joints, target = q0 + offset + scale·ã, with scale 0.55 / 0.35 / 0.44 per joint (Y:27-40); wrists are held at q0 + offset.
   * ã is a per-substep EMA (α ~ U[0.8,1]) of the action delayed by d ~ {2..6} **physics substeps** (5 ms each); no clipping (ACT:90-119).
   * The teacher outputs a residual on (q_ref − q0)/scale (PPO:223-236).
4. **Actuators and simulation.**
   * Implicit PD with kp = I_a(20π)² and kd = 4·I_a(20π), using the G1 motor armatures, effort limits 88/139/50/25/5 and velocity limits 32/20/37/22 (G1:10-199).
   * Wrist kp 4.0 and wrist-yaw default ∓0.4 in specific tasks.
   * 0.005 s physics, 4 substeps per 0.02 s control step (Y:17-18, ENV:139).
5. **Tracking rewards.**
   * Exponentials of the *mean unsquared* error: exp(−mean‖·‖/σ).
   * Upper/lower-body pos/ori use heading-aligned (yaw-only, z-zeroed root) frames; root pos/ori and body velocities use the world frame.
   * σ: pos 0.5, body ori 1.0, root ori 0.5, lin vel 0.5, ang vel 2.5, joint pos 0.25, joint vel 2.5; every weight 0.5 (Y:171-215, REW:52-322).
6. **Contact reward.**
   * Target point = object contact body pose ⊕ (offset ⊙ per-env object scale); eef point = wrist/ankle link ⊕ offset.
   * Force = filtered eef↔object contact force.
   * Reward = mean over eefs of 5·exp(−d/0.3)·exp(min(‖F‖−10,0)/40) during annotated contact frames, else 1 (REW:367-408, CMD:571-584).
   * roll_ball uses the max-over-hands variant with 0.4 m tolerance and the per-axis force branch (integer `frc_thres`).
7. **Reward bookkeeping.** Five groups (tracking, object_tracking, loco, feet, debug=0), each a **sum** of weight×term multiplied by dt=0.02, critic head per group, advantages summed (ENV:252-289, 412-416, PPO:633-639). Regularizers: action rate 0.1, joint vel 5e-4 (q̇² clamped at 5), joint pos limits 10 (90% soft range), torque limits 0.01 (60% of effort), impact 1.0, slip 0.5, air time 5.0 (penalty below 0.5 s), survival 1.0 in both loco and feet.
8. **Terminations** are *consecutive-step* counters (25 steps = 0.5 s; any good step resets them):
   * torso world pos 0.5 m / ori 1.2 rad;
   * max over all 16 keypoints in heading-aligned frames: 0.5 m / 1.2 rad;
   * object root pos 0.5 m / ori 0.8 rad;
   * lost contact while flagged: pos 0.2 m, force 1 N;
   * per-task disabling for hinged objects (TRM:12-190, Y:259-269).
9. **Observations** (actor-side):
   * command: future keypoints in the reference heading frame, future joint positions and phase, at future steps (1,2,8,16,32);
   * proprioception with Gaussian (±3σ-clipped) noise and histories; joint positions minus the calibration offset;
   * object: object xy/heading and contact targets in the robot heading frame, with step and episodic noise.
   * priv: diffs to the future reference, object pose, contact flags.
   * Every float group except `ref_joint_pos_` is running-normalized (Y:119-167, OBS, OC, HLP:138-146).
10. **Objects and randomization.**
    * USD collision geometry and masses per §9.
    * Hinged doors and chair are fixed-base articulations whose only joint torque is randomized Coulomb + viscous friction (OBJ:10-40, HRN:85-112).
    * Startup randomization: robot link masses ×[0.9,1.1], foot/hand friction, pelvis/torso COM ±2 cm; per-task object mass, friction and restitution; per-axis object scale 0.9–1.1 (contact offsets scaled, reference not).
    * Per reset: joint offsets ±0.01 rad and hinge friction/damping (RND:242-383, 731-744; HRN:12-112; SPN:292-311).
