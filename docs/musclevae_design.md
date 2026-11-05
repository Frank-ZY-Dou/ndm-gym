# MuscleVAE in NDM Gym: design

MuscleVAE's character and muscle model in `ndm_gym`, with SuperTrack and SuperMimic (§4) and a RigidFormer world model
with one token per muscle.

**Paper:** Y. Feng, X. Xu, L. Liu, *MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters*, SIGGRAPH Asia 2023
Conference Papers ([doi:10.1145/3610548.3618137](https://doi.org/10.1145/3610548.3618137),
[arXiv:2312.07340](https://arxiv.org/abs/2312.07340)). Code: [wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_),
pinned at 54c001c (2024-03-19).

## 1. Sources and licenses

- **The MuscleVAE repository** has no license. Nothing from it is copied into `ndm_gym`.
  - `tools/download_musclevae.sh` fetches it at the pinned commit; a converter turns the character and muscle files into
    local assets, as for LAFAN1 and HDMI. Its pretrained models (Google Drive folders linked from its README) are
    fetched the same way, for validation only.
  - Their checkpoints are pickled files: they are opened with `torch.load(weights_only=True)`, or read field by field
    in an isolated process if that fails, never unpickled freely.
- **The training code is not released:** `train_world_model` and `train_policy` are called but defined nowhere. Their
  algorithm is not re-implemented: SuperTrack and SuperMimic train the character (§4).
- **The muscle layout comes from MASS** (Lee et al. 2019, Apache-2.0: `human.xml`, `muscle284.xml`).
- **The reference motions are LAFAN1** walk, run and jump, retargeted to their skeleton (BVH in their repository).
  LAFAN1 is CC BY-NC-ND, so they are never redistributed.

## 2. Simulation

`ndm_gym/sim/rigid/muscle_sim.py` uses Newton (MuJoCo-Warp) directly. MuJoCo-Warp has spatial tendons, but they cannot
express MuscleVAE's blended anchors, and the pinned Isaac Lab commit rejects articulations with them. Muscles are not
solver objects: torch computes their forces, as MuscleVAE's code does, and applies them as Newton's per-body wrenches.

- **Character:** 23 bodies, 22 joints (14 ball, 8 hinge), 61.4 kg, capsule, box and sphere geoms.
  - Self-collision is off and joint damping is 10, as in theirs.
  - Converted from their JSON to MJCF/USD, with Y-up turned into Z-up.
  - Their ball joints limit three Euler angles separately (an ODE AMotor), which MuJoCo's ball joints (cone limits
    only) cannot. `ball_joints="euler"` turns each ball joint into three limited hinges in their Euler order;
    `"ball"` (the default) leaves the limits unenforced and measures how far poses exceed them.
  - Three hinges lose a degree of freedom at the middle angle ±π/2, where their ball joint does not. The references
    reach 1.56 rad (head) and 1.53 rad (left arm); of the 14,795 reset frames, 88 lie within 0.2 rad of the singularity
    and 6,819 exceed an Euler limit (head 5,519, by up to 1.16 rad; Spine1 1,732; hands 495 and 141), which the hinge
    limits then push back. `reset_exclude` ("gimbal", "limits") drops them from the resets; off by default.
  - Their damped step solves the constraints with the damping in the inertia (I + h D), whereas MuJoCo applies its
    implicit damping after the constraint solve. `joint_damping="ode"` reproduces theirs: armature h D, and the
    damping torque on the joint's relative angular velocity, −D ω_rel, as an explicit force (for an Euler hinge
    −D axis_i·ω_rel; the armature stays the constant h D, exact at a zero middle angle). `"implicit"` is the default.
    The difference is large for the feet and toes, whose inertia is far below h D.
  - Their pretrained policy needs both (`ball_joints="euler"`, `joint_damping="ode"`). Replayed from 768 evenly spaced
    starts for 512 steps (posterior mean), 10–12% of its episodes end in two runs (walk 0%, jump 7–8%, run 23–27%), MPJPE
    6.7 cm in the root's frame; in the default simulator 96% end, at a median of 84 steps (one A100).
  - The GPU simulation is not run-to-run deterministic (MuJoCo-Warp accumulates with atomics); the muscle model is.
    Equivalence checks run on the CPU or compare discrete outcomes.
  - MuJoCo's contact and limit settings follow their ODE setup (friction 0.8, a pyramidal cone, no restitution, hard
    constraints corrected by ERP 0.2 per step); the list is in the simulator's docstring.
- **Rates:** physics 120 Hz, control 20 Hz (6 substeps), as in their `config.yml`. Each substep runs the muscle model,
  then the solver; a control step's six substeps form one CUDA graph.
- **Muscles:** 284 muscles with 1,212 anchors, straight polyline paths, no wrapping.
  - Anchors within 8 cm of a joint are blended between two bodies (weights ∝ 1/√d), and the whole force goes to the
    nearer body, as in their loader.
  - Each substep: anchor positions from the body poses, lengths and velocities, then the Hill model (rigid tendon, no
    pennation) and the muscle-space PD control: target length (1 + a)·l_Tpose, force clipped to the range the 3CC-r
    fatigue model allows. Activation comes from inverting the Hill model.
  - Forces act at each anchor along its path directions, sum into one wrench per body, and enter Newton through
    `State.body_f`.
  - `ndm_gym/robots/muscles.py` vectorizes the computation across environments on the GPU.
- **Parameters:** by default those of the released code, so that their pretrained models run in our port.
  - That includes the muscle velocity computed as Δl/fps, which makes force–velocity ≈ 1 and the damping term about
    14,400× smaller.
  - The paper's version (kd = 0.1·kp, fatigue rates F 0.01 / R 0.002, velocity Δl/Δt) is a switch
    (`--muscle_model paper`).
  - Both are listed side by side in the converter's output.
- **Validation:**
  - Unit tests check every formula on random inputs against independent references from the paper's equations.
  - Their pretrained tracking policy is replayed in our simulator and must track the walk, run and jump references
    (`ndm_gym/policies/musclevae_pretrained.py`, `tools/replay_musclevae.py`).
  - Optionally, one-step forces are compared with their ODE build, run in an isolated environment, since it
    needs Python 3.8 and a C++ build.

## 3. Library changes (the G1 path unchanged, checked by the equivalence and golden tests)

- **Ball joints:** the forward kinematics (`core/kinematics.py` assumes one hinge per body), the state layout,
  observations, and metrics (geodesic joint errors).
- **`robots/musclevae_char.py`:** the character, its muscles and its PD/fatigue parameters.
- **A BVH motion source** for their retargeted clips, 100 fps to 20 Hz.
- **`tasks/muscle_tracking.py`:** the tracking task of §4 (state, reference, reward, termination).

## 4. Algorithms: SuperTrack and SuperMimic

Training starts from LAFAN1's selected configurations (table below); character-specific selections are in
[`benchmarks/musclevae/results.md`](../benchmarks/musclevae/results.md). MuscleVAE's own training (ControlVAE-style: prior,
posterior, mixture of experts) is not re-implemented. Code: `tasks/muscle_tracking.py`, `worldmodels/muscle.py`,
`policies/muscle_policy.py`, `algos/muscle_algos.py` and `muscle_guard.py`, `train_muscles.py`,
`tools/evaluate_muscles.py`.

- **Environment:** MuscleSim with `ball_joints="euler"` and `joint_damping="ode"`, MuscleVAE's resets, reference switches
  (every 96 steps and at a clip's end), termination and three clips.
- **State:** each body's center of mass and rotation, with the backward-difference velocities of the 50 ms control
  step (MuscleVAE's recorded state); the muscle state is their 15 fatigue values. Positions follow from these
  velocities exactly, so semi-implicit Euler is exact.
- **Policy:** LAFAN1's structure and sizes on MuscleVAE's inputs: their observation (371 + 15) and their target (the
  tracked frame's observation), no history. Output: the 284 muscle targets.
- **Reference:** our forward kinematics of the clips, raised as a reset raises them. This is the reset pose and what
  `tools/replay_musclevae.py` scores. Their dataset's placement is not used: the hinges cannot reach it (lower legs,
  feet and hands are 1–11 cm off on average). The termination keeps their head height, as MuscleSim does.
- **World models** (MLP 2048×3, LayerNorm, SiLU; maximal coordinates): inputs in the root's heading frame.
  - SuperTrack: per-body linear and angular accelerations and the fatigue rate, integrated by semi-implicit Euler.
    Fit by the L1 window loss (positions, velocities, rotations, rotation rates) plus an L1 fatigue term.
  - SuperMimic: per-body increments and the fatigue increment. Fit by the normalized squared error.
- **SuperTrack's loss:** the paper's local terms on all 23 bodies (root-frame positions, velocities, rotations,
  rotation rates, heights, up direction), each balanced to 1, with heights and up direction × `st_balance_w`.
  Plus the offset regularizer, the task's action-rate penalty at the reward's ratio to its full score (as on LAFAN1),
  and the failure penalty. Actions are offsets about the reference action l_ref / l_Tpose − 1.
- **SuperMimic's reward:** BeyondMimic's four body terms on all 23 bodies (positions, geodesic rotations, linear and
  angular velocities), widths × `reward_std_scale`, and the G1 task's action-rate penalty. The reference is
  re-anchored to the character's root (heading and horizontal position; velocities turned too), since a switch jumps
  to a random clip and frame. Actions are about the T-pose (a = 0). The critic sees the observation, goal, re-anchored
  body errors and the off count.
- **Terminations in imagination:** MuscleVAE's rule (head more than 0.5 m off, counted, 20 after 26 steps;
  observation above 50), straight-through as in HOI: exact counts in the forward pass, sigmoid gradients.
- **Imagination:** starts from replay states whose clip covers the horizon, with the recorded frame, episode step and
  off count. The reference then advances one frame per step; random switches are not imagined.
- **Options (off by default):** LAFAN1's `--wm_input_noise`, `--wm_smooth_w` and `--pi_rollout_clip` for both
  algorithms; for SuperTrack, LAFAN1's `--st_late_w` and `--st_ref_replay τ`, DiffMimic's demonstration replay: after an
  imagined step's losses, a rollout whose bodies are on average more than τ m from the reference frame re-anchored to its
  root (the reward's frame) continues from that anchored state, with no gradient through it and its off count restarted.
  The muscle state and the failure penalty stay the rollout's.

Settings, LAFAN1 to this character (everything not listed keeps `train.py`'s default):

| Setting | LAFAN1 | Here | Rule |
|---|---|---|---|
| SuperTrack | `--st_fail_w 5 --st_balance_w 4` | same | unit-free loss weights |
| SuperMimic | `--reward_std_scale 0.7` | same | a factor on SuperMimic's widths (m, rad, m/s, rad/s), which carry over |
| `--action_noise` | 0.1 (ST), 0.3 (SM) on u | 0.05, 0.15 on a | × 0.5: MuscleVAE's collection noise (0.05 on a) over the G1 default (0.1 on u) |
| `--st_imag_noise` | 0.1 on u | 0.05 on a | same rule |
| `--imag_start_perturb` | 0.3 m/s on the root | 0.3 m/s on every body | the G1's root kick moves the whole body with the joints unchanged |
| action-rate penalty | −0.1 × Σ₂₉ (Δu)² | −0.1 × (29/284) Σ₂₈₄ (Δa / 0.5)² | the × 0.5 rule; 284 muscles scaled to the G1's 29 actuators |

Mapping details:

- **Action units:** `tasks.muscle_tracking.ACTION_UNIT` implements the × 0.5 rule above.
- **Horizon and discount stay in steps:** H = 32 is 1.6 s here (0.64 s on the G1; MuscleVAE's BPTT is 1.2 s), and
  γ = 0.99 discounts over 5 s (2 s on the G1). Matched in time they would be 13 steps and 0.975. Likewise the replay
  holds 50 s per env (20 s on the G1), and the 8-step model windows are 0.4 s (MuscleVAE's own 8 steps).
- **Action-rate penalty:** the first SuperTrack development run had none and went bang-bang: the
  rate rose from 0.6 to 240 in G1 units within 650 policy updates (smooth G1 policies: about 2), as SuperTrack did on
  the G1 without it. The joint-limit penalty is dropped: the Euler hinges enforce the limits.
- **No world-space root terms:** after a switch the reference's global pose is unrelated to the character's, and
  no drift is scored.
- **`--st_reg` stays 0.01**, now on muscle-target offsets.
- **The muscle state is the 15 part values** (MuscleVAE's), not the 284 × 3 per-muscle states, so the model's state
  is not fully Markov.

## 5. RigidFormer with muscle tokens

- **Tokens:** 284 muscle tokens, 23 body tokens and 2 registers, 309 in all (the G1 has 32).
  - Body tokens carry what muscle features cannot: root height, gravity direction, contacts and velocities. The
    world model's outputs are body dynamics.
- **Muscle token features** (22, from the bodies by the muscle model):
  - length and its rate over the rest length, normalized length;
  - action, PD error over the rest length;
  - the first substep's clipped PD force over f0max and activation, and the Hill factors (FL, FV, PE), with the
    muscle's part fatigue standing in for its own (the model's state has only the 15 part values);
  - its part's fatigue (MA, MR, MF);
  - path midpoint and the first and last segments' directions in the heading frame;
  - plus a learned per-muscle embedding for the static attributes (f0, kp, rest lengths, group).
- **Body token features** (24): position, 6-D rotation, linear and angular velocity, and relative to the parent its
  rotation (6-D) and rotation rate.
- **Rotary positions:** each muscle's arc-length midpoint (near the joint it actuates); each body's position.
- **Attention bias:**
  - between bodies: the kinematic-tree hop distance;
  - between a muscle and a body: the hops from the nearest body the muscle attaches to;
  - between two muscles: the hops between their attachment sets.
- **Conditioning:** FiLM on the 15 part fatigue values in every block.
- **Read-out:** per-body heads for the accelerations or increments of §4's models; the 15 part values' rates or
  increments from each part's muscle tokens, pooled with the part fatigue's f0 weights.
- **Implementation:** `worldmodels.muscle.MuscleRigidFormerNet`, `train_muscles --wm_backbone rigidformer` (`--rf_dim`,
  `--rf_layers`, `--rf_heads`, `--rf_registers`, `--rf_no_tree_bias`, `--rf_no_rope`). Each step, features included,
  is recomputed in the backward pass. Tests check the features against `robots.muscles`, the relation classes, the
  start at zero, the invariance to a turn and shift of the scene, and the gradients.
- **Cost** (A100 40GB, bf16, compiled, development numbers; §4's batch sizes):

  | world model | parameters | fit, 2048 windows x 8 steps | policy BPTT, 4096 x 32 | inference at 4096 envs |
  |---|---|---|---|---|
  | MLP 2048 x 3 | 10.0 M | 32 ms | 102 ms | 0.81 ms per step, 5.1 M env-steps/s |
  | RigidFormer, width 128, 3 blocks | 0.92 M | 1300 ms | 8.9 s (peak 21.5 GB) | 71.8 ms per step, 57 k env-steps/s |
  | RigidFormer, width 64, 2 blocks | 0.20 M | 659 ms | 4.5 s | 34.9 ms per step, 117 k env-steps/s |

  MuscleSim at 4096 environments (training settings): 37.2 ms per control step, 110 k env-steps/s. With 309 tokens a
  step of the RigidFormer costs 41-88 times the MLP's, and in inference it runs at about the simulator's speed.

### Body tokens (`--rf_muscle_tokens body`)

The 23 body tokens and the registers only, so that a step costs about as much as the MLP's.

- **Muscle wrench:** each body token adds the force and the torque about its origin that the muscles exert on it,
  from the first substep's clipped PD forces through the anchors (`MuscleModel.body_wrench`), in the heading frame.
  Only the RigidFormer gets it; the MLP's inputs are unchanged.
- **Muscle encoder:** each muscle's 13 scalar features (above, without the path's geometry) go through a shared linear
  map plus a learned per-muscle embedding and a SiLU, `--rf_muscle_dim` wide (default 32). Each body token adds the
  mean of the encodings of the muscles attached to it, mapped to the token width.
- **Attention bias:** the tree's hops between bodies; any pair with a register is one more class.
- **Fatigue read-out:** one token per body part: the f0-weighted mean of its muscles' encodings, plus the mean of the
  body tokens over the bodies its muscles attach to. It shares the body tokens' read-out.
- **Recomputation:** the policy's BPTT recomputes each step in the backward pass inside the step's compiled graph; the
  window fit keeps its activations (`checkpoint` "frozen").
- **Tests:** the wrench against the muscle model's substep, the pooling matrices, the relation classes, the start at
  zero, the invariance, the gradients and the recomputation; the muscle-token model is unchanged bit for bit.
- **Cost** (A100 40GB, bf16, compiled, development numbers; §4's batch sizes, and training iterations per second of
  SuperTrack at 1024 environments with policy updates):

  | world model | parameters | fit, 2048 x 8 | BPTT, 4096 x 32 | inference at 4096 envs | training it/s |
  |---|---|---|---|---|---|
  | MLP 2048 x 3 | 10.0 M | 31 ms | 93 ms | 0.80 ms | 1.75 |
  | body tokens, width 128, 3 blocks | 0.91 M | 126 ms | 1.16 s | 10.4 ms | 0.39 |
  | body tokens, width 64, 2 blocks | 0.19 M | 94 ms | 0.78 s | 7.1 ms | 0.52 |
  | body tokens, width 128, 3 blocks, 9 registers, both fused options | 0.91 M | 110 ms | 0.78 s | 6.2 ms | 0.44 |
  | body tokens, width 64, 2 blocks, 9 registers, both fused options | 0.19 M | 106 ms | 0.46 s | 4.0 ms | 0.56 |

  A step's forward pass at 4096 environments (width 128): the network 6.0 ms, the muscle features 3.3 ms (their backward
  pass about 5.5 ms more).
- **Fused options:** `--rf_fused_train` (with `--rf_registers 9`: 32 tokens) runs the blocks' bf16 passes as the LAFAN1
  RigidFormer's fused kernels (`kernels/rf_train.py`); `--rf_fused_features` computes the muscles' lengths, length rates and
  wrench with fused kernels (`kernels/muscle_geometry.py`: a muscle's anchors walked in registers, per-body sums without
  atomics; within fp32 rounding of `robots.muscles` in float64, gradients included, `tests/test_muscle_geometry.py`). The
  features then take 1.5 ms forward and 5.0 ms forward and backward at 4096 environments (3.6 and 8.4 ms compiled
  PyTorch). With both, the window fit is about half launch-bound (GPU busy 56 of 106 ms at width 64); the BPTT is
  GPU-bound, a third of it in the muscle geometry.
