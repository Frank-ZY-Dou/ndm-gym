# World models

A world model predicts the next state from the state, the action and the simulator's context,
s<sub>t+1</sub> = f(s<sub>t</sub>, a<sub>t</sub>, ξ), and is differentiable: SuperTrack and SuperMimic train the policy
by backpropagation through it. Every world model has the same interface (`ndm_gym.worldmodels`):

| | |
|---|---|
| `step(S, a, xi_n)` | the state one control step (20 ms) later |
| `inputs(S, a, xi_n)` | the network's raw inputs |
| `targets(S, S1)` | the regression targets of a transition (they fit the output normalizer) |
| `in_norm`, `out_norm` | running normalizers of the inputs and the targets (`ndm_gym.core.normalizer.RunningNorm`) |

The G1's state (`ndm_gym.state`) is `S = (p, R, v, w, q, qd)`: the pelvis position (3), rotation matrix (3 × 3), world
linear and angular velocity of the pelvis link frame (3 + 3), joint positions and velocities (29 + 29). The context is
`xi = [friction, joint calibration offsets (29), torso center-of-mass offset (3)]`, which `normalize_xi` scales into the
world models' `xi_n`. An action is the normalized joint setpoint sent to the simulator's PD controllers.

## The G1's world models

| | class | predicts | algorithm |
|---|---|---|---|
| SuperTrack's | `worldmodels.accel.AccelWorldModel` | the root's linear and angular accelerations in its frame and the joint accelerations, integrated by semi-implicit Euler (`substeps`, `--st_substeps`) | `--algo supertrack` |
| SuperMimic's | `worldmodels.increment.IncrementWorldModel` | frame-local increments of every state component | `--algo supermimic` |
| BlinkMimic's | `worldmodels.semi_implicit.SemiImplicitIncrementWorldModel` | velocity increments; the positions from the old and new velocities plus learned corrections | `--algo supermimic --wm_structured` (in `--algo blinkmimic`) |

With zero position corrections, a step of BlinkMimic's model equals one of SuperTrack's with 4 substeps
([BlinkMimic](blinkmimic.md#2-how-the-methods-relate)).

## Backbones

Either network takes the same targets, normalizers, integration and losses.

- **MLP** (default): 3 × 2048 with LayerNorm and SiLU (`--wm_hidden`, `--wm_act`, `--wm_no_ln`), on heading- and
  position-invariant state features; 8.75 M parameters in SuperTrack's model, 8.82 M in SuperMimic's.
- **RigidFormer** (`--wm_backbone rigidformer`, sizes `--rf_dim`, `--rf_layers`, `--rf_heads`, `--rf_registers`): a
  Transformer with one token per G1 body, 3-D rotary attention on the body positions, a kinematic-tree attention bias,
  FiLM conditioning on the context and a read-out in reduced coordinates; 0.88 M parameters at the default size.
  It needs the G1's kinematics (`ndm_gym.robots.g1.kinematics`). See [RigidFormer backbone](rigidformer_backbone.md).

Both output layers start at zero, so a new world model predicts the output normalizer's mean.

## Fitting

The algorithms fit their world model on k-step windows of the replay buffer (`--wm_window`), rolled out from the
window's first state (`ndm_gym.algos.common.Algo.wm_update`), each with its own loss: SuperTrack's L1 errors of the
bodies' world-space positions, velocities, rotations and angular velocities, SuperMimic's normalized frame-local state
error. The offline
benchmark `ndm_gym.wmbench.fit` calls the same update on collected datasets ([Quickstart](quickstart.md#the-offline-world-model-benchmark)).

## Inference copies

`ndm_gym.worldmodels.g1_infer.inference_copy(wm, dtype, attention, fused)` makes a forward-only copy for simulation
throughput: bf16 parameters, the RigidFormer's token attention kernel, and with `fused` the features and the integration
as Triton kernels (CUDA). Each copy is compared with its model before use and refused on a mismatch.

## The other tasks' world models

- **Human-object interaction**: `worldmodels.hoi.HOIWorldModel`, one control step of the robot and the tracked object
  and the contact forces (the end effectors' and the feet's), from the state, the four physics substeps' filtered
  actions and the scene's context; `kind` "accel" or "increment", as above; MLP or `HOIRigidFormerNet`.
- **Muscles**: `worldmodels.muscle.MuscleWorldModel`, the next state of MuscleVAE's 23 bodies (maximal coordinates) and
  of the muscle state from the 284 muscle targets; `kind` "accel" or "increment"; MLP or `MuscleRigidFormerNet`.
