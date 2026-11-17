# Wuji Hand piano playing (plan)

This draft claims a score-conditioned piano benchmark for NDM Gym. The benchmark
will use two Wuji Hand 2 (beta2) hands on an 88-key keyboard in MuJoCo-Warp.
A single-hand phrase is an integration milestone. Benchmark results are pending.

## Task

[Eden's piano task](https://embodied-ai-nexus.github.io/Eden/tasks/manipulation/#piano)
uses two floating Wuji Hand 2 hands in Genesis, with 20 finger joints and six
wrist control axes per hand. It provides the current score frame and ten future
frames at 20 Hz control and 200 Hz physics. Use those rates and lookahead as the
starting configuration for the MuJoCo-Warp task.

- **Scene and actions.** Use a left and right hand with bounded wrist pose and
  finger position targets over a fixed keyboard. Model each key with passive
  return, damping, travel limits and collision geometry. Read the measured key
  travel after physics and detect press and release crossings with hysteresis.
- **Goals and observations.** Convert the score into timed note intervals,
  preserving releases between repeated notes and simultaneous onsets for chords.
  Observe hand and wrist positions and velocities, key travel and velocity, the
  previous action, and the current and upcoming target keys.
- **Learning objective.** Use continuous key-travel errors for desired and extra
  presses so the policy objective has gradients through the learned dynamics.
  Evaluate with discrete note events. Apply action-rate regularization and keep
  physical contact responsible for triggering notes.
- **Episodes.** Reset hands to score-specific starting poses and keys to rest.
  Evaluate full scores from nominal and perturbed starts. Training may sample
  score segments, initialized with physically reachable hand poses and key
  states. Record early terminations and count the remaining required notes as missed.

## Assets and scores

Audit the exact left and right beta2 hand models and collision meshes, the keyboard
model, and score provenance before reuse. Pin the approved sources and document
license and attribution per asset. Use a download step where redistribution terms
require it. Eden and [RoboPianist](https://proceedings.mlr.press/v229/zakka23a.html)
are task references. A small built-in score can be authored directly as timed
note events, with third-party MIDI files added only after checking their terms.

## Implementation and milestones

1. **Contact checks.** Load the scene, verify finger and key joint ordering,
   compare a scripted contact rollout in MuJoCo and MuJoCo-Warp, and check that
   keys return after release. Check repeated strikes and simultaneous presses.
2. **Single-hand integration.** Run a short scale and the opening phrase of
   *Twinkle Twinkle Little Star* through the simulator, world model, policy
   trainer and checkpoint evaluator. Confirm that the world model captures press
   and release timing near contact, where averaged dynamics can blur transitions.
3. **Two-hand benchmark.** Add an explicit left/right arrangement with melody and
   accompaniment and overlapping notes. Train SuperTrack and SuperMimic through
   NDM Gym's existing interfaces, using the generalized-coordinate MLP where
   suitable and including both hands and moving keys in its state. Use the same
   scene, score, control rate and evaluation starts for both methods.
4. **Reproduction.** Provide asset setup, named training configurations, an
   evaluation command, tests, and a benchmark runner. Add a task page and a
   target-versus-played piano roll. Verify score parsing, note matching and
   repeated-note handling without the simulator, and validate contact and reset
   behavior with simulator smoke tests.

## What will be measured

- **Main comparison.** Three training seeds (0, 1, 2) per method on the same GPU
  with a common wall-clock budget. Report mean and sample SD across seeds, the
  hardware and simulator settings, and training time and throughput. Fix the
  arrangement, budget and reset perturbation ranges before the reported runs.
- **Note accuracy.** Match events one-to-one by pitch within a proposed ±50 ms
  onset window. Count unmatched played notes as extras and unmatched targets as
  misses. Report precision, recall, F1 and median absolute onset and release
  errors for matched notes. Include a breakdown for repeated notes and passages
  requiring both hands.
- **Baseline.** Evaluate a score-driven scripted finger controller in the same
  scene and with the same evaluator. If an Eden PPO run is available, report it
  separately and state its simulator and metric differences.
- **World-model diagnostics.** Measure key-travel prediction error and note
  onset error over held-out contact rollouts at stated horizons.

## Progress

The task scope is defined. The first implementation step is the asset audit and
a scripted press-and-release rollout. The two-hand score arrangement and final
benchmark settings are open until the scene is validated.
