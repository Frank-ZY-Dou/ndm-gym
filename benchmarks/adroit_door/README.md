# Adroit Door demonstration tracking (planned)

This draft claims an Adroit Door tracking contribution under CONTRIBUTING.md's
"Claiming a task" guidance. It contains a task plan, not a finished implementation
or a claim that all contribution checks have passed.

## Task

A native Adroit hand follows a door-opening demonstration: approach the handle,
manipulate the latch, and swing the door open. The reference will describe the
hand motion and the door and latch joint coordinates. Unlike the existing XHand
grasp-and-lift examples, this task involves an articulated object with constrained
motion. This is a tracking adaptation of Adroit Door, not a new manipulation task
or an offline-RL D4RL score reproduction.

Sources: [Adroit Door](https://robotics.farama.org/envs/adroit_hand/adroit_door/)
and [DAPG demonstrations](https://github.com/aravindr93/hand_dapg).
Data, models and their versions and applicable notices will be documented before
the reproducible adapter is added; demonstration files and checkpoints will not
be redistributed in this change.

## Implementation scope

- A pinned demonstration loader, initial-state restoration, physical action replay
  and reference conversion. State playback alone is not replay validation.
- A task adapter with observations, reference goals, differentiable tracking
  reward/loss, termination and separate door-opening metrics.
- SuperMimic and SuperTrack integration through the existing world-model and
  policy interfaces, preserving other tasks' defaults.
- Training, evaluation and repository-style rendering commands, component tests
  and a reproducible benchmark protocol.

## Development status

Native CPU physical action replay and a replay video have been obtained in the
development workspace. The selected clip opens the door to approximately
1.536 rad without direct door actuation or added external forces. Soft joint-limit
excess and contact penetration are being documented against the native model;
exact agreement with the historical demonstration is not established.

These are preliminary replay observations, not learned-policy results or results
reproducible from this documentation-only commit. Experimental supervised residual
controllers are not evidence for SuperMimic or SuperTrack and are outside this draft.

## Planned experiments and validation

Before training, freeze the reference/version, development and evaluation starts,
tracking and task-completion metrics, budgets and checkpoint selection rule.
Evaluate hand tracking, door/latch angle errors, actual opening completion and
failure modes under declared initial-state perturbations. Use three independent
training seeds per reported method, final checkpoints and mean/sample standard
deviation; method comparisons use the same task and declared budgets.

Reuse applicable measured efficient paths and report hardware, settings, time per
iteration and total training cost. Videos will identify the reference, controller,
checkpoint and playback speed. Reference replay belongs in documentation; learned
policy previews will follow the repository's README conventions when available.

Pending: runnable adapter and reproduction commands; model-based training pilots;
three-seed results; efficiency measurements; and all applicable implementation,
packaging, equivalence, documentation and merge checks. No performance or
superiority claim is made by this draft.
