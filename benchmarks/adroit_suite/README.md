# Native Adroit manipulation tracking: single-seed study

This extension adds Pen, Hammer and Relocate on the shared native Adroit infrastructure. Door remains in its independent contribution and its existing three-seed report is unchanged. These are demonstration-tracking adaptations with online physical transition collection, not D4RL offline-RL benchmark results.

The [protocol](protocol_v1.json) was fixed before training. Each task runs SuperTrack and SuperMimic from fresh seed 0, with the full 20,000-iteration, 2,048-world schedule inherited from the Hand accurate configuration adaptation. This first study intentionally has one seed per method; it cannot estimate training-seed variance and does not satisfy a later three-seed claim.

## Reference selection and task interpretation

All 25 DAPG demonstrations per task are physically replayed in the pinned legacy environment, then in MuJoCo 3.15.0. Select the smallest clip index that finishes successfully and succeeds on at least five of its last ten frames. This rule uses reference feasibility only, before training, and all candidate outcomes are retained. Selected clips: Pen 1, Hammer 3, Relocate 0.

- Pen uses 24 native hand actuators; Hammer 26; Relocate 30. The manipulated object and Hammer's nail are passive.
- Tracking terms separate hand translations, hand rotations, object translations, object rotations, and the nail where present. Empty groups are absent. Widths are 0.02 m, 0.3 rad, 0.02 m, 0.3 rad and 0.01 m, respectively.
- Object angular tracking uses wrapped native scalar joint coordinates, not Cartesian SO(3) error. All reported units name this distinction.
- Original Gymnasium-Robotics 1.3.1 frame-success predicates are reported separately from tracking reward: Pen's position/orientation condition, Hammer's nail-to-goal distance, Relocate's object-to-target distance. Final, ever and successful-frame fraction are distinct metrics; these are not the full original D4RL evaluation protocol.
- The normalized reference-control residual prior is explicit. A float32-deployed zero-residual control is evaluated on the identical 256 starts. Its success is not attributed to policy learning.

## Shared implementation

The extension parameterizes the existing policy dimensions, replay dimensions, native collector and task tracking heads. SuperTrack and SuperMimic reuse their existing update implementations and the same full-budget runner. Door defaults and checkpoint structure are preserved. Native C++ collection retains complete MuJoCo integration state, original contact solver settings, hand-only controls and bounded worker count.

Cross-task tests check collection against a serial oracle across contact/reset boundaries, replay dimensions, frozen-world-model policy gradients, policy/optimizer recovery, angle periodicity, reward/evaluation agreement, passive-object reset invariance and checkpoint task identity. Real-scene checks additionally compare full physical clips and matched 2,048-world collection timings. Full-capacity fresh-process resume is checked against uninterrupted training for both methods on every task.

## Review scope

All six full-budget runs and final evaluations are complete. See [results and limitations](results_v1.md), [measured preflight performance](preflight_v1.md), [software validation](validation_v1.json), and the [contribution checklist](contributing_checks.md). Selected trained-policy previews are in the root README. Raw assets, datasets, checkpoints and recordings remain external. This branch is for local review; it does not modify the Door PR or open another PR. CONTRIBUTING's draft-claim step is therefore deferred to a subsequent user-authorized submission.
