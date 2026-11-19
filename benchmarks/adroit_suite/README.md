# Native Adroit Pen, Hammer and Relocate tracking

These three tasks extend the merged Adroit Door infrastructure with native task dimensions, demonstration replay, dense tracking terms, physical success predicates, full-budget training, evaluation and rendering. SuperTrack and SuperMimic learn from new simulator transitions; this is online demonstration tracking, not standard D4RL offline RL.

The complete study has **18 fresh runs**: three tasks × two methods × seeds 0, 1 and 2. Every run uses 20,000 iterations and 2,048 worlds, with final-only evaluation on 256 identical starts per task. See [all results and per-seed outcomes](v2/results.md), the [formal protocol](v2/formal_protocol.json), [software validation](v2/software_validation.json), and the [contribution checklist](v2/contributing_checks.md).

## Task and reference contracts

All 25 DAPG demonstrations per task are physically replayed in the pinned legacy environment and again in MuJoCo 3.15.0. Before training, select the smallest clip index that finishes successfully and succeeds on at least five of its last ten frames: Pen 1, Hammer 3, Relocate 0. All candidate outcomes are retained externally. Models, demonstrations, meshes and textures are not redistributed.

Pen has 24 native actuators, Hammer 26 and Relocate 30. Manipulated objects and the nail remain passive. Dense terms separate hand translations/rotations, object translations/rotations and nail motion; angular error is wrapped scalar joint-coordinate error. Native success is evaluated separately with the original Gymnasium-Robotics predicates. The explicit float32 demonstration-control prior is evaluated on the same starts; its behavior is not credited to learned feedback.

Full preparation commands, verified archive hashes, units, training settings and evaluation conventions are in the [task guide](../../docs/tasks/adroit_suite.md). The [native preflight](preflight_v1.md) records serial/native physical equivalence, matched collector timings and recovery checks; the follow-up records [compatibility with the later upstream integration](v2/upstream_compatibility.json).

## Results and limitations

Normalized tracking RMSE and native final success, means over training seeds 0-2 ([every seed](v2/results.md)):

| task | the prior | SuperTrack | SuperMimic |
|---|---|---|---|
| Pen | 0.4024, 58.6% | 0.0519, 96.7% | 0.2380, 79.4% |
| Hammer | 0.0389, 51.2% | 0.0485, 100.0% | 0.0762, 73.0% |
| Relocate | 0.0316, 100.0% | 0.0508, 100.0% | 0.2059, 100.0% |

Every evaluation, successful or not, remains in the report.

The [Hammer precision experiment](v2/README.md) selected a 1 mm reward kernel in a matched 6,000-iteration development screen. Its full-budget results did not pass the initially specified ≥95%-success-on-every-seed quality gate. The [scope addendum](v2/publication-scope-addendum.json) retains that negative result and completes all originally planned tasks without changing budgets, references, metrics or success predicates. Algorithm performance is not conflated with software validity.

The independent Door PR and its historical study are unchanged. The original six-run, single-seed extension is archived in [results_v1.md](results_v1.md); it is not pooled into the new three-seed study, and its failed Hammer policy remains visible as historical evidence. Three training seeds on one reference per task do not establish held-out-motion generalization.
