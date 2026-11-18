# Hammer precision study and three-seed follow-up

All 18 formal runs are complete; see [the three-seed report](results.md). The original single-seed study, including its failed SuperMimic Hammer policy, remains in [results_v1.md](../results_v1.md). No failed result is overwritten or relabelled as a repair.

## Diagnosis of the original policy

Native forward kinematics reproduces the reported zero success rate: all 256 final nail-to-goal distances are at least 10.000279 mm, above the original strict 10 mm threshold. The median is 10.166045 mm. The reference itself finishes at 9.912779 mm, a margin of only 0.087221 mm.

The original reward uses a 10 mm nail tracking width. On the nominal final policy state, the 0.274496 mm nail-coordinate error still receives 0.999247 of that term's maximum. Reducing the width to 1 mm would increase its local reward gradient from 0.05486 to 5.09156 per metre, with the same reference target. The original learned model's 16-step open-loop nail RMSE is 0.189034 mm over 128 fixed starts from the nominal physical rollout, larger than the reference's success margin. These measurements identify a weak precision signal and a fragile reference margin; they do not alone prove that reward precision is the only cause. [Diagnostic data](diagnostic_v1_model.json).

## Controlled experiment

The [protocol](development_protocol.json) was fixed before the runs. Three SuperMimic configurations use the same fresh development seed 3065, 6,000 iterations, 2,048 worlds, complete configured updates and the original 20k learning-rate schedule. Only the optional nail tracking width changes: 10 mm, 1 mm or 0.5 mm. The reference, physical model, solver, controls, native success threshold and evaluation metrics remain unchanged. Development evaluation uses its own 256 fixed starts (seed 3065).

All tracking returns and normalized errors use the original v1 reporting widths, even when training uses a narrower reward kernel. Training reward configuration is recorded in the protocol and checkpoint identity. An incompatible configuration cannot load a checkpoint silently. The option is rejected for Pen and Relocate.

A candidate must have at least 95% native final success, finite trajectories on all 256 starts, and fixed-metric normalized tracking RMSE at most 0.1. Eligible candidates are ranked by success, then tracking error. If none passes, the next hypothesis will be registered separately before another run. Short development prefixes are not counted as completed formal seeds.

The formal training source matches `ndm_gym/` in commit `1fbef65`, based on upstream `7e1c17d`. Subsequent integration with upstream `76b9914` preserves the default execution; [full-capacity compatibility checks](upstream_compatibility.json) match exactly. Named Hand accurate settings are retained in their native Adroit adaptation; the separate Door contribution is unchanged.

## Development result

All three runs completed exactly the registered 6,000-iteration budget. The original 10 mm kernel succeeded on 17/256 starts (6.64%), 1 mm on 256/256 (100%), and 0.5 mm on 110/256 (42.97%). Fixed-metric normalized tracking RMSE was respectively 0.069817, 0.097843 and 0.147558. Only 1 mm passed both registered gates and it is selected. Its overall tracking error increased about 40% relative to the same-budget control; this is a development-seed task-success gain, not an overall tracking-precision improvement. The formal results below do not establish a robust repair. [All candidate results](development_results.json).

Recomputed final native distances for 1 mm range from 9.494850 to 9.860450 mm, with median 9.671138 mm, entirely below the original 10 mm threshold. The original configuration's median is 10.185109 mm. [Native distance diagnostic](development_thresholds.json). No reference trajectory or reference control, contact property or success threshold was changed.

The [formal protocol](formal_protocol.json) now fixes 18 fresh complete runs on the integrated source. Hammer SuperMimic uses 1 mm; SuperTrack retains its original 10 mm setting and its distinct L1 objective. Pen/Relocate retain their original defaults. The initial schedule ran all six Hammer runs first and required every SuperMimic seed to pass the fixed success and error gates before continuing. Development seed 3065 is excluded from the formal seeds 0, 1 and 2. Those full-budget results now show that this gate failed; see below.

## Full-budget Hammer outcome and publication scope

All six Hammer runs completed the original 20,000 iterations. SuperTrack succeeds on 256/256 starts in every seed. SuperMimic succeeds on 54/256, 256/256 and 251/256 starts (21.09%, 100%, 98.05%); fixed normalized tracking RMSE is respectively 0.073089, 0.062580 and 0.093070. Thus the 1 mm change does **not** establish a robust task-success repair. The original effect-size gate failed and is retained in the evidence.

Native forward kinematics places all 202 failed seed-0 episodes between 10.000 and 10.326 mm from the nail goal (median over all episodes 10.036 mm); seed 2 has five failures, with maximum distance 10.030 mm. The original strict 10 mm predicate is unchanged. This is physical threshold sensitivity, not a mismatch between archived native outcomes and the evaluator.

The [scope addendum](publication-scope-addendum.json) retains the failed gate and completes all twelve originally planned Pen/Relocate runs. CONTRIBUTING does not require 95% task success. No training condition, budget, seed, reference, metric or success threshold changes; the failed Hammer result is not omitted or relabelled as a successful repair. All 18 runs, source/input/count audits and physical evaluation records are included in the [complete report](results.md).
