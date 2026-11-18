# Hammer precision repair and three-seed follow-up

Work in progress. The original single-seed study, including its failed SuperMimic Hammer policy, remains in `../results_v1.md`. No failed result is overwritten or relabelled as a repair.

## Diagnosis of the original policy

Native forward kinematics reproduces the reported zero success rate: all 256 final nail-to-goal distances are at least 10.000279 mm, above the original strict 10 mm threshold. The median is 10.166045 mm. The reference itself finishes at 9.912779 mm, a margin of only 0.087221 mm.

The original reward uses a 10 mm nail tracking width. On the nominal final policy state, the 0.274496 mm nail-coordinate error still receives 0.999247 of that term's maximum. Reducing the width to 1 mm would increase its local reward gradient from 0.05486 to 5.09156 per metre, with the same reference target. The original learned model's 16-step open-loop nail RMSE is 0.189034 mm over 128 fixed starts from the nominal physical rollout, larger than the reference's success margin. These measurements identify a weak precision signal and a fragile reference margin; they do not alone prove that reward precision is the only cause. [Diagnostic data](diagnostic_v1_model.json).

## Controlled experiment

The [protocol](development_protocol.json) was fixed before the runs. Three SuperMimic configurations use the same fresh development seed 3065, 6,000 iterations, 2,048 worlds, complete configured updates and the original 20k learning-rate schedule. Only the optional nail tracking width changes: 10 mm, 1 mm or 0.5 mm. The reference, physical model, solver, controls, native success threshold and evaluation metrics remain unchanged. Development evaluation uses its own 256 fixed starts (seed 3065).

All tracking returns and normalized errors use the original v1 reporting widths, even when training uses a narrower reward kernel. Training reward configuration is recorded in the protocol and checkpoint identity. An incompatible configuration cannot load a checkpoint silently. The option is rejected for Pen and Relocate.

A candidate must have at least 95% native final success, finite trajectories on all 256 starts, and fixed-metric normalized tracking RMSE at most 0.1. Eligible candidates are ranked by success, then tracking error. If none passes, the next hypothesis will be registered separately before another run. Short development prefixes are not counted as completed formal seeds.

The implementation is integrated locally with upstream `7e1c17d`; named Hand accurate settings are retained in their native Adroit adaptation. The separate Door contribution is unchanged. The formal three-seed settings are frozen in the protocol below; publication waits for complete results and checks.

## Development result

All three runs completed exactly the registered 6,000-iteration budget. The original 10 mm kernel succeeded on 17/256 starts (6.64%), 1 mm on 256/256 (100%), and 0.5 mm on 110/256 (42.97%). Fixed-metric normalized tracking RMSE was respectively 0.069817, 0.097843 and 0.147558. Only 1 mm passed both registered gates and it is selected. Its overall tracking error increased about 40% relative to the same-budget control; this is a task-success repair, not an overall tracking-precision improvement. [All candidate results](development_results.json).

Recomputed final native distances for 1 mm range from 9.494850 to 9.860450 mm, with median 9.671138 mm, entirely below the original 10 mm threshold. The original configuration's median is 10.185109 mm. [Native distance diagnostic](development_thresholds.json). No reference state, action, contact property or success threshold was changed.

The [formal protocol](formal_protocol.json) now fixes 18 fresh complete runs on the integrated source. Hammer SuperMimic uses 1 mm; SuperTrack retains its original 10 mm setting and its distinct L1 objective. Pen/Relocate retain their original defaults. All six Hammer runs finish first; each of the three SuperMimic seeds must pass the fixed success and error gates before remaining tasks start. Development seed 3065 is excluded from the formal seeds 0, 1 and 2. Full-budget robustness is not established until those results finish.
