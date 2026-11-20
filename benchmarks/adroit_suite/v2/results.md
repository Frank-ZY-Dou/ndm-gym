# Native Adroit tracking: complete three-seed study

Contributed by [Yifan Kang](https://github.com/YIFANK) ([#11](https://github.com/Frank-ZY-Dou/ndm-gym/pull/11)).

Pen, Hammer and Relocate each run SuperTrack and SuperMimic with fresh training seeds 0, 1 and 2: **18 complete runs**, 20,000 iterations and 2,048 worlds per run. Every final checkpoint is evaluated on the same 256 starts for its task, with all 4,608 learned-policy episodes retained. Means ± sample SD below are across three **training-seed means**, not 768 independent training seeds. The float32 reference-control prior is evaluated once on those same starts.

This is online demonstration tracking on native MuJoCo scenes, not the standard D4RL offline-RL benchmark. There is one reference per task (Pen clip 1, Hammer clip 3, Relocate clip 0), with one nominal and 255 hand-only perturbed evaluation starts. Passive-object starts are unchanged.

## Main results

All reporting uses the original tracking widths and native success predicates. Hammer SuperMimic alone trains with the selected 1 mm nail kernel; its narrower training kernel never changes the reporting scale or the strict 10 mm native success threshold.

| Task | Policy | Tracking return ↑ | Normalized tracking RMSE ↓ | Native final success (%) ↑ | Training minutes |
| --- | --- | ---: | ---: | ---: | ---: |
| Pen | Reference controls | 5.1731 | 0.4024 | 58.59 | — |
| Pen | SuperTrack | 5.9769 ± 0.0013 | 0.0519 ± 0.0043 | 96.74 ± 1.19 | 139.5 ± 3.9 |
| Pen | SuperMimic | 5.6410 ± 0.3666 | 0.2380 ± 0.1386 | 79.43 ± 28.46 | 153.3 ± 2.4 |
| Hammer | Reference controls | 24.9052 | 0.0389 | 51.17 | — |
| Hammer | SuperTrack | 24.8926 ± 0.0477 | 0.0485 ± 0.0231 | 100.00 ± 0.00 | 121.4 ± 1.4 |
| Hammer | SuperMimic | 24.8110 ± 0.0597 | 0.0762 ± 0.0155 | 73.05 ± 45.00 | 137.1 ± 6.9 |
| Relocate | Reference controls | 14.0622 | 0.0316 | 100.00 | — |
| Relocate | SuperTrack | 14.0428 ± 0.0040 | 0.0508 ± 0.0027 | 100.00 ± 0.00 | 94.3 ± 0.8 |
| Relocate | SuperMimic | 13.5146 ± 0.1276 | 0.2059 ± 0.0255 | 100.00 ± 0.00 | 109.3 ± 3.0 |

- Native final success by training seed (0, 1, 2), SuperTrack and SuperMimic: Pen 98.05, 95.70, 96.48% and 99.61, 46.88,
  91.80%; Hammer 100, 100, 100% and 21.09, 100, 98.05%; Relocate 100% for the prior and all six runs.
- Normalized tracking RMSE, SuperTrack and SuperMimic against the prior's: Pen 0.0519 and 0.2380 against 0.4024, Hammer
  0.0485 and 0.0762 against 0.0389, Relocate 0.0508 and 0.2059 against 0.0316.

Native success and tracking error are reported together.

## Every complete seed

| Task | Method | Seed | Return | Normalized RMSE | Native final successes / 256 | Training minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pen | SuperTrack | 0 | 5.978125 | 0.049232 | 251 | 136.91 |
| Pen | SuperTrack | 1 | 5.977099 | 0.049506 | 245 | 144.03 |
| Pen | SuperTrack | 2 | 5.975570 | 0.056843 | 247 | 137.56 |
| Pen | SuperMimic | 0 | 5.836302 | 0.167484 | 255 | 150.51 |
| Pen | SuperMimic | 1 | 5.868691 | 0.148871 | 120 | 154.88 |
| Pen | SuperMimic | 2 | 5.218132 | 0.397711 | 235 | 154.48 |
| Hammer | SuperTrack | 0 | 24.875085 | 0.058518 | 256 | 122.95 |
| Hammer | SuperTrack | 1 | 24.946623 | 0.022101 | 256 | 120.25 |
| Hammer | SuperTrack | 2 | 24.856146 | 0.064869 | 256 | 121.00 |
| Hammer | SuperMimic | 0 | 24.826463 | 0.073089 | 54 | 133.25 |
| Hammer | SuperMimic | 1 | 24.861505 | 0.062580 | 256 | 133.04 |
| Hammer | SuperMimic | 2 | 24.745075 | 0.093070 | 251 | 145.03 |
| Relocate | SuperTrack | 0 | 14.047388 | 0.047707 | 256 | 95.14 |
| Relocate | SuperTrack | 1 | 14.041307 | 0.052028 | 256 | 93.67 |
| Relocate | SuperTrack | 2 | 14.039852 | 0.052595 | 256 | 94.12 |
| Relocate | SuperMimic | 0 | 13.368830 | 0.235133 | 256 | 106.89 |
| Relocate | SuperMimic | 1 | 13.606029 | 0.188043 | 256 | 108.26 |
| Relocate | SuperMimic | 2 | 13.568861 | 0.194410 | 256 | 112.65 |

## Physical tracking errors

Translations and nail motion are reported in millimetres; rotations are wrapped native joint-coordinate errors in radians, not Cartesian SO(3) errors. Each entry is the mean ± sample SD of seed means; the prior has no training-seed variance.

### Pen

| Metric | Reference controls | SuperTrack | SuperMimic |
| --- | ---: | ---: | ---: |
| hand rotation joint RMSE (rad) | 0.05768 | 0.01247 ± 0.00042 | 0.05422 ± 0.01006 |
| object translation RMSE (mm) | 5.45219 | 0.76533 ± 0.09997 | 3.49323 ± 1.53528 |
| object rotation joint RMSE (rad) | 0.17926 | 0.02006 ± 0.00373 | 0.09395 ± 0.07264 |

### Hammer

| Metric | Reference controls | SuperTrack | SuperMimic |
| --- | ---: | ---: | ---: |
| hand rotation joint RMSE (rad) | 0.00478 | 0.01997 ± 0.01144 | 0.03326 ± 0.00798 |
| nail RMSE (mm) | 0.42618 | 0.33181 ± 0.03697 | 0.20632 ± 0.04383 |
| object translation RMSE (mm) | 0.58235 | 1.11082 ± 0.57780 | 1.62647 ± 0.32793 |
| object rotation joint RMSE (rad) | 0.01541 | 0.00713 ± 0.00391 | 0.01854 ± 0.00247 |

### Relocate

| Metric | Reference controls | SuperTrack | SuperMimic |
| --- | ---: | ---: | ---: |
| hand translation RMSE (mm) | 0.35773 | 0.37652 ± 0.03414 | 2.78155 ± 0.20084 |
| hand rotation joint RMSE (rad) | 0.00386 | 0.02482 ± 0.00031 | 0.07240 ± 0.00282 |
| object translation RMSE (mm) | 0.43290 | 0.52014 ± 0.05844 | 2.55310 ± 0.43392 |
| object rotation joint RMSE (rad) | 0.01606 | 0.01432 ± 0.00268 | 0.08074 ± 0.01725 |

## Hammer precision experiment and negative outcome

The [registered 6,000-iteration development screen](development_protocol.json) selected a 1 mm nail kernel: 256/256 native successes versus 17/256 for the 10 mm control, with worse overall fixed-metric tracking error. These development runs are not formal seeds. At the complete 20,000-iteration budget, the selected configuration succeeds on **54/256, 256/256 and 251/256** starts. The original ≥95%-on-every-seed quality gate therefore **failed**. It is not a CONTRIBUTING or software-validity requirement. The [scope addendum](publication-scope-addendum.json) completes the other originally planned tasks and retains every negative result; no metric, predicate, reference, seed or budget was relaxed.

The reference finishes only 0.0872 mm inside the strict native threshold. Recomputing physical forward kinematics puts all 202 failed seed-0 episodes between 10.000 and 10.326 mm from the goal; the median over all 256 starts is 10.036 mm. The archived native outcomes match this threshold crossing. This is evidence of threshold sensitivity, not a robust repair. [Diagnosis and development](README.md), [full-budget distances](formal-threshold-diagnostic.json), [failed gate](hammer-gate.json).

## Budget and implementation

Each run collects 40,960,000 training transitions plus 65,536 warmup transitions, performs 80,000 model optimizer steps and 19,700 policy updates. SuperMimic additionally performs 394,000 critic optimizer steps. This matches transition/iteration budgets, not identical optimizer work or wall time. No original v1 seed is pooled into this study.

Each job uses one L40S, eight bounded physical CPU cores and native CPU physics with eager FP32 Torch learning and fused Adam. Six jobs run concurrently at most; resource admission, exact-capacity warmups, resumable checkpoints and pressure monitoring are enabled. Data, environments, outputs, caches and temporary files stay on `/data/YifanKang/` on gpu156. Unrelated jobs are not stopped.

Measured training takes 93.7–154.9 minutes per seed. Formal guards reserve 38.17 GPU-hours; the union of active guard intervals is 6.77 hours. The elapsed span is 8.14 hours, including a 1.37-hour pause after the Hammer gate. Development, checks, diagnostics and rendering are separate costs. [Timing definitions](execution-timing.json), [all 18 resource guards](resource-summary.json).

The frozen training implementation matches `ndm_gym/` in commit `1fbef65`. Integration with upstream `76b9914` preserves the original defaults: both algorithms match the frozen full-capacity 320-step prechecks bit-for-bit in model, optimizer, replay, physical integration and RNG state, with only the source fingerprint differing. [Compatibility](upstream_compatibility.json), [training source manifest](training_source.json), [software checks](software_validation.json). The measured serial/native collector comparison and full-scene equivalence are retained in [the native preflight study](../preflight_v1.md); no universal speedup or unsupported L40S kernel support is claimed.

## Trained-policy previews

Each case uses the final run with the lowest fixed-metric tracking RMSE across all six method/seed combinations, then fixed nominal episode 0. The full nonterminal window is rendered at native speed with original scene appearance and the NDM Gym mark; no reference-only movie is presented as a trained policy. Preview selection is separate from the three-seed averages.

| Case | Selected final policy | Normalized RMSE | Original state frames (exclusive end), stride |
| --- | --- | ---: | --- |
| Pen | SuperTrack, seed 0, 20k | 0.049232 | 0:200, 2 |
| Hammer | SuperTrack, seed 1, 20k | 0.022101 | 0:624, 2 |
| Relocate | SuperTrack, seed 0, 20k | 0.047707 | 0:352, 2 |

The README uses a single three-case WebP grid from the repository compositor. Shorter clips hold their last nonterminal frame until the longest clip loops. Individual lossless previews and their provenance remain available; the grid is encoded with the repository tool at quality 80. Assets, demonstrations, checkpoints and raw recordings stay outside git. [Preview review](preview_qa.json).

## Reproduction and scope

Follow the [task guide](../../../docs/tasks/adroit_suite.md), verified demonstration hashes and [formal protocol](formal_protocol.json). Use separate output directories for seeds 0, 1 and 2, with `--nail-tracking-std .001` only for Hammer SuperMimic. Evaluate final checkpoints on seed 20261117 and 256 starts; retain every failure. The [machine-readable report](results.json) contains all per-seed means, group statistics, complete optimizer/transition counts and source/input/checkpoint evidence hashes.

Three training seeds do not establish held-out-motion generalization. Native final success, ever success and successful-frame fraction are distinct; all are archived. The single-reference, hand-only perturbation protocol is narrower than general manipulation or the original D4RL evaluation. The explicit demonstration-control prior already succeeds at many starts, especially Relocate, so its performance must not be credited to learned feedback.
