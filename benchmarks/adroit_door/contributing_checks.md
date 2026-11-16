# Contribution evidence checklist

The full-budget study and its selected trained-policy preview are complete.
Checks below state their tested scope; skipped upstream GPU cases are not
certified. Machine-specific operational scripts and their 18 passing tests are
retained separately from the independently checked submission tree. Historical
reduced-budget runs remain development evidence.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Claim the task with a draft PR | Existing native Adroit Door draft; no duplicate claim | Done |
| Development setup and pre-commit | Development tools installed; repository Ruff hook installed | Done locally |
| `make lint` | Ruff passes the current local source | Passed |
| `make docs` | Strict MkDocs build passes | Passed after final report edits |
| `make test-fast` before commit | Latest implementation, reporting and preparation source verified against the checked snapshot: 682 passed, 217 skipped, 49 deselected (626.54 s) | Passed on CPU; skipped cases are not passes |
| Full tests before merging algorithm changes | `make test`: 687 passed, 261 skipped, exit 0; unchanged upstream equivalence and golden tests pass. Door GPU checks: 44 passed, two unsupported fused-on-CPU cases skipped; exact-capacity GPU recovery verified separately | Passed within [documented coverage](validation_v1.json); upstream GPU skips are not certified |
| Keep existing algorithm defaults | Existing defaults retained; five-goal/full-budget behavior is selected by the new runner. Unchanged equivalence and golden tests pass | Passed |
| Tests for new components | Collector, replay, goal horizon, schedule, recovery, evaluation, aggregation and preparation tests are included in the checked source | Passed; see validation coverage |
| Measure the proposed efficient path | Matched Python/native collector timings and bit-exact contact/reset oracle in `collector_speed_v1.json` | Passed; full per-seed end-to-end timings in [results](full_results_v1.md) |
| Freeze study rules before runs | `full_protocol_v1.json`; final-only checkpoints; common 256 starts; all seeds reported | Done |
| Three seeds per method (maintainer/user requirement) | Fresh seeds 0/1/2, 20,000 iterations each, both methods; actual transition/optimizer counts verified | Six of six completed; none relabeled from development |
| Matched evaluation and control | Same 256 multivariate starts; zero-residual reference control; full-clip errors and opening | Both precision controls and all six final policy evaluations complete; identical starts verified |
| Native appearance and NDM Gym mark | Task renderer retains the Adroit scene and common mark; final previews use saved physical rollouts | Selected SuperTrack seed-0 preview visually checked; native scene and mark retained |
| README shows trained policy from lowest-error configuration | `presentation_plan_v1.json` fixes all-seed normalized error selection, seed 0, nominal start and a window excluding termination | SuperTrack selected by all-seed normalized error; seed 0 nominal episode, terminal frame excluded |
| Keep assets, checkpoints and large recordings outside Git | Inputs, runs, checkpoints and MP4s remain external; only compact result summaries and selected WebP previews belong in the contribution | Verified staged files: no assets, checkpoints, raw trajectories, shared libraries or MP4s; selected WebP only |

The study uses native Adroit physics and articulated states; it does not claim
literal equivalence to XHand's Cartesian representation, controller or soft
failure termination. The frozen protocol enumerates these adaptations. Opening
alone is insufficient evidence of learned feedback: the reference-control prior
already opens on all 256 evaluation starts.

L40S per-block shared memory is a separate constraint from host RAM or free VRAM.
Known incompatible upstream kernel configurations must not be retried unchanged
or counted as passed through skipping. Any unsupported full-suite cases remain
explicit in the final report.
