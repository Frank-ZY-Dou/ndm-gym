# Contribution review checklist

This contribution covers Pen, Hammer and Relocate on the merged Door foundation. All 18 fresh full-budget runs are complete. A failed scientific effect-size gate is retained as a negative result; it is not reported as a passing method-performance test.

| Requirement | Evidence and scope |
| --- | --- |
| Claim with a draft PR | Publication was explicitly deferred until the implementation and experiments were complete. The submitted contribution contains the finished study. |
| `make lint` | Checked on the final candidate; see software validation. |
| `make test-fast` before commit | Final 320-file source: 1,046 passed, 234 skipped, 74 deselected. Source hashes match the checked candidate. |
| `make test` for algorithm/shared learning changes | Integrated source: 1,056 passed, 298 skipped. Only the reporting auditor changed afterward, covered by the final fast suite. CPU-visible scope and unsupported/asset-dependent skips remain explicit; skipped tests are not counted as passed. |
| `make docs` | Strict build with final results and previews; validation records the outcome. |
| Efficient measured paths | Native CPU collector, shared Hand accurate adaptation and fused Adam. [Serial/native timing and physical equivalence](../preflight_v1.md), actual full-budget per-seed times in [results](results.md). No universal speedup or unsupported L40S shared-memory-kernel support is claimed. |
| Preserve defaults and existing behavior | Door defaults and its upstream opt-in settings are preserved. Hammer reward precision is behind an optional flag; the default retains v1 behavior. Door-only tuning/start flags are rejected by the fixed suite protocol. [Exact full-capacity compatibility](upstream_compatibility.json). |
| New components have tests | Native task dimensions, physical collection oracle, gradients, grouped reward/errors, angle wrapping, checkpoint identity and recovery, passive-object reset invariance, fixed reporting metrics and multi-seed audit rejection cases. |
| Simulator-free imports | Existing packaging checks remain in the passed suites; simulator-dependent imports are lazy. |
| Study rules and selection recorded | [Formal protocol](formal_protocol.json), fresh seeds 0–2, final checkpoints only, matched 256 starts, controls and complete update counts. The failed Hammer gate and subsequent [scope addendum](publication-scope-addendum.json) are retained; no negative outcome is dropped. |
| Trained-policy README grid | Lowest fixed-error final run per case across all six method/seed combinations; nominal episode 0, full nonterminal window, native speed and appearance, NDM Gym mark. Individual lossless clips and the repository-composed WebP grid have recorded provenance. |
| Preview inspection | First/middle/last frames of every selected clip and the grid inspected for labels, framing, native appearance and mark; [visual QA](preview_qa.json). Reference-only clips stay out of the README. |
| Data, assets and large outputs | Assets and demonstrations obtained externally with pinned versions and archive hashes; attribution in NOTICE. Checkpoints, raw recordings and operational logs remain outside git. Only reports, provenance and WebP previews are versioned. |

[Software validation](software_validation.json) distinguishes checked source versions and skipped coverage. The scientific limitations include seed-sensitive SuperMimic performance, no overall tracking gain over the prior on Hammer/Relocate, and no held-out-motion evaluation. These remain part of the contribution.
