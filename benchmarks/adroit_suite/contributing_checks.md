# Contribution review checklist

This record covers Pen, Hammer and Relocate only. Door's independent PR and three-seed report are unchanged. The requested scope is one fresh full-budget seed per new task and method; a three-seed study remains separate work.

| CONTRIBUTING requirement | Evidence / scope |
| --- | --- |
| Claim with a draft PR | Deferred: this branch is local review only, as requested. No new PR or Door PR update. |
| `make lint` | Final candidate checked; see validation report. |
| `make test-fast` before commit | Final frozen candidate: 864 passed, 221 skipped, 64 deselected; supported runtime. |
| `make test` for algorithm changes | Final frozen candidate: 871 passed, 278 skipped. CPU-visible suite; skips remain explicit, separately from Adroit GPU prechecks and runs. |
| `make docs` | Strict documentation build on final results and media; see validation report. |
| Efficient existing paths and measured replacement | Shared full Hand accurate adaptation, fused Adam, native collector; [preflight](preflight_v1.json) includes matched serial/native timings and exact physical-state equivalence. Native collection is not faster at every phase; no universal speedup is claimed. Unsupported upstream L40S shared-memory kernels are not treated as tested. |
| Preserve default behavior | Door remains the default of parameterized components; Door regression/equivalence tests run unchanged. No new learning algorithm is registered. |
| Tests for new components | Task/reference validation, serial/native equivalence, dynamic replay dimensions, real policy updates, model freezing, checkpoint identity/recovery, evaluation reward agreement and result-audit rejection tests. |
| Simulator-free imports | Existing packaging checks included in both test suites. Simulator imports remain lazy. |
| Study rules fixed before runs | [Protocol](protocol_v1.json), frozen source/input hashes, full 20,000 iterations per method, final checkpoint only, matched 256 starts and reference-control comparison. |
| README trained-policy preview | Completed: Pen SuperTrack, Hammer SuperMimic, Relocate SuperTrack; lower tracking-error method, fixed nominal episode, nonterminal full clip, native appearance, repository renderer and NDM Gym mark. Hammer nominal failure is explicitly captioned. |
| Preview review before a whole set | Reference camera samples reviewed first; first/middle/last frames of all three selected policy clips inspected. Labels, framing, native appearance and mark verified. Reference clips stay outside README. |
| Repository media / data rules | Only lossless WebP previews and compact reports are versioned; original assets, demonstrations, checkpoint files and raw recordings stay external. Attribution is in NOTICE. |

All six runs, final evaluations, source/input/count audits and visual review are recorded. Publication steps remain deferred. A passing software suite is not a claim that learned feedback improves upon the reference-control prior. One seed cannot establish seed variance, and a single demonstration per task cannot establish held-out-motion generalization.
