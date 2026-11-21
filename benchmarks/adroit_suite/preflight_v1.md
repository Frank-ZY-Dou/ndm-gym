# Preflight performance and correctness

Measurements use 2,048 worlds, eight physical CPU cores and one NVIDIA L40S per learner on a shared host. Each collection comparison uses the same task states and original solver settings. These are short preflight measurements, not complete training runtimes. Shared-host scheduling and contact phase affect the results.

| Task | Physical phase | Serial Python collection (ms) | Native collection (ms) | Python / native |
| --- | ---: | ---: | ---: | ---: |
| Pen | 0 | 212.23 | 52.30 | 4.06× |
| Pen | 100 | 402.37 | 511.50 | 0.79× |
| Pen | 199 | 380.56 | 296.86 | 1.28× |
| Hammer | 0 | 239.48 | 92.09 | 2.60× |
| Hammer | 312 | 267.47 | 198.84 | 1.35× |
| Hammer | 623 | 298.22 | 112.98 | 2.64× |
| Relocate | 0 | 248.81 | 61.73 | 4.03× |
| Relocate | 176 | 251.51 | 172.89 | 1.45× |
| Relocate | 351 | 370.54 | 179.78 | 2.06× |

The native path is optional and preserves exact physical results in the whole-clip/contact/reset checks. It is not uniformly faster at every phase; the Pen middle-phase measurement is slower. Formal training uses the validated native path for all tasks and methods, without choosing a different implementation from the eventual policy results.

| Task | Method | Warm median full iteration (ms) | Peak allocated GPU memory (GiB) |
| --- | --- | ---: | ---: |
| Pen | SuperTrack | 420.14 | 3.13 |
| Pen | SuperMimic | 462.75 | 3.32 |
| Hammer | SuperTrack | 360.27 | 3.19 |
| Hammer | SuperMimic | 410.65 | 3.40 |
| Relocate | SuperTrack | 374.76 | 3.27 |
| Relocate | SuperMimic | 335.96 | 3.50 |

Each timing includes real physical collection and full configured model/policy updates; SuperMimic also includes its critic updates. The first update is excluded from the warm median. Ten-update timing samples are not training convergence evidence. Formal elapsed times and totals are reported separately after all six complete runs.

The full-capacity restart checks compare both methods after iteration 301 through 320 with uninterrupted execution in fresh processes, including models, optimizers, simulator integration states, replay and RNG state. All three tasks passed. Those diagnostic prefixes are not counted among the six formal runs.

Raw measurements and reference eligibility decisions: [preflight_v1.json](preflight_v1.json).
