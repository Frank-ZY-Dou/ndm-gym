# Reference simulator diagnostics

Protocol: [protocol.md](protocol.md), fixed before these runs. macOS arm64,
Python 3.12.13, MuJoCo 4.0.0, PyTorch 2.15.0. 20 episodes per seed and control.
Raw logs are excluded from git. These are untrained baselines, not policy results.

| Seed | Control | Success | Drop | Mean final angle (rad) |
|---|---|---:|---:|---:|
| 0 | midpoint | 0% | 0% | 2.2265 |
| 1 | midpoint | 0% | 0% | 2.2701 |
| 2 | midpoint | 0% | 0% | 2.0653 |
| 0 | random | 0% | 95% | 2.2146 |
| 1 | random | 0% | 80% | 2.3365 |
| 2 | random | 0% | 80% | 2.4583 |

The midpoint controller retains the cube but does not achieve any sampled goal.
Random independent actions usually lose the object. No learned controller,
world-model fit, GPU throughput or comparisons with the humanoid algorithms
were evaluated. This reference backend alone does not establish learnability.

Reproduce from the repository root:

```bash
bash tools/download_allegro.sh
python tools/evaluate_reorientation_baselines.py --out logs/reorientation/baselines.json
```
