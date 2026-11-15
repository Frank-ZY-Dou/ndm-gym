# Adroit Door demonstration tracking

This draft adds native Adroit tracking of a physically replayed door-opening
reference: 28 hand controls, with passive door and latch joints. The task scores
hand, door and latch tracking separately from opening success. No door/latch
actuator, external force or per-step reference-state assignment is used.

## Implementation

- A replay exporter and validated reference archive, native MuJoCo transition
  adapter, articulated tracking reward and policy interfaces.
- Door adaptations of SuperTrack (acceleration model and imagined L1 tracking)
  and SuperMimic (increment model, imagined reward and lambda-return critic).
- An alternating online trainer with whole-episode holdouts, independent seeded
  initialization, optimizer counters, round-boundary recovery and final physical
  evaluation. Both methods use an explicitly declared reference-control residual
  prior; opening is not evidence that feedback was learned.
- Component tests and a native-scene reference renderer with the NDM Gym mark.

## Prepare and run

Follow the [task guide](../../docs/tasks/adroit_door.md) to obtain the pinned
DAPG demonstration and export its reference and native scene. Assets and outputs
stay outside the repository. Use the supported package environment for training;
the legacy demonstration exporter has its own isolated environment.

Generate a new, recorded bootstrap dataset:

```sh
python -m ndm_gym.collect_door --model /path/to/door.xml \
  --reference /path/to/reference.npz --output /path/to/new-bootstrap --seed 0
```

Run either method with explicit inputs and a new output directory:

```sh
python -m ndm_gym.train_door_online --method supertrack \
  --model /path/to/door.xml --reference /path/to/reference.npz \
  --bootstrap /path/to/new-bootstrap/episodes.npz --output /path/to/new-run \
  --seed 0 --rounds 10 --bootstrap-updates 1000 --wm-updates 500 \
  --policy-updates 250 --batch-size 256 --horizon 4
```

Use `--method supermimic` for the other method and distinct output directories
for independent seeds. This public bootstrap recipe is reproducible but does
**not** recreate the historical mixed bootstrap used in the results below.
The online trainer performs its own final physical evaluation; `eval_door` is
for the separate development trainer's checkpoint format, not online checkpoints.
Use `--help` for recovery and measured optional execution paths.

## Existing three-seed evidence

The [corrected-contract study](online_study_2943.md#corrected-model-contract-replication-round2967)
reports every seed and mean ± sample standard deviation. Each method received
6000 world-model and 2500 policy updates, batch 256 and horizon 4, and final-only
scoring on five narrow initial perturbations of one clip.

| Method | Training seeds | Opening | Tracking return, mean ± sample SD |
| --- | --- | --- | --- |
| SuperTrack | 29701, 29702, 29703 | 15/15 | 8.771478 ± 0.236312 |
| SuperMimic | 29601, 29602, 29603 | 15/15 | 8.250310 ± 0.169370 |
| Zero-residual reference controls | No training | 5/5 | 9.035571 |

Higher return is better. Both learned methods worsen tracking against the
reference-control prior. These are bounded negative results, not robust learned
feedback or a completed final benchmark. Historical shared-bootstrap acquisition
cost is unresolved. New optional training flags are not validated by these runs.

## Draft completion checklist

- [x] Native physical replay, task interfaces and genuine method integration.
- [x] Bounded independent three-seed results, with failures and prior disclosed.
- [ ] Final frozen, reproducible matched-budget three-seed study.
- [ ] Complete relevant GPU tests before merging algorithm changes; CPU skips
  do not certify them. Known upstream kernels exceed L40S shared-memory capacity.
- [ ] Efficient end-to-end execution measurement against applicable upstream paths.
- [ ] Trained-policy preview selected and captioned under CONTRIBUTING rules.

The [development notes](development_notes.md) retain measured CPU/CUDA options,
legacy results and method corrections. The [replay results](results.md) describe
source selection and physical feasibility. Neither replay nor supervised baseline
results substitute for the online methods' final study. This PR stays draft.
