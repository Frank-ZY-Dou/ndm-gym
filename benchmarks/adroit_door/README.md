# Adroit Door demonstration tracking

This task adds native Adroit tracking of a physically replayed door-opening
reference: 28 hand controls, with passive door and latch joints. The task scores
hand, door and latch tracking separately from opening success. No door/latch
actuator, external force or per-step reference-state assignment is used.

## Complete full-budget study

[All six final runs and their controls](full_results_v1.md) are complete: fresh
seeds 0–2 for SuperTrack and SuperMimic, 20,000 iterations each, 256 common starts
per final checkpoint. All 1,536 policy episodes open the door, but both learned
methods worsen overall tracking relative to reference controls. Mean return is
9.02324 ± 0.01019 for SuperTrack and 8.92959 ± 0.05251 for SuperMimic, versus
9.03533 for the untrained deployed control. Mean training time is 87.61 ± 1.00
and 90.44 ± 2.37 minutes per seed, respectively, on an L40S with eight physical
CPU cores per run. These statistics describe one demonstration, not held-out
motions or a claim of learned improvement.

The [machine-readable results](full_results_v1.json),
[validation record](validation_v1.json) and
[contribution checklist](contributing_checks.md) preserve the evidence and its
limits. The selected README policy is SuperTrack seed 0 under the frozen
all-seed error rule. Historical studies below remain development evidence.

## Implementation

- A replay exporter and validated reference archive, native MuJoCo transition
  adapter, articulated tracking reward and policy interfaces.
- Door adaptations of SuperTrack (acceleration model and imagined L1 tracking)
  and SuperMimic (increment model, imagined reward and lambda-return critic).
- A full-budget parallel trainer with independent seeded initialization, device
  replay, optimizer counters, complete physical/RNG recovery and final physical
  evaluation. Both methods use an explicitly declared reference-control residual
  prior; opening is not evidence that feedback was learned. The older
  episode-round trainer remains available for development.
- Component tests and a native-scene policy/reference renderer with the NDM Gym mark.

## Prepare inputs

Follow the [task guide](../../docs/tasks/adroit_door.md) to obtain the pinned
DAPG demonstration and export its reference and native scene. Assets and outputs
stay outside the repository. Use the supported package environment for training;
the legacy demonstration exporter has its own isolated environment.

For the full study, use the guide's `--model-output` export followed by
`tools/prepare_adroit_door.py` in MuJoCo 3.15.0. The
[preparation equivalence record](preparation_equivalence_v1.json) verifies that
this public chain regenerates every position, velocity and action array exactly,
and regenerates the study's model XML byte for byte on the same asset paths.
Fresh runtime metadata changes archive hashes; it does not change the targets.

## Full-budget protocol

The new study is frozen in [full_protocol_v1.json](full_protocol_v1.json).
It starts fresh seeds 0, 1 and 2 for each method, using the applicable capacity,
batch sizes, horizon and training schedule of `hand-supertrack-accurate` and
`hand-supermimic-accurate` at upstream `69b818e`. Each seed runs 20,000 iterations
with 2,048 worlds: 40,960,000 training transitions, 80,000 world-model updates and
19,700 policy updates. SuperMimic additionally performs 394,000 critic updates.
The 32 warmup collection steps are counted separately.

The measured runtime uses Python 3.12.3, PyTorch 2.11.0+cu126 and MuJoCo
3.15.0 on NVIDIA L40S GPUs. Each training worker gets one GPU and eight physical
Xeon Platinum 8468V cores; three seeds run concurrently on a shared host.
Per-run protocols record runtime versions and source/input hashes.

This is a native Door adaptation: articulated joint states and demonstration-control
residuals replace XHand Cartesian states and position targets. Native MuJoCo keeps
the original controller and contact solver, including 20 no-slip iterations.
There is no XHand tracking-error soft termination, nonzero soft-failure penalty,
or CUDA physics collection graph. These differences are not algorithm-equivalence
claims; the protocol lists them explicitly.

Build the optional native batch loop against the same installed MuJoCo used for
training, writing the library outside the checkout:

```sh
python tools/build_door_native.py --output /path/to/door_batch.so
python -m ndm_gym.train_door_full --method supertrack --seed 0 \
  --model /path/to/door.xml --reference /path/to/reference.npz \
  --native-library /path/to/door_batch.so --output /path/to/full-track-seed0
python -m ndm_gym.eval_door_full --checkpoint /path/to/full-track-seed0/latest.pt \
  --model /path/to/door.xml --reference /path/to/reference.npz \
  --native-library /path/to/door_batch.so --output /path/to/full-track-seed0-evaluation
```

Repeat with seeds 1 and 2 and with `--method supermimic`. Do not use
`--development-steps` for study runs. Checkpoints include optimizer state, RNG,
replay and every world's full physical integration state. `--resume` takes a
trusted checkpoint and a new output directory; protocol and source hashes must match.

Evaluation uses the final checkpoint, all 256 fixed starts and the complete clip.
It reports hand/root, passive door and latch errors as well as opening, with no
best-seed or intermediate-checkpoint selection. The zero-residual prior uses the
same starts. The [final report](full_results_v1.md) includes all three seeds, mean and sample
SD, measured training/evaluation costs and the unsuccessful first launch.
Historical numbers below remain development evidence only.

The [failed first-launch cost](failed_launch_cost_v1.json) is retained separately:
20.80 seconds wall time and 0.0172 guarded GPU-hours across three workers, all
failing on a missing import path before training began. These are allocated
elapsed costs, not GPU busy time. The corrected study starts fresh; development,
validation and rendering costs are outside these training totals.

The [precision addendum](baseline_precision_addendum_v1.json), fixed before any
final policy evaluation, retains the original float64 reference-control baseline
and adds a zero-residual baseline with the policy's deployed float32 rounding.
Claims about learned improvement use the latter. Generate it with:

```sh
python tools/eval_door_control.py --model /path/to/door.xml \
  --reference /path/to/reference.npz --native-library /path/to/door_batch.so \
  --output /path/to/study/prior-float32
```

### Measured native collection path

The optional compiled batch loop was compared with Python stepping on the same
Xeon Platinum 8468V host, eight physical cores, 2,048 worlds, MuJoCo 3.15.0 and
20 no-slip iterations. Both paths matched the serial oracle bit for bit over a
perturbed contact trajectory, terminal transitions, resets and previous actions.
The [measurement record](collector_speed_v1.json) includes all 1/2/4/8-worker
results and source/evidence hashes. Eight-worker medians over ten consecutive
steps, starting from three physical phases, were:

| Initial frame | Python stepping | Compiled batch loop |
| --- | ---: | ---: |
| 1 | 0.3168 s | 0.0640 s |
| 114 (contact) | 0.3338 s | 0.1985 s |
| 225 (includes reset) | 0.3414 s | 0.0627 s |

These are CPU collection measurements, taken sequentially on a shared host while
other training/rendering work was active. They do not measure learning, replay,
evaluation or a total training speedup. Full-seed timings are reported separately.
The [presentation plan](presentation_plan_v1.json) fixes the final preview's
configuration criterion and seed before final evaluations are available.

## Historical development three-seed evidence

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

## Study completion checklist

- [x] Native physical replay, task interfaces and genuine method integration.
- [x] Bounded independent three-seed results, with failures and prior disclosed.
- [x] Final frozen, reproducible matched-budget three-seed study.
- [x] Required CPU fast/full test commands and Door GPU component/recovery checks
  pass within the [reported coverage](validation_v1.json). Skipped upstream GPU
  tests are not certified; known kernels exceed L40S shared-memory capacity.
- [x] Matched collector-path measurements and complete end-to-end seed timings.
- [x] Trained-policy preview selected and captioned under CONTRIBUTING rules.

The [development notes](development_notes.md) retain measured CPU/CUDA options,
legacy results and method corrections. The [replay results](results.md) describe
source selection and physical feasibility. Neither replay nor supervised baseline
results substitute for the online methods' final study. This PR stays draft.

## Legacy development runner

The episode-round runner below is retained for development and historical
reproduction. It is not the full-budget study runner and its bootstrap dataset
is not needed by `train_door_full`.

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
**not** recreate the historical mixed bootstrap used in the development results above.
The online trainer performs its own final physical evaluation; `eval_door` is
for the separate development trainer's checkpoint format, not online checkpoints.
Use `--help` for recovery and measured optional execution paths.
