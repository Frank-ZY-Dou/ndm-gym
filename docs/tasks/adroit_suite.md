# Adroit manipulation tracking

Contributed by [Yifan Kang](https://github.com/YIFANK) ([#11](https://github.com/Frank-ZY-Dou/ndm-gym/pull/11)).

Pen, Hammer and Relocate extend the native Adroit infrastructure used by [Door](adroit_door.md). The algorithms learn from new simulator transitions and optimize tracking through a learned dynamics model. This is an online demonstration-tracking study, **not standard D4RL offline RL**.

The initial single-seed study is retained as historical evidence. The [three-seed follow-up](../../benchmarks/adroit_suite/v2/README.md) uses fresh complete runs for both methods and all three tasks. Door's independent contribution is unchanged. The [study protocol and results](../../benchmarks/adroit_suite/README.md) record reference selection, adaptation, controls, compute budgets and limitations.

## Native task contracts

| Task | Actuators | Scalar positions/velocities | Independently evaluated native frame success |
| --- | ---: | ---: | --- |
| Pen | 24 | 30 / 30 | Object-to-target distance below 0.075 m and pen-axis cosine above 0.95 |
| Hammer | 26 | 33 / 33 | Nail-to-goal distance below 0.01 m |
| Relocate | 30 | 36 / 36 | Object-to-target distance below 0.1 m |

These predicates match Gymnasium-Robotics 1.3.1's `info['success']` on every exported legacy demonstration frame. Final success, ever success and successful-frame fraction are reported separately. They do not by themselves reproduce the original benchmark's complete episode/evaluation protocol.

Tracking errors use native scalar coordinates, separating metres from radians and hand from object. Unlimited angular coordinates use wrapped differences. Object angular errors are **joint-coordinate errors**, not Cartesian orientation errors. Tracking reward contains one exponential term per nonempty group, with the widths fixed in the study protocol, and the action-rate penalty. SuperTrack optimizes the grouped L1 errors through its acceleration model; SuperMimic optimizes imagined reward and a bootstrapped critic through its increment model.

Both use the explicit demonstration-control residual policy. Evaluate its untrained float32-deployed control on identical starts: opening, lifting or hammering that the prior already achieves is not evidence of learned feedback.

## Prepare external references

Obtain the trusted DAPG demonstration archives from `aravindr93/hand_dapg` at commit `9a1de23df6574e1b94e1da742e3326eb1f61c013`. Keep demonstrations and Gymnasium-Robotics assets outside the checkout. The exporter requires MuJoCo 3.1.6 and Gymnasium-Robotics 1.3.1; training and supported-reference preparation use MuJoCo 3.15.0.

| Archive | SHA256 |
| --- | --- |
| `pen-v0_demos.pickle` | `84a6dae9fdac887b76460fb834eafd2bbeb4ef8bd4d6b05afb02f23698dde1d5` |
| `hammer-v0_demos.pickle` | `d3e517e717b1939696af5a599bec6554888ef9438a7214388fbd7c250895fd3f` |
| `relocate-v0_demos.pickle` | `1ad6bbe10e9550865279a44540745eccd22ef623aa0c2e95d2baf5678b3a3284` |

Pickle can execute code; only use an already trusted local archive. A matching hash establishes identity, not trust. The exporter retains all state fields the native environment setter consumes; historical diagnostic-only fields are recorded as extra metadata rather than passed to Gymnasium's stricter state schema.

```bash
# Set LEGACY_PYTHON, DEMOS and OUT to your installed interpreter and external directories.
$LEGACY_PYTHON tools/replay_adroit.py --task pen \
  --trusted-demo "$DEMOS/pen-v0_demos.pickle" \
  --sha256 84a6dae9fdac887b76460fb834eafd2bbeb4ef8bd4d6b05afb02f23698dde1d5 \
  --output "$OUT/pen-legacy"

# In the MuJoCo 3.15.0 training environment; ASSETS is the original installed Adroit asset directory.
python tools/prepare_adroit.py --legacy-dir "$OUT/pen-legacy" \
  --mesh-dir "$ASSETS/resources/meshes" --texture-dir "$ASSETS/resources/textures" \
  --output "$OUT/pen"
python tools/build_door_native.py --output "$OUT/adroit-native.so"
```

Repeat with Hammer and Relocate and their verified digests. Preparation removes retired XML capacity attributes and noncolliding tracker visual geometry, retains native inertia/controller/solver settings, and replays all controls on the supported engine. It never edits a trajectory to improve its score. The first finite clip with native final success and at least five successes in its last ten frames is selected, before training: Pen 1, Hammer 3, Relocate 0. All 25 outcomes per task remain in `preparation.json`. Engine versions can produce different contact trajectories; this study does not assert bit-exact legacy-to-supported dynamics.

## Validate and train

The native loop is optional; it releases Python's GIL across each physical chunk while preserving complete integration state and original solver settings. Match the Python oracle and measure the replacement path on the actual task before a long run. No object forces or per-step state resets are permitted.

```bash
COMMON="--model $OUT/pen/model.xml --reference $OUT/pen/reference.npz --native-library $OUT/adroit-native.so"
python tools/validate_adroit_native.py $COMMON --output "$OUT/pen-native-check.json"
python tools/benchmark_adroit_full.py $COMMON --method supertrack --steps 10 --output "$OUT/pen-timing"
python tools/validate_adroit_recovery.py $COMMON --output "$OUT/pen-recovery"

# Each invocation is a fresh complete run: 20,000 iterations, 2,048 worlds.
CUDA_VISIBLE_DEVICES=0 python -m ndm_gym.train_adroit $COMMON \
  --method supertrack --seed 0 --workers 8 --output "$OUT/pen-supertrack-seed0"
CUDA_VISIBLE_DEVICES=0 python -m ndm_gym.train_adroit $COMMON \
  --method supermimic --seed 0 --workers 8 --output "$OUT/pen-supermimic-seed0"
```

For the follow-up, repeat each task/method with seeds 0, 1 and 2 in separate output directories. Hammer SuperMimic additionally uses `--nail-tracking-std .001`; every other task/method retains the defaults. This changes only the training reward width, not the reference or native success predicate. Evaluation always reports the original widths. The narrower width did not make Hammer success robust across seeds; preserve every result. Door-specific tuning and start-offset options are rejected by the fixed suite protocol.

The formal training source matches commit `1fbef65` in `ndm_gym/`. Subsequent integration with upstream `76b9914` retains the same default execution: both algorithms' full-capacity 320-step states, optimizers, replay and RNGs match bit-for-bit, with only the source fingerprint differing. [Compatibility record](../../benchmarks/adroit_suite/v2/upstream_compatibility.json).

The shared schedule performs 40,960,000 training transitions plus 65,536 warmup transitions, 80,000 model optimizer steps and 19,700 policy updates. SuperMimic additionally performs 394,000 critic optimizer steps. The Hand accurate adaptation uses four model batches of 2,048 eight-step windows, 4,096 imagined policy starts, horizon 16, five goal frames, the existing network sizes and method-specific learning rates. Native joint coordinates/controller, finite-clip termination and CPU physics differ from XHand; this is not a literal equivalence claim.

Use a distinct explicitly selected GPU and disjoint bounded CPU cores for concurrent runs, leaving RAM/VRAM/disk headroom. Run a representative full-capacity warmup first. This path uses native CPU physics and standard Torch GPU learning; it does not depend on upstream Triton kernels whose per-block shared-memory requirements exceed an L40S. Checkpoints include simulator integration states, replay, optimizers and RNG states; `--resume` requires a new output directory and matching source/protocol/inputs.

## Evaluate and render

```bash
python -m ndm_gym.eval_adroit $COMMON --prior --output "$OUT/pen-prior"
python -m ndm_gym.eval_adroit $COMMON \
  --checkpoint "$OUT/pen-supertrack-seed0/latest.pt" --output "$OUT/pen-supertrack-eval"
python tools/render_adroit.py "$OUT/pen/reference.npz" "$OUT/pen/model.xml" "$OUT/pen-policy.webp" \
  --rollouts "$OUT/pen-supertrack-eval/physical-rollouts.npz" --episode 0 \
  --start-frame 0 --end-frame 200 --stride 2 --label "SuperTrack accurate / seed 0 / 20k"
```

Both methods and the control use the same 256 fixed starts: one nominal and 255 hand-only perturbed starts. The object and nail initial states remain unchanged. Evaluate the entire clip, retain final physical states and never silently drop failed episodes. Select the README method using the lower normalized tracking RMSE over all episodes; for the three-seed follow-up select the lowest-error final run across all six method/seed combinations and use fixed nominal episode 0, exclude termination, and retain native appearance and the NDM Gym mark. Adjust the exclusive endpoint to the selected task's action count. Three training seeds estimate variability only for this one reference; they do not establish held-out-motion generalization.

## Observed limitations

[Complete three-seed results](../../benchmarks/adroit_suite/v2/results.md), SuperTrack and SuperMimic:

- native final success by training seed (0, 1, 2): Pen 98.05, 95.70, 96.48% and 99.61, 46.88, 91.80%; Hammer 100, 100,
  100% and 21.09, 100, 98.05%; Relocate 100% for the prior and all six runs;
- normalized tracking RMSE against the prior's: Pen 0.0519 and 0.2380 against 0.4024, Hammer 0.0485 and 0.0762 against
  0.0389, Relocate 0.0508 and 0.2059 against 0.0316.

The original strict native predicates and fixed reporting widths remain unchanged. Every failure is retained. The [historical single-seed results](../../benchmarks/adroit_suite/results_v1.md) are separate evidence and are not pooled into the three-seed averages. These results do not establish robustness to larger passive-object perturbations or unseen demonstrations.
