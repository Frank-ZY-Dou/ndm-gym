# Adroit Door demonstration tracking

Contributed by [Yifan Kang](https://github.com/YIFANK) ([#3](https://github.com/Frank-ZY-Dou/ndm-gym/pull/3)).

A native Adroit hand tracks a physically replayed door-opening demonstration.
The 28 hand actuators must move the hand, turn the passive latch and open the
passive door. The task measures hand, door and latch tracking separately from
opening. There are no door/latch actuators, external assistance forces or
per-step assignments to the reference state during physical evaluation.

The full-budget trainer implements native Door adaptations of SuperTrack and
SuperMimic. Both use an **explicit demonstration-control residual prior**: the
policy observes the simulated state and adds a bounded correction to the current
reference hand action. A zero-residual policy already opens this reference, so
opening alone does not demonstrate learned feedback or improvement. The
[benchmark](../../benchmarks/adroit_door/README.md) records the frozen protocol,
measured execution paths, controls and study status.

## Prepare the physical reference

Use an isolated legacy replay environment with `numpy`, `mujoco==3.1.6` and
`gymnasium-robotics==1.3.1`. Do not override the main package's dependencies to
run this exporter. Obtain the demonstration from
[DAPG at the pinned commit](https://github.com/aravindr93/hand_dapg/tree/9a1de23df6574e1b94e1da742e3326eb1f61c013).
Only deserialize a trusted pickle: a digest establishes identity, not safety.
Demonstrations, model assets and checkpoints remain outside the repository.

From the repository root, using the legacy environment's Python:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/replay_adroit_door.py \
  --trusted-demo /path/to/hand_dapg/dapg/demonstrations/door-v0_demos.pickle \
  --sha256 d6ec9a090261f901e5c9bf10082799e30c8d34c3f28aef4026e492a1aefc96c3 \
  --clip 23 --output /path/outside/repository/door23.npz \
  --model-output /path/outside/repository/door23-legacy.xml
```

The archive contains T actions and T+1 states: action i generates state i+1.
Actions use the original controller's clipped [-1, 1] mapping. The exporter
advances physics at every substep, rejects nonfinite states and external applied
forces, and records native joint order, scene identity and contact diagnostics.
The XML includes the initialized scene and original actuator parameters.

Next, use the supported **MuJoCo 3.15.0** training environment to regenerate the
physical reference. Meshes and textures remain in the original Gymnasium
Robotics asset directories:

```sh
python tools/prepare_adroit_door.py \
  --legacy-model /path/outside/repository/door23-legacy.xml \
  --legacy-reference /path/outside/repository/door23.npz \
  --output /path/outside/repository/door23-supported
```

Use the resulting `door.xml` and `reference.npz` for all training and evaluation.
When moving between hosts, pass `--mesh-dir` and `--texture-dir` pointing to the
same original assets on the new host. Preparation removes retired XML capacity
fields and a noncolliding mocap tracker visual while retaining its explicit
inertia. It preserves the controller, contact settings and 20 no-slip iterations,
then physically replays the controls. It does not adjust trajectory states to
improve matching.

The supported engine's arrays define the study targets. The 3.1.6 and 3.15.0
engines are not numerically equivalent: their door-angle RMSE on this replay is
0.002535 rad. The [preparation check](../../benchmarks/adroit_door/preparation_equivalence_v1.json)
verifies that the public preparation chain regenerates the frozen study's
positions, velocities and actions bit for bit, and its XML byte for byte on the
same asset paths. Archive hashes can differ because metadata includes paths and
measured timing. `preparation.json` records fresh diagnostics and identities;
historical supported-reference metadata inherited legacy diagnostics and is not
used as current-engine performance evidence.

## Train SuperTrack and SuperMimic

Install the repository as described in its README. The measured full-study
runtime uses Python 3.12.3, PyTorch 2.11.0+cu126 and MuJoCo 3.15.0. Build the
optional native collection loop with a C++ compiler against that same MuJoCo
installation, keeping its output outside the checkout:

```sh
python tools/build_door_native.py --output /path/outside/repository/door_batch.so
python -m ndm_gym.train_door_full --method supertrack --seed 0 \
  --model /path/outside/repository/door23-supported/door.xml \
  --reference /path/outside/repository/door23-supported/reference.npz \
  --native-library /path/outside/repository/door_batch.so \
  --output /path/outside/repository/supertrack-seed0
```

Repeat for seeds 1 and 2 and for `--method supermimic`, using independent output
directories. Every formal seed starts fresh and runs 20,000 iterations in 2,048
worlds. The [frozen protocol](../../benchmarks/adroit_door/full_protocol_v1.json)
uses four model updates per iteration, a 300-iteration model-only phase, policy
batch 4,096, horizon 16 and five future goal frames. SuperMimic also uses its
lambda-return critic. Streaming normalization, physical-phase initialization,
GPU replay sampling, fused Adam and cosine learning-rate decay are part of this
protocol. `--development-steps` is for diagnostics, not formal study runs.

The named Hand accurate configurations supply the applicable model capacity,
optimizer and training budget. Door uses articulated native joint states and
control units rather than XHand Cartesian states and position targets. It has
finite-clip termination, no XHand soft-failure penalty and no CUDA physics
collection graph. These adaptations are explicitly enumerated in the protocol;
they are not claims of literal task or controller equivalence.

Collection uses native CPU physics with bounded persistent workers and a compiled
batch loop. The same contact solver and control mapping are retained. The
[collector measurements](../../benchmarks/adroit_door/collector_speed_v1.json)
compare this path with Python stepping on identical physical states and settings,
including a bit-exact serial oracle, contacts and resets. End-to-end seed time
includes learning and must be reported separately from collection speed.

Checkpoints include model and optimizer states, normalizers, every world's full
physical state, replay and CPU/CUDA random states. `--resume` takes a trusted
checkpoint and a new output directory; source and protocol identities must match.
Fresh-process recovery has been checked against uninterrupted updates for both
methods. On shared machines, select GPUs and CPU affinity explicitly and leave
memory, disk and compute headroom for other users.

## Evaluate the final policy and control

```sh
python -m ndm_gym.eval_door_full \
  --checkpoint /path/outside/repository/supertrack-seed0/latest.pt \
  --model /path/outside/repository/door23-supported/door.xml \
  --reference /path/outside/repository/door23-supported/reference.npz \
  --native-library /path/outside/repository/door_batch.so \
  --output /path/outside/repository/supertrack-seed0-evaluation
```

Formal evaluation requires the iteration-20,000 checkpoint and checks the actual
optimizer counts. It uses one nominal and 255 fixed, multivariate perturbed hand
starts, with the same starts for every method and seed. It retains every physical
trajectory and reports full-clip root translation error in metres, hand/door/latch
errors in radians, tracking return, finiteness and opening. Report all three seed
means and their sample standard deviation; do not choose the best seed or an
intermediate checkpoint. This is one demonstration, not held-out-motion evidence.

Run `tools/eval_door_control.py` with the same `--model`, `--reference` and
`--native-library`, plus a new `--output` directory, for the untrained
zero-residual control rounded to float32 exactly as deployed by the policy.
The [precision addendum](../../benchmarks/adroit_door/baseline_precision_addendum_v1.json)
retains the original float64 control too. Learned-improvement comparisons use the
deployed float32 control, not opening alone.

## The door and latch started open

`--door-latch-offset D` starts every training episode after the first with the door and the latch raised by U(0, D) rad
each, and `eval_door_full` and `tools/eval_door_control.py` take the same option for their 255 perturbed starts (the
nominal one stays closed). With D = 0.3 the demonstration's controls alone open 167 of 256 starts. The training settings
are options too (`--lr`, `--residual-scale`, `--action-noise`, `--wm-only-iters`; SuperTrack's `--correction-reg` and
`--correction-imag-noise`; SuperMimic's `--gamma`), recorded in the run and rebuilt by the evaluation:

```sh
python -m ndm_gym.train_door_full --method supertrack --residual-scale 0.2 --door-latch-offset 0.3 --seed 0 \
  --snapshot-every 500 --model ... --reference ... --native-library ... --output /path/outside/repository/st-offset-seed0
python -m ndm_gym.eval_door_full --checkpoint /path/outside/repository/st-offset-seed0/latest.pt --door-latch-offset 0.3 \
  --model ... --reference ... --native-library ... --output /path/outside/repository/st-offset-seed0-evaluation
```

Results on A100 40GB, both settings, every seed and the settings screen: [results](../../benchmarks/adroit_door/results_a100.md).

## Render saved physical trajectories

```sh
python tools/render_adroit_door.py \
  /path/outside/repository/door23-supported/reference.npz \
  /path/outside/repository/door23-supported/door.xml \
  /path/outside/repository/policy-preview.webp \
  --rollouts /path/outside/repository/supertrack-seed0-evaluation/physical-rollouts.npz \
  --episode 0 --start-frame 0 --end-frame 226 --stride 2 \
  --label "SuperTrack accurate / seed 0 / 20k"
```

Use an appropriate MuJoCo rendering backend. The renderer keeps the native
Adroit scene and adds the NDM Gym mark, configuration label and provenance
sidecar. The exclusive endpoint above omits the terminal state. Rendering assigns
saved states for display; it is not another physics evaluation. Without
`--rollouts`, the output is explicitly labelled as a reference, not a policy.

The [presentation rule](../../benchmarks/adroit_door/presentation_plan_v1.json)
selects the README method by normalized tracking error over all three seeds and
all 256 starts, then shows its seed 0 nominal episode. It does not select a
visually favorable seed. Keep raw trajectories, checkpoints and large recordings
external; publish only the selected compact WebP preview. Credit DAPG and
Gymnasium-Robotics alongside the preview; the repository NOTICE records the
source versions and notices.

## Interfaces and development history

`DoorReference.load` reads the non-pickle archive without importing a simulator.
`DoorTask` provides observation, goal, reward, termination and opening heads on
`(qpos, qvel)`. Its four Gaussian tracking terms separately score root translation,
hand angles, door and latch, with a normalized action-rate penalty. `setpoint`
maps normalized hand controls to physical native actuator units. `DoorPhysics`
is the single-world correctness adapter; it preserves the terminal state until
explicit reset and rejects applied assistance forces.

`DOOR_ALGOS` registers the native Door algorithms. The standalone `DoorPolicy`,
residual-update diagnostic and older episode-round trainers remain development
interfaces; they are not the full-study runner. Historical supervised pilots,
reduced-budget runs and commands are retained in the
[development notes](../../benchmarks/adroit_door/development_notes.md),
[replay results](../../benchmarks/adroit_door/results.md) and
[legacy runner guide](../../benchmarks/adroit_door/README.md#legacy-development-runner).
Their budgets and results must not be relabelled as the formal study.
