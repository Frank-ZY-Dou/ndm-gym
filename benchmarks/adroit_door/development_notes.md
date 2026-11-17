# Development history and measured execution paths

These chronological records describe the source and protocols at each stage.
For the current contribution status and entry points, see [README](README.md).

Round2966 source correction: Door SuperTrack now fits the acceleration model
with squared normalized multistep error summed over steps and averaged over
windows, matching `HandSuperTrack`'s inherited `_wm_rollout`. Generic MJCF L1
behavior remains unchanged. Its checkpoint contract is `accel-squared-v2`;
legacy L1 checkpoints require the preserved historical implementation. All
previous SuperTrack training results below used the legacy L1 model fit and
are not evidence for this corrected source. SuperMimic's model loss is unchanged.
Corrected-contract seeds are reported in the [online study](online_study_2943.md#corrected-model-contract-replication-round2967).
They are bounded negative replication; effective learned tracking remains unproven.

Native Door model-method integration (development): `DoorSuperTrack` uses the
upstream MJCF acceleration model, calibrated L1 tracking over imagined windows,
and a frozen world model retaining action derivatives. Its four tracked terms
are ARTz translation, remaining hand joints, door angle and latch angle.
`DoorSuperMimic` now uses the upstream increment parameterization and squared
normalized multistep model error, with the same policy and imagined
discounted task reward, terminal bootstrap and a lambda-return critic. The finite
reference end cuts bootstrap; there is no tracking-error termination. Each policy
update is followed by one full-batch critic update (gamma .99, lambda .95).
Both policies retain the declared reference-control residual prior.

The embodiment is the original native Adroit model: 30 coordinates (28 hand,
door hinge and latch), 28 hand actuators, original control ranges and timestep.
There is no door/latch actuation or XHand retargeting. This is an Adroit adaptation,
not a controlled algorithm-only comparison to upstream XHand.

The bounded integration pilot used 1000 world-model and 100 policy updates per
method, batch32/H4, seed2927 and only final checkpoints. SuperTrack opened on all
five narrow same-clip starts; SuperMimic opened on none, with finite trajectories.
These are development results, not robust task success or formal replication.
Supervised seeds29301–29303 remain auxiliary evidence only. The bounded model-method replication below is complete; efficient GPU paths,
learned previews and full merge checks remain pending. CPU checks do not certify skipped CUDA tests.


Historical acceleration-model bounded replication (seeds29401/29402/29403): each method used 5000
world-model and 1000 policy updates, batch32/H4, CPU float32, one thread, no
supervised updates. Offline windows come from the same physically replayed clip;
no new online data were collected. Each final checkpoint ran the same five ARRz
starts (0, ±.0075, ±.015 rad), 226 controls per episode. All seeds are retained.

| Method | Opening per seed | Door RMSE rad, mean ± sampleSD | Hand RMSE rad, mean ± sampleSD |
| --- | --- | --- | --- |
| SuperTrack | 5/5, 5/5, 5/5 | .022469 ± .010652 | .046461 ± .008854 |
| SuperMimic | 0/5, 0/5, 0/5 | .759457 ± .004697 | .188290 ± .008691 |

The statistics use each seed's average across the five starts. SuperMimic failed
opening on every start; this integration is not a successful SuperMimic solution.
No intermediate checkpoints or best seeds were selected. This limited study does
not establish convergence, generalization, or a comparison to upstream XHand.
GPU compute cost was zero. CPU update timings and process wallclock follow.

| Method | Seed | WM seconds | Policy seconds | Process seconds |
| --- | --- | --- | --- | --- |
| supertrack | 29401 | 8.925 | 6.098 | 19 |
| supertrack | 29402 | 8.925 | 6.098 | 18 |
| supertrack | 29403 | 8.934 | 6.135 | 19 |
| supermimic | 29401 | 9.035 | 12.248 | 25 |
| supermimic | 29402 | 8.916 | 11.782 | 24 |
| supermimic | 29403 | 8.983 | 12.028 | 25 |


The SuperMimic rows above used the earlier acceleration/L1 model fit. An upstream
semantic audit found that HandSuperMimic instead uses increments and squared
normalized model error summed over valid steps per window. The Door implementation
now follows that contract and rejects legacy acceleration checkpoints before
loading. Those six historical runs are preserved; they are not replications of
the corrected SuperMimic source. A new bounded pilot precedes any new frozen
three-seed protocol. The existing critic return/normalizer order agrees with
upstream; the single full-batch critic update remains explicitly declared.

A corrected-model development pilot also tested the upstream critic schedule
(5 epochs, 4 minibatches) with 1000 model and 100 policy updates, batch32/H4,
seed2927, final checkpoint only. All five physical episodes remained finite and
failed opening (nominal door RMSE .761012 rad). Increasing critic fitting did
not resolve the physical failure. Critic schedules are explicit checkpoint
parameters; the default remains one full-batch update.

A frozen-model coverage audit evaluated every valid window of those five saved
physical episodes, without fitting or checkpoint selection. At H4 the nominal
failed episode's door prediction RMSE was .0560 rad, versus .0009 rad for
persistence; the training reference gave .0045 rad versus .0658 rad. At H16 the
failed episode's errors were .1793 versus .0015 rad. Aggregate normalized state
error still beat persistence, showing why an aggregate metric alone cannot
validate the task-relevant dynamics. These same-clip development episodes were
excluded from model fitting but are not independent final test data. The audit
supports collecting physical policy-state coverage before another training
pilot; it does not establish the sole cause of failure or validate policy gradients.

The opt-in CPU development runner now accepts `--physical-replay --replay-noise
0.02`. It executes reference controls plus seeded iid action noise in native
physics, splits complete collection episodes (last episode held out) before
normalization/fitting, and samples reset-safe multi-step model windows and
physical policy-start states. This changes training data; the default remains
reference-only. `replay.npz` records both splits outside the repository. The
held-out split is saved for subsequent model-error analysis, not used for
optimization. These are same-clip development episodes, not independent task
references or evidence of robust generalization. No learned-policy collection
or online exploration is claimed.

The bounded corrected SuperMimic physical-replay pilot (seed 2927, 1000 model
updates / 100 policy updates, horizon 4, critic 5 epochs / 4 minibatches) remained
finite but opened 0/5 evaluation episodes; nominal door RMSE was 0.759919 rad.
Model and policy update times were 5.355 s and 3.682 s on one CPU thread; GPU
compute was zero. This negative result does not establish sufficient physical
coverage or effective model gradients. Matched corrected-source replication is
preregistered separately; historical acceleration-model runs remain historical.


The CPU runner also accepts `--replay-dataset /outside/repo/physical-rollouts.npz`
from `python -m ndm_gym.eval_door`. Its adjacent `result.json` must identify the
source checkpoint and matching model/reference SHA256. The last complete episode
is held out before normalization and optimization; this is same-clip validation.
Do not combine this option with `--physical-replay`.

A bounded failure-only replay pilot (seed2933,1000 model/100 policy updates,
SuperMimic critic5epochs/4minibatches) remained0/5 opening. Heldout failure H4 Door
RMSE was0.000114rad, but successful-reference error0.06570rad was near persistence
0.06579rad. This negative result motivates preserving successful interaction
coverage alongside failures; improved prediction of a stationary door is not
evidence of improved control. CPU model/policy update costs5.391s/3.711s; no GPU
compute or formal seed replacement.


Online replay development (round2940): both methods now execute current-policy
controls in native physics via `collect_policy_replay`; each of three rounds
collects four episodes and holds the last out before normalization or fitting.
Equal training windows come from retained successful replay and the three new
policy episodes. Each method used 1000 bootstrap model updates followed by
600 model / 300 policy updates, batch32/H4, policy learning rate 1e-5, final
checkpoint only, no supervised updates. Both opened all five narrow starts.
This is software/data-path feasibility, not completed effective final training
or robust learned-feedback improvement; meaningful frozen three-seed studies
per method remain pending. The command-line reference-noise collector described
above is distinct from this programmatic policy collector.

Replay sampling speed (round2941): cached flat indices preserve the original
uniform window ordering, RNG consumption, frame/reset sentinel and episode
boundaries, including unequal episode lengths. On gpu156's CPU, float32, one
thread, CUDA hidden, 300 calls after 10 warmups: batch32/H4 sampling measured
0.229 ms original versus 0.064 ms cached; batch256/H4 1.571 versus 0.168 ms;
batch256/H16 1.577 versus 0.417 ms. These are sampling timings, not whole-training
speedups. One same-seed SuperTrack bootstrap/online/evaluation pair measured
13.463 versus 13.210 seconds; final model/policy state, physical trajectories
and task metrics were bit-identical. A single ordered timing pair does not
establish an overall speed improvement. GPU paths remain unmeasured.


The [frozen online three-seed study](online_study_2943.md) reports genuine SuperMimic and SuperTrack world-model training with matched declared budgets. Both methods retain opening on the five narrow starts but worsen tracking relative to reference controls; this is bounded negative evidence, not an effective final benchmark.

### Self-contained online development command

`python -m ndm_gym.train_door_online --model /path/native-scene.xml
--reference /path/reference.npz --bootstrap /path/physical-rollouts.npz
--method supertrack --seed 2944 --output /data/YifanKang/adroit-door/new-run`
(join the lines into one command). Use `--method supermimic` for the reward/critic
method. Inputs are required; no historical audit script or checkpoint is loaded.
The bootstrap file uses `episodeN_qpos`, `episodeN_qvel`, `episodeN_actions` from
physical evaluation. Its last episode is excluded before all normalization and
sampling. Supply independently verified native-model/reference-compatible data;
file hashes identify inputs but do not establish their physical provenance.
Record acquisition/generation cost separately.

Each round collects current-policy episodes, excludes the last, fits the world
model, then optimizes the policy through that model. Training windows are equally
sampled from retained bootstrap and recent episodes. Defaults are CPU eager fp32,
one thread, B256/H4, 1000 bootstrap model updates plus ten rounds of 500 model and
250 policy updates, policy LR 1e-5. These are development settings, not an efficient
GPU certification. Supervised fitting is unavailable in this command.

`--resume /path/latest.pt` restores the model, optimizers, calibration and Torch
RNG at a completed round boundary into a **new** output directory. All protocol
settings and input content hashes must match. An atomic `latest.pt` is replaced
only after a complete round; a crash during collection/update repeats that round
from its previous saved boundary. Only the final policy is evaluated. Interrupted
round costs must be added when reporting total costs. Do not treat resumed process
time as total training time.

Protocol format 5 records policy/world-model widths and the critic schedule.
Checkpoints and final results contain actual Adam step counts for each optimizer.
Before saving, these must equal the declared bootstrap/round budgets (including
critic epochs and minibatches); inconsistent or missing per-parameter updates
raise an error. Recovery verifies saved counts. Older protocol formats require
historical source and cannot resume into this version. These checks establish
executed update counts, not physical learning success.

This command is a new reproducible development path. Its bootstrap holdout and
source grouping are explicit and differ from the historical six-episode split
used in round2943; it does not retrospectively reproduce or replace those results.
Exact optimizer/RNG recovery is tested separately for both methods. Physical CLI
warmup and current full checks remain required before using this path for a study.

Generate a new physical bootstrap without a historical learned checkpoint:

```sh
python -m ndm_gym.collect_door --model /path/native-scene.xml \
  --reference /path/reference.npz --output /data/YifanKang/adroit-door/bootstrap-new \
  --seed 2945 --noise .01 --offsets -.01 0 .01 .005
```

Pass `bootstrap-new/episodes.npz` as `--bootstrap` to the online trainer.
The collector executes reference controls with seeded IID control noise in native
physics; it never assigns states after reset. The last complete episode is held
out by the trainer. `generation.json` records input/output hashes, transitions,
physics wall time, CPU threads and simulator version. Generation is a shared cost
when both methods use the same dataset, reported separately from each training
process. The XML hash identifies that file only: preserve its referenced assets
and the original native model contract too. This generates new development data;
it does not recreate the historical mixed2934 source or its acquisition costs.
On gpu156, MuJoCo3.15.0, CPU one thread, seed2945, four226-control episodes took
0.374s for904physical transitions (GPU compute0). This collection timing excludes
process startup, model loading and file serialization; it is not a method speedup.

The online CLI also accepts `--device cuda:0 --fused-adam`, opting into the
upstream fused Adam setting for the world model, policy, and SuperMimic critic.
It changes results by rounding; the default is unchanged. Recovery requires the
same flag and protocol format (currently 6). This option does not enable CUDA
graphs. On L40S, batch256/H4, one CPU thread, fp32, 3 warmups and 20 measured
actual updates, eager/fused median milliseconds were SuperTrack WM 10.262/10.118,
policy 13.220/13.148; SuperMimic WM 10.435/10.353, policy 19.624/19.417.
These single ordered measurements include replay sampling/transfers and do not
establish a stable speedup. CPU remains the faster measured default for the small
complete online schedule. Both fused methods passed optimizer/RNG recovery and
one bounded physical cycle; this is software validation, not learning benefit.


The upstream `hand-supertrack-fast` and `hand-supermimic-fast` configurations
are XHand measurements, not Door presets. Their systems options have the
following applicability to this native CPU-physics collector:

| Upstream option | Door status and evidence |
| --- | --- |
| `fused_adam` | Explicit CUDA opt-in, measured above; no stable speedup established. |
| `reset_masked`, `no_sync`, `collect_graph` | Not implemented: native MuJoCo collection runs on the CPU and replay is stored as CPU episodes. The upstream collector requires Warp physics, device replay writes, and masked resets. |
| `update_graphs` | Not implemented: current updates read finite checks and reference-window guards on the host; SuperTrack also calibrates its loss balancer on the first update. Upstream graph capture requires its `no_sync` update contract. |
| `compile_eager` | Upstream hand option compiles collection forward kinematics; Door does not use that hand-state FK path. No Door speed claim. |
| bf16 and inference graphs | Forward-only experiment, not online training support. bf16 changed predictions; inference capture does not validate optimizer capture. |

The canonical trainer therefore uses the measured faster CPU eager path for
this small model/batch schedule. This is a scoped implementation choice, not
certification that the upstream fastest end-to-end training requirement is
complete. A future GPU collector must preserve native Adroit model, actuator,
control and reference contracts and pass physical equivalence before reuse.

The online CLI optionally compiles the differentiable world-model loss with
`--compile-wm` (TorchInductor default mode). This leaves the imagined policy and
critic updates eager. Compilation can change results by rounding and has a
first-update cost; the flag does not promise a speedup or implement the upstream
collection `compile_eager` option. End-to-end comparisons must include lazy
compilation and physical collection, with matching budgets and actual optimizer
counts. Default execution remains eager fp32. Recovery requires the same flag.

A bounded CPU public-CLI comparison (one thread, B256/H4, seed2955,
40 WM/20 policy updates, one online round and one physical evaluation) took
1.496s eager versus 45.124s compiled WM for SuperTrack; SuperMimic took
1.726s versus 23.419s. These single ordered measurements include first-update
compilation but exclude command startup/bootstrap acquisition. Actual counters
matched all budgets (including 20 Mimic critic updates); all four rollouts
opened and remained finite. Compilation changed Track tracking return from
9.028513 to 9.028517; Mimic returned 9.037642 for both. This is software/path
validation on one narrow start, not learning benefit or three-seed evidence.
Compiler Python-header warnings did not prevent completion. The short schedule
supports retaining eager default; it does not measure steady-state speed.

A separate warm WM-only comparison loaded independent deep copies of the same
eager checkpoint per method and used identical replay/RNG, three warmups and
20 measured B256/H4 updates. Median eager/compiled WM times were 9.310/6.522ms
(SuperTrack) and 9.643/6.818ms (SuperMimic), about 1.4x faster. Sampling and
optimizer work are included; policy, collection and first compilation are
excluded. Both paths finished at63WM/20policy updates (Mimic20critic).
These single ordered CPU measurements support a warm WM speed gain, not a
whole-training or GPU speed claim. Initial failed timing attempt reused mutable
optimizer checkpoint tensors; its count assertion failed, and only the corrected
independent-copy comparison is reported here.

Native CPU collection can opt into `--collection-worlds 3` on the online CLI.
This batches policy inference across at most three independent MuJoCo instances;
physics still steps serially with the original noslip solver and hand-only controls.
The default remains one world. Batch rounding may change results, so the option
belongs in the frozen protocol and cannot be changed on recovery.
A bounded single-thread CPU measurement (five starts, three repetitions, nonzero
feedback, no training) gave median serial/batched collection times of
0.684/0.593 seconds (1.15x). All recorded actions, qpos and qvel were identical
in this measurement, including a partial final batch and repeated reset.
This does not certify CUDA collection, whole-training speed or learned benefit.
The identical native model was rejected by MuJoCo-Warp because its original
`noslip_iterations=20` solver is unsupported; the solver was not disabled.

### Optional upstream policy optimizer

`python -m ndm_gym.train_door_online --upstream-policy-optimizer` selects policy Adam betas
(0.9, 0.95) and value clipping at 0.1 after the existing norm cap of 1.
World-model and critic optimizers retain their settings. The default preserves
historical Door training; SuperMimic already used value clipping. Checkpoints
record the option and reject recovery with a different configuration; checkpoints
without the field use the historical default. This changes learning, with no
speedup claim. Matched bounded CPU pilots (B256/H4, 6000 model and 2500 policy
updates, final five same-clip starts) did not establish a tracking benefit:
returns 9.02622 with upstream betas and 9.01764 with the value cap, versus
9.03557 for the zero-correction prior. These are development diagnostics.

SuperTrack additionally accepts opt-in `--correction-reg` and
`--correction-imag-noise` (both default zero). The regularizer is
`reg * (mean(c**2) + mean(abs(c)))` on the dimensionless policy correction,
excluding the demonstration prior. Imagined controls are
`clip(reference + 0.1 * (c + noise), -1, 1)` before the native actuator
midpoint/range mapping; previous-action observations retain the clean action.
Noise is absent from physical collection/evaluation. This follows upstream's
separation of policy action and imagined noise, with explicit Door control
units and clipping; numerical upstream noise values are not equivalent joint
perturbations across embodiments. SuperMimic rejects nonzero values. Checkpoint
recovery checks both options; historical absent fields mean zero. No training
benefit has been established for these options.

SuperTrack's opt-in `--task-weights` follows upstream HandSuperTrack's
`--st_task_weights`: each of Door's four L1 coordinate errors receives weight
`1 / (horizon * reward_kernel_width)`. Widths are 0.02 m for ARTz and
0.3/0.2/0.3 rad for hand/door/latch. This replaces first-batch calibration;
weights stay fixed and recovery rejects a different weighting configuration.
It remains an L1 tracking objective, rather than the exponential task reward:
the mean absolute hand error also differs from the reward's mean squared hand
error. Thus this is an upstream-supported loss scaling option, not exact
reward optimization or evidence of improved physical control. SuperMimic
rejects the option and continues to optimize imagined reward with its critic.
Historical checkpoints without the field retain first-batch calibration.
