# Adroit Door demonstration tracking

The Door reference adapter records full native hand, door and latch generalized
coordinates and velocities from actual physical control replay. Actual opening
is scored separately from tracking; matching a closed-door reference is not opening.
`ndm_gym.tasks.adroit_door.DoorTask` provides analytic tracking heads on the
upstream `MjcfWorldModel` state tuple `(qpos, qvel)`. The opt-in residual algorithm and bounded CPU training/evaluation entry points
are implemented below. Genuine bounded online runs are reported in the [benchmark overview](../../benchmarks/adroit_door/README.md); the final study remains pending.

Use an isolated replay environment with `numpy`, `mujoco==3.1.6` and
`gymnasium-robotics==1.3.1`. This verified legacy replay environment is separate
from the library's Python >=3.12 / MuJoCo >=3.3 environment. A bounded CPU comparison with MuJoCo 3.15.0 also opens the door (1.53947 rad
final), using the exported native model and the same 226 actions. Door-angle
RMSE against 3.1.6 is 0.002535 rad; maximum generalized-coordinate difference
is 0.067877, so engine equivalence is not established. The supported-runtime CPU adapter and checkpoint deployment have been verified;
this does not certify efficient GPU training. Do not override the main package dependencies
to make the legacy replay run.

Obtain the demonstration from [DAPG at the pinned commit](https://github.com/aravindr93/hand_dapg/tree/9a1de23df6574e1b94e1da742e3326eb1f61c013).
Only deserialize a trusted pickle; the digest check establishes identity, not
safety of untrusted Python objects. No demonstrations, models, checkpoints or
recordings are included in this contribution.

From the repository root, using that isolated environment's Python:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/replay_adroit_door.py \
  --trusted-demo /path/to/hand_dapg/dapg/demonstrations/door-v0_demos.pickle \
  --sha256 d6ec9a090261f901e5c9bf10082799e30c8d34c3f28aef4026e492a1aefc96c3 \
  --clip 23 --output /path/outside/repository/door23.npz
```

The archive contains T actions and T+1 states: state zero is the restored initial
scene; action i generates state i+1. Actions use the original controller's clipped
[-1, 1] mapping. The exporter advances physics at each substep and rejects
nonfinite states and external applied forces. It reports contacts, penetration,
soft-limit excess, native named joint order, timing and source identity. It does
not change native limits or contact settings. Only initialization assigns state.

`ndm_gym.motion.adroit_door.DoorReference.load` reads the resulting non-pickle
archive without importing a simulator. Errors include native hand generalized
coordinate RMS (mixed root translation and joint-angle units), door radians and
latch radians. These diagnostic errors are not a frozen training reward.

See `benchmarks/adroit_door/results.md` for development selection, physical
measurements, historical observation mismatch and outstanding training checks.
A saved-state video is visualization of the physical rollout, not independent
proof of dynamics. The renderer labels the preview as a replay reference;
three new training seeds remain pending. There is no README trained-policy claim.

## Articulated tracking heads

`DoorTask` reads the exported reference and exposes `obs`, `goal`, `setpoint`,
`reward`, `termination` and independent `opened` heads. ARTz translation is
scored in metres separately from the remaining 27 hand coordinates in radians;
door and latch each have their own angle kernel. Reward is the sum of four
Gaussian tracking terms minus normalized action-rate cost, integrated at the
reference control timestep. Goal features include reference positions and
velocities. A transition from frame f is scored against f+1. There is no
error-based early termination: the full reference horizon is scored; nonfinite
states terminate as failures. Default widths and weights are explicit in
`DoorRewardCfg`; these are implementation defaults, not a frozen study protocol.

Construct `DoorTask` with the native model's `(28, 2)` actuator control ranges.
`setpoint` clips normalized actions and maps them through those ranges to
physical native hand controls. Pass the same physical controls to MuJoCo and
`MjcfWorldModel.step`: the world model's servo features require physical units.
It does not output door/latch targets, add a reference feed-forward action or
modify physics. The existing
XHand policy has a different action dimension, so it cannot be reused unchanged.

`tools/render_adroit_door.py reference.npz resolved-model.xml preview.webp`
renders saved states using the common NDM Gym mark and the original Adroit
scene. Run with an appropriate MuJoCo rendering backend. Keep the output outside
the source tree while reviewing; the preview is explicitly labeled as a replay
reference. Rendering assigns saved states and is not a feasibility test.

### Physical transition and policy interfaces

`ndm_gym.sim.rigid.adroit_door.DoorPhysics(model, reference)` accepts the
caller-supplied pinned native MuJoCo model, including the recorded scene and
controller parameters. Construction checks native joint/actuator order, scene
position and timestep. `step(action)` applies 28 normalized hand controls and
returns `((qpos, qvel), done)`. The terminal state stays available until explicit
`reset()`; the adapter never resets or overwrites a state within an episode.
External applied forces are rejected. Optional explicit initial qpos/qvel offsets
support later perturbed-start evaluation; they do not change the reference.
This is a single-world CPU correctness path; GPU training throughput has not
been measured.

`ndm_gym.policies.adroit_door.DoorPolicy` uses the upstream MLP, `run_mlp` and
`RunningNorm` components, with 88 observation inputs, 60 inputs per goal frame
and 28 bounded action outputs. It does not use demonstration controls as policy
inputs. This standalone interface is not the trained residual baseline; final world-model
training and its three independent seeds remain outstanding.

### Explicit reference-control residual baseline

`DoorResidualPolicy` is an opt-in alternative to `DoorPolicy`: callers supply
one normalized reference hand action per control frame, and the network adds a
bounded correction before actuator clipping. Its final layer starts at zero,
so its initial behavior is exactly open-loop reference-control replay. This
is an explicit control prior, not a trained policy result or object assistance.
The simulator applies only the resulting 28 native hand controls.

A bounded CPU diagnostic on the replay-derived MuJoCo 3.15 clip compared the
five ARRz initial offsets `0, -.01, -.005, .005, .01` radians. Original controls
opened the door in all five (final angle 1.528–1.539 radians). Earlier action
imitation pilots failed all five, despite low supervised error. These narrow
start perturbations do not establish broad robustness, learned feedback, or
three-seed training performance. Residual feedback training and efficient
training-path measurements remain pending.

The opt-in `policies.door_imagination.residual_update` performs a bounded,
time-aligned residual update through the unchanged `MjcfWorldModel.step` API.
World-model parameters are frozen during the update while action derivatives
remain enabled; previous parameter gradient flags are restored even on failure.
Nonfinite losses or gradients reject the optimizer step. The caller supplies
normalized reference controls explicitly, fits observation normalization, and
owns model fitting, sampling, checkpoints and physical evaluation. This eager
fp32 correctness path has no efficient GPU training certification. One finite
update through an untrained model does not establish useful control gradients.

The opt-in `ndm_gym.algos.DOOR_ALGOS["residual"]` component owns the native
MJCF increment world model, residual policy and Adam optimizers. Its `wm_loss`
accepts physical actuator controls; `policy_update` accepts explicit state,
frame and previous normalized action batches. Callers must fit normalizers on
training data only. Checkpoints include both optimizers and the control prior;
loading rejects a different prior, horizon, full reference qpos/qvel, control
ranges, timestep, reward settings or articulated model layout before changing
parameters or optimizer state. Older checkpoints without this identity are
rejected; they must not be silently relabeled as compatible. This eager fp32 path has not been
certified as the fastest GPU path. Imagined return is diagnostic: the bounded
physical gain audit did not establish reliable model gradients. Three independent supervised baseline seeds are reported in the
[benchmark results](../../benchmarks/adroit_door/results.md); they do not validate
model-gradient learning. GPU efficiency measurements remain outstanding.

### Bounded CPU development entry point

`python -m ndm_gym.train_door --model /path/to/supported-native-door.xml
--reference /path/to/replay-derived-reference.npz --output /outside/repository/new-run
--seed 0 --wm-updates 100 --policy-updates 0` fits the upstream increment model
on the reference transitions, saves its final algorithm/optimizer states, and
executes the final hand policy in native physics. Supply the scene and controller
settings used to produce that reference. No automatic XML or dependency conversion
is performed. The output directory must be new; logs, checkpoints and recordings
belong outside the repository. This is a single-thread eager fp32 CPU correctness
path, not a certified efficient GPU training configuration.

The default policy remains the explicitly labelled reference-control baseline.
Positive `--policy-updates` runs a bounded diagnostic residual update; existing
model-gradient diagnostics have **not** established physical improvement. Fitting
on reference transitions does not constitute held-out model validation. Record
clip, seed, budgets and offsets before execution; evaluate the final checkpoint
without selecting intermediate policies. The runner reports tracking errors and
actual opening separately, saves full physical trajectories, and records update
and evaluation walltime. `--offsets` perturb the joint named **ARRz**, in radians;
ARRx is a different coordinate. The bounded three-seed supervised baseline is reported in the benchmark results;
GPU speed measurements remain outstanding.

Deploy a trusted local checkpoint without further updates using
`python -m ndm_gym.eval_door --checkpoint /outside/repository/run/final.pt
--model /path/to/supported-native-door.xml --reference /path/to/reference.npz
--output /outside/repository/new-evaluation`. Model and reference are explicit;
the saved task contract is checked before evaluation. Offsets default to the
saved protocol; `--offsets` explicitly overrides them. The output directory must
be new. Loading uses Python pickle, so checkpoints must come from a trusted
source. Saved trajectories contain actual physics states and hand controls.

### Supervised reference residual baseline

`--supervised-updates 1500 --batch-size 128 --teacher-gain 0.3
--train-offsets -.01 -.005 .005 .01` collects native physical feedback targets
and fits bounded residuals on top of the replay-control prior. This is supervised
learning, not a model-gradient improvement claim. Seeds 29301/29302/29303 were
frozen before execution and evaluated at ARRz 0, ±0.0075 and ±0.015 radians.
All final checkpoints opened all five episodes; tracking changes versus the
zero-residual baseline were mixed. These are perturbations of one reference,
not held-out motions. Collection, training and evaluation costs are separate.

The preliminary seeds used `--wm-updates 0 --policy-updates 0`. They are
supervised residual-controller baselines only, not completed SuperMimic or
SuperTrack training. Five narrow same-clip starts per seed do not establish
robustness. Subsequent corrected-contract online world-model seeds 29701/29702/29703 for SuperTrack and 29601/29602/29603 for SuperMimic
completed 6000 model and 2500 policy updates each (plus 2500 critic updates for
SuperMimic). Both opened 15/15 narrow starts, but tracking returns were lower
than the zero-update prior; these are negative replication results, not an
effective final benchmark. See the benchmark report for all seeds and costs. Exact replay equality is a reproducibility observation,
not an additional upstream contribution requirement.

### Online training devices

`python -m ndm_gym.train_door_online` accepts `--device cpu` (default) or
`--device cuda:0`. For CUDA, explicitly set `CUDA_VISIBLE_DEVICES` to the admitted
physical GPU. Both paths use eager fp32 and the original world-model-through-time
objectives; native physical collection and evaluation remain single-world CPU
MuJoCo. Replay sampling retains CPU RNG semantics and transfers each sampled
batch to the selected training device. Normalizers are always fitted on CPU
training episodes, then copied to the selected device; this avoids device-specific
statistics changing L1 gradient signs. Floating-point training can still differ
between devices. Recovery protocol format 3 records this normalization rule and
rejects older protocols rather than silently changing continuation. Round recovery saves CUDA RNG state as
well as model, optimizers and CPU RNG, and requires the same declared device.
This path uses ordinary PyTorch operations, not the incompatible upstream Triton
simulator kernels. It does not certify the full GPU suite or provide a CUDA graph
training configuration. Measure complete collection/training/evaluation time;
small models can spend more time transferring and synchronizing than computing.
