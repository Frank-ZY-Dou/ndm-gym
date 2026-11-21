# NablaMimic

## 1. What it is

[NablaMimic](https://github.com/sailab-ai/NablaMimic) (∇Mimic; W. Lu, Y. Chen, *∇Mimic: Ultra-Efficient Humanoid Control
with Differentiable Simulation*, [arXiv:2610.41654](https://arxiv.org/abs/2610.41654); Apache-2.0; read at `bfbb2a6`,
version 0.2.0) trains one policy that tracks the 40 LAFAN1 clips on the Unitree G1 by backpropagating through an exact
differentiable simulator. It has no world model. The simulator is
[NablaEngine](https://github.com/sailab-ai/NablaEngine) (the same authors, Apache-2.0): a CUDA engine written for the G1
that computes MuJoCo's contact dynamics in float32 and the analytic adjoint of a whole 20 ms control step. A patched
MuJoCo Warp, with the adjoint of an open pull request, is its second training backend and the simulator of its
evaluation.

It is not an `--algo` of this library. This page describes its authors' code, which
[`benchmarks/g1_lafan1_tracking/nablamimic/`](../../benchmarks/g1_lafan1_tracking/nablamimic/) runs unmodified on the
task this library's algorithms train on.

**What its authors report** (its README at `bfbb2a6` and the paper that README links, read 2026-11-11; one RTX PRO 6000,
three seeds, its own evaluation, [section 4](#4-its-evaluation-beside-this-librarys)): return 0.82 after 8 minutes
(0.12 ± 0.01 h per seed, 1,250 updates of 0.38 s), 0.910 ± 0.004 at 3 h, no clip with a fall from 31 minutes on. The arXiv
version (v1, 2026-10-30) describes the earlier recipe: 0.82 after 20 minutes, 0.875 at 3 h. On an H100 the paper has one
run, of the earlier recipe with absolute actions: 0.82 after 0.41 h.

**Measured here** with its code and its own evaluation, seeds 0-2, one hour of training each (mean ± sd; per seed in the
[results](../../benchmarks/g1_lafan1_tracking/results.md#nablamimics-own-code-timed)):

| configuration | GPU | s per update | first evaluation at return 0.82: minutes from update 50 (on its clock) | crossing, interpolated (min) | return after the hour |
|---|---|---|---|---|---|
| published, `configs/ours.args` | A100 80GB SXM4 | 0.67 | 14.48 ± 4.31 (15.12 ± 4.31) | 14.25 ± 3.39 | 0.876, 0.876, 0.851 |
| reduced memory, `remat` with `--replay-steps 250` ([section 5](#5-memory-on-the-gpu)) | A100 40GB SXM4 | 0.72 | 18.48 ± 1.72 (19.16 ± 1.72) | 17.93 ± 2.33 | 0.871, 0.864, 0.872 |

These numbers stand beside this library's [training speed](../../README.md#training-speed) on the A100 40GB SXM4
(SuperTrack 4.13 ± 0.18 min and SuperMimic 3.52 ± 0.30 min to return 0.82, one GPU, seeds 0-2) and are not ranked
with it: the two evaluations run on different simulators and robot models ([section 4](#4-its-evaluation-beside-this-librarys)).

## 2. How it trains

Paths are NablaMimic's at `bfbb2a6`. One update (`src/nablamimic/train.py`; the values of `configs/ours.args`):

1. **Collection.** 4096 collector environments run the policy with action noise N(0, 0.1²) and return-based reset
   sampling over 1 s bins, 2.5 steps per update, and write their states into a FIFO replay of 1000 × 4096 states
   (`collect_step`, `:478-493`).
2. **Windows.** 4096 windows of 64 control steps (1.28 s) start from states drawn uniformly from the replay and run
   through the simulator with action noise N(0, 0.05²) (`make_rollout`, `:79-190`).
3. **Objective.** The window's discounted reward (γ = 0.99) plus the critic's value at the window's end, at time-outs and
   at clip ends; a fall gets no value. The reward is BeyondMimic's tracking reward with SuperMimic's weights
   (`src/nablamimic/env.py:56-59`, `:404-426`); training adds a smooth penalty that starts before the termination
   thresholds (`train.py:70-76`). An environment whose episode ends inside a window restarts and keeps counting.
4. **Gradient.** Reverse mode through the 64 steps (the simulator's adjoint, the policy, the reward) with respect to every
   step's action: dJ/da for 64 × 4096 samples (`:572-576`).
5. **Policy.** The action targets below, then a Polyak average of the actor (decay 0.999), which its evaluation scores.
6. **Critic.** TD(λ = 0.95) targets on the same windows, fit on the state each transition starts from, 5 epochs of 4
   minibatches (`critic_step`, `:621-658`).

Its three techniques (the paper's sections 5 and 6):

- **Per-sample clipping in the backward pass.** At every step of the backward pass each environment's action cotangent is
  clipped to 3 times the batch median of its norm, so a sample near a non-differentiable contact event cannot take over
  the update (`src/nablamimic/nets.py:114-142`, `--act-cot-clip 3`).
- **Action targets, fit in several steps.** Each sample gets the target a + η dJ/da, with η set so that the targets
  move the joint setpoints by 0.03 rad RMS (annealed to 0.3 of that with the learning rate). The actor takes 8 regression
  steps of 16,384 samples toward them with an Adam of its own, learning rate 6e-4 (`action_target_update`, `:495-559`;
  `--action-target 8 1 0.03`; after PODS, Zamora Mora et al. 2021).
- **NablaEngine.** The forward step runs as fused kernels per world and writes a tape (1,472 float32 values per world and
  substep); the adjoint of the control step is one kernel that replays it. The paper gives 1.21 ms per control step of
  4096 worlds and 1.78 ms with its gradient on an RTX PRO 6000, 1.70 and 2.73 ms on an H100.

The rest of the recipe: residual actions (the setpoint is the next reference frame's joint angles plus the scaled output;
the action-rate term stays on the commanded setpoints), SuperMimic's encoder-decoder actor (2048, 1024, 512, 256; latent
32) with the reference's relative anchor orientation into the decoder, a critic (256, 128), Adam (0.9, 0.95) at 2e-4 with
100 warm-up updates and a cosine decay to a tenth over 16,000 updates, a checkpoint every 250 updates, and a wall-clock
budget (`--hours 3`).

## 3. Running it from this repository

Results named NablaMimic come from its authors' code and entry point, unmodified.
[`benchmarks/g1_lafan1_tracking/nablamimic/`](../../benchmarks/g1_lafan1_tracking/nablamimic/) holds what this repository
adds; no file of NablaMimic or NablaEngine is copied or changed, and the reader clones them as below.

| | |
|---|---|
| `run.sh` | one run: checks the checkout, takes the GPU, starts its `python -m nablamimic.train --config ... --out ...`, then its `scripts/evaluate_run.sh` |
| `remat_config.py` | writes its `configs/ours.args` without `--no-remat` ([section 5](#5-memory-on-the-gpu)) from the checkout, with its own reader and parser |
| `nm_table.py` | reads its `config.json`, `log.jsonl` and `eval_nominal.json` into the columns of the results |

`run.sh` starts only if the checkout is at the pinned commit with no tracked file modified and nothing added under
`src`, `scripts` or `patches`; the installed NablaEngine is the revision its `scripts/setup_env.sh` pins; its patched
MuJoCo Warp tree, MuJoCo Menagerie's G1 and the 40 LAFAN1 files match its own checksum lists; the GPU is free and, for a
timed run, no other GPU process runs on the machine. It holds the GPU lock of `scripts/run_train.sh` (the directory
`$LOCKS`), pins the CPUs it is given, and records the checks and the exact command beside the run. Options reach
NablaMimic through its own command line, after its `--config`. `nm_table.py` marks a run whose source was modified, that
was resumed or that ran on the CPU; `tests/test_nablamimic_table.py` tests it on run directories in the format of
NablaMimic's files, a layout checked against files its unmodified trainer and evaluator wrote. Neither its trainer nor its
evaluator runs without a GPU.

```bash
# once, as its README says (Python 3.12, a CUDA 12 nvcc for NablaEngine's one-time build): the checkout, its environment, the data
git clone https://github.com/sailab-ai/NablaMimic && cd NablaMimic && git checkout bfbb2a6773b4776c17402a34a93ec7589d9c7c18
bash scripts/setup_env.sh && source .venv/bin/activate && pip install huggingface_hub && bash scripts/download_lafan1.sh --cache

# from this repository: a run of its published configuration, seed 0, stopped after an hour by its own --hours, with its evaluation
R=benchmarks/g1_lafan1_tracking/nablamimic
NABLAMIMIC=<checkout> GPU=0 CPUS=0-7 $R/run.sh nm_s0 published 0 --hours 1
# the reduced-memory configuration of the results, for a 40 GB GPU
NABLAMIMIC=<checkout> GPU=0 CPUS=0-7 $R/run.sh nm_remat250_s0 remat 0 --hours 1 --replay-steps 250
python $R/nm_table.py <checkout>/runs/nm_s0 <checkout>/runs/nm_remat250_s0
```

## 4. Its evaluation beside this library's

Its score is `full_norm_return` (`src/nablamimic/evaluate.py:83-90`): every clip is played once from its first frame; the
reward's eight terms are summed over the clip's steps while the robot has not fallen; the sum is divided by the clip's
steps times the largest tracking reward of a step (0.1); the 40 values are averaged. A fall forfeits the rest of its
clip. `ndm_gym.evaluate`'s `norm_return` is the same formula.

| | NablaMimic (`evaluate.py`, `eval_plan.py`, `env.py`, `robot.py`, `motion.py`) | this library (`ndm_gym/evaluate.py`, `tasks/tracking.py`, `sim/rigid/isaaclab_mjwarp.py`) |
|---|---|---|
| clips | the 40 CSV files of the LAFAN1 Retargeting Dataset | the same files (equal SHA-256) |
| resampling to 50 Hz | linear, slerp for the root; 441,120 frames; each clip is shifted in height so that the 5th percentile of its lowest foot point is on the floor | linear, slerp; 441,140 frames (20 clips keep one more, their last); heights as in the files |
| start, noise | first frame, no noise, no randomization, friction 1.0, mean action | the same |
| fall | pelvis height error above 0.25 m or gravity-z difference above 0.8; a non-finite state | the same two thresholds |
| reward terms | six tracking terms (weights 1, 1, 1, 1, 0.5, 0.5; widths 0.3, 0.4, 1.0, 3.14, 0.3, 0.4), action rate −0.1 on the commanded setpoints, joint limits −2.0, each times 0.02 | the same terms, weights and widths |
| tracked bodies | 14 | the same 14 |
| anchor | the torso: the reference is re-anchored to it, and the two 0.5 terms are its position and orientation | the pelvis |
| normalization | (clip frames − 1) × 0.1, mean over clips | the same |
| robot | MuJoCo Menagerie's G1 with mjlab's feet and contacts: 31 capsules, 2 spheres, the floor; 502 contact pairs, self-collisions on | Unitree's `g1_29dof_rev_1_0` description (24 meshes, 8 spheres, 4 cylinders); self-collisions off |
| joints, actuators | 29 joints in the same order; armature, torque limits, PD gains, joint ranges, default pose and total mass (33.34 kg) equal to this library's | |
| simulator | MuJoCo Warp, forward only, 10 Newton iterations; training runs on NablaEngine | Isaac Lab 3 with Newton's MuJoCo-Warp solver, Isaac Lab's default of 100 iterations, for training and evaluation |
| time step | 4 × 5 ms, implicitfast, pyramidal cones | the same |
| weights scored | the actor's Polyak average | the checkpoint's policy (its average with `--pi_ema`) |
| checkpoints scored | every 250 updates in the first hour, then every 1000, and the last; once each, after training | every 100 iterations, live, four evaluations averaged (`--repeats 4`) |
| clock | `wall_s`: training time from the first update, after the actor's and the collector's steps are compiled | minutes from iteration 50 to the checkpoint |
| time to 0.82 | the first crossing of the three seeds' mean curve, interpolated linearly between evaluations (its paper) | per seed, the first checkpoint whose four evaluations average 0.82; mean ± std over seeds |

**What 0.82 means.** 0.82 is the LAFAN1 return SuperMimic's paper reports, and both take it as the bar. The two
evaluators agree in the definition and differ in the physics they run it on: the G1's colliders and self-collisions, the
solver's iterations, the anchor body, the clips' heights. A NablaMimic policy's 0.82 is 0.82 by NablaMimic's evaluation.
No checkpoint is scored by both evaluators. `nm_table.py` therefore labels its rows as NablaMimic's own evaluation, and
they stand beside this library's rows, not in one ranking.

**The clock.** Its first update compiles the critic's and the statistics' steps inside the clock unless its compilation
cache holds them (`train.py:773-784`, `:798`, `:855-856`; `jax_cache.py`). `nm_table.py` gives the minutes on its clock and
the minutes from update 50, as this library's training speed counts from iteration 50, and the per-seed crossing
interpolated as its paper does.

**Updates and time to 0.82 on one GPU type.** On an A100 40GB SXM4, three seeds each, each by its own evaluation and clock
([its runs](../../benchmarks/g1_lafan1_tracking/results.md#nablamimics-own-code-timed),
[the recipe's](xmimic.md#2-what-carries-over-to-this-library)):

| | NablaMimic, reduced-memory configuration | this library's recipe (`--algo blinkmimic`) |
|---|---|---|
| time per update | 0.72 s | 0.16 s (iterations 200-1000; 0.23 s with the probe sweeps of iterations 1000-1500) |
| first evaluation at 0.82 | updates 1,500, 1,500, 1,750 (evaluated every 250) | iterations 1,400, 1,500, 1,600 (every 100) |
| minutes from update / iteration 50 | 18.48 ± 1.72 | 4.48 ± 0.37 |

Its published configuration, on an A100 80GB SXM4: 0.67 s per update, updates 1,000, 1,250 and 1,750, 14.48 ± 4.31 min.

The two reach 0.82 in about as many updates, and an update through the exact simulator takes about 4.5 times as long
(0.72 against 0.16 s).
Each of its updates differentiates 4,096 windows of 64 control steps. NablaEngine's gradient of a rollout takes 5.5 ms per
control step at 4,096 worlds on this GPU
([speed tables](../../benchmarks/ndm_speed/results.md#differentiable-simulators-and-the-g1-world-models)): about 0.35 s
of the 0.72 s. This library's update differentiates 2,048 windows of 32 steps through its 3 × 1024 MLP world model;
through SuperTrack's larger 2048 × 3 MLP the gradient takes 0.7 to 1.6 ms per control step at 1,024 to 4,096 worlds. The
exact gradient needs no model and carries no model error; in these runs it does
not lower the number of updates to 0.82. The evaluators, the simulators and the checkpoint grids differ (the table above),
so the rows stand side by side and are not a ranking.

## 5. Memory on the GPU

Its README: the configurations use about 50 GB (tested on an RTX PRO 6000 and an H100); on a smaller GPU, lower
`--replay-steps` / `--collect-envs` (the replay) and `--num-envs` (the windows per update) as needed; the reported results
use the defaults.

The buffers whose size follows from its code (computed with its own modules):

| buffer | size |
|---|---|
| replay, 1000 × 4096 states of 4,408 bytes (`env.py:449-454`) | 18.1 GB |
| the windows' tapes in NablaEngine, 23,552 bytes per world and control step, 4096 × 64 | 6.2 GB |
| action targets of one set of windows and their optimizer's state (`train.py:761-765`) | 0.95 GB |
| the rest: the windows' other residuals, the environments' states (391 KB each, 375 KB of it constant buffers), XLA's workspace | not derived |

Its own options that lower it:

| option | effect | changes |
|---|---|---|
| without `--no-remat` | the windows are rematerialized in the backward pass (`jax.checkpoint` with the engine's policy, `train.py:177`, `:469`): the engine's tapes and the steps' outputs are kept, the rest of each step is computed again | the time per update; the algorithm and the hyperparameters stay, the gradients equal up to rounding (NablaEngine, `docs/api.md`, Remat) |
| `--replay-steps` | 18.1 MB per row of 4096 states: 9.0 GB less at 500 rows, 13.5 GB less at 250 | which states the windows start from |
| `--collect-envs` | the replay's columns and the collector's batch | the data per update |
| `--num-envs` | the windows per update: tapes, residuals, action targets | the method's batch |

Its base configuration sets `--no-remat`, and a configuration file cannot remove an included option, so the first row
needs a file written out in full. Its own `configs/reported/ablations/no_pods.args` is written out for that reason, and
it and `configs/shac.args` train with rematerialized windows. `remat_config.py` writes the file for `configs/ours.args`
from the checkout and checks with its parser that only `no_remat` differs: `run.sh`'s configuration `remat`.

On A100s (the peak of JAX's allocator, of its limit):

| configuration | A100 80GB | A100 40GB |
|---|---|---|
| `published` | trains: 0.67 s per update, 44.6 to 44.7 of 63.8 GB | out of memory |
| `remat` | not run | out of memory |
| `remat` with `--replay-steps 500` | not run | out of memory during training, after 600 to 1,330 updates |
| `remat` with `--replay-steps 250` | not run | trains: 0.72 s per update, 25.5 to 25.6 of 31.0 GB |

The last row is the reduced-memory configuration of the results: the published algorithm and hyperparameters with
rematerialized windows and a replay of 250 × 4096 states instead of 1000 × 4096, which is full after 100 updates
instead of 400, so that the windows start from more recent states. It is not a configuration its authors published. A
checkpoint is 217 MB, and an hour on an A100 writes 20 to 22 of them.

## 6. Licenses

NablaMimic and NablaEngine: Apache License 2.0 (Copyright 2026 The NablaMimic Authors; Copyright 2026 The NablaEngine
Authors). NablaEngine's G1 model derives from MuJoCo Menagerie's Unitree G1 (BSD-3-Clause, Unitree Robotics) with mjlab's
configuration (Apache-2.0). NablaMimic's training on MuJoCo Warp and its evaluation use a fork of MuJoCo Warp
(Apache-2.0) with NablaMimic's patches. The G1 retargets of LAFAN1 (Ubisoft La Forge) are CC BY-NC-ND 4.0,
non-commercial; NablaMimic's `scripts/download_lafan1.sh` fetches them into its checkout, and nothing here redistributes
them.

No file of NablaMimic or NablaEngine is in this repository: `run.sh` runs the reader's own checkout, and
`remat_config.py` writes its file at run time from that checkout.
