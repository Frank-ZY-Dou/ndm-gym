# BlinkMimic

## 1. What it is

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (F. Chen, 2026; MIT) trains SuperMimic's objective with the
training changes of [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao, 2026; MIT) and its own.
Both derive from supermimic_vibed, as NDM Gym does. Here `--algo blinkmimic` is SuperMimic with BlinkMimic's released
recipe (`PRESET[main]` of `scripts/presets.sh` @ `d264658`) in this library's options (`ndm_gym/algos/blinkmimic.py`,
`PRESET`). It is not a separate algorithm: every change is an option of its own, off by default, and the
algorithm-independent ones also apply to SuperTrack.

```bash
ndm-train --algo blinkmimic --log_dir logs/blm                                    # the recipe: 3500 iterations
ndm-train --algo blinkmimic --iterations 1000000 --max_hours 3 --log_dir logs/blm   # options given here take precedence
ndm-train --algo supertrack --pi_ema 0.995 --pi_opt muon --wm_mirror 0.5 ...        # single options with any algorithm
```

The run's `args.json` records `algo: supermimic` and `preset: blinkmimic`; `evaluate.py` deploys the policy average
(`--pi_ema`), which checkpoints hold as `pi`.

## 2. How the methods relate

All of them learn the policy through one learned world model of the simulator.

- **SuperTrack** (Fussell et al. 2021): supervised. The policy minimizes L1 tracking losses through the world model; no
  reward, no critic.
- **SuperMimic** (Liao et al. 2026): the policy maximizes the imagined reward through the world model, with a critic
  beyond the horizon; first-order gradients.
- **BlinkMimic**: SuperMimic's objective with InstantTrack's and its own training changes (section 3).

**BlinkMimic's semi-implicit world model is the idea of SuperTrack's acceleration world model.** BlinkMimic's model
(`--wm_structured`, `ndm_gym/worldmodels/semi_implicit.py`) predicts velocity increments and integrates the positions from
the old and new velocities, dt ((1 − c) v + c v′) with c = (n + 1) / (2n) = 0.625 for the simulator's n = 4 substeps, plus a
learned position correction. SuperTrack's model (Fussell et al. 2021, §3.4; `ndm_gym/worldmodels/accel.py`) predicts
accelerations, that is velocity increments per dt, and integrates the positions by semi-implicit Euler; with
`--st_substeps 4` it uses the same c. With zero corrections, a step of BlinkMimic's model equals a step of SuperTrack's
(`tests/test_blinkmimic.py::test_semi_implicit_integration_is_supertracks`). SuperMimic's model
(`ndm_gym/worldmodels/increment.py`) predicts the position increments directly and does not integrate velocities.

## 3. BlinkMimic's changes as options

The rows of BlinkMimic's table "What BlinkMimic changes" and of InstantTrack's change tables. Present: the option existed
before. New: added for BlinkMimic, off by default; the default paths are bit for bit unchanged
(`tests/test_algo_equivalence.py`, `tests/test_supertrack_golden.py`).

| change | BlinkMimic / InstantTrack | here | status |
|---|---|---|---|
| policy EMA; checkpoints and evaluation use the average | `--pi_ema 0.995` | `--pi_ema 0.995` | new, any algorithm |
| a share of the resets one bin before recent failure points | `--fail_reset 0.3` | `--fail_reset 0.3` | new, any algorithm |
| contact reward in imagination (evaluation's reward unchanged) | `--contact_reward 0.25` | `--contact_reward 0.25` | new, SuperMimic |
| world model 3 × 1024 | `--wm_hidden 1024,1024,1024` | `--wm_hidden 1024 1024 1024` | present |
| learning rate 4e-4, 5 world-model batches per iteration | `--lr 4e-4 --wm_batches 5` | the same | present |
| fused Adam | `--fused_adam` | `--fused_adam` | present |
| 3 × 3 rotation products fused | `--fused_rot` | `--elementwise_3x3` | present |
| more of the step compiled | `--compile_more` | `--compile_eager` | present (the collection's reward, the normalizer features, the critic input) |
| update steps as CUDA graphs | `--cuda_graph` | `--update_graphs` | new, SuperMimic and SuperTrack (the whole updates, optimizer steps and policy average included; below) |
| no host synchronization in the update loop | `--no_sync` | `--no_sync` | new, any algorithm |
| Muon on the hidden weight matrices of policy and world model | `--pi_opt muon --wm_opt muon` | the same | new, any algorithm (world model: MLP; LAFAN1 and muscle training also the RigidFormer, Muon on its blocks' attention and SwiGLU matrices) |
| action built on the next reference pose | `--pi_ref_skip` (InstantTrack: `--ref_residual 1.0`) | `--sm_residual` | present |
| no world-model-only warm-up | `--wm_only_iters 0` | the same | present |
| semi-implicit world model | `--wm_structured` | `--wm_structured` | new, SuperMimic (MLP) |
| left-right mirror of half of each world-model batch | `--wm_mirror 0.5` | `--wm_mirror 0.5` | new, any algorithm |
| a step's episodes of one bin averaged in the reset statistics | `--bin_fix` | `--bin_fix` | new, any algorithm |
| reference contact from each clip's own floor | `--ref_contact_fix` | `--ref_contact_fix` | new, with the contact reward |
| policy batch 2048; policy learning rate 4×, ramped to 1× over 300 iterations once the training return reaches 0.75 (by iteration 1400) | `--pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --lr_decay_by 1400` | the same | `--pi_batch` present, the rest new (any algorithm) |
| 3500 iterations, a checkpoint every 100 | `--iterations 3500 --ckpt_every 100` | the same | present |
| evaluation-guided resets | `--eprobe_share 0.15 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7` | the same | new, any algorithm (`ndm_gym/algos/eprobe.py`; a probe process, below) |
| the probe sweep as a CUDA graph, run alongside training | `--eprobe_graph --eprobe_async 20` | the same | new (the sweep in the probe process, the failures applied at BlinkMimic's iteration, below) |
| collection and resets as CUDA graphs; background checkpoints | `--graph_early --collect_graph --reset_masked --ckpt_light --ckpt_minutes 0` | `--graph_early --collect_graph --reset_masked --ckpt_light` | new, the one-robot LAFAN1 scene (below); `--ckpt_minutes 0` needs no counterpart |
| InstantTrack only: smaller policy (1024-512-256), goal encoding batched per rollout | `--enc_hidden --dec_hidden`, `--batch_goal_enc` | – | not in BlinkMimic's recipe |

Evaluation-guided resets, ported with a probe process:

- **What BlinkMimic does.** From iteration 1000, every 125 iterations, it sweeps the deployed policy (the average,
  deterministic) over all clips in a second, evaluation-mode simulator of its training process: one world per 300
  frames of each clip, started a random lead (200-450 frames, one per sweep) earlier on the reference and run to the
  tile's end. The first failures (termination or a non-finite action) add to per-bin counts that decay by 0.7 per sweep;
  each bin weighs the counts of itself and the 10 bins after it in its clip, and 15% of the resets follow these weights.
  With `--eprobe_graph --eprobe_async 20` the sweep is one captured step replayed on a side CUDA stream, 20 steps per
  training iteration, and its failures apply once its last step has run.
- **Here.** Isaac Lab 3 holds one `SimulationContext` per process (a second one raises), and the worlds of a scene step
  together, so probe worlds in the training scene would step with every collection step and could not take 20 steps per
  iteration without the training worlds. The probe simulator therefore runs in a worker process on the same GPU
  (`python -m ndm_gym.algos.eprobe --worker`, started by the trainer): a `TrackingSim` in evaluation mode as
  `ndm_gym.evaluate` builds it, one world per tile (1489 for LAFAN1's 40 clips). At a sweep's start the trainer draws the
  lead (as BlinkMimic, from torch's default generator) and sends a copy of the average; the worker sweeps (one captured
  step replayed, as `evaluate --graph`, with `--eprobe_graph`); the trainer applies the failures at the iteration where
  BlinkMimic does, `ceil(steps / 20)` iterations after the start (steps = lead + tile; the first sweep three fewer, its
  graph warm-up), waiting there if the worker is not done, as BlinkMimic waits for its side stream. The counts, the window
  and the focus's share of the reset distribution are BlinkMimic's (`tests/test_eprobe.py` compares them with its code).
- **Residual differences.** The sweep runs in another process instead of on a side stream of the same process: the two
  share the GPU by time slicing rather than concurrent streams, which changes the wall-clock cost only. Checkpoints keep
  the counts, the focus and its share; a sweep in flight is not kept, as in BlinkMimic. A worker whose trainer is gone
  exits.
- **Cost, measured** (`--algo blinkmimic`, seed 0, one A100 40GB, development run): 161 ms per iteration without sweeps
  (iterations 500-1000), 229 ms with them (1000-1500, four sweeps); a sweep of the 1489 probe worlds takes 12.8-14.5 s
  (604-690 steps of one CUDA graph, about 21 ms each) and adds about 8 s of training wall-clock. The sweeps at 1000-1375
  found 16, 8, 9 and 7 failures; the focus then covered 108-142 of the 8842 bins (1.2-1.6%) and drew 19-21% of the resets
  (its share 15% plus the base distribution's mass there). Each applied at BlinkMimic's iteration (1032, 1156, 1285, 1409).
  With both processes clients of one CUDA MPS server, the sweeps cost as much (about +60 ms per iteration in 1000-1500,
  with and without, on a loaded GPU).

The systems options (wall-clock only: the same computations up to rounding and random streams), in train.py's shared loop:

- `--no_sync`: no host synchronization in the loop. A non-finite policy gradient skips the optimizer step on the device
  (the log's `nonfinite` column counts them); the normalizers test their count limit on the device; starts are drawn from
  twice as many candidates, the valid ones kept by a stable sort (the default's selection when enough are valid).
- `--update_graphs` (`--cuda_graph`): the world-model update and the policy update, with the optimizer steps (capturable
  fused Adam, Muon with tensor learning rates) and the policy average's step, as CUDA graphs, captured once the replay
  buffer is full (`--graph_early`: at once, the buffer's head and fill level on the device). The survival products are a
  loop of products, since cumprod's backward tests for zeros on the host.
- `--collect_graph --reset_masked`: the collection step (observation, policy, simulator step, reward, replay write,
  normalizers) and the episode ends (bin statistics, resets by Isaac Lab's mask writes) as one CUDA graph; each replay's
  physics step first runs Isaac Lab's solver reset and forward kinematics of the environments reset by the one before,
  as an eager step does.
- `--ckpt_light`: periodic checkpoints hold the policy average only, written by a background thread; the full checkpoint
  and `last.pt` at the end. With `--last_every_min M`, `last.pt` also every M minutes of training, at the next periodic
  checkpoint, so that `--resume` can continue a run stopped before its end.

Not ported: `--ckpt_minutes 0`, which needs no counterpart (train.py checkpoints by time only at `--ckpt_hours`).

With the systems options instead of their CUDA-graph options, the final configurations of the LAFAN1 study run faster
(ms per iteration on an A100 40GB, iterations 250-450 with `--wm_only_iters 100`, so that the policy updates are timed):

| | SuperTrack, MLP | SuperMimic, MLP | SuperTrack, RigidFormer | SuperMimic, RigidFormer |
|---|---|---|---|---|
| `--wm_cudagraphs --pi_cudagraphs` (`--critic_cudagraphs`) | 304.3 | 326.5 | 846.7 | 1005.2 |
| `--no_sync --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light` | 271.7 | 296.1 | 808.2 | 998.7 |

## 4. Validation

`tests/test_systems.py`: the graphable updates give the default updates' parameters bit for bit (with the default's
cumprod as the same loop of products, which equals cumprod up to rounding); a non-finite policy step is skipped on the
device; with `--update_graphs` each optimizer has its own device learning rate, so that `--wm_lr` and `--pi_lr_mult` give
the rates they give without it (also in muscle training: `tests/test_muscle_systems.py`); the device-side freeze test,
replay head, weighted bin statistics and gated Muon step; in the simulator, the masked training step, eagerly and as a CUDA
graph, equals the indexed one bit for bit, and a run with all the systems options captures and replays its graphs.
`tests/test_eprobe.py`: the probe worlds' tiles and sweep frames, the failure counts, their decay and window against
BlinkMimic's own code, the focus's share of the reset distribution (and share 0 leaving it bit for bit), the deterministic
probe policy, the options' checks, and the iteration at which each sweep's failures apply.
`tests/test_blinkmimic.py`: the preset is BlinkMimic's `PRESET[main]` flag by flag, and command-line options override it;
unsupported combinations are refused; Muon takes exactly the hidden matrices; the policy average and its checkpoints;
the defaults give the unchanged updates bit for bit; the mirror is an involution, the G1's default pose and action scales
are symmetric, and the forward kinematics of a mirrored state are the mirrored bodies; the semi-implicit model's targets
integrate back to the next state, and its step equals SuperTrack's with 4 substeps; the contact term; the reset
statistics; the policy's learning-rate schedule; every option of the recipe together trains.
`tests/test_wm_muon_rf.py`: `--wm_opt muon` with the RigidFormer world model (LAFAN1 training): Muon takes exactly the
blocks' weight matrices as stored (q/k/v and the attention gate one matrix, the SwiGLU's value and gate one matrix), Adam
every other parameter, with and without merged tokens; one update trains both; the MLP's split and `--wm_opt adam` are
unchanged; on the GPU the update with `--update_graphs` and `--rf_fused_train`, captured as a CUDA graph, gives the eager
update bit for bit.
`tests/test_muscle_wm_muon_rf.py`: the same option in muscle training: train_muscles takes it with either token set; the
muscle RigidFormer's split is LAFAN1's rule on the same blocks (the muscle encoder, embeddings and read-out on Adam); one
update trains both parts; the update without host synchronization is the default one bit for bit and skips a non-finite
one; a checkpoint restores the optimizer's state, with `--update_graphs` into the device learning rates; on the GPU a
short run with the fused RigidFormer and the systems options captures and replays its updates.

On an A100 40GB, `--algo blinkmimic` trains at 161 ms per iteration between the probe sweeps and 229 ms over the sweep
periods (iterations 1000-1500); with warm compile caches the first 50 iterations, with compilation and graph capture,
take 20 s. Training uses 12.5 GiB of GPU memory, the probe worker 2.2 GiB. A run resumed from its `last.pt` continues where
it stopped.

Time to target on one A100 40GB, every checkpoint (one per 100 iterations) evaluated four times on all 40 clips
(`evaluate --group 18 --repeats 4 --graph`), minutes of training wall-clock as BlinkMimic counts them:

| | seeds | first return ≥ 0.82 | first return ≥ 0.83 | ≥ 0.82 and 0 failed, 3 checkpoints in a row |
|---|---|---|---|---|
| `--algo blinkmimic` | 0 / 1 / 2 | it 1500 / 1500 / 1400; 4.81 / 4.82 / 4.54 min | it 1600 / 1600 / 1700; 5.23 / 5.25 / 5.68 min | it 1700 / 2300 / 1900; 5.64 / 7.96 / 6.44 min |
| without the evaluation-guided resets (`--eprobe_share 0`) | 0 / 1 / 2 | it 1600 / 1600 / 1500 | it 1700 / 1700 / 1800 | it 2300 / 2200 / – |

BlinkMimic's own code on an A100 40GB (seed 0, its evaluator with 4 replicas) reaches 0.834 at iteration 1500, at 175 ms
per iteration between sweeps, after about 10 min of start-up with cold compile caches; on an RTX 4090, its README reports
1300 / 1500 / 1200 iterations and 2.89 / 3.28 / 2.64 min to 0.82 at 133 ms per iteration.

## 5. The other training loops

The systems options and the recipe's options that apply are also options of muscle training (`ndm_gym.train_muscles`) and
the two-robot scene (`ndm_gym.train --paired`), off by default; the default paths are unchanged. Each trainer refuses the
options it does not implement; HOI training (`ndm_gym.train_hoi`) takes none of them.

| option | muscles | two robots |
|---|---|---|
| `--no_sync --update_graphs --graph_early` | yes | yes (as LAFAN1) |
| `--collect_graph --reset_masked` | yes (`train_muscles.collection_step` in `MuscleSim.capture`) | yes (`PairedTrackingSim.step_core`, `step_tail_masked`, `reset_masked`) |
| `--ckpt_light`, `--last_every_min`, `--fused_adam`, `--elementwise_3x3`, `--combo_kernels` | yes | yes |
| `--pi_ema`, `--pi_opt muon`, `--wm_opt muon` | yes (`--wm_opt muon`: the MLP and the RigidFormer) | yes |
| `--pi_lr_mult` with `--lr_decay_len` | ramped at `--lr_decay_by` (the log has no normalized training return) | as LAFAN1 (`--lr_decay_ret` too) |
| `--fail_reset`, `--bin_fix` | no: MuscleSim draws its resets inside its reset graph | no |
| `--wm_mirror` | no: no mirror map of the 284 muscles | allowed |
| `--eprobe_*` | no | no |
| `--wm_structured`, `--contact_reward` | no (LAFAN1 SuperMimic's model and foot-contact term) | no |

Milliseconds per iteration, synchronized profiles after the world-model-only iterations (collection / model update /
policy update), A100 40GB unless noted:

| loop | as before | systems options |
|---|---|---|
| muscles, SuperTrack-MLP | 457 (108 / 189 / 160) | 344 (102 / 125 / 117) |
| muscles, SuperMimic-MLP | 621 (109 / 261 / 251) | 395 (103 / 130 / 162) |
| two robots, SuperTrack-MLP (before: `--wm_cudagraphs --pi_cudagraphs`) | 418 (161 / 117 / 140) | 306 (71 / 115 / 120) |
| the same, A100 80GB | 328 (94 / 112 / 122) | 289 (65 / 110 / 114) |

On the muscles the collection graph saves 6 of the collection's 108 ms (the simulator's kernels dominate), and
SuperMimic-MLP's model update, not captured before, varies between profiles (654 ms per iteration in a second one).
Unprofiled, the muscle runs train at 3.04 (SuperTrack-MLP) and 2.62 (SuperMimic-MLP) iterations per second with the systems
options, 2.51 and 1.87 without.

Validation: `tests/test_muscle_systems.py`, `tests/test_paired_systems.py`. The updates written without host
synchronization give the default updates' values and parameters bit for bit (muscle SuperTrack and SuperMimic); a
non-finite update is skipped on the device. The collection graphs play the eager or indexed steps' episodes: the same
episode ends, frame counters and flags, the reset states bit for bit (two robots), the states within the simulator's own
run-to-run differences across processes. A short run of each trainer with all the systems options captures and replays its
graphs.
