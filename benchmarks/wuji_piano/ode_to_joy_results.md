# Wuji Hand piano: the contributed study on an RTX 3060

The contributed study: both algorithms' starting configurations (seeds 0-2, one RTX 3060 12 GB, 20 rounds, 102,400
transitions per seed). The A100 runs of both algorithms, each after the same settings screen, are in
[results_a100.md](results_a100.md).

How to read SuperMimic's numbers on this page: it ran with the library's default discount (0.99) and critic settings,
untuned on this task, on settings shared with SuperTrack that were fixed after a pilot run of SuperTrack alone; its
critic overpredicts the return on every seed ([critic diagnostics](#critic-diagnostics)). With the configuration the
A100 page's screen selected at this horizon (`--gamma 0.95 --bptt-decay 0.95`) its Eden F1 is 0.833 ± 0.079.

Neither algorithm reached the validation target of 0.95 within the fixed budget. These measurements describe performance at round 20. Each checkpoint played the same 256 held-out starts, including the nominal start. Uncertainties are sample standard deviations across training seeds.

## Fixed-budget results

| Measurement after 20 rounds | SuperTrack | SuperMimic |
| --- | ---: | ---: |
| Eden F1 | 0.8490 ± 0.0705 | 0.4123 ± 0.2090 |
| Precision | 0.9124 ± 0.0253 | 0.7539 ± 0.1757 |
| Recall | 0.8421 ± 0.0778 | 0.3900 ± 0.1870 |
| Two-note frames with both due pitches down (%) | 69.4 ± 10.9 | 10.5 ± 12.7 |
| Two-note frames with exactly the intended keys down (%) | 65.4 ± 11.7 | 10.1 ± 12.6 |
| Exact keyboard match across all frames (%) | 71.7 ± 10.6 | 23.7 ± 11.9 |
| Note events reached per episode (of 19) | 17.76 ± 0.73 | 8.30 ± 7.83 |
| Supplemental RTX 3060 training seconds per seed | 323.1 ± 16.3 | 361.3 ± 19.1 |
| Collected control transitions per training second | 317.4 ± 15.6 | 283.9 ± 14.6 |
| Paired attacks, within 50 ms | 1,374/3,072 (44.7%) | 50/3,072 (1.6%) |
| Episodes reaching every note | 169/768 | 0/768 |
| Episodes exact for the whole song | 0/768 | 0/768 |

| Seed | SuperTrack F1 | SuperMimic F1 | SuperTrack precision / recall | SuperMimic precision / recall | SuperTrack training / evaluation s | SuperMimic training / evaluation s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.9051 | 0.4893 | 0.9206 / 0.9033 | 0.7865 / 0.4571 | 311.9 / 42.2 | 350.4 / 64.7 |
| 1 | 0.8721 | 0.5719 | 0.9326 / 0.8683 | 0.9111 / 0.5342 | 315.5 / 61.1 | 383.4 / 54.1 |
| 2 | 0.7698 | 0.1758 | 0.8840 / 0.7546 | 0.5642 / 0.1787 | 341.9 / 62.7 | 350.3 / 53.9 |

The untrained actors scored 0.1304 F1 for all six runs. The scripted controller scored 0.6325 and reached all 19 notes on 0/256 starts. The following counts use the same held-out episodes on the RTX 3060, equally weighting seeds. A paired attack requires both measured onsets within ±50 ms of the score and of each other. [Per-seed events and scripted reference](note_timing_results.md)

| Measurement | SuperTrack | SuperMimic |
| --- | ---: | ---: |
| Missing due pitches per frame | 0.283 ± 0.132 | 1.112 ± 0.343 |
| Extra pitches per frame | 0.167 ± 0.043 | 0.373 ± 0.258 |
| Frames with an extra pitch (%) | 15.7 ± 3.2 | 28.1 ± 17.6 |
| Rest frames with all keys released (%) | 99.6 ± 0.3 | 94.1 ± 4.5 |

## Score and physical task

The two-hand Wuji Hand 2 beta2 scene uses Eden's built-in 90 BPM score, with
15 melody notes and four held bass notes. The full episode lasts 11.50 s and
includes the one-second lead-in. All 230 × 88 target key states were checked
against the upstream piano-roll generator. Two pitches are due together on 164
frames, one on 36 frames and none on 30 frames. Four frames begin two new notes
together. [Eden score](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/songs.py)

Every physical key is down at 87% of its travel. For each frame, precision,
recall and F1 compare the full keyboard against the score for the step just
taken. A zero denominator is scored as 1 for that metric. Average over frames
and then episodes. A note is reached when its pitch is down at least once
during its score window. Exact frames include releases and exclude extra keys.
Attack timing and hand contacts are diagnostics. [Eden scorer](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/custom_terms/metrics.py)

Two floating Wuji Hand 2 beta2 hands have six wrist pose targets and 20 independent finger targets per hand. All 52 axes are actuated. The wrist and finger targets span the model’s action space with its effort limits. Both policies learn their own fingering. Their observation contains the 52 normalized actuator positions, their last changes, 88 key depths, the previous action and eleven frames of 88 target keys. Physics runs at 200 Hz and control at 20 Hz. [Eden configuration](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/config.py)

## Training and evaluation

Both algorithms use the NDM Gym hand actor, replay and native update methods. The actor has hidden widths 192/128. The shared, task-specific world model (not the library's acceleration or increment model) integrates actuator motion and estimates travel of each passive key using a differentiable fit to the ten fingertip positions, an actuator MLP of widths 128/128 and a key residual MLP of widths 192/128. [Piano dynamics](../../ndm_gym/worldmodels/piano.py) · [Hand algorithms](../../ndm_gym/algos/hand_algos.py)

| Setting | SuperTrack | SuperMimic |
|---|---|---|
| Score-conditioned learning signal | Minimize due-key travel error + 0.85 × maximum unintended depth + 0.20 × nearest-tip distance error | Reward equals one minus the same weighted errors |
| Imagined horizon and actor | Three steps through frozen piano dynamics, MLP policy | Three steps through frozen piano dynamics, MLP policy |
| Critic | — | 256/128 hidden units, γ = 0.99, λ = 0.95, 5 epochs × 4 minibatches of 72 per actor update |
| Collection and fitting | 32 worlds × 160 steps per round, 90 model and 50 actor updates | Same, plus 20,000 critic optimizer steps across the 20 rounds |
| Optimizer | Fused Adam, actor betas 0.9/0.95, model betas 0.9/0.999 | Same, critic betas 0.9/0.999 |
| Learning rates | 0.0008, actor decays linearly to 0.00016 in rounds 11–20 | Same, critic fixed at 0.0008 |

Each seed receives 1,800 two-step model updates and 1,000 policy updates with batches of 96. Replay spans six collection rounds. Collection starts with 32 random-action worlds, then uses 24 noisy-policy worlds and eight random-action worlds. Training starts vary by up to 0.14 of each normalized action scale. The reach error uses the closest of any ten tips, measured in units of 35 mm. Evaluation uses only physical keyboard travel.

The validation set uses 32 fixed starts (RNG 456, perturbation up to 0.15 of scale). Each untrained baseline and round-20 checkpoint plays the full song from 256 fixed starts (RNG 789, up to 0.20 of scale), without exploration or mid-song resets. Validation was recorded after each training round. The target is the first recorded mean validation F1 ≥ 0.95. The six fixed-budget runs shared one configuration apart from the update algorithm.

| Collection round | Real transitions per seed | SuperTrack validation F1 | SuperMimic validation F1 |
| ---: | ---: | ---: | ---: |
| 1 | 5,120 | 0.3832 ± 0.0209 | 0.1725 ± 0.0728 |
| 5 | 25,600 | 0.5687 ± 0.0778 | 0.1537 ± 0.1476 |
| 10 | 51,200 | 0.7603 ± 0.1322 | 0.2899 ± 0.0892 |
| 15 | 76,800 | 0.8303 ± 0.0775 | 0.3454 ± 0.0622 |
| 20 | 102,400 | 0.8401 ± 0.0778 | 0.4158 ± 0.2118 |

## Timing on the contributor’s RTX 3060

Training times are supplemental measurements on one RTX 3060 with 12 GiB. CUDA-synchronized training includes collection, updates, round-end validation and checkpoint writes. Scene initialization, untrained baseline, held-out evaluation and rendering are outside the timer. Runs used Python 3.12, PyTorch 2.13.0+cu129, MuJoCo 3.15.0, MuJoCo-Warp 3.15.0 and Warp 1.17.0.

| Seconds per seed across 20 rounds | SuperTrack | SuperMimic |
|---|---:|---:|
| Physics collection | 169.6 ± 4.2 | 174.1 ± 2.5 |
| World-model updates | 9.4 ± 0.6 | 9.5 ± 0.2 |
| Policy updates, including the critic where applicable | 11.6 ± 1.0 | 26.9 ± 0.1 |
| Validation | 132.4 ± 13.1 | 150.7 ± 16.6 |

## Timing the native training updates

The study compared eager updates with the hand trainer's CUDA graphs on one RTX 3060 (12 GB, PyTorch 2.13.0+cu129, MuJoCo-Warp 3.15.0). Every condition ran three rounds at seeds 0/1/2 and used its final checkpoint. Each seed collected 15,360 transitions (32 worlds × 160 steps × 3 rounds) and ran 90 world-model and 50 policy updates per round, with minibatches of 96. SuperMimic also trained its critic. Each round validated on 32 starts. Each final policy played the full score on the same 32 held-out starts.

| Trainer | Updates | Training s/seed | Collect s/round | Model s/round | Policy + critic¹ s/round | Validate s/round | Final Eden F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| SuperTrack | Eager | 50.5 ± 1.8 | 8.96 | 0.47 | 0.57 | 6.85 | 0.332 ± 0.099 |
| SuperTrack | CUDA graphs | 44.5 ± 2.4 | 8.59 | 0.13 | 0.16 | 5.94 | 0.293 ± 0.096 |
| SuperMimic | Eager | 51.3 ± 2.4 | 8.96 | 0.49 | 1.38 | 6.26 | 0.276 ± 0.116 |
| SuperMimic | CUDA graphs | 48.4 ± 4.5 | 8.73 | 0.13 | 0.37 | 6.90 | 0.422 ± 0.143 |

¹ The policy phase includes critic updates for SuperMimic. Training time includes graph warmup and capture, validation and checkpoint writes. It excludes scene setup, the untrained baseline and final evaluation. Training time and F1 show the mean ± sample SD across seeds. Phase times are synchronized means across nine rounds.

The graphs capture native model fitting and policy improvement and read the replay buffer's current write position and fill level on the GPU. The fused Adam optimizer reads the actor's scheduled learning rate from a device tensor. Device-side sampling changes the random draws, so enabling the option can change the learned policy. The 20-round results at the top use the default eager updates. The three-round F1 values describe early training.

Individual measurements below use the final checkpoint after three rounds. The same seeds and 32 starts were used throughout. Time is in seconds on the RTX 3060.

| Seed | SuperTrack eager s / F1 | SuperTrack graphs s / F1 | SuperMimic eager s / F1 | SuperMimic graphs s / F1 |
|---|---:|---:|---:|---:|
| 0 | 49.35 / 0.2956 | 44.40 / 0.3762 | 51.44 / 0.3339 | 51.76 / 0.5761 |
| 1 | 52.63 / 0.4444 | 42.04 / 0.1871 | 48.76 / 0.1417 | 43.34 / 0.3953 |
| 2 | 49.61 / 0.2565 | 46.94 / 0.3145 | 53.64 / 0.3510 | 50.10 / 0.2945 |

Reproduce one pair with the following commands. Use `supermimic` in both commands for the other pair.

```bash
bench=benchmarks/wuji_piano
run=data/wuji_piano/update-timing
"$bench/run.sh" supertrack "$run/supertrack-eager" --seeds 0 1 2 --rounds 3 --eval-starts 32
"$bench/run.sh" supertrack "$run/supertrack-graphs" --seeds 0 1 2 --rounds 3 --eval-starts 32 --update-graphs
```

## Recorded rollouts

Each recording uses the round-20 policy with the highest round-20 validation F1 for its algorithm. The complete nominal episodes use the same 230-frame score. Audio is synthesized from measured presses.

| Nominal replay on RTX 3060 | SuperTrack, seed 0 | SuperMimic, seed 1 |
|---|---:|---:|
| Eden F1 | 0.9190 | 0.6181 |
| Written notes reached | 17/19 | 17/19 |
| Exact keyboard frames | 189/230 | 99/230 |
| Timely two-note attacks | 2/4 | 0/4 |
| Actuator commands varying over the song | 52/52 | 52/52 |
| Measured key/frame entries differing from the saved nominal evaluation | 5/20,240 | 15/20,240 |

Open-loop predictions start at each valid recorded state and consume the actual policy actions. Errors below are mean absolute fractions of full key travel on each recording.

| Horizon | SuperTrack all keys / due keys | SuperMimic all keys / due keys |
|---|---:|---:|
| 50 ms | 0.0372 / 0.1433 | 0.0477 / 0.2143 |
| 150 ms | 0.0513 / 0.1845 | 0.0647 / 0.3313 |
| 500 ms | 0.0627 / 0.2997 | 0.0873 / 0.4695 |

## Critic diagnostics

SuperMimic validation and optimizer measurements use seeds 0–2 on the same RTX 3060. Losses come from the last update of round 20.

| Seed | Best validation F1 (round) | Final validation F1 | Final world-model loss | Final critic MSE | Final imagined reward per step |
|---|---:|---:|---:|---:|---:|
| 0 | 0.6434 (3) | 0.4938 | 0.4987 | 2.0874 | 0.0881 |
| 1 | 0.6313 (14) | 0.5776 | 0.5296 | 0.9529 | 0.1562 |
| 2 | 0.6048 (17) | 0.1760 | 0.4215 | 13.6036 | 0.0776 |

The critic’s predictions below use each round-20 nominal physical trajectory. Measured returns discount the continuous reward by 0.99 to the end of the song. Positive bias denotes overprediction.

| Seed | Critic at song start | Measured discounted return at start | Mean critic bias across frames | Mean absolute critic error |
|---|---:|---:|---:|---:|
| 0 | 35.52 | 26.70 | +11.00 | 11.04 |
| 1 | 34.74 | 41.87 | +14.46 | 15.39 |
| 2 | 46.86 | -24.97 | +60.47 | 60.47 |

[Training, evaluation and replay commands](README.md#commands)
