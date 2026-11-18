# Two freely moving Wuji hands playing Ode to Joy

## Results

| Seed | Baseline F1 | Precision | Recall | Final F1 | Both notes down | Exact two-note frames | All 19 notes reached | Time to .95 | Training | Final evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1304 | 0.9206 | 0.9033 | 0.9051 | 74.8% | 72.0% | 0/256 | unreached | 311.9 s | 42.2 s |
| 1 | 0.1304 | 0.9326 | 0.8683 | 0.8721 | 76.6% | 72.4% | 165/256 | unreached | 315.5 s | 61.1 s |
| 2 | 0.1304 | 0.8840 | 0.7546 | 0.7698 | 56.9% | 52.0% | 4/256 | unreached | 341.9 s | 62.7 s |

Across training seeds, final F1 was **0.8490 ± 0.0705**, precision 0.9124 ± 0.0253, recall 0.8421 ± 0.0778, and training time **323.1 ± 16.3 s** (mean ± sample SD). Total timed training was 16.16 GPU-minutes. Mean end-to-end training throughput was 317.4 ± 15.6 collected control transitions per second.

All due pitches were down on 69.4 ± 10.9% of the two-note frames. The entire keyboard matched the score on 65.4 ± 11.7% of those frames and 71.7 ± 10.6% of all frames. The policies reached 17.76 ± 0.73 of 19 note events per episode. 0 of 768 episodes matched all 230 frames.

At the four written two-key attacks per episode, both measured onsets were within ±50 ms of the score on 1,374/3,072 attacks. On 1,374/3,072 attacks the two keys also started within 50 ms of each other.
The median inter-key delay among the matched attacks was 50 ms, sampled at 20 Hz.

| Training component | Seconds per seed, mean ± SD |
|---|---:|
| Physics collection | 169.6 ± 4.2 |
| World-model updates | 9.4 ± 0.6 |
| Policy updates | 11.6 ± 1.0 |
| Validation | 132.4 ± 13.1 |

The complete three-seed command took approximately 19.8 wall-clock minutes including setup, untrained and trained evaluations, and checkpoint I/O.

## Task and scoring

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

## Training protocol

- Hardware was one RTX 3060 with 12 GiB, NVIDIA driver 595.99.02 and an Intel
  i7-12700F host. Software was Python 3.12.16, PyTorch 2.13.0+cu129, MuJoCo
  3.15.0, MuJoCo-Warp 3.15.0 and Warp 1.17.0. Physics ran at 200 Hz and the
  policy at 20 Hz. The local task was added on NDM Gym base commit
  `173fd778456d2743ff03ff71861f5b6c96d4dda0`.
- Both wrists are free joints with Cartesian pose servos. Each hand's six wrist
  axes and 20 finger joints receive independent policy outputs. All hand and
  keyboard collisions use the assets' native geometry. The hands start at
  Eden's default placements for its handwritten songs. The scene uses the
  built-in action scaling about rest and the actuator force limits. Policy
  actions freely span that action space. No score note is assigned to a hand
  or finger. The neutral-pose actuator transmission matrix has rank 52, confirming
  independent mechanical actuation of all 52 axes. The policy has no motion
  reference or scripted warm start.
  [Eden configuration](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/config.py)
- SuperTrack uses NDM Gym's hand actor, GPU replay, native world-model and policy
  update methods, and optimizer loop. The actor has 192- and 128-unit hidden
  layers. It receives 52 normalized actuator positions, their 52 last changes,
  all 88 key depths, the last 52 actions and 11 frames of 88 target keys.
  The learned dynamics predict actuator motion and all 88 next key depths.
  A differentiable approximation of the hand-model fingertip geometry helps the
  model locate possible contacts across the keyboard. Its two-layer actuator
  and key residual networks use widths 128/128 and 192/128, respectively.
- Training optimizes continuous due-key travel, unintended key travel and
  distance from each due key to the closest of any of the ten fingertips.
  This last term is a soft learning signal, following Eden's approach-to-key
  shaping. It supplies no hand assignment or evaluation credit. SuperTrack
  differentiates through three imagined steps of the frozen model. There is
  no additional action-amplitude, wrist, motion-style or fingertip-pad penalty.
- Seeds 0, 1 and 2 each used 20 rounds, 32 worlds and 160 steps of physics
  collection per round. That is 102,400 physical control transitions per seed.
  Each round ran 90 model updates on two-step windows (batch 96) and 50 actor
  updates (batch 96), totaling 1,800 and 1,000 updates per seed. Fused Adam
  used a model learning rate of 0.0008. The actor started at 0.0008 and
  decayed linearly to 0.00016 over the second half. On round 1, exploration
  samples all 52 controls. Subsequent rounds mix 24 noisy-policy worlds and
  eight random-action worlds.
- Each final checkpoint and its untrained actor baseline were evaluated on the
  same 256 starts (RNG 789, one nominal), with no exploration or resets during
  the complete song. Perturbations are applied only at reset. Each normalized
  coordinate varies independently by up to 0.20 of its Eden action scale,
  e.g. up to 20 mm along the keyboard and 0.10 rad at each finger joint.
  The one-second lead-in allows perturbations to settle. Validation uses 32
  starts from RNG 456, up to 0.15 of scale. The time-to-target is the first
  validation reaching mean Eden F1 of at least 0.95. This is a reporting
  target chosen for the run.
- Times use CUDA synchronization and cover collection, model and actor fitting,
  validation and checkpoint writes. Final evaluation, baseline evaluation,
  scene initialization, rendering and the exploratory six-round seed-5 pilot
  are outside timed training. This run benchmarks SuperTrack. PR #7
  also benchmarked SuperMimic on a different task and GPU.
  [PR #7 protocol](https://github.com/Yzmblog/ndm-gym/blob/c26810196aa99b93fd0a73f00df9f27778eed620/benchmarks/quadrotor_racing_tracking/README.md)

[Train, replay and render](README.md#eden-ode-to-joy-with-52-independent-controls)

## Recorded rollout

The recording uses seed 0's **final** checkpoint, chosen by its highest final validation F1 across the three seeds (0.9070). In a fresh process the nominal rollout reached 17/19 notes, scored 0.9190 F1, and matched the full keyboard on 189/230 frames. 2/4 two-key attacks had both measured onsets within 50 ms of the score and of each other. There were 24 measured onsets. The saved and replayed nominal keyboard masks differed in 5 entries. The MP4 and WebM decode to 230 frames with audio synthesized from measured key presses.

The intended note windows entirely missed in the replay were E4 at 5.00 s and E4 at 9.00 s (episode time, including the one-second lead-in).

Contacts reconstructed from the physical poses linked the left hand to G2, C3 and the right hand to C4, D4, E4, F4, G4 on correctly depressed keys. All 52 policy outputs varied during the episode.

| Open-loop world-model horizon | Mean absolute error, all 88 key depths | Mean absolute error, due key depths |
|---|---:|---:|
| 1 step (50 ms) | 0.0372 | 0.1433 |
| 3 steps (150 ms) | 0.0513 | 0.1845 |
| 10 steps (500 ms) | 0.0627 | 0.2997 |

Each prediction starts from the measured state and consumes the recorded actions. Errors are fractions of full key travel over all valid windows of this nominal episode.
