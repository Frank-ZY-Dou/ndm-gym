# Wuji Hand piano: the contributed study's Für Elise exercise on an RTX 3060

An exercise of the contributed study (seeds 0-2, one RTX 3060 12 GB): two Wuji hands play the opening of Für Elise with a
score-specific controller, neither SuperTrack nor SuperMimic. The benchmark's results, both algorithms on A100s after the
same settings screen, are in [results_a100.md](results_a100.md).

## Score and simulator

The excerpt starts with the E5–D♯5 pickup and ends on C5 at the end of bar 5.
It has 26 right-hand notes and nine left-hand notes. The two staves preserve
the pitches, rests and written durations of the public-domain Mutopia edition.
At quarter note = 72 the 3/8 excerpt spans 6⅔ seconds, rounded to 133 scored
control frames (6.65 s). A 0.50 s lead-in positions the hands and a 0.25 s tail
lets the keys return. Both intervals are excluded from the metrics. The simulator
models physical key depression without a sustain pedal or velocity target.
[Mutopia score and source](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=931)

Hardware was one NVIDIA GeForce RTX 3060 (12 GiB).
Software was PyTorch 2.13.0+cu129, MuJoCo 3.15.0, MuJoCo-Warp 3.15.0 and
Warp 1.17.0. Physics ran at 200 Hz, with a 20 Hz controller. The Wuji Hand 2
beta2 hand models and Eden's 88 passive keys were simulated together with hand
collisions enabled.

## Learning and evaluation

A pitch-to-wrist-position guide supplies an initial trajectory to both hands.
The actor learns six residual controls over three translation axes for each
wrist, indexed by the score frame, plus a shared neural correction from wrist
position, velocity, and reference position. The two index fingers play the notes.
Finger joints hold prescribed targets during evaluation. A 15 ms lift at the
start of larger wrist movements helps the index finger clear adjacent keys.
This implementation is a score-specific hybrid of physics search and a neural
key-motion model. It has not been evaluated on other pieces or tempos.
The standard SuperTrack and SuperMimic algorithms were outside this run.

- **Seeds and updates.** Three independent training runs, seeds 0–2. Each used
  32 parallel worlds and 12 training rounds. The search keeps six partial
  trajectories per control step. Each round also performs 140 Adam updates of
  the key-motion model and 80 Adam updates of the actor (batch size 384, learning
  rates 0.002 and 0.0005). The forward model predicts next-step key travel on
  the 14 pitches used. The search and evaluator check all 88 keys.
- **Selection and timing.** The earliest checkpoint with the highest
  validation F1 over 32 starts with wrist perturbations up to 2 mm is evaluated,
  and the first validation F1 of at least 0.95 is recorded. Synchronized wall time
  includes physics search, fitting, validation and checkpoint selection.
  Scene initialization and final evaluation are measured separately.
- **Final evaluation.** Deterministic full-excerpt rollouts with no exploration
  on the same 256 starting poses per seed (one nominal, 255 with independent
  translations of up to 3 mm per wrist axis, evaluation seed 789). Each
  simulation has time to settle during the unscored lead-in. The report uses
  sample SD across the three independent training seeds.
- **Eden accuracy.** A key is down at 87% travel. Each step compares the
  actual key mask with the score for the control step just taken. Precision,
  recall and F1 are averaged over frames and then starts. A zero denominator
  gives that metric a value of 1. An exact frame requires all 88 keys to agree.
  [Eden metrics](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/custom_terms/metrics.py)
- **Supplemental measures.** Active-note recall counts hits divided by due
  key-frames in each staff's distinct pitch register, without rewarding rests.
  Onset F1 matches pitches one-to-one within one control frame (±50 ms).
  Coverage counts an intended note if its key was down in any frame of its
  scored duration. These are distinct from Eden's headline F1.

## Results

| Controller | Precision | Recall | Eden F1 | Exact frames | Onset F1 (±50 ms) | Training | First validation F1 ≥0.95 | Selected round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometric baseline | 0.8570 | 0.6692 | 0.6504 | 53.4% | 0.5639 | — | — | — |
| Seed 0 | 0.9912 | 0.9676 | 0.9678 | 92.8% | 0.9274 | 133.2 s | 58.8 s | 5 |
| Seed 1 | 1.0000 | 0.9436 | 0.9474 | 93.2% | 0.9714 | 133.5 s | unreached | 5 |
| Seed 2 | 0.9862 | 0.9468 | 0.9516 | 91.7% | 0.9180 | 133.2 s | 26.5 s | 7 |
| Mean ± sample SD | 0.9925 ± 0.0070 | 0.9527 ± 0.0130 | **0.9556 ± 0.0108** | 92.6 ± 0.8% | 0.9389 ± 0.0286 | **133.3 ± 0.2 s** | — | — |

| Seed | Treble active-note recall | Bass active-note recall | Intended notes reached, nominal | Nominal onsets matched within 50 ms / played | All 35 notes reached in one rollout | Every key correct on every frame | Final-round validation F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9835 | 0.8813 | 35/35 | 33 / 37 | 255/256 | 0/256 | 0.8996 |
| 1 | 0.9587 | 0.8974 | 35/35 | 34 / 35 | 256/256 | 0/256 | 0.8997 |
| 2 | 0.9664 | 0.8461 | 35/35 | 34 / 40 | 256/256 | 0/256 | 0.8892 |

Total timed training was 6.66 GPU-minutes. The three runs shared 0.58 s of scene setup with warmed Warp kernel caches. Final evaluation of 256 starts took 38.9 ± 0.1 s per checkpoint. Each training run evaluated the untrained actor at round 0, then used 12 search rounds, 1,680 forward-model gradient updates and 960 actor gradient updates.

All seed rows above use the checkpoint selected on validation. The final-round validation column shows the effect of selection. Seeds were run sequentially, and all hand and keyboard contact states came from MuJoCo-Warp. [The exercise's trainer](exercises/train_fur_elise.py)

Loading seed 0's saved checkpoint in a fresh process reproduced the nominal rollout's binary key states exactly. Over all 256 starts, replay F1 was 0.96796 versus 0.96785 in the original evaluation. Timing and release errors remained across seeds. The hands reached all 35 notes during their written windows in 767 of 768 evaluations, and none matched every key on every scored frame.
