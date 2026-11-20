# Wuji Hand piano: the contributed study's note timing and scripted reference on an RTX 3060

The contributed study's note-timing diagnostic of its runs (both algorithms' starting configurations, seeds 0-2) and of a
scripted reference. The A100 runs of both algorithms, each after the same settings screen, are in [results_a100.md](results_a100.md).
SuperMimic's runs here are at the library's default discount (0.99) and critic settings, untuned on this task
([how to read them](ode_to_joy_results.md)).

All seven evaluations ran on an RTX 3060 and used the same 256 starting poses and full 230-frame score. The primary result uses Eden framewise F1. This diagnostic counts each written and played note once. A single held note can cover several correct frames and still miss a repeated attack.

## Note events

Match equal MIDI pitches within one control tick (±50 ms), maximizing the number of matches and then minimizing total absolute onset error. Precision, recall and F1 below pool events across starts within each seed. Timing columns give the median absolute error across matched attacks or observed releases in milliseconds. Releases extending to the final frame have unknown timing and are excluded. The repeat column covers successive attacks of the same pitch in one voice (5 per episode). The paired column counts individual notes starting on the four simultaneous two-note attacks (8 per episode). The separate paired-attack metric requires both notes to start on time.

| Method | Seed | Precision | Recall | Event F1 | Hits | Misses | Extras | Onset / release (ms) | Observed releases | Repeat hits | Paired-note hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SuperTrack | 0 | 0.6690 | 0.8598 | 0.7525 | 4,182 | 682 | 2,069 | 0 / 0 | 3,670 | 1,206/1,280 | 1,622/2,048 |
| SuperTrack | 1 | 0.4538 | 0.8575 | 0.5935 | 4,171 | 693 | 5,020 | 0 / 0 | 3,758 | 940/1,280 | 1,419/2,048 |
| SuperTrack | 2 | 0.3567 | 0.7584 | 0.4852 | 3,689 | 1,175 | 6,654 | 0 / 0 | 3,211 | 899/1,280 | 1,013/2,048 |
| SuperMimic | 0 | 0.1065 | 0.1355 | 0.1193 | 659 | 4,205 | 5,528 | 0 / 500 | 659 | 34/1,280 | 504/2,048 |
| SuperMimic | 1 | 0.5888 | 0.7500 | 0.6597 | 3,648 | 1,216 | 2,548 | 0 / 150 | 3,648 | 914/1,280 | 1,198/2,048 |
| SuperMimic | 2 | 0.0021 | 0.0029 | 0.0024 | 14 | 4,850 | 6,775 | 0 / 150 | 14 | 0/1,280 | 0/2,048 |
| scripted | — | 0.8559 | 0.6316 | 0.7268 | 3,072 | 1,792 | 517 | 0 / 50 | 2,816 | 768/1,280 | 1,280/2,048 |

The score-driven reference solves static hover and press joint positions for seven pitches. Its left wrist shifts 70 mm, and each command leads the score by one control tick. Both hands use the same actuator limits and physical key-down threshold as the policies. These hand and finger assignments belong only to the scripted controller. It was evaluated once without training.

The script scored 0.6325 Eden F1, precision 0.9977 and recall 0.5717. It reached 12.0/19 note events and held both pitches on 23.8% of two-note frames. Both onsets were timely on 512/1,024 written paired attacks. Evaluation took 26.3 seconds on the RTX 3060, excluding scene and IK setup.

## Validation under a shared time limit

A secondary timing diagnostic: the last completed round at or before 300 synchronized training seconds of each seed (a cutoff set after training), with its logged 32-start validation result. The primary tables use the 256-start final evaluations after equal experience.

| Method | Rounds (seeds 0/1/2) | Validation F1 (seeds 0/1/2) | Mean ± sample SD |
|---|---|---|---:|
| SuperTrack | 19 / 19 / 17 | 0.9153 / 0.7110 / 0.7089 | 0.7784 ± 0.1186 |
| SuperMimic | 17 / 15 / 16 | 0.4435 / 0.3387 / 0.3425 | 0.3749 ± 0.0595 |

## World-model onset predictions

From every valid starting frame of each final policy’s held-out nominal physical rollout, replay its recorded actions through the frozen model for 1, 3 or 10 steps. Initialize both windows with the measured key state. Compare threshold crossings within each window using the same one-to-one pitch matcher. The table pools overlapping windows across the three seeds per method. Timing error covers matched events only. A one-step window has a single possible onset tick. SuperTrack's final checkpoints were replayed on the same nominal start to record their actuator observations.

| Method | Horizon | Physical onsets | Predicted | Matched | Precision | Recall | Mean absolute timing error (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SuperTrack | 50 ms | 95 | 75 | 58 | 0.773 | 0.611 | 0.0 |
| SuperTrack | 150 ms | 285 | 220 | 111 | 0.505 | 0.389 | 3.6 |
| SuperTrack | 500 ms | 950 | 893 | 320 | 0.358 | 0.337 | 4.8 |
| SuperMimic | 50 ms | 75 | 100 | 25 | 0.250 | 0.333 | 0.0 |
| SuperMimic | 150 ms | 225 | 174 | 65 | 0.374 | 0.289 | 15.4 |
| SuperMimic | 500 ms | 744 | 433 | 130 | 0.300 | 0.175 | 24.2 |

[Training and reproduction](README.md) · [Key-depth errors for both models](ode_to_joy_results.md#recorded-rollouts)
