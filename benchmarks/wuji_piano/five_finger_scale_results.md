# Two Wuji hands playing Eden’s five-finger scale

Each row evaluates a final SuperTrack checkpoint on the same 256 starts. Eden’s full 152-frame score includes a one-second silent lead-in, nine notes at 80 BPM, and the intervening rests. The lower two keys are assigned to the left hand and the upper three to the right. Coverage counts a note if it crosses the existing 87% travel threshold during its own window. An exact frame requires all 88 key states to agree. [Eden score](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/songs.py) · [Eden metrics](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/custom_terms/metrics.py)

| Seed | Untrained F1 | Final precision | Final recall | Final F1 | Exact frames | Perfect episodes | All nine notes reached | Time to validation F1 ≥0.95 | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2895 | 0.9998 | 0.9759 | 0.9758 | 97.55% | 5/256 | 256/256 | 17.4 s | 59.6 s |
| 1 | 0.2895 | 0.9999 | 0.9811 | 0.9810 | 98.08% | 10/256 | 256/256 | 17.3 s | 57.9 s |
| 2 | 0.2895 | 0.9998 | 0.9707 | 0.9705 | 97.03% | 0/256 | 256/256 | 16.6 s | 57.3 s |

Mean ± sample SD across the three seeds was **0.9758 ± 0.0052 Eden F1** and **58.2 ± 1.2 s** of training. Total timed training was 2.91 GPU-minutes. The held-out evaluation took 19.5 ± 0.5 s per final checkpoint. Time includes synchronized simulation collection, gradient updates, the initial validation measurement, validation after each round and checkpointing. It excludes scene construction, the untrained held-out baseline, final evaluation, rendering and a separate seed-5 pilot.

Across the 768 episodes, 33 produced exactly the nine pitches in order with no extra onsets. 6729 of 6912 target onsets had a same-pitch measured onset within one control step (50 ms), with 2,345 unmatched measured onsets. A key that briefly releases and re-crosses the threshold during one note produces another onset. The complete key masks and per-seed histories are saved with the checkpoints.

## Training and evaluation

- Hardware was one NVIDIA GeForce RTX 3060 (12 GiB). Simulated contacts ran in MuJoCo-Warp at 200 Hz. Policies issued actions at 20 Hz in 32 parallel worlds. Seeds 0, 1 and 2 ran sequentially.
- Each run used eight rounds of 152 control steps per world, totaling 38,912 training transitions. Each round performed 105 two-step world-model updates (batch 256) and 75 four-step imagined policy updates (batch 192). The five-output NDM Gym hand policy has 128- and 96-unit hidden layers. The shared finger dynamics network has two 64-unit hidden layers. Both use Adam at 0.0015.
- The 40 physical joint motors receive position commands. Each of five policy outputs controls a four-joint finger stroke, with the other fingers held in a rest posture and wrists fixed. The model learns finger and key acceleration from physical replay. The actor learns press amplitude and timing through that frozen model. The training loss tracks the five assigned key depths. Evaluation compares every one of the 88 physical keys against Eden’s roll and includes rest frames.
- Each evaluation uses one nominal and 255 perturbed starts (RNG 789), with up to ±0.08 rad of initial finger-joint displacement and persistent motor setpoint biases up to ±0.025 rad. Validation uses RNG 456 and biases up to ±0.018 rad. The training bias limit was ±0.017 rad. Each complete evaluation episode replays the fixed exercise with no exploration or mid-song resets. Results measure this arrangement and tempo.
- The score ends during the ninth note. Its release falls outside the evaluated episode. Video audio is synthesized from measured keys. The overlay shows both the target and played pitch. [Training and replay commands](README.md#eden-five-finger-scale-with-two-hands)

## Checkpoint used in the video

The video uses seed 2, round 4, selected for the highest validation F1 (0.9978) across the three training runs. Replaying that checkpoint in a separate process gave **0.9958 F1** on the 256 held-out starts. 179/256 episodes matched every frame and 182/256 produced exactly the nine written onsets in order. Its nominal episode matched all 152 frames, including the rests. At the middle of each note the saved physical pose showed contact between the assigned finger and the intended key. This diagnostic uses the finger links and adds no contact requirement to the score.

The runs used PyTorch 2.13.0+cu129, MuJoCo 3.15.0, MuJoCo-Warp 3.15.0 and Warp 1.17.0. The untrained policies received F1 credit on the 44 silent frames. Training results, key masks, learned models and the selected replay are in the local output directory `data/wuji_piano/five-finger-scale/`.
