# Animal motion tracking: results

MIMIC-MJX's rat, fruit fly and stick insect track their animals' recorded kinematics on MuJoCo-Warp, with SuperTrack and
SuperMimic ([task](../../docs/tasks/animals.md)). First round: one configuration per algorithm, three training seeds each.

## How the numbers were obtained

- **Clips:** one policy per body on clips 0-9 of its reference; held-out: clips 10-19, never trained on.
- **Runs:** SuperTrack with `hand-supertrack-fast`'s options (`--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights`,
  Muon, the policy average, the learning-rate ramp, the hand task's systems options), SuperMimic with `hand-supermimic-fast`'s
  (`--lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95` and the same), both with `--pi_batch 2048 --wm_lr 2e-3
  --fused_fk`; 2,048 worlds; a fixed budget of iterations, the learning rate decayed over it: 4,000 (stick insect), 6,000
  (fly), 10,000 (rat); seeds 0, 1, 2. `benchmarks/mimic_mjx_animal_tracking/run_formal.sh` runs them.
- **Evaluation:** `evaluate_animal` on each run's last checkpoint: every clip played from its first frame and four more times
  from perturbed starts (0.02 rad on the joints), the policy without exploration noise, to the clip's end. Errors over the steps
  the episodes lived: the tracked bodies' and the end effectors' positions (mm; the fly's centimetres converted), the joints'
  RMS error. Ended: the share of episodes ended by a termination (MIMIC-MJX's) before their clip's end.
- **Hardware:** one RTX 4090 (WSL2), shared by three training queues at once. The training times are wall clock under that
  sharing, upper bounds of a run alone, and not engineered.
- **Table:** mean ± sample SD over the seeds.

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) | env steps |
|---|---|---|---|---|---|---|---|---|---|
| stick insect (≈3.5 cm) | SuperTrack | 0.0 ± 0.0 % | 0.41 ± 0.01 mm | 0.67 ± 0.02 mm | 8.6 ± 0.1° | 0.7 ± 1.2 % | 0.77 ± 0.01 mm | 17 ± 3 | 8.2 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.39 ± 0.01 mm | 0.59 ± 0.00 mm | 4.1 ± 0.1° | 1.3 ± 2.3 % | 0.87 ± 0.03 mm | 19 ± 3 | 8.2 M |
| fruit fly (≈2.6 mm) | SuperTrack | 0.0 ± 0.0 % | 0.062 ± 0.001 mm | 0.116 ± 0.003 mm | 4.8 ± 0.4° | 6.7 ± 3.1 % | 0.092 ± 0.008 mm | 27 ± 4 | 12.3 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.047 ± 0.002 mm | 0.078 ± 0.003 mm | 3.8 ± 0.2° | 0.0 ± 0.0 % | 0.066 ± 0.000 mm | 30 ± 4 | 12.3 M |
| rat (≈16 cm reach) | SuperTrack | 69 ± 3 % | 35.8 ± 1.4 mm | 33.1 ± 2.2 mm | 12.1 ± 0.3° | 83 ± 8 % | 52.8 ± 4.1 mm | 53 ± 6 | 20.5 M |
| | SuperMimic | 19 ± 1 % | 19.6 ± 1.7 mm | 16.8 ± 1.1 mm | 9.8 ± 0.1° | 51 ± 6 % | 43.2 ± 3.0 mm | 54 ± 10 | 20.5 M |

**In short:** the stick insect and the fly are tracked to 1-2 % of their size without a failed episode, on held-out clips too,
in under half an hour each on a shared consumer GPU; SuperMimic has the lower errors on both. The rat, torque-driven through
40 ms activation filters with 67 joints, is not solved in this budget: SuperMimic ends 19 % of its episodes (SuperTrack 69 %)
after 20.5 M environment steps, where MIMIC-MJX's PPO reaches 95 % of its test reward after about 650 M (6 to 20 hours on one
GPU). Its critic appears to carry the balance that SuperTrack's 0.16 s imagined window does not see.

Not comparable as they stand: MIMIC-MJX reports the rat's median keypoint error against stac-mjx's keypoints (about 10 mm on
held-out clips); the bodies' error here is a mean over the body frames' origins, and the rat does not yet finish its clips.

## Development notes

- **PD servos for the rat:** gains from each actuator's effective inertia and force (natural frequency 1-8 Hz) keep the open
  loop stable but leave it as short-lived as zero torque (17 % of a clip against 16 %): through 40 ms of activation lag a stable
  stiffness is two orders of magnitude below what holds the rat up, and the policy has to learn its feedforward torques. Not
  kept.
- **Iteration time** (rat, SuperTrack, 2,048 worlds, the GPU alone): collection 20 ms, world-model updates 28 ms, policy update
  43 ms with `--pi_batch 2048 --fused_fk`, from 145 ms with the default batch of 4,096 and the level-by-level kinematics.
  `--fused_fk` compiles for about 5 minutes on the rat, inside the training time above.
- **Open:** the rat (a longer budget, a longer imagined window for SuperTrack), the time to a target, and the RigidFormer.
