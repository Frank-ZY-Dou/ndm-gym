# Wuji Hand piano: SuperTrack and SuperMimic on A100s

Each algorithm's configuration, chosen by the same settings screen, at horizons 3 and 8: seeds 0-2, 20 rounds each
(102,400 control transitions), every run alone on one A100 40GB SXM4. Every final checkpoint plays the whole song from the
same 256 held-out starts. Numbers are the mean ± sample SD over the three seeds. The world model is the task-specific
`PianoDynamics` for both algorithms ([task and scoring](README.md#task-and-scoring)).

The contributed study ran the starting configurations of both algorithms on one RTX 3060 (seeds 0-2, the same budget):
Eden F1 0.8490 ± 0.0705 for SuperTrack and 0.4123 ± 0.2090 for SuperMimic ([the contributed study](ode_to_joy_results.md),
[note timing and the scripted reference](note_timing_results.md)).

## Results

| configuration (`run.sh <algorithm> OUT <options>`) | Eden F1 | precision | recall | both due pitches down | exact two-note frames | notes reached (of 19) | training per seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `supertrack --horizon 3 --st-reg 0.01` | 0.804 ± 0.140 | 0.916 ± 0.051 | 0.787 ± 0.152 | 0.618 ± 0.270 | 0.595 ± 0.264 | 17.64 ± 1.46 | 7.1 ± 0.3 min |
| `supermimic --horizon 3 --gamma 0.95 --bptt-decay 0.95` | 0.708 ± 0.115 | 0.877 ± 0.174 | 0.676 ± 0.104 | 0.414 ± 0.196 | 0.396 ± 0.211 | 16.57 ± 1.63 | 8.5 ± 0.7 min |
| `supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 0.888 ± 0.092 | 0.920 ± 0.078 | 0.891 ± 0.087 | 0.770 ± 0.150 | 0.706 ± 0.180 | 17.84 ± 1.70 | 7.7 ± 0.2 min |
| `supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.815 ± 0.024 | 0.932 ± 0.056 | 0.776 ± 0.013 | 0.430 ± 0.072 | 0.402 ± 0.104 | 14.26 ± 1.73 | 10.6 ± 0.2 min |

Every seed:

| configuration | seed | Eden F1 | precision | recall | all 19 notes reached (of 256 starts) | training (min) |
|---|---:|---:|---:|---:|---:|---:|
| `supertrack --horizon 3 --st-reg 0.01` | 0 | 0.9468 | 0.9633 | 0.9426 | 172 | 7.0 |
| `supertrack --horizon 3 --st-reg 0.01` | 1 | 0.8000 | 0.9224 | 0.7789 | 72 | 6.9 |
| `supertrack --horizon 3 --st-reg 0.01` | 2 | 0.6660 | 0.8623 | 0.6383 | 0 | 7.4 |
| `supermimic --horizon 3 --gamma 0.95 --bptt-decay 0.95` | 0 | 0.7380 | 0.9727 | 0.6776 | 0 | 9.1 |
| `supermimic --horizon 3 --gamma 0.95 --bptt-decay 0.95` | 1 | 0.5806 | 0.6762 | 0.5716 | 4 | 7.7 |
| `supermimic --horizon 3 --gamma 0.95 --bptt-decay 0.95` | 2 | 0.8051 | 0.9829 | 0.7800 | 63 | 8.7 |
| `supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 0 | 0.9464 | 0.9685 | 0.9428 | 215 | 7.7 |
| `supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 1 | 0.7825 | 0.8310 | 0.7908 | 0 | 7.9 |
| `supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 2 | 0.9361 | 0.9619 | 0.9389 | 215 | 7.6 |
| `supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0 | 0.8043 | 0.8992 | 0.7718 | 0 | 10.7 |
| `supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 1 | 0.8427 | 0.9968 | 0.7907 | 5 | 10.8 |
| `supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 2 | 0.7975 | 0.9001 | 0.7659 | 0 | 10.3 |

Training time is the trainer's own, CUDA-synchronized: collection, model and policy updates, validation after every round
and checkpoint writes, of the contributed implementation (eager updates; not engineered for speed).

## The settings screen

Each algorithm got the same screen from its starting configuration: the shared horizon (3 or 8) and three settings of its
own, 16 configurations each at seed 5, then per horizon the two best and the starting configuration at seeds 6 and 7. A
run's score is its mean validation F1 over rounds 16-20 (32 validation starts, apart from the 256 evaluation starts). The
screen ran on RTX 2080 Ti GPUs and serves the selection only.

- SuperTrack's own settings: Gaussian noise on the imagined actions (`--st-imag-noise`, 0 or 0.1), the L2 + L1 action
  regularizer (`--st-reg`, 0 or 0.01) and the reach term's weight in its loss (`--st-reach-weight`, 0.20 or 0.40).
- SuperMimic's own settings: the discount (`--gamma`, 0.99 or 0.95), the gradient through each imagined state
  (`--bptt-decay`, 1 or 0.95) and the critic's epochs per update (`--critic-epochs`, 5 or 10).

**SuperTrack, horizon 3**, every configuration at seed 5

| configuration | score | round-20 validation F1 |
|---|---:|---:|
| `--horizon 3 --st-reg 0.01` | 0.8612 | 0.8751 |
| `--horizon 3 --st-imag-noise 0.1` | 0.8423 | 0.7788 |
| `--horizon 3 --st-imag-noise 0.1 --st-reg 0.01` | 0.8136 | 0.8015 |
| `--horizon 3` | 0.8157 | 0.7810 |
| `--horizon 3 --st-reach-weight 0.4` | 0.8042 | 0.8425 |
| `--horizon 3 --st-imag-noise 0.1 --st-reg 0.01 --st-reach-weight 0.4` | 0.7973 | 0.8081 |
| `--horizon 3 --st-imag-noise 0.1 --st-reach-weight 0.4` | 0.7678 | 0.9145 |
| `--horizon 3 --st-reg 0.01 --st-reach-weight 0.4` | 0.6608 | 0.8142 |

The two best and the starting configuration at seeds 5-7; selected: `--horizon 3 --st-reg 0.01`.

| configuration | seed 5 | seed 6 | seed 7 | mean score | mean round-20 F1 |
|---|---:|---:|---:|---:|---:|
| `--horizon 3 --st-reg 0.01` | 0.8612 | 0.8949 | 0.7884 | 0.8482 | 0.8470 |
| `--horizon 3 --st-imag-noise 0.1` | 0.8423 | 0.8812 | 0.7504 | 0.8246 | 0.8212 |
| `--horizon 3` | 0.8157 | 0.8369 | 0.7877 | 0.8134 | 0.8209 |

**SuperTrack, horizon 8**, every configuration at seed 5

| configuration | score | round-20 validation F1 |
|---|---:|---:|
| `--horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 0.8722 | 0.8828 |
| `--horizon 8 --st-imag-noise 0.1 --st-reg 0.01 --st-reach-weight 0.4` | 0.8400 | 0.8541 |
| `--horizon 8 --st-imag-noise 0.1` | 0.8191 | 0.8998 |
| `--horizon 8 --st-imag-noise 0.1 --st-reach-weight 0.4` | 0.8230 | 0.8487 |
| `--horizon 8 --st-reg 0.01` | 0.7976 | 0.9296 |
| `--horizon 8 --st-reach-weight 0.4` | 0.7744 | 0.6039 |
| `--horizon 8 --st-reg 0.01 --st-reach-weight 0.4` | 0.7471 | 0.8561 |
| `--horizon 8` | 0.6713 | 0.6892 |

The two best and the starting configuration at seeds 5-7; selected: `--horizon 8 --st-imag-noise 0.1 --st-reg 0.01`.

| configuration | seed 5 | seed 6 | seed 7 | mean score | mean round-20 F1 |
|---|---:|---:|---:|---:|---:|
| `--horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 0.8722 | 0.8556 | 0.8609 | 0.8629 | 0.8866 |
| `--horizon 8 --st-imag-noise 0.1 --st-reg 0.01 --st-reach-weight 0.4` | 0.8400 | 0.8815 | 0.8806 | 0.8674 | 0.8666 |
| `--horizon 8` | 0.6713 | 0.7273 | 0.7299 | 0.7095 | 0.7290 |

**SuperMimic, horizon 3**, every configuration at seed 5

| configuration | score | round-20 validation F1 |
|---|---:|---:|
| `--horizon 3 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.7785 | 0.7612 |
| `--horizon 3 --gamma 0.95 --bptt-decay 0.95` | 0.7716 | 0.5988 |
| `--horizon 3 --gamma 0.95 --critic-epochs 10` | 0.7590 | 0.8579 |
| `--horizon 3 --gamma 0.95` | 0.6656 | 0.7252 |
| `--horizon 3 --bptt-decay 0.95 --critic-epochs 10` | 0.5921 | 0.7159 |
| `--horizon 3 --critic-epochs 10` | 0.5604 | 0.5726 |
| `--horizon 3` | 0.1292 | 0.1283 |
| `--horizon 3 --bptt-decay 0.95` | 0.0372 | 0.0484 |

The two best and the starting configuration at seeds 5-7; selected: `--horizon 3 --gamma 0.95 --bptt-decay 0.95`.

| configuration | seed 5 | seed 6 | seed 7 | mean score | mean round-20 F1 |
|---|---:|---:|---:|---:|---:|
| `--horizon 3 --gamma 0.95 --bptt-decay 0.95` | 0.7716 | 0.7934 | 0.8808 | 0.8153 | 0.7591 |
| `--horizon 3 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.7785 | 0.7624 | 0.8155 | 0.7855 | 0.7926 |
| `--horizon 3` | 0.1292 | 0.3184 | 0.0768 | 0.1748 | 0.1836 |

Why the discount: a step's reward is at most 1, so the return the critic estimates is at most 100 at γ 0.99 and 20 at
γ 0.95. In the starting configuration (γ 0.99, horizon 3, five critic epochs) the imagined return with the critic's value
at its end (minus the policy loss) went past that bound at every seed: at its highest 5.8e8, 189 and 2,250 at seeds 5, 6
and 7; with the gradient decay too, 3.1e5. With ten critic epochs it stayed at most 29.8, and those runs scored 0.56 and
0.59. With γ 0.95 it stayed at most 12.7 in every run of the screen and of the finals; with γ 0.99 at horizon 8, at most
36.3.

**SuperMimic, horizon 8**, every configuration at seed 5

| configuration | score | round-20 validation F1 |
|---|---:|---:|
| `--horizon 8 --gamma 0.95 --critic-epochs 10` | 0.8675 | 0.8807 |
| `--horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.8277 | 0.8763 |
| `--horizon 8 --gamma 0.95` | 0.7944 | 0.7253 |
| `--horizon 8 --gamma 0.95 --bptt-decay 0.95` | 0.7579 | 0.7549 |
| `--horizon 8 --bptt-decay 0.95` | 0.6830 | 0.7381 |
| `--horizon 8 --bptt-decay 0.95 --critic-epochs 10` | 0.5910 | 0.5702 |
| `--horizon 8` | 0.5709 | 0.5842 |
| `--horizon 8 --critic-epochs 10` | 0.5577 | 0.6745 |

The two best and the starting configuration at seeds 5-7; selected: `--horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10`.

| configuration | seed 5 | seed 6 | seed 7 | mean score | mean round-20 F1 |
|---|---:|---:|---:|---:|---:|
| `--horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.8277 | 0.8987 | 0.7282 | 0.8182 | 0.8867 |
| `--horizon 8 --gamma 0.95 --critic-epochs 10` | 0.8675 | 0.6362 | 0.6436 | 0.7158 | 0.7783 |
| `--horizon 8` | 0.5709 | 0.5406 | 0.7015 | 0.6043 | 0.5976 |

## The world model as the simulator

What each side steps:

- **The world model**, `PianoDynamics` (the task's own, one architecture for both algorithms), steps the policy's observed
  state, 192 numbers: the 52 actuator positions, their last changes and the 88 key depths. One step is a closed-form
  fingertip and contact model and two small residual networks.
- **MuJoCo-Warp** steps the full scene for one 50 ms control step: 10 physics steps at 200 Hz of both hands and all 88
  keys with their contacts, then the observation.

Both start from the same worlds: the SuperTrack (horizon 8) seed-0 policy's states 2 s into the song from the evaluation
starts, each world holding that frame's action. Each number is the median of 5 runs of 200 steps after 20 warm-up steps
and 5 s of calls, on one A100 40GB SXM4 with nothing else running. The four seed-0 models (both algorithms at both
horizons) step within 1 % of each other, so each count has one row.

| worlds | MuJoCo-Warp, the scene | world model, fp32, one CUDA graph | world model, bf16 networks, compiled, one CUDA graph |
|---:|---:|---:|---:|
| 1,024 | 17,379 world-steps/s (58.9 ms a step) | 3.95-3.96 M (228x) | 20.4-20.5 M (1,174-1,179x) |
| 4,096 | 21,819 world-steps/s (187.7 ms a step) | 5.77-5.81 M (265-266x) | 45.9-46.3 M (2,106-2,121x) |
| 16,384 | out of memory on the 40 GB card | | |

At 16,384 worlds MuJoCo-Warp's convex collision step asks for a 13.67 GB buffer for the scene's contact capacity (384
contacts a world), more than the card had left.

The open-loop error beside it: the model rolled out under the recorded actions from every 10th frame of the 256
evaluation episodes, the mean absolute error in fractions of a key's travel. The bf16 copies are within 0.0003 of fp32.

| world model (seed 0) | due keys, 1 / 3 / 10 steps | all 88 keys, 1 / 3 / 10 steps |
|---|---|---|
| SuperTrack, horizon 8 | 0.142 / 0.139 / 0.192 | 0.043 / 0.049 / 0.049 |
| SuperMimic, horizon 8 | 0.210 / 0.205 / 0.236 | 0.043 / 0.051 / 0.055 |
| SuperTrack, horizon 3 | 0.130 / 0.086 / 0.135 | 0.046 / 0.054 / 0.057 |
| SuperMimic, horizon 3 | 0.141 / 0.161 / 0.189 | 0.044 / 0.053 / 0.053 |

So the speed-up is that of a model of what the policy observes, not of a replacement for the physics: after 1 to 10
steps a due key's predicted depth is off by 9-24 % of its travel.
