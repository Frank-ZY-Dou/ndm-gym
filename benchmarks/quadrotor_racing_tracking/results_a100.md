# Quadrotor trajectory tracking: results on A100 40GB

The task and the contributed study on an H200: [README](../../README.md#quadrotor-trajectory-tracking), [results](results.md).

## The world model as the simulator

The world models of the most accurate configurations (`quadrotor-supertrack-accurate`, an acceleration MLP of 3 × 256;
`quadrotor-supermimic-accurate`, an increment MLP of 3 × 256), trained on the ellipse (flight-01a) with seeds 0-2 for 6,000
iterations on an A100 40GB SXM4, step the drone in place of its physics. A world-step is one 10 ms control step, the command
held; MuJoCo Warp runs its four physics steps of a control step as one CUDA graph. One A100 40GB SXM4 with nothing else on
it; every figure the median of five runs of 500 steps after 50 and 2 s of stepping; mean ± SD over the three seeds.

The fused step: the world model's inference copy (`ndm_gym/worldmodels/quad_infer.py`) with its network in fp32 (TF32
products, as training runs it) and compiled (`torch.compile`, max-autotune-no-cudagraphs); the input's features and
normalization, and the output's de-normalization and integration, as two Triton kernels around it (`quad_fused.py`); the
state kept as four tensors (position, rotation matrix, velocities) in two buffers, each step one CUDA graph from one buffer
into the other. The copy is compared with the model on the GPU before it runs and refused on a mismatch.

```bash
# one checkpoint at one world count (1,024 / 4,096 / 16,384 / 65,536), on one GPU with nothing else running
python -m ndm_gym.wmbench.speed_quadrotor --ckpt logs/quad/ellipse_st_s0/last.pt --num_envs 1024 --warm_seconds 2 --profile \
  --copy fp32 bf16 --fused --compile none max-autotune-no-cudagraphs --out speed.json
```

Million world-steps per second, SuperTrack's model / SuperMimic's; MuJoCo Warp measured beside each:

| worlds | MuJoCo Warp | the step as the study runs it (fp32) | the fused step (fp32) | fused / MuJoCo Warp |
|---:|---:|---:|---:|---:|
| 1,024 | 0.936 / 0.938 | 2.88 / 2.95 | 27.3 ± 0.2 / 28.7 ± 0.2 | 29.1 / 30.6 |
| 4,096 | 3.27 / 3.26 | 9.09 / 9.24 | 64.6 ± 0.7 / 63.9 ± 0.5 | 19.8 / 19.6 |
| 16,384 | 7.99 / 7.99 | 20.1 / 20.3 | 104 ± 3.6 / 107 ± 0.2 | 13.0 / 13.4 |
| 65,536 | 11.8 / 11.8 | 28.4 / 28.3 | 80.5 ± 0.1 / 80.4 ± 0.1 | 6.8 / 6.8 |

Where the fused step's time goes, ms per step, each part timed alone as its own CUDA graph (SuperTrack's model; SuperMimic's
within 0.01 ms):

| part | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|
| the state into the network's input (features, normalization) | 0.0038 | 0.0038 | 0.0038 | 0.0054 |
| the network (3 × 256, compiled) | 0.0321 | 0.0566 | 0.144 | 0.780 |
| the next state (de-normalization, integration) | 0.0038 | 0.0054 | 0.0099 | 0.0294 |
| the whole step | 0.0376 | 0.0637 | 0.158 | 0.814 |

- **The network is 85 to 96 % of the step.** From 16,384 to 65,536 worlds its time grows 5.4 times for four times the
  worlds: a layer's fp32 activations, 64 MB at 65,536 worlds, outgrow the A100's 40 MB L2 cache, so each layer's output goes
  to memory and back. A network fused into one kernel per tile of worlds, its activations kept on the chip, would remove
  those round trips; it is not built.
- **The study's step** converts the state to and from quaternions around the network: 0.20 ms of its 0.36 ms at 1,024 worlds
  (SuperTrack's model). The fused step keeps the rotation matrix.
- **The same errors:** the fused step's open-loop position error against the simulator after 1 / 8 / 32 / 64 / 100 steps is
  the model's: 0.015 / 0.21 / 1.35 / 3.93 / 8.25 mm (SuperTrack's model) and 1.84 / 4.11 / 6.18 / 10.8 / 20.4 mm
  (SuperMimic's), from the simulator's states under the policy with exploration noise 0.1, 4,096 worlds.
- **bf16:** with the network in bf16 the step runs 33.4 / 81.5 / 147 / 169 M world-steps/s (SuperTrack's model), but the
  position error after 100 steps grows to 14.3 ± 2.9 mm (SuperTrack's model, from 8.25) and 23.7 ± 0.5 mm (SuperMimic's,
  from 20.4); the rows above stay in fp32.
