# Quadrotor trajectory tracking: results on A100 40GB

Contributed by [Zhengming Yu](https://github.com/Yzmblog) ([#7](https://github.com/Frank-ZY-Dou/ndm-gym/pull/7)).

The task and the contributed study on an H200: [README](../../README.md#quadrotor-trajectory-tracking), [results](results.md);
the named configurations on an A100 40GB SXM4: [results](results.md#the-named-configurations-on-an-a100-40gb-sxm4). Here, on
an A100: [the world model as the simulator](#the-world-model-as-the-simulator), [the contributed study's runs again](#the-runs-again-on-an-a100).

## The world model as the simulator

The world models of the contributed study's selected configurations (SuperTrack's, an acceleration MLP of 3 × 256;
SuperMimic's, an increment MLP of 3 × 256; [the runs again](#the-runs-again-on-an-a100)), trained on the ellipse (flight-01a)
with seeds 0-2 for 6,000 iterations on an A100 40GB SXM4, step the drone in place of its physics. A world-step is one 10 ms
control step, the command held; MuJoCo Warp runs its four physics steps of a control step as one CUDA graph. One A100 40GB SXM4
with nothing else on it; every figure the median of five runs of 500 steps after 50 and 2 s of stepping; mean ± SD over the
three seeds.

The fused step: the world model's inference copy (`ndm_gym/worldmodels/quad_infer.py`) with its network in fp32 (TF32
products, as training runs it); the input's features and normalization, and the output's de-normalization and integration,
as two Triton kernels around it (`quad_fused.py`); the state kept as four tensors (position, rotation matrix, velocities) in
two buffers, each step one CUDA graph from one buffer into the other. The network runs either layer by layer, compiled
(`torch.compile`, max-autotune-no-cudagraphs), or as one kernel (`ndm_gym/kernels/mlp_fused.py`): a program carries 16 to
64 worlds through every layer, their activations in registers, and reads only the weights from L2. The copy and the kernel
are compared with the model on the GPU before they run and refused on a mismatch.

```bash
# one checkpoint at one world count (1,024 / 4,096 / 16,384 / 65,536), on one GPU with nothing else running
python -m ndm_gym.wmbench.speed_quadrotor --ckpt logs/quad/ellipse_st_s0/last.pt --num_envs 4096 --warm_seconds 2 --profile \
  --copy fp32 bf16 --fused --compile max-autotune-no-cudagraphs kernel --out speed.json
```

Million world-steps per second, SuperTrack's model / SuperMimic's; MuJoCo Warp and both networks measured in the same run:

| worlds | MuJoCo Warp | the step as the study runs it | fused, the network compiled | fused, the network as one kernel | kernel / MuJoCo Warp |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.937 / 0.938 | 2.88 / 2.95 | 27.3 ± 0.12 / 28.7 ± 0.03 | 32.1 ± 0.10 / 32.0 ± 0.02 | 34.2 / 34.1 |
| 4,096 | 3.26 / 3.27 | 9.02 / 9.30 | 63.9 ± 0.18 / 62.1 ± 0.42 | 74.3 ± 0.23 / 73.7 ± 0.70 | 22.8 / 22.6 |
| 16,384 | 7.99 / 7.99 | 20.1 / 20.4 | 105 ± 0.69 / 107 ± 0.08 | 101 ± 0.03 / 101 ± 0.14 | 12.6 / 12.7 |
| 65,536 | 11.8 / 11.8 | 28.4 / 28.3 | 80.5 ± 0.20 / 80.3 ± 0.13 | 136 ± 0.13 / 135 ± 0.19 | 11.5 / 11.5 |

Where the fused step's time goes, ms per step, each part timed alone as its own CUDA graph, the network compiled / as one
kernel (SuperTrack's model; SuperMimic's within 0.01 ms):

| part | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|
| the state into the network's input (features, normalization) | 0.0038 / 0.0038 | 0.0038 / 0.0037 | 0.0038 / 0.0038 | 0.0055 / 0.0055 |
| the network (3 × 256) | 0.0320 / 0.0259 | 0.0569 / 0.0471 | 0.142 / 0.139 | 0.780 / 0.444 |
| the next state (de-normalization, integration) | 0.0038 / 0.0038 | 0.0052 / 0.0052 | 0.0096 / 0.0092 | 0.0290 / 0.0300 |
| the whole step | 0.0375 / 0.0319 | 0.0640 / 0.0552 | 0.156 / 0.163 | 0.814 / 0.481 |

- **Layer by layer, the activations outgrow the cache.** The compiled network is 85 to 96 % of the step, and from 16,384 to
  65,536 worlds its time grows 5.5 times for four times the worlds: a layer's fp32 activations, 64 MB at 65,536 worlds,
  outgrow the A100's 40 MB L2 cache, so each layer's output goes to memory and back.
- **As one kernel, they stay on the chip.** The network takes 0.444 ms instead of 0.780 at 65,536 worlds, and 17 to 19 % less
  at 1,024 and 4,096 worlds. At 16,384 worlds it is no faster: each program reads the whole network's weights from L2 for its
  16 to 64 worlds, which there costs what the saved round trips gain. The network is still 85 % of the step at 4,096 worlds.
- **The study's step** converts the state to and from quaternions around the network: 0.20 ms of its 0.36 ms at 1,024 worlds
  (SuperTrack's model). The fused step keeps the rotation matrix.
- **The same errors:** with either network, the fused step's open-loop position error against the simulator after 1 / 8 /
  32 / 64 / 100 steps is the model's: 0.015 / 0.21 / 1.35 / 3.93 / 8.25 mm (SuperTrack's model) and 1.84 / 4.11 / 6.18 /
  10.8 / 20.4 mm (SuperMimic's), from the simulator's states under the policy with exploration noise 0.1, 4,096 worlds. The
  kernel's outputs are as far from the network computed in fp64 as TF32 products are (at most 0.0015).
- **bf16:** with the network in bf16 the step runs 34.4 / 81.3 / 156 / 169 M world-steps/s compiled and 44.6 / 110 / 174 /
  223 M as one kernel (SuperTrack's model), but the position error after 100 steps grows to 14.3 ± 2.9 mm (SuperTrack's
  model, from 8.25) and 23.7 ± 0.5 mm (SuperMimic's, from 20.4); the rows above stay in fp32.

## The runs again on an A100

Every run of the study trained and evaluated again on A100 40GB SXM4 GPUs: the same configurations, seeds, flights and
evaluations. The selected configurations, mean ± SD over seeds 0-2; position over 256 perturbed starts of the whole flight:

| task | position, SuperTrack: A100 / H200 | SuperMimic: A100 / H200 | training, SuperTrack: A100 / H200 | SuperMimic: A100 / H200 |
|---|---|---|---|---|
| ellipse | 3.44 ± 0.01 / 3.38 ± 0.03 mm | 6.82 ± 0.28 / 6.55 ± 0.39 mm | 9.6 / 5.6 min | 12.5 / 7.0 min |
| lemniscate | 2.04 ± 0.01 / 2.02 ± 0.02 mm | 4.94 ± 0.30 / 4.96 ± 0.03 mm | 9.3 / 5.8 min | 12.2 / 7.1 min |
| race track | 2.23 ± 0.02 / 2.19 ± 0.02 mm | 5.22 ± 0.12 / 5.08 ± 0.16 mm | 9.3 / 5.9 min | 12.2 / 7.3 min |
| all 18 autonomous flights | 2.55 ± 0.02 / 2.54 ± 0.02 mm | 6.27 ± 0.27 / 6.22 ± 0.34 mm | 9.3 / 5.8 min | 12.3 / 7.2 min |
| ellipse, piloted | 1.36 ± 0.02 / 1.34 ± 0.01 mm | 5.56 ± 0.07 / 5.51 ± 0.26 mm | 9.3 / 5.9 min | 12.2 / 7.3 min |
| lemniscate, piloted | 1.22 ± 0.01 / 1.20 ± 0.00 mm | 4.76 ± 0.16 / 4.71 ± 0.11 mm | 9.2 / 5.8 min | 12.1 / 7.2 min |
| race track, piloted | 3.16 ± 0.02 / 3.08 ± 0.03 mm | 8.03 ± 0.16 / 8.13 ± 0.27 mm | 9.3 / 5.9 min | 12.2 / 7.2 min |

- **The same accuracy:** a success of 100 % on every row, as on the H200, and every position error within 5 % of the H200's
  (SuperTrack +0.4 to +2.6 %, SuperMimic -1.2 to +4.1 %). With the hand task's options SuperTrack again does not learn the
  task, and SuperMimic tracks to 14.5 to 16.1 mm (12.6 to 16.9 on the H200).
- **The time scales with the GPU:** training takes 1.6 to 1.8 times the H200's for both algorithms (10.5-10.7 and 8.0-8.1
  iterations per second). Time to a success of 0.99 within 20 mm: SuperTrack 1.20 / 0.94 / 1.07 min (ellipse / lemniscate /
  race track), SuperMimic 1.52 / 0.99 / 1.16 min; every run reaches it by iteration 1,000, as on the H200.
- **Zero-shot on the pilots' flights:** the policy of the 18 autonomous flights tracks the piloted ellipse, lemniscate and
  race track to 6.55 / 5.77 / 23.06 mm (SuperTrack) and 12.80 / 13.00 / 38.41 mm (SuperMimic), within the seeds' spread of
  the H200's 5.90 / 5.46 / 22.33 and 12.40 / 13.24 / 38.14 mm.

The ellipse's policies on drones with other values than the model they trained on (mean ± SD over seeds 0-2; no rollout
terminates):

| the simulated drone | SuperTrack, A100 | SuperTrack, H200 | SuperMimic, A100 |
|---|---|---|---|
| as trained | 3.44 ± 0.02 mm | 3.38 ± 0.03 mm | 6.82 ± 0.28 mm |
| half the inertia | 4.06 ± 0.11 mm | 3.94 ± 0.16 mm | 7.45 ± 0.51 mm |
| twice the inertia | 9.52 ± 1.74 mm | 7.93 ± 1.77 mm | 17.70 ± 6.76 mm |
| no drag | 28.75 ± 0.47 mm | 28.78 ± 0.87 mm | 47.56 ± 1.50 mm |
| twice the drag | 25.69 ± 0.25 mm | 25.60 ± 0.42 mm | 45.25 ± 1.49 mm |
| 1.2 times the mass, the same largest thrust | 7.95 ± 0.06 mm | 7.78 ± 0.13 mm | 14.47 ± 0.34 mm |
