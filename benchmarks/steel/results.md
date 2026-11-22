# The G1's world models against Steel

A world model steps a whole batch of robots with one network pass. A GPU rigid-body solver computes, for every world and
every substep, the robot's kinematics and dynamics, its collisions and its contact constraints. On the Unitree G1
replaying LAFAN1 tracking, at 4,096 worlds on one A100 40GB SXM4:

- **Throughput.** The MLP world models of `lafan1-supertrack-accurate` and `lafan1-supermimic-accurate` step 14.80 M and
  14.06 M control steps per second. That is 16.9x and 16.1x [Steel](https://www.skild.ai/blogs/steel) (Skild AI's solver in
  Newton) at its fastest setting that passes the accuracy criteria below (0.875 M), and 42.4x and 40.2x MuJoCo Warp in
  Isaac Lab, the task's simulator (0.349 M).
- **Error.** After 32 control steps, their tracked bodies are 19.6 and 22.8 mm from the task's simulator, and Steel's are
  30.2 mm.
- **Where the time goes.** Steel spends 47.5 % of its kernel time on articulation dynamics, 25.9 % on collision and 20.3 %
  on contact rows. The world models spend 80 % on the network.

The world models learned from the task's simulator; Steel did not. Its error is its difference from that simulator.
Skild AI's post reports Steel's speed against PhysX and MuJoCo Warp over 20 Isaac Lab tasks; this page measures one
workload, with Steel's accuracy checked beside its speed.

## One GPU

Control steps per second over all worlds, in millions: the median of three runs, their range in parentheses. The body
error is the tracked bodies' position error after 32 control steps, against the task's simulator.

| row | body error | N = 1,024 | N = 4,096 | N = 16,384 | N = 65,536 |
|---|---|---|---|---|---|
| `lafan1-supertrack-accurate`'s world model (acceleration MLP), bf16 copy with the fused step | 19.6 mm | 5.18 (5.17–5.19) | 14.80 (14.79–14.84) | 23.42 (23.38–23.44) | 27.22 (27.20–27.31) |
| `lafan1-supermimic-accurate`'s world model (semi-implicit MLP), bf16 copy with the fused step | 22.8 mm | 5.21 (5.19–5.28) | 14.06 (14.05–14.12) | 22.90 (22.88–22.94) | 27.05 (26.97–27.11) |
| `lafan1-supertrack-accurate`'s world model, as trained (compiled) | 19.6 mm | 5.15 (5.10–5.23) | 13.32 (13.30–13.32) | 22.23 (22.21–22.24) | 26.77 (26.73–26.79) |
| `lafan1-supermimic-accurate`'s world model, as trained (compiled) | 22.8 mm | 5.06 (5.03–5.15) | 11.98 (11.98–11.99) | 21.11 (21.09–21.11) | 26.25 (26.19–26.29) |
| Steel, fastest passing setting (4 substeps, 1 iteration, its speed options) | 30.2 mm | 0.363 (0.363–0.363) | 0.875 (0.875–0.876) | 1.037 (1.037–1.037) | 1.015 (1.014–1.016) |
| Steel at the task's physics (4 substeps, 12 iterations) | 24.9 mm | 0.076 (0.076–0.076) | 0.191 (0.191–0.191) | 0.282 (0.282–0.282) | 0.282 (0.282–0.283) |
| Steel's G1 preset at a 20 ms step (1 substep, 2 iterations; fails the criteria) | 59.7 mm | 0.440 (0.440–0.440) | 1.055 (1.055–1.056) | 1.495 (1.494–1.495) | 1.391 (1.385–1.392) |
| MuJoCo Warp 3.17 in Newton, on Steel's model at the task's settings | 0.20 mm | 0.170 (0.170–0.170) | 0.357 (0.357–0.357) | 0.473 (0.473–0.473) | 0.514 (0.513–0.514) |
| MuJoCo Warp 3.15 in Isaac Lab 3, the task's simulator | the reference | 0.164 (0.164–0.164) | 0.349 (0.349–0.349) | 0.467 (0.467–0.467) | 0.513 (0.513–0.513) |

The fastest paths against Steel's fastest passing setting / MuJoCo Warp in Isaac Lab, at equal N:

| world model | N = 1,024 | N = 4,096 | N = 16,384 | N = 65,536 |
|---|---|---|---|---|
| `lafan1-supertrack-accurate` | 14.3x / 31.5x | 16.9x / 42.4x | 22.6x / 50.1x | 26.8x / 53.1x |
| `lafan1-supermimic-accurate` | 14.4x / 31.7x | 16.1x / 40.2x | 22.1x / 49.0x | 26.6x / 52.8x |

- **Steel against MuJoCo Warp.** Steel's fastest passing setting runs 2.0-2.5x MuJoCo Warp in Isaac Lab, its error after
  32 steps 30.2 mm. At the task's physics, Steel runs 0.5-0.6x of it, with 24.9 mm.
- **The two MLPs are close.** SuperTrack's is up to 5.3 % faster (4,096 worlds) and SuperMimic's 0.5 % faster at 1,024
  worlds.

## The largest batches

From 131,072 worlds, the batch doubled until a run failed; medians in millions (range):

| world model, fastest path | N = 131,072 | N = 262,144 | N = 524,288 | N = 1,048,576 | N = 2,097,152 |
|---|---|---|---|---|---|
| `lafan1-supertrack-accurate` | 27.95 (27.94–27.99) | 28.16 (28.13–28.17) | 28.16 (28.14–28.19) | 28.09 (28.08–28.11) | 28.00 (27.99–28.02) |
| `lafan1-supermimic-accurate` | 27.59 (27.59–27.64) | 27.90 (27.89–27.91) | 27.95 (27.91–27.99) | 27.91 (27.89–27.92) | 27.83 (27.82–27.84) |

Each row at its own peak:

| row | peak (its N) | against Steel's peak | against MuJoCo Warp's peak |
|---|---|---|---|
| `lafan1-supertrack-accurate` | 28.16 M (524,288) | 27.2x | 54.9x |
| `lafan1-supermimic-accurate` | 27.95 M (524,288) | 27.0x | 54.5x |
| Steel, fastest passing setting | 1.037 M (16,384) | | 2.0x |
| MuJoCo Warp in Isaac Lab | 0.513 M (65,536) | | |
| MuJoCo Warp in Newton | 0.514 M (65,536) | | |

- **The MLPs** ran up to 2,097,152 worlds, ending when the card's memory ran out (23.8 GB at that batch).
- **Steel** ran up to 65,536 worlds, using 11.0 GB of the card. At 131,072 worlds, its run was killed while it built the
  model on the host; each run had 24 GB of host memory.

## Four GPUs

16,384 worlds per GPU. Each GPU runs its own process; the processes time their 1,000 steps together. The total is the sum
over the GPUs, in millions: the median of three runs (range). The ratio is the four-GPU total over four times one GPU.

| row | one GPU | two GPUs | four GPUs | four / (4 × one) |
|---|---|---|---|---|
| `lafan1-supertrack-accurate`, fastest path | 22.56 (22.01–22.75) | 44.70 (44.45–45.24) | 89.24 (89.10–89.34) | 0.989 |
| `lafan1-supermimic-accurate`, fastest path | 22.13 (21.92–22.25) | 43.82 (43.57–44.11) | 87.06 (86.69–87.34) | 0.984 |
| Steel, fastest passing setting | 1.036 (1.035–1.036) | 2.067 (2.067–2.068) | 4.134 (4.133–4.135) | 0.998 |
| MuJoCo Warp 3.17 in Newton | 0.472 (0.472–0.473) | 0.945 (0.944–0.945) | 1.891 (1.890–1.891) | 1.001 |

Every row scales to four GPUs within 1.6 %. On four GPUs the MLPs step 21.6x and 21.1x Steel's fastest passing setting.

## Where the time goes

Kernel time per control step, from Nsight Systems traces (the CUDA graphs traced node by node), at 4,096 worlds and at each
path's largest batch.

Steel's fastest passing setting:

| phase | N = 4,096 | share | N = 65,536 | share |
|---|---|---|---|---|
| collision (narrow phase, bounding boxes, broad phase), once per control step | 1.533 ms | 25.9 % | 8.704 ms | 13.3 % |
| articulation dynamics (kinematics, inverse dynamics, drives, the mass matrix and its factor), 4 substeps | 2.806 ms | 47.5 % | 44.320 ms | 67.9 % |
| contact rows (contact response, rows, bias, limits) | 1.197 ms | 20.3 % | 9.790 ms | 15.0 % |
| the constraint sweep, 1 iteration | 0.262 ms | 4.4 % | 1.842 ms | 2.8 % |
| integration | 0.056 ms | 0.9 % | 0.532 ms | 0.8 % |
| status checks | 0.056 ms | 0.9 % | 0.113 ms | 0.2 % |
| kernels, summed | 5.910 ms | | 65.300 ms | |
| the step's wall time | 4.746 ms | | 64.637 ms | |

The world models' fastest paths, ms per control step:

| world model | N | input features | network | integration | the graph's input copies | kernels, summed | the step's wall time |
|---|---|---|---|---|---|---|---|
| `lafan1-supertrack-accurate` | 4,096 | 0.008 | 0.210 | 0.022 | 0.023 | 0.263 | 0.302 |
| `lafan1-supermimic-accurate` | 4,096 | 0.008 | 0.222 | 0.024 | 0.023 | 0.277 | 0.321 |
| `lafan1-supertrack-accurate` | 2,097,152 | 1.716 | 65.539 | 4.754 | 1.833 | 73.841 | 74.914 |
| `lafan1-supermimic-accurate` | 2,097,152 | 1.721 | 65.675 | 5.028 | 1.832 | 74.257 | 75.519 |

- **Steel's step is mostly articulation dynamics.** It takes 47.5 % of the kernel time at 4,096 worlds and 67.9 % at
  65,536. Collision takes another quarter at 4,096 worlds. In the graph, independent kernels overlap, so at 4,096 worlds
  their times sum to 124.5 % of the step.
- **The world models' step is mostly the network:** 80 % at 4,096 worlds and 88-89 % at 2,097,152. At 4,096 worlds the
  0.3 ms step also has idle gaps between its kernels (their times sum to 86-87 % of it), and the graph's input copies take
  as long as the integration.

## Setup

- **GPUs.** A100 40GB SXM4, driver 565.57.01; one GPU for the one-GPU rows, four for the four-GPU rows, nothing else
  running.
- **Workload.** Replay windows of the G1 (29 DoF) tracking LAFAN1's 40 clips. They were recorded in the task's simulator
  from SuperTrack's MLP policy, without randomization, pushes or action noise. Each world starts from a recorded state and
  takes the recorded joint targets for 32 control steps of 20 ms; the world models take the recorded actions. Then the
  next window starts. Feet go in and out of contact.
- **Timing.**
  - Each run: 20 warm-up control steps and 1 s of warm calls, then 200 timed control steps in segments of 32 (one window
    each). Restarts between windows are not timed.
  - Throughput is control steps per second over all worlds. Each row is three runs in fresh processes.
  - Four-GPU rows: 1,000 timed steps per process.
  - Steel's and MuJoCo Warp in Newton's control step is one CUDA graph; the world models run compiled, replayed as CUDA
    graphs.
- **Steel.** `SolverSteel` of Skild AI's Newton fork ([skild-ai/newton](https://github.com/skild-ai/newton) at
  `baffc2a`, unmodified) with warp-lang 1.19.0.
  - The G1 is imported by Newton's MJCF importer, with convex hulls for its meshes, joint limits on, self-collisions off,
    friction 1.0 and the task's gains, armature and effort limits.
  - Constraint capacities are sized from measured peaks. Every timed run had no dropped constraint row or contact, and
    finite states.
- **Steel's settings.**
  - *At the task's physics:* 4 substeps of 5 ms, Steel's default 12 iterations, collision every substep.
  - *Fastest passing:* the cheapest setting that passes the criteria, searched in the order of Steel's tuning guide. It
    keeps 4 substeps and takes 1 iteration, point friction (`friction_anchor_beta=0`), `contact_gap_gate=0`,
    `parallel_tree`, `reuse_kinematics` and `update_mass_matrix_interval=4`. It collides once per control step and merges
    fixed joints at import.
  - *G1 preset:* Steel's G1 example's 1 substep and 2 iterations, at the task's 20 ms step.
- **Steel's criteria.** A setting counts if all of these hold:
  - no dropped row or contact, and finite states;
  - body errors after 8 and 32 steps at most 1.1x those of Steel at 32 substeps of 0.625 ms and 32 iterations;
  - no more terminated windows than that setting, beyond two standard errors;
  - foot penetration, mean and peak, at most the recording's.
- **The merged inertia.** At this commit, Newton's merge of fixed joints adds a merged link's inertia about its frame
  origin instead of its center of mass. Upstream Newton has since fixed this
  ([newton-physics/newton#5280](https://github.com/newton-physics/newton/pull/5280)). The fastest passing setting merges
  each fixed link into its parent (the torso takes the head, the logo and three sensor frames; the pelvis its contour link
  and a sensor frame; each wrist its rubber hand), and its rows here use the fork as it is. A backport of the upstream fix
  to Skild AI's fork is proposed in [skild-ai/newton#32](https://github.com/skild-ai/newton/pull/32). With the merged
  inertia corrected, the fastest passing setting's error after 32 steps is 29.6 mm, from a separate check on an A100 40GB
  SXM4 that is not part of the timed runs.
- **MuJoCo Warp.**
  - In Isaac Lab 3 (MuJoCo Warp 3.15): the task's simulator, as LAFAN1 training runs it, and the reference of every error.
  - In Newton (MuJoCo Warp 3.17): on the same Newton model as Steel, at the task's solver settings.
- **World models.** Each configuration's three 3 h runs (seeds 7-9). The speed rows time seed 7's world model; the
  errors are the mean over the three seeds' world models as trained.
  - Fastest path: a bf16 copy with the fused integration.
  - Each path is compared with the model on the GPU before it is timed.
- **Errors.** Means over the 8,192 windows of one recording (seed 1). The two other recordings give errors within 1.9 mm
  of these.

## Reproducing

- **This repository** with Isaac Lab 3 (`bash scripts/setup.sh`), the G1 description and LAFAN1's clips
  (`tools/download_data.sh`, which also writes the G1's MJCF, `data/g1_description/g1_29dof.xml`, and the USD that Isaac Lab
  reads, `data/g1_usd`), and the two-G1 task's clip
  (`tools/convert_interhuman.py`), which `speed_pair.py --pair` reads; with `--replay` only its first frame is used.
- **Steel** in a separate Python 3.12 environment, with Skild AI's Newton fork at the commit timed here:

  ```bash
  python3.12 -m venv steel
  steel/bin/pip install "newton[sim,importers] @ git+https://github.com/skild-ai/newton.git@baffc2abb2e101549f8eb797a2adc1645390c566" warp-lang==1.19.0
  ```

  The timed runs used warp-lang 1.19.0's CUDA 12 build (`1.19.0+cu12`), which their driver (565.57.01) required.
- **The world models and the recording's policy** are on [Hugging Face](https://huggingface.co/frankzydou/ndm-gym)
  (non-commercial use, as LAFAN1): the 3 h runs of `lafan1-supertrack-accurate` and `lafan1-supermimic-accurate`,
  seeds 7-9, and the SuperTrack MLP policy that recorded the windows. The speed rows time seed 7's world models; the errors
  average seeds 7-9. The configurations can also be trained again (`ndm-train --config ...`, the README's
  [Recommended settings](../../README.md#-recommended-settings)).

  ```bash
  hf download frankzydou/ndm-gym --include "g1_lafan1_tracking/*-mlp-*/*" --local-dir weights   # the three folders below
  ```

  | file in `g1_lafan1_tracking/` | size (bytes) | sha256 |
  |---|---|---|
  | `supertrack-mlp-accurate/seed7.pt` | 45,607,685 | `e0d7acd6a77afe207b58ab53311d9563224c655faf56efce63e0def27eb9f1fc` |
  | `supertrack-mlp-accurate/seed8.pt` | 45,607,685 | `51d5ba71d4e2c9cb2fbbf2185da695e44b53af8be4f011d3a39ec9c6656aaa42` |
  | `supertrack-mlp-accurate/seed9.pt` | 45,607,685 | `fab954da6512cc9637903fe3154bd0a9a7a51c0696ff3b906831d25f02f94d15` |
  | `supermimic-mlp-accurate/seed7.pt` | 45,752,913 | `f2695063a2a483e41d13d5f41e28153c891c05c06a4c0781a475641943328cca` |
  | `supermimic-mlp-accurate/seed8.pt` | 45,752,913 | `498e9066e441e167d166ffc6afe2ef6a0fa6c931489f5a8a7133383ab6e53dc1` |
  | `supermimic-mlp-accurate/seed9.pt` | 45,752,913 | `b92b527660ef0cec87bb0d093bb16692288928680b25baf4068c451fa9d2bfab` |
  | `supertrack-mlp-recording/seed7.pt` | 71,493,553 | `96debe193e0d527461de6335673d4719499bdb9342a8de0d83770a297e67d80f` |

- **The runs** (`benchmarks/steel/run.sh`; its header lists every variable):

  ```bash
  export STEEL_PY=steel/bin/python MAIN_PY=python PAIR=data/interhuman/interhuman_1566.npz
  W=weights/g1_lafan1_tracking
  export POLICY=$W/supertrack-mlp-recording/seed7.pt CK_ST=$W/supertrack-mlp-accurate/seed7.pt CK_SM=$W/supermimic-mlp-accurate/seed7.pt
  bash benchmarks/steel/run.sh record out                # the recording and its 8,192 windows of 32 steps
  bash benchmarks/steel/run.sh accuracy out              # the errors against the recording, and Steel's criteria
  bash benchmarks/steel/run.sh one out                   # one GPU: every row, three runs each, then the largest batches
  GPUS="0 1 2 3" bash benchmarks/steel/run.sh four out   # one, two and four GPUs, started together
  TRACE_N=65536 bash benchmarks/steel/run.sh traces out  # Nsight Systems traces and the time per phase
  python benchmarks/steel/table.py out                   # the tables above
  ```

- **Not distributed:** the recording and its windows (they come from LAFAN1's motions; `run.sh record` makes them again).
- **One change since the timed runs:** the four-GPU runs used an earlier barrier that could start the processes a few
  milliseconds apart. The barrier here writes a single start. A process times only its own steps, so a skew could only
  shorten the time the processes overlap.
