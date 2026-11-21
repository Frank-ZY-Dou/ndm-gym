# mbrl-lib's planner on a world model

[mbrl-lib](https://github.com/facebookresearch/mbrl-lib) (at `3f93ccc`, MIT) ships a true-dynamics controller,
`mbrl/diagnostics/control_env.py`: CEM over action sequences, each candidate scored by rolling out the environment's own
MuJoCo model in a pool of CPU processes. It is the one place in mbrl-lib where a simulator sits inside an inner loop. Here its
candidate rollouts run, in the planner only, on MuJoCo Warp or on this library's world model of the same MuJoCo model
(`ndm_gym/worldmodels/mjcf.py`); every plan still executes on CPU MuJoCo, the controller's own environment, and no file of
mbrl-lib's is changed. A planner on a world model of the true simulator is not PETS, whose models are learned ensembles: these
are results of mbrl-lib's controller, not of its algorithms.

| Task | CPU MuJoCo (pool of 16) / MuJoCo Warp | SuperTrack's acceleration model, its speed-up over the two | SuperMimic's increment model, its speed-up over the two | Plans at the bar, each model |
|---|---|---|---|---|
| cart-pole | 1.566 / 0.3144 s | **0.0151 s**, 103.8x / 20.8x | **0.0150 s**, 104.3x / 20.9x | 30 of 30 |
| pusher | 0.950 / 0.2998 s | **0.0303 s**, 31.3x / 9.9x | **0.0286 s**, 33.2x / 10.5x | 28 of 30 |

Planning time per executed step on one A100 80GB SXM4, the median over planner seeds 0-2; the two world models timed in
one session. Plans at the bar: each model's plans with planner seeds 10-19 on three fits, at the bar of the simulators' own
plans (Returns). The two models are level: their steps alone are within 1.2 % of each other at 4,096 worlds, and on the
pusher one model's planning time moves between runs by more than the two differ (Planning time). The planning times are
not engineered; the world models' step is the one after the second round of its engineering (The step alone).

## Setup

- **Tasks.** PETS' cart-pole (`pets_cartpole`; population 500, horizon 30, 5 CEM iterations, a 200-step episode) and pusher
  (`pets_pusher`; 350, 25, 5, 150 steps), at control_env.py's PETS settings. The models step on MuJoCo 4.0.1 as mbrl-lib's own
  MuJoCo 2.3.3 steps them (its contact margins, inverse weights and convex collision).
- **World models.** Two MLPs of 3 x 1024 with the same inputs, each control step one evaluation of the network:
  - SuperTrack's acceleration model predicts the control step's generalized accelerations; the position advances with the
    mean velocity of the step's physics steps.
  - SuperMimic's increment model predicts the step's increments: the tangent from the position to the next position, and
    the velocity's change.
  - The cart-pole's models have LayerNorm. The pusher's have none and have contact inputs: the arm's collision geoms as
    spheres, each with its gap to the object (read beyond the contacts' margin), the nearest point and the relative
    velocity. For each model, LayerNorm or none was chosen by the held-out error at the planner's horizon; both chose alike.
  - Training data: the CPU planner's recordings with planner seeds 0 and 1; seed 2 is held out.
  - A fit stops when its held-out error at the horizon has not fallen 2 % below its best in five evaluations (every 2,000
    updates), and keeps its best state (Fits).
  - The pusher's fits are made again once, after four recordings of the planner on the model's own first fit (one
    planner-in-the-loop round). Three fit seeds per model.
  - The recipe (the stop rule, the pusher's contact inputs and its round) was developed on SuperTrack's acceleration model
    and applied to SuperMimic's increment model unchanged. The mean velocity (`--accel_mean_velocity`) is an option of the
    acceleration model alone; the increment model has no counterpart.
  - The planner runs the world model as a bf16 copy, its step compiled into one CUDA graph. The two models' steps share one
    code path (the inputs, the kinematics and contact gaps, the network) and differ in the last operation: an acceleration
    integrated, an increment applied.
- **Hardware.** A100 80GB SXM4 GPUs on a node of 2 x 28 cores (Xeon Gold 6330): the returns on three GPUs side by side, the
  timings on one GPU, one job at a time. torch 2.14.1 (CUDA 12.6), MuJoCo 4.0.1, MuJoCo Warp 4.0.1 on Warp 1.19.0's CUDA 12
  build: the node's driver runs CUDA 12.7, and Warp 1.19.0's wheel on PyPI needs a CUDA 13 driver. SuperMimic's runs came a
  day after SuperTrack's, on the same node, cards and environment; both models' timings come from that later session, the
  simulators' plans and timings from the first.

## Returns

Planner seeds 10-19, which no recording used; the episode's return. A plan is at the bar when it returns at least the lowest
of the 20 simulator plans less 5 % of their mean's size. Mean (standard error) over the ten plans; SuperTrack's acceleration
model and SuperMimic's increment model, three fits each.

| | CPU MuJoCo | MuJoCo Warp | SuperTrack, fit seed 0 | 1 | 2 | SuperMimic, fit seed 0 | 1 | 2 | bar |
|---|---|---|---|---|---|---|---|---|---|
| cart-pole | 174.7 (0.5) | 174.8 (0.5) | 174.5 (0.4) | 174.3 (0.5) | 174.6 (0.4) | 174.7 (0.4) | 174.8 (0.6) | 174.5 (0.4) | 162.3 |
| plans at the bar | | | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 | |
| pusher | -66.0 (0.6) | -66.4 (0.6) | -65.9 (0.8) | -76.2 (8.3) | -66.4 (0.7) | -70.6 (3.4) | -66.1 (0.5) | -66.7 (0.3) | -72.9 |
| plans at the bar | | | 10 of 10 | 8 of 10 | 10 of 10 | 8 of 10 | 10 of 10 | 10 of 10 | |

Every plan:

| cart-pole, planner seed | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | mean (standard error) | at the bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPU MuJoCo | 175.8 | 174.4 | 175.6 | 176.4 | 175.0 | 174.3 | 173.7 | 175.1 | 171.0 | 175.7 | 174.7 (0.5) | |
| MuJoCo Warp | 176.8 | 174.6 | 175.5 | 176.5 | 174.8 | 174.0 | 173.9 | 175.1 | 171.7 | 175.7 | 174.8 (0.5) | |
| SuperTrack, fit seed 0 | 175.8 | 174.9 | 175.6 | 175.1 | 174.6 | 174.9 | 173.4 | 175.0 | 172.0 | 174.0 | 174.5 (0.4) | 10 of 10 |
| SuperTrack, fit seed 1 | 175.4 | 174.9 | 175.3 | 176.6 | 173.6 | 175.9 | 172.5 | 174.2 | 170.9 | 173.9 | 174.3 (0.5) | 10 of 10 |
| SuperTrack, fit seed 2 | 176.5 | 174.9 | 175.2 | 174.9 | 174.4 | 173.9 | 173.1 | 175.2 | 172.3 | 175.7 | 174.6 (0.4) | 10 of 10 |
| SuperMimic, fit seed 0 | 176.6 | 175.5 | 175.7 | 175.7 | 174.7 | 173.5 | 173.9 | 175.4 | 171.8 | 174.1 | 174.7 (0.4) | 10 of 10 |
| SuperMimic, fit seed 1 | 175.9 | 175.2 | 176.1 | 175.8 | 174.7 | 175.4 | 172.1 | 177.5 | 171.7 | 173.5 | 174.8 (0.6) | 10 of 10 |
| SuperMimic, fit seed 2 | 174.8 | 175.6 | 175.3 | 176.4 | 174.1 | 173.6 | 173.1 | 175.2 | 171.6 | 174.9 | 174.5 (0.4) | 10 of 10 |

| pusher, planner seed | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | mean (standard error) | at the bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPU MuJoCo | -64.1 | -65.4 | -64.1 | -65.3 | -64.2 | -68.2 | -67.2 | -69.3 | -64.9 | -66.9 | -66.0 (0.6) | |
| MuJoCo Warp | -66.7 | -64.2 | -65.5 | -65.8 | -64.0 | -67.4 | -68.7 | -69.6 | -65.9 | -65.7 | -66.4 (0.6) | |
| SuperTrack, fit seed 0 | -64.6 | -64.3 | -66.5 | -65.3 | -64.8 | -70.6 | -68.3 | -62.5 | -68.8 | -63.1 | -65.9 (0.8) | 10 of 10 |
| SuperTrack, fit seed 1 | -65.5 | -65.2 | -149.1 | -67.7 | -63.9 | -65.6 | -65.6 | -67.8 | -69.4 | -82.1 | -76.2 (8.3) | 8 of 10 |
| SuperTrack, fit seed 2 | -66.1 | -63.0 | -64.9 | -70.8 | -65.7 | -65.5 | -68.9 | -68.3 | -64.9 | -66.0 | -66.4 (0.7) | 10 of 10 |
| SuperMimic, fit seed 0 | -65.1 | -64.8 | -65.1 | -67.1 | -65.9 | -92.8 | -65.1 | -88.8 | -67.0 | -64.8 | -70.6 (3.4) | 8 of 10 |
| SuperMimic, fit seed 1 | -67.9 | -64.1 | -65.1 | -65.5 | -66.3 | -63.8 | -67.1 | -68.6 | -67.6 | -64.9 | -66.1 (0.5) | 10 of 10 |
| SuperMimic, fit seed 2 | -65.4 | -67.1 | -65.5 | -67.4 | -66.8 | -67.0 | -67.7 | -67.7 | -65.7 | -66.6 | -66.7 (0.3) | 10 of 10 |

- **The cart-pole's world models plan as the simulators do.** A fit's ten plans less MuJoCo Warp's on the same seeds:
  SuperTrack's -0.32 (0.26), -0.52 (0.36) and -0.24 (0.20), SuperMimic's -0.15 (0.22), -0.06 (0.45) and -0.39 (0.25).
- **Two of each model's thirty pusher plans are below the bar.** SuperTrack's fit seed 1: -149.1 with planner seed 12 and
  -82.1 with seed 19, where the simulators return -64 to -67. SuperMimic's fit seed 0: -92.8 with seed 15 and -88.8 with
  seed 17, where they return -67 to -70. The other fits are level with MuJoCo Warp: their ten plans less its plans,
  SuperTrack's fit seeds 0 and 2 +0.48 (0.94) and -0.05 (0.64), SuperMimic's fit seeds 1 and 2 +0.28 (0.54) and -0.32
  (0.53).
- **Each model's thirty plans:** the cart-pole 174.5 (0.2) and 174.6 (0.3), the pusher -69.5 (2.8) and -67.8 (1.2)
  (SuperTrack's, SuperMimic's). On the same fit and planner seeds, SuperMimic's less SuperTrack's: +0.16 (0.17) and +1.69
  (3.13).
- **The timed plans** (Planning time; fit seed 0, planner seeds 0, 1 and 2, whose first two recorded the fits' data): the
  cart-pole 176.4 / 175.5 / 175.0 and 175.7 / 174.9 / 174.1, the pusher -78.0 / -69.0 / -68.6 and -65.1 / -65.2 / -68.0
  (SuperTrack's, SuperMimic's). SuperTrack's pusher plan with planner seed 0 is below the bar, where the simulators return
  -62.2 and -66.0. The world models' plans repeat: SuperTrack's three, run again in SuperMimic's session, return the same to
  the fourth decimal.
- **MuJoCo Warp's pusher plans vary from run to run.** Its ten plans again on the same seeds differ from the first run by
  0.95 on average and by up to 2.5; CPU MuJoCo's plans repeat to the last digit. The two simulators agree: MuJoCo Warp less
  CPU MuJoCo is +0.15 (0.13) on the cart-pole and -0.40 (0.40) on the pusher.

## Fits

Updates at the fit's stop and at the state it kept, and that state's held-out error: the mean absolute error of the predicted
state on planner seed 2's recording after 1 and 8 steps and after the planner's horizon (30 on the cart-pole, 25 on the
pusher), in units of the spread of the recordings' increments. The pusher's r0 is the first fit and r1 the fit made again
after the round; r1's units differ slightly between the models, as each model's round added its own recordings.

| fit | SuperTrack: updates | held-out error at 1 / 8 / the horizon | SuperMimic: updates | held-out error at 1 / 8 / the horizon |
|---|---|---|---|---|
| cart-pole, fit seed 0 | 48,000, 38,000 | 0.004 / 0.021 / 0.240 | 46,000, 36,000 | 0.005 / 0.036 / 0.212 |
| cart-pole, fit seed 1 | 22,000, 12,000 | 0.005 / 0.037 / 0.379 | 38,000, 28,000 | 0.007 / 0.057 / 0.465 |
| cart-pole, fit seed 2 | 56,000, 46,000 | 0.003 / 0.013 / 0.114 | 22,000, 12,000 | 0.008 / 0.058 / 0.498 |
| pusher, fit seed 0, r0 | 52,000, 42,000 | 0.046 / 0.175 / 0.509 | 36,000, 26,000 | 0.053 / 0.188 / 0.551 |
| pusher, fit seed 0, r1 | 36,000, 26,000 | 0.052 / 0.186 / 0.528 | 28,000, 18,000 | 0.049 / 0.159 / 0.461 |
| pusher, fit seed 1, r0 | 38,000, 28,000 | 0.050 / 0.196 / 0.571 | 40,000, 30,000 | 0.053 / 0.182 / 0.513 |
| pusher, fit seed 1, r1 | 34,000, 24,000 | 0.054 / 0.207 / 0.595 | 36,000, 26,000 | 0.048 / 0.164 / 0.478 |
| pusher, fit seed 2, r0 | 52,000, 42,000 | 0.046 / 0.174 / 0.495 | 20,000, 10,000 | 0.067 / 0.257 / 0.746 |
| pusher, fit seed 2, r1 | 34,000, 24,000 | 0.051 / 0.187 / 0.534 | 20,000, 10,000 | 0.066 / 0.226 / 0.619 |

The round's recordings, the planner on each model's first fits with planner seeds 3 to 6 (fit seeds 0 / 1 / 2): SuperTrack's
-65.4 / -90.2 / -65.1 / -112.6, -66.4 / -66.1 / -65.6 / -142.5 and -66.8 / -64.3 / -169.2 / -64.9, four of twelve below the
bar; SuperMimic's -65.0 / -64.1 / -64.3 / -77.8, -67.9 / -66.1 / -63.6 / -75.5 and -67.4 / -65.2 / -78.4 / -69.3, three of
twelve.

## Planning time

Seconds per executed step: the median over an episode's steps, then the median over planner seeds 0, 1 and 2 (the cart-pole's
200 steps, the pusher's 150). CPU MuJoCo is mbrl-lib's own pool with 16 processes, the fastest of 8, 16, 28 and 56 on both
tasks. The two world models' columns come from one session on one card, their plans alternating by seed. In brackets: the
speed-up over CPU MuJoCo / MuJoCo Warp.

| | CPU MuJoCo | MuJoCo Warp | SuperTrack's model (bf16) | SuperMimic's model (bf16) |
|---|---|---|---|---|
| cart-pole | 1.566 | 0.3144 | **0.0151** (103.8x / 20.8x) | **0.0150** (104.3x / 20.9x) |
| pusher | 0.950 | 0.2998 | **0.0303** (31.3x / 9.9x) | **0.0286** (33.2x / 10.5x) |

| seconds a step, planner seeds 0 / 1 / 2 | CPU MuJoCo | MuJoCo Warp | SuperTrack's model | SuperMimic's model | SuperTrack's, first session |
|---|---|---|---|---|---|
| cart-pole | 1.566 / 1.587 / 1.515 | 0.3144 / 0.3144 / 0.3146 | 0.0151 / 0.0152 / 0.0151 | 0.0150 / 0.0151 / 0.0150 | 0.0151 / 0.0151 / 0.0151 |
| pusher | 0.947 / 0.950 / 0.978 | 0.2998 / 0.3028 / 0.2960 | 0.0303 / 0.0304 / 0.0302 | 0.0317 / 0.0286 / 0.0284 | 0.0305 / 0.0309 / 0.0307 |

A plan scores population x horizon x iterations world steps (75,000 on the cart-pole, 43,750 on the pusher), and the planner's
own work (sampling, sorting, the rewards) is inside each time; mbrl-lib's CEM runs on the CPU. Built once per process and
outside these times: MuJoCo Warp's rollouts in 3.9-4.2 s, the world models' compiled steps in 10.5-16.5 s (SuperTrack's) and
10.8-21.3 s (SuperMimic's). The episode's first step counts among its steps: on CPU MuJoCo it takes 7.7-8.4 s while the pool
starts, and the medians are not moved by it.

- **SuperTrack's first session,** a day before, gave 0.0151 and 0.0307 s (103.8x / 20.8x and 31.0x / 9.8x); its plans in
  the later session return what they returned there to the fourth decimal. CPU MuJoCo's and MuJoCo Warp's columns are from
  the first session.
- **The pusher's planning times move between runs.** SuperMimic's three episodes take 0.0317, 0.0286 and 0.0284 s a
  step; the medians of their 30-step stretches fall from 38.3 to 28.4 ms in the first and are 28.3 to 28.8 ms in the other
  two, but for the second's first stretch (33.5 ms). In SuperTrack's six episodes (both sessions) they are 29.1 to 32.6 ms.
  The two models' steps alone are level (The step alone). A later session that timed each plan's parts found their work
  on the card level as well; the slow plans are in CEM's work on the CPU, its sampling and its update, which nothing of
  the world model enters. Their cause is not known yet.

## The step alone

One control step of N worlds (2 physics steps of the cart-pole, 4 of the pusher) from recorded states, 200 timed steps with
the worlds set back every 30 (the pusher's 25), the median of three runs: milliseconds per step of all N worlds.

| | worlds | CPU MuJoCo, one process | MuJoCo Warp, graph | SuperTrack's model (bf16), graph | SuperMimic's model (bf16), graph |
|---|---|---|---|---|---|
| cart-pole | 1 | 0.032 | 1.647 | 0.030 | 0.028 |
| | 64 | 1.880 | 1.923 | 0.038 | 0.037 |
| | 512 | 14.99 | 2.266 | 0.053 | 0.052 |
| | 4,096 | 119.1 | 2.742 | **0.155** | **0.155** |
| pusher | 1 | 0.057 | 1.245 | 0.081 | 0.080 |
| | 64 | 3.420 | 2.041 | 0.101 | 0.102 |
| | 512 | 27.35 | 2.469 | 0.146 | 0.145 |
| | 4,096 | 217.7 | 3.669 | **0.422** | **0.417** |

At 4,096 worlds SuperTrack's model makes 26.4 M world steps a second on the cart-pole and 9.71 M on the pusher, SuperMimic's
26.4 M and 9.83 M, against MuJoCo Warp's 1.49 M and 1.12 M. The two models' columns come from one session, the other two
from the first.

- **CPU MuJoCo's column is one process on one core**, the worlds stepped one after the other; the node's 56 cores together
  were not measured. In the planner, mbrl-lib's pool of 16 processes makes about 48,000 world steps a second on the cart-pole
  and 46,000 on the pusher (from the planning times above).
- **At one world the world models' steps are no faster than a CPU process's**: level on the cart-pole (0.030 and 0.028
  against 0.032 ms) and slower on the pusher (0.081 and 0.080 against 0.057 ms). From 64 worlds on they are the fastest of
  the three.
- **Why they are faster in a batch.** MuJoCo Warp runs each control step's 2 or 4 physics steps, each with its collision
  detection and constraint solver; a world model replaces the control step with one evaluation of its network for all
  worlds, whose cost grows slowly with the worlds until the GPU is full. At one world that evaluation's latency is all that
  is left, and a CPU core steps these small models as fast or faster.
- **The two models' steps are level:** within 4 % of each other at every count of worlds, within 1.2 % at 4,096.
  SuperTrack's step in the first session lies within 0.4 % of its column.
- **MuJoCo Warp's column was measured in a process of its own**, after every other job; the other columns in processes
  that do not load Warp. On Warp's CUDA 12 build the capture of MuJoCo Warp's step failed in a few processes in
  which a capture, the world model's compiled step and another capture followed each other (2 of 17 in test runs on RTX 2080
  Ti cards, and once on an A100 80GB), and never in a process of its own.
- **Not final.** The world models' step here is the one after the second round of its engineering. A screen for a third
  round (the kinematics on one tensor per scalar, a third fewer kernels) found no shorter step on an RTX 2080 Ti; it was not
  repeated on the A100.

## Other environments

Hopper, the inverted pendulum and the half-cheetah are in development: their world models do not yet plan at the
simulators' level on the planner's seeds.

## Reproducing

`benchmarks/mbrl_lib/scripts/setup_env.sh <this checkout>` builds two environments with uv from the pins beside it: (a)
mbrl-lib's own (Python 3.10, MuJoCo 2.3.3), which makes the local-coordinate copies of Hopper and Walker2d
(`tools/convert_mbrl_models.py`), and (b) the one every command below runs in (Python 3.12, gymnasium 0.26.3, mujoco and
mujoco-warp 4.0.1, warp-lang 1.19.0, torch 2.14.1), which imports mbrl-lib's checkout at `3f93ccc`, unmodified, and this
library. On a driver older than CUDA 13, Warp 1.19.0's CUDA 12 build from Warp's releases
(`warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl`) replaces PyPI's wheel, as for the runs here. The cart-pole below;
the pusher: `pets_pusher`, `--substeps 4`, the fit's options `--no_ln --contacts --contact_objects object --contact_pair_margin`
beside the model's `--wm_param`, `--eval_k 1 8 25`, `--reset_every 25`, and one planner-in-the-loop round
(four recordings with planner seeds 3-6 on each first fit, `collect_mbrl --sim ndm --ndm <fit> --dtype bf16 --fused`, added to
its training recordings for a fit made again from scratch).

```bash
OWN="--setting pets --own_mujoco"   # PETS' settings; the model as mbrl-lib's own MuJoCo steps it
# the CPU planner's recordings: its candidate rollouts, the executed trajectory, random and noisy episodes (seed 2 held out)
for s in 0 1 2; do python -m ndm_gym.wmbench.collect_mbrl --env pets_cartpole $OWN --seed $s --num_processes 28 --out data/cp_s$s; done
# SuperTrack's acceleration model, fit seeds 0-2
python -m ndm_gym.wmbench.fit_mjcf --model data/cp_s0_model.mjb --substeps 2 --wm_param accel --accel_mean_velocity \
  --train data/cp_s{0,1}_{rollout,executed,random,noisy}.npz --test held=data/cp_s2_rollout.npz exec=data/cp_s2_executed.npz \
  --steps 200000 --eval_every 2000 --until_flat 0.02 5 --batch 512 --window 8 --amp none --compile --cut 300 --cut_scale_floor \
  --eval_k 1 8 30 --seed 0 --out fits/cp_f0.csv --save fits/cp_f0.pt
# SuperMimic's increment model: the same with --wm_param increment in place of --wm_param accel --accel_mean_velocity
# plans: CPU MuJoCo's pool, MuJoCo Warp, the world model (planner seeds 10-19 for the returns; 0-2 with --num_steps 200 for the times)
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole $OWN --sim cpu --num_processes 16 --seed 10 --out plans/cp_cpu_s10.json
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole $OWN --device cuda:0 --sim warp --seed 10 --out plans/cp_warp_s10.json
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole $OWN --device cuda:0 --sim ndm --ndm fits/cp_f0.pt --dtype bf16 --fused \
  --seed 10 --out plans/cp_ndm_s10.json
# the step alone: CPU MuJoCo and the world model in one process, then MuJoCo Warp in a process of its own
python -m ndm_gym.wmbench.speed_mbrl --meta data/cp_s2.json --states data/cp_s2_rollout.npz --nworld 1 64 512 4096 --reset_every 30 \
  --what cpu ndm --dtype bf16 --fused --ckpt fits/cp_f0.pt --warm_seconds 1 --out speed/cp.csv
python -m ndm_gym.wmbench.speed_mbrl --meta data/cp_s2.json --states data/cp_s2_rollout.npz --nworld 1 64 512 4096 --reset_every 30 \
  --what warp --warm_seconds 1 --out speed/cp.csv
```

Each plan's json holds its return and its planning time per executed step (`plan_s_median`). `python -m
ndm_gym.wmbench.check_mbrl_rewards` checks the batched rewards and terminations the GPU planners score with against the
environments' own step, and `benchmarks/mbrl_lib/compare_sim.py` steps an environment from identical states in mbrl-lib's own
environment, in environment (b) and on MuJoCo Warp, and compares the three.
