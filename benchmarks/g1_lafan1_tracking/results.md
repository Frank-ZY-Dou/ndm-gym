# Results in detail

The [README](../../README.md#-results) reports the main A100 results. Tables here come from run files: `tools/ablations.py`
generates the H100 tables, `tools/rf_checks.py` the RigidFormer checks, `tools/grad_diag_report.py` the policy-gradient
diagnostic and `nablamimic/nm_table.py` NablaMimic's; the other tables come from the same run files.

## How the numbers were obtained

Both algorithms share the simulator, data, networks, schedule and evaluation, and get the same tuning budget: 12 runs
with seed 0 after their starting configurations, 3 h each. Among the seed-0 runs, 14 for SuperTrack and 13 for
SuperMimic (counting the starting configurations and SuperTrack's round-1 run without task penalties), those within
0.01 of the highest evaluation return at 3 h are ranked by fewer failed clips, then by return (three SuperMimic runs
tied at 0 failed clips). Selected configurations run on seeds 1-3, 3 h on one A100 40GB per run (8 physical cores,
24 GB memory cap). The RigidFormer runs take
these configurations with RigidFormer (RF-S, world-model learning rate 4e-4, chosen offline on the world-model
benchmark's test sets), compared with the MLP at 3 h and at equal iterations (12,670 and 13,540, the MLP finals' mean
counts at 3 h); the H100 supplement and the RigidFormer checks use them. SuperTrack-RF's and SuperMimic-RF's final
configurations, tuned online as the MLP was: [below](#rigidformers-final-configurations). The standard evaluation plays
each of the 40 LAFAN1 clips once from its first frame, without noise or
randomization. Differences come with Welch 95% intervals over seeds. Every section below except the training speed and the policy-gradient
diagnostic fixed its rule before its runs. Both algorithms are re-implementations on this pipeline, not
the authors' code; the configuration SuperMimic's tuning rounds selected differs from the paper's (action noise 0.3 instead
of 0.1, imagination kicks 0.3, reward kernels ×0.7). Its most accurate configuration takes the paper's action noise
0.1 and SuperTrack's offsets about the reference pose; how the named configurations were chosen after the tuning rounds:
[below](#the-most-accurate-configurations).

## The most accurate configurations

The named most accurate configurations with the MLP world model: 3 h of training each on one A100 40GB SXM4, seeds
7-9, scored as the main comparison (the last checkpoint's three standard evaluations and its pushes of 1 and 2 m/s);
mean ± sd over the seeds. No run dropped late (an evaluation during training below 0.60 after its first at or above
0.82).

| MLP world model | `lafan1-supertrack-accurate` | `lafan1-supermimic-accurate` | SuperTrack - SuperMimic |
|---|---|---|---|
| return | 0.898 ± 0.002 | 0.915 ± 0.005 | **-0.017 [-0.028, -0.006]** |
| failed clips | 0.00 ± 0.00 | 0.00 ± 0.00 | +0.00 [0.00, 0.00] |
| MPJPE-local (mm) | 10.8 ± 0.0 | 16.7 ± 0.1 | **-5.9 [-6.2, -5.6]** |
| joint angle (deg) | 2.07 ± 0.03 | 2.96 ± 0.02 | **-0.89 [-0.95, -0.84]** |
| failed at 1 m/s pushes | 2.0 ± 0.0 | 2.3 ± 0.6 | -0.3 [-1.8, 1.1] |
| failed at 2 m/s pushes | 15.7 ± 3.5 | 11.7 ± 0.6 | +4.0 [-4.4, 12.4] |
| iterations in 3 h | 63,200, 63,393, 63,324 | 66,837, 67,208, 67,058 | |

The intervals are Welch 95% intervals of SuperTrack's row minus SuperMimic's, from each seed's scores; bold where they
exclude 0.

- **How they were chosen.** Each starts from its algorithm's configuration from the tuning rounds, changed in rounds for
  the time to return 0.82, each algorithm with its own options ([training speed](#training-speed)). These configurations
  then trained 3 h
  with the learning rate's cosine over the 3 h (`--lr_decay_hours 3`), which scored higher than the schedule as published
  in a 3 h pilot run of each (seed 0). At 3 h only the cosine, SuperMimic's action noise and the lean options on SuperTrack's
  MLP were compared; the changes for the time to 0.82 were not tuned for accuracy after 3 h.
- **What was tried for which.** In the rounds for the time to 0.82 the simulator's action noise was varied for SuperMimic
  only (SuperTrack's runs kept the default 0.1), and the imagination kicks in five values for SuperMimic and one for
  SuperTrack.
- **The action noise.** SuperMimic's MLP configuration runs with 0.1, the default that SuperTrack's runs keep, in place of
  its fast configuration's 0.3: with 0.3 its three 3 h runs on these seeds dropped late (return 0.298 ± 0.054 at 3 h), and in a
  development screen of the noise at 3 h (four seeds per arm) all four runs with 0.3 dropped late (at least 20 failures in
  two consecutive probe sweeps after iteration 24,000) and none with 0.1; SuperTrack's configuration dropped at neither 0.1
  nor 0.035.
- **The lean options.** Both MLP configurations carry the same lean options (`--eprobe_lean --imag_hoist --wm_presample
  --muon_batched --gc_freeze`). SuperTrack's MLP runs without them, with the same
  cosine, scored a return of 0.895 ± 0.001 and an MPJPE-local of 11.0 ± 0.2 mm (seeds 7-9).
- **The budget is 3 h of wall clock:** about 63,310 iterations for SuperTrack, 67,030 for SuperMimic. At equal iterations
  the order is the same: SuperTrack 0.898 ± 0.003 and 10.8 ± 0.0 mm at its last checkpoints, SuperMimic 0.913 ± 0.001
  and 16.7 ± 0.2 mm at 64,000 iterations (one evaluation per run).

## The 3 h configurations at one budget

These were the named 3 h configurations until [the most accurate configurations](#the-most-accurate-configurations)
replaced them. SuperTrack's and SuperMimic's 3 h configurations, two arms each, on seeds 7, 8 and 9: 3 h of training each with the speed
options (`SPEED` of `ndm_gym/configs.py`; SuperMimic also `--critic_cudagraphs`), one A100 40GB SXM4 per run, all on one
code version. Each algorithm's second arm takes a setting of the other algorithm's configuration that it can take. Both of
SuperMimic's arms use the paper's action noise 0.1: with 0.3 and the speed options its return falls late
([below](#rigidformers-final-configurations)). Scored as the main comparison: the last checkpoint's three standard
evaluations and its pushes of 1 and 2 m/s; mean ± sd over the seeds.

| algorithm | arm | return | failed clips | MPJPE-local (mm) | joint angle (deg) | failed at 1 m/s pushes | failed at 2 m/s pushes | iterations in 3 h | late drop, seeds 7 / 8 / 9 |
|---|---|---|---|---|---|---|---|---|---|
| SuperTrack | the tuning rounds' configuration | 0.857 ± 0.004 | 0.00 ± 0.00 | 15.2 ± 1.0 | 2.02 ± 0.03 | 4.0 ± 2.6 | 19.3 ± 2.1 | 36077, 36226, 36338 | no / no / no |
| SuperTrack | with SuperMimic's imagination kicks, `--imag_start_perturb 0.3` | 0.842 ± 0.007 | 0.56 ± 0.38 | 16.1 ± 0.5 | 2.12 ± 0.03 | 2.3 ± 1.2 | 14.3 ± 5.8 | 35982, 36176, 36200 | no / no / no |
| SuperMimic | the tuning rounds' configuration with action noise 0.1 | 0.863 ± 0.002 | 0.44 ± 0.38 | 24.3 ± 0.9 | 4.05 ± 0.08 | 3.7 ± 1.5 | 10.0 ± 1.0 | 33818, 33669, 33874 | no / no / no |
| SuperMimic | the same with SuperTrack's offsets about the reference pose, `--sm_residual` | 0.872 ± 0.003 | 0.00 ± 0.00 | 21.2 ± 0.3 | 3.40 ± 0.10 | 3.0 ± 1.0 | 12.0 ± 0.0 | 33778, 33687, 33827 | no / no / no |

Each algorithm's row is its arm with the higher mean return or, when the other arm's is within 0.01 of it, the one with
the lower MPJPE-local: SuperTrack's configuration as it was; SuperMimic's with action noise 0.1 and `--sm_residual` (0.872
against 0.863, within 0.01, and 21.2 against 24.3 mm).

| | SuperTrack | SuperMimic | SuperTrack - SuperMimic |
|---|---|---|---|
| return | 0.857 ± 0.004 | 0.872 ± 0.003 | **-0.016 [-0.023, -0.008]** |
| failed clips | 0.00 ± 0.00 | 0.00 ± 0.00 | +0.00 [0.00, 0.00] |
| MPJPE-local (mm) | 15.2 ± 1.0 | 21.2 ± 0.3 | **-6.0 [-8.2, -3.9]** |
| joint angle (deg) | 2.02 ± 0.03 | 3.40 ± 0.10 | **-1.38 [-1.61, -1.15]** |
| failed at 1 m/s pushes | 4.0 ± 2.6 | 3.0 ± 1.0 | +1.0 [-4.7, 6.7] |
| failed at 2 m/s pushes | 19.3 ± 2.1 | 12.0 ± 0.0 | **+7.3 [2.2, 12.5]** |

Both rows are the better of two arms on the same seeds 7-9 (the same selection for both algorithms), not confirmed on fresh
seeds. The intervals are Welch 95% intervals of SuperTrack's row minus SuperMimic's; bold where they exclude 0.

SuperMimic maximizes the return directly and SuperTrack minimizes the pose errors directly; each leads on the measure it
optimizes. No run's return fell late (an evaluation during training below 0.60 after its first at or above 0.82).
SuperTrack's configuration scored 0.861 ± 0.003 and 14.5 ± 0.2 mm in its earlier 3 h runs on the same seeds
([below](#rigidformers-final-configurations)); here, on one code version with SuperMimic's runs, 0.857 ± 0.004 and
15.2 ± 1.0 mm.

## Training speed

Minutes to return 0.82 with MLP world models, BlinkMimic's recipe and systems options, and each algorithm's fastest
configuration ([commands](../../README.md#training-speed)), one run at a time on A100 40GB SXM4 GPUs. Every run trained
3,000 iterations with a checkpoint every 100, evaluated after training (`ndm-eval --group 18 --repeats 4 --graph`); time
starts at iteration 50, after compilation and graph capture, and ends at the first checkpoint whose four evaluations on all
40 clips average at least 0.82. Means and standard deviations use unrounded times.

One GPU, seeds 0-2: the two configurations and SuperTrack's without the lean options in one block, the arms' order rotated
with the seed.

| configuration | minutes to 0.82, per seed | mean ± std | iterations to 0.82 | MPJPE-local there (mm) | ms per iteration, 200-1000 / 1000-1500 |
|---|---|---|---|---|---|
| `lafan1-supertrack-fast` | 4.35 / 4.03 / 4.03 | 4.13 ± 0.18 | 1,800, 1,700, 1,700 | 22.4 ± 0.7 | 124.4 / 170.7 |
| the same without the lean options | 3.93 / 3.99 / 3.96 | 3.96 ± 0.03 | 1,600, 1,600, 1,600 | 23.2 ± 0.4 | 127.3 / 188.7 |
| `lafan1-supermimic-fast` | 3.83 / 3.22 / 3.52 | 3.52 ± 0.30 | 1,700, 1,500, 1,600 | 27.0 ± 1.0 | 116.3 / 164.9 |

Four GPUs, seeds 3-5: the two configurations in one session with nothing else running (with the same-session runs without
`--fk_fused` below).

| configuration | minutes to 0.82, per seed | mean ± std | iterations to 0.82 | return there | MPJPE-local there (mm) | ms per iteration, 200-1000 / 1000-1500 |
|---|---|---|---|---|---|---|
| `lafan1-supertrack-fast-4gpu` | 1.41 / 1.49 / 1.50 | 1.47 ± 0.05 | 1700, 1800, 1800 | 0.827 ± 0.001 | 21.9 ± 0.5 | 51.0 / 51.2 |
| `lafan1-supermimic-fast-4gpu` | 1.42 / 1.34 / 1.51 | 1.42 ± 0.09 | 1700, 1600, 1800 | 0.826 ± 0.002 | 26.9 ± 0.4 | 51.6 / 51.7 |

The return and MPJPE-local are those of the checkpoint where the time is read (its four evaluations' means). There
SuperTrack's MPJPE-local is lower, on one GPU and on four (22.4 against 27.0 mm, 21.9 against 26.9 mm); the times are level
within the seeds' spread (one GPU: SuperTrack's minus SuperMimic's +0.61 [-0.01, 1.24] min, Welch 95 %). In an earlier
one-GPU session and on four GPUs the global MPJPE at that point is above 1 m for both, the roots having drifted on the long
clips (SuperTrack 1.25 and 1.23 m, SuperMimic 1.11 and 1.06 m).

**The rule for an option that keeps the computation:** it is adopted when every run with it is faster per iteration than
every run without, by more than the two arms' standard deviations together, over iterations 200-1000 and over 1000-1500.
The lean options meet it on SuperTrack's one-GPU path (by 2.9 and 18.0 ms, against 0.2 and 6.2 ms), so
`lafan1-supertrack-fast` takes them, as `lafan1-supermimic-fast` does. Its time to 0.82 is level with and without them
(+0.18 [-0.27, 0.62] min, Welch 95 %): the runs with them needed 100-200 more iterations on two seeds.

The four-GPU configurations compute the forward kinematics as two Triton kernels with a hand-written backward pass
(`--fk_fused`; equal to the default up to float rounding). In their session it meets the rule for both algorithms, in both
windows:

| four GPUs, seeds 3-5 | without `--fk_fused` | with it |
|---|---|---|
| SuperTrack: minutes to 0.82; ms per iteration, 200-1000 / 1000-1500 | 1.52 ± 0.08; 51.8 / 52.1 | 1.47 ± 0.05; 51.0 / 51.2 |
| SuperMimic: minutes to 0.82; ms per iteration, 200-1000 / 1000-1500 | 1.49 ± 0.05; 52.0 / 52.2 | 1.42 ± 0.09; 51.6 / 51.7 |

Why it helps there: on four GPUs the iteration is the policy learner's step, whose imagined steps and losses run the
kinematics, and the seeds' times per iteration agree within 0.3 ms, so the 0.4-0.9 ms the fused kernels save shows in every
run. On one GPU it meets the rule over iterations 200-1000 and not over 1000-1500, where the probe sweeps from iteration
1,000 spread the baseline seeds' times by 9.3 ms (SuperTrack) and 5.0 ms (SuperMimic), so the one-GPU configurations do
without it (the one-GPU arms, seeds 3-5, four runs sharing the node):

| one GPU, ms per iteration, 200-1000 / 1000-1500 | without `--fk_fused` | with it | faster by, against the two arms' standard deviations together |
|---|---|---|---|
| SuperTrack | 127.6 / 191.1 | 123.2 / 186.4 | 4.42 against 0.28; 4.71 against 9.44 |
| SuperMimic | 116.2 / 162.1 | 115.1 / 161.1 | 1.10 against 0.15; 1.07 against 4.76 |

Time per iteration by phase on one GPU, ms, from each arm's timer run in the one-GPU block (seed 0; the rest: the
iteration less its three phases, the probe sweeps running from iteration 1000):

| configuration | iterations | collection (2.5 steps) | world-model updates | policy update with the critic | the rest |
|---|---|---|---|---|---|
| `lafan1-supertrack-fast` | 200-1000 | 42.2 | 32.9 | 49.0 | 0.4 |
| | 1000-1500 | 42.2 | 32.8 | 49.0 | 46.8 |
| the same without the lean options | 200-1000 | 42.4 | 35.0 | 50.0 | 0.0 |
| | 1000-1500 | 42.2 | 35.0 | 50.0 | 61.5 |
| `lafan1-supermimic-fast` | 200-1000 | 45.8 | 24.0 | 47.0 | -0.5 |
| | 1000-1500 | 45.6 | 24.0 | 46.9 | 48.4 |

On one GPU an iteration runs its phases one after another, and the host adds under 1 ms. Both configurations encode each
imagined step's goal once per rollout instead of inside every step (`--pi_encode_once`) and run with the lean options
(`--eprobe_lean --imag_hoist --wm_presample --muon_batched --gc_freeze`). On SuperTrack's path they take 2 ms off the
world-model updates and 1 ms off the policy update, and each of its four probe sweeps over iterations 1000-1500 takes 9.2
instead of 11.2 s, `--eprobe_lean` computing only what a sweep reads. The world-model updates take 24.0 ms of SuperMimic's
iteration and 32.9 ms of SuperTrack's.

In an earlier session both algorithms were timed with and without the lean options under the same conditions (seeds 0-2);
with them SuperMimic's policy update took 46.0 ms instead of 55.0:

| one GPU, minutes to 0.82; ms per iteration, 200-1000 / 1000-1500 | SuperTrack | SuperMimic |
|---|---|---|
| without the lean options (SuperMimic also without `--pi_encode_once`) | 4.30 ± 0.01; 127.4 / 188.7 | 4.22 ± 0.42; 127.1 / 191.9 |
| with them | 4.04 ± 0.30; 124.5 / 170.8 | 3.43 ± 0.18; 116.3 / 164.9 |

SuperTrack's two times are level within the seeds' spread; its configuration took the lean options later, in the
one-GPU block above, by its time per iteration.

With the lean options SuperTrack's iteration is shorter in both windows: its world-model updates take 32.4 ms instead of
35.0, its policy update 49.1 instead of 50.0. SuperMimic's arm also added `--pi_encode_once`, which SuperTrack's
configuration already had.

The one-GPU configurations in four sessions on the same GPU type, each time seeds 0-2, one run at a time (minutes to 0.82):

| session | SuperTrack, without the lean options | SuperTrack, with them (`lafan1-supertrack-fast`) | `lafan1-supermimic-fast` |
|---|---|---|---|
| the one-GPU block above | 3.96 ± 0.03 | 4.13 ± 0.18 | 3.52 ± 0.30 |
| the session of the lean options (above) | 4.30 ± 0.01 | 4.04 ± 0.30 | 3.43 ± 0.18 |
| a session of single changes, as its baselines | 4.18 ± 0.20 | - | 3.32 ± 0.17 |
| the session of the earlier published rows | 4.19 ± 0.18 | - | 3.53 ± 0.30 |

The four-GPU path by device, from the layout's timer (`LAYOUT_PROFILE=1`: CUDA events on each device's stream around the
iteration's steps, every 20th iteration, one `[prof-layout]` line each); seed 0, two runs of each configuration, the lines
of iterations 200-1000 averaged, ms per timed iteration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 LAYOUT_PROFILE=1 ndm-train --config lafan1-supertrack-fast-4gpu --seed 0 --iterations 1000
CUDA_VISIBLE_DEVICES=0,1,2,3 LAYOUT_PROFILE=1 ndm-train --config lafan1-supermimic-fast-4gpu --seed 0 --iterations 1000
```

| configuration | device | collection | world-model updates | policy update | exchange (incl. waiting) | idle | span |
|---|---|---|---|---|---|---|---|
| SuperTrack, four GPUs | GPU 0, actor | 33.7 | | | 15.4 | 0.0 | 49.0 |
| | GPU 1, world model | | 32.4 | | 16.4 | 0.1 | 48.9 |
| | GPU 2, policy | | | 48.0 | 0.9 | 0.1 | 49.0 |
| SuperMimic, four GPUs | GPU 0, actor | 36.4 | | | 10.3 | 0.0 | 46.7 |
| | GPU 1, world model | | 26.3 | | 20.3 | 0.1 | 46.6 |
| | GPU 2, policy | | | 45.7 | 0.8 | 0.1 | 46.6 |

Without the fused kinematics the same path read 34.9 ms for SuperTrack's world-model updates and 49.5 ms for its policy update,
46.5 ms for SuperMimic's policy update; the collections and SuperMimic's world-model updates moved by at most 0.2 ms, and the
exchanges are mostly the wait for the policy learner.

The layout runs the actor, the world-model learner and the policy learner at the same time on different GPUs, each on the
others' results of the previous iteration, so an iteration takes about its slowest step: the policy update (45.7-48.0 ms).
The actor and the world-model learner finish early and wait in the exchange for the policy learner's results; the probe sweeps run
on a fourth GPU, every 250 iterations.

## Multi-GPU layouts and the learner's restructuring, timed

BlinkMimic's recipe (`--algo blinkmimic`, the algorithm and hyperparameters XMimic runs) with the options taken from
XMimic: the iteration's steps side by side on several GPUs (`--layout`), the probe sweeps on a GPU of their own
(`--eprobe_gpu`) and the learner's restructuring ([the options](../../docs/algorithms/xmimic.md)). One node of A100 40GB
SXM4 GPUs with nothing else running: training on four of its GPUs, the evaluators on the others; XMimic's own code ran on
the same node (below). Every run's policy updated in every window read (the log's policy loss is nonzero).

**Time per iteration** (seed 0, no evaluation; milliseconds per iteration from the training log's wall clock). The learner's
options are added one group at a time, each row on top of the one above it:

| options | GPUs | ms/it, iterations 200-1000 | 1000-1500 (four probe sweeps) |
|---|---|---|---|
| the recipe | 1 | 160.8 | 227.3 |
| `--pi_encode_once` | 1 | 150.2 (-6.6 %) | |
| and `--imag_hoist` | 1 | 150.0 | |
| and `--pi_deferred_grads --wm_deferred_grads` | 1 | 163.8 | |
| and `--muon_batched` | 1 | 163.4 | |
| and `--wm_presample` | 1 | 161.3 | |
| and `--gc_freeze`: the learner's set | 1 | 161.2 | 227.9 |
| the set and `--njmax 256` | 1 | 159.2 (-1.2 % from the set) | |
| the recipe, the probe worker on GPU 1 | 2 | 160.6 | 180.1 |
| the set, the probe worker on GPU 1 | 2 | 161.1 | 180.5 |
| the recipe again, last | 1 | 160.9 | 227.4 |

The two runs of the recipe differ by 0.1 ms. `--pi_encode_once` alone takes 6.6 % off; the deferred gradients add 13.8 ms
back, and the learner's whole set ends within 0.3 % of the recipe. The probe worker on a second GPU takes 47 ms per iteration
off iterations 1000-1500 (227.3 to 180.1).

**Time to return 0.82** (seeds 0-2; checkpoints every 100 iterations, evaluated during training with `ndm-eval --watch
--group 18 --repeats 4 --graph`; minutes of training from iteration 50 to the first checkpoint whose four evaluations average
at least 0.82; GPU-minutes are those minutes times the GPUs the run trains on; the layout runs train with `--gc_freeze`):

| arm | GPUs | ms/it 200-1000 | ms/it 1000-1500 | iterations to 0.82 | minutes to 0.82 | GPU-minutes |
|---|---|---|---|---|---|---|
| the recipe | 1 | 160.81 ± 0.04 | 227.12 ± 0.81 | 1400, 1500, 1600 | 4.12, 4.45, 4.85 (4.48 ± 0.37) | 4.48 ± 0.37 |
| the learner's set | 1 | 161.11 ± 0.07 | 227.66 ± 0.77 | 1500, 1600, 1500 | 4.46, 4.89, 4.45 (4.60 ± 0.25) | 4.60 ± 0.25 |
| `--layout AL` | 2 | 86.84 ± 0.04 | 141.82 ± 1.61 | 1500, 1600, 1800 | 2.56, 2.81, 3.31 (2.89 ± 0.38) | 5.79 ± 0.77 |
| `--layout AL --eprobe_gpu 2` | 3 | 86.86 ± 0.07 | 123.30 ± 1.21 | 1600, 1500, 1500 | 2.64, 2.39, 2.41 (2.48 ± 0.14) | 7.44 ± 0.41 |
| `--layout AWL` | 3 | 87.08 ± 0.01 | 136.38 ± 1.74 | 1500, 1600, 1600 | 2.52, 2.76, 2.80 (2.69 ± 0.15) | 8.07 ± 0.46 |
| `--layout AWL --eprobe_gpu 3` | 4 | 87.17 ± 0.04 | 124.80 ± 0.92 | 1600, 1400, 1700 | 2.63, 2.21, 2.88 (2.57 ± 0.34) | 10.29 ± 1.36 |
| the same, a probe sweep every 250 iterations | 4 | 87.18 ± 0.05 | 87.32 ± 0.04 | 1500, 1600, 1800 | 2.11, 2.25, 2.54 (2.30 ± 0.22) | 9.21 ± 0.89 |
| `--layout AL --eprobe_gpu 2 --pi_encode_once` | 3 | 80.39 ± 0.08 | 118.35 ± 1.23 | 1700, 1500, 1500 | 2.70, 2.25, 2.27 (2.41 ± 0.26) | 7.23 ± 0.77 |

Arms within one standard deviation of each other are read as level:
- The recipe and the learner's set are level.
- Every layout arm reaches 0.82 1.5-1.9 times sooner than the recipe (2.3-2.9 against 4.5 minutes), at 1.3-2.3 times its
  GPU-minutes.
- AL with the probe worker, AWL with the probe worker, and the last two arms are level with each other.
- AWL is level with AWL with the probe worker, and slower than AL with the probe worker and than the last two arms.
- AL alone is level with AWL and AWL with the probe worker, and slower than AL with the probe worker and than the last two
  arms.

The last two arms:
- A probe sweep every 250 iterations (`--eprobe_every 250 --eprobe_async 4`), as XMimic's four-GPU layout sweeps, removes the
  four-GPU arm's wait for the sweeps: 87.3 ms per iteration over iterations 1000-1500, against 124.8.
- `--pi_encode_once` on AL with the probe worker takes 7.4 % off its time per iteration (80.4 against 86.9 ms).

`--njmax 256` was not timed to return 0.82: it takes 1.2 % off an iteration (above).

**XMimic's own code** ([nico-bohlinger/xmimic](https://github.com/nico-bohlinger/xmimic) at `54860e1`, unmodified, with
its published configuration and the training budget of its table, `--max_minutes 6.2`; scored after training by its own
evaluator, `python -m xmimic.evaluate --reps 4 --group 18`, on another GPU; JAX at its `train.py`'s 0.9 of each GPU's
memory). Minutes from iteration 50 as above, and on its own clock, which starts after compilation; its target is a return of
at least 0.82 with no failed clip, held for three checkpoints in a row:

| arm | GPUs | seeds | it/s 200-1000 | it/s 1100-3500 | first ≥ 0.82: iterations | minutes from iteration 50 | on its clock | its target, minutes from iteration 50 |
|---|---|---|---|---|---|---|---|---|
| its one-GPU trainer | 1 | 3 | 7.20 ± 0.00 | 5.59 ± 0.02 | 1400, 1400, 1500 | 3.45, 3.44, 3.70 (3.53 ± 0.15) | 3.63, 3.62, 3.88 | 4.62, not within the run, 4.01 |
| `AWLP` | 4 | 3 | 13.48 ± 0.08 | 13.20 ± 0.01 | 1400, 1500, 1600 | 1.68, 1.82, 1.94 (1.81 ± 0.13) | 1.82, 1.96, 2.08 | 2.69, 2.45, 2.19 (2.44 ± 0.25) |
| `FP` | 2 | 1 (seed 0) | 7.20 | 7.18 | 1300 | 2.91 | 3.09 | 3.37 |
| `ALP` | 3 | 1 (seed 0) | 12.08 | 12.03 | 1400 | 1.86 | 2.01 | 2.28 |

XMimic's rows stand beside this library's and are not ranked with them: the evaluators, the simulators, the constraint rows
per world in training (256 against 500) and the clocks differ. During the compilation of its one-GPU run with seed 1, before
its clock and its iteration 50 started, a copy of 1.6 GB of earlier runs' files read the node's disk; its measures begin
after it (its minutes match seeds 0 and 2).

## NablaMimic's own code, timed

[NablaMimic](https://github.com/sailab-ai/NablaMimic) (∇Mimic) trains its policy through a differentiable simulator,
without a world model. Its authors' code at `bfbb2a6` runs here unmodified, with its own entry point and evaluation
(MuJoCo Warp, its G1 model; the full-clip normalized return of the actor it evaluates): seeds 0-2, one hour of training
each (its own `--hours 1`, a wall-clock budget), one run at a time while the machine's other GPUs were idle.
[`nablamimic/run.sh`](nablamimic/run.sh) starts a run, [`nablamimic/nm_table.py`](nablamimic/nm_table.py) writes the
tables, and [its page](../../docs/algorithms/nablamimic.md) says how it trains and how to run it. Its authors report
return 0.82 after 8 minutes on one RTX PRO 6000, at 0.38 s per update.

Columns: the first evaluated checkpoint at return 0.82 or above (update / minutes on its own clock / minutes from update
50, as the training speed above counts from iteration 50); that crossing interpolated linearly between the two
evaluations around it (its paper's rule, here per seed); the first of three evaluations in a row at 0.82 or above without
a fall. Its checkpoints come every 250 updates, about 3 minutes here, so its times are read on that grid. Memory is the
peak of JAX's allocator, of its limit.

**Its published configuration** (`configs/ours.args`, about 50 GB of GPU memory by its README) on one A100 80GB SXM4:

| run | s per update | first ≥ 0.82: update / min on its clock / min from update 50 | crossing, min (interpolated) | ≥ 0.82 without a fall, 3 in a row | best return (update) | last evaluation: return, clips with a fall | peak GPU memory |
|---|---|---|---|---|---|---|---|
| `nm_a100_80gb_published_s0` | 0.669 | 1000 / 11.33 / 10.69 | 11.07 | 2000 / 22.59 / 21.95 | 0.877 (5000) | 0.876, 0 of 40 | 44.6 of 63.8 GB |
| `nm_a100_80gb_published_s1` | 0.671 | 1250 / 14.21 / 13.57 | 13.85 | 2750 / 31.10 / 30.46 | 0.876 (5314) | 0.876, 0 of 40 | 44.7 of 63.8 GB |
| `nm_a100_80gb_published_s2` | 0.671 | 1750 / 19.81 / 19.17 | 17.81 | 2250 / 25.44 / 24.80 | 0.870 (4500) | 0.851, 1 of 40 | 44.6 of 63.8 GB |
| mean ± sd | 0.67 ± 0.00 | 1333 / 15.12 ± 4.31 / 14.48 ± 4.31 | 14.25 ± 3.39 | | | | |

**A reduced-memory configuration** on one A100 40GB SXM4, which the published configuration does not fit: the windows
rematerialized in the backward pass (`remat`: the same algorithm and hyperparameters, more time per update) and a replay
of 250 × 4096 states instead of 1000 × 4096 (`--replay-steps 250`). It is not a configuration its authors published;
`remat` alone and `remat` with `--replay-steps 500` also run out of memory on 40 GB.

| run | s per update | first ≥ 0.82: update / min on its clock / min from update 50 | crossing, min (interpolated) | ≥ 0.82 without a fall, 3 in a row | best return (update) | last evaluation: return, clips with a fall | peak GPU memory |
|---|---|---|---|---|---|---|---|
| `nm_a100_remat250_s0` | 0.714 | 1500 / 18.13 / 17.46 | 17.25 | 3000 / 36.11 / 35.44 | 0.871 (4995) | 0.871, 0 of 40 | 25.6 of 31.0 GB |
| `nm_a100_remat250_s1` | 0.717 | 1500 / 18.21 / 17.53 | 16.02 | 3000 / 36.28 / 35.61 | 0.865 (4750) | 0.864, 0 of 40 | 25.5 of 31.0 GB |
| `nm_a100_remat250_s2` | 0.716 | 1750 / 21.14 / 20.46 | 20.53 | 3250 / 39.17 / 38.49 | 0.872 (4984) | 0.872, 0 of 40 | 25.6 of 31.0 GB |
| mean ± sd | 0.72 ± 0.00 | 1583 / 19.16 ± 1.72 / 18.48 ± 1.72 | 17.93 ± 2.33 | | | | |

**Beside this library's fastest configurations.** On one A100 40GB SXM4, SuperTrack reaches return 0.82 after
4.13 ± 0.18 min and SuperMimic after 3.52 ± 0.30 min ([Training speed](#training-speed), seeds 0-2). NablaMimic's rows
stand beside these numbers and are not ranked with them. The same in both: the 40 clips, the return's formula, each
clip's start, the fall thresholds, the reward's terms and weights, the G1's joints, gains and mass. Not the same: the
evaluator and its simulator, the G1's colliders and self-collisions, the anchor body, the clips' heights, one evaluation
per checkpoint against four, and a checkpoint every 250 updates against one every 100 iterations
([side by side](../../docs/algorithms/nablamimic.md#4-its-evaluation-beside-this-librarys)). No checkpoint is scored by
both evaluators.

## H100 supplement

Selected configurations and their RigidFormer versions before its tuning, seeds 1-3, one H100 each (16 physical cores per run; other GPUs
on the host were not controlled), scored as on the A100, with a checkpoint every 2000 iterations. Open an issue for run
files. The last three rows use training logs, with noise and randomization, before and after SuperMimic's evaluation
drop. RigidFormer runs end before those windows. SuperMimic's paper reports return 0.82 after 1.2 h with its own configuration.

<!-- h100:begin -->
| | SuperTrack | SuperTrack-RF | SuperMimic | SuperMimic-RF |
|---|---|---|---|---|
| first evaluation at or above return 0.82 (training time) | 1.25 h (1.03-1.45) | 1 of 3 seeds reach it | 1.48 h (1.32-1.60) | 0 of 3 seeds reach it |
| highest evaluation return | 0.856 ± 0.002 | 0.817 ± 0.003 | 0.836 ± 0.002 | 0.807 ± 0.009 |
| normalized return, at 3 h | 0.844 ± 0.015 | 0.815 ± 0.001 | 0.074 ± 0.076 | 0.797 ± 0.003 |
| failed clips (of 40), at 3 h | 0.6 ± 0.7 | 0.2 ± 0.2 | 39.4 ± 1.0 | 1.1 ± 0.8 |
| MPJPE, root-relative (mm), at 3 h | 16.4 ± 1.5 | 25.4 ± 1.0 | 54.8 ± 6.5 | 33.1 ± 1.3 |
| joint-angle error (deg), at 3 h | 2.10 ± 0.06 | 2.60 ± 0.03 | 10.89 ± 0.97 | 6.28 ± 0.17 |
| root-orientation error (rad), at 3 h | 0.046 ± 0.006 | 0.082 ± 0.005 | 0.098 ± 0.021 | 0.059 ± 0.005 |
| body-velocity error (m/s), at 3 h | 0.201 ± 0.011 | 0.265 ± 0.009 | 0.400 ± 0.112 | 0.234 ± 0.007 |
| failed clips under root pushes of 1 m/s every 2 s, at 3 h | 3.0 ± 2.0 | 6.0 ± 2.6 | 40.0 ± 0.0 | 3.7 ± 1.2 |
| failed clips under root pushes of 2 m/s every 2 s, at 3 h | 18.0 ± 1.7 | 17.3 ± 3.8 | 40.0 ± 0.0 | 9.0 ± 0.0 |
| iterations in 3 h | 28,768 | 8,199 | 26,892 | 8,545 |
| training episodes: normalized return (training log, iterations 16k-20k → 24k-3 h) | 0.378 → 0.441 | – | 0.282 → 0.319 | – |
| training episodes ending in a failure (%) (training log, iterations 16k-20k → 24k-3 h) | 0.28 → 0.18 | – | 0.27 → 0.19 | – |
| world-model loss (training log, iterations 16k-20k → 24k-3 h) | 0.0521 → 0.0490 | – | 0.0417 → 0.0355 | – |
<!-- h100:end -->

## SuperMimic's late drop (diagnostic)

Failed clips (of 40) / MPJPE-local before the drop (iteration 20000) and after (3 h), evaluated with parts of training's
data collection enabled. Open an issue for files (`diag/h100`).

<!-- diag:begin -->
| evaluated with | seed 1, iteration 20000 | seed 1, 3 h | seed 3, iteration 20000 | seed 3, 3 h |
|---|---|---|---|---|
| as evaluated | 0 / 30.4 mm | 38 / 58.6 mm | 1 / 29.2 mm | 40 / 55.6 mm |
| clamp to ±20 | 0 / 30.5 mm | 38 / 57.3 mm | 1 / 29.2 mm | 40 / 61.8 mm |
| setpoint noise 0.3 with the clamp | 0 / 29.1 mm | 1 / 27.5 mm | 1 / 28.8 mm | 0 / 26.2 mm |
| observation noise | 2 / 30.5 mm | 39 / 53.4 mm | 2 / 29.3 mm | 40 / 58.5 mm |
| setpoint and observation noise | 0 / 30.1 mm | 1 / 28.2 mm | 0 / 29.5 mm | 0 / 26.9 mm |
<!-- diag:end -->

The 3 h checkpoints track only with training's action noise (0.3): with it they fail 0-1 clips, without it 38-40. The
clamp and the observation noise play no part. The iteration-20000 checkpoints fail 0-2 clips under every condition. The
paper uses action noise 0.1, which it describes as serving only the world-model fit (§3; noise-free evaluation in
Appendix F.1).

## Noise check: SuperMimic with action noise 0.1

SuperMimic's full configuration with the paper's action noise 0.1 instead of 0.3, seeds 1-3, 3 h on the H100, scored
as above. Rule, fixed before the runs: a drop is an evaluation below 0.60 after the first at or above return 0.82 (with 0.3, evaluations
fell below 0.60 by iteration 22000-24000 and ended at 0.02-0.15). No seed drops: the lowest subsequent evaluation is
0.817-0.829; at 3 h, return is 0.854 ± 0.003 with 0.2 failed clips. Open an issue for run files.

<!-- noise:begin -->
| | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| first evaluation at or above return 0.82 | iteration 12,000 (1.31 h) | iteration 14,000 (1.52 h) | iteration 10,000 (0.99 h) | |
| lowest evaluation after it | 0.829 | 0.821 | 0.817 | |
| drops (an evaluation below 0.60 after it) | no | no | no | |
| normalized return, at 3 h | 0.857 | 0.853 | 0.851 | 0.854 ± 0.003 |
| failed clips (of 40), at 3 h | 0.0 | 0.3 | 0.3 | 0.2 ± 0.2 |
| MPJPE, root-relative (mm), at 3 h | 26.5 | 26.2 | 24.6 | 25.8 ± 1.0 |
| joint-angle error (deg), at 3 h | 4.36 | 4.53 | 4.09 | 4.33 ± 0.23 |
| root-orientation error (rad), at 3 h | 0.046 | 0.045 | 0.042 | 0.044 ± 0.002 |
| body-velocity error (m/s), at 3 h | 0.192 | 0.185 | 0.191 | 0.189 ± 0.004 |
| failed clips under root pushes of 1 m/s every 2 s, at 3 h | 5.0 | 1.0 | 3.0 | 3.0 ± 2.0 |
| failed clips under root pushes of 2 m/s every 2 s, at 3 h | 12.0 | 14.0 | 9.0 | 11.7 ± 2.5 |
| iterations in 3 h | 27,450 | 27,462 | 28,751 | 27,888 |
<!-- noise:end -->

## RigidFormer check 1: dynamics shifts

The equal-iteration checkpoints of the MLP finals and of their RigidFormer versions before its tuning, seeds 1-3, each evaluated once
with the feet's and the ground's friction at 0.2, 0.5 or 2.0 (training draws 0.3-1.2) or with 5 or 10 kg added to the
torso. Rule: RigidFormer counts as more robust under a condition when at least one of return, failed clips and
MPJPE-local is significantly better (Welch 95% interval) and none is significantly worse. Generated by
`tools/rf_checks.py 1`.

<!-- rfcheck1:begin -->
| algorithm | case | metric | MLP | RigidFormer | RF - MLP [95% CI] | verdict |
|---|---|---|---|---|---|---|
| SuperTrack | no shift: the watcher's evaluation (context) | return | 0.830 | 0.824 | -0.006 [-0.037, +0.025] | context, no verdict |
|  |  | failed clips | 1.00 | 0.33 | -0.67 [-2.72, +1.38] |  |
|  |  | MPJPE-local (mm) | 20.1 | 23.1 | +3.0 [+1.9, +4.1] |  |
| SuperTrack | friction 0.2 | return | 0.264 | 0.315 | +0.051 [-0.043, +0.145] | no significant difference |
|  |  | failed clips | 36.33 | 34.00 | -2.33 [-7.94, +3.27] |  |
|  |  | MPJPE-local (mm) | 49.5 | 51.2 | +1.7 [-5.6, +9.0] |  |
| SuperTrack | friction 0.5 | return | 0.815 | 0.809 | -0.006 [-0.029, +0.016] | RigidFormer worse (MPJPE-local (mm)) |
|  |  | failed clips | 1.67 | 0.67 | -1.00 [-2.31, +0.31] |  |
|  |  | MPJPE-local (mm) | 21.1 | 23.2 | +2.1 [+0.9, +3.3] |  |
| SuperTrack | friction 2.0 | return | 0.730 | 0.742 | +0.011 [-0.012, +0.034] | no significant difference |
|  |  | failed clips | 4.67 | 2.00 | -2.67 [-7.30, +1.96] |  |
|  |  | MPJPE-local (mm) | 36.2 | 37.4 | +1.2 [-3.5, +6.0] |  |
| SuperTrack | payload 5 kg | return | 0.692 | 0.700 | +0.008 [-0.027, +0.042] | no significant difference |
|  |  | failed clips | 11.33 | 9.00 | -2.33 [-6.73, +2.06] |  |
|  |  | MPJPE-local (mm) | 33.3 | 35.3 | +2.0 [-7.5, +11.5] |  |
| SuperTrack | payload 10 kg | return | 0.217 | 0.239 | +0.022 [-0.105, +0.150] | no significant difference |
|  |  | failed clips | 36.33 | 35.67 | -0.67 [-7.92, +6.58] |  |
|  |  | MPJPE-local (mm) | 54.3 | 55.4 | +1.1 [-12.5, +14.6] |  |
| SuperMimic | no shift: the watcher's evaluation (context) | return | 0.828 | 0.828 | +0.000 [-0.021, +0.021] | context, no verdict |
|  |  | failed clips | 1.00 | 0.00 | -1.00 [-3.48, +1.48] |  |
|  |  | MPJPE-local (mm) | 28.1 | 30.1 | +2.0 [-1.2, +5.2] |  |
| SuperMimic | friction 0.2 | return | 0.443 | 0.493 | +0.049 [-0.047, +0.145] | no significant difference |
|  |  | failed clips | 26.67 | 23.00 | -3.67 [-13.20, +5.86] |  |
|  |  | MPJPE-local (mm) | 40.8 | 42.4 | +1.7 [-0.9, +4.2] |  |
| SuperMimic | friction 0.5 | return | 0.827 | 0.824 | -0.003 [-0.030, +0.024] | no significant difference |
|  |  | failed clips | 0.33 | 0.00 | -0.33 [-1.77, +1.10] |  |
|  |  | MPJPE-local (mm) | 28.0 | 29.8 | +1.8 [-0.8, +4.3] |  |
| SuperMimic | friction 2.0 | return | 0.803 | 0.777 | -0.026 [-0.063, +0.012] | no significant difference |
|  |  | failed clips | 0.00 | 1.00 | +1.00 [-1.48, +3.48] |  |
|  |  | MPJPE-local (mm) | 34.4 | 37.4 | +3.0 [-0.8, +6.9] |  |
| SuperMimic | payload 5 kg | return | 0.780 | 0.774 | -0.007 [-0.080, +0.067] | no significant difference |
|  |  | failed clips | 3.67 | 3.00 | -0.67 [-5.92, +4.59] |  |
|  |  | MPJPE-local (mm) | 33.0 | 36.1 | +3.2 [-0.0, +6.4] |  |
| SuperMimic | payload 10 kg | return | 0.550 | 0.553 | +0.003 [-0.096, +0.102] | no significant difference |
|  |  | failed clips | 20.00 | 19.33 | -0.67 [-10.63, +9.29] |  |
|  |  | MPJPE-local (mm) | 48.7 | 49.7 | +1.0 [-3.6, +5.6] |  |
<!-- rfcheck1:end -->

SuperTrack-RF's MPJPE-local is already 3.0 mm higher without a shift (first rows), so its friction-0.5 verdict reflects
that gap rather than a loss of robustness.

## RigidFormer check 2: sample efficiency

Fresh seeds 4-6, 8000 iterations, the full configurations with the MLP and with RigidFormer (RF-S, world-model learning
rate 4e-4), all twelve runs on A100 80GB GPUs with one code version. Rule: at iterations 4000, 6000 and 8000,
RigidFormer counts as more sample-efficient when at least one of return, failed clips and MPJPE-local is significantly
better (Welch 95% interval) and none is significantly worse; the hypothesis holds for an algorithm when that is the case
at two of the three checkpoints. Generated by `tools/rf_checks.py 2`.

<!-- rfcheck2:begin -->
| algorithm | case | metric | MLP | RigidFormer | RF - MLP [95% CI] | verdict |
|---|---|---|---|---|---|---|
| SuperTrack | iteration 4000 | return | 0.723 | 0.718 | -0.005 [-0.039, +0.029] | no significant difference |
|  |  | failed clips | 5.00 | 5.00 | +0.00 [-4.97, +4.97] |  |
|  |  | MPJPE-local (mm) | 40.0 | 40.8 | +0.8 [-3.3, +4.9] |  |
| SuperTrack | iteration 6000 | return | 0.775 | 0.783 | +0.008 [-0.042, +0.058] | RigidFormer better (failed clips) |
|  |  | failed clips | 3.33 | 1.00 | -2.33 [-4.38, -0.28] |  |
|  |  | MPJPE-local (mm) | 31.7 | 31.7 | -0.0 [-6.9, +6.8] |  |
| SuperTrack | iteration 8000 | return | 0.811 | 0.802 | -0.010 [-0.031, +0.012] | no significant difference |
|  |  | failed clips | 1.33 | 1.00 | -0.33 [-2.38, +1.72] |  |
|  |  | MPJPE-local (mm) | 26.5 | 26.8 | +0.3 [-2.1, +2.8] |  |
| SuperTrack | hypothesis (2 of 3 checkpoints) | | | | | does not hold (1 of 3) |
| SuperMimic | iteration 4000 | return | 0.733 | 0.741 | +0.008 [-0.018, +0.034] | no significant difference |
|  |  | failed clips | 3.33 | 2.67 | -0.67 [-5.99, +4.66] |  |
|  |  | MPJPE-local (mm) | 44.1 | 44.3 | +0.2 [-3.0, +3.3] |  |
| SuperMimic | iteration 6000 | return | 0.791 | 0.788 | -0.003 [-0.022, +0.016] | no significant difference |
|  |  | failed clips | 1.67 | 1.33 | -0.33 [-1.64, +0.98] |  |
|  |  | MPJPE-local (mm) | 35.9 | 36.6 | +0.7 [-5.6, +7.0] |  |
| SuperMimic | iteration 8000 | return | 0.802 | 0.803 | +0.001 [-0.021, +0.024] | no significant difference |
|  |  | failed clips | 1.00 | 0.67 | -0.33 [-1.77, +1.10] |  |
|  |  | MPJPE-local (mm) | 33.1 | 32.8 | -0.3 [-3.0, +2.4] |  |
| SuperMimic | hypothesis (2 of 3 checkpoints) | | | | | does not hold (0 of 3) |
<!-- rfcheck2:end -->

## RigidFormer check 3: unseen motions

LAFAN1's own split: subject 5's 7 clips held out, training on the other 33. Seeds 1-3, each trained to the
equal-iteration count (SuperTrack 12670, SuperMimic 13540), the full configurations with the MLP and with RigidFormer
(RF-S, world-model learning rate 4e-4). All twelve runs on four H200s, two per GPU, with one code version; none died or
was rerun. Every checkpoint got one standard evaluation on all 40 clips, and the held-out and training clips' values
come from its per-clip results. Rule: at the last checkpoint, on the held-out clips, RigidFormer counts as generalizing
better when at least one of return, failed clips and MPJPE-local is significantly better (Welch 95% interval) and none
is significantly worse, and as generalizing worse in the mirror case. The training clips, the held-out minus training
gap and the earlier checkpoints are context, without a verdict. Generated by `tools/rf_checks.py 3`. At the last
checkpoint every run fails the held-out fallAndGetUp1_subject5; the training clip fallAndGetUp2_subject3 fails in two
runs, and dance2_subject4, walk3_subject3 (training) and jumps1_subject5 (held-out) in one each.

<!-- rfcheck3:begin -->
| algorithm | case | metric | MLP | RigidFormer | RF - MLP [95% CI] | verdict |
|---|---|---|---|---|---|---|
| SuperTrack | held-out clips, it 12670 | return | 0.685 | 0.650 | -0.035 [-0.146, +0.075] | no significant difference |
|  |  | failed clips | 1.00 | 1.33 | +0.33 [-1.10, +1.77] |  |
|  |  | MPJPE-local (mm) | 28.7 | 36.0 | +7.3 [-0.0, +14.6] |  |
| SuperTrack | training clips, it 12670 (context) | return | 0.840 | 0.827 | -0.013 [-0.026, +0.001] | context, no verdict |
|  |  | failed clips | 0.33 | 0.33 | +0.00 [-1.31, +1.31] |  |
|  |  | MPJPE-local (mm) | 20.2 | 21.9 | +1.7 [+0.6, +2.9] |  |
| SuperTrack | held-out minus training (context) | return | -0.155 | -0.177 | -0.022 [-0.117, +0.072] | context, no verdict |
|  |  | failed clips | 0.67 | 1.00 | +0.33 [-1.10, +1.77] |  |
|  |  | MPJPE-local (mm) | 8.5 | 14.1 | +5.6 [-1.4, +12.5] |  |
| SuperMimic | held-out clips, it 13540 | return | 0.713 | 0.704 | -0.010 [-0.027, +0.008] | no significant difference |
|  |  | failed clips | 1.00 | 1.00 | +0.00 [+0.00, +0.00] |  |
|  |  | MPJPE-local (mm) | 31.3 | 32.5 | +1.1 [-3.0, +5.3] |  |
| SuperMimic | training clips, it 13540 (context) | return | 0.825 | 0.836 | +0.011 [-0.004, +0.027] | context, no verdict |
|  |  | failed clips | 0.67 | 0.00 | -0.67 [-2.10, +0.77] |  |
|  |  | MPJPE-local (mm) | 28.1 | 28.1 | -0.0 [-3.0, +2.9] |  |
| SuperMimic | held-out minus training (context) | return | -0.111 | -0.132 | -0.021 [-0.029, -0.013] | context, no verdict |
|  |  | failed clips | 0.33 | 1.00 | +0.67 [-0.77, +2.10] |  |
|  |  | MPJPE-local (mm) | 3.2 | 4.4 | +1.2 [-0.9, +3.3] |  |

Held-out clips at every checkpoint (mean over seeds 1-3; context):

| algorithm | model | iteration | return | failed clips | MPJPE-local (mm) |
|---|---|---|---|---|---|
| SuperTrack | MLP | 2000 | 0.551 | 1.33 | 60.4 |
| SuperTrack | RigidFormer | 2000 | 0.499 | 1.67 | 78.8 |
| SuperTrack | MLP | 4000 | 0.645 | 1.00 | 45.6 |
| SuperTrack | RigidFormer | 4000 | 0.632 | 1.00 | 47.1 |
| SuperTrack | MLP | 6000 | 0.665 | 1.00 | 35.1 |
| SuperTrack | RigidFormer | 6000 | 0.618 | 1.67 | 39.7 |
| SuperTrack | MLP | 8000 | 0.671 | 1.00 | 36.4 |
| SuperTrack | RigidFormer | 8000 | 0.633 | 1.67 | 35.1 |
| SuperTrack | MLP | 10000 | 0.677 | 1.00 | 35.7 |
| SuperTrack | RigidFormer | 10000 | 0.677 | 1.00 | 39.4 |
| SuperTrack | MLP | 12000 | 0.631 | 1.67 | 32.9 |
| SuperTrack | RigidFormer | 12000 | 0.652 | 1.33 | 32.1 |
| SuperTrack | MLP | 12670 | 0.685 | 1.00 | 28.7 |
| SuperTrack | RigidFormer | 12670 | 0.650 | 1.33 | 36.0 |
| SuperMimic | MLP | 2000 | 0.568 | 1.00 | 58.2 |
| SuperMimic | RigidFormer | 2000 | 0.557 | 1.00 | 59.5 |
| SuperMimic | MLP | 4000 | 0.660 | 1.33 | 46.0 |
| SuperMimic | RigidFormer | 4000 | 0.663 | 1.00 | 44.6 |
| SuperMimic | MLP | 6000 | 0.694 | 1.00 | 36.8 |
| SuperMimic | RigidFormer | 6000 | 0.690 | 1.00 | 38.4 |
| SuperMimic | MLP | 8000 | 0.706 | 1.00 | 34.1 |
| SuperMimic | RigidFormer | 8000 | 0.717 | 1.00 | 36.2 |
| SuperMimic | MLP | 10000 | 0.705 | 1.00 | 34.7 |
| SuperMimic | RigidFormer | 10000 | 0.706 | 1.00 | 34.9 |
| SuperMimic | MLP | 12000 | 0.709 | 1.00 | 32.9 |
| SuperMimic | RigidFormer | 12000 | 0.672 | 1.33 | 32.4 |
| SuperMimic | MLP | 13540 | 0.713 | 1.00 | 31.3 |
| SuperMimic | RigidFormer | 13540 | 0.704 | 1.00 | 32.5 |
<!-- rfcheck3:end -->

## RigidFormer tuned online

SuperTrack-RF and SuperMimic-RF tuned online as the MLP was, 12 tuning runs each as the MLP had (selected: SuperTrack
`--st_balance_w 2`, SuperMimic `--imag_start_perturb 0.4`; RF-S, world-model learning rate 4e-4), on seeds 1-3, one A100
40GB each, against the MLP finals and against RigidFormer before its tuning. Equal iterations: the MLP finals' counts at
3 h. Mean ± std over seeds;
difference [Welch 95% interval], bold when it excludes 0.

At equal iterations no metric differs significantly from the MLP finals. At 3 h without the speed options, where
RigidFormer reaches about 4,400 iterations, return is 0.08-0.09 lower and MPJPE 14-19 mm higher. Against RigidFormer
before its tuning, only SuperTrack's failures under 2 m/s pushes at 3 h differ.

| algorithm | comparison | return | failed clips | MPJPE-local (mm) | joint angle (deg) | failed at 1 m/s pushes | failed at 2 m/s pushes |
|---|---|---|---|---|---|---|---|
| SuperTrack | selected RigidFormer - MLP finals, equal iterations | 0.824 ± 0.013; -0.008 [-0.034, 0.018] | 0.33 ± 0.33; -0.56 [-2.36, 1.25] | 22.8 ± 2.1; +2.8 [-1.8, 7.3] | 2.35 ± 0.09; +0.03 [-0.16, 0.22] | 9.0 ± 1.7; +1.3 [-3.4, 6.1] | 23.3 ± 3.5; -1.3 [-9.1, 6.5] |
| SuperTrack | selected RigidFormer - MLP finals, at 3 h | 0.742 ± 0.024; **-0.090 [-0.142, -0.039]** | 3.22 ± 0.38; **+2.33 [0.57, 4.10]** | 39.4 ± 2.8; **+19.4 [12.9, 25.9]** | 3.14 ± 0.10; **+0.83 [0.62, 1.03]** | 7.7 ± 2.1; +0.0 [-5.0, 5.0] | 29.3 ± 1.2; **+4.7 [2.0, 7.3]** |
| SuperTrack | selected RigidFormer - RigidFormer before tuning, equal iterations | 0.824 ± 0.013; +0.006 [-0.023, 0.035] | 0.33 ± 0.33; -0.33 [-1.09, 0.42] | 22.8 ± 2.1; -0.3 [-5.1, 4.4] | 2.35 ± 0.09; -0.13 [-0.33, 0.06] | 9.0 ± 1.7; +2.3 [-5.0, 9.6] | 23.3 ± 3.5; +1.0 [-11.5, 13.5] |
| SuperTrack | selected RigidFormer - RigidFormer before tuning, at 3 h | 0.742 ± 0.024; +0.007 [-0.050, 0.064] | 3.22 ± 0.38; -1.33 [-2.76, 0.09] | 39.4 ± 2.8; +2.5 [-3.2, 8.3] | 3.14 ± 0.10; -0.07 [-0.27, 0.13] | 7.7 ± 2.1; -2.7 [-7.4, 2.1] | 29.3 ± 1.2; **+8.7 [3.4, 14.0]** |
| SuperMimic | selected RigidFormer - MLP finals, equal iterations | 0.815 ± 0.022; -0.009 [-0.053, 0.036] | 0.11 ± 0.19; -1.11 [-3.92, 1.70] | 31.5 ± 3.9; +3.1 [-6.2, 12.5] | 5.69 ± 1.59; +0.79 [-3.15, 4.74] | 1.7 ± 0.6; +0.0 [-1.3, 1.3] | 8.3 ± 1.2; -1.0 [-3.6, 1.6] |
| SuperMimic | selected RigidFormer - MLP finals, at 3 h | 0.744 ± 0.007; **-0.080 [-0.106, -0.054]** | 2.22 ± 0.38; +1.00 [-1.60, 3.60] | 42.3 ± 1.4; **+13.9 [10.9, 16.9]** | 7.71 ± 0.40; **+2.82 [1.83, 3.80]** | 2.7 ± 1.2; +1.0 [-1.4, 3.4] | 7.7 ± 2.1; -1.7 [-5.9, 2.6] |
| SuperMimic | selected RigidFormer - RigidFormer before tuning, equal iterations | 0.815 ± 0.022; -0.012 [-0.063, 0.038] | 0.11 ± 0.19; +0.11 [-0.37, 0.59] | 31.5 ± 3.9; +1.5 [-7.0, 10.0] | 5.69 ± 1.59; -0.02 [-3.68, 3.63] | 1.7 ± 0.6; -1.3 [-7.5, 4.9] | 8.3 ± 1.2; +0.0 [-2.4, 2.4] |
| SuperMimic | selected RigidFormer - RigidFormer before tuning, at 3 h | 0.744 ± 0.007; -0.004 [-0.020, 0.012] | 2.22 ± 0.38; +0.33 [-0.72, 1.39] | 42.3 ± 1.4; -1.3 [-6.1, 3.6] | 7.71 ± 0.40; +0.16 [-0.71, 1.03] | 2.7 ± 1.2; -1.3 [-3.8, 1.1] | 7.7 ± 2.1; -3.0 [-15.0, 9.0] |

Checkpoints of the selected configurations: SuperTrack last: iterations 12670, 12670, 12670; SuperTrack 3 h: iterations 4300, 4475, 4341; SuperMimic last: iterations 13540, 13540, 13540; SuperMimic 3 h: iterations 4348, 4380, 4462.

## RigidFormer's final configurations

SuperTrack-RF and SuperMimic-RF tuned online as the MLP was, with every speed option, on 3 h runs with seed 0 (one A100
40GB each; the RigidFormer after 21 and 19 tuning runs in all, SuperTrack-RF's and SuperMimic-RF's, the MLP after 12),
ranked by return and, within 0.01, by fewer failed clips: SuperTrack-RF `--st_balance_w 2 --wm_batches 5`,
SuperMimic-RF `--imag_start_perturb 0.4 --wm_batches 7`, both `--wm_lr 4e-4`. The selected configurations then ran on
fresh seeds 7-9, 3 h each, scored as the main comparison, next to the MLP configurations with the speed options that
apply to them. Differences with Welch 95% intervals; bold when the interval excludes 0.

| | SuperTrack-RF | SuperMimic-RF | SuperTrack-RF - SuperMimic-RF |
|---|---|---|---|
| return | 0.832 ± 0.003 | 0.804 ± 0.007 | **+0.028 [0.014, 0.042]** |
| failed clips | 0.44 ± 0.51 | 0.78 ± 0.38 | -0.33 [-1.39, 0.72] |
| MPJPE-local (mm) | 21.0 ± 1.3 | 33.1 ± 0.5 | **-12.1 [-14.8, -9.3]** |
| joint angle (deg) | 2.24 ± 0.05 | 6.05 ± 0.62 | **-3.80 [-5.33, -2.28]** |
| failed at 1 m/s pushes | 5.3 ± 1.2 | 1.3 ± 0.6 | **+4.0 [1.6, 6.4]** |
| failed at 2 m/s pushes | 24.3 ± 3.1 | 9.0 ± 2.0 | **+15.3 [9.1, 21.6]** |

Iterations reached in 3 h: SuperTrack-RF 12654, 12677, 12696; SuperMimic-RF 10710, 10694, 10726.

| algorithm | comparison | return | failed clips | MPJPE-local (mm) | joint angle (deg) | failed at 1 m/s pushes | failed at 2 m/s pushes |
|---|---|---|---|---|---|---|---|
| SuperTrack | RigidFormer - MLP | 0.832 ± 0.003; **-0.029 [-0.036, -0.022]** | 0.44 ± 0.51; +0.44 [-0.82, 1.71] | 21.0 ± 1.3; **+6.5 [3.4, 9.6]** | 2.24 ± 0.05; **+0.29 [0.18, 0.40]** | 5.3 ± 1.2; +2.0 [-0.4, 4.4] | 24.3 ± 3.1; **+6.7 [0.3, 13.0]** |
| SuperTrack | MLP iterations in 3 h | 35621, 35794, 35796 | | | | | |
| SuperMimic | RigidFormer - MLP | 0.804 ± 0.007; **+0.667 [0.246, 1.088]** | 0.78 ± 0.38; **-36.67 [-47.55, -25.79]** | 33.1 ± 0.5; -22.6 [-52.3, 7.1] | 6.05 ± 0.62; -3.64 [-7.76, 0.47] | 1.3 ± 0.6; **-38.0 [-40.4, -35.6]** | 9.0 ± 2.0; **-31.0 [-36.0, -26.0]** |
| SuperMimic | MLP iterations in 3 h | 33184, 33205, 33244 | | | | | |

With the speed options the MLP runs about 3x more iterations in 3 h. SuperTrack's MLP return is 0.861 ± 0.003 and its
MPJPE-local 14.5 ± 0.2 mm in these runs (seeds 7-9; the tuning rounds' configuration with the speed options). SuperMimic's MLP with action noise 0.3
drops late, as on the H100 (return 0.137 at about 33k iterations); its highest per-checkpoint evaluation averages 0.841
over the seeds, SuperMimic-RF's 0.811. With action noise 0.1 and `--sm_residual` it does not drop (0.872 ± 0.003, the
[3 h configurations at one budget](#the-3-h-configurations-at-one-budget), where SuperTrack's configuration scores
0.857 ± 0.004).

Held-out clips, as in check 3: both final configurations and the MLP's trained on the 33 clips of subjects 1-4 to equal
iterations, seeds 7-9, scored on subject 5's 7 clips at the last checkpoint, with check 3's rule.

| algorithm | case | metric | MLP | RigidFormer | RF - MLP [95% CI] | verdict |
|---|---|---|---|---|---|---|
| SuperTrack | held-out clips, it 12670 | return | 0.682 | 0.684 | +0.002 [-0.021, +0.026] | no significant difference |
|  |  | failed clips | 1.00 | 1.00 | +0.00 [+0.00, +0.00] |  |
|  |  | MPJPE-local (mm) | 32.3 | 30.6 | -1.7 [-16.2, +12.8] |  |
| SuperTrack | training clips, it 12670 (context) | return | 0.830 | 0.835 | +0.004 [-0.027, +0.036] | context, no verdict |
|  |  | failed clips | 0.33 | 0.67 | +0.33 [-0.98, +1.64] |  |
|  |  | MPJPE-local (mm) | 20.2 | 20.2 | +0.0 [-3.6, +3.6] |  |
| SuperTrack | held-out minus training (context) | return | -0.148 | -0.150 | -0.002 [-0.040, +0.036] | context, no verdict |
|  |  | failed clips | 0.67 | 0.33 | -0.33 [-1.64, +0.98] |  |
|  |  | MPJPE-local (mm) | 12.1 | 10.4 | -1.7 [-12.9, +9.5] |  |
| SuperMimic | held-out clips, it 13540 | return | 0.708 | 0.703 | -0.005 [-0.009, -0.000] | RigidFormer worse (return) |
|  |  | failed clips | 1.00 | 1.00 | +0.00 [+0.00, +0.00] |  |
|  |  | MPJPE-local (mm) | 33.7 | 31.4 | -2.2 [-5.8, +1.3] |  |
| SuperMimic | training clips, it 13540 (context) | return | 0.818 | 0.827 | +0.009 [-0.002, +0.020] | context, no verdict |
|  |  | failed clips | 0.33 | 0.00 | -0.33 [-1.77, +1.10] |  |
|  |  | MPJPE-local (mm) | 31.5 | 29.8 | -1.8 [-3.5, +0.0] |  |
| SuperMimic | held-out minus training (context) | return | -0.110 | -0.124 | -0.014 [-0.027, -0.000] | context, no verdict |
|  |  | failed clips | 0.67 | 1.00 | +0.33 [-1.10, +1.77] |  |
|  |  | MPJPE-local (mm) | 2.1 | 1.7 | -0.5 [-4.5, +3.5] |  |

## Policy-gradient diagnostic (RigidFormer tuning, outside the study)

Three options against spikes in the policy gradient through the world model, off by default: `--bptt_clip_k k` caps each
rollout's state gradient after every imagined step at k times the batch median, `--rf_rope_detach` removes the gradient
through RigidFormer's rotary positions, and `--wm_bptt_fp32` runs the world model without bf16 autocast in the policy
update. From a finished run's last checkpoint, 1000 policy updates at learning rate 0 on a replay recorded with its
policy, the same batches for every setting (`grad_diag.sh`, `tools/grad_stats.py`). Norms are the policy gradient's.
"cos to mean direction" averages single updates' cosine with the mean direction of 500 other updates without any option,
"cos of the > 3x ones" does so for the updates above three times the median norm, "consistency" is the length of the
mean unit gradient (1 when all updates align), and "mean direction against base" is the cosine of this setting's mean
direction with that of the run's own setting. `final_*_s1` are seed 1 of the MLP finals and `abl_*_rf_s1` that of
their RigidFormer versions before its tuning (A100 40GB), `c2_*_s4` seed 4 of check 2 (A100 80GB); `_rf` marks RigidFormer. Generated
by `tools/grad_diag_report.py`.

On every RigidFormer run the cap lowers the 99th percentile of the gradient norm and raises single updates' cosine with
the mean direction (SuperTrack-RF seed 1: 17.0 to 4.5 with k = 10 and 4.1 with k = 3, cosine 0.90 to 0.97). On every
run, with k = 10 the mean direction stays that of the uncapped gradient (cosine 0.997-0.999); with k = 3 it turns a
little (0.96-0.995). Without autocast the updates above three times the median fall on SuperMimic-RF seed 1 (19 to 12)
and rise on the other three RigidFormer runs, and an update takes about 50% longer. The rotary detach changes little.
The RigidFormer tuning therefore tries the cap with k = 10 and k = 3; the rows of SuperTrack-RF seed 1 with the cap or
without autocast and the MLP's seed-1 rows were measured after that choice.

<!-- graddiag:begin -->
| run | setting | median norm | p99 | max | > 3x median | > 10x median | cos to mean direction | cos of the > 3x ones | consistency | mean direction against base | s per update |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_sm_rf_s1 | base | 1.91 | 9.1 | 79.7 | 19 | 4 | 0.840 | 0.18 | 0.841 | 0.999 | 1.05 |
| abl_sm_rf_s1 | rf_rope_detach=1 | 1.78 | 8.4 | 76.2 | 15 | 4 | 0.839 | 0.19 | 0.857 | 0.979 | 1.04 |
| abl_sm_rf_s1 | bptt_clip_k=3 | 1.69 | 5.1 | 43.3 | 11 | 2 | 0.867 | 0.26 | 0.895 | 0.969 | 1.07 |
| abl_sm_rf_s1 | bptt_clip_k=10 | 1.85 | 5.1 | 26.9 | 8 | 2 | 0.867 | 0.26 | 0.869 | 0.997 | 1.05 |
| abl_sm_rf_s1 | wm_bptt_fp32=1 | 1.91 | 6.0 | 24.6 | 12 | 1 | 0.840 | 0.21 | 0.842 | 0.998 | 1.58 |
| abl_st_rf_s1 | base | 3.29 | 17.0 | 110.1 | 17 | 4 | 0.902 | 0.16 | 0.902 | 1.000 | 0.99 |
| abl_st_rf_s1 | rf_rope_detach=1 | 3.27 | 15.7 | 108.9 | 19 | 4 | 0.902 | 0.19 | 0.908 | 0.994 | 0.98 |
| abl_st_rf_s1 | bptt_clip_k=3 | 3.13 | 4.1 | 34.2 | 4 | 1 | 0.972 | 0.16 | 0.980 | 0.992 | 1.00 |
| abl_st_rf_s1 | bptt_clip_k=10 | 3.20 | 4.5 | 32.4 | 2 | 1 | 0.970 | 0.14 | 0.971 | 0.999 | 1.00 |
| abl_st_rf_s1 | wm_bptt_fp32=1 | 3.30 | 21.1 | 189.4 | 29 | 6 | 0.901 | 0.20 | 0.901 | 0.999 | 1.52 |
| c2_sm_rf_s4 | base | 1.88 | 5.4 | 56.5 | 10 | 3 | 0.870 | 0.25 | 0.870 | 1.000 | 0.95 |
| c2_sm_rf_s4 | rf_rope_detach=1 | 1.83 | 4.9 | 46.9 | 8 | 1 | 0.878 | 0.21 | 0.883 | 0.994 | 0.94 |
| c2_sm_rf_s4 | bptt_clip_k=3 | 1.76 | 3.4 | 64.9 | 5 | 2 | 0.907 | 0.19 | 0.942 | 0.964 | 0.96 |
| c2_sm_rf_s4 | bptt_clip_k=10 | 1.84 | 3.3 | 48.3 | 5 | 1 | 0.915 | 0.12 | 0.916 | 0.999 | 0.96 |
| c2_sm_rf_s4 | wm_bptt_fp32=1 | 1.88 | 7.1 | 41.9 | 15 | 5 | 0.868 | 0.20 | 0.869 | 0.999 | 1.45 |
| c2_sm_s4 | base | 2.19 | 5.0 | 54.6 | 4 | 2 | 0.917 | 0.12 | 0.917 | 1.000 | 0.25 |
| c2_sm_s4 | bptt_clip_k=3 | 2.04 | 2.8 | 8.7 | 3 | 0 | 0.950 | 0.33 | 0.962 | 0.988 | 0.26 |
| c2_sm_s4 | bptt_clip_k=10 | 2.12 | 3.1 | 16.8 | 5 | 0 | 0.947 | 0.19 | 0.948 | 0.999 | 0.26 |
| c2_sm_s4 | wm_bptt_fp32=1 | 2.19 | 5.7 | 17.9 | 8 | 0 | 0.914 | 0.21 | 0.915 | 1.000 | 0.27 |
| c2_st_rf_s4 | base | 3.46 | 6.6 | 48.3 | 4 | 1 | 0.930 | 0.22 | 0.930 | 1.000 | 1.04 |
| c2_st_rf_s4 | rf_rope_detach=1 | 3.47 | 6.2 | 57.6 | 4 | 1 | 0.931 | 0.19 | 0.934 | 0.997 | 1.03 |
| c2_st_rf_s4 | bptt_clip_k=3 | 3.23 | 4.2 | 54.9 | 4 | 2 | 0.971 | 0.14 | 0.978 | 0.993 | 1.06 |
| c2_st_rf_s4 | bptt_clip_k=10 | 3.36 | 4.5 | 19.7 | 4 | 0 | 0.968 | 0.18 | 0.969 | 0.999 | 1.06 |
| c2_st_rf_s4 | wm_bptt_fp32=1 | 3.45 | 7.0 | 213.9 | 7 | 2 | 0.926 | 0.19 | 0.927 | 1.000 | 1.58 |
| c2_st_s4 | base | 2.70 | 5.3 | 43.9 | 6 | 2 | 0.924 | 0.14 | 0.924 | 1.000 | 0.28 |
| c2_st_s4 | bptt_clip_k=3 | 2.48 | 4.0 | 70.3 | 5 | 1 | 0.951 | 0.20 | 0.966 | 0.985 | 0.29 |
| c2_st_s4 | bptt_clip_k=10 | 2.61 | 4.6 | 144.1 | 6 | 1 | 0.951 | 0.10 | 0.952 | 0.998 | 0.29 |
| c2_st_s4 | wm_bptt_fp32=1 | 2.70 | 7.3 | 62.4 | 9 | 2 | 0.922 | 0.14 | 0.922 | 1.000 | 0.31 |
| final_sm_s1 | base | 1.76 | 4.4 | 13.5 | 9 | 0 | 0.880 | 0.14 | 0.880 | 1.000 | 0.36 |
| final_sm_s1 | bptt_clip_k=3 | 1.59 | 4.4 | 218.9 | 8 | 3 | 0.896 | 0.13 | 0.914 | 0.981 | 0.35 |
| final_sm_s1 | bptt_clip_k=10 | 1.72 | 3.9 | 94.1 | 5 | 1 | 0.897 | 0.18 | 0.898 | 0.999 | 0.33 |
| final_sm_s1 | wm_bptt_fp32=1 | 1.78 | 4.2 | 101.4 | 9 | 1 | 0.882 | 0.29 | 0.883 | 0.999 | 0.33 |
| final_st_s1 | base | 3.49 | 5.6 | 64.9 | 4 | 1 | 0.961 | 0.14 | 0.961 | 1.000 | 0.28 |
| final_st_s1 | bptt_clip_k=3 | 3.38 | 3.6 | 10.6 | 1 | 0 | 0.984 | 0.17 | 0.989 | 0.995 | 0.31 |
| final_st_s1 | bptt_clip_k=10 | 3.44 | 3.9 | 8.5 | 0 | 0 | 0.982 | - | 0.983 | 0.999 | 0.31 |
| final_st_s1 | wm_bptt_fp32=1 | 3.48 | 8.1 | 84.1 | 7 | 1 | 0.956 | 0.15 | 0.956 | 1.000 | 0.30 |
<!-- graddiag:end -->
