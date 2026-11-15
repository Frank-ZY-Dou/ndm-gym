# XMimic: why it is fast, and its techniques as options

[XMimic](https://github.com/nico-bohlinger/xmimic) (N. Bohlinger, J. Peters, *XMimic and XSoma: Learning Humanoid Motion
Tracking and Locomotion in Under a Minute*, 2026; MIT;
[paper](https://nico-bohlinger.github.io/papers/2026_xmimic_xsoma.pdf)) is [BlinkMimic](../blinkmimic.md)'s trainer rebuilt
in JAX: the same algorithm, hyperparameters, task and evaluation, and other machinery. This page says where its speed comes
from (section 1), how much of each part carries over to this library's PyTorch pipeline, measured (section 2), and what
the options are that hold its techniques here (sections 3 and 4): options of `ndm-train`, all off by default
(`ndm_gym/algos/xmimic.py`, `ndm_gym/algos/layouts.py`). The algorithm itself is `--algo blinkmimic`.

## 1. Why it is fast

An iteration collects 2.5 control steps of 4,096 worlds, fits the world model 5 times and updates the policy by
backpropagating through 32 imagined steps. That is a long chain of small GPU operations that depend on each other, so its
time follows the number of operations more than the GPU's compute. XMimic shortens the chain and runs its parts side by
side. Its published runs (its README at `54860e1`; H100 NVL GPUs; the target is BlinkMimic's, a return of at least 0.82
with no failed clip, held for three checkpoints):

| | GPUs | iterations per second | minutes to the first 0.82 | minutes to the target | GPU-minutes to the target |
|---|---|---|---|---|---|
| BlinkMimic | 1 | 9.1 | 2.25 | 3.19 | 3.2 |
| XMimic, one GPU | 1 | 12.0 | 1.71 | 2.47 | 2.5 |
| probe sweeps in a second process on the same GPU (FP1) | 1 | 11.8 | 1.83 | 2.42 | 2.4 |
| probe sweeps on a GPU of their own (FP) | 2 | 16.0 | 1.59 | 2.10 | 4.2 |
| actor \| learner (AL) | 2 | 25.2 | 0.93 | 1.39 | 2.8 |
| actor \| learner \| prober (ALP) | 3 | 25.8 | 0.90 | 1.28 | 3.8 |
| actor \| world model \| policy (AWL) | 3 | 36.4 | 0.71 | 1.09 | 3.3 |
| actor \| world model \| policy \| prober (AWLP) | 4 | 38.8 | 0.70 | 0.86 | 3.4 |

Four things make the difference:

1. **The iteration's steps side by side on several GPUs** (12.0 → 25.2 → 36.4 iterations per second). On one GPU an
   iteration takes the sum of its steps: collection, world-model updates, policy update. With an actor, a world-model
   learner and a policy learner on GPUs of their own it takes about the longest of them. Each works on the others'
   results of the previous iteration: the actor collects with a policy one iteration old, and the policy learns in a
   world model one or two iterations old. The time to the target falls from 2.47 minutes to 1.39 on two GPUs, 1.09 on
   three and 0.86 on four. The GPU time does not fall: 3.4 GPU-minutes on four GPUs against 2.5 on one. On four GPUs
   the actor's physics steps (about 24 ms) and the policy update (about 23 ms) fill the 25 ms iteration, so from there
   on the simulator and the policy update are the limit.
2. **A shorter chain on one GPU** (9.1 → 12.0 iterations per second: the same algorithm 1.3 times faster). Whole
   iterations run as compiled programs, 50 at a time, and nothing returns to Python in between. In the policy update,
   what does not depend on the imagined state is computed once before the 32-step rollout, the weight gradients are
   one matrix product per layer over all steps instead of one per step, Muon orthogonalizes matrices of equal shape
   together, and matrix products return bf16, so that the backward pass runs in bf16 too (its policy update on an H100:
   41.6 → 29.9 ms). The training simulator keeps 256 constraint rows per world instead of 500 (7 % per physics step).
3. **The probe sweeps off the trainer's GPU.** BlinkMimic's evaluation-guided resets replay the averaged policy on all
   clips every 125 iterations, and on one GPU a sweep halves the training speed while it runs. On a GPU of their own
   the sweeps no longer slow the trainer (12.0 → 16.0 iterations per second over a run); a second process on the same
   GPU changes little (11.8).
4. **The host out of the way.** The actor in its own thread, the collection loops' trip counts known to the host,
   Python's garbage collector frozen after the setup: together 20 → 39-41 iterations per second on four GPUs.

Its authors note that the kernel-level gains target Hopper GPUs and may be smaller on others.

## 2. What carries over to this library

This library's `--algo blinkmimic` is BlinkMimic's recipe in PyTorch. Its collection and its updates already run as
captured CUDA graphs of compiled step functions, with bf16 products under autocast. Measured on LAFAN1 on one node of
A100 40GB SXM4 GPUs with nothing else running; XMimic's own code (`54860e1`, unmodified, with its published configuration
and its own evaluator) ran on the same node.

Time per iteration (seed 0, no evaluation; milliseconds from the training log's wall clock). The learner's options are added
one group at a time, each row on top of the one above it:

| options | GPUs | ms per iteration, iterations 200-1000 | iterations 1000-1500, probe sweeps running |
|---|---|---|---|
| the recipe | 1 | 160.8 | 227.3 |
| `--pi_encode_once` | 1 | 150.2 (-6.6 %) | |
| and `--imag_hoist` | 1 | 150.0 | |
| and `--pi_deferred_grads --wm_deferred_grads` | 1 | 163.8 | |
| and `--muon_batched` | 1 | 163.4 | |
| and `--wm_presample` | 1 | 161.3 | |
| and `--gc_freeze`: the learner's whole set | 1 | 161.2 | 227.9 |
| the set and `--njmax 256` | 1 | 159.2 | |
| the recipe with `--eprobe_gpu 1` | 2 | 160.6 | 180.1 |

Training time to return 0.82 (seeds 0-2: minutes from iteration 50 to the first checkpoint whose four evaluations average
at least 0.82; GPU-minutes are those minutes times the GPUs the run trains on; the layout runs use `--gc_freeze`):

| | GPUs | ms per iteration, iterations 200-1000 | iterations to 0.82 | minutes to 0.82 | GPU-minutes |
|---|---|---|---|---|---|
| the recipe | 1 | 160.8 | 1,400-1,600 | 4.48 ± 0.37 | 4.48 ± 0.37 |
| the learner's whole set | 1 | 161.1 | 1,500-1,600 | 4.60 ± 0.25 | 4.60 ± 0.25 |
| `--layout AL` | 2 | 86.8 | 1,500-1,800 | 2.89 ± 0.38 | 5.79 ± 0.77 |
| `--layout AL --eprobe_gpu 2` | 3 | 86.9 | 1,500-1,600 | 2.48 ± 0.14 | 7.44 ± 0.41 |
| `--layout AWL` | 3 | 87.1 | 1,500-1,600 | 2.69 ± 0.15 | 8.07 ± 0.46 |
| `--layout AWL --eprobe_gpu 3` | 4 | 87.2 | 1,400-1,700 | 2.57 ± 0.34 | 10.29 ± 1.36 |
| `--layout AL --eprobe_gpu 2 --pi_encode_once` | 3 | 80.4 | 1,500-1,700 | 2.41 ± 0.26 | 7.23 ± 0.77 |
| XMimic's own code, one GPU | 1 | 138.9 | 1,400-1,500 | 3.53 ± 0.15 | 3.53 ± 0.15 |
| XMimic's own code, `AWLP` | 4 | 74.2 | 1,400-1,600 | 1.81 ± 0.13 | 7.25 ± 0.52 |

XMimic's rows stand beside the others and are not ranked with them: the evaluators, the simulators, the constraint rows
per world in training (256 against 500) and the clocks differ.

What each of the four does here:

1. **The split carries over.** With the actor and the policy learner on two GPUs an iteration takes 86.8 ms instead of
   160.8, and the staleness costs few iterations (0.82 after 1,500-1,800 against 1,400-1,600). Every layout reaches 0.82
   1.5 to 1.9 times sooner than the recipe, at 1.3 to 2.3 times its GPU-minutes. A third GPU for the world model does
   not shorten the iteration here (87.1 ms): the policy update alone takes about that long. `--pi_encode_once` shortens
   it (80.4 ms).
2. **The shorter chain carries over in part.** Encoding the goals of all imagined steps once takes 6.6 % off an
   iteration. Hoisting the rest of the step-independent work, batching Muon and drawing an iteration's windows in one
   pass change it by about 2 ms or less. The deferred gradients add 13.8 ms: under `--amp bf16` their operands are bf16
   and the product is taken in fp32, as torch has no bf16 product with an fp32 result. The learner's whole set ends
   level with the recipe, per iteration (161.2 against 160.8 ms) and in the time to 0.82 (4.60 ± 0.25 against
   4.48 ± 0.37 min). The reason is the starting point: every update here already is one captured CUDA graph with bf16
   products, so most of what the restructuring removes from a direct translation is not there to remove. XMimic's own
   one-GPU program still has the shorter iteration (138.9 ms against 150.2 with `--pi_encode_once`). A per-phase
   profile of both codes on this GPU type puts that difference in the collection step, not in the updates: its
   program steps MuJoCo Warp on MuJoCo's own data, with 256 constraint rows per world, where this library steps it
   through Isaac Lab on Newton, with a state conversion at every substep and 500 rows. `--njmax 256` takes 1.2 % off
   here and changes results.
3. **The sweeps off the trainer's GPU carry over.** While sweeps run, an iteration takes 227.3 ms on one GPU and 180.1
   with the probe worker on a second (`--eprobe_gpu`). `--eprobe_lean`, this library's own, makes the sweep step itself
   cheaper by leaving out what a sweep does not read.
4. **The host.** `--gc_freeze` freezes the garbage collector; the layout runs use it.

In short, on this task: with two to four GPUs, splitting the iteration is the largest gain (0.82 in 2.4-2.9 minutes
instead of 4.5), and it is paid in GPU time. On one GPU, what carries over is encoding the goals once (6.6 %) and, with
a second GPU free, the probe sweeps off the trainer's. The rest of the one-GPU difference to XMimic's own code is in how
the simulator is stepped, which no option here changes.

## 3. The options

| option | what it does | in XMimic | results |
|---|---|---|---|
| `--layout AL`, `--layout AWL` | the iteration's steps at the same time on two or three GPUs, each on the others' results of the previous iteration ([section 4](#4-the-iteration-split-across-gpus)) | its layouts AL and AWL; with `--eprobe_gpu`, ALP and AWLP | change: the policy update's batch and world model are one iteration older |
| `--eprobe_gpu i` | the probe sweeps of the evaluation-guided resets on the i-th GPU the trainer sees, not on the trainer's | the prober GPU of its layouts FP, ALP and AWLP | the same sweeps |
| `--njmax 256` | 256 constraint rows per world in the training simulator instead of the scene's 500; the evaluation and the probes keep 500 | `config.py` (`njmax`) | change: a world that needs more rows loses the rest for that step |
| `--gc_freeze` | Python's garbage collector frozen once the run is set up | `runtime.py` (`freeze_gc`) | host time only |
| `--pi_encode_once` | the policy's goal encoder once over the goals of all imagined steps; SuperTrack had it, SuperMimic has it now | `algorithm.py` and `fast.py` (`imagine`) | the default's values up to rounding |
| `--imag_hoist` (with `--pi_encode_once`) | what does not depend on the imagined state is computed before the rollout: the reference's terms of all frames and the noise of all steps; each observation is normalized once, when it is produced | `fast.py` (`imagine`) | the noise is one draw per kind instead of one per step (another random stream); with the same noise, the default's values and gradients up to rounding |
| `--pi_deferred_grads` (with `--imag_hoist`), `--wm_deferred_grads`; `--deferred_bf16` | weight gradients as one matrix product per layer over all steps, formed after the backward pass: the policy's decoder, the world model | `fast_nets.py` (`dense`, `deferred_grads`) | the same forward pass, the gradients up to rounding. Under `--amp bf16` the products' operands are bf16 and the product is taken in fp32; `--deferred_bf16` takes it in bf16 |
| `--muon_batched` | Muon's Newton-Schulz iteration batched over matrices of equal shape | `fast_nets.py` (`newton_schulz_batched`) | equal up to the batched products' rounding |
| `--wm_presample` | the windows of an iteration's world-model updates sampled and mirrored in one pass; with `--update_graphs` the iteration's updates are one CUDA graph | `fast.py` (`wm_updates`) | the same windows, bit for bit while the fit draws no random numbers itself |

The learner's options (`--imag_hoist`, `--pi_deferred_grads`, `--wm_deferred_grads`, `--deferred_bf16`, `--muon_batched`,
`--wm_presample`) apply to SuperTrack's and SuperMimic's updates with the MLP world model on the one-robot scene. They are
refused with the RigidFormer world model, with `--paired`, and with `--wm_cudagraphs` or `--pi_cudagraphs` (`--update_graphs`
captures whole updates). `--pi_encode_once` is the policy's alone and runs with either world model.

This library's own, not from XMimic: `--eprobe_lean`. A probe sweep's step then computes what the sweep reads (the policy,
the simulator's step, the termination test) and not the step's reward and flattened states; the sweeps record the same
failures.

```bash
CUDA_VISIBLE_DEVICES=0,1 ndm-train --algo blinkmimic --eprobe_gpu 1 --eprobe_lean
ndm-train --algo blinkmimic --pi_encode_once --imag_hoist --muon_batched --wm_presample --gc_freeze
```

## 4. The iteration split across GPUs

`--layout` (default `1GPU`: the loop as it is) runs an iteration's steps at the same time on the first GPUs the trainer
sees, each on the others' results of the previous iteration. It needs the systems options `--no_sync --update_graphs
--graph_early --collect_graph --reset_masked`, which `--algo blinkmimic` has, and covers SuperMimic and SuperTrack with the
MLP world model. `--paired`, the RigidFormer, `--resume`, `--pi_updates` other than 1 and the learner's options of section 3
are refused.

| | GPU 0: actor | GPU 1 | GPU 2 |
|---|---|---|---|
| `AL` | simulator, collection, replay buffer, world model with its updates, draw of the next imagination batch | policy learner: policy update, policy average, critic | |
| `AWL` | simulator, collection, replay buffer, draw of the next imagination batch | world-model learner: the world-model updates, on a replica of the replay buffer | policy learner |

```bash
CUDA_VISIBLE_DEVICES=0,1 ndm-train --algo blinkmimic --layout AL                        # actor | policy learner
CUDA_VISIBLE_DEVICES=0,1,2 ndm-train --algo blinkmimic --layout AL --eprobe_gpu 2       # the probe worker on a third GPU (XMimic's ALP)
CUDA_VISIBLE_DEVICES=0,1,2,3 ndm-train --algo blinkmimic --layout AWL --eprobe_gpu 3    # XMimic's AWLP
```

One process and one host thread. Every device has an object of the run's algorithm; the collection, the world-model update
and the policy update are the CUDA graphs of the one-GPU loop, each captured on its device.

What iteration i reads (where an entry is not the one-GPU loop's, that loop reads iteration i's own):

| iteration i's | reads | `AL` | `AWL` |
|---|---|---|---|
| collection | the policy | as iteration i - 1's update left it, as on one GPU | the same |
| world-model updates | the replay | up to iteration i's collection, as on one GPU | up to iteration i - 1's collection |
| | the world model's normalizers | iteration i's, as on one GPU | iteration i - 1's |
| policy update | the imagination batch | drawn at the end of iteration i - 1 | the same |
| | the world model | as iteration i - 1's updates left it | as iteration i - 1's updates left it, which saw the replay up to iteration i - 2's collection |
| | the policy's normalizers | iteration i - 1's | the same |
| | the critic | its own, as on one GPU | the same |

The first iteration has no batch yet and makes no policy update. This is XMimic's staleness. At the end of every iteration
the batch, the normalizers, the world model and the policy's weights cross the devices as in-place copies into the tensors
the other device's graphs read; the copies are exact, so the actor's policy is the learner's bit for bit. The probe sweeps
run in the worker process, on the actor's GPU or on `--eprobe_gpu`'s; as on one GPU, the trainer applies a sweep's failures
at BlinkMimic's iteration and waits there if the worker is not done.

## 5. Tests

- `tests/test_xmimic.py`: the options are off by default and refused where they do not apply; the probe worker starts on
  the chosen one of the trainer's GPUs; `--njmax` sets the training simulator's rows.
- `tests/test_xmimic_learner.py`: each learner option against the default updates on the CPU, in eager mode and fp32; under
  bf16 autocast, with the updates compiled, and in a short training run on a GPU.
- `tests/test_layouts.py`: the split loop gives, bit for bit, the networks of a one-object loop with the previous
  iteration's batch, world model and normalizers swapped in by hand, on the CPU; on two and three GPUs with the updates as
  CUDA graphs.
- `tests/test_eprobe.py`: with `--eprobe_lean` a sweep records the same failures.
