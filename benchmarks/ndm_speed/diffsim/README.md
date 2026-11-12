# Differentiable simulators against the G1 world models: the benchmark scripts

The scripts behind the `ndm_speed` results' differentiable-simulator section: the G1 forward step and the gradient of a
32-step rollout's loss, at 1,024, 4,096 and 16,384 worlds, for NablaEngine, MuJoCo Warp with the adjoint of its pull
request #1535, and MJX, each as published and tuned. One process per measurement; every line goes to one CSV.

| script | what it times | environment |
|---|---|---|
| `bench_nablaengine.py` | NablaEngine v0.2.0 (`be48f14`), forward and its tape plus the adjoint | NablaEngine's own (`scripts/setup_env.sh`) |
| `bench_mjwarp_adjoint.py` | MuJoCo Warp with PR #1535 (`etaoxing/mujoco_warp` at `4af093f`; as published: `357a75d`) | mujoco_warp from that checkout, editable; mujoco 3.17.0, warp-lang 1.18.0 |
| `bench_mjx.py` | MJX-JAX, forward and `jax.grad` through the rollout | mujoco-mjx 3.17.0 with jax 0.12.0.1 (the benchmark model is a MuJoCo 3.17.0 binary) |
| `common.py` | the shared model, states, loss and CSV columns | |
| `check_workload_cpu.py` | that every simulator loads the same model and loss, on the CPU | any of the above |
| `run_matrix.sh` | the whole matrix on one GPU | sources the environments |
| `run_paper_bench.sh` | NablaEngine's own timing kit, its scripts unmodified | the kit's environments |

The world models' rows come from `ndm_gym.wmbench.speed_pair` in this library's environment.

Run, with `DIFFSIM_ROOT` the folder that holds `env.sh` (the caches and `HOME` inside it) and each simulator's environment
with an `activate.sh`:

```bash
DIFFSIM_ROOT=/path/to/diffsim bash run_matrix.sh 0 results/matrix_a100.csv "nablaengine mjwarp mjx" "1024 4096 16384" "as_published tuned"
```

The first run of each simulator compiles: NablaEngine's library is prebuilt; MuJoCo Warp's kernels take a few minutes
and are then cached; MJX compiles every program, minutes each, and does not cache them.
