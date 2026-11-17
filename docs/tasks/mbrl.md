# mbrl-lib's planner on a world model

A sampling planner scores hundreds of candidates at every step. On a simulator, each candidate's control step runs its
physics steps; on a world model it is one network evaluation for all candidates at once.

[mbrl-lib](https://github.com/facebookresearch/mbrl-lib) (MIT) ships a true-dynamics controller,
`mbrl/diagnostics/control_env.py`: CEM over action sequences, each candidate scored by rolling out the environment's own
MuJoCo model in a pool of CPU processes. Here its candidates are scored on this library's world model of the same MuJoCo
model (`ndm_gym.worldmodels.mjcf`) or on MuJoCo Warp. Every plan still executes on CPU MuJoCo, the controller's own
environment, and no file of mbrl-lib's is changed. These are results of mbrl-lib's controller, not of PETS, whose models
are learned ensembles.

| task | CPU MuJoCo, 16 processes | MuJoCo Warp | SuperTrack's model | SuperMimic's model | plans at the bar, fit seeds 0 / 1 / 2 |
|---|---|---|---|---|---|
| cart-pole | 1.566 s | 0.314 s | **0.0151 s** (103.8x / 20.8x) | **0.0150 s** (104.3x / 20.9x) | 10 / 10 / 10 and 10 / 10 / 10 of 10 |
| pusher | 0.950 s | 0.300 s | **0.0303 s** (31.3x / 9.9x) | **0.0286 s** (33.2x / 10.5x) | 10 / 8 / 10 and 8 / 10 / 10 of 10 |

PETS' cart-pole and pusher at `control_env.py`'s PETS settings: seconds per executed step, the median over planner seeds
0-2, on one A100 80GB SXM4; in brackets the speed-up over CPU MuJoCo / MuJoCo Warp.

- **World models:** SuperTrack's acceleration model and SuperMimic's increment model, MLPs fitted on the same recordings
  by the same rule, as bf16 copies whose steps are one CUDA graph each, timed in one session.
- **Plans at the bar:** a plan returns at least the lowest of the 20 simulator plans less 5 % of their mean's size
  (planner seeds 10-19, which no recording used).
- **Level:** the two models' steps are within 1.2 % at 4,096 worlds. On the pusher one model's planning time moves between
  runs by more than the two differ, in mbrl-lib's CEM work on the CPU.

The setup, the returns, the fits and the step's timing alone are in the
[results](https://github.com/Frank-ZY-Dou/ndm-gym/blob/main/benchmarks/mbrl_lib/results.md).

## Setup

mbrl-lib at `3f93ccc` runs in an environment of its own, with torch and this library importable.
`benchmarks/mbrl_lib/scripts/setup_env.sh` builds it with uv from the pinned versions the results were measured with
(Python 3.12, MuJoCo and MuJoCo Warp 4.0.1, Warp 1.19.0, torch 2.14.1), beside mbrl-lib's own environment (MuJoCo 2.3.3).

```bash
bash benchmarks/mbrl_lib/scripts/setup_env.sh "$PWD"
# the controller with its candidates on mbrl-lib's pool of CPU MuJoCo processes, on MuJoCo Warp and on a fit of the world model
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole --setting pets --own_mujoco --sim cpu --num_processes 16 --out plan_cpu.json
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole --setting pets --own_mujoco --sim warp --out plan_warp.json
python -m ndm_gym.wmbench.plan_mbrl --env pets_cartpole --setting pets --own_mujoco --sim ndm --ndm fit.pt --fused --out plan_ndm.json
```

`--own_mujoco` steps the models as mbrl-lib's own MuJoCo steps them. Hopper and Walker2d need local-coordinate copies of
gymnasium's models, made once in mbrl-lib's own environment (`python tools/convert_mbrl_models.py`, then
`--local_models data/mbrl_models`); `setup_env.sh` makes them.

## The pipeline

| step | module | what it does |
|---|---|---|
| record | `ndm_gym.wmbench.collect_mbrl` | the CPU planner's candidate rollouts at recorded plan starts and its executed trajectory, replayed on CPU MuJoCo |
| fit | `ndm_gym.wmbench.fit_mjcf` | a world model of the environment's MuJoCo model on the recordings, one step per control step (`--substeps`): SuperTrack's accelerations (`--wm_param accel`) or SuperMimic's increments (`--wm_param increment`) |
| plan | `ndm_gym.wmbench.plan_mbrl` | `control_env.py`, unmodified, with `--sim cpu`, `warp` or `ndm` (the fit of `--ndm`) |
| time | `ndm_gym.wmbench.speed_mbrl` | the model's control step alone on CPU MuJoCo, on MuJoCo Warp and on the world model, from recorded states |
| check | `ndm_gym.wmbench.check_mbrl_rewards` | the batched rewards and terminations against the environments' own step |

`ndm_gym.sim.mbrl_envs` builds the environments as `control_env.py` does, and `ndm_gym.sim.mbrl_ndm` runs the batched
rollouts on MuJoCo Warp and on the world model with the environments' rewards and terminations. The commands for both
tasks and both world models are in the results'
[Reproducing](https://github.com/Frank-ZY-Dou/ndm-gym/blob/main/benchmarks/mbrl_lib/results.md#reproducing).
