# NDM Gym

**Neural Dynamics Modeling Gym**: neural world models for physics-based humanoid control.

NDM Gym trains controllers that learn through a neural world model. Every algorithm shares one pipeline (Isaac Lab 3
with MuJoCo-Warp, the Unitree G1 with 29 DoF), so a comparison changes only the algorithm. The tasks are LAFAN1 motion
tracking, two G1s tracking a two-person clip, HDMI's human-object interaction and MuscleVAE's muscle-actuated character;
on the last two the neural dynamics models also stand in for the simulator.

- **Algorithms:** SuperTrack and SuperMimic, which both train the policy by backpropagation through the world model;
  BlinkMimic's recipe and systems options (`--algo blinkmimic`, and each change as an option of its own). See
  [Algorithms](algorithms.md).
- **World models:** SuperTrack's acceleration model and SuperMimic's increment model, each with an MLP or a RigidFormer
  backbone. See [World models](world_models.md).
- **Tasks:** motion tracking on LAFAN1; two G1s on a two-person clip; HDMI's 13 human-object interaction tasks, with
  contact forces predicted by the world model; MuscleVAE's character with 284 Hill-type muscles and fatigue. See
  [Tasks](tasks.md).
- **NDMs as simulators:** throughput and open-loop accuracy against the physics engine.
- **Evaluation:** tracking errors on every clip, push robustness, held-out clips, side-by-side videos, an offline
  world-model benchmark.

## Install

```bash
pip install "ndm_gym @ git+https://github.com/Frank-ZY-Dou/ndm-gym"   # the library: world models, policies, algorithms
```

```bash
git clone --recursive https://github.com/Frank-ZY-Dou/ndm-gym.git && cd ndm-gym
bash scripts/setup.sh                                                  # everything: Isaac Lab 3, the data
```

[Installation](installation.md) says what each gives; [Quickstart](quickstart.md) trains, evaluates and uses the world
models from Python.

## Layout

| Path | What |
|---|---|
| `ndm_gym/core/` | rotations, differentiable articulation FK, MLPs, running normalizers, Muon |
| `ndm_gym/robots/` | Unitree G1 29-DoF spec (`g1.py`); MuscleVAE's character and muscles (`musclevae_char.py`, `muscles.py`) |
| `ndm_gym/motion/` | motion library (50 Hz goals) and loaders (`sources/lafan1.py`, `hoi.py`, `interhuman.py`, `musclevae.py`) |
| `ndm_gym/sim/` | Isaac Lab 3 + MuJoCo-Warp tracking simulator; HDMI's scenes; MuscleVAE on Newton; Isaac Lab's launcher |
| `ndm_gym/tasks/` | `tracking.py` (SuperMimic's setup); `hdmi.py`, `hdmi_tasks.py` (HDMI's 13 tasks); `muscle_tracking.py` |
| `ndm_gym/worldmodels/` | `accel.py` (SuperTrack), `increment.py` (SuperMimic), `semi_implicit.py` (BlinkMimic), `rigidformer.py` (backbone), `hoi.py`, `muscle.py` |
| `ndm_gym/kernels/` | fused Triton kernels: the RigidFormer's blocks, the muscles' geometry |
| `ndm_gym/policies/` | tracking policy and critic; the HOI and muscle policies; MuscleVAE's released policy |
| `ndm_gym/algos/` | `common.py` (world-model fitting, imagination starts), `supertrack.py`, `supermimic.py`, `blinkmimic.py`, `eprobe.py`, `hoi_algos.py`, `muscle_algos.py`, `ppo_hoi.py` |
| `ndm_gym/wmbench/` | offline world-model benchmark; HOI and muscle datasets, fits and throughput |
| `ndm_gym/train.py`, `ndm_gym/evaluate.py` | `ndm-train`, `ndm-eval`; `train_hoi.py`, `train_ppo_hoi.py`, `train_muscles.py` for the other tasks |
| `scripts/`, `tools/` | install and launch scripts; data download and conversion, rendering, plots |
| `tests/` | equivalence tests against the upstream code, component tests |

Results, ablations and trained weights are in the [README](https://github.com/Frank-ZY-Dou/ndm-gym#-results) and on
[Hugging Face](https://huggingface.co/frankzydou/ndm-gym).

## Citation

```bibtex
@misc{ndmgym2026,
  title        = {NDM Gym: Neural Dynamics Modeling Gym},
  author       = {{NDM Gym Contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/Frank-ZY-Dou/ndm-gym}}
}
```

MIT license. The LAFAN1 motions, HDMI's and MuscleVAE's files are downloaded by the scripts in `tools/` and not
redistributed; see the [README](https://github.com/Frank-ZY-Dou/ndm-gym#-license-and-acknowledgments).
