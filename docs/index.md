# NDM Gym

**Neural Dynamics Modeling Gym**: neural world models for controlling physical agents.

NDM Gym trains controllers of physical agents through neural world models. Algorithms share one pipeline within each task: LAFAN1 motion
tracking, two G1s on a two-person clip, HDMI's human-object interaction (Isaac Lab 3 with MuJoCo-Warp, the Unitree G1 with 29 DoF) and
MuscleVAE's muscle-actuated character (Newton with MuJoCo-Warp) and hand-object tracking (MuJoCo-Warp). The neural dynamics models (NDMs) also serve as simulators.

- [Algorithms](algorithms.md): SuperTrack, SuperMimic and BlinkMimic's recipe (`--algo blinkmimic`).
- [World models](world_models.md): acceleration and increment models, MLP and RigidFormer backbones, inference copies.
- [Tasks](tasks.md): LAFAN1 tracking, two G1s, HDMI's 13 human-object interaction tasks and MuscleVAE's character.
- [Evaluation](quickstart.md): tracking errors, push robustness, held-out clips, videos and offline world-model fits.

## Install

```bash
pip install "ndm_gym @ git+https://github.com/Frank-ZY-Dou/ndm-gym"   # the library: world models, policies, algorithms
```

```bash
git clone --recursive https://github.com/Frank-ZY-Dou/ndm-gym.git && cd ndm-gym
bash scripts/setup.sh                                                  # everything: Isaac Lab 3, the data
```

See [installation options](installation.md) and the [Quickstart](quickstart.md) for training, evaluation and Python usage.

## Layout

| Path | What |
|---|---|
| `ndm_gym/core/` | rotations, differentiable articulation FK, MLPs, running normalizers, Muon |
| `ndm_gym/robots/` | Unitree G1 29-DoF spec (`g1.py`); MuscleVAE's character and muscles (`musclevae_char.py`, `muscles.py`); XHand (`xhand.py`) |
| `ndm_gym/motion/` | motion library (50 Hz goals) and loaders (`sources/lafan1.py`, `hoi.py`, `interhuman.py`, `musclevae.py`, `spider.py`) |
| `ndm_gym/sim/` | Isaac Lab 3 + MuJoCo-Warp tracking simulator; HDMI's scenes; MuscleVAE on Newton; a hand and an object on MuJoCo-Warp; Isaac Lab's launcher |
| `ndm_gym/tasks/` | `tracking.py` (SuperMimic's setup); `hdmi.py`, `hdmi_tasks.py` (HDMI's 13 tasks); `muscle_tracking.py`; `hand_tracking.py` |
| `ndm_gym/worldmodels/` | `accel.py` (SuperTrack), `increment.py` (SuperMimic), `semi_implicit.py` (BlinkMimic), `rigidformer.py` (backbone), `hoi.py`, `muscle.py`, `hand.py` |
| `ndm_gym/kernels/` | fused Triton kernels: the RigidFormer's blocks, the muscles' geometry |
| `ndm_gym/policies/` | tracking policy and critic; the HOI, muscle and hand policies; MuscleVAE's released policy |
| `ndm_gym/algos/` | `common.py` (world-model fitting, imagination starts), `supertrack.py`, `supermimic.py`, `blinkmimic.py`, `eprobe.py`, `hoi_algos.py`, `muscle_algos.py`, `hand_algos.py`, `ppo_hoi.py` |
| `ndm_gym/wmbench/` | offline world-model benchmark; HOI and muscle datasets, fits and throughput |
| `ndm_gym/train.py`, `ndm_gym/evaluate.py` | `ndm-train`, `ndm-eval`; `train_hoi.py`, `train_ppo_hoi.py`, `train_muscles.py`, `train_hand.py` for the other tasks |
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
