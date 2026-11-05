# Installation

Install the library with pip, or include the simulator and data with the setup script.

| | `pip install` | `scripts/setup.sh` |
|---|---|---|
| world models (MLP, RigidFormer), policies, algorithms, kinematics, in Python | yes | yes |
| offline world-model fits on collected datasets (`ndm_gym.wmbench.fit`) | yes, with the data and a CUDA GPU | yes |
| training, evaluation, data collection (`ndm-train`, `ndm-eval`, `ndm_gym.wmbench.collect`) | no | yes |
| Isaac Lab 3, Newton and MuJoCo-Warp at the tested versions | no | yes |
| the G1 description and the LAFAN1 motions in `data/` | no (`tools/download_data.sh`) | yes |

## The library

Python 3.12 or newer:

```bash
pip install "ndm_gym @ git+https://github.com/Frank-ZY-Dou/ndm-gym"
```

This installs PyTorch (2.9 or newer), NumPy and MuJoCo for MJCF kinematics. On Linux, pip uses PyTorch's CUDA build from
PyPI. For a CPU-only install, install PyTorch from its CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "ndm_gym @ git+https://github.com/Frank-ZY-Dou/ndm-gym"
```

From a clone, `pip install -e .` enables editable installation with the repository's `data/`:

```bash
git clone --recursive https://github.com/Frank-ZY-Dou/ndm-gym.git && cd ndm-gym
pip install -e ".[test]"
bash tools/download_data.sh --no-usd    # the G1 description (BSD-3-Clause) and the LAFAN1 motions (CC BY-NC-ND 4.0)
```

`--no-usd` skips USD conversion, which needs Isaac Lab and is used only by the simulator. G1 kinematics, RigidFormer
and the motion library read `data/g1_description/g1_29dof.xml`. Without a clone, supply a path to
`ArticulationKinematics` instead of using `data/` (see
[Quickstart](quickstart.md#the-rigidformer)).

Optional dependencies:

| extra | installs | for |
|---|---|---|
| `ndm_gym[render]` | Pillow, matplotlib | the rendering and plotting scripts in `tools/` (they also call ffmpeg) |
| `ndm_gym[sim]` | MuJoCo-Warp, Warp, Newton and MuJoCo at the versions of Isaac Lab's lock | the simulator's packages that pip can install; Isaac Lab 3 itself comes from `scripts/setup.sh` |
| `ndm_gym[test]` | pytest, Pillow, PyYAML | `tests/` |
| `ndm_gym[docs]` | MkDocs Material, mkdocstrings | this site: `mkdocs serve` |
| `ndm_gym[dev]` | test, docs, ruff, pre-commit | development |

Fused GPU kernels (`ndm_gym.kernels`, the world models' inference copies) need a CUDA GPU and Triton, included in
PyTorch's CUDA builds for Linux.

Without the simulator, the modules and commands that need it raise `ndm_gym.sim.SimulatorNotInstalled`:

```console
$ ndm-train --algo supertrack
ndm_gym.sim.SimulatorNotInstalled: ndm-train needs Isaac Lab 3 with Newton and MuJoCo-Warp: run scripts/setup.sh, which
installs them (a pip install of this repository gives the library without the simulator). Not installed: isaaclab,
isaaclab_newton, newton, warp, mujoco_warp
```

## Everything

Linux with an NVIDIA GPU (driver for CUDA 12.6 or newer):

```bash
git clone --recursive https://github.com/Frank-ZY-Dou/ndm-gym.git && cd ndm-gym
bash scripts/setup.sh                 # Isaac Lab 3 and locked dependencies into ~/ndm_gym/venv, motions and G1 into data/
source ~/ndm_gym/venv/bin/activate    # ndm-train and ndm-eval on PATH
```

`scripts/setup.sh` (set `WS` to install elsewhere than `~/ndm_gym`):

1. installs uv, clones Isaac Lab 3 at the pinned commit and installs it with its locked dependencies into `$WS/venv`;
2. replaces torch, torchvision and warp by the CUDA 12 builds of the same versions (the lock pins CUDA 13 builds, which
   need driver 580 or newer);
3. downloads the LAFAN1 motions and the G1 description into `data/`, checks them against `tools/download_data.sha256`
   and converts the G1 to MJCF and USD (`tools/download_data.sh`);
4. installs `ndm_gym` editable, with the test extras.

The script stops at the first failure and can be rerun. See [Tasks](tasks.md) for other data downloads.

## Checks

```bash
make lint         # ruff, as in CI
make test-fast    # the tests without those marked slow; they read data/, the simulator's tests need a GPU and Isaac Lab
make docs         # this site, into site/ (pip install -r docs/requirements.txt)
```
