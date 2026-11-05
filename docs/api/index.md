# API reference

Generated from the docstrings of the modules that import with the library alone (`pip install ndm_gym`: PyTorch, NumPy,
MuJoCo). The layers build on each other: `core` → `robots` → `motion` → `sim` → `tasks` → `worldmodels` → `policies` →
`algos` → `train` / `evaluate`.

| | |
|---|---|
| [Core](core.md) | rotations, differentiable kinematics, networks, normalizers, Muon |
| [Robots and state](robots.md) | the G1's specification; the state layout |
| [Motion](motion.md) | the motion library and its sources |
| [Tasks](tasks.md) | LAFAN1 tracking: observations, reward, terminations |
| [World models](worldmodels.md) | SuperTrack's, SuperMimic's, BlinkMimic's; the RigidFormer; inference copies; HOI and muscles |
| [Policies](policies.md) | the tracking policy and critic; the HOI and muscle policies |
| [Algorithms](algos.md) | the algorithm interface, SuperTrack, SuperMimic, BlinkMimic's options |
| [World-model benchmark](wmbench.md) | offline fits and scores on collected datasets |
| [Simulator](sim.md) | `SimulatorNotInstalled`; Isaac Lab's launcher |

## What each module needs

Every module imports with the library alone, except:

| needs | modules |
|---|---|
| the simulator (Isaac Lab 3, Newton, MuJoCo-Warp: `scripts/setup.sh`) | `ndm_gym.sim.rigid.isaaclab_mjwarp`, `ndm_gym.sim.rigid.hdmi_sim` (they raise `SimulatorNotInstalled`), `ndm_gym.sim.rigid.muscle_sim` (Newton and MuJoCo-Warp), and the modules that import them at the top: `ndm_gym.algos.hoi_algos`, `ndm_gym.train_hoi` |
| Triton (PyTorch's CUDA builds for Linux) and a CUDA GPU | `ndm_gym.kernels.*`, `ndm_gym.core.token_attention`, `ndm_gym.worldmodels.g1_fused`, `ndm_gym.worldmodels.hoi_fused`: the fused kernels, imported where an option asks for them |

The commands that run the simulator (`ndm_gym.train`, `ndm_gym.evaluate`, `ndm_gym.wmbench.collect`, ...) import
without it, so that their arguments and their simulator-free parts can be used (`ndm_gym.wmbench.fit` builds its
arguments from `ndm_gym.train`'s); running them raises `SimulatorNotInstalled`. `tests/test_packaging.py` checks this
list.
