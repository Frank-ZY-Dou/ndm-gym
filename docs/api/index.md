# API reference

Generated from docstrings of modules that import with the library's `pip install` (PyTorch, NumPy, MuJoCo).
Dependency layers: `core` → `robots` → `motion` → `sim` → `tasks` → `worldmodels` → `policies` →
`algos` → `train` / `evaluate`.

| | |
|---|---|
| [Core](core.md) | rotations, differentiable kinematics, networks, normalizers, Muon |
| [Robots and state](robots.md) | the G1's specification; the state layout |
| [Motion](motion.md) | the motion library and its sources |
| [Tasks](tasks.md) | LAFAN1 tracking: observations, reward, terminations |
| [World models](worldmodels.md) | SuperTrack's, SuperMimic's, BlinkMimic's; the RigidFormer; inference copies; HOI, muscles and the hand |
| [Policies](policies.md) | the tracking policy and critic; the HOI, muscle and hand policies |
| [Algorithms](algos.md) | the algorithm interface, SuperTrack, SuperMimic, BlinkMimic's options |
| [World-model benchmark](wmbench.md) | offline fits and scores on collected datasets |
| [Simulator](sim.md) | `SimulatorNotInstalled`; Isaac Lab's launcher |

## What each module needs

Every module imports with the library alone, except:

| needs | modules |
|---|---|
| the simulator (Isaac Lab 3, Newton, MuJoCo-Warp: `scripts/setup.sh`) | `ndm_gym.sim.rigid.isaaclab_mjwarp`, `ndm_gym.sim.rigid.hdmi_sim` (they raise `SimulatorNotInstalled`), `ndm_gym.sim.rigid.muscle_sim` (Newton and MuJoCo-Warp), `ndm_gym.sim.rigid.hand_sim` (MuJoCo-Warp), and the modules that import them at the top: `ndm_gym.algos.hoi_algos`, `ndm_gym.train_hoi` |
| Triton (PyTorch's CUDA builds for Linux) and a CUDA GPU | `ndm_gym.kernels.*`, `ndm_gym.core.token_attention`, `ndm_gym.worldmodels.g1_fused`, `ndm_gym.worldmodels.hoi_fused`: the fused kernels, imported where an option asks for them |

Simulator commands (`ndm_gym.train`, `ndm_gym.evaluate`, `ndm_gym.wmbench.collect`, ...) import without the simulator,
exposing arguments and simulator-free functions. For example, `ndm_gym.wmbench.fit` uses `ndm_gym.train`'s arguments.
Running these commands requires the simulator or raises `SimulatorNotInstalled`. `tests/test_packaging.py` checks this list.
