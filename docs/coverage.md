# What NDM Gym covers

Every method, simulator, dataset and robot model this library implements, runs, accelerates or builds on, by name with its
link, one line each; the papers are under "Papers and repositories" in the README. Entries marked *(ongoing)* are not in
this repository yet.

## Methods

| Method | Link | Here |
|---|---|---|
| SuperTrack | [paper](https://doi.org/10.1145/3478513.3480527) | implemented |
| SuperMimic | [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) | implemented |
| BlinkMimic | [fuyichen0725/blinkmimic](https://github.com/fuyichen0725/blinkmimic) | its recipe as options |
| InstantTrack | [Huy-Quang-Dao/instantTrack](https://github.com/Huy-Quang-Dao/instantTrack) | BlinkMimic's base |
| XMimic | [nico-bohlinger/xmimic](https://github.com/nico-bohlinger/xmimic) | its changes as options; its own code timed |
| ∇Mimic (NablaMimic) | [sailab-ai/NablaMimic](https://github.com/sailab-ai/NablaMimic) | its own code timed |
| RWM | [leggedrobotics/rsl_rl_rwm](https://github.com/leggedrobotics/rsl_rl_rwm), [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model), [robotic_world_model_lite](https://github.com/leggedrobotics/robotic_world_model_lite) | its code through `--algo rwm` *(ongoing)* |
| RigidFormer | [paper](https://arxiv.org/abs/2605.09196) | world-model backbone |
| NeRD | [NVlabs/neural-robot-dynamics](https://github.com/NVlabs/neural-robot-dynamics) | NeRD-style world model *(ongoing)* |
| HDMI | [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI) | its 13 human-object interaction tasks |
| MuscleVAE | [wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_) | its muscle-actuated character |
| SPIDER | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object clips and scenes |
| RoboKit | [openrobokit/robokit](https://github.com/openrobokit/robokit) | its planner on world models |
| mbrl-lib | [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | its controller on world models, at PETS' settings, and its algorithms (PETS, MBPO) *(ongoing)* |
| MimicKit | [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) | the framework whose methods run on world models *(ongoing)* |
| DeepMimic | [xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic) | MimicKit's tracking method *(ongoing)* |
| AMP | [paper](https://doi.org/10.1145/3450626.3459670), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| AWR | [paper](https://arxiv.org/abs/1910.00177), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| ASE | [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE), in MimicKit | accelerated with SuperTrack *(ongoing)* |
| C·ASE | [Frank-ZY-Dou/CASE](https://github.com/Frank-ZY-Dou/CASE) | ported to MimicKit; accelerated with SuperTrack *(ongoing)* |
| LCP | [paper](https://arxiv.org/abs/2410.11825), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| ADD | [paper](https://doi.org/10.1145/3757377.3763819), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| CFC | [project page](https://people.csail.mit.edu/frankzydou/projects/CFC/) | two-level world model with forces as the interface *(ongoing)* |
| Eden | [embodied-ai-nexus/Eden](https://github.com/embodied-ai-nexus/Eden) | its tasks on world models *(ongoing)* |

## Simulators and engines

| Simulator | Link | Here |
|---|---|---|
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | tracking simulator |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | models, CPU simulation |
| MuJoCo Warp | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | GPU simulation; the baseline of the speed comparisons |
| Newton | [newton-physics/newton](https://github.com/newton-physics/newton) | MuscleVAE's simulation; MimicKit's *(ongoing)* |
| NablaEngine | [sailab-ai/NablaEngine](https://github.com/sailab-ai/NablaEngine) | differentiable simulator in the speed comparison |
| Genesis | [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) | Eden's simulator *(ongoing)* |
| SPlisHSPlasH | [InteractiveComputerGraphics/SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) | fluid ground truth for CFC *(ongoing)* |

## Data

| Data | Link | Here |
|---|---|---|
| LAFAN1 | [ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | the G1's 40 tracking clips |
| InterHuman (InterGen) | [tr3e/InterGen](https://github.com/tr3e/InterGen) | the two-person clip |
| OMOMO | [lijiaman/omomo_release](https://github.com/lijiaman/omomo_release) | RoboKit's G1 box carrying |
| OmniRetarget (holosoma) | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | RoboKit's G1 climbing |

## Robots and hands

| Robot | Type | Model from | Here |
|---|---|---|---|
| Unitree G1 | humanoid, 29 DoF | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | tracking, two G1s, human-object interaction, RoboKit's G1 tasks |
| Unitree Go2 | quadruped | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | RoboKit's Go2 jumping |
| LimX TRON 1 | wheel-legged biped | [limxdynamics/tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | driving and a ramp, planned on world models *(ongoing)* |
| XHand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Allegro hand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Inspire hand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Wuji Hand 1, Wuji Hand 2.2 | dexterous hands | [openrobokit/robokit](https://github.com/openrobokit/robokit) | hand-object tracking with our own retargets, RoboKit's cube *(ongoing)* |
| Sharpa hand | dexterous hand | [openrobokit/robokit](https://github.com/openrobokit/robokit) | RoboKit's pen *(ongoing)* |
| Adroit hand | dexterous hand, 28 actuators | [the contributed pull request](https://github.com/Frank-ZY-Dou/ndm-gym/pull/3) | door opening *(ongoing)* |
| MuscleVAE's character | muscle-actuated humanoid | [wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_) | tracking with muscles |
| ASE's humanoid | sword-and-shield character | [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) | MimicKit's methods *(ongoing)* |

## Tools

| Tool | Link | Here |
|---|---|---|
| HoloMotion | [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | retargets the two-person clip |
| supermimic_vibed | [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | the code this library started from |
