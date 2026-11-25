# What NDM Gym covers

Every method, simulator, dataset, robot model and tool this library implements, runs, accelerates or builds on, by name with its
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
| DiffMimic | [paper](https://arxiv.org/abs/2304.03274) | its demonstration replay as an option (`--st_ref_replay`) |
| NeRD | [NVlabs/neural-robot-dynamics](https://github.com/NVlabs/neural-robot-dynamics) | NeRD-style world model *(ongoing)* |
| MIMIC-MJX | [paper](https://arxiv.org/abs/2511.20532), [talmolab/vnl-playground](https://github.com/talmolab/vnl-playground) | its rat, fruit fly and stick insect, their edits and terminations |
| HDMI | [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI) | its 13 human-object interaction tasks |
| MuscleVAE | [wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_) | its muscle-actuated character |
| SPIDER | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object clips and scenes |
| RoboKit | [openrobokit/robokit](https://github.com/openrobokit/robokit) | its planner on world models |
| PETS | [paper](https://arxiv.org/abs/1805.12114) | its planner settings of the cart-pole and the pusher, in mbrl-lib's controller |
| mbrl-lib | [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | its controller on world models, at PETS' settings; its algorithms (PETS, MBPO) *(ongoing)* |
| MimicKit | [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) | the framework whose methods run on world models *(ongoing)* |
| DeepMimic | [xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic) | MimicKit's tracking method *(ongoing)* |
| AMP | [paper](https://doi.org/10.1145/3450626.3459670), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| AWR | [paper](https://arxiv.org/abs/1910.00177), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| ASE | [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE), in MimicKit | accelerated with SuperTrack *(ongoing)* |
| C·ASE | [Frank-ZY-Dou/CASE](https://github.com/Frank-ZY-Dou/CASE) | ported to MimicKit; accelerated with SuperTrack *(ongoing)* |
| LCP | [paper](https://arxiv.org/abs/2410.11825), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| ADD | [paper](https://doi.org/10.1145/3757377.3763819), in MimicKit | timed with the systems options; accelerated *(ongoing)* |
| CFC | [project page](https://people.csail.mit.edu/frankzydou/projects/CFC/) | two-level world model with forces as the interface *(ongoing)* |
| Eden | [embodied-ai-nexus/Eden](https://github.com/embodied-ai-nexus/Eden) | its piano task's *Ode to Joy* and framewise scoring, on MuJoCo-Warp ([contributed](https://github.com/Frank-ZY-Dou/ndm-gym/pull/9)); its other tasks on world models *(ongoing)* |

## Simulators and engines

| Simulator | Link | Here |
|---|---|---|
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | tracking simulator |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | models, CPU simulation; MJX in the differentiable-simulator comparison |
| MuJoCo Warp | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | GPU simulation; the baseline of the speed comparisons |
| Newton | [newton-physics/newton](https://github.com/newton-physics/newton) | MuscleVAE's simulation; MimicKit's *(ongoing)* |
| Steel (Skild AI) | [skild-ai/steel](https://github.com/skild-ai/steel), [skild-ai/newton](https://github.com/skild-ai/newton) | Newton's `SolverSteel`: timed beside the G1's world models ([results](../benchmarks/steel/results.md)); a simulation backend *(ongoing)* |
| NablaEngine | [sailab-ai/NablaEngine](https://github.com/sailab-ai/NablaEngine) | differentiable simulator in the speed comparison |
| Genesis | [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) | timed beside the differentiable simulators; Eden's simulator *(ongoing)* |
| SPlisHSPlasH | [InteractiveComputerGraphics/SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) | fluid ground truth for CFC *(ongoing)* |

## Data

| Data | Link | Here |
|---|---|---|
| LAFAN1 | [ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | the G1's 40 tracking clips |
| LAFAN1 Retargeting Dataset | [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | LAFAN1 on the G1, and the G1's URDF |
| InterHuman (InterGen) | [tr3e/InterGen](https://github.com/tr3e/InterGen) | the two-person clip |
| SMPL | [paper](https://doi.org/10.1145/2816795.2818013) | InterHuman's body model, read from the user's licensed copy |
| OMOMO | [lijiaman/omomo_release](https://github.com/lijiaman/omomo_release) | RoboKit's G1 box carrying |
| InterMimic | [project page](https://sirui-xu.github.io/InterMimic/) | OMOMO's clip as it processed it |
| OmniRetarget (holosoma) | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | RoboKit's G1 climbing |
| DAPG's demonstrations | [aravindr93/hand_dapg](https://github.com/aravindr93/hand_dapg), [paper](https://arxiv.org/abs/1709.10087) | the Adroit door's demonstration (clip 23); the pen, hammer and relocate demonstrations |
| SPIDER's retargets | [retarget/retarget_full](https://huggingface.co/datasets/retarget/retarget_full) | the hands' clips |
| HRDexDB, OakInk, DexYCB | [HRDexDB](https://arxiv.org/abs/2604.14944), [OakInk](https://arxiv.org/abs/2203.15709), [DexYCB](https://arxiv.org/abs/2104.04631) | the demonstrations behind the hands' clips |
| Race Against the Machine | [tii-racing/drone-racing-dataset](https://github.com/tii-racing/drone-racing-dataset), [paper](https://doi.org/10.1109/LRA.2024.3371288) | the quadrotor's flights, autonomous and piloted |
| On Your Own | [paper](https://doi.org/10.1109/LRA.2026.3653405), [release v3.0.0](https://github.com/tii-racing/drone-racing-dataset/releases/tag/v3.0.0) | the professional pilot's race-track flights |
| MIMIC-MJX's clips | [talmolab/MIMIC-MJX](https://huggingface.co/datasets/talmolab/MIMIC-MJX) | stac-mjx's kinematics of a rat, a fruit fly and a stick insect, for animal motion tracking |
| Lifelike agility and play | [Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play), [paper](https://doi.org/10.1038/s42256-024-00861-3) | a Labrador's motion capture, retargeted to the Go2 for quadruped tracking |
| robomimic | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic), [its datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets) | the proficient-human demonstrations of Lift, Square and Transport, tracked by Panda arms in robosuite's scenes on MuJoCo-Warp *(ongoing)* |

## Robots and hands

| Robot | Type | Model from | Here |
|---|---|---|---|
| Unitree G1 | humanoid, 29 DoF | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | tracking, two G1s, human-object interaction, RoboKit's G1 tasks |
| Unitree G1, mjlab's model | humanoid, primitive colliders | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | the differentiable-simulator comparison |
| Unitree H1 | humanoid, 19 DoF | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros), through the [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | LAFAN1 tracking on the dataset's H1 retargets, on MuJoCo-Warp *(ongoing)* |
| Unitree Go2 | quadruped | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | RoboKit's Go2 jumping; quadruped motion tracking |
| LimX TRON 1 | wheel-legged biped | [limxdynamics/tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | driving and a ramp, planned on world models *(ongoing)* |
| XHand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Allegro hand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Inspire hand | dexterous hand | [facebookresearch/spider](https://github.com/facebookresearch/spider) | hand-object tracking |
| Wuji Hand 1, Wuji Hand 2.2 | dexterous hands | [openrobokit/robokit](https://github.com/openrobokit/robokit) | hand-object tracking with our own retargets, RoboKit's cube *(ongoing)* |
| Wuji Hand 2 (beta2) | dexterous hand, two floating | [wuji-technology/wuji-description](https://github.com/wuji-technology/wuji-description) | piano playing, Eden's *Ode to Joy* ([contributed](https://github.com/Frank-ZY-Dou/ndm-gym/pull/9)) |
| RoboPianist's keyboard | 88 passive keys | [google-research/robopianist](https://github.com/google-research/robopianist), through [Eden's assets](https://huggingface.co/datasets/Kashu7100/eden_assets) | piano playing |
| Sharpa hand | dexterous hand | [openrobokit/robokit](https://github.com/openrobokit/robokit) | RoboKit's pen *(ongoing)* |
| Adroit hand | dexterous hand, 24-30 actuators by task | [Farama-Foundation/Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) | door opening ([contributed](https://github.com/Frank-ZY-Dou/ndm-gym/pull/3)); pen turning, hammering and ball relocation ([contributed](https://github.com/Frank-ZY-Dou/ndm-gym/pull/11)); from DAPG demonstrations |
| MIMIC-MJX's rat, stick insect | anatomically detailed animal bodies (67 and 41 joints, torque actuators) | [talmolab/vnl-playground](https://github.com/talmolab/vnl-playground) | animal motion tracking |
| flybody's fruit fly | insect body, 36 leg servos | [TuragaLab/flybody](https://github.com/TuragaLab/flybody), through vnl-playground | animal motion tracking |
| Race Against the Machine's racer | 5-inch racing quadrotor | written here from the paper's specification (`ndm_gym/robots/quadrotor.py`) | trajectory tracking |
| MuscleVAE's character | muscle-actuated humanoid | [wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_) | tracking with muscles |
| Cart-pole, pusher | PETS' MuJoCo models | [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | mbrl-lib's planner on world models |
| ASE's humanoid | sword-and-shield character | [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) | MimicKit's methods *(ongoing)* |

## Tools

| Tool | Link | Here |
|---|---|---|
| HoloMotion | [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | retargets the two-person clip |
| supermimic_vibed | [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | the code this library started from |
| MASS | [lsw9021/MASS](https://github.com/lsw9021/MASS) | the bone meshes of the muscle renders |
| supermimic-quadruped (`quadmotion`) | [Houserz/supermimic-quadruped](https://github.com/Houserz/supermimic-quadruped) | retargets the Labrador's motion capture to the Go2, from the user's checkout |
| unitree_mujoco | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | the Go2's joint damping |
