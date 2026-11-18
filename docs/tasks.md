# Tasks

The following tasks share the pipeline. Scripts in `tools/` prepare the data; none of the motions are redistributed.

| task | robot or character | simulator | data | command |
|---|---|---|---|---|
| [LAFAN1 motion tracking](tasks/lafan1.md) | Unitree G1, 29 DoF | Isaac Lab 3, MuJoCo-Warp | `tools/download_data.sh` | `ndm-train` |
| [Two G1s](tasks/two_g1s.md), a two-person clip | two Unitree G1s | Isaac Lab 3, MuJoCo-Warp | `tools/convert_interhuman.py` | `ndm-train --motions interhuman --paired` |
| [Human-object interaction](tasks/hoi.md), HDMI's 13 tasks | HDMI's G1 variant | Isaac Lab 3, MuJoCo-Warp | `tools/download_hdmi.sh`, `tools/convert_hdmi.py` | `python -m ndm_gym.train_hoi` |
| [Muscle-actuated character](tasks/muscles.md) | MuscleVAE's character | Newton, MuJoCo-Warp | `tools/download_musclevae.sh`, `tools/convert_musclevae.py` | `python -m ndm_gym.train_muscles` |
| [Hand-object tracking](tasks/hand.md), SPIDER's clips | XHand, Allegro hand or Inspire hand on a floating palm | MuJoCo-Warp | `tools/download_hand_clips.py` | `python -m ndm_gym.train_hand` |
| [Quadrotor trajectory tracking](tasks/quadrotor.md), Race Against the Machine's flights | a 5-inch racing quadrotor | MuJoCo-Warp | `tools/download_drone_racing.py` | `python -m ndm_gym.train_quadrotor` |
| [Adroit Door tracking](tasks/adroit_door.md), physical door-opening reference | native Adroit hand, passive door and latch | native MuJoCo CPU collection, CUDA learning | `tools/replay_adroit_door.py`, `tools/prepare_adroit_door.py` | `python -m ndm_gym.train_door_full` |
| [Adroit Pen, Hammer and Relocate](tasks/adroit_suite.md) | native Adroit hand, passive manipulated objects | native MuJoCo CPU collection, CUDA learning | `tools/replay_adroit.py`, `tools/prepare_adroit.py` | `python -m ndm_gym.train_adroit` |

The world models also stand in for simulators:

| task | robots | simulator replaced | data | command |
|---|---|---|---|---|
| [RoboKit's planner on a world model](tasks/robokit.md), its MPC examples | Unitree G1, Go2 | MuJoCo Warp, in RoboKit's sampling MPC | recorded from the examples (`wmbench.collect_robokit`) | `python -m ndm_gym.wmbench.plan_robokit` |
| [mbrl-lib's planner on a world model](tasks/mbrl.md), PETS' cart-pole and pusher | mbrl-lib's MuJoCo models | CPU MuJoCo and MuJoCo Warp, in mbrl-lib's CEM controller | recorded from the controller (`wmbench.collect_mbrl`) | `python -m ndm_gym.wmbench.plan_mbrl` |
