# Tasks

Five tasks share the pipeline. Scripts in `tools/` prepare the data; none of the motions are redistributed.

| task | robot or character | simulator | data | command |
|---|---|---|---|---|
| [LAFAN1 motion tracking](tasks/lafan1.md) | Unitree G1, 29 DoF | Isaac Lab 3, MuJoCo-Warp | `tools/download_data.sh` | `ndm-train` |
| [Two G1s](tasks/two_g1s.md), a two-person clip | two Unitree G1s | Isaac Lab 3, MuJoCo-Warp | `tools/convert_interhuman.py` | `ndm-train --motions interhuman --paired` |
| [Human-object interaction](tasks/hoi.md), HDMI's 13 tasks | HDMI's G1 variant | Isaac Lab 3, MuJoCo-Warp | `tools/download_hdmi.sh`, `tools/convert_hdmi.py` | `python -m ndm_gym.train_hoi` |
| [Muscle-actuated character](tasks/muscles.md) | MuscleVAE's character | Newton, MuJoCo-Warp | `tools/download_musclevae.sh`, `tools/convert_musclevae.py` | `python -m ndm_gym.train_muscles` |
| [Hand-object tracking](tasks/hand.md), SPIDER's clips | XHand, floating palm | MuJoCo-Warp | `tools/download_hand_clips.py` | `python -m ndm_gym.train_hand` |
