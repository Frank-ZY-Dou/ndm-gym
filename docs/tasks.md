# Tasks

Four tasks share the pipeline. Scripts in `tools/` prepare the data; none of the motions are redistributed.

| task | command | data | design |
|---|---|---|---|
| LAFAN1 motion tracking | `ndm-train` | `tools/download_data.sh` | below |
| two G1s, a two-person clip | `ndm-train --motions interhuman --paired` | `tools/convert_interhuman.py` | below |
| human-object interaction (HDMI) | `python -m ndm_gym.train_hoi` | `tools/download_hdmi.sh`, `tools/convert_hdmi.py` | [design](hoi_design.md), [HDMI's task](hdmi_task_spec.md) |
| muscle-actuated character (MuscleVAE) | `python -m ndm_gym.train_muscles` | `tools/download_musclevae.sh`, `tools/convert_musclevae.py` | [design](musclevae_design.md) |

## LAFAN1 motion tracking

The Unitree G1 (29 DoF, `ndm_gym/robots/g1.py`) tracks 40 retargeted LAFAN1 clips using SuperMimic's setup
(`ndm_gym/tasks/tracking.py`). The simulator and world models share observations, analytic reward and terminations.
Domain randomization covers friction, joint calibration and the torso's center of mass. `--clips` selects clips from
a file (one name per line); `--motion_dir` reads another directory of retargeted CSVs.

```bash
bash tools/download_data.sh    # LAFAN1 G1 retargets (CC BY-NC-ND 4.0) and the G1 description, checked against tools/download_data.sha256
ndm-train --algo supertrack --log_dir logs/st
ndm-eval --run_dir logs/st --watch
```

## Two G1s

One policy controls two G1s tracking a retargeted two-person InterHuman clip in one scene, with robot-robot contacts.
InterHuman (CC BY-NC-SA 4.0) is not redistributed: `tools/convert_interhuman.py`
retargets a clip with HoloMotion's HoloRetarget, in an environment of its own (its docstring has the command).

```bash
ndm-train --algo supertrack --motions interhuman --paired --motion_dir interhuman_1566.npz --log_dir logs/pair_st
```

## Human-object interaction

HDMI's 13 tasks retain its observations, rewards, terminations and randomization (`ndm_gym/tasks/hdmi.py`,
`ndm_gym/tasks/hdmi_tasks.py`), scenes and G1 variant on Isaac Lab 3 and MuJoCo-Warp
(`ndm_gym/sim/rigid/hdmi_sim.py`). The world models also predict the contact forces. HDMI's PPO teacher is the
model-free baseline (`ndm_gym.train_ppo_hoi`).

The tasks (`--task`): `carry_and_place_bread_box`, `carry_box_over_shoulder`, `move_foam`, `move_stool-climb`,
`move_suitcase`, `open_door-feet`, `open_foldchair-sit`, `push_box`, `push_door-hand`, `roll_ball-hand`,
`topple_wood_board_and_cross`, `traverse_door`, `truman`.

```bash
bash tools/download_hdmi.sh && python tools/convert_hdmi.py    # HDMI's clips and objects into data/hdmi/ (not redistributed)
python -m ndm_gym.train_hoi --task move_suitcase --algo supertrack --log_dir logs/hoi_st
python -m ndm_gym.train_hoi --task move_suitcase --algo supermimic --log_dir logs/hoi_sm
python -m ndm_gym.train_ppo_hoi --task move_suitcase --log_dir logs/hoi_ppo
python -m ndm_gym.evaluate_hoi --run_dir logs/hoi_st --watch
```

Collect transitions, fit offline, measure open-loop errors and benchmark throughput against physics:

```bash
python -m ndm_gym.wmbench.collect_hoi --task move_suitcase --seed 0 --out hoi_s0.pt
python -m ndm_gym.wmbench.collect_hoi --task move_suitcase --seed 1 --out hoi_s1.pt
python -m ndm_gym.wmbench.fit_hoi --task move_suitcase --kind accel --backbone rigidformer --train hoi_s0.pt --test held=hoi_s1.pt --out fit.csv
python -m ndm_gym.wmbench.speed_hoi --task move_suitcase --mode physics --num_envs 4096 --out speed.csv
python -m ndm_gym.wmbench.speed_hoi --task move_suitcase --mode ndm --backbone rigidformer --compile reduce-overhead --num_envs 4096 --out speed.csv
```

## Muscle-actuated character

MuscleVAE's character: 23 bodies, 284 Hill-type muscles with fatigue, on Newton with MuJoCo-Warp directly (no Isaac
Lab; `ndm_gym/sim/rigid/muscle_sim.py`), tracking MuscleVAE's three LAFAN1 clips. The world models predict the
character's state and the muscles' fatigue; the RigidFormer has a token per body and per muscle
(`--rf_muscle_tokens all`) or per body, with the muscles' wrench and pooled encodings (`--rf_muscle_tokens body`).

```bash
bash tools/download_musclevae.sh && python tools/convert_musclevae.py   # character, muscles, motions (not redistributed)
python -m ndm_gym.train_muscles --algo supertrack --log_dir logs/mus_st
python -m ndm_gym.train_muscles --algo supermimic --log_dir logs/mus_sm

# MuscleVAE's released policy in this simulator, rendered in their look (bash tools/download_mass_meshes.sh for the bones)
bash tools/download_musclevae_pretrained.sh
python tools/replay_musclevae.py --out_dir replay
MUJOCO_GL=egl python tools/render_musclevae.py walk.mp4 --states replay/states_lafan_walk_0.npz --ref
```

Selected configurations and results:
[`benchmarks/musclevae/results.md`](../benchmarks/musclevae/results.md).
