# Human-object interaction

HDMI's 13 tasks retain its observations, rewards, terminations and randomization (`ndm_gym/tasks/hdmi.py`,
`ndm_gym/tasks/hdmi_tasks.py`), scenes and G1 variant on Isaac Lab 3 and MuJoCo-Warp
(`ndm_gym/sim/rigid/hdmi_sim.py`). The world models also predict the contact forces. HDMI's PPO teacher is the
model-free baseline (`ndm_gym.train_ppo_hoi`).

The tasks (`--task`): `carry_and_place_bread_box`, `carry_box_over_shoulder`, `move_foam`, `move_stool-climb`,
`move_suitcase`, `open_door-feet`, `open_foldchair-sit`, `push_box`, `push_door-hand`, `roll_ball-hand`,
`topple_wood_board_and_cross`, `traverse_door`, `truman`.

| | | |
|---|---|---|
| ![move_suitcase](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/move_suitcase.webp)<br>`move_suitcase` | ![carry_box_over_shoulder](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/carry_box_over_shoulder.webp)<br>`carry_box_over_shoulder` | ![carry_and_place_bread_box](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/carry_and_place_bread_box.webp)<br>`carry_and_place_bread_box` |
| ![move_foam](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/move_foam.webp)<br>`move_foam` | ![move_stool-climb](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/move_stool-climb.webp)<br>`move_stool-climb` | ![push_box](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/push_box.webp)<br>`push_box` |
| ![push_door-hand](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/push_door-hand.webp)<br>`push_door-hand` | ![open_door-feet](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/open_door-feet.webp)<br>`open_door-feet` | ![open_foldchair-sit](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/open_foldchair-sit.webp)<br>`open_foldchair-sit` |
| ![roll_ball-hand](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/roll_ball-hand.webp)<br>`roll_ball-hand` | ![topple_wood_board_and_cross](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/topple_wood_board_and_cross.webp)<br>`topple_wood_board_and_cross` | ![truman](https://raw.githubusercontent.com/Frank-ZY-Dou/ndm-gym/main/docs/media/hoi/policies/truman.webp)<br>`truman` |

One policy per task on twelve of HDMI's thirteen tasks, trained in the world model and run in the simulator (evaluation seed 1). Each episode runs to the end of its clip; 8 s of it are shown.

Renders of policies tracking HDMI's reference motions, with the robot and object models of HDMI's release (its repository has no license), for non-commercial illustration only; not covered by this repository's MIT license ([see NOTICE](../../NOTICE)).

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

The task as implemented, item by item: [design](../hoi_design.md), [HDMI task specification](../hdmi_task_spec.md). Simulation throughput:
[`results.md`](../../benchmarks/ndm_speed/results.md).
