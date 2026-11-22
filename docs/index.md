# NDM Gym

**Neural Dynamics Modeling Gym**: neural world models for controlling physical agents.

NDM Gym trains controllers of physical agents through neural world models. Algorithms share one pipeline within each task: LAFAN1
motion tracking, two G1s on a two-person clip, HDMI's human-object interaction (Isaac Lab 3 with MuJoCo-Warp, the Unitree G1 with
29 DoF), MuscleVAE's muscle-actuated character (Newton with MuJoCo-Warp), hand-object tracking with the XHand, the Allegro hand
and the Inspire hand, a racing quadrotor tracking flown trajectories, a Unitree Go2 tracking a dog's motion capture and two
Wuji hands playing the piano (all four MuJoCo-Warp), and the Adroit hand tracking DAPG's door, pen, hammer and relocate
demonstrations (MuJoCo on the CPU). The neural dynamics models (NDMs) also serve as simulators, for instance in RoboKit's MPC
planner and mbrl-lib's CEM planner. On the G1 they step 16-17x faster than Steel's fastest passing setting and 40-42x faster
than MuJoCo Warp at 4,096 worlds on an A100 ([results](../benchmarks/steel/results.md)).

- [Leaderboard](leaderboard.md): each task's fastest and most accurate configuration, and the named configurations.
- [Algorithms](algorithms.md): SuperTrack, SuperMimic and BlinkMimic's recipe (`--algo blinkmimic`);
  [XMimic](algorithms/xmimic.md)'s techniques as options; XMimic's and [NablaMimic](algorithms/nablamimic.md)'s own code
  timed beside.
- [World models](world_models.md): acceleration and increment models, MLP and RigidFormer backbones, inference copies.
- [Tasks](tasks.md): LAFAN1 tracking, two G1s, HDMI's 13 human-object interaction tasks, MuscleVAE's character, hand-object
  tracking, [quadrotor trajectory tracking](tasks/quadrotor.md), [quadruped motion tracking](tasks/quadruped.md),
  [Wuji Hand piano playing](tasks/wuji_piano.md), the Adroit
  [door](tasks/adroit_door.md) and [pen, hammer and relocate](tasks/adroit_suite.md), and [RoboKit's](tasks/robokit.md) and
  [mbrl-lib's](tasks/mbrl.md) planners on a world model.
- [Evaluation](quickstart.md): tracking errors, push robustness, held-out clips, videos and offline world-model fits.
- [Coverage](coverage.md): every method, simulator, dataset and robot model the library covers, with its link.

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
| `ndm_gym/robots/` | Unitree G1 29-DoF spec (`g1.py`); MuscleVAE's character and muscles (`musclevae_char.py`, `muscles.py`); dexterous hands (`hands.py`); objects (`objects.py`); a racing quadrotor (`quadrotor.py`); the Unitree Go2 (`go2.py`) |
| `ndm_gym/motion/` | motion library (50 Hz goals) and loaders (`sources/lafan1.py`, `hoi.py`, `interhuman.py`, `musclevae.py`, `spider.py`, `drone_racing.py`, `lifelike.py`) |
| `ndm_gym/sim/` | Isaac Lab 3 + MuJoCo-Warp tracking simulator; HDMI's scenes; MuscleVAE on Newton; a hand and an object on MuJoCo-Warp; a quadrotor on MuJoCo-Warp; the Go2 on MuJoCo-Warp; Isaac Lab's launcher; RoboKit's simulator on a world model and its recordings; mbrl-lib's environments on a world model |
| `ndm_gym/tasks/` | `tracking.py` (SuperMimic's setup), `contact.py`; `hdmi.py`, `hdmi_tasks.py` (HDMI's 13 tasks); `muscle_tracking.py`; `hand_tracking.py`; `quadrotor_tracking.py`; `quadruped_tracking.py` |
| `ndm_gym/worldmodels/` | `accel.py` (SuperTrack), `increment.py` (SuperMimic), `semi_implicit.py` (BlinkMimic), `rigidformer.py` (backbone), `hoi.py`, `muscle.py`, `hand.py`, `quadrotor.py`, `mjcf.py` (MuJoCo models in generalized coordinates) |
| `ndm_gym/kernels/` | fused Triton kernels: the RigidFormer's blocks, the muscles' geometry |
| `ndm_gym/policies/` | tracking policy and critic; the HOI, muscle, hand and quadruped policies; MuscleVAE's released policy |
| `ndm_gym/algos/` | `common.py` (world-model fitting, imagination starts), `supertrack.py`, `supermimic.py`, `blinkmimic.py`, `eprobe.py`, `xmimic.py`, `layouts.py`, `hoi_algos.py`, `muscle_algos.py`, `hand_algos.py`, `quadrotor_algos.py`, `quadruped_algos.py`, `ppo_hoi.py` |
| `ndm_gym/wmbench/` | offline world-model benchmark; HOI and muscle datasets, fits and throughput; RoboKit's and mbrl-lib's recordings, fits, timing and plans |
| `ndm_gym/train.py`, `ndm_gym/evaluate.py` | `ndm-train`, `ndm-eval`; `train_hoi.py`, `train_ppo_hoi.py`, `evaluate_hoi.py`, `train_muscles.py`, `train_hand.py`, `evaluate_hand.py`, `train_quadrotor.py`, `evaluate_quadrotor.py`, `train_quadruped.py`, `evaluate_quadruped.py` for the other tasks |
| `scripts/`, `tools/` | install and launch scripts; data download and conversion, rendering, plots |
| `tests/` | equivalence tests against the upstream code, component tests |

The [Leaderboard](leaderboard.md) carries the README's leaderboard and recommended settings; the results with their
galleries, the ablations and the trained weights are in the [README](https://github.com/Frank-ZY-Dou/ndm-gym#-results) and on
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

MIT license. The LAFAN1 motions, HDMI's and MuscleVAE's files and SPIDER's hand clips are downloaded by the scripts in
`tools/`, InterHuman's clip is converted from the user's own copy, and none of them is redistributed; see the
[README](https://github.com/Frank-ZY-Dou/ndm-gym#-license-and-acknowledgments).
The papers, data, robot models and code it builds on are cited in the README's
[Papers and repositories](https://github.com/Frank-ZY-Dou/ndm-gym#-papers-and-repositories).
