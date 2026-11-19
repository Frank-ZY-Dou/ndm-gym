# Third-party notices

NDM Gym started from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT, ETH Zurich, Chong Zhang); its
copyright line is in [`LICENSE`](LICENSE).

## Submodules

The git submodule `third_party/supermimic` ([zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) @
`ee0608f`, MIT, Chong Zhang), used unmodified for the equivalence tests, keeps its own license file and copyright notices.

## RoboKit

`tests/test_robokit_rec.py` and `tests/test_robokit_ndm.py` adapt the cart model and planner settings of the SimMPC guide of
[RoboKit](https://github.com/openrobokit/robokit) (MIT, Copyright (c) 2025 OpenRoboKit). This library's RoboKit modules
(`ndm_gym/sim/robokit_*.py`, `ndm_gym/wmbench/*robokit*.py`) patch RoboKit's classes at run time and copy none of its files.

## mbrl-lib, Gymnasium and dm_control

This library's mbrl-lib modules (`ndm_gym/sim/mbrl_*.py`, `ndm_gym/wmbench/*mbrl*.py`) run
[mbrl-lib](https://github.com/facebookresearch/mbrl-lib) at `3f93ccc` from its own checkout and patch its classes at run
time; they copy none of its files. `ndm_gym/sim/mbrl_envs.py` ports a few functions to batched torch:

- pets_reacher's end-effector position (`mbrl/env/pets_reacher.py`, `get_EE_pos`) and the rewards and terminations of
  mbrl-lib's environments: mbrl-lib, MIT License, Copyright (c) Facebook, Inc. and its affiliates (text below);
- the rewards and healthy checks of Hopper, Walker2d and InvertedPendulum:
  [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 0.26.3, MIT
  License, Copyright (c) 2016 OpenAI and Copyright (c) 2022 Farama Foundation;
- `rewards.tolerance` and the cart-pole, cheetah, cup, finger and walker rewards of the DeepMind Control Suite:
  [dm_control](https://github.com/google-deepmind/dm_control), Copyright 2017 The dm_control Authors, under the Apache
  License 2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)).

## Quadruped motion tracking

The quadruped tracking task ([`docs/tasks/quadruped.md`](docs/tasks/quadruped.md)) copies no third-party file. What it
reads is downloaded by the user at pinned commits (`tools/download_lifelike.sh`) and is not redistributed:

- the Labrador motion capture of "Lifelike agility and play in quadrupedal robots using reinforcement learning and
  generative pre-trained models" (Han et al., Nature Machine Intelligence 2024;
  [Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play) @
  `cc9eaa9`, `data/raw_mocap_data`; its licence file is MIT, its README says the code and data are for research purposes
  only);
- the Unitree Go2 of [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) @ `370c439`
  (`unitree_go2`, BSD-3-Clause).

`tools/convert_lifelike.py` calls the `quadmotion` package of
[supermimic-quadruped](https://github.com/Houserz/supermimic-quadruped) (MIT, Copyright (c) 2026 Yuxi Liu and Houserz;
the results used `46a81c2`) from the user's own checkout to retarget the motion capture. The joints' damping of `ndm_gym/robots/go2.py` (0.1 N m s / rad) is the value of Unitree's own model
of the Go2 ([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), `unitree_robots/go2/go2.xml`).

## Humanoid motion tracking (H1)

The humanoid tracking task ([`docs/tasks/humanoid.md`](docs/tasks/humanoid.md)) copies no third-party file. What it reads
is downloaded by the user at the pinned revision of the G1's data (`tools/download_h1.sh`, checked against
`tools/download_h1.sha256`) and is not redistributed:

- the H1 retargets of LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0) in the
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) @ `ce15729` (`h1/`,
  CC BY-NC-ND 4.0);
- the Unitree H1 description of the same dataset (`robot_description/h1`), Unitree's
  [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (`robots/h1_description`, BSD-3-Clause): the H1's
  geometry, masses, inertias, joint ranges and torque limits.

The H1's servo gains in `ndm_gym/robots/humanoid.py` are the values of Unitree's own H1 training configuration
([unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), `legged_gym/envs/h1/h1_config.py`, BSD-3-Clause),
and its joints' armature (0.1 kg m²) and damping (1 N m s / rad) those of Unitree's own MuJoCo model of the H1
([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), `unitree_robots/h1/h1.xml`, BSD-3-Clause;
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_h1/h1.xml`, BSD-3-Clause, has the same).
The training options `--bin_starts` and `--fail_reset` of `ndm_gym/train_humanoid.py` (`ndm_gym/sim/rigid/humanoid_sim.py`)
reuse the library G1's reset statistics (`ndm_gym/sim/rigid/reset_stats.py`, ported from BlinkMimic and InstantTrack; see
below).

Two options kept for comparison, off by default, take values from other projects and no code:

- `--gains tracking`: the H1's servo gains of [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) @ `101492b`
  (`humanoidverse/config/robot/h1/h1.yaml`, MIT License) and of
  [ProtoMotions](https://github.com/NVlabs/ProtoMotions) @ `4a905b9` (`protomotions/config/robot/h1.yaml`; NVIDIA
  License, non-commercial), which give the same values;
- `--foot menagerie`: the end points and radii of the three foot capsules of
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_h1/h1.xml` (BSD-3-Clause, Unitree
  Robotics).

The exclusion of the `fallAndGetUp` clips follows a practice of other humanoid tracking work, cited on the results page:
H2O (T. He et al., 2024) and GMT (Z. Chen et al., 2025) remove motions that their robot cannot perform.

## Ported code

The options of `--algo blinkmimic` ([`docs/blinkmimic.md`](docs/blinkmimic.md)) are ported from
[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) @ `d264658`, which is derived from
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack), both under the MIT License reproduced below. The ported
files carry BlinkMimic's notice: `ndm_gym/algos/blinkmimic.py`, `ndm_gym/algos/eprobe.py`, `ndm_gym/core/muon.py`,
`ndm_gym/sim/rigid/reset_stats.py`, `ndm_gym/tasks/contact.py` and `ndm_gym/worldmodels/semi_implicit.py`. Adapted parts of
`ndm_gym/train.py`, `evaluate.py`, `buffer.py`, `algos/common.py`, `core/normalizer.py`, `sim/rigid/isaaclab_mjwarp.py`,
`tests/test_eprobe.py` and `tests/test_blinkmimic.py` are under the same license.

## BlinkMimic

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mbrl-lib

```
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Wuji Hand piano

- The unmounted Wuji Hand 2 beta2 hand descriptions and STL meshes come from Wuji Technology under MIT. The downloader fetches the upstream license at the same revision. [Wuji Technology, revision c2cd7f8](https://github.com/wuji-technology/wuji-description/tree/c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95)
- The passive keyboard comes from Eden’s asset dataset. Its authors credit RoboPianist, Copyright 2023 The RoboPianist Authors, Apache-2.0. The downloader fetches the keyboard and the accompanying attribution. [Pinned keyboard and notice](https://huggingface.co/datasets/Kashu7100/eden_assets/tree/f409faaf65d8a996425cc71a035c30845c7314d8/objects/piano)
- The benchmark adapts Eden’s *Ode to Joy* arrangement and five-finger exercise (`eden/tasks/piano/songs.py`), its framewise scoring (`custom_terms/metrics.py`) and its piano configuration to MuJoCo-Warp: Copyright 2026 Kashu Yamazaki, Apache-2.0. `ndm_gym/tasks/piano.py` and `benchmarks/wuji_piano/five_finger_scale.py` state what they take and how they differ; the simulator integration, learning objectives and evaluation code are this repository's. [Eden at revision 148d972](https://github.com/embodied-ai-nexus/Eden/tree/148d972beaee94b801c6a55808f00445bedb381b) · [Eden Apache-2.0 license](benchmarks/wuji_piano/EDEN_LICENSE)
