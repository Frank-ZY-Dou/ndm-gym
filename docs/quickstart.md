# Quickstart

## Train and evaluate

With the simulator installed (`bash scripts/setup.sh`, then `source ~/ndm_gym/venv/bin/activate`):

```bash
# train (scripts/run_train.sh adds a memory cap, CPU pinning and a GPU lock)
ndm-train --algo supertrack --st_fail_w 5 --st_balance_w 4 --max_hours 3 --log_dir logs/st
ndm-train --algo supermimic --reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3 --max_hours 3 --log_dir logs/sm
ndm-train --algo supertrack --wm_backbone rigidformer --max_hours 3 --log_dir logs/st_rf   # the RigidFormer world model

# evaluate every checkpoint as it is written; push robustness of one checkpoint
ndm-eval --run_dir logs/st --watch
ndm-eval --run_dir logs/st --ckpt logs/st/ckpt_012000.pt --push_mag 0 1 2 --out_json push.json

# compare: curves, then side-by-side videos of recorded evaluations (pip install "ndm_gym[render] @ git+https://github.com/Frank-ZY-Dou/ndm-gym", ffmpeg)
python tools/plot.py logs/st logs/sm --labels SuperTrack SuperMimic --out compare.png
ndm-eval --run_dir logs/st --ckpt <ckpt> --record st.npz      # and the same for sm.npz
MUJOCO_GL=egl python tools/compare_videos.py st.npz sm.npz --labels SuperTrack SuperMimic --out videos/
```

`ndm-train --help` lists all options. Other [tasks](tasks.md) use:
`python -m ndm_gym.train_hoi --task move_suitcase --algo supertrack`, `python -m ndm_gym.train_muscles --algo supertrack`.

## World models in Python

The library's `pip install` is enough. See [World models](world_models.md) for the state, action and context layouts;
joints follow `ndm_gym.robots.g1.G1_JOINT_NAMES`. The differentiable `step` predicts the state one control step (20 ms) later:

```python
import torch

from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots.g1 import NUM_J
from ndm_gym.state import normalize_xi
from ndm_gym.worldmodels.accel import AccelWorldModel          # SuperTrack's: predicts accelerations
from ndm_gym.worldmodels.increment import IncrementWorldModel  # SuperMimic's: predicts state increments

N = 256
S = (torch.randn(N, 3), so3_exp(torch.randn(N, 3)), torch.randn(N, 3), torch.randn(N, 3), torch.randn(N, NUM_J), torch.randn(N, NUM_J))
a = torch.randn(N, NUM_J, requires_grad=True)
xi = torch.cat((torch.full((N, 1), 0.75), torch.zeros(N, NUM_J + 3)), -1)

wm = AccelWorldModel()                   # the MLP world model (3 x 2048), as training builds it
S1 = wm.step(S, a, normalize_xi(xi))     # the next state, same layout
S1[0].square().sum().backward()          # into a.grad, through the world model
```

A new model's output layer is zero, so these gradients are zero until it is fit or loaded.
`wm.inputs(S, a, xi_n)` (`xi_n = normalize_xi(xi)`) and `wm.targets(S, S1)` give the network's raw inputs and
regression targets for fitting the normalizers `wm.in_norm` and `wm.out_norm` (`update`). Algorithms fit k-step replay
windows through `ndm_gym.algos.common.Algo.wm_update`.

### The RigidFormer

The RigidFormer backbone reads the G1's kinematic tree, built by MuJoCo from the G1's MJCF
(`data/g1_description/g1_29dof.xml`, written by `tools/download_data.sh`):

```python
from ndm_gym.robots import g1

kin = g1.kinematics("cpu")                               # or ArticulationKinematics(<mjcf path>, g1.G1_JOINT_NAMES, g1.TRACKED_BODIES, "cpu")
wm = AccelWorldModel(rigidformer=dict(kin=kin))          # SuperTrack-RF's world model: RF-S, 0.88 M parameters
wm = IncrementWorldModel(rigidformer=dict(kin=kin))      # SuperMimic-RF's world model: RF-S too
```

### From a checkpoint

`world_model_from_args` builds a model from a checkpoint's saved arguments:

```python
from ndm_gym.algos import world_model_from_args

ck = torch.load("logs/st/last.pt", map_location="cpu")
wm = world_model_from_args(ck["args"])
wm.load_state_dict(ck["wm"])
```

`last.pt` and the full checkpoints hold the world model; with `--ckpt_light`, the periodic ones hold the policy only and
`last.pt` is written at the end (with `--last_every_min M`, also every M minutes of training).

## The offline world-model benchmark

Collect transitions with a frozen policy in the simulator, then fit and score world models without it (a CUDA GPU and
`data/`):

```bash
python -m ndm_gym.wmbench.collect --ckpt logs/st/ckpt_006000.pt --out wm_train.pt --seed 0    # simulator
python -m ndm_gym.wmbench.collect --ckpt logs/st/ckpt_006000.pt --out wm_test.pt --seed 1     # simulator
python -m ndm_gym.wmbench.fit --algo supertrack --train wm_train.pt --test st=wm_test.pt --out wm_st.csv
python -m ndm_gym.wmbench.fit --algo supertrack --wm_backbone rigidformer --wm_lr 4e-4 --train wm_train.pt --test st=wm_test.pt --out wm_st_rf.csv
```

`fit` uses `ndm-train`'s options (`ndm_gym.train.add_train_args`) and world-model update.
