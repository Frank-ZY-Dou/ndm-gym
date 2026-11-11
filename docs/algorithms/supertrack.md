# SuperTrack

`--algo supertrack`. The policy minimizes L1 tracking losses through the world model, with no reward, no survival weight and no critic. A re-implementation on
this pipeline, not the authors' code ([`supertrack.py`](../../ndm_gym/algos/supertrack.py)).

## Policy update

The policy outputs PD offsets about the reference pose. It is trained by backpropagation through the frozen world model over `--horizon` imagined steps
(paper §3.3, §3.5, Alg. 2). Term weights are fixed once, so that every term contributes 1 at the start of training.

- **Tracking terms, root-local:** the bodies' positions, velocities, rotations and angular velocities; their heights and up vectors.
- **World-space root terms:** position and rotation, the paper's §4.2 extension for tasks that score global drift.
- **Added to the paper:** the task's action-rate and joint-limit penalties, at the reward's ratio to its full tracking score.
- **Regularization:** L2 and L1 penalties on the PD offsets, and Gaussian noise on them in imagination.
- **Optional:** a failure penalty on imagined steps past the task's termination thresholds.

## World model

The acceleration model ([`accel.py`](../../ndm_gym/worldmodels/accel.py)) predicts the root's linear and angular accelerations in its frame and the joint
accelerations, integrated by semi-implicit Euler (paper §3.4, Eq. 8-12). It is fit with an L1 window loss on the bodies' world-space states (Alg. 1). It takes an MLP or
a RigidFormer backbone ([World models](../world_models.md)).

## Options

| option | default | effect |
|---|---|---|
| `--st_fail_w` | 0 | cost of every imagined step past the termination thresholds, in calibrated per-step tracking losses |
| `--st_fail_margin` | 1 | the failure penalty's thresholds as a fraction of the task's termination thresholds (< 1: earlier) |
| `--st_balance_w` | 1 | extra weight on the height and up-vector terms |
| `--st_com` | off | adds a center-of-mass (position, velocity) tracking term |
| `--st_quad_w` | 0 | weight of e²/s next to \|e\| (stronger pull on large errors) |
| `--st_late_w` | 0 | extra weight on the last imagined steps (linear ramp) |
| `--st_reg` | 0.01 | weight of the L2 and L1 PD-offset regularizers |
| `--st_imag_noise` | 0.1 | Gaussian noise on PD offsets in imagination |
| `--st_no_root_terms` | off | drops the world-space root terms (the paper's base loss) |
| `--st_no_task_regs` | off | drops the task's action-rate and joint-limit penalties |
| `--st_substeps` | 1 | world-model integration substeps (1 = paper, 4 = simulator) |
| `--st_wm_loss` | `l1` | world-model window loss (`l1`, `l2`) |

Options of both algorithms: [Algorithms](../algorithms.md#options). HOI training has its own: `python -m ndm_gym.train_hoi --help`.

## Configurations

| task | configuration |
|---|---|
| LAFAN1 motion tracking | `--st_fail_w 5 --st_balance_w 4` ([results](../../README.md#lafan1-motion-tracking)); [the fastest](../../README.md#training-speed) |
| Two G1s | [README](../../README.md#two-g1s-one-two-person-clip) |
| Human-object interaction | [commands](../tasks/hoi.md) |
| Muscle-actuated character | [README](../../README.md#muscle-actuated-character) |
| Hand-object tracking | `--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights` ([README](../../README.md#hand-object-tracking)) |

## Paper

L. Fussell, K. Bergamin, D. Holden. *SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning.* ACM Transactions on Graphics 40(6), 2021.
[DOI](https://doi.org/10.1145/3478513.3480527)
