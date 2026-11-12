# SuperMimic

`--algo supermimic`. The policy maximizes the imagined tracking return through the world model, with a soft survival weight and a critic beyond the horizon (first-order
gradients). A re-implementation on this pipeline, not the authors' code: it was equivalence-tested against an unofficial reproduction before the authors
released their [official code](https://github.com/HybridRobotics/supermimic) ([`supermimic.py`](../../ndm_gym/algos/supermimic.py)).

## Policy update

The policy outputs setpoints about the default pose (`--sm_residual`: offsets about the reference pose). It is trained by backpropagation through the frozen world model
over `--horizon` imagined steps (paper §3, Eq. 1-3).

- **Objective:** the task's reward on imagined states, discounted by `--gamma` and weighted by a soft survival probability.
- **Beyond the horizon:** the critic's value of the last imagined state. The critic is fit to λ-returns (`--lam`) of the imagined rollouts.
- **Exploration:** Gaussian noise on the simulator's actions during collection.

## World model

The increment model ([`increment.py`](../../ndm_gym/worldmodels/increment.py)) predicts frame-local increments of every state component, the positions directly, without
integrating velocities. It is fit on a normalized frame-local state error. It takes an MLP or a RigidFormer backbone ([World models](../world_models.md)).
BlinkMimic's semi-implicit variant is `--wm_structured` ([BlinkMimic](../blinkmimic.md#2-how-the-methods-relate)).

## Options

| option | default | effect |
|---|---|---|
| `--reward_std_scale` | 1 | scale of the reward's tracking-kernel widths during training (the policy objective and reset sampling); evaluation keeps the task's |
| `--action_noise` | 0.1 | std of the simulator's action noise during collection |
| `--gamma` | 0.99 | discount |
| `--lam` | 0.95 | λ of the critic's targets |
| `--sm_residual` | off | the policy outputs offsets about the reference pose (SuperTrack's action parameterization) |
| `--contact_reward` | 0 | weight of a foot-contact term in the imagined reward; the evaluation's reward is unchanged |
| `--ref_contact_fix` | off | with `--contact_reward`: the reference's foot contact from each clip's own floor instead of a fixed 4.5 cm |
| `--no_critic` | off | drops the critic |
| `--critic_epochs` | 5 | the critic's epochs per update |
| `--critic_minibatches` | 4 | the critic's minibatches per epoch |

Options of both algorithms: [Algorithms](../algorithms.md#options). HOI training has its own: `python -m ndm_gym.train_hoi --help`.

## Configurations

| task | configuration |
|---|---|
| LAFAN1 motion tracking | `--reward_std_scale 0.7 --action_noise 0.3 --imag_start_perturb 0.3` ([results](../../README.md#lafan1-motion-tracking)); [the fastest](../../README.md#training-speed) |
| Two G1s | [README](../../README.md#two-g1s-one-two-person-clip) |
| Human-object interaction | [commands](../tasks/hoi.md) |
| Muscle-actuated character | [README](../../README.md#muscle-actuated-character) |
| Hand-object tracking | `ndm_gym.train_hand`'s defaults ([README](../../README.md#hand-object-tracking)) |

## Paper

Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath. *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model.* CoRL 2026.
[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)
