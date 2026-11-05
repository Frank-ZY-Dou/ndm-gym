# Algorithms

`--algo` selects the algorithm (`ndm_gym.algos.ALGOS`). Algorithms share the simulator, observations, networks, replay
buffer and schedule.

| `--algo` | policy update | world model | critic |
|---|---|---|---|
| `supertrack` | supervised L1 tracking losses through the world model over the imagined horizon | SuperTrack's acceleration model | none |
| `supermimic` | the imagined reward backpropagated through the world model (first-order gradients) | SuperMimic's increment model | beyond the horizon |
| `blinkmimic` | SuperMimic with BlinkMimic's recipe: a preset of the options below | SuperMimic's semi-implicit model | beyond the horizon |

See [how the methods relate](blinkmimic.md#2-how-the-methods-relate) for the objectives and semi-implicit world models.

## The training loop

`ndm_gym/train.py` (`ndm-train`) uses SuperMimic's Algorithm A.1 for all algorithms:

1. Collect transitions with the current policy in `--num_envs` simulator environments.
2. Fit k-step replay windows (`--wm_window`, `--wm_batches` per iteration).
3. Update the policy on imagined rollouts from replay states (`--horizon` steps, `--pi_batch` rollouts).

The first `--wm_only_iters` iterations fit only the world model. `--max_hours` stops with a checkpoint; `--resume`
continues from a run's `last.pt`.

## Options

`ndm-train --help` lists all options; without the simulator, `python -m ndm_gym.wmbench.fit --help` lists
`add_train_args`. Definitions are in `ndm_gym/train.py` (`add_train_args`, `add_speed_args`) and
`ndm_gym/algos/blinkmimic.py` (`add_blinkmimic_args`). Each help text names the algorithms that use it. Defaults preserve
the original behavior, checked bit for bit by upstream equivalence and SuperTrack golden tests.

| | options |
|---|---|
| SuperTrack's objective | `--st_fail_w`, `--st_fail_margin` (failure penalty), `--st_balance_w`, `--st_com`, `--st_quad_w`, `--st_late_w`, `--st_reg`, `--st_imag_noise`, `--st_no_root_terms`, `--st_no_task_regs`, `--st_substeps`, `--st_wm_loss` |
| SuperMimic's objective | `--reward_std_scale`, `--action_noise`, `--gamma`, `--lam`, `--sm_residual`, `--contact_reward`, `--ref_contact_fix`, `--no_critic`, `--critic_epochs` |
| imagination, both | `--horizon`, `--imag_start_perturb`, `--imag_fail_starts`, `--bptt_clip_k`, `--bptt_decay`, `--pi_rollout_clip`, `--wm_bptt_fp32` |
| world model | `--wm_backbone`, `--wm_hidden`, `--wm_act`, `--wm_no_ln`, `--wm_lr`, `--wm_window`, `--wm_batches`, `--wm_batch`, `--wm_sample`, `--wm_fail_windows`, `--wm_input_noise`, `--wm_smooth_w`, `--rf_*` |
| optimization | `--lr`, `--pi_lr_mult`, `--lr_decay_iters`, `--lr_decay_hours`, `--pi_opt`, `--wm_opt`, `--pi_ema`, `--pi_updates` |
| resets | `--fail_reset`, `--bin_fix`, `--eprobe_*` (evaluation-guided resets) |
| motions | `--motions`, `--motion_dir`, `--clips`, `--paired` |
| speed (same results up to rounding, or noted) | `--fused_adam`, `--elementwise_3x3`, `--combo_kernels`, `--compile_eager`, `--wm_cudagraphs`, `--pi_cudagraphs`, `--critic_cudagraphs`, `--goal_table`, `--pi_encode_once`, `--rf_fused_train`, `--rf_fit_keep_ops`, `--rf_bptt_keep_blocks` |
| systems (BlinkMimic) | `--no_sync`, `--update_graphs`, `--graph_early`, `--collect_graph`, `--reset_masked`, `--ckpt_light` (with `--last_every_min`) |

See [BlinkMimic's options](blinkmimic.md#3-blinkmimics-changes-as-options) for the preset and overrides. Selected task
configurations are in the
[README](https://github.com/Frank-ZY-Dou/ndm-gym#-results).

## The interface

Subclass `ndm_gym.algos.common.Algo`, which owns the world model, policy and optimizers, fits replay windows and samples
imagination starts. Define:

| | |
|---|---|
| `world_model_cls(a)` | the world model for the training arguments `a` (also a checkpoint's `args`) |
| `setpoint(a, f)` | the normalized joint setpoint sent to the simulator for policy output `a` at reference frame `f` |
| `wm_loss(S_hat, w)` | the world-model loss of predicted windows against replay windows |
| `policy_update(buffer)` | one policy (and critic) update from replay states |
| `state_dict()`, `load_state_dict(ck)` | its checkpoint state |

Register the subclass in `ALGOS` (`ndm_gym/algos/__init__.py`). HOI and muscle classes follow the same pattern
(`ndm_gym/algos/hoi_algos.py`, `ndm_gym/algos/muscle_algos.py`).
