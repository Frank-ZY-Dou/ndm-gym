# Algorithms

`--algo` selects the algorithm (`ndm_gym.algos.ALGOS`); the simulator, observations, networks, replay buffer and
schedule are shared, so a comparison changes only the algorithm.

| `--algo` | policy update | world model | critic |
|---|---|---|---|
| `supertrack` | supervised L1 tracking losses through the world model over the imagined horizon | SuperTrack's acceleration model | none |
| `supermimic` | the imagined reward backpropagated through the world model (first-order gradients) | SuperMimic's increment model | beyond the horizon |
| `blinkmimic` | SuperMimic with BlinkMimic's recipe: a preset of the options below | SuperMimic's semi-implicit model | beyond the horizon |

**How the methods relate.** SuperTrack trains the policy by supervised L1 tracking through the world model, with no
reward and no critic. SuperMimic backpropagates the imagined reward through the world model, with a critic beyond the
horizon (first-order gradients). [BlinkMimic](blinkmimic.md) is SuperMimic's objective with InstantTrack's and its own
training changes; its semi-implicit world model, which predicts velocities and integrates positions, is the idea of
SuperTrack's acceleration world model.

## The training loop

`ndm_gym/train.py` (`ndm-train`) runs SuperMimic's Algorithm A.1 for every algorithm: collect transitions in the
simulator with the current policy (`--num_envs` environments), fit the world model on k-step windows of the replay
buffer (`--wm_window`, `--wm_batches` per iteration), then update the policy on rollouts imagined in the world model
from replay states (`--horizon` steps, `--pi_batch` rollouts). The first `--wm_only_iters` iterations fit the world model
only. `--max_hours` stops with a checkpoint; `--resume` continues from a run's `last.pt`.

## Options

`ndm-train --help` lists every option (without the simulator, `python -m ndm_gym.wmbench.fit --help` lists those of
`add_train_args`). They are defined in `ndm_gym/train.py` (`add_train_args`, `add_speed_args`) and
`ndm_gym/algos/blinkmimic.py` (`add_blinkmimic_args`); each help text starts with the algorithms that read it. An
option's default keeps the behavior from before the option; the equivalence tests against the upstream reproduction
and the SuperTrack golden tests check the defaults bit for bit.

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

BlinkMimic's changes, each an option of its own and off by default, are listed with their counterparts in
[BlinkMimic](blinkmimic.md#3-blinkmimics-changes-as-options); `--algo blinkmimic` sets its recipe, and options given
with it take precedence. The configurations selected for each task are in the
[README](https://github.com/Frank-ZY-Dou/ndm-gym#-results).

## The interface

An algorithm subclasses `ndm_gym.algos.common.Algo`, which owns the world model, the policy and their optimizers, fits
the world model on replay windows and samples imagination starts. A subclass defines:

| | |
|---|---|
| `world_model_cls(a)` | the world model for the training arguments `a` (also a checkpoint's `args`) |
| `setpoint(a, f)` | the normalized joint setpoint sent to the simulator for policy output `a` at reference frame `f` |
| `wm_loss(S_hat, w)` | the world-model loss of predicted windows against replay windows |
| `policy_update(buffer)` | one policy (and critic) update from replay states |
| `state_dict()`, `load_state_dict(ck)` | its checkpoint state |

and is registered in `ALGOS` (`ndm_gym/algos/__init__.py`). The HOI and muscle tasks have their own algorithm classes
on the same pattern (`ndm_gym/algos/hoi_algos.py`, `ndm_gym/algos/muscle_algos.py`).
