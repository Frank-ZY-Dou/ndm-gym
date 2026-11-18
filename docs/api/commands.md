# Commands

The training and evaluation commands: `ndm-train` and `ndm-eval` (console entry points), and `python -m` for the other
tasks. Their arguments are built by these functions, so that other tools take them too (`ndm_gym.wmbench.fit` takes
`ndm_gym.train`'s). The commands that run the simulator need it ([Installation](../installation.md)); their argument
definitions import without it.

::: ndm_gym.train
    options:
      members: [main, parse_args, add_train_args, add_speed_args]

::: ndm_gym.evaluate
    options:
      members: [main, parse_args, evaluate, evaluate_group]

::: ndm_gym.configs

::: ndm_gym.train_hoi
    options:
      members: [main, parse_args]

::: ndm_gym.evaluate_hoi
    options:
      members: [main, parse_args]

::: ndm_gym.train_ppo_hoi
    options:
      members: [main, parse_args]

::: ndm_gym.train_muscles
    options:
      members: [main, parse_args]

::: ndm_gym.train_hand
    options:
      members: [main, parse_args]

::: ndm_gym.evaluate_hand
    options:
      members: [main, rollout]

::: ndm_gym.train_quadrotor
    options:
      members: [main, parse_args, build]

::: ndm_gym.evaluate_quadrotor
    options:
      members: [main, rollout]

::: ndm_gym.train_quadruped
    options:
      members: [main, parse_args, build]

::: ndm_gym.evaluate_quadruped
    options:
      members: [main, rollout_sums, summarize_sums, minutes_to_success]

::: ndm_gym.train_robomimic
    options:
      members: [main, parse_args, build]

::: ndm_gym.evaluate_robomimic
    options:
      members: [main, rollout]
