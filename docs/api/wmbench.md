# World-model benchmark

Offline: transition datasets collected once in the simulator (`collect`), world models fit and scored on them without it
(`fit`).

::: ndm_gym.wmbench.fit

::: ndm_gym.wmbench.collect
    options:
      members: [pack, unpack]

::: ndm_gym.wmbench.fit_hoi

RoboKit's examples on a world model ([task](../tasks/robokit.md)): recordings, fits of any MuJoCo model's world model, their
throughput against MuJoCo Warp, and plans.

::: ndm_gym.wmbench.collect_robokit

::: ndm_gym.wmbench.fit_mjcf

::: ndm_gym.wmbench.speed_mjcf

::: ndm_gym.wmbench.plan_robokit

mbrl-lib's controller on a world model ([task](../tasks/mbrl.md)): recordings, plans, the
step's throughput against CPU MuJoCo and MuJoCo Warp, and the check of the batched rewards.

::: ndm_gym.wmbench.collect_mbrl

::: ndm_gym.wmbench.plan_mbrl

::: ndm_gym.wmbench.speed_mbrl

::: ndm_gym.wmbench.check_mbrl_rewards
