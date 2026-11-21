# Simulator

The simulator modules `ndm_gym.sim.rigid.isaaclab_mjwarp` and `hdmi_sim` need Isaac Lab 3, Newton and MuJoCo-Warp,
`muscle_sim` Newton and MuJoCo-Warp, `hand_sim`, `quadrotor_sim`, `quadruped_sim` and `robomimic_sim` MuJoCo-Warp alone ([Installation](../installation.md)).
Without them, they and the commands that run them raise
`SimulatorNotInstalled`.

::: ndm_gym.sim

::: ndm_gym.sim.launcher

RoboKit's simulator subclasses need RoboKit and MuJoCo Warp, imported when they are built ([task](../tasks/robokit.md)).

::: ndm_gym.sim.robokit_ndm

::: ndm_gym.sim.robokit_rec

mbrl-lib's environments for its controller on batched simulators need mbrl-lib (at `3f93ccc`) with gymnasium 0.26,
imported when they are built ([task](../tasks/mbrl.md)).

::: ndm_gym.sim.mbrl_envs

::: ndm_gym.sim.mbrl_ndm
