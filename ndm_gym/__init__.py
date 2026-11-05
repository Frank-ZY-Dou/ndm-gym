"""NDM Gym (Neural Dynamics Modeling Gym): learned differentiable dynamics models for physics-based character and
humanoid control.

Layers: ``core`` (math, kinematics, networks) -> ``robots`` -> ``motion`` -> ``sim`` -> ``tasks`` -> ``worldmodels``
-> ``policies`` -> ``algos`` (SuperTrack, SuperMimic) -> ``train`` / ``evaluate``.

A ``pip install`` of this repository installs the library, which needs PyTorch, NumPy and MuJoCo; the simulator (Isaac Lab 3 with
Newton and MuJoCo-Warp) and the data come from ``scripts/setup.sh``. Without the simulator, the simulator modules and
the commands that run it raise :class:`ndm_gym.sim.SimulatorNotInstalled`.
"""

__version__ = "0.2.0"
