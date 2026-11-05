"""Isaac Lab's launcher (``isaaclab.app``) for the commands that run the simulator: training, evaluation, collection.

With Isaac Lab installed, ``add_launcher_args`` and ``launch_simulation`` are Isaac Lab's own. Without it the commands
still import, with their argument definitions (``ndm_gym.wmbench.fit`` reads ``ndm_gym.train``'s), and both raise
:class:`ndm_gym.sim.SimulatorNotInstalled`, which says how to install the simulator.
"""

from __future__ import annotations

import importlib.util

from . import require_simulator

if importlib.util.find_spec("isaaclab") is None:

    def add_launcher_args(parser, *args, **kwargs):
        """Raises :class:`ndm_gym.sim.SimulatorNotInstalled`: Isaac Lab is not installed."""
        require_simulator(parser.prog)

    def launch_simulation(*args, **kwargs):
        """Raises :class:`ndm_gym.sim.SimulatorNotInstalled`: Isaac Lab is not installed."""
        require_simulator("launch_simulation")

else:
    from isaaclab.app import add_launcher_args, launch_simulation  # noqa: F401
