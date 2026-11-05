"""Simulators: articulated rigid bodies (``rigid``).

The simulator stack (Isaac Lab 3 with Newton and MuJoCo-Warp, at the versions of Isaac Lab's lock) is installed by
``scripts/setup.sh``, not by a ``pip install`` of this repository. The modules that need it check for it on import
(``require_simulator``), as Isaac Lab's launcher does in the commands that run the simulator (``launcher``).
"""

from __future__ import annotations

import importlib.util

SIMULATOR = ("isaaclab", "isaaclab_newton", "newton", "warp", "mujoco_warp")  # the import names of the simulator stack


class SimulatorNotInstalled(ModuleNotFoundError):
    """A simulator feature was used where the simulator is not installed."""


def require_simulator(feature: str, modules: tuple[str, ...] = SIMULATOR) -> None:
    """Raise :class:`SimulatorNotInstalled` for ``feature`` (a module or a command) unless all of ``modules`` are
    installed. Looks the modules up without importing them."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise SimulatorNotInstalled(f"{feature} needs Isaac Lab 3 with Newton and MuJoCo-Warp: run scripts/setup.sh, which installs them "
                                    f"(a pip install of this repository gives the library without the simulator). Not installed: {', '.join(missing)}",
                                    name=missing[0])  # fmt: skip
