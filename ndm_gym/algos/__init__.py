"""Learning algorithms on the shared pipeline, selected by ``--algo``."""

from .supermimic import SuperMimic
from .supertrack import SuperTrack

ALGOS = {"supermimic": SuperMimic, "supertrack": SuperTrack}


def world_model_from_args(a: dict):
    """The (untrained) world model a run with training arguments ``a`` (a checkpoint's ``args``) builds."""
    return ALGOS[a["algo"]].world_model_cls(a)

# Native articulated Door has its own state/task contract, like HAND_ALGOS.
from .door import DoorResidual  # noqa: E402

from .door_supertrack import DoorSuperTrack  # noqa: E402
from .door_supermimic import DoorSuperMimic  # noqa: E402

DOOR_ALGOS = {"residual": DoorResidual, "supertrack": DoorSuperTrack, "supermimic": DoorSuperMimic}
