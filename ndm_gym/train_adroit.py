"""Native Pen/Hammer/Relocate with the shared full-budget Door training engine.

Task identity is read from the validated physical reference. Each method uses
its own world-model parameterization and policy objective. Existing Door defaults
and checkpoint schemas remain unchanged.
"""
from .motion.adroit import AdroitReference
from .sim.rigid.adroit import AdroitPhysics
from functools import partial

from .tasks.adroit import AdroitTask, training_arguments, TRAINING_HELP
from .train_door_full import main as train


def main(argv=None):
    cfg, remaining = training_arguments(argv)
    return train(remaining, reference_cls=AdroitReference, task_cls=partial(AdroitTask, cfg=cfg), physics_cls=AdroitPhysics, extra_help=TRAINING_HELP)


if __name__ == '__main__':
    main()
