"""Native Pen/Hammer/Relocate with the shared full-budget Door training engine.

Task identity is read from the validated physical reference. Each method uses
its own world-model parameterization and policy objective. Existing Door defaults
and checkpoint schemas remain unchanged.
"""
from .motion.adroit import AdroitReference
from .sim.rigid.adroit import AdroitPhysics
from .tasks.adroit import AdroitTask
from .train_door_full import main as train


def main(argv=None):
    return train(argv, reference_cls=AdroitReference, task_cls=AdroitTask, physics_cls=AdroitPhysics)


if __name__ == '__main__':
    main()
