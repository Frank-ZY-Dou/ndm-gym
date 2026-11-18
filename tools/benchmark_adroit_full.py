"""Exact-capacity warmup/timing using the shared full-budget benchmark engine."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_door_full import main
from ndm_gym.motion.adroit import AdroitReference
from ndm_gym.sim.rigid.adroit import AdroitPhysics
from ndm_gym.tasks.adroit import AdroitTask, training_arguments, TRAINING_HELP

if __name__ == '__main__':
    cfg, remaining = training_arguments()
    main(remaining, reference_cls=AdroitReference, task_cls=partial(AdroitTask, cfg=cfg), physics_cls=AdroitPhysics, extra_help=TRAINING_HELP)
