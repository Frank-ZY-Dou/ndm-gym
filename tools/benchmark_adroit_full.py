"""Exact-capacity warmup/timing using the shared full-budget benchmark engine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_door_full import main
from ndm_gym.motion.adroit import AdroitReference
from ndm_gym.sim.rigid.adroit import AdroitPhysics
from ndm_gym.tasks.adroit import AdroitTask

if __name__ == '__main__':
    main(reference_cls=AdroitReference, task_cls=AdroitTask, physics_cls=AdroitPhysics)
