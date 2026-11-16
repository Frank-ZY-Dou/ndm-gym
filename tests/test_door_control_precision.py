"""The float32 control must reproduce an actual deployed zero-residual policy."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import torch

from test_adroit_door_physics import setup
from ndm_gym.algos.door import DoorResidual
from ndm_gym.eval_door_full import evaluate
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.tasks.adroit_door import DoorTask
from ndm_gym.worldmodels.mjcf import MjcfLayout

SPEC = importlib.util.spec_from_file_location('control', Path(__file__).parents[1]/'tools/eval_door_control.py')
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


def test_control_actions_and_physics_match_deployed_zero_policy():
    model, ref = setup()
    ref = replace(ref, actions=np.full_like(ref.actions, 1/3))
    task = DoorTask(ref, model.actuator_ctrlrange)
    algo = DoorResidual(task, replace(MjcfLayout.from_model(model), dt=task.dt),
                        torch.tensor(ref.actions, dtype=torch.float32), horizon=1, hidden=(8,), wm_hidden=(8,))
    control = CONTROL.deployed_control_reference(ref)
    assert control.actions.dtype == np.float32
    assert control.qpos is ref.qpos and control.qvel is ref.qvel
    with ParallelNativeDoorCollector(model, ref, 256, 1) as collector:
        deployed = evaluate(algo, collector, ref, prior=False)
    with ParallelNativeDoorCollector(model, ref, 256, 1) as collector:
        baseline = evaluate(algo, collector, control, prior=True)
    for actual, expected in zip(baseline, deployed):
        np.testing.assert_array_equal(actual, expected)
