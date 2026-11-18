"""Evaluate the zero-residual reference control in deployed float32 precision.

Keep the original float64 reference-control evaluation as a separate control.
This supplement changes no training trajectory, evaluation start or target.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.eval_door_full import evaluation_starts, initialize_evaluation, evaluate, score
from ndm_gym.motion.adroit_door import DoorReference, JOINT_NAMES
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.train_door_full import make_algorithm


def deployed_control_reference(reference):
    """The same rounding/clamp as the policy's stored reference action buffer."""
    actions = torch.as_tensor(reference.actions, dtype=torch.float32).clamp(-1, 1).numpy()
    return replace(reference, actions=actions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--door-latch-offset', type=float, default=0.,
                   help='Starts of eval_door_full --door-latch-offset (default 0: the original starts)')
    a = p.parse_args()
    if not np.isfinite(a.door_latch_offset) or a.door_latch_offset < 0:
        p.error('--door-latch-offset must be finite and nonnegative')
    import mujoco
    torch.set_num_threads(1)
    reference = DoorReference.load(a.reference)
    control = deployed_control_reference(reference)
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    algo, _ = make_algorithm('supertrack', model, reference, 'cpu')
    a.output.mkdir(parents=True, exist_ok=False)
    offsets, velocities = evaluation_starts(a.door_latch_offset)
    np.savez_compressed(a.output/'starts.npz', qpos=offsets, qvel=velocities)
    started = time.perf_counter()
    with ParallelNativeDoorCollector(model, reference, 256, a.workers, a.native_library.resolve()) as collector:
        initialize_evaluation(collector, offsets, velocities)
        q, v, actions = evaluate(algo, collector, control, prior=True)
    results = score(q, v, actions, reference, algo.task.cfg)
    np.savez_compressed(a.output/'physical-rollouts.npz', **{
        f'episode{i}_{key}': value[:, i] for i in range(256)
        for key, value in [('qpos', q), ('qvel', v), ('actions', actions)]})
    report = dict(prior=True, method=None, seed=None, iteration=None, full_budget=False,
                  checkpoint_sha256=None, control_precision='float32, matching deployed reference buffer',
                  starts_sha256=hashlib.sha256((a.output/'starts.npz').read_bytes()).hexdigest(),
                  input_sha256={k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
                                for k in ('model', 'reference', 'native_library')},
                  source_joint_order=JOINT_NAMES, evaluations=results,
                  means={k: float(np.mean([r[k] for r in results])) for k in results[0]},
                  wall_seconds=time.perf_counter()-started,
                  evaluation_scope='Same 256 starts and physical targets; precision-matched zero-residual control supplement')
    if a.door_latch_offset:
        report['door_latch_offset'] = a.door_latch_offset
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['means']))


if __name__ == '__main__':
    main()
