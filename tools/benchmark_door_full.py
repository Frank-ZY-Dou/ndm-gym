"""Exact-capacity end-to-end development timing, not a final training seed.

Uses the full runner's collector, normalizers, replay and update functions.
Measured iterations represent the post-WM-only phase. Startup, initialization,
warmup and checkpoint costs are separate, and no convergence claim is made.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from ndm_gym.motion.adroit_door import DoorReference
from ndm_gym.tasks.adroit_door import DoorTask
from ndm_gym.sim.rigid.adroit_door import DoorPhysics
from ndm_gym.motion.door_ring import DoorReplayRing
from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
from ndm_gym.train_door_full import collect, make_algorithm, save
from ndm_gym.train_door_online import optimizer_counts, updates


def main(argv=None, *, reference_cls=DoorReference, task_cls=DoorTask, physics_cls=DoorPhysics, extra_help=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=extra_help)
    for name in ('model', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--native-library', type=Path, help='Omit to time Python stepping as the control')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--method', choices=('supertrack', 'supermimic'), required=True)
    a = p.parse_args(argv)
    if a.steps < 1 or not 1 <= a.workers <= 2048:
        p.error('Invalid timing budget')
    import mujoco
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(3065)
    started = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    ref = reference_cls.load(a.reference)
    algo, _ = make_algorithm(a.method, model, ref, 'cuda:0', task_cls=task_cls)
    ring = DoorReplayRing(256, 2048, 'cuda:0', nq=model.nq, nv=model.nv, nu=model.nu)
    with ParallelNativeDoorCollector(model, ref, 2048, a.workers, a.native_library, physics_cls=physics_cls) as collector:
        collector.stagger_from_physical_replay(3065)
        for _ in range(32):
            collect(algo, collector, ring)
        torch.cuda.synchronize()
        startup = time.perf_counter()-started
        records = []
        for step in range(a.steps):
            tick = time.perf_counter()
            collect(algo, collector, ring)
            torch.cuda.synchronize()
            sampling = time.perf_counter()-tick
            tick = time.perf_counter()
            updates(algo, ring, 4096, 16, 4, 1, wm_batch=2048, wm_window=8)
            torch.cuda.synchronize()
            learning = time.perf_counter()-tick
            row = dict(step=step, collection_seconds=sampling, learning_seconds=learning,
                       iteration_seconds=sampling+learning)
            records.append(row)
            print(json.dumps(row), flush=True)
        expected = dict(wm=4*a.steps, policy=a.steps)
        if a.method == 'supermimic':
            expected['critic'] = 20*a.steps
        assert optimizer_counts(algo) == expected
        tick = time.perf_counter()
        save(a.output/'development.pt', algo, collector, ring,
             {'development_only': True, 'method': a.method}, a.steps, {})
        checkpoint = time.perf_counter()-tick
        result = dict(development_only=True, method=a.method, worlds=2048, workers=a.workers,
                      native_library=str(a.native_library), startup_seconds=startup,
                      checkpoint_seconds=checkpoint, invocation_seconds=time.perf_counter()-started,
                      optimizer_counts=expected, records=records,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      warm_median_seconds=float(np.median([r['iteration_seconds'] for r in records[1:] or records])),
                      limitations='Short post-WM-only timing, no full seed/evaluation; original Door adaptation differences remain.')
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    del algo, ring
    gc.collect()


if __name__ == '__main__':
    main()
