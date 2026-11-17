"""Real-contact development benchmark; never launches training or infers total ETA."""
import argparse
import json
import resource
import time
from pathlib import Path

import mujoco
import numpy as np

from ndm_gym.motion.adroit_door import DoorReference
from ndm_gym.sim.rigid.door_collector import (
    NativeDoorCollector, ParallelNativeDoorCollector,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--worlds', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--native-library', help='Trusted library built for this MuJoCo runtime')
    a = p.parse_args()
    if a.worlds < 8 or a.repeats < 1:
        p.error('worlds >= 8 and repeats >= 1 required')
    model = mujoco.MjModel.from_xml_path(a.model)
    ref = DoorReference.load(Path(a.reference))
    oracle = NativeDoorCollector(model, ref, 8)
    pools = [ParallelNativeDoorCollector(model, ref, 8, n, a.native_library) for n in (1, 2, 4, 8)]
    previous = np.zeros((8, 28))
    contacts = 0
    snapshots = []
    try:
        for frame in range(ref.frames + 2):
            actions = np.repeat(ref.actions[frame % len(ref.actions)][None], 8, axis=0)
            actions[1:] = np.clip(actions[1:] + np.random.default_rng(frame).normal(0, .03, (7, 28)), -1, 1)
            expected = oracle.step(actions)
            for pool in pools:
                actual = pool.step(actions)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
                np.testing.assert_array_equal(actual['previous_actions'], previous)
                for key in oracle.observe():
                    np.testing.assert_array_equal(pool.observe()[key], oracle.observe()[key])
            previous[:] = actions
            previous[expected['done']] = 0
            contacts = max(contacts, *(s.data.ncon for s in oracle.worlds))
            if frame in (0, len(ref.actions) // 2, len(ref.actions) - 2):
                c = pools[0].chunks[0]
                snapshots.append((c.states[0].copy(), c.qpos[0].copy(), c.qvel[0].copy(), int(c.frames[0])))
    finally:
        for pool in pools:
            pool.close()
    if contacts < 1:
        raise RuntimeError('Contact equivalence not exercised')
    measurements = []
    for workers in (1, 2, 4, 8):
        started = time.perf_counter()
        with ParallelNativeDoorCollector(model, ref, a.worlds, workers, a.native_library) as pool:
            init = time.perf_counter() - started
            for phase, (state, q, v, frame) in enumerate(snapshots):
                for chunk in pool.chunks:
                    chunk.states[:] = state
                    chunk.qpos[:] = q
                    chunk.qvel[:] = v
                    chunk.frames[:] = frame
                timings = []
                for _ in range(a.repeats):
                    frames = pool.observe()['frames']
                    actions = ref.actions[frames]
                    started = time.perf_counter()
                    pool.step(actions)
                    timings.append(time.perf_counter() - started)
                measurements.append(dict(workers=workers, phase=phase, initial_frame=frame,
                                         initialization_seconds=init, seconds=timings,
                                         median_seconds=float(np.median(timings)),
                                         p95_seconds=float(np.percentile(timings, 95))))
    result = dict(development_only=True, worlds=a.worlds, contacts=contacts, native_library=a.native_library,
                  equivalence='bit-exact serial oracle, perturbed full contact trajectory, terminal/reset and previous actions',
                  noslip_iterations=int(model.opt.noslip_iterations), mujoco=mujoco.__version__,
                  measurements=measurements,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  limitations='Collection only. No learner, replay, GPU, compile/checkpoint or end-to-end ETA certification.')
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
