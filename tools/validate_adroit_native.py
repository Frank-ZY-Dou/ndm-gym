"""Contact/reset oracle and matched collection timings on the selected native scene."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.motion.adroit import AdroitReference
from ndm_gym.sim.rigid.adroit import AdroitPhysics
from ndm_gym.sim.rigid.door_collector import PackedNativeDoorCollector, ParallelNativeDoorCollector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(a.model))
    ref = AdroitReference.load(a.reference)
    serial = PackedNativeDoorCollector(model, ref, 5, physics_cls=AdroitPhysics)
    rng = np.random.default_rng(20261117)
    with ParallelNativeDoorCollector(model, ref, 5, 2, a.native_library, physics_cls=AdroitPhysics) as fast:
        for frame in range(ref.frames+3):
            actions = np.repeat(ref.actions[frame % (ref.frames-1)][None], 5, axis=0)
            actions += rng.normal(0, .001, actions.shape)
            expected, actual = serial.step(actions), fast.step(actions)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(serial.states, np.concatenate([c.states for c in fast.chunks]))
    rows = []
    for phase in [0, (ref.frames-1)//2, ref.frames-2]:
        with ParallelNativeDoorCollector(model, ref, 2048, 8, a.native_library, physics_cls=AdroitPhysics) as fast:
            fast.stagger_from_physical_replay(0)
            state = fast.state_dict()
            # Replicate one physically generated phase to compare the same contact load.
            index = int(np.flatnonzero(state['frames'] == phase)[0])
            for key in ('states', 'qpos', 'qvel', 'frames', 'previous_actions'):
                state[key][:] = state[key][index]
            actions = np.repeat(ref.actions[phase:phase+1], 2048, axis=0)
            with ParallelNativeDoorCollector(model, ref, 2048, 8, physics_cls=AdroitPhysics) as slow:
                timings = {}
                for label, collector in [('python', slow), ('native', fast)]:
                    samples = []
                    for _ in range(5):
                        collector.load_state_dict(state)
                        begin = time.perf_counter()
                        result = collector.step(actions)
                        samples.append(time.perf_counter()-begin)
                    timings[label] = float(np.median(samples))
                    if label == 'python':
                        oracle = result
                    else:
                        for key in oracle:
                            np.testing.assert_array_equal(result[key], oracle[key])
                rows.append(dict(phase=phase, seconds=timings))
    a.output.write_text(json.dumps(dict(task=ref.metadata['task'], worlds=2048, workers=8,
                                       oracle='bit-exact whole physical clip, contact, reset, complete integration state',
                                       timings=rows, scope='collection only, same host sequential timings'), indent=2)+'\n')
    print(a.output.read_text())


if __name__ == '__main__':
    main()
