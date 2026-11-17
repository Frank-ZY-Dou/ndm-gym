"""Development-only native contact equivalence and packed capacity measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import mujoco
import numpy as np

from ndm_gym.motion.adroit_door import DoorReference
from ndm_gym.sim.rigid.door_collector import NativeDoorCollector, PackedNativeDoorCollector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1:
        parser.error('worlds and steps must be positive')
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    ref = DoorReference.load(args.reference)
    separate = NativeDoorCollector(model, ref, 2)
    packed = PackedNativeDoorCollector(model, ref, 2)
    rng = np.random.default_rng(3062)
    max_contacts = 0
    started = time.perf_counter()
    # Complete physical trajectory plus reset: replay and perturbed controls.
    for step in range(ref.frames + 1):
        prior = ref.actions[step % (ref.frames - 1)]
        actions = np.stack([prior, np.clip(prior + rng.normal(0, .03, 28), -1, 1)])
        expected, actual = separate.step(actions), packed.step(actions)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for key in separate.observe():
            np.testing.assert_array_equal(packed.observe()[key], separate.observe()[key])
        max_contacts = max(max_contacts, *(world.data.ncon for world in separate.worlds))
    equivalence_seconds = time.perf_counter() - started
    if max_contacts == 0:
        raise RuntimeError('No contacts exercised; cannot certify contact equivalence')
    del separate, packed
    started = time.perf_counter()
    collector = PackedNativeDoorCollector(model, ref, args.worlds)
    construction_seconds = time.perf_counter() - started
    timings = []
    for step in range(args.steps):
        actions = np.repeat(ref.actions[step % len(ref.actions)][None], args.worlds, axis=0)
        started = time.perf_counter()
        collector.step(actions)
        timings.append(time.perf_counter() - started)
    result = dict(development_only=True, mujoco=mujoco.__version__, worlds=args.worlds,
                  steps=args.steps, noslip_iterations=int(model.opt.noslip_iterations),
                  equivalence_transitions=2 * (ref.frames + 1), max_contacts=int(max_contacts),
                  equivalence='bit-exact all returned arrays and post-reset observations',
                  equivalence_seconds=equivalence_seconds, construction_seconds=construction_seconds,
                  collection_seconds=timings, transitions_per_second=args.worlds / np.mean(timings),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  packed_state_bytes=collector.states.nbytes,
                  projected_20032_collection_seconds=float(np.mean(timings) * 20032),
                  projection_limit='Early-trajectory collection only; excludes learning and late-contact cost',
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.model, args.reference)})
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
