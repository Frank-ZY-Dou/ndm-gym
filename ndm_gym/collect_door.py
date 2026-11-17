"""Generate native physical reference-control episodes for Door model bootstrap.

This is a new reproducible dataset, not regeneration of historical mixed2934.
CPU physics only; no learned controller or supervised fitting.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .motion.adroit_door import DoorReference
from .motion.door_replay import collect_physical_replay
from .train_door_online import fingerprint


def save_replay(replay, path):
    """Write complete episodes in the public DoorReplay.load format."""
    np.savez_compressed(path, **{f'episode{i}_{key}': value.numpy()
                               for i, episode in enumerate(replay.episodes)
                               for key, value in zip(('qpos', 'qvel', 'actions'), episode)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--offsets', type=float, nargs='+', default=[-.01, 0., .01, .005])
    p.add_argument('--noise', type=float, default=.01, help='IID normalized-control noise standard deviation')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if (len(args.offsets) < 2 or not np.isfinite(args.offsets + [args.noise]).all()
            or args.noise < 0):
        raise ValueError('At least two finite offsets and nonnegative finite noise required')
    import mujoco
    from .sim.rigid.adroit_door import DoorPhysics

    torch.set_num_threads(1)
    reference = DoorReference.load(args.reference)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    sim = DoorPhysics(model, reference)
    source = SimpleNamespace(reference_actions=torch.tensor(reference.actions, dtype=torch.float32).clamp(-1, 1))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    replay = collect_physical_replay(source, sim, args.offsets, args.noise, args.seed)
    elapsed = time.perf_counter() - started
    save_replay(replay, args.output / 'episodes.npz')
    result = dict(format=1, seed=args.seed, offsets=args.offsets, noise=args.noise,
                  model_sha256=fingerprint(args.model), reference_sha256=fingerprint(args.reference),
                  dataset_sha256=fingerprint(args.output / 'episodes.npz'),
                  physics_seconds=elapsed, transitions=sum(len(e[2]) for e in replay.episodes),
                  cpu_threads=1, gpu_compute_seconds=0, mujoco=mujoco.__version__,
                  actions='clipped reference plus seeded IID Gaussian noise; no state assignment after reset',
                  holdout='online trainer excludes last episode before normalization and sampling',
                  historical_reproduction=False)
    (args.output / 'generation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
