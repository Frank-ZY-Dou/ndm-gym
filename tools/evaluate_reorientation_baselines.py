"""Reference backend diagnostics; see benchmarks/in_hand_reorientation/protocol.md."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', default='data/menagerie/wonik_allegro')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from ndm_gym.sim.rigid.reorientation import ReorientationSim

    rows = []
    for seed in (0, 1, 2):
        for policy in ('midpoint', 'random'):
            sim = ReorientationSim(a.model_dir, seed=seed)
            rng = torch.Generator().manual_seed(seed + 100)
            episodes = []
            for _ in range(20):
                sim.reset()
                for step in range(sim.cfg.episode_steps):
                    action = torch.zeros(16) if policy == 'midpoint' else 2 * torch.rand(16, generator=rng) - 1
                    _, _, info = sim.step(action)
                    if (info['terminated'] | info['truncated']).item():
                        break
                episodes.append(dict(steps=step + 1, angle=info['angle'].item(),
                                     success=info['success'].item(), dropped=info['dropped'].item()))
            rows.append(dict(seed=seed, policy=policy, episodes=episodes,
                             success=np.mean([x['success'] for x in episodes]),
                             dropped=np.mean([x['dropped'] for x in episodes]),
                             final_angle=np.mean([x['angle'] for x in episodes])))
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print({k: v for k, v in r.items() if k != 'episodes'})


if __name__ == '__main__':
    main()
