"""Measure physical key onsets in open-loop predictions from each checkpoint's held-out nominal rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from note_events import TICK_MS, match_notes
from ndm_gym.worldmodels.piano import NUM_ACTUATORS as A, PianoDynamics
from train_ode_to_joy import DEV, KEY_DOWN_FRACTION, build, fit_geometry


def attacks(frames):
    times, pitches = np.nonzero(frames[1:] & ~frames[:-1])
    return [(int(p), int(t), 0) for t, p in zip(times, pitches, strict=True)]


@torch.no_grad()
def measure(wm, trace):
    state, action = (torch.as_tensor(trace[k], device=DEV) for k in ("states", "actions"))
    physical = (state[:, 2 * A :] >= KEY_DOWN_FRACTION).cpu().numpy()
    result = {}
    for horizon in (1, 3, 10):
        n = len(action) - horizon + 1
        pred = state[:n]
        window = [physical[:n]]
        for j in range(horizon):
            pred = wm(pred, action[j : j + n])
            window.append((pred[:, 2 * A :] >= KEY_DOWN_FRACTION).cpu().numpy())
        window = np.stack(window, axis=1)
        true_count = pred_count = 0
        errors = []
        for i in range(n):
            actual = attacks(physical[i : i + horizon + 1])
            modelled = attacks(window[i])
            true_count += len(actual)
            pred_count += len(modelled)
            errors.extend(abs(actual[t][1] - modelled[p][1]) * TICK_MS for t, p in match_notes(actual, modelled))
        result[horizon] = dict(actual=true_count, predicted=pred_count, matched=len(errors), mean_abs_ms=float(np.mean(errors)) if errors else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand-dir", required=True, type=Path)
    p.add_argument("--keyboard", required=True, type=Path)
    p.add_argument("--trace-root", type=Path, help="Use seedN/rollout.npz from a fresh replay")
    p.add_argument("seeds", type=Path, nargs="+")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = PianoDynamics(fit_geometry(build(args.hand_dir, args.keyboard))).to(DEV)
    for seed in args.seeds:
        ck = torch.load(seed / "last.pt", map_location=DEV, weights_only=True)
        model.load_state_dict(ck["wm"])
        folder = args.trace_root / seed.name if args.trace_root else seed
        with np.load(folder / "rollout.npz") as trace:
            print(json.dumps(dict(seed=str(seed), horizons=measure(model, trace))))


if __name__ == "__main__":
    main()
