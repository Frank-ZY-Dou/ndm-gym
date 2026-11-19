"""Score-driven inverse-kinematics reference, evaluated in the same free-hand piano scene as the policies."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch
import warp as wp

from ode_to_joy import DIGITS, LOWEST_PITCH, action_layout, build, piano_roll
from train_ode_to_joy import DEV, FreeHands, evaluate, seconds, starts


def commands(model, score):
    ids, rest, scale = action_layout(model)
    pose = mujoco.MjData(model)
    hover = np.zeros_like(rest)
    # A seven-centimetre wrist shift places the left index and pinky above the two bass keys.
    hover[1] = -0.070 / scale[1]
    pose.qpos[int(model.joint("left_free").qposadr[0]) + 1] -= 0.070
    active = [(0, 1, 48), (0, 4, 43), *((1, i, pitch) for i, pitch in enumerate((60, 62, 64, 65, 67)))]
    strikes = []
    for hand, side in enumerate(("left", "right")):
        for digit, (name, _) in enumerate(DIGITS):
            offset = hand * 26 + 6 + digit * 4
            axes = slice(offset, offset + 4)
            qadr = model.jnt_qposadr[model.actuator_trnid[ids[axes], 0]]
            lo, hi = model.actuator_ctrlrange[ids[axes]].T
            lo = np.maximum(-1, (lo - rest[axes]) / scale[axes])
            hi = np.minimum(1, (hi - rest[axes]) / scale[axes])
            site = model.site(f"{side}/{side[0]}_{name}_tip").id
            pitch = next((p for h, d, p in active if (h, d) == (hand, digit)), None)
            pose.qpos[qadr] = rest[axes]
            mujoco.mj_kinematics(model, pose)
            x, y = pose.site_xpos[site, :2]
            if pitch is not None:
                y = model.body(f"key_{pitch - LOWEST_PITCH}").pos[1]

            def solve(z):
                target = np.array([x, y, z])

                def residual(a):
                    pose.qpos[qadr] = rest[axes] + a * scale[axes]
                    mujoco.mj_kinematics(model, pose)
                    return np.r_[1000 * (pose.site_xpos[site] - target), a * 0.25]

                return least_squares(residual, np.clip(np.zeros(4), lo, hi), bounds=(lo, hi)).x

            hover[axes] = solve(0.042 if pitch is not None else 0.052)
            if pitch is not None:
                strikes.append((pitch, axes, solve(0.004)))
    trajectory = np.tile(hover, (len(score), 1))
    # Command one 50 ms tick ahead to allow the position servos to respond.
    ahead = np.concatenate((score[1:], np.zeros_like(score[:1])))
    for pitch, axes, strike in strikes:
        trajectory[ahead[:, pitch - LOWEST_PITCH], axes] = strike
    return trajectory


class Script:
    def __init__(self, trajectory):
        self.actions = torch.as_tensor(trajectory, device=DEV)

    def action(self, state, prev, f):
        return self.actions[f].expand(len(state), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand-dir", type=Path, required=True)
    p.add_argument("--keyboard", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--eval-starts", type=int, default=256)
    args = p.parse_args()
    if min(args.worlds, args.eval_starts) < 1:
        p.error("world and evaluation counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("*.npz")) or (args.out / "results.json").exists():
        p.error("choose a fresh baseline output directory")
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    with wp.ScopedDevice(DEV):
        m = build(args.hand_dir, args.keyboard)
        score = piano_roll()
        policy = Script(commands(m, score))
        sim = FreeHands(m, args.worlds)
        initial = starts(args.eval_starts, 789, 0.20)
        t = seconds()
        stats, masks, trace = evaluate(sim, policy, torch.as_tensor(score, device=DEV), initial, record=True)
        elapsed = seconds() - t
    np.savez_compressed(args.out / "evaluation.npz", played=masks.cpu().numpy(), goal=score, starts=initial.cpu().numpy())
    np.savez_compressed(args.out / "rollout.npz", **trace)
    result = dict(test=stats, evaluation_s=elapsed, gpu=torch.cuda.get_device_name(), worlds=args.worlds)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
