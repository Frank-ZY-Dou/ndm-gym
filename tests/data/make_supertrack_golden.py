"""Writes tests/data/supertrack_golden.pt: the outputs of this repository's SuperTrack components on synthetic clips and
states (no motion data), for tests/test_supertrack_golden.py. The clips are stored with the outputs, so the test rebuilds
the same motion library. Needs the G1 description in data/ (scripts/setup.sh), not the LAFAN1 motions.

Run from the repository root: python tests/data/make_supertrack_golden.py
"""

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ndm_gym.algos.supertrack import LossBalancer, ReferenceLocal, local_features, track_terms, wm_window_terms  # noqa: E402
from ndm_gym.core.rotations import mat_to_quat, so3_exp  # noqa: E402
from ndm_gym.motion.library import MotionLib  # noqa: E402
from ndm_gym.motion.sources import Clip  # noqa: E402
from ndm_gym.robots import g1  # noqa: E402
from ndm_gym.state import flatten_state, normalize_xi, unflatten_state  # noqa: E402
from ndm_gym.worldmodels.accel import AccelWorldModel  # noqa: E402

OUT = os.path.join(REPO, "tests", "data", "supertrack_golden.pt")
FPS, T, N = 30.0, 121, 64  # 4 s clips; N states


def rot(i: int, angle: torch.Tensor) -> torch.Tensor:
    """Rotations about axis i (0: x, 1: y, 2: z) by the angles (T,)."""
    v = torch.zeros(angle.shape[0], 3)
    v[:, i] = angle
    return so3_exp(v)


def synthetic_clips(kin) -> list[dict]:
    """Three clips: the root moves along its heading while it turns, at about 0.78 m with a small sway, and every joint
    swings about the default pose by at most half its distance to the nearer limit (and 0.3 rad)."""
    q0 = torch.tensor(g1.default_joint_pos_list())
    room = torch.minimum(q0 - kin.q_lo.cpu(), kin.q_hi.cpu() - q0).clamp(min=0.0)
    t = torch.arange(T) / FPS
    clips = []
    for name, speed, turn in (("synthetic_walk", 0.6, 0.15), ("synthetic_run", 1.5, -0.3), ("synthetic_turn", 0.0, 0.9)):
        yaw = 0.5 + turn * t
        step = speed / FPS * torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
        xy = torch.cat((torch.zeros(1, 2), torch.cumsum(step[:-1], 0)))
        z = 0.78 + 0.02 * torch.sin(2 * math.pi * 1.5 * t)
        R = rot(2, yaw) @ rot(1, 0.04 * torch.sin(2 * math.pi * 1.5 * t)) @ rot(0, 0.05 * torch.sin(2 * math.pi * 0.75 * t))
        amp = 0.5 * room.clamp(max=0.6) * (0.5 + 0.5 * torch.rand(g1.NUM_J))
        freq, phase = 0.4 + 1.2 * torch.rand(g1.NUM_J), 2 * math.pi * torch.rand(g1.NUM_J)
        joints = q0 + amp * torch.sin(2 * math.pi * freq * t[:, None] + phase)
        clips.append({"name": name, "root_p": torch.cat((xy, z[:, None]), -1), "root_quat": mat_to_quat(R), "joints": joints, "fps": FPS})
    return clips


def main():
    torch.manual_seed(0)
    kin = g1.kinematics("cpu")
    clips = synthetic_clips(kin)
    motion = MotionLib([Clip(**c) for c in clips], kin, "cpu")
    f = motion.clip_start[torch.arange(N) % len(clips)] + 40 + torch.randint(0, 50, (N,))
    p, R, v, w, q, qd = motion.ref_state(f)
    S = (p + 0.05 * torch.randn(N, 3), so3_exp(0.1 * torch.randn(N, 3)) @ R, v + 0.3 * torch.randn(N, 3), w + 0.3 * torch.randn(N, 3),
         q + 0.1 * torch.randn(N, g1.NUM_J), qd + 0.5 * torch.randn(N, g1.NUM_J))  # fmt: skip
    S1 = motion.ref_state(f + 1)
    S_flat, S1_flat = flatten_state(S), flatten_state(S1)
    S, S1 = unflatten_state(S_flat), unflatten_state(S1_flat)  # the states exactly as the test rebuilds them
    u = torch.randn(N, g1.NUM_J)
    xi = torch.cat((0.3 + 0.9 * torch.rand(N, 1), 0.02 * torch.rand(N, g1.NUM_J) - 0.01, 0.1 * torch.rand(N, 3) - 0.05), -1)
    wm = AccelWorldModel(hidden=(64, 64))
    wm.in_norm.update(wm.inputs(S, u, normalize_xi(xi)))
    wm.out_norm.update(wm.targets(S, S1))
    torch.nn.init.normal_(wm.net[-1].weight, std=0.05)  # zero at initialization: the step would not depend on the network
    torch.nn.init.normal_(wm.net[-1].bias, std=0.05)
    with torch.no_grad():
        S_next_flat = flatten_state(wm.step(S, u, normalize_xi(xi)))
        S_next = unflatten_state(S_next_flat)
        ref = ReferenceLocal(motion, kin).feats
        wterms = wm_window_terms(kin, S_next, S1)
        bal = LossBalancer(4)
        bal.calibrate(wterms.mean(0), torch.ones(4))
        golden = {
            "clips": clips, "f": f, "S": S_flat, "S1": S1_flat, "u": u, "xi": xi, "wm_state": wm.state_dict(),
            "wm_inputs": wm.inputs(S, u, normalize_xi(xi)), "wm_targets": wm.targets(S, S1), "S_next": S_next_flat,
            "ref_at_f": tuple(x[f] for x in ref), "local_S": local_features(*kin.forward(*S, all_bodies=True)),
            "track_terms": track_terms(kin, S, ref, f), "wm_window_terms": wterms, "bal_w": bal.w.clone(),
        }  # fmt: skip
    torch.save(golden, OUT)
    print("saved", OUT, {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in golden.items()})


if __name__ == "__main__":
    main()
