"""Writes tests/data/supertrack_update_golden.pt: what two world-model updates and two policy updates of
``ndm_gym.algos.SuperTrack`` return and leave in its loss weights and normalizers, as ``updates`` below runs them on CPU,
in eager mode, from fixed seeds. The imagination noise, the task's action-rate and joint-limit penalties and the failure
penalty (``--st_fail_w``) are on; the replay holds noisy states of the synthetic clips of supertrack_golden.pt, about 30%
of them within a horizon of their clip's end (not valid imagination starts). The file pins the behavior of the code
that wrote it, not a reference (SuperTrack has no public implementation): tests/test_supertrack_golden.py runs
``updates`` again and compares. A deliberate change of SuperTrack's default path rewrites the file with the change.

Run from the repository root: python tests/data/make_supertrack_update_golden.py
"""

import argparse
import os
import sys
from unittest import mock

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import ndm_gym.core.nets as nets  # noqa: E402
from ndm_gym.algos.supertrack import SuperTrack  # noqa: E402
from ndm_gym.buffer import ReplayBuffer  # noqa: E402
from ndm_gym.core.rotations import so3_exp  # noqa: E402
from ndm_gym.motion.library import MotionLib  # noqa: E402
from ndm_gym.motion.sources import Clip  # noqa: E402
from ndm_gym.robots import g1  # noqa: E402
from ndm_gym.state import flatten_state, unflatten_state  # noqa: E402
from ndm_gym.tasks.tracking import MDP  # noqa: E402

CLIPS = os.path.join(REPO, "tests", "data", "supertrack_golden.pt")
OUT = os.path.join(REPO, "tests", "data", "supertrack_update_golden.pt")
T, N = 12, 32  # replay rows x environments


def training_args() -> argparse.Namespace:
    """Every argument SuperTrack reads, at ndm_gym.train's defaults except the world model's width, the batch sizes and
    the failure penalty (on, at weight 3: about 80% of the imagined rollouts cross the task's termination thresholds)."""
    return argparse.Namespace(
        algo="supertrack", st_reg=0.01, st_imag_noise=0.1, st_no_root_terms=False, st_no_task_regs=False, st_substeps=1, st_balance_w=1.0,
        st_com=False, st_quad_w=0.0, st_late_w=0.0, st_fail_w=3.0, st_fail_margin=1.0, st_wm_loss="l1", imag_start_perturb=0.0,
        bptt_clip_k=0.0, wm_bptt_fp32=False, imag_fail_starts=0.0, imag_fail_window=50, horizon=32, goal_frames=10, hist=10, wm_window=8,
        wm_hidden=[64, 64], wm_act="silu", wm_no_ln=False, wm_input_noise=0.0, wm_smooth_w=0.0, wm_smooth_eps=0.1, wm_smooth_dirs=1,
        wm_backbone="mlp", wm_sample="uniform", wm_prio_alpha=0.6, wm_fail_windows=0.0, wm_fail_window=50, wm_batch=32, pi_batch=64,
        lr=2e-4, fused_adam=False, wm_cudagraphs=False, pi_cudagraphs=False,
    )  # fmt: skip


def replay(motion: MotionLib, horizon: int, seed: int) -> ReplayBuffer:
    """T steps of N environments: noisy reference states, their next reference states, random setpoints and contexts;
    about 30% of the states lie within ``horizon`` frames of their clip's end."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    buf = ReplayBuffer(T, N, "cpu", priorities=False)  # as train.py builds it for uniform window sampling
    for t in range(T):
        f = motion.clip_start[torch.arange(N) % len(motion.names)] + 40 + t + torch.randint(0, 50, (N,), generator=g)
        end = motion.clip_end[motion.clip_id[f]]
        f = torch.where(torch.rand(N, generator=g) < 0.3, end - torch.randint(1, horizon, (N,), generator=g), f)
        p, R, v, w, q, qd = motion.ref_state(f)
        S = (p + 0.05 * rn(N, 3), so3_exp(0.1 * rn(N, 3)) @ R, v + 0.3 * rn(N, 3), w + 0.3 * rn(N, 3), q + 0.1 * rn(N, 29), qd + 0.5 * rn(N, 29))
        xi = torch.cat((0.3 + 0.9 * torch.rand(N, 1, generator=g), 0.02 * torch.rand(N, 29, generator=g) - 0.01,
                        0.1 * torch.rand(N, 3, generator=g) - 0.05), -1)  # fmt: skip
        buf.add({"s": flatten_state(S), "s1": flatten_state(motion.ref_state(f + 1)), "a": rn(N, 29), "a_prev": rn(N, 29), "xi": xi,
                 "r": torch.rand(N, generator=g), "f": f, "ep_step": torch.full((N,), t, dtype=torch.int32),
                 "done": torch.rand(N, generator=g) < 0.05})  # fmt: skip
    return buf


def updates(clips: list[dict], tweak=None) -> dict:
    """SuperTrack on the synthetic clips: the normalizers fit on the replay through Algo.update_wm_norms and
    update_pi_norms (as train.py's collection fits them), the world model's last layer drawn at random (zero at
    initialization, the policy's gradient would not pass through the network), then wm_update twice (the second with the
    probe) and policy_update twice (the first calibrates the loss weights), each after torch.manual_seed. ``tweak(algo)``
    runs before the policy updates (the generator's check that the penalties matter)."""
    amp = nets.AMP_DTYPE
    nets.AMP_DTYPE = None
    try:
        torch.manual_seed(0)
        kin = g1.kinematics("cpu")
        motion = MotionLib([Clip(**c) for c in clips], kin, "cpu")
        mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list()))
        args = training_args()
        with mock.patch.object(torch, "compile", lambda fn, **kw: fn):  # eager
            algo = SuperTrack(args, kin, motion, mdp, "cpu")
        torch.nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
        buf = replay(motion, args.horizon, seed=7)
        w = buf.data
        S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
        algo.update_wm_norms(S0, S1, w["a"].flatten(0, 1), w["xi"].flatten(0, 1))
        o = mdp.add_noise(mdp.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten()))
        algo.update_pi_norms(o, motion.goal(w["f"].flatten(), args.goal_frames))
        out = {"wm_updates": [], "pi_updates": []}
        for seed, probe in ((1, False), (2, True)):
            torch.manual_seed(seed)
            out["wm_updates"].append(tuple(x.detach().clone() for x in algo.wm_update(buf, probe=probe)))
        if tweak is not None:
            tweak(algo)
        for seed in (3, 4):
            torch.manual_seed(seed)
            out["pi_updates"].append(tuple(x.detach().clone() for x in algo.policy_update(buf)))
        out["bal_wm"], out["bal_pi"] = algo.bal_wm.state_dict(), algo.bal_pi.state_dict()
        out["norms"] = {f"{m}.{k}": v.clone() for m in ("wm", "pi") for k, v in getattr(algo, m).state_dict().items() if "norm." in k}
        return out
    finally:
        nets.AMP_DTYPE = amp


def main():
    clips = torch.load(CLIPS)["clips"]
    golden = updates(clips)
    torch.save(golden, OUT)
    print("saved", OUT)
    # the imagination noise and every penalty matter: without one, the policy losses move far beyond the test's tolerance
    tweaks = {
        "no imagination noise": lambda a: setattr(a.args, "st_imag_noise", 0.0),
        "no joint-limit penalty": lambda a: setattr(a, "w_limit", 0.0),
        "no action-rate penalty": lambda a: setattr(a, "w_rate", 0.0),
        "no failure penalty": lambda a: setattr(a.args, "st_fail_w", 0.0),
    }
    for name, tweak in tweaks.items():
        other = updates(clips, tweak)
        rel = [float((b[0] - a[0]).abs() / a[0].abs()) for a, b in zip(golden["pi_updates"], other["pi_updates"])]
        print(f"{name}: policy losses change by {rel[0]:.2e}, {rel[1]:.2e} (relative)")
    print("imagined survival:", [float(u[4]) for u in golden["pi_updates"]])


if __name__ == "__main__":
    main()
