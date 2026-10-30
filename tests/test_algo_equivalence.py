"""One world-model update and one policy+critic update of ``ndm_gym.algos.SuperMimic`` equal the upstream training
functions (third_party/supermimic/supermimic/train.py) bit for bit: CPU, eager mode, same weights, same replay
buffer, same RNG state."""

import argparse
import math
import os
import time

import pytest
import torch
import torch.nn as nn

from conftest import DEV, REPO, UPSTREAM, random_states, with_ndm_fk

EXACT = dict(rtol=0, atol=0)
T, N = 48, 64  # replay steps x environments


def _args():
    return argparse.Namespace(wm_window=8, wm_batch=32, pi_batch=64, horizon=32, goal_frames=10, hist=10, gamma=0.99, lam=0.95,
                              no_critic=False, critic_epochs=5, critic_minibatches=4, lr=2e-4, lr_decay_iters=250_000, lr_decay_hours=None,
                              imag_start_perturb=0.0, wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, wm_input_noise=0.0,
                              wm_sample="uniform", wm_prio_alpha=0.6, wm_backbone="mlp", imag_fail_starts=0.0,
                              imag_fail_window=50, wm_fail_windows=0.0, wm_fail_window=50, sm_residual=False, bptt_clip_k=0.0,
                              wm_bptt_fp32=False)  # fmt: skip


def _fill(buffer_cls, motion, seed):
    """Deterministic replay data: noisy reference states, their next reference states, random setpoints."""
    from ndm_gym.state import flatten_state

    buf = buffer_cls(T, N, DEV)
    g = torch.Generator().manual_seed(seed)
    for t in range(T):
        f, S, xi = random_states(motion, N, seed=1000 + t)
        f = torch.minimum(f + t, motion.clip_end[motion.clip_id[f]] - 40)
        buf.add({"s": flatten_state(S), "s1": flatten_state(motion.ref_state(f + 1)), "a": torch.randn(N, 29, generator=g),
                 "a_prev": torch.randn(N, 29, generator=g), "xi": xi, "r": torch.rand(N, generator=g), "f": f,
                 "ep_step": torch.full((N,), t, dtype=torch.int32), "done": torch.rand(N, generator=g) < 0.05})  # fmt: skip
    return buf


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


@pytest.mark.slow
def test_supermimic_updates_match_upstream(eager, upstream, kin, motion, mdp):
    from ndm_gym.algos.supermimic import SuperMimic
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.state import normalize_xi, unflatten_state

    args = _args()
    # ---- upstream: rebuild train.py's globals and exec its update functions
    vk = with_ndm_fk(upstream["kinematics"].G1Kinematics(DEV))  # same FK on both sides; FK itself: test_upstream_equivalence
    vm = upstream["motion"].MotionLib.from_dir(os.path.join(REPO, "data", "lafan1_g1"), vk, DEV)
    vmdp = upstream["mdp"].MDP(vk, vm, torch.tensor(upstream["robot"].default_joint_pos_list(), device=DEV))
    n, md = upstream["networks"], upstream["mdp"]
    torch.manual_seed(0)
    wm, pi = n.WorldModel(), n.Policy(md.OBS_DIM, 73 * 10, hist=10)
    critic = n.Critic(md.INV_DIM + md.CRITIC_EXTRA + 73 * 10)
    ns = dict(args=args, wm=wm, pi=pi, critic=critic, opt_wm=torch.optim.Adam(wm.parameters(), lr=args.lr),
              opt_pi=torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95)), opt_v=torch.optim.Adam(critic.parameters(), lr=args.lr),
              mdp=vmdp, motion=vm, K=10, Hh=10, HZ=32, dev=DEV, unflatten_state=md.unflatten_state, normalize_xi=md.normalize_xi,
              torch=torch, nn=nn, math=math, time=time, os=os, buffer=_fill(upstream["buffer"].ReplayBuffer, vm, seed=7))  # fmt: skip
    src = open(os.path.join(UPSTREAM, "supermimic", "train.py")).read()
    exec(src[src.index("# per-component floors"): src.index("# ------------------------------------------------------------------------------------------------\nsim_cfg")], ns)
    # ---- refactored
    torch.manual_seed(0)
    algo = SuperMimic(args, kin, motion, mdp, DEV)
    buf = _fill(ReplayBuffer, motion, seed=7)
    for m0, m1 in ((wm, algo.wm), (pi, algo.pi), (critic, algo.critic)):
        for (k0, p0), (k1, p1) in zip(m0.state_dict().items(), m1.state_dict().items()):
            assert k0 == k1
            torch.testing.assert_close(p1, p0, **EXACT)
    # normalizers from the replay data, as collect() fits them
    w = buf.data
    S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
    x_in = algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1)))
    for m in (wm, algo.wm):
        m.in_norm.update(x_in), m.out_norm.update(m.targets(S0, S1))
    for m in (pi, algo.pi):
        m.obs_norm.update(mdp.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten())), m.goal_norm.update(motion.goal(w["f"].flatten(), 10))

    torch.manual_seed(1)
    old_wm = ns["wm_update"]()
    torch.manual_seed(1)
    new_wm = algo.wm_update(buf)
    for a, b in zip(new_wm, old_wm):
        torch.testing.assert_close(a, b, **EXACT)
    torch.manual_seed(2)
    old_pi = ns["policy_update"]()
    torch.manual_seed(2)
    new_pi = algo.policy_update(buf)
    for a, b in zip(new_pi, old_pi):
        torch.testing.assert_close(a.cpu(), b.cpu(), **EXACT)
    for m0, m1 in ((wm, algo.wm), (pi, algo.pi), (critic, algo.critic)):
        for p0, p1 in zip(m0.parameters(), m1.parameters()):
            torch.testing.assert_close(p1, p0, **EXACT)
