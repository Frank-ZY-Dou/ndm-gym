"""``--sm_residual``: SuperMimic with SuperTrack's action parameterization (offsets about the reference pose of the
next frame), for ablations; the default path stays the upstream one."""

import pytest
import torch

from conftest import DEV, random_states
from test_algo_equivalence import _args

EXACT = dict(rtol=0, atol=0)


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def test_residual_flag_is_read_from_the_run_arguments():
    from ndm_gym.algos import ALGOS

    assert ALGOS["supertrack"].is_residual({}) is True
    assert ALGOS["supermimic"].is_residual({}) is False  # checkpoints from before the flag
    assert ALGOS["supermimic"].is_residual({"sm_residual": True}) is True


def test_supermimic_residual_setpoints(eager, kin, motion, mdp):
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.algos.supermimic import SuperMimic
    from ndm_gym.robots.g1 import G1_JOINT_NAMES, action_scale
    from ndm_gym.state import normalize_xi
    from ndm_gym.tasks.tracking import OBS_DIM

    args = _args()
    args.sm_residual = True
    torch.manual_seed(0)
    algo = SuperMimic(args, kin, motion, mdp, DEV)
    assert algo.residual
    alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=DEV)
    f, S, xi = random_states(motion, 16, seed=3)
    ref = (motion.q[f + 1] - mdp.default_q) / alpha  # the reference pose of the target frame as a setpoint
    torch.testing.assert_close(reference_setpoints(motion, mdp.default_q, alpha)[f + 1], ref, **EXACT)
    a = torch.randn(16, 29)
    torch.testing.assert_close(algo.setpoint(a, f), a + ref, **EXACT)

    # imagination executes the same setpoint: the world model, the next observation and the action-rate term see it
    o_hist, a_prev, g = torch.randn(16, args.hist, OBS_DIM), torch.randn(16, 29), motion.goal(f, args.goal_frames)
    xi_n = normalize_xi(xi)
    S1, u, _, f1, _ = algo._imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
    torch.testing.assert_close(u, algo.pi(o_hist, g) + ref, **EXACT)
    for x, y in zip(S1, algo.wm.step(S, u, xi_n)):
        torch.testing.assert_close(x, y, **EXACT)
    assert torch.equal(f1, f + 1)

    args.sm_residual = False
    plain = SuperMimic(args, kin, motion, mdp, DEV)
    assert not plain.residual
    assert plain.setpoint(a, f) is a
