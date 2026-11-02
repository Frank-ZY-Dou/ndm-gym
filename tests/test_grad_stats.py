"""tools/grad_stats.py --anatomy: an update's policy gradient taken apart by imagined step and by rollout equals brute
force (each step's and each rollout's own vector-Jacobian product through the policy) and sums to the update's
gradient, and the taps leave the update unchanged (CPU, eager, a small-MLP world model, the synthetic clips)."""

import importlib.util
import os

import pytest
import torch

from conftest import REPO
from test_algo_equivalence import EXACT
from test_bptt_tricks import _algo


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _tool():
    spec = importlib.util.spec_from_file_location("grad_stats", os.path.join(REPO, "tools", "grad_stats.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_gradient(algo, buf, seed):
    """One policy update at learning rate 0 and its gradient before the update clips it (as grad_stats captures it)."""
    from ndm_gym.algos.common import set_lr

    for o in algo.optimizers:
        set_lr(o, 0.0)
    params = list(algo.pi.parameters())
    raw = [None] * len(params)
    hooks = [p.register_post_accumulate_grad_hook(lambda p, i=i: raw.__setitem__(i, p.grad.detach().clone())) for i, p in enumerate(params)]
    torch.manual_seed(seed)
    out = algo.policy_update(buf)
    for h in hooks:
        h.remove()
    return torch.cat([g.flatten() for g in raw]), out


def _vjp(pi, o, g, adj):
    from ndm_gym.policies.tracking_policy import TrackingPolicy

    with torch.enable_grad():
        grads = torch.autograd.grad(TrackingPolicy.forward(pi, o, g), list(pi.parameters()), adj)
    return torch.cat([x.flatten() for x in grads])


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_anatomy_parts_match_brute_force(eager, kin, synthetic_motion, synthetic_mdp, name):
    tool = _tool()
    base, buf = _algo(name, kin, synthetic_motion, synthetic_mdp)
    g0, out0 = _raw_gradient(base, buf, seed=3)
    algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp)
    critic0 = {k: v.clone() for k, v in algo.critic.state_dict().items()} if name == "supermimic" else None
    anatomy = tool.Anatomy(algo, chunk=16)
    anatomy.start()
    g, out = _raw_gradient(algo, buf, seed=3)
    torch.testing.assert_close(g, g0, **EXACT)  # the taps only read gradients
    for a, b in zip(out, out0):
        torch.testing.assert_close(a, b, equal_nan=True, **EXACT)
    ref = torch.randn(g.shape, generator=torch.Generator().manual_seed(4))
    ref = ref / ref.norm()
    Ob, G, A = (torch.stack([r[k] for r in anatomy.steps]) for k in ("o", "g", "adj"))
    H, B = A.shape[:2]
    assert (H, B) == (algo.HZ, algo.args.pi_batch) and len(anatomy.states) == H
    P = anatomy.parts(g, ref)
    tol = dict(rtol=1e-4, atol=1e-6 * float(g.norm()) ** 2)
    torch.testing.assert_close(P["step_sum"], g, rtol=1e-4, atol=1e-5 * float(g.abs().max()))
    torch.testing.assert_close(P["rollout_dot_g"].sum(), g @ g, **tol)
    torch.testing.assert_close(P["step_gram"].diagonal(), P["step_sq"], rtol=1e-4, atol=0.0)
    torch.testing.assert_close(P["step_gram"].sum(), g @ g, **tol)  # |sum_t g_t|^2
    for t in (0, 13, H - 1):
        gt = _vjp(algo.pi, Ob[t], G[t], A[t])
        torch.testing.assert_close(P["step_gram"][t, 5], gt @ _vjp(algo.pi, Ob[5], G[5], A[5]), **tol)
        torch.testing.assert_close(P["step_sq"][t], gt @ gt, **tol)
        torch.testing.assert_close(P["step_dot_g"][t], gt @ g, **tol)
        torch.testing.assert_close(P["step_dot_ref"][t], gt @ ref, rtol=1e-4, atol=1e-6 * float(g.norm()))
    for b in (0, 17, B - 1):
        gb = _vjp(algo.pi, Ob[:, b], G[:, b], A[:, b])
        torch.testing.assert_close(P["rollout_sq"][b], gb @ gb, **tol)
        torch.testing.assert_close(P["rollout_dot_g"][b], gb @ g, **tol)
        torch.testing.assert_close(P["rollout_dot_ref"][b], gb @ ref, rtol=1e-4, atol=1e-6 * float(g.norm()))
    torch.testing.assert_close(P["adj"], A.norm(dim=-1), **EXACT)
    # SuperMimic's rollouts past their soft termination (survival weight 0) get no gradient at all
    assert P["state_adj"].shape == (H, B) and bool((P["state_adj"] >= 0).all()) and float((P["state_adj"] > 0).float().mean()) > 0.5
    # weighted sums of the rollouts' gradients, and the chunks' own sums
    w = torch.rand(B, generator=torch.Generator().manual_seed(5))
    agg, groups = anatomy.rg.weighted({"one": torch.ones(B), "w": w})
    close = dict(rtol=1e-4, atol=1e-5 * float(g.abs().max()))
    torch.testing.assert_close(agg["one"], g, **close)
    torch.testing.assert_close(agg["w"], _vjp(algo.pi, Ob.flatten(0, 1), G.flatten(0, 1), (A * w[None, :, None]).flatten(0, 1)), **close)
    assert len(groups) == B // 16
    torch.testing.assert_close(sum(groups), g, **close)
    s = anatomy.take_apart(g, ref)
    assert s["check_steps"] < 1e-4 and s["check_rollouts"] < 1e-4
    # --fp32_twin's comparison of every aggregation (on CPU without autocast both passes compute the same)
    if critic0 is not None:
        algo.critic.load_state_dict(critic0)  # the update refit the critic's normalizer, as grad_stats restores it
    algo.args.wm_bptt_fp32 = True
    anatomy.start()
    g32, _ = _raw_gradient(algo, buf, seed=3)
    assert all(abs(c - 1) < 1e-5 for c in anatomy.twin(g32, ref).values())
    assert abs(sum(s["step_share"]) - 1) < 1e-3 and len(s["step_norm"]) == H and len(s["top"]) == 3
    assert abs(s["variants"]["rerun"][2] - 1) < 1e-6 and set(s["variants"]) == {"rerun", "rnorm", "trim1", "rclip2", "rclip4", "rclip8",
                                                                                 "rclip4_proxy", "mom4"}  # fmt: skip


def test_geometric_median():
    tool = _tool()
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [100.0, 100.0]])
    z = tool.geometric_median(X)
    assert bool(((z > 0.5) & (z < 1)).all())  # the outlier moves the mean to 20.4, the geometric median within the square
    torch.testing.assert_close(tool.geometric_median(X[:4]), torch.tensor([0.5, 0.5]))  # symmetric points: their center
    y = torch.randn(7, 5, generator=torch.Generator().manual_seed(0))
    dist = lambda p: float((y - p).norm(dim=1).sum())  # noqa: E731
    zy = tool.geometric_median(y)
    assert all(dist(zy) <= dist(zy + 1e-3 * d) + 1e-9 for d in torch.eye(5).unbind(0) + (-torch.eye(5)).unbind(0))  # a minimum
