"""Flag-gated tricks against policy-gradient spikes in the BPTT through the world model: ``--bptt_clip_k`` (cap each
rollout's state gradient after every imagined step), ``--rf_rope_detach`` (no gradient through RigidFormer's rotary
positions) and ``--wm_bptt_fp32`` (no bf16 autocast in the world model during the BPTT). At their defaults the
algorithms are unchanged (test_algo_equivalence, test_supertrack_golden)."""

import pytest
import torch
import torch.nn as nn

from conftest import DEV
from test_algo_equivalence import EXACT, _args, _fill

ST_ARGS = dict(st_reg=0.01, st_imag_noise=0.1, st_no_root_terms=False, st_no_task_regs=False, st_substeps=1, st_balance_w=1.0, st_com=False,
               st_quad_w=0.0, st_late_w=0.0, st_fail_w=0.0, st_fail_margin=1.0, st_wm_loss="l1")  # fmt: skip


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _algo(name, kin, motion, mdp, **kw):
    """A small-MLP algorithm whose world model's last layer is random (zero at initialization, the policy gradient
    would not pass through the network), with normalizers fit on the replay data as collect() fits them."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.state import normalize_xi, unflatten_state

    args = _args()
    vars(args).update(algo=name, wm_hidden=[64, 64], **ST_ARGS, **kw)
    torch.manual_seed(0)
    algo = ALGOS[name](args, kin, motion, mdp, DEV)
    nn.init.normal_(algo.wm.net[-1].weight, std=0.05)
    buf = _fill(ReplayBuffer, motion, seed=7)
    w = buf.data
    S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
    algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1))))
    algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    algo.pi.obs_norm.update(mdp.obs(S0, w["a_prev"].flatten(0, 1), w["xi"].flatten(0, 1), w["f"].flatten()))
    algo.pi.goal_norm.update(motion.goal(w["f"].flatten(), args.goal_frames))
    return algo, buf


def test_cap_rollout_grads():
    from ndm_gym.algos.common import cap_rollout_grads

    torch.manual_seed(0)
    x, R = torch.randn(64, 3, requires_grad=True), torch.randn(64, 3, 3, requires_grad=True)
    y, Ry = cap_rollout_grads((x, R), 3.0)
    assert torch.equal(y, x) and torch.equal(Ry, R)
    g, gR = torch.randn(64, 3), torch.randn(64, 3, 3)
    g[5] *= 100.0  # an exploding rollout
    g[7] = 0.0  # a rollout without gradient: not counted in the median
    torch.autograd.backward((y, Ry), (g, gR))
    n = g.norm(dim=1)
    cap = 3.0 * n[n > 0].median()
    keep = n <= cap
    assert not keep[5] and keep[7]
    torch.testing.assert_close(x.grad[keep], g[keep], **EXACT)  # every rollout below the cap is untouched
    torch.testing.assert_close(x.grad[~keep].norm(dim=1), cap.expand(int((~keep).sum())))  # the others are scaled to it
    torch.testing.assert_close(x.grad[5] / x.grad[5].norm(), g[5] / n[5])  # in their own direction
    nR = gR.flatten(1).norm(dim=1)
    torch.testing.assert_close(R.grad[nR <= 3.0 * nR.median()], gR[nR <= 3.0 * nR.median()], **EXACT)  # per component, per row
    z = torch.randn(8, 3, requires_grad=True)
    (zy,) = cap_rollout_grads((z,), 2.0)
    zy.backward(torch.zeros(8, 3))
    torch.testing.assert_close(z.grad, torch.zeros(8, 3), **EXACT)  # no gradient at all: nothing to scale


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_bptt_clip_k_in_policy_update(eager, kin, motion, mdp, name):
    """A cap no rollout reaches changes nothing, bit for bit; a low cap changes the policy gradient, not the loss."""
    out, params = {}, {}
    for k in (0.0, 1e9, 0.5):
        algo, buf = _algo(name, kin, motion, mdp, bptt_clip_k=k)
        torch.manual_seed(2)
        out[k] = algo.policy_update(buf)
        params[k] = [p.detach().clone() for p in algo.pi.parameters()]
    for a, b in zip(out[0.0], out[1e9]):
        torch.testing.assert_close(a, b, equal_nan=True, **EXACT)
    for a, b in zip(params[0.0], params[1e9]):
        torch.testing.assert_close(a, b, **EXACT)
    torch.testing.assert_close(out[0.5][0], out[0.0][0], **EXACT)  # the loss: same forward pass
    assert out[0.5][3] < out[0.0][3]  # the policy-gradient norm before clipping


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_wm_bptt_fp32_switches_autocast_only_in_the_policy_update(eager, kin, motion, mdp, monkeypatch, name):
    import ndm_gym.worldmodels.accel as accel
    import ndm_gym.worldmodels.increment as increment

    for flag in (False, True):
        algo, buf = _algo(name, kin, motion, mdp, wm_bptt_fp32=flag)
        seen = []
        mod = accel if name == "supertrack" else increment
        run = mod.run_mlp

        def spy(net, x, amp=True):
            seen.append(amp)
            return run(net, x, amp=amp)

        monkeypatch.setattr(mod, "run_mlp", spy)
        torch.manual_seed(1)
        algo.wm_update(buf)
        assert seen and all(seen)  # world-model fitting keeps its precision
        seen.clear()
        torch.manual_seed(2)
        algo.policy_update(buf)
        assert len(seen) == algo.HZ and all(a == (not flag) for a in seen)
        assert algo.wm.amp is True  # restored for everything else
        monkeypatch.setattr(mod, "run_mlp", run)


def test_rope_detach_cuts_only_the_rotary_path_in_the_frozen_backward(kin):
    from ndm_gym.algos.common import set_requires_grad
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    torch.manual_seed(0)
    nets = [RigidFormerNet(kin, root_out=6, joint_out=1, checkpoint="never", rope_detach=d, dim=24, layers=2, heads=2, registers=2)
            for d in (False, True)]  # fmt: skip
    for m in (nets[0].root_head, nets[0].joint_head):  # zero at initialization: the output would not depend on the trunk
        nn.init.normal_(m.weight, std=0.1)
    nets[1].load_state_dict(nets[0].state_dict())
    D = nets[0].in_dim
    x0 = torch.randn(4, D + 3 * nets[0].n_body)
    for frozen in (False, True):
        grads, ys = [], []
        for net in nets:
            set_requires_grad(net, not frozen)
            x = x0.clone().requires_grad_(True)
            y = net(x)
            y.square().sum().backward()
            ys.append(y.detach()), grads.append(x.grad)
        torch.testing.assert_close(ys[1], ys[0], **EXACT)  # the forward pass is unchanged
        if not frozen:  # world-model training: unchanged
            torch.testing.assert_close(grads[1], grads[0], **EXACT)
        else:  # the policy's BPTT: the positions reach the output through the token features only
            assert grads[0][:, D:].abs().max() > 0
            torch.testing.assert_close(grads[1][:, D:], torch.zeros_like(grads[1][:, D:]), **EXACT)
            torch.testing.assert_close(grads[1][:, :D], grads[0][:, :D])
    with pytest.raises(ValueError):
        RigidFormerNet(kin, root_out=6, joint_out=1, rope=False, rope_detach=True, **dict(dim=24, layers=1, heads=2))


def test_rope_detach_training_argument():
    from ndm_gym.worldmodels.rigidformer import rigidformer_kwargs

    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=96, rf_layers=2, rf_heads=3,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    assert rigidformer_kwargs(a)["rope_detach"] is False  # checkpoints from before the flag
    assert rigidformer_kwargs({**a, "rf_rope_detach": True})["rope_detach"] is True
    with pytest.raises(ValueError):
        rigidformer_kwargs({**a, "wm_backbone": "mlp", "rf_rope_detach": True})
