"""SuperMimic's --critic_compile (train.py speed option, off by default), on the CPU in eager mode (torch.compile as the identity),
where the option computes the default's values with the same operations: the graphable policy update with --critic_compile is
the default's bit for bit (its survival products are the same loop of products), the eager one up to float rounding in its policy
loss and gradient (the option's loop of products where the default has cumprod); the critic's update is the default's bit for bit
in both, its inputs normalized once per update and cast once to the first product's dtype (the normalizer does not change during
the minibatch steps), as the CPU's bf16 autocast shows. The test at the end needs a GPU: the option compiled, in fp32 and under
CUDA's bf16 autocast, against the default up to rounding."""

import pytest
import torch
import torch.nn as nn

from conftest import DEV, random_states, synthetic_clips
from test_algo_equivalence import EXACT
from test_bptt_tricks import ST_ARGS, _algo
from test_xmimic_learner import _close

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the options compiled for the GPU")


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


@pytest.fixture()
def cpu_autocast(monkeypatch):
    """--amp bf16 on the CPU: the CPU's bf16 autocast in the place of CUDA's in ``run_mlp``, and ``nets.deferred_dtype``'s test of
    where the autocast acts (``nets._autocasts``) then holds for CPU tensors (as test_xmimic_learner's)."""
    from ndm_gym.core import nets

    real = torch.autocast
    monkeypatch.setattr(torch, "autocast", lambda device_type, dtype=None, **kw: real("cpu", dtype=dtype, **kw))
    monkeypatch.setattr(nets, "_autocasts", lambda device: True)
    monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)


def _update(algo, buf, graphable: bool, seed: int = 3):
    """One policy update -> (its outputs: the policy loss, value loss, imagined reward and survival; the policy's gradient norm
    and gradients; the critic's parameters after the update's minibatch steps)."""
    torch.manual_seed(seed)
    res = algo.policy_update_graphable(buf) if graphable else algo.policy_update(buf)
    pi_grads, critic = [p.grad.clone() for p in algo.pi.parameters()], [p.detach().clone() for p in algo.critic.parameters()]
    return [res[0], res[1], res[2], res[4]], [res[3]] + pi_grads, critic


def _compare(base, out, graphable: bool):
    """The graphable updates bit for bit; the eager ones' survival products are the default's cumprod (on the CPU accumulated in
    float64) against the option's loop of fp32 products: the policy loss, imagined reward and survival up to float rounding
    (1e-6), the policy gradient too (1e-5 of each gradient's largest entry), the value loss and the critic bit for bit (the
    critic's update reads the rewards and terminations, not the survival)."""
    for i, (a, b) in enumerate(zip(base[0], out[0])):
        if graphable or i == 1:
            torch.testing.assert_close(b, a, **EXACT)
        else:
            _close(b, a, 1e-6)
    for a, b in zip(base[1], out[1]):
        if graphable:
            torch.testing.assert_close(b, a, **EXACT)
        else:
            _close(b, a, 1e-5)
    for a, b in zip(base[2], out[2]):
        torch.testing.assert_close(b, a, **EXACT)


GRAPHABLE = dict(fused_adam=True, no_sync=True)
PUBLISHED = dict(sm_residual=True, pi_encode_once=True, contact_reward=0.25, ref_contact_fix=True, wm_structured=True)  # two-g1-supermimic's


@pytest.mark.parametrize("graphable", [False, True])
@pytest.mark.parametrize("extra", [{}, PUBLISHED], ids=["default", "published"])
def test_critic_compile_is_supermimics_update(eager, kin, synthetic_motion, synthetic_mdp, graphable, extra):
    """The policy update with --critic_compile against the default: the outputs, the policy gradient and the critic after its
    minibatch steps (the same random stream: the option draws nothing of its own)."""
    kw = {**extra, **(GRAPHABLE if graphable else {})}
    base = _update(*_algo("supermimic", kin, synthetic_motion, synthetic_mdp, **kw), graphable)
    algo, buf = _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, **kw)
    assert algo.critic_compile
    _compare(base, _update(algo, buf, graphable), graphable)


def test_critic_compile_is_supermimics_update_paired(eager, kin, tmp_path):
    """--paired: with the pairs' library (tests/test_paired_tracking.write_pairs) the graphable update with --critic_compile is
    the default's bit for bit, as on one robot."""
    from test_paired_tracking import _pair_motion

    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP

    motion = _pair_motion(kin, tmp_path)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=DEV))
    base = _update(*_algo("supermimic", kin, motion, mdp, **PUBLISHED, **GRAPHABLE), True)
    out = _update(*_algo("supermimic", kin, motion, mdp, critic_compile=True, **PUBLISHED, **GRAPHABLE), True)
    _compare(base, out, True)


def test_critic_compile_under_the_cpus_autocast(cpu_autocast, eager, kin, synthetic_motion, synthetic_mdp):
    """Under the bf16 autocast the minibatch steps' inputs, normalized once and cast once to bf16, are the autocast's casts of
    the rows each step normalizes: the critic after its steps is the default's bit for bit."""
    from ndm_gym.core import nets

    assert nets.deferred_dtype(DEV) == torch.bfloat16
    base = _update(*_algo("supermimic", kin, synthetic_motion, synthetic_mdp, **PUBLISHED, **GRAPHABLE), True)
    out = _update(*_algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, **PUBLISHED, **GRAPHABLE), True)
    _compare(base, out, True)


def test_critic_compile_refusals(eager, kin, synthetic_motion, synthetic_mdp):
    with pytest.raises(ValueError, match="critic_compile"):
        _algo("supermimic", kin, synthetic_motion, synthetic_mdp, critic_compile=True, no_critic=True)
    algo, _ = _algo("supermimic", kin, synthetic_motion, synthetic_mdp)
    assert not algo.critic_compile and not hasattr(algo, "critic_loss_c")


def test_train_parses_the_options(monkeypatch):
    """ndm_gym.train's parser: --critic_compile with SuperMimic only."""
    import sys

    pytest.importorskip("isaaclab")
    from ndm_gym import train

    monkeypatch.setattr(sys, "argv", ["ndm_gym.train", "--algo", "supermimic", "--critic_compile"])
    assert train.parse_args().critic_compile
    monkeypatch.setattr(sys, "argv", ["ndm_gym.train", "--algo", "supertrack", "--critic_compile"])
    with pytest.raises(SystemExit):
        train.parse_args()


# ---------------------------------------------------------------- on a GPU, compiled
def _gpu(synthetic_motion, kw: dict):
    """On the GPU: the synthetic motion library, a replay of noisy reference states and a SuperMimic agent with ``kw`` (the same
    initial weights and normalizers for every call), as test_xmimic_learner's ``_gpu_runs``."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin = g1.kinematics("cuda")
    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    buf, g = ReplayBuffer(48, 64, "cuda", priorities=False), torch.Generator().manual_seed(7)
    for t in range(48):
        f, S, xi = random_states(synthetic_motion, 64, seed=100 + t)
        f = torch.minimum(f + t, synthetic_motion.clip_end[synthetic_motion.clip_id[f]] - 40)
        tr = {"s": flatten_state(S), "s1": flatten_state(synthetic_motion.ref_state(f + 1)), "a": torch.randn(64, 29, generator=g),
              "a_prev": torch.randn(64, 29, generator=g), "xi": xi, "r": torch.rand(64, generator=g), "f": f,
              "ep_step": torch.full((64,), t, dtype=torch.int32), "done": torch.zeros(64, dtype=torch.bool)}  # fmt: skip
        buf.add({k: v.cuda() for k, v in tr.items()})
    w = {k: v.flatten(0, 1) for k, v in buf.data.items()}
    S0, S1 = unflatten_state(w["s"]), unflatten_state(w["s1"])
    args = _args()
    vars(args).update(algo="supermimic", wm_hidden=[64, 64], **ST_ARGS, **kw)
    torch.manual_seed(0)
    algo = ALGOS["supermimic"](args, kin, motion, mdp, "cuda")
    nn.init.normal_(algo.wm.net[-1].weight, std=0.05)  # zero at initialization: no gradient through the network
    algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"], normalize_xi(w["xi"]))), algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    algo.update_pi_norms(mdp.obs(S0, w["a_prev"], w["xi"], w["f"]), motion.goal(w["f"], algo.K))
    return algo, buf


@GPU
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_critic_compile_compiled(monkeypatch, synthetic_motion, amp):
    """--critic_compile compiled on the GPU against the default (its critic input compiled too, --compile_eager), with
    --pi_encode_once and the structured world model as published: the graphable policy update's loss, imagined reward and
    survival (1e-5 of the largest entry; under the bf16 autocast 1e-2, the bf16 rounding of the critic's activations), gradient
    norm and policy gradient (1e-4; 2e-2), and, on the same normalized inputs, the critic's targets and a minibatch step's loss
    and gradients (the same tolerances; on the CPU through Inductor in fp32: 1e-7 and 6e-7). The critic's parameters after its
    20 Adam steps are not compared: Adam's step is about the learning rate for any gradient, so a component whose gradient
    rounds to the other sign moves the other way."""
    from ndm_gym.core import nets
    from ndm_gym.core.normalizer import RunningNorm

    monkeypatch.setattr(RunningNorm, "no_sync", True)
    if amp:
        monkeypatch.setattr(nets, "AMP_DTYPE", torch.bfloat16)
    fwd, grad = (1e-2, 2e-2) if amp else (1e-5, 1e-4)
    kw = dict(compile_eager=True, **PUBLISHED, **GRAPHABLE)
    runs = []
    for on in (False, True):
        algo, buf = _gpu(synthetic_motion, dict(critic_compile=on, **kw))
        runs.append((algo, _update(algo, buf, True)))
    (base_algo, base), (algo, out) = runs
    for a, b in zip(base[0][:1] + base[0][2:], out[0][:1] + out[0][2:]):  # the value loss: the last step's, after 20 Adam steps
        _close(b, a, fwd)
    for a, b in zip(base[1], out[1]):
        _close(b, a, grad)
    torch.manual_seed(5)
    H, B = algo.HZ, 64
    X = torch.randn(H + 1, B, algo.critic.norm.mean.shape[0], device="cuda") * 2 + 1
    rews, dones = torch.rand(H, B, device="cuda"), (torch.rand(H, B, device="cuda") < 0.05).float()
    algo.critic.load_state_dict(base_algo.critic.state_dict())
    with torch.no_grad():
        V = base_algo.critic(X)
        R, want = V[-1], []
        for t in range(H - 1, -1, -1):
            R = rews[t] + algo.args.gamma * (1 - dones[t]) * ((1 - algo.args.lam) * V[t + 1] + algo.args.lam * R)
            want.append(R)
        targets, Xn = algo.critic_targets_c(X, rews, dones)
    _close(targets, torch.stack(want[::-1]), fwd)
    assert Xn.dtype == (torch.bfloat16 if amp else torch.float32) and Xn.shape == (H * B, X.shape[-1])
    idx = torch.randperm(H * B, device="cuda")[: H * B // 4]
    Xf, Y = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)

    def step(critic, fn):
        critic.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in critic.parameters()]

    l0, g0 = step(base_algo.critic, lambda: ((base_algo.critic(Xf[idx]) - Y[idx]) ** 2).mean())
    l1, g1 = step(algo.critic, lambda: algo.critic_loss_c(Xn, Y, idx))
    _close(l1, l0, fwd)
    for a, b in zip(g0, g1):
        _close(b, a, grad)
