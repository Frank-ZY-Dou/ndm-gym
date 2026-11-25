"""RigidFormerNet's input features as one differentiable kernel (``--rf_fused_inputs``, worldmodels/g1_fused.py:
``rf_features`` and its hand-written backward pass) against ``RigidFormerNet.inputs`` and its autograd on the GPU
(skipped without CUDA), in fp32: the features, the gradients of the state and of the setpoints over batch sizes (one,
odd, not a multiple of a program's samples), with merged tokens and with leading batch dimensions; the clipped torque's
gradient on both sides of the limit; both world models' rollouts with the option against the default path, eager and
compiled; a frozen world model's compiled step with the fused inputs and the fused read-out as one graph (their tables
built when the network is moved). The flag's mapping and refusals, and the CPU path (the default's bits), are checked
without a GPU.
"""

import pytest
import torch

DEV = "cuda"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused features need a CUDA GPU")


def _net(**kw):
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet

    return RigidFormerNet(kinematics("cpu"), 6, 1, **kw)


def _states(n, g, dev=DEV, lead=()):
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.robots.g1 import NUM_J

    r = lambda *s, scale=1.0: scale * torch.randn(*lead, n, *s, generator=g).to(dev)  # noqa: E731
    S = (r(3), so3_exp(r(3)), r(3), r(3), r(NUM_J, scale=0.5), r(NUM_J, scale=2.0))
    return S, r(NUM_J), r(1 + NUM_J + 3)


def _grads(net, S, a, xi, G):
    ins = [t.clone().requires_grad_(True) for t in (*S, a)]
    x = net.inputs(tuple(ins[:6]), ins[6], xi)
    return x.detach(), torch.autograd.grad((x * G).sum(), ins, allow_unused=True)


def _close(got, ref, what, rel=2e-4):
    ref = torch.zeros_like(got) if ref is None else ref
    err, scale = float((got - ref).abs().max()), max(float(ref.abs().max()), 1.0)
    assert err <= rel * scale, f"{what}: max error {err:.3e} at scale {scale:.3e}"


@GPU
@pytest.mark.parametrize("n", [1, 7, 64, 333, 2048])
@pytest.mark.parametrize("merge", ["none", "joints"])
def test_features_and_gradients_match_autograd(n, merge):
    """The kernel's features against the compiled kinematics', and every input's gradient of a random linear functional
    against autograd's; merged tokens change the network, not its inputs."""
    g = torch.Generator().manual_seed(n)
    ref, fused = _net(merge=merge).to(DEV), _net(merge=merge, fused_inputs=True).to(DEV)
    S, a, xi = _states(n, g)
    G = torch.randn(n, ref.in_dim, generator=g).to(DEV)
    (x0, g0), (x1, g1) = _grads(ref, S, a, xi, G), _grads(fused, S, a, xi, G)
    _close(x1, x0, f"features, n {n}", rel=1e-5)
    for name, u, v in zip(("p", "R", "v", "w", "q", "qd", "a"), g1, g0):
        assert u is not None, name
        _close(u, v, f"gradient of {name}, n {n}")
    assert not g1[0][:, :2].any()  # the features do not depend on the horizontal position


@GPU
def test_leading_dimensions():
    """States of shape (T, N, ...), as the world-model fit's windows: the same features and gradients."""
    g = torch.Generator().manual_seed(5)
    ref, fused = _net().to(DEV), _net(fused_inputs=True).to(DEV)
    S, a, xi = _states(33, g, lead=(4,))
    G = torch.randn(4, 33, ref.in_dim, generator=g).to(DEV)
    (x0, g0), (x1, g1) = _grads(ref, S, a, xi, G), _grads(fused, S, a, xi, G)
    assert x1.shape == x0.shape
    _close(x1, x0, "features", rel=1e-5)
    for name, u, v in zip(("p", "R", "v", "w", "q", "qd", "a"), g1, g0):
        _close(u, v, f"gradient of {name}")


@GPU
def test_clipped_torque():
    """The PD torque feature is clipped at the effort limit: its gradient passes inside the limits only, as autograd's."""
    g = torch.Generator().manual_seed(9)
    ref, fused = _net().to(DEV), _net(fused_inputs=True).to(DEV)
    S, a, xi = _states(512, g)
    a = 4.0 * a  # setpoints far enough for many clipped torques
    G = torch.zeros(512, ref.in_dim, device=DEV)
    from ndm_gym.worldmodels.rigidformer import TOKEN_DIM

    G[:, [TOKEN_DIM * b + 19 for b in range(1, ref.n_body)]] = 1.0
    (x0, g0), (x1, g1) = _grads(ref, S, a, xi, G), _grads(fused, S, a, xi, G)
    _close(x1, x0, "features", rel=1e-5)
    tau = x0[:, [TOKEN_DIM * b + 19 for b in range(1, ref.n_body)]]
    assert 0.2 < float((tau.abs() == 1).float().mean()) < 0.98  # both clipped and unclipped torques
    for name, u, v in zip(("p", "R", "v", "w", "q", "qd", "a"), g1, g0):
        _close(u, v, f"gradient of {name}")


@GPU
@pytest.mark.parametrize("which", ["accel", "increment"])
@pytest.mark.parametrize("compiled", [False, True])
def test_world_model_rollout(motion, which, compiled):
    """A 4-step open-loop rollout of each world model with the option against the default path in fp32: the states and
    the gradients of a linear functional of them with respect to the start state and the setpoints."""
    import ndm_gym.core.nets as nets
    from conftest import random_states
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    n, steps = 200, 4
    _, S0, xi = random_states(motion, n, seed=11)
    g = torch.Generator().manual_seed(12)
    u, dy = 0.5 * torch.randn(steps, n, 29, generator=g).to(DEV), torch.randn(steps, n, 76, generator=g).to(DEV)
    S0, xi_n = tuple(t.to(DEV) for t in S0), normalize_xi(xi).to(DEV)
    cls = AccelWorldModel if which == "accel" else IncrementWorldModel
    wms = []
    for fused in (False, True):
        torch.manual_seed(0)
        wms.append(cls(rigidformer=dict(kin=kinematics("cpu"), fused_inputs=fused)).to(DEV))
    ref, fus = wms
    with torch.no_grad():  # the heads start at zero: give the rollout a signal
        for m in (ref.net.root_head, ref.net.joint_head):
            m.weight.normal_(0.0, 0.02)
    ref.in_norm.update(ref.inputs(S0, u[0], xi_n))
    fus.load_state_dict(ref.state_dict())
    for wm in wms:
        wm.requires_grad_(False)

    def run(wm):
        S = tuple(t.clone().requires_grad_(True) for t in S0)
        us = u.clone().requires_grad_(True)
        step = torch.compile(wm.step) if compiled else wm.step
        out, St = [], S
        for k in range(steps):
            St = step(St, us[k], xi_n)
            out.append(torch.cat((St[0], St[1].flatten(-2), *St[2:]), -1))
        y = torch.stack(out)
        (y * dy).sum().backward()
        return [y.detach()] + [t.grad for t in S + (us,)]

    amp = nets.AMP_DTYPE
    try:
        nets.AMP_DTYPE = None  # fp32: the features' rounding, not the network's bf16
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        a, b = run(ref), run(fus)
    finally:
        nets.AMP_DTYPE = amp
        torch.backends.cuda.matmul.allow_tf32 = tf32
    for i, (x, y) in enumerate(zip(b, a)):
        _close(x, y, f"{which} compiled {compiled}: output {i}", rel=2e-3)


@GPU
@pytest.mark.parametrize("which", ["accel", "increment"])
def test_compiled_step_with_the_fused_kernels_is_one_graph(which, motion):
    """The tables of the fused inputs and of the fused read-out are built when the network is moved to a GPU (none on the
    CPU, where neither kernel runs), so a frozen world model's step under bf16 autocast (the policy's BPTT) traces as one
    graph: torch.compile(fullgraph=True) refuses a break, which tables built at the step's first pass caused (they read
    buffers on the host). Without its tables the network refuses the pass."""
    import ndm_gym.core.nets as nets
    from conftest import random_states
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.state import normalize_xi
    from ndm_gym.worldmodels.accel import AccelWorldModel
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    n = 64
    _, S0, xi = random_states(motion, n, seed=13)
    u = 0.5 * torch.randn(n, 29, generator=torch.Generator().manual_seed(14)).to(DEV)
    S0, xi_n = tuple(t.to(DEV) for t in S0), normalize_xi(xi).to(DEV)
    cls = AccelWorldModel if which == "accel" else IncrementWorldModel
    wm = cls(rigidformer=dict(kin=kinematics("cpu"), fused_train=True, fused_readout=True, fused_inputs=True, checkpoint="frozen_readout"))
    assert wm.net._head_tables is None and wm.net._fk is None
    with torch.no_grad():  # the heads start at zero: give the step a signal
        for m in (wm.net.root_head, wm.net.joint_head):
            m.weight.normal_(0.0, 0.02)
    wm = wm.to(DEV).requires_grad_(False)
    assert wm.net._head_tables["tob"].device.type == "cuda" and all(t.device.type == "cuda" for t in wm.net._fk)
    amp = nets.AMP_DTYPE
    try:
        nets.AMP_DTYPE = torch.bfloat16
        torch._dynamo.reset()
        step = torch.compile(wm.step, fullgraph=True)
        us = u.clone().requires_grad_(True)
        S2 = step(step(S0, us, xi_n), us, xi_n)
        sum(t.float().sum() for t in S2).backward()
        assert bool(torch.isfinite(us.grad).all()) and float(us.grad.abs().max()) > 0
        tables, wm.net._head_tables = wm.net._head_tables, None
        with pytest.raises(RuntimeError, match="fused_readout"):
            wm.step(S0, u, xi_n)
        wm.net._head_tables, wm.net._fk = tables, None
        with pytest.raises(RuntimeError, match="fused_inputs"):
            wm.step(S0, u, xi_n)
    finally:
        nets.AMP_DTYPE = amp
    assert wm.cpu().net._head_tables is None and wm.net._fk is None


def test_flag_mapping_refusals_and_cpu_path():
    """--rf_fused_inputs reaches RigidFormerNet; it is refused with an external wrench or anchor points; on the CPU a
    network with it computes its inputs as without (the same bits)."""
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.rigidformer import RigidFormerNet, rigidformer_kwargs

    a = dict(wm_backbone="rigidformer", wm_hidden=[2048, 2048, 2048], wm_act="silu", wm_no_ln=False, rf_dim=128, rf_layers=3, rf_heads=4,
             rf_registers=2, rf_no_tree_bias=False, rf_no_rope=False, amp="bf16")  # fmt: skip
    assert rigidformer_kwargs(a)["fused_inputs"] is False and rigidformer_kwargs({**a, "rf_fused_inputs": True})["fused_inputs"] is True
    for bad in (dict(ext_wrench=True), dict(anchors=4)):
        with pytest.raises(ValueError, match="fused_inputs"):
            RigidFormerNet(kinematics("cpu"), 6, 1, fused_inputs=True, **bad)
    g = torch.Generator().manual_seed(1)
    S, act, xi = _states(9, g, dev="cpu")
    assert torch.equal(_net(fused_inputs=True).inputs(S, act, xi), _net().inputs(S, act, xi))
