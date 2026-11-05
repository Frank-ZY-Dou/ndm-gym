"""--wm_opt muon with the RigidFormer world model (LAFAN1 training; core/muon.py, RigidFormerNet.hidden_matrices): Muon on
every block's weight matrices as stored (the packed q/k/v/gate projection, the attention's output projection, the SwiGLU's
packed value and gate projection and its output projection), Adam on every other parameter, each parameter in exactly one
of the two, with one token per body and with merged tokens; through the algorithms one world-model update trains both
parts, Muon's matrices at Adam's update RMS (match_rms_adamw: --wm_lr applies unchanged). The MLP's split is unchanged and
MuonAdam's explicit matrices give its steps bit for bit; --wm_opt adam is one Adam over all the world model's parameters;
ndm_gym.train takes the option with the RigidFormer, as muscle training does (tests/test_muscle_wm_muon_rf.py). On the GPU (skipped without CUDA): the world-model update with --update_graphs (capturable
Muon and Adam) and --rf_fused_train, captured as a CUDA graph and replayed, gives the eager update's values and
parameters bit for bit."""

import importlib.util

import pytest
import torch
import torch.nn as nn

from conftest import synthetic_clips
from test_algo_equivalence import EXACT, _args, _fill
from test_bptt_tricks import ST_ARGS, _algo

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
needs_isaaclab = pytest.mark.skipif(not HAVE_ISAACLAB, reason="the trainers' arguments come with Isaac Lab's launcher")
# a small RigidFormer within the fused kernels' shapes (32 tokens, or 16 merged, with the registers; head width 32)
RF = dict(wm_hidden=[2048, 2048, 2048], wm_backbone="rigidformer", rf_dim=64, rf_layers=2, rf_heads=2, rf_registers=2, rf_no_tree_bias=False,
          rf_no_rope=False)  # fmt: skip
MATRICES = ("qkvg", "proj", "w12", "w3")  # a block's weight matrices, Muon's
BLOCK_ADAM = ("norm1.weight", "qkvg.bias", "q_norm.weight", "k_norm.weight", "proj.bias", "film.weight", "film.bias", "norm2.weight", "w12.bias",
              "w3.bias")  # fmt: skip
NET_ADAM = ("ctx_net.0.weight", "ctx_net.0.bias", "embed.weight", "embed.bias", "body_emb", "registers", "rel_bias", "head_norm.weight",
            "head_w12.weight", "head_w12.bias", "head_w3.weight", "head_w3.bias", "root_head.weight", "root_head.bias", "joint_head.weight",
            "joint_head.bias")  # fmt: skip


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _split(opt) -> tuple[list[int], list[int]]:
    """The ids of MuonAdam's Muon parameters and Adam parameters, in their optimizers' order."""
    return [id(p) for g in opt.muon.param_groups for p in g["params"]], [id(p) for g in opt.adam.param_groups for p in g["params"]]


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


# ---------------------------------------------------------------------------------------------------- the parameter split
@pytest.mark.parametrize("merge", ["none", "joints"])
def test_rigidformer_hidden_matrices(merge):
    """Muon takes the blocks' four weight matrices each (q/k/v/gate packed as stored: 4 dim x dim; the SwiGLU's value and
    gate: 2 hidden x dim), Adam everything else (embeddings, context network, registers, tree bias, norms, biases, FiLM, the
    read-out's fusion and heads; with merged tokens also the read-out's body embedding); each parameter in exactly one."""
    from ndm_gym.core.muon import MuonAdam
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.accel import AccelWorldModel

    torch.manual_seed(0)
    wm = AccelWorldModel(rigidformer=dict(kin=kinematics("cpu"), dim=64, layers=2, heads=2, registers=2, merge=merge))
    net, hid = wm.net, int(2.5 * 64)
    want = [blk.get_parameter(f"{m}.weight") for blk in net.blocks for m in MATRICES]
    assert [id(p) for p in net.hidden_matrices()] == [id(p) for p in want]
    assert [tuple(p.shape) for p in want] == [(4 * 64, 64), (64, 64), (2 * hid, 64), (64, hid)] * 2
    opt = MuonAdam(wm, None, lr=1e-3, betas=(0.9, 0.999), matrices=net.hidden_matrices())
    muon, adam = _split(opt)
    names = {id(p): n for n, p in wm.named_parameters()}
    assert {names[i] for i in muon} == {f"net.blocks.{b}.{m}.weight" for b in range(2) for m in MATRICES} and len(muon) == 8
    top = NET_ADAM + (("unmerge_emb",) if merge != "none" else ())
    assert {names[i] for i in adam} == {f"net.{n}" for n in top} | {f"net.blocks.{b}.{n}" for b in range(2) for n in BLOCK_ADAM}
    assert len(set(muon)) == len(muon) and len(set(adam)) == len(adam) and not set(muon) & set(adam)
    assert sorted(muon + adam) == sorted(id(p) for p in wm.parameters())
    assert opt.muon.defaults["adjust_lr_fn"] == "match_rms_adamw" and opt.muon.defaults["weight_decay"] == 0
    assert opt.adam.defaults["betas"] == (0.9, 0.999) and not opt.capturable


def test_explicit_matrices_are_the_mlp_split():
    """MuonAdam with the MLPs' hidden matrices given explicitly steps as with the MLPs, bit for bit; it takes the MLPs or
    the matrices, matrices of the model, at least one."""
    from ndm_gym.core.muon import MuonAdam, hidden_matrices

    torch.manual_seed(0)
    nets = [nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 12), nn.SiLU(), nn.Linear(12, 3))
            for _ in range(2)]  # fmt: skip
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [MuonAdam(nets[0], [nets[0]], lr=1e-2, betas=(0.9, 0.999)),
            MuonAdam(nets[1], None, lr=1e-2, betas=(0.9, 0.999), matrices=hidden_matrices([nets[1]]))]  # fmt: skip
    assert [len(x) for x in _split(opts[1])] == [2, 6]
    x = torch.randn(32, 6)
    for _ in range(3):
        for n, o in zip(nets, opts):
            o.zero_grad()
            n(x).square().mean().backward()
            o.step()
    for p, q in zip(nets[0].parameters(), nets[1].parameters()):
        torch.testing.assert_close(q, p, **EXACT)
    m = nets[0]
    for bad in (dict(mlps=[m], matrices=hidden_matrices([m])), dict(mlps=None), dict(mlps=None, matrices=[nn.Parameter(torch.zeros(3, 3))]),
                dict(mlps=None, matrices=[]), dict(mlps=None, matrices=[m[0].bias])):  # fmt: skip
        with pytest.raises(ValueError):
            MuonAdam(m, lr=1e-2, **bad)


# ---------------------------------------------------------------------------------------------------- through the algorithms
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
@pytest.mark.parametrize("merge", ["none", "joints"])
def test_world_model_update_with_muon(eager, kin, synthetic_motion, synthetic_mdp, name, merge):
    """--wm_opt muon with --wm_backbone rigidformer: the algorithm's world-model optimizer is MuonAdam on the blocks'
    matrices; one world-model update moves every parameter, each Muon matrix by about 0.2 x the learning rate in RMS (the
    step rescaled to Adam's update RMS; 0.16-0.19 measured), each Adam parameter's entries by the learning rate (Adam's
    first step; the read-out's embedding of the bodies that share a token only, the other rows have no gradient)."""
    from ndm_gym.core.muon import MuonAdam

    algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, wm_opt="muon", rf_merge=merge, **RF)
    with torch.no_grad():  # FiLM starts at zero, which holds the context network's gradient at zero
        for blk in algo.wm.net.blocks:
            blk.film.weight.normal_(0.0, 0.05)
    opt = algo.opt_wm
    assert isinstance(opt, MuonAdam) and not opt.capturable and opt.adam.defaults["betas"] == (0.9, 0.999)
    muon, adam = _split(opt)
    assert muon == [id(p) for p in algo.wm.net.hidden_matrices()] and sorted(muon + adam) == sorted(id(p) for p in algo.wm.parameters())
    lr = algo.args.lr
    assert [g["lr"] for g in opt.param_groups] == [lr, lr]
    before = {id(p): p.detach().clone() for p in algo.wm.parameters()}
    torch.manual_seed(1)
    loss, _ = algo.wm_update(buf)
    assert bool(torch.isfinite(loss))
    for n, p in algo.wm.named_parameters():
        d = p.detach() - before[id(p)]
        if id(p) in muon:
            assert 0.1 * lr < _rms(d) < 0.3 * lr, (n, _rms(d) / lr)
            continue
        if n == "net.unmerge_emb":
            d = d[algo.wm.net.shared.squeeze(-1) > 0]
        assert float(d.abs().max()) <= 1.01 * lr and _rms(d) > 0.99 * lr, (n, _rms(d) / lr)  # 1.01: the difference's rounding


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_mlp_world_model_split_unchanged(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--wm_opt muon with the MLP world model: Muon on its Linear layers but the first and last, as before."""
    algo, _ = _algo(name, kin, synthetic_motion, synthetic_mdp, wm_opt="muon")
    linears = [m for m in algo.wm.net if isinstance(m, nn.Linear)]
    muon, adam = _split(algo.opt_wm)
    assert muon == [id(lin.weight) for lin in linears[1:-1]] and len(muon) == 1
    assert adam == [id(p) for p in algo.wm.parameters() if id(p) not in muon]


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_rigidformer_adam_unchanged(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--wm_opt adam (the default) with the RigidFormer: one Adam over all the world model's parameters in their order, with
    the default betas; the option given explicitly gives the same update bit for bit."""
    out = {}
    for explicit in (False, True):
        algo, buf = _algo(name, kin, synthetic_motion, synthetic_mdp, **RF, **({"wm_opt": "adam"} if explicit else {}))
        opt = algo.opt_wm
        assert type(opt) is torch.optim.Adam and len(opt.param_groups) == 1 and opt.defaults["betas"] == (0.9, 0.999)
        assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in algo.wm.parameters()]
        torch.manual_seed(1)
        out[explicit] = list(algo.wm_update(buf)) + [p.detach().clone() for p in algo.wm.parameters()]
    for a, b in zip(out[False], out[True]):
        torch.testing.assert_close(b, a, equal_nan=True, **EXACT)  # SuperTrack's e8 is nan


# ---------------------------------------------------------------------------------------------------- the trainers' checks
@needs_isaaclab
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_lafan1_training_takes_muon_with_the_rigidformer(name):
    """ndm_gym.train's checks accept --wm_opt muon with the RigidFormer, with the fused blocks and the systems options."""
    from ndm_gym.algos.blinkmimic import check_args
    from test_blinkmimic import _parser

    argv = ["--algo", name, "--wm_backbone", "rigidformer", "--wm_opt", "muon", "--rf_fused_train", "--rf_bptt_keep_blocks", "--rf_fit_keep_ops",
            "--fused_adam", "--no_sync", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked", "--rf_merge", "joints"]  # fmt: skip
    a = _parser().parse_args(argv)
    check_args(a)
    assert (a.wm_opt, a.wm_backbone, a.rf_fused_train, a.update_graphs) == ("muon", "rigidformer", True, True)


# ---------------------------------------------------------------------------------------------------- GPU: graphs, fused blocks
def _cuda_algo(name, synthetic_motion, **kw):
    """``_algo`` on the GPU: the algorithm with the small RigidFormer (heads at random), the replay data of ``_fill`` in a
    buffer with device-side sampling bounds (--update_graphs --graph_early), normalizers fit on it."""
    from ndm_gym.algos import ALGOS
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP

    kin = g1.kinematics("cuda")
    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    args = _args()
    vars(args).update({"algo": name, **ST_ARGS, **RF, **kw})
    torch.manual_seed(0)
    algo = ALGOS[name](args, kin, motion, mdp, "cuda")
    for m in (algo.wm.net.root_head, algo.wm.net.joint_head):
        nn.init.normal_(m.weight, std=0.05)
    host = _fill(ReplayBuffer, synthetic_motion, seed=7)
    buf = ReplayBuffer(host.T, host.N, "cuda", priorities=False)
    for k in host.fields:
        buf.data[k].copy_(host.data[k])
    buf.ptr, buf.size = host.ptr, host.size
    buf.use_device_ptr(device_size=True)
    w = buf.data
    S0, S1 = unflatten_state(w["s"].flatten(0, 1)), unflatten_state(w["s1"].flatten(0, 1))
    algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"].flatten(0, 1), normalize_xi(w["xi"].flatten(0, 1))))
    algo.wm.out_norm.update(algo.wm.targets(S0, S1))
    return algo, buf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="--update_graphs and --rf_fused_train need a GPU")
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_graphed_fused_update_with_muon(eager, monkeypatch, synthetic_motion, name):
    """--wm_opt muon --update_graphs --rf_fused_train --rf_fit_keep_ops (--rf_merge joints): MuonAdam is capturable (device
    learning rates, which set_lr fills for both parts); the world-model update (wm_update_graphable, the blocks' passes by
    the fused kernels under bf16 autocast) run by Graphed, three eager calls on a side stream, then captured and replayed,
    gives the eager update's losses and parameters bit for bit at every call, and trains every parameter."""
    import ndm_gym.core.nets as nets
    import ndm_gym.kernels.rf_train as rf_train
    from ndm_gym.algos.blinkmimic import Graphed
    from ndm_gym.algos.common import set_lr
    from ndm_gym.core.muon import MuonAdam

    fused, block_train = [], rf_train.rf_block_train
    monkeypatch.setattr(rf_train, "rf_block_train", lambda *a, **k: fused.append(1) or block_train(*a, **k))
    kw = dict(wm_opt="muon", update_graphs=True, no_sync=True, fused_adam=True, rf_fused_train=True, rf_fit_keep_ops=True, rf_merge="joints")
    amp = nets.AMP_DTYPE
    nets.AMP_DTYPE = torch.bfloat16  # train.py --amp bf16: the passes the fused kernels run
    try:
        runs = {}
        for graphed in (False, True):
            algo, buf = _cuda_algo(name, synthetic_motion, **kw)
            opt = algo.opt_wm
            assert isinstance(opt, MuonAdam) and opt.capturable and all(torch.is_tensor(g["lr"]) for g in opt.param_groups)
            set_lr(opt, 3e-4)
            assert all(float(g["lr"]) == pytest.approx(3e-4) for g in opt.param_groups)
            fn = lambda algo=algo, buf=buf: algo.wm_update_graphable(buf)  # noqa: E731
            step = Graphed(fn, lambda: True, "the world-model update") if graphed else fn
            p0, n_fused = [p.detach().clone() for p in algo.wm.parameters()], len(fused)
            out = []
            for i in range(6):  # Graphed: three eager calls, the capture and its replay, two replays
                torch.manual_seed(100 + i)
                loss, e8 = step()
                out.append([loss.clone(), e8.clone()] + [p.detach().clone() for p in algo.wm.parameters()])
            assert len(fused) > n_fused  # the blocks' passes ran by the fused kernels
            assert all(not torch.equal(a, b) for a, b in zip(p0, algo.wm.parameters()))
            finite = [i for i, t in enumerate(out[-1]) if bool(torch.isfinite(t).all())]
            assert finite == [0] + ([1] if name == "supermimic" else []) + list(range(2, len(out[-1]))), finite  # SuperTrack's e8 is nan
            if graphed:
                assert step.graph is not None
            runs[graphed] = out
    finally:
        nets.AMP_DTYPE = amp
    for i, (a, b) in enumerate(zip(runs[False], runs[True])):
        for x, y in zip(a, b):
            torch.testing.assert_close(y, x, **EXACT, equal_nan=True, msg=lambda m, i=i: f"call {i}: {m}")
