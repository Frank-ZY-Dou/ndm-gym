"""--wm_opt muon with the muscle character's RigidFormer world model (ndm_gym.train_muscles; algos.common.wm_optimizer,
MuscleRigidFormerNet.hidden_matrices), on the CPU with MuscleVAE's converted muscle table. train_muscles takes the option
with either token set, the fused options and the systems options, its agents' world-model optimizer is then MuonAdam, and
the combinations it refused before are still refused. Muon takes exactly the blocks' weight matrices, as LAFAN1 training's
split assigns them (the same rule on the same blocks, the same optimizer settings): the packed q/k/v/gate projection, the
attention's output projection, the SwiGLU's packed value and gate projection and its output projection; Adam every other
parameter (the token embeddings, the context network, the per-token and register embeddings, the relation bias, with
--rf_muscle_tokens body the muscle encoder and its maps, the blocks' norms, biases and FiLM, the read-out's fusion and
heads), each parameter in exactly one of the two. One world-model update moves every Muon matrix by about 0.2 x the
learning rate in RMS (match_rms_adamw) and every Adam entry with a gradient by the learning rate; the update written
without host synchronization (--no_sync, the body --update_graphs captures) gives the default update bit for bit and
skips a non-finite one on the device. A checkpoint written and read back as train_muscles does restores the optimizer's
state (with --update_graphs, into the device learning rates the graphs read), and the resumed agent continues as the
uninterrupted one bit for bit; a run with another --wm_opt is refused. On a GPU (marked slow): a short train_muscles run
with the fused RigidFormer, Muon and the systems options captures and replays its updates, skips none and saves MuonAdam's
state."""

import csv
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from test_algo_equivalence import EXACT
from test_muscle_training import HAVE_DATA, HAVE_LAUNCHER, HAVE_SIM, REPO, muscle_replay, small_muscle_agent, synthetic_reference
from test_muscle_worldmodels import needs_table

from ndm_gym.algos.common import set_lr
from ndm_gym.core.muon import MuonAdam

# a small muscle RigidFormer with Muon (train_muscles' --wm_backbone rigidformer, --rf_*, --wm_opt muon)
RF = dict(wm_backbone="rigidformer", muscle_model="released", rf_dim=32, rf_layers=2, rf_heads=2, rf_registers=2, rf_no_tree_bias=False,
          rf_no_rope=False, wm_opt="muon")  # fmt: skip
MATRICES = ("qkvg", "proj", "w12", "w3")  # a block's weight matrices, Muon's
BLOCK_ADAM = ("norm1.weight", "qkvg.bias", "q_norm.weight", "k_norm.weight", "proj.bias", "film.weight", "film.bias", "norm2.weight", "w12.bias",
              "w3.bias")  # fmt: skip
READOUT = ("head_norm.weight", "head_w12.weight", "head_w12.bias", "head_w3.weight", "head_w3.bias", "body_head.weight", "body_head.bias",
           "fatigue_head.weight", "fatigue_head.bias")  # fmt: skip
NET_ADAM = {"all": ("ctx_net.0.weight", "ctx_net.0.bias", "embed_body.weight", "embed_body.bias", "embed_muscle.weight", "embed_muscle.bias",
                    "tok_emb", "registers", "rel_bias") + READOUT,
            "body": ("ctx_net.0.weight", "ctx_net.0.bias", "embed_body.weight", "embed_body.bias", "muscle_in.weight", "muscle_in.bias", "muscle_emb",
                     "muscle_out.weight", "muscle_out.bias", "part_in.weight", "part_in.bias", "tok_emb", "registers", "rel_bias") + READOUT}  # fmt: skip
SYSTEMS = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked", "--ckpt_light"]
# a RigidFormer with one token per body and the fused kernels, with Muon
FUSED_RF = ["--wm_backbone", "rigidformer", "--rf_muscle_tokens", "body", "--rf_dim", "64", "--rf_layers", "2", "--rf_heads", "2", "--rf_registers",
            "9", "--rf_fused_train", "--rf_fused_features", "--wm_opt", "muon"]  # fmt: skip


def rf_agent(algo: str, tokens: str, ref, seed: int = 0, **opts):
    """A CPU learner (small_muscle_agent) with the small muscle RigidFormer and Muon, FiLM and the read-out's heads at
    random (both start at zero, which holds the gradients of the context network, and of everything before the heads, at
    zero); ``opts``: further training arguments."""
    torch.manual_seed(seed)
    agent = small_muscle_agent(algo, ref, rf_muscle_tokens=tokens, **({"rf_muscle_dim": 16} if tokens == "body" else {}), **RF, **opts)
    net = agent.wm.net
    with torch.no_grad():
        for m in [blk.film for blk in net.blocks] + [net.body_head, net.fatigue_head]:
            m.weight.normal_(0.0, 0.05)
    return agent


def split(wm, opt: MuonAdam) -> tuple[list[str], list[str]]:
    """The names of MuonAdam's Muon parameters and Adam parameters of ``wm``, in their optimizers' order."""
    names = {id(p): n for n, p in wm.named_parameters()}
    return [names[id(p)] for g in opt.muon.param_groups for p in g["params"]], [names[id(p)] for g in opt.adam.param_groups for p in g["params"]]


def opt_tensors(opt: MuonAdam) -> list[torch.Tensor]:
    """Every tensor of MuonAdam's state (Muon's momenta, Adam's step counts and moments), copied."""
    return [v.clone() for o in (opt.muon, opt.adam) for st in o.state.values() for v in st.values() if torch.is_tensor(v)]


def wm_step(agent, buf, seed: int):
    """One world-model update (--no_sync: the graphable one, which must step) after seeding the window draw; its loss."""
    torch.manual_seed(seed)
    if not agent.no_sync:
        return agent.wm_update(buf)
    loss, ok = agent.wm_update_graphable(buf)
    assert bool(ok)
    return loss


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


# ---------------------------------------------------------------------------------------------------- the trainer's checks
@needs_table
@pytest.mark.skipif(not HAVE_LAUNCHER, reason="ndm_gym.train's arguments come with Isaac Lab's launcher")
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_train_muscles_takes_muon_with_the_rigidformer(algo, capsys):
    """train_muscles accepts --wm_opt muon with --wm_backbone rigidformer, with either --rf_muscle_tokens, with the fused
    blocks and features and with the systems options, and the agent its arguments build steps the world model with
    MuonAdam on the blocks' matrices (capturable with --update_graphs); the MLP's Muon is unchanged; the combinations
    refused before are refused with their messages."""
    from ndm_gym.algos.muscle_algos import MUSCLE_ALGOS
    from ndm_gym.train_muscles import parse_args

    ref = synthetic_reference()
    base = ["--algo", algo, "--wm_backbone", "rigidformer", "--wm_opt", "muon"]
    for extra in ([], ["--rf_muscle_tokens", "all"], ["--rf_muscle_tokens", "body", "--rf_muscle_dim", "16"],
                  FUSED_RF + SYSTEMS + ["--pi_opt", "muon", "--pi_ema", "0.995", "--wm_lr", "1e-3"]):  # fmt: skip
        a = parse_args(base + extra)
        assert (a.wm_opt, a.wm_backbone, a.rf_muscle_tokens) == ("muon", "rigidformer", "body" if "body" in extra else "all")
        agent = MUSCLE_ALGOS[algo](a, ref, "cpu")
        opt = agent.opt_wm
        assert isinstance(opt, MuonAdam) and opt.capturable == a.update_graphs == ("--update_graphs" in extra)
        assert [id(p) for g in opt.muon.param_groups for p in g["params"]] == [id(p) for p in agent.wm.net.hidden_matrices()]
    a = parse_args(["--algo", algo, "--wm_opt", "muon"])
    assert a.wm_backbone == "mlp" and isinstance(MUSCLE_ALGOS[algo](a, ref, "cpu").opt_wm, MuonAdam)
    capsys.readouterr()
    for bad, why in ((base + ["--wm_hidden", "512"], "--wm_hidden configures the other world-model backbone than --wm_backbone rigidformer"),
                     (["--algo", algo, "--wm_opt", "muon", "--rf_dim", "64"], "--rf_dim configures the other world-model backbone"),
                     (base + ["--rf_fused_train"], "--rf_fused_train runs the body tokens' blocks"),
                     (base + ["--rf_merge", "joints"], "--rf_merge is an option of LAFAN1 training"),
                     (base + ["--no_sync"], "--no_sync skips a non-finite update on the device, with the fused Adam: needs --fused_adam"),
                     (base + ["--update_graphs", "--fused_adam"], "--update_graphs captures the updates written without host synchronization")):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(bad)
        assert why in capsys.readouterr().err, bad


# ---------------------------------------------------------------------------------------------------- the parameter split
@needs_table
@pytest.mark.parametrize("tokens", ["all", "body"])
def test_muscle_rigidformer_split_is_lafan1s(tokens):
    """The muscle RigidFormer's split is LAFAN1's: LAFAN1 training's optimizer (algos.common.wm_optimizer) on the G1
    RigidFormer of the same size puts on Muon exactly its blocks' four weight matrices and every other parameter on Adam;
    the same rule, by a parameter's name within its block, gives the muscle model's split (with either token set), the same
    matrices' shapes and the same optimizer settings. Explicitly: Muon the blocks' matrices in order (the model's
    hidden_matrices), Adam the listed parameters, each parameter in exactly one of the two."""
    from ndm_gym.algos.common import wm_optimizer
    from ndm_gym.robots.g1 import kinematics
    from ndm_gym.worldmodels.accel import AccelWorldModel

    torch.manual_seed(0)
    g1 = AccelWorldModel(rigidformer=dict(kin=kinematics("cpu"), dim=32, layers=2, heads=2, registers=2))
    g1_opt = wm_optimizer(g1, SimpleNamespace(wm_opt="muon", lr=1e-3), {"lr": 1e-3}, False, {})
    g1_muon, g1_adam = split(g1, g1_opt)
    in_block = lambda n: n.split(".", 3)[3] if n.startswith("net.blocks.") else None  # noqa: E731  the name within its block
    assert {in_block(n) for n in g1_muon} == {f"{m}.weight" for m in MATRICES} and {in_block(n) for n in g1_adam} == set(BLOCK_ADAM) | {None}
    agent = rf_agent("supertrack", tokens, synthetic_reference())
    wm, opt = agent.wm, agent.opt_wm
    muon, adam = split(wm, opt)
    for n, _ in wm.named_parameters():  # LAFAN1's rule
        assert (n in muon) == (in_block(n) in {in_block(m) for m in g1_muon}) != (n in adam), n
    shapes = [(128, 32), (32, 32), (160, 32), (32, 80)] * 2  # per block (4 dim, dim), (dim, dim), (2 hidden, dim), (dim, hidden)
    assert [tuple(wm.get_parameter(n).shape) for n in muon] == [tuple(g1.get_parameter(n).shape) for n in g1_muon] == shapes
    assert opt.muon.defaults == g1_opt.muon.defaults and opt.adam.defaults == g1_opt.adam.defaults and not opt.capturable
    assert opt.muon.defaults["adjust_lr_fn"] == "match_rms_adamw" and opt.muon.defaults["weight_decay"] == 0
    assert opt.adam.defaults["betas"] == (0.9, 0.999)
    # explicitly
    assert muon == [f"net.blocks.{b}.{m}.weight" for b in range(2) for m in MATRICES]
    assert [id(p) for g in opt.muon.param_groups for p in g["params"]] == [id(p) for p in wm.net.hidden_matrices()]
    assert set(adam) == {f"net.{n}" for n in NET_ADAM[tokens]} | {f"net.blocks.{b}.{n}" for b in range(2) for n in BLOCK_ADAM}
    assert len(set(muon)) == len(muon) and len(set(adam)) == len(adam) and not set(muon) & set(adam)
    assert sorted(muon + adam) == sorted(n for n, _ in wm.named_parameters())


# ---------------------------------------------------------------------------------------------------- the updates
@needs_table
@pytest.mark.parametrize("tokens", ["all", "body"])
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_muscle_world_model_update_with_muon(algo, tokens):
    """One world-model update at the learning rate set_lr gives both parts (train_muscles' schedule and --wm_lr) moves
    every parameter, each with a gradient: each Muon matrix by about 0.2 x the learning rate in RMS (the step rescaled to
    Adam's update RMS; 0.17-0.19 measured), each Adam parameter by Adam's first step of its (clipped) gradient g,
    -lr g / (|g| + eps), the learning rate per entry but where g is near eps (relation classes of few token pairs) or zero
    (classes that do not occur). Two more updates train on; Muon keeps a momentum per matrix, Adam its moments and step
    count per parameter."""
    ref = synthetic_reference()
    agent = rf_agent(algo, tokens, ref)
    net, opt = agent.wm.net, agent.opt_wm
    buf = muscle_replay(ref)
    agent.update_wm_norms({k: v.flatten(0, 1) for k, v in buf.data.items()})
    lr, eps = 2.0**-10, opt.adam.defaults["eps"]
    set_lr(opt, lr)
    assert [g["lr"] for g in opt.param_groups] == [lr, lr]
    muon = {id(p) for p in net.hidden_matrices()}
    before = {n: p.detach().clone() for n, p in agent.wm.named_parameters()}
    loss = wm_step(agent, buf, 1)
    assert loss is not None and bool(torch.isfinite(loss))
    for n, p in agent.wm.named_parameters():
        d, g = p.detach() - before[n], p.grad  # the gradient the step took (clipped in place)
        assert float(g.abs().max()) > 0, n
        if id(p) in muon:
            assert 0.1 * lr < _rms(d) < 0.3 * lr, (n, _rms(d) / lr)
        else:  # atol: the difference's rounding
            torch.testing.assert_close(d, -lr * g / (g.abs() + eps), rtol=0, atol=1e-3 * lr, msg=lambda m, n=n: f"{n}: {m}")
            assert float(d.abs().max()) > 0.99 * lr, n
    for seed in (2, 3):
        assert bool(torch.isfinite(wm_step(agent, buf, seed)))
    assert all(set(opt.muon.state[p]) == {"momentum_buffer"} for p in net.hidden_matrices())
    assert all(int(opt.adam.state[p]["step"]) == 3 for g in opt.adam.param_groups for p in g["params"])


@needs_table
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_graphable_muscle_update_with_muon(algo):
    """--no_sync (with --fused_adam; the update --update_graphs captures): wm_update_graphable, MuonAdam's step gated on
    the device, gives the default update's losses, parameters and optimizer state bit for bit; a non-finite update is
    counted and leaves the parameters, Muon's momenta and Adam's moments and step counts unchanged."""
    ref = synthetic_reference()
    agents = [rf_agent(algo, "body", ref, fused_adam=True, no_sync=no_sync) for no_sync in (False, True)]
    agents[1].wm.load_state_dict(agents[0].wm.state_dict())
    bufs = (muscle_replay(ref), muscle_replay(ref))
    for agent, buf in zip(agents, bufs):
        agent.update_wm_norms({k: v.flatten(0, 1) for k, v in buf.data.items()})
    for seed in (10, 11, 12):
        torch.testing.assert_close(wm_step(agents[1], bufs[1], seed), wm_step(agents[0], bufs[0], seed), **EXACT)
    for x, y in zip(agents[0].wm.parameters(), agents[1].wm.parameters()):
        torch.testing.assert_close(y, x, **EXACT)
    s0, s1 = opt_tensors(agents[0].opt_wm), opt_tensors(agents[1].opt_wm)
    n_muon, n_adam = (sum(len(g["params"]) for g in o.param_groups) for o in (agents[0].opt_wm.muon, agents[0].opt_wm.adam))
    assert len(s0) == len(s1) == n_muon + 3 * n_adam  # Muon's momenta; Adam's step counts and moments
    for x, y in zip(s0, s1):
        torch.testing.assert_close(y, x, **EXACT)
    agent, buf = agents[1], bufs[1]
    snap = lambda: [p.detach().clone() for p in agent.wm.parameters()] + opt_tensors(agent.opt_wm)  # noqa: E731
    s0 = snap()
    buf.data["s1"][:] = float("nan")
    assert not bool(agent.wm_update_graphable(buf)[1]) and agent.wm_skips.take() == (1, 1)
    assert all(torch.equal(x, y) for x, y in zip(s0, snap()))


# ---------------------------------------------------------------------------------------------------- checkpoints
@needs_table
@pytest.mark.parametrize("graphs", [False, True])
def test_checkpoint_restores_muon(tmp_path, graphs):
    """A checkpoint of an agent with the RigidFormer and Muon, written and read back as train_muscles writes last.pt and
    --resume reads it, loaded into an agent of another initialization: the networks and both parts of the optimizer's
    state are the saved ones (with --update_graphs in the device learning-rate tensors built with the optimizer, which the
    graphs read), and the resumed agent's next updates are the uninterrupted agent's bit for bit; without the optimizer's
    state they are not. check_resume refuses a run with another --wm_opt."""
    from ndm_gym.algos.blinkmimic import RESUME_MATCH, check_resume

    ref = synthetic_reference()
    kw = dict(fused_adam=True, no_sync=True, update_graphs=True) if graphs else {}
    agent = rf_agent("supertrack", "body", ref, **kw)
    buf = muscle_replay(ref)
    agent.update_wm_norms({k: v.flatten(0, 1) for k, v in buf.data.items()})
    set_lr(agent.opt_wm, 2.0**-11)
    for seed in (10, 11):
        wm_step(agent, buf, seed)
    path = tmp_path / "last.pt"
    torch.save({**agent.state_dict(), "iteration": 2, "args": vars(agent.args)}, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    resumed = rf_agent("supertrack", "body", ref, seed=1, **kw)
    lrs = [g["lr"] for g in resumed.opt_wm.param_groups]
    assert all(torch.is_tensor(x) == graphs for x in lrs)
    assert not all(torch.equal(p, q) for p, q in zip(resumed.wm.parameters(), agent.wm.parameters()))
    resumed.load_state_dict(ck)
    assert all(torch.equal(q, p) for p, q in zip(agent.wm.state_dict().values(), resumed.wm.state_dict().values()))
    if graphs:
        assert all(g["lr"] is x for g, x in zip(resumed.opt_wm.param_groups, lrs))
    assert [float(g["lr"]) for g in resumed.opt_wm.param_groups] == [2.0**-11] * 2
    sd0, sd1 = agent.opt_wm.state_dict(), resumed.opt_wm.state_dict()
    for part in ("muon", "adam"):
        assert len(sd0[part]["param_groups"]) == len(sd1[part]["param_groups"]) == 1
        for g0, g1 in zip(sd0[part]["param_groups"], sd1[part]["param_groups"]):
            assert g0.keys() == g1.keys() and all(torch.equal(*map(torch.as_tensor, (g0[k], g1[k]))) if k == "lr" else g0[k] == g1[k] for k in g0)
        assert sd0[part]["state"].keys() == sd1[part]["state"].keys() and len(sd0[part]["state"]) == len(sd0[part]["param_groups"][0]["params"])
        for i, st in sd0[part]["state"].items():
            assert st.keys() == sd1[part]["state"][i].keys() and all(torch.equal(v, sd1[part]["state"][i][k]) for k, v in st.items())
    assert all("momentum_buffer" in resumed.opt_wm.muon.state[p] for p in resumed.wm.net.hidden_matrices())
    fresh = rf_agent("supertrack", "body", ref, seed=1, **kw)  # the networks only
    fresh.wm.load_state_dict(ck["wm"]), fresh.bal_wm.load_state_dict(ck["bal_wm"])
    set_lr(fresh.opt_wm, 2.0**-11)
    for seed in (20, 21):
        for a in (agent, resumed, fresh):
            wm_step(a, buf, seed)
    assert all(torch.equal(p, q) for p, q in zip(agent.wm.parameters(), resumed.wm.parameters()))
    t0, t1 = opt_tensors(agent.opt_wm), opt_tensors(resumed.opt_wm)
    assert len(t0) == len(t1) and all(torch.equal(x, y) for x, y in zip(t0, t1))
    assert not all(torch.equal(p, q) for p, q in zip(agent.wm.parameters(), fresh.wm.parameters()))
    check_resume(ck["args"], SimpleNamespace(**RESUME_MATCH | {"wm_opt": "muon"}))
    with pytest.raises(ValueError, match="--wm_opt must match the checkpoint's run"):
        check_resume(ck["args"], SimpleNamespace(**RESUME_MATCH | {"wm_opt": "adam"}))


# ---------------------------------------------------------------------------------------------------- GPU: a short run
@pytest.mark.slow
@pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_SIM and HAVE_DATA and HAVE_LAUNCHER), reason="needs a GPU, Newton and the assets")
@pytest.mark.parametrize("algo", ["supertrack", "supermimic"])
def test_train_muscles_rigidformer_muon_run(tmp_path, algo):
    """A short train_muscles run with the study's RigidFormer (fused blocks and features), --wm_opt muon and the systems
    options, as test_muscle_systems' run with the MLP: the collection step, the world-model update (MuonAdam's capturable
    step) and the policy update are captured and replayed, no update is skipped, and last.pt holds MuonAdam's state, a
    momentum per block matrix."""
    log = tmp_path / "run"
    cmd = [sys.executable, "-m", "ndm_gym.train_muscles", "--algo", algo, "--num_envs", "64", "--pi_batch", "256", "--wm_batch", "256",
           "--warmup_steps", "16", "--wm_only_iters", "1", "--log_every", "2", "--ckpt_every", "4", "--iterations", "8", *FUSED_RF, *SYSTEMS,
           "--log_dir", str(log)]  # fmt: skip
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for what in ("one collection step", "the world-model update", "the policy update"):
        assert f"captured {what}" in r.stdout, what
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == [2, 4, 6, 8] and all(x["wm_skipped"] == "0" and x["pi_skipped"] == "0" for x in rows)
    last = torch.load(log / "last.pt", map_location="cpu", weights_only=False)
    assert last["iteration"] == 8 and last["args"]["wm_opt"] == "muon" and set(last["opt_wm"]) == {"muon", "adam"}
    assert len(last["opt_wm"]["muon"]["state"]) == 2 * len(MATRICES)  # --rf_layers 2
