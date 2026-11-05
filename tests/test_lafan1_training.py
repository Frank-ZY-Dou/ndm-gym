"""The LAFAN1 training path around the algorithms' updates: the normalizer fits of collection (Algo.update_wm_norms and
update_pi_norms, train.py's collect), the simulator's transition records (flatten_states, TrackingSim.step's previous
action), --resume, the speed options --critic_cudagraphs, --compile_eager, --goal_table and --pi_encode_once, and the tool
that rebuilds a run's arguments (tools/grad_stats.py). The simulator's tests run train.py or the simulator in a process of
this file (``--worker``) or run ``python -m ndm_gym.train``, on the synthetic clips of
tests/data/supertrack_golden.pt; they are marked slow and need a GPU and Isaac Lab."""

import argparse
import copy
import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from conftest import DEV, REPO, random_states, synthetic_clips

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
HAVE_SIM = torch.cuda.is_available() and HAVE_ISAACLAB


@pytest.fixture()
def eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _algo(name, kin, motion, mdp, **kw):
    from ndm_gym.algos import ALGOS
    from test_algo_equivalence import _args
    from test_bptt_tricks import ST_ARGS

    args = _args()
    vars(args).update({"algo": name, "wm_hidden": [64, 64], **ST_ARGS, **kw})
    torch.manual_seed(0)
    return ALGOS[name](args, kin, motion, mdp, DEV)


# ---------------------------------------------------------------------------------------------------- normalizers
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_norm_updates_fit_the_normalizers(eager, kin, synthetic_motion, synthetic_mdp, name, graphs):
    """Algo.update_wm_norms and update_pi_norms, which collection calls on every transition, update the world model's
    input and output normalizers and the policy's observation and goal normalizers exactly as RunningNorm.update on the
    world model's inputs and targets, the observations and the goals. With CUDA graphs (--wm_cudagraphs, --pi_cudagraphs)
    the new statistics are written into the tensors the graphs read: every normalizer keeps its tensors."""
    from ndm_gym.state import normalize_xi

    motion, mdp = synthetic_motion, synthetic_mdp
    algo = _algo(name, kin, motion, mdp, wm_cudagraphs=graphs, pi_cudagraphs=graphs)
    norms = {"in": algo.wm.in_norm, "out": algo.wm.out_norm, "obs": algo.pi.obs_norm, "goal": algo.pi.goal_norm}
    ref = copy.deepcopy(norms)
    held = {k: (n.mean, n.var, n.count) for k, n in norms.items()}
    g = torch.Generator().manual_seed(0)
    for seed in (1, 2):
        f, S0, xi = random_states(motion, 64, seed)
        S1, a = motion.ref_state(f + 1), torch.randn(64, 29, generator=g)
        o, goal = mdp.add_noise(mdp.obs(S0, torch.randn(64, 29, generator=g), xi, f)), motion.goal(f, algo.K)
        algo.update_wm_norms(S0, S1, a, xi)
        algo.update_pi_norms(o, goal)
        ref["in"].update(algo.wm.inputs(S0, a, normalize_xi(xi))), ref["out"].update(algo.wm.targets(S0, S1))
        ref["obs"].update(o), ref["goal"].update(goal)
    for k, n in norms.items():
        assert float(n.count) == 128, k
        for attr in ("mean", "var", "count"):
            torch.testing.assert_close(getattr(n, attr), getattr(ref[k], attr), rtol=0, atol=0, msg=lambda m: f"{k}.{attr}: {m}")
        if graphs:
            assert all(x is y for x, y in zip((n.mean, n.var, n.count), held[k])), k


# ---------------------------------------------------------------------------------------------------- options and tools
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_critic_cudagraphs_follow_the_eager_loop(eager, synthetic_motion):
    """SuperMimic --critic_cudagraphs: the critic's minibatch steps run eagerly on a side stream at the first policy update
    and as a captured CUDA graph from the second; over three updates the critic, its normalizer and its losses follow the
    default loop up to Adam's rounding (the capturable implementation computes its step sizes on the device)."""
    from ndm_gym.algos.supermimic import SuperMimic
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import flatten_state, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin = g1.kinematics("cuda")
    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    buf, g = ReplayBuffer(12, 64, "cuda", priorities=False), torch.Generator().manual_seed(7)
    for t in range(12):
        f, S, xi = random_states(synthetic_motion, 64, seed=100 + t)
        f = f + t
        tr = {"s": flatten_state(S), "s1": flatten_state(synthetic_motion.ref_state(f + 1)), "a": torch.randn(64, 29, generator=g),
              "a_prev": torch.randn(64, 29, generator=g), "xi": xi, "r": torch.rand(64, generator=g), "f": f,
              "ep_step": torch.full((64,), t, dtype=torch.int32), "done": torch.rand(64, generator=g) < 0.05}  # fmt: skip
        buf.add({k: v.cuda() for k, v in tr.items()})
    w = {k: v.flatten(0, 1) for k, v in buf.data.items()}
    S0, S1 = unflatten_state(w["s"]), unflatten_state(w["s1"])
    out = {}
    for graphs in (False, True):
        args = _args()
        vars(args).update(algo="supermimic", wm_hidden=[64, 64], critic_cudagraphs=graphs)
        torch.manual_seed(0)
        algo = SuperMimic(args, kin, motion, mdp, "cuda")
        algo.update_wm_norms(S0, S1, w["a"], w["xi"])
        algo.update_pi_norms(mdp.obs(S0, w["a_prev"], w["xi"], w["f"]), motion.goal(w["f"], algo.K))
        out[graphs] = []
        for seed in (2, 3, 4):
            torch.manual_seed(seed)
            v_loss = algo.policy_update(buf)[1]
            out[graphs].append([v_loss.clone(), *(t.detach().clone() for t in algo.critic.state_dict().values())])
        assert (algo.critic_graph is not None and "graph" in algo.critic_graph) == graphs
    for step, (a, b) in enumerate(zip(out[False], out[True])):
        for x, y in zip(a, b):
            torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-6, msg=lambda m: f"update {step}: {m}")


@pytest.mark.skipif(not HAVE_ISAACLAB, reason="ndm_gym.train imports Isaac Lab's launcher")
def test_tools_know_every_training_argument(monkeypatch):
    """tools/grad_stats.py rebuilds ndm_gym.train's arguments to read a run's args.json. A default run's, as train.py
    writes it (with the speed options of add_speed_args), holds no argument it does not know and does not differ from the
    arguments it expects."""
    from ndm_gym import train

    monkeypatch.setattr(sys, "argv", ["ndm_gym.train", "--algo", "supertrack"])
    written = json.loads(json.dumps(vars(train.parse_args())))  # what train.py writes to args.json
    speed = {"fused_adam", "elementwise_3x3", "wm_cudagraphs", "pi_cudagraphs", "critic_cudagraphs", "compile_eager"}
    assert speed | {"rf_fused_rope"} <= set(written)
    monkeypatch.syspath_prepend(os.path.join(REPO, "tools"))
    import grad_stats

    assert grad_stats.run_arguments(written, [], launched=set()) == written


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_compile_eager_compiles_the_eager_feature_functions(monkeypatch, kin, synthetic_motion, synthetic_mdp, name):
    """--compile_eager: the world model's normalizer features and SuperMimic's critic input of the policy loss are the
    compiled functions (torch.compile recorded here, as the identity), without it the eager ones; with the identity
    in place of the compiler both give the same bits (the option only changes which function runs)."""
    from test_bptt_tricks import _algo as bptt_algo

    compiled = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compiled.append(getattr(fn, "__name__", str(fn))) or fn)
    out = {}
    for flag in (False, True):
        compiled.clear()
        algo, buf = bptt_algo(name, kin, synthetic_motion, synthetic_mdp, compile_eager=flag)
        assert ("inputs" in compiled) == flag and algo.compile_eager == flag
        calls = []
        if name == "supermimic":
            critic_input_c = algo.critic_input_c
            algo.critic_input_c = lambda *a: calls.append(1) or critic_input_c(*a)
        torch.manual_seed(3)
        out[flag] = [t.detach().clone() for t in algo.policy_update(buf)[:2]] + [p.grad.clone() for p in algo.pi.parameters()]
        if name == "supermimic":  # the critic's targets use the compiled input either way; the policy loss's with the option
            assert len(calls) == 1 + flag, calls
    for a, b in zip(out[False], out[True]):
        torch.testing.assert_close(b, a, rtol=0, atol=0, equal_nan=True)  # SuperTrack's value loss is nan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled for the GPU, as training runs them")
@pytest.mark.parametrize("backbone", ["mlp", "rigidformer"])
def test_compile_eager_functions_match_the_eager_ones(synthetic_motion, backbone):
    """--compile_eager's compiled functions on the GPU equal the eager ones up to float rounding: the collection's reward
    and its terms (TrackingSim.reward_fn), the world model's normalizer features (Algo.norm_inputs; the RigidFormer's
    forward kinematics) and SuperMimic's critic input with its gradients to the imagined state."""
    from ndm_gym.algos.supermimic import SuperMimic
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import normalize_xi
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args

    kin = g1.kinematics("cuda")
    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
    args = _args()
    vars(args).update(algo="supermimic", wm_hidden=[64, 64], compile_eager=True)
    if backbone == "rigidformer":
        vars(args).update(wm_hidden=[2048, 2048, 2048], wm_backbone="rigidformer", rf_dim=64, rf_layers=1, rf_heads=2, rf_registers=2,
                          rf_no_tree_bias=False, rf_no_rope=False)  # fmt: skip
    torch.manual_seed(0)
    algo = SuperMimic(args, kin, motion, mdp, "cuda")
    f, S, xi = random_states(synthetic_motion, 256, seed=11)
    f, S, xi = f.cuda(), tuple(t.cuda() for t in S), xi.cuda()
    g = torch.Generator().manual_seed(12)
    a, a_prev = torch.randn(256, 29, generator=g).cuda(), torch.randn(256, 29, generator=g).cuda()
    tol = dict(rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        r, terms = torch.compile(mdp.reward)(S, f + 1, a, a_prev, terms=True)
        r0, terms0 = mdp.reward(S, f + 1, a, a_prev, terms=True)
        torch.testing.assert_close(r, r0, **tol)
        for k in terms0:
            torch.testing.assert_close(terms[k], terms0[k], **tol, msg=lambda m, k=k: f"{k}: {m}")
        torch.testing.assert_close(algo.norm_inputs(S, a, normalize_xi(xi)), algo.wm.inputs(S, a, normalize_xi(xi)), **tol)
    grads = []
    for fn in (algo.critic_input_c, algo._critic_input):
        Sg = tuple(t.clone().requires_grad_(True) for t in S)
        y = fn(Sg, f, normalize_xi(xi), motion.goal(f, algo.K))
        (y * torch.linspace(-1, 1, y.shape[-1], device="cuda")).sum().backward()
        grads.append([y.detach()] + [t.grad for t in Sg])
    for x, y in zip(*grads):
        torch.testing.assert_close(x, y, **tol)


def _close(b, a, rel: float, msg=None):
    """b equals a up to ``rel`` times a's largest magnitude, entry by entry (a gradient's small entries round as its large
    ones do)."""
    torch.testing.assert_close(b, a, rtol=rel, atol=rel * float(a.abs().max()), msg=msg)


def _goal_table_motion(kin, dev=DEV, chunk: int = 65536):
    """The synthetic clips' motion library with every frame's goal cached (--goal_table)."""
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip

    motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, dev)
    motion.cache_goals(10, chunk=chunk)
    return motion


def test_goal_table_reads_the_goals(kin, synthetic_motion):
    """--goal_table: MotionLib.cache_goals computes every frame's goal once, in chunks of frames; goal() then reads the
    table, bit for bit the direct computation's goals of any frames. Another number of future frames and a second cache
    are refused."""
    motion = _goal_table_motion(kin, chunk=37)  # chunks that end inside the clips
    assert motion.goal_table.shape == (motion.num_frames, 730)
    g = torch.Generator().manual_seed(0)
    for n in (1, 7, 256):
        f = torch.randint(0, motion.num_frames, (n,), generator=g)
        torch.testing.assert_close(motion.goal(f, 10), synthetic_motion.goal(f, 10), rtol=0, atol=0)
    with pytest.raises(ValueError, match="holds 10"):
        motion.goal(f, 5)
    with pytest.raises(ValueError, match="cached already"):
        motion.cache_goals(10)


@pytest.mark.parametrize("name", ["supertrack", "supermimic"])
def test_goal_table_updates_are_the_default_ones(eager, kin, synthetic_motion, synthetic_mdp, name):
    """--goal_table in eager mode, where every goal was the direct computation's: the policy update (the goals of its
    start, of every imagined step and of the critic) is the default one bit for bit. Compiled imagination steps computed
    their goals up to rounding (test_goal_options_compiled)."""
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP
    from test_bptt_tricks import _algo as bptt_algo

    motion = _goal_table_motion(kin)
    out = {}
    for table in (False, True):
        m, mdp = (motion, MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=DEV))) if table else (synthetic_motion, synthetic_mdp)
        algo, buf = bptt_algo(name, kin, m, mdp)
        torch.manual_seed(3)
        out[table] = [t.detach().clone() for t in algo.policy_update(buf)] + [p.grad.clone() for p in algo.pi.parameters()]
    for a, b in zip(out[False], out[True]):
        torch.testing.assert_close(b, a, rtol=0, atol=0, equal_nan=True)  # SuperTrack's value loss and imagined reward are nan


@pytest.mark.parametrize("graphable", [False, True])
def test_pi_encode_once_is_the_policy_update(eager, kin, synthetic_motion, synthetic_mdp, graphable):
    """SuperTrack --pi_encode_once: the policy's goal encoder runs once over every imagined step's goal and the steps run
    its decoder (TrackingPolicy.encode and decode, the two halves of its forward, bit for bit). The policy update, default
    and graphable, gives the default's loss, gradient norm and policy gradient up to float rounding (the encoder's products
    over all steps at once). With --pi_rollout_clip, which reruns the policy on every step's goal, it is refused."""
    from test_bptt_tricks import _algo as bptt_algo

    kw = dict(fused_adam=True, no_sync=True) if graphable else {}
    out = {}
    for once in (False, True):
        algo, buf = bptt_algo("supertrack", kin, synthetic_motion, synthetic_mdp, pi_encode_once=once, **kw)
        assert algo.encode_once == once
        torch.manual_seed(3)
        res = algo.policy_update_graphable(buf) if graphable else algo.policy_update(buf)
        out[once] = [res[0], res[3]] + [p.grad.clone() for p in algo.pi.parameters()]
    for a, b in zip(out[False], out[True]):  # here 9e-7 of each gradient's largest entry at most
        _close(b, a, 1e-5)
    f, S, xi = random_states(synthetic_motion, 16, seed=5)
    goal = synthetic_motion.goal(f, 10)
    obs = synthetic_mdp.add_noise(synthetic_mdp.obs(S, torch.zeros(16, 29), xi, f)).unsqueeze(1).expand(-1, 10, -1)
    torch.testing.assert_close(algo.pi.decode(algo.pi.encode(goal), obs), algo.pi(obs, goal), rtol=0, atol=0)
    with pytest.raises(ValueError, match="pi_rollout_clip"):
        bptt_algo("supertrack", kin, synthetic_motion, synthetic_mdp, pi_encode_once=True, pi_rollout_clip=2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled for the GPU, as training runs them")
def test_goal_options_compiled(synthetic_motion):
    """--goal_table and SuperTrack's --pi_encode_once with the imagination compiled on the GPU, as training runs it: the
    policy update's loss, gradient norm and policy gradient equal the default's up to float rounding."""
    from ndm_gym.algos.supertrack import SuperTrack
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.tasks.tracking import MDP
    from test_algo_equivalence import _args
    from test_bptt_tricks import ST_ARGS

    kin = g1.kinematics("cuda")
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
    out = {}
    for table, once in ((False, False), (True, False), (True, True)):
        motion = MotionLib([Clip(**c) for c in synthetic_clips()], kin, "cuda")
        if table:
            motion.cache_goals(10)
        mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device="cuda"))
        args = _args()
        vars(args).update(algo="supertrack", wm_hidden=[64, 64], pi_encode_once=once, **ST_ARGS)
        torch.manual_seed(0)
        algo = SuperTrack(args, kin, motion, mdp, "cuda")
        torch.nn.init.normal_(algo.wm.net[-1].weight, std=0.05)  # zero at initialization: no gradient through the network
        algo.wm.in_norm.update(algo.wm.inputs(S0, w["a"], normalize_xi(w["xi"]))), algo.wm.out_norm.update(algo.wm.targets(S0, S1))
        algo.update_pi_norms(mdp.obs(S0, w["a_prev"], w["xi"], w["f"]), motion.goal(w["f"], algo.K))
        torch.manual_seed(3)
        res = algo.policy_update(buf)
        out[table, once] = [res[0], res[3]] + [p.grad.clone() for p in algo.pi.parameters()]
    for key in ((True, False), (True, True)):  # here 3e-6 of each gradient's largest entry at most; the default repeats bit for bit
        for a, b in zip(out[False, False], out[key]):
            _close(b, a, 1e-4, msg=lambda m, key=key: f"--goal_table, --pi_encode_once {key}: {m}")


# ---------------------------------------------------------------------------------------------------- transitions
@pytest.mark.skipif(not HAVE_ISAACLAB, reason="the simulator module imports Isaac Lab")
def test_flatten_states_match_flatten_state(synthetic_motion):
    """The transition record of TrackingSim.step (and HDMISim's): flatten_states converts the rotations of all its states
    at once and returns each state's flatten_state (robots) or flatten_object (objects), bit for bit and in order."""
    from ndm_gym.core.rotations import so3_exp
    from ndm_gym.sim.rigid.isaaclab_mjwarp import flatten_states
    from ndm_gym.state import flatten_object, flatten_state

    _, S, _ = random_states(synthetic_motion, 16, seed=3)
    _, S1, _ = random_states(synthetic_motion, 16, seed=4)
    g = torch.Generator().manual_seed(5)
    rn = lambda: torch.randn(16, 3, generator=g)  # noqa: E731
    So, So1 = ((rn(), so3_exp(rn()), rn(), rn()) for _ in range(2))
    want = [flatten_state(S), flatten_state(S1), flatten_object(So), flatten_object(So1)]
    for got, n in ((flatten_states(S, S1), 2), (flatten_states(S, S1, So, So1), 4)):
        assert len(got) == n and all(torch.equal(x, y) for x, y in zip(got, want[:n]))


# ---------------------------------------------------------------------------------------------------- GPU, simulator
TRAIN = ["--algo", "supertrack", "--num_envs", "32", "--buffer_steps", "64", "--warmup_steps", "16", "--wm_batches", "1", "--wm_batch", "64",
         "--pi_batch", "64", "--wm_hidden", "256", "256", "--wm_only_iters", "1", "--log_every", "1", "--ckpt_every", "1"]  # fmt: skip


def write_motions(d) -> str:
    """The synthetic clips as LAFAN1 files (rows of root position, root quaternion xyzw and 29 joint angles at 30 fps)."""
    import numpy as np

    os.makedirs(d, exist_ok=True)
    for c in synthetic_clips(sim=True):
        q = c["root_quat"]  # wxyz
        np.savetxt(os.path.join(d, c["name"] + ".csv"), torch.cat((c["root_p"], q[:, 1:], q[:, :1], c["joints"]), -1).double().numpy(), delimiter=",")
    return str(d)


def run(*cmd: str):
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, timeout=1800)
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr  # Warp's teardown messages follow it
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    return r


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
def test_resumed_run_refills_the_replay_and_updates(tmp_path):
    """A run resumed from its last.pt collects --warmup_steps before its first iteration (the replay buffer is not saved),
    then fits the world model and updates the policy; its log continues without a second header."""
    motions, log = write_motions(tmp_path / "motions"), tmp_path / "run"
    base = ["-m", "ndm_gym.train", *TRAIN, "--motion_dir", motions, "--log_dir", str(log)]
    run(*base, "--iterations", "2")
    assert torch.load(log / "last.pt", map_location="cpu", weights_only=False)["iteration"] == 2
    run(*base, "--iterations", "3", "--resume", str(log / "last.pt"))
    assert torch.load(log / "last.pt", map_location="cpu", weights_only=False)["iteration"] == 3
    lines = (log / "train_log.csv").read_text().strip().splitlines()
    assert sum(x.startswith("iteration,") for x in lines) == 1
    rows = list(csv.DictReader(lines))
    assert [int(r["iteration"]) for r in rows] == [1, 2, 3] and rows[2]["pi_loss"] not in ("nan", "")
    # iteration 3: the resumed run's 16 warm-up steps and the iteration's own 2 (utd 0.4), as iteration 1 of the first run
    steps = [int(r["sim_steps"]) for r in rows]
    assert steps[0] == steps[2] - steps[1] == (16 + 2) * 32, steps


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
def test_collect_fits_the_normalizers_on_its_transitions(tmp_path):
    """train.py's collection fits the world model's normalizers on each transition it adds to the replay (its states,
    setpoints and context, bit for bit) and the policy's on the goals and observations the policy is given."""
    motions, out = write_motions(tmp_path / "motions"), tmp_path / "seen.json"
    run(os.path.abspath(__file__), "--worker", "collect", "--out", str(out), "--motion_dir", motions, "--log_dir", str(tmp_path / "run"))
    seen = json.load(open(out))
    assert seen == {"wm": 18, "pi": 18}, seen  # 16 warm-up steps, 2 in the iteration


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_step_keeps_its_own_previous_action(tmp_path):
    """TrackingSim.step keeps a copy of the action as the next transition's previous action: the caller may change its
    tensor afterwards, and a reset clears the copy (to zero), not the caller's tensor."""
    out = tmp_path / "steps.pt"
    run(os.path.abspath(__file__), "--worker", "step", "--out", str(out))
    rec = torch.load(out)
    assert rec["done"][:-1].any(), "no episode ended: the test needs resets"
    assert torch.equal(rec["after"], rec["a"])  # the caller's tensors, read back after their steps: untouched
    for t in range(1, len(rec["a"])):
        assert torch.equal(rec["a_prev"][t], torch.where(rec["done"][t - 1].unsqueeze(-1), 0.0, rec["a"][t - 1])), t


def worker_collect(out: str, motion_dir: str, log_dir: str):
    """train.main() for one iteration (eager, 32 envs), with Algo.update_wm_norms and update_pi_norms checked at every
    call against the transition just added to the replay and the simulator's state before the step."""
    sys.path.insert(0, REPO)
    torch.compile = lambda fn, **kw: fn  # eager: the test is about collection

    from ndm_gym import train
    from ndm_gym.algos.common import Algo
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim
    from ndm_gym.state import unflatten_state

    last, seen = {}, {"wm": 0, "pi": 0}
    add, wm_norms, pi_norms, init = ReplayBuffer.add, Algo.update_wm_norms, Algo.update_pi_norms, TrackingSim.__init__

    def sim_init(self, *a, **kw):
        init(self, *a, **kw)
        last["env"] = self

    def add_(self, tr):
        last["tr"] = {k: v.clone() for k, v in tr.items() if torch.is_tensor(v)}
        return add(self, tr)

    def wm_norms_(self, S0, S1, a, xi):
        tr = last.pop("tr")  # the transition just added
        want = (*unflatten_state(tr["s"]), *unflatten_state(tr["s1"]), tr["a"], tr["xi"])
        if not all(torch.equal(x, y) for x, y in zip((*S0, *S1, a, xi), want)):
            raise AssertionError("update_wm_norms is not given the transition just added to the replay")
        seen["wm"] += 1
        return wm_norms(self, S0, S1, a, xi)

    def pi_norms_(self, o, g):
        env = last["env"]
        if not torch.equal(g, self.motion.goal(env.f, self.K)):
            raise AssertionError("update_pi_norms is not given the goals of the simulator's frames")
        clean = self.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        if not bool(((o - clean).abs() <= self.mdp.obs_noise + 1e-5).all()) or torch.equal(o, clean):
            raise AssertionError("update_pi_norms is not given the noisy observations of the simulator's state")
        seen["pi"] += 1
        return pi_norms(self, o, g)

    TrackingSim.__init__, ReplayBuffer.add, Algo.update_wm_norms, Algo.update_pi_norms = sim_init, add_, wm_norms_, pi_norms_
    sys.argv = ["ndm_gym.train", *TRAIN, "--iterations", "1", "--motion_dir", motion_dir, "--log_dir", log_dir]
    train.main()
    json.dump(seen, open(out, "w"))


def worker_step(out: str):
    """TrackingSim on the synthetic clips, 8 envs, episodes of at most 5 steps: 12 steps with fresh random actions, each
    action tensor changed in place after its step, as a caller may reuse it."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, n = "cuda:0", 8
    kin = g1.kinematics(dev)
    motion = MotionLib([Clip(**c) for c in synthetic_clips(sim=True)], kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    sim_cfg = make_sim(dev, launcher)
    rec = {"a": [], "after": [], "a_prev": [], "done": []}
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = TrackingSim(sim_cfg, n, motion, mdp, max_episode_steps=5)
        for _ in range(12):
            a = 0.1 * torch.randn(n, 29, device=dev)
            rec["a"].append(a.cpu())
            tr = env.step(a, env.state())
            rec["after"].append(a.cpu()), rec["a_prev"].append(tr["a_prev"].cpu()), rec["done"].append(tr["done"].cpu())
            a.mul_(-3.0).add_(1.0)  # the caller reuses its tensor
    torch.save({k: torch.stack(v) for k, v in rec.items()}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["collect", "step"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion_dir")
    ap.add_argument("--log_dir")
    cli = ap.parse_args()
    if cli.worker == "collect":
        worker_collect(cli.out, cli.motion_dir, cli.log_dir)
    else:
        worker_step(cli.out)


def test_reset_frames_stay_before_each_clip_end():
    """TrackingSim.sample_frames: a clip of 50 k + 1 frames has a last bin of one frame, its last; a reset drawn there
    starts one frame earlier, since its first step would read past the clip. Bins with room are drawn as before."""
    pytest.importorskip("isaaclab")
    from types import SimpleNamespace

    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim

    # clip 0: 51 frames (bins at 0 and 50, the second only its last frame); clip 1: 120 frames (bins at 51, 101, 151)
    motion = SimpleNamespace(bin_start=torch.tensor([0, 50, 51, 101, 151]), bin_clip=torch.tensor([0, 0, 1, 1, 1]),
                             clip_end=torch.tensor([50, 170]), num_bins=5)  # fmt: skip
    sim = SimpleNamespace(motion=motion, bin_G=torch.zeros(5), G_max=1.0, bin_w_avg=torch.full((5,), 0.2), device="cpu", fail_reset=0.0,
                          focus_q=None, focus_share=0.0, focus_static=False)  # fmt: skip
    torch.manual_seed(0)
    f, bins = TrackingSim.sample_frames(sim, 4000)
    last = motion.clip_end[motion.bin_clip[bins]]
    assert bool((f <= last - 1).all()) and bool((f[bins == 1] == 49).all()) and int((bins == 1).sum()) > 0
    # the bins with room: the same frames as the draw without the clamp
    torch.manual_seed(0)
    sim.bin_w_avg = torch.full((5,), 0.2)
    soft = torch.softmax(-25.0 * sim.bin_G / sim.G_max, 0)
    w = 0.999 * sim.bin_w_avg + 0.001 * soft
    probs = 0.5 / 5 + 0.4 * soft + 0.1 * w / w.sum()
    b2 = torch.multinomial(probs, 4000, replacement=True)
    start = motion.bin_start[b2]
    end = torch.minimum(start + 50, motion.clip_end[motion.bin_clip[b2]])
    f2 = start + (torch.rand(4000) * (end - start).float()).long()
    assert torch.equal(b2, bins) and torch.equal(f2[bins != 1], f[bins != 1])
