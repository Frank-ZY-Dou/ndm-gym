"""The HOI world-model benchmark's scoring (wmbench/fit_hoi.py's window_scores, which wmbench/score_hoi.py shares) on
CPU: the scores of a step with known errors fed back on itself, the object's rotation error exact; speed_hoi.py's
inference paths (needs Isaac Lab's packages): --rf_kernels refused on the MLP, the paths without it described as
before; with a GPU, --rf_kernels refused on a bf16 inference copy, and score_hoi through the eager path reproducing a
fit's last rows on synthetic datasets."""

import importlib.util
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from conftest import REPO
from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots.g1 import NUM_J
from ndm_gym.state import OBJ_DIM, STATE_DIM, flatten_object, flatten_state
from ndm_gym.tasks.hdmi import DECIMATION, NUM_ACTIONS
from ndm_gym.worldmodels.hoi import ContextLayout, HOIWorldModel

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
needs_isaaclab = pytest.mark.skipif(not HAVE_ISAACLAB, reason="speed_hoi.py and the task constants import Isaac Lab's packages")
needs_gpu = pytest.mark.skipif(not (HAVE_ISAACLAB and torch.cuda.is_available()), reason="needs a GPU and Isaac Lab's packages")
MOVE_SUITCASE = ContextLayout(55, False, 0)  # HDMISim.context_layout of move_suitcase
E = 2  # its end effectors


def _states(n: int, g: torch.Generator):
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    S = (torch.cat((r(n, 2), 0.78 + 0.02 * r(n, 1)), -1), so3_exp(r(n, 3)), 0.3 * r(n, 3), 0.3 * r(n, 3), 0.1 * r(n, NUM_J), 0.5 * r(n, NUM_J))
    So = (S[0] + torch.tensor([0.4, 0.0, 0.1]) + 0.05 * r(n, 3), so3_exp(r(n, 3)), 0.1 * r(n, 3), 0.1 * r(n, 3))
    return S, So


@pytest.mark.parametrize("hinged", [False, True])
def test_window_scores_of_a_drifting_step(hinged):
    """A step that drifts by known amounts at every call, fed back on itself from the windows' first states, against
    recordings that stand still: errors k times the drifts (a translation of the root moves every keypoint by it), the
    forces' errors their offsets; the step sees each window step's actions and context."""
    from ndm_gym.robots import g1
    from ndm_gym.tasks.hdmi import KEYPOINT_NAMES
    from ndm_gym.wmbench.fit_hoi import SCORE_COLUMNS, score_fields, window_scores

    n, k = 64, 5
    g = torch.Generator().manual_seed(int(hinged))
    S, So = _states(n, g)
    o = torch.cat((0.3 * torch.randn(n, 1, generator=g), torch.randn(n, 1, generator=g)), -1) if hinged else flatten_object(So)
    still = lambda x: x.unsqueeze(1).expand(n, k, *x.shape[1:]).clone()  # noqa: E731  (n, k, ...): the same at every step
    w = {"s": still(flatten_state(S)), "s1": still(flatten_state(S)), "o": still(o), "o1": still(o), "u": torch.randn(n, k, DECIMATION, NUM_ACTIONS, generator=g),
         "ctx": torch.randn(n, k, MOVE_SUITCASE.dim, generator=g), "f_eef": torch.zeros(n, k, E, 3), "f_feet": torch.zeros(n, k, DECIMATION, 2, 3)}  # fmt: skip
    dp, dq, dpo, th, angle, fe, ff = torch.tensor([0.003, -0.004, 0.0]), 0.002, torch.tensor([0.0, 0.006, -0.008]), 0.03, 0.01, 1.5, 0.25
    rot = so3_exp(torch.tensor([0.0, 0.0, th]))
    kin = g1.kinematics("cpu", tracked=list(KEYPOINT_NAMES))

    def drifting(dp, dq, seen):
        def step(S, So, u, ctx):
            seen.append((u, ctx))
            p, R, v, w_, q, qd = S
            So1 = (So[0] + angle, So[1]) if hinged else (So[0] + dpo, rot @ So[1], So[2], So[3])
            return (p + dp, R, v, w_, q + dq, qd), So1, (torch.full((n, E, 3), fe), torch.full((n, DECIMATION, 2, 3), -ff))

        return step

    seen = []
    s = window_scores(drifting(dp, 0.0, seen), w, kin, hinged)
    assert len(seen) == k and all(torch.equal(u, w["u"][:, j]) and torch.equal(c, w["ctx"][:, j]) for j, (u, c) in enumerate(seen))
    assert list(s) == SCORE_COLUMNS.split(",")[2:] and s["n"] == n
    approx = lambda x: pytest.approx(x, rel=1e-4, abs=1e-3)  # noqa: E731  (fp32)
    assert s["keypoint_mm"] == approx(1e3 * k * float(dp.norm())) and s["keypoint_p90_mm"] == approx(1e3 * k * float(dp.norm()))
    assert s["joint_mrad"] == 0.0
    if hinged:
        assert math.isnan(s["obj_mm"]) and s["obj_mrad"] == approx(1e3 * k * angle)
    else:
        assert s["obj_mm"] == approx(1e3 * k * float(dpo.norm())) and s["obj_mrad"] == pytest.approx(1e3 * k * th, rel=1e-3)
    assert s["eef_force_n"] == approx(fe) and s["feet_force_n"] == approx(ff)
    fields = score_fields(s).split(",")
    assert len(fields) == 8 and fields[0] == f"{s['keypoint_mm']:.3f}" and fields[5] == f"{s['eef_force_n']:.4f}" and fields[7] == str(n)
    assert fields[3] == ("nan" if hinged else f"{s['obj_mm']:.3f}")
    s = window_scores(drifting(torch.zeros(3), dq, []), w, kin, hinged)  # the joints drift: they move the keypoints too
    assert s["joint_mrad"] == approx(1e3 * k * dq) and s["keypoint_mm"] > 0


@pytest.mark.parametrize("angle", [1e-3, 3e-2])
def test_window_scores_object_rotation_is_exact(angle):
    """A prediction off by a known small rotation about a random axis is scored at that angle within 1e-6 rad, window by
    window (the arccos of the trace, used before, lost such angles: in TF32 and, near zero, in fp32)."""
    from ndm_gym.robots import g1
    from ndm_gym.tasks.hdmi import KEYPOINT_NAMES
    from ndm_gym.wmbench.fit_hoi import window_scores

    n = 32
    g = torch.Generator().manual_seed(7)
    S, So = _states(n, g)
    one = lambda x: x.unsqueeze(1).clone()  # noqa: E731  one-step windows that stand still
    w = {"s": one(flatten_state(S)), "s1": one(flatten_state(S)), "o": one(flatten_object(So)), "o1": one(flatten_object(So)),
         "u": torch.zeros(n, 1, DECIMATION, NUM_ACTIONS), "ctx": torch.zeros(n, 1, MOVE_SUITCASE.dim), "f_eef": torch.zeros(n, 1, E, 3),
         "f_feet": torch.zeros(n, 1, DECIMATION, 2, 3)}  # fmt: skip
    turn = so3_exp(angle * torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1))
    kin = g1.kinematics("cpu", tracked=list(KEYPOINT_NAMES))
    for i in range(n):  # one window at a time: every window's angle, not only their mean

        def step(S, So, u, ctx):
            return S, (So[0], So[1] @ turn[i : i + 1], So[2], So[3]), (torch.zeros(1, E, 3), torch.zeros(1, DECIMATION, 2, 3))

        s = window_scores(step, {key: v[i : i + 1] for key, v in w.items()}, kin, False)
        assert abs(s["obj_mrad"] - 1e3 * angle) <= 1e-3 and s["obj_mm"] == 0.0


def _model(rigidformer: bool):
    from ndm_gym.robots.g1 import G1_JOINT_NAMES
    from ndm_gym.tasks.hdmi import JOINT_NAMES
    from ndm_gym.worldmodels.hoi import rigidformer_args

    torch.manual_seed(0)
    if not rigidformer:
        return HOIWorldModel("accel", False, E, MOVE_SUITCASE, hidden=(32, 32))
    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    consts = {"eef_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "q0": [0.0] * NUM_J, "act_scale": [0.4] * NUM_ACTIONS,
              "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip
    return HOIWorldModel("accel", False, E, MOVE_SUITCASE, rigidformer=rigidformer_args(consts, 128, 3, 4, 2))


def _path(**kw):
    return SimpleNamespace(**({"compile": "none", "inference_copy": "none", "rf_attention": "token", "fused": False, "rf_kernels": "none"} | kw))


@needs_isaaclab
@pytest.mark.parametrize("kernels", ["blocks", "whole"])
def test_rf_kernels_refused_on_the_mlp(kernels):
    from ndm_gym.wmbench.speed_hoi import inference_model

    for copy in ("none", "amp"):
        with pytest.raises(ValueError, match="the MLP backbone has neither"):
            inference_model(_model(False), _path(rf_kernels=kernels, inference_copy=copy))


@needs_isaaclab
def test_paths_without_rf_kernels_as_before():
    """--rf_kernels none (the default): the model as built or its inference copy, compiled or not, and the detail string
    as speed_hoi wrote it before the option existed."""
    from ndm_gym.wmbench.speed_hoi import inference_model

    for rf in (False, True):
        wm = _model(rf)
        m, step, path = inference_model(wm, _path())
        assert m is wm and step == wm.step and path == "compile none"
        m, _, path = inference_model(wm, _path(compile="reduce-overhead", inference_copy="bf16", rf_attention="sdpa"))
        assert m is not wm and m.net_dtype == torch.bfloat16
        assert path == "compile reduce-overhead inference copy bf16" + (" attention sdpa" if rf else "")
    with pytest.raises(ValueError, match="--fused needs --inference_copy"):
        inference_model(_model(True), _path(fused=True))


@needs_gpu
@pytest.mark.parametrize("kernels", ["blocks", "whole"])
def test_rf_kernels_on_the_fp32_models_and_refused_on_a_bf16_copy(kernels):
    """fused_copy takes the RigidFormer as trained or its fp32 inference copy, and refuses a bf16 copy (the kernels run
    a model with fp32 parameters); the whole network is refused on compute capability 9.x unless NDM_RF_WHOLE_ON_SM90=1."""
    from ndm_gym.kernels.rf_block import FusedBlock, FusedHOIRigidFormerNet
    from ndm_gym.wmbench.speed_hoi import inference_model

    wm = _model(True).cuda()
    with pytest.raises(ValueError, match="fp32 parameters"):
        inference_model(wm, _path(inference_copy="bf16", fused=True, rf_kernels=kernels))
    if kernels == "whole" and torch.cuda.get_device_capability()[0] == 9 and os.environ.get("NDM_RF_WHOLE_ON_SM90") != "1":
        with pytest.raises(RuntimeError, match="refused on compute capability 9"):
            inference_model(wm, _path(rf_kernels=kernels))
        return
    for copy in ("none", "amp"):
        m, _, path = inference_model(wm, _path(inference_copy=copy, fused=copy == "amp", rf_kernels=kernels))
        assert isinstance(m.net, FusedHOIRigidFormerNet) if kernels == "whole" else all(isinstance(b, FusedBlock) for b in m.net.blocks)
        assert path.endswith(f" rf_kernels {kernels}")


def _dataset(path, T: int, N: int, seed: int):
    """A synthetic dataset as collect_hoi.py saves one (move_suitcase's fields), with episode ends."""
    g = torch.Generator().manual_seed(seed)
    S, So = _states(N, g)
    s, o = [flatten_state(S)], [flatten_object(So)]
    for _ in range(T):  # random walks
        s.append(s[-1] + 0.01 * torch.randn(N, STATE_DIM, generator=g))
        o.append(o[-1] + 0.01 * torch.randn(N, OBJ_DIM, generator=g))
    s, o = torch.stack(s), torch.stack(o)
    r = lambda *shape: torch.randn(T, N, *shape, generator=g)  # noqa: E731
    data = {"s": s[:-1].clone(), "s1": s[1:].clone(), "o": o[:-1].clone(), "o1": o[1:].clone(), "u": r(DECIMATION, NUM_ACTIONS),
            "ctx": r(MOVE_SUITCASE.dim), "f_eef": r(E, 3), "f_feet": r(DECIMATION, 2, 3), "done": torch.rand(T, N, generator=g) < 0.03}  # fmt: skip
    torch.save({"meta": {"task": "move_suitcase", "context_layout": tuple(MOVE_SUITCASE)}, "data": data}, path)


@needs_gpu
@pytest.mark.parametrize("backbone,amp", [("mlp", "bf16"), ("mlp", "none"), ("rigidformer", "bf16")])
def test_score_hoi_reproduces_the_fit(tmp_path, backbone, amp):
    """score_hoi through the eager path (--compile none, no copy, the fit's --amp) writes the fit's last rows."""
    train, test, fit = tmp_path / "train.pt", tmp_path / "test.pt", tmp_path / "fit.csv"
    _dataset(train, 24, 48, 0)
    _dataset(test, 24, 48, 1)
    size = ["--hidden", "64", "64"] if backbone == "mlp" else ["--rf_dim", "64", "--rf_layers", "1", "--rf_heads", "2"]
    windows = ["--eval_windows", "256", "--eval_k", "4", "8"]

    def run(*a):
        p = subprocess.run([sys.executable, "-m", *a], cwd=REPO, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-4000:]

    run("ndm_gym.wmbench.fit_hoi", "--task", "move_suitcase", "--kind", "accel", "--backbone", backbone, *size, "--train", str(train),
        "--test", f"held={test}", "--steps", "6", "--batch", "64", "--window", "4", "--eval_every", "3", "--eager", "--amp", amp, *windows,
        "--out", str(fit))  # fmt: skip
    run("ndm_gym.wmbench.score_hoi", "--model", str(tmp_path / "fit_wm.pt"), "--test", f"held={test}", *windows, "--out", str(tmp_path / "score.csv"))
    fit_rows = [r.split(",") for r in fit.read_text().splitlines()[1:]]
    last = [",".join(r[3:]) for r in fit_rows if r[0] == "6"]
    scored = (tmp_path / "score.csv").read_text().splitlines()
    assert scored[0].endswith(",path") and len(last) == 2
    assert [r.rsplit(",", 1)[0] for r in scored[1:]] == last
    assert all(r.endswith(f",compile none amp {amp}") for r in scored[1:])
