"""The tools of the Steel comparison (benchmarks/steel/results.md): collect.py's evaluation conditions, speed_pair.py's
replay workload and barrier, and replay_engine.py's windows. Their default paths stay as they were."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from conftest import DEV
from test_wmbench_packing import _recording


def _collect_args(**kw):
    base = dict(eval_conditions=False, action_noise=None, push_mag=0.0)
    return SimpleNamespace(**(base | kw))


def test_collection_conditions_default_is_training():
    from ndm_gym.wmbench.collect import collection_conditions

    assert collection_conditions(_collect_args()) == (False, 0.1, True)  # the earlier default action noise, sensor noise on
    assert collection_conditions(_collect_args(action_noise=0.3)) == (False, 0.3, True)
    assert collection_conditions(_collect_args(action_noise=0.0, push_mag=0.5)) == (False, 0.0, True)


def test_collection_conditions_evaluation():
    from ndm_gym.wmbench.collect import collection_conditions

    assert collection_conditions(_collect_args(eval_conditions=True)) == (True, 0.0, False)
    assert collection_conditions(_collect_args(eval_conditions=True, action_noise=0.0)) == (True, 0.0, False)
    for bad in (dict(action_noise=0.1), dict(push_mag=0.5)):
        with pytest.raises(ValueError):
            collection_conditions(_collect_args(eval_conditions=True, **bad))


def test_segments_and_schedule():
    from ndm_gym.wmbench.speed_pair import replay_schedule, segment_lengths

    assert segment_lengths(200, 32) == [32] * 6 + [8]
    assert segment_lengths(64, 32) == [32, 32]
    sched = replay_schedule(10, 4, 3)
    assert sched.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]


def test_timed_segments_times_only_the_segments():
    from ndm_gym.wmbench.speed_pair import timed_segments

    log = []
    t = timed_segments(lambda j: log.append(("step", j)), lambda k: log.append(("restart", k)), [3, 2], warmup=4, sync=lambda: None)
    assert t >= 0.0
    # segment 0's start, the warm-up from it, then each segment restarts and runs its steps
    assert log == [("restart", 0)] + [("step", j % 3) for j in range(4)] + [("restart", 0)] + [("step", j) for j in range(3)] + \
        [("restart", 1)] + [("step", j) for j in range(2)]


def test_replay_windows_follow_the_recording(tmp_path):
    from ndm_gym.buffer import FIELDS
    from ndm_gym.wmbench.collect import G1_RULES, pack
    from ndm_gym.wmbench.speed_pair import replay_windows

    data = _recording(T=40, N=6, ds=FIELDS["s"][0], da=FIELDS["a"][0], dx=FIELDS["xi"][0], seed=3)
    path = tmp_path / "rec.pt"
    torch.save({"meta": {}, "data": pack(data, G1_RULES)}, path)
    k = 5
    w = replay_windows(str(path), 16, k, seed=1, dev=DEV, full=True)
    assert w["s0"].shape == (16, FIELDS["s"][0]) and w["a"].shape == (16, k, FIELDS["a"][0]) and w["s1"].shape == (16, k, FIELDS["s"][0])
    for i in range(16):
        t0, e = int(w["t0"][i]), int(w["env"][i])
        assert torch.equal(w["s0"][i], data["s"][t0, e])
        assert torch.equal(w["a"][i], data["a"][t0:t0 + k, e])
        assert torch.equal(w["s1"][i], data["s1"][t0:t0 + k, e])
        assert torch.equal(w["f"][i], data["f"][t0:t0 + k, e])
        assert torch.equal(w["xi"][i], data["xi"][t0, e])
        assert not bool(data["done"][t0:t0 + k - 1, e].any())  # no episode ends inside a window
    again = replay_windows(str(path), 16, k, seed=1, dev=DEV)
    assert all(torch.equal(again[key], w[key]) for key in ("s0", "a", "xi"))  # the same draw for every tool


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="speed_pair's parser takes Isaac Lab's launcher arguments")
def test_speed_pair_replay_options(monkeypatch, capsys):
    """--replay is off by default and takes one G1 per env in the physics or ndm mode, timing the step."""
    from ndm_gym.wmbench import speed_pair

    base = ["speed_pair", "--pair", "clip.npz", "--out", "o.csv"]
    monkeypatch.setattr(sys, "argv", base + ["--mode", "physics", "--single"])
    a = speed_pair.parse_args()
    assert (a.replay, a.replay_len, a.replay_windows, a.replay_seed) == (None, 32, 8192, 0)
    for mode in ("physics", "ndm"):
        monkeypatch.setattr(sys, "argv", base + ["--mode", mode, "--single", "--replay", "rec.pt"])
        assert speed_pair.parse_args().replay == "rec.pt"
    for extra in (["--mode", "physics", "--replay", "rec.pt"], ["--mode", "env", "--replay", "rec.pt"],
                  ["--mode", "ndm", "--single", "--replay", "rec.pt", "--grad_horizon", "32"],
                  ["--mode", "ndm", "--single", "--replay", "rec.pt", "--replay_len", "0"]):  # fmt: skip
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit):
            speed_pair.parse_args()
    capsys.readouterr()


def test_replay_windows_from_an_export(tmp_path):
    """--replay with replay_engine export's .npz reads its windows: the same as drawn from the recording with the same seed."""
    import numpy as np

    from ndm_gym.buffer import FIELDS
    from ndm_gym.wmbench.collect import G1_RULES, pack
    from ndm_gym.wmbench.speed_pair import replay_windows

    data = _recording(T=40, N=6, ds=FIELDS["s"][0], da=FIELDS["a"][0], dx=FIELDS["xi"][0], seed=4)
    rec = tmp_path / "rec.pt"
    torch.save({"meta": {}, "data": pack(data, G1_RULES)}, rec)
    w = replay_windows(str(rec), 12, 5, seed=2, dev=DEV, full=True)
    exp = tmp_path / "win.npz"
    np.savez(exp, **{k: v.numpy() for k, v in w.items()}, seed=2)
    again = replay_windows(str(exp), 10, 4, seed=2, dev=DEV, full=True)
    assert torch.equal(again["s0"], w["s0"][:10]) and torch.equal(again["a"], w["a"][:10, :4]) and torch.equal(again["s1"], w["s1"][:10, :4])
    for bad in (dict(n=13, length=4, seed=2), dict(n=10, length=6, seed=2), dict(n=10, length=4, seed=3)):
        with pytest.raises(ValueError):
            replay_windows(str(exp), bad["n"], bad["length"], seed=bad["seed"], dev=DEV)


def test_barrier_meets_and_starts_together(tmp_path):
    """--barrier: every process waits for the others, then for one common start."""
    import threading
    import time

    from ndm_gym.wmbench.speed_pair import barrier

    path, n, lead, hold = tmp_path / "b", 3, 0.2, 0.5
    arrived, gos, starts, errors = [], [], [], []

    def caller():
        arrived.append(time.time())
        try:
            gos.append(barrier(str(path), n, lead=lead))
            starts.append(time.time())
        except Exception as e:  # a caller's error fails the test, not only its thread
            errors.append(e)

    ts = [threading.Thread(target=caller, daemon=True) for _ in range(n)]
    for t in ts[:-1]:
        t.start()
    deadline = time.time() + 60  # each wait ends as soon as its condition holds; the bound leaves room for a loaded machine
    while len(list(path.glob("ready.*"))) < n - 1:
        assert time.time() < deadline and not errors, errors
        time.sleep(0.01)
    time.sleep(hold)  # the last caller is still away: the others wait for it
    assert not starts and not errors and not (path / "go").exists()
    ts[-1].start()
    for t in ts:
        t.join(timeout=60)
    assert not errors and len(starts) == n, errors
    go = float((path / "go").read_text())
    assert gos == [go] * n and go >= max(arrived) + lead - 0.001  # one start for every caller, lead s after the last arrival
    assert min(starts) >= go and max(starts) - min(starts) < hold  # nobody before it, and closer together than the arrivals were


def test_barrier_one_start_for_callers_that_arrive_together(tmp_path):
    """--barrier: callers that arrive together each see all n and write a start; the first one written is every caller's (several
    write in most rounds)."""
    import threading
    import time

    from ndm_gym.wmbench.speed_pair import barrier

    n = 4

    def caller(path, at_once, out, errors):
        at_once.wait()  # released together: several see n before any start is written
        try:
            out.append((barrier(str(path), n, lead=0.2), time.time()))
        except Exception as e:  # a caller's error fails the test, not only its thread
            errors.append(e)

    for r in range(3):
        path, at_once, out, errors = tmp_path / f"b{r}", threading.Barrier(n, timeout=60), [], []
        ts = [threading.Thread(target=caller, args=(path, at_once, out, errors), daemon=True) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        assert not errors and len(out) == n, errors
        go = float((path / "go").read_text())
        assert all(g == go and s >= go for g, s in out) and not list(path.glob("go.*"))  # one start, nobody before it, no temp left


def test_gpu_driver_override(monkeypatch):
    from ndm_gym.wmbench.speed_pair import gpu_driver

    monkeypatch.setenv("NDM_GPU_DRIVER", "565.57.01")
    assert gpu_driver("GPU-x") == "565.57.01"


def test_replay_engine_export(tmp_path, monkeypatch):
    """replay_engine export: the windows that --replay draws, with every recorded state and the G1's constants an engine needs."""
    import numpy as np

    from ndm_gym.buffer import FIELDS
    from ndm_gym.robots import g1
    from ndm_gym.wmbench import replay_engine
    from ndm_gym.wmbench.collect import G1_RULES, pack
    from ndm_gym.wmbench.speed_pair import replay_windows

    data = _recording(T=40, N=6, ds=FIELDS["s"][0], da=FIELDS["a"][0], dx=FIELDS["xi"][0], seed=5)
    rec, out = tmp_path / "rec.pt", tmp_path / "win.npz"
    torch.save({"meta": {"algo": "supertrack"}, "data": pack(data, G1_RULES)}, rec)
    monkeypatch.setattr(sys, "argv", ["replay_engine", "export", "--recording", str(rec), "--windows", "12", "--len", "5", "--seed", "3",
                                      "--out", str(out)])  # fmt: skip
    replay_engine.main()
    e = np.load(out, allow_pickle=False)
    w = replay_windows(str(rec), 12, 5, seed=3, dev="cpu", full=True)
    for k in ("s0", "a", "xi", "s1", "f", "t0", "env"):
        assert np.array_equal(e[k], w[k].numpy()), k
    assert int(e["seed"]) == 3 and list(e["joint_names"]) == g1.G1_JOINT_NAMES
    assert e["kp"].shape == e["kd"].shape == e["alpha"].shape == e["default_q"].shape == (len(g1.G1_JOINT_NAMES),)
