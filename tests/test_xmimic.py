"""XMimic's options of ndm_gym.train (algos.xmimic, docs/algorithms/xmimic.md): the probe worker's GPU (--eprobe_gpu), the training
simulator's constraint rows (--njmax) and the collector's freeze (--gc_freeze), on CPU."""

import argparse
import importlib.util
import os
import subprocess
import types

import pytest

HAVE_SIM_CFG = importlib.util.find_spec("isaaclab") is not None


def _parse(argv):
    from ndm_gym.algos.xmimic import add_xmimic_args

    p = argparse.ArgumentParser()
    p.add_argument("--eprobe_share", type=float, default=0.0)
    # the arguments of ndm_gym.train that the learner options' checks read, with its defaults (--algo: SuperMimic's)
    p.add_argument("--algo", default="supermimic")
    p.add_argument("--wm_backbone", default="mlp")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--pi_encode_once", action="store_true")
    p.add_argument("--pi_rollout_clip", type=float, default=0.0)
    p.add_argument("--pi_cudagraphs", action="store_true")
    p.add_argument("--wm_cudagraphs", action="store_true")
    p.add_argument("--amp", default="bf16")
    p.add_argument("--pi_opt", default="adam")
    p.add_argument("--wm_opt", default="adam")
    p.add_argument("--wm_sample", default="uniform")
    p.add_argument("--wm_fail_windows", type=float, default=0.0)
    add_xmimic_args(p)
    return p.parse_args(argv)


def test_defaults_are_off():
    from ndm_gym.algos.xmimic import check_args

    a = _parse([])
    assert (a.eprobe_gpu, a.njmax, a.gc_freeze) == (-1, 0, False)
    check_args(a)
    check_args(_parse(["--eprobe_share", "0.15", "--eprobe_gpu", "1", "--njmax", "256", "--gc_freeze"]))


@pytest.mark.parametrize("argv", [["--eprobe_gpu", "1"], ["--eprobe_share", "0.15", "--eprobe_gpu", "-2"], ["--njmax", "-1"]])
def test_unsupported_settings_are_refused(argv):
    from ndm_gym.algos.xmimic import check_args

    with pytest.raises(ValueError):
        check_args(_parse(argv))


def test_probe_device_is_an_entry_of_the_trainers_devices():
    from ndm_gym.algos.xmimic import probe_device

    assert probe_device(1, "4,6,7") == "6" and probe_device(0, "4,6,7") == "4" and probe_device(2, " 4, 6 ,7") == "7"
    assert probe_device(3, None) == "3" and probe_device(3, "") == "3"  # CUDA_VISIBLE_DEVICES unset: the device itself
    with pytest.raises(ValueError):
        probe_device(1, "5")


def test_probe_worker_starts_on_the_chosen_gpu(tmp_path, monkeypatch):
    """EvalProbe starts its worker with CUDA_VISIBLE_DEVICES set to the chosen entry, and with the trainer's environment
    (env=None) without --eprobe_gpu."""
    from ndm_gym.algos import eprobe

    started = []

    def popen(cmd, **kw):
        started.append(kw.get("env"))
        return types.SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(eprobe, "probe_tiles", lambda motion, tile: [0, 1, 2])
    monkeypatch.setattr(eprobe, "FailureFocus", lambda *a: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    base = dict(eprobe_tile=300, eprobe_win=10, eprobe_decay=0.7)
    eprobe.EvalProbe(argparse.Namespace(**base, eprobe_gpu=1), None, None, str(tmp_path))
    eprobe.EvalProbe(argparse.Namespace(**base, eprobe_gpu=-1), None, None, str(tmp_path))
    eprobe.EvalProbe(argparse.Namespace(**base), None, None, str(tmp_path))  # a run's arguments from before the option
    assert started[0]["CUDA_VISIBLE_DEVICES"] == "5" and started[0]["TMPDIR"] == os.environ["TMPDIR"]
    assert started[1] is None and started[2] is None


@pytest.mark.skipif(not HAVE_SIM_CFG, reason="the simulator's configuration classes are Isaac Lab's")
def test_njmax_sets_the_training_simulators_rows():
    from ndm_gym.sim.rigid.isaaclab_mjwarp import make_sim

    rows = lambda **kw: make_sim("cpu", None, **kw).physics.solver_cfg.njmax  # noqa: E731
    assert rows() == 500 and rows(paired=True) == 1000  # the scenes' own
    assert rows(njmax=256) == 256 and rows(paired=True, njmax=512) == 512
    assert rows(njmax=None) == 500
    assert make_sim("cpu", None, njmax=256).physics.solver_cfg.nconmax == 128  # the contacts' room is unchanged
