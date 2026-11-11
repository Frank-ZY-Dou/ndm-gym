"""The library without the simulator, as a ``pip install`` of this repository installs it: every module imports except the simulator
and the modules that import it at the top, which raise ``ndm_gym.sim.SimulatorNotInstalled`` (or name the simulator's
missing package), as do the commands that run the simulator; the world models run forward and backward on CPU.

Each check runs in a subprocess in which the simulator's packages cannot be imported (None in ``sys.modules``, which
``importlib.util.find_spec`` also reports as missing), so the tests hold on a machine where they are installed too."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import REPO

BLOCKED = ("isaaclab", "isaaclab_newton", "isaaclab_physx", "isaaclab_ov", "isaaclab_visualizers", "isaacsim", "omni", "carb", "pxr", "newton",
           "warp", "mujoco_warp")  # fmt: skip
# the modules that need the simulator on import: the simulator itself and the modules that import it at the top. A module
# that is not a simulator imports the simulator inside the functions that run it, and Isaac Lab's launcher from
# ndm_gym.sim.launcher, so that the library imports without the simulator.
NEEDS_SIMULATOR = {
    "ndm_gym.sim.rigid.isaaclab_mjwarp",
    "ndm_gym.sim.rigid.hdmi_sim",
    "ndm_gym.sim.rigid.muscle_sim",  # Newton and MuJoCo-Warp directly
    "ndm_gym.sim.rigid.hand_sim",  # MuJoCo-Warp directly
    "ndm_gym.algos.hoi_algos",  # ndm_constants from sim.rigid.hdmi_sim
    "ndm_gym.train_hoi",  # isaaclab.app
}
NEEDS_TRITON = {"ndm_gym.core.token_attention", "ndm_gym.kernels.muscle_geometry", "ndm_gym.kernels.rf_block", "ndm_gym.kernels.rf_train",
                "ndm_gym.worldmodels.g1_fused", "ndm_gym.worldmodels.hoi_fused"}  # fmt: skip  # GPU kernels, imported where a flag asks for them


def run_blocked(code: str, tmp_path) -> dict:
    """Run ``code`` in a fresh interpreter without the simulator; it prints one JSON line, returned here."""
    prelude = f"import sys\nfor _m in {BLOCKED!r}:\n    sys.modules[_m] = None\n"
    env = {**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_modules_import_without_simulator(tmp_path):
    """Every module of the package imports without the simulator, except those that need it, which fail with
    SimulatorNotInstalled or by naming one of its packages, and those that need Triton (without it)."""
    out = run_blocked(
        """
        import importlib, json, pkgutil, sys
        import ndm_gym
        from ndm_gym.sim import SimulatorNotInstalled

        res = {"version": ndm_gym.__version__}
        for m in pkgutil.walk_packages(ndm_gym.__path__, "ndm_gym."):
            try:
                importlib.import_module(m.name)
                res[m.name] = "ok"
            except SimulatorNotInstalled as e:
                res[m.name] = "simulator: " + str(e)
            except ImportError as e:
                res[m.name] = f"missing {e.name}: {e}"
        sim = ("isaaclab", "isaacsim", "newton", "warp", "mujoco_warp")
        res["loaded"] = sorted(n for n in sys.modules if n.split(".")[0] in sim and sys.modules[n])
        print(json.dumps(res))
        """,
        tmp_path,
    )
    version = out.pop("version")
    assert version.count(".") == 2, version
    assert out.pop("loaded") == []
    have_triton = subprocess.run([sys.executable, "-c", "import triton"], capture_output=True).returncode == 0
    failed = {m: why for m, why in out.items() if why != "ok"}
    unexpected = {}
    for m, why in failed.items():
        sim = why.startswith("simulator: ") or any(why.startswith(f"missing {b}:") or why.startswith(f"missing {b}.") for b in BLOCKED)
        if sim and m in NEEDS_SIMULATOR:
            assert why.startswith("missing") or "scripts/setup.sh" in why, why
        elif not (m in NEEDS_TRITON and not have_triton and why.startswith("missing triton")):
            unexpected[m] = why
    assert not unexpected, f"modules that do not import without the simulator: {unexpected}"
    assert NEEDS_SIMULATOR <= set(failed), f"these import without the simulator now; take them off NEEDS_SIMULATOR: {NEEDS_SIMULATOR - set(failed)}"
    for m in ("ndm_gym.worldmodels.rigidformer", "ndm_gym.algos", "ndm_gym.wmbench.fit", "ndm_gym.train", "ndm_gym.evaluate", "ndm_gym.sim.launcher"):
        assert out[m] == "ok", (m, out[m])


def test_commands_need_the_simulator(tmp_path):
    """The commands that run the simulator raise SimulatorNotInstalled, which names scripts/setup.sh, when they build
    their arguments; the offline world-model fit (no simulator) parses its arguments."""
    out = run_blocked(
        """
        import json, runpy, sys
        from ndm_gym.sim import SimulatorNotInstalled

        res = {}
        for prog, mod, argv in (("ndm-train", "ndm_gym.train", ["--algo", "supertrack"]), ("ndm-eval", "ndm_gym.evaluate", ["--run_dir", "logs/x"]),
                                ("collect", "ndm_gym.wmbench.collect", ["--ckpt", "c.pt", "--out", "d.pt"])):
            sys.argv = [prog, *argv]
            try:
                runpy.run_module(mod, run_name="__main__")
                res[mod] = "ran"
            except SimulatorNotInstalled as e:
                res[mod] = str(e)
        from ndm_gym.sim.launcher import launch_simulation
        try:
            launch_simulation(None, None)
        except SimulatorNotInstalled as e:
            res["launch_simulation"] = str(e)
        from ndm_gym.wmbench import fit

        sys.argv = ["fit", "--algo", "supermimic", "--train", "a.pt", "--test", "t=b.pt", "--out", "x.csv"]
        res["fit"] = vars(fit.parse_args())["algo"]
        print(json.dumps(res))
        """,
        tmp_path,
    )
    assert out.pop("fit") == "supermimic"
    for what, msg in out.items():
        assert "needs Isaac Lab 3" in msg and "run scripts/setup.sh" in msg and "isaaclab" in msg, (what, msg)
    assert out["ndm_gym.train"].startswith("ndm-train needs"), out["ndm_gym.train"]


def world_model_steps(runs: list[list[str]], tmp_path) -> dict:
    """For each run's training arguments, the world model training builds (``world_model_from_args``), its class and a
    step forward and backward on CPU (random states and actions): output shapes, whether the gradients are finite and
    reach the actions. The zero-initialized last layers get small random weights first, which would stop that gradient."""
    out = run_blocked(
        f"""
        import argparse, json
        import torch
        from ndm_gym.algos import world_model_from_args
        from ndm_gym.algos.blinkmimic import add_blinkmimic_args
        from ndm_gym.core.rotations import so3_exp
        from ndm_gym.robots.g1 import NUM_J
        from ndm_gym.state import XI_DIM
        from ndm_gym.train import add_speed_args, add_train_args

        parser = argparse.ArgumentParser()
        add_train_args(parser)
        add_speed_args(parser)
        add_blinkmimic_args(parser)
        torch.manual_seed(0)
        res = {{}}
        for argv in {runs!r}:
            wm = world_model_from_args(vars(parser.parse_args(argv)))
            for p in wm.parameters():
                if not p.detach().any():
                    p.data.normal_(0.0, 0.01)
            N, J = 8, NUM_J
            S = (torch.randn(N, 3), so3_exp(torch.randn(N, 3)), torch.randn(N, 3), torch.randn(N, 3), 0.3 * torch.randn(N, J), torch.randn(N, J))
            act = torch.randn(N, J, requires_grad=True)
            S1 = wm.step(S, act, torch.randn(N, XI_DIM))
            (sum(x.square().mean() for x in S1) + wm.targets(S, S1).square().mean()).backward()
            grads = [p.grad for p in wm.parameters() if p.grad is not None]
            res[" ".join(argv)] = dict(cls=type(wm).__name__, shapes=[list(x.shape) for x in S1], n_grads=len(grads),
                                       act_grad=float(act.grad.abs().sum()), finite=all(bool(torch.isfinite(g).all()) for g in grads + [act.grad]))
        print(json.dumps(res))
        """,
        tmp_path,
    )
    for argv, r in out.items():
        assert r["shapes"] == [[8, 3], [8, 3, 3], [8, 3], [8, 3], [8, 29], [8, 29]], (argv, r)
        assert r["finite"] and r["act_grad"] > 0 and r["n_grads"] > 0, (argv, r)
    return {argv: r["cls"] for argv, r in out.items()}


def test_mlp_world_models_without_simulator(tmp_path):
    """SuperTrack's and SuperMimic's MLP world models, and BlinkMimic's semi-implicit one."""
    runs = [["--algo", "supertrack"], ["--algo", "supermimic"], ["--algo", "supermimic", "--wm_structured"]]
    assert world_model_steps(runs, tmp_path) == {"--algo supertrack": "AccelWorldModel", "--algo supermimic": "IncrementWorldModel",
                                                 "--algo supermimic --wm_structured": "SemiImplicitIncrementWorldModel"}  # fmt: skip


def test_rigidformer_world_models_without_simulator(tmp_path):
    """SuperTrack-RF's and SuperMimic-RF's world models (RigidFormer backbone), which need the G1's MJCF for its
    kinematics (data/, tools/download_data.sh)."""
    from ndm_gym.robots import g1

    if not os.path.exists(g1.MJCF_PATH):
        pytest.skip(f"the RigidFormer needs the G1's MJCF {g1.MJCF_PATH} (tools/download_data.sh)")
    runs = [["--algo", a, "--wm_backbone", "rigidformer"] for a in ("supertrack", "supermimic")]
    assert world_model_steps(runs, tmp_path) == {"--algo supertrack --wm_backbone rigidformer": "AccelWorldModel",
                                                 "--algo supermimic --wm_backbone rigidformer": "IncrementWorldModel"}  # fmt: skip
