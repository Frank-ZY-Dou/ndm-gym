"""The guard against non-finite simulator states (ndm_gym.algos.hoi_guard) and HDMI's joint velocity limits on HDMISim,
with the CUDA graph of the control step and without. A NaN joint velocity written into one env flags exactly that env at
its next step; the env is reset through the normal path, which also clears MuJoCo-Warp's warm start of its world, and
is finite at the step after; every other env follows an uninjected run, in which the same env is reset at the same step
(the same random draws). MuJoCo-Warp is not deterministic on CUDA, so the runs are compared by their discrete outcomes
and within the difference of two uninjected runs. A knee written far above its velocity limit is back at the limit after
the substep. Each run is a process of this file (``--worker``); marked slow, needs a GPU and Isaac Lab."""

import argparse
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None
TASK, N, STEPS, K, E = "move_suitcase", 64, 24, 8, 5  # env E gets the NaN before step K


def worker(mode: str, inject: bool, out: str, seed: int, overspeed: float = 0.0, no_limit: bool = False):
    """One run of STEPS control steps with the reference as the action. ``inject``: a NaN joint velocity in env E before
    step K; ``overspeed``: that joint velocity (rad/s) in env E's left knee instead; ``no_limit``: HDMISim without its
    joint velocity limits (the clamps and the links' recomputation patched out)."""
    sys.path.insert(0, REPO)
    import warp as wp
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.algos.hoi_guard import guard_terms, nonfinite
    from ndm_gym.sim.rigid.hdmi_sim import HDMISim, make_hdmi_sim
    from ndm_gym.tasks.hdmi import HDMITask
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.wmbench.collect_hoi import merge_states

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(seed)
    dev, cfg = "cuda:0", TASKS[TASK]
    sim_cfg = make_hdmi_sim(dev)
    if no_limit:
        HDMISim._limit_joint_velocities = HDMISim._link_velocities = lambda self: None
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = HDMISim(sim_cfg, cfg, N, graph=mode == "graph")
        lim = env.vel_limit[env.j_hdmi]  # HDMI order
        task = HDMITask(cfg, N, env.joint_limits, env.robot_mass, noise=False, device=dev)
        guard = torch.compile(nonfinite) if mode == "graph" else nonfinite  # the compiled guard, as the trainers trace it
        solver = env.sim.backend_manager._solver
        q0 = env.q0[env.j_act]
        prev = env.read()
        task.reset(torch.arange(N, device=dev), prev)
        rec = {"bad": [], "failed": [], "done": [], "pos": [], "vel": [], "force": [], "warm": [], "speed": [], "speed_sub": []}
        names = None
        for step in range(STEPS):
            t = env.t.clone()
            a = (env.clip.joint_pos[t.clamp(max=env.L - 1)] - q0) / env.scale  # the reference as the action: no random draws
            if inject and step == K:
                qd = env.robot.data.joint_vel.torch[E : E + 1].clone()
                qd[0, 0] = float("nan")
                env.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=torch.tensor([E], device=dev))
            if overspeed and step == K:
                qd = env.robot.data.joint_vel.torch[E : E + 1].clone()
                qd[0, env.robot.joint_names.index("left_knee_joint")] = overspeed
                env.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=torch.tensor([E], device=dev))
            x, truncated = env.step(a, prev)
            o = task.step(x)
            failed, _ = guard(x, o.obs, a, t)
            names = names or [n for n, _, _ in guard_terms(x, o.obs)]
            bad = failed.any(-1)
            done = o.terminated | bad | truncated
            if not inject and not overspeed and step == K:
                done[E] = True  # the uninjected run resets the same env at the same step
            now = x.now
            cat = lambda *vs: torch.cat([v.reshape(N, -1) for v in vs], -1).cpu()  # noqa: E731
            rec["pos"].append(cat(now.body_pos, now.body_rot, now.joint_pos, now.obj_pos, now.obj_rot, now.eef_pos, now.target_pos))
            rec["vel"].append(cat(now.body_lin_vel, now.body_ang_vel, now.joint_vel))
            rec["force"].append(cat(now.eef_force, x.feet_force))
            rec["speed"].append((now.joint_vel.abs() / lim).amax(-1).cpu())  # the largest joint speed per limit, per env
            rec["speed_sub"].append((x.joint_vel_sub.abs() / lim).amax((1, 2)).cpu())
            rec["bad"].append(bad.cpu()), rec["failed"].append(failed.cpu()), rec["done"].append(done.cpu())
            prev = now
            warm = torch.full((N,), float("nan"))
            if done.any():
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                env.reset(ids)
                env.refresh()
                prev = merge_states(done, prev, env.read())
                task.reset(ids, prev)
                warm = wp.to_torch(solver.mjw_data.qacc_warmstart).abs().amax(-1).cpu()  # each world's largest warm-start entry
            rec["warm"].append(warm)
        torch.save({k: torch.stack(v) for k, v in rec.items()} | {"names": names}, out)


def run(tmp_path, mode: str, inject: bool, name: str, *extra: str) -> dict:
    out = str(tmp_path / f"{name}.pt")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--mode", mode, "--out", out] + (["--inject"] if inject else []) + list(extra)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return torch.load(out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="HDMISim needs a GPU and Isaac Lab")
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_nan_joint_velocity_flags_and_resets_one_env(tmp_path, mode):
    a, a2, b = run(tmp_path, mode, False, "a"), run(tmp_path, mode, False, "a2"), run(tmp_path, mode, True, "b")
    others = torch.arange(N) != E
    # the uninjected runs flag nothing; the injected run flags env E at step K only, by its state and observations
    assert not a["bad"].any() and not a2["bad"].any()
    assert b["bad"][K].nonzero().flatten().tolist() == [E] and not b["bad"][:K].any() and not b["bad"][K + 1 :].any()
    failed = [n for n, f in zip(b["names"], b["failed"][K, E]) if f]
    assert "joint_vel" in failed and "body_pos" in failed, failed
    assert not torch.isfinite(b["pos"][K, E]).all()
    # env E is reset at step K with its world's warm start cleared (the reset's writes do it), finite from the next step on
    assert b["done"][K, E] and all(torch.isfinite(b[g][K + 1 :, E]).all() for g in ("pos", "vel", "force"))
    assert b["warm"][K, E] == 0 and a["warm"][K, E] == 0 and (b["warm"][K, others] > 0).all()
    # every env follows the uninjected run: the same flags and resets throughout; the states of the steps up to two after
    # the injection, before contacts make two uninjected runs drift apart (they reach centimetres by step 20), within 100
    # times the difference of two uninjected runs there; env E after its reset as the uninjected run's env E
    assert torch.equal(b["done"], a["done"]) and torch.equal(a["done"], a2["done"])
    early = slice(0, K + 3)
    for g, floor in (("pos", 1e-4), ("vel", 1e-3), ("force", 1e-2)):
        tol = 100 * (a[g][early] - a2[g][early]).abs().max().item() + floor
        d = (b[g] - a[g]).abs()
        assert d[early][:, others].max() <= tol, (g, d[early][:, others].max(), tol)
        assert d[K + 1 : K + 3, E].max() <= tol, (g, d[K + 1 : K + 3, E].max(), tol)
        assert all(torch.isfinite(r[g][:, others]).all() for r in (a, a2, b))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="HDMISim needs a GPU and Isaac Lab")
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_joint_velocity_limits(tmp_path, mode):
    """HDMI's joint velocity limits hold after every substep, in the step's state and its substep joints, also for a
    knee written to 100 rad/s (its limit is 20); without the clamps the simulator keeps it above the limit."""
    lim, free = run(tmp_path, mode, False, "lim", "--overspeed", "100"), run(tmp_path, mode, False, "free", "--overspeed", "100", "--no_limit")
    assert lim["speed"].max() <= 1 + 1e-6 and lim["speed_sub"].max() <= 1 + 1e-6, (lim["speed"].max(), lim["speed_sub"].max())
    assert free["speed_sub"][K, E] > 1.5, free["speed_sub"][K, E]
    assert not lim["bad"].any() and not lim["done"].any()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", choices=["eager", "graph"], required=True)
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overspeed", type=float, default=0.0)
    ap.add_argument("--no_limit", action="store_true")
    cli = ap.parse_args()
    worker(cli.mode, cli.inject, cli.out, cli.seed, cli.overspeed, cli.no_limit)
