"""ndm_gym.evaluate's CUDA-graph loop (--graph, _loop_graph) against the eager functions: at every step the observation,
action, reward and tracking metrics of the step are those the eager functions give for the state the step read, and the
accumulated return, step counts and errors are their sums. Runs in a process of this file (``--worker``) on the
synthetic clips of tests/data/supertrack_golden.pt; marked slow, needs a GPU and Isaac Lab. Also the watcher's mean over
--repeats and its --shard split."""

import argparse
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from conftest import REPO, synthetic_clips

HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("isaaclab") is not None
EXACT_KEYS = ("f", "a_prev", "alive")  # bookkeeping: equal


def worker(out: str):
    """6 envs (the synthetic clips twice) under a residual policy with random weights in bf16 autocast, as a bf16 run's
    checkpoint evaluates, with pushes of 0.5 m/s every 7 steps; _loop_graph's hooks recompute every step eagerly."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    import ndm_gym.core.nets as nets
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.evaluate import METRICS, PushCfg, _loop_graph, track_metrics
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.state import unflatten_state
    from ndm_gym.tasks.tracking import MAX_STEP_REWARD, MDP, OBS_DIM

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # as evaluate.main
    nets.AMP_DTYPE = torch.bfloat16  # as evaluate() for a bf16 run
    dev, K, Hh = "cuda:0", 10, 10
    kin = g1.kinematics(dev)
    motion = MotionLib([Clip(**dict(c, name=f"{c['name']}_{i}")) for i in range(2) for c in synthetic_clips(sim=True)], kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    sim_cfg = make_sim(dev, launcher)
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        C = len(motion.names)
        env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True)
        pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev).eval()
        ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
        ids = torch.arange(C, device=dev)
        env.reset(ids, motion.clip_start.clone())  # as evaluate()
        clip_end = motion.clip_end.clone()
        ret, steps = torch.zeros(C, device=dev), torch.zeros(C, device=dev)
        err = {k: torch.zeros(C, device=dev) for k in METRICS}
        push = PushCfg(0.5, 7, 0)
        gen = torch.Generator(device=dev).manual_seed(push.seed)
        prev, eager, diffs = {}, {"hist": None, "ret": torch.zeros_like(ret), "steps": torch.zeros_like(steps)}, []
        eager["err"] = {k: torch.zeros(C, device=dev) for k in METRICS}

        def before():
            prev.update(f=env.f.clone(), a_prev=env.a_prev.clone(), alive=~env.done_eval)

        def after(o):
            f, a_prev, alive = prev["f"], prev["a_prev"], prev["alive"]
            obs = mdp.obs(o["S"], a_prev, env.xi, f)
            h = eager["hist"]
            eager["hist"] = h = obs.unsqueeze(1).repeat(1, Hh, 1) if h is None else torch.cat((h[:, 1:], obs.unsqueeze(1)), 1)
            a = pi(h, motion.goal(f, K)) + ref_u[f + 1]
            tr = o["out"]
            S1 = unflatten_state(tr["s1"])
            r, terms = mdp.reward(S1, f + 1, o["a"], a_prev, terms=True)
            m = track_metrics(kin, motion, S1, f + 1)
            m["r_track"], m["action_rate"], m["joint_limit"] = terms["track"] / MAX_STEP_REWARD, terms["action_rate"], terms["joint_limit"]
            eager["ret"] += torch.where(alive, r, 0.0)
            eager["steps"] += alive.float()
            for k, v in m.items():
                eager["err"][k] += torch.where(alive, v, 0.0)
            d = {"obs": o["o"] - obs, "a": o["a"] - a, "r": tr["r"] - r, **{k: o["m"][k] - v for k, v in m.items()},
                 "f": (tr["f"] != f).float(), "a_prev": (tr["a_prev"] != a_prev).float(), "alive": (o["alive"] != alive).float()}  # fmt: skip
            diffs.append({k: float(v.abs().max()) for k, v in d.items()})

        _loop_graph(env, kin, motion, mdp, pi, K, Hh, True, ref_u, clip_end, ret, steps, err, push, gen, ids, hooks=(before, after))
        sums = {"ret": float((ret - eager["ret"]).abs().max()), "steps": float((steps - eager["steps"]).abs().max()),
                **{k: float((err[k] - eager["err"][k]).abs().max()) for k in METRICS}}  # fmt: skip
        torch.save({"steps": diffs, "sums": sums, "alive_steps": steps.cpu()}, out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_graph_loop_matches_the_eager_functions(tmp_path):
    out = tmp_path / "diffs.pt"
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--out", str(out)], cwd=REPO, capture_output=True, text=True,
                       timeout=1800)  # fmt: skip
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr  # Warp's teardown messages follow it
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    res = torch.load(out)
    assert len(res["steps"]) >= 10 and float(res["alive_steps"].max()) >= 10, (len(res["steps"]), res["alive_steps"])  # replays after step 3
    worst = {k: max(d[k] for d in res["steps"]) for k in res["steps"][0]}
    assert all(worst[k] == 0 for k in EXACT_KEYS), worst
    assert all(v <= 1e-5 for k, v in worst.items() if k not in EXACT_KEYS), worst
    assert all(v <= 1e-4 for v in res["sums"].values()), res["sums"]


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="ndm_gym.evaluate imports Isaac Lab's launcher")
def test_mean_of_repeats():
    """--watch --repeats: a checkpoint's eval.csv row holds the means over its evaluations and the clips failed in any."""
    from ndm_gym.evaluate import METRICS, mean_of_repeats

    reps = [{"norm_return": 0.8, "norm_return_framew": 0.7, "n_failed": 1, "failed": ["b"], **{k: 1.0 for k in METRICS}},
            {"norm_return": 0.9, "norm_return_framew": 0.8, "n_failed": 2, "failed": ["b", "a"], **{k: 3.0 for k in METRICS}}]  # fmt: skip
    m = mean_of_repeats(reps)
    assert m["norm_return"] == pytest.approx(0.85) and m["norm_return_framew"] == pytest.approx(0.75)
    assert m["n_failed"] == 1.5 and m["failed"] == ["a", "b"] and m["repeats"] is reps and all(m[k] == 2.0 for k in METRICS)


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="ndm_gym.evaluate imports Isaac Lab's launcher")
def test_shards_partition_the_checkpoints():
    """--shard I/N: the N shards of a run's checkpoints are disjoint and cover them all."""
    from ndm_gym.evaluate import in_shard

    paths = [f"/run/ckpt_{it:06d}.pt" for it in range(100, 6100, 100)]
    for n in (1, 2, 3):
        parts = [[p for p in paths if in_shard(p, (i, n, 100))] for i in range(n)]
        assert sorted(sum(parts, [])) == paths and all(len(x) >= len(paths) // n for x in parts)
    assert [p[-9:-3] for p in paths if in_shard(p, (1, 2, 100))][:3] == ["000100", "000300", "000500"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", required=True)
    ap.add_argument("--out", required=True)
    worker(ap.parse_args().out)
