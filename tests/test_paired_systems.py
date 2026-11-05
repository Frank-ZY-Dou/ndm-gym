"""BlinkMimic's collection graph in the two-robot scene (train.py --paired --collect_graph --reset_masked;
PairedTrackingSim.step_core, step_tail_masked, reset_masked; marked slow: a GPU and Isaac Lab, in processes of this file
or of train.py): the masked training step, eagerly and as one CUDA graph, plays the episodes of the step with its resets
by index; a short paired run with all the systems options captures and replays its graphs."""

import argparse
import csv
import os
import subprocess
import sys

import pytest
import torch
from test_paired_tracking import HAVE_SIM, REPO, write_pairs

SYSTEMS = ["--no_sync", "--fused_adam", "--update_graphs", "--graph_early", "--collect_graph", "--reset_masked", "--ckpt_light"]


def run(*cmd: str):
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, timeout=1800)
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr  # Warp's teardown messages follow it
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    return r


def worker_steps(pairs: str, out: str, mode: str):
    """PairedTrackingSim on the synthetic pairs, 6 envs (12 robots), episodes of at most 5 steps, no reset noise: 24 steps
    with fixed random actions, by step() with its resets by index (``indexed``), by step_core and step_tail_masked with
    reset_masked (``masked``), or by those as one CUDA graph (``graph``: GraphedStep, as train.py --collect_graph
    --reset_masked). Each env's k-th reset goes to its k-th entry of fixed tables of frames (one pair per env, both sides
    at the same time) and joint offsets, so that all runs play the same episodes."""
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import GraphedStep, PairedTrackingSim, make_sim, motion_pairs
    from ndm_gym.tasks.tracking import MDP

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, n, steps = "cuda:0", 12, 24
    kin = g1.kinematics(dev)
    motion = MotionLib(interhuman.load(pairs), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    P = n // 2
    g = torch.Generator().manual_seed(1)
    acts = 0.3 * torch.randn(steps, n, 29, generator=g)
    pc = motion_pairs(motion).cpu()  # (2, pairs)
    k = torch.arange(P) % pc.shape[1]  # env e plays pair k[e]
    start = motion.clip_start.cpu()
    length = (motion.clip_end - motion.clip_start).cpu()[pc[0, k]]
    u = (torch.rand(P, steps + 1, generator=g) * (length - 1).unsqueeze(-1).float()).long()  # >= 1 step before the end
    frames = torch.cat((start[pc[0, k]].unsqueeze(-1) + u, start[pc[1, k]].unsqueeze(-1) + u)).to(dev)  # (2P, resets)
    offsets = (0.02 * torch.rand(P, steps + 1, 29, generator=g) - 0.01).repeat(2, 1, 1).to(dev)
    sim_cfg = make_sim(dev, launcher, paired=True)
    rec = {k_: [] for k_ in ("s", "s1", "r", "term", "done", "f", "ep_step", "a_prev", "xi", "start_bin")}
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = PairedTrackingSim(sim_cfg, n, motion, mdp, max_episode_steps=5, reset_noise=False)
        count, ar = torch.zeros(n, dtype=torch.long, device=dev), torch.arange(n, device=dev)
        reset, reset_masked = env.reset, env.reset_masked

        def reset_(ids, frames_=None):
            reset(ids, frames[ids, count[ids]])
            env.xi[ids, 1:30] = offsets[ids, count[ids]]
            count[ids] += 1

        def reset_masked_(mask, frames_=None):
            reset_masked(mask, frames[ar, count])
            env.xi[:, 1:30] = torch.where(mask.unsqueeze(-1), offsets[ar, count], env.xi[:, 1:30])
            count.add_(mask.long())

        env.reset, env.reset_masked = reset_, reset_masked_
        env.reset(ar)
        env.refresh()
        a = torch.zeros(n, 29, device=dev)

        def masked():
            tr = env.step_tail_masked(env.step_core(a, env.capture_state()))
            return {k_: tr[k_] for k_ in ("s", "s1", "r", "term", "done")}

        graphed = GraphedStep(env, masked, resets=True, name="the masked step")
        for t in range(steps):
            a.copy_(acts[t])
            if mode == "indexed":
                tr = env.step(a, env.state())
            else:
                env.robot.data.root_link_vel_w  # current, as GraphedStep makes it before each call
                tr = graphed() if mode == "graph" else masked()
            for k_ in ("s", "s1", "r", "term", "done"):
                rec[k_].append(tr[k_].cpu())
            for k_ in ("f", "ep_step", "a_prev", "xi", "start_bin"):
                rec[k_].append(getattr(env, k_).cpu())
        rec = {k_: torch.stack(v) for k_, v in rec.items()}
        rec.update(count=count.cpu(), bin_G=env.bin_G.cpu())
    torch.save(rec, out)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the simulator needs a GPU and Isaac Lab")
def test_paired_masked_steps_are_the_indexed_ones(tmp_path):
    """--paired --collect_graph --reset_masked: step_core with step_tail_masked, eagerly and as one CUDA graph, gives the
    transitions, episode bookkeeping and bin statistics of step() with its resets by index, for the same actions and reset
    targets: the same episodes (an env's two robots together), the reset states bit for bit, and the states and rewards
    within the simulator's own run-to-run differences: MuJoCo-Warp is not bitwise reproducible across processes, and two
    indexed runs differ by up to 1.4e-3 in the states over these 24 steps (A100; the robots collide), as the masked and
    graphed runs do from them; the tolerance is about four times that."""
    write_pairs(tmp_path / "pairs", 1.2)
    recs = {}
    for mode in ("indexed", "masked", "graph"):
        out = tmp_path / f"{mode}.pt"
        run(os.path.abspath(__file__), "--worker", mode, "--pairs", str(tmp_path / "pairs"), "--out", str(out))
        recs[mode] = torch.load(out)
    a = recs["indexed"]
    P = a["done"].shape[1] // 2
    assert int(a["count"].min()) >= 3, "too few resets"
    assert torch.equal(a["done"][:, :P], a["done"][:, P:])
    starts = torch.cat((torch.ones_like(a["done"][:1]), a["done"][:-1]))  # the steps from a reset state
    for mode in ("masked", "graph"):
        b = recs[mode]
        for k in ("term", "done", "f", "ep_step", "start_bin", "count", "a_prev", "xi"):
            assert torch.equal(b[k], a[k]), (mode, k)
        assert torch.equal(b["s"][starts], a["s"][starts]), (mode, "reset states")
        for k in ("s", "s1", "r", "bin_G"):
            torch.testing.assert_close(b[k], a[k], rtol=5e-3, atol=5e-3, msg=lambda m, k=k, mode=mode: f"{mode} {k}: {m}")


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the trainer needs a GPU and Isaac Lab")
def test_paired_systems_options_train(tmp_path):
    """A paired SuperTrack run with all the systems options (and --pi_ema): the collection step, the world-model update and
    the policy update are captured and replayed, episodes end and reset inside the collection graph, no update is skipped."""
    write_pairs(tmp_path / "pairs", 1.2)
    log = tmp_path / "run"
    args = ["--algo", "supertrack", "--motions", "interhuman", "--paired", "--motion_dir", str(tmp_path / "pairs"), "--num_envs", "64",
            "--pi_batch", "256", "--wm_batch", "256", "--warmup_steps", "16", "--wm_only_iters", "1", "--log_every", "1", "--ckpt_every", "4",
            "--pi_ema", "0.995", *SYSTEMS, "--iterations", "8", "--log_dir", str(log)]  # fmt: skip
    r = run("-m", "ndm_gym.train", *args)
    for what in ("one collection step", "the world-model update", "the policy update"):
        assert f"captured {what}" in r.stdout, what
    rows = list(csv.DictReader(open(log / "train_log.csv")))
    assert [int(x["iteration"]) for x in rows] == list(range(1, 9)) and all(x["nonfinite"] == "0" for x in rows)
    assert any(x["ep_len"] not in ("nan", "") for x in rows), "no episode ended"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["indexed", "masked", "graph"], required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    worker_steps(cli.pairs, cli.out, cli.worker)
