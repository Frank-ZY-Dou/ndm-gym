"""Two-person clips and the two-robot scene (the two-G1 boxing example): the converted format's loader
(``ndm_gym.motion.sources.interhuman``), the pairing of a motion library's clips and the pairs' reset bins
(``motion_pairs``, ``pair_bins``), PairedTrackingSim's paired start frames and resets; and, in a process of this file
(``--worker``, marked slow, GPU and Isaac Lab), the scene itself: an env's two robots share their episodes, and they
collide. The pairs are made of the synthetic clips of tests/data/supertrack_golden.pt, the second side turned to face the
first."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import DEV, REPO, synthetic_clips

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
HAVE_SIM = torch.cuda.is_available() and HAVE_ISAACLAB


def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)  # fmt: skip


def write_pairs(d, gap: float, names=None, permute: bool = False) -> list[str]:
    """One two-person .npz per synthetic clip (sim=True): side 1 the clip, side 2 the clip turned by 180 degrees about
    the vertical through side 1's first root position and moved ``gap`` m along side 1's first heading, so that the two
    face each other ``gap`` m apart. ``permute``: the joints stored in reversed order (dof_names says so)."""
    from ndm_gym.robots.g1 import G1_JOINT_NAMES

    os.makedirs(d, exist_ok=True)
    out = []
    for c in synthetic_clips(sim=True):
        p, q, j = c["root_p"], c["root_quat"], c["joints"]
        w, x, y, z = q[0]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        head = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros(())))
        c0 = p[0] * torch.tensor([1.0, 1.0, 0.0])
        rz = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        p2 = (p - c0) @ rz.T + c0 + gap * head
        q2 = _qmul(torch.tensor([0.0, 0.0, 0.0, 1.0]).expand_as(q), q)
        order = list(range(29))[::-1] if permute else list(range(29))
        name = (names or {}).get(c["name"], c["name"])
        path = os.path.join(d, f"{name}.npz")
        rec = {"source": "synthetic", "fps": float(c["fps"]), "scale": 1.0, "dof_names": [G1_JOINT_NAMES[i] for i in order]}
        np.savez(path, root_p=torch.stack((p, p2), 1).numpy(), root_quat=torch.stack((q, q2), 1).numpy(),
                 dof=torch.stack((j, j), 1)[..., order].numpy(), dof_names=np.array([G1_JOINT_NAMES[i] for i in order]),
                 fps=np.array(float(c["fps"])), record=np.array(json.dumps(rec)))  # fmt: skip
        out.append(path)
    return out


def test_load_pair(tmp_path):
    """A two-person file gives its two sides as two clips (p1, p2), joints in the G1's order whatever the file's, with
    the pair and side in their meta; a directory gives every file's, sorted by name."""
    from ndm_gym.motion.sources import interhuman

    paths = write_pairs(tmp_path, 1.0, permute=True)
    clips = interhuman.load(paths[0])
    src = synthetic_clips(sim=True)[0]
    pair = os.path.splitext(os.path.basename(paths[0]))[0]
    assert [c.name for c in clips] == [f"{pair}_p1", f"{pair}_p2"]
    assert [(c.meta["pair"], c.meta["side"]) for c in clips] == [(pair, 0), (pair, 1)]
    for c in clips:
        assert torch.equal(c.joints, src["joints"]) and c.fps == float(src["fps"])
    assert torch.equal(clips[0].root_p, src["root_p"]) and torch.equal(clips[0].root_quat, src["root_quat"])
    assert len(interhuman.load(str(tmp_path))) == 2 * len(paths)


def test_load_errors(tmp_path):
    from ndm_gym.motion.sources import interhuman

    path = write_pairs(tmp_path, 1.0)[0]
    d = dict(np.load(path))
    np.savez(tmp_path / "extra.npz", **d, extra=np.zeros(1))
    with pytest.raises(ValueError, match="arrays"):
        interhuman.load(str(tmp_path / "extra.npz"))
    np.savez(tmp_path / "names.npz", **{**d, "dof_names": np.array(["a"] * 29)})
    with pytest.raises(ValueError, match="joints"):
        interhuman.load(str(tmp_path / "names.npz"))
    np.savez(tmp_path / "short.npz", **{**d, "dof": d["dof"][:-1]})
    with pytest.raises(ValueError, match="dof has shape"):
        interhuman.load(str(tmp_path / "short.npz"))
    with pytest.raises(ValueError, match="no two-person clips"):
        interhuman.load(str(tmp_path / "empty"))


def _pair_motion(kin, tmp_path):
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman

    write_pairs(tmp_path, 1.0)
    return MotionLib(interhuman.load(str(tmp_path)), kin, DEV)


@pytest.mark.skipif(not HAVE_ISAACLAB, reason="the scene's module needs Isaac Lab")
def test_motion_pairs_and_bins(kin, tmp_path):
    """motion_pairs: each pair's two clips, in the order of their first sides; a lone side or sides of different
    lengths are errors. pair_bins: every first side's bin with the second side's at the same time."""
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip
    from ndm_gym.sim.rigid.isaaclab_mjwarp import motion_pairs, pair_bins

    m = _pair_motion(kin, tmp_path)
    ids = motion_pairs(m)
    assert ids.tolist() == [[0, 2, 4], [1, 3, 5]]
    a, b, off = pair_bins(m, ids)
    assert torch.equal(m.bin_clip[a], ids[0].repeat_interleave(len(a) // 3)) and torch.equal(m.bin_clip[b], ids[1].repeat_interleave(len(a) // 3))
    assert torch.equal(m.bin_start[b] - m.bin_start[a], off)
    c = synthetic_clips(sim=True)[0]
    lone = MotionLib([Clip(**{**c, "meta": {"pair": "x", "side": 0}})], kin, DEV)
    with pytest.raises(ValueError, match="both sides"):
        motion_pairs(lone)
    short = {**c, "root_p": c["root_p"][:-10], "root_quat": c["root_quat"][:-10], "joints": c["joints"][:-10]}
    uneven = MotionLib([Clip(**{**c, "meta": {"pair": "x", "side": 0}}), Clip(**{**short, "name": "y", "meta": {"pair": "x", "side": 1}})], kin, DEV)
    with pytest.raises(ValueError, match="different lengths"):
        motion_pairs(uneven)
    with pytest.raises(ValueError, match="not one side"):
        motion_pairs(MotionLib([Clip(**c)], kin, DEV))


@pytest.mark.skipif(not HAVE_ISAACLAB, reason="the scene's module needs Isaac Lab")
def test_paired_start_frames_and_resets(kin, tmp_path):
    """PairedTrackingSim.sample_frames: a pair starts both sides at the same time of its two clips, before their ends,
    in the bins of those frames; reset refuses to start one robot of an env without the other."""
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, motion_pairs, pair_bins

    m = _pair_motion(kin, tmp_path)
    ids = motion_pairs(m)
    a, b, off = pair_bins(m, ids)
    sim = SimpleNamespace(motion=m, pbin_a=a, pbin_b=b, pbin_off=off, bin_G=torch.rand(m.num_bins), G_max=1.0,
                          bin_w_avg=torch.full((len(a),), 1.0 / len(a)), device=DEV, P=2)  # fmt: skip
    torch.manual_seed(0)
    fa, fb, ba, bb = PairedTrackingSim.sample_frames(sim, 3000)
    ca, cb = m.clip_id[fa], m.clip_id[fb]
    assert bool((ids[1][torch.searchsorted(ids[0], ca)] == cb).all())
    assert torch.equal(fa - m.clip_start[ca], fb - m.clip_start[cb])
    assert bool((fa < m.clip_end[ca]).all()) and torch.equal(m.frame_bin[fa], ba) and torch.equal(m.frame_bin[fb], bb)
    assert set(ca.tolist()) == set(ids[0].tolist())
    with pytest.raises(ValueError, match="reset together"):
        PairedTrackingSim.reset(sim, torch.tensor([0, 1, 2]))


def run(*cmd: str):
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, timeout=1800)
    err = r.stderr[r.stderr.find("Traceback") :] if "Traceback" in r.stderr else r.stderr
    assert r.returncode == 0, r.stdout[-2000:] + err[:4000]
    return r


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the scene needs a GPU and Isaac Lab")
def test_paired_scene_shares_episodes(tmp_path):
    """Training mode, 3 envs (one per synthetic pair), episodes of at most 6 steps, random actions: an env's two robots
    always share their frame (at the clips' offset) and episode step, and their episodes end together."""
    out = tmp_path / "steps.pt"
    write_pairs(tmp_path / "pairs", 1.2)
    run(os.path.abspath(__file__), "--worker", "steps", "--pairs", str(tmp_path / "pairs"), "--out", str(out))
    rec = torch.load(out)
    P = rec["f"].shape[1] // 2
    assert rec["done"].any(), "no episode ended: the test needs resets"
    assert torch.equal(rec["done"][:, :P], rec["done"][:, P:]) and torch.equal(rec["ep_step"][:, :P], rec["ep_step"][:, P:])
    # the second robot plays the partner clip (the files' order: clip 2k's partner is 2k + 1) at the same time
    assert torch.equal(rec["clip"][:, P:], rec["clip"][:, :P] + 1) and torch.equal(rec["t"][:, P:], rec["t"][:, :P])
    assert torch.isfinite(rec["s1"]).all()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SIM, reason="the scene needs a GPU and Isaac Lab")
def test_paired_robots_collide(tmp_path):
    """Two pairs of the same clip, face to face, one 0.2 m apart (their bodies overlapping) and one 1.2 m apart, each in
    its own env, start with the reference's velocities and hold their first poses: the near pair is pushed apart by its
    contacts, more than 10 cm beyond what the far pair's distance does in the same time."""
    out = tmp_path / "dist.json"
    write_pairs(tmp_path / "near", 0.2, names={c["name"]: "near" for c in synthetic_clips(sim=True)[:1]})
    for f in os.listdir(tmp_path / "near"):
        if not f.startswith("near"):
            os.remove(tmp_path / "near" / f)
    write_pairs(tmp_path / "far", 1.2, names={c["name"]: "far" for c in synthetic_clips(sim=True)[:1]})
    for f in os.listdir(tmp_path / "far"):
        if not f.startswith("far"):
            os.remove(tmp_path / "far" / f)
    run(os.path.abspath(__file__), "--worker", "contact", "--pairs", str(tmp_path / "near"), str(tmp_path / "far"), "--out", str(out))
    d = json.load(open(out))
    assert (d["near"][1] - d["near"][0]) - (d["far"][1] - d["far"][0]) > 0.1, d


def worker(kind: str, pairs: list[str], out: str):
    sys.path.insert(0, REPO)
    from isaaclab.app import add_launcher_args, launch_simulation

    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev = "cuda:0"
    kin = g1.kinematics(dev)
    sim_cfg = make_sim(dev, launcher, paired=True)
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        if kind == "steps":
            motion = MotionLib(interhuman.load(pairs[0]), kin, dev)
            mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
            env = PairedTrackingSim(sim_cfg, 6, motion, mdp, max_episode_steps=6)
            rec = {"f": [], "clip": [], "t": [], "ep_step": [], "done": [], "s1": []}
            for _ in range(20):
                c = motion.clip_id[env.f]
                rec["f"].append(env.f.cpu()), rec["clip"].append(c.cpu()), rec["t"].append((env.f - motion.clip_start[c]).cpu())
                rec["ep_step"].append(env.ep_step.cpu())
                tr = env.step(0.1 * torch.randn(6, 29, device=dev), env.state())
                rec["done"].append(tr["done"].cpu()), rec["s1"].append(tr["s1"].cpu())
            torch.save({k: torch.stack(v) for k, v in rec.items()}, out)
            return
        # contact: each pair in its own scene (one env), evaluation mode, both robots holding their reference poses
        motion = MotionLib([c for d in pairs for c in interhuman.load(d)], kin, dev)
        mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
        env = PairedTrackingSim(sim_cfg, len(motion.names), motion, mdp, eval_mode=True)
        ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
        clip_of = env.instance_clip
        env.reset(torch.arange(env.N, device=dev), motion.clip_start[clip_of])
        env.refresh()
        P, res = env.P, {}
        S = env.state()
        d0 = (S[0][P:, :2] - S[0][:P, :2]).norm(dim=-1)
        for _ in range(10):
            env.step(ref_u[env.f + 1], env.state())
            env.f = motion.clip_start[clip_of]  # hold the first frame's pose
        S = env.state()
        d1 = (S[0][P:, :2] - S[0][:P, :2]).norm(dim=-1)
        for i in range(P):
            res[motion.meta[int(clip_of[i])]["pair"]] = [float(d0[i]), float(d1[i])]
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["steps", "contact"], required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    worker(cli.worker, cli.pairs, cli.out)
