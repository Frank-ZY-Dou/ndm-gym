"""Check that state() reflects a reset immediately (no stale data)."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
import torch
from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
from ndm_gym.robots import g1
from ndm_gym.tasks.tracking import MDP
from ndm_gym.motion.library import MotionLib
from ndm_gym.motion.sources import lafan1
from ndm_gym.robots.g1 import default_joint_pos_list
dev = "cuda:0"
kin = g1.kinematics(dev); motion = MotionLib(lafan1.load_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1")), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, 64, motion, mdp, eval_mode=True)
    ids = torch.arange(64, device=dev)
    for _ in range(30):
        env.step(torch.zeros(64, 29, device=dev))
    fr = motion.clip_start[ids % 40] + 200
    env.reset(ids, fr)
    S = env.state(); ref = motion.ref_state(fr)
    print("no update:", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
    env.scene.update(0.0)
    S = env.state()
    print("after update(0):", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
    out = env.step(torch.zeros(64, 29, device=dev))
    print("step s vs ref:", [round((a - b).abs().max().item(), 4) for a, b in zip(env.mdp.__class__.__mro__ and __import__('ndm_gym.state', fromlist=['x']).unflatten_state(out['s']), ref)])
