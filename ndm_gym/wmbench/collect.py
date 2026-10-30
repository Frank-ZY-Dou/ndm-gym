"""Collect a transition dataset with a frozen policy under training conditions (domain randomization, reset noise,
return-based resets, action noise), optionally with horizontal root-velocity pushes, in replay-buffer layout.

The file stores what each transition adds (``pack``): a field that repeats another one shifted in time (the next
state ``s1`` is the next row's ``s`` unless an episode ended or a push changed the state, ``a_prev`` the previous
row's ``a`` unless an episode started, ``xi`` holds and ``f``, ``ep_step`` count within an episode) keeps only the
rows that differ from that, with their positions; ``unpack`` restores every value bit for bit (checked when saving).
``fit.load_buffer`` reads both this and the earlier layout of every field in full.

Run: ``python -m ndm_gym.wmbench.collect --ckpt logs/<run>/ckpt_012000.pt --out data_wm/<name>.pt``
"""

from __future__ import annotations

import argparse
import os

import torch
from isaaclab.app import add_launcher_args, launch_simulation

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# field -> how the other rows predict it: ("next", f): field f one row later; ("prev", f): field f one row earlier;
# ("hold", None): its own previous row; ("count", None): its own previous row plus one
G1_RULES = {"s1": ("next", "s"), "a_prev": ("prev", "a"), "xi": ("hold", None), "f": ("count", None), "ep_step": ("count", None)}


def _predict(data: dict, x: torch.Tensor, rule: tuple) -> tuple[torch.Tensor, int]:
    """The (T, N, ...) prediction of field x and the row it cannot predict (the last for "next", else the first)."""
    kind, src = rule
    if kind == "next":
        s = data[src]
        return torch.cat((s[1:], s[-1:])), x.shape[0] - 1
    if kind == "prev":
        s = data[src]
        return torch.cat((s[:1], s[:-1])), 0
    if kind == "hold":
        return torch.cat((x[:1], x[:-1])), 0
    if kind == "count":
        return torch.cat((x[:1], x[:-1] + 1)), 0
    raise ValueError(f"unknown rule {rule}")


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int32) if x.dtype == torch.float32 else x


def pack(data: dict, rules: dict) -> dict:
    """``data``: name -> (T, N, ...) tensor. The fields of ``rules`` become {"rule", "shape": (T, N), "idx": flat
    positions of the rows their prediction misses (int32), "rows": those rows}; the others stay as they are. The result
    is checked to unpack to ``data`` bit for bit."""
    out = {k: v for k, v in data.items() if k not in rules}
    for k, rule in rules.items():
        x = data[k]
        T, N = x.shape[:2]
        if T * N >= 2**31:
            raise ValueError(f"{k}: {T} x {N} rows do not fit int32 positions")
        pred, row = _predict(data, x, rule)
        miss = _bits(x) != _bits(pred)
        miss = miss.flatten(2).any(-1) if miss.dim() > 2 else miss
        miss[row] = True
        idx = miss.flatten().nonzero().squeeze(-1)
        out[k] = {"rule": rule, "shape": (T, N), "idx": idx.int(), "rows": x.flatten(0, 1)[idx]}
    back = unpack(out, data["s"].device)
    for k, v in data.items():
        if back[k].dtype != v.dtype or not torch.equal(_bits(back[k]), _bits(v)):
            raise RuntimeError(f"packing {k} does not restore it")
    return out


def unpack(data: dict, device) -> dict:
    """The fields of ``pack``'s output (or of a file written before packing, every field in full) on ``device``."""
    out = {k: v.to(device) for k, v in data.items() if not isinstance(v, dict)}
    for k, p in data.items():
        if not isinstance(p, dict):
            continue
        kind, src = p["rule"]
        (T, N), idx, rows = p["shape"], p["idx"].to(device).long(), p["rows"].to(device)
        if kind in ("next", "prev"):
            if src not in out:
                raise KeyError(f"{k} is predicted from {src}, which is not stored in full")
            x, _ = _predict(out, out[src], p["rule"])
            x = x.flatten(0, 1).index_copy(0, idx, rows).view(T, N, *x.shape[2:])
        else:  # "hold" / "count": every row repeats (or counts on from) the last stored row of its env at or before it
            miss = torch.zeros(T * N, dtype=torch.bool, device=device).index_fill_(0, idx, True).view(T, N)
            t = torch.arange(T, device=device).unsqueeze(1).expand(T, N)
            last = torch.where(miss, t, -1).cummax(0).values
            rank = (miss.flatten().cumsum(0) - 1).view(T, N)
            x = rows[rank.gather(0, last)]
            if kind == "count":
                x = x + (t - last).to(x.dtype).view(T, N, *([1] * (x.dim() - 2)))
        out[k] = x
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="policy checkpoint")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400, help="control steps per environment")
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--push_mag", type=float, default=0.0, help="horizontal root-velocity pushes (m/s); 0 disables")
    parser.add_argument("--push_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
    add_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.buffer import ReplayBuffer
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP, OBS_DIM

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    kin = g1.kinematics(dev)
    motion = MotionLib(lafan1.load_dir(args.motion_dir), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    ck = torch.load(args.ckpt, map_location=dev)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    nets.AMP_DTYPE = torch.bfloat16 if a_["amp"] == "bf16" else None
    pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    residual = ALGOS[a_["algo"]].is_residual(a_)

    sim_cfg = make_sim(dev, args)
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = TrackingSim(sim_cfg, args.num_envs, motion, mdp)
        ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
        buffer = ReplayBuffer(args.steps, args.num_envs, dev)
        obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
        ids = torch.arange(args.num_envs, device=dev)
        n_term = 0
        for t in range(args.steps):
            if args.push_mag > 0 and t > 0 and t % args.push_every == 0:
                ang = 2 * torch.pi * torch.rand(args.num_envs, device=dev)
                S = env.state()
                dv = args.push_mag * torch.stack((torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)), -1)
                env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2] + dv, S[3]), -1), env_ids=ids)
                env.refresh()
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            a = pi(obs_hist, motion.goal(env.f, K))
            if residual:
                a = a + ref_u[env.f + 1]
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            out = env.step(a, S)  # S: the state just read, nothing written since
            buffer.add(out)
            n_term = n_term + out["term"].sum()  # on the device: no synchronization per step
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        n_term = int(n_term)
        meta = {"ckpt": args.ckpt, "algo": a_["algo"], "iteration": ck["iteration"], "num_envs": args.num_envs, "steps": args.steps,
                "action_noise": args.action_noise, "push_mag": args.push_mag, "push_every": args.push_every, "seed": args.seed,
                "terminations": n_term}  # fmt: skip
        packed = pack(buffer.data, G1_RULES)
        cpu = lambda v: {**v, "idx": v["idx"].cpu(), "rows": v["rows"].cpu()} if isinstance(v, dict) else v.cpu()  # noqa: E731
        torch.save({"meta": meta, "data": {k: cpu(v) for k, v in packed.items()}}, args.out)
        print(f"saved {args.out}: {args.steps} x {args.num_envs} transitions, {n_term} terminations", flush=True)


if __name__ == "__main__":
    main()
