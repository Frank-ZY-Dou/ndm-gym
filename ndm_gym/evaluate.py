"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.
Tracking errors (tracked bodies) are averaged over each clip's steps until it ends or fails, then over clips.
Optional robustness test: horizontal root-velocity pushes at a fixed period, identical for every policy.

Run as a watcher: ``python -m ndm_gym.evaluate --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass

from ndm_gym.sim.launcher import add_launcher_args, launch_simulation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ("mpjpe_g", "mpjpe_l", "e_q", "e_root_pos", "e_root_ori", "e_body_vel", "e_body_angvel", "r_track", "action_rate", "joint_limit")


@dataclass
class PushCfg:
    mag: float  # horizontal root-velocity change (m/s)
    every: int  # control steps between pushes
    seed: int  # seed of the push directions


def track_metrics(kin, motion, S, f) -> dict:
    """Per-env tracking errors of state S against reference frame f (tracked bodies; m, rad, m/s, rad/s)."""
    from ndm_gym.core.rotations import rot_angle_sq

    p, R, v, w, q, qd = S
    bp, bR, bv, bw = kin.forward(p, R, v, w, q, qd)
    rbp, rp = motion.body_p[f], motion.root_p[f]
    return {
        "mpjpe_g": (bp - rbp).norm(dim=-1).mean(-1),
        "mpjpe_l": ((bp - p.unsqueeze(-2)) - (rbp - rp.unsqueeze(-2))).norm(dim=-1).mean(-1),
        "e_q": (q - motion.q[f]).abs().mean(-1),
        "e_root_pos": (p - rp).norm(dim=-1),
        "e_root_ori": rot_angle_sq(R, motion.root_R[f]).sqrt(),
        "e_body_vel": (bv - motion.body_v[f]).norm(dim=-1).mean(-1),
        "e_body_angvel": (bw - motion.body_w[f]).norm(dim=-1).mean(-1),
    }


def _loop_graph(env, kin, motion, mdp, pi, K: int, Hh: int, residual: bool, ref_u, clip_end, ret, steps, err, push, gen, ids, hooks=None):
    """evaluate()'s loop with its step as one CUDA graph (--graph): the same computations, so the same values. Steps 1
    and 2 run eagerly (the second on the capture stream), step 3 is captured, later steps replay it. The step's state
    lives in fixed tensors updated in place (the observation history, the sums, the simulator's counters:
    TrackingSim.step(inplace=True)); pushes and the end test run between the steps as in the eager loop. ``hooks``:
    (before, after) called around each step, ``after`` with the step's outputs (a test's)."""
    import torch

    from ndm_gym.sim.rigid.isaaclab_mjwarp import after_replay, before_replay, capture_step, on_capture_stream
    from ndm_gym.state import CONTROL_DT, unflatten_state
    from ndm_gym.tasks.tracking import MAX_STEP_REWARD

    C, dev = len(motion.names), env.device
    hist = None

    def body():
        nonlocal hist
        S = env.state(env.robot.data._root_link_vel_w.data)  # the root velocity property's buffer, made current before the step
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        if hist is None:
            hist = o.unsqueeze(1).repeat(1, Hh, 1)
        else:
            hist.copy_(torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
        a = pi(hist, motion.goal(env.f, K))
        if residual:
            a = a + ref_u[env.f + 1]
        alive = ~env.done_eval
        out = env.step(a, S, inplace=True)
        ret.add_(torch.where(alive, out["r"], 0.0))
        steps.add_(alive.float())
        m = track_metrics(kin, motion, unflatten_state(out["s1"]), out["f"] + 1)
        m["r_track"] = out["terms"]["track"] / MAX_STEP_REWARD
        m["action_rate"], m["joint_limit"] = out["terms"]["action_rate"], out["terms"]["joint_limit"]
        for k, v in m.items():
            err[k].add_(torch.where(alive, v, 0.0))
        env.f.copy_(torch.minimum(env.f, clip_end - 1))
        return {"S": S, "o": o, "a": a, "alive": alive, "out": out, "m": m}

    graph = outs = None
    t = 0
    while not bool(env.done_eval.all()):
        if push is not None and t > 0 and t % push.every == 0:
            ang = 2 * torch.pi * torch.rand(C, generator=gen, device=dev)
            S = env.state()
            dv = push.mag * torch.stack((torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)), -1)
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2] + dv, S[3]), -1), env_ids=ids)
            env.refresh()
        t += 1
        if hooks is not None:
            hooks[0]()
        env.robot.data.root_link_vel_w  # recomputed if stale, else kept (a velocity a push wrote), exactly as state() decides
        if t <= 2:
            o = body() if t == 1 else on_capture_stream(env.sim, body)
        else:
            before_replay(env.sim)
            first = graph is None
            if first:  # the capture runs the step's Python side once; the replay below does its GPU work
                graph, outs, wp_record = capture_step(env.sim, body)
            graph.replay()
            if not first:
                after_replay(env.sim, env.scene, CONTROL_DT, 1)
            o = outs
        if hooks is not None:
            hooks[1](o)


def evaluate(env, kin, motion, mdp, ck: dict, record: str | None = None, push: PushCfg | None = None, graph: bool = False) -> dict:
    """``graph``: each step as one CUDA graph (_loop_graph; the same values; no ``record``). In the two-robot scene
    (PairedTrackingSim) each instance plays its own side of its env's two-person clip (``env.instance_clip``), the
    results are per instance (named by their clips), and ``pairs`` gives each two-person clip's outcome."""
    import numpy as np
    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.core.rotations import mat_to_quat
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.state import flatten_state, unflatten_state
    from ndm_gym.tasks.tracking import MAX_STEP_REWARD, OBS_DIM

    dev = env.device
    C = len(motion.names)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    residual = ALGOS[a_["algo"]].is_residual(a_)
    ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)  # residual policies output offsets about it
    nets.AMP_DTYPE = torch.bfloat16 if a_["amp"] == "bf16" else None
    pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    clip_of = getattr(env, "instance_clip", None)  # the two-robot scene: each instance's clip; otherwise instance c plays clip c
    if clip_of is not None and (push is not None or graph):
        raise ValueError("the two-robot scene's evaluation has no pushes and no --graph")
    names = motion.names if clip_of is None else [motion.names[i] for i in clip_of.tolist()]
    clip_len = motion.clip_len if clip_of is None else motion.clip_len[clip_of]
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone() if clip_of is None else motion.clip_start[clip_of])
    clip_end = motion.clip_end.clone() if clip_of is None else motion.clip_end[clip_of]
    ret = torch.zeros(C, device=dev)
    steps = torch.zeros(C, device=dev)
    err = {k: torch.zeros(C, device=dev) for k in METRICS}
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": [], "s": [], "u": [], "xi": []}
    gen = torch.Generator(device=dev).manual_seed(push.seed) if push is not None else None
    t = 0
    if graph and record:
        raise ValueError("--graph does not record trajectories")
    with torch.no_grad():
        if graph:  # the whole evaluation, step by step in a CUDA graph; the eager loop below then has nothing left to do
            _loop_graph(env, kin, motion, mdp, pi, K, Hh, residual, ref_u, clip_end, ret, steps, err, push, gen, ids)
        while not bool(env.done_eval.all()):
            if push is not None and t > 0 and t % push.every == 0:
                ang = 2 * torch.pi * torch.rand(C, generator=gen, device=dev)
                S = env.state()
                dv = push.mag * torch.stack((torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)), -1)
                env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2] + dv, S[3]), -1), env_ids=ids)
                env.refresh()
            t += 1
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(env.f, K))
            if residual:
                a = a + ref_u[env.f + 1]
            alive = ~env.done_eval.clone()
            if record:
                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
                # full state and executed setpoint, for world-model diagnostics (tools/wm_diagnose.py)
                traj["s"].append(flatten_state(S).cpu()), traj["u"].append(a.cpu()), traj["xi"].append(env.xi.cpu())
            out = env.step(a, S)  # S: the state just read, nothing written since
            ret += torch.where(alive, out["r"], 0.0)
            steps += alive.float()
            m = track_metrics(kin, motion, unflatten_state(out["s1"]), out["f"] + 1)
            m["r_track"] = out["terms"]["track"] / MAX_STEP_REWARD  # tracking part of the reward, in [0, 1]
            m["action_rate"], m["joint_limit"] = out["terms"]["action_rate"], out["terms"]["joint_limit"]  # unweighted
            for k, v in m.items():
                err[k] += torch.where(alive, v, 0.0)
            env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    err = {k: v / steps for k, v in err.items()}
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        **{k: v.mean().item() for k, v in err.items()},
        "per_clip": {names[i]: round(norm[i].item(), 4) for i in range(C)},
        "per_clip_err": {names[i]: {k: round(v[i].item(), 5) for k, v in err.items()} for i in range(C)},
    }
    if clip_of is not None:  # a two-person clip succeeds when neither robot terminated
        P = C // 2
        both = ~(env.failed_eval[:P] | env.failed_eval[P:])
        res["pair_success"] = both.float().mean().item()
        res["pairs"] = {}
        for i in range(P):
            res["pairs"][motion.meta[int(clip_of[i])]["pair"]] = {
                "success": bool(both[i]),
                "failed": [names[j] for j in (i, P + i) if env.failed_eval[j]],
                "norm_return": round(0.5 * (norm[i] + norm[P + i]).item(), 4),
                "mpjpe_g": round(0.5 * (err["mpjpe_g"][i] + err["mpjpe_g"][P + i]).item(), 5),
            }
    if record:
        starts = motion.clip_start if clip_of is None else motion.clip_start[clip_of]  # per recorded column
        np.savez_compressed(record, names=np.array(names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=starts.cpu().numpy(), clip_len=clip_len.cpu().numpy())  # fmt: skip
    return res


def evaluate_group(env, kin, motion, mdp, cks: list[dict], repeats: int, graph: bool = False) -> list[list[dict]]:
    """``evaluate`` of the checkpoints ``cks``, each ``repeats`` times, at once (--group): the simulator holds
    C * len(cks) * repeats worlds, world (m * repeats + r) * C + c playing clip c under checkpoint m. The evaluation is
    latency-bound at C worlds, so a group costs about one evaluation's wall-clock. The checkpoints share their policies'
    layout and inputs. Returns each checkpoint's ``repeats`` results, as evaluate's (one-robot scene, no pushes)."""
    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.state import unflatten_state
    from ndm_gym.tasks.tracking import MAX_STEP_REWARD, OBS_DIM

    dev, C, M = env.device, len(motion.names), len(cks)
    N, rows = env.N, C * repeats
    if N != C * M * repeats or getattr(env, "instance_clip", None) is not None:
        raise ValueError(f"evaluate_group: a one-robot scene of {C * M * repeats} worlds, not {N}")
    a_ = cks[0]["args"]
    K, Hh, residual = a_["goal_frames"], a_["hist"], ALGOS[a_["algo"]].is_residual(a_)
    for ck in cks:
        b = ck["args"]
        if (b["goal_frames"], b["hist"], b["amp"], ALGOS[b["algo"]].is_residual(b)) != (K, Hh, a_["amp"], residual):
            raise ValueError("evaluate_group: the checkpoints' policies differ in their inputs, precision or residual setpoints")
    ref_u = reference_setpoints(motion, mdp.default_q, env.alpha)
    nets.AMP_DTYPE = torch.bfloat16 if a_["amp"] == "bf16" else None
    pis = []
    for ck in cks:
        pi = TrackingPolicy(OBS_DIM, 73 * K, hist=Hh).to(dev)
        pi.load_state_dict(ck["pi"])
        pis.append(pi.eval())

    def policy(hist, goal):  # each checkpoint's policy on its block of worlds
        return torch.cat([p(hist[m * rows : (m + 1) * rows], goal[m * rows : (m + 1) * rows]) for m, p in enumerate(pis)])

    ids = torch.arange(N, device=dev)
    clip_of = ids % C
    clip_len, clip_end = motion.clip_len[clip_of], motion.clip_end[clip_of]
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[clip_of])
    ret, steps = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    err = {k: torch.zeros(N, device=dev) for k in METRICS}
    hist = None
    with torch.no_grad():
        if graph:  # the whole evaluation, step by step in a CUDA graph; the eager loop below then has nothing left to do
            _loop_graph(env, kin, motion, mdp, policy, K, Hh, residual, ref_u, clip_end, ret, steps, err, None, None, ids)
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = policy(hist, motion.goal(env.f, K))
            if residual:
                a = a + ref_u[env.f + 1]
            alive = ~env.done_eval.clone()
            out = env.step(a, S)
            ret += torch.where(alive, out["r"], 0.0)
            steps += alive.float()
            m = track_metrics(kin, motion, unflatten_state(out["s1"]), out["f"] + 1)
            m["r_track"] = out["terms"]["track"] / MAX_STEP_REWARD
            m["action_rate"], m["joint_limit"] = out["terms"]["action_rate"], out["terms"]["joint_limit"]
            for k, v in m.items():
                err[k] += torch.where(alive, v, 0.0)
            env.f = torch.minimum(env.f, clip_end - 1)
    norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    err = {k: v / steps for k, v in err.items()}
    out = []
    for m in range(M):
        reps = []
        for r in range(repeats):
            sl = slice((m * repeats + r) * C, (m * repeats + r + 1) * C)
            failed = env.failed_eval[sl]
            reps.append({
                "norm_return": norm[sl].mean().item(),
                "norm_return_framew": (ret[sl].sum() / ((clip_len[sl] - 1).sum() * MAX_STEP_REWARD)).item(),
                "n_failed": int(failed.sum().item()),
                "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
                **{k: v[sl].mean().item() for k, v in err.items()},
                "per_clip": {motion.names[i]: round(norm[sl][i].item(), 4) for i in range(C)},
                "per_clip_err": {motion.names[i]: {k: round(v[sl][i].item(), 5) for k, v in err.items()} for i in range(C)},
            })  # fmt: skip
        out.append(reps)
    return out


def mean_of_repeats(reps: list[dict]) -> dict:
    """A checkpoint's result over ``--repeats`` evaluations (the simulation is not bit-reproducible, so borderline clips can
    flip): the means of the returns, failed counts and errors; ``failed`` lists the clips failed in any of them, ``repeats``
    holds the evaluations."""
    out = {k: sum(r[k] for r in reps) / len(reps) for k in ("norm_return", "norm_return_framew", "n_failed", *METRICS)}
    return {**out, "failed": sorted({c for r in reps for c in r["failed"]}), "repeats": reps}


def in_shard(path: str, shard: tuple[int, int, int]) -> bool:
    """--shard I/N: whether checkpoint ``path`` (ckpt_<iteration>.pt) is I modulo N in the run's checkpoint order (iteration
    over the run's --ckpt_every); shard is (I, N, ckpt_every)."""
    i, n, every = shard
    return (int(os.path.basename(path)[5:11]) // every) % n == i


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
    parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
    parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
    parser.add_argument("--motions", type=str, default="lafan1", choices=["lafan1", "interhuman"], help="the clips' source, as train.py's")
    parser.add_argument("--paired", action="store_true", help="the two-robot scene, as train.py's (two-person clips, both sides at once)")
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    parser.add_argument("--push_mag", type=float, nargs="+", default=None,
                        help="--ckpt mode robustness test: one evaluation per horizontal root-velocity push magnitude (m/s)")  # fmt: skip
    parser.add_argument("--push_every", type=int, default=100, help="control steps between pushes (100 = 2 s)")
    parser.add_argument("--push_seed", type=int, default=0, help="seed of the push directions (same pushes for every policy)")
    parser.add_argument("--out_json", type=str, default=None, help="--ckpt mode: also write the full result here")
    parser.add_argument("--repeats", type=int, default=1,
                        help="evaluate every setting (--ckpt) or checkpoint (eval.csv: their means) this many times (simulation is not "
                             "bit-reproducible across runs)")  # fmt: skip
    parser.add_argument("--friction", type=float, default=None,
                        help="--ckpt mode dynamics shift: friction of the feet and the ground instead of 1.0 (training draws 0.3-1.2)")  # fmt: skip
    parser.add_argument("--payload", type=float, default=0.0, help="--ckpt mode dynamics shift: extra torso mass in kg (training: none)")
    parser.add_argument("--graph", action="store_true", help="each step as one CUDA graph (the same values, fewer launches; not with --record)")
    parser.add_argument("--group", type=int, default=1,
                        help="--watch and all-checkpoint modes: evaluate up to this many checkpoints (each --repeats times) at once in "
                             "one simulator, the same wall-clock as one (evaluate_group)")  # fmt: skip
    parser.add_argument("--shard", type=str, default=None,
                        help="--watch and all-checkpoint modes: I/N, only the checkpoints whose index (iteration over the run's "
                             "--ckpt_every) is I modulo N, so that N evaluators on N GPUs share one run's eval.csv")  # fmt: skip
    add_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import interhuman, lafan1
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import PairedTrackingSim, TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    kin = g1.kinematics(dev)
    motion = MotionLib(interhuman.load(args.motion_dir) if args.motions == "interhuman" else lafan1.load_dir(args.motion_dir), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))
    C = len(motion.names)
    assert args.push_mag is None or (args.ckpt and not args.record), "--push_mag needs --ckpt and no --record"
    assert args.repeats == 1 or not args.record, "--repeats: not with --record"
    assert (args.friction is None and not args.payload) or (args.ckpt and not args.record), "--friction / --payload need --ckpt and no --record"
    assert not args.paired or (args.motions == "interhuman" and args.push_mag is None and not args.graph), \
        "--paired: two-person clips, no pushes, no --graph"  # fmt: skip
    assert args.group == 1 or not (args.ckpt or args.paired or args.record), "--group: the --watch or all-checkpoint modes, one robot"
    shard = None
    if args.shard is not None:
        assert not (args.ckpt or args.record), "--shard: the --watch or all-checkpoint modes"
        i, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= i < n, f"--shard I/N needs 0 <= I < N, got {args.shard}"
        every = json.load(open(os.path.join(args.run_dir, "args.json")))["ckpt_every"]
        shard = (i, n, every)

    sim_cfg = make_sim(dev, args, paired=args.paired)
    with launch_simulation(sim_cfg, args):
        n_worlds = C * args.group * args.repeats if args.group > 1 else C
        env = (PairedTrackingSim if args.paired else TrackingSim)(sim_cfg, n_worlds, motion, mdp, eval_mode=True, eval_friction=args.friction,
                                                                  eval_payload=args.payload)  # fmt: skip
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location=dev)
            results = []
            for mag in args.push_mag if args.push_mag is not None else [None]:
                push = PushCfg(mag, args.push_every, args.push_seed) if mag is not None else None
                for rep in range(args.repeats):
                    res = evaluate(env, kin, motion, mdp, ck, args.record, push, graph=args.graph)
                    res["push_mag"], res["repeat"], res["iteration"], res["wall_time"] = mag, rep, ck["iteration"], ck["wall_time"]
                    res["friction"], res["payload"] = env.eval_friction, args.payload
                    print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_clip")}, indent=1), flush=True)
                    results.append(res)
            if args.out_json:
                json.dump(results, open(args.out_json, "w"), indent=1)
            return
        out_csv = os.path.join(args.run_dir, "eval.csv")
        done = set()
        if os.path.exists(out_csv):
            done = {int(ln.split(",")[0]) for ln in open(out_csv).read().strip().splitlines()[1:]}
        elif shard is None:
            open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed," + ",".join(METRICS) + ",failed\n")
        else:  # the first of the shards to start writes the header; rows are appended whole by each
            try:
                with open(out_csv, "x") as f:
                    f.write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed," + ",".join(METRICS) + ",failed\n")
            except FileExistsError:
                pass
        while True:
            ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
            todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
            todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
            if shard is not None:
                todo = [c for c in todo if in_shard(c, shard)]
            groups = [todo[i : i + args.group] for i in range(0, len(todo), args.group)] if args.group > 1 else [[c] for c in todo]
            for paths in groups:
                cks = [torch.load(c, map_location=dev) for c in paths]
                t0 = time.time()
                if args.group > 1:  # a partial group is filled with copies of its last checkpoint, whose results are dropped
                    reps = evaluate_group(env, kin, motion, mdp, cks + cks[-1:] * (args.group - len(cks)), args.repeats, graph=args.graph)
                else:
                    reps = [[evaluate(env, kin, motion, mdp, cks[0], graph=args.graph) for _ in range(args.repeats)]]
                for c, ck, rs in zip(paths, cks, reps):
                    _watch_row(out_csv, c, ck, rs[0] if args.repeats == 1 else mean_of_repeats(rs), C, t0)
                    done.add(int(os.path.basename(c)[5:11]))
            if not args.watch:
                break
            if shard is None or not todo:  # a shard scans again at once after a pass with work
                time.sleep(60)


def _watch_row(out_csv: str, c: str, ck: dict, res: dict, C: int, t0: float):
    """The watcher's eval.csv row, JSON file and log line of checkpoint ``c``."""
    it = int(os.path.basename(c)[5:11])
    line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
            f"{res['norm_return_framew']:.4f},{res['n_failed']}," + ",".join(f"{res[k]:.5f}" for k in METRICS) + f",{'|'.join(res['failed'])}\n")  # fmt: skip
    open(out_csv, "a").write(line)
    json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
    print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}, "
          f"mpjpe_g {1e3 * res['mpjpe_g']:.1f} mm, mpjpe_l {1e3 * res['mpjpe_l']:.1f} mm ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
