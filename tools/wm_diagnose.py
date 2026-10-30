"""World-model generalization along real rollouts: k-step open-loop prediction error of one or more checkpoints'
world models on trajectories recorded by ``python -m ndm_gym.evaluate --ckpt ... --record traj.npz``.

For every recorded clip and start step t, each world model rolls out k steps from the simulator state s_t on the
executed setpoints u_t..u_{t+k-1}; the error is measured against s_{t+k} (tracked-body position, mean over bodies;
mean absolute joint angle). Windows are grouped by the clip's outcome: the last ``--pre_fail_s`` seconds before a
failure, the rest of failed clips, and clips that finished.

Usage: python tools/wm_diagnose.py --traj traj_st.npz --ckpt logs/a/ckpt_012000.pt logs/b/ckpt_012000.pt \
           --labels ST-WM SM-WM --out results/wm_diag_st_traj.json [--set_arg st_substeps=1]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ndm_gym.algos import world_model_from_args  # noqa: E402
from ndm_gym.robots import g1  # noqa: E402
from ndm_gym.state import CONTROL_DT, normalize_xi, unflatten_state  # noqa: E402


def parse_set_args(items):
    """``key=value`` pairs (JSON values) added to the checkpoints' training arguments, for checkpoints older than a flag."""
    return {k: json.loads(v) for k, v in (it.split("=", 1) for it in items)}


@torch.no_grad()
def kstep_errors(wm, kin, s, u, xi, starts, k, batch=8192):
    """Body-position (m) and joint (rad) error after k open-loop steps from each start index (T, ...) arrays."""
    eb, eq = [], []
    for i in range(0, len(starts), batch):
        t0 = starts[i : i + batch]
        S = unflatten_state(s[t0])
        for j in range(k):
            S = wm.step(S, u[t0 + j], normalize_xi(xi[t0 + j]))
        St = unflatten_state(s[t0 + k])
        eb.append((kin.forward(*S)[0] - kin.forward(*St)[0]).norm(dim=-1).mean(-1))
        eq.append((S[4] - St[4]).abs().mean(-1))
    return torch.cat(eb), torch.cat(eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pre_fail_s", type=float, default=1.0)
    ap.add_argument("--set_arg", nargs="*", default=[], help="key=value added to checkpoint args that lack the key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.ckpt) == len(a.labels)
    dev = "cuda:0"
    d = dict(np.load(a.traj))
    names = list(d["names"])
    s, u, xi, alive = (torch.tensor(d[n], device=dev) for n in ("s", "u", "xi", "alive"))  # (T, C, ...)
    kin = g1.kinematics(dev)
    extra = parse_set_args(a.set_arg)
    wms = {}
    for path, lab in zip(a.ckpt, a.labels):
        ck = torch.load(path, map_location=dev)
        args = dict(ck["args"])
        for key, val in extra.items():
            assert key not in args, f"{path}: --set_arg {key} would override the checkpoint's own value {args[key]}"
            args[key] = val
        wm = world_model_from_args(args).to(dev)
        wm.load_state_dict(ck["wm"])
        wm.eval()
        wms[lab] = wm
    pre = int(round(a.pre_fail_s / CONTROL_DT))
    res = {"traj": a.traj, "k": a.k, "pre_fail_s": a.pre_fail_s, "clips": {}, "summary": {}}
    groups = {lab: {"pre_fail": [], "failed_other": [], "finished": []} for lab in wms}
    for c, name in enumerate(names):
        T_alive = int(alive[:, c].sum().item())  # steps taken while alive (incl. the terminating one)
        clip_steps = int(d["clip_len"][c]) - 1
        failed = T_alive < clip_steps
        # windows end at the post-step state of the last alive step, which is recorded unless the loop ended there
        last = min(T_alive, s.shape[0] - 1) - a.k
        if last <= 0:
            continue
        starts = torch.arange(0, last + 1, a.stride, device=dev)
        entry = {"failed": failed, "steps": T_alive}
        for lab, wm in wms.items():
            eb, eq = kstep_errors(wm, kin, s[:, c], u[:, c], xi[:, c], starts, a.k)
            entry[lab] = {"body_m": eb.cpu().tolist(), "q_rad": eq.cpu().tolist()}
            end = starts + a.k
            if failed:
                near = end > T_alive - pre
                groups[lab]["pre_fail"] += eb[near].cpu().tolist()
                groups[lab]["failed_other"] += eb[~near].cpu().tolist()
            else:
                groups[lab]["finished"] += eb.cpu().tolist()
        entry["start_steps"] = starts.cpu().tolist()
        res["clips"][name] = entry
    for lab, g in groups.items():
        res["summary"][lab] = {k: {"n": len(v), "mean_mm": 1e3 * float(np.mean(v)) if v else None,
                                   "p90_mm": 1e3 * float(np.percentile(v, 90)) if v else None} for k, v in g.items()}  # fmt: skip
    json.dump(res, open(a.out, "w"))
    print(f"{a.k}-step body-position error (mm) on {os.path.basename(a.traj)}")
    for lab, sm in res["summary"].items():
        print(f"  {lab:12s} " + "  ".join(f"{k}: mean {v['mean_mm']:.1f} p90 {v['p90_mm']:.1f} (n={v['n']})" for k, v in sm.items() if v["n"]))


if __name__ == "__main__":
    main()
