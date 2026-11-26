"""The motion check of an animal tracking run: its recorded rollouts beside the reference's same frames, the same for both
algorithms (final.py records the test set's rollouts: evaluate_animal --record).

    python benchmarks/mimic_mjx_animal_tracking/motion.py <run dir> [--rollout final/train/rollout.npz] [--episodes 10] [--out <json>]

The episodes are the first ``--episodes`` worlds of the record (each clip from its first frame, unperturbed) that reached their
clip's end. Per episode, over its steps, for the policy and for the reference's frames of the same steps, of the hinges the task
tracks:

- ``jerk``: the RMS over the steps of each hinge's third difference over dt^3, its mean over the hinges (rad/s^3), and ``acc``
  the same of the second difference (rad/s^2);
- the hinges' velocity power (the first difference over dt, its mean removed) above ``F_HIGH`` Hz and in total, summed over the
  hinges: the run's ``high_share`` is the sum over its episodes above over the sum in total (power over power);
- ``limit``: the share of (step, limited hinge) pairs within 1 % of the hinge's range from a limit, and ``new_limit`` the share at
  which the reference's same frame is not at that limit (low and high apart).

The run's jerk, acc and limits are their means over the episodes. The check (``CHECK``), the same for both algorithms: the jerk
at most 1.5 times the reference's, the share above F_HIGH at most 2 times, the steps at a limit at most the reference's.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hz: the lowest whole frequency above which the reference's joint velocity holds at most 5 % of its power (clips 0-9)
F_HIGH = {"stick": 27, "fly": 73}
LIMIT = 0.01  # within this share of a hinge's range from a limit
CHECK = {"jerk": 1.5, "high_share": 2.0}  # at most these times the reference's; the steps at a limit at most the reference's


def diff_rms(q: np.ndarray, n: int, dt: float) -> float:
    """The RMS over the steps of the n-th difference over dt^n of each column of q (T, J), its mean over the columns."""
    d = np.diff(q, n, axis=0) / dt**n
    return float(np.sqrt((d**2).mean(0)).mean())


def high_power(q: np.ndarray, dt: float, fc: float) -> tuple[float, float]:
    """The velocity power of the columns of q (T, J) above fc and in total (summed over the columns and the frequencies above 0)."""
    v = np.diff(q, axis=0) / dt
    v = v - v.mean(0)
    p = np.abs(np.fft.rfft(v, axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), dt)
    return float(p[f > fc].sum()), float(p[f > 0].sum())


def limit_shares(q: np.ndarray, rq: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float, float]:
    """The shares of (step, hinge) pairs at a limit of the policy's q and the reference's rq (T, J) and the policy's new ones."""
    rng = hi - lo
    pl, ph, rl, rh = q - lo < LIMIT * rng, hi - q < LIMIT * rng, rq - lo < LIMIT * rng, hi - rq < LIMIT * rng
    return float((pl | ph).mean()), float((rl | rh).mean()), float(((pl & ~rl) | (ph & ~rh)).mean())


def measure(record: dict, jq: np.ndarray, limited: np.ndarray, lo: np.ndarray, hi: np.ndarray, dt: float, fc: float, episodes: int) -> dict:
    """The run's motion numbers from its record (``qpos`` (N, T, nq) after each step, ``frame`` (N, T), ``lived``, ``ended``,
    ``ref_qpos``): the policy's and the reference's, their ratios and the check."""
    ref = record["ref_qpos"].astype(np.float64)
    rows = []
    for i in range(min(episodes, len(record["qpos"]))):
        if bool(record["ended"][i]):
            continue
        n = int(record["lived"][i])
        q, rq = record["qpos"][i, :n].astype(np.float64)[:, jq], ref[record["frame"][i, :n]][:, jq]
        row = {}
        for who, x in (("policy", q), ("reference", rq)):
            high, total = high_power(x, dt, fc)
            row[who] = {"acc": diff_rms(x, 2, dt), "jerk": diff_rms(x, 3, dt), "high": high, "total": total}
        row["policy"]["limit"], row["reference"]["limit"], row["policy"]["new_limit"] = limit_shares(q[:, limited], rq[:, limited], lo, hi)
        rows.append(row)
    if not rows:
        raise ValueError("no episode of the record reached its clip's end: nothing to measure")
    out = {"episodes": len(rows), "f_high": fc}
    for who in ("policy", "reference"):
        out[who] = {k: float(np.mean([r[who][k] for r in rows])) for k in ("acc", "jerk", "limit")}
        out[who]["high_share"] = sum(r[who]["high"] for r in rows) / sum(r[who]["total"] for r in rows)
    out["policy"]["new_limit"] = float(np.mean([r["policy"]["new_limit"] for r in rows]))
    p, r = out["policy"], out["reference"]
    out["ratio"] = {k: p[k] / r[k] for k in ("acc", "jerk", "high_share")}
    out["passes"] = bool(out["ratio"]["jerk"] <= CHECK["jerk"] and out["ratio"]["high_share"] <= CHECK["high_share"] and p["limit"] <= r["limit"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a run's directory (its args.json)")
    ap.add_argument("--rollout", default=os.path.join("final", "train", "rollout.npz"), help="the record, relative to the run's directory")
    ap.add_argument("--episodes", type=int, default=10, help="the first worlds of the record: each clip from its first frame")
    ap.add_argument("--data_dir", default=None, help="the bodies and clips (default: the run's)")
    ap.add_argument("--out", default=None, help="the numbers as JSON (default: motion.json beside the record)")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from ndm_gym.train_animal import build, parse_clips

    run = json.load(open(os.path.join(a.run, "args.json")))
    if run["animal"] not in F_HIGH:
        raise SystemExit(f"motion.py: no frequency band for the {run['animal']} (F_HIGH: {', '.join(F_HIGH)})")
    record = dict(np.load(os.path.join(a.run, a.rollout), allow_pickle=True))
    model, _, _, task = build(run["animal"], parse_clips(run["clip_ids"]), a.data_dir or run["data_dir"], "cpu", run["reward_std_scale"],
                              run["action_scale"], run["goal_stride"], ref_qvel=run.get("ref_qvel", "stored"))  # fmt: skip
    if not np.array_equal(record["ref_qpos"], task.ref.qpos.numpy()):
        raise SystemExit("motion.py: the record's reference is not the run's (other clips or data)")
    jq = task.jq.numpy()
    hinge = {int(model.jnt_qposadr[j]): j for j in range(model.njnt) if model.jnt_type[j] == 3}
    joints = [hinge[int(x)] for x in jq]
    limited = np.array([bool(model.jnt_limited[j]) for j in joints])
    lo, hi = (np.array([model.jnt_range[j][k] for j in joints])[limited] for k in (0, 1))
    out = {"run": os.path.abspath(a.run), "animal": run["animal"], "algo": run["algo"], "seed": run["seed"],
           **measure(record, jq, limited, lo, hi, task.ctrl_dt, F_HIGH[run["animal"]], a.episodes)}  # fmt: skip
    path = a.out or os.path.join(a.run, os.path.dirname(a.rollout), "motion.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    p, r, x = out["policy"], out["reference"], out["ratio"]
    print(f"{run['animal']} {run['algo']} seed {run['seed']}: jerk {x['jerk']:.3f} x, share above {out['f_high']} Hz {x['high_share']:.3f} x the "
          f"reference's; steps at a limit {100 * p['limit']:.2f} % (the reference's {100 * r['limit']:.2f} %), new {100 * p['new_limit']:.2f} %; "
          f"{'passes' if out['passes'] else 'fails'} the check ({out['episodes']} episodes)")


if __name__ == "__main__":
    main()
