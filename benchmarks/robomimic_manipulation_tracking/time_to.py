"""The training time at which a run reaches the benchmark's target for good: every kept checkpoint of the run (train_robomimic
--ckpt_keep) up to its stop (<run>/stop.json from stop_rule.py, when it exists) evaluated as the last one is (evaluate_robomimic,
256 perturbed rollouts, without the reference's targets); the wall time of the first checkpoint from which every later one has
a success of at least 0.99 with the tracked objects within 10 mm (the hand task's target). The last checkpoint's evaluation is
the run's own (<run>/eval) when it exists. Writes <run>/time_to.json (minutes_to_target: null if never). The times are the
runs' own wall clock, checkpoints every --ckpt_every iterations. A run without kept checkpoints, or with an evaluation in
<run>/eval_ckpts that is not this script's, is an error.

    python benchmarks/robomimic_manipulation_tracking/time_to.py <run directory> [...]
"""

import glob
import json
import os
import subprocess
import sys

SUCCESS, OBJECT_MM = 0.99, 10.0
NUM_ENVS = 256  # evaluate_robomimic's perturbed rollouts per demonstration, as in the last checkpoint's evaluation


def evaluate(ckpt: str, out: str) -> dict:
    path = os.path.join(out, "metrics.json")
    if not os.path.exists(path):
        cmd = [sys.executable, "-m", "ndm_gym.evaluate_robomimic", "--ckpt", ckpt, "--out", out, "--no_reference", "--num_envs", str(NUM_ENVS)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(path))


def check(m: dict, it: int, path: str) -> dict:
    """The metrics of checkpoint ``it`` as this script evaluates it (the perturbed rollouts, the physics trained in)."""
    if m["iteration"] != it or m["num_envs"] != NUM_ENVS:
        sys.exit(f"{path} is not this script's evaluation of checkpoint {it} ({NUM_ENVS} perturbed rollouts): remove it")
    return m


def time_to(run: str, evaluate=evaluate) -> dict:
    """The run's reading (written to <run>/time_to.json): every kept checkpoint's iteration, wall time, success and object
    error, whether it meets the target, and the first from which every later one does (None if none). A run without kept
    checkpoints, or with an evaluation that is not this script's, is an error."""
    own = os.path.join(run, "eval", "metrics.json")
    last = json.load(open(own)) if os.path.exists(own) else None
    stop_path = os.path.join(run, "stop.json")
    stop = json.load(open(stop_path))["stop_iteration"] if os.path.exists(stop_path) else None
    ckpts, rows = sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))), []
    if not ckpts:
        sys.exit(f"{run}: no ckpt_*.pt (train_robomimic --ckpt_keep keeps them)")
    for ckpt in ckpts:
        it = int(os.path.basename(ckpt)[5:11])
        if stop is not None and it > stop:
            continue
        if last and last["iteration"] == it:
            m = check(last, it, own)
        else:
            out = os.path.join(run, "eval_ckpts", f"{it:06d}")
            m = check(evaluate(ckpt, out), it, os.path.join(out, "metrics.json"))
        p = m["policy"]
        ok = p["success_rate"] >= SUCCESS and p["object_mm"] <= OBJECT_MM  # (a NaN error, every rollout terminated: not met)
        rows.append({"iteration": it, "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "object_mm": p["object_mm"],
                     "target": ok})  # fmt: skip
    first = next((rows[i] for i in range(len(rows)) if all(r["target"] for r in rows[i:])), None)
    res = {"minutes_to_target": first and first["minutes"], "iteration_to_target": first and first["iteration"], "checkpoints": rows}
    with open(os.path.join(run, "time_to.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def main():
    for run in sys.argv[1:]:
        res = time_to(run)
        print(run, res["minutes_to_target"], res["iteration_to_target"])


if __name__ == "__main__":
    main()
