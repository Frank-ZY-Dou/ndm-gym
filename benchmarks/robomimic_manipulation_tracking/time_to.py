"""The training time at which a run reaches the benchmark's target for good: every kept checkpoint of the run (train_robomimic
--ckpt_keep) evaluated as the last one is (evaluate_robomimic, 256 perturbed rollouts, without the reference's targets); the
wall time of the first checkpoint from which every later one has a success of at least 0.99 with the tracked objects within
10 mm (the hand task's target). The last checkpoint's evaluation is the run's own (<run>/eval) when it exists. Writes
<run>/time_to.json (minutes_to_target: null if never). The times are the runs' own wall clock, checkpoints every
--ckpt_every iterations.

    python benchmarks/robomimic_manipulation_tracking/time_to.py <run directory> [...]
"""

import glob
import json
import os
import subprocess
import sys

SUCCESS, OBJECT_MM = 0.99, 10.0


def evaluate(ckpt: str, out: str) -> dict:
    path = os.path.join(out, "metrics.json")
    if not os.path.exists(path):
        cmd = [sys.executable, "-m", "ndm_gym.evaluate_robomimic", "--ckpt", ckpt, "--out", out, "--no_reference"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(path))


def main():
    for run in sys.argv[1:]:
        own = os.path.join(run, "eval", "metrics.json")
        last = json.load(open(own)) if os.path.exists(own) else None
        rows = []
        for ckpt in sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))):
            it = int(os.path.basename(ckpt)[5:11])
            m = last if last and last["iteration"] == it else evaluate(ckpt, os.path.join(run, "eval_ckpts", f"{it:06d}"))
            p = m["policy"]
            ok = p["success_rate"] >= SUCCESS and p["object_mm"] <= OBJECT_MM
            rows.append({"iteration": it, "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "object_mm": p["object_mm"], "target": ok})
        first = next((rows[i] for i in range(len(rows)) if all(r["target"] for r in rows[i:])), None)
        res = {"minutes_to_target": first and first["minutes"], "iteration_to_target": first and first["iteration"], "checkpoints": rows}
        with open(os.path.join(run, "time_to.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(run, res["minutes_to_target"], res["iteration_to_target"])


if __name__ == "__main__":
    main()
