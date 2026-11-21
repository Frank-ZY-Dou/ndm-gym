"""The training time at which a run reaches the benchmark's target for good, and whether the run has settled when it ends:
every kept checkpoint of the run (train_robomimic --ckpt_keep) evaluated as the last one is (evaluate_robomimic, 256
perturbed rollouts, without the open-loop rollouts).

- The time to the target is the wall time of the first checkpoint from which every later one has a success of at least
  0.99 with the tracked objects within 10 mm (the manipulation task's target, the hand task's).
- A run has settled if its last five checkpoints agree: their successes lie within 0.05 of one another and their objects'
  errors within 1 mm or a quarter of their mean, whichever is larger (the benchmark's stopping rule: README.md).

The last checkpoint's evaluation is the run's own (<run>/eval) when it exists. Writes <run>/time_to.json
(minutes_to_target: null if never). The times are the runs' own wall clock, checkpoints every --ckpt_every iterations. A
run without kept checkpoints, or with an evaluation in <run>/eval_ckpts that is not this script's, is an error.

    python benchmarks/aloha_bimanual_tracking/time_to.py <run directory> [...]
"""

import glob
import json
import os
import subprocess
import sys

SUCCESS, OBJECT_MM = 0.99, 10.0
LAST, SUCCESS_BAND, OBJECT_BAND_MM, OBJECT_BAND = 5, 0.05, 1.0, 0.25  # the settled test over the last checkpoints
NUM_ENVS = 256  # evaluate_robomimic's perturbed rollouts, as in the last checkpoint's evaluation


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


def settled(rows: list[dict]) -> bool:
    """Whether the last checkpoints agree (the module's docstring); False with fewer than ``LAST`` or an error that is not finite."""
    last = rows[-LAST:]
    success, obj = [r["success_rate"] for r in last], [r["object_mm"] for r in last]
    if len(last) < LAST or any(x != x for x in obj):
        return False
    return max(success) - min(success) <= SUCCESS_BAND and max(obj) - min(obj) <= max(OBJECT_BAND_MM, OBJECT_BAND * sum(obj) / len(obj))


def main():
    for run in sys.argv[1:]:
        own = os.path.join(run, "eval", "metrics.json")
        last = json.load(open(own)) if os.path.exists(own) else None
        ckpts, rows = sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))), []
        if not ckpts:
            sys.exit(f"{run}: no ckpt_*.pt (train_robomimic --ckpt_keep keeps them)")
        for ckpt in ckpts:
            it = int(os.path.basename(ckpt)[5:11])
            if last and last["iteration"] == it:
                m = check(last, it, own)
            else:
                out = os.path.join(run, "eval_ckpts", f"{it:06d}")
                m = check(evaluate(ckpt, out), it, os.path.join(out, "metrics.json"))
            p = m["policy"]
            ok = p["success_rate"] >= SUCCESS and p["object_mm"] <= OBJECT_MM
            rows.append({"iteration": it, "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "stage_rates": p["stage_rates"],
                         "object_mm": p["object_mm"], "terminated": p["terminated"], "target": ok})  # fmt: skip
        first = next((rows[i] for i in range(len(rows)) if all(r["target"] for r in rows[i:])), None)
        res = {"minutes_to_target": first and first["minutes"], "iteration_to_target": first and first["iteration"], "settled": settled(rows),
               "checkpoints": rows}  # fmt: skip
        with open(os.path.join(run, "time_to.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(run, res["minutes_to_target"], res["iteration_to_target"], "settled" if res["settled"] else "not settled")


if __name__ == "__main__":
    main()
