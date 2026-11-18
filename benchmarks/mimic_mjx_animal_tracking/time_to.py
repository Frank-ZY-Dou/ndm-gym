"""The training time at which a run reaches the benchmark's target for good: every kept checkpoint of the run (train_animal
--ckpt_keep) evaluated as its last one is (evaluate_animal on the run's clips, four perturbed repeats); the wall time of the
first checkpoint from which every later one ends at most 2 % of its episodes with the bodies' error within 3 % of the body's
size (the task's: the bodies' median largest distance from the root; stick insect 0.53 mm, fruit fly 0.079 mm, rat 4.7 mm).
Writes <run>/time_to.json (minutes_to_target: null if never). The times are the runs' own wall clock, checkpoints every
--ckpt_every iterations.

    python benchmarks/mimic_mjx_animal_tracking/time_to.py <run directory> [...]
"""

import glob
import json
import os
import subprocess
import sys

SIZE_MM = {"stick": 17.58, "fly": 2.623, "rat": 156.6}  # the tasks' sizes (train_animal's log), in evaluate_animal's mm
ENDED, BODY = 0.02, 0.03


def main():
    for run in sys.argv[1:]:
        ckpts, out = sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))), os.path.join(run, "eval_ckpts")
        todo = [c for c in ckpts if not os.path.exists(os.path.join(out, os.path.basename(c)[5:11], "metrics.json"))]
        if todo:  # one process for the run's checkpoints (evaluate_animal --ckpts)
            cmd = [sys.executable, "-m", "ndm_gym.evaluate_animal", "--ckpts", *todo, "--repeats", "4", "--out", out]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rows = []
        for c in ckpts:
            m = json.load(open(os.path.join(out, os.path.basename(c)[5:11], "metrics.json")))
            ok = m["ended_share"] <= ENDED and m["body_mm"] <= BODY * SIZE_MM[m["animal"]]
            rows.append({"iteration": m["iteration"], "minutes": m["minutes"], "ended_share": m["ended_share"], "body_mm": m["body_mm"], "target": ok})
        first = None
        for i in range(len(rows)):
            if all(r["target"] for r in rows[i:]):
                first = rows[i]
                break
        res = {"minutes_to_target": first and first["minutes"], "iteration_to_target": first and first["iteration"], "checkpoints": rows}
        with open(os.path.join(run, "time_to.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(run, res["minutes_to_target"], res["iteration_to_target"])


if __name__ == "__main__":
    main()
