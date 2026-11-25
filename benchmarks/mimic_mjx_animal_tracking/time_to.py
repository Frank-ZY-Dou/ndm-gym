"""The training time at which a run reaches the benchmark's target for good: every kept checkpoint of the run (train_animal
--ckpt_keep) evaluated as its last one is (evaluate_animal on the run's clips, four perturbed repeats); the wall time of the
first checkpoint from which every later one ends at most 2 % of its episodes with the bodies' error within 3 % of the body's
size (the task's: the bodies' median largest distance from the root; stick insect 0.53 mm, fruit fly 0.079 mm, rat 4.7 mm).
Writes <run>/time_to.json (minutes_to_target: null if never). The times are the runs' own wall clock, checkpoints every
--ckpt_every iterations. A run without kept checkpoints, or with an evaluation in <run>/eval_ckpts that is not this script's,
is an error.

    python benchmarks/mimic_mjx_animal_tracking/time_to.py <run directory> [...]
"""

import glob
import json
import os
import subprocess
import sys

SIZE_MM = {"stick": 17.58, "fly": 2.623, "rat": 156.6}  # the tasks' sizes (train_animal's log), in evaluate_animal's mm
ENDED, BODY = 0.02, 0.03
REPEATS = 4  # each clip's perturbed repeats, as in the last checkpoint's evaluation


def main():
    for run in sys.argv[1:]:
        ckpts, out = sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))), os.path.join(run, "eval_ckpts")
        if not ckpts:
            sys.exit(f"{run}: no ckpt_*.pt (train_animal --ckpt_keep keeps them)")
        todo = [c for c in ckpts if not os.path.exists(os.path.join(out, os.path.basename(c)[5:11], "metrics.json"))]
        if todo:  # one process for the run's checkpoints (evaluate_animal --ckpts)
            cmd = [sys.executable, "-m", "ndm_gym.evaluate_animal", "--ckpts", *todo, "--repeats", str(REPEATS), "--out", out]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rows = []
        for c in ckpts:
            path = os.path.join(out, os.path.basename(c)[5:11], "metrics.json")
            m = json.load(open(path))
            if m["iteration"] != int(os.path.basename(c)[5:11]) or m["episodes"] != m["clips"] * (1 + REPEATS):
                sys.exit(f"{path} is not this script's evaluation of {c} (the run's clips, {REPEATS} perturbed repeats): remove it")
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
