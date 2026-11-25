"""Each run's stop checkpoint by the benchmark's stop rule, the same for both algorithms: every kept checkpoint of the run
(train_robomimic --ckpt_keep, every 250 iterations) in order, evaluated on 256 perturbed starts of evaluation seed 1 (other
starts than the results', which are seed 5), without the reference's targets. With S(i) and E(i) the mean success and object
error of the evaluations at i, i - 250 and i - 500, the run stops at the first i where they and the same means 1,000
iterations earlier exist, every error among them finite, E(i) > 0.97 E(i - 1,000) and either S(i) >= 0.99 or
S(i) < S(i - 1,000) + 0.01; otherwise at its last checkpoint (the cap). Writes <run>/stop.json (the stop iteration and its
checkpoint, the evaluations read); the evaluations go to <run>/quick/<iteration>/.

    python benchmarks/robomimic_manipulation_tracking/stop_rule.py <run directory> [...]

The results' runs applied the rule while they trained (each checkpoint evaluated as it appeared, the training ended at the
stop); applied afterwards to a run trained to the cap it picks the same checkpoint, up to the evaluations' run-to-run spread.
"""

import glob
import json
import math
import os
import subprocess
import sys

EVERY, WINDOW, SMOOTH, NUM_ENVS, SEED = 250, 1000, 3, 256, 1


def means(q: dict, i: int) -> tuple[float, float] | None:
    """The mean success and object error of the evaluations at i, i - 250 and i - 500; None if one is missing or an error
    is not finite (every rollout terminated)."""
    its = [i - k * EVERY for k in range(SMOOTH)]
    if any(j not in q for j in its) or not all(math.isfinite(q[j][1]) for j in its):
        return None
    return sum(q[j][0] for j in its) / SMOOTH, sum(q[j][1] for j in its) / SMOOTH


def stop(q: dict) -> int | None:
    """The first iteration at which the rule holds over the evaluations ``q`` (iteration -> (success, object mm)), or None."""
    for i in sorted(q):
        now, before = means(q, i), means(q, i - WINDOW)
        if now is not None and before is not None:
            (s, e), (s0, e0) = now, before
            if e > 0.97 * e0 and (s >= 0.99 or s < s0 + 0.01):
                return i
    return None


def evaluate(ckpt: str, out: str) -> dict:
    path = os.path.join(out, "metrics.json")
    if not os.path.exists(path):
        cmd = [sys.executable, "-m", "ndm_gym.evaluate_robomimic", "--ckpt", ckpt, "--out", out, "--num_envs", str(NUM_ENVS),
               "--seed", str(SEED), "--no_reference"]  # fmt: skip
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(path))


def stop_of(run: str, evaluate=evaluate) -> dict:
    """The run's stop (written to <run>/stop.json): its checkpoints evaluated in order until the rule holds."""
    q, at = {}, None
    ckpts = sorted((int(os.path.basename(c)[5:11]), c) for c in glob.glob(os.path.join(run, "ckpt_*.pt")))
    if not ckpts:
        raise SystemExit(f"{run}: no kept checkpoint (train_robomimic --ckpt_keep)")
    for it, c in ckpts:
        p = evaluate(c, os.path.join(run, "quick", str(it)))["policy"]
        q[it] = (p["success_rate"], p["object_mm"] if p["object_mm"] is not None else math.nan)
        if (at := stop(q)) is not None:
            break
    res = {"stop_iteration": at if at is not None else ckpts[-1][0], "by_rule": at is not None,
           "evaluations": {str(k): {"success_rate": s, "object_mm": e} for k, (s, e) in sorted(q.items())}}  # fmt: skip
    res["stop_ckpt"] = os.path.join(run, f"ckpt_{res['stop_iteration']:06d}.pt")
    with open(os.path.join(run, "stop.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    for run in sys.argv[1:]:
        res = stop_of(run)
        print(run, res["stop_iteration"], "rule" if res["by_rule"] else "cap")


if __name__ == "__main__":
    main()
