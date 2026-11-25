"""Where a run of the benchmark stops and the training time at which it reaches the target for good: every kept checkpoint
of the run (train_robomimic --ckpt_keep) evaluated as the reported one is (evaluate_robomimic, 256 perturbed rollouts from
the same starts, without the open-loop rollouts).

- A run has settled at a checkpoint if that checkpoint and the four kept before it agree: their successes lie within 0.05
  of one another and their objects' errors within 1 mm or a quarter of their mean, whichever is larger.
- The run stops at the first checkpoint, from the iterations of its learning rate's cosine on (``--lr_decay_iters``), at
  which it has settled (the benchmark's stopping rule: README.md). A run trained past that checkpoint stops there all the
  same: the later checkpoints are evaluated and counted (how many have a success within 0.05 of the stop's), and are not
  the run's. A run that never settles stops at its last checkpoint, not settled.
- The time to the target is the wall time of the first checkpoint from which every later one, up to the stop, has a
  success of at least 0.99 with the tracked objects within 10 mm (the manipulation task's target, the hand task's).

Writes <run>/time_to.json (stop_iteration, stop_minutes, settled; minutes_to_target: null if never; every checkpoint).
The times are the runs' own wall clock, checkpoints every --ckpt_every iterations. A run without kept checkpoints, or with
an evaluation in <run>/eval_ckpts that is not this script's, is an error.

    python benchmarks/aloha_bimanual_tracking/time_to.py <run directory> [...]
    python benchmarks/aloha_bimanual_tracking/time_to.py --part 0/2 <run directory> [...]

``--part i/n`` evaluates every n-th checkpoint from the i-th and writes no time_to.json: the parts of a run on several
GPUs, before the call without it (a run to the cap of 48,000 iterations has 240 checkpoints).
"""

import glob
import json
import os
import subprocess
import sys

SUCCESS, OBJECT_MM = 0.99, 10.0
LAST, SUCCESS_BAND, OBJECT_BAND_MM, OBJECT_BAND = 5, 0.05, 1.0, 0.25  # the settled test over a checkpoint and the four before it
NUM_ENVS = 256  # evaluate_robomimic's perturbed rollouts, as in the reported checkpoint's evaluation


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


def stop(rows: list[dict], start: int) -> tuple[int, bool]:
    """The index of the checkpoint the run stops at and whether it has settled there (the module's docstring)."""
    for i, r in enumerate(rows):
        if r["iteration"] >= start and settled(rows[: i + 1]):
            return i, True
    return len(rows) - 1, False


def main():
    argv, part = sys.argv[1:], None
    if argv and argv[0] == "--part":
        part, argv = tuple(int(x) for x in argv[1].split("/")), argv[2:]
    for run in argv:
        ckpts, rows = sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))), []
        if not ckpts:
            sys.exit(f"{run}: no ckpt_*.pt (train_robomimic --ckpt_keep keeps them)")
        for ckpt in ckpts if part is None else ckpts[part[0] :: part[1]]:
            it = int(os.path.basename(ckpt)[5:11])
            out = os.path.join(run, "eval_ckpts", f"{it:06d}")
            m = check(evaluate(ckpt, out), it, os.path.join(out, "metrics.json"))
            p = m["policy"]
            ok = p["success_rate"] >= SUCCESS and p["object_mm"] <= OBJECT_MM
            rows.append({"iteration": it, "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "stage_rates": p["stage_rates"],
                         "final_state_rate": p["final_state_rate"], "object_mm": p["object_mm"], "terminated": p["terminated"], "target": ok})  # fmt: skip
        if part is not None:
            print(run, f"part {part[0]}/{part[1]}:", len(rows), "checkpoints evaluated")
            continue
        at, ok = stop(rows, json.load(open(os.path.join(run, "args.json")))["lr_decay_iters"])
        own, later = rows[: at + 1], rows[at + 1 :]
        first = next((own[i] for i in range(len(own)) if all(r["target"] for r in own[i:])), None)
        near = sum(abs(r["success_rate"] - rows[at]["success_rate"]) <= SUCCESS_BAND for r in later)
        res = {"stop_iteration": rows[at]["iteration"], "stop_minutes": rows[at]["minutes"], "settled": ok,
               "minutes_to_target": first and first["minutes"], "iteration_to_target": first and first["iteration"],
               "after_stop": {"checkpoints": len(later), "success_within_band": near}, "checkpoints": rows}  # fmt: skip
        with open(os.path.join(run, "time_to.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(run, "stops at", res["stop_iteration"], "settled" if ok else "not settled", "| to the target:", res["minutes_to_target"],
              res["iteration_to_target"], f"| after the stop {near} of {len(later)} within {SUCCESS_BAND}")  # fmt: skip


if __name__ == "__main__":
    main()
