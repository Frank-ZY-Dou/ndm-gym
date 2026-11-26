"""The animal tracking benchmark's reading of one run at its stop (README.md): the stop checkpoint evaluated on the run's
clips and on held-out clips, each clip from its first frame and from 19 perturbed starts (200 episodes for ten clips,
evaluation seed 0: other starts than the stop rule's), and the time to the target read backward from it.

    python benchmarks/mimic_mjx_animal_tracking/final.py <run dir> --stop <iteration> --reason rule|cap|end [--heldout 10-19]
        [--data_dir <data>] [--record] [--gpu "<GPU>"] [--code <commit>]

The target: at most 1 % of the episodes ended (two of 200) with the bodies' error of the completed episodes within 5 % of the
body's size (``meets``). The time to the target is the training clock (the checkpoint's ``wall_time``: from the first
iteration after the warm-up collection, checkpoint writes included) of the first kept checkpoint from which every later one
up to the stop meets the target, on the run's clips and these starts; read backward from the stop checkpoint (the kept
checkpoints before it, latest first, until one misses), which finds the checkpoint that evaluating them all would. None when
the stop checkpoint misses the target.

Writes ``<run>/final/train`` and ``<run>/final/heldout`` (metrics.json; ``--record``: rollout.npz too), ``<run>/ttt/<iteration>``,
and ``<run>/result.json`` with the training's iterations per second from iteration 301 to the stop (its log).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

ENDED_MAX, BODY_SHARE = 0.01, 0.05  # the target
FINAL = {"repeats": 19, "seed": 0}
REASONS = ("rule", "cap", "end")  # why a run stopped (watch.py's stop.json)


def meets(m: dict) -> bool:
    """The target: at most 1 % of the episodes ended, the bodies' error of the completed episodes within 5 % of the size."""
    return m["ended_share"] <= ENDED_MAX and m["body_mm"] is not None and m["body_mm"] <= BODY_SHARE * m["size_mm"]


def time_to_target(evaluate, stop: int, every: int, m_stop: dict) -> dict:
    """The first kept checkpoint from which every later one up to ``stop`` meets the target, read backward: ``m_stop`` the stop
    checkpoint's metrics, ``evaluate(iteration)`` an earlier one's (on the same starts), called latest first until one misses
    or the first kept checkpoint (``every``) is read. Its iteration and training minutes (None: the stop checkpoint misses)."""
    rows, first, it, m = [], None, stop, m_stop
    while True:
        ok = meets(m)
        rows.append({"iteration": it, "minutes": m["minutes"], "ended_share": m["ended_share"], "body_mm": m["body_mm"], "target": ok})
        if not ok:
            break
        first = rows[-1]
        if it - every < every:
            break
        it -= every
        m = evaluate(it)
    return {"iteration": first and first["iteration"], "minutes": first and first["minutes"], "checkpoints": rows[::-1],
            "target": {"ended_share_max": ENDED_MAX, "body_share_of_size": BODY_SHARE}}  # fmt: skip


def train_rate(log: str, first: int, last: int) -> float | None:
    """Iterations per second over iterations first + 1 .. last from the training log's rows (a row's rate covers the iterations
    since the row before it; ``first`` a row's iteration)."""
    with open(log) as f:
        rows = [(int(float(x["iteration"])), float(x["it_per_s"])) for x in csv.DictReader(f)]
    sel, prev, secs = [], first, 0.0
    for it, rate in rows:
        if first < it <= last and rate > 0:
            secs += (it - prev) / rate
            sel.append(it)
        if it > first:
            prev = it
    return (sel[-1] - first) / secs if sel and secs > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run")
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--reason", required=True, choices=REASONS)
    p.add_argument("--heldout", nargs="+", default=["10-19"], help="the held-out clips")
    p.add_argument("--data_dir", default=None, help="the clips (default: the run's)")
    p.add_argument("--record", action="store_true", help="also write the evaluations' rollout.npz (tools/render_animal.py)")
    p.add_argument("--gpu", default=None, help="the training's GPU, recorded (the evaluations' is recorded as eval_gpu)")
    p.add_argument("--code", default=None, help="the code's commit, recorded")
    a = p.parse_args()

    import numpy as np
    import torch

    from ndm_gym.evaluate_animal import Evaluator
    from ndm_gym.train_animal import parse_clips

    def ckpt(it: int) -> dict:
        path = os.path.join(a.run, f"ckpt_{it:06d}.pt")
        return {**torch.load(path, map_location="cuda:0", weights_only=False), "path": os.path.abspath(path)}

    def write(out: str, m: dict, rollout: dict | None = None):
        os.makedirs(out, exist_ok=True)
        json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
        if rollout is not None:
            np.savez_compressed(os.path.join(out, "rollout.npz"), **rollout)

    stop = ckpt(a.stop)
    run = stop["args"]
    every = int(run["ckpt_every"])
    train = Evaluator(run, None, FINAL["repeats"], 0.02, FINAL["seed"], a.data_dir)
    m_train, roll = train(stop, record=a.record)
    write(os.path.join(a.run, "final", "train"), m_train, roll)
    held = Evaluator(run, parse_clips(a.heldout), FINAL["repeats"], 0.02, FINAL["seed"], a.data_dir)
    m_held, roll = held(stop, record=a.record)
    write(os.path.join(a.run, "final", "heldout"), m_held, roll)
    del held

    def evaluate(it: int) -> dict:
        m, _ = train(ckpt(it))
        write(os.path.join(a.run, "ttt", str(it)), m)
        return m

    ttt = time_to_target(evaluate, a.stop, every, m_train)
    json.dump(ttt, open(os.path.join(a.run, "ttt.json"), "w"), indent=1)
    res = {"run": os.path.basename(os.path.normpath(a.run)), "animal": run["animal"], "algo": run["algo"], "seed": run["seed"],
            "target_rate_w": float(run.get("target_rate_w", 0.0)), "stop_iteration": a.stop, "stop_reason": a.reason, "cap": run["iterations"],
            "gpu": a.gpu, "eval_gpu": torch.cuda.get_device_name(0),
           "code": a.code, "it_per_s_301_to_stop": train_rate(os.path.join(a.run, "train_log.csv"), 300, a.stop),
           "final": {"train": m_train, "heldout": m_held}, "time_to_target": ttt}  # fmt: skip
    json.dump(res, open(os.path.join(a.run, "result.json"), "w"), indent=1)
    print(json.dumps({"run": res["run"], "stop": a.stop, "ended": m_train["ended_share"], "body_mm": m_train["body_mm"],
                      "heldout_ended": m_held["ended_share"], "ttt_min": ttt["minutes"]}))  # fmt: skip


if __name__ == "__main__":
    main()
