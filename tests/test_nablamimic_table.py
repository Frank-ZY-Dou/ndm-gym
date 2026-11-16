"""benchmarks/g1_lafan1_tracking/nablamimic/nm_table.py on run directories written here in the format of NablaMimic's own files at
bfbb2a6: config.json (its train.py:342-354, 389), log.jsonl (one record per 10 updates, train.py:860-891), eval_nominal.json
(evaluate.py:89-116, 148-168), DONE and DIVERGED (train.py:894, 899). The layout was checked against files its unmodified trainer and
evaluator wrote (docs/algorithms/nablamimic.md, section 3). No GPU, no data."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "g1_lafan1_tracking", "nablamimic"))
import nm_table as table  # noqa: E402

COMMIT = "bfbb2a6773b4776c17402a34a93ec7589d9c7c18"
CLIPS = ["dance1_subject1", "walk1_subject1", "fallAndGetUp1_subject1", "run1_subject2"]
FILE_ARGS = ["--num-envs", "4096", "--horizon", "64", "--no-remat", "--replay-steps", "1000", "--collect-envs", "4096", "--hours", "3"]


def write_run(root, name, evals, s_update=0.4, start=12.0, over=("--seed", "0", "--hours", "1"), end="DONE", updates=1500, **cfg):
    """A run directory: evals = [(update, return, clips with a fall)]; the log's clock is start + s_update per update, a checkpoint's
    the clock before its update."""
    run = root / name
    (run / "ckpt").mkdir(parents=True)
    conf = {
        "config": "configs/ours.args", "out": str(run), "resume": None, "seed": 0, "clips": ["all"], "num_envs": 4096, "horizon": 64,
        "iters": 400000, "hours": 1.0, "physics": "engine", "no_remat": True, "residual_action": True, "replay_steps": 1000,
        "collect_envs": 4096, "utd": 0.4, "ckpt_every": 250, "devices": 1, "processes": 1, "git_commit": COMMIT, "git_dirty": False,
        "gpu": "NVIDIA A100-SXM4-40GB", "argv": FILE_ARGS + list(over), "config_args": FILE_ARGS, "config_file": "configs/ours.args",
        "segments": [{"start": "2026-11-11T16:14:06", "commit": COMMIT, "dirty": False, "resume": None, "gpu": "NVIDIA A100-SXM4-40GB",
                      "engine": {"version": "0.1.0", "commit": "c61a6fe", "key": "21bd49cd05b24ded259d", "arch": "sm_80", "built": False}}],
    }  # fmt: skip
    conf.update(cfg)
    (run / "config.json").write_text(json.dumps(conf, indent=2))
    with open(run / "log.jsonl", "w") as f:
        for it in range(0, updates, 10):
            rec = {"iter": it, "wall_s": start + s_update * (it + 1), "iter_s": s_update if it else start, "env_steps": 262144 * (it + 1),
                   "actor_loss": -1.5, "grad_norm": 0.0009, "critic_loss": 0.18, "rew_mean": 0.64, "term_rate": 0.016}  # fmt: skip
            if it % 100 == 0 or it in (10, 30):
                rec["gpu_mem_gb"] = {"bytes_in_use": 30.1, "peak_bytes_in_use": 36.5 + it / 1e4, "bytes_limit": 42.5}
            f.write(json.dumps(rec) + "\n")
    if evals is not None:
        ev = {"clips": CLIPS, "terms": ["body_pos", "body_ori", "action_rate", "joint_limit"], "run": str(run)}
        for it, ret, falls in evals:
            per_clip = [{"clip": c, "alive_frac": 0.5 if i < falls else 1.0, "terms": {"body_pos": 0.9}, "fall_s": 8.9 if i < falls else None,
                         "cause": 1 if i < falls else None, "return": ret} for i, c in enumerate(CLIPS)]  # fmt: skip
            ev[f"it{it}/nominal"] = {
                "full_norm_return": ret, "alive_frac": 1 - falls / 8, "failed_frac": falls / 4, "per_clip": per_clip, "iter": it,
                "fall_cause_counts": {"height": falls, "tilt": 0, "both": 0, "nonfinite": 0}, "wall_s": start + s_update * it,
                "seconds": 60.0, "gpu": conf["gpu"], "solver_iterations": 10, "action_rate_space": "setpoint", "actor_key": "actor_ema",
            }  # fmt: skip
        (run / "eval_nominal.json").write_text(json.dumps(ev))
    if end:
        (run / end).write_text(str(updates))
    return str(run)


EVALS = [(250, 0.5, 3), (500, 0.7, 1), (750, 0.8, 0), (1000, 0.83, 1), (1250, 0.85, 0), (1500, 0.86, 0), (1750, 0.87, 0)]


def test_time_to_the_target(tmp_path):
    r = table.read_run(write_run(tmp_path, "a_s0", EVALS))
    ev = r["ev"]
    assert [e[0] for e in ev] == [250, 500, 750, 1000, 1250, 1500, 1750] and r["n_clips"] == 4 and [e[3] for e in ev] == [3, 1, 0, 1, 0, 0, 0]
    assert table.first_at(ev, 0.82) == (1000, pytest.approx(412.0))  # the checkpoint's clock: 12 s of start-up + 0.4 s x 1000
    assert table.crossing(ev, 0.82) == pytest.approx(312.0 + 100.0 * 0.02 / 0.03)  # linear between 0.80 at 750 and 0.83 at 1000
    assert table.held(ev, 0.82) == (1250, pytest.approx(512.0))  # 1000 has a fall
    assert table.first_at(ev, 0.9) is None and table.crossing(ev, 0.9) is None and table.held(ev, 0.9) is None
    assert table.crossing(ev, 0.4) == pytest.approx(112.0)  # the first evaluation is already above: its own time
    assert r["s_update"] == pytest.approx(0.4) and r["w_from"] == pytest.approx(12.0 + 0.4 * 51)
    assert r["mem"] == {"peak_bytes_in_use": pytest.approx(36.5 + 0.14), "bytes_limit": 42.5}
    assert r["commit"] == "bfbb2a6" and r["gpu"] == "NVIDIA A100-SXM4-40GB" and r["state"] == "DONE 1500" and r["notes"] == []
    assert r["conf"] == "ours.args --seed 0 --hours 1; 4096 x 64 windows kept, replay 1000 x 4096"
    cells = table.cells(r, 0.82)
    assert cells[0] == "1000 / 6.87 / 6.33" and cells[1] == f"{(312 + 200 / 3) / 60:.2f}" and cells[2] == "1250 / 8.53 / 7.99"
    assert cells[3] == "0.870 (1750)" and cells[4] == "0.870, 0 of 4, 0.20 h (1750)"
    short = table.read_run(write_run(tmp_path, "short_s0", EVALS, updates=40))  # no record of update 50: no minutes from it
    assert short["w_from"] is None and short["s_update"] is None and table.cells(short, 0.82)[0] == "1000 / 6.87 / –"
    early = table.read_run(write_run(tmp_path, "early_s0", [(20, 0.9, 0), (40, 0.9, 0), (60, 0.9, 0)], updates=70))
    assert table.cells(early, 0.82)[0] == "20 / 0.33 / –" and table.cells(early, 0.82)[2] == "20 / 0.33 / –"  # at the target before update 50


def test_table_and_arms(tmp_path, capsys):
    runs = [write_run(tmp_path, "a_s0", EVALS), write_run(tmp_path, "a_s1", EVALS, s_update=0.5, over=("--seed", "1", "--hours", "1")),
            write_run(tmp_path, "b_remat_s0", EVALS[:3], no_remat=False, config_file="/runs/b_remat_s0.config.args", end=None, updates=800),
            write_run(tmp_path, "c_s0", None)]  # fmt: skip
    assert table.main(runs) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("| ")]
    rows = {line.split(" | ")[0].strip("| "): [c.strip() for c in line.strip("|").split(" | ")] for line in lines}
    assert out.startswith("NablaMimic's own code and evaluation") and "minutes from update 50" in out
    assert rows["a_s0"][4:7] == ["0.400", "1000 / 6.87 / 6.33", "6.31"] and rows["a_s1"][5:7] == ["1000 / 8.53 / 7.91", "7.84"]
    assert "b_remat_s0.config.args --seed 0 --hours 1; 4096 x 64 windows rematerialized" in rows["b_remat_s0"][1]
    assert rows["b_remat_s0"][5] == "–" and rows["b_remat_s0"][-1] == "not finished, log to update 790"
    assert rows["c_s0"][5:10] == ["not evaluated"] * 5
    arm = rows["a"]  # the second table's row of the two seeds
    assert arm[1:4] == ["2", "0.45 ± 0.07", "1000, 1000"] and arm[4] == "6.87, 8.53 (7.70 ± 1.18)" and arm[5] == "6.33, 7.91 (7.12 ± 1.12)"
    assert rows["b_remat"][3:7] == ["–", "not reached", "not reached", "not reached"] and rows["b_remat"][7] == "0"


def test_notes(tmp_path):
    cpu = table.read_run(write_run(tmp_path, "cpu_s0", EVALS, gpu="cpu"))
    assert cpu["notes"] == ["ran on the CPU: a test, not a result"]
    dirty = table.read_run(write_run(tmp_path, "dirty_s0", EVALS, git_dirty=True))
    assert dirty["notes"] == ["its source was modified: not NablaMimic's code"]
    seg = {"start": "x", "commit": COMMIT, "dirty": False, "resume": "ckpt/it_000250.pkl", "gpu": "NVIDIA A100-SXM4-40GB"}
    resumed = table.read_run(write_run(tmp_path, "res_s0", EVALS, segments=[seg, seg]))
    assert resumed["notes"] == ["resumed (2 segments): not one timed run"]
    run = write_run(tmp_path, "pin_s0", EVALS, end="DIVERGED")
    (tmp_path / "pin_s0.launch.txt").write_text("date_utc 2026-11-11T16:25:03Z\npin 0123456789abcdef\ncommit " + COMMIT + "\n")
    r = table.read_run(run)
    assert r["notes"] == ["its commit is not the pinned 0123456"] and r["state"] == "DIVERGED 1500"
    other_gpu = write_run(tmp_path, "ev_s0", EVALS)
    ev = json.loads(open(os.path.join(other_gpu, "eval_nominal.json")).read())
    for k in ev:
        if k.startswith("it"):
            ev[k]["gpu"] = "NVIDIA A40"
    open(os.path.join(other_gpu, "eval_nominal.json"), "w").write(json.dumps(ev))
    assert table.read_run(other_gpu)["gpu"] == "NVIDIA A100-SXM4-40GB (evaluated on NVIDIA A40)"
    plain = write_run(tmp_path, "plain_s0", EVALS)  # started without --config: its config.json has no argv, config_args, segments
    cfg = json.loads(open(os.path.join(plain, "config.json")).read())
    for k in ("argv", "config_args", "config_file", "segments"):
        del cfg[k]
    open(os.path.join(plain, "config.json"), "w").write(json.dumps(cfg, indent=2))
    r = table.read_run(plain)
    assert r["conf"] == "no --config; 4096 x 64 windows kept, replay 1000 x 4096" and r["notes"] == []


def test_log_lines(tmp_path):
    run = write_run(tmp_path, "cut_s0", EVALS, end=None)
    with open(os.path.join(run, "log.jsonl"), "a") as f:
        f.write('{"iter": 1500, "wall_s": 61')  # a run killed while writing
    r = table.read_run(run)
    assert r["notes"] == ["the log's last line is cut"] and r["state"] == "not finished, log to update 1490"
    with open(os.path.join(run, "log.jsonl"), "a") as f:
        f.write('\n{"iter": 1510, "wall_s": 620.0, "iter_s": 0.4}\n')
    with pytest.raises(SystemExit, match="not a JSON record"):
        table.read_run(run)
    with pytest.raises(SystemExit, match="no config.json"):
        table.read_run(str(tmp_path))
    resumed = write_run(tmp_path, "twice_s0", EVALS)
    with open(os.path.join(resumed, "log.jsonl"), "a") as f:  # a resumed run logs an update again: the later record counts
        f.write('{"iter": 1000, "wall_s": 999.0, "iter_s": 0.4}\n')
    assert table.read_log(resumed)[0][1000]["wall_s"] == 999.0
