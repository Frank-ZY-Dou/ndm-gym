"""The scripts of benchmarks/steel (results.md, Reproducing) on small inputs written here: the tables from run.sh's logs and
CSV rows, the per-phase sums of a kernel summary, and the accuracy criteria. Standard library scripts, run as a reader runs
them."""

import json
import os
import subprocess
import sys

import pytest

STEEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "steel")


def _run(script, *args):
    out = subprocess.run([sys.executable, os.path.join(STEEL, script), *map(str, args)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _log(d, tag, gpu, line, rc=0):
    with open(os.path.join(d, "logs", f"{tag}.log"), "w") as f:
        f.write(f"2026-01-01T00:00:00Z start {tag} on GPU {gpu}: cmd\n{line}\n2026-01-01T00:00:10Z end {tag}: exit {rc}\n")


def test_table_medians_ratios_and_several_gpus(tmp_path):
    one = tmp_path / "one"
    (one / "logs").mkdir(parents=True)
    steel_csv = ["engine,setting,worlds,steps,seconds,steps_per_s,ms_per_step,valid,gpu,driver,peak_mem_gb,build_s,detail"]
    pair_csv = ["pair,mode,envs,robots,steps,seconds,robot_steps_per_s,pair_steps_per_s,ms_per_step,amp,gpu,driver,peak_mem_gb,detail"]
    for rep, (fast, task, mj, st) in enumerate([(870000, 190000, 349000, 14800000), (875000, 191000, 350000, 14700000),
                                               (880000, 192000, 351000, 14900000)], 1):  # fmt: skip
        for name, v in (("fastest-passing", fast), ("task-physics", task)):
            _log(one, f"{name}_n4096_r{rep}_g0", 0, f"[steel_g1] speed {name} 4096 worlds: {v:,} control steps/s, 1.0 ms per step, valid 1")
            steel_csv.append(f"steel,{name},4096,200,1.0,{v}.0,1.0,1,A100,565,0.7,1.0,x")
        _log(one, f"mujoco-isaaclab_n4096_r{rep}_g0", 0, f"[speed_pair] physics_single_replay 4096 envs: {mj:,} robot-steps/s (x)")
        pair_csv.append(f"clip.npz,physics_single_replay,4096,4096,200,1.0,{mj}.0,{mj}.0,1.0,bf16,A100,565,0.1,x")
        _log(one, f"st-fastest_n4096_r{rep}_g0", 0, f"[speed_pair] ndm_single_replay 4096 envs: {st:,} robot-steps/s (x)")
        pair_csv.append(f"clip.npz,ndm_single_replay,4096,4096,200,1.0,{st}.0,{st}.0,1.0,bf16,A100,565,0.1,x")
    (one / "steel_rows.csv").write_text("\n".join(steel_csv) + "\n")
    (one / "pair_rows.csv").write_text("\n".join(pair_csv) + "\n")
    four = tmp_path / "four"
    (four / "logs").mkdir(parents=True)
    rows = ["engine,setting,worlds,steps,seconds,steps_per_s,ms_per_step,valid,gpu,driver,peak_mem_gb,build_s,detail"]
    for rep in (1, 2, 3):
        for k in (1, 2, 4):
            bar = four / "barrier" / f"fastest-passing_k{k}_r{rep}"
            bar.mkdir(parents=True)
            (bar / "go").write_text("1767225600.000")  # 2026-01-01T00:00:00Z: the stamps' end (10 s) is after start + window
            for g in range(k):
                (bar / f"ready.{g}.0").write_text("")
                v = 1000000 + 1000 * g + rep
                _log(four, f"fastest-passing_n16384_r{rep}k{k}_g{g}", g,
                     f"[steel_g1] speed fastest-passing 16384 worlds: {v:,} control steps/s, 16 ms per step, valid 1")
                rows.append(f"steel,fastest-passing,16384,1000,5.0,{v}.0,16.0,1,A100,565,2.8,1.0,x")
    (four / "steel_rows.csv").write_text("\n".join(rows) + "\n")
    out = _run("table.py", tmp_path, "--json", tmp_path / "t.json")
    res = json.loads((tmp_path / "t.json").read_text())
    assert res["one"]["fastest-passing|4096"]["median"] == 875000 and res["one"]["st-fastest|4096"]["median"] == 14800000
    assert "16.9x" in out and "42.3x" in out  # 14.8 M over 0.875 M and over 0.350 M
    four_k = res["four"]["fastest-passing"]
    assert four_k["1"]["median"] == 1000002 and four_k["4"]["median"] == 4 * 1000000 + 1000 * 6 + 4 * 2
    assert "one go file in 9" in out and "ready files equal k in 9" in out


def test_phases_sum_per_control_step(tmp_path):
    steel = tmp_path / "steel.csv"
    steel.write_text("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
                     "50,4000000,10,0,0,0,0,0,compute_shape_aabbs_kernel\n"
                     "40,6000000,40,0,0,0,0,0,tree_fk_id_kernel\n"
                     "10,1000000,10,0,0,0,0,0,pgs_solve_sparse\n"
                     "1,500000,1,0,0,0,0,0,setup_kernel\n")  # fmt: skip
    out = _run("phases.py", steel)
    assert "control steps in the trace: 10" in out
    assert "collision                 0.400 ms" in out and "articulation dynamics     0.600 ms" in out and "sweep                     0.100 ms" in out
    assert "sum                       1.100 ms" in out and "unmatched (setup, restarts): 0.5 ms in all, 1 kernels" in out
    ndm = tmp_path / "ndm.csv"
    ndm.write_text("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
                   "80,8000000,40,0,0,0,0,0,ampere_bf16_s16816gemm_bf16\n"
                   "10,1000000,20,0,0,0,0,0,_g1_integrate_kernel\n"
                   "5,500000,20,0,0,0,0,0,_g1_mlp_features_kernel\n"
                   "5,500000,20,0,0,0,0,0,multi_tensor_apply_kernel<copy>\n")  # fmt: skip
    out = _run("phases_ndm.py", ndm)
    assert "control steps in the trace: 20" in out and "network                   0.400 ms" in out and "integration               0.050 ms" in out
    assert "input features            0.025 ms" in out and "graph input copies        0.025 ms" in out and "sum                       0.500 ms" in out


@pytest.mark.parametrize("dropped,expect", [(False, "PASS"), (True, "fail")])
def test_gates(tmp_path, dropped, expect):
    def row(name, e8, e32, term, pen):
        return {"source": name, "windows": 100, "finite_windows": 100, "body_mm@1": [0.5, 0.0], "body_mm@8": [e8, 0.1], "body_mm@32": [e32, 0.2],
                "joint_rad@32": [0.01, 0.0], "penetration_mean_mm": pen, "penetration_peak_mm": 10 * pen,
                "terminated_windows": [term, 0.01]}  # fmt: skip

    scores = [row("recording", 0, 0, 0.0, 1.5), row("steel_reference", 5.0, 29.3, 0.03, 0.1), row("steel_fastest-passing", 5.4, 30.2, 0.03, 1.4)]
    (tmp_path / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in scores) + "\n")
    for name in ("steel_reference", "steel_fastest-passing"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"chunks": [{"dropped": dropped}]}))
    out = _run("gates.py", tmp_path / "scores.jsonl", tmp_path)
    line = next(ln for ln in out.splitlines() if ln.startswith("steel_fastest-passing"))
    assert line.endswith(expect), line
