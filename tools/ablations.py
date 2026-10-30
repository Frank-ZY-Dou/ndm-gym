"""Ablation tables for benchmarks/g1_lafan1_tracking/ablations.md and the README.

Preliminary single-factor comparisons among the tuning runs (seed 0) and among the offline world-model benchmark
fits, with single-seed differences flagged only when they exceed the seed-to-seed noise measured in the final round:
``python tools/ablations.py --logs logs --wmbench results/wmbench --results results/abl results/final --md benchmarks/g1_lafan1_tracking/ablations.md``

The generated tables replace the part of the markdown file between the ``<!-- generated:begin -->`` and
``<!-- generated:end -->`` markers. The README's H100 table needs the H100 run files (available on request: open an issue): their
``logs`` among ``--logs`` and their ``results/h100`` among ``--results``.
"""

import argparse
import csv
import json
import math
import os
import statistics

DEG = 180 / math.pi
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"
README_BEGIN, README_END = "<!-- ablations:begin -->", "<!-- ablations:end -->"
RESULTS_BEGIN, RESULTS_END = "<!-- results:begin -->", "<!-- results:end -->"
H100_BEGIN, H100_END = "<!-- h100:begin -->", "<!-- h100:end -->"
DIAG_BEGIN, DIAG_END = "<!-- diag:begin -->", "<!-- diag:end -->"
NOISE_BEGIN, NOISE_END = "<!-- noise:begin -->", "<!-- noise:end -->"

# algorithm, change, base run, run: each pair differs in exactly one factor, or the change names the combination
POLICY = [
    ("SuperTrack", "task penalties (action rate; joint limits): off -> on", "r1_st_notaskreg_s0", "r1_st_s0"),
    ("SuperTrack", "failure penalty 0 -> 1", "r1_st_s0", "r2_st_fail_s0"),
    ("SuperTrack", "failure penalty 1 -> 3", "r2_st_fail_s0", "r2b_st_fail3_s0"),
    ("SuperTrack", "balance weights x1 -> x4 (failure penalty 1)", "r2_st_fail_s0", "r2b_st_fail_bal4_s0"),
    ("SuperTrack", "balance weights x1 -> x4 (failure penalty 3)", "r2b_st_fail3_s0", "r2c_st_f3b4_s0"),
    ("SuperTrack", "failure penalty 1 -> 3 (balance x4)", "r2b_st_fail_bal4_s0", "r2c_st_f3b4_s0"),
    ("SuperTrack", "failure penalty 3 -> 5 (balance x4)", "r2c_st_f3b4_s0", "r2c_st_f5b4_s0"),
    ("SuperTrack", "termination margin 1.0 -> 0.8 (penalty 3; balance x4)", "r2c_st_f3b4_s0", "r2c_st_f3b4m08_s0"),
    ("SuperTrack", "world-model pre-failure windows 0 -> 25% (penalty 3; balance x4)", "r2c_st_f3b4_s0", "r2c_st_f3b4wfw_s0"),
    ("SuperTrack", "pre-failure imagination starts 0 -> 25% (penalty 1)", "r2_st_fail_s0", "r2b_st_fail_fs_s0"),
    ("SuperTrack", "imagination kicks 0 -> 0.3 m/s (penalty 1)", "r2_st_fail_s0", "r2b_st_fail_k_s0"),
    ("SuperTrack", "action noise 0.1 -> 0.3 (no penalty)", "r1_st_s0", "r2_st_a_s0"),
    ("SuperTrack", "imagination kicks 0 -> 0.3 m/s (no penalty)", "r1_st_s0", "r2_st_k_s0"),
    ("SuperTrack", "action noise 0.3 + kicks 0.3 (no penalty)", "r1_st_s0", "r2_st_ak_s0"),
    ("SuperTrack", "all tuning: round 1 -> selected (penalty 5; balance x4)", "r1_st_s0", "r2c_st_f5b4_s0"),
    ("SuperMimic", "action noise 0.1 -> 0.3", "r1_sm_s0", "r2_sm_a_s0"),
    ("SuperMimic", "imagination kicks 0 -> 0.3 m/s", "r1_sm_s0", "r2_sm_k_s0"),
    ("SuperMimic", "action noise 0.3 + kicks 0.3", "r1_sm_s0", "r2_sm_ak_s0"),
    ("SuperMimic", "action noise 0.1 -> 0.05", "r1_sm_s0", "r2b_sm_noise005_s0"),
    ("SuperMimic", "pre-failure imagination starts 0 -> 25%", "r1_sm_s0", "r2b_sm_fs_s0"),
    ("SuperMimic", "pre-failure starts 0 -> 25% (noise 0.3 + kicks 0.3)", "r2_sm_ak_s0", "r2b_sm_ak_fs_s0"),
    ("SuperMimic", "kicks 0.3 -> 0.5 m/s (noise 0.3)", "r2_sm_ak_s0", "r2b_sm_ak05_s0"),
    ("SuperMimic", "action noise 0.3 -> 0.2 (kicks 0.3)", "r2_sm_ak_s0", "r2b_sm_a02k_s0"),
    ("SuperMimic", "reward kernel width x1 -> x0.7", "r1_sm_s0", "r2c_sm_rs07_s0"),
    ("SuperMimic", "reward kernel width x1 -> x0.5", "r1_sm_s0", "r2c_sm_rs05_s0"),
    ("SuperMimic", "discount 0.99 -> 0.995", "r1_sm_s0", "r2c_sm_g0995_s0"),
    ("SuperMimic", "noise 0.3 + kicks 0.3 (kernel x0.7)", "r2c_sm_rs07_s0", "r2c_sm_rs07ak_s0"),
    ("SuperMimic", "all tuning: round 1 -> selected (kernel x0.7; noise 0.3; kicks 0.3)", "r1_sm_s0", "r2c_sm_rs07ak_s0"),
]
# single-factor tuning pairs shown in the summary (base run, run), in order
SUMMARY_PAIRS = [
    ("r1_st_notaskreg_s0", "r1_st_s0"), ("r1_st_s0", "r2_st_fail_s0"), ("r2_st_fail_s0", "r2b_st_fail3_s0"), ("r2c_st_f3b4_s0", "r2c_st_f5b4_s0"),
    ("r2_st_fail_s0", "r2b_st_fail_bal4_s0"), ("r2b_st_fail3_s0", "r2c_st_f3b4_s0"), ("r2c_st_f3b4_s0", "r2c_st_f3b4m08_s0"),
    ("r2c_st_f3b4_s0", "r2c_st_f3b4wfw_s0"), ("r2_st_fail_s0", "r2b_st_fail_fs_s0"), ("r2_st_fail_s0", "r2b_st_fail_k_s0"), ("r1_st_s0", "r2_st_a_s0"),
    ("r1_sm_s0", "r2c_sm_rs07_s0"), ("r1_sm_s0", "r2c_sm_rs05_s0"), ("r1_sm_s0", "r2_sm_a_s0"), ("r1_sm_s0", "r2_sm_k_s0"),
    ("r1_sm_s0", "r2b_sm_noise005_s0"), ("r1_sm_s0", "r2b_sm_fs_s0"), ("r1_sm_s0", "r2c_sm_g0995_s0"),
]
# round 3, single seed at 3 h: algorithm, change, base run, run
ROUND3 = [
    ("SuperTrack", "learning rate decayed over the 3 h", "r2c_st_f5b4_s0", "r3_st_lrh_s0"),
    ("SuperTrack", "world model without LayerNorm", "r2c_st_f5b4_s0", "r3_st_noln_s0"),
    ("SuperTrack", "both", "r2c_st_f5b4_s0", "r3_st_lrhnoln_s0"),
    ("SuperTrack", "action noise 0.3 + imagination kicks 0.3 (on top of the decay)", "r3_st_lrh_s0", "r3_st_lrh_aug_s0"),
    ("SuperMimic", "learning rate decayed over the 3 h", "r2c_sm_rs07ak_s0", "r3_sm_lrh_s0"),
    ("SuperMimic", "world model without LayerNorm", "r2c_sm_rs07ak_s0", "r3_sm_noln_s0"),
    ("SuperMimic", "both", "r2c_sm_rs07ak_s0", "r3_sm_lrhnoln_s0"),
]
CHECKPOINTS = [(10000, "10k"), ("last", "3h")]
METRICS = [("norm_return", "return", 1.0), ("n_failed", "failed", 1.0), ("mpjpe_l", "mpjpe_mm", 1e3), ("e_q", "e_q_deg", DEG),
           ("action_rate", "action_rate", 1.0)]

# world model, change, base fit, fit
WORLD_MODEL = [
    ("ST-WM", "model: ST-WM -> SM-WM (same ST data)", "b1_st_l1_base", "b1_sm_on_st"),
    ("ST-WM", "window loss L1 -> L2", "b1_st_l1_base", "b1_st_l2"),
    ("ST-WM", "window sampling uniform -> by error", "b1_st_l1_base", "b1_st_prio"),
    ("ST-WM", "integration substeps 1 -> 4", "b1_st_l1_base", "b1_st_sub4"),
    ("ST-WM", "input noise 0 -> 0.03", "b1_st_l1_base", "b2_st_innoise"),
    ("ST-WM", "activation SiLU -> GELU", "b1_st_l1_base", "b3_st_gelu"),
    ("ST-WM", "activation SiLU -> ELU", "b1_st_l1_base", "b3_st_elu"),
    ("ST-WM", "activation SiLU -> Mish", "b1_st_l1_base", "b3_st_mish"),
    ("ST-WM", "LayerNorm: on -> off", "b1_st_l1_base", "b3_st_noln"),
    ("ST-WM", "backbone MLP -> RigidFormer", "b1_st_l1_base", "b4_st_rf"),
    ("ST-WM", "pre-failure windows 0 -> 25%", "b1_st_l1_base", "b5_st_failwin"),
    ("ST-WM", "data: + a second same-distribution set", "b1_st_l1_base", "b2_st_mix_same"),
    ("ST-WM", "data: + a set collected with action noise 0.3", "b1_st_l1_base", "b2_st_mix_noise"),
    ("ST-WM", "data: ST policy -> SM policy", "b1_st_l1_base", "b2_st_on_sm"),
    ("SM-WM", "LayerNorm: on -> off", "b1_sm_on_st", "b8_sm_noln"),
    ("SM-WM", "backbone MLP -> RigidFormer", "b1_sm_on_st", "b4_sm_rf"),
    ("ST-WM", "backbone MLP -> RigidFormer, tuned (lr 4e-4, as trained online)", "b1_st_l1_base", "rf_st_lr4e4"),
    ("SM-WM", "backbone MLP -> RigidFormer, tuned (lr 4e-4, as trained online)", "b1_sm_on_st", "rf_sm_lr4e4"),
    ("SM-WM", "pre-failure windows 0 -> 25%", "b1_sm_on_st", "b5_sm_failwin"),
    ("SM-WM", "data: + a set collected with action noise 0.3", "b1_sm_on_st", "b2_sm_mix_noise"),
    ("SM-WM", "data: ST policy -> SM policy", "b1_sm_on_st", "b2_sm_on_sm"),
]
WM_COLUMNS = [("st", 32, "ST policy, 32 steps"), ("sm", 32, "SM policy, 32 steps"), ("push", 32, "pushed, 32 steps"),
              ("early", 32, "early policy, 32 steps"), ("st_pf", 8, "ST pre-failure, 8 steps"), ("sm_pf", 8, "SM pre-failure, 8 steps")]


# the ablation study: algorithm, variant, change from the full configuration[,
# checkpoint: "h3" is the one --ckpt_hours saved at 3 h, otherwise the last]; runs are abl_<variant>_s<seed>, the full configurations
# are FULL's runs
STUDY = [
    ("SuperTrack", "st_nofail", "without the failure penalty"),
    ("SuperTrack", "st_nobal", "without the balance weights"),
    ("SuperTrack", "st_nofailbal", "without the failure penalty and the balance weights"),
    ("SuperTrack", "st_notaskreg", "without the task penalties"),
    ("SuperTrack", "st_noroot", "without the world-space root terms"),
    ("SuperMimic", "sm_rs1", "reward kernels at the upstream width"),
    ("SuperMimic", "sm_noise01", "simulator action noise 0.1"),
    ("SuperMimic", "sm_nokick", "without imagination kicks"),
    ("SuperMimic", "sm_residual", "offsets about the reference pose (SuperTrack's actions)"),
    ("SuperTrack", "st_smperturb", "with SuperMimic's action noise 0.3 and imagination kicks"),
    ("SuperMimic", "sm_r1", "round-1 configuration"),  # SuperTrack's round-1 configuration is st_nofailbal
    ("SuperTrack", "st_noimagnoise", "without the imagination noise on the offsets"),
    ("SuperMimic", "sm_nocritic", "without the critic"),
    ("SuperTrack", "st_paper", "the paper's objective (without the penalty, balance weights, task penalties, root terms)"),
    ("SuperTrack", "st_rf", "RigidFormer world model (SuperTrack-RF), at 3 h (equal wall-clock)", "h3"),
    ("SuperTrack", "st_rf", "RigidFormer world model (SuperTrack-RF), at the MLP runs' 3 h iteration count (equal iterations)"),
    ("SuperMimic", "sm_rf", "RigidFormer world model (SuperMimic-RF), at 3 h (equal wall-clock)", "h3"),
    ("SuperMimic", "sm_rf", "RigidFormer world model (SuperMimic-RF), at the MLP runs' 3 h iteration count (equal iterations)"),
]
FULL = {"SuperTrack": "final_st", "SuperMimic": "final_sm"}
STUDY_METRICS = [("norm_return", "return", 1.0, ".3f"), ("n_failed", "failed clips", 1.0, ".2f"), ("mpjpe_l", "MPJPE-local (mm)", 1e3, ".1f"),
                 ("e_q", "joint angle (deg)", DEG, ".2f"), ("push1", "failed at 1 m/s pushes", 1.0, ".1f"),
                 ("push2", "failed at 2 m/s pushes", 1.0, ".1f")]


def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0 or x >= 1.0:
        return max(0.0, min(1.0, x))
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_quantile(p, df):
    """Quantile p > 0.5 of Student's t distribution with df (possibly fractional) degrees of freedom."""
    cdf = lambda t: 1.0 - 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))  # noqa: E731
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def welch(full, variant):
    """variant - full: difference of the means and its Welch 95% confidence interval."""
    va, vb = statistics.variance(full) / len(full), statistics.variance(variant) / len(variant)
    d = statistics.mean(variant) - statistics.mean(full)
    if va + vb == 0:
        return d, d, d
    df = (va + vb) ** 2 / (va**2 / (len(full) - 1) + vb**2 / (len(variant) - 1))
    h = t_quantile(0.975, df) * math.sqrt(va + vb)
    return d, d - h, d + h


def run_scores(log_dirs, result_dirs, run, mark="", metrics=None):
    """A run's last checkpoint, or with ``mark`` (e.g. "h3") the one its ``--ckpt_hours`` saved: each standard metric
    averaged over all its standard evaluations (the eval watcher's, ``rep[_<mark>]_<run>.json`` and the 0 m/s entry of
    ``push[_<mark>]_<run>.json``), their number, and the failed clips under 1 and 2 m/s pushes (None without push tests);
    None if the run has no evaluation of that checkpoint. ``metrics`` replaces METRICS."""
    metrics = metrics or METRICS
    try:
        rows = evaluations(log_dirs, run)
    except FileNotFoundError:
        return None
    if not rows:  # started, not evaluated yet
        return None
    names = (f"rep_{mark}_{run}.json", f"push_{mark}_{run}.json") if mark else (f"rep_{run}.json", f"push_{run}.json")
    if mark:  # the checkpoint is the one its post-processing evaluated; the eval watcher's evaluation of it is in eval.csv
        its = {res["iteration"] for d in result_dirs for name in names if os.path.exists(os.path.join(d, name))
               for res in json.load(open(os.path.join(d, name)))}  # fmt: skip
        assert len(its) <= 1, (run, mark, its)
        last = next((r for r in rows if its and int(r["iteration"]) in its), None)
        if last is None:
            return None
    else:
        last = rows[-1]
    it = int(last["iteration"])
    std = [{col: float(last[col]) for col, *_ in metrics}]
    push = {}
    for d in result_dirs:
        for name in names:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for res in json.load(f):
                    assert res["iteration"] == it, (path, res["iteration"], it)
                    if not res.get("push_mag"):
                        std.append({col: float(res[col]) for col, *_ in metrics})
                    else:
                        push.setdefault(res["push_mag"], []).append(res["n_failed"])
    out = {col: statistics.mean(e[col] for e in std) for col, *_ in metrics}
    out["n_evals"], out["iteration"] = len(std), it
    out["push1"] = statistics.mean(push[1.0]) if 1.0 in push else None
    out["push2"] = statistics.mean(push[2.0]) if 2.0 in push else None
    return out


def seed_scores(log_dirs, result_dirs, run, mark=""):
    """``run_scores`` of a run of the ablation study, None until it has three standard evaluations and both pushes."""
    out = run_scores(log_dirs, result_dirs, run, mark)
    if out is None or out["n_evals"] < 3 or out["push1"] is None or out["push2"] is None:
        return None
    return out


def round3_rows(log_dirs, result_dirs, noise):
    rows = []
    for algo, change, base, run in ROUND3:
        b, r = run_scores(log_dirs, result_dirs, base), run_scores(log_dirs, result_dirs, run)
        if b is None or r is None:
            continue
        row = {"group": algo, "change": change, "base": base, "run": run, "base_evals": b["n_evals"], "evals": r["n_evals"]}
        for col, name, k in METRICS:
            row[f"base_{name}"], row[name] = round(b[col] * k, 4), round(r[col] * k, 4)
            row[f"d_{name}"] = round((r[col] - b[col]) * k, 4)
            row[f"sig_{name}"] = int(abs(r[col] - b[col]) * k > noise[algo][name])
        for m in ("push1", "push2"):
            row[f"base_{m}"], row[m] = b[m], r[m]
            row[f"d_{m}"] = None if b[m] is None or r[m] is None else round(r[m] - b[m], 2)
            row[f"sig_{m}"] = int(row[f"d_{m}"] is not None and m in noise[algo] and abs(row[f"d_{m}"]) > noise[algo][m])
        rows.append(row)
    return rows


def study_rows(log_dirs, result_dirs):
    full = {algo: [seed_scores(log_dirs, result_dirs, f"{prefix}_s{s}") for s in (1, 2, 3)] for algo, prefix in FULL.items()}
    rows = []
    for algo, variant, change, *mark in [(a, FULL[a], "full configuration") for a in FULL] + STUDY:
        mark = mark[0] if mark else ""
        runs = [f"{variant}_s{s}" if variant in FULL.values() else f"abl_{variant}_s{s}" for s in (1, 2, 3)]
        seeds = [seed_scores(log_dirs, result_dirs, r, mark) for r in runs]
        row = {"group": algo, "variant": variant, "checkpoint": mark or "last", "change": change,
               "complete": int(all(seeds) and all(full[algo]))}  # fmt: skip
        for col, name, k, _ in STUDY_METRICS:
            if not row["complete"]:
                continue
            xs, ref = [s[col] * k for s in seeds], [s[col] * k for s in full[algo]]
            row[f"{col}_mean"], row[f"{col}_sd"] = statistics.mean(xs), statistics.stdev(xs)
            if variant not in FULL.values():
                d, lo, hi = welch(ref, xs)
                row[f"d_{col}"], row[f"lo_{col}"], row[f"hi_{col}"], row[f"sig_{col}"] = d, lo, hi, int(lo > 0 or hi < 0)
        rows.append(row)
    return rows


def evaluations(log_dirs, run):
    for d in log_dirs:
        path = os.path.join(d, run, "eval.csv")
        if os.path.exists(path):
            with open(path) as f:
                return sorted(csv.DictReader(f), key=lambda r: int(r["iteration"]))
    raise FileNotFoundError(f"no eval.csv of {run} in {log_dirs}")


def at(log_dirs, run, checkpoint):
    rows = evaluations(log_dirs, run)
    return rows[-1] if checkpoint == "last" else next(r for r in rows if int(r["iteration"]) == checkpoint)


def seed_noise(log_dirs, result_dirs):
    """Twice the standard deviation of the difference of two single-seed runs, sqrt(2) * sigma, with sigma the
    seed-to-seed standard deviation of the final round's last checkpoints (seeds 1-3); for the failed clips under
    pushes too when the final round's push tests are in ``result_dirs``."""
    out = {}
    for algo, tag in (("SuperTrack", "st"), ("SuperMimic", "sm")):
        last = [evaluations(log_dirs, f"final_{tag}_s{s}")[-1] for s in (1, 2, 3)]
        out[algo] = {name: 2 * math.sqrt(2) * statistics.stdev(float(r[col]) * k for r in last) for col, name, k in METRICS}
        scores = [run_scores(log_dirs, result_dirs, f"final_{tag}_s{s}") for s in (1, 2, 3)]
        for m in ("push1", "push2"):
            if all(sc is not None and sc[m] is not None for sc in scores):
                out[algo][m] = 2 * math.sqrt(2) * statistics.stdev(sc[m] for sc in scores)
    return out


def policy_rows(log_dirs, noise):
    rows = []
    for algo, change, base, run in POLICY:
        for checkpoint, tag in CHECKPOINTS:
            b, r = at(log_dirs, base, checkpoint), at(log_dirs, run, checkpoint)
            row = {"group": algo, "change": change, "base": base, "run": run, "checkpoint": tag,
                   "base_it": int(b["iteration"]), "it": int(r["iteration"])}
            for col, name, k in METRICS:
                bv, rv = float(b[col]) * k, float(r[col]) * k
                row[f"base_{name}"], row[name], row[f"d_{name}"] = round(bv, 4), round(rv, 4), round(rv - bv, 4)
                row[f"sig_{name}"] = int(abs(rv - bv) > noise[algo][name])
            rows.append(row)
    return rows


LATE_STEPS = (12000, 14000, 16000, 18000, 20000)


def fit_scores(wmbench, name):
    """Scores of a fit: each 32-step error averaged over its last five evaluations (LATE_STEPS; at a single checkpoint
    it moves by 6-10% from one evaluation to the next), every other error at the final weights: its rescoring
    (``rs_<name>.csv``, with the pre-failure test sets) when there is one, else its last step."""
    with open(os.path.join(wmbench, f"{name}.csv")) as f:
        own = list(csv.DictReader(f))
    out = {}
    for test, k, _ in WM_COLUMNS:
        if k == 32:
            xs = [float(r["body_mm"]) for r in own if r["test"] == test and int(r["k"]) == k and int(r["step"]) in LATE_STEPS]
            assert len(xs) == len(LATE_STEPS), (name, test, k, len(xs))
            out[(test, k)] = statistics.mean(xs)
    rescored = os.path.join(wmbench, f"rs_{name}.csv")
    if os.path.exists(rescored):
        with open(rescored) as f:
            rows = list(csv.DictReader(f))
    else:
        rows = own
    last = max(int(r["step"]) for r in rows)
    for r in rows:
        if int(r["step"]) == last:
            out.setdefault((r["test"], int(r["k"])), float(r["body_mm"]))
    return out


def world_model_rows(wmbench):
    rows = []
    for model, change, base, fit in WORLD_MODEL:
        b, f = fit_scores(wmbench, base), fit_scores(wmbench, fit)
        row = {"group": model, "change": change, "base": base, "fit": fit}
        for test, k, _ in WM_COLUMNS:
            row[f"base_{test}_k{k}"], row[f"{test}_k{k}"] = round(b[(test, k)], 2), round(f[(test, k)], 2)
            row[f"d_{test}_k{k}"] = round(100 * (f[(test, k)] / b[(test, k)] - 1), 1)
        rows.append(row)
    return rows


WM_SEEDS = ("", "_s1", "_s2")  # seed 0 (the first fit) and the refits
WM_OOD = ("sm", "push", "early")  # the out-of-distribution test sets of table C


def wm_fits(wmbench, name):
    return [f"{name}{s}" for s in WM_SEEDS if os.path.exists(os.path.join(wmbench, f"{name}{s}.csv"))]


def wm_seed_rows(wmbench):
    """Table C with all fits: each fit's 32-step errors (mean of its last five evaluations), each test set divided by the
    base fits' mean on it, out of distribution averaged over its three test sets; the change against the base fits in
    % with a Welch 95% confidence interval when both sides have at least two fits."""
    rows = []
    for model, change, base, fit in WORLD_MODEL:
        bases, fits = wm_fits(wmbench, base), wm_fits(wmbench, fit)
        bs, fs = [fit_scores(wmbench, n) for n in bases], [fit_scores(wmbench, n) for n in fits]
        ref = {t: statistics.mean(b[(t, 32)] for b in bs) for t in ("st",) + WM_OOD}

        def norm(sc):
            return {"in": sc[("st", 32)] / ref["st"], "ood": statistics.mean(sc[(t, 32)] / ref[t] for t in WM_OOD)}

        bn, fn = [norm(b) for b in bs], [norm(f) for f in fs]
        row = {"group": model, "change": change, "base": base, "fit": fit, "n_base": len(bs), "n_fit": len(fs)}
        for key in ("in", "ood"):
            b, f = [x[key] for x in bn], [x[key] for x in fn]
            row[f"d_{key}"] = 100 * (statistics.mean(f) - statistics.mean(b))
            if len(b) >= 2 and len(f) >= 2:
                _, lo, hi = welch(b, f)
                row[f"lo_{key}"], row[f"hi_{key}"], row[f"sig_{key}"] = 100 * lo, 100 * hi, int(lo > 0 or hi < 0)
        rows.append(row)
    return rows


def signed(x, fmt):
    return ("+" if x > 0 else "") + format(x, fmt)


SUMMARY_METRICS = [("return", ".3f", 1), ("failed", ".0f", -1), ("mpjpe_mm", ".1f", -1), ("e_q_deg", ".2f", -1),
                   ("action_rate", ".2f", -1)]  # name, format, sign of the better direction
# effects that decide a verdict; joint-angle error is left out (it measures the same precision as MPJPE, and its seed
# spread in the final round is so small that its threshold would flag noise)
VERDICT_METRICS = ("return", "failed", "mpjpe_mm", "action_rate")
EFFECT_WORDS = {("return", 1): "higher return", ("return", -1): "lower return", ("failed", 1): "fewer failed clips",
                ("failed", -1): "more failed clips", ("mpjpe_mm", 1): "lower MPJPE", ("mpjpe_mm", -1): "higher MPJPE",
                ("e_q_deg", 1): "lower joint-angle error", ("e_q_deg", -1): "higher joint-angle error",
                ("action_rate", 1): "smoother actions", ("action_rate", -1): "less smooth actions",
                ("push1", 1): "fewer failed clips at 1 m/s pushes", ("push1", -1): "more failed clips at 1 m/s pushes",
                ("push2", 1): "fewer failed clips at 2 m/s pushes", ("push2", -1): "more failed clips at 2 m/s pushes"}


def effects(rows):
    """Effects on the verdict metrics that exceed the seed noise in every row (comparison point) with the same sign, as
    (metric, +1 better / -1 worse) pairs."""
    out = []
    for name, _, better in (m for m in SUMMARY_METRICS if m[0] in VERDICT_METRICS):
        if all(r[f"sig_{name}"] for r in rows) and len({r[f"d_{name}"] > 0 for r in rows}) == 1:
            out.append((name, better if rows[0][f"d_{name}"] > 0 else -better))
    return out


def summary_pair_rows(pol):
    rows = []
    for base, run in SUMMARY_PAIRS:
        pair = [next(r for r in pol if r["base"] == base and r["run"] == run and r["checkpoint"] == tag) for _, tag in CHECKPOINTS]
        row = {"group": pair[0]["group"], "change": pair[0]["change"], "base": base, "run": run}
        for name, _, _ in SUMMARY_METRICS:
            for r, (_, tag) in zip(pair, CHECKPOINTS):
                row[f"d_{name}_{tag}"], row[f"sig_{name}_{tag}"] = r[f"d_{name}"], r[f"sig_{name}"]
        row["effects"] = ";".join(f"{n}{'+' if d > 0 else '-'}" for n, d in effects(pair)) or "none"
        rows.append(row)
    return rows


def _cell(d, sig, fmt):
    return f"**{signed(d, fmt)}**" if sig else signed(d, fmt)


def _effect_text(code):
    """"return+;failed+" (+ better, - worse) -> words."""
    return "none" if code == "none" else ", ".join(EFFECT_WORDS[(c[:-1], 1 if c[-1] == "+" else -1)] for c in code.split(";"))


# rows of the README's verdict table: algorithm, factor, source, key[, note] -- a tuning pair (base run, run), a round-3
# run, or offline world-model fits (one, or several that share a row when their verdicts agree)
README_ROWS = [
    ("SuperTrack", "failure penalty (0 → 1)", "pair", ("r1_st_s0", "r2_st_fail_s0")),
    ("SuperTrack", "balance weights (×1 → ×4)", "pair", ("r2_st_fail_s0", "r2b_st_fail_bal4_s0")),
    ("SuperTrack", "task penalties (action rate, joint limits)", "pair", ("r1_st_notaskreg_s0", "r1_st_s0")),
    ("SuperTrack", "SuperMimic's action noise and imagination kicks (0.3 each, with the learning-rate decay)", "round3", "r3_st_lrh_aug_s0"),
    ("SuperTrack", "more action noise (0.1 → 0.3)", "pair", ("r1_st_s0", "r2_st_a_s0"), "push robustness not measured"),
    ("SuperTrack", "stronger failure penalty (1 → 3)", "pair", ("r2_st_fail_s0", "r2b_st_fail3_s0")),
    ("SuperTrack", "pre-failure imagination starts (25%)", "pair", ("r2_st_fail_s0", "r2b_st_fail_fs_s0")),
    ("SuperTrack", "imagination kicks alone (0.3 m/s)", "pair", ("r2_st_fail_s0", "r2b_st_fail_k_s0"), "push robustness not measured"),
    ("SuperTrack", "tighter termination margin (1.0 → 0.8)", "pair", ("r2c_st_f3b4_s0", "r2c_st_f3b4m08_s0")),
    ("SuperTrack", "world-model pre-failure windows (25%)", "pair", ("r2c_st_f3b4_s0", "r2c_st_f3b4wfw_s0")),
    ("SuperTrack", "learning rate decayed over the 3 h", "round3", "r3_st_lrh_s0"),
    ("SuperTrack", "world model without LayerNorm", "round3", "r3_st_noln_s0"),
    ("SuperMimic", "narrower reward kernels (×0.7)", "pair", ("r1_sm_s0", "r2c_sm_rs07_s0")),
    ("SuperMimic", "less action noise (0.1 → 0.05)", "pair", ("r1_sm_s0", "r2b_sm_noise005_s0"), "push robustness not measured"),
    ("SuperMimic", "more action noise (0.1 → 0.3)", "pair", ("r1_sm_s0", "r2_sm_a_s0"), "push robustness not measured"),
    ("SuperMimic", "imagination kicks (0.3 m/s)", "pair", ("r1_sm_s0", "r2_sm_k_s0"), "push robustness not measured"),
    ("SuperMimic", "pre-failure imagination starts (25%)", "pair", ("r1_sm_s0", "r2b_sm_fs_s0")),
    ("SuperMimic", "discount 0.99 → 0.995", "pair", ("r1_sm_s0", "r2c_sm_g0995_s0")),
    ("SuperMimic", "learning rate decayed over the 3 h", "round3", "r3_sm_lrh_s0"),
    ("SuperMimic", "world model without LayerNorm", "round3", "r3_sm_noln_s0"),
    ("World model", "more training data", "wm", ("b2_st_mix_same",)),
    ("World model", "no LayerNorm (both world models)", "wm", ("b3_st_noln", "b8_sm_noln")),
    ("World model", "RigidFormer backbone at the default learning rate, SuperTrack's model", "wm", ("b4_st_rf",)),
    ("World model", "RigidFormer backbone (lr 4e-4, as trained online), SuperTrack's model (SuperTrack-RF)", "wm", ("rf_st_lr4e4",)),
    ("World model", "RigidFormer backbone (lr 4e-4, as trained online), SuperMimic's model (SuperMimic-RF)", "wm", ("rf_sm_lr4e4",)),
    ("World model", "L2 loss; pre-failure windows (both world models)", "wm", ("b1_st_l2", "b5_st_failwin", "b5_sm_failwin")),
    ("World model", "RigidFormer backbone at the default learning rate, SuperMimic's model", "wm", ("b4_sm_rf",)),
    ("World model", "training data from a noisier policy (both world models; the three fits differ widely)", "wm", ("b2_st_mix_noise", "b2_sm_mix_noise")),
    ("World model", "error-prioritized windows; input noise; 4 integration substeps; GELU / ELU / Mish instead of SiLU", "wm",
     ("b1_st_prio", "b2_st_innoise", "b1_st_sub4", "b3_st_gelu", "b3_st_elu", "b3_st_mish")),
]
# preliminary README rows replaced, once its row is complete, by the leave-one-out variant on the same factor
SUPERSEDED_BY = {("SuperTrack", "failure penalty (0 → 1)"): "st_nofail", ("SuperTrack", "balance weights (×1 → ×4)"): "st_nobal",
                 ("SuperTrack", "task penalties (action rate, joint limits)"): "st_notaskreg",
                 ("SuperTrack", "SuperMimic's action noise and imagination kicks (0.3 each, with the learning-rate decay)"): "st_smperturb",
                 ("SuperMimic", "narrower reward kernels (×0.7)"): "sm_rs1", ("SuperMimic", "imagination kicks (0.3 m/s)"): "sm_nokick",
                 ("SuperMimic", "more action noise (0.1 → 0.3)"): "sm_noise01"}  # fmt: skip
VERDICT_MARK = {"helps": "✅", "trade-off": "⚖️", "hurts": "❌", "none": "➖"}
WM_MARGIN = 5.0  # % change of the open-loop error that counts while the fit-to-fit noise is not measured


def verdict(effs):
    signs = {d for _, d in effs}
    return "none" if not signs else "helps" if signs == {1} else "hurts" if signs == {-1} else "trade-off"


def _push_words(effs):
    """Effects as words, with the two push magnitudes merged when they agree."""
    push = [d for m, d in effs if m in ("push1", "push2")]
    words = [EFFECT_WORDS[e] for e in effs if e[0] not in ("push1", "push2")]
    if len(push) == 2 and push[0] == push[1]:
        words.append("fewer failed clips under 1 and 2 m/s pushes" if push[0] == 1 else "more failed clips under 1 and 2 m/s pushes")
    else:
        words += [EFFECT_WORDS[e] for e in effs if e[0] in ("push1", "push2")]
    return ", ".join(words) or "none beyond seed noise"


# rows of the README table from the leave-one-out study: variant, checkpoint mark, algorithm, factor, and whether the
# variant adds the factor to the full configuration (+1) or removes it (-1)
README_STUDY = [
    ("st_nofail", "", "SuperTrack", "failure penalty", -1),
    ("st_nobal", "", "SuperTrack", "balance weights", -1),
    ("st_nofailbal", "", "SuperTrack", "failure penalty and balance weights together", -1),
    ("st_notaskreg", "", "SuperTrack", "task penalties (action rate, joint limits)", -1),
    ("st_noroot", "", "SuperTrack", "world-space root terms", -1),
    ("st_noimagnoise", "", "SuperTrack", "imagination noise on the offsets", -1),
    ("st_paper", "", "SuperTrack", "the four additions to the paper's objective together", -1),
    ("st_smperturb", "", "SuperTrack", "SuperMimic's action noise and imagination kicks", 1),
    ("st_rf", "h3", "SuperTrack", "RigidFormer world model (SuperTrack-RF), at 3 h", 1),
    ("st_rf", "", "SuperTrack", "RigidFormer world model (SuperTrack-RF), at equal iterations", 1),
    ("sm_rs1", "", "SuperMimic", "narrower reward kernels (×0.7)", -1),
    ("sm_noise01", "", "SuperMimic", "simulator action noise 0.3 instead of 0.1", -1),
    ("sm_nokick", "", "SuperMimic", "imagination kicks", -1),
    ("sm_nocritic", "", "SuperMimic", "the critic", -1),
    ("sm_r1", "", "SuperMimic", "the tuned changes together (against round 1)", -1),
    ("sm_residual", "", "SuperMimic", "offsets about the reference pose (SuperTrack's actions)", 1),
    ("sm_rf", "h3", "SuperMimic", "RigidFormer world model (SuperMimic-RF), at 3 h", 1),
    ("sm_rf", "", "SuperMimic", "RigidFormer world model (SuperMimic-RF), at equal iterations", 1),
]
# the study's metrics that decide a verdict (joint-angle error is reported in table D but, as in the preliminary rows,
# does not decide): column, effect-word key, +1 when higher is better
STUDY_VERDICT = [("norm_return", "return", 1), ("n_failed", "failed", -1), ("mpjpe_l", "mpjpe_mm", -1), ("push1", "push1", -1),
                 ("push2", "push2", -1)]  # fmt: skip


def study_readme_rows(study):
    out = []
    for variant, mark, algo, factor, sign in README_STUDY:
        row = next((r for r in study if r["variant"] == variant and r["checkpoint"] == (mark or "last")), None)
        if row is None or not row["complete"]:
            continue
        effs = [(key, sign * better * (1 if row[f"d_{col}"] > 0 else -1)) for col, key, better in STUDY_VERDICT if row[f"sig_{col}"]]
        out.append({"group": f"{algo}, seeds 1-3", "factor": factor, "verdict": verdict(effs), "effect": _push_words(effs),
                    "codes": ";".join(f"{m}{'+' if d > 0 else '-'}" for m, d in effs), "note": "", "d_in_lo": "", "d_in_hi": "",
                    "d_ood_lo": "", "d_ood_hi": ""})  # fmt: skip
    order = list(VERDICT_MARK)
    return sorted(out, key=lambda r: (not r["group"].startswith("SuperTrack"), order.index(r["verdict"])))  # SuperTrack first


def wm_seed_verdict(r):
    """Three fits a side: the counted changes (Welch interval excludes 0) as for the policy rows; one fit: the ±5% margin."""
    if r["n_fit"] < 2 or r["n_base"] < 2:
        if r["d_ood"] < -WM_MARGIN:
            return "trade-off" if r["d_in"] > WM_MARGIN else "helps"
        return "hurts" if r["d_ood"] > WM_MARGIN or r["d_in"] > WM_MARGIN else "none"
    return verdict([(k, 1 if r[f"d_{k}"] < 0 else -1) for k in ("in", "ood") if r[f"sig_{k}"]])


def readme_rows(pairs, r3, wm_seed):
    out = []
    for algo, factor, source, key, *note in README_ROWS:
        if source == "pair":
            row = next(r for r in pairs if (r["base"], r["run"]) == key)
            effs = [] if row["effects"] == "none" else [(c[:-1], 1 if c[-1] == "+" else -1) for c in row["effects"].split(";")]
            effs = [e for e in effs if e[0] in VERDICT_METRICS]
            text = ", ".join(EFFECT_WORDS[e] for e in effs) or "none beyond seed noise"
        elif source == "round3":
            row = next((r for r in r3 if r["run"] == key), None)
            if row is None:
                continue
            effs = effects([row]) + [(m, 1 if row[f"d_{m}"] < 0 else -1) for m in ("push1", "push2") if row[f"sig_{m}"]]
            text = _push_words(effs)
        else:
            fits = [next(r for r in wm_seed if r["fit"] == f) for f in key]
            vs = {wm_seed_verdict(r) for r in fits}
            single = {r["n_fit"] < 2 for r in fits}
            assert len(vs) == 1 and len(single) == 1, (factor, vs, single)  # a shared row needs a shared verdict and kind
            cells = []
            for k in ("in", "ood"):
                ds = [r[f"d_{k}"] for r in fits]
                sig = {bool(r.get(f"sig_{k}")) for r in fits}
                assert len(sig) == 1, (factor, k, sig)  # and the same significance in each column
                txt = " to ".join(dict.fromkeys(signed(x, ".0f") for x in (min(ds), max(ds)))) + "%"
                cells.append(f"**{txt}**" if sig.pop() else txt)
            out.append({"group": "World model, offline", "factor": factor, "verdict": vs.pop(),
                        "effect": f"open-loop error {cells[0]} in distribution, {cells[1]} out of it" + (" (one fit)" if single.pop() else ""),
                        "codes": "", "note": "", "d_in_lo": round(min(r["d_in"] for r in fits), 1), "d_in_hi": round(max(r["d_in"] for r in fits), 1),
                        "d_ood_lo": round(min(r["d_ood"] for r in fits), 1), "d_ood_hi": round(max(r["d_ood"] for r in fits), 1)})  # fmt: skip
            continue
        out.append({"group": algo, "factor": factor, "verdict": verdict(effs), "effect": text + (f" ({note[0]})" if note else ""),
                    "codes": ";".join(f"{m}{'+' if d > 0 else '-'}" for m, d in effs), "note": note[0] if note else "",
                    "d_in_lo": "", "d_in_hi": "", "d_ood_lo": "", "d_ood_hi": ""})
    order = list(VERDICT_MARK)
    groups = list(dict.fromkeys(r["group"] for r in out))
    return [r for g in groups for r in sorted((x for x in out if x["group"] == g), key=lambda x: order.index(x["verdict"]))]


RESULT_METRICS = METRICS + [("e_root_ori", "e_root_ori", 1.0), ("e_body_vel", "e_body_vel", 1.0)]
RESULT_ROWS = [("normalized return", "norm_return", 1.0, ".3f"), ("failed clips (of 40)", "n_failed", 1.0, ".1f"),
               ("MPJPE, root-relative (mm)", "mpjpe_l", 1e3, ".1f"), ("joint-angle error (deg)", "e_q", DEG, ".2f"),
               ("root-orientation error (rad)", "e_root_ori", 1.0, ".3f"), ("body-velocity error (m/s)", "e_body_vel", 1.0, ".3f"),
               ("failed clips under root pushes of 1 m/s every 2 s", "push1", 1.0, ".1f"),
               ("failed clips under root pushes of 2 m/s every 2 s", "push2", 1.0, ".1f"), ("iterations in 3 h", "iteration", 1.0, ",.0f")]  # fmt: skip


def results_markdown(log_dirs, result_dirs):
    """The README's main comparison: each full configuration's seeds 1-3, every metric of the last checkpoint averaged over
    its three standard evaluations (as in table D), mean ± std over seeds."""
    sc = {algo: [run_scores(log_dirs, result_dirs, f"{p}_s{s}", metrics=RESULT_METRICS) for s in (1, 2, 3)] for algo, p in FULL.items()}
    out = ["| | SuperTrack | SuperMimic |", "|---|---|---|"]
    for label, col, k, fmt in RESULT_ROWS:
        cells = []
        for algo in ("SuperTrack", "SuperMimic"):
            xs = [x[col] * k for x in sc[algo]]
            m = statistics.mean(xs)
            cells.append(format(m, fmt) if col == "iteration" else f"{m:{fmt}} ± {statistics.stdev(xs):{fmt}}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


H100 = [("SuperTrack", "h100_final_st"), ("SuperTrack-RF", "h100_final_st_rf"), ("SuperMimic", "h100_final_sm"),
        ("SuperMimic-RF", "h100_final_sm_rf")]  # the H100 supplement; run files available on request (open an issue)
H100_MARK = 0.82  # the paper's mark: SuperMimic reaches this evaluation return after 1.2 h on an H100
H100_WINDOWS = ((16000, 20000), (24000, None))  # training-log windows before and after SuperMimic's evaluations drop
H100_TRAIN = [("training episodes: normalized return", "ep_ret_norm", 1.0, ".3f"), ("training episodes ending in a failure (%)", "term_frac", 100.0, ".2f"),
              ("world-model loss", "wm_loss", 1.0, ".4f")]  # fmt: skip


def h100_markdown(log_dirs, result_dirs):
    """The README's H100 table: the full configurations and their RigidFormer versions, seeds 1-3, 3 h on one H100 each,
    scored as the A100 table, plus the first evaluation at or above the paper's mark, the highest evaluation and the
    training-log windows around SuperMimic's drop (runs that do not reach them show a dash); None until every run has its
    three standard evaluations and both push tests."""
    sc, first, best = {}, {}, {}
    for name, p in H100:
        runs = [f"{p}_s{s}" for s in (1, 2, 3)]
        sc[name] = [run_scores(log_dirs, result_dirs, r, metrics=RESULT_METRICS) for r in runs]
        if any(x is None or x["n_evals"] < 3 or x["push1"] is None or x["push2"] is None for x in sc[name]):
            return None
        evs = [evaluations(log_dirs, r) for r in runs]
        first[name] = [next((float(e["train_wall_h"]) for e in ev if float(e["norm_return"]) >= H100_MARK), None) for ev in evs]
        best[name] = [max(float(e["norm_return"]) for e in ev) for ev in evs]
    out = ["| | " + " | ".join(name for name, _ in H100) + " |", "|---|" + "---|" * len(H100)]

    def first_cell(hs):
        if None in hs:
            return f"{sum(h is not None for h in hs)} of 3 seeds reach it"
        return f"{statistics.mean(hs):.2f} h ({min(hs):.2f}-{max(hs):.2f})"

    out.append(f"| first evaluation at or above return {H100_MARK} (training time) | " + " | ".join(first_cell(first[n]) for n, _ in H100) + " |")
    out.append("| highest evaluation return | " + " | ".join(f"{statistics.mean(best[n]):.3f} ± {statistics.stdev(best[n]):.3f}" for n, _ in H100) + " |")
    for label, col, k, fmt in RESULT_ROWS:
        cells = []
        for name, _ in H100:
            xs = [x[col] * k for x in sc[name]]
            m = statistics.mean(xs)
            cells.append(format(m, fmt) if col == "iteration" else f"{m:{fmt}} ± {statistics.stdev(xs):{fmt}}")
        out.append(f"| {label}{'' if col == 'iteration' else ', at 3 h'} | " + " | ".join(cells) + " |")
    (a0, b0), (a1, _) = H100_WINDOWS
    for label, col, k, fmt in H100_TRAIN:
        cells = []
        for name, p in H100:
            means = []
            for s in (1, 2, 3):
                log = train_log(log_dirs, f"{p}_s{s}")
                xs = [[float(r[col]) * k for r in log if lo < int(r["iteration"]) <= (hi or math.inf)] for lo, hi in H100_WINDOWS]
                means.append([statistics.mean(x) if x else None for x in xs])
            if any(None in m for m in means):  # the run ends before the later window
                cells.append("–")
            else:
                cells.append(" → ".join(format(statistics.mean(m[w] for m in means), fmt) for w in (0, 1)))
        out.append(f"| {label} (training log, iterations {a0 // 1000}k-{b0 // 1000}k → {a1 // 1000}k-3 h) | " + " | ".join(cells) + " |")
    return "\n".join(out)


# The noise check: SuperMimic's full configuration with the paper's action noise 0.1, seeds 1-3, 3 h on the H100; a seed
# drops when an evaluation after its first at or above H100_MARK scores below NOISE_DROP
NOISE_RUN, NOISE_DROP = "h100_final_sm_noise01", 0.60


def noise_markdown(log_dirs, result_dirs):
    """results.md: per seed of the noise check, the first evaluation at or above the mark, the lowest one after it and
    the declared rule's verdict, then the 3 h scores as in the H100 table; None until every run has its three standard
    evaluations and both push tests."""
    runs = [f"{NOISE_RUN}_s{s}" for s in (1, 2, 3)]
    sc = [run_scores(log_dirs, result_dirs, r, metrics=RESULT_METRICS) for r in runs]
    if any(x is None or x["n_evals"] < 3 or x["push1"] is None or x["push2"] is None for x in sc):
        return None
    first, low = [], []
    for r in runs:
        ev = evaluations(log_dirs, r)
        i = next((k for k, e in enumerate(ev) if float(e["norm_return"]) >= H100_MARK), None)
        first.append(None if i is None else ev[i])
        low.append(None if i is None or i + 1 == len(ev) else min(float(e["norm_return"]) for e in ev[i + 1 :]))
    out = ["| | seed 1 | seed 2 | seed 3 | mean ± std |", "|---|---|---|---|---|"]
    out.append(f"| first evaluation at or above return {H100_MARK} | "
               + " | ".join("–" if f is None else f"iteration {int(f['iteration']):,} ({float(f['train_wall_h']):.2f} h)" for f in first) + " | |")  # fmt: skip
    out.append("| lowest evaluation after it | " + " | ".join("–" if x is None else f"{x:.3f}" for x in low) + " | |")
    out.append(f"| drops (an evaluation below {NOISE_DROP:.2f} after it) | " + " | ".join("–" if x is None else ("yes" if x < NOISE_DROP else "no") for x in low) + " | |")
    for label, col, k, fmt in RESULT_ROWS:
        xs = [x[col] * k for x in sc]
        cells = [format(v, fmt) for v in xs]
        mean = format(statistics.mean(xs), fmt) if col == "iteration" else f"{statistics.mean(xs):{fmt}} ± {statistics.stdev(xs):{fmt}}"
        out.append(f"| {label}{'' if col == 'iteration' else ', at 3 h'} | " + " | ".join(cells) + f" | {mean} |")
    return "\n".join(out)


# SuperMimic's H100 diagnostic: <seed>_<E|L>_<condition>.json (available on request: open an issue), E the iteration-20000
# checkpoint and L the 3 h one; the E checkpoint's standard evaluation is its row in eval.csv
DIAG = [("as evaluated", "std"), ("clamp to ±20", "clamp"), ("setpoint noise 0.3 with the clamp", "act"),
        ("observation noise", "obs"), ("setpoint and observation noise", "both")]  # fmt: skip
DIAG_SEEDS = (1, 3)


def diag_markdown(diag_dir, log_dirs):
    """README: failed clips and MPJPE-local of SuperMimic's H100 checkpoints before (iteration 20000) and after (3 h) its
    evaluations drop, under parts of the training collection; None when a file is missing."""
    cols = [(s, ck) for s in DIAG_SEEDS for ck in ("E", "L")]
    head = [f"seed {s}, {'iteration 20000' if ck == 'E' else '3 h'}" for s, ck in cols]
    out = ["| evaluated with | " + " | ".join(head) + " |", "|---|" + "---|" * len(cols)]
    for label, cond in DIAG:
        cells = []
        for s, ck in cols:
            path = os.path.join(diag_dir, f"s{s}_{ck}_{cond}.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)[0]
            elif ck == "E" and cond == "std":
                r = at(log_dirs, f"h100_final_sm_s{s}", 20000)
            else:
                return None
            cells.append(f"{int(r['n_failed'])} / {float(r['mpjpe_l']) * 1e3:.1f} mm")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# Table E: what the online runs log about their world model and policy gradient, MLP (the full configurations) against
# RigidFormer (SuperTrack-RF and SuperMimic-RF, which stop at the MLP runs' 3 h iteration count), seeds 1-3
RF_ONLINE = [("SuperTrack", "final_st", "abl_st_rf"), ("SuperMimic", "final_sm", "abl_sm_rf")]
RF_ONLINE_ERRORS = [("wm_loss", "world-model loss (its training objective)", 1.0, ".4f"), ("wm_q8_rad", "8-step joint-angle error (deg)", DEG, ".3f"),
                    ("wm_body8_m", "8-step body-position error (mm)", 1e3, ".2f"), ("imag_rew", "imagined reward per step", 1.0, ".3f")]  # fmt: skip
RF_LATE = 2000  # the errors are averaged over each run's last this many iterations
RF_PI_FROM = 1000  # the policy-gradient statistics use the log rows after this iteration (policy updates start at 500)
RF_SPIKE = 3.0  # a log row is a spike when its policy-gradient norm exceeds this multiple of the run's median


def train_log(log_dirs, run):
    for d in log_dirs:
        path = os.path.join(d, run, "train_log.csv")
        if os.path.exists(path):
            with open(path) as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(f"no train_log.csv of {run} in {log_dirs}")


def rf_online_rows(log_dirs):
    """Table E rows, or the list of missing runs when a training log is missing."""
    logs, missing = {}, []
    for algo, mlp, rf in RF_ONLINE:
        for prefix in (mlp, rf):
            for s in (1, 2, 3):
                try:
                    logs[f"{prefix}_s{s}"] = train_log(log_dirs, f"{prefix}_s{s}")
                except FileNotFoundError:
                    missing.append(f"{prefix}_s{s}")
    if missing:
        return None, missing

    def late(log, col):
        end = int(log[-1]["iteration"])
        xs = [float(r[col]) for r in log if int(r["iteration"]) > end - RF_LATE]
        xs = [x for x in xs if not math.isnan(x)]
        return statistics.mean(xs) if xs else None

    def grad(log):
        g = [float(r["pi_gradnorm"]) for r in log if int(r["iteration"]) > RF_PI_FROM]
        med = statistics.median(g)
        return med, sum(x > RF_SPIKE * med for x in g), max(g), len(g)

    rows = []
    for algo, mlp, rf in RF_ONLINE:
        runs = {key: [logs[f"{p}_s{s}"] for s in (1, 2, 3)] for key, p in (("mlp", mlp), ("rf", rf))}
        for col, label, k, fmt in RF_ONLINE_ERRORS:
            v = {key: [late(log, col) for log in rs] for key, rs in runs.items()}
            if any(x is None for xs in v.values() for x in xs):  # not logged by this algorithm (SuperTrack has no imagined reward)
                continue
            m = {key: [x * k for x in xs] for key, xs in v.items()}
            rows.append({"group": algo, "quantity": f"{label}, last {RF_LATE} iterations", "fmt": fmt,
                         "mlp": m["mlp"], "rf": m["rf"], "change": 100 * (statistics.mean(m["rf"]) / statistics.mean(m["mlp"]) - 1)})  # fmt: skip
        g = {key: [grad(log) for log in rs] for key, rs in runs.items()}
        n = sorted({x[3] for xs in g.values() for x in xs})
        rows.append({"group": algo, "quantity": "policy-gradient norm before clipping to 1, median", "fmt": ".2f",
                     "mlp": [x[0] for x in g["mlp"]], "rf": [x[0] for x in g["rf"]],
                     "change": 100 * (statistics.mean(x[0] for x in g["rf"]) / statistics.mean(x[0] for x in g["mlp"]) - 1)})  # fmt: skip
        rows.append({"group": algo, "quantity": f"log rows above {RF_SPIKE:g}x that median (of {n[0]}-{n[-1]}), per seed", "fmt": "d",
                     "mlp": [x[1] for x in g["mlp"]], "rf": [x[1] for x in g["rf"]], "change": None})  # fmt: skip
        rows.append({"group": algo, "quantity": "largest log row, per seed", "fmt": ".1f",
                     "mlp": [x[2] for x in g["mlp"]], "rf": [x[2] for x in g["rf"]], "change": None})  # fmt: skip
    return rows, []


def rf_online_markdown(rows, missing):
    out = [f"**E. RigidFormer online** (seeds 1-3, from the training logs, equal iterations): MLP (the full configurations) against "
           "RigidFormer (SuperTrack-RF and SuperMimic-RF). The world-model errors are measured on the replay batches it trains on, "
           f"before the update on that batch: each run's own policy data, not held out. They are averaged over each run's last {RF_LATE} iterations; "
           f"mean ± std over seeds. The policy-gradient statistics use the log rows after iteration {RF_PI_FROM}, each the mean "
           "over 50 iterations.", ""]
    if rows is None:
        return out + ["Pending: training logs of " + ", ".join(f"`{r}`" for r in missing) + ".", ""]
    out += ["| algorithm | quantity | MLP | RigidFormer | change |", "|---|---|---|---|---|"]
    for r in rows:
        if r["change"] is None:
            cells = [", ".join(format(x, r["fmt"]) for x in r[key]) for key in ("mlp", "rf")]
        else:
            cells = [f"{statistics.mean(r[key]):{r['fmt']}} ± {statistics.stdev(r[key]):{r['fmt']}}" for key in ("mlp", "rf")]
        out.append(f"| {r['group']} | {r['quantity']} | {cells[0]} | {cells[1]} | {'' if r['change'] is None else signed(r['change'], '.0f') + '%'} |")
    return out + [""]


def readme_markdown(rows):
    """The README's verdict table: the leave-one-out study's rows and the offline world-model rows (the preliminary
    single-seed rows stay in ablations.md)."""
    out = ["✅ helps · ⚖️ trade-off · ❌ hurts · ➖ no clear effect. Policy rows: leave-one-out study, seeds 1-3, 3 h; an effect "
           "counts when its Welch 95% interval against the full configuration excludes 0 (return, failed clips, MPJPE, failed "
           "clips under 1 and 2 m/s pushes). World-model rows: open-loop 32-step error of offline fits, three per variant "
           "(bold: significant). Preliminary single-seed runs and all numbers: "
           "[`ablations.md`](benchmarks/g1_lafan1_tracking/ablations.md).", "",
           "| | | factor | effect |", "|---|---|---|---|"]
    out += [f"| {VERDICT_MARK[r['verdict']]} | {r['group'].removesuffix(', seeds 1-3')} | {r['factor']} | {r['effect']} |" for r in rows]
    return "\n".join(out)


def markdown(pol, pairs, r3, wm, wm_seed, noise, study, rf_online):
    thr = "; ".join(f"{a} return {n['return']:.3f}, failed clips {n['failed']:.1f}, MPJPE-local {n['mpjpe_mm']:.1f} mm, "
                    f"joint angle {n['e_q_deg']:.2f} deg"
                    + (f", failed clips at 1 / 2 m/s pushes {n['push1']:.1f} / {n['push2']:.1f}" if "push1" in n else "")
                    for a, n in noise.items())
    out = ["### Summary", "",
           "**D. Leave-one-out study** (seeds 1-3, 3 h; strongest evidence): mean ± std over seeds, each seed averaged over three "
           "standard evaluations; difference to the full configuration with its Welch 95% confidence interval, in bold when the "
           "interval excludes 0.", ""]
    done = [r for r in study if r["complete"]]
    if done:
        out += ["| algorithm | variant | " + " | ".join(lab for _, lab, *_ in STUDY_METRICS) + " |", "|---|---|" + "---|" * len(STUDY_METRICS)]
        for r in done:
            cells = []
            for col, _, _, fmt in STUDY_METRICS:
                c = f"{r[f'{col}_mean']:{fmt}} ± {r[f'{col}_sd']:{fmt}}"
                if f"d_{col}" in r:
                    d = f"{signed(r[f'd_{col}'], fmt)} [{r[f'lo_{col}']:{fmt}}, {r[f'hi_{col}']:{fmt}}]"
                    c += f"; {'**' + d + '**' if r[f'sig_{col}'] else d}"
                cells.append(c)
            out.append(f"| {r['group']} | {r['change']} (`{r['variant']}`) | " + " | ".join(cells) + " |")
        out.append("")
    pending = list(dict.fromkeys(r["variant"] for r in study if not r["complete"]))
    if pending:
        out += ["Pending: " + ", ".join(f"`{v}`" for v in pending) + ".", ""]
    out += rf_online_markdown(*rf_online)
    out += ["**A. Tuning pairs** (preliminary): two seed-0 runs that differ in one factor, compared at iteration 10000 (equal "
            "samples) and at 3 h (equal wall-clock; rounds 1-2 ran with the slower FK and reach fewer iterations). A difference is "
            f"bold when it exceeds twice the standard deviation of the difference of two single-seed runs ({thr}; sigma from the "
            "final round's three seeds); an effect is listed only when it does so at both points with the same sign.", "",
            "| algorithm | change | runs | return, 10k / 3 h | failed clips | MPJPE-local (mm) | joint angle (deg) | action rate | "
            "effect beyond seed noise |", "|---|---|---|---|---|---|---|---|---|"]
    for r in pairs:
        cells = [" / ".join(_cell(r[f"d_{n}_{t}"], r[f"sig_{n}_{t}"], f) for _, t in CHECKPOINTS) for n, f, _ in SUMMARY_METRICS]
        out.append(f"| {r['group']} | {r['change']} | `{r['base']}` -> `{r['run']}` | " + " | ".join(cells) + f" | {_effect_text(r['effects'])} |")
    out += ["", "**B. Round 3** (preliminary, single seed, 3 h): the variant's metrics are averaged over its standard evaluations "
            "(eval count in the runs column); same noise thresholds as A. Failed clips under pushes are base -> run.", "",
            "| algorithm | change | runs (evals) | return | failed clips | MPJPE-local (mm) | joint angle (deg) | action rate | failed at 1 m/s | "
            "failed at 2 m/s | effect beyond seed noise |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in r3:
        cells = [_cell(r[f"d_{n}"], r[f"sig_{n}"], f) for n, f, _ in SUMMARY_METRICS]
        pushes = ["-" if r[m] is None else (f"{r[f'base_{m}']:.1f} -> {r[m]:.1f}" if r[f"base_{m}"] is not None else f"{r[m]:.1f}")
                  for m in ("push1", "push2")]
        eff = effects([r]) + [(m, 1 if r[f"d_{m}"] < 0 else -1) for m in ("push1", "push2") if r[f"sig_{m}"]]
        eff = ", ".join(EFFECT_WORDS[e] for e in eff) or "none"
        out.append(f"| {r['group']} | {r['change']} | `{r['base']}` ({r['base_evals']}) -> `{r['run']}` ({r['evals']}) | "
                   + " | ".join(cells) + f" | {pushes[0]} | {pushes[1]} | {eff} |")
    out += ["", "**C. World model, offline**: the fits of each variant (seed 0 and the refits with seeds 1 and 2; fits column) against "
            "the base fits, on fixed data. Each fit's 32-step body-position error is averaged over its last five evaluations (12k-20k "
            "updates) and divided, per test set, by the base fits' mean on it; out of distribution is the mean over the SM-policy, "
            "pushed and early-policy test sets. Change in % with its Welch 95% confidence interval, in bold when the interval "
            "excludes 0; a single fit has no interval. The fits on SM-policy data are in the detailed table only, since for them "
            "these sets swap roles. The pre-failure columns compare the first fits at their final weights.", "",
            "| model | change | fits (variant / base) | in distribution, 32 steps | out of distribution, 32 steps | "
            "pre-failure (ST policy), 8 steps | pre-failure (SM policy), 8 steps |", "|---|---|---|---|---|---|---|"]
    first = {r["fit"]: r for r in wm}
    for r in (x for x in wm_seed if not x["change"].startswith("data: ST policy -> SM policy")):  # in/out of distribution swap there

        def wm_cell(k):
            d = f"{signed(r[f'd_{k}'], '.1f')}%"
            if f"lo_{k}" not in r:
                return d
            d += f" [{r[f'lo_{k}']:.1f}, {r[f'hi_{k}']:.1f}]"
            return f"**{d}**" if r[f"sig_{k}"] else d

        f0 = first[r["fit"]]
        out.append(f"| {r['group']} | {r['change']} | {r['n_fit']} / {r['n_base']} | {wm_cell('in')} | {wm_cell('ood')} | "
                   f"{signed(f0['d_st_pf_k8'], '.1f')}% | {signed(f0['d_sm_pf_k8'], '.1f')}% |")
    out += ["", "### Detailed tables", "", "Single-seed differences in **bold** exceed the seed noise (thresholds above).", ""]
    for tag, title in (("10k", "at iteration 10000 (equal iterations = equal simulator steps)"),
                       ("3h", "at the 3 h checkpoint (equal wall-clock; iterations reached in the run column)")):
        out += [f"#### Policy, {title}", "",
                "| algorithm | change | runs | return | failed clips | MPJPE-local (mm) | joint angle (deg) |", "|---|---|---|---|---|---|---|"]
        for r in (x for x in pol if x["checkpoint"] == tag):
            def cell(name, fmt):
                d = signed(r[f"d_{name}"], fmt)
                return f"{r[f'base_{name}']:{fmt}} -> {r[name]:{fmt}} ({'**' + d + '**' if r[f'sig_{name}'] else d})"

            its = f" ({r['base_it']} / {r['it']} it)" if tag == "3h" else ""
            out.append(f"| {r['group']} | {r['change']} | `{r['base']}` -> `{r['run']}`{its} | {cell('return', '.3f')} | "
                       f"{cell('failed', '.0f')} | {cell('mpjpe_mm', '.1f')} | {cell('e_q_deg', '.2f')} |")
        out.append("")
    out += ["#### World model (offline benchmark), first fits: body-position error (mm) and its change against the first base fit", "",
            "| model | change | fit | " + " | ".join(lab for *_, lab in WM_COLUMNS) + " |", "|---|---|---|" + "---|" * len(WM_COLUMNS)]
    for r in wm:
        out.append(f"| {r['group']} | {r['change']} | `{r['fit']}` | "
                   + " | ".join(f"{r[f'{t}_k{k}']:.1f} ({signed(r[f'd_{t}_k{k}'], '.1f')}%)" for t, k, _ in WM_COLUMNS) + " |")
    bases = {}
    for r in wm:
        bases.setdefault(r["base"], r)
    out += ["", "Base fits: " + "; ".join(f"`{b}` " + ", ".join(f"{t} {k} steps {r[f'base_{t}_k{k}']:.1f} mm" for t, k, _ in WM_COLUMNS)
                                          for b, r in bases.items()) + "."]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True, help="directories holding <run>/eval.csv (searched in order)")
    p.add_argument("--wmbench", required=True, help="directory of the offline world-model benchmark results")
    p.add_argument("--md", required=True, help="markdown file whose generated section is replaced")
    p.add_argument("--results", nargs="*", default=[], help="directories holding rep_<run>.json and push_<run>.json")
    p.add_argument("--readme", default=None, help="README whose verdict table and main-comparison table (ablations and results markers) are replaced")
    p.add_argument("--csv_dir", default=None, help="also write the tables as CSV")
    p.add_argument("--diag", default=None,
                   help="directory of SuperMimic's H100 diagnostic (diag/h100 of the H100 run files, available on request: open an issue)")
    p.add_argument("--results_md", default=None, help="markdown file whose H100 and diagnostic tables (h100 and diag markers) are replaced")
    a = p.parse_args()
    noise = seed_noise(a.logs, a.results)
    pol, wm, study = policy_rows(a.logs, noise), world_model_rows(a.wmbench), study_rows(a.logs, a.results)
    wm_seed = wm_seed_rows(a.wmbench)
    pairs, r3 = summary_pair_rows(pol), round3_rows(a.logs, a.results, noise)
    rf_online = rf_online_rows(a.logs)
    text = open(a.md).read()
    i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
    open(a.md, "w").write(text[:i] + "\n" + markdown(pol, pairs, r3, wm, wm_seed, noise, study, rf_online) + "\n" + text[j:])
    done = {r["variant"] for r in study if r["complete"] and r["checkpoint"] == "last"}
    prelim = [r for r in readme_rows(pairs, r3, wm_seed) if SUPERSEDED_BY.get((r["group"], r["factor"])) not in done]
    verdicts = study_readme_rows(study) + prelim
    if a.readme:
        text = open(a.readme).read()
        shown = [r for r in verdicts if r["group"].endswith(", seeds 1-3") or r["group"] == "World model, offline"]
        i, j = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
        text = text[:i] + "\n" + readme_markdown(shown) + "\n" + text[j:]
        i, j = text.index(RESULTS_BEGIN) + len(RESULTS_BEGIN), text.index(RESULTS_END)
        text = text[:i] + "\n" + results_markdown(a.logs, a.results) + "\n" + text[j:]
        open(a.readme, "w").write(text)
    if a.results_md:
        text = open(a.results_md).read()
        h100 = h100_markdown(a.logs, a.results)
        if h100 is None:
            print("H100 table not replaced: some h100_final_* run lacks its evaluations (pass logs/ and results/h100 of the H100 run files)")
        else:
            i, j = text.index(H100_BEGIN) + len(H100_BEGIN), text.index(H100_END)
            text = text[:i] + "\n" + h100 + "\n" + text[j:]
        diag = diag_markdown(a.diag, a.logs) if a.diag else None
        if diag is None:
            print("diagnostic table not replaced: pass --diag with every file of diag/h100 of the H100 run files")
        else:
            i, j = text.index(DIAG_BEGIN) + len(DIAG_BEGIN), text.index(DIAG_END)
            text = text[:i] + "\n" + diag + "\n" + text[j:]
        noise = noise_markdown(a.logs, a.results)
        if noise is None:
            print(f"noise-check table not replaced: some {NOISE_RUN}_s* run lacks its evaluations (pass the H100 run files)")
        else:
            i, j = text.index(NOISE_BEGIN) + len(NOISE_BEGIN), text.index(NOISE_END)
            text = text[:i] + "\n" + noise + "\n" + text[j:]
        open(a.results_md, "w").write(text)
    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        rf_csv = [{**r, "mlp": " ".join(map(str, r["mlp"])), "rf": " ".join(map(str, r["rf"]))} for r in rf_online[0] or []]
        for name, rows in (("abl_policy", pol), ("abl_wm", wm), ("abl_wm_seeds", wm_seed), ("abl_study", study), ("abl_pairs", pairs), ("abl_round3", r3),
                           ("abl_verdicts", verdicts), ("abl_rf_online", rf_csv)):
            with open(os.path.join(a.csv_dir, f"{name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
    print(f"{len(pol)} policy rows, {len(wm)} world-model rows, {sum(r['complete'] for r in study)}/{len(study)} study rows -> {a.md}")


if __name__ == "__main__":
    main()
