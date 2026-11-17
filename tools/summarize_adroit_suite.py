"""Audit six full-budget single-seed runs against their frozen study manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.door_schedule import FullHandSchedule


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(train, evaluation, *, task, method, manifest, checkpoint_sha):
    p = train["protocol"]
    if train["status"] != "trained_full_budget" or p["task"] != task or p["method"] != method or p["seed"] != 0:
        raise ValueError("Incomplete or mismatched formal run")
    if p["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Training source differs from frozen manifest")
    expected_inputs = dict(manifest["inputs"][task], native_library=manifest["native_library_sha256"])
    if p["input_hashes"] != expected_inputs or evaluation["input_hashes"] != expected_inputs:
        raise ValueError("Inputs differ from frozen manifest")
    if (evaluation["task"], evaluation["method"], evaluation["seed"], evaluation["iteration"], evaluation["prior"]) != (
        task,
        method,
        0,
        20000,
        False,
    ):
        raise ValueError("Evaluation is not this final policy")
    if evaluation["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Evaluated checkpoint mismatch")
    FullHandSchedule().assert_counts(train["counts"], 20000, method=method)
    if p["schedule"] != vars(FullHandSchedule()):
        raise ValueError("Changed full schedule")
    episodes = evaluation["evaluations"]
    if len(episodes) != 256 or any(not row["finite"] for row in episodes):
        raise ValueError("Missing or nonfinite physical episodes")
    for key, value in evaluation["means"].items():
        values = np.asarray([row[key] for row in episodes], dtype=float)
        if not np.isfinite(values).all() or not np.isclose(value, values.mean(), rtol=1e-12, atol=1e-12):
            raise ValueError("Invalid aggregate metric: " + key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, formal = a.study_root, a.study_root / "formal-v1"
    manifest, execution = read(formal / "manifest.json"), read(formal / "execution.json")
    if len(execution) != 6 or any(v["status"] != "completed" or v["returncode"] != 0 for v in execution.values()):
        raise ValueError("All six formal guards must complete successfully")
    rows, priors, selected, evidence = [], {}, {}, {}
    for task in ("pen", "hammer", "relocate"):
        prior_dir = root / ("prechecks-" + task) / "prior"
        prior = read(prior_dir / "result.json")
        if not prior["prior"] or prior["task"] != task or len(prior["evaluations"]) != 256:
            raise ValueError("Invalid deployed prior evaluation")
        if any(not r["finite"] for r in prior["evaluations"]):
            raise ValueError("Nonfinite prior episode")
        priors[task] = prior["means"]
        baseline_starts = np.load(prior_dir / "starts.npz", allow_pickle=False)
        for method in ("supertrack", "supermimic"):
            name = task + "-" + method + "-seed0"
            run, ev = formal / name, formal / (name + "-evaluation")
            train, evaluation = read(run / "result.json"), read(ev / "result.json")
            checkpoint_sha = sha(run / "latest.pt")
            check_run(train, evaluation, task=task, method=method, manifest=manifest, checkpoint_sha=checkpoint_sha)
            with np.load(ev / "starts.npz", allow_pickle=False) as starts:
                for key in baseline_starts.files:
                    np.testing.assert_array_equal(starts[key], baseline_starts[key])
            with np.load(ev / "physical-rollouts.npz", allow_pickle=False) as trajectories:
                for i in range(256):
                    for key in ("qpos", "qvel", "actions"):
                        if not np.isfinite(trajectories[f"episode{i}_{key}"]).all():
                            raise ValueError("Nonfinite archived trajectory")
            rows.append(
                dict(
                    task=task,
                    method=method,
                    seed=0,
                    clip=evaluation["clip"],
                    counts=train["counts"],
                    means=evaluation["means"],
                    training_seconds=train["invocation_wall_seconds"],
                    evaluation_seconds=evaluation["wall_seconds"],
                    checkpoint_sha256=checkpoint_sha,
                    guarded_seconds=execution[name]["ended"] - execution[name]["started"],
                )
            )
            for path in (run / "result.json", run / "protocol.json", ev / "result.json", ev / "starts.npz", ev / "physical-rollouts.npz"):
                evidence[str(path)] = sha(path)
        baseline_starts.close()
        choices = [r for r in rows if r["task"] == task]
        selected[task] = min(choices, key=lambda r: r["means"]["normalized_tracking_rmse"])["method"]
        for path in (
            prior_dir / "result.json",
            root / task / "preparation.json",
            root / ("prechecks-" + task) / "native.json",
            root / ("prechecks-" + task) / "recovery/result.json",
        ):
            evidence[str(path)] = sha(path)
    for path in (formal / "manifest.json", formal / "execution.json"):
        evidence[str(path)] = sha(path)
    result = dict(
        status="complete_single_seed_study",
        rows=rows,
        priors=priors,
        selected_preview_methods=selected,
        frozen_source_sha256=manifest["source_sha256"],
        evidence_sha256=evidence,
        study_wall_seconds=max(x["ended"] for x in execution.values()) - min(x["started"] for x in execution.values()),
        guarded_gpu_hours=sum(x["ended"] - x["started"] for x in execution.values()) / 3600,
        scope="Six fresh full-budget runs, one seed per task/method. No training-seed standard deviation or held-out-motion claim. Guarded costs exclude earlier development/prechecks and CPU tests.",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], selected=selected, wall_seconds=result["study_wall_seconds"])))


if __name__ == "__main__":
    main()
