"""Reject incomplete and mixed-source Adroit results before publication."""

import copy

import pytest

from ndm_gym.door_schedule import FullHandSchedule
from tools.summarize_adroit_suite import check_run


def fixture():
    inputs = dict(model="m", reference="r", native_library="n")
    manifest = dict(source_sha256="s", inputs={"pen": dict(model="m", reference="r")}, native_library_sha256="n")
    train = dict(
        status="trained_full_budget",
        protocol=dict(task="pen", method="supertrack", seed=0, source_sha256="s", input_hashes=inputs, schedule=vars(FullHandSchedule())),
        counts=FullHandSchedule().counts(20000, method="supertrack"),
    )
    # counts() also returns schedule metadata; check_run accepts only measured count fields.
    train["counts"] = {
        k: train["counts"][k]
        for k in ("training_transitions", "warmup_transitions", "wm_optimizer_steps", "policy_updates", "critic_optimizer_steps")
    }
    evaluation = dict(
        task="pen",
        method="supertrack",
        seed=0,
        iteration=20000,
        prior=False,
        input_hashes=inputs,
        checkpoint_sha256="ck",
        means=dict(finite=1.0, tracking_return=1.0),
        evaluations=[dict(finite=True, tracking_return=1.0) for _ in range(256)],
    )
    return train, evaluation, manifest


def test_accept_complete_single_seed():
    train, evaluation, manifest = fixture()
    check_run(train, evaluation, task="pen", method="supertrack", manifest=manifest, checkpoint_sha="ck")


@pytest.mark.parametrize("corruption", ["budget", "source", "seed", "checkpoint", "episode", "aggregate"])
def test_reject_incomplete_or_mixed_evidence(corruption):
    train, evaluation, manifest = copy.deepcopy(fixture())
    if corruption == "budget":
        train["counts"]["policy_updates"] -= 1
    if corruption == "source":
        train["protocol"]["source_sha256"] = "other"
    if corruption == "seed":
        evaluation["seed"] = 1
    if corruption == "checkpoint":
        evaluation["checkpoint_sha256"] = "other"
    if corruption == "episode":
        evaluation["evaluations"].pop()
    if corruption == "aggregate":
        evaluation["means"]["tracking_return"] = 2.0
    with pytest.raises((ValueError, AssertionError)):
        check_run(train, evaluation, task="pen", method="supertrack", manifest=manifest, checkpoint_sha="ck")
