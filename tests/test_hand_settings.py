"""benchmarks/xhand_tracking/settings.json: every algorithm's default and every hand's arguments for a task parse with
train_hand's parser, select that algorithm, and name a hand and a task of the task lists."""

import glob
import json
import os

import pytest

from ndm_gym import train_hand

BENCHMARK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "xhand_tracking")
with open(os.path.join(BENCHMARK, "settings.json")) as f:
    SETTINGS = json.load(f)
CASES = [(algo, None, None) for algo in SETTINGS]
CASES += [(algo, robot, task) for algo, entry in SETTINGS.items() for robot, tasks in entry.items() if robot != "default" for task in tasks]


def listed() -> set[tuple[str, str]]:
    pairs = set()
    for path in glob.glob(os.path.join(BENCHMARK, "*tasks.json")):
        with open(path) as f:
            suite = json.load(f)
        pairs |= {(robot, task["id"]) for robot in suite["robots"] for task in suite["tasks"]}
    return pairs


@pytest.mark.parametrize("algo, robot, task", CASES)
def test_the_arguments_parse_and_name_a_listed_hand_and_task(algo, robot, task):
    arguments = SETTINGS[algo]["default"]
    if robot is not None:
        assert (robot, task) in listed()
        arguments += " " + SETTINGS[algo][robot][task]
    assert train_hand.parse_args(["--clip", "clip", *arguments.split()]).algo == algo
