"""Do not turn repeated evaluations or changed metrics into a three-seed claim."""
import copy
from dataclasses import asdict

import pytest

from ndm_gym.tasks.adroit import AdroitRewardCfg
from tools.summarize_adroit_multiseed import TASKS, METHODS, SEEDS, aggregate, check_rewards
from tools.summarize_adroit_suite import check_run
from test_adroit_summary import fixture


def rows():
    return [dict(task=t, method=m, seed=s, means=dict(normalized_tracking_rmse=.05+.01*s, native_success_final=1.))
            for t in TASKS for m in METHODS for s in SEEDS]


def test_three_seed_statistics_and_preview_selection():
    groups, previews = aggregate(rows())
    for task in TASKS:
        for method in METHODS:
            metric = groups[task][method]['normalized_tracking_rmse']
            assert metric['mean'] == pytest.approx(.06)
            assert metric['sample_sd'] == pytest.approx(.01)
        assert previews[task]['seed'] == 0


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'wrong_seed'])
def test_reject_invalid_seed_sets(corruption):
    data = rows()
    if corruption == 'missing':
        data.pop()
    elif corruption == 'duplicate':
        data[-1] = copy.deepcopy(data[-2])
    else:
        data[-1]['seed'] = 3
    with pytest.raises(ValueError, match='seed'):
        aggregate(data)


def test_fixed_reporting_metric_does_not_follow_training_precision():
    reward = asdict(AdroitRewardCfg(nail_std=.001))
    train = dict(protocol=dict(task_reward=reward))
    evaluation = dict(training_reward=copy.deepcopy(reward), reporting_reward=asdict(AdroitRewardCfg()), evaluation_seed=20261117)
    check_rewards(train, evaluation, .001)
    evaluation['reporting_reward']['nail_std'] = .001
    with pytest.raises(ValueError, match='Reporting'):
        check_rewards(train, evaluation, .001)


def test_nonzero_seed_must_match_explicit_expectation():
    train, evaluation, manifest = fixture()
    train['protocol']['seed'] = evaluation['seed'] = 2
    check_run(train, evaluation, task='pen', method='supertrack', seed=2, manifest=manifest, checkpoint_sha='ck')
    with pytest.raises(ValueError):
        check_run(train, evaluation, task='pen', method='supertrack', seed=1, manifest=manifest, checkpoint_sha='ck')
