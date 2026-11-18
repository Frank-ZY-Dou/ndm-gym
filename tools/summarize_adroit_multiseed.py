"""Audit the full Adroit three-seed suite, retaining failed physical episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.tasks.adroit import AdroitRewardCfg
from tools.summarize_adroit_suite import check_run, read, sha


TASKS = ('pen', 'hammer', 'relocate')
METHODS = ('supertrack', 'supermimic')
SEEDS = (0, 1, 2)


def aggregate(rows):
    expected = {(t, m, s) for t in TASKS for m in METHODS for s in SEEDS}
    actual = [(r['task'], r['method'], r['seed']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicated formal seed')
    groups, previews = {}, {}
    for task in TASKS:
        groups[task] = {}
        for method in METHODS:
            selected = sorted((r for r in rows if (r['task'], r['method']) == (task, method)), key=lambda r: r['seed'])
            keys = selected[0]['means'].keys()
            if any(r['means'].keys() != keys for r in selected):
                raise ValueError('Different evaluation metrics across seeds')
            groups[task][method] = {
                key: dict(mean=mean(r['means'][key] for r in selected),
                          sample_sd=stdev(r['means'][key] for r in selected)) for key in keys}
        best = min((r for r in rows if r['task'] == task), key=lambda r: r['means']['normalized_tracking_rmse'])
        previews[task] = dict(method=best['method'], seed=best['seed'], normalized_tracking_rmse=best['means']['normalized_tracking_rmse'])
    return groups, previews


def check_rewards(train, evaluation, width):
    expected = asdict(AdroitRewardCfg(nail_std=width))
    if train['protocol']['task_reward'] != expected or evaluation['training_reward'] != expected:
        raise ValueError('Training reward differs from frozen selection')
    if evaluation['reporting_reward'] != asdict(AdroitRewardCfg()) or evaluation['evaluation_seed'] != 20261117:
        raise ValueError('Reporting metric or evaluation starts changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, formal = args.study_root, args.study_root/'formal'
    manifest, execution = read(formal/'manifest.json'), read(formal/'execution.json')
    if len(execution) != 18 or any(x['status'] != 'completed' or x['returncode'] != 0 for x in execution.values()):
        raise ValueError('All 18 full-budget jobs must finish')
    for path, digest in manifest['source_files'].items():
        if sha(root/'formal-source'/path) != digest:
            raise ValueError('Frozen source changed: '+path)
    rows, priors, evidence = [], {}, {}
    for task in TASKS:
        baseline = root/'prechecks'/(task+'-prior')
        prior = read(baseline/'result.json')
        if not prior['prior'] or prior['task'] != task or len(prior['evaluations']) != 256 or any(not x['finite'] for x in prior['evaluations']):
            raise ValueError('Invalid reference-control evaluation')
        if prior['reporting_reward'] != asdict(AdroitRewardCfg()):
            raise ValueError('Prior reporting scale changed')
        priors[task] = prior['means']
        with np.load(baseline/'starts.npz', allow_pickle=False) as z:
            baseline_starts = {key: z[key] for key in z.files}
        for method in METHODS:
            width = manifest['protocol']['reward_widths'][task+'/'+method]
            for seed in SEEDS:
                name = f'{task}-{method}-seed{seed}'
                run, ev = formal/name, formal/(name+'-evaluation')
                train, evaluation = read(run/'result.json'), read(ev/'result.json')
                checkpoint = sha(run/'latest.pt')
                check_run(train, evaluation, task=task, method=method, seed=seed, manifest=manifest, checkpoint_sha=checkpoint)
                check_rewards(train, evaluation, width)
                with np.load(ev/'starts.npz', allow_pickle=False) as starts:
                    if set(starts.files) != set(baseline_starts):
                        raise ValueError('Different reset fields')
                    for key, values in baseline_starts.items():
                        np.testing.assert_array_equal(starts[key], values)
                with np.load(ev/'physical-rollouts.npz', allow_pickle=False) as saved:
                    success = saved['native_success']
                    if success.ndim != 2 or success.shape[1] != 256 or success.dtype != np.bool_:
                        raise ValueError('Invalid archived native outcomes')
                    for i, episode in enumerate(evaluation['evaluations']):
                        if (bool(success[-1, i]) != episode['native_success_final']
                                or bool(success[:, i].any()) != episode['native_success_ever']
                                or not np.isclose(success[:, i].mean(), episode['native_success_frame_fraction'], rtol=0, atol=1e-12)):
                            raise ValueError('Reported success differs from archived outcomes')
                        for key in ('qpos', 'qvel', 'actions'):
                            data = saved[f'episode{i}_{key}']
                            length = len(success)+(key != 'actions')
                            if len(data) != length or not np.isfinite(data).all():
                                raise ValueError('Incomplete or nonfinite trajectory')
                rows.append(dict(task=task, method=method, seed=seed, clip=evaluation['clip'],
                                 training_reward=train['protocol']['task_reward'], means=evaluation['means'],
                                 native_success_count=sum(x['native_success_final'] for x in evaluation['evaluations']),
                                 nominal_native_success=evaluation['evaluations'][0]['native_success_final'],
                                 counts=train['counts'], checkpoint_sha256=checkpoint,
                                 training_seconds=train['invocation_wall_seconds'], evaluation_seconds=evaluation['wall_seconds'],
                                 guarded_seconds=execution[name]['ended']-execution[name]['started']))
                for path in (run/'result.json', run/'protocol.json', ev/'result.json', ev/'physical-rollouts.npz', ev/'starts.npz'):
                    evidence[str(path)] = sha(path)
        evidence[str(baseline/'result.json')] = sha(baseline/'result.json')
    groups, previews = aggregate(rows)
    gate = read(formal/'hammer-gate.json')
    if not gate['passed']:
        raise ValueError('Hammer quality gate failed')
    for row in rows:
        if row['task'] == 'hammer' and row['method'] == 'supermimic':
            if row['means']['native_success_final'] < .95 or row['means']['normalized_tracking_rmse'] > .1:
                raise ValueError('Hammer repair did not pass on every seed')
    for path in (formal/'manifest.json', formal/'execution.json', formal/'hammer-gate.json'):
        evidence[str(path)] = sha(path)
    result = dict(status='complete_three_seed_study', rows=rows, groups=groups, priors=priors, previews=previews,
                  frozen_source_sha256=manifest['source_sha256'], evidence_sha256=evidence,
                  wall_seconds=max(x['ended'] for x in execution.values())-min(x['started'] for x in execution.values()),
                  guarded_gpu_hours=sum(x['ended']-x['started'] for x in execution.values())/3600,
                  scope='18 fresh complete runs. Mean and sample SD of three seed means, not 768 independent training seeds. One reference per task; no held-out-motion generalization claim. Development, checks and rendering costs are separate.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], previews=previews, guarded_gpu_hours=result['guarded_gpu_hours'])))


if __name__ == '__main__':
    main()
