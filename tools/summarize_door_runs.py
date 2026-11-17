"""Validate completed Door runs against a frozen protocol before aggregation.

Standard-library only. This does not certify simulator equivalence or tracking
benefit. Supply a JSON manifest with ``seeds``, ``evaluations`` (ordered offsets),
``protocols`` (the complete expected protocol for each method, excluding seed),
and ``runs`` (result.json paths). Historical development runs are not promoted.
Uncheckpointed work and shared data generation costs must be reported separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ('opened_final', 'opened_ever', 'translation_rmse_m', 'hand_rmse_rad',
           'door_rmse_rad', 'latch_rmse_rad')
MATCHED = ('rounds', 'bootstrap_updates', 'wm_updates', 'policy_updates',
           'train_offsets', 'offsets', 'model_sha256', 'reference_sha256',
           'bootstrap_sha256', 'selection', 'horizon', 'batch_size', 'wm_batch_size',
           'wm_window', 'wm_hidden', 'policy_hidden', 'collection_worlds')


def summarize(manifest, results):
    seeds = manifest['seeds']
    offsets = manifest['evaluations']
    protocols = manifest['protocols']
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(seed) is not int for seed in seeds)):
        raise ValueError('Exactly three independent declared training seeds required')
    if (len(offsets) != 256 or len(set(offsets)) != 256
            or not all(type(x) in (int, float) and math.isfinite(x) for x in offsets)):
        raise ValueError('Exactly 256 distinct finite frozen evaluation offsets required')
    if set(protocols) != {'supertrack', 'supermimic'}:
        raise ValueError('Both method protocols required')
    for key in MATCHED:
        if key not in protocols['supertrack'] or key not in protocols['supermimic']:
            raise ValueError(f'Missing matched protocol field: {key}')
        if protocols['supertrack'][key] != protocols['supermimic'][key]:
            raise ValueError(f'Unmatched task/evaluation/budget field: {key}')
    for method, protocol in protocols.items():
        for key in ('rounds', 'bootstrap_updates', 'wm_updates', 'policy_updates'):
            if type(protocol[key]) is not int or protocol[key] < 1:
                raise ValueError(f'Invalid positive integer budget: {method}/{key}')
        if method == 'supermimic':
            for key in ('critic_epochs', 'critic_minibatches'):
                if type(protocol[key]) is not int or protocol[key] < 1:
                    raise ValueError(f'Invalid positive integer budget: {method}/{key}')
    rows = {}
    for result in results:
        protocol = result['protocol']
        method, seed = protocol['method'], protocol['seed']
        if (method not in protocols or type(seed) is not int
                or seed not in seeds or (method, seed) in rows):
            raise ValueError('Unexpected or duplicate method/seed')
        if {k: v for k, v in protocol.items() if k != 'seed'} != protocols[method]:
            raise ValueError('Run differs from frozen complete protocol')
        if protocol['offsets'] != offsets or protocol['selection'] != 'final only':
            raise ValueError('Evaluation/selection differs from frozen protocol')
        expected = dict(wm=protocol['bootstrap_updates'] + protocol['rounds'] * protocol['wm_updates'],
                        policy=protocol['rounds'] * protocol['policy_updates'])
        if method == 'supermimic':
            expected['critic'] = expected['policy'] * protocol['critic_epochs'] * protocol['critic_minibatches']
        counts = result['optimizer_counts']
        if (counts != expected or any(type(value) is not int for value in counts.values())
                or min(expected.values()) < 1):
            raise ValueError('Incomplete or zero optimizer budget')
        evaluations = result['evaluations']
        if (any(type(e['ARRz_offset_rad']) not in (int, float)
                or not math.isfinite(e['ARRz_offset_rad']) for e in evaluations)
                or [e['ARRz_offset_rad'] for e in evaluations] != offsets):
            raise ValueError('Physical evaluation offsets/order differ')
        if not all(e['finite'] is True for e in evaluations):
            raise ValueError('Nonfinite physical evaluation; retain failure separately')
        for e in evaluations:
            if any(type(e[k]) is not bool for k in ('opened_final', 'opened_ever')):
                raise ValueError('Completion fields must be booleans')
            if any(type(e[k]) not in (int, float) or not math.isfinite(e[k]) or e[k] < 0
                   for k in METRICS[2:]):
                raise ValueError('Invalid tracking metric')
        seconds = result['process_seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured process time')
        rows[method, seed] = dict(seed=seed, optimizer_counts=expected,
                                 resumed_from=result.get('resumed_from'), process_seconds=seconds,
                                 **{k: statistics.mean(e[k] for e in evaluations) for k in METRICS})
    if len(rows) != 6:
        raise ValueError('Six completed runs required; partial results are development evidence')
    methods = {}
    for method in protocols:
        values = [rows[method, seed] for seed in seeds]
        times = [r['process_seconds'] for r in values]
        methods[method] = dict(seeds=values, invocation_seconds=dict(
            total=sum(times), mean=statistics.mean(times), sample_sd=statistics.stdev(times)),
            summary={
            k: dict(mean=statistics.mean(r[k] for r in values),
                    sample_sd=statistics.stdev(r[k] for r in values)) for k in METRICS})
    return dict(methods=methods, evaluations_per_seed=256,
                reported_invocation_seconds=sum(r['process_seconds'] for r in rows.values()),
                cost_scope='process_seconds covers only reported invocation; resumed prior work, interrupted attempts, '
                           'shared bootstrap generation and GPU cost require separate measured accounting',
                claim='Completed frozen-protocol replication; no automatic tracking-benefit or upstream-equivalence claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    paths = [(args.manifest.parent / path).resolve() for path in manifest['runs']]
    payloads = [path.read_bytes() for path in paths]
    report = summarize(manifest, [json.loads(raw) for raw in payloads])
    report.update(manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  result_sha256={str(path): hashlib.sha256(payload).hexdigest()
                                 for path, payload in zip(paths, payloads)})
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
