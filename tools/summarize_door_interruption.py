"""Summarize guard-stopped Door progress without loading or certifying checkpoints.

The guard may kill the stage writer before it records failure. This report is
separate evidence: it never modifies the original stage, metrics or checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(guard, metrics):
    if guard.get('status') != 'failed' or not guard.get('reason'):
        raise ValueError('A terminal failed guard result is required')
    previous_round = 0
    previous_counts = {}
    for row in metrics:
        round_number = row['completed_rounds']
        counts = row['optimizer_counts']
        if type(round_number) is not int or round_number <= previous_round:
            raise ValueError('Completed rounds must strictly increase')
        if (not counts or any(type(v) is not int or v < 0 for v in counts.values())
                or (previous_counts and counts.keys() != previous_counts.keys())
                or any(counts[k] < v for k, v in previous_counts.items())):
            raise ValueError('Optimizer counters must be nonnegative monotonic integers')
        previous_round, previous_counts = round_number, counts
    return dict(status='interrupted', reason=guard['reason'],
                completed_rounds_from_metrics=previous_round,
                optimizer_counts_from_metrics=previous_counts,
                checkpoint_validation='NOT PERFORMED; requires admitted learner recovery',
                final_evaluation='NOT CERTIFIED by progress metrics',
                cost_scope='No invocation time or GPU compute inferred from progress metrics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard_raw, metrics_raw = args.guard.read_bytes(), args.metrics.read_bytes()
    # A torn final JSONL record is an error, never silently discarded.
    report = summarize(json.loads(guard_raw), [json.loads(line) for line in metrics_raw.splitlines()])
    report['source_sha256'] = dict(guard=hashlib.sha256(guard_raw).hexdigest(),
                                   metrics=hashlib.sha256(metrics_raw).hexdigest())
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
