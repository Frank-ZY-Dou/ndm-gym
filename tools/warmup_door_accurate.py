"""Run the actual native Door CLI at accurate capacity, then restore in a fresh process.

Development admission only: one round and four bootstrap WM updates, no formal
benchmark claim. Run inside a resource-limited launcher; this tool does
not grant resource admission. Outputs/checkpoints must be on the admitted disk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def command(args, method, output, resume=None):
    values = [sys.executable, '-m', 'ndm_gym.train_door_online',
              '--method', method, '--config', 'door-hand-accurate-capacity-v1',
              '--model', str(args.model), '--reference', str(args.reference),
              '--bootstrap', str(args.bootstrap), '--output', str(output),
              '--device', args.device, '--seed', str(args.seed),
              '--rounds', '1', '--bootstrap-updates', '4', '--offsets', '0']
    if resume is not None:
        values += ['--resume', str(resume)]
    return values


def verify_result(result, method):
    expected = dict(wm=8, policy=1)
    if method == 'supermimic':
        expected['critic'] = 20
    if result['optimizer_counts'] != expected:
        raise ValueError('Actual optimizer counts differ from warmup budget')
    protocol = result['protocol']
    capacity = dict(config='door-hand-accurate-capacity-v1', horizon=16,
                    wm_hidden=[1024, 1024, 1024], policy_hidden=[1024, 512, 256],
                    wm_batch_size=2048, batch_size=4096, wm_window=8,
                    rounds=1, bootstrap_updates=4, wm_updates=4, policy_updates=1,
                    upstream_policy_optimizer=True,
                    learning_rate=4e-4 if method == 'supertrack' else 2e-4,
                    policy_lr=4e-4 if method == 'supertrack' else 2e-4)
    if method == 'supermimic':
        capacity.update(critic_epochs=5, critic_minibatches=4)
    if any(protocol.get(key) != value for key, value in capacity.items()):
        raise ValueError('Warmup did not use the declared capacity and budget')
    if not result['evaluations'] or not all(row['finite'] for row in result['evaluations']):
        raise ValueError('Native physical evaluation was absent or nonfinite')
    if result['three_seed_evidence']:
        raise ValueError('A development warmup cannot be three-seed evidence')


def physical_results(rows):
    """Compare deterministic physical metrics while retaining measured timings in reports."""
    return [{key: value for key, value in row.items() if key != 'wall_seconds'}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'bootstrap', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3023)
    args = parser.parse_args()
    args.model, args.reference, args.bootstrap, args.output = (
        path.resolve() for path in (args.model, args.reference, args.bootstrap, args.output))
    # Fail before a learner process if an input is absent or output already exists.
    for path in (args.model, args.reference, args.bootstrap):
        if not path.is_file():
            raise ValueError(f'Missing input: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(claim='Exact-capacity shortened-schedule CLI admission only; not final seeds or tracking benefit',
                  source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((Path(__file__).resolve().parents[1] / 'ndm_gym').rglob('*.py'))},
                  runs=[])
    for method in ('supertrack', 'supermimic'):
        original = None
        for stage in ('train', 'restore'):
            output = args.output / f'{method}-{stage}'
            cmd = command(args, method, output,
                          args.output / f'{method}-train/latest.pt' if stage == 'restore' else None)
            started = time.perf_counter()
            # Each call is a fresh interpreter, so restore cannot reuse the trained object.
            with (args.output / f'{method}-{stage}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            seconds = time.perf_counter() - started
            result = json.loads((output / 'result.json').read_text())
            verify_result(result, method)
            if original is not None and (physical_results(result['evaluations']) != physical_results(original['evaluations'])
                                         or result['protocol'] != original['protocol']):
                raise ValueError('Fresh-process restoration changed physical evaluation or protocol')
            original = result
            report['runs'].append(dict(method=method, stage=stage, command=cmd,
                                       wall_seconds=seconds, optimizer_counts=result['optimizer_counts'],
                                       evaluations=result['evaluations']))
            # Preserve partial successful stages if a later stage fails.
            (args.output / 'stages.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'PASS'
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
