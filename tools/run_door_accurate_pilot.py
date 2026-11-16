"""Sequential native Door accurate-capacity pilot; run inside the resource guard.

No retries or resource admission. Optional bootstrap/round checkpoints must have
an adjacent protocol.json from this same pilot. This is development evidence,
not formal seeds. Keep outputs on the disk admitted for the experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

METHODS = ('supertrack', 'supermimic')
OFFSETS = [0., -.015, -.0075, .0075, .015]


def validate_evaluations(evaluations):
    """Require the declared physical starts and finite measured tracking errors.

    Opening is recorded, never an acceptance threshold. A failed opening is
    valid negative evidence; a missing/reordered/nonfinite evaluation is not.
    """
    if (any(type(row.get('ARRz_offset_rad')) not in (int, float)
            or not math.isfinite(row['ARRz_offset_rad']) for row in evaluations)
            or [row.get('ARRz_offset_rad') for row in evaluations] != OFFSETS):
        raise ValueError('Pilot physical evaluation offsets/order differ')
    for row in evaluations:
        if row.get('finite') is not True:
            raise ValueError('Nonfinite native physical evaluation')
        if any(type(row.get(key)) is not bool for key in ('opened_final', 'opened_ever')):
            raise ValueError('Missing/invalid physical completion fields')
        for key in ('translation_rmse_m', 'hand_rmse_rad', 'door_rmse_rad', 'latch_rmse_rad'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Missing/nonfinite physical metric: {key}')


def command(args, method, output):
    cmd = [sys.executable, '-m', 'ndm_gym.train_door_online', '--method', method,
           '--config', 'door-hand-accurate-capacity-v1', '--model', str(args.model),
           '--reference', str(args.reference), '--bootstrap', str(args.bootstrap),
           '--output', str(output), '--device', args.device, '--seed', str(args.seed),
           '--rounds', '32']
    checkpoint = getattr(args, method + '_resume')
    if checkpoint:
        cmd += ['--resume', str(checkpoint)]
    return cmd


def expected_protocol(args, method):
    """Identity and capacity required for recovered and newly produced evidence."""
    expected = dict(method=method, seed=args.seed, config='door-hand-accurate-capacity-v1',
                    rounds=32, bootstrap_updates=1200, wm_updates=4, policy_updates=1,
                    horizon=16, batch_size=4096, wm_batch_size=2048, wm_window=8,
                    wm_hidden=[1024, 1024, 1024], policy_hidden=[1024, 512, 256],
                    device=args.device)
    for name in ('model', 'reference', 'bootstrap'):
        expected[name + '_sha256'] = hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
    return expected


def validate_resume(args, method):
    checkpoint = getattr(args, method + '_resume')
    if checkpoint is None:
        return
    if not checkpoint.is_file():
        raise ValueError(f'Missing checkpoint: {checkpoint}')
    protocol = json.loads((checkpoint.parent / 'protocol.json').read_text())
    expected = expected_protocol(args, method)
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{method} checkpoint is not this pilot protocol')
    # The learner additionally compares the complete protocol and restores RNG,
    # optimizer state and actual update counters before any collection/update.


def validate_result(args, method, path):
    result = json.loads(path.read_text())
    expected_identity = expected_protocol(args, method)
    if any(result.get('protocol', {}).get(k) != v for k, v in expected_identity.items()):
        raise ValueError('Pilot result protocol/input identity mismatch')
    expected = dict(wm=1328, policy=32)
    if method == 'supermimic':
        expected['critic'] = 640
    if (result['optimizer_counts'] != expected
            or any(type(v) is not int for v in result['optimizer_counts'].values())):
        raise ValueError('Incomplete pilot optimizer budget')
    validate_evaluations(result['evaluations'])
    return expected


def run(args):
    for name in ('model', 'reference', 'bootstrap'):
        if not getattr(args, name).is_file():
            raise ValueError(f'Missing input: {getattr(args, name)}')
    for method in METHODS:
        validate_resume(args, method)
        completed = getattr(args, method + '_completed', None)
        if completed:
            if getattr(args, method + '_resume'):
                raise ValueError('Choose completed result or checkpoint, not both')
            validate_result(args, method, completed)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(claim='Development pilot only; no formal-seed or tracking-benefit claim',
                  prior_checkpoint_cost='excluded; report interrupted/prior work separately',
                  source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((Path(__file__).resolve().parents[1] / 'ndm_gym').rglob('*.py'))},
                  stages=[])
    report_path = args.output / 'stages.json'
    for method in METHODS:
        completed = getattr(args, method + '_completed', None)
        if completed:
            counts = validate_result(args, method, completed)
            report['stages'].append(dict(
                method=method, status='reused_completed', result_path=str(completed.resolve()),
                result_sha256=hashlib.sha256(completed.read_bytes()).hexdigest(),
                optimizer_counts=counts, invocation_wall_seconds=0.,
                cost_scope='No learner invoked; original result and prior costs retained separately'))
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            continue
        cmd = command(args, method, args.output / method)
        stage = dict(method=method, command=cmd, status='running')
        report['stages'].append(stage)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        started = time.perf_counter()
        try:
            with (args.output / f'{method}.log').open('x') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            expected = validate_result(args, method, args.output / method / 'result.json')
            stage.update(status='completed', optimizer_counts=expected,
                         result_sha256=hashlib.sha256((args.output / method / 'result.json').read_bytes()).hexdigest())
        except BaseException as error:
            stage.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            stage['invocation_wall_seconds'] = time.perf_counter() - started
            report_path.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'completed'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'bootstrap', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for method in METHODS:
        parser.add_argument('--' + method + '-resume', type=Path)
        parser.add_argument('--' + method + '-completed', type=Path,
                            help='Reuse validated completed pilot result.json; no learner invocation or new compute claim')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3025)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
