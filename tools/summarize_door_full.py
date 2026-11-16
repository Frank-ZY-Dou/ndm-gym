"""Summarize all six frozen full-budget runs, refusing incomplete evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ('supertrack', 'supermimic')
METRICS = ('tracking_return', 'translation_rmse_m', 'hand_rmse_rad',
           'door_rmse_rad', 'latch_rmse_rad', 'opened_final', 'opened_ever')
SCHEDULE = dict(iterations=20000, worlds=2048, warmup_steps=32, wm_batches=4,
                wm_only_iters=300, critic_epochs=5, critic_minibatches=4)
SETTINGS = dict(goal_frames=5, horizon=16, wm_batch=2048, wm_window=8,
                pi_batch=4096, action_noise=.1, lr_decay_iterations=20000)
ERROR_SCALES = dict(translation_rmse_m=.02, hand_rmse_rad=.3, door_rmse_rad=.2, latch_rmse_rad=.3)


def read(path, hashes):
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def summarize(root):
    root = Path(root)
    hashes, methods = {}, {}
    execution_times = []
    manifest = read(root/'manifest.json', hashes)
    if manifest['methods'] != list(METHODS) or manifest['seeds'] != [0, 1, 2] or manifest['full_iterations'] != 20000:
        raise ValueError('Study manifest differs from the frozen methods/seeds/budget')
    for name, expected in manifest['files'].items():
        with Path(name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Frozen study file changed: '+name)
    source = Path(manifest['source'])/'ndm_gym'
    source_files = sorted(source.rglob('*.py'))
    if not source_files or any(str(p) not in manifest['files'] for p in source_files):
        raise ValueError('Source files are missing from the frozen manifest')
    source_hash = hashlib.sha256(b''.join(
        str(p.relative_to(source)).encode()+p.read_bytes() for p in source_files)).hexdigest()
    shared = None
    starts_hash = None
    for method in METHODS:
        rows = []
        for seed in range(3):
            key = f'{method}-seed{seed}'
            train = read(root/key/'result.json', hashes)
            evaluation = read(root/(key+'-evaluation')/'result.json', hashes)
            execution = read(root/(key+'-execution.json'), hashes)
            guard = read(root/(key+'-guard')/'result.json', hashes)
            if train['status'] != 'trained_full_budget' or guard['status'] != 'completed' or execution['status'] != 'completed':
                raise ValueError(f'{key}: incomplete training/guard/execution')
            protocol = train['protocol']
            if protocol['source_sha256'] != source_hash:
                raise ValueError(f'{key}: runtime source differs from frozen manifest')
            if not set(protocol['input_hashes'].values()) <= set(manifest['files'].values()):
                raise ValueError(f'{key}: input hashes are absent from frozen manifest')
            if protocol['schedule'] != SCHEDULE or any(protocol[k] != v for k, v in SETTINGS.items()):
                raise ValueError(f'{key}: schedule/settings differ from frozen full budget')
            if protocol['method'] != method or type(protocol['seed']) is not int or protocol['seed'] != seed:
                raise ValueError(f'{key}: training identity mismatch')
            counts = dict(training_transitions=40960000, warmup_transitions=65536,
                          wm_optimizer_steps=80000, policy_updates=19700,
                          critic_optimizer_steps=394000 if method == 'supermimic' else 0)
            if train['counts'] != counts or any(type(v) is not int for v in train['counts'].values()):
                raise ValueError(f'{key}: incomplete optimizer/transition counts')
            matched = {k: v for k, v in protocol.items() if k not in ('method', 'seed', 'lr', 'source_config')}
            if shared is not None and matched != shared:
                raise ValueError(f'{key}: protocols do not match')
            shared = matched
            if protocol['lr'] != (4e-4 if method == 'supertrack' else 2e-4):
                raise ValueError(f'{key}: learning rate differs from frozen protocol')
            if (evaluation['full_budget'] is not True or evaluation['iteration'] != 20000
                    or evaluation['method'] != method or type(evaluation['seed']) is not int
                    or evaluation['seed'] != seed or evaluation['prior'] is not False):
                raise ValueError(f'{key}: evaluation identity/budget mismatch')
            checkpoint = root/key/'latest.pt'
            with checkpoint.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != evaluation['checkpoint_sha256']:
                raise ValueError(f'{key}: evaluation checkpoint mismatch')
            offsets = root/(key+'-evaluation')/'starts.npz'
            digest = hashlib.sha256(offsets.read_bytes()).hexdigest()
            if digest != evaluation['starts_sha256'] or starts_hash not in (None, digest):
                raise ValueError(f'{key}: evaluation starts mismatch')
            starts_hash = digest
            episodes = evaluation['evaluations']
            if len(episodes) != 256:
                raise ValueError(f'{key}: requires all 256 starts')
            for row in episodes:
                if row['finite'] is not True:
                    raise ValueError(f'{key}: nonfinite physical episode')
                for metric in METRICS:
                    value = row[metric]
                    if metric.startswith('opened_'):
                        valid = type(value) is bool
                    else:
                        valid = type(value) in (int, float) and math.isfinite(value)
                    if not valid:
                        raise ValueError(f'{key}: invalid metric {metric}')
            means = {m: statistics.mean(row[m] for row in episodes) for m in METRICS}
            if any(not math.isclose(means[m], evaluation['means'][m], abs_tol=1e-12) for m in METRICS):
                raise ValueError(f'{key}: reported means differ from raw episodes')
            costs = dict(training_seconds=train['invocation_wall_seconds'],
                         evaluation_seconds=evaluation['wall_seconds'],
                         guarded_elapsed_seconds=execution['finished']-execution['started'])
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in costs.values()):
                raise ValueError(f'{key}: invalid elapsed time')
            execution_times.append((execution['started'], execution['finished']))
            normalized_squared_error = statistics.mean(
                statistics.mean((r[m]/scale)**2 for m, scale in ERROR_SCALES.items()) for r in episodes)
            rows.append(dict(seed=seed, means=means, counts=counts,
                             normalized_squared_error=normalized_squared_error, **costs))
        methods[method] = dict(seeds=rows, metrics={m: dict(
            mean=statistics.mean(r['means'][m] for r in rows),
            sample_sd=statistics.stdev(r['means'][m] for r in rows)) for m in METRICS},
            training_seconds=dict(mean=statistics.mean(r['training_seconds'] for r in rows),
                                  sample_sd=statistics.stdev(r['training_seconds'] for r in rows)),
            normalized_tracking_rmse=math.sqrt(statistics.mean(r['normalized_squared_error'] for r in rows)))
    controls = {}
    for folder in ('prior', 'prior-float32'):
        prior = read(root/folder/'result.json', hashes)
        if prior['prior'] is not True or prior['checkpoint_sha256'] is not None or prior['full_budget'] is not False:
            raise ValueError('Control must be an untrained zero-residual prior')
        digest = hashlib.sha256((root/folder/'starts.npz').read_bytes()).hexdigest()
        if digest != starts_hash or digest != prior['starts_sha256']:
            raise ValueError('Control must use the same evaluation starts')
        if folder == 'prior-float32':
            if prior.get('control_precision') != 'float32, matching deployed reference buffer':
                raise ValueError('Learning comparison requires the deployed float32 control')
            if not set(prior['input_sha256'].values()) <= set(manifest['files'].values()):
                raise ValueError('Deployed control inputs differ from the frozen study')
        episodes = prior['evaluations']
        if len(episodes) != 256 or any(r['finite'] is not True for r in episodes):
            raise ValueError('Control requires all 256 finite episodes')
        for row in episodes:
            for metric in METRICS:
                value = row[metric]
                valid = (type(value) is bool if metric.startswith('opened_')
                         else type(value) in (int, float) and math.isfinite(value))
                if not valid:
                    raise ValueError('Invalid control metric: '+metric)
        controls[folder] = dict(means={m: statistics.mean(r[m] for r in episodes) for m in METRICS},
                               evaluation_seconds=prior['wall_seconds'])
    prior_means = controls['prior-float32']['means']
    for method in methods.values():
        method['difference_from_deployed_float32_prior'] = {m: method['metrics'][m]['mean']-prior_means[m] for m in METRICS}
    selected = min(METHODS, key=lambda name: methods[name]['normalized_tracking_rmse'])
    return dict(methods=methods, deployed_float32_prior=controls['prior-float32'],
                reference_control_float64=controls['prior'], evidence_sha256=hashes, starts_sha256=starts_hash,
                study_wall_seconds=max(t[1] for t in execution_times)-min(t[0] for t in execution_times),
                preview_selection=dict(method=selected, seed=0, episode=0,
                                       rule='presentation_plan_v1.json; all-seed normalized tracking RMSE; terminal frame excluded'),
                guarded_gpu_hours=sum(r['guarded_elapsed_seconds'] for m in methods.values() for r in m['seeds'])/3600,
                cost_scope='Guarded elapsed time includes admission, exact-config warmup, process startup, training and evaluation. '
                           'Training seconds are the runner internal timer, starting after reference/model loading; '
                           'evaluation seconds are the physical evaluation and artifact-writing timer. '
                           'Failed launch attempts, development experiments and rendering must be reported separately.',
                scope='One demonstration; 256 common starts; final checkpoints; all three seeds; no held-out-motion claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.root)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
