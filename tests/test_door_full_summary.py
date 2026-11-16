"""Final aggregation rejects partial, mismatched and nonfinite evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('full_summary', Path(__file__).parents[1]/'tools/summarize_door_full.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def study(tmp_path):
    source = tmp_path/'source'/'ndm_gym'
    source.mkdir(parents=True)
    (source/'fake.py').write_bytes(b'fixture source')
    source_hash = hashlib.sha256(b'fake.pyfixture source').hexdigest()
    model = tmp_path/'model'
    model.write_bytes(b'fixture model')
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = dict(methods=list(MODULE.METHODS), seeds=[0, 1, 2], full_iterations=20000,
                    source=str(source.parent), files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in (source/'fake.py', model)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    for method in MODULE.METHODS:
        for seed in range(3):
            key = f'{method}-seed{seed}'
            train = tmp_path/key
            evaluation = tmp_path/(key+'-evaluation')
            guard = tmp_path/(key+'-guard')
            for folder in (train, evaluation, guard):
                folder.mkdir()
            (train/'latest.pt').write_bytes(b'checkpoint')
            (evaluation/'starts.npz').write_bytes(b'common starts')
            protocol = dict(method=method, seed=seed, lr=4e-4 if method == 'supertrack' else 2e-4,
                            source_sha256=source_hash, input_hashes={'model': model_hash},
                            schedule=MODULE.SCHEDULE, **MODULE.SETTINGS)
            counts = dict(training_transitions=40960000, warmup_transitions=65536,
                          wm_optimizer_steps=80000, policy_updates=19700,
                          critic_optimizer_steps=394000 if method == 'supermimic' else 0)
            (train/'result.json').write_text(json.dumps(dict(status='trained_full_budget', protocol=protocol,
                                                           counts=counts, invocation_wall_seconds=100+seed)))
            means = {m: (.1+seed if not m.startswith('opened_') else True) for m in MODULE.METRICS}
            report = dict(full_budget=True, iteration=20000, method=method, seed=seed, prior=False,
                          checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),
                          starts_sha256=hashlib.sha256(b'common starts').hexdigest(),
                          evaluations=[dict(means, finite=True) for _ in range(256)], means=means, wall_seconds=5)
            (evaluation/'result.json').write_text(json.dumps(report))
            (guard/'result.json').write_text(json.dumps(dict(status='completed')))
            (tmp_path/(key+'-execution.json')).write_text(json.dumps(dict(status='completed', started=0, finished=120)))
    prior = tmp_path/'prior'
    prior.mkdir()
    (prior/'starts.npz').write_bytes(b'common starts')
    report.update(prior=True, full_budget=False, checkpoint_sha256=None)
    (prior/'result.json').write_text(json.dumps(report))
    deployed = tmp_path/'prior-float32'
    deployed.mkdir()
    (deployed/'starts.npz').write_bytes(b'common starts')
    report.update(control_precision='float32, matching deployed reference buffer', input_sha256={'model': model_hash})
    (deployed/'result.json').write_text(json.dumps(report))
    return tmp_path


def test_three_seed_sample_sd(study):
    report = MODULE.summarize(study)
    for row in report['methods'].values():
        assert row['metrics']['hand_rmse_rad'] == dict(mean=1.1, sample_sd=1.)
        assert [r['seed'] for r in row['seeds']] == [0, 1, 2]
    assert report['guarded_gpu_hours'] == .2
    assert report['preview_selection']['method'] == 'supertrack'
    assert report['preview_selection']['seed'] == 0
    assert report['methods']['supertrack']['difference_from_deployed_float32_prior']['hand_rmse_rad'] == -1.
    assert 'reference_control_float64' in report


@pytest.mark.parametrize('corruption', ['missing', 'partial', 'counts', 'identity', 'nan', 'starts', 'checkpoint', 'source',
                                      'horizon', 'prior_starts', 'prior_finite', 'frozen_input', 'deployed_precision'])
def test_reject_bad_evidence(study, corruption):
    train = study/'supertrack-seed0'/'result.json'
    evaluation = study/'supertrack-seed0-evaluation'/'result.json'
    t, e = json.loads(train.read_text()), json.loads(evaluation.read_text())
    if corruption == 'missing':
        (study/'supermimic-seed2'/'result.json').unlink()
    elif corruption == 'partial':
        e['evaluations'].pop()
    elif corruption == 'counts':
        t['counts']['policy_updates'] = 1
    elif corruption == 'identity':
        e['seed'] = False
    elif corruption == 'nan':
        e['evaluations'][0]['hand_rmse_rad'] = float('nan')
    elif corruption == 'starts':
        (study/'supertrack-seed0-evaluation'/'starts.npz').write_bytes(b'other starts')
    elif corruption == 'checkpoint':
        (study/'supertrack-seed0'/'latest.pt').write_bytes(b'other checkpoint')
    elif corruption == 'source':
        t['protocol']['source_sha256'] = 'different'
    elif corruption == 'horizon':
        t['protocol']['horizon'] = 4
    elif corruption == 'prior_starts':
        (study/'prior'/'starts.npz').write_bytes(b'other starts')
    elif corruption == 'prior_finite':
        p = study/'prior'/'result.json'
        r = json.loads(p.read_text())
        r['evaluations'][0]['finite'] = False
        p.write_text(json.dumps(r))
    elif corruption == 'frozen_input':
        (study/'model').write_bytes(b'changed model')
    elif corruption == 'deployed_precision':
        p = study/'prior-float32'/'result.json'
        r = json.loads(p.read_text())
        r['control_precision'] = 'float64'
        p.write_text(json.dumps(r))
    train.write_text(json.dumps(t))
    evaluation.write_text(json.dumps(e))
    with pytest.raises((ValueError, FileNotFoundError)):
        MODULE.summarize(study)
