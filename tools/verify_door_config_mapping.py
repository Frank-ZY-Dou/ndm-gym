"""Fail closed on stale pinned provenance or Door parser/mapping drift.

Simulator-free; verifies declared values and coverage, not runtime equivalence.
Run: python3 tools/verify_door_config_mapping.py --repo .
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

from audit_door_hand_configs import audit


def verify(repo):
    repo = Path(repo)
    directory = repo / 'benchmarks/adroit_door/config_audit'
    pinned = json.loads((directory / 'upstream-69b818e.json').read_text())
    mapping = json.loads((directory / 'door-mapping3017.json').read_text())
    if audit(repo, pinned['upstream_commit']) != pinned:
        raise ValueError('Pinned upstream provenance differs from git objects')
    if mapping['upstream_commit'] != pinned['upstream_commit']:
        raise ValueError('Mapping upstream identity differs from provenance')
    spec = importlib.util.spec_from_file_location('door_configs', repo / 'ndm_gym/door_configs.py')
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)
    tree = ast.parse((repo / 'ndm_gym/train_door_online.py').read_text())
    parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parser')
    ns = dict(argparse=argparse, Path=Path, __doc__='Door configuration verification')
    exec(compile(ast.Module(body=[parser], type_ignores=[]), '<actual Door parser>', 'exec'), ns)
    if mapping['door_preset'] != configs.NAME:
        raise ValueError('Mapping preset name differs from actual preset')
    counts = {}
    for method in ('supertrack', 'supermimic'):
        row = mapping['methods'][method]
        upstream = pinned['configurations'][f'hand-{method}-accurate']['expanded']
        if set(row['parameters']) != set(upstream):
            raise ValueError(f'{method}: upstream argument coverage differs')
        args = ['--method', method, '--config', configs.NAME]
        for name, value in [('model', 'model'), ('reference', 'reference'),
                            ('bootstrap', 'bootstrap'), ('output', 'out')]:
            args.extend(['--' + name, value])
        resolved = json.loads(json.dumps(vars(configs.parse_args(ns['parser'](), args)), default=str))
        if resolved != row['door_resolved']:
            raise ValueError(f'{method}: actual parser differs from recorded Door values')
        counts[method] = dict(supported=0, adapted=0, unavailable=0)
        for name, parameter in row['parameters'].items():
            status = parameter['status']
            if status not in counts[method] or not parameter['reason'].strip():
                raise ValueError(f'{method}/{name}: invalid status or missing reason')
            if parameter['upstream'] != upstream[name]:
                raise ValueError(f'{method}/{name}: stale upstream value')
            option = parameter['door_option']
            if option is not None and (option not in resolved or parameter['door_value'] != resolved[option]):
                raise ValueError(f'{method}/{name}: stale mapped Door option')
            counts[method][status] += 1
    return dict(status='PASS', upstream_commit=pinned['upstream_commit'], parameters=counts,
                limitation='Declaration consistency only; runtime behavior/equivalence not certified')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'))
    print(json.dumps(verify(p.parse_args().repo), indent=2))
