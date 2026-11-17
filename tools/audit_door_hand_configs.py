"""Resolve pinned Hand parser defaults without importing a simulator or torch.

Run from any directory: python3 tools/audit_door_hand_configs.py --repo . --output FILE.
Reads git objects, never checks out or overwrites upstream or Door sources.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


def audit(repo, revision):
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', revision], text=True).strip()
    sources = {}

    def read(path):
        value = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{path}'], text=True)
        sources[path] = hashlib.sha256(value.encode()).hexdigest()
        return value

    ns = dict(argparse=argparse, os=os, REPO='<upstream-root>')
    parser = argparse.ArgumentParser()
    for path, function in [('ndm_gym/train.py', 'add_train_args'),
                           ('ndm_gym/algos/hand_fast.py', 'add_hand_fast_args')]:
        tree = ast.parse(read(path))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
        ns[function](parser)
    hand = ast.parse(read('ndm_gym/train_hand.py'))
    defaults = next(ast.literal_eval(n.value) for n in hand.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HAND_DEFAULTS' for t in n.targets))
    parse = next(n for n in hand.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
    for n in parse.body:
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
                and n.value.func.value.id == 'parser' and n.value.func.attr == 'add_argument'):
            exec(compile(ast.Module(body=[n], type_ignores=[]), '<hand arguments>', 'exec'), dict(ns, parser=parser))
    parser.set_defaults(**defaults)
    next(a for a in parser._actions if a.dest == 'algo').required = False
    cfg = {}
    exec(compile(read('ndm_gym/configs.py'), '<pinned configs>', 'exec'), cfg)
    cfg['add_config_arg'](parser)
    result = {}
    for method in ('supertrack', 'supermimic'):
        name = f'hand-{method}-accurate'
        values = vars(parser.parse_args(cfg['expand'](['--config', name, '--clip', '<required-reference>'], 'train_hand')))
        result[name] = dict(expanded=values, measured=cfg['CONFIGS']['train_hand'][name][1])
    for path in ('README.md', 'ndm_gym/algos/hand_algos.py', 'ndm_gym/worldmodels/hand.py',
                 'ndm_gym/policies/hand_policy.py', 'docs/algorithms/xmimic.md', 'docs/algorithms/nablamimic.md'):
        read(path)
    return dict(upstream_commit=commit, source_sha256=sources, configurations=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--revision', default='69b818e')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.repo, a.revision), indent=2, sort_keys=True) + '\n')
