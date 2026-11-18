"""Actual-capacity fresh-process recovery across the WM-only/policy boundary."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def identical(a, b, path='root'):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            identical(a[key], b[key], path+'/'+str(key))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            identical(x, y, path+'/'+str(index))
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, path


def main(argv=None, *, training_module='ndm_gym.train_door_full', training_args=()):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for method in ('supertrack', 'supermimic'):
        base = [sys.executable, '-m', training_module, '--model', str(a.model),
                '--reference', str(a.reference), '--native-library', str(a.native_library),
                '--method', method, '--seed', '3065', '--workers', '8', *training_args]
        for stage, steps in [('prefix', 301), ('resumed', 320), ('continuous', 320)]:
            output = a.output/f'{method}-{stage}'
            cmd = base + ['--output', str(output), '--development-steps', str(steps)]
            if stage == 'resumed':
                cmd += ['--resume', str(a.output/f'{method}-prefix/latest.pt')]
            with (a.output/f'{method}-{stage}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output/'result.json').read_text())
            assert result['status'] == 'development_prefix'
            results.append(dict(method=method, stage=stage, counts=result['counts'],
                                wall_seconds=result['invocation_wall_seconds']))
            (a.output/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
        resumed = torch.load(a.output/f'{method}-resumed/latest.pt', map_location='cpu', weights_only=False)
        continuous = torch.load(a.output/f'{method}-continuous/latest.pt', map_location='cpu', weights_only=False)
        for key in ('protocol', 'iteration', 'algo', 'collector', 'replay', 'torch_rng', 'cuda_rng'):
            identical(resumed[key], continuous[key], key)
        del resumed, continuous
        print(method+' full-state recovery PASS', flush=True)
    (a.output/'result.json').write_text(json.dumps(dict(status='PASS', runs=results,
        scope='320-iteration development prefixes at full capacity; not full-budget seeds'), indent=2)+'\n')


if __name__ == '__main__':
    main()
