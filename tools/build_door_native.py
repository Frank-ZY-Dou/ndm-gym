"""Build the optional native loop against this interpreter's MuJoCo library.

Writes only the explicit output path; use an experiment/cache directory outside
the source tree. No fast-math or solver substitutions are enabled.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    import mujoco
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(mujoco.__file__).parent
    libraries = sorted(root.glob('libmujoco.*'))
    if len(libraries) != 1:
        raise RuntimeError(f'Expected one MuJoCo shared library: {libraries}')
    source = Path(__file__).resolve().parents[1] / 'ndm_gym/sim/rigid/native/door_batch.cc'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['c++', '-std=c++17', '-O3', '-fPIC', '-ffp-contract=off',
           '-dynamiclib' if sys.platform == 'darwin' else '-shared',
           '-I'+str(root/'include'), str(source), str(libraries[0]),
           '-Wl,-rpath,'+str(root), '-o', str(a.output)]
    subprocess.run(cmd, check=True)
    print(a.output)


if __name__ == '__main__':
    main()
