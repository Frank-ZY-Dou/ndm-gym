#!/bin/bash
# setup_env.sh <this repository's checkout>: the benchmark's two environments under MBRL_BASE (default ~/mbrl_c2), with
# uv: (a) Python 3.10 with mbrl-lib's own pins and MuJoCo 2.3.3; (b) Python 3.12 with MuJoCo 4.0.1 and MuJoCo Warp 4.0.1,
# which imports mbrl-lib's checkout (3f93ccc, unmodified) and ndm_gym from the given checkout. Then the local-coordinate
# copies of Hopper and Walker2d, made in (a) into the checkout's data/mbrl_models. Nothing is written outside MBRL_BASE and
# that directory. Every package is held at the version of env_a.txt and env_b.txt beside this script (the environments'
# lists of the runs in results.md). Needs uv at MBRL_BASE/uv/bin/uv and the network.
set -euo pipefail
trap 'echo SETUP_FAILED at line $LINENO' ERR
B=${MBRL_BASE:-$HOME/mbrl_c2}
CODE=$(cd "$1" && pwd)
PINS=$(cd "$(dirname "$0")" && pwd)
[ -f $CODE/ndm_gym/wmbench/plan_mbrl.py ] || { echo "setup_env.sh: $CODE is not this repository's checkout"; exit 2; }
mkdir -p $B && cd $B
unset PYOPENGL_PLATFORM MUJOCO_GL DISPLAY
export HOME=$B/home XDG_CACHE_HOME=$B/cache/xdg XDG_DATA_HOME=$B/home/.local/share XDG_CONFIG_HOME=$B/home/.config
export UV_HTTP_TIMEOUT=300 UV_CACHE_DIR=$B/cache/uv UV_PYTHON_INSTALL_DIR=$B/uv/python UV_LINK_MODE=copy UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed
export PIP_CACHE_DIR=$B/cache/pip PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
mkdir -p $HOME $XDG_CACHE_HOME $XDG_DATA_HOME $XDG_CONFIG_HOME src
UV=$B/uv/bin/uv
[ -x $UV ] || { echo "setup_env.sh: no uv at $UV"; exit 2; }
COMMIT=3f93cccfc8d635f74e335a2f07aab6e9a48fc021
echo "== mbrl-lib at $COMMIT"
[ -d src/mbrl-lib/.git ] || git clone -q https://github.com/facebookresearch/mbrl-lib src/mbrl-lib
git -C src/mbrl-lib checkout -q $COMMIT
test "$(git -C src/mbrl-lib rev-parse HEAD)" = $COMMIT
test -z "$(git -C src/mbrl-lib status --porcelain)"

echo "== environment (b): Python 3.12"
[ -d venv_b ] || $UV venv -q --python 3.12 venv_b
$UV pip install -q --python venv_b/bin/python -c $PINS/env_b.txt torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
$UV pip install -q --python venv_b/bin/python -c $PINS/env_b.txt numpy==1.26.4 hydra-core==1.3.8 omegaconf==2.3.1 gymnasium==0.26.3 gym==0.26.2 \
  mujoco==4.0.1 mujoco-warp==4.0.1 warp-lang==1.19.0 "etils[epath]==1.15.0" absl-py==2.5.1 dm-control==1.0.52 scipy==1.17.1 \
  sk-video==1.1.10 termcolor==3.3.0 imageio==2.38.1 matplotlib==3.11.2 tqdm==4.70.1 pyyaml==6.0.3 pytest==9.2.0
P=$(venv_b/bin/python -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$B/src/mbrl-lib" > $P/mbrl_lib_checkout.pth
echo "$CODE" > $P/ndm_gym_code.pth
MUJOCO_GL=disable venv_b/bin/python -I -B -c "import torch, warp, mujoco, mujoco_warp, numpy, hydra, omegaconf, gymnasium, dm_control, mbrl, ndm_gym; print('torch', torch.__version__, torch.version.cuda, 'warp', warp.__version__, 'mujoco', mujoco.__version__, 'ndm_gym', ndm_gym.__file__)"
$UV pip freeze --python venv_b/bin/python > freeze_b.txt
diff <(sort freeze_b.txt) <(sort $PINS/env_b.txt) > /dev/null || echo "NOTE: environment (b) differs from env_b.txt (diff freeze_b.txt)"

echo "== environment (a): Python 3.10, pip 24.0, mbrl-lib's own pins"
[ -d venv_a ] || $UV venv -q --seed --python 3.10 venv_a
venv_a/bin/python -I -m pip install -q "pip==24.0"
venv_a/bin/python -I -m pip install -q -c $PINS/env_a.txt torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple  # torch from its own index, what it needs at the pinned versions from PyPI
venv_a/bin/python -I -m pip install -q -c $PINS/env_a.txt -r src/mbrl-lib/requirements/main.txt numpy==1.26.4 mujoco==2.3.3 importlib_resources==7.1.0
P=$(venv_a/bin/python -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$B/src/mbrl-lib" > $P/mbrl_lib_checkout.pth
venv_a/bin/python -I -B -c "import torch, mujoco, numpy, hydra, omegaconf, gymnasium, mbrl; print('torch', torch.__version__, torch.version.cuda, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
venv_a/bin/python -I -m pip freeze > freeze_a.txt
diff <(sort freeze_a.txt) <(sort $PINS/env_a.txt) > /dev/null || echo "NOTE: environment (a) differs from env_a.txt (diff freeze_a.txt)"
test -z "$(git -C src/mbrl-lib status --porcelain)"

echo "== the local-coordinate copies of Hopper and Walker2d (MuJoCo 2.3.3)"
(cd $CODE && $B/venv_a/bin/python -I -B tools/convert_mbrl_models.py --out data/mbrl_models)
echo SETUP_DONE
