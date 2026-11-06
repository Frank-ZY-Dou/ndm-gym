# Contributing

## Setup

Install as in the [README](README.md), then `pip install -e ".[dev]"` and `pre-commit install`.

## Checks

- `make lint`: ruff, also run by CI on every push.
- `make test-fast` before every commit; `make test` before merging changes to `ndm_gym/core`,
  `ndm_gym/worldmodels` or `ndm_gym/algos`.
- `make docs`: the documentation site (`mkdocs.yml`; the guides in `docs/`, the API reference from the docstrings), also
  built by CI.

## Conventions

- Components sit behind the interfaces in `ndm_gym/algos/common.py` (`Algo`: `world_model_cls`, `setpoint`, `wm_loss`,
  `policy_update`) and `ndm_gym/worldmodels/`; a new algorithm is registered in `ndm_gym/algos/__init__.py`.
- An optional change to an algorithm goes behind a flag whose default keeps the current behavior. The equivalence
  tests against `third_party/supermimic` (its `data/` is not in git: `ln -s ../../data third_party/supermimic/data`) and the SuperTrack golden
  tests keep passing unchanged.
- Every new component comes with a test in `tests/`.
- The library imports without the simulator, as a `pip install` of the repository gives it: a command takes Isaac Lab's launcher from
  `ndm_gym.sim.launcher`, and a module that is not a simulator imports the simulator inside the functions that use it
  (`tests/test_packaging.py`).
- A study lives in `benchmarks/<study>/` with its results; its rules and selection criteria are fixed before the runs.
- Logs, checkpoints, recordings and videos stay out of the repository; the LAFAN1 motions (CC BY-NC-ND 4.0) are not
  redistributed.
