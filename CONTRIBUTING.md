# Contributing

## Setup

Install as in the [README](README.md), then `pip install -e ".[dev]"` and `pre-commit install`.

## Claiming a task

If you have a task in mind, claim it before it is finished: open a draft pull request that says what you plan to do. We can
then look at it early, and others see that it is taken, which avoids duplicate work.

## Checks

- `make lint`: ruff, also run by CI on every push.
- `make test-fast` before every commit; `make test` before merging changes to `ndm_gym/core`,
  `ndm_gym/worldmodels` or `ndm_gym/algos`.
- `make docs`: the documentation site (`mkdocs.yml`; the guides in `docs/`, the API reference from the docstrings), also
  built by CI.

## Efficiency

We aim for the most efficient implementation of everything here and ask the same of contributions: please build on the
fastest paths this repository has measured, or on a faster one you have found and measured yourself.

- Training: the speed options (`ndm-train --help` notes for each whether it changes results), combined as the named
  configurations combine them (`ndm_gym/configs.py`, [Recommended settings](README.md#-recommended-settings)).
- World models as simulators: their fastest inference path (bf16, one CUDA graph per step, the fused kernels of
  `benchmarks/ndm_speed`).
- A review goes faster when a new component or option comes with its speed beside the path it replaces: the time per step
  or per iteration, the GPU type and the settings.
- A speed-up that changes results, even by rounding, is best kept behind a flag whose help says so.

## Visualizations

Figures and videos share the repository's look, so that results read alike across tasks:

- Renders come from the repository's tools (`tools/render.py` and the task renderers `render_hoi.py`, `render_hand.py`,
  `render_musclevae.py`), which add the NDM Gym mark. A task taken from another project keeps that project's look (for
  example MuscleVAE's bones, red muscles and checkered floor), so that its results can be set beside the original's.
- The README shows policies trained here, one per task: the configuration with the lowest tracking error, in a window where
  the episode does not end, captioned with its configuration. Reference motions and other projects' policies belong in the
  documentation.
- Charts follow `tools/make_speedup_chart.py`: drawn at twice their display size and saved as lossless WebP. README videos are
  WebP previews (`tools/compare_videos.py`); MP4s and other large media stay out of the repository.
- LAFAN1 renders carry its attribution and non-commercial note (CC BY-NC-ND 4.0).
- A sample frame or clip in the pull request, before a whole set is rendered, saves rework.

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
