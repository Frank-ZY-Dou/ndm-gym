# Contributing

## Setup

Install as in the [README](README.md), then `pip install -e ".[dev]"` and `pre-commit install`.

## Claiming a task

If you have a task in mind, claim it before it is finished: open a draft pull request that says what you plan to do. We can
then look at it early, and others see that it is taken, which avoids duplicate work.

## Common problems

What most often holds back a pull request here, and what to do instead.

**Comparisons**

- Treat both algorithms the same: the same tuning rounds and budget, options, seeds and evaluation starts. Settings tuned
  for one algorithm and reused by the other are not yet a comparison; give the other a round of the same size.
- A configuration said to follow another task's settings follows them exactly, or the text names each change.
- Stop runs by one rule fixed before the runs and the same for both algorithms (a target, or a plateau of the main error),
  with a cap. A fixed budget is the measure only when the comparison says so; a run still improving at its end gives no
  final number.
- Report every seed. Evaluate the last checkpoint, or one chosen by a rule fixed beforehand, never one picked on the test
  starts.
- Define success as the original benchmark does, or call it a proxy.

**Timing**

- Time each run alone on its GPU. Times are reported from A100- or H100-class GPUs; we can re-time a contribution on ours.
- Save checkpoints often enough to read the time to the target, and say what the clock includes.
- Call a configuration "fastest" only with a per-phase time table of that exact path; otherwise write "time to the target,
  not engineered".

**Results text**

- Results only: no plans or changes of plan, machine names, driver versions, local paths or links to private forks.
- Give every number its seeds and GPU type. List SuperTrack first, both numbers side by side, in neutral words: no
  verdicts, no remark about one algorithm alone.
- Open a task's section with its settings table (goal | configuration | result) and one conclusion line; details go to the
  results page.
- Describe the world model as it is, and say when it is specific to the task rather than the library's.

**Code**

- Rebase on the current `main` and run `make test-fast` there: a script written against an older interface can break.
- Declare new dependencies in `pyproject.toml`'s extras; tests that need them use `pytest.importorskip`.
- Implement or refuse every option a script accepts; none is silently ignored. Missing data or an unsupported setting
  raises an error instead of falling back to a default.
- Pin downloads to a revision and check their SHA-256 sums.
- Add tests that run on the CPU (a stub world model is fine).
- Keep scripts that do not use the library out of a benchmark, or in a folder of their own.

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
  `render_musclevae.py`, `render_robokit.py`, `render_mbrl.py`), which add the NDM Gym mark. A task taken from another
  project keeps that project's look (for
  example MuscleVAE's bones, red muscles and checkered floor), so that its results can be set beside the original's.
- The README shows policies trained here, in one grid per task: per case the run with the lowest tracking error, in a
  window where the episode does not end, captioned with its configuration. Reference motions and other projects' policies belong in the
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
