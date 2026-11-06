# In-hand object reorientation (reference backend)

This experimental task adds an Allegro Hand v3 right hand and a freely moving
50 g, 5 cm cube. It currently provides a serial MuJoCo CPU reference simulator,
pure-tensor task objectives, and reproducible midpoint/random control diagnostics.
It does **not** yet provide SuperTrack/SuperMimic training or a GPU batched backend.
The existing humanoid training commands do not select this task.

## Setup and diagnostics

```bash
pip install -e ".[dev]"
bash tools/download_allegro.sh
python tools/evaluate_reorientation_baselines.py --out logs/reorientation/baselines.json
python -m pytest tests/test_reorientation.py
```

The downloader pins Menagerie revision `98a70c569c4c1b6da467a20012f766ee1f8dc521`.
Its Allegro assets are BSD-2-Clause, retained under ignored `data/menagerie/`
with their license; no meshes are redistributed. MuJoCo alone is sufficient,
including on macOS. Isaac Lab is not needed by this reference backend.

## Task contract

The palm is fixed in Menagerie's supplied pose. Each episode starts at midpoint
joint positions, zero velocities, and cube center `(0, 0, 0.1)` m with identity
orientation. Goals are uniformly sampled SO(3) quaternions in scalar-first order.
Cube symmetries are not equivalent: the task models a labeled cube.
Reset has no settling phase; initial free motion is part of the episode.

Actions are 16 absolute joint position targets normalized to `[-1, 1]`, clipped
and mapped to actuator limits. Each control step runs ten 2 ms physics steps.
State tensors are `(N, 45)`: 16 joint positions, 16 velocities, object position
(3), quaternion (4, wxyz), linear velocity (3), angular velocity (3). The reference
backend has N=1. The target is available as `sim.goal`; it is not part of state.

Reward is `exp(-angular_error) - 0.001 * mean(action**2)`; drops receive -1.
Success requires error <= 0.2 rad for 10 consecutive control steps. Moving the
object center more than 0.15 m from its initial position is a drop; nonfinite
pose/error is a failure. Success and drop terminate; 250 steps truncate.
Drops override success. Terminal transitions are returned before any reset,
and the caller must reset after a terminal/truncated step.

The validation protocol and results live in `benchmarks/in_hand_reorientation/`
in the repository. These diagnostics test simulation/episode semantics and do
not establish that a learned policy can reorient an object. Integration with
world-model training, grasp initialization studies, GPU batching and learned
controller comparisons remain future work.
