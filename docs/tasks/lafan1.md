# LAFAN1 motion tracking

The Unitree G1 (29 DoF, `ndm_gym/robots/g1.py`) tracks 40 retargeted LAFAN1 clips using SuperMimic's setup
(`ndm_gym/tasks/tracking.py`). The simulator and world models share observations, analytic reward and terminations.
Domain randomization covers friction, joint calibration and the torso's center of mass. `--clips` selects clips from
a file (one name per line); `--motion_dir` reads another directory of retargeted CSVs.

```bash
bash tools/download_data.sh    # LAFAN1 G1 retargets (CC BY-NC-ND 4.0) and the G1 description, checked against tools/download_data.sha256
ndm-train --algo supertrack --log_dir logs/st
ndm-eval --run_dir logs/st --watch
```

Selected configurations and results: [README](../../README.md#lafan1-motion-tracking), [`results.md`](../../benchmarks/g1_lafan1_tracking/results.md),
[ablations](../../benchmarks/g1_lafan1_tracking/ablations.md).
