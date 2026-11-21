# Quadruped motion tracking: results

Contributed by [Jionghao Wang](https://github.com/shanemankiw) ([#4](https://github.com/Frank-ZY-Dou/ndm-gym/pull/4),
[#14](https://github.com/Frank-ZY-Dou/ndm-gym/pull/14)).

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make a Unitree Go2 track the
retargeted motion capture of a Labrador ([the task](../../docs/tasks/quadruped.md), [the rules](README.md)). Tracking
results on this task are not reported here.

The benchmark's tasks:

| task | clip | length | trunk speed, mean / top |
|---|---|---|---|
| `walk` | `dog_quad_walk_002_1` | 6.0 s | 0.88 / 1.25 m/s |
| `run` | `dog_quad_run_002_0` | 5.5 s | 1.63 / 2.95 m/s |
| `play` | `dog_play_002_0` | 10.1 s | 0.87 / 2.15 m/s |
| `all` | the 65 clips, one policy | 488 s | |

```bash
benchmarks/go2_lifelike_tracking/run.sh <task> <seed> logs/go2_st --config go2-supertrack       # all; go2-supermimic
benchmarks/go2_lifelike_tracking/run.sh <task> <seed> logs/go2_st --config go2-supertrack-clip  # walk, run, play; go2-supermimic-clip
```
