# Wuji Hand piano playing

Contributed by [Anna Mei](https://github.com/annamei1) ([#9](https://github.com/Frank-ZY-Dou/ndm-gym/pull/9)).

Two floating Wuji Hand 2 beta2 hands play Eden's 90 BPM arrangement of *Ode to Joy*: 15 melody notes and four held bass
notes, two pitches due together on 164 of the 230 control frames, 11.5 s with a one-second lead-in.
[Benchmark, commands and results](../../benchmarks/wuji_piano/README.md)

| Component | Configuration |
|---|---|
| Physics | MuJoCo-Warp at 200 Hz, gravity-compensated hands and 88 independently hinged passive keys |
| Controls | 20 finger targets and six wrist pose targets per hand at 20 Hz: 52 independent controls |
| Observation | 52 actuator positions, their last changes, 88 key depths and the previous action |
| Score | the current 88-key target and the next ten, without fingering or hand assignments |
| World model | `PianoDynamics` (`ndm_gym/worldmodels/piano.py`), task-specific and the same for both algorithms, not the library's acceleration or increment model: a polynomial fit of the fingertip geometry and hand-set contact constants, with residual networks for the actuators and the keys |
| Training | SuperTrack and SuperMimic through the hand trainers' updates, real transitions from 32 GPU worlds, 102,400 per run |
| Evaluation | the last checkpoint, 256 fixed starts including the nominal one, the whole song, without exploration |

Eden's precision, recall and F1 compare the measured key states with the score on each frame; a key is down at 87 % of
its travel. The event diagnostic counts attacks and releases at 50 ms resolution.

Eden F1 on the same 256 held-out starts (mean ± sample SD, seeds 0-2, one A100 40GB SXM4 per run), each algorithm's
configuration after the same settings screen:

| horizon | SuperTrack | SuperMimic |
|---:|---|---|
| 3 | `--st-reg 0.01`: 0.903 ± 0.015 | `--gamma 0.95 --bptt-decay 0.95`: 0.833 ± 0.079 |
| 8 | `--st-imag-noise 0.1 --st-reg 0.01`: 0.801 ± 0.027 | `--gamma 0.95 --bptt-decay 0.95 --critic-epochs 10`: 0.820 ± 0.135 |

SuperTrack at horizon 3 is the most accurate; at horizon 8 SuperMimic's mean is the higher. SuperTrack's seeds lie closer
together and its runs take less time at both horizons (7.2 and 7.7 min, SuperMimic's 8.7 and 10.4).

Anna Mei's study ran both starting configurations on one RTX 3060 (seeds 0-2, the same budget): SuperTrack
0.8490 ± 0.0705 and SuperMimic 0.4123 ± 0.2090.
[Every seed, the settings screen and the world model's speed](../../benchmarks/wuji_piano/results_a100.md)

```bash
bench=benchmarks/wuji_piano
run=data/wuji_piano/ode-to-joy
python "$bench/download_assets.py"
"$bench/run.sh" supertrack "$run/supertrack" --horizon 3 --st-reg 0.01
"$bench/run.sh" supermimic "$run/supermimic" --horizon 3 --gamma 0.95 --bptt-decay 0.95
"$bench/run.sh" replay "$run/replay" "$run/supertrack/seed0/last.pt" --eval-starts 32
```

[Setup and reproduction](../../benchmarks/wuji_piano/README.md#commands)
