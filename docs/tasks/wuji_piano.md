# Wuji Hand piano playing

Two floating Wuji Hand 2 beta2 hands play a 90 BPM arrangement of *Ode to Joy*. It contains 15 melody attacks and four bass attacks. Two notes are due during 164 of the 230 control frames. The full episode is 11.5 seconds, including a one-second lead-in. [Benchmark and recorded rollout](../../benchmarks/wuji_piano/README.md)

| Component | Configuration |
|---|---|
| Physics | MuJoCo-Warp at 200 Hz, gravity-compensated hands and 88 independently hinged passive keys |
| Controls | 20 finger targets and six wrist pose targets per hand at 20 Hz |
| Observation | 52 actuator positions, their last changes, 88 key depths, and the previous action |
| Score | Current 88-key target and the next ten targets, without fingering or hand assignments |
| Learned dynamics | Actuator motion, approximate fingertip geometry and a learned residual predicting the physical key depths |
| Training | NDM Gym hand SuperTrack and SuperMimic, three imagined steps, real transitions from 32 GPU worlds |
| Evaluation | Last checkpoint, 256 fixed starts including nominal, complete song, policy exploration disabled |

Eden precision, recall and F1 compare measured key states with the target on each frame. A key is down at 87% travel. The separate event diagnostic measures individual attacks and releases at 50 ms resolution. On this song SuperTrack had the highest framewise F1 (0.849). The scripted reference had the highest fraction of timely paired attacks (50.0%). The held-out starts test sensitivity to initial hand placement on one score. [Event timing and reference controller](../../benchmarks/wuji_piano/note_timing_results.md)

From a checkout with the assets installed, run one training configuration or reproduce a checkpoint rollout:

```bash
bench=benchmarks/wuji_piano
run=data/wuji_piano/ode-to-joy
python "$bench/download_assets.py"
"$bench/run.sh" supertrack "$run/supertrack"
"$bench/run.sh" replay "$run/replay" "$run/supertrack/seed0/last.pt"
```

The benchmark includes the exact environment, perturbation distribution, training times, per-seed results, and audio rendering commands. [Setup and reproduction](../../benchmarks/wuji_piano/README.md#eden-ode-to-joy-with-52-independent-controls)
