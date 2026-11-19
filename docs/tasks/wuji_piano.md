# Wuji Hand piano playing

| Goal | Configuration | Result |
|---|---|---|
| Play Eden’s *Ode to Joy* with melody and bass | Two floating Wuji Hand 2 hands, 52 controls, 20 rounds, three seeds on an RTX 3060 | SuperTrack 0.8490 ± 0.0705 and SuperMimic 0.4123 ± 0.2090 Eden F1 |

On the same 256 starts per checkpoint, both due keys were down during 69.4% of the two-note frames with SuperTrack, 10.5% with SuperMimic and 23.8% with the scripted reference. [Per-seed results and training settings](../../benchmarks/wuji_piano/ode_to_joy_results.md) · [Attack and release timing](../../benchmarks/wuji_piano/note_timing_results.md)

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

Eden precision, recall and F1 compare measured key states with the target on each frame. A key is down at 87% travel. The separate event diagnostic measures individual attacks and releases at 50 ms resolution. The fractions of timely paired attacks were 44.7% for SuperTrack, 1.6% for SuperMimic and 50.0% for the scripted reference (same RTX 3060 evaluation). The held-out starts test sensitivity to initial hand placement on one score. [Event timing and reference controller](../../benchmarks/wuji_piano/note_timing_results.md)

From a checkout with the assets installed, run one training configuration or reproduce a checkpoint rollout:

```bash
bench=benchmarks/wuji_piano
run=data/wuji_piano/ode-to-joy
python "$bench/download_assets.py"
"$bench/run.sh" supertrack "$run/supertrack"
"$bench/run.sh" replay "$run/replay" "$run/supertrack/seed0/last.pt"
```

The benchmark includes the exact environment, perturbation distribution, training times, per-seed results, and audio rendering commands. [Setup and reproduction](../../benchmarks/wuji_piano/README.md#eden-ode-to-joy-with-52-independent-controls)
