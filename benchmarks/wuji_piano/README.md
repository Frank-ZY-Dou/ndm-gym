# Wuji Hand piano playing

Contributed by [Anna Mei](https://github.com/annamei1) ([#9](https://github.com/Frank-ZY-Dou/ndm-gym/pull/9)).

Two freely moving Wuji Hand 2 beta2 hands play Eden’s 90 BPM *Ode to Joy* on 88 passive keys in MuJoCo-Warp: the
melody and a held bass, 19 notes in an 11.5 s episode with a one-second lead-in. Each hand has six wrist pose axes and
20 finger joints, 52 independent controls at 20 Hz over 200 Hz physics. The policy sees the score's current frame and
the next ten, with no fingering and no hand assignment. Both algorithms train through the same task-specific world model,
`PianoDynamics` ([`ndm_gym/worldmodels/piano.py`](../../ndm_gym/worldmodels/piano.py)): a polynomial fit of the
fingertip geometry and hand-set contact constants locate the keys a fingertip can press, and two residual networks
correct the actuators' motion and the keys' depths. It is not the library's acceleration or increment model.

## Results

Each algorithm's configuration after the same settings screen; seeds 0-2, 102,400 control transitions per run, each run
alone on one A100 40GB SXM4, the same 256 held-out starts ([every seed, the screen and the world model's speed](results_a100.md)):

| configuration (`run.sh <algorithm> OUT <options>`) | Eden F1 | notes reached (of 19) | training per seed |
|---|---:|---:|---:|
| `supertrack --horizon 3 --st-reg 0.01` | 0.903 ± 0.015 | 18.53 | 7.2 min |
| `supermimic --horizon 3 --gamma 0.95 --bptt-decay 0.95` | 0.833 ± 0.079 | 17.42 | 8.7 min |
| `supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01` | 0.801 ± 0.027 | 16.57 | 7.7 min |
| `supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10` | 0.820 ± 0.135 | 15.94 | 10.4 min |

SuperTrack at horizon 3 is the most accurate, above SuperMimic at either horizon; at horizon 8 SuperMimic's mean F1 is
the higher. SuperTrack's seeds lie closer together and its runs take less time at both horizons (not engineered); an
earlier set of finals differs on the order at horizon 8 and on the seeds' spread
([both sets](results_a100.md#results)).

The training times are the contributed implementation's (eager updates), not engineered for speed. Anna Mei's study
ran both algorithms' starting configurations on one RTX 3060 (seeds 0-2, the same budget): Eden F1 0.8490 ± 0.0705 for
SuperTrack and 0.4123 ± 0.2090 for SuperMimic ([the contributed study](ode_to_joy_results.md));
there the scripted reference, with assigned fingers, scored 0.6325 ([note timing](note_timing_results.md)).
SuperMimic there ran with the library's default discount (0.99) and critic settings, untuned on this task, and its
critic overpredicts the return on every seed (mean bias +11.0, +14.5 and +60.5;
[critic diagnostics](ode_to_joy_results.md#critic-diagnostics)).
The settings both algorithms share were set in that study, where a pilot run of SuperTrack preceded its runs and
SuperMimic's followed them; the screen on the A100 page varied the horizon and three settings of each algorithm's own.

![SuperTrack's horizon-3 policy, seed 0, playing the arrangement from the nominal start](../../docs/media/wuji_piano/ode.webp)

![Target notes and the measured presses of that episode](../../docs/media/wuji_piano/ode-roll.webp)

## Task and scoring

The score is Eden's: 15 melody notes and four held bass notes; two pitches are due together on 164 of the 230 control
frames, one on 36 and none on 30, and four frames begin two notes together. A key is down at 87 % of its travel.
Eden's framewise precision, recall and F1 compare the whole keyboard with the score after every control step, a zero
denominator counting as 1, averaged over frames and then episodes. A note is reached when its pitch is down at least
once in its window. [Eden's score](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/songs.py) ·
[Eden's scorer](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/custom_terms/metrics.py)

Every run collects 20 rounds of 32 worlds × 160 control steps (102,400 transitions), with 90 world-model updates on
two-step windows and 50 policy updates per round, both in batches of 96. The final checkpoint plays the whole song from
256 fixed starts (one nominal, the others with every control perturbed by up to 0.20 of its scale), without exploration.

## Commands

On Linux with Python 3.12, a CUDA build of PyTorch, and FFmpeg (or the render extra's `imageio-ffmpeg`) for the video:

```bash
python -m pip install -e '.[hand,contacts,render]'
python benchmarks/wuji_piano/download_assets.py          # 57 files at pinned revisions, each checked against assets.sha256
bench=benchmarks/wuji_piano
run=data/wuji_piano/ode-to-joy
"$bench/run.sh" supertrack "$run/supertrack" --horizon 3 --st-reg 0.01                     # seeds 0-2
"$bench/run.sh" supermimic "$run/supermimic" --horizon 3 --gamma 0.95 --bptt-decay 0.95
# at horizon 8: supertrack --horizon 8 --st-imag-noise 0.1 --st-reg 0.01
#               supermimic --horizon 8 --gamma 0.95 --bptt-decay 0.95 --critic-epochs 10
"$bench/run.sh" scripted "$run/scripted"                  # the scripted reference: assigned fingers, solved key poses
python "$bench/summarize_ode_to_joy.py" "$run/supertrack"
python "$bench/note_events.py" "$run"/{supertrack,supermimic}/seed{0,1,2}/evaluation.npz "$run/scripted/evaluation.npz"
# replay a checkpoint (its nominal start recorded), render it with audio from the detected keys, plot its piano roll
"$bench/run.sh" replay "$run/replay" "$run/supertrack/seed0/last.pt" --eval-starts 32
hands=data/wuji_piano/wuji-description-c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95/hand2/hand2_beta2/body/mjcf
MUJOCO_GL=egl python "$bench/render_fur_elise.py" --song ode_to_joy --background light --hand-dir "$hands" \
  --keyboard data/wuji_piano/keyboard/piano.xml --rollout "$run/replay/seed0/rollout.npz" --out "$run/ode.mp4"
python "$bench/piano_roll_figure.py" "$run/replay/seed0/rollout.npz" "$run/ode-roll.webp"
# the world model as the simulator: MuJoCo-Warp's control step against the model's, at 1,024 and 4,096 worlds
python "$bench/speed_piano.py" --hand-dir "$hands" --keyboard data/wuji_piano/keyboard/piano.xml \
  --ckpt "$run/supertrack/seed0/last.pt" "$run/supermimic/seed0/last.pt" --out "$run/speed.json"
```

The runner passes trainer options through after the output directory (`--seeds 0`, `--horizon 8`, ...; `python
"$bench/train_ode_to_joy.py" --help`). Use a fresh output directory for each run. `WUJI_PIANO_ASSETS` sets another
asset root and `PYTHON` the interpreter. Each seed directory holds `history.json` (every round's validation and
timings), `results.json` (settings and the 256-start evaluation), `evaluation.npz` (every measured key mask),
`rollout.npz` (the nominal episode) and the checkpoints.

The hand models are Wuji Technology's (MIT), the keyboard is RoboPianist's through Eden's assets (Apache-2.0), and the
score and scoring follow Eden (Apache-2.0): [third-party notices](../../THIRD_PARTY_NOTICES.md#wuji-hand-piano).
