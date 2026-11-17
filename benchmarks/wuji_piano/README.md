# Wuji Hand piano playing

This draft claims a score-conditioned piano-playing benchmark for NDM Gym, using [Eden's piano task](https://embodied-ai-nexus.github.io/Eden/tasks/manipulation/) as a reference.

- Start with one Wuji Hand 2 in MuJoCo-Warp and the opening phrase of *Twinkle Twinkle Little Star*. Check the hand and keyboard asset terms before adding assets.
- Simulate spring-loaded keys and verify press and release detection with a scripted rollout, including repeated notes.
- Train a policy conditioned on upcoming notes and model key travel in imagined rollouts. Start with SuperTrack and evaluate SuperMimic on the same score and scene.
- Report note precision, recall, and F1 at a stated onset tolerance, plus onset error and training time. Include commands to train and evaluate, tests for note timing, and a target-versus-played piano roll.
