# Reference backend validation protocol

Fixed before running the baseline: Allegro right hand, 50 g / 5 cm cube,
MuJoCo 2 ms substeps, 50 Hz position targets. Uniform SO(3) goals, no cube
symmetry reduction. Seeds 0, 1, 2, 20 episodes each, maximum 250 steps.
Success requires angle <= 0.2 rad for 10 consecutive steps. Failure means
object displacement > 0.15 m from its initial position or nonfinite state.
Report success/drop fractions and final angular error (radians) across every
episode, including failures; no checkpoint selection or episode filtering.
Compare midpoint controls against independent uniform random controls.
These are simulator diagnostics, not a learned-controller benchmark.
Raw JSON stays outside the repository. No tuning on these seeds.
