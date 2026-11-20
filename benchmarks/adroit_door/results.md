# Native Adroit Door control replay (development feasibility)

Contributed by [Yifan Kang](https://github.com/YIFANK) ([#3](https://github.com/Frank-ZY-Dou/ndm-gym/pull/3)).

This is reference preparation, not a trained policy result or a performance
comparison. Native hand; no Allegro retargeting, attachments, object actuators or
external applied forces. Bounded supervised residual training results are recorded below. Existing defaults are unchanged.

Source: DAPG `9a1de23df6574e1b94e1da742e3326eb1f61c013`,
`dapg/demonstrations/door-v0_demos.pickle`, SHA256
`d6ec9a090261f901e5c9bf10082799e30c8d34c3f28aef4026e492a1aefc96c3`.
[Primary source](https://github.com/aravindr93/hand_dapg/tree/9a1de23df6574e1b94e1da742e3326eb1f61c013).
DAPG repository Apache-2.0; no data or model assets are redistributed here.
Gymnasium-Robotics separately supplies native Adroit assets with its own notices.

Development inspection replayed all 25 demonstrations. Eight reached 1.35 rad;
clip 23 was selected after this inspection for its opening and lower measured
penetration than clip 11. This is development selection, not a held-out result.
Do not use these clips as independent evaluation evidence.

Verified physics: Gymnasium-Robotics 1.3.1, MuJoCo 3.1.6, Python 3.9.6,
2 ms physics / five substeps per control (100 Hz). Clip 23: 226 controls,
final door angle 1.536375553 rad, maximum 1.582222727 rad;
maximum latch angle 1.676481657 rad. Two substep runs had zero qpos discrepancy.
520 of 1130 substeps had contacts. Native soft constraints allow measured joint
limit excess 0.048877443 (ARRy, rad) and penetration 4.406944 mm. These are
observations, not hard-limit/zero-penetration certification or invented acceptance
tolerances. Model/controller parameters were not adjusted.

Historical observations are not synchronized complete states: the initial state
and first hand observation differ, and per-frame qpos[0]/qvel are absent.
Door observation RMSE is 0.037483721 rad (max 0.169990095); latch RMSE
0.020168960 rad (max 0.052608175). Exported references are explicitly
**replay-derived**, including full initial and post-control qpos/qvel.

The new exporter measured 0.07320 s physics loop / 0.32388 ms per control on
Mac mini CPU, one OMP/OpenBLAS thread. This includes per-substep audits, excludes
loading/serialization and is one measurement, not an optimized training benchmark.
No GPU or learned-model speed claim. Future training must use upstream measured
speed paths and report GPU/settings and time per iteration.

Supported-runtime comparison: Python 3.12.3 / MuJoCo 3.15.0, CPU only,
same restored state and 226 native controls on the exported original model.
All states remained finite; final door 1.539469218 rad, maximum 1.580814108 rad,
latch maximum 1.676828541 rad. Door RMSE versus 3.1.6 is 0.002534777 rad;
maximum qpos difference is 0.067876976. This establishes opening under the
new engine, not engine equivalence or a validated training integration.
Obsolete XML capacity attributes were removed for the new loader; physical
parameters, controls and force arrays were unchanged.

The native analytic `DoorTask` now consumes the same `(qpos, qvel)` state tuple
as `MjcfWorldModel`. It maps normalized policy actions to the model's physical
actuator ranges for both physical and imagined transitions. Actual-model CPU
admission checked all 226 mappings (maximum float32-versus-float64 control
difference 1.1921e-7), observation/goal shapes, and finite action gradients through
an untrained frozen world model. This is interface admission, not learned-model
accuracy or policy feasibility under perturbed starts. Its eager reward head
measured 0.0919 ms per batch of 226 states, averaged over 100 calls, CPU one
thread; this is not a training or GPU speed benchmark.

A lossless WebP reference preview was generated with
`tools/render_adroit_door.py`, using the common renderer's NDM Gym mark and
native Adroit scene. It is labeled as a physical replay reference, not a trained
policy; large media and reference arrays remain outside the repository.

Development runner (round2916): `python -m ndm_gym.train_door` now binds the
native physics, registered residual algorithm and final-checkpoint physical
evaluation. Frozen integration protocol: seed2916, 100 model updates, batch32,
horizon4, zero policy updates, eager fp32 CPU one thread, MuJoCo3.15.0.
Model-update median1.779ms, total0.25688s; each complete226-control physical
evaluation took0.168–0.173s. These are correctness-path measurements, not a GPU
speed claim or evidence of learned feedback. The five starts (ARRz0,±.005,±.01rad)
all opened, finaldoor1.53035–1.53947rad. Nominal doorRMSE1.67e-9rad;
perturbed doorRMSE.00605–.00752rad. Actual opening is scored separately.

Historical correction: private round2908/2910 pilot scripts perturbed index1,
ARRx, despite their ARRz labels. Those measurements remain preserved as ARRx
evidence. The new runner indexes ARRz by name (index3), with a component test.
The closed negative model-gradient diagnostic does not establish useful physical
policy gradients. The supervised baseline below uses analytic feedback targets
instead. Efficient GPU configuration/measurements and current full merge checks
remain uncertified.


## Frozen three-seed supervised baseline

Independent seeds 29301/29302/29303 each fit 1500 minibatches of 128 targets,
collected in native physics from ARRz offsets ±0.005/±0.01 rad with fixed joint
feedback gain 0.3. Final checkpoints only; all seeds reported, with no superiority
selection threshold. The policy retains the replay-action prior and learns bounded
residuals. MuJoCo 3.15.0, Torch 2.11.0, fp32 eager CPU, one thread, CUDA hidden;
no GPU efficiency claim. Training took 2.0202/1.9705/1.9643 seconds, excluding
collection and evaluation. Exact commands and source hashes are retained in the
external frozen protocol `protocol2923.json`.

Evaluation uses all 226 controls at ARRz 0, ±0.0075, ±0.015 rad, without resets
inside the episode. These are held-out initial offsets of the same reference clip,
not held-out motions. Every seed opened 5/5 episodes with finite states. Fresh
checkpoint deployment reproduced every qpos/qvel/action exactly for all 15 episodes.

| Seed | Hand RMSE (rad) | Door RMSE (rad) | Latch RMSE (rad) |
|---|---:|---:|---:|
| 29301 | .00377011 | .00416773 | .00196904 |
| 29302 | .00513703 | .00138272 | .00384658 |
| 29303 | .00384766 | .00526962 | .00448036 |
| Mean ± sample SD | .00425160 ± .00076778 | .00360669 ± .00200327 | .00343199 ± .00130598 |

The same-offset zero-residual baseline opens 5/5; its hand/door/latch RMSE is 0.00278775/0.00417984/0.00228456 rad. Tracking changes are mixed; no method-superiority or broad robustness claim. Three seeds establish this bounded supervised baseline only, not completion of all contribution checks.

These preliminary supervised runs used `wm_updates=0` and `policy_updates=0`.
They do not complete SuperMimic/SuperTrack world-model training or the final
model-based three-seed requirement. That work remains pending. Bit-exact
redeployment is a reproducibility observation, not an upstream requirement.

Round2934: implemented --mixed-replay, exact half-batch windows per source, complete-episode holdouts per source before normalization/optimization. Default paths and native Adroit unchanged. Tests7PASS1.08s, CPU only. CPU1thread budget120s/4GiB; remote data/cache/tmp under /data/YifanKang. Preflight /data2.6Tfree,RAM535Giavailable,swap33Mi; unrelated jobs preserved.
Preregistered model-only seed2934,1000WM/0policy/0supervised updates,batch32/H4,final only. Successful-reference heldout H4/H16 doorRMSE .01183765/.05075030 vs persistence .06640600/.26047397rad. Failed-policy heldout .00604863/.01670800 vs .00091903/.00148161rad. Opening dynamics retained but failure predictions worse than persistence. No policy pilot or repeated formal seeds. Audit all valid windows saved audit2934.json. Initial lint failed missing PATH ruff; corrected environment check pending. Current fullfast not rerun; prior2932 historical. FullGPU/efficiency unresolved; Draft#3 untouched.
Next: preregister bounded predicted-versus-physical action-gradient branching audit on heldout success/failure states, verify exact restoration and compare unchanged/equal-norm random controls before another learning pilot. No review_ready or all-CONTRIBUTING claim.
