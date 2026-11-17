# Frozen online Door study, round2943

## Corrected model-contract replication (round2967)

The current model-contract evidence supersedes the historical SuperTrack table
below: SuperTrack uses `accel-squared-v2`, seeds29701–29703. SuperMimic's
`increment-squared-v1` source was unchanged; its seeds29601–29603 are retained,
not rerun or relabeled. Seed sets are independent, not paired across methods.
Both use the frozen matched budgets below:6000 WM/2500 policy updates, B256/H4,
9040 online physical transitions and1130 final-evaluation transitions per seed,
plus the shared bootstrap. No supervised updates or intermediate selection.

All30 evaluations are finite; both methods open15/15 narrow same-clip starts.
Zero-update reference controls already open5/5, with mean return9.035571.
Both trained methods have worse tracking. This is bounded negative replication,
not demonstrated tracking benefit or a completed effective final benchmark.

|Method|Seed|Opened|Return|Hand RMSE rad|Door RMSE rad|Latch RMSE rad|Translation RMSE m|CPU wall s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|supertrack|29701|5/5|8.970134|0.009486|0.006822|0.020717|0.003149|211.491723|
|supertrack|29702|5/5|8.834152|0.020492|0.005385|0.031962|0.005791|141.139209|
|supertrack|29703|5/5|8.510147|0.044529|0.021200|0.031802|0.011636|208.844188|
|supermimic|29601|5/5|8.191209|0.046498|0.048528|0.019488|0.024590|212.934160|
|supermimic|29602|5/5|8.441311|0.043897|0.046729|0.012498|0.012486|212.918067|
|supermimic|29603|5/5|8.118409|0.049593|0.039363|0.035652|0.024731|155.530907|

Mean ± sampleSD across the three training seeds (each averaged over five starts):

- supertrack: return 8.771478 ± 0.236312; hand RMSE rad 0.024836 ± 0.017921; door RMSE rad 0.011136 ± 0.008746; latch RMSE rad 0.028160 ± 0.006446; translation RMSE m 0.006859 ± 0.004343; CPU wall s 187.158374 ± 39.875744.
- supermimic: return 8.250310 ± 0.169370; hand RMSE rad 0.046663 ± 0.002852; door RMSE rad 0.044873 ± 0.004856; latch RMSE rad 0.022546 ± 0.011876; translation RMSE m 0.020602 ± 0.007029; CPU wall s 193.794378 ± 33.137139.

Training GPU cost is0hours. CPU times include the reused-bootstrap fitting,
online collection and evaluation, but exclude original shared-bootstrap
acquisition, guard warmup and rendering; shared acquisition cost is unresolved.
These are observed process times, not a controlled speed benchmark.
Source evidence: `formal-protocol2967.json`, `formal-summary2967.json` and
`optimizer-counts2967.json` in the continuation evidence. Later opt-in optimizer,
correction and task-weight options are not validated by these historical runs.
The commands in the benchmark README exercise the public trainer; they do not
reconstruct unidentified historical bootstrap acquisition.

## Historical round2943 study

The remaining tables and check records describe their original source only;
they do not certify the current patch or replace the corrected table above.


Native Adroit hand, original 28 hand-only actuators, 30 qpos/qvel including passive door/latch, original clip23 scene/reference/control timing. No XHand or Allegro retargeting. No artificial door actuation. Different embodiments are not algorithm-only controls.

This is a bounded matched online replication with genuine world-model and policy optimization, not evidence of effective learned feedback or broad task robustness. All six final policies open all five narrow same-clip starts; the executable reference-control prior already does this. Tracking degrades after learning, especially for SuperMimic. No superiority claim.

Protocol was fixed before full-schedule pilot29600 and formal29601–29603: fresh per-method/per-seed initialization; shared six-episode bootstrap dataset; 1000 bootstrap WM updates, then10 rounds of current-policy physical collection +500 WM +250 imagined-policy updates. B256,H4,policy LR1e-5,eager fp32,hidden32x32,WM hidden64x64. SuperMimic executes its increment/squared model objective, imagined reward and terminal critic with TD(lambda) targets; executed constructor defaults are one critic epoch/one full minibatch per policy update. SuperTrack uses its acceleration model/tracking objective. Neither performs supervised updates. The two methods retain their respective upstream objectives, so matched update counts do not imply identical FLOPs.

Each round collects four complete episodes at ARRz offsets[-.01,-.005,.005,.01]; last episode is held out before training/normalization. Equal sampled windows from three retained success-source training episodes and three current-policy training episodes. Final only; evaluation offsets[0,-.015,-.0075,.0075,.015],full226-control horizon,opening>=1.35rad reported separately from tracking. No intermediate policy selection. Shared historical dataset acquisition/generation costs are not included in new process times and remain a total-cost accounting limitation.

|Method|Training seed|Opened|Return|Hand RMSE rad|Door RMSE rad|Latch RMSE rad|Translation RMSE m|Wall s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|supertrack|29601|5/5|8.969786|0.006992|0.002496|0.015447|0.003415|167.347|
|supertrack|29602|5/5|8.990882|0.007559|0.003003|0.013423|0.002789|164.506|
|supertrack|29603|5/5|8.970539|0.010358|0.004322|0.018025|0.003263|163.872|
|supermimic|29601|5/5|8.191209|0.046498|0.048528|0.019488|0.024590|212.934|
|supermimic|29602|5/5|8.441311|0.043897|0.046729|0.012498|0.012486|212.918|
|supermimic|29603|5/5|8.118409|0.049593|0.039363|0.035652|0.024731|155.531|

Sample SD is across three independently trained seeds, after averaging the five starts within each seed.

- supertrack: return 8.977069 ± 0.011968; opened100% ±0pp; observed CPU process wall 165.241 ± 1.851s; training GPU0hours.
- supermimic: return 8.250310 ± 0.169370; opened100% ±0pp; observed CPU process wall 193.794 ± 33.137s; training GPU0hours.

Retained matched zero-update reference-control prior: mean return9.035571,5/5 opened. Comparison is descriptive, not a new trained baseline. All30 final physical evaluations are finite. Fresh training costs include bootstrap; guard startup/warmup and preview rendering are separate. Jobs overlapped with CPU checks, and observed timings are not a controlled method speed benchmark. Online collection is40x226=9040 physical transitions per seed; final evaluation5x226=1130. Shared bootstrap data are additional reused inputs, not fresh interactions.

Reproducibility/evidence: `protocol2943.json`, executed `schedule2943.py`, `formal-summary2943.json`, and `full-source2943-manifest.json` in the continuation evidence directory. Remote root `/data/YifanKang/adroit-door/d0/`; output per method/seed `formal2943-{method}-{seed}`. Each includes raw final/iteration checkpoints, saved replay, complete physical rollouts, result JSON and deployment audit. Existing `gpu156_guard.py` used4GiB RAM/1GiB disk/900s,one CPU thread per job,max two jobs; CPU24/25,CUDA hidden in training. GPU0/1 were capability probes/guard locks only; training GPU cost is zero. Exact B256/H4 warmup executes actual WM/policy updates and physical evaluation. No unrelated job changes.

Raw final checkpoint `protocol` inherited stale bootstrap CLI fields; `online_protocol`, optimizer state and actual executed script record the true run. Raw files are preserved. `deployment.pt` contains corrected metadata only, with raw/corrected SHA256 in `deployment-audit.json`; all weights/optimizer states unchanged. Every saved rollout is bit-exact after reloading, and metrics match except wallclock. This is an explicit metadata limitation, not new training or model selection.

Fixed first seed29601,episode0 previews for each method are native-scene NDM-marked WebP,57 frames,remote only. They visualize saved actual scoring trajectories, not new simulated evidence. Both long-title attempts are preserved; short-title replacements are readable. No README best-policy claim. Rendering used explicit otherwise idle GPU2 and is separate from CPU training.

Current executable/test/text source matches local candidate. Full326-file manifest records14 untransferred upstream media and a different teaser on the remote check tree; no executable/text mismatches. Applicable checks: make lint PASS; strict make docs PASS; component/packaging22PASS; original marked slow upstream equivalence1PASS/2warnings24.59s; make test-fast543PASS/209SKIP/49DESELECTED/53warnings588.65s,exit0. Original tests/assertions/default algorithms unchanged. Skips are not GPU passes. Earlier fast run interrupted intentionally when registry discrepancy was found, then current code rerun.

Door SuperMimic and SuperTrack now both appear in public DOOR_ALGOS; a test verifies equality with the training-method registry and unchanged default ALGOS keys. Existing sampler equivalence test covers unequal episode lengths/RNG draws; actual-policy replay test covers deployment/reset semantics.

Still unresolved: end-to-end GPU/bf16/graph/named-speed-path integration and measurement for Door; current full GPU suite (historical L40S101376byte/block capacity is below known111104–123392byte kernel needs); robust learned-feedback benefit; self-contained public reproduction of historical bootstrap dataset generation and full shared-cost accounting. Current CPU checks do not certify these. All CONTRIBUTING requirements are NOT claimed passed. Existing Draft PR#3 stays draft and untouched; no push, duplicate PR or ready/merge action.

Next implementation should make the genuine alternating trainer a self-contained repository command using explicit model/reference/replay inputs and canonical checkpoint metadata, preserving the executed historical study; then test recovery and any measured device path before further training. Do not repeat unchanged formal seeds or start supervised detours.
