# Two G1s

One policy controls two G1s tracking a retargeted two-person InterHuman clip in one scene, with robot-robot contacts.
InterHuman (CC BY-NC-SA 4.0) is not redistributed: `tools/convert_interhuman.py`
retargets a clip with HoloMotion's HoloRetarget, in an environment of its own (its docstring has the command).

```bash
ndm-train --config two-g1-supertrack --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_st
ndm-train --config two-g1-supermimic --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_sm
# three GPUs: actor, world-model learner and policy learner one GPU each
CUDA_VISIBLE_DEVICES=0,1,2 ndm-train --config two-g1-supertrack-3gpu --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_st_3gpu
CUDA_VISIBLE_DEVICES=0,1,2 ndm-train --config two-g1-supermimic-3gpu --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_sm_3gpu
```

| configuration | GPUs | world model |
|---|---|---|
| `two-g1-supertrack`, `two-g1-supermimic` | 1 | MLP; the policy's goal encoded once per update (`--goal_table --pi_encode_once`) |
| `two-g1-supertrack-3gpu`, `two-g1-supermimic-3gpu` | 3 | the same, with `--layout AWL` |
| `two-g1-supertrack-rf`, `two-g1-supermimic-rf` | 1 | the small RigidFormer |

The two-robot scene also takes the failure resets (`--fail_reset`, `--bin_fix`, over the pair bins), the probe sweeps
(`--eprobe_*`) and the layouts `AL` and `AWL`, each off by default ([BlinkMimic's options](../blinkmimic.md),
[the layouts](../algorithms/xmimic.md)).

Selected configurations and results: [README](../../README.md#two-g1s-one-two-person-clip), [`results.md`](../../benchmarks/g1_boxing/results.md).
