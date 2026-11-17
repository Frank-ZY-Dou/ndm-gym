# Two G1s

One policy controls two G1s tracking a retargeted two-person InterHuman clip in one scene, with robot-robot contacts.
InterHuman (CC BY-NC-SA 4.0) is not redistributed: `tools/convert_interhuman.py`
retargets a clip with HoloMotion's HoloRetarget, in an environment of its own (its docstring has the command).

```bash
ndm-train --config two-g1-supertrack --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_st
ndm-train --config two-g1-supermimic --motion_dir data/interhuman/interhuman_1566.npz --log_dir logs/pair_sm
```

Selected configurations and results: [README](../../README.md#two-g1s-one-two-person-clip), [`results.md`](../../benchmarks/g1_boxing/results.md).
