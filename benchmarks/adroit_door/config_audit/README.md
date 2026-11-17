# Historical episode-runner configuration audit

`upstream-69b818e.json` records the expanded upstream named Hand configurations.
`door-mapping3017.json` maps those arguments to the legacy `train_door_online`
episode-runner preset. `tools/verify_door_config_mapping.py` checks declaration
consistency, not runtime equivalence, convergence or contribution acceptance.

These files do **not** describe the newer `train_door_full` study. Its frozen
budget, method settings, native-Door adaptations and evaluation rules are in
[`../full_protocol_v1.json`](../full_protocol_v1.json). Historical reduced-budget
runs must not be relabeled as full-budget evidence.
