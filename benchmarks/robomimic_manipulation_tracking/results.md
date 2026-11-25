# Manipulation demonstration tracking: results

Contributed by [Jingdong Zhang](https://github.com/Evergreen0929) ([#10](https://github.com/Frank-ZY-Dou/ndm-gym/pull/10)).

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make one Franka Panda (two for
Transport) track robomimic's first proficient-human demonstration of Lift, Square and Transport on MuJoCo-Warp
([the task](../../docs/tasks/robomimic.md), [the rules](README.md)).

## Setup

- **Reference:** demonstration 0 of each task (robomimic's v1.5 proficient-human files, revision `74fa018`), its last state
  held for 0.5 s: Lift 3.45 s, Square 6.85 s, Transport 36.2 s at 20 Hz.
- **Configurations** (`ndm_gym/configs.py`), each chosen by the same rules for both algorithms ([below](#configurations)):
  - `robomimic-supertrack`: `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0 --st_task_weights --target_rate_w
    64` with the hand task's systems options;
  - `robomimic-supermimic`: `--algo supermimic --bptt_decay 0.95 --target_rate_w 64` (learning rate 2e-4, action noise 0.1,
    discount 0.99) with the same systems options;
  - both: `--task <task> --demo 0 --solver_iterations 20`, 2,048 worlds, a cap of 6,000 iterations with the learning
    rate's cosine decay over it, a checkpoint kept every 250; training seeds 10, 11, 12.
- **Training:** one run per A100 40GB SXM4, three runs at a time on three GPUs of one machine, nothing else on them. Each
  kept checkpoint was evaluated as it appeared, on a fourth GPU, for the stop rule; the training ended at the stop.
- **Stop rule, the same for both algorithms:** each kept checkpoint on 256 perturbed starts (evaluation seed 1). With S(i)
  and E(i) the mean success and object error of the evaluations at i, i - 250 and i - 500, a run stops at the first i where
  E(i) > 0.97 E(i - 1,000) and either S(i) >= 0.99 or S(i) < S(i - 1,000) + 0.01; otherwise at 6,000. The checkpoint at
  the stop, not the best one, is the run's result.
- **Evaluation:** `ndm_gym.evaluate_robomimic` on the stop checkpoint: 256 rollouts of the whole demonstration from
  perturbed starts (0.01 rad on the arms' joints, evaluation seed 5, other starts than the stop rule's), without
  exploration noise. A rollout succeeds if it never terminates and meets robosuite's success test at the demonstration's
  last frame, read from the simulator as robosuite reads it. Errors are means over the frames of the rollouts that do not
  terminate. MuJoCo-Warp's contact order is not deterministic, so two evaluations of one checkpoint on the same starts
  differ: on Transport by about 4 points of success either way. Each number here is one evaluation.
- **Success test:** robosuite's own. Checked against robosuite 1.5.0's `_check_success` on 13,957 states of the three
  datasets and 22,320 states of 36 of our evaluation rollouts: the same verdict on every state. The contributed version
  counted an object inside its bin's box as placed; robosuite requires contact with the bin's base (Transport).
- **Time to the target:** a success of at least 0.99 with the objects within 10 mm (the hand task's target); the training
  clock (from the first iteration, checkpoint writes included) of the first kept checkpoint from which every later one up
  to the stop meets it on the evaluation's starts. Not engineered: the hand task's systems options, no per-phase table.
- **Reference's targets, open loop:** the same rollouts with the demonstration's joint positions as the servos' targets
  and no policy.
- Mean ± sample SD over the three training seeds.

```bash
benchmarks/robomimic_manipulation_tracking/run_formal.sh
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s??
```

## Results

| | Lift | Square | Transport |
|---|---|---|---|
| success ↑, SuperTrack / SuperMimic | 100 / 100 % | 100 / 100 % | 99.9 ± 0.2 / 89.2 ± 15.7 % |
| object position ↓, SuperTrack / SuperMimic | 0.26 ± 0.01 / 1.01 ± 0.24 mm | 1.48 ± 0.70 / 2.55 ± 0.48 mm | 4.24 ± 0.61 / 4.50 ± 1.50 mm |
| object position at the last frame ↓ | 0.1 ± 0.1 / 3.1 ± 1.1 mm | 4.3 ± 1.4 / 4.2 ± 1.2 mm | 10.1 ± 1.5 / 39.5 ± 48.7 mm |
| object rotation ↓ | 0.23 ± 0.02 / 0.54 ± 0.12° | 0.77 ± 0.10 / 1.86 ± 0.65° | 9.92 ± 11.26 / 5.80 ± 5.96° |
| grippers' keypoints ↓ | 1.18 ± 0.06 / 3.79 ± 1.72 mm | 1.68 ± 0.33 / 6.46 ± 0.67 mm | 3.34 ± 0.53 / 9.46 ± 1.51 mm |
| episodes ended ↓ | 0 / 0 % | 0 / 0 % | 0.1 ± 0.2 / 9.9 ± 14.4 % |
| reference's targets, open loop: success | 0 % | 5.9 ± 1.7 % | 0 % |
| stop (iteration) | 3,333 ± 382 / 3,917 ± 1,010 | 3,667 ± 764 / 4,417 ± 1,010 | 2,917 ± 804 / 4,667 ± 1,528 |
| training to the stop (min) | 7.0 ± 0.8 / 9.1 ± 2.4 | 15.6 ± 3.3 / 20.0 ± 4.5 | 16.2 ± 4.5 / 27.9 ± 9.2 |
| iterations / s (301 to the stop) | 7.83 / 7.06 | 3.89 / 3.64 | 2.98 / 2.76 |

- **Lift and Square:** both algorithms succeed in every rollout of every seed. SuperTrack's object error is the lower: 0.26
  against 1.01 mm on Lift, 1.48 against 2.55 mm on Square. The reference's joint positions alone succeed in 0 and 6 % of the
  rollouts.
- **Transport:** success 99.9 ± 0.2 % for SuperTrack and 89.2 ± 15.7 % for SuperMimic (seeds 10, 11, 12: 100, 100 and 99.6 %
  against 98.0, 98.4 and 71.1 %), object error 4.24 and 4.50 mm, episodes ended 0.1 and 9.9 %. SuperTrack's runs reach the
  target, SuperMimic's do not (below).
- **The arms:** the grippers' keypoints stay closer to the demonstration's with SuperTrack on every task (1.18 / 1.68 /
  3.34 mm against 3.79 / 6.46 / 9.46 mm).

Every seed (10 / 11 / 12):

| task | | SuperTrack | SuperMimic |
|---|---|---|---|
| Lift | stop | 3,250 / 3,750 / 3,000 | 3,000 / 5,000 / 3,750 |
| | success; object | 100 / 100 / 100 %; 0.26 / 0.25 / 0.27 mm | 100 / 100 / 100 %; 1.14 / 0.73 / 1.16 mm |
| Square | stop | 4,500 / 3,500 / 3,000 | 3,250 / 5,000 / 5,000 |
| | success; object | 100 / 100 / 100 %; 0.91 / 2.26 / 1.27 mm | 100 / 100 / 100 %; 2.33 / 3.11 / 2.22 mm |
| Transport | stop | 2,000 / 3,250 / 3,500 | 6,000 / 3,000 / 5,000 |
| | success; object | 100 / 100 / 99.6 %; 3.55 / 4.70 / 4.46 mm | 98.0 / 98.4 / 71.1 %; 3.13 / 4.25 / 6.11 mm |
| | episodes ended | 0 / 0 / 0.4 % | 2.0 / 1.2 / 26.6 % |

SuperMimic's Transport seed 12 stopped at 5,000 with a success of 71.1 % (26.6 % of its rollouts ended early); its seed 10
stopped at 6,000, the last kept checkpoint, by the rule.

## Smooth motion

**Why the policies' motion was rough.** The demonstrations were teleoperated through robosuite's operational-space controller
(stiffness 150, critically damped: a natural frequency of about 2 Hz), which smooths the operator's hand. Nothing in either
objective kept a policy's position targets from changing faster. Behind the rollouts of the configurations without the term the
targets, recovered from the arms' motion, changed 4.6 to 12.9 times as much per 50 ms step as the demonstration's (seeds 0-2;
78 to 97 % of the change above 5 Hz). The servos pass part of that on to the arms as jerk. The demonstration's own targets, played
through the same servos without a policy, move the arms more smoothly than the demonstration itself (jerk 0.27 / 0.44 / 0.80
times on Lift / Square / Transport).

**The term, the same for both algorithms.** A cost on the policy's part of the targets' change per step: each actuator's change
over its tracking kernel's width (an arm joint 0.4 rad, a finger 0.04 m), squared, the mean over the actuators, times
`--target_rate_w` (64 in both configurations). A change of the width on every actuator costs the weight; following the
demonstration costs nothing. SuperMimic's reward carries it in the place of the contributed small term on the action's change;
SuperTrack's loss carries its mean over the imagined steps. Off by default, and then the updates are as before
([the option](../../docs/tasks/robomimic.md#smooth-targets)).

**The motion check.** The rollout of each run from the unperturbed start against the demonstration's same frames (`motion.py`),
the means over the three seeds. A row passes if:

- the arms' joint jerk is at most 1.5 times the demonstration's;
- the share of their velocity power above 5 Hz is at most 2 times. The share is power over power: the power above 5 Hz summed
  over the arms' joints, over their total power (each joint's velocity less its mean), so that a joint that barely moves weighs
  as little as its power;
- the position targets' change per step is at most 2 times the demonstration's;
- no arm joint comes to a limit (within 1 % of its range) that the demonstration does not reach;
- a held object's slip in its gripper is at most 1 mm per step more than in the demonstration.

Every row passes. With the share as the mean over the joints of each joint's own share the same rows pass (0.12 to 0.99
times).

Without the term (seeds 3, 4 and 6) and with it (seeds 10, 11 and 12); a ratio is the policy's value over the demonstration's
over the same frames; the slip in mm per step, in brackets the demonstration's:

| task | algorithm | jerk ×, without / with | share above 5 Hz ×, without / with | targets' change ×, with | slip, without / with |
|---|---|---|---|---|---|
| Lift | SuperTrack | 2.85 / 0.41 | 8.1 / 0.07 | 1.11 | 0.93 / 0.02 (0.03) |
| | SuperMimic | 1.78 / 0.38 | 2.2 / 0.04 | 1.11 | 0.27 / 0.20 (0.03) |
| Square | SuperTrack | 7.20 / 0.86 | 31.0 / 0.38 | 1.19 | 0.80 / 0.30 (0.30) |
| | SuperMimic | 7.58 / 0.97 | 24.6 / 0.35 | 1.24 | 0.70 / 0.45 (0.30) |
| Transport | SuperTrack | 9.58 / 0.97 | 83.8 / 0.73 | 1.10 | 0.60 / 0.28 (0.29) |
| | SuperMimic | 10.86 / 1.30 | 34.1 / 0.98 | 1.25 | 0.67 / 0.45 (0.27) |

SuperMimic's Transport seeds 11 and 12 have an unperturbed rollout that does not pass the success test (seed 11 plays to the
end, seed 12 ends at 590 of 724 frames); both are in the rows. The largest jerk ratio is 1.30 (SuperMimic's Transport).

Against the same configurations without the term, on other seeds (3, 4 and 6):

- **Object error.** SuperTrack: from 0.37 to 0.26 mm on Lift, from 1.30 to 1.48 mm on Square, from 2.99 to 4.24 mm on Transport.
  SuperMimic: from 0.89 to 1.01 mm, from 8.33 to 2.55 mm, from 4.14 (2 of 3 runs) to 4.50 mm.
- **Success.** Transport: from 59.6 ± 42.3 to 99.9 ± 0.2 % (SuperTrack) and from 34.4 ± 44.9 to 89.2 ± 15.7 % (SuperMimic).
  Square: from 77.5 ± 39.0 to 100 % (SuperMimic). Every other row is 100 % with and without.
- We have not separated what in the term does this; it also lowers a held object's slip in its gripper (above).

## Time to the target

Per run, the training time at which the target holds from then on (above), with the object error at that checkpoint
beside it.

| task | algorithm | minutes to the target ↓ | iteration | object error there | seeds 10 / 11 / 12 (min) |
|---|---|---|---|---|---|
| Lift | SuperTrack | 1.03 ± 0.00 | 500 / 500 / 500 | 0.58 ± 0.13 mm | 1.03 / 1.03 / 1.03 |
| | SuperMimic | 1.11 ± 0.01 | 500 / 500 / 500 | 2.68 ± 1.66 mm | 1.11 / 1.11 / 1.10 |
| Square | SuperTrack | 2.12 ± 0.01 | 500 / 500 / 500 | 5.02 ± 2.09 mm | 2.13 / 2.12 / 2.12 |
| | SuperMimic | 4.53 ± 3.08 | 1,750 / 500 / 750 | 5.34 ± 3.81 mm | 8.02 / 2.22 / 3.34 |
| Transport | SuperTrack | 15.76 ± 5.24 | 1,750 / 3,250 / 3,500 | 3.75 ± 1.45 mm | 9.76 / 18.04 / 19.48 |
| | SuperMimic | not reached | | | |

SuperTrack reaches the target at the second kept checkpoint (500 iterations) in every Lift and Square run, SuperMimic at
the same on Lift and at 1,750, 500 and 750 on Square. On Transport SuperTrack's three runs reach it after 9.8, 18.0 and 19.5 min
(iterations 1,750, 3,250 and 3,500; in seeds 11 and 12 at the stop's own checkpoint), SuperMimic's do not.

## The world model as the simulator

One step of the world model, one network evaluation, stands in for the 25 physics steps of the 50 ms control step it
predicts, each with its collision detection and its constraint solve (20 iterations). `ndm_gym.wmbench.speed_robomimic`
times both on one A100 40GB SXM4 with nothing else on the machine: MuJoCo-Warp's control step of the training's scene (one
CUDA graph) and the world model of each configuration's seed-0 run without the smoothness term (its stop checkpoint; the
term changes neither the network nor its step) in fp32 (its step one CUDA graph) and as a bf16
inference copy (compiled, one CUDA graph). Every world starts from a frame of the final's nominal test rollout and holds
the reference's targets of that frame. World-steps per second, the median of 5 runs of 200 steps; not engineered (no
per-phase table).

| task | world model | worlds | MuJoCo-Warp | fp32 | bf16 | bf16 against MuJoCo-Warp |
|---|---|---|---|---|---|---|
| Transport | SuperTrack's acceleration model | 1,024 | 10,662 | 0.634 M | 4.78 M | 448x |
| | | 4,096 | 15,196 | 1.48 M | 11.63 M | 765x |
| | SuperMimic's increment model | 1,024 | 10,115 | 0.619 M | 4.42 M | 437x |
| | | 4,096 | 14,614 | 1.44 M | 11.06 M | 757x |
| Lift | SuperTrack's acceleration model | 1,024 | 29,621 | 1.51 M | 10.46 M | 353x |
| | | 4,096 | 78,335 | 2.56 M | 20.02 M | 256x |
| | SuperMimic's increment model | 1,024 | 29,001 | 1.42 M | 10.08 M | 347x |
| | | 4,096 | 79,472 | 2.54 M | 20.36 M | 256x |
| Square | SuperTrack's acceleration model | 1,024 | 15,290 | 1.20 M | 7.85 M | 513x |
| | | 4,096 | 24,679 | 2.24 M | 17.89 M | 725x |
| | SuperMimic's increment model | 1,024 | 14,647 | 1.18 M | 7.54 M | 515x |
| | | 4,096 | 23,389 | 2.19 M | 17.22 M | 736x |

The control step's cost grows with the scene and its contacts: at 4,096 worlds Lift's one arm and cube take 52 ms a step,
Transport's two arms, bins and objects 270-285 ms. The world model's step grows only with the state it predicts: 0.21,
0.23 and 0.35 ms on Lift, Square and Transport (SuperTrack's bf16 copies). MuJoCo-Warp's numbers differ a little between
the two models of a task because their worlds start from each final's own rollout.

## Configurations

Each algorithm's configuration was chosen by the same rules for both, on other training seeds and starts than the results'.

**The arms.** The hand task's accurate preset and four single changes: the exploration (action noise), the learning rate, the
failure's weight or the return's horizon, and one change specific to the algorithm (SuperTrack's imagination noise,
SuperMimic's gradient scale through each imagined state, `--bptt_decay`). Each ran on the three tasks with the stop rule
above (training seed 5) and was evaluated at its stop on 256 starts of evaluation seed 2; the highest mean success over the
three tasks wins, and within one point of it the lower mean object error. The fifth arm of each algorithm was added after the
first four, so that both have the same kinds of change, and the choice was made again. These runs ran on RTX 2080 Ti GPUs and
serve the selection only (one seed per configuration; -: every rollout terminated):

| configuration | Lift: success, object, stop | Square | Transport | mean success | mean object |
|---|---|---|---|---|---|
| SuperTrack, the preset (imagination noise 0.2) | 100 %, 1.98 mm, 2,000 | 8.6 %, 20.44 mm, 2,500 | 92.2 %, 8.52 mm, 6,000 | 66.9 % | 10.31 mm |
| SuperTrack, imagination noise 0 (a finalist) | 100 %, 0.41 mm, 2,250 | 100 %, 1.86 mm, 3,500 | 21.9 %, 4.24 mm, 3,000 | 74.0 % | 2.17 mm |
| SuperTrack, learning rate 2e-4 | 100 %, 0.94 mm, 2,000 | 0 %, -, 2,000 | 14.5 %, 20.53 mm, 2,500 | 38.2 % | - |
| SuperTrack, failure weight 5 | 0 %, 11.14 mm, 1,750 | 100 %, 3.13 mm, 5,750 | 95.3 %, 8.60 mm, 6,000 | 65.1 % | 7.63 mm |
| SuperTrack, action noise 0.3 (a finalist) | 100 %, 1.31 mm, 2,750 | 100 %, 4.15 mm, 3,750 | 29.7 %, 11.27 mm, 2,750 | 76.6 % | 5.58 mm |
| SuperMimic, the preset (a finalist) | 100 %, 0.77 mm, 4,750 | 85.2 %, 8.59 mm, 3,000 | 30.1 %, 4.02 mm, 5,250 | 71.7 % | 4.46 mm |
| SuperMimic, action noise 0.3 | 100 %, 1.41 mm, 3,250 | 6.6 %, 30.94 mm, 1,750 | 65.6 %, 9.24 mm, 4,250 | 57.4 % | 13.86 mm |
| SuperMimic, learning rate 4e-4 | 100 %, 0.59 mm, 3,750 | 23.0 %, 19.84 mm, 4,500 | 25.8 %, 3.97 mm, 4,000 | 49.6 % | 8.13 mm |
| SuperMimic, discount 0.995 | 100 %, 1.64 mm, 3,500 | 98.4 %, 6.44 mm, 2,250 | 0 %, -, 6,000 | 66.1 % | - |
| SuperMimic, `--bptt_decay 0.95` (a finalist) | 100 %, 0.66 mm, 4,000 | 100 %, 2.38 mm, 3,500 | 61.3 %, 4.45 mm, 4,500 | 87.1 % | 2.49 mm |

**The finalists.** The four finalists ran on A100 40GB SXM4 GPUs, training seeds 0, 1 and 2, 256 test starts of evaluation seed
0 (the mean ± sample SD over the seeds). Each algorithm keeps the one with the higher success averaged over the three tasks,
and a tie within one point would go to the lower mean object error:

| configuration | Lift: success; object | Square | Transport | mean success | mean object |
|---|---|---|---|---|---|
| SuperTrack, imagination noise 0 (kept) | 100 %; 0.41 ± 0.11 mm | 100 %; 1.45 ± 0.13 mm | 56.9 ± 16.3 %; 5.42 ± 4.25 mm | 85.6 % | 2.43 mm |
| SuperTrack, action noise 0.3 | 66.5 ± 57.6 %; 15.28 ± 20.33 mm | 34.0 ± 57.2 %; 27.55 ± 21.81 mm | 60.5 ± 51.9 %; 11.00 ± 5.33 mm | 53.7 % | 17.94 mm |
| SuperMimic, the preset | 100 %; 0.92 ± 0.18 mm | 69.5 ± 52.8 %; 7.62 ± 7.06 mm | 54.9 ± 37.4 %; 6.73 ± 2.10 mm | 74.8 % | 5.09 mm |
| SuperMimic, `--bptt_decay 0.95` (kept) | 100 %; 0.97 ± 0.45 mm | 100 %; 2.55 ± 0.59 mm | 52.5 ± 14.0 %; 4.66 ± 1.83 mm | 84.2 % | 2.73 mm |

**The smoothness weight.** One weight for both algorithms, the lowest of 0, 2, 4, 8, 16, 32 and 64 at which both algorithms' runs
pass the motion check on all three tasks and keep their accuracy (Lift and Square at the target; Transport's success not more
than 25 points below the same configuration without the term): 64. The screen ran on RTX 2080 Ti GPUs, training seed 7 for
every weight, seeds 8 and 9 for 32 and 64, and 13 and 14 for 64 (selection only; a row is the mean over its seeds).

The arms' jerk over the demonstration's, SuperTrack / SuperMimic (Lift's is at most 0.97 times for both from weight 2 on):

| weight | Square | Transport |
|---|---|---|
| 0 | 7.15 / 5.65 | 11.04 / 13.82 |
| 2 | 1.92 / 2.19 | 4.26 / 3.73 |
| 4 | 1.66 / 1.90 | 2.46 / 3.44 |
| 8 | 1.56 / 1.59 | 1.85 / 2.80 |
| 16 | 1.45 / 1.26 | 2.62 / 1.90 |
| 32 | 1.11 / 1.02 | 1.22 / 1.71 |
| 64 | 0.89 / 0.89 | 1.01 / 1.20 |

Below 64 the jerk is over 1.5 times on Transport for at least one algorithm. At 64 every row passes the check, also with the share as
the mean over the joints of each joint's own share.

**Without the term**, the two kept configurations on fresh seeds (3, 4 and 6; 256 starts of evaluation seed 3, A100 40GB SXM4).
The published configurations are these two with the smoothness term; their rows above are on seeds 10, 11 and 12.

| | Lift | Square | Transport |
|---|---|---|---|
| success ↑, SuperTrack / SuperMimic | 100 / 100 % | 100 / 77.5 ± 39.0 % | 59.6 ± 42.3 / 34.4 ± 44.9 % |
| object position ↓ | 0.37 ± 0.05 / 0.89 ± 0.28 mm | 1.30 ± 0.11 / 8.33 ± 7.39 mm | 2.99 ± 1.03 / 4.14 ± 1.64 mm (SuperMimic: 2 of 3 runs) |
| to the target ↓ | 1.04 / 1.11 min | 2.86 / 14.99 min (SuperMimic: 2 of 3 runs) | not reached / not reached |

## The contributed runs

The contributor's runs ([Jingdong Zhang](https://github.com/Evergreen0929)): seeds 0-2, each run alone on an A100 80GB
SXM4. They differ from the runs above in five ways:

- a fixed budget of 1,500 iterations (the learning rate on a schedule of 4,000, a checkpoint kept every 100), the last
  checkpoint evaluated;
- SuperTrack at imagination noise 0.1, the trainer's default;
- SuperMimic without `--bptt_decay 0.95`;
- no smoothness term (`--target_rate_w`) on either algorithm;
- the earlier success tests: Square's without robosuite's release condition, Transport's by the objects' positions inside
  their bins instead of their contact with the bins' bases.

| | Lift | Square | Transport |
|---|---|---|---|
| success, SuperTrack / SuperMimic | 100 / 100 % | 100 / 84.9 % | 70.4 / 29.7 % |
| object position, SuperTrack / SuperMimic | 0.62 / 0.87 mm | 1.57 / 5.34 mm | 5.69 / 10.56 mm (SuperMimic: 2 of 3 seeds) |
| episodes ended, SuperTrack / SuperMimic | 0 / 0 % | 0 / 10.7 % | 25.7 / 69.0 % |
| to the target, SuperTrack / SuperMimic | 0.9 / 1.1 min | 3.7 / 4.3 min (SuperMimic: 2 of 3 runs) | not reached / not reached |

In the contributor's words: "The Transport runs of both algorithms and Square's SuperMimic run of seed 1 still change
between their last checkpoints (below), so these are the fixed budget's numbers, not final ones." Each checkpoint's success
(%), by its iteration, in bold where the checkpoint meets the target (the first 300 iterations fit the world model alone):

| task | algorithm | seed | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1,000 | 1,100 | 1,200 | 1,300 | 1,400 | 1,500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 0 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | SuperMimic | 0 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 0 | 0 | 0 | 0 | 6.2 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| Square | SuperTrack | 0 | 4.7 | 5.9 | 6.6 | 98.4 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | 93.0 | **100** | **100** | **100** |
| | | 1 | 2.0 | 10.2 | 4.3 | **100** | 91.0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 7.4 | 9.0 | 6.2 | 87.9 | 97.7 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | SuperMimic | 0 | 7.8 | 6.6 | 8.6 | 0 | 100 | **100** | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 6.2 | 9.0 | 7.8 | 70.7 | 100 | **100** | **100** | 66.4 | **100** | 60.9 | 98.4 | 15.6 | **99.6** | 79.3 | 54.7 |
| | | 2 | 6.6 | 5.9 | 8.2 | 0 | 100 | **100** | **100** | **100** | **100** | 10.5 | **100** | **100** | **100** | **100** | **100** |
| Transport | SuperTrack | 0 | 0 | 0 | 0 | 59.0 | 82.0 | 96.5 | 72.7 | 83.6 | 34.4 | 48.0 | 60.9 | 55.9 | 39.5 | 43.0 | 88.7 |
| | | 1 | 0 | 0 | 0 | **100** | 95.3 | 7.4 | 83.6 | 62.9 | 58.6 | 69.1 | 54.7 | 20.7 | 48.4 | 61.3 | 55.5 |
| | | 2 | 0 | 0 | 0 | 73.0 | 94.9 | 69.5 | 93.4 | 25.4 | 59.8 | 78.5 | 40.2 | 60.5 | 62.1 | 73.4 | 67.2 |
| | SuperMimic | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.5 | 0 | 0.8 | 0 |
| | | 1 | 0 | 0 | 0 | 0 | 1.2 | 0 | 0 | 0 | 0.4 | 27.7 | 55.1 | 64.1 | 84.0 | 21.9 | 62.5 |
| | | 2 | 0 | 0 | 0 | 0 | 0 | 0.8 | 12.9 | 0.4 | 23.0 | 2.3 | 31.2 | 30.9 | 34.0 | 27.3 | 26.6 |

On Lift every run of both algorithms meets the target at each checkpoint from 400 iterations, SuperMimic's run of seed 1
from 600. On Square, SuperTrack's runs first meet it at 500 / 400 / 600 iterations: the run of seed 0 misses it at 1,200
(93.0 %) and holds it from 1,300, the run of seed 1 misses it at 500 (91.0 %) and holds it from 600, the run of seed 2 holds
it from 600. SuperMimic's Square runs succeed in every rollout at 500 iterations with the nut 11.0 to 11.8 mm from its
demonstration, and first meet the target at 600: the run of seed 0 misses it at 700 (no rollout succeeds) and holds it
from 800, the run of seed 2 misses it at 1,000 (10.5 %) and holds it from 1,100, and the run of seed 1 meets it at 4 of its
15 checkpoints and not at the last (54.7 %). On Transport one checkpoint meets the target (SuperTrack's seed 1 at 400
iterations: 100 %, the objects within 8.3 mm); from 400 iterations SuperTrack's checkpoints have a success of 7.4 to
100 %, and SuperMimic's highest are 3.5 / 84.0 / 34.0 % (seeds 0 / 1 / 2).

Every run's metrics, time to the target and training summary, as JSON records: [`a100_runs/`](a100_runs/README.md). The
grid of their renders, per case the seed with the highest success, then the lowest object error, from the unperturbed start
(Transport at four times real time; the six play the whole demonstration): [`a100_runs/media/grid.webp`](a100_runs/media/grid.webp).

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/robomimic/` holds, per task and
algorithm, the run with the lowest object error among those whose own rollout from the unperturbed start plays the whole
demonstration, passes the success test and passes the motion check (SuperTrack: Lift seed 11, Square seed 10, Transport
seed 10; SuperMimic: Lift seed 11, Square seed 12, Transport seed 10; SuperMimic's Transport seeds 11 and 12 do not pass the
success test), Transport at four times real time, and the grid of the six (`tools/make_grid.py`). The teaser's tiles are, per
task, the lowest object error among both algorithms' runs of that kind.
