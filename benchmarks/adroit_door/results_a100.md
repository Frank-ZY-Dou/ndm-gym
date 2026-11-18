# Adroit Door tracking: results on A100 40GB

The task, its preparation and the contributed study on an L40S: [README](README.md), [full results](full_results_v1.md).
Every run here: one A100 40GB SXM4 and eight physical cores of a Xeon Platinum 8357B, 20,000 iterations in 2,048 worlds,
the contributed implementation as it is (eager fp32 learning on the GPU, native MuJoCo collection on the CPU), evaluated
with `eval_door_full` on its 256 starts (one nominal, 255 with the hand perturbed). Means ± sample SD over training
seeds 0-2 of each seed's 256-start mean; every seed is listed.

## The demonstration's start

The contributed configuration (`--method supertrack` or `--method supermimic`), beside the demonstration's controls
replayed without training (float32, as the policy holds them).

| metric | replayed controls | SuperTrack | SuperMimic |
|---|---:|---:|---:|
| tracking return ↑ | 9.03533 | 9.01798 ± 0.01329 | 8.93393 ± 0.04424 |
| root translation RMSE ↓ (mm) | 0.5789 | 0.4829 ± 0.0389 | 1.1062 ± 0.3333 |
| hand RMSE ↓ (rad) | 0.00387 | 0.02328 ± 0.00979 | 0.05383 ± 0.00936 |
| door RMSE ↓ (rad) | 0.00420 | 0.00827 ± 0.00239 | 0.00929 ± 0.00146 |
| latch RMSE ↓ (rad) | 0.00251 | 0.00650 ± 0.00441 | 0.02391 ± 0.01188 |
| final opening ↑ | 256/256 | 768/768 | 768/768 |
| training minutes per seed | - | 116.2 ± 1.9 | 146.3 ± 3.3 |

| method | seed | return | hand (rad) | door (rad) | opened | training (min) |
|---|---:|---:|---:|---:|---:|---:|
| SuperTrack | 0 | 9.00423 | 0.03311 | 0.01103 | 256/256 | 116.58 |
| SuperTrack | 1 | 9.01895 | 0.02319 | 0.00682 | 256/256 | 114.09 |
| SuperTrack | 2 | 9.03076 | 0.01353 | 0.00697 | 256/256 | 117.86 |
| SuperMimic | 0 | 8.91685 | 0.05783 | 0.00979 | 256/256 | 149.97 |
| SuperMimic | 1 | 8.90078 | 0.06052 | 0.01043 | 256/256 | 145.30 |
| SuperMimic | 2 | 8.98417 | 0.04313 | 0.00764 | 256/256 | 143.51 |

The replayed controls already open every start, and both trained policies track a little below them: a correction
learned on top of a control that already works has little to gain on one clip.

## The door and latch started open (`--door-latch-offset 0.3`)

Every training episode and every evaluation start except the nominal one begins with the door and the latch raised by
U(0, 0.3) rad each (the hand perturbed as above). The replayed controls then open the door on 167 of 256 starts (65.2 %)
at a tracking return of 6.1016. Each algorithm's configurations were chosen in a screen of eight arms (seed 0, 6,000
iterations, the same shared arms for both): the most accurate and the one first at 99 % opening. Training time: the
trainer's clock at the first snapshot (every 500 iterations) with 99 % (or 95 %) of the starts open, evaluation on
another GPU not included.

| method | configuration | opened | return ↑ | hand RMSE ↓ (rad) | door RMSE ↓ (rad) | to 99 % | to 95 % |
|---|---|---:|---:|---:|---:|---:|---:|
| SuperTrack | `--method supertrack --residual-scale 0.2 --door-latch-offset 0.3` | 99.3 ± 0.8 % | 8.165 ± 0.054 | 0.0730 ± 0.0039 | 0.1403 ± 0.0054 | 37.6 ± 10.6 min | 22.9 ± 2.8 min |
| SuperTrack | `--method supertrack --lr 8e-4 --door-latch-offset 0.3` | 91.1 ± 6.4 % | 7.794 ± 0.079 | 0.0819 ± 0.0041 | 0.1673 ± 0.0110 | 35.5 ± 13.6 min | 16.0 ± 1.5 min |
| SuperMimic | `--method supermimic --door-latch-offset 0.3` | 99.5 ± 0.9 % | 7.927 ± 0.066 | 0.0783 ± 0.0049 | 0.1320 ± 0.0060 | 13.1 ± 7.9 min | 6.7 ± 0.1 min |
| SuperMimic | `--method supermimic --lr 1e-4 --door-latch-offset 0.3` | 99.1 ± 0.9 % | 7.846 ± 0.032 | 0.0898 ± 0.0030 | 0.1333 ± 0.0046 | 15.3 ± 5.3 min | 11.6 ± 1.9 min |

| method | configuration (with `--door-latch-offset 0.3`) | seed | opened | return | hand (rad) | door (rad) | to 99 % |
|---|---|---:|---:|---:|---:|---:|---:|
| SuperTrack | `--method supertrack --residual-scale 0.2` | 0 | 256/256 | 8.225 | 0.0710 | 0.1341 | 28.5 min |
| SuperTrack | `--method supertrack --residual-scale 0.2` | 1 | 255/256 | 8.148 | 0.0774 | 0.1441 | 49.3 min |
| SuperTrack | `--method supertrack --residual-scale 0.2` | 2 | 252/256 | 8.122 | 0.0706 | 0.1427 | 35.0 min |
| SuperTrack | `--method supertrack --lr 8e-4` | 0 | 215/256 | 7.706 | 0.0792 | 0.1797 | 22.6 min |
| SuperTrack | `--method supertrack --lr 8e-4` | 1 | 246/256 | 7.859 | 0.0867 | 0.1590 | 34.2 min |
| SuperTrack | `--method supertrack --lr 8e-4` | 2 | 239/256 | 7.815 | 0.0799 | 0.1632 | 49.7 min |
| SuperMimic | `--method supermimic` | 0 | 256/256 | 7.969 | 0.0749 | 0.1300 | 6.8 min |
| SuperMimic | `--method supermimic` | 1 | 252/256 | 7.852 | 0.0840 | 0.1388 | 22.0 min |
| SuperMimic | `--method supermimic` | 2 | 256/256 | 7.961 | 0.0760 | 0.1273 | 10.4 min |
| SuperMimic | `--method supermimic --lr 1e-4` | 0 | 255/256 | 7.883 | 0.0865 | 0.1304 | 13.8 min |
| SuperMimic | `--method supermimic --lr 1e-4` | 1 | 251/256 | 7.835 | 0.0904 | 0.1386 | 21.2 min |
| SuperMimic | `--method supermimic --lr 1e-4` | 2 | 255/256 | 7.822 | 0.0925 | 0.1307 | 10.9 min |


These times are the contributed implementation's, not engineered: its updates take about 0.24 s (SuperTrack) and
0.33 s (SuperMimic) an iteration and its CPU physics about 0.1 s.

## The world model as the simulator

The world models of the most accurate configurations with the door and latch started open (SuperTrack
`--residual-scale 0.2`, SuperMimic the contributed configuration; `--door-latch-offset 0.3`), training seed 0, iteration
20,000, step the door in place of its physics. A world-step is one 10 ms control step; every world starts from a state of
the model's evaluation rollouts, holding that frame's control. One A100 40GB SXM4 with no other process on any GPU and
eight physical cores; the median of five timed runs. MuJoCo on the CPU runs the contributed native collector on the eight
cores, MuJoCo Warp runs in float32 with its five physics steps as one CUDA graph, and the world model runs in fp32 as
trained (one CUDA graph) and as a bf16 inference copy (compiled, one CUDA graph).

```bash
# both checkpoints at 1,024-65,536 worlds: one GPU with nothing else running, addressed by its UUID, and eight physical cores
# of its NUMA node; <in dir> as the script's header says
CUDA_VISIBLE_DEVICES=GPU-<uuid> PY=python taskset -c <eight cores> benchmarks/adroit_door/scripts/door_speed.sh <in dir> <out dir>
# or one checkpoint at one world count
python -m ndm_gym.wmbench.speed_door --ckpt <run>/latest.pt --rollouts <run>-evaluation/physical-rollouts.npz \
  --model door23-supported/door.xml --reference door23-supported/reference.npz --native-library door_batch.so \
  --nworld 16384 --profile --out speed.json
```

World-steps per second, SuperTrack's model (an acceleration MLP, 3 × 1,024) / SuperMimic's (an increment MLP, 3 × 1,024);
MuJoCo Warp measured beside each, the CPU once:

| worlds | MuJoCo, CPU (8 cores) | MuJoCo Warp | NDM fp32 | NDM bf16 | bf16 / MuJoCo Warp | bf16 / CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 15,733 | 119,676 / 122,565 | 1.89 M / 1.99 M | 9.29 M / 11.95 M | 77.6 / 97.5 | 590 / 760 |
| 4,096 | 16,955 | 274,901 / 284,868 | 2.87 M / 2.84 M | 20.85 M / 21.55 M | 75.8 / 75.7 | 1,230 / 1,271 |
| 16,384 | 16,962 | 396,344 / 422,031 | 3.41 M / 3.38 M | 27.62 M / 27.14 M | 69.7 / 64.3 | 1,629 / 1,600 |
| 65,536 | - | 440,309 / 470,236 | 3.50 M / 3.45 M | 28.47 M / 27.99 M | 64.6 / 59.5 | - |

Where the bf16 step's time goes (kernel time, SuperTrack / SuperMimic): at 16,384 worlds matrix products 0.386 / 0.404 ms
and the rest (normalization, the integration or the increment, casts) 0.217 / 0.206 ms, 64 % / 66 % products; at 1,024
worlds 60 % / 57 %, at 65,536 worlds 64 % / 66 %. The step is mostly the network's products; about a third stays in
compiled elementwise kernels. The fp32 step takes 4.8 ms at 16,384 worlds, eight times the bf16 one.

Open-loop error, fp32 → bf16: 2,816 starts (every 16th frame of the 256 evaluation episodes) under the recorded controls;
the hand's joint RMSE and the door's and the latch's absolute error in rad, the root's translation error in mm.

| steps | SuperTrack hand | door | latch | root | SuperMimic hand | door | latch | root |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00624 → 0.00626 | 0.00071 → 0.00072 | 0.00216 → 0.00218 | 0.325 → 0.328 | 0.00679 → 0.00678 | 0.00095 → 0.00095 | 0.00201 → 0.00202 | 0.310 → 0.307 |
| 8 | 0.02411 → 0.02416 | 0.00315 → 0.00313 | 0.00875 → 0.00873 | 1.943 → 1.951 | 0.02849 → 0.02853 | 0.00320 → 0.00323 | 0.00920 → 0.00921 | 1.272 → 1.294 |
| 16 | 0.03174 → 0.03244 | 0.00585 → 0.00578 | 0.01540 → 0.01534 | 3.182 → 3.206 | 0.03900 → 0.03911 | 0.00551 → 0.00554 | 0.01824 → 0.01817 | 2.151 → 2.204 |
| 32 | 0.05667 → 0.05756 | 0.01181 → 0.01168 | 0.02578 → 0.02585 | 4.583 → 4.554 | 0.04942 → 0.04933 | 0.00966 → 0.00975 | 0.02504 → 0.02529 | 3.214 → 3.165 |
| 64 | 0.07044 → 0.07144 | 0.02322 → 0.02350 | 0.04232 → 0.04288 | 7.373 → 7.541 | 0.05618 → 0.05563 | 0.00991 → 0.00985 | 0.03528 → 0.03548 | 4.261 → 4.220 |

The bf16 copies stay within 2.5 % of fp32 (the largest difference: SuperMimic's root at 16 steps; SuperTrack's, 2.3 %, its
root at 64 steps).

MuJoCo Warp without no-slip and without multi-contact or native CCD, which it refuses with this model's margins; the CPU
collector and the world models' training data keep no-slip.
