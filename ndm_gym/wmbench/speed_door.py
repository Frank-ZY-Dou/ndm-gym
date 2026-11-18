"""The Adroit door's world model as the simulator: its throughput beside MuJoCo's on the CPU (the native collector) and MuJoCo
Warp's on the GPU, and its open-loop error, for a checkpoint of train_door_full (its latest.pt).

Throughput, in world-steps per second (one step: one 10 ms control step, the native model's five physics steps, and one step
of the world model), every world started from a state of the checkpoint's evaluation rollouts (``--rollouts``: eval_door_full's
physical-rollouts.npz) and holding that frame's control:

- ``cpu``: the contributed native collector (``ParallelNativeDoorCollector`` with the compiled batch loop, ``--workers``
  threads on this process's cores), its whole ``step`` (no-slip on, as the training and evaluation model);
- ``warp``: MuJoCo Warp (``sim.rigid.door_warp``: float32, no no-slip, no multi-contact or native CCD, the 128-edge EPA
  buffer), its five physics steps as one CUDA graph;
- ``ndm_fp32``: the world model's step as trained, one CUDA graph;
- ``ndm_bf16``: the fast inference path of the G1 and HDMI rows (``worldmodels.mjcf.inference_copy`` in bf16, checked there
  against the fp32 network; the step compiled with ``--compile`` and replayed as one CUDA graph). The model has no contact or
  geometry inputs, so ``MjcfWorldModel.fused`` (the kinematics' fused path) has nothing to act on.

Each is the median of ``--repeats`` runs of ``--steps`` steps after ``--warmup``, synchronized around the timed loop (the CPU
path: wall time). ``--profile``: the bf16 step's kernels by kind (matrix products against the rest), from torch.profiler on
the compiled step outside the graph.

Open-loop error: from frames of the rollouts (every ``--start_stride``-th frame of every episode), the world model rolled out
under the recorded controls; at each of ``--horizons``, the hand's joint RMSE (rad), the door's and the latch's absolute error
(rad) and the root's translation error (mm) against the recorded state, in fp32 and in the bf16 copy.

Run: ``python -m ndm_gym.wmbench.speed_door --ckpt RUN/latest.pt --rollouts RUN-evaluation/physical-rollouts.npz --model door.xml
--reference reference.npz --native-library door_batch.so --nworld 1024 4096 16384 --out speed.json``
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def rollouts(path):
    d = np.load(path)
    n = len([k for k in d.files if k.endswith('_qpos')])
    return (np.stack([d[f'episode{i}_qpos'] for i in range(n)]), np.stack([d[f'episode{i}_qvel'] for i in range(n)]),
            np.stack([d[f'episode{i}_actions'] for i in range(n)]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('ckpt', 'rollouts', 'model', 'reference', 'native-library', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--nworld', type=int, nargs='+', default=[1024, 4096, 16384])
    p.add_argument('--what', nargs='+', default=['cpu', 'warp', 'ndm_fp32', 'ndm_bf16'], choices=['cpu', 'warp', 'ndm_fp32', 'ndm_bf16'])
    p.add_argument('--workers', type=int, default=8, help='cpu: the native collector threads (the cores this process is pinned to)')
    p.add_argument('--compile', default='max-autotune-no-cudagraphs', help='ndm_bf16: torch.compile mode, or none')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--cpu_steps', type=int, default=20)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 8, 16, 32, 64])
    p.add_argument('--start_stride', type=int, default=16)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    import mujoco
    import torch

    from ndm_gym.motion.adroit_door import DoorReference
    from ndm_gym.train_door_full import make_algorithm
    from ndm_gym.worldmodels.mjcf import inference_copy

    dev = 'cuda:0'
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    ref = DoorReference.load(a.reference)
    algo, _ = make_algorithm(ck['protocol']['method'], model, ref, dev, ck['protocol'].get('options'))
    algo.load_state_dict(ck['algo'])
    wm = algo.wm.eval().requires_grad_(False)
    fast = inference_copy(wm, torch.bfloat16)
    Q, V, A = rollouts(a.rollouts)  # (episodes, 227, 30), (episodes, 227, 30), (episodes, 226, 28)
    out = dict(ckpt=str(a.ckpt), method=ck['protocol']['method'], iteration=ck['iteration'], gpu=torch.cuda.get_device_name(0),
               wm_kind=wm.kind, wm_hidden=[m.out_features for m in wm.net if hasattr(m, 'out_features')][:-1],
               torch=torch.__version__, mujoco=mujoco.__version__, cpu_workers=a.workers, rows=[], open_loop={})

    # ---- open-loop error
    f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    H = max(a.horizons)
    starts = [(e, f) for e in range(len(Q)) for f in range(0, Q.shape[1] - 1 - H, a.start_stride)]
    ei, fi = np.array(starts).T
    with torch.no_grad():
        for name, net in (('fp32', wm), ('bf16', fast)):
            q, v, rows = f32(Q[ei, fi]), f32(V[ei, fi]), {}
            for h in range(1, H + 1):
                q, v = net.step(q, v, algo.setpoint(f32(A[ei, fi + h - 1]).clamp(-1, 1)))
                if h in a.horizons:
                    t = f32(Q[ei, fi + h])
                    rows[h] = dict(starts=len(starts), hand_rmse_rad=float((q[:, 1:28] - t[:, 1:28]).square().mean().sqrt()),
                                   door_abs_rad=float((q[:, 28] - t[:, 28]).abs().mean()),
                                   latch_abs_rad=float((q[:, 29] - t[:, 29]).abs().mean()),
                                   root_mm=1000 * float((q[:, 0] - t[:, 0]).abs().mean()))
            out['open_loop'][name] = rows

    rng = np.random.default_rng(a.seed)

    def timed(step, sync, steps) -> float:
        rates = []
        for _ in range(a.repeats):
            for _ in range(min(a.warmup, steps)):
                step()
            sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            sync()
            rates.append(steps / (time.perf_counter() - t0))
        return statistics.median(rates)

    for n in a.nworld:
        pick = rng.integers(0, len(starts), n)
        q0, v0 = Q[ei[pick], fi[pick]], V[ei[pick], fi[pick]]
        act = np.clip(A[ei[pick], fi[pick]], -1, 1)
        for what in a.what:
            row = dict(what=what, nworld=n)
            try:
                if what == 'cpu':
                    from ndm_gym.sim.rigid.door_collector import ParallelNativeDoorCollector
                    with ParallelNativeDoorCollector(model, ref, n, a.workers, a.native_library.resolve()) as c:
                        c.stagger_from_physical_replay(a.seed)  # native integration states (warm starts) at the reference's phases
                        frames = c.observe()['frames']
                        u = np.clip(ref.actions[np.minimum(frames, len(ref.actions) - 1)], -1, 1)
                        rate = timed(lambda: c.step(u), lambda: None, a.cpu_steps)
                    row['detail'] = f'native collector, {a.workers} threads, stagger from the reference replay, its controls'
                elif what == 'warp':
                    from ndm_gym.sim.rigid.door_warp import WarpDoorCollector
                    import warp as wp
                    c = WarpDoorCollector(model, ref, n, dev)
                    c.view['qpos'].copy_(f32(q0)), c.view['qvel'].copy_(f32(v0)), c.view['qacc_warmstart'].zero_()
                    c.view['ctrl'].copy_(c.mid + f32(act) * c.half)
                    c._saved = dict(qpos=c.view['qpos'].clone(), qvel=c.view['qvel'].clone(), frames=c.frames.clone(),
                                    qacc_warmstart=c.view['qacc_warmstart'].clone(), previous_actions=c.previous_actions.clone())
                    c._capture()
                    with wp.ScopedDevice(c.wdev):
                        rate = timed(lambda: wp.capture_launch(c.graph), lambda: wp.synchronize_device(c.wdev), a.steps)
                    c.check()
                    row['detail'] = 'MuJoCo Warp, five physics steps one CUDA graph'
                    del c
                else:
                    net = wm if what == 'ndm_fp32' else fast
                    q, v, u = f32(q0), f32(v0), algo.setpoint(f32(act))
                    fn = net.step if (what == 'ndm_fp32' or a.compile == 'none') else torch.compile(net.step, mode=a.compile, dynamic=False)

                    def body():
                        q1, v1 = fn(q, v, u)
                        q.copy_(q1), v.copy_(v1)
                    with torch.no_grad():
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            for _ in range(3):
                                body()
                        torch.cuda.current_stream().wait_stream(s)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            body()
                        q.copy_(f32(q0)), v.copy_(f32(v0))
                        rate = timed(g.replay, torch.cuda.synchronize, a.steps)
                        if a.profile and what == 'ndm_bf16':
                            from torch.profiler import ProfilerActivity, profile
                            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                                for _ in range(20):
                                    body()
                                torch.cuda.synchronize()
                            gemm = other = 0.
                            for e in prof.key_averages():
                                t = e.device_time_total
                                if any(k in e.key.lower() for k in ('gemm', 'cutlass', 'mm_', 'matmul', 'sm80_xmma', 'ampere')):
                                    gemm += t
                                else:
                                    other += t
                            row['kernels'] = dict(matrix_products_ms=gemm / 20 / 1e3, other_ms=other / 20 / 1e3)
                    row['detail'] = ('fp32, eager step, one CUDA graph' if what == 'ndm_fp32' else
                                     f'bf16 inference copy, compiled ({a.compile}), one CUDA graph')
                row.update(steps_per_s=rate, world_steps_per_s=rate * n, ms_per_step=1e3 / rate)
            except Exception as error:  # out of memory at a large count, for instance: written, not hidden
                row['error'] = f'{type(error).__name__}: {error}'
                torch.cuda.empty_cache()
            out['rows'].append(row)
            print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
