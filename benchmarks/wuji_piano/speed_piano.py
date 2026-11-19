"""The piano's world model as the simulator: its throughput beside the scene's MuJoCo-Warp physics, and its open-loop error,
for train_ode_to_joy checkpoints.

One control step (50 ms) of each simulator, every world holding its action:

- ``mujoco_warp``: the scene of ``ode_to_joy.build`` as ``train_ode_to_joy.FreeHands`` steps it: the controls written, the 10
  physics steps at 200 Hz as one CUDA graph, the observation read (fp32);
- ``ndm_fp32``: one step of the checkpoint's ``PianoDynamics`` on the policy's observed state (192 numbers: 52 actuator
  positions, their last changes and 88 key depths, not the physics state), its eager step as one CUDA graph (TF32 matrix
  products, as trained);
- ``ndm_bf16``: the same with its two residual networks in bf16 (the fingertip geometry and contact terms stay fp32),
  compiled (``--compile``) and replayed as one CUDA graph.

Every world starts from the first checkpoint's policy's own physics state ``--frame`` frames into the song from the 256
evaluation starts (RNG 789, cycled to the world count) and holds that frame's action; each world model steps the observed
state of the same worlds. Each throughput is the median of ``--repeats`` runs of ``--steps`` steps after ``--warmup`` steps
and ``--warm-seconds`` of calls (``wmbench.speed_mjcf.timed``, ``wmbench.warm.keep_busy``), synchronized around each run.

Open-loop error, per checkpoint: its policy plays the 256 evaluation episodes in the physics; from every ``--stride``-th
frame, the world model (fp32 and bf16) is rolled out under the recorded actions; the mean absolute error of the due keys'
and of all 88 keys' depths (fractions of key travel) at each of ``--horizons`` steps, scored as ``train_ode_to_joy``'s
``model_errors`` (which uses the nominal episode alone).

Run: python benchmarks/wuji_piano/speed_piano.py --hand-dir HANDS --keyboard piano.xml --ckpt A/seed0/last.pt B/seed0/last.pt
     --out speed.json
"""

import argparse
import copy
import json
import statistics
from pathlib import Path

import mujoco_warp as mjw
import torch
from torch import nn
import warp as wp

import train_ode_to_joy as tj
from ndm_gym.wmbench.speed_mjcf import timed
from ndm_gym.worldmodels.piano import NUM_ACTUATORS as A, KEY_COUNT
from piano_updates import D, OdeSuperMimic, OdeSuperTrack


class Bf16Net(nn.Module):
    """A residual network of the world model in bf16, its input and output in fp32."""

    def __init__(self, net):
        super().__init__()
        self.net = net.to(torch.bfloat16)

    def forward(self, x):
        return self.net(x.to(torch.bfloat16)).float()


def bf16_copy(wm):
    fast = copy.deepcopy(wm)
    fast.actuator, fast.key_residual = Bf16Net(fast.actuator), Bf16Net(fast.key_residual)
    return fast.eval().requires_grad_(False)


def load(path, geometry, roll):
    ck = torch.load(path, map_location=tj.DEV, weights_only=True)
    algo = (OdeSuperMimic if ck["algo"] == "supermimic" else OdeSuperTrack)(copy.deepcopy(geometry), roll)
    algo.pi.load_state_dict(ck["pi"])
    algo.wm.load_state_dict(ck["wm"])
    algo.wm.eval().requires_grad_(False)
    return ck, algo


@torch.no_grad()
def play(sim, algo, starts, frames):
    """The policy from ``starts`` for ``frames`` frames; the observed states (frames + 1) and actions (frames), each (T, worlds, .)."""
    state, prev = sim.reset(starts), torch.zeros_like(starts)
    states, actions = [state.clone()], []
    for f in range(frames):
        prev = algo.action(state, prev, f)
        state = sim.step(prev)
        states.append(state.clone()), actions.append(prev.clone())
    return torch.stack(states), torch.stack(actions)


@torch.no_grad()
def open_loop(model, states, actions, goal, horizons, stride):
    """Due-key and all-key depth MAE at each horizon, from every ``stride``-th frame of every episode."""
    t0 = torch.arange(0, actions.shape[0] - max(horizons) + 1, stride, device=tj.DEV)
    s = states[t0].reshape(-1, D)
    out = {}
    for h in range(1, max(horizons) + 1):
        s = model(s, actions[t0 + h - 1].reshape(-1, A))
        if h in horizons:
            err = (s[:, 2 * A :] - states[t0 + h, :, 2 * A :].reshape(-1, KEY_COUNT)).abs()
            due = goal[t0 + h - 1][:, None].expand(-1, states.shape[1], -1).reshape(-1, KEY_COUNT)
            out[h] = dict(all_key_mae=err.mean().item(), due_key_mae=err[due].mean().item(), windows=len(s))
    return out


def graphed(fn, state, action):
    """One step of ``fn`` on ``state`` (fed back) and ``action`` as a CUDA graph; returns its replay."""
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture
                state.copy_(fn(state, action))
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            state.copy_(fn(state, action))
    return g.replay


def physics(m, n, starts, algo, a):
    """``n`` worlds of the scene played by ``algo``'s policy for ``a.frame`` frames from ``starts``; their observed state, the
    frame's action and the scene's step timed holding that action."""
    sim = tj.FreeHands(m, n)
    states, actions = play(sim, algo, starts, a.frame)
    state0, act = states[-1].clone(), actions[-1].clone()
    del states, actions
    runs = timed(lambda: sim.step(act), torch.cuda.synchronize, a.warmup, a.steps, a.repeats, warm_seconds=a.warm_seconds)
    return state0, act, runs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hand-dir", required=True, type=Path)
    p.add_argument("--keyboard", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path, nargs="+", help="The first checkpoint's policy also sets the timing's states.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nworld", type=int, nargs="+", default=[1024, 4096], help="16,384 worlds of the scene need more than a 40 GB GPU")
    p.add_argument("--frame", type=int, default=40)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warm-seconds", type=float, default=5.0)
    p.add_argument("--compile", default="max-autotune-no-cudagraphs", help="ndm_bf16: torch.compile's mode, or none")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--stride", type=int, default=10)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    torch.backends.cuda.matmul.allow_tf32 = True
    out = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, warp=wp.config.version, settings={k: str(v) for k, v in vars(a).items()},
               rows=[], open_loop={})  # fmt: skip
    with wp.ScopedDevice(tj.DEV):
        m = tj.build(a.hand_dir, a.keyboard)
        roll = torch.as_tensor(tj.piano_roll(), device=tj.DEV)
        test = tj.starts(256, 789, 0.20)
        geometry = tj.fit_geometry(m)
        algos = [load(c, geometry, roll) for c in a.ckpt]
        sim = tj.FreeHands(m, len(test))
        for path, (ck, algo) in zip(a.ckpt, algos, strict=True):  # each policy's evaluation episodes, each model's open-loop error
            states, actions = play(sim, algo, test, len(roll))
            errors = {
                name: open_loop(model, states, actions, roll, a.horizons, a.stride)
                for name, model in (("fp32", algo.wm), ("bf16", bf16_copy(algo.wm)))
            }
            out["open_loop"][str(path)] = dict(algo=ck["algo"], seed=ck["seed"], round=ck["round"], **errors)
            print(json.dumps({str(path): out["open_loop"][str(path)]}), flush=True)
        del sim, states, actions
        for n in a.nworld:
            state0, act, runs = physics(m, n, test.repeat((n + len(test) - 1) // len(test), 1)[:n], algos[0][1], a)
            rows = [
                ("mujoco_warp", None, runs, f"FreeHands: 10 physics steps as one CUDA graph, controls and observation; mujoco_warp {mjw.__version__}")
            ]
            torch.cuda.empty_cache()
            for path, (ck, algo) in zip(a.ckpt, algos, strict=True):
                for name, model in (("ndm_fp32", algo.wm), ("ndm_bf16", bf16_copy(algo.wm))):
                    fn = model if name == "ndm_fp32" or a.compile == "none" else torch.compile(model, mode=a.compile, dynamic=False)
                    state = state0.clone()
                    step = graphed(fn, state, act)
                    runs = timed(step, torch.cuda.synchronize, a.warmup, a.steps, a.repeats, warm_seconds=a.warm_seconds)
                    detail = (
                        "eager step, one CUDA graph, TF32 matrix products"
                        if name == "ndm_fp32"
                        else f"residual networks in bf16, compiled ({a.compile}), one CUDA graph"
                    )
                    rows.append((name, path, runs, detail))
            for name, path, runs, detail in rows:
                sec = statistics.median(runs)
                row = dict(what=name, ckpt=str(path) if path else None, nworld=n, steps=a.steps, seconds=runs, median_s=sec,
                           world_steps_per_s=n * a.steps / sec, ms_per_step=1e3 * sec / a.steps, detail=detail)  # fmt: skip
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
