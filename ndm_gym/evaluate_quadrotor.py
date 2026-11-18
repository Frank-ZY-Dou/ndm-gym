"""Evaluate a quadrotor tracking policy (train_quadrotor's last.pt or a kept checkpoint) on its flights, in the simulator.

    python -m ndm_gym.evaluate_quadrotor --ckpt logs/<name>/last.pt --out logs/<name>/evaluation

Per flight: ``--num_envs`` rollouts of the whole flight from perturbed starts (training's reset noise at the flight's first
frame), without exploration noise and without resets, and as many with the reference's feedforward command alone in the
policy's place (zero actions: the thrust and the body rates of the reference, open loop). A rollout succeeds if it never
terminates (the position stays within the task's distance of the reference). The errors are means over the frames of the
rollouts that succeed, ``pos_max_mm`` the largest position error of those rollouts after the flight's first second (the
perturbed start); ``flown_s`` is the mean time in the air before a termination (the flight's length for a success).
One rollout from the unperturbed first frame is saved as ``rollout_<flight>.npz`` (the state at every control step, beside
the same start flown under the reference's command and the flight's gates, for tools/render_quadrotor.py). ``--flight``: other flights than the checkpoint's (held-out ones). ``--inertia_scale``,
``--drag_scale``, ``--mass_scale``: the simulated drone's inertia, drag and mass times these, the policy and the reference's
feedforward command as trained.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os


SETTLE = 1.0  # s: the largest position error is taken after it (the perturbed start)


def rollout(sim, algo, task, start: int, policy: bool = True):
    """Every world from the frame ``start`` (perturbed as the simulator's noise says) to its flight's last frame, under the
    policy or, without it, under the reference's feedforward command. Returns the states (T, N, STATE_DIM), whether each
    world terminated and the step of its termination (T - 1 without one)."""
    import torch

    from ndm_gym.robots.quadrotor import NUM_U
    from ndm_gym.tasks.quadrotor_tracking import unflatten_state

    every = torch.ones(sim.N, dtype=torch.bool, device=sim.device)
    sim.reset(every, torch.full((sim.N,), start, dtype=torch.long, device=sim.device))
    steps = int(task.ref.end[start]) - start
    a_prev = torch.zeros(sim.N, NUM_U, device=sim.device)
    failed = torch.zeros(sim.N, dtype=torch.bool, device=sim.device)
    until = torch.full((sim.N,), steps, dtype=torch.long, device=sim.device)
    states = [sim.state()]
    for t in range(steps):
        f = sim.f.clone()
        a = algo.act(unflatten_state(states[-1]), a_prev, f) if policy else a_prev
        s1, term, _ = sim.step(task.setpoint(a, f), reset=False)
        until = torch.where(term & ~failed, t + 1, until)
        failed |= term
        a_prev = a
        states.append(s1)
    return torch.stack(states), failed, until


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256, help="rollouts per flight from perturbed starts")
    parser.add_argument("--flight", type=str, nargs="+", default=None, help="the flights to evaluate on (default: the checkpoint's)")
    parser.add_argument("--data_dir", type=str, default=None, help="the flights' directory (default: the checkpoint's)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_reference", action="store_true", help="skip the rollouts under the reference's feedforward command")
    parser.add_argument("--inertia_scale", type=float, default=1.0, help="the simulated drone's inertia times this (the task's and the policy's unchanged)")
    parser.add_argument("--drag_scale", type=float, default=1.0, help="the simulated drone's linear drag times this")
    parser.add_argument("--mass_scale", type=float, default=1.0, help="the simulated drone's mass times this")
    args = parser.parse_args()

    import numpy as np
    import torch

    from ndm_gym.algos.quadrotor_algos import QUAD_ALGOS
    from ndm_gym.robots.quadrotor import CONTROL_DT, QuadrotorSpec
    from ndm_gym.sim.rigid.quadrotor_sim import QuadSim
    from ndm_gym.tasks.quadrotor_tracking import unflatten_state
    from ndm_gym.motion.sources.drone_racing import load_gates
    from ndm_gym.train_quadrotor import build, flight_paths

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    flights = args.flight or targs.flight
    spec = QuadrotorSpec()
    shifted = dataclasses.replace(spec, inertia=tuple(args.inertia_scale * x for x in spec.inertia), drag=args.drag_scale * spec.drag,
                                  mass=args.mass_scale * spec.mass, thrust_to_weight=spec.thrust_to_weight / args.mass_scale)  # (the same largest thrust)
    model, _, task = build(flights, args.data_dir or targs.data_dir, dev, goal_stride=targs.goal_stride, model_spec=shifted)
    targs.update_graphs = False  # (plain optimizers: none is stepped here)
    sim = QuadSim(model, task, args.num_envs, dev)
    nominal = QuadSim(model, task, 1, dev, pos_noise=0.0, rot_noise=0.0, vel_noise=0.0, ang_noise=0.0)
    algo = QUAD_ALGOS[targs.algo](targs, task, dev, substeps=sim.substeps)
    algo.wm.load_state_dict(ck["wm"]), algo.pi.load_state_dict(ck["pi"])
    ref = task.ref

    def summarize(states: torch.Tensor, failed: torch.Tensor, until: torch.Tensor, start: int) -> dict:
        frames = torch.arange(start, start + states.shape[0], device=dev).unsqueeze(-1)
        e = task.errors(unflatten_state(states), frames)
        ok = ~failed
        mean = lambda x: float(x[:, ok].mean()) if bool(ok.any()) else math.nan  # noqa: E731
        settled = e["pos"][round(SETTLE / CONTROL_DT) :, ok]  # (after the perturbed start has been flown out)
        return {"success_rate": float(ok.float().mean()), "flown_s": float(until.float().mean() * CONTROL_DT),
                "pos_mm": 1000 * mean(e["pos"]), "pos_max_mm": 1000 * float(settled.max()) if settled.numel() else math.nan,
                "rot_deg": math.degrees(mean(e["rot_sq"].sqrt())), "vel_m_s": mean(e["vel"]), "ang_rad_s": mean(e["ang"])}  # fmt: skip

    os.makedirs(args.out, exist_ok=True)
    metrics = {"iteration": ck["iteration"], "wall_time_s": ck.get("wall_time"), "control_dt": CONTROL_DT, "num_envs": args.num_envs,
               "model": {"inertia_scale": args.inertia_scale, "drag_scale": args.drag_scale, "mass_scale": args.mass_scale}, "flights": {}}
    paths = flight_paths(flights, args.data_dir or targs.data_dir)
    for name, path, start in zip(flights, paths, torch.unique(ref.start).tolist(), strict=True):
        name = os.path.splitext(os.path.basename(name))[0]
        end = int(ref.end[start])
        m = {"frames": end - start + 1, "seconds": (end - start) * CONTROL_DT, "top_speed_m_s": float(ref.v[start : end + 1].norm(dim=-1).max())}
        states, failed, until = rollout(nominal, algo, task, start)
        m["nominal"] = summarize(states, failed, until, start)
        open_loop = rollout(nominal, algo, task, start, policy=False)[0]  # the same start under the reference's command
        np.savez(os.path.join(args.out, f"rollout_{name}.npz"), state=states[:, 0].cpu().numpy(), reference_p=ref.p[start : end + 1].cpu().numpy(),
                 reference_R=ref.R[start : end + 1].cpu().numpy(), reference_v=ref.v[start : end + 1].cpu().numpy(),
                 reference_command_state=open_loop[:, 0].cpu().numpy(), gates=load_gates(path) if os.path.exists(path) else np.zeros((0, 12)), control_dt=CONTROL_DT)  # fmt: skip
        m["policy"] = summarize(*rollout(sim, algo, task, start), start)
        if not args.no_reference:
            m["reference_command"] = summarize(*rollout(sim, algo, task, start, policy=False), start)
        metrics["flights"][name] = m
    rows = [m["policy"] for m in metrics["flights"].values()]
    metrics["policy"] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}  # the mean over the flights
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
