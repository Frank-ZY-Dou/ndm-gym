"""Evaluate a manipulation tracking policy (train_robomimic's last.pt or a kept checkpoint) on its demonstrations, in the simulator.

    python -m ndm_gym.evaluate_robomimic --ckpt logs/<name>/last.pt --out logs/<name>/evaluation

Per demonstration: ``--num_envs`` rollouts of the whole demonstration from perturbed starts (training's reset noise on the
arms' joints at its first frame), without exploration noise and without resets, and as many with the reference's joint positions
as the servos' targets in the policy's place (zero actions, open loop). A rollout succeeds if it never terminates and meets
robosuite's success test at the demonstration's last frame. The errors are means over the frames of the rollouts that do not
terminate; ``object_final_mm`` the tracked objects' mean error at the last frame; ``policy`` and ``reference_targets`` at the
top level are the means over the demonstrations. One rollout from each demonstration's unperturbed first frame is saved as
``rollout_<demo>.npz`` (the state at every control step, beside the same start under the reference's targets). ``--demo``:
other demonstrations of the task than the checkpoint's (held-out ones, of its scene).
"""

from __future__ import annotations

import argparse
import json
import math
import os


def rollout(sim, algo, task, start: int, policy: bool = True):
    """Every world from the frame ``start`` (perturbed as the simulator's noise says) to its demonstration's last, under the
    policy or, without it, under the reference's targets. Returns the states (T, N, D), whether each world terminated, the step of its
    termination (T - 1 without one) and whether it succeeds at the end."""
    import torch

    from ndm_gym.tasks.robomimic_tracking import unflatten_state

    every = torch.ones(sim.N, dtype=torch.bool, device=sim.device)
    sim.reset(every, torch.full((sim.N,), start, dtype=torch.long, device=sim.device))
    steps = int(task.ref.end[start]) - start
    a_prev = torch.zeros(sim.N, task.nu, device=sim.device)
    failed = torch.zeros(sim.N, dtype=torch.bool, device=sim.device)
    until = torch.full((sim.N,), steps, dtype=torch.long, device=sim.device)
    states = [sim.state()]
    for t in range(steps):
        f = sim.f.clone()
        a = algo.act(unflatten_state(states[-1], task.nq), a_prev, f) if policy else a_prev
        s1, term, _ = sim.step(task.setpoint(a, f), reset=False)
        until = torch.where(term & ~failed, t + 1, until)
        failed |= term
        a_prev = a
        states.append(s1)
    success = task.success(unflatten_state(states[-1], task.nq)) & ~failed
    return torch.stack(states), failed, until, success


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256, help="rollouts from perturbed starts")
    parser.add_argument("--demo", type=int, nargs="+", default=None, help="the demonstrations to evaluate on (default: the checkpoint's)")
    parser.add_argument("--data_dir", type=str, default=None, help="the demonstrations' directory (default: the checkpoint's)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_reference", action="store_true", help="skip the rollouts under the reference's targets")
    args = parser.parse_args()

    import numpy as np
    import torch

    from ndm_gym.algos.robomimic_algos import ROBOMIMIC_ALGOS
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim
    from ndm_gym.tasks.robomimic_tracking import CONTROL_DT, unflatten_state
    from ndm_gym.train_robomimic import build

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    demo = targs.demo if args.demo is None else args.demo
    model, task, (nconmax, njmax) = build(targs.task, demo, args.data_dir or targs.data_dir, dev, solver_iterations=getattr(targs, "solver_iterations", 0),
                                          solver_ls_iterations=getattr(targs, "solver_ls_iterations", 0))  # fmt: skip  # the physics trained in
    targs.update_graphs = False  # (plain optimizers: none is stepped here)
    sim = RobomimicSim(model, task, args.num_envs, dev, nconmax=nconmax, njmax=njmax)
    nominal = RobomimicSim(model, task, 1, dev, nconmax=nconmax, njmax=njmax, joint_noise=0.0)
    algo = ROBOMIMIC_ALGOS[targs.algo](targs, model, task, dev, substeps=sim.substeps)
    algo.wm.load_state_dict(ck["wm"]), algo.pi.load_state_dict(ck["pi"])

    def summarize(states, failed, until, success, start: int) -> dict:
        frames = torch.arange(start, start + states.shape[0], device=dev).unsqueeze(-1)
        S = unflatten_state(states, task.nq)
        e = task.errors(S, task.fk(S[0]), frames)
        ok = ~failed
        mean = lambda x: float(x[:, ok].mean()) if bool(ok.any()) else math.nan  # noqa: E731
        return {"success_rate": float(success.float().mean()), "terminated": float(failed.float().mean()),
                "tracked_s": float(until.float().mean() * CONTROL_DT), "object_mm": 1000 * mean(e["object_pos"].mean(-1)),
                "object_final_mm": 1000 * float(e["object_pos"][-1].mean()), "object_deg": math.degrees(mean(e["object_ori_sq"].sqrt().mean(-1))),
                "keypoint_mm": 1000 * mean(e["keypoint"].mean(-1)), "joint_rad": mean(e["joint"]), "finger_mm": 1000 * mean(e["finger"])}  # fmt: skip

    os.makedirs(args.out, exist_ok=True)
    demos = demo if isinstance(demo, list) else [demo]
    metrics = {"task": targs.task, "demos": demos, "iteration": ck["iteration"], "wall_time_s": ck.get("wall_time"), "control_dt": CONTROL_DT,
               "num_envs": args.num_envs, "per_demo": {}}  # fmt: skip
    ref = task.ref
    for name, start in zip(demos, torch.unique(ref.start).tolist(), strict=True):
        end, m = int(ref.end[start]), {}
        m["frames"] = end - start + 1
        states, *rest = rollout(nominal, algo, task, start)
        m["nominal"] = summarize(states, *rest, start)
        open_loop = rollout(nominal, algo, task, start, policy=False)[0]
        np.savez(os.path.join(args.out, f"rollout_{name}.npz"), state=states[:, 0].cpu().numpy(), reference_qpos=ref.qpos[start : end + 1].cpu().numpy(),
                 reference_targets_state=open_loop[:, 0].cpu().numpy(), nq=task.nq, control_dt=CONTROL_DT)  # fmt: skip
        m["policy"] = summarize(*rollout(sim, algo, task, start), start)
        if not args.no_reference:
            m["reference_targets"] = summarize(*rollout(sim, algo, task, start, policy=False), start)
        metrics["per_demo"][str(name)] = m
    for key in ("policy", "reference_targets"):  # the means over the demonstrations
        rows = [m[key] for m in metrics["per_demo"].values() if key in m]
        if rows:
            metrics[key] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
