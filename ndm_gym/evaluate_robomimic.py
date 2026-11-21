"""Evaluate a manipulation tracking policy (train_robomimic's last.pt or a kept checkpoint) on its demonstrations, in the simulator.

    python -m ndm_gym.evaluate_robomimic --ckpt logs/<name>/last.pt --out logs/<name>/evaluation

Per demonstration: ``--num_envs`` rollouts of the whole demonstration from perturbed starts (training's reset noise on the
arms' joints at its first frame), without exploration noise and without resets, and as many with the reference's joint positions
as the servos' targets in the policy's place (zero actions, open loop); every demonstration's rollouts run side by side, each
world to its own demonstration's last frame. A rollout succeeds if it never terminates and meets robosuite's success test at
the demonstration's last frame. The ALOHA tasks: if it never terminates and ACT's reward, read off the simulator's contacts
after every control step, reaches its maximum at one of them, as ACT counts an episode a success (``stage_rates``: the
shares of the rollouts whose highest reward reaches each stage; ``final_state_rate``: the share that never terminates and
meets the state test of the last stage at the demonstration's last frame, robots.aloha_scenes, as the robomimic tasks read
their success); and as many rollouts again under the position targets the demonstration recorded
(``demonstration_targets``, open loop). The errors are means over the frames of the rollouts that do not terminate;
``object_final_mm`` the tracked objects' mean error at the last frame; ``policy`` and ``reference_targets`` at the top level are
the means over the demonstrations. One rollout from each demonstration's unperturbed first frame is saved as
``rollout_<demo>.npz`` (the state at every control step, beside the same start under the reference's targets; the step of its
termination, -1 without one, and whether it succeeds: tools/render_robomimic.py renders it). ``--demo``:
other demonstrations of the task than the checkpoint's (held-out ones, of its scene).
"""

from __future__ import annotations

import argparse
import json
import math
import os


def rollout(sim, algo, task, starts, targets: str = "policy"):
    """Every world from its frame of ``starts`` (N,) (perturbed as the simulator's noise says) to its demonstration's last,
    under the ``policy`` or, without it, under the ``reference``'s joint positions as the targets or the ``demonstration``'s
    recorded ones; the worlds whose demonstration ends earlier hold its last frame until the longest ends. Returns the states
    (T, N, D), each world's number of steps, whether it terminated within them, the step of its termination (its number of
    steps without one), whether it succeeds (the module's docstring), its highest staged reward before it ends (zeros
    for a scene without one) and whether it never terminates and meets the scene's state test at its last frame."""
    import torch

    from ndm_gym.robots.aloha_scenes import MAX_REWARD
    from ndm_gym.tasks.robomimic_tracking import unflatten_state

    every = torch.ones(sim.N, dtype=torch.bool, device=sim.device)
    sim.reset(every, starts)
    lens = task.ref.end[starts] - starts
    a_prev = torch.zeros(sim.N, task.na, device=sim.device)
    failed = torch.zeros(sim.N, dtype=torch.bool, device=sim.device)
    best = torch.zeros(sim.N, dtype=torch.long, device=sim.device)
    until = lens.clone()
    states = [sim.state()]
    for t in range(int(lens.max())):
        f = sim.f.clone()
        a = algo.act(unflatten_state(states[-1], task.nq), a_prev, f) if targets == "policy" else a_prev
        s1, term, _ = sim.step(task.recorded_setpoint(f) if targets == "demonstration" else task.setpoint(a, f), reset=False)
        term = term & (t < lens)  # (a world past its last frame holds it)
        until = torch.where(term & ~failed, t + 1, until)
        failed |= term
        if task.scene.contacts:
            best = torch.where((t < lens) & ~failed, torch.maximum(best, task.staged_reward(sim.touching())), best)
        a_prev = a
        states.append(s1)
    states = torch.stack(states)
    final = task.success(unflatten_state(states[lens, torch.arange(sim.N, device=sim.device)], task.nq)) & ~failed
    success = (best == MAX_REWARD) & ~failed if task.scene.contacts else final
    return states, lens, failed, until, success, best, final


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256, help="rollouts per demonstration from perturbed starts")
    parser.add_argument("--demo", type=int, nargs="+", default=None, help="the demonstrations to evaluate on (default: the checkpoint's)")
    parser.add_argument("--data_dir", type=str, default=None, help="the demonstrations' directory (default: the checkpoint's)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_reference", action="store_true", help="skip the rollouts under the reference's targets")
    parser.add_argument("--solver_iterations", type=int, default=-1,
                        help="the simulated scene's solver iterations: -1 the checkpoint's training physics, 0 the scene's own (robosuite's), "
                        "else a cap (its line search: --solver_ls_iterations)")  # fmt: skip
    parser.add_argument("--solver_ls_iterations", type=int, default=0)
    args = parser.parse_args()
    if args.solver_ls_iterations and args.solver_iterations <= 0:
        parser.error("--solver_ls_iterations caps the line search of a --solver_iterations cap (> 0)")
    if args.num_envs < 1:
        parser.error("--num_envs >= 1")

    import numpy as np
    import torch

    from ndm_gym.algos.robomimic_algos import ROBOMIMIC_ALGOS
    from ndm_gym.motion.sources.robomimic import demo_path, load_demo
    from ndm_gym.robots.aloha_scenes import MAX_REWARD
    from ndm_gym.sim.rigid.robomimic_sim import RobomimicSim
    from ndm_gym.tasks.robomimic_tracking import unflatten_state
    from ndm_gym.train_robomimic import build

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    demo = targs.demo if args.demo is None else args.demo
    demos = demo if isinstance(demo, list) else [demo]
    data_dir = args.data_dir or targs.data_dir
    # robosuite draws Lift's cube and Transport's hammer per demonstration: the policy's scene only (ACT's scene is one)
    if args.demo is not None and not targs.task.startswith("aloha_"):
        trained = targs.demo if isinstance(targs.demo, list) else [targs.demo]
        path = demo_path(data_dir, targs.task)
        scene = load_demo(path, trained[0])["xml"]
        if any(load_demo(path, i)["xml"] != scene for i in demos):
            raise ValueError(f"--demo {demos}: not recorded in the scene of the checkpoint's demonstrations {trained}")
    if args.solver_iterations < 0:  # the physics trained in
        solver = (targs.solver_iterations, targs.solver_ls_iterations)
    else:
        solver = (args.solver_iterations, args.solver_ls_iterations if args.solver_iterations else 0)
    model, task, (nconmax, njmax) = build(targs.task, demos, data_dir, dev, solver_iterations=solver[0],
                                          solver_ls_iterations=solver[1])  # fmt: skip
    targs.update_graphs = False  # (plain optimizers: none is stepped here)
    ref, D = task.ref, len(demos)
    firsts = torch.unique(ref.start)  # each demonstration's first frame, in --demo's order
    sim = RobomimicSim(model, task, args.num_envs * D, dev, nconmax=nconmax, njmax=njmax)
    nominal = RobomimicSim(model, task, D, dev, nconmax=nconmax, njmax=njmax, joint_noise=0.0)
    algo = ROBOMIMIC_ALGOS[targs.algo](targs, model, task, dev, substeps=sim.substeps)
    algo.wm.load_state_dict(ck["wm"]), algo.pi.load_state_dict(ck["pi"])

    def summarize(out, worlds: torch.Tensor, start: int) -> dict:
        """The metrics of the ``worlds`` (all of one demonstration, starting at ``start``) of a rollout's output."""
        states, lens, failed, until, success, best, final = (x[:, worlds] if x.dim() > 1 else x[worlds] for x in out)
        n = int(lens[0])
        S = unflatten_state(states[: n + 1], task.nq)
        e = task.errors(S, task.fk(S[0]), torch.arange(start, start + n + 1, device=dev).unsqueeze(-1))
        ok = ~failed
        mean = lambda x: float(x[:, ok].mean()) if bool(ok.any()) else math.nan  # noqa: E731
        m = {"success_rate": float(success.float().mean()), "terminated": float(failed.float().mean()),
             "tracked_s": float(until.float().mean() * task.control_dt), "object_mm": 1000 * mean(e["object_pos"].mean(-1)),
             "object_final_mm": 1000 * float(e["object_pos"][-1].mean()), "object_deg": math.degrees(mean(e["object_ori_sq"].sqrt().mean(-1))),
             "keypoint_mm": 1000 * mean(e["keypoint"].mean(-1)), "joint_rad": mean(e["joint"]), "finger_mm": 1000 * mean(e["finger"])}  # fmt: skip
        if task.scene.contacts:  # the shares of the rollouts whose highest staged reward reaches 1, 2, ...; the state test
            m["stage_rates"] = [float((best >= k).float().mean()) for k in range(1, MAX_REWARD + 1)]
            m["final_state_rate"] = float(final.float().mean())
        return m

    os.makedirs(args.out, exist_ok=True)
    metrics = {"task": targs.task, "demos": demos, "iteration": ck["iteration"], "wall_time_s": ck.get("wall_time"), "control_dt": task.control_dt,
               "num_envs": args.num_envs, "solver_iterations": [int(model.opt.iterations), int(model.opt.ls_iterations)], "per_demo": {}}  # fmt: skip
    group = torch.arange(args.num_envs * D, device=dev) // args.num_envs  # each world's demonstration
    runs = {"nominal": rollout(nominal, algo, task, firsts), "policy": rollout(sim, algo, task, firsts[group])}
    open_loop = rollout(nominal, algo, task, firsts, "reference")[0]
    if not args.no_reference:
        runs["reference_targets"] = rollout(sim, algo, task, firsts[group], "reference")
        if ref.target is not None:
            runs["demonstration_targets"] = rollout(sim, algo, task, firsts[group], "demonstration")
    for d, (name, start) in enumerate(zip(demos, firsts.tolist(), strict=True)):
        end = int(ref.end[start])
        m = {"frames": end - start + 1, "nominal": summarize(runs["nominal"], torch.tensor([d], device=dev), start)}
        for key in ("policy", "reference_targets", "demonstration_targets"):
            if key in runs:
                m[key] = summarize(runs[key], torch.nonzero(group == d).squeeze(-1), start)
        n = end - start
        _, _, failed, until, success, _, _ = runs["nominal"]
        np.savez(os.path.join(args.out, f"rollout_{name}.npz"), state=runs["nominal"][0][: n + 1, d].cpu().numpy(),
                 reference_qpos=ref.qpos[start : end + 1].cpu().numpy(), reference_targets_state=open_loop[: n + 1, d].cpu().numpy(), nq=task.nq,
                 control_dt=task.control_dt, task=targs.task, demo=name, terminated_at=int(until[d]) if bool(failed[d]) else -1,
                 success=bool(success[d]))  # fmt: skip
        metrics["per_demo"][str(name)] = m
    for key in ("policy", "reference_targets", "demonstration_targets"):  # the means over the demonstrations
        rows = [m[key] for m in metrics["per_demo"].values() if key in m]
        if rows:
            metrics[key] = {k: np.nanmean([r[k] for r in rows], 0).tolist() for k in rows[0]}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
