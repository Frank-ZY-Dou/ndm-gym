"""Evaluate a hand-object tracking policy (train_hand's last.pt) on its clip, in the simulator.

    python -m ndm_gym.evaluate_hand --ckpt logs/<name>/last.pt --out logs/<name>/evaluation

One rollout from the unperturbed first frame to the last, without resets, is saved as ``rollout.npz`` (the scene's
qpos at every control step, for tools/render_hand.py) with its errors in ``metrics.json``, beside the same rollout
with the reference pose as the position targets in the policy's place (``reference_targets_qpos``). ``--num_envs``
rollouts from perturbed starts (training's reset noise) give the success rate and the errors, and as many with the
reference pose as the targets those to compare with. A rollout succeeds if it never terminates (the object and the
palm stay within the task's distances of the reference) and the object ends within ``--success_pos`` of its last
reference position.
"""

from __future__ import annotations

import argparse
import json
import os


def rollout(sim, algo, task, policy: bool = True):
    """Every world from its current start to the reference's last frame, under the policy or, without it, under the
    reference pose as the position targets. Returns the states (T, N, STATE_DIM), the scene's qpos (T, N, nq) and
    whether each world terminated."""
    import torch

    from ndm_gym.tasks.hand_tracking import unflatten_state

    a_prev = torch.zeros(sim.N, task.kin.q_lo.shape[0], device=sim.device)
    failed = torch.zeros(sim.N, dtype=torch.bool, device=sim.device)
    states, qpos = [sim.state()], [sim.qpos.clone()]
    for _ in range(task.ref.frames - 1):
        f = sim.f.clone()
        a = algo.act(unflatten_state(states[-1]), a_prev, f) if policy else a_prev
        s1, term, _ = sim.step(task.setpoint(a, f), reset=False)
        failed |= term
        a_prev = a
        states.append(s1), qpos.append(sim.qpos.clone())
    return torch.stack(states), torch.stack(qpos), failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256, help="rollouts from perturbed starts")
    parser.add_argument("--success_pos", type=float, default=0.05, help="the object's largest final distance from its reference (m)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import torch

    from ndm_gym.algos.hand_algos import HAND_ALGOS
    from ndm_gym.sim.rigid.hand_sim import HandSim
    from ndm_gym.state import CONTROL_DT
    from ndm_gym.tasks.hand_tracking import unflatten_state
    from ndm_gym.train_hand import build

    torch.manual_seed(args.seed)
    dev = "cuda:0"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    model, kin, task = build(targs.clip, dev)
    algo = HAND_ALGOS[targs.algo](targs, kin, task, dev)
    algo.load_state_dict(ck)
    ref, frames = task.ref, torch.arange(task.ref.frames, device=dev)

    def summarize(states: torch.Tensor, failed: torch.Tensor) -> dict:
        S = unflatten_state(states)
        e = task.errors(S, kin.forward(S[0]), frames.unsqueeze(-1))
        final = e["object_pos"][-1]
        ok = ~failed & (final < args.success_pos)
        mm = lambda x: float(x.mean() * 1000)  # noqa: E731
        return {"success_rate": float(ok.float().mean()), "terminated": float(failed.float().mean()),
                "object_pos_mm": mm(e["object_pos"]), "object_pos_final_mm": mm(final), "object_pos_max_mm": float(e["object_pos"].max() * 1000),
                "object_ori_rad": float(e["object_ori_sq"].sqrt().mean()), "fingertip_mm": mm(e["keypoint"][..., 1:]),
                "palm_mm": mm(e["keypoint"][..., 0]), "palm_ori_rad": float(e["palm_ori_sq"].sqrt().mean()),
                "object_lift_m": float((S[2][-1, :, 2] - S[2][0, :, 2]).mean())}  # fmt: skip

    sim = HandSim(model, task, 1, dev, joint_noise=0.0, object_noise=0.0)
    states, qpos, failed = rollout(sim, algo, task)
    metrics = {"clip": targs.clip, "iteration": ck["iteration"], "frames": ref.frames, "control_dt": CONTROL_DT,
               "reference_lift_m": float(ref.p[-1, 2] - ref.p[0, 2]), "nominal": summarize(states, failed)}  # fmt: skip
    sim.reset(torch.ones(1, dtype=torch.bool, device=dev))
    reference_targets_qpos = rollout(sim, algo, task, policy=False)[1]
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "rollout.npz"), qpos=qpos[:, 0].cpu().numpy(), reference_targets_qpos=reference_targets_qpos[:, 0].cpu().numpy(),
             reference_q=ref.q.cpu().numpy(), reference_p=ref.p.cpu().numpy(), reference_R=ref.R.cpu().numpy(), control_dt=CONTROL_DT)  # fmt: skip
    sim = HandSim(model, task, args.num_envs, dev)
    states, _, failed = rollout(sim, algo, task)
    metrics["perturbed"] = {"num_envs": args.num_envs, **summarize(states, failed)}
    sim.reset(torch.ones(args.num_envs, dtype=torch.bool, device=dev))
    metrics["reference_targets"] = {"num_envs": args.num_envs, **summarize(*rollout(sim, algo, task, policy=False)[::2])}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
