"""Bounded CPU Door development runner; not the measured GPU training path.

Supply a supported native MJCF and a physically replayed reference. Saved final
policies are evaluated by controls in physics; reference states are never assigned
after reset. Model fitting on this clip is not held-out model validation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algos.door import DoorResidual
from .algos.door_supertrack import DoorSuperTrack
from .algos.door_supermimic import DoorSuperMimic
from .motion.adroit_door import JOINT_NAMES, DoorReference
from .motion.door_replay import BalancedDoorReplay, DoorReplay, collect_physical_replay
from .sim.rigid.adroit_door import DoorPhysics
from .tasks.adroit_door import DoorTask
from .worldmodels.mjcf import MjcfLayout


def evaluate(algo, sim, reference, offsets):
    """Evaluate every complete horizon; offsets are ARRz radians, index by name."""
    results, rollouts = [], {}
    for episode, offset in enumerate(offsets):
        dq = np.zeros(30)
        dq[JOINT_NAMES.index("ARRz")] = offset
        state = sim.reset(qpos_offset=dq)
        device = algo.reference_actions.device
        previous = torch.zeros(28, device=device)
        qs, vs, actions, rewards = [state[0]], [state[1]], [], []
        started = time.perf_counter()
        with torch.no_grad():
            for frame, prior in enumerate(algo.reference_actions):
                st = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in state)
                action = algo.pi(algo.task.obs(st, previous),
                                 algo.task.goal(st, torch.tensor(frame, device=device)), prior)
                state, _ = sim.step(action.cpu().numpy())
                nxt = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in state)
                rewards.append(float(algo.task.reward(nxt, torch.tensor(frame + 1, device=device), action, previous)))
                previous = action
                qs.append(state[0])
                vs.append(state[1])
                actions.append(action.cpu().numpy().copy())
        q, v = np.asarray(qs), np.asarray(vs)
        delta = q[1:] - reference.qpos[1:]
        results.append({"ARRz_offset_rad": offset, "wall_seconds": time.perf_counter() - started,
                        "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
                        "opened_final": bool(q[-1, 28] >= algo.task.cfg.opening_rad),
                        "opened_ever": bool(q[:, 28].max() >= algo.task.cfg.opening_rad),
                        "final_door_rad": float(q[-1, 28]), "tracking_return": sum(rewards),
                        "translation_rmse_m": float(np.sqrt(np.mean(delta[:, 0]**2))),
                        "hand_rmse_rad": float(np.sqrt(np.mean(delta[:, 1:28]**2))),
                        "door_rmse_rad": float(np.sqrt(np.mean(delta[:, 28]**2))),
                        "latch_rmse_rad": float(np.sqrt(np.mean(delta[:, 29]**2)))})
        for name, value in (("qpos", q), ("qvel", v), ("actions", actions)):
            rollouts[f"episode{episode}_{name}"] = value
    return results, rollouts


def collect_teacher(algo, sim, offsets, gain):
    """Collect bounded joint feedback in physics, with no post-reset state writes."""
    if not offsets or not np.isfinite(offsets).all() or not np.isfinite(gain) or gain < 0:
        raise ValueError("Finite nonempty offsets and nonnegative gain required")
    observations, goals, targets = [], [], []
    for offset in offsets:
        dq = np.zeros(30)
        dq[JOINT_NAMES.index("ARRz")] = offset
        state = sim.reset(qpos_offset=dq)
        previous = torch.zeros(28)
        for frame, prior in enumerate(algo.reference_actions):
            st = tuple(torch.as_tensor(x, dtype=torch.float32) for x in state)
            correction = (gain * (algo.task.q[frame, :28] - st[0][:28])
                          / algo.task.ctrl_half).clamp(-.1, .1)
            observations.append(algo.task.obs(st, previous))
            goals.append(algo.task.goal(st, torch.tensor(frame)))
            targets.append(correction / .1)
            previous = (prior + correction).clamp(-1, 1)
            state, _ = sim.step(previous.numpy())
    batch = tuple(torch.stack(x) for x in (observations, goals, targets))
    if not all(torch.isfinite(x).all() for x in batch):
        raise FloatingPointError("Nonfinite teacher collection")
    return batch


DOOR_METHODS = {"residual": DoorResidual, "supertrack": DoorSuperTrack, "supermimic": DoorSuperMimic}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Supported native MJCF with scene already restored")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory outside the repository")
    p.add_argument("--method", choices=tuple(DOOR_METHODS), default="residual")
    p.add_argument("--critic-epochs", type=int, default=1)
    p.add_argument("--critic-minibatches", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wm-updates", type=int, default=100)
    p.add_argument("--policy-updates", type=int, default=0,
                   help="Diagnostic only: model gradients have not established physical benefit")
    p.add_argument("--supervised-updates", type=int, default=0,
                   help="Fit bounded analytic joint-feedback targets; changes learned policy")
    p.add_argument("--train-offsets", type=float, nargs="+", default=[-.01, -.005, .005, .01])
    p.add_argument("--physical-replay", action="store_true",
                   help="Collect physical reference-control episodes; hold out last episode; changes training data")
    p.add_argument("--replay-dataset", type=Path,
                   help="Saved physical checkpoint evaluation episodes; last episode held out; changes data")
    p.add_argument("--mixed-replay", action="store_true",
                   help="Equal reference-control/saved-policy windows; last episode held out per source")
    p.add_argument("--replay-noise", type=float, default=.02)
    p.add_argument("--teacher-gain", type=float, default=.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--offsets", type=float, nargs="+", default=[0., -.01, -.005, .005, .01],
                   help="Initial ARRz perturbations in radians; fixed before execution")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.wm_updates, args.policy_updates, args.supervised_updates) < 0 or args.batch_size < 1:
        raise ValueError("Updates must be nonnegative and batch size positive")
    if not all(np.isfinite(args.offsets)):
        raise ValueError("Offsets must be finite")
    if (not np.isfinite(args.train_offsets).all() or not np.isfinite(args.teacher_gain)
            or args.teacher_gain < 0):
        raise ValueError("Teacher parameters must be finite and gain nonnegative")
    if args.method != "residual" and args.supervised_updates:
        raise ValueError("Model-method runs cannot include supervised updates")
    if args.mixed_replay and (args.replay_dataset is None or args.batch_size % 2):
        raise ValueError("Mixed replay requires saved dataset and even batch size")
    if args.physical_replay and args.replay_dataset is not None and not args.mixed_replay:
        raise ValueError("Choose physical collection or saved replay, not both")
    import mujoco

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    reference = DoorReference.load(args.reference)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    sim = DoorPhysics(model, reference)
    task = DoorTask(reference, model.actuator_ctrlrange)
    layout = replace(MjcfLayout.from_model(model), dt=task.dt)
    actions = torch.as_tensor(reference.actions, dtype=torch.float32).clamp(-1, 1)
    critic = dict(critic_epochs=args.critic_epochs, critic_minibatches=args.critic_minibatches) if args.method == "supermimic" else {}
    algo = DOOR_METHODS[args.method](task, layout, actions, hidden=(32, 32), horizon=args.horizon, **critic)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(device="CPU", threads=1, dtype="float32", path="eager",
                    selection="final checkpoint only", heldout_validation=False,
                    three_seed_evidence=False, mujoco=mujoco.__version__, torch=torch.__version__)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    controls = task.setpoint(actions)
    if not args.physical_replay and args.replay_dataset is None:
        algo.wm.update_norms(task.q[:-1], task.v[:-1], controls, task.q[1:], task.v[1:])
    replay = None
    if args.physical_replay or args.replay_dataset is not None:
        if args.replay_dataset is None and len(args.train_offsets) < 2:
            raise ValueError("Physical replay requires at least two episodes for holdout")
        if args.replay_dataset is None:
            collected = collect_physical_replay(algo, sim, args.train_offsets, args.replay_noise, args.seed)
        else:
            provenance = json.loads((args.replay_dataset.parent / "result.json").read_text())
            for name in ("model", "reference"):
                digest = hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                if provenance.get(f"{name}_sha256") != digest:
                    raise ValueError(f"Replay {name} identity mismatch")
            protocol["replay_source_checkpoint_sha256"] = provenance["checkpoint_sha256"]
            collected = DoorReplay.load(args.replay_dataset)
            if any(len(e[2]) != len(actions) for e in collected.episodes):
                raise ValueError("Replay episode length must match reference")
            protocol["replay_dataset_sha256"] = hashlib.sha256(args.replay_dataset.read_bytes()).hexdigest()
        protocol["heldout_episode"] = len(collected.episodes)-1
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        replay, heldout = collected.split([len(collected.episodes)-1])
        if args.mixed_replay:
            if len(args.train_offsets) < 2:
                raise ValueError("Mixed replay requires reference episode holdout")
            reference_replay = collect_physical_replay(
                algo, sim, args.train_offsets, args.replay_noise, args.seed)
            reference_train, reference_heldout = reference_replay.split([len(reference_replay.episodes)-1])
            replay = BalancedDoorReplay([reference_train, replay])
            heldout = DoorReplay(reference_heldout.episodes + heldout.episodes)
            protocol["replay_sources"] = ["reference_controls", "saved_policy"]
            protocol["replay_sampling"] = "equal windows per source in each batch"
            protocol["heldout_episode_per_source"] = [len(reference_replay.episodes)-1, len(collected.episodes)-1]
            (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        saved = {}
        for split, data in (("train", replay), ("heldout", heldout)):
            for i, episode in enumerate(data.episodes):
                for name, value in zip(("qpos", "qvel", "actions"), episode):
                    saved[f"{split}{i}_{name}"] = value.numpy()
        np.savez_compressed(args.output / "replay.npz", **saved)
        q, v, a = (torch.cat([e[i][:-1] if i < 2 else e[i] for e in replay.episodes]) for i in range(3))
        nq, nv = (torch.cat([e[i][1:] for e in replay.episodes]) for i in range(2))
        algo.wm.update_norms(q, v, task.setpoint(a), nq, nv)
    previous = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]))
    timings = {"wm": [], "policy": [], "supervised": []}
    if args.supervised_updates:
        start = time.perf_counter()
        teacher = collect_teacher(algo, sim, args.train_offsets, args.teacher_gain)
        protocol["collection_seconds"] = time.perf_counter() - start
        np.savez_compressed(args.output / "teacher-data.npz",
                            **dict(zip(("obs", "goal", "target"), (x.numpy() for x in teacher))))
        algo.pi.norm.update(torch.cat(teacher[:2], -1))
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with (args.output / "updates.jsonl").open("w") as log:
        for kind, count in (("wm", args.wm_updates), ("policy", args.policy_updates),
                            ("supervised", args.supervised_updates)):
            for step in range(count):
                start = time.perf_counter()
                f = torch.randint(task.frames - args.horizon, (args.batch_size,))
                if replay is not None:
                    rq, rv, ra, f, rp = replay.sample(args.batch_size, args.horizon)
                if kind == "wm" and replay is not None:
                    loss = algo.wm_loss(rq, rv, task.setpoint(ra))
                elif kind == "wm":
                    loss = algo.wm_loss(torch.stack((task.q[f], task.q[f+1])),
                                        torch.stack((task.v[f], task.v[f+1])), controls[f][None])
                if kind == "wm":
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite model loss")
                    algo.opt_wm.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(algo.wm.parameters(), 1., error_if_nonfinite=True)
                    algo.opt_wm.step()
                elif kind == "supervised":
                    indices = torch.randint(len(teacher[0]), (args.batch_size,))
                    loss = algo.supervised_update(*(x[indices] for x in teacher))
                else:
                    loss, _ = algo.policy_update((rq[0], rv[0]), f, rp) if replay is not None else algo.policy_update((task.q[f], task.v[f]), f, previous[f])
                elapsed = time.perf_counter() - start
                timings[kind].append(elapsed)
                log.write(json.dumps(dict(kind=kind, update=step+1, loss=float(loss.detach()), seconds=elapsed)) + "\n")
                log.flush()
    torch.save({"algo": algo.state_dict(), "rng": torch.get_rng_state(), "protocol": protocol},
               args.output / "final.pt")
    metrics, rollouts = evaluate(algo, sim, reference, args.offsets)
    np.savez_compressed(args.output / "physical-rollouts.npz", **rollouts)
    result = {"evaluations": metrics, "timing": {k: {"updates": len(v), "total_seconds": sum(v),
              "median_seconds": float(np.median(v)) if v else None} for k, v in timings.items()},
              "method": args.method, "learned_policy": args.policy_updates + args.supervised_updates > 0, "three_seed_evidence": False}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
