"""Matched complete-clip evaluation of Adroit policies and deployed control prior.

Reports tracking errors and native success separately. Single-seed results do
not estimate training-seed variance. No states are overwritten after reset.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .door_schedule import FullHandSchedule
from .motion.adroit import AdroitReference
from .sim.rigid.adroit import AdroitPhysics, success
from .sim.rigid.door_collector import ParallelNativeDoorCollector
from .tasks.adroit import AdroitTask
from .train_door_full import make_algorithm
from .train_door_online import fingerprint, optimizer_counts


def evaluation_offsets(reference, seed=20261117, episodes=256):
    q = np.zeros((episodes, reference.qpos.shape[1]))
    v = np.zeros((episodes, reference.qvel.shape[1]))
    rng = np.random.default_rng(seed)
    for j in range(reference.actions.shape[1]):
        translation = reference.metadata['joint_types'][j] == 2
        q[1:, j] = rng.uniform(-1, 1, episodes-1)*(.001 if translation else .015)
        v[1:, j] = rng.uniform(-1, 1, episodes-1)*(.001 if translation else .01)
    return q, v


def initialize(collector, offsets):
    import mujoco
    chunk = collector.chunks[0]
    sim = AdroitPhysics(chunk.sim.model, chunk.sim.reference)
    state = collector.state_dict()
    for i, (dq, dv) in enumerate(zip(*offsets)):
        q, v = sim.reset(dq, dv)
        mujoco.mj_getState(sim.model, sim.data, state['states'][i], chunk.spec)
        state['qpos'][i], state['qvel'][i] = q, v
    collector.load_state_dict(state)


def score(q, v, actions, ref, task, native_success):
    n = actions.shape[1]
    if q.shape != (ref.frames, n, task.nq) or v.shape != (ref.frames, n, task.nv):
        raise ValueError('Evaluation must cover every reference frame')
    delta = q[1:]-ref.qpos[1:, None]
    periodic = np.asarray(ref.metadata['joint_periodic'])
    delta[..., periodic] = np.arctan2(np.sin(delta[..., periodic]), np.cos(delta[..., periodic]))
    squared = {name: np.mean(delta[..., ids]**2, axis=-1) for name, ids in task.groups.items()}
    previous = np.concatenate((np.zeros_like(actions[:1]), actions[:-1]))
    reward = sum(np.exp(-s/width**2) for s, width in zip(squared.values(), task.tracking_widths))
    reward -= task.cfg.action_rate_weight*np.mean((actions-previous)**2, axis=-1)
    metrics = {'tracking_return': reward.sum(0)*task.dt,
               'normalized_tracking_rmse': np.sqrt(np.mean(np.stack([s/w**2 for s, w in zip(squared.values(), task.tracking_widths)]), (0, 1))),
               **{name+('_rmse_m' if 'translation' in name or name == 'nail' else '_joint_rmse_rad'):
                  np.sqrt(s.mean(0)) for name, s in squared.items()},
               'native_success_final': native_success[-1], 'native_success_ever': native_success.any(0),
               'native_success_frame_fraction': native_success.mean(0),
               'finite': np.isfinite(q).all((0, 2)) & np.isfinite(v).all((0, 2)) & np.isfinite(actions).all((0, 2))}
    return [{k: values[i].item() for k, values in metrics.items()} for i in range(n)]


@torch.no_grad()
def evaluate(algo, collector, reference, *, prior=False):
    obs = collector.observe()
    q, v, actions = [obs['qpos']], [obs['qvel']], []
    n, device = len(collector.previous_actions), algo.reference_actions.device
    for f in range(reference.frames-1):
        frames = torch.full((n,), f, device=device, dtype=torch.long)
        if prior:
            action = algo.reference_actions[frames].cpu().numpy()  # deployed float32 controls
        else:
            state = tuple(torch.as_tensor(obs[k], device=device, dtype=torch.float32) for k in ('qpos', 'qvel'))
            previous = torch.as_tensor(collector.previous_actions, device=device, dtype=torch.float32)
            action = algo.pi(algo.task.obs(state, previous), algo.task.goal(state, frames),
                             algo.reference_actions[frames]).cpu().numpy()
        batch = collector.step(action)
        q.append(batch['next_qpos'])
        v.append(batch['next_qvel'])
        actions.append(batch['actions'])
        obs = collector.observe()
    return np.asarray(q), np.asarray(v), np.asarray(actions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--prior', action='store_true')
    p.add_argument('--allow-development', action='store_true')
    p.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(argv)
    import mujoco
    torch.set_num_threads(1)
    ref = AdroitReference.load(a.reference)
    if ref.metadata.get('model_sha256') != fingerprint(a.model):
        raise ValueError('Physical reference model hash mismatch')
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False) if a.checkpoint else None
    method = checkpoint['protocol']['method'] if checkpoint else 'supertrack'
    algo, _ = make_algorithm(method, model, ref, a.device, task_cls=AdroitTask)
    if checkpoint:
        for key in ('model', 'reference', 'native_library'):
            if fingerprint(getattr(a, key)) != checkpoint['protocol']['input_hashes'][key]:
                raise ValueError('Input identity mismatch: '+key)
        if checkpoint['protocol'].get('task') != ref.metadata['task']:
            raise ValueError('Checkpoint task mismatch')
        if checkpoint['iteration'] != 20000 and not a.allow_development:
            raise ValueError('Formal evaluation requires complete budget')
        algo.load_state_dict(checkpoint['algo'])
        expected = FullHandSchedule().counts(checkpoint['iteration'], method=method)
        actual = optimizer_counts(algo)
        for key, field in [('wm', 'wm_optimizer_steps'), ('policy', 'policy_updates'), ('critic', 'critic_optimizer_steps')]:
            if actual.get(key, 0) != expected[field]:
                raise ValueError('Incomplete optimizer budget')
    a.output.mkdir(parents=True, exist_ok=False)
    offsets = evaluation_offsets(ref)
    np.savez_compressed(a.output/'starts.npz', qpos=offsets[0], qvel=offsets[1])
    started = time.perf_counter()
    with ParallelNativeDoorCollector(model, ref, 256, a.workers, a.native_library.resolve(), physics_cls=AdroitPhysics) as collector:
        initialize(collector, offsets)
        q, v, actions = evaluate(algo, collector, ref, prior=a.prior)
    # Forward kinematics for scoring saved states only, not a second physics run.
    data = mujoco.MjData(model)
    successes = np.empty((len(actions), 256), dtype=bool)
    for f in range(len(actions)):
        for i in range(256):
            data.qpos[:], data.qvel[:] = q[f+1, i], v[f+1, i]
            mujoco.mj_forward(model, data)
            successes[f, i] = success(model, data, ref.metadata['task'])
    results = score(q, v, actions, ref, algo.task, successes)
    np.savez_compressed(a.output/'physical-rollouts.npz', native_success=successes,
                        **{f'episode{i}_{k}': array[:, i] for i in range(256)
                           for k, array in [('qpos', q), ('qvel', v), ('actions', actions)]})
    report = dict(task=ref.metadata['task'], clip=ref.metadata['clip'], prior=a.prior,
                  method=method if checkpoint else None, seed=checkpoint['protocol']['seed'] if checkpoint else None,
                  iteration=checkpoint['iteration'] if checkpoint else None,
                  checkpoint_sha256=fingerprint(a.checkpoint) if checkpoint else None,
                  input_hashes={k: fingerprint(getattr(a, k)) for k in ('model', 'reference', 'native_library')},
                  starts_sha256=fingerprint(a.output/'starts.npz'), evaluations=results,
                  means={k: float(np.mean([x[k] for x in results])) for k in results[0]},
                  wall_seconds=time.perf_counter()-started,
                  scope='single clip, 256 fixed hand-perturbed starts; no held-out-motion or seed-variance claim')
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['means']))


if __name__ == '__main__':
    main()
