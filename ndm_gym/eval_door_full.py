"""Frozen physical evaluation for full-budget Door checkpoints and their prior.

256 starts: one nominal and 255 seeded multivariate hand/root perturbations.
The passive door/latch initial state is unchanged. No fitting, action noise,
intermediate-checkpoint selection or state assignment after initialization.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .door_schedule import FullHandSchedule
from .motion.adroit_door import DoorReference, JOINT_NAMES
from .sim.rigid.adroit_door import DoorPhysics
from .sim.rigid.door_collector import ParallelNativeDoorCollector
from .train_door_full import make_algorithm
from .train_door_online import fingerprint, optimizer_counts


def evaluation_offsets(seed=20261115):
    rng = np.random.default_rng(seed)
    q, v = np.zeros((256, 30)), np.zeros((256, 30))
    q[1:, 0] = rng.uniform(-.001, .001, 255)
    q[1:, 1:28] = rng.uniform(-.015, .015, (255, 27))
    v[1:, 0] = rng.uniform(-.001, .001, 255)
    v[1:, 1:28] = rng.uniform(-.01, .01, (255, 27))
    return q, v


def evaluation_starts(door_latch_offset=0.):
    """The 256 starts of ``evaluation_offsets``; with ``door_latch_offset`` d > 0 the door and latch angles of starts
    1-255 are also raised by U(0, d) each (seed 20261116), inside their ranges at the closed start. d = 0: unchanged."""
    q, v = evaluation_offsets()
    if door_latch_offset:
        q[1:, 28:30] = door_latch_offset * np.random.default_rng(20261116).uniform(0., 1., (256, 2))[1:]
    return q, v


def initialize_evaluation(collector, offsets, velocities):
    import mujoco
    chunk = collector.chunks[0]
    sim = DoorPhysics(chunk.sim.model, chunk.sim.reference)
    state = collector.state_dict()
    for index, (dq, dv) in enumerate(zip(offsets, velocities)):
        q, v = sim.reset(qpos_offset=dq, qvel_offset=dv)
        mujoco.mj_getState(sim.model, sim.data, state['states'][index], chunk.spec)
        state['qpos'][index], state['qvel'][index] = q, v
    collector.load_state_dict(state)


def score(qpos, qvel, actions, reference, cfg):
    """Independent NumPy scoring; all full-episode states and all starts retained."""
    if qpos.shape != (reference.frames, 256, 30) or qvel.shape != qpos.shape:
        raise ValueError('Expected all 256 complete physical trajectories')
    if actions.shape != (reference.frames - 1, 256, 28):
        raise ValueError('Expected all physical hand controls')
    d = qpos[1:] - reference.qpos[1:, None]
    previous = np.concatenate((np.zeros_like(actions[:1]), actions[:-1]))
    reward = (np.exp(-d[..., 0]**2/cfg.translation_std**2)
              + np.exp(-np.mean(d[..., 1:28]**2, -1)/cfg.hand_std**2)
              + np.exp(-d[..., 28]**2/cfg.door_std**2)
              + np.exp(-d[..., 29]**2/cfg.latch_std**2)
              - cfg.action_rate_weight*np.mean((actions-previous)**2, -1))*reference.metadata['control_dt']
    metrics = dict(tracking_return=reward.sum(0),
                   translation_rmse_m=np.sqrt(np.mean(d[..., 0]**2, 0)),
                   hand_rmse_rad=np.sqrt(np.mean(d[..., 1:28]**2, (0, 2))),
                   door_rmse_rad=np.sqrt(np.mean(d[..., 28]**2, 0)),
                   latch_rmse_rad=np.sqrt(np.mean(d[..., 29]**2, 0)),
                   final_door_rad=qpos[-1, :, 28],
                   opened_final=qpos[-1, :, 28] >= cfg.opening_rad,
                   opened_ever=qpos[..., 28].max(0) >= cfg.opening_rad,
                   finite=np.isfinite(qpos).all((0, 2)) & np.isfinite(qvel).all((0, 2))
                   & np.isfinite(actions).all((0, 2)))
    return [{key: value[i].item() for key, value in metrics.items()} for i in range(256)]


@torch.no_grad()
def evaluate(algo, collector, reference, *, prior=False):
    obs = collector.observe()
    qpos, qvel, actions = [obs['qpos']], [obs['qvel']], []
    device = algo.reference_actions.device
    for frame in range(reference.frames - 1):
        if prior:
            action = np.repeat(reference.actions[frame:frame+1].clip(-1, 1), 256, axis=0)
        else:
            state = tuple(torch.as_tensor(obs[key], dtype=torch.float32, device=device)
                          for key in ('qpos', 'qvel'))
            previous = torch.as_tensor(collector.previous_actions, dtype=torch.float32, device=device)
            frames = torch.full((256,), frame, dtype=torch.long, device=device)
            action = algo.pi(algo.task.obs(state, previous), algo.task.goal(state, frames),
                             algo.reference_actions[frames]).cpu().numpy()
        batch = collector.step(action)
        qpos.append(batch['next_qpos'])
        qvel.append(batch['next_qvel'])
        actions.append(batch['actions'])
        obs = collector.observe()
    return np.asarray(qpos), np.asarray(qvel), np.asarray(actions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Trusted full-run checkpoint')
    source.add_argument('--prior', action='store_true')
    p.add_argument('--allow-development', action='store_true', help='Explicitly evaluate an incomplete prefix')
    p.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--door-latch-offset', type=float, default=0.,
                   help='Also raise the door and latch angles of the perturbed starts by U(0, d) rad (default 0: the original starts)')
    a = p.parse_args(argv)
    if not np.isfinite(a.door_latch_offset) or a.door_latch_offset < 0:
        p.error('--door-latch-offset must be finite and nonnegative')
    import mujoco
    torch.set_num_threads(1)
    ref = DoorReference.load(a.reference)
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False) if a.checkpoint else None
    method = ck['protocol']['method'] if ck else 'supertrack'
    algo, _ = make_algorithm(method, model, ref, a.device, ck['protocol'].get('options') if ck else None)
    if ck:
        for key in ('model', 'reference', 'native_library'):
            if fingerprint(getattr(a, key)) != ck['protocol']['input_hashes'][key]:
                raise ValueError('Checkpoint input mismatch: '+key)
        if ck['iteration'] != 20000 and not a.allow_development:
            raise ValueError('Final study requires iteration 20000')
        algo.load_state_dict(ck['algo'])
        counts = FullHandSchedule(**ck['protocol']['schedule']).counts(ck['iteration'], method=method)
        actual = optimizer_counts(algo)
        assert actual['wm'] == counts['wm_optimizer_steps']
        assert actual['policy'] == counts['policy_updates']
        assert actual.get('critic', 0) == counts['critic_optimizer_steps']
    a.output.mkdir(parents=True, exist_ok=False)
    qoffset, voffset = evaluation_starts(a.door_latch_offset)
    np.savez_compressed(a.output/'starts.npz', qpos=qoffset, qvel=voffset)
    started = time.perf_counter()
    with ParallelNativeDoorCollector(model, ref, 256, a.workers, a.native_library.resolve()) as collector:
        initialize_evaluation(collector, qoffset, voffset)
        q, v, actions = evaluate(algo, collector, ref, prior=a.prior)
    results = score(q, v, actions, ref, algo.task.cfg)
    rollouts = {f'episode{i}_{key}': value[:, i] for i in range(256)
                for key, value in [('qpos', q), ('qvel', v), ('actions', actions)]}
    np.savez_compressed(a.output/'physical-rollouts.npz', **rollouts)
    report = dict(prior=a.prior, method=method if ck else None,
                  seed=ck['protocol']['seed'] if ck else None,
                  iteration=ck['iteration'] if ck else None,
                  full_budget=bool(ck and ck['iteration'] == 20000),
                  checkpoint_sha256=fingerprint(a.checkpoint) if ck else None,
                  starts_sha256=fingerprint(a.output/'starts.npz'),
                  source_joint_order=JOINT_NAMES, evaluations=results,
                  means={key: float(np.mean([r[key] for r in results])) for key in results[0]},
                  wall_seconds=time.perf_counter()-started,
                  evaluation_scope='one nominal and 255 multivariate perturbed hand starts on one clip; no held-out motions')
    if a.door_latch_offset:
        report.update(door_latch_offset=a.door_latch_offset,
                      evaluation_scope='one nominal and 255 perturbed hand, door and latch starts on one clip; no held-out motions')
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['means']))


if __name__ == '__main__':
    main()
