"""Full-budget native Door adaptation of the Hand accurate configurations.

One iteration collects one step in 2048 persistent physical worlds, fits four
model batches and (after 300 iterations) makes one policy update. Development
prefixes are explicit and never identified as a finished full-budget run.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algos.common import lr_at, set_lr
from .door_schedule import FullHandSchedule
from .motion.adroit_door import DoorReference
from .motion.door_ring import DoorReplayRing
from .sim.rigid.door_collector import ParallelNativeDoorCollector
from .sim.rigid.adroit_door import DoorPhysics
from .tasks.adroit_door import DoorTask
from .train_door import DOOR_METHODS
from .train_door_online import configure_fused_adam, fingerprint, optimizer_counts, updates
from .worldmodels.mjcf import MjcfLayout


# Opt-in tuning settings (train_door_full flags); a run records the ones it changes in protocol['options'], which
# eval_door_full passes back here. Absent: the original configuration's values.
TUNABLE = dict(lr=None, residual_scale=.1, correction_reg=.01, correction_imag_noise=.2, gamma=.99)
SUPERTRACK_ONLY, SUPERMIMIC_ONLY = ('correction_reg', 'correction_imag_noise'), ('gamma',)


def make_algorithm(method, model, reference, device, options=None, *, task_cls=DoorTask):
    options = dict(options or {})
    if set(options) - set(TUNABLE):
        raise ValueError(f'Unknown Door tuning settings: {sorted(set(options) - set(TUNABLE))}')
    if set(options) & set(SUPERTRACK_ONLY if method == 'supermimic' else SUPERMIMIC_ONLY):
        raise ValueError(f'{method}: settings of the other method: {sorted(options)}')
    task = task_cls(reference, model.actuator_ctrlrange, device=device, goal_frames=5)
    rate = options.get('lr') or (4e-4 if method == 'supertrack' else 2e-4)
    settings = (dict(task_weights=True, correction_reg=options.get('correction_reg', .01),
                     correction_imag_noise=options.get('correction_imag_noise', .2))
                if method == 'supertrack' else dict(critic_epochs=5, critic_minibatches=4))
    algo = DOOR_METHODS[method](task, replace(MjcfLayout.from_model(model), dt=task.dt),
                               torch.tensor(reference.actions, dtype=torch.float32).clamp(-1, 1),
                               device=device, hidden=(1024, 512, 256), wm_hidden=(1024, 1024, 1024),
                               horizon=16, lr=rate, upstream_policy_optimizer=True, **settings)
    if 'residual_scale' in options:  # the policy's bound on its correction (DoorResidualPolicy, not in its state)
        algo.pi.residual_scale = float(options['residual_scale'])
    if 'gamma' in options:
        algo.gamma = float(options['gamma'])
    configure_fused_adam(algo, device.startswith('cuda'))
    return algo, rate


def door_latch_restarts(collector, offset, seed):
    """Opt-in training starts with the door and latch raised by U(0, offset) each (--door-latch-offset): after every
    step, the worlds it reset restart from such a start (their own random stream). None for offset 0: the original
    collection, whose resets start at the reference's first state."""
    if not offset:
        return None
    rng = np.random.default_rng([seed, 20261116])

    def restart(done):
        indices = np.flatnonzero(done)
        offsets = np.zeros((len(indices), 30))
        offsets[:, 28:30] = offset * rng.uniform(0., 1., (len(indices), 2))
        collector.restart_worlds(indices, offsets)
    restart.rng = rng
    return restart


@torch.no_grad()
def collect(algo, collector, ring, restart=None, action_noise=.1):
    observation = collector.observe()
    device = algo.reference_actions.device
    q, v = (torch.as_tensor(observation[key], dtype=torch.float32, device=device)
            for key in ('qpos', 'qvel'))
    frames = torch.as_tensor(observation['frames'], dtype=torch.long, device=device)
    previous = torch.as_tensor(collector.previous_actions, dtype=torch.float32, device=device)
    obs, goal = algo.task.obs((q, v), previous), algo.task.goal((q, v), frames)
    algo.pi.norm.update(torch.cat((obs, goal), -1))
    actions = algo.pi(obs, goal, algo.reference_actions[frames])
    actions = (actions + action_noise * torch.randn_like(actions)).clamp(-1, 1)
    # One device-to-host transfer for all worlds, not one per world/substep.
    batch = collector.step(actions.cpu().numpy())
    ring.append(batch)
    nq, nv = (torch.as_tensor(batch[key], dtype=torch.float32, device=device)
              for key in ('next_qpos', 'next_qvel'))
    algo.wm.update_norms(q, v, algo.setpoint(actions), nq, nv)
    if restart is not None:
        restart(batch['done'])


def check_counts(algo, collector, schedule, iteration, method):
    expected = schedule.counts(iteration, method=method)
    got = optimizer_counts(algo)
    actual = dict(training_transitions=collector.transitions - expected['warmup_transitions'],
                  warmup_transitions=expected['warmup_transitions'], wm_optimizer_steps=got['wm'],
                  policy_updates=got['policy'], critic_optimizer_steps=got.get('critic', 0))
    schedule.assert_counts(actual, iteration, method=method)
    return actual


def save(path, algo, collector, ring, protocol, iteration, timing, extra=None):
    """Atomic snapshot at an iteration boundary, including physical and replay state."""
    checkpoint = dict(format=1, protocol=protocol, iteration=iteration, algo=algo.state_dict(),
                      collector=collector.state_dict(), replay=ring.state_dict(),
                      torch_rng=torch.get_rng_state(),
                      cuda_rng=(torch.cuda.get_rng_state(algo.reference_actions.device)
                                if algo.reference_actions.is_cuda else None), timing=timing)
    if extra:
        checkpoint.update(extra)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore(checkpoint, algo, collector, ring, protocol):
    if checkpoint['format'] != 1 or checkpoint['protocol'] != protocol:
        raise ValueError('Full Door checkpoint protocol mismatch')
    algo.load_state_dict(checkpoint['algo'])
    collector.load_state_dict(checkpoint['collector'])
    ring.load_state_dict(checkpoint['replay'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if algo.reference_actions.is_cuda:
        torch.cuda.set_rng_state(checkpoint['cuda_rng'], algo.reference_actions.device)
    return checkpoint['iteration'], checkpoint['timing']


def main(argv=None, *, reference_cls=DoorReference, task_cls=DoorTask, physics_cls=DoorPhysics, extra_help=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=extra_help)
    for name in ('model', 'reference', 'native-library', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--method', choices=tuple(k for k in DOOR_METHODS if k != 'residual'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    p.add_argument('--resume', type=Path, help='Trusted local checkpoint; new output directory required')
    p.add_argument('--development-steps', type=int, help='Explicit incomplete prefix; does not alter full LR schedule')
    p.add_argument('--door-latch-offset', type=float, default=0.,
                   help='Training episodes start with the door and latch raised by U(0, d) rad each (default 0: the original task)')
    p.add_argument('--snapshot-every', type=int, default=0,
                   help='Also save the algorithm alone every N iterations (snapshot<N>.pt) for live evaluation; 0: none')
    p.add_argument('--lr', type=float, help='Learning rate of every optimizer (default 4e-4 SuperTrack, 2e-4 SuperMimic)')
    p.add_argument('--residual-scale', type=float, help="Bound of the policy's correction about the demonstration's control (default 0.1)")
    p.add_argument('--action-noise', type=float, default=.1, help='Collection action noise (default 0.1)')
    p.add_argument('--wm-only-iters', type=int, default=300, help='Iterations with world-model updates only (default 300)')
    p.add_argument('--correction-reg', type=float, help='SuperTrack: weight of the correction regularizer (default 0.01)')
    p.add_argument('--correction-imag-noise', type=float, help='SuperTrack: imagined correction noise (default 0.2)')
    p.add_argument('--gamma', type=float, help='SuperMimic: discount (default 0.99)')
    a = p.parse_args(argv)
    options = {k: getattr(a, k) for k in ('lr', 'residual_scale', 'correction_reg', 'correction_imag_noise', 'gamma')
               if getattr(a, k) is not None}
    if any(not np.isfinite(v) or v < 0 or (k in ('lr', 'residual_scale', 'gamma') and v <= 0) for k, v in options.items()) \
            or not np.isfinite(a.action_noise) or a.action_noise < 0 or a.wm_only_iters < 0 or options.get('gamma', .5) >= 1:
        p.error('Tuning settings must be finite: lr, residual scale > 0, 0 < gamma < 1, the others >= 0')
    if reference_cls is not DoorReference and (options or a.action_noise != .1 or a.wm_only_iters != 300 or a.door_latch_offset):
        p.error('Door-only tuning/start options are not supported by the fixed Adroit suite protocol')
    schedule = FullHandSchedule(wm_only_iters=a.wm_only_iters)
    end = a.development_steps if a.development_steps is not None else schedule.iterations
    if not 1 <= end <= schedule.iterations or not 1 <= a.workers <= schedule.worlds:
        p.error('Invalid development prefix or worker count')
    if not np.isfinite(a.door_latch_offset) or a.door_latch_offset < 0 or a.snapshot_every < 0:
        p.error('--door-latch-offset must be finite and nonnegative, --snapshot-every nonnegative')
    import mujoco
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    reference = reference_cls.load(a.reference)
    if reference_cls is not DoorReference and reference.metadata.get('model_sha256') != fingerprint(a.model):
        raise ValueError('Physical reference model hash mismatch')
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    if a.door_latch_offset and a.door_latch_offset >= model.jnt_range[28:30, 1].min():
        p.error('--door-latch-offset must stay inside the door and latch ranges')
    started = time.perf_counter()
    try:
        algo, rate = make_algorithm(a.method, model, reference, a.device, options, task_cls=task_cls)
    except ValueError as error:
        p.error(str(error))
    ring = DoorReplayRing(256, schedule.worlds, a.device, nq=model.nq, nv=model.nv, nu=model.nu)
    protocol = dict(format=1, method=a.method, seed=a.seed, schedule=asdict(schedule),
                    workers=a.workers, device=a.device, goal_frames=5, horizon=16,
                    wm_batch=2048, wm_window=8, pi_batch=4096, lr=rate, action_noise=a.action_noise,
                    lr_decay_iterations=20000, source_config='hand-'+a.method+'-accurate',
                    upstream_commit='69b818e', torch=str(torch.__version__), mujoco=mujoco.__version__,
                    adaptation='native articulated state and action units; demonstration-control residual; finite-clip terminal; native CPU physics',
                    unavailable='XHand FK, error-based soft termination/failure penalty, CUDA physics/collection graphs',
                    input_hashes={key: fingerprint(getattr(a, key)) for key in ('model', 'reference', 'native_library')})
    if reference_cls is not DoorReference:
        protocol['task'] = reference.metadata['task']
        protocol['task_reward'] = asdict(algo.task.cfg)
        protocol['reference_schema'] = reference.metadata['kind']
    if a.door_latch_offset:
        protocol['door_latch_offset'] = a.door_latch_offset
    if options:
        protocol['options'] = options
    source = Path(__file__).parent
    protocol['source_sha256'] = hashlib.sha256(b''.join(
        str(f.relative_to(source)).encode()+f.read_bytes() for f in sorted(source.rglob('*.py')))).hexdigest()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    sync = torch.cuda.synchronize if a.device.startswith('cuda') else lambda: None
    with ParallelNativeDoorCollector(model, reference, schedule.worlds, a.workers,
                                     a.native_library.resolve(), physics_cls=physics_cls) as collector:
        iteration = 0
        timing = dict(collection_seconds=0., update_seconds=0., checkpoint_seconds=0.)
        restart = door_latch_restarts(collector, a.door_latch_offset, a.seed)
        if a.resume:
            checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
            iteration, timing = restore(checkpoint, algo, collector, ring, protocol)
            if restart is not None:
                restart.rng.bit_generator.state = checkpoint['door_latch_rng']
            check_counts(algo, collector, schedule, iteration, a.method)
            if not iteration < end:
                raise ValueError('Resume must advance at least one iteration')
        else:
            collector.stagger_from_physical_replay(a.seed)
            for _ in range(schedule.warmup_steps):
                collect(algo, collector, ring, restart, a.action_noise)
        sync()
        startup = time.perf_counter() - started
        with (a.output/'iterations.jsonl').open('w') as log:
            for iteration in range(iteration + 1, end + 1):
                current_lr = lr_at(iteration, rate, schedule.iterations)
                for optimizer in (algo.opt_wm, algo.opt_pi, *([algo.opt_v] if hasattr(algo, 'opt_v') else [])):
                    set_lr(optimizer, current_lr)
                tick = time.perf_counter()
                collect(algo, collector, ring, restart, a.action_noise)
                sync()
                collected = time.perf_counter() - tick
                tick = time.perf_counter()
                updates(algo, ring, 4096, 16, schedule.wm_batches, int(schedule.policy_due(iteration)),
                        wm_batch=2048, wm_window=8)
                sync()
                updated = time.perf_counter() - tick
                timing['collection_seconds'] += collected
                timing['update_seconds'] += updated
                log.write(json.dumps(dict(iteration=iteration, collection_seconds=collected,
                                          update_seconds=updated, lr=current_lr))+'\n')
                log.flush()
                if iteration % 100 == 0 or iteration == end:
                    counts = check_counts(algo, collector, schedule, iteration, a.method)
                    print(json.dumps(dict(iteration=iteration, counts=counts)), flush=True)
                if iteration % 1000 == 0 or iteration == end:
                    tick = time.perf_counter()
                    save(a.output/'latest.pt', algo, collector, ring, protocol, iteration, timing,
                         None if restart is None else dict(door_latch_rng=restart.rng.bit_generator.state))
                    timing['checkpoint_seconds'] += time.perf_counter() - tick
                if a.snapshot_every and iteration % a.snapshot_every == 0:
                    tick = time.perf_counter()
                    snapshot = a.output/f'snapshot{iteration}.pt'
                    torch.save(dict(format=1, protocol=protocol, iteration=iteration, algo=algo.state_dict(), timing=dict(timing),
                                    seconds=time.perf_counter()-started), snapshot.with_suffix('.tmp'))
                    snapshot.with_suffix('.tmp').replace(snapshot)
                    timing['snapshot_seconds'] = timing.get('snapshot_seconds', 0.) + time.perf_counter() - tick
        result = dict(status='trained_full_budget' if iteration == schedule.iterations else 'development_prefix',
                      counts=check_counts(algo, collector, schedule, iteration, a.method),
                      timing=timing, invocation_wall_seconds=time.perf_counter()-started,
                      startup_seconds=startup, evaluation='pending separate frozen evaluation', protocol=protocol)
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
