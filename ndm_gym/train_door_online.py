"""Alternating native Door training with explicit inputs and round-boundary recovery.

Eager fp32 on an explicitly selected CPU or CUDA device. Bootstrap is an explicitly supplied physical episode file;
its generation cost is additional to this command's measured training time.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .motion.adroit_door import DoorReference
from .motion.door_replay import BalancedDoorReplay, DoorReplay, collect_policy_replay, collect_policy_replay_batched
from .tasks.adroit_door import DoorTask
from .train_door import DOOR_METHODS, evaluate
from .worldmodels.mjcf import MjcfLayout


def fingerprint(path):
    """Content identity, independent of input path."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(algo, replay):
    """Fit canonical CPU statistics on training episodes, then copy buffers to the device.

    Tiny device-dependent statistics can change L1 gradient signs. Only the three
    normalizers are copied back; learned parameters and optimizer state stay intact.
    """
    q, v, a = (torch.cat([e[i][:-1] if i < 2 else e[i] for e in replay.episodes]) for i in range(3))
    nq, nv = (torch.cat([e[i][1:] for e in replay.episodes]) for i in range(2))
    canonical = algo.wm if not algo.reference_actions.is_cuda else copy.deepcopy(algo.wm).cpu()
    ctrl = algo.task.ctrl_mid.cpu() + a.clamp(-1, 1) * algo.task.ctrl_half.cpu()
    canonical.update_norms(q, v, ctrl, nq, nv)
    if canonical is not algo.wm:
        for name in ('in_norm', 'out_norm', 'err_norm'):
            getattr(algo.wm, name).load_state_dict(getattr(canonical, name).state_dict())


def updates(algo, replay, batch, horizon, wm, policy):
    """Run real optimizer updates; reject nonfinite model loss/gradients."""
    for _ in range(wm):
        q, v, a, _, _ = (x.to(algo.reference_actions.device) for x in replay.sample(batch, horizon))
        loss = getattr(algo, "training_wm_loss", algo.wm_loss)(q, v, algo.task.setpoint(a))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite model loss")
        algo.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(algo.wm.parameters(), 1., error_if_nonfinite=True)
        algo.opt_wm.step()
    for _ in range(policy):
        q, v, _, frame, previous = (x.to(algo.reference_actions.device) for x in replay.sample(batch, horizon))
        algo.policy_update((q[0], v[0]), frame, previous)


def configure_wm_compile(algo, enabled=False):
    """Compile the differentiable WM loss; retain eager default and optimizer semantics.

    Compilation is lazy, so first-update cost belongs in end-to-end timings.
    This is independent of upstream's collection-FK compile_eager option.
    """
    if enabled:
        algo.training_wm_loss = torch.compile(algo.wm_loss, mode="default")


def configure_fused_adam(algo, enabled=False):
    """Opt into upstream's fused Adam setting before updates or recovery.

    All optimizers keep their original hyperparameters. CUDA rounding can change
    results; this does not enable graph capture or alter the learning objective.
    """
    if not enabled:
        return
    if not algo.reference_actions.is_cuda:
        raise ValueError("Fused Adam requires a CUDA device")
    optimizers = [algo.opt_wm, algo.opt_pi]
    if hasattr(algo, 'opt_v'):
        optimizers.append(algo.opt_v)
    if any(opt.state for opt in optimizers):
        raise ValueError("Configure fused Adam before updates or recovery")
    for opt in optimizers:
        opt.defaults.update(fused=True, foreach=False)
        for group in opt.param_groups:
            group.update(fused=True, foreach=False)


def optimizer_counts(algo):
    """Read actual Adam steps, rejecting partially updated parameter groups.

    Counts come from optimizer state rather than requested loop budgets. Every
    parameter of these dense models must participate in every optimizer update.
    """
    counts = {}
    for name, attr in (('wm', 'opt_wm'), ('policy', 'opt_pi'), ('critic', 'opt_v')):
        if not hasattr(algo, attr):
            continue
        opt = getattr(algo, attr)
        steps = {int(opt.state.get(p, {}).get('step', 0))
                 for group in opt.param_groups for p in group['params']}
        if len(steps) != 1:
            raise ValueError(f'Inconsistent {name} optimizer step counts: {steps}')
        counts[name] = steps.pop()
    return counts


def verify_budget(algo, bootstrap_updates, wm_updates, policy_updates, completed_rounds):
    """Reject a checkpoint/result whose actual optimizer work differs from its budget."""
    expected = dict(wm=bootstrap_updates + completed_rounds * wm_updates,
                    policy=completed_rounds * policy_updates)
    if hasattr(algo, 'opt_v'):
        expected['critic'] = expected['policy'] * algo.critic_epochs * algo.critic_minibatches
    actual = optimizer_counts(algo)
    if actual != expected:
        raise ValueError(f'Optimizer budget mismatch: actual {actual}, expected {expected}')
    return actual


def restore(algo, checkpoint, protocol):
    """Restore optimizer/model/RNG state only for the exact declared protocol."""
    # This newly added opt-in field was absent in historical default runs.
    saved_protocol = {'task_weights': False, 'upstream_policy_optimizer': False, 'correction_reg': 0., 'correction_imag_noise': 0., **checkpoint['protocol']}
    requested_protocol = {'task_weights': False, 'upstream_policy_optimizer': False, 'correction_reg': 0., 'correction_imag_noise': 0., **protocol}
    if saved_protocol != requested_protocol:
        raise ValueError("Recovery protocol/input identity mismatch")
    algo.load_state_dict(checkpoint['algo'])
    if 'optimizer_counts' in checkpoint and optimizer_counts(algo) != checkpoint['optimizer_counts']:
        raise ValueError('Checkpoint optimizer count mismatch')
    torch.set_rng_state(checkpoint['rng'])
    if algo.reference_actions.is_cuda:
        torch.cuda.set_rng_state(checkpoint['cuda_rng'], algo.reference_actions.device)
    return checkpoint['completed_rounds']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'bootstrap', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--method', choices=('supertrack', 'supermimic'), required=True)
    p.add_argument('--resume', type=Path, help='Round-boundary checkpoint; output must be a new directory')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda:0'),
                   help='CUDA index is relative to CUDA_VISIBLE_DEVICES; CPU-fitted statistics, '
                        'but device rounding can change training results; no speedup guaranteed')
    p.add_argument('--fused-adam', action='store_true',
                   help='CUDA fused Adam for every optimizer, as upstream --fused_adam; '
                        'changes results by rounding; no speedup guaranteed')
    p.add_argument('--upstream-policy-optimizer', action='store_true',
                   help='Policy Adam betas (.9,.95) and gradient value cap .1 after norm cap 1; '
                        'changes learning; default preserves historical Door settings')
    p.add_argument('--task-weights', action='store_true',
                   help='SuperTrack reward-width L1 weights, following upstream st_task_weights')
    p.add_argument('--correction-reg', type=float, default=0.,
                   help='SuperTrack L1+L2 penalty on dimensionless correction before residual scale')
    p.add_argument('--correction-imag-noise', type=float, default=0.,
                   help='SuperTrack imagined Gaussian correction std before residual scale; physical rollout unchanged')
    p.add_argument('--compile-wm', action='store_true',
                   help='Compile WM loss and backward with TorchInductor; includes first-update compilation cost; '
                        'rounding can change results; no speedup guaranteed')
    p.add_argument('--collection-worlds', type=int, default=1,
                   help='Bounded native CPU simulators with batched policy inference; physics stays serial; '
                        'batch rounding can change results; no speedup guaranteed')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--bootstrap-updates', type=int, default=1000)
    p.add_argument('--wm-updates', type=int, default=500)
    p.add_argument('--policy-updates', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--policy-lr', type=float, default=1e-5)
    p.add_argument('--train-offsets', type=float, nargs='+', default=[-.01, -.005, .005, .01])
    p.add_argument('--offsets', type=float, nargs='+', default=[0., -.015, -.0075, .0075, .015])
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (min(args.rounds, args.batch_size, args.horizon) < 1 or args.batch_size % 2
            or min(args.bootstrap_updates, args.wm_updates, args.policy_updates) < 0
            or len(args.train_offsets) < 2 or not args.offsets
            or not np.isfinite(args.train_offsets + args.offsets + [args.policy_lr]).all()
            or args.policy_lr <= 0):
        raise ValueError('Invalid online budget/offsets')
    import mujoco
    from .sim.rigid.adroit_door import DoorPhysics

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    reference = DoorReference.load(args.reference)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if args.collection_worlds < 1:
        raise ValueError('Positive collection worlds required')
    sim = DoorPhysics(model, reference)
    task = DoorTask(reference, model.actuator_ctrlrange, device=args.device)
    algo = DOOR_METHODS[args.method](task, replace(MjcfLayout.from_model(model), dt=task.dt),
                                   torch.tensor(reference.actions, dtype=torch.float32).clamp(-1, 1),
                                   hidden=(32, 32), horizon=args.horizon, device=args.device,
                                   upstream_policy_optimizer=args.upstream_policy_optimizer,
                                   task_weights=args.task_weights,
                                   correction_reg=args.correction_reg,
                                   correction_imag_noise=args.correction_imag_noise)
    configure_fused_adam(algo, args.fused_adam)
    configure_wm_compile(algo, args.compile_wm)
    bootstrap = DoorReplay.load(args.bootstrap)
    bootstrap, _ = bootstrap.split([len(bootstrap.episodes)-1])
    if any(len(e[2]) != len(reference.actions) for e in bootstrap.episodes):
        raise ValueError('Bootstrap horizons must match reference')
    protocol = {k: v for k, v in vars(args).items() if k not in ('model', 'reference', 'bootstrap', 'output', 'resume')}
    protocol.update(format=6, policy_hidden=[32, 32], wm_hidden=[64, 64],
                    critic_epochs=getattr(algo, 'critic_epochs', 0),
                    critic_minibatches=getattr(algo, 'critic_minibatches', 0), normalization='cpu training episodes', dtype='float32', path=('compiled-wm' if args.compile_wm else 'eager'), threads=1,
                    selection='final only', bootstrap_holdout='last episode',
                    online_holdout='last episode each round', supervised_updates=0,
                    torch=str(torch.__version__), mujoco=mujoco.__version__)
    protocol.update({name + '_sha256': fingerprint(getattr(args, name))
                     for name in ('model', 'reference', 'bootstrap')})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for group in algo.opt_pi.param_groups:
        group['lr'] = args.policy_lr
    first = 0
    if args.resume:
        first = restore(algo, torch.load(args.resume, map_location='cpu', weights_only=False), protocol)
        if not 0 <= first <= args.rounds:
            raise ValueError('Invalid completed round count')
    else:
        normalize(algo, bootstrap)
        updates(algo, bootstrap, args.batch_size, args.horizon, args.bootstrap_updates, 0)
        torch.manual_seed(args.seed + 100000)
    for iteration in range(first, args.rounds):
        if args.collection_worlds == 1:
            collected = collect_policy_replay(algo, sim, args.train_offsets)
        else:
            collected = collect_policy_replay_batched(
                algo, lambda: DoorPhysics(model, reference), args.train_offsets, args.collection_worlds)
        recent, heldout = collected.split([len(collected.episodes)-1])
        replay = BalancedDoorReplay([bootstrap, recent])
        saved = {f'{split}{i}_{key}': value.numpy() for split, data in [('train', recent), ('heldout', heldout)]
                 for i, ep in enumerate(data.episodes) for key, value in zip(('qpos', 'qvel', 'actions'), ep)}
        np.savez_compressed(args.output / f'replay{iteration}.npz', **saved)
        normalize(algo, replay)
        updates(algo, replay, args.batch_size, args.horizon, args.wm_updates, args.policy_updates)
        counts = verify_budget(algo, args.bootstrap_updates, args.wm_updates,
                               args.policy_updates, iteration + 1)
        checkpoint = dict(optimizer_counts=counts, algo=algo.state_dict(), rng=torch.get_rng_state(), protocol=protocol,
                          cuda_rng=(torch.cuda.get_rng_state(algo.reference_actions.device)
                                    if algo.reference_actions.is_cuda else None),
                          completed_rounds=iteration+1,
                          wm_updates=args.bootstrap_updates+(iteration+1)*args.wm_updates,
                          policy_updates=(iteration+1)*args.policy_updates, supervised_updates=0)
        temporary = args.output / 'checkpoint.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / 'latest.pt')
        print(json.dumps({'completed_rounds': iteration+1}), flush=True)
    counts = verify_budget(algo, args.bootstrap_updates, args.wm_updates, args.policy_updates, args.rounds)
    metrics, rollouts = evaluate(algo, sim, reference, args.offsets)
    np.savez_compressed(args.output / 'physical-rollouts.npz', **rollouts)
    result = dict(protocol=protocol, optimizer_counts=counts, evaluations=metrics, process_seconds=time.perf_counter()-started,
                  resumed_from=str(args.resume) if args.resume else None,
                  shared_bootstrap_generation_cost='excluded; report separately', three_seed_evidence=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
