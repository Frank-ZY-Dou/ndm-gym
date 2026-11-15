"""Bounded Door residual policy update through the upstream MJCF world model.

This eager correctness path is opt-in. A fitted model and physical evaluation
are required before interpreting its imagined reward as control improvement.
"""

import torch


def residual_update(policy, optimizer, world_model, task, state, frame,
                    previous_action, reference_actions, horizon=4):
    """Maximize Door reward over a nonterminal, time-aligned imagined window.

    Uses ``MjcfWorldModel.step(qpos, qvel, physical_ctrl, dt)`` unchanged.
    Frozen world-model weights retain action derivatives, as in SuperTrack.
    Caller owns sampling, normalization, data collection and training budgets.
    Nonfinite updates leave policy parameters and optimizer state unchanged.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be positive")
    if frame.ndim != 1 or torch.any(frame < 0) or torch.any(frame + horizon >= task.frames):
        raise ValueError("Imagined window must remain within the reference")
    if reference_actions.shape != (task.frames - 1, task.action_dim):
        raise ValueError("Expected one reference action per transition")
    if not torch.isfinite(reference_actions).all() or torch.any(reference_actions.abs() > 1):
        raise ValueError("Reference controls must be finite normalized actions")
    state = tuple(x.detach() for x in state)
    previous_action = previous_action.detach()
    flags = [p.requires_grad for p in world_model.parameters()]
    try:
        world_model.requires_grad_(False)
        rewards = []
        for _ in range(horizon):
            action = policy(task.obs(state, previous_action), task.goal(state, frame), reference_actions[frame])
            state = world_model.step(*state, task.setpoint(action), task.dt)
            frame = frame + 1
            rewards.append(task.reward(state, frame, action, previous_action))
            previous_action = action
        loss = -torch.stack(rewards).sum(0).mean()
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Door imagined loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return loss.detach(), grad.detach()
    finally:
        for parameter, flag in zip(world_model.parameters(), flags):
            parameter.requires_grad_(flag)
