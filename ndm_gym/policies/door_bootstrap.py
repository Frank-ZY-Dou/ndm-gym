"""Action-imitation bootstrap for Door; fitting loss is not physical tracking evidence.

Uses replay controls only as supervised targets, never as simulator feed-forward.
The initial previous control is zero, matching a newly reset deployed policy.
"""

import torch


def bootstrap_batch(task, reference, goal_frames=1):
    """One correctly aligned row per physical transition, excluding terminal state."""
    actions = torch.as_tensor(reference.actions, dtype=task.q.dtype, device=task.q.device)
    if torch.any(actions.abs() > 1):
        raise ValueError("Bootstrap targets must be normalized controls in [-1, 1]")
    previous = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]))
    state = task.q[:-1], task.v[:-1]
    frame = torch.arange(len(actions), device=task.q.device)
    return task.obs(state, previous), task.goal(state, frame, goal_frames), actions


def bootstrap_update(policy, optimizer, batch):
    """One eager fp32 update; caller owns budget, sampling, normalizers and logs.

    Reject nonfinite losses/gradients before modifying parameters or Adam state.
    No implicit normalizer updates: fit once on the declared training data.
    """
    obs, goal, target = batch
    optimizer.zero_grad(set_to_none=True)
    loss = (policy(obs, goal) - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Door bootstrap loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), grad.detach()
