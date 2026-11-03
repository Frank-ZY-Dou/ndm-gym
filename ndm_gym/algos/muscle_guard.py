"""Guard of MuscleVAE training (ndm_gym.train_muscles) against non-finite or diverged simulator states and updates, as the
HOI trainers guard theirs. MuscleSim already ends an episode whose state is not finite or whose observation exceeds 50
(MuscleVAE's rule); the guard keeps such a step out of learning: per env and control step, on the device, an env is
flagged (``flagged``) when a value of its state after the step is beyond the bounds below (NaN fails the comparison).
The trainer terminates and resets a flagged env (the reset also clears MuJoCo-Warp's solver state of that world), and
its transition enters the replay with the next state replaced by the state before (``bad``), which keeps it out of the
world model's windows, the imagination's starts and the normalizers (``masked_update``). Updates with a non-finite
gradient are skipped and counted (``SkippedUpdates``); ``check`` stops training at a log row, before any checkpoint, on a
non-finite parameter or buffer, too many flagged envs or a long run of skipped updates."""

from __future__ import annotations

import torch
import torch.nn as nn

# generous bounds: a healthy step stays orders of magnitude below them, so only a diverged or non-finite state fails
POS_BOUND = 1e3  # positions (m); quaternion entries and the muscle state (both within [-1, 1] when healthy)
VEL_BOUND = 1e3  # velocities (m/s) and rotation rates (rad/s)
MAX_EVENT_RATE = 1e-4  # flagged envs per env-step of a log window above which training stops
MAX_SKIPPED = 20  # consecutive skipped updates above which training stops


def flagged(s1: torch.Tensor, fat1: torch.Tensor) -> torch.Tensor:
    """(N,) whether the state after a step (N, 23, 13) or the muscle state (N, 15) is non-finite or beyond the bounds."""
    ok = (s1[..., :7].abs() <= POS_BOUND).flatten(1).all(-1) & (s1[..., 7:].abs() <= VEL_BOUND).flatten(1).all(-1) & (fat1.abs() <= POS_BOUND).all(-1)
    return ~ok


@torch.no_grad()
def masked_update(norm, x: torch.Tensor, keep: torch.Tensor):
    """``norm.update(x)`` (a core.normalizer.RunningNorm) from the rows where ``keep`` (N,) holds, written into the
    normalizer's buffers and without a host synchronization: the rows left out are replaced by zeros rather than
    multiplied by them (NaN x 0 is NaN); without rows, or once the normalizer is frozen at max_count, nothing changes."""
    x = x.reshape(-1, x.shape[-1]).float()
    k = keep.reshape(-1, 1)
    n = k.sum().float()
    m = torch.where(k, x, 0.0).sum(0) / n.clamp(min=1.0)
    v = torch.where(k, (x - m) ** 2, 0.0).sum(0) / n.clamp(min=1.0)
    tot = norm.count + n
    tc = tot.clamp(min=1.0)
    delta = m - norm.mean
    upd = (n > 0) & (norm.count < norm.max_count)
    mean = norm.mean + delta * n / tc
    var = (norm.var * norm.count + v * n + delta**2 * norm.count * n / tc) / tc
    norm.mean.copy_(torch.where(upd, mean, norm.mean))
    norm.var.copy_(torch.where(upd, var, norm.var))
    norm.count.copy_(torch.where(upd, tot, norm.count))


class SkippedUpdates:
    """Updates skipped for a non-finite gradient: their number since the last log row and the longest run of consecutive
    ones in that window (a run carries over log rows)."""

    def __init__(self):
        self.n = self.run = self.longest = 0

    def __call__(self, skipped: bool):
        if skipped:
            self.n, self.run = self.n + 1, self.run + 1
            self.longest = max(self.longest, self.run)
        else:
            self.run = 0

    def take(self) -> tuple[int, int]:
        """The window's number and longest run, then a new window."""
        n, longest = self.n, self.longest
        self.n, self.longest = 0, self.run
        return n, longest


def nonfinite_tensors(modules: dict[str, nn.Module | None]) -> list[str]:
    """Names of the floating-point parameters and buffers of ``modules`` that hold a non-finite value (one host sync)."""
    named = [(f"{m}.{n}", t) for m, mod in modules.items() if mod is not None for n, t in (*mod.named_parameters(), *mod.named_buffers())]
    named = [(n, t) for n, t in named if t.is_floating_point()]
    ok = torch.stack([torch.isfinite(t).all() for _, t in named]).tolist()
    return [n for (n, _), good in zip(named, ok) if not good]


def check(iteration: int, modules: dict, n_flagged: int = 0, env_steps: int = 0, skipped: dict[str, int] | None = None):
    """Stop training (no checkpoint is written after it) if a parameter or buffer of ``modules`` is not finite, if more
    than MAX_EVENT_RATE envs per env-step were flagged in a log window (``n_flagged`` in ``env_steps``), or after a run of
    more than MAX_SKIPPED consecutive skipped updates (``skipped``: kind -> the window's longest run)."""
    bad = nonfinite_tensors(modules)
    if bad:
        raise RuntimeError(f"iteration {iteration}: non-finite parameters or buffers, no checkpoint written: {', '.join(bad[:20])}")
    if env_steps > 0 and n_flagged / env_steps > MAX_EVENT_RATE:
        raise RuntimeError(f"iteration {iteration}: {n_flagged} envs flagged as non-finite or diverged in {env_steps} env-steps, more than "
                           f"{MAX_EVENT_RATE:g} per env-step (muscle_guard.MAX_EVENT_RATE); no checkpoint written")  # fmt: skip
    for kind, run in (skipped or {}).items():
        if run > MAX_SKIPPED:
            raise RuntimeError(f"iteration {iteration}: {run} consecutive {kind} updates skipped for non-finite gradients, more than "
                               f"{MAX_SKIPPED} (muscle_guard.MAX_SKIPPED); no checkpoint written")  # fmt: skip


__all__ = ["SkippedUpdates", "check", "flagged", "masked_update", "nonfinite_tensors"]
