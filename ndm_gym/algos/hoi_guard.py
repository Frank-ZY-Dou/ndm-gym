"""Guard of the HOI trainers against non-finite or exploded simulator states (docs/hoi_design.md §1, engine deviations).

HDMI has no such guard: PhysX with 8/4 solver iterations and enforced joint velocity limits (spec §0, §3.2) does not
produce these states, whereas MuJoCo-Warp has no bad-state reset and Newton's MuJoCo solver ignores joint velocity
limits. Per env and control step, on the device and inside the compiled task step (``nonfinite``), an env is flagged
when a value of the state after the step, the substep joints, the contact forces, the torques or the observation groups
the step returns is beyond its bound below (NaN fails the comparison). The trainers terminate a flagged env (no
bootstrap) and reset it through the normal path; the reset's writes also clear MuJoCo-Warp's solver state of that world
(Isaac Lab's NewtonManager.forward resets the solver's internals, qacc_warmstart among them, of the worlds written since
the last step). ``NonfiniteMonitor`` counts the flagged envs, records the first event of a run directory and, with
``check``, stops training at a log row when an event rate, a run of skipped updates or a non-finite parameter is out of
bounds.
"""

from __future__ import annotations

import collections
import math
import os

import torch
import torch.nn as nn

from ..tasks.hdmi import HDMIInput

# generous bounds: a healthy step stays orders of magnitude below them, so only a diverged or non-finite state fails
POS_BOUND = 1e3  # positions relative to the env origin (m); also joint and hinge angles (rad) and rotation-matrix entries
VEL_BOUND = 1e3  # link (m/s, rad/s), joint and hinge velocities (rad/s); the G1's joint speed limits are 20-37 rad/s
FORCE_BOUND = 1e6  # contact forces (N), actuator and hinge torques (N m)
OBS_BOUND = 1e6  # observation groups: they hold the raw policy actions, which HDMI does not clip (spec §3.1)
MAX_EVENT_RATE = 1e-4  # flagged envs per env-step of a log window above which training stops (``check``)
MAX_SKIPPED = 20  # consecutive skipped updates (non-finite gradients) above which training stops (``check``)
HISTORY = 5  # steps of the first event's record


def guard_terms(x: HDMIInput, obs: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor, float]]:
    """The quantities checked at one step, in a fixed order, with their bounds: the state after the step (``x.now``),
    the joints after substeps 3 and 4, the feet's and end effectors' contact forces, the actuators' torque, a hinge's
    angle, rate and torque, and the observation groups ``obs``."""
    n = x.now
    terms = [("body_pos", n.body_pos, POS_BOUND), ("body_rot", n.body_rot, POS_BOUND), ("body_lin_vel", n.body_lin_vel, VEL_BOUND),
             ("body_ang_vel", n.body_ang_vel, VEL_BOUND), ("joint_pos", n.joint_pos, POS_BOUND), ("joint_vel", n.joint_vel, VEL_BOUND),
             ("obj_pos", n.obj_pos, POS_BOUND), ("obj_rot", n.obj_rot, POS_BOUND), ("eef_pos", n.eef_pos, POS_BOUND),
             ("target_pos", n.target_pos, POS_BOUND), ("eef_force", n.eef_force, FORCE_BOUND), ("joint_pos_sub", x.joint_pos_sub, POS_BOUND),
             ("joint_vel_sub", x.joint_vel_sub, VEL_BOUND), ("feet_force", x.feet_force, FORCE_BOUND),
             ("applied_torque", x.applied_torque, FORCE_BOUND)]  # fmt: skip
    if n.obj_joint_pos is not None:
        terms += [("hinge_pos", n.obj_joint_pos, POS_BOUND), ("hinge_vel", n.obj_joint_vel, VEL_BOUND),
                  ("hinge_torque", x.obj_joint_torque, FORCE_BOUND)]  # fmt: skip
    return terms + [(f"obs/{k}", v, OBS_BOUND) for k, v in obs.items()]


def nonfinite(x: HDMIInput, obs: dict[str, torch.Tensor], action: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The guard of one step (module docstring), traced into the trainers' compiled task steps: which checked quantity
    fails per env, (N, F) bool, ~(|v| <= bound) so that NaN fails; and the step's record for ``NonfiniteMonitor``,
    (N, D): the checked quantities, the action and HDMI's frame counter before the step ``t``, flattened."""
    terms = guard_terms(x, obs)
    N = action.shape[0]
    failed = torch.stack([~(v.abs() <= b).reshape(N, -1).all(-1) for _, v, b in terms], -1)
    record = torch.cat([v.reshape(N, -1).float() for _, v, _ in terms] + [action.reshape(N, -1), t.reshape(N, 1).float()], -1)
    return failed, record


def record_layout(x: HDMIInput, obs: dict[str, torch.Tensor], action: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Names and per-env shapes of ``nonfinite``'s record, from tensors of a step's shapes (e.g. a reset's input)."""
    return [(name, tuple(v.shape[1:])) for name, v, _ in guard_terms(x, obs)] + [("action", tuple(action.shape[1:])), ("t", ())]


class SkippedUpdates:
    """Updates skipped for a non-finite gradient: their number since the last log row and the longest run of
    consecutive ones in that window (a run carries over log rows)."""

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


class DeviceSkippedUpdates:
    """``SkippedUpdates`` on the device (train_muscles --no_sync): each call takes a 0-dim bool tensor and updates the counts
    in place (a CUDA graph's replays update them); ``take`` reads them (one synchronization, at log rows)."""

    def __init__(self, device):
        z = lambda: torch.zeros((), dtype=torch.long, device=device)  # noqa: E731
        self.n_t, self.run_t, self.longest_t = z(), z(), z()

    def __call__(self, skipped: torch.Tensor):
        s = skipped.long()
        self.n_t.add_(s)
        self.run_t.copy_((self.run_t + 1) * s)
        self.longest_t.copy_(torch.maximum(self.longest_t, self.run_t))

    @property
    def n(self) -> int:
        return int(self.n_t)

    @property
    def run(self) -> int:
        return int(self.run_t)

    def take(self) -> tuple[int, int]:
        """The window's number and longest run, then a new window."""
        n, longest = torch.stack((self.n_t, self.longest_t)).tolist()
        self.n_t.zero_()
        self.longest_t.copy_(self.run_t)
        return n, longest


def nonfinite_tensors(modules: dict[str, nn.Module | None], tensors: dict[str, torch.Tensor] | None = None) -> list[str]:
    """Names of the floating-point parameters and buffers of ``modules`` (and of ``tensors``) that hold a non-finite
    value; one host synchronization."""
    named = [(f"{m}.{n}", t) for m, mod in modules.items() if mod is not None for n, t in (*mod.named_parameters(), *mod.named_buffers())]
    named = [(n, t) for n, t in named + list((tensors or {}).items()) if t.is_floating_point()]
    ok = torch.stack([torch.isfinite(t).all() for _, t in named]).tolist()
    return [n for (n, _), good in zip(named, ok) if not good]


def check(iteration: int, modules: dict, tensors: dict | None = None, n_flagged: int = 0, env_steps: int = 0, skipped: dict[str, int] | None = None):
    """Stop training before any checkpoint is written (no silent fallbacks) if a parameter or buffer of
    ``modules`` (or a tensor of ``tensors``) is not finite, if more than MAX_EVENT_RATE envs per env-step were flagged in
    a log window (``n_flagged`` in ``env_steps``), or after a run of more than MAX_SKIPPED consecutive skipped updates
    (``skipped``: kind -> the window's longest run)."""
    bad = nonfinite_tensors(modules, tensors)
    if bad:
        raise RuntimeError(f"iteration {iteration}: non-finite parameters or buffers, no checkpoint written: {', '.join(bad[:20])}")
    if env_steps > 0 and n_flagged / env_steps > MAX_EVENT_RATE:
        raise RuntimeError(f"iteration {iteration}: {n_flagged} envs flagged as non-finite or diverged in {env_steps} env-steps, more than "
                           f"{MAX_EVENT_RATE:g} per env-step (hoi_guard.MAX_EVENT_RATE); no checkpoint written")  # fmt: skip
    for kind, run in (skipped or {}).items():
        if run > MAX_SKIPPED:
            raise RuntimeError(f"iteration {iteration}: {run} consecutive {kind} updates skipped for non-finite gradients, more than "
                               f"{MAX_SKIPPED} (hoi_guard.MAX_SKIPPED); no checkpoint written")  # fmt: skip


class NonfiniteMonitor:
    """The guard's bookkeeping in a trainer, on the device between the trainer's existing synchronizations: the number of
    flagged envs (``take``, at log rows), the records of the last HISTORY steps of all envs (the compiled step's outputs
    themselves, no copies) and the first flagged env's history, captured at its step; ``dump`` writes that event to
    <log_dir>/nonfinite_event.pt, once per run directory. ``layout``: ``record_layout``; ``rows``: the record's rows (the
    simulator's envs, which may be more than the learner's)."""

    def __init__(self, log_dir: str, layout: list[tuple[str, tuple[int, ...]]], fields: list[str], rows: int, device, compile: bool = True):
        self.path = os.path.join(log_dir, "nonfinite_event.pt")
        self.dumped = os.path.exists(self.path)  # a resumed run keeps the first event of its directory
        self.layout, self.fields = layout, fields
        width = sum(math.prod(s) for _, s in layout)
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        self.ring = collections.deque([z(rows, width) for _ in range(HISTORY)], maxlen=HISTORY)  # zeros before the first steps
        self.count, self.steps, self.captured = z((), dtype=torch.long), z((), dtype=torch.long), z((), dtype=torch.bool)
        self.snap, self.meta = z(HISTORY, width), z(2 + len(fields), dtype=torch.long)  # meta: env, step, failed fields
        self._capture = torch.compile(self._capture_impl, dynamic=False) if compile else self._capture_impl

    @staticmethod
    def _capture_impl(bad, failed, records: list[torch.Tensor], count, steps, captured, snap, meta):
        any_bad = bad.any()
        first = any_bad & ~captured
        # the first flagged env as an index tensor: indexing with a 0-d tensor would make the compiled function read it on the host
        e = torch.argmax(bad.to(torch.int32)).view(1)
        snap = torch.where(first, torch.cat([r.index_select(0, e) for r in records]), snap)
        meta = torch.where(first, torch.cat((e, steps.view(1), failed.index_select(0, e).flatten().long())), meta)
        return count + bad.sum(), steps + 1, captured | any_bad, snap, meta

    def step(self, bad: torch.Tensor, failed: torch.Tensor, record: torch.Tensor):
        """One step's guard: the learner's envs' flags ``bad`` (N,) and ``failed`` (N, F), the step's ``record``."""
        self.ring.append(record)
        out = self._capture(bad, failed, list(self.ring), self.count, self.steps, self.captured, self.snap, self.meta)
        self.count, self.steps, self.captured, self.snap, self.meta = out

    def take(self) -> int:
        """The flagged envs since the last call (a synchronization; the trainers call it at log rows)."""
        n = int(self.count)
        self.count = torch.zeros_like(self.count)
        return n

    def event_step(self) -> int:
        """The step (0-based, counted from the monitor's construction) of the captured event."""
        return int(self.meta[1])

    def dump(self, iteration: int):
        """Write the captured event and print one line: the env, the iteration, the frame counter before the step, the
        failed quantities and the env's last HISTORY records split by name (oldest first; zeros before the run's first
        step, and an episode boundary shows as a jump of ``t``)."""
        meta, snap = self.meta.tolist(), self.snap.cpu()
        parts, i = {}, 0
        for name, shape in self.layout:
            k = math.prod(shape)
            parts[name] = snap[:, i : i + k].reshape(HISTORY, *shape)
            i += k
        failed = [f for f, on in zip(self.fields, meta[2:]) if on]
        event = {"env": meta[0], "iteration": iteration, "step": meta[1], "t": int(parts["t"][-1]), "failed": failed, "history": parts}
        torch.save(event, self.path)
        self.dumped = True
        print(f"non-finite state: env {meta[0]} at iteration {iteration} (frame {event['t']}): {', '.join(failed)}; recorded in {self.path}",
              flush=True)  # fmt: skip
