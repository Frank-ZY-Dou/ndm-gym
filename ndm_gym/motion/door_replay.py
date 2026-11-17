"""Episode-preserving physical Door replay, independent of simulator imports."""

import numpy as np
import torch


class DoorReplay:
    """Complete episodes, split before fitting; windows never cross resets.

    Actions are normalized actuator controls. Frame zero is reference frame zero
    in every episode; previous action at reset is the policy's zero sentinel.
    """

    def __init__(self, episodes):
        self.episodes = []
        for q, v, actions in episodes:
            q, v, actions = (torch.as_tensor(x, dtype=torch.float32).clone() for x in (q, v, actions))
            if (q.ndim != 2 or q.shape[1] != 30 or v.shape != q.shape
                    or actions.shape != (len(q)-1, 28) or len(actions) < 1
                    or not all(torch.isfinite(x).all() for x in (q, v, actions))
                    or torch.any(actions.abs() > 1)):
                raise ValueError("Expected finite complete Door episode and normalized actions")
            self.episodes.append((q, v, actions))
        if not self.episodes:
            raise ValueError("Replay requires complete episodes")
        self._flat_q, self._flat_v, self._flat_actions = (
            torch.cat([episode[i] for episode in self.episodes]) for i in range(3))
        self._window_indices = {}

    @classmethod
    def load(cls, path):
        """Load complete physical episodes written by saved-checkpoint evaluation."""
        with np.load(path, allow_pickle=False) as data:
            indices = sorted({int(k.split("_")[0][7:]) for k in data.files
                              if k.startswith("episode") and k.endswith("_qpos")})
            if indices != list(range(len(indices))):
                raise ValueError("Expected contiguous complete episode indices")
            return cls([tuple(data[f"episode{i}_{name}"] for name in ("qpos", "qvel", "actions"))
                        for i in indices])

    def split(self, heldout):
        """Hold out explicit episode indices; no transitions shared across splits."""
        indices = set(heldout)
        if not indices or not indices < set(range(len(self.episodes))):
            raise ValueError("Holdout must be a nonempty proper subset of episode indices")
        return (DoorReplay([e for i, e in enumerate(self.episodes) if i not in indices]),
                DoorReplay([e for i, e in enumerate(self.episodes) if i in indices]))

    def sample(self, batch, horizon, generator=None):
        """Uniform valid windows, returning time-major states/actions and starts."""
        if batch < 1 or horizon < 1:
            raise ValueError("Positive batch and horizon required")
        if horizon not in self._window_indices:
            indices = []
            q_offset = action_offset = 0
            for q, _, actions in self.episodes:
                indices.extend((q_offset + frame, action_offset + frame, frame)
                               for frame in range(len(actions)-horizon+1))
                q_offset += len(q)
                action_offset += len(actions)
            if not indices:
                raise ValueError("No complete windows for requested horizon")
            self._window_indices[horizon] = torch.tensor(indices)
        windows = self._window_indices[horizon]
        selected = windows[torch.randint(len(windows), (batch,), generator=generator)]
        q_index, action_index, frames = selected.unbind(1)
        steps = torch.arange(horizon+1)[:, None]
        previous = torch.where((frames > 0)[:, None],
                               self._flat_actions[(action_index-1).clamp_min(0)],
                               torch.zeros(28))
        return (self._flat_q[q_index+steps], self._flat_v[q_index+steps],
                self._flat_actions[action_index+steps[:-1]], frames, previous)



class BalancedDoorReplay(DoorReplay):
    """Equal source counts per batch; complete windows retain their reset semantics."""

    def __init__(self, sources):
        if len(sources) != 2:
            raise ValueError("Exactly two declared replay sources required")
        self.sources = tuple(sources)
        super().__init__([e for source in sources for e in source.episodes])

    def sample(self, batch, horizon, generator=None):
        if batch < 2 or batch % 2:
            raise ValueError("Balanced replay requires a positive even batch")
        parts = [source.sample(batch // 2, horizon, generator) for source in self.sources]
        return tuple(torch.cat([part[i] for part in parts], dim=1 if i < 3 else 0)
                     for i in range(5))


def collect_physical_replay(algo, sim, offsets, noise, seed):
    """Execute reference controls with fixed iid noise; never assign states after reset."""
    from .adroit_door import JOINT_NAMES

    if not offsets or not np.isfinite(offsets).all() or not np.isfinite(noise) or noise < 0:
        raise ValueError("Finite offsets and nonnegative noise required")
    rng = np.random.default_rng(seed)
    episodes = []
    for offset in offsets:
        dq = np.zeros(30)
        dq[JOINT_NAMES.index("ARRz")] = offset
        state = sim.reset(qpos_offset=dq)
        qs, vs, actions = [state[0].copy()], [state[1].copy()], []
        for prior in algo.reference_actions:
            action = np.clip(prior.detach().cpu().numpy() + rng.normal(0, noise, 28), -1, 1)
            state, _ = sim.step(action)
            qs.append(state[0].copy())
            vs.append(state[1].copy())
            actions.append(action)
        episodes.append((np.asarray(qs), np.asarray(vs), np.asarray(actions)))
    return DoorReplay(episodes)


def collect_policy_replay(algo, sim, offsets):
    """Collect current closed-loop policy episodes, with deployment reset semantics.

    No action noise or state assignment after reset. Holdouts must be split by
    the caller before normalization or optimization, as for reference replay.
    """
    from .adroit_door import JOINT_NAMES

    if not offsets or not np.isfinite(offsets).all():
        raise ValueError("Finite nonempty offsets required")
    episodes = []
    for offset in offsets:
        dq = np.zeros(30)
        dq[JOINT_NAMES.index("ARRz")] = offset
        state = sim.reset(qpos_offset=dq)
        previous = torch.zeros_like(algo.reference_actions[0])
        qs, vs, actions = [state[0].copy()], [state[1].copy()], []
        with torch.no_grad():
            for frame, prior in enumerate(algo.reference_actions):
                st = tuple(torch.as_tensor(x, dtype=torch.float32, device=prior.device) for x in state)
                action = algo.pi(algo.task.obs(st, previous),
                                 algo.task.goal(st, torch.tensor(frame, device=prior.device)), prior)
                state, _ = sim.step(action.cpu().numpy())
                qs.append(state[0].copy())
                vs.append(state[1].copy())
                actions.append(action.cpu().numpy().copy())
                previous = action
        episodes.append((np.asarray(qs), np.asarray(vs), np.asarray(actions)))
    return DoorReplay(episodes)


def collect_policy_replay_batched(algo, sim_factory, offsets, worlds):
    """Batch policy inference over bounded independent native CPU simulators.

    Physics steps remain serial; no worker threads or device physics are implied.
    Episode order and reset sentinel match serial collection. Batched linear
    algebra may change actions by rounding, so callers must opt in explicitly.
    The factory must return a fresh simulator with independent mutable state.
    """
    from .adroit_door import JOINT_NAMES

    if not isinstance(worlds, int) or worlds < 1:
        raise ValueError("Positive integer collection worlds required")
    if not offsets or not np.isfinite(offsets).all():
        raise ValueError("Finite nonempty offsets required")
    sims = [sim_factory() for _ in range(min(worlds, len(offsets)))]
    if len({id(sim) for sim in sims}) != len(sims):
        raise ValueError("Factory must return independent simulators")
    episodes = []
    for start in range(0, len(offsets), worlds):
        active = sims[:len(offsets[start:start+worlds])]
        states = []
        for sim, offset in zip(active, offsets[start:start+worlds]):
            dq = np.zeros(30)
            dq[JOINT_NAMES.index("ARRz")] = offset
            states.append(sim.reset(qpos_offset=dq))
        buffers = [([q.copy()], [v.copy()], []) for q, v in states]
        previous = torch.zeros_like(algo.reference_actions[0]).expand(len(active), -1)
        with torch.no_grad():
            for frame, prior in enumerate(algo.reference_actions):
                st = tuple(torch.as_tensor(np.stack([s[i] for s in states]),
                                           dtype=torch.float32, device=prior.device) for i in range(2))
                frames = torch.full((len(active),), frame, device=prior.device, dtype=torch.long)
                action = algo.pi(algo.task.obs(st, previous), algo.task.goal(st, frames),
                                 prior.expand(len(active), -1))
                controls = action.cpu().numpy()
                states = []
                for sim, control, (qs, vs, actions) in zip(active, controls, buffers):
                    state, _ = sim.step(control)
                    states.append(state)
                    qs.append(state[0].copy())
                    vs.append(state[1].copy())
                    actions.append(control.copy())
                previous = action
        episodes.extend(tuple(np.asarray(x) for x in buffer) for buffer in buffers)
    return DoorReplay(episodes)
