"""Device-resident, bounded time/world replay for persistent Door collection."""
import torch


class DoorReplayRing:
    """Uniform valid windows; terminal next states survive and resets never join.

    One append copies one whole world batch. Sampling never copies the dataset
    or synchronizes per-world tensors with the host. Capacity is control steps.
    """

    def __init__(self, capacity, worlds, device='cpu', *, nq=30, nv=30, nu=28):
        if type(capacity) is not int or type(worlds) is not int or min(capacity, worlds) < 1:
            raise ValueError('positive integer capacity and worlds required')
        self.capacity, self.worlds = capacity, worlds
        self.device = torch.device(device)
        self.data = {key: torch.empty((capacity, worlds, size), device=device)
                     for key, size in [('qpos', nq), ('qvel', nv), ('next_qpos', nq),
                                       ('next_qvel', nv), ('actions', nu), ('previous_actions', nu)]}
        self.frames = torch.empty((capacity, worlds), dtype=torch.long, device=device)
        self.done = torch.empty((capacity, worlds), dtype=torch.bool, device=device)
        self.total = 0

    def state_dict(self):
        """Snapshot initialized slots only, including terminal next states."""
        length = min(self.total, self.capacity)
        return dict(capacity=self.capacity, worlds=self.worlds, total=self.total,
                    data={k: v[:length].detach().cpu().clone() for k, v in self.data.items()},
                    frames=self.frames[:length].cpu().clone(), done=self.done[:length].cpu().clone())

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or state['worlds'] != self.worlds:
            raise ValueError('Replay dimensions mismatch')
        if type(state['total']) is not int or state['total'] < 0:
            raise ValueError('Invalid replay write count')
        length = min(state['total'], self.capacity)
        source = dict(state['data'], frames=state['frames'], done=state['done'])
        target = dict(self.data, frames=self.frames, done=self.done)
        if source.keys() != target.keys() or any(
                source[k].shape != v[:length].shape or source[k].dtype != v.dtype
                for k, v in target.items()):
            raise ValueError('Replay checkpoint schema mismatch')
        for k, v in target.items():
            v[:length].copy_(source[k])
        self.total = state['total']

    def append(self, batch):
        converted = {key: torch.as_tensor(batch[key], device=self.device, dtype=value.dtype)
                     for key, value in self.data.items()}
        converted['frames'] = torch.as_tensor(batch['frames'], device=self.device, dtype=torch.long)
        converted['done'] = torch.as_tensor(batch['done'], device=self.device, dtype=torch.bool)
        stores = dict(self.data, frames=self.frames, done=self.done)
        if any(converted[key].shape != store.shape[1:] for key, store in stores.items()):
            raise ValueError('batch shape does not match ring worlds/state contract')
        slot = self.total % self.capacity
        for key, store in stores.items():
            store[slot].copy_(converted[key])
        self.total += 1

    def sample(self, batch, horizon, generator=None):
        if type(batch) is not int or type(horizon) is not int or min(batch, horizon) < 1:
            raise ValueError('positive integer batch/horizon required')
        length = min(self.total, self.capacity)
        if horizon > length:
            raise ValueError('No complete windows for requested horizon')
        starts = torch.arange(self.total - length, self.total - horizon + 1, device=self.device)
        indices = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        # A terminal at the last transition is valid; an earlier one crosses reset.
        valid = ~self.done[indices[:, :-1]].any(dim=1)
        choices = valid.nonzero()
        if len(choices) == 0:
            raise ValueError('No reset-safe windows for requested horizon')
        chosen = choices[torch.randint(len(choices), (batch,), device=self.device, generator=generator)]
        times = indices[chosen[:, 0]].T
        worlds = chosen[:, 1]
        q = self.data['qpos'][times, worlds]
        v = self.data['qvel'][times, worlds]
        q = torch.cat((q, self.data['next_qpos'][times[-1], worlds][None]))
        v = torch.cat((v, self.data['next_qvel'][times[-1], worlds][None]))
        return (q, v, self.data['actions'][times, worlds], self.frames[times[0], worlds],
                self.data['previous_actions'][times[0], worlds])
