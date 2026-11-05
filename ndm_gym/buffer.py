"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .robots.g1 import NUM_J
from .state import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (NUM_J, torch.float32),
    "a_prev": (NUM_J, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    """``fields``: name -> (shape, dtype) of a transition, FIELDS (LAFAN1 tracking) by default; ``window_keys``: the
    fields a k-step window returns; ``priorities``: keep the window priorities of ``sample_windows_prioritized`` (without
    them, ``add`` skips their upkeep, a reduction over the whole buffer at every step)."""

    window_keys = ("s", "s1", "a", "xi", "done")

    def __init__(self, T: int, N: int, device: str, fields: dict | None = None, priorities: bool = True):
        self.T, self.N, self.device = T, N, device
        self.fields = FIELDS if fields is None else fields
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        # priority of the window starting at each (row, env), for error-prioritized world-model fitting
        self.prio = torch.ones(T, N, device=device) if priorities else None
        # train.py --update_graphs (BlinkMimic's --cuda_graph): a device copy of the write head, so that sampling captured in
        # a CUDA graph follows it; --graph_early: of the fill level too, sampling bounds follow it (uniform floats scaled by
        # the fill level instead of randint with a host bound: the same distribution, another random stream)
        self.ptr_t = self.size_t = None

    def use_device_ptr(self, device_size: bool = False):
        if self.prio is not None:
            raise ValueError("device-side sampling bounds and window priorities cannot be combined")
        self.ptr_t = torch.full((), self.ptr, dtype=torch.long, device=self.device)
        if device_size:
            self.size_t = torch.full((), self.size, dtype=torch.long, device=self.device)

    def _randint(self, n, B: int) -> torch.Tensor:
        """B uniform integers in [0, n): n a host int (torch.randint), or with --graph_early a device tensor."""
        if self.size_t is None:
            return torch.randint(0, n, (B,), device=self.device)
        return torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def _size(self):
        return self.size if self.size_t is None else self.size_t

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        if self.prio is not None:
            self.prio[self.ptr] = self.prio.max()  # new data is sampled at least as often as the hardest seen so far
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)
        if self.size_t is not None:
            self.size_t.fill_(self.size)

    def add_device(self, tr: dict):
        """``add`` at the device-side write head (train.py --collect_graph: capturable): advances ``ptr_t`` and ``size_t``
        on the device; ``advance_host`` advances the host's ``ptr`` and ``size`` after each replay."""
        row = self.ptr_t.view(1)
        for k in self.fields:
            self.data[k].index_copy_(0, row, tr[k].unsqueeze(0).to(self.data[k].dtype))
        self.ptr_t.copy_((self.ptr_t + 1) % self.T)
        self.size_t.copy_(torch.clamp(self.size_t + 1, max=self.T))

    def advance_host(self):
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = self.ptr if self.ptr_t is None else self.ptr_t
        return (ptr - self._size() + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint(self._size() - k + 1, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in self.window_keys}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_windows_prioritized(self, B: int, k: int, alpha: float):
        """Like ``sample_windows`` but with start probability proportional to priority^alpha (no importance
        weighting: emphasizing hard transitions is the point). Also returns the starts for ``update_priorities``."""
        self._need_priorities()
        n_t = self.size - k + 1
        rows0 = self._row(torch.arange(n_t, device=self.device))
        idx = torch.multinomial(self.prio[rows0].flatten() ** alpha, B, replacement=True)
        t0, env = idx // self.N, idx % self.N
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in self.window_keys}
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["start"] = (rows[:, 0], env)
        return out

    def update_priorities(self, start: tuple, err: torch.Tensor, eps: float = 1e-6):
        self._need_priorities()
        self.prio[start] = err.detach().float() + eps

    def _need_priorities(self):
        if self.prio is None:
            raise ValueError("this replay buffer keeps no window priorities (ReplayBuffer(..., priorities=False))")

    def sample_start_points(self, B: int):
        """B start points drawn uniformly: their age (0 = oldest) and env."""
        age = self._randint(self._size(), B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        return age, env

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        return self.starts_at(*self.sample_start_points(B), hist)

    def at(self, key: str, age: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Field ``key`` at ``age`` (0 = oldest) of ``env``."""
        return self.data[key][self._row(age), env]

    def terminations(self, failed):
        """Age (0 = oldest) and env of every stored termination, and the age of its episode's first stored state.

        failed(s1, f) -> bool: which of the given ``done`` transitions are terminations rather than time-outs.
        """
        rows, envs = self.data["done"].nonzero(as_tuple=True)
        if len(rows):
            keep = failed(self.data["s1"][rows, envs], self.data["f"][rows, envs])
            rows, envs = rows[keep], envs[keep]
        age = (rows - (self.ptr - self.size)) % self.T
        first = age - torch.minimum(self.data["ep_step"][rows, envs].long(), age)  # episode start, or the buffer's
        return age, envs, first

    def pre_failure_points(self, B: int, window: int, failed):
        """B start points (age, env) from the last ``window`` steps before the replay's terminations: a termination
        drawn uniformly, then a state among the ``window`` up to and including its last one, inside its episode and the
        buffer. None while the replay holds no termination (``failed`` as in ``terminations``)."""
        age, envs, first = self.terminations(failed)
        if len(age) == 0:
            return None
        i = torch.randint(0, len(age), (B,), device=self.device)
        reach = (age[i] - first[i]).clamp(max=window - 1)
        back = (torch.rand(B, device=self.device) * (reach + 1)).long()
        return age[i] - back, envs[i]

    def pre_failure_starts(self, B: int, hist: int, window: int, failed):
        """B start points like ``sample_starts`` from ``pre_failure_points``; None while the replay holds no termination."""
        points = self.pre_failure_points(B, window, failed)
        return None if points is None else self.starts_at(*points, hist)

    def pre_failure_windows(self, B: int, k: int, window: int, failed, generator: torch.Generator | None = None):
        """Up to B k-step windows like ``sample_windows`` whose last step lies in the last ``window`` steps up to and
        including a termination: a termination drawn uniformly, then the window's end, inside its episode and the
        buffer (terminations with a shorter episode are skipped). None while the replay holds none that fits."""
        age, envs, first = self.terminations(failed)
        lo = torch.maximum(age - (window - 1), first + k - 1)  # earliest end: in the window, k steps after the start
        fits = lo <= age
        age, envs, lo = age[fits], envs[fits], lo[fits]
        if len(age) == 0:
            return None
        i = torch.randint(0, len(age), (B,), device=self.device, generator=generator)
        end = lo[i] + (torch.rand(B, device=self.device, generator=generator) * (age[i] - lo[i] + 1)).long()
        return self.windows_at(end - (k - 1), envs[i], k)

    def windows_at(self, t0: torch.Tensor, env: torch.Tensor, k: int):
        """The k-step windows starting at age ``t0`` of ``env``, as ``sample_windows`` returns them."""
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in self.window_keys}
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(len(t0), 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["start"] = (rows[:, 0], env)
        return out

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """The states at ``age`` (0 = oldest) of ``env``, with the indices of their observation history."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
