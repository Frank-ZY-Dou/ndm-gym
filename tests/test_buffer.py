"""Replay-buffer sampling of imagination starts before terminations (``--imag_fail_starts``)."""

import torch

from conftest import DEV


def _replay(steps: int, ends: dict):
    """Replay of T=30 rows and 3 envs after ``steps`` steps. ``ends`` maps (env, step) to "fail" or "timeout" for the
    last transition of an episode. f holds the global step (to identify rows); s1[..., 0] = 1 marks a failure."""
    from ndm_gym.buffer import FIELDS, ReplayBuffer

    b = ReplayBuffer(30, 3, DEV)
    ep = torch.zeros(3, dtype=torch.int32)
    for t in range(steps):
        tr = {k: torch.zeros((3,) + ((s,) if isinstance(s, int) else s), dtype=dt) for k, (s, dt) in FIELDS.items()}
        tr["f"][:] = t
        tr["ep_step"] = ep.clone()
        for e in range(3):
            kind = ends.get((e, t))
            if kind is not None:
                tr["done"][e] = True
                tr["s1"][e, 0] = 1.0 if kind == "fail" else 0.0
        b.add(tr)
        ep = torch.where(tr["done"], 0, ep + 1).to(torch.int32)
    return b


def failed(s1, f):
    return s1[:, 0] > 0.5


def test_starts_lie_in_the_window_before_a_failure():
    b = _replay(37, {(0, 33): "fail", (1, 30): "timeout", (0, 19): "timeout"})  # wraps: rows hold steps 7..36
    s, h = b.pre_failure_starts(4000, 3, 5, failed)
    f = s["f"]
    assert set(f.tolist()) == {29, 30, 31, 32, 33}  # the 5 steps up to the failure, across the ring's wrap
    torch.testing.assert_close(s["s"], b.data["s"][f % 30, 0])  # env 0, the failing one
    counts = torch.bincount(f - 29, minlength=5).float()
    assert counts.min() > 0.8 * counts.mean()  # uniform over the window
    assert bool((h["f"] >= 20).all()) and bool((h["f"] <= f.unsqueeze(-1)).all())  # history inside the episode (20..33)


def test_window_is_cut_at_the_episode_start_and_buffer_start():
    b = _replay(12, {(2, 10): "fail", (2, 8): "timeout"})  # the failing episode is steps 9..10
    s, _ = b.pre_failure_starts(1000, 3, 50, failed)
    assert set(s["f"].tolist()) == {9, 10}
    b = _replay(40, {(1, 12): "fail"})  # rows hold steps 10..39; the failing episode began at step 0
    s, _ = b.pre_failure_starts(1000, 3, 50, failed)
    assert set(s["f"].tolist()) == {10, 11, 12}  # cut at the oldest stored step


def test_windows_end_in_the_window_before_a_failure():
    b = _replay(37, {(0, 33): "fail", (1, 30): "timeout", (0, 19): "timeout"})  # failing episode: steps 20..33
    w = b.pre_failure_windows(4000, 8, 10, failed)
    first = b.data["f"][w["start"][0], w["start"][1]]
    assert bool((w["start"][1] == 0).all())  # only the failing env
    # ends among the last 10 steps (24..33) and 7 steps after the episode's start (27..), so starts 20..26
    assert set(first.tolist()) == set(range(20, 27))
    assert bool((w["mask"] == 1).all())  # no episode end inside, only possibly at the last step
    assert b.pre_failure_windows(10, 20, 10, failed) is None  # the failing episode is shorter than a window


def test_no_failure_in_replay():
    b = _replay(20, {(0, 5): "timeout", (1, 9): "timeout"})
    assert b.pre_failure_starts(10, 3, 5, failed) is None
