"""Discrete note matching, final-frame censoring and batched physical strike/reset behavior."""

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "wuji_piano"


def test_events_one_to_one_and_censored_release(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH))
    pytest.importorskip("scipy")
    from note_events import event_stats, match_notes

    # Both matches take precedence over one exact match. Each detected strike is used at most once.
    assert match_notes([(1, 0, 2), (1, 2, 3)], [(1, 1, 2), (1, 3, 4)]) == [(0, 0), (1, 1)]
    assert len(match_notes([(1, 0, 1), (1, 2, 3)], [(1, 1, 2)])) == 1
    assert not match_notes([(1, 0, 1)], [(2, 0, 1), (1, 2, 3)])
    goal = np.zeros((8, 88), bool)
    goal[1:3, 42] = goal[4:, 42] = goal[1:4, 22] = True
    mask = goal.copy()
    mask[3, 22] = False
    mask[0, 80] = True
    stats = event_stats(goal, mask[None])
    assert (stats["matched"], stats["extra"], stats["missed"]) == (3, 1, 0)
    assert (stats["repeat_hits"], stats["paired_note_hits"]) == (1, 2)
    assert stats["observed_releases"] == 2
    assert stats["release_median_ms"] == 25
    assert event_stats(goal, np.zeros_like(mask)[None])["missed"] == 3


def test_batched_ode_strike_release_and_reset(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("scipy")
    pytest.importorskip("mujoco_warp")
    monkeypatch.syspath_prepend(str(BENCH))
    from download_assets import HAND_REV
    from scripted_ode import commands
    from train_ode_to_joy import A, DEV, FreeHands, build, piano_roll
    import warp as wp

    assets = Path(os.environ.get("WUJI_PIANO_ASSETS", ROOT / "data/wuji_piano"))
    hands = assets / f"wuji-description-{HAND_REV}/hand2/hand2_beta2/body/mjcf"
    if not (hands / "left.xml").exists():
        pytest.skip("download assets with benchmarks/wuji_piano/download_assets.py")
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        m = build(hands, assets / "keyboard/piano.xml")
        sim = FreeHands(m, 4)
        actions = torch.as_tensor(commands(m, piano_roll()), device=DEV)
        neutral = torch.zeros((4, A), device=DEV)
        initial = sim.reset(neutral).clone()
        down = []
        for a in actions[:42]:
            down.append(sim.step(a.expand(4, -1))[:, 2 * A :] >= 0.87)
        played = torch.stack(down)
        assert played[25, :, 48 - 21].all() and played[25, :, 64 - 21].all()
        assert not played[31, :, 64 - 21].any() and played[40, :, 64 - 21].all()
        torch.testing.assert_close(sim.reset(neutral), initial)
