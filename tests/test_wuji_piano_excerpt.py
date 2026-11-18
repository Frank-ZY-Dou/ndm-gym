"""Verify bar boundaries, hand assignments and frame grading for the two-hand excerpt."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def excerpt(monkeypatch):
    pytest.importorskip("mujoco_warp")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks" / "wuji_piano"))
    import train_fur_elise

    return train_fur_elise


def test_excerpt_score(excerpt):
    roll = excerpt.score()
    music = roll[excerpt.WARMUP : excerpt.WARMUP + excerpt.MUSIC_FRAMES]
    assert music.shape == (133, 2)
    assert not roll[: excerpt.WARMUP].any()
    assert not roll[-5:].any()
    # The pickup is E5 and D-sharp5 over a resting left hand.
    assert np.array_equal(music[:8], [[76, 0]] * 4 + [[75, 0]] * 4)
    assert np.array_equal(music[-4:], [[72, 0]] * 4)
    onsets = (music != 0) & (music != np.vstack(([0, 0], music[:-1])))
    assert onsets.sum(0).tolist() == [26, 9]
    assert music[:, 1][onsets[:, 1]].tolist() == [45, 52, 57, 40, 52, 56, 45, 52, 57]


def test_excerpt_grades_the_music_without_padding(excerpt):
    parts = torch.tensor(excerpt.score())
    midi = torch.arange(21, 109)
    goal = (parts[:, :, None] == midi).any(1)
    depth = goal[:, None].float()
    depth[: excerpt.WARMUP] = 1
    depth[-5:] = 1
    perfect = excerpt.stats(depth, goal)
    assert perfect["all"] == [[1, 1, 1]]
    assert perfect["perfect_rollouts"] == 1
    depth[excerpt.WARMUP, 0, 76 - 21] = 0
    missed = excerpt.stats(depth, goal)
    assert missed["all"][0][2] == pytest.approx(132 / 133)
    assert missed["perfect_rollouts"] == 0


def test_eden_five_finger_score(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks" / "wuji_piano"))
    from five_finger_scale import piano_roll

    target = piano_roll()
    expected = np.zeros((152, 88), dtype=bool)
    for i, pitch in enumerate((60, 62, 64, 65, 67, 65, 64, 62, 60)):
        first = 20 + 15 * i
        expected[first : first + 12, pitch - 21] = True
    np.testing.assert_array_equal(target, expected)


def test_ode_is_polyphonic_with_edens_rests_and_durations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks" / "wuji_piano"))
    from piano_score import piano_roll

    roll = piano_roll()
    assert roll.shape == (230, 88)
    onsets = roll & ~np.concatenate((np.zeros_like(roll[:1]), roll[:-1]))
    assert onsets.sum() == 19
    assert np.flatnonzero(onsets.sum(1) == 2).tolist() == [20, 74, 127, 180]
    assert (roll.sum(1) == 2).sum() == 164
    assert (np.flatnonzero(roll[20]) + 21).tolist() == [48, 64]
    assert np.all(roll[20:70, 48 - 21])
    assert not roll[30, 64 - 21] and roll[34, 64 - 21]
