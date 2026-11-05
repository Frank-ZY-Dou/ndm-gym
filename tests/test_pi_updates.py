"""--pi_updates (train.py): one policy update per iteration by default; fewer than one is refused."""

import sys

import pytest


def _parse(*extra):
    from ndm_gym.train import parse_args

    argv = sys.argv
    sys.argv = ["train", "--algo", "supertrack", "--log_dir", "/nonexistent", *extra]
    try:
        return parse_args()
    finally:
        sys.argv = argv


def test_pi_updates_default_and_refusal():
    assert _parse().pi_updates == 1
    assert _parse("--pi_updates", "2").pi_updates == 2
    with pytest.raises(SystemExit):
        _parse("--pi_updates", "0")


def test_pi_updates_refused_by_hoi_and_muscle_training():
    from ndm_gym import train_hoi, train_muscles

    assert "pi_updates" in train_hoi.UNSUPPORTED and "pi_updates" in train_muscles.UNSUPPORTED
