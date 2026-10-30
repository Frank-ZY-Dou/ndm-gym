"""Clip lists (``ndm_gym.train --clips``): reading a list file, selecting clips by name, and the LAFAN1 split of
benchmarks/g1_lafan1_tracking/splits (subjects 1-4 to train, subject 5 to test)."""

import os

import pytest
import torch

from ndm_gym.motion.sources import Clip, read_clip_list, select

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "benchmarks", "g1_lafan1_tracking", "splits")


def _clip(name):
    return Clip(name, torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2), torch.zeros(2, 29), 30.0)


def test_read_and_select(tmp_path):
    p = tmp_path / "clips.txt"
    p.write_text("# a comment\nwalk_b\n\n  run_a  # trailing comment\n")
    names = read_clip_list(str(p))
    assert names == ["walk_b", "run_a"]
    clips = [_clip(n) for n in ("run_a", "jump_c", "walk_b")]
    assert [c.name for c in select(clips, names)] == ["run_a", "walk_b"]  # in the clips' order, not the list's


def test_errors(tmp_path):
    clips = [_clip("run_a"), _clip("walk_b")]
    with pytest.raises(ValueError, match="no clip named"):
        select(clips, ["run_a", "swim_z"])
    p = tmp_path / "dup.txt"
    p.write_text("run_a\nrun_a\n")
    with pytest.raises(ValueError, match="more than once"):
        read_clip_list(str(p))
    p = tmp_path / "empty.txt"
    p.write_text("# nothing\n\n")
    with pytest.raises(ValueError, match="no clips"):
        read_clip_list(str(p))


def test_lafan1_split():
    data = os.path.join(REPO, "data", "lafan1_g1")
    every = sorted(os.path.splitext(f)[0] for f in os.listdir(data) if f.endswith(".csv"))
    train = read_clip_list(os.path.join(SPLITS, "lafan1_train_subjects1-4.txt"))
    test = read_clip_list(os.path.join(SPLITS, "lafan1_test_subject5.txt"))
    assert len(every) == 40 and len(test) == 7
    assert sorted(train + test) == every  # disjoint and complete
    assert all(n.endswith("_subject5") for n in test)
    assert not any(n.endswith("_subject5") for n in train)
