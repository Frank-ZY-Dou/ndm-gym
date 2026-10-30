"""The verdict rule of the RigidFormer checks (tools/rf_checks.py): better when at least one of return, failed clips and
MPJPE-local is significantly better and none is significantly worse; and the held-out split of an all-clip evaluation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from rf_checks import compare, split_scores  # noqa: E402


def _seeds(ret, failed, mpjpe):
    return [{"norm_return": r, "n_failed": f, "mpjpe_l": m} for r, f, m in zip(ret, failed, mpjpe)]


MLP = _seeds((0.80, 0.81, 0.82), (2, 3, 4), (0.020, 0.021, 0.022))


@pytest.mark.parametrize(
    "rf, verdict, better",
    [
        (_seeds((0.90, 0.91, 0.92), (2, 3, 4), (0.020, 0.021, 0.022)), "RigidFormer better (return)", True),
        (_seeds((0.80, 0.81, 0.82), (2, 3, 4), (0.030, 0.031, 0.032)), "RigidFormer worse (MPJPE-local (mm))", False),
        (_seeds((0.90, 0.91, 0.92), (2, 3, 4), (0.030, 0.031, 0.032)), "mixed", False),
        (_seeds((0.80, 0.82, 0.81), (3, 2, 4), (0.021, 0.020, 0.022)), "no significant difference", False),
    ],
)
def test_verdicts(rf, verdict, better):
    rows, v, b = compare(MLP, rf)
    assert v.startswith(verdict) and b == better
    assert [r["metric"] for r in rows] == ["return", "failed clips", "MPJPE-local (mm)"]
    assert rows[0]["diff"] == pytest.approx(sum(x["norm_return"] for x in rf) / 3 - 0.81)


def test_split_scores():
    res = {"per_clip": {"a": 0.9, "b": 0.5, "c": 0.7}, "failed": ["b"],
           "per_clip_err": {"a": {"mpjpe_l": 0.02}, "b": {"mpjpe_l": 0.06}, "c": {"mpjpe_l": 0.04}}}  # fmt: skip
    s = split_scores(res, ["b", "c"])
    assert s["norm_return"] == pytest.approx(0.6) and s["n_failed"] == 1 and s["mpjpe_l"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        split_scores(res, ["a", "z"])
