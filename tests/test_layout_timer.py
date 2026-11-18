"""LAYOUT_PROFILE=1's timer for train.py --layout (ndm_gym.algos.layouts.LayoutTimer): each step timed on the device that owns it
under AL and AWL, every 20th iteration after the warm-up, one line per timed iteration (on a GPU), and train.py's hooks: off
without the variable, refused without a layout, and in the loop's order around the split's steps."""

import inspect
import re
import types

import pytest
import torch

from ndm_gym.algos.layouts import LayoutTimer


def _split(layout, devices):
    return types.SimpleNamespace(layout=layout, devices=devices)


def test_each_step_is_timed_on_the_device_that_owns_it():
    awl = LayoutTimer(_split("AWL", ["cuda:0", "cuda:1", "cuda:2"]))
    assert awl.role == {"collect": torch.device("cuda:0"), "wm": torch.device("cuda:1"), "policy": torch.device("cuda:2")}
    al = LayoutTimer(_split("AL", ["cuda:0", "cuda:1"]))
    assert al.role == {"collect": torch.device("cuda:0"), "wm": torch.device("cuda:0"), "policy": torch.device("cuda:1")}


def test_every_20th_iteration_after_the_warm_up():
    t = LayoutTimer(_split("AWL", ["cuda:0", "cuda:1", "cuda:2"]))
    assert [it for it in range(101) if t.due(it, 5)] == [20, 40, 60, 80, 100]
    assert not t.due(20, 25) and t.due(40, 25)


def test_train_times_the_split_only_when_asked_and_only_with_a_layout():
    """ndm_gym.train's hooks (its source, since the loop needs the simulator): LAYOUT_PROFILE=1 without --layout is refused;
    the timer exists only with a layout and the variable; its marks bracket the collection, the world-model updates, the
    policy update and the exchange, in the loop's order, and the line is printed after the exchange."""
    pytest.importorskip("isaaclab")
    from ndm_gym import train

    src = inspect.getsource(train.main)
    assert 'raise ValueError("LAYOUT_PROFILE=1 times the devices of --layout AL or AWL' in src
    assert src.index("layout_timer = None") < src.index("layout_timer = LayoutTimer(split)") < src.index("algo = split.algo")
    marks = ['layout_timer.mark("c0", "collect")', "collect(n, stats", 'layout_timer.mark("c1", "collect")', 'layout_timer.mark("w1", "wm")',
             'layout_timer.mark("p0", "policy")', 'layout_timer.mark("p1", "policy")', 'layout_timer.mark("x0")', "split.exchange()",
             'layout_timer.mark("x1")', "print(layout_timer.report(it), flush=True)"]  # fmt: skip
    assert all(src.count(m) == 1 for m in marks), [m for m in marks if src.count(m) != 1]
    at = [src.index(m) for m in marks]
    assert at == sorted(at), marks
    assert len(re.findall(r"\bif lt:", src)) == 7  # every mark under the timed iteration's test, nothing else


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events")
def test_a_timed_iteration_prints_one_line_per_device():
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else ["cuda:0", "cuda:0"]
    t = LayoutTimer(_split("AL", devices), every=1)
    t.mark("start")
    for a, b, role in (("c0", "c1", "collect"), ("w0", "w1", "wm"), ("p0", "p1", "policy")):
        t.mark(a, role)
        with torch.cuda.device(t.role[role]):
            x = torch.randn(256, 256, device=t.role[role])
            x = x @ x
        t.mark(b, role)
    t.mark("x0")
    t.mark("x1")
    line = t.report(3)
    first = "collect+wm" if devices[0] != devices[1] else "collect+wm+policy"
    assert line.startswith(f"[prof-layout] it 3 | GPU0 ({first}): collect ") and " | GPU1 (" in line
    assert all(f"{k} " in line for k in ("wm", "policy", "exchange"))
    assert re.search(r"idle -?\d+\.\d\d of \d+\.\d\d ms$", line) and t.ev == {}
