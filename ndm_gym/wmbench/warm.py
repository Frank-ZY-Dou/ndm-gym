"""Warm-up of a GPU timing by wall time. An A100 that sat idle for a few seconds (a compile, a data load) steps its clock
down (from 1410 to 1095 MHz, to 210 when idle longer) and runs its next 100 to 200 ms of work 28 % slower; a warm-up counted
in calls is over before that when a call takes a fraction of a millisecond, and the timed loop then measures the lower
clock in part. ``keep_busy`` repeats the call until a time has passed, so that the
timed loop starts at the workload's own clock."""

from __future__ import annotations

import time


def keep_busy(fn, sync, seconds: float) -> int:
    """Call ``fn`` until ``seconds`` have passed, syncing the device every 16 calls (so that the time is the device's) and
    at the end; returns the calls made (none for seconds <= 0)."""
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
        if n % 16 == 0:
            sync()
    if n:
        sync()
    return n
