"""Synchronized wall timing for native Door training phases (no simulator import)."""
import time


def measured_call(function, *, synchronize=lambda: None, clock=time.perf_counter):
    """Return value and elapsed seconds after outstanding device work completes.

    A failed phase returns no successful measurement. Its work belongs to the
    external invocation/guard cost, including work lost since the last checkpoint.
    """
    synchronize()
    started = clock()
    value = function()
    synchronize()
    return value, clock() - started
