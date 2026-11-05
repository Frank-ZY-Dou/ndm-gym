# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/env.py) under its MIT License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Per-bin reset statistics of ``TrackingSim``: failure-focused resets (``--fail_reset``, InstantTrack), the averaging of a
step's episodes that share a bin (``--bin_fix``, BlinkMimic) and the mixing in of the evaluation-guided focus
(``--eprobe_share``, BlinkMimic; algos/eprobe.py)."""

from __future__ import annotations

import torch

from ...motion.library import MotionLib


def bin_ema(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, old: float) -> torch.Tensor:
    """One EMA step (weight ``old`` on the old value) of the per-bin statistic ``stat`` for every bin among ``bins``,
    toward the mean of that bin's samples ``x``. ``index_put_`` with repeated bins keeps one arbitrary sample instead."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x)
    cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


def bin_ema_w(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, wt: torch.Tensor, old: float) -> torch.Tensor:
    """``bin_ema`` over the samples whose weight ``wt`` is 1 (the others 0): fixed size, no host-side selection
    (train.py --reset_masked)."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


def previous_bins(motion: MotionLib) -> torch.Tensor:
    """Each bin's previous bin in its clip; a clip's first bin maps to itself."""
    prev = torch.arange(motion.num_bins, device=motion.bin_clip.device) - 1
    return torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)


def mix_failures(probs: torch.Tensor, bin_fail: torch.Tensor, bin_prev: torch.Tensor, share: float) -> torch.Tensor:
    """The reset distribution ``probs`` over bins with the part ``share`` moved to the bins before recent failure points:
    each bin's termination rate ``bin_fail`` goes to its previous bin (``previous_bins``). Unchanged while no bin has one."""
    fail = torch.zeros_like(bin_fail).index_add_(0, bin_prev, bin_fail)
    fail = fail / fail.sum().clamp(min=1e-6)
    return (1 - share) * probs + share * torch.where(fail.sum() > 0, fail, probs)


def mix_focus(probs: torch.Tensor, focus_q: torch.Tensor, share: float) -> torch.Tensor:
    """The reset distribution ``probs`` over bins with the part ``share`` drawn from ``focus_q`` instead (``--eprobe_share``)."""
    return (1 - share) * probs + share * focus_q
