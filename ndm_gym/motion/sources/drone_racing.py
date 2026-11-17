"""Flights of Race Against the Machine (Bosello et al., RA-L 2024; github.com/tii-racing/drone-racing-dataset; the data CC BY
4.0), as tools/download_drone_racing.py converts them: per autonomous flight ``<flight>.npz`` with the drone's motion-capture
state at 275 Hz, ``t`` (n,) seconds, ``p`` (n, 3) metres and ``quat`` (n, 4) (w, x, y, z), the world's z up, and the racing
gates' poses ``gates`` (g, 12), each a centre and a rotation matrix (``load_gates``; the task does not read them).

A flight is a take-off, a hover, two laps of an ellipse or a lemniscate or three of a 3-D race track, a hover and a landing.
``load_flight`` returns the part in the air, smoothed and resampled at the control rate.
"""

from __future__ import annotations

import numpy as np
import torch

from ...core.rotations import quat_slerp

MIN_HEIGHT = 0.5  # m: the flight's part above it is the reference (the take-off and the landing are left out)
SMOOTH = 0.012  # s: the standard deviation of the Gaussian that smooths the poses (the capture is at 275 Hz)


GATE_OPENING, GATE_FRAME = 1.524, 2.1336  # m: the gates' square opening (5 ft) and outer frame (7 ft), the paper's


def load_gates(path: str) -> np.ndarray:
    """The flight's gates (g, 12): per gate its centre (3) and its rotation matrix (9, row-major), the opening across the
    matrix's first column; none (0, 12) for a file converted without them."""
    with np.load(path, allow_pickle=False) as d:
        return d["gates"].astype(np.float64) if "gates" in d else np.zeros((0, 12))


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """``x`` (n, d) filtered along time with a Gaussian of ``sigma`` samples, the ends padded with their values."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate((np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)))
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def _fill_gaps(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``t`` (n,) and ``x`` (n, d) with samples put in where the capture dropped frames: between two samples more than 1.5
    of the capture's periods apart, linearly interpolated ones a period apart. A flight without gaps is returned as it is."""
    period = float(np.median(np.diff(t)))
    gaps = np.nonzero(np.diff(t) > 1.5 * period)[0]
    if len(gaps) == 0:
        return t, x
    ts, xs, last = [], [], 0
    for i in gaps:
        k = int(round((t[i + 1] - t[i]) / period)) - 1  # the dropped frames
        w = np.arange(1, k + 1)[:, None] / (k + 1)
        ts += [t[last : i + 1], t[i] + w[:, 0] * (t[i + 1] - t[i])]
        xs += [x[last : i + 1], x[i] + w * (x[i + 1] - x[i])]
        last = i + 1
    return np.concatenate((*ts, t[last:])), np.concatenate((*xs, x[last:]))


def load_flight(path: str, dt: float, min_height: float = MIN_HEIGHT, smooth: float = SMOOTH) -> dict[str, torch.Tensor]:
    """The flight's part in the air every ``dt`` seconds: position ``p`` (T, 3) and quaternion ``quat`` (T, 4), the poses
    smoothed at the capture's rate (the frames it dropped filled in first: the filter runs over samples a period apart),
    then interpolated at the samples' own times (linearly, the quaternion spherically)."""
    with np.load(path, allow_pickle=False) as d:
        t, p, quat = d["t"].astype(np.float64), d["p"].astype(np.float64), d["quat"].astype(np.float64)
    if t.ndim != 1 or p.shape != (t.shape[0], 3) or quat.shape != (t.shape[0], 4) or not (np.diff(t) > 0).all():
        raise ValueError(f"{path}: not a converted flight (t (n,), p (n, 3), quat (n, 4), increasing times)")
    if not (np.isfinite(p).all() and np.isfinite(quat).all()):
        raise ValueError(f"{path}: non-finite poses")
    flip = np.cumprod(np.where((quat[1:] * quat[:-1]).sum(1) < 0, -1.0, 1.0))  # one hemisphere along the flight
    quat[1:] *= flip[:, None]
    t, pq = _fill_gaps(t, np.concatenate((p, quat), 1))
    p, quat = pq[:, :3], pq[:, 3:]
    sigma = smooth / float(np.median(np.diff(t)))
    p, quat = _smooth(p, sigma), _smooth(quat, sigma)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    air = np.nonzero(p[:, 2] > min_height)[0]
    if len(air) < 2:
        raise ValueError(f"{path}: the flight never rises above {min_height} m")
    x = np.arange(t[air[0]], t[air[-1]], dt)
    i0 = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
    w = torch.tensor(np.clip((x - t[i0]) / (t[i0 + 1] - t[i0]), 0.0, 1.0))
    P, Q = torch.tensor(p), torch.tensor(quat)
    pos = P[i0] + (P[i0 + 1] - P[i0]) * w[:, None]
    q = quat_slerp(Q[i0], Q[i0 + 1], w)
    return {"p": pos.float(), "quat": (q / q.norm(dim=-1, keepdim=True)).float()}
