"""Motion measures of hand-object rollouts against their reference, the same for every algorithm (the check before any hand media).

    python tools/hand_motion_check.py <label>=<evaluation dir> [...] [--cutoff_hz 5] [--limit_frac 0.01] [--json out.jsonl]

An evaluation dir is ``ndm_gym.evaluate_hand --save_targets``'s output: ``rollout.npz`` (the unperturbed rollout under the policy:
the scene's qpos, the policy's actions and position targets at every control step; the same rollout with the reference pose as
the position targets in the policy's place; the clip's kinematic reference) and ``metrics.json`` (the clip). Several dirs may
share a label (a configuration's clips and seeds): the label's line is the mean over its dirs. Measures, per rollout:

- joint acceleration and jerk: the RMS over the control steps of the second and third differences of each finger joint's position,
  over dt^2 and dt^3, then the mean over the finger joints (rad/s^2, rad/s^3); the palm's three hinges the same way, apart;
- the high-frequency share of joint velocity: per finger joint the share of the spectral power of its velocity (its mean removed)
  above ``--cutoff_hz``, then the mean over the finger joints;
- action change per step: the RMS of the step-to-step change of each finger joint's position target, then the mean (rad); for the
  policy also of its raw action (in the action's own units);
- joint-limit hits: the share of (step, finger joint) pairs within ``--limit_frac`` of the joint's range from a limit;
- fingertip slip: for each fingertip in contact with the object at two consecutive steps, the displacement of its site in the
  object's frame between them; the mean over those (mm per step);
- object slip: per hand, at the steps where at least two of its fingertips touch the object, the step-to-step displacement of
  the object in its palm's frame; the mean over those (mm per step).

With two hands the finger joints and fingertips are both hands', the palm's hinges both palms'.

Contacts come from MuJoCo on the CPU at each recorded qpos (``mj_forward``); a fingertip is the body of its site. The kinematic
reference has no contacts: it gives the joint measures, its own pose's change per step standing for the action change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ndm_gym.motion.sources.spider import load_scene  # noqa: E402
from ndm_gym.robots.hands import NUM_PALM, OBJECT_BODY, hand_joints, hand_keypoints  # noqa: E402


def joint_measures(q: np.ndarray, dt: float, cutoff_hz: float, lo: np.ndarray, hi: np.ndarray, frac: float) -> dict:
    """Acceleration, jerk, the high-frequency share of the velocity and limit hits of joint positions ``q`` (T, J)."""
    acc = np.diff(q, 2, axis=0) / dt**2
    jerk = np.diff(q, 3, axis=0) / dt**3
    v = np.diff(q, axis=0) / dt
    v = v - v.mean(0)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freq = np.fft.rfftfreq(v.shape[0], dt)
    total = power[freq > 0].sum(0)
    share = np.where(total > 0, power[freq > cutoff_hz].sum(0) / np.where(total > 0, total, 1), 0.0)
    rng = hi - lo
    near = (q - lo < frac * rng) | (hi - q < frac * rng)
    rms = lambda x: np.sqrt((x**2).mean(0))  # noqa: E731
    return {"acc": float(rms(acc).mean()), "jerk": float(rms(jerk).mean()), "hf_share": float(share.mean()), "limit_hits": float(near.mean())}


def change(x: np.ndarray) -> float:
    """The RMS over the steps of the step-to-step change of each column of ``x`` (T, J), then the mean over the columns."""
    return float(np.sqrt((np.diff(x, axis=0) ** 2).mean(0)).mean())


class Scene:
    """A clip's scene on the CPU: the hands' joints (their finger and palm-hinge columns), their fingertip and palm sites, the
    object's body and geoms."""

    def __init__(self, clip: str):
        self.model = load_scene(clip)
        self.data = mujoco.MjData(self.model)
        joints, starts = hand_joints(self.model)
        self.qadr = np.array([int(self.model.jnt_qposadr[j]) for j in joints])
        self.lo, self.hi = self.model.jnt_range[joints].T
        stops = starts[1:] + [len(joints)]
        self.fingers = np.array([c for a, b in zip(starts, stops, strict=True) for c in range(a + NUM_PALM, b)])
        self.palm_hinges = np.array([c for a in starts for c in range(a + 3, a + NUM_PALM)])
        hands = hand_keypoints(self.model)
        self.palm_sites = [h[0] for h in hands]
        self.tip_sites = [s for h in hands for s in h[1:]]
        self.tip_hand = np.array([i for i, h in enumerate(hands) for _ in h[1:]])
        self.tip_bodies = [int(self.model.site_bodyid[s]) for s in self.tip_sites]
        self.obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        if self.obj < 0:
            raise ValueError(f"the scene of {clip} has no body {OBJECT_BODY}")

    def contacts_and_frames(self, qpos: np.ndarray) -> tuple:
        """Per step: which fingertips touch the object (T, F), the fingertips' sites in the object's frame (T, F, 3) and the
        object in each palm's frame (T, H, 3)."""
        m, d = self.model, self.data
        T, F = qpos.shape[0], len(self.tip_sites)
        touch, tips_obj, obj_palm = np.zeros((T, F), bool), np.zeros((T, F, 3)), np.zeros((T, len(self.palm_sites), 3))
        for t in range(T):
            d.qpos[:] = qpos[t]
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
                for k, b in enumerate(self.tip_bodies):
                    if (b1 == b and b2 == self.obj) or (b2 == b and b1 == self.obj):
                        touch[t, k] = True
            R, p = d.xmat[self.obj].reshape(3, 3), d.xpos[self.obj]
            for k, s in enumerate(self.tip_sites):
                tips_obj[t, k] = R.T @ (d.site_xpos[s] - p)
            for h, s in enumerate(self.palm_sites):
                obj_palm[t, h] = d.site_xmat[s].reshape(3, 3).T @ (p - d.site_xpos[s])
        return touch, tips_obj, obj_palm


def slips(touch: np.ndarray, tips_obj: np.ndarray, obj_palm: np.ndarray, tip_hand: np.ndarray | None = None) -> dict:
    """Fingertip slip and object slip (mm per step) from ``Scene.contacts_and_frames`` (``obj_palm`` (T, 3) for one hand or
    (T, H, 3); ``tip_hand``: each fingertip's hand, by default the first); None where no step qualifies."""
    obj_palm = obj_palm.reshape(len(obj_palm), -1, 3)
    tip_hand = np.zeros(touch.shape[1], int) if tip_hand is None else tip_hand
    both = touch[1:] & touch[:-1]
    step = np.linalg.norm(np.diff(tips_obj, axis=0), axis=-1)
    holding = np.stack([touch[:, tip_hand == h].sum(1) >= 2 for h in range(obj_palm.shape[1])], 1)  # (T, H)
    held = holding[1:] & holding[:-1]
    ostep = np.linalg.norm(np.diff(obj_palm, axis=0), axis=-1)
    return {"fingertip_slip_mm": float(step[both].mean() * 1000) if both.any() else None,
            "object_slip_mm": float(ostep[held].mean() * 1000) if held.any() else None,
            "contact_steps": int(touch.any(1).sum()), "held_steps": int(holding.any(1).sum())}  # fmt: skip


def measure(path: str, cutoff_hz: float, frac: float, scenes: dict, clip_root: str | None = None) -> dict:
    """Every measure of one evaluation dir: the policy's rollout, the reference-targets rollout and the kinematic reference.
    ``clip_root``: the clips' directory here, in the place of the evaluation's own up to ``processed/`` (an evaluation made on
    another machine)."""
    meta = json.load(open(os.path.join(path, "metrics.json")))
    r = np.load(os.path.join(path, "rollout.npz"))
    for k in ("qpos", "targets", "actions", "reference_targets", "reference_targets_qpos", "reference_q"):
        if k not in r:
            raise SystemExit(f"{path}/rollout.npz has no {k}: evaluate with ndm_gym.evaluate_hand --save_targets")
    clip = meta["clip"]
    if clip_root:
        clip = os.path.join(clip_root, clip[clip.index("processed/") :])
    sc = scenes.setdefault(clip, Scene(clip))
    dt = float(r["control_dt"])
    fi, ph = sc.fingers, sc.palm_hinges
    fl, fh = sc.lo[fi], sc.hi[fi]
    out = {"dir": path, "clip": clip, "iteration": meta.get("iteration"), "frames": int(r["qpos"].shape[0])}
    for name, qp, tg in (("policy", r["qpos"], r["targets"]), ("reference_targets", r["reference_targets_qpos"], r["reference_targets"])):
        q = qp[:, sc.qadr]
        m = joint_measures(q[:, fi], dt, cutoff_hz, fl, fh, frac)
        palm = joint_measures(q[:, ph], dt, cutoff_hz, sc.lo[ph], sc.hi[ph], frac)
        m.update({"palm_acc": palm["acc"], "palm_jerk": palm["jerk"], "target_change": change(tg[:, fi])})
        m.update(slips(*sc.contacts_and_frames(qp), sc.tip_hand))
        out[name] = m
    out["policy"]["action_change"] = change(r["actions"][:, fi])
    kq = r["reference_q"]
    k = joint_measures(kq[:, fi], dt, cutoff_hz, fl, fh, frac)
    k.update({"target_change": change(kq[:, fi])})
    out["reference_kinematic"] = k
    return out


KEYS = ("acc", "jerk", "hf_share", "target_change", "limit_hits", "fingertip_slip_mm", "object_slip_mm")


def mean_of(rows: list, part: str) -> dict:
    out = {}
    for k in KEYS + ("action_change", "palm_acc", "palm_jerk"):
        xs = [r[part][k] for r in rows if part in r and r[part].get(k) is not None]
        out[k] = float(np.mean(xs)) if xs else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="<label>=<evaluation dir>")
    p.add_argument("--cutoff_hz", type=float, default=5.0)
    p.add_argument("--limit_frac", type=float, default=0.01)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--clip_root", type=str, default=None, help="the clips' directory here (evaluations made on another machine)")
    a = p.parse_args()
    scenes, by = {}, {}
    for item in a.dirs:
        label, _, path = item.partition("=")
        if not path:
            sys.exit(f"{item}: give <label>=<evaluation dir>")
        by.setdefault(label, []).append(measure(path, a.cutoff_hz, a.limit_frac, scenes, a.clip_root))
    cell = lambda x, f: "-" if x is None else format(x, f)  # noqa: E731
    fmt = {
        "acc": ".1f",
        "jerk": ".0f",
        "hf_share": ".3f",
        "target_change": ".4f",
        "limit_hits": ".4f",
        "fingertip_slip_mm": ".2f",
        "object_slip_mm": ".2f",
    }
    print(f"finger joints, mean over each label's rollouts (cutoff {a.cutoff_hz:g} Hz, limits within {a.limit_frac:g} of the range)")
    print(f"{'label':28s} {'rollout':19s} " + " ".join(f"{k:>17s}" for k in KEYS))
    out = []
    for label, rows in by.items():
        for part in ("policy", "reference_targets", "reference_kinematic"):
            m = mean_of(rows, part)
            print(f"{label:28s} {part:19s} " + " ".join(f"{cell(m[k], fmt[k]):>17s}" for k in KEYS))
            out.append({"label": label, "rollout": part, "n": len(rows), **m})
    if a.json:
        with open(a.json, "w") as f:
            for label, rows in by.items():
                for r in rows:
                    f.write(json.dumps({"label": label, **r}) + "\n")
    for q in out:
        print(json.dumps(q))


if __name__ == "__main__":
    main()
