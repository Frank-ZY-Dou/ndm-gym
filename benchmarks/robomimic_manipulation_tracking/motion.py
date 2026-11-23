"""How a policy's rollout moves beside its demonstration: the same measures for both algorithms and for the demonstration.

    python benchmarks/robomimic_manipulation_tracking/motion.py <name>=<rollout_0.npz> [...] [--data_dir data/robomimic] [--json out.jsonl]

A rollout is ``evaluate_robomimic``'s ``rollout_<demo>.npz``: the scene's state at every 50 ms control step from the
unperturbed start, beside the demonstration's qpos. Its frames up to a termination are read (all of them if it does not
terminate), and beside them the demonstration's same frames. Several rollouts may share a name (a configuration's seeds, in
the order given). Per rollout and for the demonstration:

- acceleration and jerk: the RMS over the steps of the second and third differences of each arm joint's position, over dt^2
  and dt^3, then the mean over the arms' joints (rad/s^2, rad/s^3);
- high-frequency share: per arm joint the share of its velocity's spectral power above 5 Hz (its mean removed; samples at
  20 Hz hold up to 10 Hz), then the mean over the joints;
- limit hits: the share of step and joint pairs within 1 % of the joint's range from a limit, for the arms' joints and,
  apart, for the grippers' fingers (an open gripper rests on its limit);
- slip: over successive steps at which both fingers of one gripper touch a tracked object (MuJoCo's contacts at each
  recorded qpos; a finger is its body with the bodies below it, its pad), that object's displacement in the gripper's
  end-effector frame between the two steps; the mean over those pairs of steps (mm per step). A held object's wobble counts
  as much as its sliding.

A rollout of fewer than 8 frames has no measures. Prints a table: per task and name each rollout's values, then the
demonstration's over all its frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUTOFF_HZ, LIMIT_FRAC, MIN_FRAMES = 5.0, 0.01, 8


def joint_measures(q: np.ndarray, dt: float, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Acceleration, jerk, the high-frequency share of the velocity and the limit hits of joint positions ``q`` (T, J)."""
    rms = lambda x: np.sqrt((x**2).mean(0))  # noqa: E731
    v = np.diff(q, axis=0) / dt
    v = v - v.mean(0)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freq = np.fft.rfftfreq(v.shape[0], dt)
    total = power[freq > 0].sum(0)
    share = np.where(total > 0, power[freq > CUTOFF_HZ].sum(0) / np.where(total > 0, total, 1), 0.0)
    near = (q - lo < LIMIT_FRAC * (hi - lo)) | (hi - q < LIMIT_FRAC * (hi - lo))
    return {"acc": float(rms(np.diff(q, 2, axis=0) / dt**2).mean()), "jerk": float(rms(np.diff(q, 3, axis=0) / dt**3).mean()),
            "hf_share": float(share.mean()), "limit_hits": float(near.mean())}  # fmt: skip


def below(m: mujoco.MjModel, root: int) -> np.ndarray:
    """Which bodies are ``root`` or below it (nbody,)."""
    out = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        k = b
        while k != 0 and k != root:
            k = int(m.body_parentid[k])
        out[b] = k == root
    return out


class Scene:
    """A task's scene on the CPU: the arms' and the fingers' joints, each gripper's end effector and fingers, the tracked
    objects' bodies."""

    def __init__(self, model: mujoco.MjModel, task: str):
        from ndm_gym.robots.robomimic_scenes import TASKS, RobomimicScene

        self.model, self.data = model, mujoco.MjData(model)
        sc = RobomimicScene.from_model(model, task)
        arm = np.array(sc.arm)
        q, lo, hi = np.array(sc.act_q), np.array(sc.act_lo), np.array(sc.act_hi)
        self.arm_q, self.arm_lo, self.arm_hi = q[arm], lo[arm], hi[arm]
        self.fin_q, self.fin_lo, self.fin_hi = q[~arm], lo[~arm], hi[~arm]
        bodies = np.array(sc.keypoint_bodies).reshape(-1, 3)  # per gripper: its end effector and its two fingers
        self.eef = [int(b) for b in bodies[:, 0]]
        self.finger = -np.ones(model.nbody, int)  # each body's finger (2 * gripper + side), -1: none
        for g, (_, left, right) in enumerate(bodies):
            for side, root in enumerate((left, right)):
                self.finger[below(model, int(root))] = 2 * g + side
        self.objects = [int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in TASKS[task]["objects"]]
        self.object = -np.ones(model.nbody, int)  # each body's tracked object, -1: none
        for k, root in enumerate(self.objects):
            self.object[below(model, root)] = k
        self.cache = self.whole = None  # the demonstration's grasps and its measures over all its frames

    def grasps(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per step: whether both fingers of gripper g touch tracked object k (T, G, K), and k's position in g's end-effector
        frame (T, G, K, 3)."""
        m, d = self.model, self.data
        T, G, K = qpos.shape[0], len(self.eef), len(self.objects)
        hold, rel = np.zeros((T, G, K), bool), np.zeros((T, G, K, 3))
        for t in range(T):
            d.qpos[:] = qpos[t]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            touch = np.zeros((G, 2, K), bool)
            for i in range(d.ncon):
                b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
                for x, y in ((b1, b2), (b2, b1)):
                    if self.finger[x] >= 0 and self.object[y] >= 0:
                        touch[self.finger[x] // 2, self.finger[x] % 2, self.object[y]] = True
            hold[t] = touch.all(1)
            for g, e in enumerate(self.eef):
                R, p = d.xmat[e].reshape(3, 3), d.xpos[e]
                for k, b in enumerate(self.objects):
                    rel[t, g, k] = R.T @ (d.xpos[b] - p)
        return hold, rel


def slip(hold: np.ndarray, rel: np.ndarray) -> tuple[float | None, int]:
    """The mean step-to-step displacement (mm) of a held object in its gripper's frame, and the pairs of steps it is over."""
    both = hold[1:] & hold[:-1]
    step = np.linalg.norm(np.diff(rel, axis=0), axis=-1)
    n = int(both.sum())
    return (float(step[both].mean() * 1000) if n else None), n


def measures(sc: Scene, qpos: np.ndarray, dt: float, grasps: tuple) -> dict:
    out = joint_measures(qpos[:, sc.arm_q], dt, sc.arm_lo, sc.arm_hi)
    out["finger_limit_hits"] = joint_measures(qpos[:, sc.fin_q], dt, sc.fin_lo, sc.fin_hi)["limit_hits"]
    out["slip_mm"], out["held_pairs"] = slip(*grasps)
    return out


def scene_of(data_dir: str, task: str, demo: int) -> Scene:
    """The scene of a task's demonstration, from tools/download_robomimic.py's output."""
    from ndm_gym.motion.sources.robomimic import demo_path, load_demo
    from ndm_gym.robots.robomimic_scenes import load_scene

    xml = load_demo(demo_path(data_dir, task), demo)["xml"]
    return Scene(load_scene(xml, os.path.abspath(os.path.join(data_dir, "robosuite_assets"))), task)


def measure(name: str, path: str, data_dir: str, scenes: dict) -> dict:
    """One rollout's measures (``policy``), the demonstration's over the same frames (``demonstration``) and over all its
    frames (``demonstration_whole``)."""
    with np.load(path) as r:
        nq, dt, task, demo, term = int(r["nq"]), float(r["control_dt"]), str(r["task"]), int(r["demo"]), int(r["terminated_at"])
        state, ref, success = r["state"][:, :nq], r["reference_qpos"], bool(r["success"])
    if state.shape != ref.shape:
        raise SystemExit(f"{path}: the rollout's states {state.shape}, the demonstration's {ref.shape}")
    if (task, demo) not in scenes:
        scenes[(task, demo)] = scene_of(data_dir, task, demo)
    sc = scenes[(task, demo)]
    if sc.model.nq != nq:
        raise SystemExit(f"{path}: {nq} coordinates, the scene of {task} demonstration {demo} has {sc.model.nq}")
    if sc.cache is None:
        sc.cache = sc.grasps(ref)
        sc.whole = measures(sc, ref, dt, sc.cache)
    T = len(state) if term < 0 else term + 1
    out = {"name": name, "path": path, "task": task, "demo": demo, "frames": T, "of": len(state), "terminated_at": term, "success": success,
           "policy": None, "demonstration": None, "demonstration_whole": sc.whole}  # fmt: skip
    if T >= MIN_FRAMES:
        out["policy"] = measures(sc, state[:T], dt, sc.grasps(state[:T]))
        out["demonstration"] = measures(sc, ref[:T], dt, tuple(x[:T] for x in sc.cache))
    return out


def table(rows: list) -> str:
    """Per task and name each rollout's values in the order given, then the demonstration's over all its frames."""
    cols = (("acc", "acceleration (rad/s²)", "{:.2f}", 1), ("jerk", "jerk (rad/s³)", "{:.1f}", 1), ("hf_share", "high-frequency share", "{:.3f}", 1),
            ("slip_mm", "slip (mm per step)", "{:.2f}", 1), ("limit_hits", "arms' limit hits (%)", "{:.2f}", 100),
            ("finger_limit_hits", "fingers' limit hits (%)", "{:.1f}", 100))  # fmt: skip
    show = lambda v, f, s: "-" if v is None else f.format(s * v)  # noqa: E731
    lines = ["| task | | frames | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (3 + len(cols))]
    for task in dict.fromkeys(r["task"] for r in rows):
        for name in dict.fromkeys(r["name"] for r in rows if r["task"] == task):
            rs = [r for r in rows if r["task"] == task and r["name"] == name]
            frames = " / ".join(str(r["frames"]) if r["terminated_at"] < 0 else f"{r['frames']} of {r['of']}" for r in rs)
            cells = [" / ".join(show(None if r["policy"] is None else r["policy"][k], f, s) for r in rs) for k, _, f, s in cols]
            lines.append(f"| {task} | {name} | {frames} | " + " | ".join(cells) + " |")
        r = next(r for r in rows if r["task"] == task)
        whole = " | ".join(show(r["demonstration_whole"][k], f, s) for k, _, f, s in cols)
        lines.append(f"| {task} | the demonstration | {r['of']} | {whole} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollouts", nargs="+", help="<name>=<rollout_0.npz>; rollouts of one name are listed together")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "robomimic"), help="tools/download_robomimic.py's output")
    ap.add_argument("--json", default=None, help="one JSON line per rollout")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    rows, scenes = [], {}
    for item in a.rollouts:
        name, sep, path = item.partition("=")
        if not sep or not name:
            ap.error(f"{item!r}: <name>=<rollout_0.npz>")
        rows.append(measure(name, path, a.data_dir, scenes))
    if a.json:
        with open(a.json, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(table(rows))


if __name__ == "__main__":
    main()
