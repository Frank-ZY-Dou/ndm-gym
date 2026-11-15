"""Export full control-boundary states from physical native Adroit Door replay.

Run in the separately pinned replay environment; no assets are redistributed.
The input is a trusted, explicitly hash-verified local DAPG pickle (pickle can
execute code). Hash verification establishes identity, not trustworthiness.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.motion.adroit_door import DoorReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-demo", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Expected digest of trusted demo; not a safety sandbox")
    parser.add_argument("--clip", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True, help="New .npz file outside the source tree")
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    if version("mujoco") != "3.1.6" or version("gymnasium-robotics") != "1.3.1":
        parser.error("verified replay requires mujoco==3.1.6 and gymnasium-robotics==1.3.1")
    payload = args.trusted_demo.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.sha256:
        parser.error("demonstration SHA256 mismatch; not deserializing")
    demos = pickle.loads(payload)
    if args.clip < 0 or args.clip >= len(demos):
        parser.error("clip out of range")
    demo = demos[args.clip]
    actions = np.asarray(demo["actions"])
    if actions.ndim != 2 or actions.shape[1] != 28 or not len(actions) or not np.isfinite(actions).all():
        parser.error("expected nonempty finite native actions (T, 28)")

    import mujoco
    from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv

    env = AdroitHandDoorEnv()
    try:
        env.reset(seed=0)
        env.set_env_state(demo["init_state_dict"])
        model, data = env.model, env.data
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
        if model.nq != 30 or model.nv != 30 or names[-2:] != ["door_hinge", "latch"]:
            raise ValueError("Unexpected native model joint layout")
        if any(int(j) >= 28 for j in model.actuator_trnid[:, 0]):
            raise ValueError("Only native hand joint actuation is permitted")
        q, v = [data.qpos.copy()], [data.qvel.copy()]
        penetration, limit_excess, contact_steps = 0., 0., 0
        start = time.perf_counter()
        for action in actions:
            data.ctrl[:] = env.act_mean + np.clip(action, -1, 1) * env.act_rng
            for _ in range(env.frame_skip):
                if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise ValueError("Unexpected external applied forces")
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Nonfinite physical state")
                contact_steps += int(data.ncon > 0)
                for contact in data.contact[:data.ncon]:
                    penetration = max(penetration, -float(contact.dist))
                for j in np.flatnonzero(model.jnt_limited):
                    x = data.qpos[model.jnt_qposadr[j]]
                    lo, hi = model.jnt_range[j]
                    limit_excess = max(limit_excess, float(lo - x), float(x - hi))
            q.append(data.qpos.copy())
            v.append(data.qvel.copy())
        elapsed = time.perf_counter() - start
        meta = {"kind": "adroit-door-physical-replay-v1", "clip": args.clip, "demo_sha256": digest,
                "mujoco": version("mujoco"), "gymnasium_robotics": version("gymnasium-robotics"),
                "control_dt": model.opt.timestep * env.frame_skip, "joint_names": names,
                "frame_skip": env.frame_skip, "timestep": model.opt.timestep,
                "max_penetration_m": penetration, "max_soft_limit_excess": limit_excess,
                "contact_substeps": contact_steps, "physics_wall_seconds": elapsed,
                "seconds_per_control_step": elapsed / len(actions),
                "scene_door_body_pos": np.asarray(demo["init_state_dict"]["door_body_pos"]).tolist()}
        reference = DoorReference(np.asarray(q), np.asarray(v), actions, meta)
        reference.save(args.output)
        print(json.dumps({**meta, "frames": reference.frames, "door_final_rad": float(q[-1][28]),
                          "opened": bool(reference.opened(q[-1]))}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
