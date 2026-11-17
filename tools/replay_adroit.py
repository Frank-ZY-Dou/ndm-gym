"""Export all hash-verified native Adroit demonstrations with physical success.

Run in MuJoCo 3.1.6 / Gymnasium-Robotics 1.3.1. Pickle inputs must already be
trusted: checking their digest establishes identity, not safety. Never publishes
assets/data. Selection for training happens after supported-engine replay.
"""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.motion.adroit import AdroitReference, DIMS
from ndm_gym.sim.rigid.adroit import scene_identity, success


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=tuple(DIMS), required=True)
    p.add_argument('--trusted-demo', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if version('mujoco') != '3.1.6' or version('gymnasium-robotics') != '1.3.1':
        p.error('Export requires the pinned legacy replay environment')
    payload = a.trusted_demo.read_bytes()
    if hashlib.sha256(payload).hexdigest() != a.sha256:
        p.error('Demo digest mismatch; not deserializing')
    demos = pickle.loads(payload)
    import mujoco
    module = importlib.import_module('gymnasium_robotics.envs.adroit_hand.adroit_'+a.task)
    cls = getattr(module, 'AdroitHand'+a.task.capitalize()+'Env')
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []
    env = cls()
    try:
        for i, demo in enumerate(demos):
            env.reset(seed=0)
            # DAPG includes diagnostic fields no longer accepted by Gymnasium's
            # strict Dict space. Retain every state field the native setter reads.
            state = {k: demo['init_state_dict'][k] for k in env._state_space.spaces}
            env.set_env_state(state)
            m, d = env.model, env.data
            path = a.output/f'clip{i}.xml'
            mujoco.mj_saveLastXML(str(path), m)
            tree = ET.parse(path)
            compiler = tree.getroot().find('compiler')
            for key in ('meshdir', 'texturedir'):
                directory = Path(compiler.get(key, '.'))
                if not directory.is_absolute():
                    directory = Path(env.fullpath).resolve().parent/directory
                compiler.set(key, str(directory.resolve())+'/')
            path.write_text(ET.tostring(tree.getroot(), encoding='unicode'))
            q, v, achieved = [d.qpos.copy()], [d.qvel.copy()], []
            actions = np.asarray(demo['actions'])
            for action in actions:
                _, _, _, _, info = env.step(action)
                measured = success(m, d, a.task)
                if measured != bool(info['success']):
                    raise ValueError('Independent success predicate differs from original environment')
                achieved.append(measured)
                q.append(d.qpos.copy()); v.append(d.qvel.copy())
            meta = dict(kind='adroit-suite-physical-replay-v1', task=a.task, clip=i,
                        demo_sha256=a.sha256, mujoco=version('mujoco'), gymnasium_robotics=version('gymnasium-robotics'),
                        joint_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)],
                        joint_types=m.jnt_type.tolist(), joint_periodic=((m.jnt_type == 3)&~m.jnt_limited.astype(bool)).tolist(),
                        legacy_extra_state_fields=sorted(set(demo['init_state_dict'])-set(state)),
                        scene=scene_identity(m), frame_skip=env.frame_skip, control_dt=m.opt.timestep*env.frame_skip)
            AdroitReference(np.asarray(q), np.asarray(v), actions, meta).save(a.output/f'clip{i}.npz')
            reports.append(dict(clip=i, steps=len(actions), success_final=achieved[-1], success_steps=sum(achieved),
                                success_last10=sum(achieved[-10:]), finite=bool(np.isfinite(q).all() and np.isfinite(v).all())))
    finally:
        env.close()
    (a.output/'replay.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(dict(task=a.task, clips=len(reports), successful=sum(x['success_final'] for x in reports))))


if __name__ == '__main__':
    main()
