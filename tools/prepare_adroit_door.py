"""Prepare a native Door model and physical reference for MuJoCo 3.15.0.

Consumes the legacy export and original installed assets, without redistributing
them. Replays the same controls in the supported engine; never edits a trajectory
to match its target. New metadata measures this replay rather than inheriting
legacy engine timings/contact diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndm_gym.motion.adroit_door import DoorReference
from ndm_gym.sim.rigid.adroit_door import DoorPhysics


def prepare_xml(path, *, mesh_dir=None, texture_dir=None):
    """Remove retired capacity attributes and a noncolliding mocap visual only."""
    root = ET.parse(path).getroot()
    for node in root.iter():
        for key in ('apirate', 'njmax', 'nconmax', 'nstack'):
            node.attrib.pop(key, None)
    compiler = root.find('compiler')
    if compiler is None:
        raise ValueError('Expected exported compiler settings')
    for key, override in (('meshdir', mesh_dir), ('texturedir', texture_dir)):
        directory = Path(override) if override else Path(compiler.get(key, '.'))
        if not directory.is_absolute():
            directory = Path(path).resolve().parent/directory
        if not directory.is_dir():
            raise ValueError('Missing native asset directory: '+str(directory))
        compiler.set(key, str(directory.resolve())+'/')
    tracker = root.find(".//body[@name='vive_tracker']")
    if tracker is not None:
        if tracker.get('mocap') != 'true' or tracker.find('inertial') is None:
            raise ValueError('Tracker removal requires mocap body with explicit inertia')
        for geom in tracker.findall('geom'):
            if geom.get('contype') != '0' or geom.get('conaffinity') != '0':
                raise ValueError('Refusing to remove a colliding tracker geom')
            tracker.remove(geom)
    return ET.tostring(root, encoding='unicode')


def replay(model, source):
    import mujoco
    sim = DoorPhysics(model, source)
    q, v = [sim.state()[0]], [sim.state()[1]]
    actions = source.actions.clip(-1, 1)
    contacts, penetration, limit_excess = 0, 0., 0.
    started = time.perf_counter()
    for action in actions:
        sim.data.ctrl[:] = sim.mid + action*sim.half
        for _ in range(sim.frame_skip):
            if np.any(sim.data.xfrc_applied) or np.any(sim.data.qfrc_applied):
                raise ValueError('External forces are forbidden')
            mujoco.mj_step(model, sim.data)
            if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                raise FloatingPointError('Nonfinite physical reference')
            contacts += int(sim.data.ncon > 0)
            for contact in sim.data.contact[:sim.data.ncon]:
                penetration = max(penetration, -float(contact.dist))
            for j in np.flatnonzero(model.jnt_limited):
                x = sim.data.qpos[model.jnt_qposadr[j]]
                lo, hi = model.jnt_range[j]
                limit_excess = max(limit_excess, float(lo-x), float(x-hi))
        state = sim.state()
        q.append(state[0]); v.append(state[1])
    elapsed = time.perf_counter()-started
    keys = ('kind', 'clip', 'demo_sha256', 'control_dt', 'joint_names',
            'frame_skip', 'timestep', 'scene_door_body_pos', 'gymnasium_robotics')
    meta = {key: source.metadata[key] for key in keys if key in source.metadata}
    meta.update(mujoco=mujoco.__version__, source_mujoco=source.metadata.get('mujoco'),
                contact_substeps=contacts, max_penetration_m=penetration,
                max_soft_limit_excess=limit_excess, physics_wall_seconds=elapsed,
                seconds_per_control_step=elapsed/len(actions))
    return DoorReference(np.asarray(q), np.asarray(v), actions, meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-model', 'legacy-reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mesh-dir', type=Path, help='Original asset directory when exports moved between hosts')
    p.add_argument('--texture-dir', type=Path, help='Original asset directory when exports moved between hosts')
    a = p.parse_args()
    import mujoco
    if mujoco.__version__ != '3.15.0':
        p.error('Supported study preparation requires mujoco==3.15.0')
    xml = prepare_xml(a.legacy_model, mesh_dir=a.mesh_dir, texture_dir=a.texture_dir)
    model = mujoco.MjModel.from_xml_string(xml)
    source = DoorReference.load(a.legacy_reference)
    reference = replay(model, source)
    a.output.mkdir(parents=True, exist_ok=False)
    model_path, ref_path = a.output/'door.xml', a.output/'reference.npz'
    model_path.write_text(xml)
    reference.metadata.update(derived_from=a.legacy_reference.name,
                              derived_from_sha256=hashlib.sha256(a.legacy_reference.read_bytes()).hexdigest(),
                              model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    reference.save(ref_path)
    report = dict(metadata=reference.metadata, frames=reference.frames,
                  final_door_rad=float(reference.qpos[-1, 28]),
                  legacy_model_sha256=hashlib.sha256(a.legacy_model.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                  note='Archive hashes include runtime metadata/paths; physical array equality is checked separately.')
    (a.output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
