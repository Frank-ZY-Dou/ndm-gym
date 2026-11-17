"""Replay exported native Adroit controls on the pinned supported engine.

Select the first physically successful clip under the preregistered rule. All
candidate outcomes are recorded; policy performance cannot influence selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_adroit_door import prepare_xml
from ndm_gym.motion.adroit import AdroitReference
from ndm_gym.sim.rigid.adroit import AdroitPhysics, scene_identity, success


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-dir', 'mesh-dir', 'texture-dir', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    import mujoco
    if mujoco.__version__ != '3.15.0':
        p.error('Preparation requires MuJoCo 3.15.0')
    a.output.mkdir(parents=True, exist_ok=False)
    reports, selected = [], None
    paths = sorted(a.legacy_dir.glob('clip*.npz'), key=lambda p: int(p.stem[4:]))
    if len(paths) != 25:
        raise ValueError('Expected all 25 demonstrations')
    for path in paths:
        source = AdroitReference.load(path)
        xml = prepare_xml(path.with_suffix('.xml'), mesh_dir=a.mesh_dir, texture_dir=a.texture_dir)
        model = mujoco.MjModel.from_xml_string(xml)
        meta = dict(source.metadata, mujoco=mujoco.__version__, scene=scene_identity(model),
                    legacy_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reference = AdroitReference(source.qpos, source.qvel, source.actions, meta)
        sim = AdroitPhysics(model, reference)
        q, v, successes = [sim.state()[0]], [sim.state()[1]], []
        for action in reference.actions:
            (next_q, next_v), _ = sim.step(action)
            q.append(next_q); v.append(next_v)
            successes.append(success(model, sim.data, meta['task']))
        good = successes[-1] and sum(successes[-10:]) >= 5
        report = dict(clip=meta['clip'], steps=len(successes), success_final=successes[-1],
                      success_steps=sum(successes), success_last10=sum(successes[-10:]), eligible=good,
                      finite=bool(np.isfinite(q).all() and np.isfinite(v).all()),
                      max_legacy_qpos_difference=float(np.max(np.abs(np.asarray(q)-source.qpos))))
        reports.append(report)
        if selected is None and good and report['finite']:
            selected = meta['clip']
            modelpath = a.output/'model.xml'
            modelpath.write_text(xml)
            meta['model_sha256'] = hashlib.sha256(modelpath.read_bytes()).hexdigest()
            AdroitReference(np.asarray(q), np.asarray(v), source.actions, meta).save(a.output/'reference.npz')
        print(json.dumps(report), flush=True)
    result = dict(selected_clip=selected, candidates=reports, criterion='first eligible clip; final success and >=5 of last10')
    (a.output/'preparation.json').write_text(json.dumps(result, indent=2)+'\n')
    if selected is None:
        raise RuntimeError('No physically successful reference; training is prohibited')


if __name__ == '__main__':
    main()
