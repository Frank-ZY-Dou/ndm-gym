"""Supported preparation preserves physics settings and replays real controls."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from test_adroit_door_physics import setup
from ndm_gym.sim.rigid.adroit_door import DoorPhysics

SPEC = importlib.util.spec_from_file_location('prepare_door', Path(__file__).parents[1]/'tools/prepare_adroit_door.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_only_legacy_capacity_and_noncontact_visual_change(tmp_path):
    (tmp_path/'meshes').mkdir()
    (tmp_path/'textures').mkdir()
    text = '''<mujoco><compiler meshdir="meshes" texturedir="textures"/>
    <option iterations="20" noslip_iterations="20" apirate="200"/><size njmax="100" nconmax="50" nstack="1000"/>
    <worldbody><body name="vive_tracker" mocap="true"><inertial mass="0.064" pos="0 0 0" diaginertia="1 1 1"/>
    <geom contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    path = tmp_path/'legacy.xml'
    path.write_text(text)
    root = ET.fromstring(PREPARE.prepare_xml(path))
    assert root.find('option').attrib == dict(iterations='20', noslip_iterations='20')
    tracker = root.find(".//body[@name='vive_tracker']")
    assert tracker.find('geom') is None
    assert tracker.find('inertial').get('mass') == '0.064'
    assert Path(root.find('compiler').get('meshdir')).is_absolute()
    path.write_text(text.replace('contype="0"', 'contype="1"'))
    with pytest.raises(ValueError, match='colliding'):
        PREPARE.prepare_xml(path)
    path.write_text(text.replace('mocap="true"', 'mocap="false"'))
    with pytest.raises(ValueError, match='mocap'):
        PREPARE.prepare_xml(path)


def test_replayed_arrays_equal_native_adapter_and_metadata_is_fresh():
    model, source = setup()
    source.metadata.update(mujoco='legacy', physics_wall_seconds=-999., contact_substeps=-1)
    ref = PREPARE.replay(model, source)
    sim = DoorPhysics(model, source)
    q, v = [sim.state()[0]], [sim.state()[1]]
    for action in source.actions:
        state, _ = sim.step(action)
        q.append(state[0])
        v.append(state[1])
    np.testing.assert_array_equal(ref.qpos, q)
    np.testing.assert_array_equal(ref.qvel, v)
    assert np.any(ref.qpos[1:] != source.qpos[1:])
    assert ref.metadata['physics_wall_seconds'] > 0
    assert ref.metadata['contact_substeps'] >= 0
    assert ref.metadata['source_mujoco'] == 'legacy'
