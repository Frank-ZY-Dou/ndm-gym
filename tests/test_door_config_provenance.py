"""Prevent silent configuration/provenance drift without a learner runtime."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    spec = importlib.util.spec_from_file_location('verify_mapping', ROOT / 'tools/verify_door_config_mapping.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
finally:
    sys.path.pop(0)


class ProvenanceTest(unittest.TestCase):
    def test_current_full_mapping(self):
        result = verifier.verify(ROOT)
        self.assertEqual(result['status'], 'PASS')
        for counts in result['parameters'].values():
            self.assertEqual(sum(counts.values()), 101)

    def test_reject_stale_or_incomplete_mapping(self):
        audit_dir = Path('benchmarks/adroit_door/config_audit')
        pinned = json.loads((ROOT / audit_dir / 'upstream-69b818e.json').read_text())
        original = json.loads((ROOT / audit_dir / 'door-mapping3017.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in [audit_dir / 'upstream-69b818e.json', Path('ndm_gym/door_configs.py'),
                             Path('ndm_gym/train_door_online.py')]:
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, root / relative)
            target = root / audit_dir / 'door-mapping3017.json'
            for corruption in ('missing', 'upstream', 'resolved', 'option', 'reason', 'identity'):
                row = json.loads(json.dumps(original))
                method = row['methods']['supertrack']
                if corruption == 'missing':
                    del method['parameters']['horizon']
                elif corruption == 'upstream':
                    method['parameters']['horizon']['upstream'] = 4
                elif corruption == 'resolved':
                    method['door_resolved']['horizon'] = 4
                elif corruption == 'option':
                    method['parameters']['horizon']['door_value'] = 4
                elif corruption == 'reason':
                    method['parameters']['horizon']['reason'] = ''
                else:
                    row['upstream_commit'] = 'stale'
                target.write_text(json.dumps(row))
                with self.subTest(corruption=corruption), patch.object(verifier, 'audit', return_value=pinned):
                    with self.assertRaises(ValueError):
                        verifier.verify(root)


if __name__ == '__main__':
    unittest.main()
