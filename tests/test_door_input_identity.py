"""Simulator-free checkpoint physical-input identity contract."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'door_identity', Path(__file__).parents[1] / 'ndm_gym/motion/door_identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model, self.reference = self.root / 'model.xml', self.root / 'reference.npz'
        self.model.write_bytes(b'native Adroit XML')
        self.reference.write_bytes(b'physical reference')
        self.protocol = identity.input_identity(self.model, self.reference)

    def test_identical_relocated_inputs_accepted(self):
        relocated = self.root / 'relocated.xml'
        relocated.write_bytes(self.model.read_bytes())
        self.assertEqual(identity.verify_input_identity(self.protocol, relocated, self.reference), self.protocol)

    def test_changed_model_and_reference_refused(self):
        for path in (self.model, self.reference):
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                identity.verify_input_identity(self.protocol, self.model, self.reference)
            path.write_bytes(original)

    def test_historical_missing_digest_not_silently_certified(self):
        for key in self.protocol:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Checkpoint lacks'):
                identity.verify_input_identity({k: v for k, v in self.protocol.items() if k != key},
                                               self.model, self.reference)

    def test_missing_file_not_accepted_from_saved_digest(self):
        self.reference.unlink()
        with self.assertRaises(FileNotFoundError):
            identity.verify_input_identity(self.protocol, self.model, self.reference)
