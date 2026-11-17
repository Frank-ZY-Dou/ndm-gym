"""Admission reports must reject stale capacity, budget and physical failures."""
import copy
import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / 'tools/warmup_door_accurate.py'
_SPEC = importlib.util.spec_from_file_location('door_warmup', _PATH)
warmup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(warmup)


class WarmupAdmissionTest(unittest.TestCase):
    def test_recovery_compares_physics_and_preserves_timing(self):
        original = [dict(finite=True, door_rmse_rad=0.01, wall_seconds=0.2)]
        restored = [dict(finite=True, door_rmse_rad=0.01, wall_seconds=0.3)]
        self.assertEqual(warmup.physical_results(original), warmup.physical_results(restored))
        self.assertEqual(restored[0]['wall_seconds'], 0.3)
        restored[0]['door_rmse_rad'] = 0.02
        self.assertNotEqual(warmup.physical_results(original), warmup.physical_results(restored))

    def test_reject_incomplete_or_mislabelled_admission(self):
        for method in ('supertrack', 'supermimic'):
            result = dict(optimizer_counts=dict(wm=8, policy=1),
                          protocol=dict(config='door-hand-accurate-capacity-v1', horizon=16,
                                        wm_hidden=[1024]*3, policy_hidden=[1024, 512, 256],
                                        wm_batch_size=2048, batch_size=4096, wm_window=8,
                                        rounds=1, bootstrap_updates=4, wm_updates=4, policy_updates=1,
                                        upstream_policy_optimizer=True,
                                        learning_rate=4e-4 if method == 'supertrack' else 2e-4,
                                        policy_lr=4e-4 if method == 'supertrack' else 2e-4),
                          evaluations=[dict(finite=True)], three_seed_evidence=False)
            if method == 'supermimic':
                result['optimizer_counts']['critic'] = 20
                result['protocol'].update(critic_epochs=5, critic_minibatches=4)
            warmup.verify_result(result, method)
            for section, key, value in [
                ('optimizer_counts', 'wm', 7), ('optimizer_counts', 'policy', 0),
                ('protocol', 'horizon', 4), ('protocol', 'batch_size', 256),
                ('protocol', 'upstream_policy_optimizer', False),
                ('protocol', 'learning_rate', 1e-3),
            ]:
                corrupted = copy.deepcopy(result)
                corrupted[section][key] = value
                with self.subTest(method=method, section=section, key=key), self.assertRaises(ValueError):
                    warmup.verify_result(corrupted, method)
            for key, value in [('evaluations', []), ('evaluations', [dict(finite=False)]),
                               ('three_seed_evidence', True)]:
                corrupted = copy.deepcopy(result)
                corrupted[key] = value
                with self.subTest(method=method, key=key), self.assertRaises(ValueError):
                    warmup.verify_result(corrupted, method)
