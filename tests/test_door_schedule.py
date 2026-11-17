"""Simulator-free budget and phase boundaries, checked against Hand source."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('door_schedule', ROOT / 'ndm_gym/door_schedule.py')
schedule = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = schedule
SPEC.loader.exec_module(schedule)


class FullScheduleTest(unittest.TestCase):
    def test_full_budget(self):
        s = schedule.FullHandSchedule()
        self.assertEqual(s.counts(20000, method='supermimic'), dict(
            training_transitions=40960000, warmup_transitions=65536,
            wm_optimizer_steps=80000, policy_updates=19700,
            critic_optimizer_steps=394000))
        self.assertEqual(s.counts(20000, method='supertrack')['critic_optimizer_steps'], 0)

    def test_boundary_and_recovery(self):
        s = schedule.FullHandSchedule()
        self.assertFalse(s.policy_due(300))
        self.assertTrue(s.policy_due(301))
        for boundary in (0, 300, 301, 20000):
            counts = s.counts(boundary, method='supermimic')
            s.assert_counts(counts, boundary, method='supermimic')
            counts['wm_optimizer_steps'] += 1200
            with self.assertRaises(ValueError):
                s.assert_counts(counts, boundary, method='supermimic')

    def test_invalid(self):
        for kwargs in ({'worlds': 0}, {'iterations': 1024, 'wm_only_iters': 2000},
                       {'wm_batches': True}):
            with self.assertRaises(ValueError):
                schedule.FullHandSchedule(**kwargs)
        with self.assertRaises(ValueError):
            schedule.FullHandSchedule().policy_due(0)
