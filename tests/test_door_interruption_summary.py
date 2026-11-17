"""Interrupted progress is distinct from checkpoint recovery or final scoring."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'interruption', Path(__file__).parents[1] / 'tools/summarize_door_interruption.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        self.guard = dict(status='failed', reason='host memory pressure exceeds 1%')

    def test_interruption_preserves_progress_without_certifying_checkpoint(self):
        result = module.summarize(self.guard, [dict(completed_rounds=18,
                                  optimizer_counts=dict(wm=1272, policy=18))])
        self.assertEqual(result['completed_rounds_from_metrics'], 18)
        self.assertEqual(result['optimizer_counts_from_metrics']['wm'], 1272)
        self.assertIn('NOT PERFORMED', result['checkpoint_validation'])

    def test_interruption_before_first_round(self):
        self.assertEqual(module.summarize(self.guard, [])['completed_rounds_from_metrics'], 0)

    def test_nonterminal_or_success_is_rejected(self):
        for status in ('running', 'passed', 'completed'):
            with self.assertRaises(ValueError):
                module.summarize(dict(status=status, reason='x'), [])

    def test_duplicate_round_or_regressed_counter_is_rejected(self):
        first = dict(completed_rounds=1, optimizer_counts=dict(wm=1204, policy=1))
        for second in (first, dict(completed_rounds=2, optimizer_counts=dict(wm=1200, policy=2)),
                       dict(completed_rounds=2, optimizer_counts=dict(wm=True, policy=2)),
                       dict(completed_rounds=2, optimizer_counts=dict(wm=1208))):
            with self.assertRaises(ValueError):
                module.summarize(self.guard, [first, second])
