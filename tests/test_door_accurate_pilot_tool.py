"""Simulator-free orchestration failure and recovery admission checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1] / 'tools/run_door_accurate_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def evaluations(self):
        return [dict(ARRz_offset_rad=offset, finite=True, opened_final=False,
                     opened_ever=False, translation_rmse_m=.01, hand_rmse_rad=.2,
                     door_rmse_rad=.3, latch_rmse_rad=.4) for offset in pilot.OFFSETS]

    def test_negative_opening_is_valid_evidence(self):
        pilot.validate_evaluations(self.evaluations())

    def completed(self, method):
        path = Path(self.tmp.name) / (method + '-result.json')
        counts = dict(wm=1328, policy=32)
        if method == 'supermimic':
            counts['critic'] = 640
        path.write_text(json.dumps(dict(protocol=pilot.expected_protocol(self.args, method),
                                       optimizer_counts=counts, evaluations=self.evaluations())))
        setattr(self.args, method + '_completed', path)
        return path

    def test_completed_first_method_is_not_retrained_on_second_failure(self):
        path = self.completed('supertrack')
        original = path.read_bytes()
        with patch.object(pilot.subprocess, 'run', side_effect=subprocess.CalledProcessError(9, 'learner')) as call:
            with self.assertRaises(subprocess.CalledProcessError):
                pilot.run(self.args)
        self.assertEqual(call.call_count, 1)
        cmd = call.call_args.args[0]
        self.assertEqual(cmd[cmd.index('--method') + 1], 'supermimic')
        report = json.loads((self.args.output / 'stages.json').read_text())
        self.assertEqual([s['status'] for s in report['stages']], ['reused_completed', 'failed'])
        self.assertEqual(report['stages'][0]['invocation_wall_seconds'], 0.)
        self.assertEqual(report['stages'][0]['result_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_completed_result_refused_before_any_launch(self):
        path = self.completed('supertrack')
        result = json.loads(path.read_text())
        result['optimizer_counts']['policy'] = 31
        path.write_text(json.dumps(result))
        with patch.object(pilot.subprocess, 'run') as call:
            with self.assertRaisesRegex(ValueError, 'optimizer budget'):
                pilot.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_completed_result_and_resume_are_mutually_exclusive(self):
        self.checkpoint()
        self.completed('supertrack')
        with self.assertRaisesRegex(ValueError, 'not both'):
            pilot.run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_both_completed_results_require_no_training(self):
        for method in pilot.METHODS:
            self.completed(method)
        with patch.object(pilot.subprocess, 'run') as call:
            report = pilot.run(self.args)
        call.assert_not_called()
        self.assertEqual(report['status'], 'completed')
        self.assertTrue(all(s['status'] == 'reused_completed' for s in report['stages']))

    def test_missing_reordered_or_duplicate_physical_starts_refused(self):
        rows = self.evaluations()
        for invalid in (rows[:-1], rows[::-1], [rows[0]]*5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pilot.validate_evaluations(invalid)

    def test_finite_flag_does_not_hide_invalid_metrics(self):
        for value in (float('nan'), float('inf'), -.1, True, None):
            rows = self.evaluations()
            rows[0]['door_rmse_rad'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.validate_evaluations(rows)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.args = argparse.Namespace(output=root / 'out', device='cuda:0', seed=3025,
                                       supertrack_resume=None, supermimic_resume=None)
        for name in ('model', 'reference', 'bootstrap'):
            value = root / name
            value.write_text(name)
            setattr(self.args, name, value)

    def checkpoint(self):
        folder = Path(self.tmp.name) / 'checkpoint'
        folder.mkdir()
        checkpoint = folder / 'latest.pt'
        checkpoint.write_bytes(b'opaque; learner validates payload')
        protocol = dict(method='supertrack', seed=3025, config='door-hand-accurate-capacity-v1',
                        rounds=32, bootstrap_updates=1200, wm_updates=4, policy_updates=1,
                        horizon=16, batch_size=4096, wm_batch_size=2048, wm_window=8,
                        wm_hidden=[1024]*3, policy_hidden=[1024, 512, 256], device='cuda:0')
        for name in ('model', 'reference', 'bootstrap'):
            protocol[name + '_sha256'] = hashlib.sha256(getattr(self.args, name).read_bytes()).hexdigest()
        (folder / 'protocol.json').write_text(json.dumps(protocol))
        self.args.supertrack_resume = checkpoint
        return protocol

    def test_checkpoint_input_identity_and_budget(self):
        protocol = self.checkpoint()
        pilot.validate_resume(self.args, 'supertrack')
        protocol['rounds'] = 16
        (self.args.supertrack_resume.parent / 'protocol.json').write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, 'not this pilot'):
            pilot.run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_changed_physical_input_refused(self):
        self.checkpoint()
        self.args.model.write_text('different physics')
        with self.assertRaisesRegex(ValueError, 'not this pilot'):
            pilot.run(self.args)

    def test_command_preserves_seed_capacity_and_resume(self):
        self.checkpoint()
        cmd = pilot.command(self.args, 'supertrack', self.args.output)
        self.assertEqual(cmd[cmd.index('--seed') + 1], '3025')
        self.assertEqual(cmd[cmd.index('--config') + 1], 'door-hand-accurate-capacity-v1')
        self.assertEqual(cmd[-2:], ['--resume', str(self.args.supertrack_resume)])

    def test_rejects_boolean_physical_offset_equal_to_zero(self):
        rows = self.evaluations()
        rows[0]['ARRz_offset_rad'] = False
        with self.assertRaisesRegex(ValueError, 'offsets/order'):
            pilot.validate_evaluations(rows)

    def test_failed_stage_never_launches_second_or_retries(self):
        with patch.object(pilot.subprocess, 'run', side_effect=subprocess.CalledProcessError(9, 'learner')) as call:
            with self.assertRaises(subprocess.CalledProcessError):
                pilot.run(self.args)
        self.assertEqual(call.call_count, 1)
        report = json.loads((self.args.output / 'stages.json').read_text())
        self.assertEqual(report['stages'][0]['status'], 'failed')
        self.assertGreater(report['stages'][0]['invocation_wall_seconds'], 0)
        self.assertEqual(len(report['stages']), 1)

    def test_first_completed_stage_survives_second_failure(self):
        def launch(cmd, **kwargs):
            if cmd[cmd.index('--method') + 1] == 'supermimic':
                raise subprocess.CalledProcessError(1, cmd)
            out = Path(cmd[cmd.index('--output') + 1])
            out.mkdir()
            (out / 'result.json').write_text(json.dumps(dict(
                protocol=pilot.expected_protocol(self.args, 'supertrack'),
                optimizer_counts=dict(wm=1328, policy=32), evaluations=self.evaluations())))
        with patch.object(pilot.subprocess, 'run', side_effect=launch):
            with self.assertRaises(subprocess.CalledProcessError):
                pilot.run(self.args)
        report = json.loads((self.args.output / 'stages.json').read_text())
        self.assertEqual([s['status'] for s in report['stages']], ['completed', 'failed'])

    def test_wrong_result_identity_stops_before_second_method(self):
        for field, value in (('method', 'supermimic'), ('seed', 999),
                             ('model_sha256', 'wrong'), ('batch_size', 256)):
            with self.subTest(field=field):
                self.args.output = Path(self.tmp.name) / field
                def launch(cmd, **kwargs):
                    out = Path(cmd[cmd.index('--output') + 1])
                    out.mkdir()
                    protocol = pilot.expected_protocol(self.args, 'supertrack')
                    protocol[field] = value
                    (out / 'result.json').write_text(json.dumps(dict(
                        protocol=protocol, optimizer_counts=dict(wm=1328, policy=32),
                        evaluations=self.evaluations())))
                with patch.object(pilot.subprocess, 'run', side_effect=launch) as call:
                    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                        pilot.run(self.args)
                self.assertEqual(call.call_count, 1)
                report = json.loads((self.args.output / 'stages.json').read_text())
                self.assertEqual(report['stages'][0]['status'], 'failed')

    def test_invalid_physical_result_stops_before_second_method(self):
        def launch(cmd, **kwargs):
            out = Path(cmd[cmd.index('--output') + 1])
            out.mkdir()
            rows = self.evaluations()
            rows[0]['latch_rmse_rad'] = float('nan')
            (out / 'result.json').write_text(json.dumps(dict(
                protocol=pilot.expected_protocol(self.args, 'supertrack'),
                optimizer_counts=dict(wm=1328, policy=32), evaluations=rows)))
        with patch.object(pilot.subprocess, 'run', side_effect=launch) as call:
            with self.assertRaisesRegex(ValueError, 'physical metric'):
                pilot.run(self.args)
        self.assertEqual(call.call_count, 1)
        report = json.loads((self.args.output / 'stages.json').read_text())
        self.assertEqual(report['stages'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
