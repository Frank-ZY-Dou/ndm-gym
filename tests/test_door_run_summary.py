"""Formal evidence aggregation rejects incomplete and incomparable studies."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location(
    'door_summary', Path(__file__).resolve().parents[1] / 'tools/summarize_door_runs.py')
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def fixture():
    offsets = [i / 100000 for i in range(256)]
    common = {key: 1 for key in summary.MATCHED}
    common.update(offsets=offsets, selection='final only')
    protocols = {method: dict(common, method=method, critic_epochs=5, critic_minibatches=4)
                 for method in ('supertrack', 'supermimic')}
    manifest = dict(seeds=[101, 102, 103], evaluations=offsets, protocols=protocols)
    results = []
    for method, protocol in protocols.items():
        for index, seed in enumerate(manifest['seeds']):
            counts = dict(wm=2, policy=1)
            if method == 'supermimic':
                counts['critic'] = 20
            results.append(dict(protocol=dict(protocol, seed=seed), optimizer_counts=counts,
                                process_seconds=2., evaluations=[dict(
                                    ARRz_offset_rad=offset, finite=True, opened_final=index > 0,
                                    opened_ever=True, translation_rmse_m=.01,
                                    hand_rmse_rad=.1, door_rmse_rad=.02, latch_rmse_rad=.03)
                                    for offset in offsets]))
    return manifest, results


class DoorSummaryTest(unittest.TestCase):
    def test_rejects_boolean_physical_offset_equal_to_zero(self):
        manifest, results = fixture()
        results[0]['evaluations'][0]['ARRz_offset_rad'] = False
        with self.assertRaisesRegex(ValueError, 'offsets/order'):
            summary.summarize(manifest, results)

    def test_reports_all_seeds_and_sample_sd(self):
        manifest, results = fixture()
        report = summary.summarize(manifest, results)
        for method in report['methods'].values():
            self.assertEqual([r['seed'] for r in method['seeds']], [101, 102, 103])
            self.assertAlmostEqual(method['summary']['opened_final']['mean'], 2/3)
            self.assertAlmostEqual(method['summary']['opened_final']['sample_sd'], (1/3)**.5)
        self.assertIn('interrupted attempts', report['cost_scope'])

    def test_measured_invocation_cost_retains_resumed_scope(self):
        manifest, results = fixture()
        for index, result in enumerate(results):
            result['process_seconds'] = index + 1
        results[0]['resumed_from'] = '/trusted/checkpoint.pt'
        report = summary.summarize(manifest, results)
        self.assertEqual(report['reported_invocation_seconds'], 21)
        self.assertEqual(report['methods']['supertrack']['invocation_seconds'],
                         dict(total=6, mean=2, sample_sd=1))
        self.assertEqual(report['methods']['supermimic']['invocation_seconds'],
                         dict(total=15, mean=5, sample_sd=1))
        self.assertEqual(report['methods']['supertrack']['seeds'][0]['resumed_from'],
                         '/trusted/checkpoint.pt')
        self.assertIn('resumed prior work', report['cost_scope'])
        self.assertIn('GPU cost require separate', report['cost_scope'])

    def test_rejects_boolean_invocation_cost(self):
        manifest, results = fixture()
        results[0]['process_seconds'] = True
        with self.assertRaises(ValueError):
            summary.summarize(manifest, results)

    def test_rejects_boolean_or_fractional_update_evidence(self):
        original, runs = fixture()
        # JSON booleans compare equal to integers in Python; neither booleans
        # nor fractional counts certify that an optimizer actually ran.
        for invalid in (True, 1.0):
            results = copy.deepcopy(runs)
            results[0]['optimizer_counts']['policy'] = invalid
            with self.assertRaises(ValueError):
                summary.summarize(original, results)
        for method, key in (('supertrack', 'rounds'),
                            ('supermimic', 'critic_epochs')):
            for invalid in (True, 1.0, 0, -1):
                manifest = copy.deepcopy(original)
                manifest['protocols'][method][key] = invalid
                if key == 'rounds':
                    manifest['protocols']['supermimic'][key] = invalid
                with self.assertRaises(ValueError):
                    summary.summarize(manifest, runs)

    def test_boolean_tracking_errors_are_not_physical_measurements(self):
        manifest, original = fixture()
        for key in summary.METRICS[2:]:
            for invalid in (True, False, "0.1", None):
                results = copy.deepcopy(original)
                results[0]['evaluations'][0][key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    summary.summarize(manifest, results)

    def test_boolean_or_nonnumeric_evaluation_offsets_are_refused(self):
        original, results = fixture()
        for invalid in (True, "0.1", None):
            manifest = copy.deepcopy(original)
            manifest['evaluations'][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summary.summarize(manifest, results)

    def test_rejects_partial_and_duplicate_runs(self):
        manifest, results = fixture()
        for runs in (results[:-1], results + [results[0]], [results[0]] * 6):
            with self.assertRaises(ValueError):
                summary.summarize(manifest, runs)

    def test_rejects_invalid_physics_counts_and_protocol(self):
        manifest, original = fixture()
        for mutation in (
            lambda r: r['optimizer_counts'].update(wm=1),
            lambda r: r['protocol'].update(seed=999),
            lambda r: r['protocol'].update(extra_flag=True),
            lambda r: r['evaluations'][0].update(finite=False),
            lambda r: r['evaluations'][0].update(door_rmse_rad=float('nan')),
            lambda r: r['evaluations'][0].update(opened_final=1),
            lambda r: r['evaluations'].reverse(),
            lambda r: r.update(process_seconds=float('inf')),
        ):
            results = copy.deepcopy(original)
            mutation(results[0])
            with self.assertRaises(ValueError):
                summary.summarize(manifest, results)

    def test_rejects_unmatched_budget_and_tiny_evaluation(self):
        original, results = fixture()
        for mutation in (
            lambda m: m['protocols']['supertrack'].update(rounds=2),
            lambda m: m.update(seeds=[101, 101, 102]),
            lambda m: m.update(evaluations=[0]),
        ):
            manifest = copy.deepcopy(original)
            mutation(manifest)
            with self.assertRaises(ValueError):
                summary.summarize(manifest, results)

    def test_cli_hashes_inputs_and_refuses_overwrite(self):
        manifest, results = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest['runs'] = []
            for index, result in enumerate(results):
                name = f'result{index}.json'
                (root / name).write_text(json.dumps(result))
                manifest['runs'].append(name)
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            output = root / 'summary.json'
            command = [sys.executable, summary.__file__, str(path), '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads(output.read_text())
            self.assertEqual(len(report['result_sha256']), 6)
            self.assertEqual(len(report['manifest_sha256']), 64)
            before = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            manifest['runs'].pop()
            path.write_text(json.dumps(manifest))
            output.unlink()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse(output.exists())
