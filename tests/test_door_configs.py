"""Preset parsing does not need a simulator, model, or torch import."""
import argparse
import ast
import importlib.util
import math
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / 'ndm_gym' / 'door_configs.py'
_SPEC = importlib.util.spec_from_file_location('door_configs', _PATH)
configs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(configs)


class DoorConfigsTest(unittest.TestCase):
    def parser(self):
        p = argparse.ArgumentParser()
        p.add_argument('--method', required=True, choices=('supertrack', 'supermimic'))
        p.add_argument('--horizon', type=int, default=4)
        return p

    def test_old_default_and_override(self):
        self.assertEqual(configs.parse_args(self.parser(), ['--method', 'supertrack']).horizon, 4)
        a = configs.parse_args(self.parser(), ['--method', 'supertrack', '--config', configs.NAME, '--horizon', '20'])
        self.assertEqual(a.horizon, 20)
        self.assertEqual(a.wm_hidden, [1024]*3)
        self.assertEqual(a.policy_hidden, [1024, 512, 256])
        self.assertEqual((a.wm_batch_size, a.batch_size, a.wm_window), (2048, 4096, 8))

    def test_method_schedule(self):
        for method, lr in [('supertrack', 4e-4), ('supermimic', 2e-4)]:
            a = configs.parse_args(self.parser(), ['--method', method, '--config', configs.NAME])
            self.assertEqual((a.horizon, a.wm_updates, a.policy_updates), (16, 4, 1))
            self.assertEqual((a.critic_epochs, a.critic_minibatches), (5, 4))
            self.assertEqual(a.policy_lr, lr)
            self.assertEqual(a.task_weights, method == 'supertrack')

    def test_refuse_unavailable(self):
        for option in configs.UNAVAILABLE:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                configs.parse_args(self.parser(), ['--method', 'supertrack', '--' + option])
        with self.assertRaises(SystemExit):
            configs.parse_args(self.parser(), ['--method', 'supertrack', '--config', 'hand-supertrack-accurate'])


class PinnedAuditTest(unittest.TestCase):
    def test_named_configs_include_inherited_defaults_and_source_identity(self):
        spec = importlib.util.spec_from_file_location('audit', _PATH.parents[1] / 'tools/audit_door_hand_configs.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.audit(_PATH.parents[1], '69b818e')
        self.assertTrue(result['upstream_commit'].startswith('69b818e'))
        for method in ('supertrack', 'supermimic'):
            values = result['configurations'][f'hand-{method}-accurate']['expanded']
            self.assertEqual(values['wm_hidden'], [1024]*3)
            self.assertEqual((values['horizon'], values['wm_batch'], values['pi_batch']), (16, 2048, 4096))
            self.assertTrue(values['collect_graph'] and values['update_graphs'])
            self.assertEqual(values['pi_opt'], 'adam')
        self.assertEqual(set(result['source_sha256']), {
            'ndm_gym/train.py', 'ndm_gym/algos/hand_fast.py', 'ndm_gym/train_hand.py',
            'ndm_gym/configs.py', 'README.md', 'ndm_gym/algos/hand_algos.py',
            'ndm_gym/worldmodels/hand.py', 'ndm_gym/policies/hand_policy.py',
            'docs/algorithms/xmimic.md', 'docs/algorithms/nablamimic.md'})


class ActualTrainerParserTest(unittest.TestCase):
    def test_real_trainer_defaults_and_preset(self):
        path = _PATH.with_name('train_door_online.py')
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parser')
        ns = dict(argparse=argparse, Path=Path, __doc__='native Door parser')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        required = ['--method', 'supermimic', '--model', 'm', '--reference', 'r', '--bootstrap', 'b', '--output', 'o']
        old = configs.parse_args(ns['parser'](), required)
        self.assertEqual((old.batch_size, old.horizon, old.policy_lr), (256, 4, 1e-5))
        self.assertEqual((old.wm_hidden, old.policy_hidden), ([64, 64], [32, 32]))
        new = configs.parse_args(ns['parser'](), required + ['--config', configs.NAME])
        self.assertEqual((new.batch_size, new.wm_batch_size, new.horizon, new.wm_window), (4096, 2048, 16, 8))
        self.assertEqual((new.critic_epochs, new.critic_minibatches), (5, 4))


class ActualBudgetValidationTest(unittest.TestCase):
    def test_default_and_accurate_budgets_and_invalid_overrides(self):
        # Execute the actual pure argument-validation code without importing the
        # simulator or learner. Real main/asset-boundary tests live in the runtime suite.
        path = _PATH.with_name('train_door_online.py')
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('parser', 'validate_args')]
        ns = dict(argparse=argparse, Path=Path, math=math, __doc__='Door validation')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        for method in ('supertrack', 'supermimic'):
            required = ['--method', method, '--model', 'm', '--reference', 'r',
                        '--bootstrap', 'b', '--output', 'o']
            for preset in ([], ['--config', configs.NAME]):
                args = configs.parse_args(ns['parser'](), required + preset)
                ns['validate_args'](args)
                for option, value in [('--wm-updates', '-1'), ('--policy-updates', '-1'),
                                      ('--bootstrap-updates', '-1'), ('--wm-batch-size', '0'),
                                      ('--wm-batch-size', '3'), ('--wm-window', '0'),
                                      ('--batch-size', '3'), ('--learning-rate', 'nan'),
                                      ('--policy-lr', 'inf'), ('--collection-worlds', '0'),
                                      ('--correction-reg', 'nan'), ('--correction-imag-noise', '-1'),
                                      ('--fused-adam', None)]:
                    with self.subTest(method=method, preset=preset, option=option):
                        invalid = configs.parse_args(ns['parser'](), required + preset + ([option] if value is None else [option, value]))
                        with self.assertRaises(ValueError):
                            ns['validate_args'](invalid)

    def test_method_specific_options_rejected_before_assets(self):
        path = _PATH.with_name('train_door_online.py')
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('parser', 'validate_args')]
        ns = dict(argparse=argparse, Path=Path, math=math, __doc__='Door validation')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        required = ['--method', 'supermimic', '--model', 'm', '--reference', 'r',
                    '--bootstrap', 'b', '--output', 'o']
        for flags in [['--task-weights'], ['--correction-reg', '.01'],
                      ['--correction-imag-noise', '.2'], ['--critic-minibatches', '3'],
                      ['--critic-minibatches', '0']]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                ns['validate_args'](configs.parse_args(ns['parser'](), required + flags))
