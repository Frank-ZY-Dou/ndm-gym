"""Timing boundaries must include queued device work and reject failed phases."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('door_timing', Path(__file__).parents[1] / 'ndm_gym/door_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TimingTests(unittest.TestCase):
    def test_device_completion_is_inside_elapsed_interval(self):
        events = []
        ticks = iter([10., 13.5])

        def clock():
            events.append('clock')
            return next(ticks)

        def work():
            events.append('work')
            return 'result'

        value, seconds = timing.measured_call(work, synchronize=lambda: events.append('sync'), clock=clock)
        self.assertEqual(value, 'result')
        self.assertEqual(seconds, 3.5)
        self.assertEqual(events, ['sync', 'clock', 'work', 'sync', 'clock'])

    def test_failed_work_is_not_a_completed_phase(self):
        events = []

        def fail():
            raise RuntimeError('interrupted update')

        with self.assertRaisesRegex(RuntimeError, 'interrupted update'):
            timing.measured_call(fail, synchronize=lambda: events.append('sync'), clock=lambda: 0.)
        self.assertEqual(events, ['sync'])

    def test_device_error_is_not_reported_as_success(self):
        calls = iter([False, True])

        def sync():
            if next(calls):
                raise RuntimeError('device failure')

        with self.assertRaisesRegex(RuntimeError, 'device failure'):
            timing.measured_call(lambda: 5, synchronize=sync)


if __name__ == '__main__':
    unittest.main()
