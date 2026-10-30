"""Regression tests for held-pose goal padding at motion clip boundaries."""

import unittest

import torch

from ndm_gym.motion.library import MotionLib


class MotionGoalTest(unittest.TestCase):
    def setUp(self):
        self.lib = lib = MotionLib.__new__(MotionLib)
        lib.clip_start = torch.tensor([0, 4])
        lib.clip_end = torch.tensor([3, 9])
        lib.clip_id = torch.tensor([0] * 4 + [1] * 6)
        t = torch.arange(10, dtype=torch.float32)
        c, s = (t * 0.2).cos(), (t * 0.2).sin()
        z, o = torch.zeros_like(t), torch.ones_like(t)
        lib.root_R = torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(10, 3, 3)
        lib.root_p = torch.stack((t, t.square() * 0.1, t * 0.3), -1)
        lib.root_v = torch.stack((t + 1, t + 2, t + 3), -1)
        lib.root_w = lib.root_v * 0.2
        lib.root_v_local = (lib.root_R.transpose(-1, -2) @ lib.root_v[..., None])[..., 0]
        lib.root_w_local = (lib.root_R.transpose(-1, -2) @ lib.root_w[..., None])[..., 0]
        lib.q = t[:, None] + torch.arange(29, dtype=torch.float32)[None, :] * 0.01
        lib.qd = lib.q + 2

    def expected(self, frames, count):
        lib = self.lib
        rows = []
        for frame in frames.tolist():
            end = lib.clip_end[lib.clip_id[frame]].item()
            heading_t = lib.root_R[frame].T  # Fixture rotations contain yaw only.
            row = []
            for requested in range(frame + 1, frame + count + 1):
                future = min(requested, end)
                recorded = requested <= end
                row.append(torch.cat((
                    heading_t @ (lib.root_p[future] - lib.root_p[frame]),
                    (heading_t @ lib.root_R[future])[:, :2].reshape(6),
                    lib.root_v_local[future] if recorded else torch.zeros(3),
                    lib.root_w_local[future] if recorded else torch.zeros(3),
                    lib.q[future],
                    lib.qd[future] * 0.1 if recorded else torch.zeros(29),
                )))
            rows.append(torch.stack(row))
        return torch.stack(rows)

    def test_in_clip_goals_keep_recorded_features(self):
        frames = torch.tensor([0, 4])
        actual = self.lib.goal(frames, 2)
        self.assertEqual(actual.shape, (2, 146))
        torch.testing.assert_close(actual.reshape(2, 2, 73), self.expected(frames, 2))

    def test_mixed_clip_boundaries_hold_pose_and_zero_only_padding(self):
        frames = torch.tensor([2, 6])
        actual = self.lib.goal(frames, 4).reshape(2, 4, 73)
        torch.testing.assert_close(actual, self.expected(frames, 4))
        for row, last_recorded in enumerate((0, 2)):
            end = self.lib.clip_end[row]
            recorded_velocity = torch.cat((self.lib.root_v_local[end],
                                           self.lib.root_w_local[end], self.lib.qd[end] * 0.1))
            torch.testing.assert_close(
                torch.cat((actual[row, last_recorded, 9:15], actual[row, last_recorded, 44:])),
                recorded_velocity,
            )
            padded = actual[row, last_recorded + 1:]
            self.assertEqual(torch.count_nonzero(padded[:, 9:15]).item(), 0)
            self.assertEqual(torch.count_nonzero(padded[:, 44:]).item(), 0)
            for pose_slice in (slice(0, 9), slice(15, 44)):
                torch.testing.assert_close(padded[:, pose_slice],
                    actual[row, last_recorded, pose_slice].expand(padded.shape[0], -1))

    def test_terminal_queries_do_not_mutate_source_or_reference_state(self):
        lib = self.lib
        source = {name: value.clone() for name, value in vars(lib).items()}
        frames = lib.clip_end.clone()
        before = tuple(value.clone() for value in lib.ref_state(frames))
        for _ in range(2):
            actual = lib.goal(frames, 3).reshape(2, 3, 73)
            torch.testing.assert_close(actual, self.expected(frames, 3))
        torch.testing.assert_close(frames, source["clip_end"], rtol=0, atol=0)
        for name, value in source.items():
            torch.testing.assert_close(getattr(lib, name), value, rtol=0, atol=0)
        for actual, expected in zip(lib.ref_state(frames), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
