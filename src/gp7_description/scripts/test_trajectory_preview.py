#!/usr/bin/env python3
"""Headless timing/ownership checks for the RViz-only trajectory player."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_preview import Playback
from joint_trajectories import quintic_given_duration, trapezoidal_given_duration


class PlaybackTests(unittest.TestCase):
    def test_all_four_legs_match_existing_functions(self):
        player = Playback(duration=5.0, hold=1.0, now=100.0)
        for leg, (name, reverse, factory) in enumerate(
            [
                ('quintic', False, quintic_given_duration),
                ('quintic', True, quintic_given_duration),
                ('trapezoidal', False, trapezoidal_given_duration),
                ('trapezoidal', True, trapezoidal_given_duration),
            ]
        ):
            start, goal = (
                (player.goal, player.start) if reverse else (player.start, player.goal)
            )
            trajectory = factory(start, goal, 5.0)
            for t in [0.0, 0.4, 1.25, 2.5, 4.9]:
                state = player.evaluate(100.0 + leg * 6.0 + t)
                expected = trajectory.evaluate(t)
                self.assertEqual(state['profile'], name)
                self.assertEqual(state['direction'], 'return' if reverse else 'forward')
                np.testing.assert_allclose(
                    state['position'], expected.position, atol=1e-13
                )
                np.testing.assert_allclose(
                    state['velocity'], expected.velocity, atol=1e-13
                )
                np.testing.assert_allclose(
                    state['acceleration'], expected.acceleration, atol=1e-13
                )

    def test_holds_and_cycle_boundary(self):
        player = Playback(now=0.0)
        for leg in range(4):
            state = player.evaluate(leg * 6.0 + 5.5)
            self.assertTrue(state['holding'])
            np.testing.assert_allclose(
                state['position'],
                player.goal if leg % 2 == 0 else player.start,
                atol=1e-13,
            )
            np.testing.assert_array_equal(state['velocity'], np.zeros(6))
            np.testing.assert_array_equal(state['acceleration'], np.zeros(6))
        state = player.evaluate(24.0)
        self.assertEqual(state['cycle'], 1)
        self.assertEqual(state['profile'], 'quintic')
        np.testing.assert_allclose(state['position'], player.start)

    def test_single_profile_repeats(self):
        player = Playback()
        for name in ['quintic', 'trapezoidal']:
            player.restart(name, 10.0)
            for leg in range(4):
                state = player.evaluate(10.0 + leg * 6.0 + 2.0)
                self.assertEqual(state['profile'], name)
                self.assertEqual(state['direction'], 'return' if leg % 2 else 'forward')

    def test_stop_freezes_pose_and_restart_is_explicit(self):
        player = Playback(now=0.0)
        before = player.evaluate(2.0)['position'].copy()
        player.stop(2.0)
        state = player.evaluate(100.0)
        self.assertFalse(state['running'])
        np.testing.assert_array_equal(state['position'], before)
        np.testing.assert_array_equal(state['velocity'], np.zeros(6))
        player.restart('trapezoidal', 100.0)
        np.testing.assert_array_equal(player.evaluate(100.0)['position'], player.start)

    def test_zero_hold_boundary_is_continuous(self):
        player = Playback(duration=2.0, hold=0.0)
        np.testing.assert_allclose(
            player.evaluate(2.0)['position'], player.goal, atol=1e-13
        )
        np.testing.assert_allclose(
            player.evaluate(4.0)['position'], player.start, atol=1e-13
        )

    def test_bad_times_and_arguments(self):
        for kwargs in [
            dict(duration=0),
            dict(duration=np.inf),
            dict(hold=-1),
            dict(hold=np.nan),
        ]:
            with self.assertRaises(ValueError):
                Playback(**kwargs)
        player = Playback(now=10.0)
        for now in [9.0, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                player.evaluate(now)
        with self.assertRaises(ValueError):
            player.restart('unknown', 10.0)
        with self.assertRaises(ValueError):
            player.restart('quintic', np.nan)


if __name__ == '__main__':
    unittest.main(verbosity=2)
