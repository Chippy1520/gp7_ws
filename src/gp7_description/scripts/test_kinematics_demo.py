#!/usr/bin/env python3
"""Headless checks for the pose-only FK/IK demonstration."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from kinematics import rpy_matrix
from kinematics_demo import DemoState, matrix_pose, pose_matrix, quaternion_matrix, slider_description
import xml.etree.ElementTree as ET


class DemoTests(unittest.TestCase):
    def target(self):
        T = np.eye(4)
        T[:3, :3] = rpy_matrix([np.pi, 0., 0.])
        T[:3, 3] = [.5, .15, .45]
        return T

    def test_fk_updates_pose(self):
        state = DemoState()
        q = np.array([.2, -.2, .1, .3, -.3, .1])
        state.fk_input(q)
        np.testing.assert_allclose(state.status()['tcp_matrix'], state.model.fk(q))
        np.testing.assert_allclose(state.target, state.model.fk(q))
        self.assertEqual(state.mode, 'FK')

    def test_all_ik_branches_match_one_target(self):
        state = DemoState()
        T = self.target()
        self.assertTrue(state.solve(T))
        self.assertEqual(len(state.solutions), 8)
        first = state.q.copy()
        for index in range(len(state.solutions)):
            state.select(index)
            np.testing.assert_allclose(state.model.fk(state.q), T, atol=2e-9)
            self.assertLess(state.status()['position_error_m'], 1e-9)
            self.assertLess(state.status()['orientation_error_rad'], 1e-9)
        self.assertGreater(np.linalg.norm(first-state.q), .1)

    def test_slider_ranges_cover_all_example_branches(self):
        robot = ET.fromstring(slider_description())
        state = DemoState()
        state.solve(self.target())
        for solution in state.solutions:
            for name, value in zip(state.status()['joint_names'], solution):
                limit = robot.find(f"joint[@name='{name}']/limit")
                self.assertGreaterEqual(value, float(limit.get('lower')))
                self.assertLessEqual(value, float(limit.get('upper')))
            np.testing.assert_allclose(state.model.fk(solution), self.target(), atol=2e-9)

    def test_nearby_drag_stays_on_nearby_branch(self):
        state = DemoState()
        T = self.target()
        state.solve(T)
        before = state.q.copy()
        T[0, 3] += .001
        self.assertTrue(state.solve(T))
        self.assertLess(np.linalg.norm(state.q-before), .1)

    def test_unreachable_preserves_robot(self):
        state = DemoState()
        state.solve(self.target())
        before = state.q.copy()
        T = self.target()
        T[0, 3] = 10.
        self.assertFalse(state.solve(T))
        np.testing.assert_array_equal(state.q, before)
        self.assertTrue(state.error)
        self.assertEqual(state.status()['branches'], 0)

    def test_fk_inputs_cannot_overwrite_ik_pose(self):
        state = DemoState()
        state.solve(self.target())
        before = state.q.copy()
        state.fk_input(np.full(6, .1))
        np.testing.assert_array_equal(state.q, before)
        state.use_fk()
        np.testing.assert_array_equal(state.q, np.full(6, .1))

    def test_quaternion_round_trip_and_extreme_scaling(self):
        state = DemoState()
        rng = np.random.default_rng(52)
        for q in [np.zeros(6)] + list(rng.uniform(-np.pi, np.pi, (100, 6))):
            T = state.model.fk(q)
            np.testing.assert_allclose(pose_matrix(matrix_pose(T)), T, atol=2e-14)
        np.testing.assert_allclose(quaternion_matrix([1e308, 0, 0, 0]), rpy_matrix([np.pi, 0, 0]), atol=1e-15)

    def test_invalid_input_rejected_without_joint_change(self):
        state = DemoState()
        before = state.q.copy()
        for q in [np.full(6, np.nan), np.zeros(5)]:
            with self.assertRaises(ValueError):
                state.fk_input(q)
        with self.assertRaises(ValueError):
            quaternion_matrix([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            state.select(0)
        np.testing.assert_array_equal(state.q, before)


if __name__ == '__main__':
    unittest.main()
