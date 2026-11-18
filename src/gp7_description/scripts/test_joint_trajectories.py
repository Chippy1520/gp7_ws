#!/usr/bin/env python3
"""Numerical and boundary tests for the given-duration reference equations."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_trajectories import quintic_given_duration, trapezoidal_given_duration


class GivenDurationTests(unittest.TestCase):
    def test_quintic_known_coefficients(self):
        trajectory = quintic_given_duration(0.0, 0.18, 2.0)
        np.testing.assert_allclose(
            trajectory.coefficients,
            [0.0, 0.0, 0.0, 0.225, -0.16875, 0.03375],
            atol=1e-14,
        )

    def test_quintic_general_boundary_conditions(self):
        trajectory = quintic_given_duration(
            0.2,
            -0.7,
            3.7,
            start_velocity=-0.1,
            end_velocity=0.15,
            start_acceleration=0.2,
            end_acceleration=-0.3,
        )
        start, end = trajectory.evaluate(0.0), trajectory.evaluate(3.7)
        np.testing.assert_allclose(
            [start.position, start.velocity, start.acceleration],
            [0.2, -0.1, 0.2],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [end.position, end.velocity, end.acceleration],
            [-0.7, 0.15, -0.3],
            atol=1e-12,
        )

    def test_quintic_multiple_joints(self):
        trajectory = quintic_given_duration(
            [0.0, 1.0, -1.0],
            [0.2, -0.1, 2.0],
            2.5,
            start_velocity=0.1,
            end_velocity=[0.2, 0.0, -0.1],
        )
        self.assertEqual(trajectory.coefficients.shape, (6, 3))
        samples = trajectory.evaluate([0.0, 1.0, 2.5])
        self.assertEqual(samples.position.shape, (3, 3))
        np.testing.assert_allclose(
            samples.position[[0, -1]], [[0.0, 1.0, -1.0], [0.2, -0.1, 2.0]], atol=1e-12
        )
        np.testing.assert_allclose(
            samples.velocity[[0, -1]], [[0.1] * 3, [0.2, 0.0, -0.1]], atol=1e-12
        )

    def test_quintic_constant_velocity_special_case(self):
        trajectory = quintic_given_duration(
            0.0, 1.0, 2.0, start_velocity=0.5, end_velocity=0.5
        )
        time = np.linspace(0.0, 2.0, 101)
        sample = trajectory.evaluate(time)
        np.testing.assert_allclose(sample.position, 0.5 * time, atol=1e-13)
        np.testing.assert_allclose(sample.velocity, 0.5, atol=1e-13)
        np.testing.assert_allclose(sample.acceleration, 0.0, atol=1e-13)

    def test_trapezoid_default_blends(self):
        trajectory = trapezoidal_given_duration(0.0, 0.18, 2.0)
        self.assertEqual(trajectory.acceleration_duration, 0.5)
        self.assertEqual(trajectory.deceleration_duration, 0.5)
        self.assertAlmostEqual(float(trajectory.cruise_velocity), 0.12)
        np.testing.assert_allclose(
            trajectory.evaluate([0.0, 0.5, 1.5, 2.0]).position,
            [0.0, 0.03, 0.15, 0.18],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            trajectory.evaluate([0.0, 0.5, 1.5, 2.0]).velocity,
            [0.0, 0.12, 0.12, 0.0],
            atol=1e-14,
        )
        self.assertIsNone(trajectory.evaluate(1.0).jerk)

    def test_trapezoid_nonzero_boundary_velocities(self):
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                trajectory = trapezoidal_given_duration(
                    0.0,
                    sign * 0.4,
                    2.0,
                    acceleration_duration=0.4,
                    deceleration_duration=0.8,
                    start_velocity=sign * 0.03,
                    end_velocity=sign * 0.06,
                )
                start, end = trajectory.evaluate(0.0), trajectory.evaluate(2.0)
                np.testing.assert_allclose(
                    [start.position, start.velocity], [0.0, sign * 0.03], atol=1e-14
                )
                np.testing.assert_allclose(
                    [end.position, end.velocity], [sign * 0.4, sign * 0.06], atol=1e-14
                )
                area = (
                    0.5 * (trajectory.start_velocity + trajectory.cruise_velocity) * 0.4
                    + trajectory.cruise_velocity * 0.8
                    + 0.5 * (trajectory.cruise_velocity + trajectory.end_velocity) * 0.8
                )
                self.assertAlmostEqual(float(area), sign * 0.4)

    def test_triangular_asymmetric_blends(self):
        trajectory = trapezoidal_given_duration(
            0.0, 0.18, 2.0, acceleration_duration=0.8, deceleration_duration=1.2
        )
        self.assertAlmostEqual(float(trajectory.cruise_velocity), 0.18)
        self.assertAlmostEqual(float(trajectory.evaluate(2.0).position), 0.18)
        self.check_joins(trajectory)

    def check_joins(self, trajectory):
        for boundary in [
            trajectory.acceleration_duration,
            trajectory.duration - trajectory.deceleration_duration,
        ]:
            left, right = (
                trajectory.evaluate(boundary - 1e-9),
                trajectory.evaluate(boundary + 1e-9),
            )
            np.testing.assert_allclose(left.position, right.position, atol=1e-8)
            np.testing.assert_allclose(left.velocity, right.velocity, atol=1e-8)

    def test_trapezoid_join_continuity_and_acceleration_convention(self):
        trajectory = trapezoidal_given_duration(0.0, 0.18, 2.0)
        self.check_joins(trajectory)
        self.assertEqual(float(trajectory.evaluate(0.5).acceleration), 0.0)
        self.assertEqual(
            float(trajectory.evaluate(1.5).acceleration), float(trajectory.deceleration)
        )
        self.assertEqual(
            float(trajectory.evaluate(2.0).acceleration), float(trajectory.deceleration)
        )

    def test_analytical_derivatives_against_finite_differences(self):
        trajectories = [
            quintic_given_duration(0.1, 0.5, 2.0),
            trapezoidal_given_duration(0.1, 0.5, 2.0),
        ]
        for trajectory in trajectories:
            for t in [0.2, 0.8, 1.8]:
                h = 1e-5
                left, center, right = [
                    trajectory.evaluate(x) for x in [t - h, t, t + h]
                ]
                self.assertAlmostEqual(
                    float((right.position - left.position) / (2 * h)),
                    float(center.velocity),
                    places=8,
                )
                self.assertAlmostEqual(
                    float((right.velocity - left.velocity) / (2 * h)),
                    float(center.acceleration),
                    places=8,
                )
                if center.jerk is not None:
                    self.assertAlmostEqual(
                        float((right.acceleration - left.acceleration) / (2 * h)),
                        float(center.jerk),
                        places=7,
                    )

    def test_vector_shapes_and_stationary_joints(self):
        for factory in [quintic_given_duration, trapezoidal_given_duration]:
            trajectory = factory([0.0, 1.0, -0.2], [0.2, 1.0, -0.5], 2.0)
            sample = trajectory.evaluate(np.linspace(0.0, 2.0, 31))
            self.assertEqual(sample.position.shape, (31, 3))
            np.testing.assert_allclose(sample.position[:, 1], 1.0)
            np.testing.assert_allclose(sample.velocity[:, 1], 0.0)
            np.testing.assert_allclose(sample.acceleration[:, 1], 0.0)
            np.testing.assert_allclose(
                trajectory.evaluate(2.0).position, [0.2, 1.0, -0.5], atol=1e-13
            )

    def test_constant_trajectory(self):
        for factory in [quintic_given_duration, trapezoidal_given_duration]:
            sample = factory(0.3, 0.3, 2.0).evaluate(np.linspace(0.0, 2.0, 101))
            np.testing.assert_allclose(sample.position, 0.3, atol=1e-14)
            np.testing.assert_allclose(sample.velocity, 0.0, atol=1e-14)

    def test_no_wrapping_or_automatic_retiming(self):
        for factory in [quintic_given_duration, trapezoidal_given_duration]:
            trajectory = factory(2.9, 3.5, 0.01)
            self.assertEqual(trajectory.duration, 0.01)
            self.assertAlmostEqual(float(trajectory.evaluate(0.01).position), 3.5)
            self.assertGreater(float(trajectory.evaluate(0.005).velocity), 1.0)

    def test_input_copies(self):
        for factory in [quintic_given_duration, trapezoidal_given_duration]:
            start = np.array([0.0, 1.0])
            goal = np.array([0.2, 2.0])
            trajectory = factory(start, goal, 2.0)
            start[:] = 100.0
            goal[:] = 200.0
            np.testing.assert_allclose(
                trajectory.evaluate(2.0).position, [0.2, 2.0], atol=1e-13
            )

    def test_invalid_inputs(self):
        for factory in [quintic_given_duration, trapezoidal_given_duration]:
            for T in [0.0, -1.0, np.nan, np.inf, [2.0]]:
                with (
                    self.subTest(factory=factory.__name__, T=T),
                    self.assertRaises(ValueError),
                ):
                    factory(0.0, 1.0, T)
            for start, end in [
                ([], []),
                ([0.0, 1.0], [2.0]),
                ([[0.0]], [[1.0]]),
                (0.0, np.nan),
            ]:
                with self.assertRaises(ValueError):
                    factory(start, end, 2.0)
            with self.assertRaises(ValueError):
                factory([0.0, 1.0], [1.0, 2.0], 2.0, start_velocity=[1.0, 2.0, 3.0])
            trajectory = factory(0.0, 1.0, 2.0)
            for t in [-0.1, 2.1, np.nan, np.inf, [[1.0]]]:
                with self.assertRaises(ValueError):
                    trajectory.evaluate(t)

    def test_unsupported_trapezoids_rejected(self):
        for arguments in [
            dict(acceleration_duration=0.0),
            dict(acceleration_duration=1.5, deceleration_duration=1.0),
            dict(start_velocity=-0.1),
            dict(start_velocity=1.0),
        ]:
            with self.assertRaises(ValueError):
                trapezoidal_given_duration(0.0, 0.1, 2.0, **arguments)
        with self.assertRaises(ValueError):
            trapezoidal_given_duration(1.0, 1.0, 2.0, end_velocity=0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
