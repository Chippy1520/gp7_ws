#!/usr/bin/env python3
"""Direct q(t) trajectories for a GIVEN duration; no limit enforcement or control.

Positions may be scalars or one-dimensional joint vectors. Angles and their
rates use radians and seconds. No angular wrapping is performed. These are
reference equations, not collision-checked or dynamically validated motions.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    """Analytical position and derivatives at one time or an array of times."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    # Trapezoidal acceleration jumps have no finite ordinary jerk value.
    jerk: np.ndarray | None


def _as_finite_array(value, name):
    value = np.array(value, dtype=float, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def _positive_scalar(value, name):
    value = _as_finite_array(value, name)
    if value.ndim != 0 or value <= 0:
        raise ValueError(f'{name} must be a positive scalar')
    return float(value)


def _validated_endpoints(q_start, q_goal):
    q_start = _as_finite_array(q_start, 'q_start')
    q_goal = _as_finite_array(q_goal, 'q_goal')
    if q_start.ndim > 1 or q_start.size == 0 or q_start.shape != q_goal.shape:
        raise ValueError('Endpoints must be scalars or matching nonempty joint vectors')
    return q_start, q_goal


def _boundary_array(value, shape, name):
    value = _as_finite_array(value, name)
    try:
        return np.broadcast_to(value, shape).copy()
    except ValueError as e:
        raise ValueError(f'{name} must be scalar or match the joint vector') from e


def _validated_time(value, duration):
    value = _as_finite_array(value, 'time')
    if value.ndim > 1 or np.any(value < 0) or np.any(value > duration):
        raise ValueError('Time must be scalar or a 1-D array within [0, duration]')
    return value


def _evaluate_polynomial(coefficients, time):
    # Coefficients are ascending powers; Horner evaluates the direct q(t).
    expanded_time = time.reshape(time.shape + (1,) * (coefficients.ndim - 1))
    result = np.zeros(time.shape + coefficients.shape[1:])
    for coefficient in coefficients[::-1]:
        result = result * expanded_time + coefficient
    return result


@dataclass(frozen=True)
class QuinticTrajectory:
    """Direct polynomial coefficients in ascending order: a0 through a5."""

    duration: float
    # Shape (6,) for one joint or (6, number_of_joints) for a vector.
    coefficients: np.ndarray

    def evaluate(self, time):
        time = _validated_time(time, self.duration)
        derivatives = [self.coefficients]
        for _ in range(3):
            previous = derivatives[-1]
            derivatives.append(
                np.array([i * previous[i] for i in range(1, len(previous))])
            )
        return TrajectorySample(*[_evaluate_polynomial(c, time) for c in derivatives])


def quintic_given_duration(
    q_start,
    q_goal,
    duration,
    *,
    start_velocity=0.0,
    end_velocity=0.0,
    start_acceleration=0.0,
    end_acceleration=0.0,
):
    """Solve M(T) a = b for general endpoint position, velocity, acceleration.

    Returns a0..a5, not a normalized-progress polynomial. All joints use the
    supplied duration. No velocity, acceleration, jerk or position limits are
    applied. Nonzero boundary derivatives may produce overshoot or reversal.
    """
    q_start, q_goal = _validated_endpoints(q_start, q_goal)
    T = np.float64(_positive_scalar(duration, 'duration'))
    v0 = _boundary_array(start_velocity, q_start.shape, 'start_velocity')
    vf = _boundary_array(end_velocity, q_start.shape, 'end_velocity')
    acc0 = _boundary_array(start_acceleration, q_start.shape, 'start_acceleration')
    accf = _boundary_array(end_acceleration, q_start.shape, 'end_acceleration')
    # Rows follow the syllabus: q(0), q_dot(0), q_ddot(0), q(T), q_dot(T), q_ddot(T).
    boundary_values = np.stack([q_start, v0, acc0, q_goal, vf, accf])
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            boundary_matrix = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                ],
                dtype=float,
            )
            coefficients = np.linalg.solve(boundary_matrix, boundary_values)
            if not np.isfinite(coefficients).all():
                raise ValueError('Nonfinite polynomial coefficients')
    except (np.linalg.LinAlgError, FloatingPointError) as e:
        raise ValueError(
            'Duration or boundary values exceed the numerical range'
        ) from e
    return QuinticTrajectory(float(T), coefficients)


@dataclass(frozen=True)
class TrapezoidalTrajectory:
    """Piecewise position with linear ramps in velocity."""

    duration: float
    acceleration_duration: float
    deceleration_duration: float
    q_start: np.ndarray
    q_goal: np.ndarray
    start_velocity: np.ndarray
    end_velocity: np.ndarray
    cruise_velocity: np.ndarray
    acceleration: np.ndarray
    deceleration: np.ndarray

    def evaluate(self, time):
        time = _validated_time(time, self.duration)
        time = time.reshape(time.shape + (1,) * self.q_start.ndim)
        ta, td, T = (
            self.acceleration_duration,
            self.deceleration_duration,
            self.duration,
        )
        remaining = T - time
        position_at_cruise_start = (
            self.q_start + 0.5 * (self.start_velocity + self.cruise_velocity) * ta
        )
        accelerating = time < ta
        decelerating = time >= T - td

        # Evaluate each position equation before selecting the active phase.
        position_accelerating = (
            self.q_start
            + self.start_velocity * time
            + 0.5 * self.acceleration * time**2
        )
        position_cruising = position_at_cruise_start + self.cruise_velocity * (
            time - ta
        )
        position_decelerating = (
            self.q_goal
            - self.end_velocity * remaining
            + 0.5 * self.deceleration * remaining**2
        )
        position = np.where(
            accelerating,
            position_accelerating,
            np.where(decelerating, position_decelerating, position_cruising),
        )

        velocity_accelerating = self.start_velocity + self.acceleration * time
        velocity_decelerating = self.end_velocity - self.deceleration * remaining
        velocity = np.where(
            accelerating,
            velocity_accelerating,
            np.where(decelerating, velocity_decelerating, self.cruise_velocity),
        )
        acceleration = np.where(
            accelerating,
            self.acceleration,
            np.where(decelerating, self.deceleration, 0.0),
        )
        return TrajectorySample(position, velocity, acceleration, None)


def trapezoidal_given_duration(
    q_start,
    q_goal,
    duration,
    *,
    acceleration_duration=None,
    deceleration_duration=None,
    start_velocity=0.0,
    end_velocity=0.0,
):
    """Piecewise q(t) with specified blend durations and no limit enforcement.

    Defaults: acceleration and deceleration each take T/4. Setting both to
    T/2 produces a triangular velocity profile. Endpoint speeds must permit
    a nonreversing accelerate-cruise-decelerate shape in each moving joint.
    A zero-displacement joint requires zero endpoint velocities.

    Acceleration jumps at joins. Evaluation uses the right-hand segment at
    internal joins and the deceleration segment's left-hand value at t=T.
    Arbitrary endpoint accelerations are not supported by this profile.
    """
    q_start, q_goal = _validated_endpoints(q_start, q_goal)
    T = _positive_scalar(duration, 'duration')
    ta = _positive_scalar(
        T / 4 if acceleration_duration is None else acceleration_duration,
        'acceleration_duration',
    )
    td = _positive_scalar(
        T / 4 if deceleration_duration is None else deceleration_duration,
        'deceleration_duration',
    )
    if ta + td > T:
        raise ValueError(
            'Acceleration and deceleration durations must sum to at most T'
        )
    v0 = _boundary_array(start_velocity, q_start.shape, 'start_velocity')
    vf = _boundary_array(end_velocity, q_start.shape, 'end_velocity')
    displacement = q_goal - q_start
    # Area under the velocity profile must equal the signed displacement.
    cruise = (displacement - 0.5 * (v0 * ta + vf * td)) / (T - 0.5 * (ta + td))
    direction = np.sign(displacement)
    u0, uf, peak = direction * v0, direction * vf, direction * cruise
    tolerance = (
        64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs([v0, vf, cruise]))))
    )
    stationary_with_velocity = np.any((direction == 0) & ((v0 != 0) | (vf != 0)))
    reversed_boundary_velocity = np.any(u0 < -tolerance) or np.any(uf < -tolerance)
    cruise_below_boundary_speed = np.any(peak < np.maximum(u0, uf) - tolerance)
    if (
        stationary_with_velocity
        or reversed_boundary_velocity
        or cruise_below_boundary_speed
    ):
        raise ValueError(
            'Blend times and endpoint velocities do not form a nonreversing trapezoid'
        )
    return TrapezoidalTrajectory(
        T,
        ta,
        td,
        q_start,
        q_goal,
        v0,
        vf,
        cruise,
        (cruise - v0) / ta,
        (vf - cruise) / td,
    )
