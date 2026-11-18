#!/usr/bin/env python3
"""Isolated RViz playback of direct-q(t) references. No actuator commands."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_trajectories import quintic_given_duration, trapezoidal_given_duration
from kinematics import ExportedModel, NAMES


class Playback:
    """Steady-time pose playback; stop/restart are NOT physical stop plans."""

    def __init__(self, duration=5.0, hold=1.0, now=0.0):
        self.duration = float(duration)
        self.hold = float(hold)
        if not np.isfinite(self.hold) or self.hold < 0:
            raise ValueError('Hold duration must be finite and nonnegative')
        self.start = np.zeros(6)
        self.goal = np.array([0.15, 0.12, -0.12, 0.12, 0.18, -0.10])
        self.trajectories = {}
        for name, factory in [
            ('quintic', quintic_given_duration),
            ('trapezoidal', trapezoidal_given_duration),
        ]:
            self.trajectories[name, False] = factory(
                self.start, self.goal, self.duration
            )
            self.trajectories[name, True] = factory(
                self.goal, self.start, self.duration
            )
        self.stopped_q = self.start.copy()
        self.restart('compare', now)

    def restart(self, profile, now):
        if profile not in ('compare', 'quintic', 'trapezoidal'):
            raise ValueError('Unknown playback profile')
        if not np.isfinite(now):
            raise ValueError('Start time must be finite')
        self.profile, self.origin, self.running = profile, float(now), True

    def evaluate(self, now):
        if not np.isfinite(now) or now < self.origin:
            raise ValueError('Playback time must be finite and not precede start')
        if not self.running:
            return dict(
                position=self.stopped_q.copy(),
                velocity=np.zeros(6),
                acceleration=np.zeros(6),
                profile=self.profile,
                direction='stopped',
                phase_time_s=0.0,
                holding=True,
                cycle=0,
                running=False,
            )
        elapsed = now - self.origin
        leg_duration = self.duration + self.hold
        phase_index = int(elapsed // leg_duration)
        phase_elapsed = elapsed % leg_duration
        legs_per_cycle = 4 if self.profile == 'compare' else 2
        leg_index = phase_index % legs_per_cycle
        if self.profile == 'compare':
            profile_name = 'quintic' if leg_index < 2 else 'trapezoidal'
        else:
            profile_name = self.profile
        reverse = bool(leg_index % 2)
        holding = phase_elapsed >= self.duration
        sample = self.trajectories[profile_name, reverse].evaluate(
            min(phase_elapsed, self.duration)
        )
        return dict(
            position=sample.position,
            velocity=np.zeros(6) if holding else sample.velocity,
            acceleration=np.zeros(6) if holding else sample.acceleration,
            profile=profile_name,
            direction='return' if reverse else 'forward',
            phase_time_s=min(phase_elapsed, self.duration),
            holding=holding,
            cycle=phase_index // legs_per_cycle,
            running=True,
        )

    def stop(self, now):
        self.stopped_q = self.evaluate(now)['position'].copy()
        self.running = False


class TrajectoryPreview(Node):
    def __init__(self, duration, hold):
        super().__init__('gp7_trajectory_preview')
        self.playback = Playback(duration, hold, time.monotonic())
        self.model = ExportedModel()
        self.joints = self.create_publisher(JointState, '/joint_states', 10)
        self.readout = self.create_publisher(JointState, '/gp7_trajectory/readout', 10)
        self.markers = self.create_publisher(MarkerArray, '/gp7_trajectory/markers', 10)
        self.status = self.create_publisher(String, '/gp7_trajectory/status', 10)
        for name, profile in [
            ('restart_comparison', 'compare'),
            ('play_quintic', 'quintic'),
            ('play_trapezoidal', 'trapezoidal'),
        ]:
            self.create_service(
                Trigger,
                '/gp7_trajectory/' + name,
                lambda request, response, p=profile: self.restart(p, response),
            )
        self.create_service(Trigger, '/gp7_trajectory/stop', self.stop)
        self.path = self._create_reference_path(duration)
        self.create_timer(0.02, self.publish)
        self.get_logger().info(
            'RViz references only. Comparison loops automatically. '
            'No limits, collisions or Gazebo commands. '
            'Restart resets the displayed pose to Home.'
        )

    def _create_reference_path(self, duration):
        """Draw FK along the reference; this is not a collision-free path claim."""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.ns, marker.id, marker.type = 'reference_tcp_path', 0, Marker.LINE_STRIP
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.004
        marker.color.r = marker.color.g = marker.color.b = 0.35
        marker.color.a = 0.7
        trajectory = self.playback.trajectories['quintic', False]
        for joint_angles in trajectory.evaluate(
            np.linspace(0.0, duration, 121)
        ).position:
            position = self.model.fk(joint_angles)[:3, 3]
            marker.points.append(
                Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            )
        return marker

    def restart(self, profile, response):
        self.playback.restart(profile, time.monotonic())
        response.success, response.message = (
            True,
            profile + ' restarted from Home (visualization only)',
        )
        return response

    def stop(self, request, response):
        self.playback.stop(time.monotonic())
        response.success, response.message = (
            True,
            'Display frozen; NOT a physical deceleration plan',
        )
        return response

    def publish(self):
        state = self.playback.evaluate(time.monotonic())
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = NAMES + ['gripper_joint_positive', 'gripper_joint_negative']
        msg.position = state['position'].tolist() + [0.0, 0.0]
        msg.velocity = state['velocity'].tolist() + [0.0, 0.0]
        self.joints.publish(msg)
        self.readout.publish(msg)
        report = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        report.update(
            duration_s=self.playback.duration,
            hold_s=self.playback.hold,
            stamp_sec=msg.header.stamp.sec,
            stamp_nanosec=msg.header.stamp.nanosec,
            joint_names=NAMES,
            scope='unchecked references, not measured robot feedback',
        )
        self.status.publish(String(data=json.dumps(report)))
        markers = [self.path, *self._tcp_axes(), self._readout_label(state)]
        self.markers.publish(MarkerArray(markers=markers))

    def _tcp_axes(self):
        """Follow the displayed TF directly, without a second moving target."""
        markers = []
        for axis, rgb in enumerate(
            [(1.0, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.25, 1.0)]
        ):
            marker = Marker()
            marker.header.frame_id, marker.frame_locked = 'gripper_tcp', True
            marker.ns, marker.id, marker.type = 'actual_tcp_axes', axis, Marker.ARROW
            marker.pose.orientation.w = 1.0
            end = [0.0, 0.0, 0.0]
            end[axis] = 0.10
            marker.points = [Point(), Point(x=end[0], y=end[1], z=end[2])]
            marker.scale.x, marker.scale.y, marker.scale.z = 0.004, 0.009, 0.015
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 1.0
            markers.append(marker)
        return markers

    def _readout_label(self, state):
        label = Marker()
        label.header.frame_id = 'base_link'
        label.ns, label.id, label.type = 'readout', 0, Marker.TEXT_VIEW_FACING
        label.pose.position.x, label.pose.position.z = 0.35, 1.35
        label.pose.orientation.w = 1.0
        label.scale.z, label.color.a = 0.028, 1.0
        q = np.rad2deg(state['position'])
        v = state['velocity']
        label.text = (
            f'{state["profile"]}:{state["direction"]}\n'
            f't={state["phase_time_s"]:.2f}/{self.playback.duration:g}s;hold={state["holding"]}\n'
            f'S,L,U[deg]=({q[0]:.1f},{q[1]:.1f},{q[2]:.1f})\n'
            f'R,B,T[deg]=({q[3]:.1f},{q[4]:.1f},{q[5]:.1f})\n'
            f'vS,L,U[rad/s]=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})\n'
            f'vR,B,T[rad/s]=({v[3]:.3f},{v[4]:.3f},{v[5]:.3f})\n'
            'REFERENCE_ONLY:no_limits_or_collisions'
        )
        return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--hold', type=float, default=1.0)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = TrajectoryPreview(args.duration, args.hold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
