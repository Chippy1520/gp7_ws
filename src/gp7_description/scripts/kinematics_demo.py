#!/usr/bin/env python3
"""RViz pose-only FK/IK bridge. Never publishes Gazebo or actuator commands."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Resolve trusted sibling modules even when PYTHONSAFEPATH disables script-dir imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker, MarkerArray

from kinematics import ExportedModel, NAMES, URDF, ik_exported, so3_log, validate_pose


def quaternion_matrix(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError('Expected a finite nonzero quaternion [x,y,z,w]')
    q = q / np.max(np.abs(q))
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_pose(T):
    R = T[:3, :3]
    # Symmetric eigenproblem avoids the trace singularity at 180 degrees.
    K = np.array([[R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1], R[2,0]+R[0,2], R[2,1]-R[1,2]],
                  [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2], R[0,2]-R[2,0]],
                  [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], R[1,0]-R[0,1]],
                  [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], np.trace(R)]])
    _, vectors = np.linalg.eigh(K)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def pose_matrix(pose):
    T = np.eye(4)
    q = pose.orientation
    T[:3, :3] = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return validate_pose(T)


def slider_description():
    """Display-only principal-angle ranges, never written to the robot URDF."""
    robot = ET.parse(URDF).getroot()
    for joint in robot.findall('joint'):
        if joint.get('name') in NAMES:
            limit = joint.find('limit')
            limit.set('lower', str(-math.pi))
            limit.set('upper', str(math.pi))
    return ET.tostring(robot, encoding='unicode')


class DemoState:
    def __init__(self):
        self.model = ExportedModel()
        self.q = np.zeros(6)
        self.last_fk = self.q.copy()
        self.target = self.model.fk(self.q)
        self.mode = 'FK'
        self.solutions = []
        self.branch = 0
        self.error = ''

    def fk_input(self, q):
        self.model.fk(q)  # Validate before changing state.
        self.last_fk = np.array(q, dtype=float, copy=True)
        if self.mode == 'FK':
            self.q = self.last_fk.copy()
            self.target = self.model.fk(self.q)
            self.solutions = []
            self.error = ''

    def solve(self, target):
        target = validate_pose(target)
        solutions = ik_exported(target, seed=self.q, model=self.model)
        self.target = target.copy()
        # All geometric branches fit the standard GUI principal-angle display.
        self.solutions = [(q + math.pi) % (2 * math.pi) - math.pi for q in solutions]
        self.mode = 'IK'
        self.branch = 0
        self.error = '' if solutions else 'No supported IK solution; robot pose unchanged'
        if solutions:
            self.q = self.solutions[0].copy()
        return bool(solutions)

    def select(self, index):
        if not 0 <= index < len(self.solutions):
            raise ValueError('No such IK branch')
        self.branch = index
        self.q = self.solutions[index].copy()

    def use_fk(self):
        self.mode = 'FK'
        self.fk_input(self.last_fk)

    def status(self):
        actual = self.model.fk(self.q)
        return {'mode': self.mode, 'joint_names': NAMES, 'q_rad': self.q.tolist(),
                'q_deg': np.rad2deg(self.q).tolist(), 'tcp_matrix': actual.tolist(),
                'target_matrix': self.target.tolist(), 'branches': len(self.solutions),
                'branch': self.branch, 'position_error_m': float(np.linalg.norm(actual[:3,3]-self.target[:3,3])),
                'orientation_error_rad': float(np.linalg.norm(so3_log(self.target[:3,:3]@actual[:3,:3].T))),
                'error': self.error, 'scope': 'Visualization only; no hardware-limit or collision filtering'}


class KinematicsDemo(Node):
    def __init__(self):
        super().__init__('gp7_kinematics_demo')
        self.state = DemoState()
        self.joints = self.create_publisher(JointState, '/joint_states', 10)
        self.ik_feedback = self.create_publisher(JointState, '/gp7_demo/ik_joint_states', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.slider_model = self.create_publisher(String, '/gp7_demo/slider_description', qos)
        self.slider_model.publish(String(data=slider_description()))
        self.markers = self.create_publisher(MarkerArray, '/gp7_demo/markers', 10)
        self.report = self.create_publisher(String, '/gp7_demo/status', 10)
        self.create_subscription(JointState, '/gp7_demo/fk_joint_states', self.fk_input, 10)
        self.create_subscription(PoseStamped, '/gp7_demo/target_pose', self.target_input, 10)
        self.create_service(SetBool, '/gp7_demo/use_ik', self.mode_input)
        self.create_service(Trigger, '/gp7_demo/next_branch', self.next_branch)
        self.server = InteractiveMarkerServer(self, 'gp7_demo/target')
        self.menu = MenuHandler()
        self.menu.insert('FK: use joint sliders', callback=lambda _: self.state.use_fk())
        self.menu.insert('IK: load reachable example', callback=self.example_target)
        self.menu.insert('IK: next solution branch', callback=self.branch_menu)
        self.add_target()
        self.create_timer(.05, self.publish)
        self.get_logger().info('POSE DEMO ONLY. FK: standard sliders. IK: RViz Interact tool, drag TCP handles. No actuator commands.')

    def add_target(self):
        marker = InteractiveMarker()
        marker.header.frame_id = 'base_link'
        marker.name = 'tcp_target'
        marker.description = 'TCP_target(menu)'
        marker.pose = matrix_pose(self.state.target)
        marker.scale = .23
        center = InteractiveMarkerControl()
        center.always_visible = True
        center.interaction_mode = InteractiveMarkerControl.MENU
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.pose.orientation.w = 1.
        sphere.scale.x = sphere.scale.y = sphere.scale.z = .025
        sphere.color.r, sphere.color.g, sphere.color.a = 1., .7, 1.
        center.markers.append(sphere)
        marker.controls.append(center)
        rotations = [(0., 0., 0., 1.), (0., 0., math.sqrt(.5), math.sqrt(.5)),
                     (0., -math.sqrt(.5), 0., math.sqrt(.5))]
        for axis, quaternion in zip('xyz', rotations):
            for mode, prefix in [(InteractiveMarkerControl.MOVE_AXIS, 'move'),
                                 (InteractiveMarkerControl.ROTATE_AXIS, 'rotate')]:
                control = InteractiveMarkerControl()
                control.name = prefix + '_' + axis
                control.interaction_mode = mode
                control.orientation_mode = InteractiveMarkerControl.FIXED
                control.orientation.x, control.orientation.y, control.orientation.z, control.orientation.w = quaternion
                marker.controls.append(control)
        self.server.insert(marker, feedback_callback=self.feedback)
        self.menu.apply(self.server, 'tcp_target')
        self.server.applyChanges()

    def example_target(self, feedback):
        target = np.eye(4)
        target[:3, :3] = np.diag([1., -1., -1.])
        target[:3, 3] = [.5, .15, .45]
        self.state.solve(target)

    def branch_menu(self, feedback):
        response = self.next_branch(None, Trigger.Response())
        if not response.success:
            self.get_logger().info(response.message)

    def fk_input(self, msg):
        values = dict(zip(msg.name, msg.position))
        if not all(name in values for name in NAMES):
            return
        try:
            self.state.fk_input([values[name] for name in NAMES])
        except ValueError as e:
            self.get_logger().warning(str(e))

    def target_input(self, msg):
        if msg.header.frame_id != 'base_link':
            self.get_logger().warning('Target rejected: explicitly use frame_id base_link')
            return
        try:
            self.state.solve(pose_matrix(msg.pose))
        except ValueError as e:
            self.get_logger().warning(str(e))

    def feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            try:
                self.state.solve(pose_matrix(feedback.pose))
            except ValueError as e:
                self.get_logger().warning(str(e))

    def mode_input(self, request, response):
        if request.data:
            self.state.solve(self.state.model.fk(self.state.q))
        else:
            self.state.use_fk()
        response.success = True
        response.message = self.state.mode + ' pose demo; switching may jump, never send to actuators'
        return response

    def next_branch(self, request, response):
        if self.state.mode != 'IK' or not self.state.solutions:
            response.success, response.message = False, 'First specify a solvable IK target'
        else:
            self.state.select((self.state.branch + 1) % len(self.state.solutions))
            response.success, response.message = True, f'Branch {self.state.branch+1}/{len(self.state.solutions)}'
        return response

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = NAMES + ['gripper_joint_positive', 'gripper_joint_negative']
        msg.position = self.state.q.tolist() + [0., 0.]
        self.joints.publish(msg)
        # Feedback only in IK: continuous FK feedback would fight user edits.
        # Quantized GUI echoes remain inputs, not authority over an IK solution.
        if self.state.mode == 'IK':
            self.ik_feedback.publish(msg)
        self.server.setPose('tcp_target', matrix_pose(self.state.target))
        self.server.applyChanges()
        status = self.state.status()
        self.report.publish(String(data=json.dumps(status)))
        markers = []
        for axis, color in enumerate([(1., 0., 0.), (0., .7, 0.), (0., .25, 1.)]):
            m = Marker()
            m.header.frame_id = 'gripper_tcp'
            m.frame_locked = True
            m.ns, m.id, m.type = 'actual_tcp_axes', axis, Marker.ARROW
            m.pose.orientation.w = 1.
            end = [0., 0., 0.]
            end[axis] = .10
            m.points = [Point(), Point(x=end[0], y=end[1], z=end[2])]
            m.scale.x, m.scale.y, m.scale.z = .004, .009, .015
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 1.
            markers.append(m)
            target_axis = Marker()
            target_axis.header.frame_id = 'base_link'
            target_axis.header.stamp = stamp
            target_axis.ns, target_axis.id, target_axis.type = 'target_tcp_axes', axis, Marker.ARROW
            target_axis.pose = matrix_pose(self.state.target)
            end[axis] = .14
            target_axis.points = [Point(), Point(x=end[0], y=end[1], z=end[2])]
            target_axis.scale.x, target_axis.scale.y, target_axis.scale.z = .002, .006, .012
            target_axis.color.r, target_axis.color.g, target_axis.color.b = color
            target_axis.color.a = .6
            markers.append(target_axis)
        label = Marker()
        label.header.frame_id = 'base_link'
        label.header.stamp = stamp
        label.ns, label.id, label.type = 'readout', 0, Marker.TEXT_VIEW_FACING
        label.pose.position.x, label.pose.position.z = .3, 1.35
        label.pose.orientation.w = 1.
        label.scale.z, label.color.a = .03, 1.
        xyz = np.array(status['tcp_matrix'])[:3,3]
        q = status['q_deg']
        detail = 'FK:joint->TCP'
        if status['mode'] == 'IK':
            detail = ('IK:no_solution;robot_unchanged' if status['error']
                      else f"IK:branch={status['branch']+1}/{status['branches']}")
        # Compact strings also avoid excessive space-glyph advances in Ogre fonts.
        label.text = (detail + '\n'
                      f"TCP[m]=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f})\n"
                      f"S,L,U[deg]=({q[0]:.1f},{q[1]:.1f},{q[2]:.1f})\n"
                      f"R,B,T[deg]=({q[3]:.1f},{q[4]:.1f},{q[5]:.1f})\n"
                      f"Error={status['position_error_m']*1000:.5f}mm,"
                      f"{math.degrees(status['orientation_error_rad']):.5f}deg\n"
                      'Preview_only:no_collision_checks')
        markers.append(label)
        self.markers.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = KinematicsDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
