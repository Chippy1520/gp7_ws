"""Offline regressions for the simulation-only Sprint readiness path."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import pick_place_motion as motion
from pick_place_config import load_config
from pick_place_planner import joint_segment


class WorkspacePreflightTests(unittest.TestCase):
    def setUp(self):
        self.cube = np.eye(4)
        self.cube[:3, 3] = [.50, -.35, .025]  # Outside the retired green patch.
        self.executor = SimpleNamespace(config=load_config(profile='sprint'),
                                        q=np.zeros(8), velocity=np.zeros(8), checker=Mock())
        self.executor.config['execution']['workspace_only'] = True

    def test_preflight_only_and_height_routing(self):
        actor = motion.GazeboMotion(self.executor, {}, {}, .25, .05, .1, False)
        with patch.object(motion, 'fresh', return_value={'cube': self.cube}), \
                patch.object(motion, 'gripper_bottom', side_effect=[.1, .3]), \
                patch.object(motion, 'approach') as plan:
            actor.preflight()
            self.assertTrue(actor.report['workspace_preflight']['needs_rise'])
            actor.preflight()
            self.assertFalse(actor.report['workspace_preflight']['needs_rise'])
            plan.assert_not_called()
        self.assertFalse(actor.segments)
        self.executor.checker.validate.assert_not_called()

    def test_off_floor_and_outside_workspace_are_rejected(self):
        for xyz, error in [([.5, -.35, .08], 'resting on the ground'),
                           ([0., 0., .025], 'outside the blue')]:
            with self.subTest(xyz=xyz):
                self.cube[:3, 3] = xyz
                with self.assertRaisesRegex(RuntimeError, error):
                    motion.check_workspace_start(self.executor, self.cube)

    def test_flat_rotated_cube_is_accepted_and_grasp_aligns_to_its_faces(self):
        for roll in (0., 90., 180.):
            for yaw in (-131., 37., 112.):
                with self.subTest(roll=roll, yaw=yaw):
                    self.cube[:3, :3] = Rotation.from_euler('xz', [roll, yaw], degrees=True).as_matrix()
                    with patch.object(motion, 'gripper_bottom', return_value=.3):
                        self.assertFalse(motion.check_workspace_start(self.executor, self.cube)['needs_rise'])
                    tcp = self.cube.copy()
                    tcp[:3, :3] = motion.sprint_grasp_rotation(self.cube)
                    np.testing.assert_allclose(tcp[:3, 2], [0., 0., -1.], atol=1e-10)
                    self.executor.q[6:] = .01
                    self.executor.model = Mock(fk=Mock(return_value=tcp))
                    self.executor.native = Mock(snapshot=Mock(return_value={'cube': self.cube}))
                    motion.Executor.check_grasp_geometry(self.executor)

    def test_cube_on_an_edge_is_rejected_even_if_it_touches_ground(self):
        self.cube[:3, :3] = Rotation.from_euler('x', 45., degrees=True).as_matrix()
        self.cube[2, 3] = .025 * np.abs(self.cube[2, :3]).sum()
        with self.assertRaisesRegex(RuntimeError, 'flat cube resting on the ground'):
            motion.check_workspace_start(self.executor, self.cube)

    def test_sprint_segments_keep_kinematics_without_collision_sweeps(self):
        segment = joint_segment('APPROACH_PICKUP', np.zeros(6), np.zeros(6), 1.)
        result = motion.checked(self.executor, segment)
        self.assertFalse(result['collision_checked'])
        self.executor.checker.validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
