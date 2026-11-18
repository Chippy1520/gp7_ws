"""Offline regressions for the simulation-only Sprint readiness path."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

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

    def test_sprint_segments_keep_kinematics_without_collision_sweeps(self):
        segment = joint_segment('APPROACH_PICKUP', np.zeros(6), np.zeros(6), 1.)
        result = motion.checked(self.executor, segment)
        self.assertFalse(result['collision_checked'])
        self.executor.checker.validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
