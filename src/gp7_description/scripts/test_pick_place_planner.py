#!/usr/bin/env python3
"""Offline regressions for constrained arm/wrist and vertical references."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kinematics import ExportedModel
from joint_trajectories import quintic_given_duration
from pick_place_config import arm_limits,load_config
from pick_place_planner import compensated_segment,joint_segment,vertical_segment,validate_segment,top_down_ik,wrist_candidates


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=copy.deepcopy(load_config());cls.config['planning']['sample_period_s']=.1
        cls.model=ExportedModel()
        cls.start=top_down_ik([.45,-.2,.325],config=cls.config)
        cls.goal=top_down_ik([.5,.5,.325],seed=cls.start,config=cls.config)

    def test_arm_is_exact_global_quintic_with_fixed_orientation(self):
        segment=compensated_segment('TRANSFER',self.start,self.goal,20.,self.config)
        times=np.linspace(0,20,77)
        arm=quintic_given_duration(self.start[:3],self.goal[:3],20.)
        for derivative,field in [(0,'position'),(1,'velocity'),(2,'acceleration')]:
            np.testing.assert_allclose(segment.evaluate(times,derivative)[:,:3],getattr(arm.evaluate(times),field),atol=1e-13,rtol=0)
        report=validate_segment(segment,self.config)
        self.assertLess(report['dense_interpolation_orientation_error_rad'],1e-5)
        self.assertGreater(report['min_wrist_singular_value'],.05)
        np.testing.assert_allclose(segment.evaluate(0.),self.start,atol=1e-7)
        np.testing.assert_allclose(segment.evaluate(20.),self.goal,atol=1e-7)
        np.testing.assert_allclose(segment.evaluate([0.,20.],1),0.,atol=1e-9)

    def test_vertical_fk_and_stopped_endpoints(self):
        segment=vertical_segment('DESCEND_PICK',self.start,.025,28.,self.config)
        report=validate_segment(segment,self.config)
        self.assertLess(report['dense_interpolation_xy_error_m'],1e-5)
        self.assertLess(report['dense_interpolation_orientation_error_rad'],1e-5)
        np.testing.assert_allclose(self.model.fk(segment.evaluate(28.))[:3,3],[.45,-.2,.025],atol=1e-8)
        np.testing.assert_allclose(segment.evaluate([0.,28.],1),0.,atol=1e-9)

    def test_singularity_and_continuous_rate_limits_reject(self):
        with self.assertRaises(ValueError):
            wrist_candidates(self.model,np.zeros(3),self.model.fk(np.zeros(6))[:3,:3],arm_limits(self.config))
        segment=joint_segment('TOO_FAST',np.zeros(6),np.ones(6)*.1,.05)
        with self.assertRaisesRegex(ValueError,'velocity'):validate_segment(segment,self.config)

    def test_infeasible_vertical_and_mismatched_orientation_reject(self):
        with self.assertRaises(ValueError):vertical_segment('IMPOSSIBLE',self.start,3.,10.,self.config)
        with self.assertRaises(ValueError):compensated_segment('BAD',self.start,np.zeros(6),20.,self.config)


if __name__=='__main__':unittest.main(verbosity=2)
