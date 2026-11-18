#!/usr/bin/env python3
"""Offline tests for explicit expanded-model, rate-bounded simulation profiles."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xml.etree.ElementTree as E
import numpy as np
from build_pick_place_world import build
from kinematics import ExportedModel,ik_exported
from pick_place_config import load_config,PKG,NAMES,arm_limits
from pick_place_profiles import PROFILES
from probe_jacobian_polar_route import CartesianPath,differential_segment


class ProfileTests(unittest.TestCase):
    def test_profile_worlds_retain_model_limits_and_effort_caps(self):
        urdf=E.parse(PKG/'urdf/gp7_with_gripper.urdf').getroot()
        for name in PROFILES:
            with self.subTest(profile=name),tempfile.TemporaryDirectory() as d:
                config=load_config(profile=name)
                world=build(output_path=Path(d)/'world.sdf',profile=name)
                robot=E.parse(world).getroot().find("world/model[@name='gp7']")
                for joint in NAMES:
                    limit=urdf.find(f"joint[@name='{joint}']/limit")
                    self.assertGreaterEqual(config['joint_limits'][joint][0],float(limit.get('lower')))
                    self.assertLessEqual(config['joint_limits'][joint][1],float(limit.get('upper')))
                    if joint in ['joint_s','joint_t']:
                        self.assertEqual(config['joint_limits'][joint],[float(limit.get(k)) for k in ['lower','upper']])
                    self.assertLessEqual(config['velocity_limit_rad_s'],float(limit.get('velocity')))
                    self.assertLessEqual(float(robot.findtext(f"joint[@name='{joint}']/axis/limit/effort")),float(limit.get('effort')))
                    controller=next(p for p in robot.findall('plugin') if p.findtext('joint_name')==joint
                                    and p.get('name')=='gz::sim::systems::JointPositionController')
                    self.assertEqual(controller.get('name'),'gz::sim::systems::JointPositionController')
                    self.assertEqual(controller.findtext('use_velocity_commands'),'true')
                    self.assertEqual(float(controller.findtext('cmd_max')),config['velocity_limit_rad_s'])
                    self.assertIsNone(controller.find('velocity_topic'))
                self.assertFalse(config['execution']['velocity_feedforward'])

    def test_expanded_model_allows_rear_ik_without_unbounded_angles(self):
        config=load_config(profile='fast');target=np.eye(4)
        target[:3,:3]=np.diag([1.,-1.,-1.]);target[:3,3]=[-.45,-.2,.25]
        solutions=ik_exported(target,model=ExportedModel(),limits=arm_limits(config))
        self.assertTrue(solutions)
        self.assertTrue(all(np.all(q>=arm_limits(config)[:,0]) and np.all(q<=arm_limits(config)[:,1]) for q in solutions))

    def test_polar_alternative_winding_preserves_endpoint(self):
        start=np.array([-.45,-.2,.275]);goal=np.array([.5,.5,.275])
        raw=np.arctan2(goal[1],goal[0])-np.arctan2(start[1],start[0])
        short=CartesianPath(start,goal,2.,True);long=CartesianPath(start,goal,2.,True,angular_delta=raw)
        np.testing.assert_allclose(long.evaluate(0)[0],start,atol=1e-12)
        np.testing.assert_allclose(long.evaluate(2)[0],goal,atol=1e-12)
        self.assertGreater(np.linalg.norm(long.evaluate(1)[0]-short.evaluate(1)[0]),.5)
        with self.assertRaises(ValueError):CartesianPath(start,goal,2.,True,angular_delta=.1)

    def test_aggressive_request_is_retimed_not_clipped(self):
        config=load_config(profile='fast');model=ExportedModel();target=np.eye(4)
        target[:3,:3]=np.diag([1.,-1.,-1.]);target[:3,3]=[.45,-.2,.025]
        q=ik_exported(target,model=model,limits=arm_limits(config))[0]
        start=model.fk(q)[:3,3];goal=start+np.array([0.,0.,.01])
        segment,report=differential_segment('TEST_RETIME',q,CartesianPath(start,goal,.05),config)
        self.assertGreater(report['time_scale_factor'],1.)
        self.assertGreater(segment.duration,.05)
        self.assertLessEqual(max(report['continuous_peak_velocity_rad_s']),config['velocity_limit_rad_s']+1e-8)
        self.assertLessEqual(max(report['continuous_peak_acceleration_rad_s2']),config['acceleration_limit_rad_s2']+1e-8)
        np.testing.assert_allclose(model.fk(segment.evaluate(segment.duration))[:3,3],goal,atol=1e-5)


if __name__=='__main__':unittest.main()
