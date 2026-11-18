#!/usr/bin/env python3
"""Geometry regressions: real FCL/CAD, contact exceptions and negative cases."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kinematics import ExportedModel
from unittest.mock import patch
import xml.etree.ElementTree as E
from pick_place_collision import CollisionChecker, link_fk, NAMES, PKG
from pick_place_planner import Segment, spline_from_states
from pick_place_planner import top_down_ik


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.checker=CollisionChecker();cls.model=ExportedModel()

    def test_home_and_task_poses_are_clear(self):
        for q in [np.zeros(6),*[top_down_ik(p) for p in [[.45,-.2,.025],[.45,-.2,.325],[.5,.5,.225],[.5,.5,.325]]]]:
            result=self.checker.check(q)
            self.assertTrue(result['safe'],result['violations'])

    def test_cube_in_palm_is_detected(self):
        q=np.zeros(6);cube=self.model.fk(q);cube[:3,3]-=.06*cube[:3,2]
        result=self.checker.check(q,cube_transform=cube)
        self.assertFalse(result['safe'])
        self.assertTrue(any('gripper_palm' in v for v in result['violations']))

    def test_support_exceptions_cannot_hide_penetration_or_wrong_footprint(self):
        cube=np.eye(4);cube[:3,3]=[.45,-.2,.010]
        result=self.checker.check(np.zeros(6),cube_transform=cube,holding=True,support='ground')
        self.assertIn(('invalid_support_exception','ground'),result['violations'])
        cube[:3,3]=[.70,.5,.225]
        result=self.checker.check(np.zeros(6),cube_transform=cube,holding=True,support='box_a')
        self.assertIn(('invalid_support_exception','box_a'),result['violations'])

    def test_full_mode_still_checks_global_self_and_world_pairs(self):
        from pick_place_collision import collision_object
        c=self.checker;q=np.zeros(6)
        frames=link_fk(c.urdf,dict(zip(NAMES,q)))
        # A synthetic static obstacle at the palm must still be checked even
        # though the actual cube remains safely resting far away.
        geometry,local=c.boxes['gripper_palm']
        with patch.dict(c.environment,ground=collision_object(geometry,frames['gripper_palm']@local)):
            result=c.check(q)
        self.assertTrue(any(v[:3]==('environment','gripper_palm','ground') for v in result['violations']))
        # Non-adjacent shapes deliberately superposed; use actual FCL boxes,
        # bypassing CAD refinement solely for this synthetic geometry fixture.
        geometry,local=c.boxes['base_link']
        overlap=np.linalg.inv(frames['link_u'])@frames['base_link']@local
        with patch.dict(c.boxes,link_u=(geometry,overlap)),patch.object(c,'meshes',{}):
            result=c.check(q)
        self.assertTrue(any(v[:3]==('self','base_link','link_u') for v in result['violations']))
        self.assertFalse(c.cube_only)

    def test_invalid_inputs_reject_before_fcl(self):
        with self.assertRaises(ValueError):self.checker.check([0,0])
        invalid=np.eye(4);invalid[0,0]=float('nan')
        with self.assertRaises(ValueError):self.checker.check(np.zeros(6),cube_transform=invalid)
        with self.assertRaises(ValueError):self.checker.check(np.zeros(6),support='ignore_everything')


class CubeOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker=CollisionChecker(cube_only=True)
        cls.q=np.zeros(6)
        cls.frames=link_fk(cls.checker.urdf,dict(zip(NAMES,[0.]*6)))

    @staticmethod
    def cube(x,y,z):
        T=np.eye(4);T[:3,3]=[x,y,z];return T

    def test_resting_support_is_controlled_and_checks_unheld_cube(self):
        c=self.checker
        self.assertTrue(c.check(self.q)['safe'])
        ground=self.cube(.45,-.2,.025)
        self.assertFalse(c.check(self.q,cube_transform=ground)['safe'])
        self.assertTrue(c.check(self.q,cube_transform=ground,support='ground')['safe'])
        pedestal=self.cube(.5,.5,.225)
        self.assertFalse(c.check(self.q,cube_transform=pedestal)['safe'])
        self.assertTrue(c.check(self.q,cube_transform=pedestal,support='box_a')['safe'])
        for T,support in [(self.cube(.5,.5,.15),'ground'),(self.cube(.45,-.2,-.1),'box_a')]:
            self.assertFalse(c.check(self.q,cube_transform=T,support=support)['safe'])

    def test_support_depth_rotation_and_footprint(self):
        for holding in (False,True):
            for T,support in [(self.cube(.45,-.2,.01),'ground'),
                              (self.cube(.5,.5,.21),'box_a'),
                              (self.cube(.64,.5,.225),'box_a')]:
                result=self.checker.check(self.q,cube_transform=T,holding=holding,support=support)
                self.assertIn(('invalid_support_exception',support),result['violations'])
        tiny=self.cube(.45,-.2,.025-5e-6)
        self.assertTrue(self.checker.check(self.q,cube_transform=tiny,support='ground')['safe'])
        rotated=self.cube(.5,.5,.225)
        angle=np.pi/4;rotated[:3,:3]=[[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]
        self.assertIn(('invalid_support_exception','box_a'),self.checker.check(
            self.q,cube_transform=rotated,support='box_a')['violations'])
        for tolerance in (np.nan,.1,-1):
            with self.assertRaises(ValueError):
                self.checker.check(self.q,support_penetration_tolerance=tolerance)

    def test_arm_and_palm_never_get_contact_exception(self):
        for name in ('gripper_palm','link_u'):
            geometry,local=self.checker.cube_robot_shapes[name][0]
            T=self.frames[name]@local
            for options in ({},{'holding':True},{'contact_allowed':True}):
                result=self.checker.check(self.q,cube_transform=T,**options)
                self.assertTrue(any(name in v for v in result['violations']),result)

    def test_finger_exception_only_when_explicitly_allowed(self):
        name='gripper_finger_positive'
        T=self.frames[name]@self.checker.cube_robot_shapes[name][0][1]
        result=self.checker.check(self.q,cube_transform=T)
        self.assertTrue(any(name in v for v in result['violations']))
        for options in ({'holding':True},{'contact_allowed':True}):
            result=self.checker.check(self.q,cube_transform=T,**options)
            self.assertFalse(any(name in v for v in result['violations']))

    def test_extra_pad_shapes_are_loaded_and_checked(self):
        # Modify only an in-memory world; no generated world files are touched.
        world=E.parse(PKG/'worlds/gp7_pick_place.sdf')
        link=world.getroot().find("world/model[@name='gp7']/link[@name='gripper_palm']")
        E.SubElement(link,'collision',name='extra_pad').extend([
            E.fromstring('<pose>2 2 2 0 0 0</pose>'),
            E.fromstring('<geometry><box><size>.02 .02 .02</size></box></geometry>')])
        original_parse=E.parse
        def parse(path):
            return original_parse(path) if str(path).endswith('.urdf') else world
        with patch('pick_place_collision.E.parse',side_effect=parse):
            checker=CollisionChecker(cube_only=True)
        self.assertEqual(len(checker.cube_robot_shapes['gripper_palm']),2)
        T=self.frames['gripper_palm']@checker.cube_robot_shapes['gripper_palm'][1][1]
        result=checker.check(self.q,cube_transform=T,holding=True)
        self.assertTrue(any('gripper_palm' in v for v in result['violations']))

    def test_mode_skips_only_global_pairs_and_keeps_padding(self):
        import pick_place_collision as collision
        with patch.object(collision,'separation',wraps=collision.separation) as measured:
            result=self.checker.check(self.q)
        expected=sum(map(len,self.checker.cube_robot_shapes.values()))+2
        self.assertEqual(measured.call_count,expected)
        self.assertIn('excludes robot self/world checks',result['scope'])
        # Move a cube just outside the base box: inflation must still reject it.
        local=self.checker.cube_robot_shapes['base_link'][0][1]
        size=self.checker.cube_robot_shapes['base_link'][0][0].side
        T=self.frames['base_link']@local
        T[:3,3]+=T[:3,0]*(size[0]/2+.025+.01)
        self.assertFalse(any('base_link' in v for v in self.checker.check(self.q,cube_transform=T)['violations']))
        self.assertTrue(any('base_link' in v for v in self.checker.check(
            self.q,cube_transform=T,motion_padding=.02)['violations']))

    def test_validate_20ms_keeps_joint_rate_bound_and_scope(self):
        end=np.array([.001,0,0,0,0,0]);zero=np.zeros(6)
        spline=spline_from_states([0,1],[zero,end],[zero,zero],[zero,zero])
        segment=Segment('small_safe_move',1.,spline)
        result=self.checker.validate(segment,period=.02)
        self.assertEqual(result['collision_samples'],51)
        self.assertGreater(result['between_sample_point_displacement_bound_m'],0)
        self.assertIn('cube_only',result['scope'])
        finer=self.checker.validate(segment,period=.005)
        self.assertAlmostEqual(result['between_sample_point_displacement_bound_m'],
                               4*finer['between_sample_point_displacement_bound_m'])


if __name__=='__main__':unittest.main(verbosity=2)
