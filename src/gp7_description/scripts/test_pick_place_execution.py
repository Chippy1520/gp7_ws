#!/usr/bin/env python3
"""Focused safety/interface checks; fixtures are not physical execution evidence."""
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from gz.msgs10.pose_v_pb2 import Pose_V
from rosgraph_msgs.msg import Clock
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pick_place_feedback import GazeboFeedback,CONTACT_TOPICS
from pick_place_executor import Executor,Reference
from pick_place_planner import joint_segment
from verify_pick_place_controllers import ControllerCheck,JAW_NAMES
from pick_place_config import load_config,NAMES


class ExecutionTests(unittest.TestCase):
    def test_velocity_feedforward_is_bounded_before_any_commands(self):
        e=ControllerCheck.__new__(ControllerCheck);e.config=load_config(profile='fast')
        e.names=NAMES+JAW_NAMES;e.pubs={n:Mock() for n in e.names};e.velocity_pubs={n:Mock() for n in NAMES}
        for velocity in [np.full(8,np.nan),np.ones(8),np.zeros(5)]:
            with self.assertRaisesRegex(ValueError,'Invalid velocity feedforward'):e.publish(np.zeros(8),velocity)
        for p in [*e.pubs.values(),*e.velocity_pubs.values()]:p.publish.assert_not_called()
        e.publish(np.zeros(8),np.zeros(8))
        for p in [*e.pubs.values(),*e.velocity_pubs.values()]:self.assertEqual(p.publish.call_count,1)

    def test_velocity_command_ownership_is_not_bypassed(self):
        e=ControllerCheck.__new__(ControllerCheck);e.q=np.zeros(8);e.sim=0.;e.spin=Mock();e.node=Mock()
        e.names=['joint_s'];e.pubs={'joint_s':Mock()};e.velocity_pubs={'joint_s':Mock()};e.node.get_name.return_value='owner'
        for p in [*e.pubs.values(),*e.velocity_pubs.values()]:p.get_subscription_count.return_value=1
        e.node.get_publishers_info_by_topic.side_effect=lambda topic:[SimpleNamespace(node_name=
            'gp7_sim_bridge' if topic=='/joint_states' else ('foreign' if topic.endswith('cmd_vel_ff') else 'owner'))]
        with self.assertRaisesRegex(RuntimeError,'Competing command owner: .*cmd_vel_ff'):e.ready()

    def test_abort_clears_feedforward_even_without_a_measured_position(self):
        e=Executor.__new__(Executor);e.commanding=True;e.q=None;e.node=Mock();e.velocity_pubs={'joint_s':Mock()}
        e.finish();self.assertEqual(e.velocity_pubs['joint_s'].publish.call_args.args[0].data,0.)

    def test_readiness_waits_for_graph_without_relaxing_ownership(self):
        e=ControllerCheck.__new__(ControllerCheck)
        e.q=np.zeros(8);e.sim=0.;e.spin=Mock();e.node=Mock();e.names=['joint_s'];e.authorized=False
        e.pubs={'joint_s':Mock()};e.pubs['joint_s'].get_subscription_count.return_value=1
        e.node.get_name.return_value='owner'
        def graph(topic):
            if topic=='/joint_states':
                return [] if e.spin.call_count==1 else [SimpleNamespace(node_name='gp7_sim_bridge')]
            return [SimpleNamespace(node_name='owner')]
        e.node.get_publishers_info_by_topic.side_effect=graph
        e.ready();self.assertTrue(e.authorized);self.assertEqual(e.spin.call_count,2)
        e.pubs['joint_s'].publish.assert_not_called()
        e.authorized=False
        e.node.get_publishers_info_by_topic.side_effect=lambda topic:[SimpleNamespace(node_name='gp7_sim_bridge' if topic=='/joint_states' else 'competing')]
        with self.assertRaisesRegex(RuntimeError,'Competing command owner'):e.ready()
        e.finish();e.pubs['joint_s'].publish.assert_not_called()

    def test_contact_readiness_requires_publishers_not_subscribers_or_contact_events(self):
        feedback=GazeboFeedback.__new__(GazeboFeedback);feedback.node=Mock()
        feedback.node.topic_info.return_value=([object()],[])
        self.assertEqual(feedback.require_contact_publishers(timeout=0),list(CONTACT_TOPICS.values()))
        for side,missing in CONTACT_TOPICS.items():
            with self.subTest(side=side):
                feedback.node.topic_info.side_effect=lambda topic:([],[object()]) if topic==missing else ([object()],[])
                with self.assertRaisesRegex(RuntimeError,'Missing Gazebo contact publishers: '+side):
                    feedback.require_contact_publishers(timeout=0)

    def test_carry_settles_on_contact_not_unreachable_preload_position(self):
        executor=Executor.__new__(Executor)
        executor.config={}
        executor.q=np.r_[np.zeros(6),.010,.010];executor.velocity=np.zeros(8)
        executor.jaw_target=np.array([.012,.012]);executor.carrying=True
        executor.sim=0.;executor.pause_since=None;executor.metrics=[]
        executor.refresh=Mock();executor.publish=Mock();executor.native=Mock()
        executor.native.snapshot.side_effect=lambda:{'contacts':{'positive':time.monotonic(),'negative':time.monotonic()},'cube':np.eye(4)}
        def advance_fixture_time():
            executor.sim+=.1;executor.q[4]=.0022
        executor.spin=advance_fixture_time
        executor.follow(joint_segment('LIFT_HOLD_FIXTURE',np.zeros(6),np.zeros(6),.1))
        self.assertEqual(executor.metrics[-1]['phase'],'LIFT_HOLD_FIXTURE')
        self.assertLess(executor.sim,1.)
        np.testing.assert_allclose(executor.hold_position(),np.zeros(6))

    def test_named_pose_uses_individual_simulation_stamp(self):
        feedback=GazeboFeedback.__new__(GazeboFeedback);feedback.lock=threading.Lock()
        feedback.history={'cube':[], 'tcp':[]}
        message=Pose_V();p=message.pose.add();p.name='box_b';p.orientation.w=1.
        p.header.stamp.sec=12;p.header.stamp.nsec=25000000;p.position.z=.025
        feedback._cube(message)
        self.assertEqual(feedback.cube_stamp,12.025)
        self.assertEqual(feedback.cube[2,3],.025)
        previous=feedback.cube.copy();p.name='not_the_cube';feedback._cube(message)
        np.testing.assert_array_equal(feedback.cube,previous)

    def test_pose_histories_are_associated_at_shared_time(self):
        feedback=GazeboFeedback.__new__(GazeboFeedback);feedback.lock=threading.Lock()
        feedback.history={'cube':[], 'tcp':[]}
        feedback.cube=feedback.tcp=None;feedback.cube_stamp=feedback.tcp_stamp=None
        feedback.cube_wall=feedback.tcp_wall=0.;feedback.paused=False;feedback.stats_wall=0.
        feedback.contacts={};feedback.contact_pairs=[]
        for seconds,x in [(1.,0.),(1.08,.08)]:
            message=Pose_V();p=message.pose.add();p.name='box_b';p.orientation.w=1.
            p.header.stamp.sec=1;p.header.stamp.nsec=round((seconds-1)*1e9);p.position.x=x
            feedback._cube(message)
        for seconds in [1.,1.02]:
            message=Pose_V();p=message.pose.add();p.name='gp7::gripper_palm';p.orientation.w=1.
            p.header.stamp.sec=1;p.header.stamp.nsec=round((seconds-1)*1e9)
            feedback._palm(message)
        original=feedback._at
        def outside_callback_lock(history,stamp):
            acquired=feedback.lock.acquire(blocking=False)
            self.assertTrue(acquired,'Interpolation must not block transport callbacks')
            if acquired:feedback.lock.release()
            return original(history,stamp)
        with patch.object(feedback,'_at',side_effect=outside_callback_lock) as interpolate:
            data=feedback.snapshot();again=feedback.snapshot()
            self.assertEqual(interpolate.call_count,2,'Unchanged poses should not be interpolated repeatedly')
            again['cube'][0,3]=100.
            self.assertAlmostEqual(feedback.snapshot()['cube'][0,3],.02)
        self.assertAlmostEqual(data['cube_stamp'],data['tcp_stamp'])
        self.assertAlmostEqual(data['cube_stamp'],1.02)
        self.assertAlmostEqual(data['cube'][0,3],.02)

    def test_feedback_before_first_clock_waits_without_commands(self):
        executor=Executor.__new__(Executor);executor.node=Mock();executor.native=Mock()
        now=time.monotonic()
        executor.native.snapshot.return_value={'paused':False,'stats_wall':now,
            'cube':np.eye(4),'tcp':np.eye(4),'cube_wall':now,'tcp_wall':now,
            'cube_stamp':1.,'tcp_stamp':1.}
        executor.error=None;executor.pause_since=None;executor.sim=None
        executor.q=np.zeros(8);executor.feedback_wall=now;executor.clock_wall=0.
        executor.carrying=False;executor.last_geometry=-1.;executor.checker=Mock()
        with patch('pick_place_executor.rclpy.spin_once'):executor.spin()
        executor.checker.check.assert_not_called()

    def test_read_only_finish_cannot_publish_commands(self):
        executor=Executor.__new__(Executor);executor.commanding=False;executor.node=Mock()
        executor.pubs={'joint_s':Mock()};executor.q=np.zeros(8)
        executor.finish();executor.pubs['joint_s'].publish.assert_not_called()
        executor.node.destroy_node.assert_called_once()

    def test_backward_clock_aborts_without_stale_hold(self):
        executor=Executor.__new__(Executor);executor.sim=10.;executor.error=None
        clock=Clock();clock.clock.sec=1
        ControllerCheck.clock(executor,clock)
        self.assertIn('Backward',executor.error)
        executor.commanding=True;executor.q=np.zeros(8);executor.node=Mock();executor.pubs={'joint_s':Mock()}
        executor.finish();executor.pubs['joint_s'].publish.assert_not_called()

    def test_paused_world_does_not_advance_reference(self):
        executor=Executor.__new__(Executor);executor.node=Mock();executor.native=Mock()
        executor.native.snapshot.return_value={'paused':True,'stats_wall':time.monotonic()}
        executor.error=None;executor.pause_since=None;executor.config={'execution':{'max_clock_pause_wall_s':30}}
        executor.sim=4.;
        with patch('pick_place_executor.rclpy.spin_once'):executor.spin()
        self.assertEqual(executor.sim,4.);self.assertIsNotNone(executor.pause_since)

    def test_reference_combines_real_quintics_without_feedback_fabrication(self):
        arm=joint_segment('CHECK',np.zeros(6),np.ones(6)*.1,4.)
        reference=Reference(arm,[0.,0.],[.01,.01])
        np.testing.assert_allclose(reference.evaluate(0.).position,np.zeros(8),atol=1e-12)
        np.testing.assert_allclose(reference.evaluate(4.).position,np.r_[np.ones(6)*.1,.01,.01],atol=1e-12)
        np.testing.assert_allclose(reference.evaluate(4.).velocity,0.,atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
