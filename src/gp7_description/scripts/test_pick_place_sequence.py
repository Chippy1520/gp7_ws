#!/usr/bin/env python3
"""Offline sequence/API regressions. No real ROS nodes, commands or GUI tests."""
import json
import xml.etree.ElementTree as E
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import task as sequence
import pick_place_motion as backend
import build_pick_place_world as world_builder
from pick_place_executor import Executor
from pick_place_planner import joint_segment


class SequenceTests(unittest.TestCase):
    def test_live_pickup_uses_measured_position_without_relaxing_ground_guard(self):
        config={'cube':{'initial_center_xyz_m':[.45,-.2,.025]}}
        cube=np.eye(4);cube[:3,3]=[.4,-.1,.025]
        with self.assertRaisesRegex(RuntimeError,'not at pickup'):
            backend.accept_pickup(config,cube)
        backend.accept_pickup(config,cube,use_live=True)
        self.assertEqual(config['cube']['initial_center_xyz_m'],[.4,-.1,.025])
        cube[2,3]=.1
        with self.assertRaisesRegex(RuntimeError,'upright cube resting on the ground'):
            backend.accept_pickup(config,cube,use_live=True)
        cube[2,3]=.025;cube[:3,:3]=np.diag([1.,-1.,-1.])
        with self.assertRaisesRegex(RuntimeError,'upright cube resting on the ground'):
            backend.accept_pickup(config,cube,use_live=True)
        cube[0,3]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'Invalid measured cube pose'):
            backend.accept_pickup(config,cube,use_live=True)

    def test_reused_kinematics_still_checks_cube_and_refines_padding(self):
        segment=joint_segment('LIFT',np.zeros(6),np.zeros(6),1.)
        check=Mock(side_effect=[ValueError('conservative padding'),{'safe':True}])
        e=SimpleNamespace(config={},checker=SimpleNamespace(cube_only=True,validate=check),grasp_transform=np.eye(4))
        with patch.object(backend,'validate_segment') as validate:
            r=backend.checked(e,segment,holding=True,kinematic_report={'phase':'LIFT','duration_s':1.})
        validate.assert_not_called();self.assertEqual(check.call_count,2)
        self.assertEqual([c.kwargs['period'] for c in check.call_args_list],[.02,.01])
        self.assertTrue(r['collision_checked'])
        with self.assertRaises(ValueError):
            backend.checked(e,segment,kinematic_report={'phase':'WRONG','duration_s':1.})

    def test_force_profile_is_bounded_and_does_not_change_cube(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'trial.sdf'
            world_builder.build(output_path=output,grip_force_n=4.)
            world=E.parse(output).getroot().find('world')
            self.assertEqual(float(world.findtext("model[@name='box_b']/link/inertial/mass")),.1)
            robot=world.find("model[@name='gp7']")
            for side in ['positive','negative']:
                controller=next(p for p in robot.findall('plugin') if p.findtext('joint_name')=='gripper_joint_'+side)
                self.assertEqual(float(controller.findtext('p_gain')),2000.)
                self.assertEqual(float(controller.findtext('i_gain')),0.)
                self.assertEqual(float(controller.findtext('cmd_max')),4.)
                self.assertEqual(float(controller.findtext('cmd_min')),-4.)
            for force in [0.,-1.,11.,float('nan')]:
                with self.assertRaises(ValueError):world_builder.set_grip_preload(robot,force)

    def test_high_level_order_and_separate_delays(self):
        robot = Mock(stop_after_lift=False)
        sequence.pick_and_place(robot)
        self.assertEqual([c[0] for c in robot.method_calls], [
            'raise_if_needed', 'delay', 'approach_pickup', 'delay',
            'descend_to_cube', 'delay', 'grasp', 'delay', 'lift', 'delay',
            'transfer_polar', 'delay', 'lower_onto_pedestal', 'delay',
            'release', 'delay', 'retreat', 'verify_placement'])
        self.assertEqual([c.args[0] for c in robot.delay.call_args_list],
                         [.5, .5, .5, 1., 1., 1., 1., .5])

    def test_stop_after_lift_does_not_transfer_or_release(self):
        robot = Mock(stop_after_lift=True)
        sequence.pick_and_place(robot)
        robot.lift.assert_called_once()
        robot.transfer_polar.assert_not_called()
        robot.release.assert_not_called()

    def motion(self, executor):
        if not isinstance(getattr(executor,'config',None),dict):executor.config={}
        return backend.GazeboMotion(executor, sequence.TIMINGS, {}, .25, .05, .1, False)

    def test_short_assisted_hold_is_stretched_not_jaw_rate_clipped(self):
        e=SimpleNamespace(q=np.r_[np.zeros(6),[.00006,.00006]],refresh=Mock(),sim=0.,carrying=False,
                          attachment=Mock(),jaw_target=np.zeros(2),spin=Mock(side_effect=RuntimeError('reference captured')))
        hold=joint_segment('DELAY',np.zeros(6),np.zeros(6),.0025)
        with self.assertRaisesRegex(RuntimeError,'reference captured'):Executor.follow(e,hold)
        self.assertGreater(e.curve.duration,.0025)
        times=np.linspace(0,e.curve.duration,101)
        self.assertLessEqual(max(np.max(abs(e.curve.evaluate(t).velocity[6:])) for t in times),.016)
        np.testing.assert_allclose(e.curve.segment.evaluate(times),0.)

    def test_profile_delay_still_runs_a_guarded_hold(self):
        e=SimpleNamespace(config={'execution':{'delay_scale':.02}},follow=Mock(),hold_position=Mock(return_value=np.zeros(6)))
        with patch.object(backend,'fresh'):self.motion(e).delay(.5)
        self.assertAlmostEqual(e.follow.call_args.args[0].duration,.5*.02)

    def test_delay_uses_guarded_constant_joint_reference(self):
        e = SimpleNamespace(q=np.r_[np.zeros(6), [.01, .01]], follow=Mock(),
                            hold_position=Mock(return_value=np.zeros(6)))
        with patch.object(backend, 'fresh'):
            self.motion(e).delay(.5)
        segment = e.follow.call_args.args[0]
        self.assertEqual((segment.name, segment.duration), ('DELAY', .5))
        np.testing.assert_allclose(segment.evaluate([0., .5]), 0.)
        self.assertEqual(e.follow.call_args.kwargs, {})
        self.assertEqual(len(e.follow.call_args.args), 1)  # Retain jaw target.

    def test_delay_propagates_guard_failure(self):
        e = SimpleNamespace(q=np.zeros(8), follow=Mock(side_effect=RuntimeError('slip')),
                            hold_position=Mock(return_value=np.zeros(6)))
        with patch.object(backend, 'fresh'), self.assertRaisesRegex(RuntimeError, 'slip'):
            self.motion(e).delay(.5)

    def test_invalid_and_zero_delays_cannot_command(self):
        e = Mock()
        for value in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.motion(e).delay(value)
        self.motion(e).delay(0.)
        e.follow.assert_not_called()

    def test_stage_order_rejected_before_motion(self):
        e = Mock()
        motion = self.motion(e)
        motion.pending = ['APPROACH_PICKUP']
        with self.assertRaisesRegex(RuntimeError, 'Unsafe stage order'):
            motion.descend_to_cube()
        e.follow.assert_not_called()

    def test_delay_does_not_accumulate_controller_tracking_bias(self):
        bias=np.zeros(8);bias[4]=.0022
        e=SimpleNamespace(q=bias.copy(), follow=Mock(), refresh=Mock(), sim=0.,
                          carrying=False, jaw_target=np.zeros(2),
                          hold_position=Mock(return_value=np.zeros(6)),
                          spin=Mock(side_effect=RuntimeError('passed start guard')))
        with patch.object(backend,'fresh'):
            self.motion(e).delay(.5)
        hold=e.follow.call_args.args[0]
        # A biased controller follows the hold target with the same small error.
        e.q[:6]=hold.evaluate(hold.duration)+bias[:6]
        descent=joint_segment('DESCEND_PICK',np.zeros(6),np.zeros(6),1.)
        with self.assertRaisesRegex(RuntimeError,'passed start guard'):
            Executor.follow(e,descent)
        np.testing.assert_allclose(hold.evaluate(0.),np.zeros(6))

    def test_delay_still_rejects_excessive_drift_before_commands(self):
        q=np.zeros(8);q[4]=.0042
        e=SimpleNamespace(q=q,refresh=Mock(),publish=Mock(),
                          hold_position=Mock(return_value=np.zeros(6)))
        e.follow=lambda segment:Executor.follow(e,segment)
        with patch.object(backend,'fresh'), self.assertRaisesRegex(RuntimeError,'0.003 rad'):
            self.motion(e).delay(.5)
        e.publish.assert_not_called()

    def test_hold_position_returns_reference_copy_or_initial_measurement(self):
        e=Executor.__new__(Executor)
        e.q=np.full(8,.0022);e.arm_target=np.zeros(6)
        target=e.hold_position();target[0]=1.
        np.testing.assert_allclose(e.arm_target,0.)
        np.testing.assert_allclose(e.hold_position(),0.)
        e.arm_target=None
        np.testing.assert_allclose(e.hold_position(),e.q[:6])

    def test_grasp_and_release_hold_the_preceding_reference(self):
        e=SimpleNamespace(q=np.full(8,.0022),hold_position=Mock(return_value=np.zeros(6)),
                          follow=Mock(side_effect=RuntimeError('reference captured')))
        with self.assertRaisesRegex(RuntimeError,'reference captured'):
            Executor.grasp(e)
        np.testing.assert_allclose(e.follow.call_args.args[0].evaluate(0.),0.)
        e.follow=Mock();e.supported=Mock();e.detach_supported=Mock()
        motion=self.motion(e);motion.pending=['RELEASE']
        with patch.object(backend,'fresh'):
            motion.release()
        e.supported.assert_called_once()
        np.testing.assert_allclose(e.follow.call_args.args[0].evaluate(0.),0.)

    def test_carry_preserves_preload_but_release_starts_at_measured_jaws(self):
        for carrying, goal, expected in [(True, .012, .012), (False, 0., .010)]:
            e = SimpleNamespace(q=np.r_[np.zeros(6), [.010, .010]],
                                refresh=Mock(), sim=0., carrying=carrying,
                                jaw_target=np.array([goal, goal]),
                                spin=Mock(side_effect=RuntimeError('reference captured')))
            segment = joint_segment('HOLD', np.zeros(6), np.zeros(6), 1.)
            with self.assertRaisesRegex(RuntimeError, 'reference captured'):
                Executor.follow(e, segment)
            np.testing.assert_allclose(e.curve.evaluate(0.).position[6:], expected)

    def test_read_only_mode_never_calls_sequence_and_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            e = SimpleNamespace(ready=Mock(), wait_native=Mock(return_value={}),
                                native=SimpleNamespace(require_contact_publishers=Mock(return_value=[])),checker=SimpleNamespace(),
                                q=np.zeros(8), metrics=[], rows=[], observations=[], finish=Mock())
            callback = Mock()
            argv = ['sequence', '--output', directory]
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, {
                'ROS_DOMAIN_ID': '77', 'GZ_PARTITION': 'gp7_project_lab'}), \
                    patch.object(backend.rclpy, 'init'), patch.object(backend.rclpy, 'shutdown'), \
                    patch.object(backend, 'live_audit', return_value={}), \
                    patch.object(backend, 'controller_gate', return_value=True), \
                    patch.object(backend, 'Executor', return_value=e), \
                    patch.object(backend.GazeboMotion, 'prepare'):
                backend.run(callback, timings=sequence.TIMINGS, pickup_height=.25,
                            hover_clearance=.05, retreat_distance=.1)
            callback.assert_not_called()
            result = json.loads((Path(directory)/'result.json').read_text())
            self.assertEqual(result['status'], 'READY_APPROACH_AND_DESCENT')
            self.assertTrue(e.checker.cube_only)
            self.assertEqual(e.support,'ground')
            e.finish.assert_called_once()

    def test_mocked_interrupt_holds_before_shutdown_and_always_writes_report(self):
        for broken_cleanup in [False,True]:
            with self.subTest(cleanup_failure=broken_cleanup), tempfile.TemporaryDirectory() as directory:
                events=[]
                def finish():
                    events.append('finish')
                    if broken_cleanup:raise RuntimeError('mocked cleanup failure')
                e=SimpleNamespace(ready=Mock(),wait_native=Mock(return_value={}),refresh=Mock(),sim=3.,
                    native=SimpleNamespace(require_contact_publishers=Mock(return_value=[])),carrying=True,checker=SimpleNamespace(),
                    q=np.r_[np.zeros(6),[.010,.010]],metrics=[],rows=[],observations=[],finish=finish)
                def stop(motion):
                    os.kill(os.getpid(),backend.signal.SIGINT)
                    events.append('cooperative flag')
                    self.assertTrue(e.abort_requested())
                    Executor.spin(e)
                with patch.object(sys,'argv',['sequence','--execute','--output',directory]), \
                        patch.dict(os.environ,{'ROS_DOMAIN_ID':'77','GZ_PARTITION':'gp7_project_lab'}), \
                        patch.object(backend.rclpy,'init') as init, \
                        patch.object(backend.rclpy,'shutdown',side_effect=lambda:events.append('shutdown')), \
                        patch.object(backend,'service',side_effect=lambda *a:events.append('pause') or 'data: true'), \
                        patch.object(backend,'live_audit',return_value={}),patch.object(backend,'controller_gate',return_value=True), \
                        patch.object(backend,'Executor',return_value=e),patch.object(backend.GazeboMotion,'prepare'):
                    with self.assertRaises(RuntimeError if broken_cleanup else KeyboardInterrupt):
                        backend.run(stop,timings=sequence.TIMINGS,pickup_height=.25,hover_clearance=.05,retreat_distance=.1)
                init.assert_called_once_with(signal_handler_options=backend.SignalHandlerOptions.NO)
                self.assertEqual(events,['cooperative flag','pause','finish','shutdown'])
                result=json.loads((Path(directory)/'result.json').read_text())
                self.assertTrue(result['physics_paused_after_abort'])
                self.assertTrue(result['operator_stop_requested'])
                self.assertEqual(result['status'],'FAIL' if broken_cleanup else 'ABORTED')
                self.assertEqual('cleanup_error' in result,broken_cleanup)

    def test_assisted_sprint_waits_do_not_leak_into_physical_mode(self):
        for assisted in [False,True]:
            with self.subTest(assisted=assisted),tempfile.TemporaryDirectory() as directory:
                e=SimpleNamespace(ready=Mock(),wait_native=Mock(return_value={}),
                    native=SimpleNamespace(require_contact_publishers=Mock(return_value=[])),checker=SimpleNamespace(),
                    q=np.zeros(8),metrics=[],rows=[],observations=[],finish=Mock())
                argv=['sequence','--profile','sprint','--output',directory]+(['--detachable-grasp'] if assisted else [])
                with patch.object(sys,'argv',argv),patch.dict(os.environ,{'ROS_DOMAIN_ID':'77','GZ_PARTITION':'gp7_project_lab'}), \
                        patch.object(backend.rclpy,'init'),patch.object(backend.rclpy,'shutdown'), \
                        patch.object(backend,'live_audit',return_value={}),patch.object(backend,'controller_gate',return_value=True), \
                        patch.object(backend,'Executor',return_value=e),patch.object(backend.GazeboMotion,'prepare'):
                    backend.run(Mock(),timings=sequence.TIMINGS,pickup_height=.25,hover_clearance=.05,retreat_distance=.1)
                report=json.loads((Path(directory)/'result.json').read_text());settings=report['execution_settings']
                self.assertEqual(settings['delay_scale'],.005 if assisted else .02)
                self.assertEqual(settings['settle_duration_s'],.05 if assisted else .1)
                self.assertEqual(settings['grasp_contact_hold_s'],.12 if assisted else .25)
                self.assertEqual(report['rate_limits'],{'velocity_limit_rad_s':2.5,'acceleration_limit_rad_s2':8.})
                self.assertEqual(report['timings']['grasp'],1.5)

    def test_carry_is_planned_fresh_from_the_measured_grasp(self):
        segment=joint_segment('LIFT',np.zeros(6),np.zeros(6),1.)
        e=SimpleNamespace(config={},q=np.zeros(8),grasp_transform=np.eye(4),follow=Mock())
        motion=self.motion(e);motion.pending=['LIFT']
        with patch.object(backend,'fresh'),patch.object(motion,'_save'), \
                patch.object(backend,'carry',return_value=([segment],[{'phase':'LIFT'}])) as plan:
            motion.lift()
        plan.assert_called_once()
        self.assertIs(plan.call_args.args[0],e)
        self.assertEqual(motion.report['carry_plan_source'],'fresh measured grasp')
        e.follow.assert_called_once()

    def test_missing_contact_publishers_block_before_planning_or_execution(self):
        for late in [False,True]:
            with self.subTest(after_planning=late), tempfile.TemporaryDirectory() as directory:
                check=Mock(side_effect=([[],RuntimeError('Missing Gazebo contact publishers')]
                                       if late else RuntimeError('Missing Gazebo contact publishers')))
                e=SimpleNamespace(ready=Mock(),wait_native=Mock(return_value={}),
                                  native=SimpleNamespace(require_contact_publishers=check),carrying=False,checker=SimpleNamespace(),
                                  q=np.zeros(8),metrics=[],rows=[],observations=[],finish=Mock())
                callback=Mock()
                with patch.object(sys,'argv',['sequence','--execute','--output',directory]), \
                        patch.dict(os.environ,{'ROS_DOMAIN_ID':'77','GZ_PARTITION':'gp7_project_lab'}), \
                        patch.object(backend.rclpy,'init'),patch.object(backend.rclpy,'shutdown'), \
                        patch.object(backend,'live_audit',return_value={}), \
                        patch.object(backend,'controller_gate',return_value=True), \
                        patch.object(backend,'Executor',return_value=e), \
                        patch.object(backend.GazeboMotion,'prepare') as prepare:
                    with self.assertRaisesRegex(RuntimeError,'Missing Gazebo contact publishers'):
                        backend.run(callback,timings=sequence.TIMINGS,pickup_height=.25,
                                    hover_clearance=.05,retreat_distance=.1)
                self.assertEqual(prepare.call_count,int(late))
                callback.assert_not_called();e.finish.assert_called_once()
                self.assertEqual(json.loads((Path(directory)/'result.json').read_text())['status'],'FAIL')

    def test_shortened_calibrated_carry_targets_and_scope(self):
        config = backend.load_config()
        cube = np.eye(4)
        cube[:3, 3] = config['cube']['initial_center_xyz_m']
        grasp = np.diag([1., -1., -1., 1.])
        grasp[:3, 3] = [.001, .002, .00007]
        e = SimpleNamespace(q=np.zeros(8), config=config, grasp_transform=grasp,
                            native=Mock(), model=Mock())
        e.native.snapshot.return_value = {'cube': cube}
        e.model.fk.return_value = cube @ np.linalg.inv(grasp)
        paths = []
        def build(name, q, path, config):
            paths.append(path)
            return joint_segment(name, q, q, path.duration), {}
        with patch.object(backend, 'differential_segment', side_effect=build), \
                patch.object(backend, 'checked', return_value={}):
            phases, _ = backend.carry(e, True, sequence.TIMINGS, .05, .1)
            self.assertEqual([s.name for s in phases], ['LIFT'])
            paths.clear()
            phases, _ = backend.carry(e, False, sequence.TIMINGS, .05, .1)
        self.assertEqual([(s.name, s.duration) for s in phases],
                         [(name,sequence.TIMINGS[name.lower()]) for name in ['LIFT','TRANSFER','LOWER','RETREAT']])
        self.assertTrue(paths[1].polar)
        placed = cube.copy()
        placed[:3, 3] = config['placement']['center_xyz_m']
        np.testing.assert_allclose(paths[2].goal, (placed @ np.linalg.inv(grasp))[:3, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
