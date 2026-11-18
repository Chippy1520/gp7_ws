#!/usr/bin/env python3
"""Offline assist protocol tests; not physical or timing evidence."""
import json
import tempfile
import time
import unittest
import xml.etree.ElementTree as E
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from pick_place_attachment import DetachableGrasp
from pick_place_executor import Executor
from build_pick_place_world import build


class AttachmentTests(unittest.TestCase):
    def test_fresh_ack_required_not_cached_true(self):
        node=Mock();a=DetachableGrasp(node);a._state(Boolean(data=True))
        node.advertise.return_value.publish.side_effect=lambda msg:a._state(msg)
        guard=Mock();a.set(True,guard)
        guard.assert_called();node.advertise.return_value.publish.assert_called_once()
        self.assertTrue(a.attached());a.received=time.monotonic()-1;self.assertFalse(a.attached())

    def test_attach_is_after_confirmed_grasp_and_holds_measured_jaw_travel(self):
        events=[];e=Executor.__new__(Executor);e.hold_position=Mock(return_value=np.zeros(6))
        e.follow=Mock(side_effect=lambda *a,**k:events.append('settled-contact-grasp'))
        e.spin=Mock();e.attachment=Mock();e.attachment.set.side_effect=lambda *a:events.append('attach')
        e.q=np.r_[np.zeros(6),.010,.010];e.model=Mock();e.model.fk.return_value=np.eye(4)
        e.native=Mock();e.native.snapshot.return_value={'cube':np.eye(4),'contact_pairs':[]};e.metrics=[]
        e.grasp();self.assertEqual(events,['settled-contact-grasp','attach'])
        self.assertTrue(e.carrying);np.testing.assert_equal(e.jaw_target,[.010,.010])

    def test_bad_grasp_never_attaches(self):
        e=Executor.__new__(Executor);e.hold_position=Mock(return_value=np.zeros(6));e.follow=Mock(side_effect=RuntimeError('no contacts'))
        e.attachment=Mock()
        with self.assertRaisesRegex(RuntimeError,'no contacts'):e.grasp()
        e.attachment.set.assert_not_called()

    def test_no_support_means_no_detach(self):
        e=Executor.__new__(Executor);e.attachment=Mock();e.carrying=True;e.supported=Mock(side_effect=RuntimeError('unsupported'))
        with self.assertRaisesRegex(RuntimeError,'unsupported'):e.detach_supported()
        e.attachment.set.assert_not_called();self.assertTrue(e.carrying)

    def test_sprint_kinematic_player_and_jaw_reference_limits(self):
        from pick_place_profiles import PROFILES
        with tempfile.TemporaryDirectory() as d:
            p=build(output_path=Path(d)/'world.sdf',profile='sprint',grip_force_n=8,detachable_grasp=True)
            world=E.parse(p).getroot().find('world');robot=world.find("model[@name='gp7']")
            self.assertEqual(float(world.findtext('physics/max_step_size')),.001)
            self.assertFalse(any(x.get('name')=='gz::sim::systems::Physics' for x in world.findall('plugin')))
            player=robot.find("plugin[@name='gp7::KinematicPlayback']")
            for name in ['gripper_joint_positive','gripper_joint_negative']:
                j=robot.find(f"joint[@name='{name}']")
                self.assertEqual(float(j.findtext('axis/limit/lower')),0.)
                entry=next(x for x in player.findall('kinematic_joint') if x.findtext('name')==name)
                self.assertEqual(float(entry.findtext('speed')),.016)
                self.assertFalse(any(x.findtext('joint_name')==name for x in robot.findall('plugin')))
            self.assertLessEqual(1.875*.012/PROFILES['sprint']['timings']['grasp'],.016)
            self.assertLessEqual(1.875*.0101/PROFILES['sprint']['timings']['release'],.016)

    def test_assist_is_not_silently_added_to_physical_world(self):
        with tempfile.TemporaryDirectory() as d:
            for active in [False,True]:
                p=build(output_path=Path(d)/f'{active}.sdf',detachable_grasp=active)
                self.assertEqual('gp7::KinematicPlayback' in p.read_text(),active)
                self.assertNotIn('gp7::DetachableGrasp',p.read_text())
                self.assertEqual(json.loads(p.with_suffix('.json').read_text())['detachable_grasp_assist'],active)


class AssistedReleaseTests(unittest.TestCase):
    def test_opening_uses_ramp_not_carry_preload_shortcut(self):
        from unittest.mock import patch
        import numpy as np
        from pick_place_executor import Executor
        from pick_place_planner import joint_segment
        e=Executor.__new__(Executor);e.refresh=lambda:None;e.q=np.r_[np.zeros(6),[.010,.010]]
        e.carrying=True;e.jaw_target=np.array([.010,.010]);e.sim=0.
        segment=joint_segment('RELEASE',np.zeros(6),np.zeros(6),1.2)
        with patch('pick_place_executor.Reference',side_effect=RuntimeError('capture')) as reference:
            with self.assertRaisesRegex(RuntimeError,'capture'):e.follow(segment,[0.,0.])
        np.testing.assert_allclose(reference.call_args.args[1],[.010,.010])
        np.testing.assert_allclose(reference.call_args.args[2],[0.,0.])

    def test_support_then_open_then_detach(self):
        from unittest.mock import Mock,patch
        import numpy as np
        from pick_place_motion import GazeboMotion
        events=[]
        e=SimpleNamespace(attachment=object(),carrying=True,supported=lambda:events.append('support'),hold_position=lambda:np.zeros(6))
        def opened(*args):
            self.assertTrue(e.carrying);events.append('open')
        e.follow=opened;e.detach_supported=lambda:events.append('detach')
        m=GazeboMotion.__new__(GazeboMotion);m.e=e;m.timings={'release':1.2};m.pending=['RELEASE'];m._expect=Mock()
        with patch('pick_place_motion.fresh'):m.release()
        self.assertEqual(events,['support','open','detach'])

if __name__=='__main__':unittest.main()
