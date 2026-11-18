#!/usr/bin/env python3
"""Pure serialization and mocked native transport tests; no Gazebo motion."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import BPoly
from gz.msgs10.joint_trajectory_pb2 import JointTrajectory
from gz.msgs10.stringmsg_pb2 import StringMsg
from joint_trajectories import TrajectorySample, quintic_given_duration
import pick_place_playback as playback


def quintic(duration=1.0):
    return quintic_given_duration(np.arange(8.), np.arange(8.) + 1, duration,
                                 start_velocity=.1, end_velocity=-.2,
                                 start_acceleration=.3, end_acceleration=-.4)


def reference(knots, duration=1.):
    base = quintic(duration)
    boundaries = [[getattr(base.evaluate(t), field) for field in
                   ('position', 'velocity', 'acceleration')] for t in knots]
    spline = BPoly.from_derivatives(knots, boundaries)
    return SimpleNamespace(duration=duration, segment=SimpleNamespace(spline=spline),
                           evaluate=lambda t: TrajectorySample(
                               spline(t), spline(t, 1), spline(t, 2), None))


class MessageTests(unittest.TestCase):
    def test_native_round_trip_and_quintic_boundary_derivatives(self):
        curve = quintic(2.5)
        msg = playback.trajectory_message(curve, 12.25, segment_id='unique')
        self.assertIsInstance(msg, JointTrajectory)
        decoded = JointTrajectory.FromString(msg.SerializeToString())
        self.assertEqual(decoded, msg)
        self.assertEqual(list(msg.joint_names), list(playback.JOINT_NAMES))
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nsec), (12, 250000000))
        self.assertEqual(msg.header.data[0].key, 'segment_id')
        self.assertEqual(list(msg.header.data[0].value), ['unique'])
        self.assertEqual(len(msg.points), 2)
        for point, t in zip(msg.points, [0., 2.5]):
            sample = curve.evaluate(t)
            for target, source in [('positions', 'position'), ('velocities', 'velocity'),
                                   ('accelerations', 'acceleration')]:
                np.testing.assert_allclose(getattr(point, target), getattr(sample, source))

    def test_unequal_knots_preserved_and_missing_boundaries_added(self):
        curve = reference([0., .07, .6, 1.])
        for knots in ([0., .07, .6, 1.], [.07, .6]):
            curve.segment.spline = SimpleNamespace(x=knots)
            msg = playback.trajectory_message(curve, 0.)
            self.assertEqual([p.time_from_start.sec * 10**9 + p.time_from_start.nsec
                              for p in msg.points], [0, 70000000, 600000000, 1000000000])
            for point, t in zip(msg.points, [0., .07, .6, 1.]):
                np.testing.assert_allclose(point.positions, curve.evaluate(t).position)

    def test_nanosecond_rounding_carries_in_header_and_native_duration(self):
        msg = playback.trajectory_message(quintic(1.9999999996), 7.9999999996)
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nsec), (8, 0))
        stamp = msg.points[-1].time_from_start
        self.assertEqual(stamp.DESCRIPTOR.full_name, 'gz.msgs.Duration')
        self.assertEqual((stamp.sec, stamp.nsec), (2, 0))
        msg = playback.trajectory_message(quintic(1.1234567894), .1234567894)
        self.assertEqual(msg.header.stamp.nsec, 123456789)
        self.assertEqual(msg.points[-1].time_from_start.nsec, 123456789)

    def test_invalid_start_duration_and_knots(self):
        for start in (-1, float('nan'), float('inf')):
            with self.subTest(start=start), self.assertRaises(ValueError):
                playback.trajectory_message(quintic(), start)
        for duration in (0., -1., float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                playback.trajectory_message(SimpleNamespace(duration=duration), 0.)
        for knots in ([0., .4, .4, 1.], [0., .7, .3, 1.], [-.1, 1.],
                      [0., 1.1], [0., float('nan')], [0., float('inf')], [0., 1e-12, 1.]):
            curve = SimpleNamespace(duration=1., segment=SimpleNamespace(spline=SimpleNamespace(x=knots)))
            with self.subTest(knots=knots), self.assertRaises(ValueError):
                playback.trajectory_message(curve, 0.)

    def test_invalid_sample_fields(self):
        for field in ('position', 'velocity', 'acceleration', 'jerk'):
            for invalid in ([0.] * 7, [0.] * 7 + [float('nan')], [float('inf')] * 8):
                values = dict(position=np.zeros(8), velocity=np.zeros(8),
                              acceleration=np.zeros(8), jerk=np.zeros(8))
                values[field] = invalid
                curve = SimpleNamespace(duration=1., evaluate=lambda t: SimpleNamespace(**values))
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    playback.trajectory_message(curve, 0.)

    def test_custom_names_and_identifier_validation(self):
        names = [f'j{i}' for i in range(8)]
        self.assertEqual(list(playback.trajectory_message(quintic(), 0, names).joint_names), names)
        for bad in ([], ['j'] * 8, names[:-1], names[:-1] + ['']):
            with self.assertRaises(ValueError):
                playback.trajectory_message(quintic(), 0, bad)
        with self.assertRaises(ValueError):
            playback.trajectory_message(quintic(), 0, segment_id='')


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.node = Mock()
        self.pub, self.hold_pub = Mock(), Mock()
        self.pub.publish.return_value = self.hold_pub.publish.return_value = True
        self.node.advertise.side_effect = [self.pub, self.hold_pub]
        self.client = playback.PlaybackClient(self.node)
        self.callback = self.node.subscribe.call_args.args[2]

    def status(self, state, identifier=None, **extra):
        self.callback(StringMsg(data=json.dumps(dict(state=state,
                      id=self.client.latest_id if identifier is None else identifier, **extra))))

    def test_topics_hold_and_default_node(self):
        self.assertIs(self.client.node, self.node)
        self.assertIs(self.client.pub, self.pub)
        self.assertEqual(self.node.advertise.call_args_list[0].args,
                         (playback.TRAJECTORY_TOPIC, JointTrajectory))
        self.assertEqual(self.node.advertise.call_args_list[1].args[0], playback.HOLD_TOPIC)
        self.assertEqual(self.node.subscribe.call_args.args[:2], (StringMsg, playback.STATUS_TOPIC))
        self.client.hold()
        self.assertTrue(self.hold_pub.publish.call_args.args[0].data)
        self.hold_pub.publish.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'hold'):
            self.client.hold()
        with patch.object(playback, 'Node', return_value=Mock()) as factory:
            playback.PlaybackClient()
            factory.assert_called_once_with()

    def test_accepted_and_unique_ids(self):
        ids = []
        for _ in range(2):
            pump = Mock(side_effect=lambda: self.status('accepted'))
            ids.append(self.client.execute(quintic(), 3., pump))
            pump.assert_called()
            msg = self.pub.publish.call_args.args[0]
            self.assertEqual(msg.header.data[0].value[0], ids[-1])
        self.assertNotEqual(*ids)

    def test_rejected_and_publish_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'bad boundary'):
            self.client.execute(quintic(), 0, lambda: self.status('rejected', reason='bad boundary'))
        self.pub.publish.return_value = False
        pump = Mock()
        with self.assertRaisesRegex(RuntimeError, 'publish'):
            self.client.execute(quintic(), 0, pump)
        pump.assert_not_called()

    def test_retry_identical_payload_ignores_other_ids(self):
        clock = [0.]
        def pump():
            clock[0] += .025
            self.status('accepted', identifier='other')
            if self.pub.publish.call_count >= 3:
                self.status('accepted')
        with patch.object(playback.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(playback.time, 'sleep'):
            self.client.execute(quintic(), 0, pump)
        self.assertEqual(self.pub.publish.call_count, 3)
        payloads = [call.args[0].SerializeToString() for call in self.pub.publish.call_args_list]
        self.assertEqual(len(set(payloads)), 1)

    def test_timeout_and_malformed_json(self):
        self.callback(StringMsg(data='{broken'))
        self.callback(StringMsg(data='[]'))
        self.assertEqual(self.client.status, {})
        clock = [0.]
        def pump():
            clock[0] += .01
        with patch.object(playback.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(playback.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                self.client.execute(quintic(), 0, pump)
        self.assertGreaterEqual(clock[0], 2.)
        self.assertGreater(self.pub.publish.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
