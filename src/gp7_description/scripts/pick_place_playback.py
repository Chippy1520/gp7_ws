#!/usr/bin/env python3
"""Upload complete simulation trajectories over native Gazebo Transport.

No ROS bridge or per-frame commands are used. ``execute`` waits only for the
upload acknowledgement, not for motion completion. ``pump`` must be nonblocking
(or bounded) so acknowledgement waiting can enforce its wall-clock deadline.
"""
import json
import math
import threading
import time
import uuid

from gz.transport13 import Node
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.joint_trajectory_pb2 import JointTrajectory
from gz.msgs10.stringmsg_pb2 import StringMsg
from pick_place_config import NAMES

JOINT_NAMES = tuple(NAMES) + ('gripper_joint_positive', 'gripper_joint_negative')
TRAJECTORY_TOPIC = '/gp7/kinematic/trajectory'
HOLD_TOPIC = '/gp7/kinematic/hold'
STATUS_TOPIC = '/gp7/kinematic/status'


def _seconds(value, name, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _nanoseconds(value):
    # Round once, then carry into seconds (never emit nsec == 1_000_000_000).
    return round(value * 1_000_000_000)


def _set_time(message, nanoseconds):
    message.sec, message.nsec = divmod(nanoseconds, 1_000_000_000)


def _vector(value, name):
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain eight finite scalars') from exc
    if len(result) != 8 or not all(math.isfinite(item) for item in result):
        raise ValueError(f'{name} must contain eight finite scalars')
    return result


def trajectory_message(curve, start, names=None, segment_id='test'):
    """Serialize position/velocity/acceleration boundary data at every knot.

    Endpoint-only generic quintics and piecewise quintic BPoly references are
    both reconstructed exactly by quintic Hermite interpolation in the plugin.
    Jerk is not transmitted; references may leave it unspecified (None).
    """
    start = _seconds(start, 'start')
    duration = _seconds(curve.duration, 'duration', positive=True)
    names = list(JOINT_NAMES if names is None else names)
    if len(names) != 8 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 8:
        raise ValueError('names must contain eight distinct nonempty joint names')
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError('segment_id must be a nonempty string')
    spline = getattr(getattr(curve, 'segment', None), 'spline', None)
    raw = getattr(spline, 'x', None)
    knots = [] if raw is None else [_seconds(t, 'knot') for t in raw]
    if any(t > duration for t in knots) or any(b <= a for a, b in zip(knots, knots[1:])):
        raise ValueError('spline knots must be strictly increasing within [0, duration]')
    if not knots or knots[0] != 0:
        knots.insert(0, 0.0)
    if knots[-1] != duration:
        knots.append(duration)
    times_ns = [_nanoseconds(t) for t in knots]
    if any(b <= a for a, b in zip(times_ns, times_ns[1:])):
        raise ValueError('knots must remain strictly increasing at nanosecond resolution')

    message = JointTrajectory()
    message.joint_names.extend(names)
    _set_time(message.header.stamp, _nanoseconds(start))
    entry = message.header.data.add()
    entry.key = 'segment_id'
    entry.value.append(segment_id)
    for t, ns in zip(knots, times_ns):
        sample = curve.evaluate(t)
        point = message.points.add()
        for field, source in [('positions', 'position'), ('velocities', 'velocity'),
                              ('accelerations', 'acceleration')]:
            getattr(point, field).extend(_vector(getattr(sample, source), source))
        if getattr(sample, 'jerk', None) is not None:
            _vector(sample.jerk, 'jerk')
        _set_time(point.time_from_start, ns)
    return message


class PlaybackClient:
    """Native transport upload/acknowledgement client; never starts a server."""

    def __init__(self, node=None):
        self.node = Node() if node is None else node
        self.status = {}
        self.latest_id = None
        self._lock = threading.Lock()
        self.pub = self.node.advertise(TRAJECTORY_TOPIC, JointTrajectory)
        self._hold_pub = self.node.advertise(HOLD_TOPIC, Boolean)
        self.node.subscribe(StringMsg, STATUS_TOPIC, self._status_callback)

    def _status_callback(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if isinstance(status, dict):
            with self._lock:
                self.status = status

    def execute(self, curve, start, pump):
        segment_id = uuid.uuid4().hex
        message = trajectory_message(curve, start, segment_id=segment_id)
        with self._lock:
            self.latest_id = segment_id
        deadline = time.monotonic() + 2.0
        next_publish = 0.0
        while True:
            with self._lock:
                status = dict(self.status)
            if status.get('id') == segment_id:
                if status.get('state') == 'accepted':
                    return segment_id
                if status.get('state') == 'rejected':
                    raise RuntimeError(f'Trajectory {segment_id} rejected: {status.get("reason", "unspecified reason")}')
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f'Timed out waiting for trajectory {segment_id} acknowledgement')
            if now >= next_publish:
                if not self.pub.publish(message):
                    raise RuntimeError(f'Failed to publish trajectory {segment_id}')
                next_publish = now + 0.05
            pump()
            time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

    def hold(self):
        """Request the plugin hold its current simulated joint positions."""
        if not self._hold_pub.publish(Boolean(data=True)):
            raise RuntimeError('Failed to publish kinematic hold')
