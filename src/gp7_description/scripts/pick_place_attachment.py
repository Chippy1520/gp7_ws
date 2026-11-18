#!/usr/bin/env python3
"""Opt-in detachable-constraint assistance, not friction-grasp validation."""
import threading
import time
from gz.msgs10.boolean_pb2 import Boolean


class DetachableGrasp:
    def __init__(self,node):
        self.node=node;self.lock=threading.Lock();self.state=None;self.received=0.
        self.publisher=node.advertise('/pick_place/grasp/enable',Boolean)
        node.subscribe(Boolean,'/pick_place/grasp/attached',self._state)

    def _state(self,msg):
        with self.lock:self.state=msg.data;self.received=time.monotonic()

    def attached(self):
        with self.lock:return self.state is True and time.monotonic()-self.received<.3

    def set(self,enabled,guard):
        # Require a fresh acknowledgement AFTER this command, not a cached state.
        started=time.monotonic();last=-1.
        while time.monotonic()-started<2.:
            guard();now=time.monotonic()
            if now-last>.05:
                self.publisher.publish(Boolean(data=bool(enabled)));last=now
            with self.lock:
                if self.received>started and self.state is bool(enabled):return
        raise RuntimeError('No fresh detachable-assist acknowledgement: '+str(enabled))
