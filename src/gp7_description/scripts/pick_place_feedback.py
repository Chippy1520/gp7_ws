#!/usr/bin/env python3
"""Read-only named Gazebo cube/palm/contact feedback, with simulation stamps."""
import threading
from collections import deque
from bisect import bisect_left
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.world_stats_pb2 import WorldStatistics

CONTACT_TOPICS = {
    'positive': '/world/gp7_lab/model/gp7/link/gripper_finger_positive/sensor/pick_place_contacts/contact',
    'negative': '/world/gp7_lab/model/gp7/link/gripper_finger_negative/sensor/pick_place_contacts/contact',
    'support': '/world/gp7_lab/model/box_b/link/body/sensor/pick_place_support_contacts/contact',
}


def transform(message):
    q=message.orientation
    T=np.eye(4);T[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    p=message.position;T[:3,3]=[p.x,p.y,p.z]
    return T


def stamp(message):return message.header.stamp.sec+message.header.stamp.nsec*1e-9


class GazeboFeedback:
    def __init__(self):
        self.node=Node();self.lock=threading.Lock()
        self.history={key:deque(maxlen=200) for key in ['cube','tcp']}
        self.cube=self.tcp=None;self.cube_stamp=self.tcp_stamp=None
        self.cube_wall=self.tcp_wall=0.;self.paused=False;self.stats_wall=0.
        self.contacts={'positive':0.,'negative':0.,'support':0.};self.contact_pairs=[]
        self.node.subscribe(Pose_V,'/pick_place/box_b/pose',self._cube)
        for topic in ['/model/gp7/pose','/world/gp7_lab/model/gp7/pose']:
            self.node.subscribe(Pose_V,topic,self._palm)
        self.node.subscribe(WorldStatistics,'/world/gp7_lab/stats',self._stats)
        for side,topic in CONTACT_TOPICS.items():
            self.node.subscribe(Contacts,topic,lambda msg,side=side:self._contact(msg,side))

    def require_contact_publishers(self,timeout=2.):
        """Empty contacts are normal before grasp; absent publishers are not."""
        end=time.monotonic()+timeout
        while True:
            missing=[side for side,topic in CONTACT_TOPICS.items() if not self.node.topic_info(topic)[0]]
            if not missing:return list(CONTACT_TOPICS.values())
            if time.monotonic()>=end:
                raise RuntimeError('Missing Gazebo contact publishers: '+', '.join(missing)+
                                   '. Reload the prepared world by restarting its server; do not bypass contact checks.')
            time.sleep(.05)

    def _cube(self,msg):
        # PoseArray conversion is deliberately not used here: preserve name,
        # identity and simulation timestamp of the actual dynamic object.
        matching=[p for p in msg.pose if p.name=='box_b']
        if len(matching)!=1:return
        with self.lock:
            self.cube=transform(matching[0]);self.cube_stamp=stamp(matching[0]);self.cube_wall=time.monotonic()
            self._remember('cube',self.cube_stamp,self.cube)

    def _palm(self,msg):
        for p in msg.pose:
            if p.name=='gp7::gripper_palm':
                T=transform(p);T[:3,3]+=T[:3,:3]@np.array([0,0,.075])
                with self.lock:
                    self.tcp=T;self.tcp_stamp=stamp(p);self.tcp_wall=time.monotonic()
                    self._remember('tcp',self.tcp_stamp,self.tcp)

    def _stats(self,msg):
        with self.lock:self.paused=msg.paused;self.stats_wall=time.monotonic()

    def _contact(self,msg,side):
        pairs=[]
        for c in msg.contact:
            a=getattr(c.collision1,'name',str(c.collision1));b=getattr(c.collision2,'name',str(c.collision2))
            if side=='support':
                if ('box_a' in a and 'box_b' in b) or ('box_a' in b and 'box_b' in a):pairs.append([a,b])
            elif 'box_b' in a or 'box_b' in b:pairs.append([a,b])
        if pairs:
            with self.lock:self.contacts[side]=time.monotonic();self.contact_pairs=pairs

    def _remember(self,key,time_s,T):
        history=self.history[key]
        if history and time_s<history[-1][0]:history.clear()
        history.append((time_s,T.copy()))
        self.pose_revision=getattr(self,'pose_revision',0)+1

    @staticmethod
    def _at(history,time_s):
        times=[item[0] for item in history];index=bisect_left(times,time_s)
        if index<len(times) and abs(times[index]-time_s)<1e-9:return history[index][1].copy()
        if index==0 or index==len(times):return None
        a,A=history[index-1];b,B=history[index]
        if b-a>.1:return None
        T=np.eye(4)
        T[:3,:3]=Slerp([a,b],Rotation.from_matrix(np.stack([A[:3,:3],B[:3,:3]])))(time_s).as_matrix()
        T[:3,3]=[(np.interp(time_s,[a,b],[A[k,3],B[k,3]])) for k in range(3)]
        return T

    def snapshot(self):
        # Never hold the transport callback lock while SciPy interpolates.
        with self.lock:
            raw=[self.cube_stamp,self.tcp_stamp]
            revision=getattr(self,'pose_revision',0)
            cached=getattr(self,'pose_cache',None)
            histories=None if cached is not None and cached[0]==revision else {key:list(value) for key,value in self.history.items()}
            data={'cube_wall':self.cube_wall,'tcp_wall':self.tcp_wall,
                  'paused':self.paused,'stats_wall':self.stats_wall,
                  'contacts':dict(self.contacts),'contact_pairs':list(self.contact_pairs)}
        common=min(raw) if all(v is not None for v in raw) else None
        if histories is None:
            cube,tcp=cached[1:]
        else:
            cube=self._at(histories['cube'],common) if common is not None else None
            tcp=self._at(histories['tcp'],common) if common is not None else None
            self.pose_cache=(revision,cube,tcp)
        cube=None if cube is None else cube.copy()
        tcp=None if tcp is None else tcp.copy()
        data.update({'cube':cube,'tcp':tcp,'cube_stamp':common,'tcp_stamp':common,
                     'raw_latest_stamps':raw,
                     'association':'shared simulation time; measured position interpolation and scipy Slerp'})
        return data
