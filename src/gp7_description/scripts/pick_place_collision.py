#!/usr/bin/env python3
"""FCL checks using actual simulator boxes and CAD refinement for self geometry.

Adjacent kinematic links are the only fixed self-contact exclusions. Gazebo's
self_collide=false is never treated as self-collision evidence.
"""
from itertools import product
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as E
import numpy as np
import fcl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_model import fk as link_fk
from kinematics import rpy_matrix, validate_pose
from pick_place_config import PKG, NAMES


def pose(text):
    values=np.fromstring(text or '0 0 0 0 0 0',sep=' ')
    T=np.eye(4);T[:3,3]=values[:3];T[:3,:3]=rpy_matrix(values[3:]);return T


def collision_object(geometry,T):
    return fcl.CollisionObject(geometry,fcl.Transform(T[:3,:3],T[:3,3]))


def separation(first,second):
    if fcl.collide(first,second,fcl.CollisionRequest(),fcl.CollisionResult()):return -1.
    return float(fcl.distance(first,second,fcl.DistanceRequest(),fcl.DistanceResult()))


def load_stl(path, box_geometry, box_pose):
    raw=Path(path).read_bytes();count=int.from_bytes(raw[80:84],'little')
    triangles=np.frombuffer(raw,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]),count=count,offset=84)['vertices']
    vertices=triangles.astype(float).reshape(-1,3)
    local=(vertices-box_pose[:3,3])@box_pose[:3,:3]
    if np.any(abs(local)>np.asarray(box_geometry.side)/2+1e-8):
        raise ValueError('CAD mesh is not contained in the broad-phase simulator box')
    model=fcl.BVHModel();model.beginModel(count,len(vertices));model.addSubModel(vertices,np.arange(len(vertices)).reshape(-1,3));model.endModel()
    return model


def _batch_link_fk(robot, q, jaws, progress=lambda: None):
    """Same URDF tree/origins/Rodrigues products as audit_model.fk, in batches.

    All arrays belong to this invocation: a progress callback may safely run a
    scalar check (or another validate) without replacing in-flight transforms.
    Callers bound batch size so no unbounded whole-path FK precedes a guard.
    """
    from audit_model import origin
    positions=dict(zip(NAMES+['gripper_joint_positive','gripper_joint_negative'],
                       [*q.T,*np.broadcast_to(jaws,(len(q),2)).T]))
    links={link.get('name') for link in robot.findall('link')}
    pending=list(robot.findall('joint'))
    roots=links-{j.find('child').get('link') for j in pending}
    assert len(roots)==1,roots
    frames={roots.pop():np.broadcast_to(np.eye(4),(len(q),4,4)).copy()}
    while pending:
        count=len(pending)
        for joint in pending[:]:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            if parent not in frames:continue
            assert child not in frames,f'duplicate child {child}'
            motion=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
            kind=joint.get('type')
            if kind!='fixed':
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                assert abs(np.linalg.norm(axis)-1)<1e-8
                value=np.broadcast_to(positions.get(joint.get('name'),0),(len(q),))
                if kind=='prismatic':motion[:,:3,3]=value[:,None]*axis
                else:
                    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    motion[:,:3,:3]=(np.eye(3)+np.sin(value)[:,None,None]*K+
                                     (1-np.cos(value))[:,None,None]*(K@K))
            frames[child]=(frames[parent]@origin(joint))@motion
            pending.remove(joint)
            progress()
        assert len(pending)<count,'disconnected tree or cycle'
    assert set(frames)==links
    return frames


class CollisionChecker:
    """Default legacy full checker; opt in with ``cube_only=True``.

    Cube-only covers every robot collision primitive conservatively, but omits
    robot self/world pairs. A supplied unheld cube pose needs explicit resting
    support (``ground`` before grasp, ``box_a`` after release). Only the original
    world cube with no supplied pose infers ground support. ``contact_allowed``
    permits finger/cube contact during grasp/release, never palm/arm contact.
    Support exceptions allow contact, not penetration beyond the configured
    small tolerance; they are intentional contact exclusions from clearance.
    """
    def __init__(self,world_path=None,clearance=.005,self_clearance=.001,cube_only=False):
        self.cube_only=bool(cube_only)
        self.clearance=clearance
        self.self_clearance=self_clearance
        self.urdf=E.parse(PKG/'urdf/gp7_with_gripper.urdf').getroot()
        world=E.parse(world_path or PKG/'worlds/gp7_pick_place.sdf').getroot().find('world')
        robot=world.find("model[@name='gp7']")
        self.adjacent={frozenset([j.find('parent').get('link'),j.find('child').get('link')]) for j in self.urdf.findall('joint')}
        self.boxes={};self.meshes={};self.environment={}
        # Keep the legacy first-box/CAD path intact, but cover every simulator
        # collision shape (including extra finger pads) in cube-only mode.
        self.cube_robot_shapes={};self.world_shapes={}
        self.cube_corners=np.array(list(product([-.025,.025],repeat=3)))
        for link in robot.findall('link'):
            name=link.get('name');collision=link.find('collision')
            if collision is None:continue
            size=np.fromstring(collision.findtext('geometry/box/size'),sep=' ')
            self.boxes[name]=(fcl.Box(*size),pose(collision.findtext('pose')))
            self.cube_robot_shapes[name]=[
                (self._primitive(c.find('geometry')),pose(c.findtext('pose')))
                for c in link.findall('collision')]
            uri=link.findtext('visual/geometry/mesh/uri')
            # Resolve the same supplied mesh from this checkout, not the author's old path.
            uri=str(PKG/'meshes'/Path(uri).name) if uri else uri
            if uri and not self.cube_only:self.meshes[name]=load_stl(uri,*self.boxes[name])
        for model in world.findall('model'):
            if model.get('name')=='gp7':continue
            T=pose(model.findtext('pose'))
            for link in model.findall('link'):
                for collision in link.findall('collision'):
                    size=np.fromstring(collision.findtext('geometry/box/size'),sep=' ')
                    placement=T@pose(link.findtext('pose'))@pose(collision.findtext('pose'))
                    geometry=fcl.Box(*size)
                    obj=collision_object(geometry,placement)
                    self.environment[model.get('name')]=obj
                    self.world_shapes.setdefault(model.get('name'),[]).append((obj,placement,size))
        self.cube_geometry=fcl.Box(.05,.05,.05)
        # Triangle inequality: sum of every fixed origin length overbounds any
        # ancestor-to-point chain. Add geometry radius, jaw stroke/cube margin.
        origin_lengths=sum(np.linalg.norm(np.fromstring(j.find('origin').get('xyz'),sep=' '))
                           for j in self.urdf.findall('joint'))
        radii=[np.linalg.norm(T[:3,3])+np.linalg.norm(geometry.side)/2
               for shapes in self.cube_robot_shapes.values() for geometry,T in shapes]
        self.point_reach_bound=float(origin_lengths+max(radii)+.1)
        parents={j.find('child').get('link'):j for j in self.urdf.findall('joint')}
        self.ancestors={}
        for name in self.boxes:
            chain=set();current=name
            while current in parents:
                joint=parents[current]
                if joint.get('name') in NAMES:chain.add(NAMES.index(joint.get('name')))
                current=joint.find('parent').get('link')
            self.ancestors[name]=chain

    @staticmethod
    def _primitive(geometry):
        """Exact boxes, conservative boxes for rounded pad primitives; fail closed."""
        if geometry.find('box') is not None:
            return fcl.Box(*np.fromstring(geometry.findtext('box/size'),sep=' '))
        if geometry.find('sphere') is not None:
            diameter=2*float(geometry.findtext('sphere/radius'))
            return fcl.Box(diameter,diameter,diameter)
        if geometry.find('cylinder') is not None:
            diameter=2*float(geometry.findtext('cylinder/radius'))
            return fcl.Box(diameter,diameter,float(geometry.findtext('cylinder/length')))
        raise ValueError('Unsupported robot collision primitive (cube checks must cover every shape)')

    def _check_cube_only(self,frames,cube_transform,holding,contact_allowed,support,
                         motion_padding,joint_motion,tolerance):
        # Only the untouched, world-defined resting cube gets inferred support.
        # Predicted/released cube poses require the caller to name their support.
        if cube_transform is None:
            cube_transform=self.world_shapes['box_b'][-1][1]
            if support is None:support='ground'
        cube=collision_object(self.cube_geometry,cube_transform)
        minimum=np.inf;violations=[]
        for name,shapes in self.cube_robot_shapes.items():
            if (holding or contact_allowed) and name in ('gripper_finger_positive','gripper_finger_negative'):
                continue
            if joint_motion is not None:
                moving=(self.ancestors[name]^set(range(6)) if holding else self.ancestors[name])
                padding=self.point_reach_bound*sum(joint_motion[j] for j in moving)
            else:padding=motion_padding*(2 if holding else 1)
            for geometry,local in shapes:
                gap=separation(collision_object(geometry,frames[name]@local),cube)
                minimum=min(minimum,gap)
                if gap<self.clearance+padding:violations.append(('environment',name,'box_b',gap))
        corners=self.cube_corners@cube_transform[:3,:3].T+cube_transform[:3,3]
        for name in ('ground','box_a'):
            for obstacle,placement,size in self.world_shapes[name]:
                gap=separation(cube,obstacle)
                if name==support:
                    # Check ALL corners in support coordinates, not merely the
                    # cube center. Tolerance never waives deep penetration.
                    local=(corners-placement[:3,3])@placement[:3,:3]
                    footprint=np.all(abs(local[:,:2])<=size[:2]/2-motion_padding+1e-12)
                    if local[:,2].min()<size[2]/2-tolerance or not footprint:
                        violations.append(('invalid_support_exception',name))
                    continue
                minimum=min(minimum,gap)
                if gap<self.clearance+motion_padding:violations.append(('carried_cube' if holding else 'cube',name,gap))
        return {'safe':not violations,'minimum_checked_separation_m':float(minimum),
                'violations':violations,'scope':'cube_only: cube vs all robot collision shapes and ground/pedestal; excludes robot self/world checks'}

    def check(self,q,jaws=(0.,0.),cube_transform=None,holding=False,support=None,motion_padding=0.,joint_motion=None,support_penetration_tolerance=1e-5,contact_allowed=False):
        return self._check_state(q,jaws,cube_transform,holding,support,motion_padding,
                                 joint_motion,support_penetration_tolerance,contact_allowed)

    def _check_state(self,q,jaws=(0.,0.),cube_transform=None,holding=False,support=None,motion_padding=0.,joint_motion=None,support_penetration_tolerance=1e-5,contact_allowed=False,*,frames=None):
        q=np.asarray(q,dtype=float);jaws=np.asarray(jaws,dtype=float)
        if q.shape!=(6,) or jaws.shape!=(2,) or not np.isfinite(q).all() or not np.isfinite(jaws).all():
            raise ValueError('Expected finite six-joint and two-jaw states')
        if cube_transform is not None:cube_transform=validate_pose(cube_transform)
        if support not in [None,'ground','box_a']:raise ValueError('Unsupported contact exception')
        if self.cube_only and (not np.isfinite(support_penetration_tolerance) or not 0<=support_penetration_tolerance<=1e-3):
            raise ValueError('Support penetration tolerance must be finite and at most 1 mm')
        if frames is None:
            frames=link_fk(self.urdf,dict(zip(NAMES+['gripper_joint_positive','gripper_joint_negative'],[*q,*jaws])))
        if self.cube_only:
            return self._check_cube_only(frames,cube_transform,holding,contact_allowed,support,
                                         motion_padding,joint_motion,support_penetration_tolerance)
        robot={name:collision_object(geometry,frames[name]@local) for name,(geometry,local) in self.boxes.items()}
        meshes={};violations=[];minimum=np.inf
        items=list(robot.items())
        for i,(name,first) in enumerate(items):
            for other,second in items[i+1:]:
                if frozenset([name,other]) in self.adjacent:continue
                gap=separation(first,second)
                # Common ancestor rotations cancel in relative self geometry.
                unique=self.ancestors[name]^self.ancestors[other]
                relative_padding=(self.point_reach_bound*sum(joint_motion[j] for j in unique)
                                  if joint_motion is not None else 2*motion_padding)
                if gap<self.self_clearance+relative_padding:
                    # Coarse bounding boxes can overlap empty CAD space. Check
                    # the actual CAD meshes for this pair/configuration instead
                    # of creating a broad collision-exclusion list.
                    if name in self.meshes:
                        meshes.setdefault(name,collision_object(self.meshes[name],frames[name]))
                    if other in self.meshes:
                        meshes.setdefault(other,collision_object(self.meshes[other],frames[other]))
                    gap=separation(meshes.get(name,first),meshes.get(other,second))
                minimum=min(minimum,gap)
                if gap<self.self_clearance+relative_padding:violations.append(('self',name,other,gap))
        cube=self.environment['box_b'] if cube_transform is None else collision_object(self.cube_geometry,cube_transform)
        for name,first in robot.items():
            for other,second in {**self.environment,'box_b':cube}.items():
                if name=='base_link' and other=='ground':continue
                if other=='box_b' and (holding or contact_allowed) and name in ['gripper_finger_positive','gripper_finger_negative']:continue
                gap=separation(first,second);minimum=min(minimum,gap)
                if joint_motion is not None:
                    moving=(self.ancestors[name]^set(range(6)) if other=='box_b' and holding else self.ancestors[name])
                    relative_padding=self.point_reach_bound*sum(joint_motion[j] for j in moving)
                else:relative_padding=motion_padding*(2 if other=='box_b' and holding else 1)
                threshold=self.clearance+relative_padding
                if gap<threshold:violations.append(('environment',name,other,gap))
        if holding:
            for name in ['ground','box_a']:
                gap=separation(cube,self.environment[name])
                if name==support:
                    if cube_transform is None:raise ValueError('Support exception requires a predicted cube transform')
                    corners=np.array(list(product([-.025,.025],repeat=3)))
                    corners=corners@cube_transform[:3,:3].T+cube_transform[:3,3]
                    height=0. if name=='ground' else .2
                    footprint=(name=='ground' or np.all(corners[:,:2] >= .35+motion_padding)
                               and np.all(corners[:,:2] <= .65-motion_padding))
                    if corners[:,2].min() < height-support_penetration_tolerance or not footprint:
                        violations.append(('invalid_support_exception',name))
                    continue
                minimum=min(minimum,gap)
                if gap<self.clearance+motion_padding:violations.append(('carried_cube',name,gap))
        return {'safe':not violations,'minimum_checked_separation_m':float(minimum),'violations':violations}

    def validate(self,segment,jaws=(0.,0.),holding=False,tcp_cube=None,support=None,period=.005,cube_world=None,contact_allowed=False,*,progress_callback=None):
        """Certify the unchanged time grid/padding with bounded-size FK batches.

        Optional zero-argument guard runs at entry, completion and approximately
        every 30 ms of preparation (including FK). Exceptions abort validation.
        Cooperative, not a hard deadline: an individual spline/FCL operation or
        callback cannot be preempted. Callback must be bounded and not mutate the
        checker/segment; scalar check() and nested validate() are safe.
        """
        from kinematics import ExportedModel
        from pick_place_planner import continuous_extrema
        last_progress=0.
        def progress(force=False):
            nonlocal last_progress
            if progress_callback is not None and (force or time.monotonic()-last_progress>=.03):
                progress_callback()
                last_progress=time.monotonic()
        progress(True)
        if holding and (tcp_cube is None or np.linalg.norm(tcp_cube[:3,3])>.05):
            raise ValueError('A calibrated, nearby TCP-to-cube transform is required')
        if not np.isfinite(period) or period<=0:raise ValueError('Sample period must be positive and finite')
        model=ExportedModel() if holding else None
        times=np.linspace(0,segment.duration,int(np.ceil(segment.duration/period))+1)
        peak_speed=np.max(abs(continuous_extrema(segment.spline,1)),axis=1)
        half_interval=float(np.max(np.diff(times)))/2
        joint_motion=peak_speed*half_interval
        padding=self.point_reach_bound*float(sum(peak_speed))*half_interval
        minimum=np.inf
        jaws=np.asarray(jaws,dtype=float)
        if jaws.shape!=(2,) or not np.isfinite(jaws).all():
            raise ValueError('Expected finite six-joint and two-jaw states')
        progress()
        for start in range(0,len(times),64):
            batch_times=times[start:start+64]
            qs=np.asarray(segment.evaluate(batch_times),dtype=float)
            if qs.shape!=(len(batch_times),6) or not np.isfinite(qs).all():
                raise ValueError('Expected finite six-joint and two-jaw states')
            progress()
            batch=_batch_link_fk(self.urdf,qs,jaws,progress)
            # Match ExportedModel.fk's final grouped tool product; no Jacobian
            # is needed. URDF link_t has exactly the same six arm products.
            cubes=(batch['link_t']@model.tool)@tcp_cube if holding else None
            progress()
            for i,(t,q) in enumerate(zip(batch_times,qs)):
                frames={name:values[i] for name,values in batch.items()}
                cube=cubes[i] if holding else cube_world
                result=self._check_state(q,jaws,cube,holding,support,motion_padding=padding,
                                         joint_motion=joint_motion,contact_allowed=contact_allowed,frames=frames)
                minimum=min(minimum,result['minimum_checked_separation_m'])
                progress()
                if not result['safe']:raise ValueError(f'{segment.name} collision at t={t:.6g}: {result["violations"]}')
        progress(True)
        return {'phase':segment.name,'collision_samples':len(times),'sample_period_max_s':period,
                'minimum_checked_separation_m':float(minimum),
                'between_sample_point_displacement_bound_m':padding,
                'scope':('cube_only: cube vs all robot collision shapes and ground/pedestal; excludes robot self/world checks; ' if self.cube_only else '')+
                        'FCL checks inflated by a conservative joint-rate/geometry displacement bound; reference geometry only, not physical tracking certification'}
