#!/usr/bin/env python3
"""Pure hybrid references: direct polynomial arm + continuously compensated wrist.

No ROS or simulator commands. Limits, interpolation and FK are checked offline.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import BPoly, PPoly
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinematics import ExportedModel, dh, dh_rows, refine_ik, ik_exported, validate_pose
from joint_trajectories import quintic_given_duration
from pick_place_config import arm_limits, load_config


@dataclass
class Segment:
    name: str
    duration: float
    spline: BPoly
    arm: object = None
    desired_rotation: object = None
    vertical_xy: object = None

    def evaluate(self, time, derivative=0):
        time=np.asarray(time,dtype=float)
        if np.any(time<0) or np.any(time>self.duration):raise ValueError('Time outside segment')
        result=self.spline(time,nu=derivative)
        if self.arm is not None:
            sample=self.arm.evaluate(time)
            field=['position','velocity','acceleration','jerk'][derivative]
            result[..., :3]=getattr(sample,field)
        return result


def rotation_error(desired, actual):
    return Rotation.from_matrix(desired@actual.T).as_rotvec()


def wrist_candidates(model, arm, desired, limits):
    # This project's nominal DH tool rotation is identity. Exact exported
    # fixed/tool rotations are included by the wrist-only residual below.
    frame=np.eye(4)
    for row in dh_rows(np.r_[arm,np.zeros(3)])[:3]:frame=frame@dh(*row)
    W=frame[:3,:3].T@desired
    sine=np.hypot(W[0,2],W[1,2])
    if sine<1e-8:raise ValueError('Nominal wrist singularity on compensated path')
    a,b,c=np.arctan2(W[1,2],W[0,2]),np.arctan2(sine,W[2,2]),np.arctan2(W[2,1],-W[2,0])
    results=[]
    for base in [np.array([a,b,c]),np.array([a+np.pi,-b,c+np.pi])]:
        turns=[range(int(np.ceil((low-x)/ (2*np.pi))),int(np.floor((high-x)/(2*np.pi)))+1)
               for x,(low,high) in zip(base,limits[3:])]
        for winding in product(*turns):
            seed=base+2*np.pi*np.array(winding)
            fit=least_squares(lambda wrist:rotation_error(desired,model.fk(np.r_[arm,wrist])[:3,:3]),
                              seed,bounds=(limits[3:,0],limits[3:,1]),
                              ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=40)
            if np.linalg.norm(fit.fun)<1e-8 and not any(np.linalg.norm(fit.x-q)<1e-7 for q in results):
                results.append(fit.x)
    if not results:raise ValueError('No in-limit exact-export wrist solution with the arm held fixed')
    return results


def continuous_wrist_sequence(model, arms, rotation, limits, first_wrist, last_wrist, max_step=.15):
    # Dynamic programming evaluates the complete candidate sequence, not a greedy
    # nearest-only walk that might run into a later limit or branch discontinuity.
    layers=[wrist_candidates(model,a,rotation,limits) for a in arms]
    costs=np.array([np.linalg.norm(q-first_wrist)**2 if np.linalg.norm(q-first_wrist)<1e-4 else np.inf for q in layers[0]])
    parents=[]
    for previous,current in zip(layers,layers[1:]):
        next_cost=np.full(len(current),np.inf);parent=np.full(len(current),-1,dtype=int)
        for j,q in enumerate(current):
            for i,p in enumerate(previous):
                if np.max(abs(q-p))>max_step:continue
                value=costs[i]+np.linalg.norm(q-p)**2
                if value<next_cost[j]:next_cost[j],parent[j]=value,i
        costs=next_cost;parents.append(parent)
    costs=np.array([v if np.linalg.norm(q-last_wrist)<1e-4 else np.inf for v,q in zip(costs,layers[-1])])
    if not np.isfinite(costs).any():raise ValueError('No continuous wrist branch connecting both endpoint configurations')
    index=int(np.argmin(costs));chosen=[layers[-1][index]]
    for layer,parent in zip(layers[-2::-1],parents[::-1]):
        index=int(parent[index]);chosen.append(layer[index])
    return np.array(chosen[::-1])


def jacobian_rate(model,q,velocity):
    delta=1e-5
    return (model.evaluate(q+delta*velocity)[1]-model.evaluate(q-delta*velocity)[1])/(2*delta)


def spline_from_states(times,q,velocity,acceleration):
    return BPoly.from_derivatives(times,[[a,b,c] for a,b,c in zip(q,velocity,acceleration)])


def compensated_segment(name,start,goal,duration,config=None,model=None):
    config=config or load_config();model=model or ExportedModel();limits=arm_limits(config)
    start,goal=np.asarray(start),np.asarray(goal)
    rotation=model.fk(start)[:3,:3]
    if np.linalg.norm(rotation_error(rotation,model.fk(goal)[:3,:3]))>1e-6:
        raise ValueError('Compensated endpoints must have the same orientation')
    arm=quintic_given_duration(start[:3],goal[:3],duration)
    times=np.linspace(0,duration,max(3,int(np.ceil(duration/config['planning']['sample_period_s']))+1))
    sample=arm.evaluate(times)
    wrists=continuous_wrist_sequence(model,sample.position,rotation,limits,start[3:],goal[3:])
    q=np.c_[sample.position,wrists];velocity=np.c_[sample.velocity,np.zeros_like(wrists)];acceleration=np.c_[sample.acceleration,np.zeros_like(wrists)]
    for i,position in enumerate(q):
        J=model.evaluate(position)[1][3:]
        if np.linalg.svd(J[:,3:],compute_uv=False)[-1]<config['planning']['wrist_min_singular_value']:
            raise ValueError('Wrist conditioning margin violated')
        velocity[i,3:]=np.linalg.solve(J[:,3:],-J[:,:3]@velocity[i,:3])
        Jdot=jacobian_rate(model,position,velocity[i])[3:]
        acceleration[i,3:]=np.linalg.solve(J[:,3:],-J[:,:3]@acceleration[i,:3]-Jdot@velocity[i])
    return Segment(name,duration,spline_from_states(times,q,velocity,acceleration),arm,rotation)


def joint_segment(name,start,goal,duration):
    curve=quintic_given_duration(start,goal,duration)
    times=np.array([0.,duration]);values=[curve.evaluate(t) for t in times]
    spline=spline_from_states(times,[s.position for s in values],[s.velocity for s in values],[s.acceleration for s in values])
    return Segment(name,duration,spline)


def vertical_segment(name,start,z_goal,duration,config=None,model=None):
    config=config or load_config();model=model or ExportedModel();limits=arm_limits(config)
    initial=model.fk(start);z=quintic_given_duration(initial[2,3],z_goal,duration)
    times=np.linspace(0,duration,max(3,int(np.ceil(duration/config['planning']['sample_period_s']))+1))
    samples=z.evaluate(times);positions=[];velocities=[];accelerations=[];previous=np.array(start)
    for t,height,v,a in zip(times,samples.position,samples.velocity,samples.acceleration):
        target=initial.copy();target[2,3]=height
        q=refine_ik(model,target,previous,tolerance=1e-10)
        if q is None or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Vertical IK infeasible within task limits')
        if np.max(abs(q-previous))>.15:raise ValueError('Vertical IK branch discontinuity')
        J=model.evaluate(q)[1]
        dq=np.linalg.solve(J,np.array([0,0,v,0,0,0]))
        ddq=np.linalg.solve(J,np.array([0,0,a,0,0,0])-jacobian_rate(model,q,dq)@dq)
        positions.append(q);velocities.append(dq);accelerations.append(ddq);previous=q
    return Segment(name,duration,spline_from_states(times,positions,velocities,accelerations),
                   desired_rotation=initial[:3,:3],vertical_xy=initial[:2,3])


def continuous_extrema(spline,derivative=0):
    """Polynomial extrema, including interior roots, not sample-only rate checks."""
    polynomial=PPoly.from_bernstein_basis(spline)
    output=[]
    for joint in range(6):
        scalar=PPoly(polynomial.c[:,:,joint],polynomial.x)
        function=scalar.derivative(derivative)
        roots=function.derivative().roots(extrapolate=False)
        times=np.r_[polynomial.x,roots[np.isfinite(roots)]]
        values=function(times)
        output.append([float(values.min()),float(values.max())])
    return np.array(output)


def validate_segment(segment,config=None,model=None):
    config=config or load_config();model=model or ExportedModel();limits=arm_limits(config)
    position=continuous_extrema(segment.spline)
    velocity=continuous_extrema(segment.spline,1);acceleration=continuous_extrema(segment.spline,2)
    if np.any(position[:,0]<limits[:,0]-1e-9) or np.any(position[:,1]>limits[:,1]+1e-9):raise ValueError('Continuous joint position bound violated')
    if np.max(abs(velocity))>config['velocity_limit_rad_s']+1e-8:raise ValueError('Continuous all-joint velocity bound violated; increase duration')
    if np.max(abs(acceleration))>config['acceleration_limit_rad_s2']+1e-8:raise ValueError('Continuous all-joint acceleration bound violated; increase duration')
    times=np.linspace(0,segment.duration,max(3,int(np.ceil(segment.duration/.01))+1))
    angles=segment.evaluate(times);max_orientation=max_xy=0.;min_condition=1.
    for q in angles:
        T,J=model.evaluate(q)
        if segment.desired_rotation is not None:
            max_orientation=max(max_orientation,float(np.linalg.norm(rotation_error(segment.desired_rotation,T[:3,:3]))))
            min_condition=min(min_condition,float(np.linalg.svd(J[3:,3:],compute_uv=False)[-1]))
        if segment.vertical_xy is not None:max_xy=max(max_xy,float(np.linalg.norm(T[:2,3]-segment.vertical_xy)))
    if max_orientation>config['planning']['orientation_reference_tolerance_rad']:raise ValueError('Interpolated orientation error exceeds reference tolerance')
    if segment.desired_rotation is not None and min_condition<config['planning']['wrist_min_singular_value']:raise ValueError('Interpolated wrist conditioning violated')
    if max_xy>1e-5:raise ValueError('Interpolated vertical segment drifts in XY')
    return {'phase':segment.name,'duration_s':segment.duration,'continuous_position_bounds_rad':position.tolist(),
            'continuous_peak_velocity_rad_s':np.max(abs(velocity),axis=1).tolist(),
            'continuous_peak_acceleration_rad_s2':np.max(abs(acceleration),axis=1).tolist(),
            'dense_interpolation_orientation_error_rad':max_orientation,'dense_interpolation_xy_error_m':max_xy,
            'min_wrist_singular_value':min_condition,'collision_checked':False}


def top_down_ik(xyz,seed=None,config=None,model=None):
    config=config or load_config();model=model or ExportedModel()
    target=np.eye(4);target[:3,:3]=np.diag([1.,-1.,-1.]);target[:3,3]=xyz
    solutions=ik_exported(target,seed=seed,model=model,limits=arm_limits(config))
    if not solutions:raise ValueError('No task-envelope top-down solution')
    return solutions[0]
