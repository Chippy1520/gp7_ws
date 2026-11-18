#!/usr/bin/env python3
"""OFFLINE ONLY: differential-IK vertical/polar path feasibility, not execution.
The initial aligned pickup waypoint is solved once. Starting-state recovery,
physical grasp and measured grasp calibration are deliberately not implemented.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from joint_trajectories import quintic_given_duration
from kinematics import ExportedModel
from pick_place_config import PKG, load_config, arm_limits
from pick_place_planner import Segment, spline_from_states, validate_segment, top_down_ik, continuous_extrema, jacobian_rate
from pick_place_collision import CollisionChecker


class CartesianPath:
    def __init__(self, start, goal, duration, polar=False,angular_delta=None):
        self.start=np.asarray(start,dtype=float);self.goal=np.asarray(goal,dtype=float)
        self.duration=duration;self.polar=polar
        if polar:
            a=np.arctan2(start[1],start[0]);b=np.arctan2(goal[1],goal[0])
            shortest=np.arctan2(np.sin(b-a),np.cos(b-a))
            if angular_delta is not None and (not np.isfinite(angular_delta) or abs(np.arctan2(np.sin(angular_delta-shortest),np.cos(angular_delta-shortest)))>1e-8):
                raise ValueError('Polar winding must reach the same endpoint')
            b=a+(shortest if angular_delta is None else angular_delta)
            self.curve=quintic_given_duration([np.hypot(*start[:2]),a,start[2]],
                                             [np.hypot(*goal[:2]),b,goal[2]],duration)
        else:self.curve=quintic_given_duration(self.start,self.goal,duration)

    def evaluate(self,t):
        s=self.curve.evaluate(float(t))
        if not self.polar:return s.position,s.velocity
        r,a,z=s.position;dr,da,dz=s.velocity;c=np.cos(a);sn=np.sin(a)
        return np.array([r*c,r*sn,z]),np.array([dr*c-r*sn*da,dr*sn+r*c*da,dz])

    def acceleration(self,t):
        """Analytical Cartesian acceleration, including polar centripetal terms."""
        s=self.curve.evaluate(float(t))
        if not self.polar:return s.acceleration
        r,a,z=s.position;dr,da,dz=s.velocity;ddr,dda,ddz=s.acceleration
        c,sn=np.cos(a),np.sin(a)
        return np.array([ddr*c-2*dr*sn*da-r*c*da*da-r*sn*dda,
                         ddr*sn+2*dr*c*da-r*sn*da*da+r*c*dda,ddz])


def differential_segment(name, start, path, config):
    model=ExportedModel();rotation=model.fk(start)[:3,:3];limits=arm_limits(config)
    evaluations=0
    def velocity(t,q):
        nonlocal evaluations
        evaluations+=1
        T,J=model.evaluate(q);p,v=path.evaluate(t)
        error=np.r_[p-T[:3,3],Rotation.from_matrix(rotation@T[:3,:3].T).as_rotvec()]
        twist=np.r_[v,np.zeros(3)]+4.*error
        # Constant small damping; singularity feasibility is checked separately.
        return J.T@np.linalg.solve(J@J.T+1e-12*np.eye(6),twist)
    began=time.monotonic()
    solution=solve_ivp(velocity,[0,path.duration],start,method='DOP853',
                       rtol=1e-10,atol=1e-12,first_step=min(.01,path.duration),max_step=.2,dense_output=True)
    if not solution.success:raise ValueError(solution.message)
    knot_step=min(.05,path.duration/40)
    times=np.linspace(0,path.duration,int(np.ceil(path.duration/knot_step))+1)
    q=solution.sol(times).T
    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError(name+': joint limits violated')
    dq=np.array([velocity(t,a) for t,a in zip(times,q)])
    ddq=[]
    for t,position,rate in zip(times,q,dq):
        J=model.evaluate(position)[1]
        desired_rate=np.r_[path.evaluate(t)[1],np.zeros(3)]
        desired_acceleration=np.r_[path.acceleration(t),np.zeros(3)]
        # Differentiate J*qdot = desired_twist + 4*pose_error. One-sided
        # differencing qdot at the endpoints incorrectly made rest acceleration
        # nonzero and introduced a discontinuity at subsequent guarded holds.
        ddq.append(np.linalg.solve(J,desired_acceleration+4*(desired_rate-J@rate)
                                    -jacobian_rate(model,position,rate)@rate))
    dq[[0,-1]]=0.  # Exact rest boundaries of the Cartesian quintic, not IK residual.
    ddq=np.array(ddq);ddq[[0,-1]]=0.
    segment=Segment(name,path.duration,spline_from_states(times,q,dq,ddq),
                    desired_rotation=rotation,
                    vertical_xy=path.start[:2] if not path.polar and np.allclose(path.start[:2],path.goal[:2]) else None)
    scale=1.
    if config.get('simulation_profile'):
        scale=max(1.,float(np.max(abs(continuous_extrema(segment.spline,1))))/config['velocity_limit_rad_s'],
                  np.sqrt(float(np.max(abs(continuous_extrema(segment.spline,2))))/config['acceleration_limit_rad_s2']))
        if scale>1.:
            scale*=1.05
            segment.spline=BPoly(segment.spline.c,segment.spline.x*scale)
            segment.duration*=scale
    report=validate_segment(segment,config,model)
    report.update(requested_duration_s=path.duration,time_scale_factor=float(scale))
    # Check between construction knots, not only the integrated knot values.
    errors=[];conditions=[]
    for t in np.linspace(0,segment.duration,int(np.ceil(segment.duration/.01))+1):
        T,J=model.evaluate(segment.evaluate(t));p,_=path.evaluate(t/scale)
        errors.append(np.linalg.norm(T[:3,3]-p))
        conditions.append(np.linalg.svd(J,compute_uv=False)[-1])
    if max(errors)>1e-5:raise ValueError(name+': interpolated Cartesian path error exceeds 10 micrometres')
    if min(conditions)<.005:raise ValueError(name+': full Jacobian singularity margin violated')
    report.update(generator='DOP853 differential IK with FK error feedback; no per-knot nonlinear IK',
                  construction_wall_s=time.monotonic()-began,rhs_evaluations=evaluations,
                  max_cartesian_error_m=max(errors),min_full_jacobian_singular_value=min(conditions))
    return segment,report


def main():
    config=load_config();output=PKG.parents[1]/'verification/pick_place/jacobian_polar_probe'
    output.mkdir(parents=True,exist_ok=True)
    result={'status':'FAIL','scope':'offline nominal geometry only; NO robot commands, grasp or startup recovery',
            'startup':'Measured home TCP was 0.815 m and wrist singular; requires a separate checked transition to 0.50 m aligned pickup',
            'grasp':'UNCALIBRATED TCP-at-cube-centre hypothesis','phases':[]}
    segments=[];started=time.monotonic()
    try:
        pickup=np.array(config['cube']['initial_center_xyz_m']);above=pickup.copy();above[2]=.5
        place=np.array(config['placement']['center_xyz_m']);hover=place.copy();hover[2]+=.05
        lifted=pickup.copy();lifted[2]=hover[2];retreat=place.copy();retreat[2]+=.10
        q=top_down_ik(above,config=config);initial=q.copy();last=above
        checker=CollisionChecker(clearance=config['planning']['collision_clearance_m'],self_clearance=config['planning']['self_collision_clearance_m'])
        tcp_cube=np.diag([1.,-1.,-1.,1.]);placed=np.eye(4);placed[:3,3]=place
        specs=[('DESCEND_PICK',pickup,40.,False),('LIFT',lifted,28.,False),
               ('TRANSFER',hover,30.,True),('LOWER',place,10.,False),('RETREAT',retreat,16.,False)]
        for name,goal,duration,polar in specs:
            print('PREPARE',name,flush=True)
            path=CartesianPath(last,goal,duration,polar)
            segment,report=differential_segment(name,q,path,config)
            held=name in ['LIFT','TRANSFER','LOWER']
            support='ground' if name=='LIFT' else 'box_a' if name=='LOWER' else None
            print('CHECK COLLISIONS',name,flush=True)
            report['collision']=checker.validate(segment,jaws=(.010,.010) if held else (0.,0.),
                holding=held,tcp_cube=tcp_cube if held else None,support=support,
                cube_world=placed if name=='RETREAT' else None,
                period=config['planning']['collision_sample_period_s'])
            result['phases'].append(report);segments.append(segment)
            q=segment.evaluate(duration);last=goal
            print('PASS OFFLINE',name,flush=True)
        result.update(status='PASS_OFFLINE_ALIGNED_PICKUP_TO_RETREAT',initial_aligned_q=initial.tolist(),wall_seconds=time.monotonic()-started)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        figure=Figure(figsize=(9,7));axes=figure.add_subplot(111,projection='3d');model=ExportedModel()
        display=[]
        for segment in segments:
            times=np.linspace(0,segment.duration,201);angles=segment.evaluate(times)
            points=np.array([model.fk(a)[:3,3] for a in angles])
            axes.plot(*points.T,label=segment.name)
            display.append({'name':segment.name,'time':times.tolist(),'q':angles.tolist(),'tcp':points.tolist()})
        axes.set(xlabel='Base X [m]',ylabel='Base Y [m]',zlabel='Base Z [m]',title='OFFLINE Jacobian path — startup transition not included')
        axes.legend();figure.tight_layout();figure.savefig(output/'path.png',dpi=140)
        (output/'display_plan.json').write_text(json.dumps({'stages':display},indent=2)+'\n')
    except Exception as error:
        result['error']=str(error);raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
