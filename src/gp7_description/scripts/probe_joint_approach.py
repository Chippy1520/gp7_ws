#!/usr/bin/env python3
"""Offline current-start joint approach + Jacobian descent check. Never commands ROS."""
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from audit_model import fk as link_fk
from kinematics import ExportedModel,ik_exported
from pick_place_config import PKG,NAMES,load_config,arm_limits
from pick_place_planner import joint_segment,continuous_extrema,validate_segment
from pick_place_collision import CollisionChecker
from probe_jacobian_polar_route import CartesianPath,differential_segment


def joint_approach(start,goal,config):
    unit=joint_segment('APPROACH_PICKUP',start,goal,1.)
    velocity=float(np.max(abs(continuous_extrema(unit.spline,1))))
    acceleration=float(np.max(abs(continuous_extrema(unit.spline,2))))
    duration=max(config['planning'].get('minimum_approach_duration_s',2.),1.1*max(velocity/config['velocity_limit_rad_s'],
                          np.sqrt(acceleration/config['acceleration_limit_rad_s2'])))
    return joint_segment('APPROACH_PICKUP',start,goal,duration)


def gripper_bottom(checker,q):
    frames=link_fk(checker.urdf,dict(zip(NAMES+['gripper_joint_positive','gripper_joint_negative'],[*q,0.,0.])))
    minimum=np.inf
    for name in ['gripper_palm','gripper_finger_positive','gripper_finger_negative']:
        geometry,local=checker.boxes[name];T=frames[name]@local
        corners=np.array(list(product([-.5,.5],repeat=3)))*np.asarray(geometry.side)
        minimum=min(minimum,float(((T[:3,:3]@corners.T).T+T[:3,3])[:,2].min()))
    return minimum


def main():
    config=load_config();root=PKG.parents[1]/'verification/pick_place'
    out=root/'joint_approach_probe';out.mkdir(parents=True,exist_ok=True)
    measured=json.loads((out/'measured_start.json').read_text())
    q=np.array(measured['q'][:6]);v=np.array(measured['velocity'])
    if len(v)!=8 or max(abs(v))>.0008 or max(abs(np.array(measured['q'][6:])))>.0002:
        raise RuntimeError('Probe requires a settled measured start with open jaws')
    model=ExportedModel();checker=CollisionChecker(clearance=config['planning']['collision_clearance_m'],self_clearance=config['planning']['self_collision_clearance_m'])
    clearance=config['pedestal']['base_xyz_m'][2]+config['pedestal']['size_m'][2]+config['planning']['collision_clearance_m']
    initial_bottom=gripper_bottom(checker,q)
    result={'status':'FAIL','scope':'OFFLINE current measured-start approach and revised descent only; no robot commands',
            'measured_start':measured,'initial_tcp':model.fk(q).tolist(),
            'minimum_required_gripper_height_m':clearance,'initial_gripper_bottom_m':initial_bottom,
            'candidate_policy':'first passing height from ascending finite list; NOT a global optimum','candidates':[]}
    started=time.monotonic()
    try:
        if initial_bottom<clearance:raise RuntimeError('This measured start needs a separate clearance-rise plan; no direct approach permitted')
        result['clearance_rise']='SKIPPED: initial gripper is above pedestal clearance; complete approach still checked'
        selected=None
        for height in [.25,.30,.35,.40,.45,.50]:
            xyz=np.array(config['cube']['initial_center_xyz_m']);xyz[2]=height
            T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=xyz
            solutions=ik_exported(T,seed=q,model=model,limits=arm_limits(config))
            item={'height_m':height,'ik_branches':len(solutions),'status':'REJECTED'}
            for goal in solutions:
                segment=joint_approach(q,goal,config)
                try:
                    report=validate_segment(segment,config,model)
                    bottom=min(gripper_bottom(checker,segment.evaluate(t)) for t in np.linspace(0,segment.duration,241))
                    peak=np.max(abs(continuous_extrema(segment.spline,1)),axis=1)
                    inflation=checker.point_reach_bound*float(peak.sum())*(segment.duration/240)/2
                    if bottom-inflation<clearance:raise ValueError('Gripper sweeps below the pedestal clearance plane, including between-sample bound')
                    report['gripper_height_between_sample_inflation_m']=inflation
                    report['collision']=checker.validate(segment,period=config['planning']['collision_sample_period_s'])
                    J=model.evaluate(goal)[1];minimum=float(np.linalg.svd(J,compute_uv=False)[-1])
                    if minimum<.005:raise ValueError('Endpoint unsuitable for Jacobian stage')
                    item.update(status='PASS',report=report,minimum_sampled_gripper_bottom_m=bottom,
                                endpoint_min_jacobian_singular_value=minimum)
                    selected=segment;break
                except ValueError as error:item['reason']=str(error)
            result['candidates'].append(item)
            if selected is not None:break
        if selected is None:raise RuntimeError('No checked direct joint approach among candidate heights')
        approach_end=selected.evaluate(selected.duration)
        path=CartesianPath(model.fk(approach_end)[:3,3],config['cube']['initial_center_xyz_m'],28.)
        descent,report=differential_segment('DESCEND_PICK',approach_end,path,config)
        report['collision']=checker.validate(descent,period=config['planning']['collision_sample_period_s'])
        result['descent']=report
        previous=json.loads((root/'jacobian_polar_probe/display_plan.json').read_text())
        earlier_grasp=np.array(previous['stages'][0]['q'][-1])
        difference=float(max(abs(descent.evaluate(descent.duration)-earlier_grasp)))
        if difference>1e-7:raise RuntimeError('New descent does not join earlier Jacobian lift branch')
        result.update(status='PASS_OFFLINE_DIRECT_APPROACH_AND_DESCENT',selected_height_m=height,
                      approach_duration_s=selected.duration,downstream_joint_join_error_rad=difference,
                      downstream_scope='Endpoint matches earlier five-stage offline trial; downstream stages NOT revalidated in this run')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        figure=Figure(figsize=(9,7));axes=figure.add_subplot(111,projection='3d')
        display=[]
        for segment in [selected,descent]:
            times=np.linspace(0,segment.duration,241);angles=segment.evaluate(times)
            points=np.array([model.fk(a)[:3,3] for a in angles])
            axes.plot(*points.T,label=segment.name+' (checked now)')
            display.append({'name':segment.name,'time':times.tolist(),'q':angles.tolist(),'tcp':points.tolist()})
        for stage in previous['stages'][1:]:
            points=np.array(stage['tcp']);axes.plot(*points.T,'--',label=stage['name']+' (prior offline trial)')
        axes.set(xlabel='Base X [m]',ylabel='Base Y [m]',zlabel='Base Z [m]',title='OFFLINE: joint-space approach, then Cartesian task path')
        axes.legend(fontsize=8);figure.tight_layout();figure.savefig(out/'path.png',dpi=140)
        (out/'display_plan.json').write_text(json.dumps({'stages':display},indent=2)+'\n')
    except Exception as error:
        result['error']=str(error);raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        files=[Path(__file__),PKG/'scripts/probe_jacobian_polar_route.py',PKG/'config/pick_place.yaml',PKG/'worlds/gp7_pick_place.sdf',PKG/'urdf/gp7_with_gripper.urdf']
        result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'selected_height_m':height,'approach_duration_s':selected.duration,'output':str(out)}))


if __name__=='__main__':main()
