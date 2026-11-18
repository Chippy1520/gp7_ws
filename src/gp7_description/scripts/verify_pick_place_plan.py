#!/usr/bin/env python3
"""Build and validate a nominal route offline. Never publishes motion commands."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kinematics import ExportedModel
from pick_place_config import PKG,load_config
from pick_place_planner import (joint_segment,vertical_segment,compensated_segment,
                                top_down_ik,validate_segment)
from pick_place_collision import CollisionChecker


def nominal_route(start=None,config=None,stop_after_lift=False):
    config=config or load_config();start=np.zeros(6) if start is None else np.array(start)
    home=np.zeros(6);pickup=np.array(config['cube']['initial_center_xyz_m'])
    above=pickup.copy();above[2]=config['placement']['hover_center_xyz_m'][2]
    above_q=top_down_ik(above,seed=home,config=config)
    unrotated=above_q.copy();unrotated[3:]=0.
    segments=[joint_segment('HOME',start,home,6.),joint_segment('ABOVE_PICKUP',home,unrotated,15.),
              joint_segment('ORIENT_DOWN',unrotated,above_q,14.)]
    descend=vertical_segment('DESCEND_PICK',above_q,pickup[2],28.,config)
    segments.append(descend);grasp_q=descend.evaluate(descend.duration)
    lift=vertical_segment('LIFT',grasp_q,above[2],28.,config);segments.append(lift)
    # Nominal TCP-at-cube-centre hypothesis, never a calibrated physical grasp.
    tcp_cube=np.eye(4);tcp_cube[:3,:3]=np.diag([1.,-1.,-1.])
    if stop_after_lift:return segments,tcp_cube
    lifted=lift.evaluate(lift.duration)
    hover_q=top_down_ik(config['placement']['hover_center_xyz_m'],seed=lifted,config=config)
    transfer=compensated_segment('TRANSFER',lifted,hover_q,24.,config);segments.append(transfer)
    lower=vertical_segment('LOWER',transfer.evaluate(transfer.duration),config['placement']['center_xyz_m'][2],16.,config)
    segments.append(lower)
    retreat=vertical_segment('RETREAT',lower.evaluate(lower.duration),.425,20.,config);segments.append(retreat)
    segments.append(joint_segment('RETURN_HOME',retreat.evaluate(retreat.duration),home,24.))
    return segments,tcp_cube


def verify_route(start=None):
    config=load_config();model=ExportedModel();out=PKG.parents[1]/'verification/pick_place';out.mkdir(parents=True,exist_ok=True)
    report={'status':'FAIL','scope':'offline nominal reference feasibility; no physical grasp or execution certified',
            'grasp_transform':'UNCALIBRATED TCP-at-cube-centre hypothesis','phases':[]}
    try:
        segments,tcp_cube=nominal_route(start,config)
        checker=CollisionChecker(clearance=config['planning']['collision_clearance_m'],
                                 self_clearance=config['planning']['self_collision_clearance_m'])
        report['collision_policy']={'environment_clearance_m':checker.clearance,'self_clearance_m':checker.self_clearance,
                                     'self_geometry':'CAD mesh refinement within checked simulator broad-phase boxes',
                                     'environment_geometry':'actual task SDF boxes','point_reach_bound_m':checker.point_reach_bound}
        placed=np.eye(4);placed[:3,3]=config['placement']['center_xyz_m']
        arrays={}
        for segment in segments:
            phase=validate_segment(segment,config,model)
            carried=segment.name in ['LIFT','TRANSFER','LOWER']
            support='ground' if segment.name=='LIFT' else 'box_a' if segment.name=='LOWER' else None
            phase['collision']=checker.validate(segment,jaws=(.012,.012) if carried else (0.,0.),
                holding=carried,tcp_cube=tcp_cube if carried else None,support=support,
                period=config['planning']['collision_sample_period_s'],cube_world=placed if segment.name in ['RETREAT','RETURN_HOME'] else None)
            phase['collision_checked']=True
            report['phases'].append(phase)
            times=np.linspace(0,segment.duration,int(np.ceil(segment.duration/.02))+1)
            arrays[segment.name+'_time']=times
            arrays[segment.name+'_q']=segment.evaluate(times)
            arrays[segment.name+'_qdot']=segment.evaluate(times,1)
            arrays[segment.name+'_qddot']=segment.evaluate(times,2)
            print('PASS offline',segment.name,flush=True)
        # Explicit physical contact exceptions are confined to stationary-arm
        # jaw closure/opening and the grasped cube. No attachment is performed.
        contacts=[]
        for name,index,support in [('GRASP',3,'ground'),('RELEASE',6,'box_a')]:
            q=segments[index].evaluate(segments[index].duration)
            cube=model.fk(q)@tcp_cube
            for stroke in np.linspace(0,.012,121):
                result=checker.check(q,jaws=(stroke,stroke),cube_transform=cube,holding=True,support=support)
                if not result['safe']:raise ValueError(name+' contact sweep: '+str(result['violations']))
            contacts.append({'phase':name,'arm_stationary':True,'jaw_samples':121,'allowed_contacts':'fingers-cube and cube-support only'})
        report['contact_checks']=contacts
        report['status']='PASS_NOMINAL_OFFLINE_ONLY'
        report['requires_before_execution']=['measured start-state revalidation','physical grasp/contact calibration',
            'measured relative grasp transform','ROS-time executor with feedback/collision/clock abort gates']
        np.savez_compressed(out/'nominal_reference.npz',**arrays)
    except Exception as error:
        report['error']=str(error);raise
    finally:
        report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PKG/'scripts/pick_place_planner.py',PKG/'scripts/pick_place_collision.py',PKG/'config/pick_place.yaml',PKG/'worlds/gp7_pick_place.sdf',PKG/'urdf/gp7_with_gripper.urdf',*sorted((PKG/'meshes').glob('*.stl'))]}
        (out/'nominal_plan_gate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':print(json.dumps(verify_route(),indent=2))
