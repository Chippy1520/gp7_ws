#!/usr/bin/env python3
"""Offline, non-pickle nominal spline cache. Not physical grasp certification."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from scipy.interpolate import BPoly
from joint_trajectories import QuinticTrajectory
from kinematics import ExportedModel
from pick_place_config import PKG
from pick_place_planner import Segment, validate_segment, joint_segment
from pick_place_collision import CollisionChecker
from verify_pick_place_plan import nominal_route

SCHEMA = 1
LIFT_NAMES = ['HOME', 'ABOVE_PICKUP', 'ORIENT_DOWN', 'DESCEND_PICK', 'LIFT']
FULL_NAMES = LIFT_NAMES + ['TRANSFER', 'LOWER', 'RETREAT', 'RETURN_HOME']


def emit(event, **fields):
    print('GP7_MONITOR '+json.dumps({'event':event, **fields}, allow_nan=False, separators=(',', ':')), flush=True)


def status(stage, state, message):
    emit('status', stage=stage, state=state, message=message)


def fingerprint(config):
    paths = [PKG/'scripts'/name for name in (
        'pick_place_cache.py', 'pick_place_executor.py', 'pick_place_feedback.py',
        'pick_place_planner.py', 'verify_pick_place_plan.py', 'pick_place_collision.py',
        'pick_place_config.py', 'kinematics.py', 'joint_trajectories.py',
        'verify_pick_place_controllers.py', 'audit_model.py')]
    paths += list((PKG/'config').glob('*')) + list((PKG/'worlds').glob('*.sdf'))
    paths += list((PKG/'urdf').glob('*.urdf')) + list((PKG/'meshes').glob('*.stl'))
    sources = {str(p.relative_to(PKG)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(paths) if p.is_file()}
    return {'source_sha256':sources, 'config':config}


def checker_for(config):
    return CollisionChecker(clearance=config['planning']['collision_clearance_m'],
                            self_clearance=config['planning']['self_collision_clearance_m'])


def validate_nominal(segments, tcp_cube, config):
    checker=checker_for(config);model=ExportedModel();reports=[]
    placed=np.eye(4);placed[:3,3]=config['placement']['center_xyz_m']
    for segment in segments:
        status(segment.name, 'preparing', 'Validating nominal limits, interpolation and collisions')
        report=validate_segment(segment,config,model)
        held=segment.name in ['LIFT','TRANSFER','LOWER']
        support='ground' if segment.name=='LIFT' else 'box_a' if segment.name=='LOWER' else None
        report['collision']=checker.validate(segment,jaws=(.012,.012) if held else (0.,0.),
            holding=held,tcp_cube=tcp_cube if held else None,support=support,
            period=config['planning']['collision_sample_period_s'],
            cube_world=placed if segment.name in ['RETREAT','RETURN_HOME'] else None)
        report['collision_checked']=True;reports.append(report)
    contacts=[]
    for name,index,support in [('GRASP',3,'ground'),('RELEASE',6,'box_a')]:
        if index>=len(segments):continue
        q=segments[index].evaluate(segments[index].duration);cube=model.fk(q)@tcp_cube
        for stroke in np.linspace(0,.012,121):
            result=checker.check(q,jaws=(stroke,stroke),cube_transform=cube,holding=True,support=support)
            if not result['safe']:raise ValueError(name+' contact sweep: '+str(result['violations']))
        contacts.append({'phase':name,'jaw_samples':121})
    return reports,contacts


def save_cache(directory, segments, tcp_cube, config, reports, contacts, stop_after_lift):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays={'tcp_cube':tcp_cube};metadata=[]
    for i,s in enumerate(segments):
        prefix=f's{i}_';entry={'name':s.name,'duration':s.duration}
        arrays[prefix+'c']=s.spline.c;arrays[prefix+'x']=s.spline.x
        for field in ['desired_rotation','vertical_xy']:
            if getattr(s,field) is not None:arrays[prefix+field]=getattr(s,field)
        if s.arm is not None:arrays[prefix+'arm']=s.arm.coefficients
        metadata.append(entry)
    # Unique arrays + metadata replaced last: concurrent readers see a complete generation.
    filename='splines-'+uuid.uuid4().hex+'.npz';np.savez_compressed(directory/filename,**arrays)
    document={'schema':SCHEMA,'status':'PASS_NOMINAL_OFFLINE_ONLY',
        'stop_after_lift':bool(stop_after_lift),'fingerprint':fingerprint(config),
        'arrays':filename,'arrays_sha256':hashlib.sha256((directory/filename).read_bytes()).hexdigest(),
        'segments':metadata,'phases':reports,'contact_checks':contacts,
        'start_tolerance_rad':.003,'scope':'nominal only; measured grasp always replanned'}
    temporary=directory/('metadata-'+uuid.uuid4().hex+'.json')
    temporary.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,directory/'metadata.json')


def load_cache(directory, config, stop_after_lift=False):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
    if meta['schema']!=SCHEMA or meta['status']!='PASS_NOMINAL_OFFLINE_ONLY':raise ValueError('Invalid nominal cache gate')
    if meta['fingerprint']!=fingerprint(config):raise ValueError('Nominal cache source/geometry/config fingerprint is stale')
    if meta['stop_after_lift']!=bool(stop_after_lift):raise ValueError('Nominal cache stage scope mismatch')
    names=LIFT_NAMES if stop_after_lift else FULL_NAMES
    if [s['name'] for s in meta['segments']]!=names:raise ValueError('Invalid cached stage sequence')
    if [s['phase'] for s in meta['phases']]!=names or not all(s['collision_checked'] for s in meta['phases']):
        raise ValueError('Missing cached validation evidence')
    if [s['phase'] for s in meta['contact_checks']] != (['GRASP'] if stop_after_lift else ['GRASP','RELEASE']):
        raise ValueError('Missing cached contact sweep evidence')
    filename=meta['arrays']
    if Path(filename).name!=filename:raise ValueError('Invalid cache array path')
    path=directory/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['arrays_sha256']:raise ValueError('Cache array checksum mismatch')
    segments=[]
    with np.load(path,allow_pickle=False) as data:
        for key in data.files:
            if not np.issubdtype(data[key].dtype,np.number) or not np.isfinite(data[key]).all():raise ValueError('Invalid cache arrays')
        for i,entry in enumerate(meta['segments']):
            p=f's{i}_';duration=float(entry['duration']);x=data[p+'x'];c=data[p+'c']
            if duration<=0 or x.ndim!=1 or x[0]!=0 or x[-1]!=duration or np.any(np.diff(x)<=0):raise ValueError('Invalid spline time domain')
            if c.shape!=(6,len(x)-1,6):raise ValueError('Invalid spline coefficients')
            arm=QuinticTrajectory(duration,data[p+'arm'].copy()) if p+'arm' in data else None
            if arm is not None and arm.coefficients.shape!=(6,3):raise ValueError('Invalid arm polynomial')
            rotation=data[p+'desired_rotation'].copy() if p+'desired_rotation' in data else None
            xy=data[p+'vertical_xy'].copy() if p+'vertical_xy' in data else None
            segments.append(Segment(entry['name'],duration,BPoly(c.copy(),x.copy()),arm,rotation,xy))
        tcp_cube=data['tcp_cube'].copy()
    for a,b in zip(segments,segments[1:]):
        if not np.allclose(a.evaluate(a.duration),b.evaluate(0),atol=1e-8,rtol=0):raise ValueError('Discontinuous cache')
    return segments,tcp_cube,meta


def prepare(directory, config, stop_after_lift=False):
    started=time.monotonic();status('PREPARE','preparing','Preparing offline nominal route; no robot commands')
    try:
        result=load_cache(directory,config,stop_after_lift);mode='cache_load'
    except (OSError,ValueError,KeyError,TypeError):
        before=fingerprint(config)
        segments,tcp_cube=nominal_route(config=config,stop_after_lift=stop_after_lift)
        reports,contacts=validate_nominal(segments,tcp_cube,config)
        if before!=fingerprint(config):raise RuntimeError('Sources changed during preparation')
        save_cache(directory,segments,tcp_cube,config,reports,contacts,stop_after_lift)
        result=load_cache(directory,config,stop_after_lift);mode='built_and_validated'
    seconds=time.monotonic()-started
    status('PREPARE','ready',f'{mode}: {seconds:.3f} wall seconds (nominal offline only)')
    return (*result,{'mode':mode,'wall_seconds':seconds})


def measured_start(segments, q, velocity, config, checker, cube):
    """Never splice arbitrary feedback into a cached curve. Revalidate tiny HOME connector."""
    q=np.asarray(q);velocity=np.asarray(velocity)
    if q.shape!=(8,) or velocity.shape!=(8,) or not np.isfinite(q).all() or not np.isfinite(velocity).all():raise ValueError('Invalid measured start')
    if np.max(abs(q[:6]-segments[0].evaluate(0)))>.003:raise ValueError('Measured start outside cache tolerance (0.003 rad); prepare/reposition explicitly')
    if np.max(abs(velocity[:6]))>.008 or np.max(abs(velocity[6:]))>.0008 or np.max(abs(q[6:]))>.0002:
        raise ValueError('Cached route requires settled measured arm and open jaws')
    if cube is None or np.linalg.norm(cube[:3,3]-config['cube']['initial_center_xyz_m'])>.003:
        raise ValueError('Cached route requires measured initial cube geometry')
    connector=joint_segment('HOME',q[:6],segments[0].evaluate(segments[0].duration),segments[0].duration)
    validate_segment(connector,config)
    checker.validate(connector,period=config['planning']['collision_sample_period_s'],cube_world=cube)
    return [connector,*segments[1:]]


def display_plan(segments, output):
    model=ExportedModel();stages=[]
    for s in segments:
        times=np.linspace(0,s.duration,min(241,max(2,int(np.ceil(s.duration/.15))+1)))
        q=s.evaluate(times)
        stages.append({'name':s.name,'duration':s.duration,'time':times.tolist(),'q':q.tolist(),
            'qd':s.evaluate(times,1).tolist(),'qdd':s.evaluate(times,2).tolist(),
            'tcp':[model.fk(v)[:3,3].tolist() for v in q]})
    path=Path(output).resolve();path.write_text(json.dumps({'display_only':True,'stages':stages},allow_nan=False)+'\n')
    emit('plan',path=str(path));return path
