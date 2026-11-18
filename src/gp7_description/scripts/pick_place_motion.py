#!/usr/bin/env python3
"""Shared Gazebo planning, guarded motion stages and execution lifecycle.
Edit the high-level sequence and timing settings in task.py.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import signal
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation
from kinematics import ik_exported
from pick_place_config import PKG,load_config,arm_limits
from pick_place_executor import Executor
from pick_place_planner import joint_segment,continuous_extrema,validate_segment
from pick_place_pickup_area import load_grasp_workspace,in_grasp_workspace
from pick_place_profiles import PROFILES
from pick_place_cache import display_plan,status,emit
from probe_joint_approach import joint_approach,gripper_bottom
from probe_jacobian_polar_route import CartesianPath,differential_segment
from verify_pick_place_controllers import live_audit
from prepare_pick_place_scene import service
from google.protobuf import text_format
from gz.msgs10.boolean_pb2 import Boolean


def fresh(e):
    end=time.monotonic()+5
    while time.monotonic()<end:
        time.sleep(.002);rclpy.spin_once(e.node,timeout_sec=.005)
        if getattr(e, 'abort_requested', lambda: False)():
            raise KeyboardInterrupt('Operator requested Stop')
        if e.error:raise RuntimeError(e.error)
        data=e.native.snapshot();now=time.monotonic()
        if (e.sim is not None and e.q is not None and data['cube'] is not None
                and data['tcp'] is not None and abs(e.sim-data['cube_stamp'])<.05
                and max(now-e.feedback_wall,now-e.clock_wall,now-data['cube_wall'],now-data['tcp_wall'])<.1):return data
    raise RuntimeError('No fresh synchronized joint/clock/cube/TCP feedback')


def checked(e,segment,holding=False,support=None,cube=None,kinematic_report=None,tcp_cube=None):
    print('CHECK',segment.name,flush=True)
    started=time.monotonic()
    report=validate_segment(segment,e.config) if kinematic_report is None else dict(kinematic_report)
    if kinematic_report is not None and (report['phase']!=segment.name or report['duration_s']!=segment.duration):
        raise ValueError('Kinematic validation does not match this segment')
    if e.config.get('execution', {}).get('workspace_only'):
        report.update(collision_checked=False, preparation_wall_s=time.monotonic()-started)
        return report
    if not holding and support is None:support='ground'
    periods=[.02,.01,.005,.0025,.00125,.000625] if getattr(e.checker,'cube_only',False) else [e.config['planning']['collision_sample_period_s']]
    guard=None
    if getattr(e,'commanding',False) is True:
        def guard():
            # Differential IK can leave ROS callbacks queued while native poses
            # advance. Synchronize both streams BEFORE applying association gates.
            fresh(e);e.spin()
    for period in periods:
        try:
            report['collision']=e.checker.validate(segment,jaws=(.010,.010) if holding else (0.,0.),
                holding=holding,tcp_cube=(e.grasp_transform if tcp_cube is None else tcp_cube) if holding else None,
                support=support,cube_world=cube,period=period,
                progress_callback=guard)
            break
        except ValueError:
            if period==periods[-1]:raise
    report['preparation_wall_s']=time.monotonic()-started
    report['collision_checked']=True
    report['reused_kinematic_validation']=kinematic_report is not None
    return report


def accept_pickup(config,cube,use_live=False):
    if cube.shape!=(4,4) or not np.isfinite(cube).all():
        raise RuntimeError('Invalid measured cube pose')
    if not use_live:
        if np.linalg.norm(cube[:3,3]-config['cube']['initial_center_xyz_m'])>.003:
            raise RuntimeError('Cube is not at pickup')
        return
    # This runner picks from the floor, not from arbitrary surfaces.
    bottom=cube[2,3]-.025*abs(cube[2,:3]).sum()
    if abs(bottom)>.001 or cube[2,2]<np.cos(np.deg2rad(1.)):
        raise RuntimeError('Live pickup requires an upright cube resting on the ground')
    config['cube']['initial_center_xyz_m']=cube[:3,3].tolist()


def clearance_height(config):
    pedestal=config['pedestal']
    return pedestal['base_xyz_m'][2]+pedestal['size_m'][2]+config['planning']['collision_clearance_m']


def check_workspace_start(e,cube):
    accept_pickup(e.config,cube,True)
    survey=load_grasp_workspace()
    if survey is None:raise RuntimeError('Sprint workspace is missing or stale; regenerate it before running')
    if not in_grasp_workspace(*cube[:2,3],survey):
        raise RuntimeError('Cube center is outside the blue Sprint pickup area')
    if Rotation.from_matrix(cube[:3,:3]).magnitude()>np.deg2rad(1.):
        raise RuntimeError('Sprint workspace assumes the cube is aligned with the world axes')
    bottom=gripper_bottom(e.checker,e.q[:6]);height=clearance_height(e.config)
    return {'gripper_bottom_m':bottom,'clearance_height_m':height,'needs_rise':bool(bottom<height)}


def approach(e,cube,timings,pickup_height,*,prefer_hover=False):
    q=e.q[:6].copy();segments=[];reports=[]
    floor=clearance_height(e.config)
    quick=e.config.get('execution', {}).get('workspace_only',False)
    bottom=gripper_bottom(e.checker,q)
    if bottom<floor:
        start=e.model.fk(q)[:3,3];goal=start.copy();goal[2]+=floor-bottom+.03
        segment,kinematics=differential_segment('CLEARANCE_RISE',q,CartesianPath(start,goal,timings['rise']),e.config)
        reports.append(checked(e,segment,cube=cube,kinematic_report=kinematics));segments.append(segment);q=segment.evaluate(segment.duration)
    target=cube.copy();target[:3,:3]=cube[:3,:3]@np.diag([1.,-1.,-1.])
    # Rank only cheap endpoint/spline estimates. Integrate and collision-check
    # lazily, stopping at the first *pair* that passes, not a whole-cycle optimum.
    limits=arm_limits(e.config)
    diagnostics={'policy':'ranked finite approach candidates; first validated approach/descent pair, not global cycle optimum',
                 'counts':dict.fromkeys(['no_ik','joint_limits','conditioning','clearance','approach_check','descent'],0),
                 'candidate_count':0,'attempted_count':0,'rejections':[],'selected':None}
    e.approach_diagnostics=diagnostics
    def reject(reason,height,branch,error):
        diagnostics['counts'][reason]+=1
        diagnostics['rejections'].append({'reason':reason,'height_m':float(height),'branch':branch,'error':str(error)})
    heights=pickup_height + np.arange(6)*.05
    batches=[heights]
    if prefer_hover:
        # Try one hover before enumerating alternate heights. The same path
        # checks still apply, and the fallback never repeats this height.
        nearest=int(np.argmin(abs(heights-.40)))
        batches=[heights[nearest:nearest+1],np.delete(heights,nearest)]
        diagnostics['policy']='preferred hover first; ranked fallback with full path checks'
    for batch in batches:
        candidates=[]
        for height in batch:
            target[2,3]=height
            try:goals=list(ik_exported(target,seed=q,model=e.model,limits=limits))
            except (ValueError,RuntimeError) as error:
                reject('no_ik',height,None,error);continue
            if not goals:reject('no_ik',height,None,'IK returned no solutions within configured arm limits')
            for branch,goal in enumerate(goals):
                diagnostics['candidate_count']+=1
                goal=np.asarray(goal,dtype=float)
                if goal.shape!=(6,) or not np.isfinite(goal).all() or np.any(goal<limits[:,0]) or np.any(goal>limits[:,1]):
                    reject('joint_limits',height,branch,'IK endpoint outside configured arm limits or nonfinite');continue
                try:
                    sigma=float(np.linalg.svd(e.model.evaluate(goal)[1],compute_uv=False)[-1])
                    if not np.isfinite(sigma) or sigma<.005:
                        reject('conditioning',height,branch,f'Endpoint Jacobian sigma_min={sigma:.6g} < .005');continue
                except np.linalg.LinAlgError as error:
                    reject('conditioning',height,branch,error);continue
                try:
                    segment=joint_approach(q,goal,e.config)
                    if timings['approach'] is not None:
                        segment=joint_segment(segment.name,q,goal,max(segment.duration,timings['approach']))
                except (ValueError,RuntimeError) as error:
                    reject('approach_check',height,branch,error);continue
                travel=float(np.linalg.norm(goal-q))
                margin=float(np.min(np.minimum(goal-limits[:,0],limits[:,1]-goal)/(limits[:,1]-limits[:,0])))
                score=float(segment.duration+.1*travel+.0025/sigma+.01/(margin+.01))
                stats={'height_m':float(height),'branch':branch,'score':score,'approach_duration_s':float(segment.duration),
                       'joint_travel_rad':travel,'endpoint_sigma_min':sigma,'normalized_limit_margin':margin}
                candidates.append((score,len(candidates),goal,segment,stats))
        for _,_,goal,segment,stats in sorted(candidates,key=lambda item:item[:2]):
            height=stats['height_m'];branch=stats['branch'];diagnostics['attempted_count']+=1
            try:
                if not quick:
                    peak=np.max(abs(continuous_extrema(segment.spline,1)),axis=1)
                    bound=e.checker.point_reach_bound*float(peak.sum())*(segment.duration/240)/2
                    low=min(gripper_bottom(e.checker,a) for a in segment.evaluate(np.linspace(0,segment.duration,241)))
                    if low-bound<floor:
                        reject('clearance',height,branch,f'Gripper lower bound {low-bound:.6g} m < required {floor:.6g} m (sample minimum {low:.6g}, inflation {bound:.6g})');continue
                report=checked(e,segment,cube=cube)
            except (ValueError,RuntimeError) as error:
                reject('approach_check',height,branch,error);continue
            try:
                start=e.model.fk(goal)[:3,3]
                descent,kinematics=differential_segment('DESCEND_PICK',goal,CartesianPath(start,cube[:3,3],timings['descent']),e.config)
                descent_report=checked(e,descent,cube=cube,kinematic_report=kinematics)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                reject('descent',height,branch,error);continue
            # Do not commit rejected approaches/reports, including on descent FCL failure.
            diagnostics['selected']=stats
            report['branch_selection']=diagnostics
            segments.extend([segment,descent]);reports.extend([report,descent_report])
            print('APPROACH SELECTED',json.dumps(stats),flush=True)
            diagnostics['used_preferred_hover']=bool(prefer_hover and len(batch)==1)
            return segments,reports
    recent=diagnostics['rejections'][-8:]
    raise RuntimeError('No validated approach/descent pair; rejection counts='+json.dumps(diagnostics['counts'])+
                       '; recent source errors='+json.dumps(recent))


def carry(e,lift_only,timings,hover_clearance,retreat_distance,*,q_start=None,cube=None,grasp_transform=None,check_collisions=True):
    q=e.q[:6].copy() if q_start is None else np.array(q_start).copy()
    cube=e.native.snapshot()['cube'].copy() if cube is None else cube.copy()
    grasp=e.grasp_transform if grasp_transform is None else grasp_transform
    placed=cube.copy();placed[:3,3]=e.config['placement']['center_xyz_m']
    placed[2,3]=e.config['pedestal']['size_m'][2]+.025*abs(placed[2,:3]).sum()
    hover=placed.copy();hover[2,3]+=hover_clearance
    lifted=cube.copy();lifted[2,3]=hover[2,3]
    retreat=placed.copy();retreat[2,3]+=retreat_distance
    specs=[('LIFT',lifted,timings['lift'],False,'ground')]
    if not lift_only:specs += [('TRANSFER',hover,timings['transfer'],True,None),('LOWER',placed,timings['lower'],False,'box_a'),('RETREAT',retreat,timings['retreat'],False,'box_a')]
    segments=[];reports=[]
    for name,goal,duration,polar,support in specs:
        target=goal@np.linalg.inv(grasp);start=e.model.fk(q)[:3,3]
        windings=[None]
        if polar:
            raw=np.arctan2(target[1,3],target[0,3])-np.arctan2(start[1],start[0])
            short=np.arctan2(np.sin(raw),np.cos(raw))
            windings=[short] if abs(raw-short)<1e-8 else [short,raw]
        failures=[]
        for winding in windings:
            try:
                path=CartesianPath(start,target[:3,3],duration,polar,angular_delta=winding)
                segment,kinematics=differential_segment(name,q,path,e.config)
                report=(checked(e,segment,name!='RETREAT',support,placed,kinematic_report=kinematics,tcp_cube=grasp)
                        if check_collisions else kinematics)
                if polar:report['polar_angular_delta_rad']=float(winding)
                break
            except ValueError as error:failures.append(str(error))
        else:raise ValueError(name+': no feasible polar winding/path; '+ '; '.join(failures))
        reports.append(report);segments.append(segment);q=segment.evaluate(segment.duration)
    return segments,reports



class GazeboMotion:
    """Small stage API; feedback and collision guards remain in Executor."""
    def __init__(self,e,timings,report,pickup_height,hover_clearance,retreat_distance,stop_after_lift):
        self.e=e;self.timings=timings;self.report=report
        self.pickup_height=pickup_height;self.hover_clearance=hover_clearance
        self.retreat_distance=retreat_distance;self.stop_after_lift=stop_after_lift
        self.segments={};self.pending=[]

    def preflight(self):
        data=fresh(self.e)
        if max(abs(self.e.q[6:]))>.0002 or max(abs(self.e.velocity))>.008:
            raise RuntimeError('Start must be settled with open jaws')
        self.report['workspace_preflight']=check_workspace_start(self.e,data['cube'])
        self.report['pickup_pose_world']=data['cube'].tolist()

    def prepare(self):
        started=time.monotonic()
        data=fresh(self.e)
        if max(abs(self.e.q[6:]))>.0002 or max(abs(self.e.velocity))>.008:
            raise RuntimeError('Start must be settled with open jaws')
        if self.e.config['execution'].get('workspace_only'):
            self.report['workspace_preflight'] = check_workspace_start(self.e, data['cube'])
        else:
            accept_pickup(self.e.config,data['cube'],self.report.get('use_live_cube',False))
        self.report['pickup_pose_world']=data['cube'].tolist()
        prefer_hover=(self.report.get('use_live_cube',False)
                      and self.e.config.get('simulation_profile')=='sprint'
                      and self.e.config['execution'].get('arm_actuation')=='kinematic_playback')
        segments,reports=approach(self.e,data['cube'],self.timings,self.pickup_height,prefer_hover=prefer_hover)
        self.segments={s.name:s for s in segments}
        self.pending=[s.name for s in segments]+['GRASP','LIFT','TRANSFER','LOWER','RELEASE','RETREAT','VERIFY']
        self.report['approach_checks']=reports
        self.report['approach_preparation_wall_s']=time.monotonic()-started
        self._save('approach_checks',reports)
        self.report['nominal_preparation_wall_s']=time.monotonic()-started
        display=self.e.output/'nominal_display_plan.json'
        display_plan(segments,display)
        emit('plan',path=str(display))

    def _save(self,name,data):
        (self.e.output/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')

    def _expect(self,name):
        if not self.pending or self.pending[0]!=name:
            raise RuntimeError(f'Unsafe stage order: requested {name}; next is {self.pending[:1]}')

    def _move(self,name,support=None,jaws=None):
        self._expect(name);fresh(self.e)
        self.e.support=support if support is not None else ('ground' if not self.e.carrying else None)
        self.e.follow(self.segments[name],jaws);self.pending.pop(0)

    def raise_if_needed(self):
        if 'CLEARANCE_RISE' in self.segments:
            self._move('CLEARANCE_RISE',jaws=[0.,0.])

    def approach_pickup(self):
        self._move('APPROACH_PICKUP',jaws=[0.,0.])

    def descend_to_cube(self):
        self._move('DESCEND_PICK',jaws=[0.,0.])

    def grasp(self):
        self._expect('GRASP');self.e.grasp(duration=self.timings['grasp']);self.pending.pop(0)

    def lift(self):
        self._expect('LIFT');fresh(self.e)
        started=time.monotonic()
        print('CHECK CARRY: planning from the measured grasp',flush=True)
        segments,reports=carry(self.e,self.stop_after_lift,self.timings,self.hover_clearance,self.retreat_distance)
        self.report['carry_plan_source']='fresh measured grasp'
        self.report['carry_preparation_wall_s']=time.monotonic()-started
        self.segments.update({s.name:s for s in segments})
        self.report['carry_checks']=reports;self._save('carry_checks',reports)
        self._move('LIFT',support='ground')

    def transfer_polar(self):
        self._move('TRANSFER')

    def lower_onto_pedestal(self):
        self._move('LOWER',support='box_a')

    def release(self):
        self._expect('RELEASE');fresh(self.e);self.e.supported()
        assisted=getattr(self.e,'attachment',None) is not None
        # DART restores separate-model collision on detachment: clear the
        # fingers first, with the supported cube still constrained and guarded.
        if not assisted:self.e.carrying=False
        self.e.support='box_a'
        q=self.e.hold_position()
        self.e.follow(joint_segment('RELEASE',q,q,self.timings['release']),[0.,0.])
        if assisted:self.e.detach_supported()
        self.e.contact_allowed=False;self.pending.pop(0)

    def retreat(self):
        self._move('RETREAT',support='box_a',jaws=[0.,0.])
        if 'cycle_begin_sim_time' in self.report:
            self.report['motion_cycle_sim_s']=self.e.sim-self.report['cycle_begin_sim_time']
        if 'cycle_begin_wall_time' in self.report:
            self.report['motion_cycle_wall_s']=time.monotonic()-self.report['cycle_begin_wall_time']

    def verify_placement(self):
        self._expect('VERIFY');self.report['placement']=self.e.final_stability()
        self.pending.pop(0)
        self.report['status']='PASS_ASSISTED_SEQUENCE' if self.report.get('detachable_grasp_assist') else 'PASS_PHYSICAL_SEQUENCE'

    def delay(self,seconds):
        """Simulation-time hold, with the same feedback/contact/abort guards.

        Like a motion, this may include additional measured settling time.
        Unlike time.sleep(), it retains arm hold and jaw preload while monitoring.
        """
        seconds=float(seconds)
        if not np.isfinite(seconds) or seconds<0:raise ValueError('Delay must be finite and nonnegative')
        if seconds==0:return
        seconds*=self.e.config.get('execution',{}).get('delay_scale',1.)
        if seconds==0:return
        fresh(self.e);q=self.e.hold_position()
        self.e.follow(joint_segment('DELAY',q,q,seconds))


def controller_gate():
    gate=json.loads((PKG.parents[1]/'verification/pick_place/controller_gate.json').read_text())
    return gate['status']=='PASS' and gate.get('arm_actuation') in ['gazebo_position','kinematic_playback']


def run(sequence,*,timings,pickup_height,hover_clearance,retreat_distance,
        options=None,use_profile_timings=True,world_audit=None):
    """Default is no-motion preflight; explicit --execute runs the user sequence."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--detachable-grasp',action='store_true',help='Explicit simulation assist: contact-triggered constraint, supported detach; not friction validation')
    parser.add_argument('--stop-after-lift',action='store_true')
    parser.add_argument('--use-live-cube',action='store_true',help='Plan from the measured upright cube on the ground')
    parser.add_argument('--profile',choices=sorted(PROFILES),help='Bounded simulation-only speed/workspace trial profile')
    parser.add_argument('--output',type=Path,default=PKG.parents[1]/'verification/pick_place/cartesian_run')
    args=parser.parse_args() if options is None else options
    if args.profile and use_profile_timings:timings={**timings,**PROFILES[args.profile]['timings']}
    keys={'rise','approach','descent','grasp','lift','transfer','lower','release','retreat'}
    if set(timings)!=keys:raise ValueError('Timing keys must be '+str(sorted(keys)))
    for key,value in timings.items():
        if key=='approach' and value is None:continue
        if not np.isfinite(value) or value<=0:raise ValueError('Duration must be finite and positive: '+key)
    for value in [pickup_height,hover_clearance,retreat_distance]:
        if not np.isfinite(value) or value<=0:raise ValueError('Heights must be finite and positive')
    if os.environ.get('ROS_DOMAIN_ID')!='77' or os.environ.get('GZ_PARTITION')!='gp7_project_lab':
        raise RuntimeError('Use ROS domain 77 and partition gp7_project_lab')
    if not controller_gate():
        raise RuntimeError('Controller gate has not passed')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    config=load_config(profile=args.profile);e=None;report={'status':'FAIL','mode':'physical friction grasp; no attachment/reset/teleport','timings':timings,'use_live_cube':args.use_live_cube,'simulation_profile':args.profile,'rate_limits':{k:config[k] for k in ['velocity_limit_rad_s','acceleration_limit_rad_s2']}}
    report['detachable_grasp_assist']=args.detachable_grasp
    if args.detachable_grasp:
        from pick_place_config import activate_kinematic_playback
        activate_kinematic_playback(config)
        config['execution'].update(PROFILES.get(args.profile,{}).get('assisted_execution',{}))
        report['mode']='PURE KINEMATIC PLAYBACK: simulation-tick q(t), geometric cube attachment/support; NO physics, PID, effort servo or friction-grasp validation'
    quick=bool(args.detachable_grasp and args.use_live_cube and args.profile=='sprint')
    if quick:config['execution']['workspace_only']=True
    report['arm_actuation']=config['execution']['arm_actuation']
    report['jaw_actuation']=config['execution']['jaw_actuation']
    report['actuation_scope']=('Native kinematic model state and analytical derivatives; physics disabled; no motor/grasp-force validation'
                               if config['execution'].get('kinematic_playback') else
                               'Idealized Gazebo position execution; no torque or contact-force validation.')
    report['execution_settings']=config['execution'].copy()
    # Keep the context alive while KeyboardInterrupt performs guarded pause/hold.
    # rclpy's default SIGINT handler shuts it down before cleanup can publish.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stop_requested=False
    def request_stop(signum,frame):
        # Never raise across Gazebo / pybind transport calls. Stop cooperatively
        # at the next Python execution gate with the ROS context still valid.
        nonlocal stop_requested
        stop_requested=True
    previous_sigint=signal.signal(signal.SIGINT,request_stop)
    try:
        audit=live_audit if world_audit is None else world_audit
        report['world']=audit(config,allow_cube_reposition=True) if args.use_live_cube else audit(config)
        e=Executor(config,output,cube_only=True,detachable_grasp=args.detachable_grasp)
        e.abort_requested=lambda:stop_requested
        e.checker.cube_only=True;e.support='ground'
        report['collision_scope']=('disabled for workspace-only kinematic Sprint' if quick else
                                   'cube versus robot/environment; whole-arm/self sweeps omitted for fixed-scene exercise')
        e.ready(require_home=False)
        report['frames']=e.wait_native()
        report['contact_topics']=[] if config['execution'].get('kinematic_playback') else e.native.require_contact_publishers()
        motion=GazeboMotion(e,timings,report,pickup_height,hover_clearance,retreat_distance,args.stop_after_lift)
        if quick and not args.execute:
            motion.preflight()
        else:
            motion.prepare()
        if stop_requested:raise KeyboardInterrupt('Operator requested Stop')
        if not args.execute:
            report['status']='READY_WORKSPACE' if quick else 'READY_APPROACH_AND_DESCENT'
            if quick:
                message=('Cube is in the workspace. Arm will rise first.' if report['workspace_preflight']['needs_rise']
                         else 'Cube is in the workspace. Arm is above clearance height.')
            else:message='Approach/descent validated. Carry is planned after grasp.'
            status('PREPARE','ready',message)
            return
        if not config['execution'].get('kinematic_playback'):e.native.require_contact_publishers()
        e.refresh();report['cycle_begin_sim_time']=e.sim;report['cycle_begin_wall_time']=time.monotonic()
        sequence(motion)
        report['complete_with_stability_sim_s']=e.sim-report['cycle_begin_sim_time']
        report['complete_with_stability_wall_s']=time.monotonic()-report['cycle_begin_wall_time']
        if args.stop_after_lift and motion.pending[0:1]==['TRANSFER']:
            report['status']='PASS_GRASP_AND_LIFT'
        elif motion.pending:raise RuntimeError('Sequence ended before completion')
    except (Exception,KeyboardInterrupt) as error:
        report['status']='ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAIL'
        report['operator_stop_requested']=stop_requested
        report['error']=str(error) or 'Interrupted';print('ABORT:',report['error'],flush=True)
        if e and e.carrying:
            try:
                reply=service('/world/gp7_lab/control','gz.msgs.WorldControl','gz.msgs.Boolean','pause: true',config['gz_partition'])
                ack=Boolean();text_format.Parse(reply,ack);report['physics_paused_after_abort']=ack.data
            except Exception as pause_error:report['pause_error']=str(pause_error)
        raise
    finally:
        try:
            if e:
                report['phases']=e.metrics;report['last_measured_q']=None if e.q is None else e.q.tolist()
                report['max_carry_orientation_error_rad']=getattr(e,'max_cube_orientation_error',0.)
                for name,rows in [('tracking',e.rows),('cube',e.observations)]:
                    with (output/(name+'.csv')).open('w',newline='') as stream:csv.writer(stream).writerows(rows)
                e.finish()
        except Exception as cleanup_error:
            report['cleanup_error']=str(cleanup_error);report['status']='FAIL'
            raise
        finally:
            # A failed hold/teardown must not erase the failure evidence.
            try:
                (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
                rclpy.shutdown()
            finally:
                signal.signal(signal.SIGINT,previous_sigint)
