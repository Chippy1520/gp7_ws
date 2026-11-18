#!/usr/bin/env python3
"""Simulation-time pick/place with explicitly selected physical or kinematic scope."""
import argparse
from itertools import product
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_trajectories import quintic_given_duration,TrajectorySample
from kinematics import ExportedModel,ik_exported
from pick_place_config import PKG,load_config,arm_limits
from pick_place_planner import joint_segment,vertical_segment,compensated_segment,validate_segment
from pick_place_collision import CollisionChecker
from pick_place_feedback import GazeboFeedback
from pick_place_attachment import DetachableGrasp
from verify_pick_place_controllers import ControllerCheck,live_audit
from pick_place_cache import prepare, load_cache, measured_start, display_plan, status, emit


class Reference:
    def __init__(self,segment,jaw_start,jaw_goal):
        self.segment=segment;self.duration=segment.duration
        self.jaws=quintic_given_duration(jaw_start,jaw_goal,self.duration)
    def evaluate(self,t):
        j=self.jaws.evaluate(t)
        return TrajectorySample(np.r_[self.segment.evaluate(t),j.position],
            np.r_[self.segment.evaluate(t,1),j.velocity],np.r_[self.segment.evaluate(t,2),j.acceleration],None)


def angle(first,second):return float(np.linalg.norm(Rotation.from_matrix(first@second.T).as_rotvec()))


class Executor(ControllerCheck):
    def __init__(self,config,output,cube_only=False,detachable_grasp=False):
        super().__init__(config)
        self.native=GazeboFeedback();self.model=ExportedModel()
        self.attachment=DetachableGrasp(self.native.node) if detachable_grasp else None
        self.checker=CollisionChecker(clearance=config['planning']['collision_clearance_m'],
                                      self_clearance=config['planning']['self_collision_clearance_m'],cube_only=cube_only)
        self.output=output;self.commanding=False;self.carrying=False;self.grasp_transform=None;self.cube_rotation=None
        self.jaw_target=np.zeros(2);self.arm_target=None;self.metrics=[];self.observations=[]
        self.max_cube_orientation_error=0.;self.pause_since=None;self.last_geometry=-1.
        self.support=None;self.contact_allowed=False;self.last_cube=None
        self.last_monitor_wall=0.

    def spin(self):
        if getattr(self,'abort_requested',lambda:False)():
            raise KeyboardInterrupt('Operator requested Stop')
        self.heartbeat()
        if getattr(self,'playback',None) is not None and self.playback.status.get('state')=='rejected':
            raise RuntimeError('Kinematic playback rejected: '+str(self.playback.status))
        # A ready ROS queue makes spin_once return immediately. Yield to the
        # Gazebo transport callback threads instead of busy-polling/GIL starvation.
        time.sleep(.002)
        rclpy.spin_once(self.node,timeout_sec=.005)
        data=self.native.snapshot();now=time.monotonic()
        paused=data['paused'] and now-data['stats_wall']<1.
        if self.error:raise RuntimeError(self.error)
        if paused:
            if self.pause_since is None:self.pause_since=now
            if now-self.pause_since>self.config['execution']['max_clock_pause_wall_s']:raise RuntimeError('Gazebo pause timeout')
            return
        self.pause_since=None
        if self.q is not None and now-self.feedback_wall>.5:raise RuntimeError('Stale measured joint state')
        if self.sim is not None and now-self.clock_wall>3.:raise RuntimeError('Lost advancing simulation clock')
        if self.sim is None or self.q is None:return
        if data['cube'] is None or data['tcp'] is None:
            if self.commanding:raise RuntimeError('No time-associated measured cube/TCP poses')
            return
        if self.commanding and abs(self.sim-data['cube_stamp'])>.1:
            raise RuntimeError('Associated pose time differs from ROS simulation time by over 0.1 s')
        limits=arm_limits(self.config)
        if np.any(self.q[:6]<limits[:,0]-1e-4) or np.any(self.q[:6]>limits[:,1]+1e-4):
            raise RuntimeError('Measured joint position outside task intervals')
        if not self.config['execution'].get('synchronous_position_inputs') and (max(abs(self.velocity[:6]))>self.config['velocity_limit_rad_s']+.005 or max(abs(self.velocity[6:]))>.0205):
            raise RuntimeError(f'Measured velocity exceeded simulation gate: arm={self.velocity[:6].tolist()}, jaws={self.velocity[6:].tolist()}')
        if now-data['cube_wall']>.5 or now-data['tcp_wall']>.5:raise RuntimeError('Stale named cube/palm feedback')
        if abs(data['cube_stamp']-data['tcp_stamp'])>.03:raise RuntimeError('Cube/palm timestamps are not associated')
        cube=data['cube'];tcp=data['tcp']
        if self.carrying:
            relative=np.linalg.inv(tcp)@cube
            if np.linalg.norm(relative[:3,3]-self.grasp_transform[:3,3])>.004:
                raise RuntimeError('Measured cube slip exceeds 4 mm')
            orientation=angle(cube[:3,:3],self.cube_rotation)
            self.max_cube_orientation_error=max(self.max_cube_orientation_error,orientation)
            if orientation>np.deg2rad(1.):raise RuntimeError('Measured cube orientation exceeded 1 degree')
            if getattr(self,'attachment',None) is not None:
                if not self.attachment.attached():raise RuntimeError('Lost verified detachable grasp constraint')
            elif any(now-data['contacts'][s]>.2 for s in ['positive','negative']):raise RuntimeError('Lost physical finger/cube contact')
        if not self.config.get('execution', {}).get('workspace_only') and self.sim-self.last_geometry>=.01:
            self.last_geometry=self.sim
            result=self.checker.check(self.q[:6],self.q[6:],cube_transform=cube,
                holding=self.contact_allowed,support=self.support,support_penetration_tolerance=.001)
            if not result['safe']:raise RuntimeError('Measured geometry violation: '+str(result['violations']))
            self.observations.append([self.phase,self.sim,*cube[:3,3],*Rotation.from_matrix(cube[:3,:3]).as_quat(),
                                       self.max_cube_orientation_error])

    def wait_native(self):
        end=time.monotonic()+12
        while time.monotonic()<end:
            self.spin();data=self.native.snapshot()
            if data['cube'] is not None and data['tcp'] is not None:break
        else:raise RuntimeError('Missing actual named Gazebo pose streams')
        expected=self.model.fk(self.q[:6]);actual=data['tcp']
        pe=float(np.linalg.norm(expected[:3,3]-actual[:3,3]));re=angle(expected[:3,:3],actual[:3,:3])
        if pe>.002 or re>.005:raise RuntimeError('Measured TCP and exported FK disagree')
        return {'tcp_fk_position_error_m':pe,'tcp_fk_orientation_error_rad':re}

    def refresh(self):
        # Offline planning must not make queued ROS messages look like fresh
        # feedback or advance a phase whose reference has not started yet.
        end=time.monotonic()+5
        while time.monotonic()<end:
            rclpy.spin_once(self.node,timeout_sec=.005)
            if self.error:raise RuntimeError(self.error)
            if time.monotonic()-self.feedback_wall<.1 and time.monotonic()-self.clock_wall<.1:return
        raise RuntimeError('No fresh ROS state/clock after offline planning')

    def follow(self,segment,jaw_goal=None,closing=False):
        self.refresh()
        # Planning/display serialization must never splice a curve onto a moved arm.
        if np.max(abs(self.q[:6]-segment.evaluate(0)))>.003:
            raise RuntimeError('Measured phase start differs from validated reference by over 0.003 rad')
        self.commanding=True
        self.phase=segment.name;self.begin=self.sim
        if getattr(self,'playback',None) is not None:self.begin+=.05
        status(self.phase,'running','Following validated reference with measured feedback gates')
        jaw_goal=self.jaw_target.copy() if jaw_goal is None else np.asarray(jaw_goal,dtype=float)
        # Preserve squeeze preload across carried moves/holds; measured travel is
        # blocked by the cube and must not restart the closure ramp each stage.
        preserve=self.carrying and not closing and np.array_equal(jaw_goal,self.jaw_target)
        jaw_start=jaw_goal if preserve else self.q[6:].copy()
        jaw_limit=.016 if getattr(self,'attachment',None) is not None else .02
        minimum_jaw_time=1.875*np.max(abs(jaw_goal-jaw_start))/jaw_limit
        if segment.name=='DELAY' and getattr(self,'attachment',None) is not None and minimum_jaw_time>segment.duration:
            # A short guarded hold may correct residual jaw travel after detach.
            # Stretch only the constant-arm hold, never a validated moving path.
            if not np.allclose(segment.evaluate(0),segment.evaluate(segment.duration),atol=1e-12,rtol=0.):
                raise RuntimeError('DELAY must be a constant-arm hold')
            segment=joint_segment('DELAY',segment.evaluate(0),segment.evaluate(0),minimum_jaw_time*1.01)
        if 1.875*np.max(abs(jaw_goal-jaw_start))/segment.duration>jaw_limit+1e-9:
            raise RuntimeError('Quintic jaw reference exceeds scene jaw velocity limit; increase phase duration')
        self.curve=Reference(segment,jaw_start,jaw_goal);self.jaw_target=jaw_goal.copy()
        if getattr(self,'playback',None) is not None:self.playback.execute(self.curve,self.begin,self.spin)
        expected=self.curve.evaluate(segment.duration).position
        stable=None;last_publish=-1.;deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            self.spin()
            if self.pause_since is not None:continue
            elapsed=max(0.,self.sim-self.begin)
            if getattr(self,'playback',None) is None and self.sim>=last_publish+self.config.get('execution',{}).get('command_period_s',.02):
                sample=self.curve.evaluate(min(elapsed,segment.duration))
                if self.config.get('execution',{}).get('velocity_feedforward'):self.publish(sample.position,sample.velocity)
                else:self.publish(sample.position)
                last_publish=self.sim
            data=self.native.snapshot()
            self.monitor_sample(data,elapsed,segment.duration)
            arms=np.all(abs(self.q[:6]-expected[:6])<.003) and np.all(abs(self.velocity[:6])<.008)
            if self.config.get('execution',{}).get('jaw_actuation') in ['gazebo_position','kinematic_playback']:
                jaws=np.all(abs(self.q[6:]-expected[6:])<.0002) and np.all(abs(self.velocity[6:])<.0008)
                if self.carrying:jaws=jaws and self.attachment.attached()
            elif self.carrying and getattr(self,'attachment',None) is not None:
                jaws=self.attachment.attached() and np.all(abs(self.q[6:]-expected[6:])<.0002) and np.all(abs(self.velocity[6:])<.0008)
            elif closing or self.carrying:
                # A held rigid cube prevents reaching the free-space preload
                # target. Completion requires real contacts and settled jaws.
                jaws=all(time.monotonic()-data['contacts'][s]<.1 for s in ['positive','negative'])
                jaws=jaws and np.all(self.q[6:]>.008) and np.all(self.q[6:]<.0118) and np.all(abs(self.velocity[6:])<.0008)
            else:jaws=np.all(abs(self.q[6:]-expected[6:])<.0002) and np.all(abs(self.velocity[6:])<.0008)
            good=elapsed>=segment.duration and arms and jaws
            stable=self.sim if good and stable is None else (stable if good else None)
            settings=self.config.get('execution',{})
            # The bounded preload controller has I=0: no ten-second PI buildup.
            # Contact, jaw velocity and arm settling gates still apply.
            hold=settings.get('settle_duration_s',.3)
            if closing:
                preload=self.config.get('grasp',{}).get('preload_force_n')
                hold=settings.get('grasp_contact_hold_s',.3) if preload is not None else 10.
                if settings.get('jaw_actuation') in ['gazebo_position','kinematic_playback']:hold=settings.get('settle_duration_s',.3)
            if segment.name=='DELAY':hold=0.  # The explicit guarded hold is already the delay.
            if stable is not None and self.sim-stable>=hold:
                result={'phase':self.phase,'duration_s':segment.duration,'time_s':self.sim,'measured_q':self.q.tolist(),
                        'arm_error_rad':float(max(abs(self.q[:6]-expected[:6]))),
                        'cube_pose':data['cube'].tolist()}
                self.arm_target=expected[:6].copy()
                self.metrics.append(result);print('MEASURED phase complete:',self.phase,flush=True)
                self.curve=None;return
            if elapsed>segment.duration+25:raise RuntimeError(self.phase+' failed to settle/contact')
        raise RuntimeError('Wall timeout: '+self.phase)

    def monitor_sample(self,data,elapsed,duration):
        # Wall-time throttle bounds JSON overhead independently of simulation speed.
        now=time.monotonic()
        if now-getattr(self,'last_monitor_wall',0.)<.1:return
        self.last_monitor_wall=now
        ref=self.curve.evaluate(min(elapsed,duration))
        tcp=data.get('tcp')
        if tcp is None:return
        emit('sample',stage=self.phase,sim_time=float(self.sim),elapsed=float(elapsed),duration=float(duration),
             reference={'q':ref.position.tolist(),'qd':ref.velocity.tolist(),'qdd':ref.acceleration.tolist()},
             measured={'q':self.q.tolist(),'qd':self.velocity.tolist()},tcp=tcp[:3,3].tolist(),
             forces={'positive':None,'negative':None,'source':
                     'not applicable: idealized jaws' if self.config.get('execution',{}).get('jaw_actuation') in ['gazebo_position','kinematic_playback']
                     else 'unavailable: contact wrench frame/sign not validated'})

    def hold_position(self):
        """Retain the reference, not its measured tracking error, between stages."""
        target=getattr(self,'arm_target',None)
        return self.q[:6].copy() if target is None else target.copy()

    def grasp(self,duration=3.):
        self.contact_allowed=True;self.support='ground'
        q=self.hold_position()
        hold=joint_segment('GRASP',q,q,duration)
        ideal=getattr(self,'config',{}).get('execution',{}).get('jaw_actuation') in ['gazebo_position','kinematic_playback']
        self.follow(hold,[.010,.010] if ideal else [.012,.012],closing=True)
        if ideal:self.check_grasp_geometry()
        if getattr(self,'attachment',None) is not None:
            self.attachment.set(True,self.spin)
            # The verified constraint now carries the load. Hold contact travel,
            # rather than forcing 2 mm of visual overclosure through the box.
            self.jaw_target=self.q[6:].copy()
        data=self.native.snapshot();derived=self.model.fk(self.q[:6])
        if np.linalg.norm(derived[:3,3]-data['cube'][:3,3])>.003:raise RuntimeError('Grasp-centre calibration exceeded 3 mm')
        self.grasp_transform=np.linalg.inv(derived)@data['cube']
        self.cube_rotation=data['cube'][:3,:3].copy()
        if angle(self.cube_rotation,np.eye(3))>np.deg2rad(1.):raise RuntimeError('Cube was tilted during grasp')
        self.carrying=True
        self.metrics.append({'phase':'CALIBRATED_GEOMETRIC_GRASP' if ideal else 'CALIBRATED_CONTACT_GRASP','assisted':getattr(self,'attachment',None) is not None,'tcp_cube':self.grasp_transform.tolist(),
                             'basis':'measured joint FK and named actual cube, cross-checked with actual palm',
                             'grasp_check':'geometry' if ideal else 'physical contact',
                             'contacts':data['contact_pairs'],'measured_jaws_m':self.q[6:].tolist()})

    def check_grasp_geometry(self):
        """Measured face alignment for the approved idealized 50 mm cube grasp."""
        cube=self.native.snapshot()['cube'];tcp=self.model.fk(self.q[:6])
        relative=np.linalg.inv(tcp)@cube
        radius=.025*np.abs(relative[1,:3]).sum()
        centre=relative[1,3]
        gaps=np.array([.035-self.q[6]-centre-radius,.035-self.q[7]+centre-radius])
        bottom=cube[2,3]-.025*np.abs(cube[2,:3]).sum()
        if (not np.isfinite(relative).all() or np.linalg.norm(relative[:3,3])>.003 or np.max(abs(gaps))>.0005
                or abs(bottom)>.001 or angle(cube[:3,:3],np.eye(3))>np.deg2rad(1.)):
            raise RuntimeError('Idealized grasp geometry invalid: cube must be aligned, grounded and between settled jaws')
        return {'jaw_face_gaps_m':gaps.tolist(),'cube_bottom_m':float(bottom)}

    def detach_supported(self):
        if getattr(self,'attachment',None) is None:return
        self.supported();self.carrying=False
        def guard():
            self.spin();self.supported()
            if angle(self.native.snapshot()['cube'][:3,:3],self.cube_rotation)>np.deg2rad(1.):
                raise RuntimeError('Cube tilted during supported detachment')
        try:self.attachment.set(False,guard)
        except Exception:
            self.carrying=True
            raise
        self.metrics.append({'phase':'SUPPORTED_DETACH','time_s':self.sim})

    def carry_plan(self,stop_after_lift=False):
        start=self.q[:6].copy();cube=self.native.snapshot()['cube'].copy()
        lifted=cube.copy();lifted[2,3]=self.config['placement']['hover_center_xyz_m'][2]
        lift_tcp=lifted@np.linalg.inv(self.grasp_transform)
        lift=vertical_segment('LIFT',start,lift_tcp[2,3],28.,self.config)
        if stop_after_lift:
            # Calibrated grasp differs from nominal; validate only the requested lift.
            report=validate_segment(lift,self.config)
            report['collision']=self.checker.validate(lift,jaws=(.012,.012),holding=True,
                tcp_cube=self.grasp_transform,support='ground',
                period=self.config['planning']['collision_sample_period_s'])
            (self.output/'calibrated_plan.json').write_text(json.dumps([report],indent=2)+'\n')
            display_plan([lift],self.output/'calibrated_display_plan.json')
            return [lift]
        desired=cube.copy();desired[:3,3]=self.config['placement']['hover_center_xyz_m']
        target=desired@np.linalg.inv(self.grasp_transform)
        solutions=ik_exported(target,seed=lift.evaluate(lift.duration),model=self.model,limits=arm_limits(self.config))
        if not solutions:raise RuntimeError('Calibrated hover has no feasible IK')
        transfer=compensated_segment('TRANSFER',lift.evaluate(lift.duration),solutions[0],24.,self.config)
        desired[:3,3]=self.config['placement']['center_xyz_m'];target=desired@np.linalg.inv(self.grasp_transform)
        lower=vertical_segment('LOWER',transfer.evaluate(transfer.duration),target[2,3],16.,self.config)
        retreat=vertical_segment('RETREAT',lower.evaluate(lower.duration),target[2,3]+.2,20.,self.config)
        home=joint_segment('RETURN_HOME',retreat.evaluate(retreat.duration),np.zeros(6),24.)
        phases=[lift,transfer,lower,retreat,home];reports=[]
        for segment in phases:
            report=validate_segment(segment,self.config)
            held=segment.name in ['LIFT','TRANSFER','LOWER']
            support='ground' if segment.name=='LIFT' else 'box_a' if segment.name=='LOWER' else None
            report['collision']=self.checker.validate(segment,jaws=(.012,.012) if held else (0.,0.),holding=held,
                tcp_cube=self.grasp_transform if held else None,support=support,
                cube_world=desired if not held else None,period=self.config['planning']['collision_sample_period_s'])
            reports.append(report)
        (self.output/'calibrated_plan.json').write_text(json.dumps(reports,indent=2)+'\n')
        display_plan(phases,self.output/'calibrated_display_plan.json')
        return phases

    def supported(self):
        data=self.native.snapshot();cube=data['cube']
        if getattr(self,'config',{}).get('execution',{}).get('kinematic_playback'):
            bottom=cube[2,3]-.025*np.abs(cube[2,:3]).sum()
            if abs(bottom-.2)>.0005:raise RuntimeError('Cube bottom is not within 0.5 mm of pedestal support plane; refusing release')
        elif time.monotonic()-data['contacts']['support']>.1:raise RuntimeError('No measured cube/pedestal support contact; refusing release')
        if np.linalg.norm(cube[:2,3]-np.array([.5,.5]))>.005:raise RuntimeError('Placement XY error exceeds 5 mm')
        if abs(cube[2,3]-.225)>.003:raise RuntimeError('Cube is not at supported placement height')

    def final_stability(self):
        self.phase='PLACEMENT_STABILITY';start=self.sim;positions=[];orientations=[]
        while self.sim-start<2.:
            self.spin();data=self.native.snapshot();self.supported()
            positions.append(data['cube'][:3,3].tolist());orientations.append(angle(data['cube'][:3,:3],np.eye(3)))
            if orientations[-1]>np.deg2rad(1.):raise RuntimeError('Released cube orientation outside 1 degree')
        return {'stable_simulation_seconds':self.sim-start,'final_cube_center_m':positions[-1],
                'scope':'geometric pose stability, not physical support/friction validation' if self.config.get('execution',{}).get('kinematic_playback') else 'physical contact and pose stability',
                'max_upright_error_rad':max(orientations),'max_carry_orientation_error_rad':self.max_cube_orientation_error}

    def finish(self):
        if self.commanding:
            for pub in getattr(self,'velocity_pubs',{}).values():pub.publish(Float64(data=0.))
        if self.commanding and self.q is not None and np.isfinite(self.q).all():
            # Damped measured-position hold; retain closure when carrying.
            # On a world reset, never publish the stale pre-reset pose.
            reset=self.error and 'Backward' in self.error
            if not reset:
                hold=np.r_[self.q[:6],self.jaw_target if self.carrying else self.q[6:]]
                self.send_positions(hold)
        self.node.destroy_node()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true')
    mode.add_argument('--prepare',action='store_true',help='Offline validated reusable nominal spline preparation; no ROS node or commands')
    parser.add_argument('--resume-grasp',action='store_true',help='Explicitly resume a stopped, ground-supported grasp; never home through the cube')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stop-after-lift',action='store_true',help='Hold the measured lifted pose with jaws closed; do not transfer or release')
    args=parser.parse_args();config=load_config()
    output=args.output or PKG.parents[1]/'verification/pick_place/execution';output.mkdir(parents=True,exist_ok=True)
    cache=output.parent/('prepared_lift' if args.stop_after_lift else 'prepared_full')
    if args.prepare:
        if args.resume_grasp:parser.error('--prepare cannot calibrate a resumed grasp')
        try:
            phases,_,_,timing=prepare(cache,config,args.stop_after_lift)
            display_plan(phases,output/'nominal_display_plan.json')
            (output/'prepare_result.json').write_text(json.dumps({'status':'READY_NOMINAL_OFFLINE_ONLY',
                'cache':str(cache.resolve()),**timing},indent=2)+'\n')
        except Exception as error:
            status('PREPARE','aborted',str(error));raise
        return
    if int(os.environ.get('ROS_DOMAIN_ID','0'))!=77 or os.environ.get('GZ_PARTITION')!='gp7_project_lab':raise RuntimeError('Wrong simulation domain/partition')
    gate=json.loads((output.parent/'controller_gate.json').read_text())
    if gate['status']!='PASS':raise RuntimeError('Controller gate has not passed')
    rclpy.init();executor=None;report={'status':'FAIL','mode':'physical friction grasp; no attachment or teleport'}
    try:
        report['live_audit']=live_audit(config);executor=Executor(config,output)
        if args.resume_grasp:executor.contact_allowed=True;executor.support='ground'
        executor.ready(require_home=not args.resume_grasp);report['frame_crosscheck']=executor.wait_native()
        phases=[]
        if args.resume_grasp:
            data=executor.native.snapshot();actual=executor.model.fk(executor.q[:6])
            if np.linalg.norm(data['cube'][:3,3]-[.45,-.2,.025])>.003 or np.linalg.norm(actual[:3,3]-data['cube'][:3,3])>.003:
                raise RuntimeError('Resume requires a stopped ground-supported cube aligned with the TCP')
            if max(abs(executor.velocity))>.008:raise RuntimeError('Resume requires settled measured joints')
            report['resume']='explicit ground-supported grasp; no homing or cube reset'
        else:
            status('PRECHECK','preparing','Loading fingerprint-validated nominal gate')
            phases,tcp_cube,cache_gate=load_cache(cache,config,args.stop_after_lift)
            executor.refresh()
            phases=measured_start(phases,executor.q.copy(),executor.velocity.copy(),config,
                                  executor.checker,executor.native.snapshot()['cube'])
            report['nominal_cache']=str(cache.resolve())
            display_plan(phases,output/'nominal_display_plan.json')
        status('PRECHECK','ready','Measured startup gates passed; calibrated carry remains required')
        if not args.execute:
            report['status']='READ_ONLY_READY';return
        # Scoped metadata replaces the legacy nominal gate with the same collision,
        # contact and freshness requirements. Resume also requires fresh evidence.
        load_cache(cache,config,args.stop_after_lift)
        if not args.resume_grasp:
            executor.checker.validate(phases[0],period=config['planning']['collision_sample_period_s'])
            for segment in phases[:4]:executor.follow(segment,[0.,0.])
        executor.grasp()
        status('CALIBRATED_PLAN','preparing','Replanning requested stages from measured physical grasp')
        carried=executor.carry_plan(stop_after_lift=args.stop_after_lift)
        for segment in carried[:3]:
            executor.support='ground' if segment.name=='LIFT' else 'box_a' if segment.name=='LOWER' else None
            executor.follow(segment)
            if segment.name=='LIFT' and args.stop_after_lift:
                data=executor.native.snapshot()
                report['lift']={'actual_cube_pose':data['cube'].tolist(),
                                'measured_jaws_m':executor.q[6:].tolist(),
                                'max_carry_orientation_error_rad':executor.max_cube_orientation_error,
                                'end_condition':'measured-position hold with closure retained; no transfer or release'}
                report['status']='PASS_GRASP_AND_LIFT_ONLY'
                status('LIFT','holding','Measured lifted hold; jaws retain closure; no transfer/release')
                return
            if segment.name=='TRANSFER':
                data=executor.native.snapshot();R=data['cube'][:3,:3]
                corners=np.array(list(product([-.025,.025],repeat=3)))@R.T+data['cube'][:3,3]
                clearance=float(corners[:,2].min()-.2)
                if abs(clearance-.1)>.005:raise RuntimeError('Measured hover clearance outside 0.10 m ± 5 mm')
                report['hover_bottom_clearance_m']=clearance
        executor.supported()
        executor.carrying=False;executor.support='box_a'
        q=carried[2].evaluate(carried[2].duration)
        executor.follow(joint_segment('RELEASE',q,q,3.),[0.,0.])
        executor.contact_allowed=False;executor.support=None
        for segment in carried[3:]:executor.follow(segment,[0.,0.])
        report['placement']=executor.final_stability();report['status']='PASS_ONE_PHYSICAL_CYCLE'
        status('RETURN_HOME','complete','Physical cycle and placement stability checks passed')
    except Exception as error:
        report['error']=str(error);status(executor.phase if executor else 'PRECHECK','aborted',str(error));print('ABORT:',error,flush=True);raise
    finally:
        if executor:
            report['phases']=executor.metrics
            with (output/'tracking.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['phase','sim_time',*[f'ref_{n}' for n in executor.names],*[f'q_{n}' for n in executor.names],*[f'v_{n}' for n in executor.names]]);writer.writerows(executor.rows)
            with (output/'cube.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['phase','sim_time','x','y','z','qx','qy','qz','qw','max_carry_orientation_error_rad']);writer.writerows(executor.observations)
            executor.finish()
        report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),PKG/'scripts/pick_place_feedback.py',PKG/'scripts/pick_place_planner.py',
             PKG/'scripts/pick_place_collision.py',PKG/'config/pick_place.yaml']}
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n');rclpy.shutdown()


if __name__=='__main__':main()
