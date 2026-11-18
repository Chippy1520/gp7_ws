#!/usr/bin/env python3
"""Gate 1: audit live task SDF and measured small-motion ROS-time tracking only."""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as E
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64
from google.protobuf import text_format
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.scene_pb2 import Scene
from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
from gz.transport13 import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_trajectories import quintic_given_duration
from pick_place_config import PKG, NAMES, load_config, arm_limits, activate_kinematic_playback
from prepare_pick_place_scene import service

OUT = PKG.parents[1]/'verification/pick_place'
JAW_NAMES = ['gripper_joint_positive', 'gripper_joint_negative']


def audit_position_controller(plugin,name,speed,index=None):
    """Reject stale/custom/effort-controlled worlds before any motion command."""
    expected={'joint_name':name,'topic':f'/gp7/{name}/cmd_pos','use_velocity_commands':'true'}
    if index is not None:
        expected.update(topic='/gp7/joint_positions',use_actuator_msg='true',actuator_number=str(index))
    speed_valid=(plugin.find('cmd_max') is None if index is not None else
                 np.isclose(float(plugin.findtext('cmd_max','nan')),speed,rtol=0.,atol=1e-12))
    if (plugin.get('name')!='gz::sim::systems::JointPositionController'
            or plugin.get('filename')!='gz-sim-joint-position-controller-system'
            or any(plugin.findtext(key)!=value for key,value in expected.items())
            or plugin.find('velocity_topic') is not None
            or not speed_valid):
        raise RuntimeError('Live world lacks standard PID-bypass position control: '+name+'; reset the owned scene')


def audit_arm_controller(plugin,name,config):
    index=NAMES.index(name) if config['execution'].get('synchronous_position_inputs') else None
    audit_position_controller(plugin,name,config['velocity_limit_rad_s'],index)


def live_audit(config,allow_cube_reposition=False,live_sdf_response=None,live_scene_response=None):
    partition = config['gz_partition']
    if live_sdf_response is None:
        response = service('/world/gp7_lab/generate_world_sdf', 'gz.msgs.SdfGeneratorConfig',
                           'gz.msgs.StringMsg', '', partition)
        message = StringMsg(); text_format.Parse(response, message)
    else:
        message = live_sdf_response
    (OUT/'live_world.sdf').write_text(message.data)
    world = E.fromstring(message.data).find('world')
    robot = world.find("model[@name='gp7']")
    kinematic=config['execution'].get('kinematic_playback',False)
    if kinematic:
        if any(p.get('name') in ['gz::sim::systems::Physics','gz::sim::systems::Contact'] for p in world.findall('plugin')):
            raise RuntimeError('Kinematic playback requires a scene without physics/contact response; reset the owned scene')
        players=[p for p in robot.findall('plugin') if p.get('name')=='gp7::KinematicPlayback']
        if len(players)!=1 or any(p.get('name') in ['gp7::MeasuredVelocityPID','gz::sim::systems::JointPositionController','gp7::DetachableGrasp'] for p in robot.findall('plugin')):
            raise RuntimeError('Expected one kinematic player and no effort/velocity servos or physical grasp constraints')
        indices=sorted(int(j.findtext('index')) for j in players[0].findall('kinematic_joint') if int(j.findtext('index'))>=0)
        if indices!=list(range(8)):raise RuntimeError('Kinematic player must own all eight joints exactly once')
    if not kinematic and config['grasp'].get('ideal_attachment_approved') and not any(p.get('name')=='gp7::DetachableGrasp' for p in robot.findall('plugin')):
        raise RuntimeError('Explicit detachable assist requested but the live world lacks its plugin')
    actual = np.array([[float(robot.findtext(f"joint[@name='{name}']/axis/limit/{key}"))
                        for key in ['lower', 'upper']] for name in NAMES])
    np.testing.assert_allclose(actual, arm_limits(config), atol=1e-12, rtol=0)
    for name in NAMES:
        if kinematic:continue
        cap = float(robot.findtext(f"joint[@name='{name}']/axis/limit/velocity"))
        # Live SDF serialization normalizes the unlimited -1 sentinel to inf.
        valid_cap=(cap==-1. or np.isposinf(cap)) if config['execution'].get('synchronous_position_inputs') else np.isclose(cap,config['velocity_limit_rad_s'])
        if not valid_cap:
            raise RuntimeError('Live velocity cap mismatch: '+name)
        if config['execution'].get('synchronous_position_inputs'):
            if float(robot.findtext(f"joint[@name='{name}']/axis/limit/effort"))!=1e6:
                raise RuntimeError('Expected artificial kinematic servo authority, not a motor torque cap: '+name)
        plugins=[p for p in robot.findall('plugin') if p.findtext('joint_name')==name
                 and p.get('name')!='gz::sim::systems::JointStatePublisher']
        if len(plugins)!=1:raise RuntimeError('Expected exactly one arm controller: '+name)
        audit_arm_controller(plugins[0],name,config)
    if config['execution']['jaw_actuation']=='gazebo_position':
        for name in JAW_NAMES:
            plugins=[p for p in robot.findall('plugin') if p.findtext('joint_name')==name]
            if len(plugins)!=1:raise RuntimeError('Expected exactly one jaw controller: '+name)
            audit_position_controller(plugins[0],name,.016,6+JAW_NAMES.index(name))
            limits=robot.find(f"joint[@name='{name}']/axis/limit")
            np.testing.assert_allclose([float(limits.findtext(k)) for k in ['lower','upper']],
                                       [-.0002,.012],rtol=0.,atol=1e-12)
            cap=float(limits.findtext('velocity'))
            if not (cap==-1. or np.isposinf(cap)):raise RuntimeError('Expected kinematic jaw velocity interface')
            if float(limits.findtext('effort'))!=1e6:raise RuntimeError('Expected artificial jaw servo authority')
    if live_scene_response is None:
        scene = Scene()
        text_format.Parse(service('/world/gp7_lab/scene/info','gz.msgs.Empty','gz.msgs.Scene','',partition),scene)
    else:
        scene = live_scene_response
    models = {m.name:m for m in scene.model}
    p=models['box_a'].pose.position
    np.testing.assert_allclose([p.x,p.y,p.z],[.5,.5,.1],atol=1e-8,rtol=0)
    p=models['box_b'].pose.position
    if not allow_cube_reposition:
        np.testing.assert_allclose([p.x,p.y,p.z],[.45,-.2,.025],atol=.001,rtol=0)
    if sum(len(link.visual) for link in models['box_b'].link)!=9:
        raise RuntimeError('Coloured cube visuals/marks were lost')
    return {'arm_actuation':config['execution']['arm_actuation'],
            'actual_intervals_rad':actual.tolist(),'cube_center_m':[p.x,p.y,p.z],
            'pedestal_center_m':[.5,.5,.1],'cube_visual_count':9}


class ControllerCheck:
    def __init__(self, config):
        self.config=config; self.names=NAMES+JAW_NAMES
        self.node=rclpy.create_node('gp7_task_controller_check',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.q=self.velocity=None;self.sim=None;self.feedback_wall=0.;self.clock_wall=0.
        self.error=None;self.max_speed=np.zeros(8);self.rows=[];self.authorized=False
        self.phase='readiness';self.curve=None;self.begin=0.;self.last_record=-1.
        self.sub=self.node.create_subscription(JointState,'/joint_states',self.feedback,qos_profile_sensor_data)
        self.clock_sub=self.node.create_subscription(Clock,'/clock',self.clock,qos_profile_sensor_data)
        self.pubs={name:self.node.create_publisher(Float64,f'/gp7/{name}/cmd_pos',10) for name in self.names}
        self.velocity_pubs=({name:self.node.create_publisher(Float64,f'/gp7/{name}/cmd_vel_ff',10) for name in NAMES}
                            if config['execution'].get('velocity_feedforward') else {})
        self.position_node=Node() if config['execution'].get('synchronous_position_inputs') else None
        self.position_pub=None if self.position_node is None else self.position_node.advertise('/gp7/joint_positions',Actuators)
        self.playback=None
        if config['execution'].get('kinematic_playback'):
            from pick_place_playback import PlaybackClient
            self.playback=PlaybackClient()
            self.playback_heartbeat=self.playback.node.advertise('/gp7/kinematic/heartbeat',GzBoolean)
            self.last_heartbeat=0.

    def clock(self,msg):
        stamp=msg.clock.sec+msg.clock.nanosec*1e-9
        if self.sim is not None and stamp<self.sim-1e-8:self.error='Backward simulation time jump'
        if self.sim is None or stamp>self.sim:self.clock_wall=time.monotonic()
        self.sim=stamp

    def feedback(self,msg):
        values={name:(p,v) for name,p,v in zip(msg.name,msg.position,msg.velocity)}
        if not all(n in values for n in self.names):return
        self.q=np.array([values[n][0] for n in self.names]);self.velocity=np.array([values[n][1] for n in self.names])
        self.feedback_wall=time.monotonic();self.max_speed=np.maximum(self.max_speed,abs(self.velocity))
        if not np.isfinite(self.q).all() or not np.isfinite(self.velocity).all():self.error='Nonfinite feedback'
        if self.curve is not None:
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            ref=self.curve.evaluate(np.clip(stamp-self.begin,0.,self.curve.duration)).position
            if max(abs(ref[:6]-self.q[:6]))>self.config['execution']['tracking_abort_rad']:
                self.error='Tracking error exceeded configured gate'
            if self.config['execution'].get('jaw_actuation') in ['gazebo_position','kinematic_playback'] and max(abs(ref[6:]-self.q[6:]))>.001:
                self.error='Idealized jaw tracking error exceeded 1 mm'
            if stamp-self.last_record>=.019:
                self.rows.append([self.phase,stamp,*ref,*self.q,*self.velocity]);self.last_record=stamp

    def spin(self):
        self.heartbeat()
        rclpy.spin_once(self.node,timeout_sec=.01)
        if self.error:raise RuntimeError(self.error)
        if self.q is not None and time.monotonic()-self.feedback_wall>.5:raise RuntimeError('Stale joint feedback')
        if self.sim is not None and time.monotonic()-self.clock_wall>3.:raise RuntimeError('Simulation clock stopped; gate interrupted')

    def heartbeat(self):
        if getattr(self,'playback',None) is not None and time.monotonic()-self.last_heartbeat>=.05:
            self.playback_heartbeat.publish(GzBoolean(data=True));self.last_heartbeat=time.monotonic()

    def publish(self,q,velocity=None):
        limits=arm_limits(self.config)
        if (not np.isfinite(q).all() or np.any(q[:6]<limits[:,0]) or np.any(q[:6]>limits[:,1])
                or np.any(q[6:] < -1e-9) or np.any(q[6:] > .012+1e-9)):
            raise ValueError('Invalid reference; refusing rather than clamping')
        if getattr(self,'velocity_pubs',{}):
            velocity=np.zeros(6) if velocity is None else np.asarray(velocity)[:6]
            if velocity.shape!=(6,) or not np.isfinite(velocity).all() or max(abs(velocity))>self.config['velocity_limit_rad_s']+1e-9:
                raise ValueError('Invalid velocity feedforward; refusing rather than clamping')
            for name,value in zip(NAMES,velocity):self.velocity_pubs[name].publish(Float64(data=float(value)))
        self.send_positions(q)

    def send_positions(self,q):
        if getattr(self,'playback',None) is not None:
            self.playback.hold();return
        if getattr(self,'position_pub',None) is not None:
            self.position_pub.publish(Actuators(position=list(map(float,q))))
        else:
            for name,value in zip(self.names,q):self.pubs[name].publish(Float64(data=float(value)))

    def ready(self,require_home=True):
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            self.spin()
            if getattr(self,'playback',None) is not None:
                writers,readers=self.playback.node.topic_info('/gp7/kinematic/trajectory')
                if len(writers)>1:raise RuntimeError('Competing kinematic trajectory command owner')
                if len(writers)!=1 or len(readers)!=1:continue
                if self.playback.status.get('state')=='rejected':raise RuntimeError('Kinematic player rejected state: '+str(self.playback.status))
            if getattr(self,'position_node',None) is not None:
                writers,readers=self.position_node.topic_info('/gp7/joint_positions')
                if len(writers)>1:raise RuntimeError('Competing Gazebo vector command owner')
                if len(writers)!=1 or not readers:continue
            pubs=[*self.pubs.values(),*getattr(self,'velocity_pubs',{}).values()]
            if self.q is None or self.sim is None or not all(p.get_subscription_count()>0 for p in pubs):continue
            owners=self.node.get_publishers_info_by_topic('/joint_states')
            if not owners or any(not o.node_name or o.node_name=='_NODE_NAME_UNKNOWN_' for o in owners):continue
            if len(owners)!=1 or owners[0].node_name!='gp7_sim_bridge':raise RuntimeError('Unexpected measured state owners: '+str([o.node_name for o in owners]))
            discovered=True
            topics=[f'/gp7/{name}/cmd_pos' for name in self.names]
            topics += [f'/gp7/{name}/cmd_vel_ff' for name in getattr(self,'velocity_pubs',{})]
            for topic in topics:
                owners=self.node.get_publishers_info_by_topic(topic)
                if not owners or any(not o.node_name or o.node_name=='_NODE_NAME_UNKNOWN_' for o in owners):discovered=False;break
                if len(owners)!=1 or owners[0].node_name!=self.node.get_name():raise RuntimeError('Competing command owner: '+topic)
            if discovered:break
        else:raise RuntimeError('Missing measured feedback, clock, bridge subscribers or ownership discovery')
        if require_home and np.max(abs(self.q[:6]))>.05:raise RuntimeError('Unexpected startup state; no automatic repositioning')
        self.authorized=True

    def move(self,label,target,duration=4.):
        self.phase=label;self.begin=self.sim
        if getattr(self,'playback',None) is not None:self.begin+=.05
        self.curve=quintic_given_duration(self.q.copy(),np.asarray(target),duration)
        if getattr(self,'playback',None) is not None:self.playback.execute(self.curve,self.begin,self.spin)
        stable=None;next_publish=-1.;deadline=time.monotonic()+90
        tolerances=np.array([.003]*6+[.0002]*2);speed=np.array([.008]*6+[.0008]*2)
        while time.monotonic()<deadline:
            self.spin();elapsed=max(0.,self.sim-self.begin)
            if elapsed>duration+15:raise RuntimeError(label+' did not settle')
            if getattr(self,'playback',None) is None and self.sim>=next_publish:
                self.publish(self.curve.evaluate(min(elapsed,duration)).position);next_publish=self.sim+.02
            good=elapsed>=duration and np.all(abs(self.q-target)<tolerances) and np.all(abs(self.velocity)<speed)
            stable=self.sim if good and stable is None else (stable if good else None)
            if stable is not None and self.sim-stable>=.3:
                record={'phase':label,'target':list(target),'measured':self.q.tolist(),
                        'max_arm_endpoint_error_rad':float(max(abs(self.q[:6]-target[:6]))),
                        'simulation_time_s':self.sim,'elapsed_sim_s':elapsed}
                print(json.dumps(record),flush=True);return record
        raise RuntimeError('Wall timeout: '+label)

    def finish(self):
        if getattr(self,'authorized',False) and self.q is not None:
            for pub in getattr(self,'velocity_pubs',{}).values():pub.publish(Float64(data=0.))
            # On failure hold the measured state, never reset all joints to zero.
            q=self.q.copy()
            if np.all(np.isfinite(q)):
                self.send_positions(q)
        self.node.destroy_node()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['fast','faster','sprint'])
    parser.add_argument('--detachable-grasp',action='store_true')
    args=parser.parse_args()
    config=load_config(profile=args.profile);OUT.mkdir(parents=True,exist_ok=True)
    if args.detachable_grasp:
        activate_kinematic_playback(config)
    if int(os.environ.get('ROS_DOMAIN_ID','0'))!=config['ros_domain_id']:
        raise RuntimeError('Use the configured simulation ROS domain')
    report={'status':'FAIL','arm_actuation':config['execution']['arm_actuation'],
            'jaw_actuation':config['execution']['jaw_actuation'],
            'scope':'Small-motion measured tracking of standard inputs only; NOT torque, grasp or full-workspace validation'}
    check=None;rclpy.init()
    try:
        report['live_world_audit']=live_audit(config)
        check=ControllerCheck(config);check.ready()
        stages=[('startup_hold',[0.]*8),('arm_positive',[.10,.08,-.08,.08,.10,-.06,0,0]),
                ('jaw_stroke_only',[.10,.08,-.08,.08,.10,-.06,.010,.010]),
                ('return_open',[0.]*8),('arm_negative',[-.08,-.06,.06,-.06,-.08,.06,0,0]),
                ('final_home',[0.]*8)]
        report['stages']=[check.move(name,target) for name,target in stages]
        if not config['execution'].get('synchronous_position_inputs') and (max(check.max_speed[:6])>config['velocity_limit_rad_s']+.005 or max(check.max_speed[6:])>.0205):
            raise RuntimeError('Measured velocity gate exceeded')
        report['after_scene_audit']=live_audit(config)
        report['status']='PASS'
    except Exception as error:
        report['error']=str(error);raise
    finally:
        if check:
            report['max_measured_speed']=dict(zip(check.names,check.max_speed.tolist()))
            with (OUT/'controller_tracking.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['phase','sim_time_s',*[f'ref_{n}' for n in check.names],*[f'q_{n}' for n in check.names],*[f'v_{n}' for n in check.names]])
                writer.writerows(check.rows)
            check.finish()
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PKG/'scripts/build_pick_place_world.py',PKG/'config/pick_place.yaml']}
        (OUT/'controller_gate.json').write_text(json.dumps(report,indent=2)+'\n')
        rclpy.shutdown()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
