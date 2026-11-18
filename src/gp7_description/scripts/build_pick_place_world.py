#!/usr/bin/env python3
"""Generate the task world without editing the legacy world or prepared objects."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as E
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_place_config import PKG, NAMES, load_config, activate_kinematic_playback
from pick_place_pickup_area import add_pickup_area

GRASP_FRICTION = 1.2  # Trial pad/cube coefficient, not measured material data.


def child(parent, tag, text=None):
    element = parent.find(tag)
    if element is None:
        element = E.SubElement(parent, tag)
    if text is not None:
        element.text = str(text)
    return element


def add_finger_pads(link, side):
    # Recess the backing 1 mm; two separated pads retain the original envelope.
    direction = 1 if side == 'positive' else -1
    for kind in ['collision', 'visual']:
        backing = link.find(kind)
        child(backing, 'pose', f'0 {direction*.0005} .03 0 0 0')
        child(backing.find('geometry/box'), 'size', '.02 .007 .06')
        for name, x in [('left', -.006), ('right', .006)]:
            pad = E.SubElement(link, kind, name=f'contact_pad_{name}_{kind}')
            child(pad, 'pose', f'{x} {-direction*.0035} .03 0 0 0')
            child(child(child(pad, 'geometry'), 'box'), 'size', '.008 .001 .06')
            if kind == 'visual':
                material = child(pad, 'material')
                child(material, 'ambient', '.08 .08 .08 1')
                child(material, 'diffuse', '.08 .08 .08 1')


def set_grip_preload(robot,force_n):
    """Nominal force at 2 mm blocked error; command cap, not a wrench measurement."""
    force_n=float(force_n)
    if not np.isfinite(force_n) or not 0<force_n<=10:
        raise ValueError('Simulation jaw preload must be finite and in (0, 10] N')
    for plugin in robot.findall('plugin'):
        if plugin.findtext('joint_name') not in ['gripper_joint_positive','gripper_joint_negative']:continue
        joint=robot.find(f"joint[@name='{plugin.findtext('joint_name')}']")
        mass=float(robot.findtext(f"link[@name='{joint.findtext('child')}']/inertial/mass"))
        damping=float(joint.findtext('axis/dynamics/damping','0'))
        # Restore approximately critical unloaded damping as preload stiffness rises.
        derivative=max(2.,2*np.sqrt((force_n/.002)*mass)-damping)
        for key,value in {'p_gain':force_n/.002,'i_gain':0.,'i_max':0.,'i_min':0.,'d_gain':derivative,
                          'cmd_max':force_n,'cmd_min':-force_n}.items():child(plugin,key,value)


def relocate_meshes(root):
    """Resolve saved CAD mesh references inside this checkout, not the author's home."""
    for uri in root.findall('.//geometry/mesh/uri'):
        name = Path(unquote(urlparse(uri.text).path)).name
        mesh = PKG / 'meshes' / name
        if not mesh.is_file():
            raise FileNotFoundError('Required robot mesh is missing: ' + str(mesh))
        uri.text = mesh.as_uri()


def build(config_path=None, output_path=None, grip_force_n=None,profile=None,detachable_grasp=False):
    config = load_config(config_path,profile=profile)
    source = PKG/'worlds/gp7_bringup.sdf'
    root = E.parse(source).getroot()
    relocate_meshes(root)
    world = root.find('world')
    output = Path(output_path) if output_path else PKG/'worlds/gp7_pick_place.sdf'
    add_pickup_area(world, profile, detachable_grasp, config_path,
                    output.with_suffix('.pickup-area.obj'))
    if detachable_grasp:
        activate_kinematic_playback(config)
        child(world.find('physics'),'max_step_size',.001)
    inputs = [source]
    for name in ['box_a', 'box_b']:
        asset = PKG/f'worlds/pick_place_assets/{name}.sdf'
        model = E.parse(asset).getroot().find('model')
        inputs.append(asset)
        previous = world.find(f"model[@name='{name}']")
        world.remove(previous)
        world.append(copy.deepcopy(model))
    pedestal = world.find("model[@name='box_a']")
    cube = world.find("model[@name='box_b']")
    base = np.array(config['pedestal']['base_xyz_m'])
    size = np.array(config['pedestal']['size_m'])
    expected = base + [0, 0, size[2]/2]
    if not np.allclose(np.fromstring(pedestal.findtext('pose'), sep=' ')[:3], expected, atol=1e-12, rtol=0):
        raise ValueError('Prepared pedestal does not match the fixed task configuration')
    if not np.allclose(np.fromstring(cube.findtext('pose'), sep=' ')[:3], config['cube']['initial_center_xyz_m'], atol=1e-12, rtol=0):
        raise ValueError('Prepared cube differs from configured pickup; refusing to move it')
    robot = world.find("model[@name='gp7']")
    urdf=E.parse(PKG/'urdf/gp7_with_gripper.urdf').getroot()
    links = [robot.find(f"link[@name='gripper_finger_{side}']") for side in ['positive', 'negative']]
    for side, link in zip(['positive', 'negative'], links):
        add_finger_pads(link, side)
    for link in links + cube.findall('link'):
        for collision in link.findall('collision'):
            ode = child(child(child(collision, 'surface'), 'friction'), 'ode')
            child(ode, 'mu', GRASP_FRICTION)
            child(ode, 'mu2', GRASP_FRICTION)
    for name in NAMES:
        joint = robot.find(f"joint[@name='{name}']")
        lower, upper = config['joint_limits'][name]
        limit = joint.find('axis/limit')
        child(limit, 'lower', lower)
        child(limit, 'upper', upper)
        child(limit, 'velocity', config['velocity_limit_rad_s'])
        effort=float(urdf.find(f"joint[@name='{name}']/limit").get('effort'))
        child(limit,'effort',min(float(limit.findtext('effort')),effort))
        controller = next(p for p in robot.findall('plugin') if p.findtext('joint_name') == name)
        robot.remove(controller)
        controller=E.SubElement(robot,'plugin',filename='gz-sim-joint-position-controller-system',
                                name='gz::sim::systems::JointPositionController')
        for key,value in {'joint_name':name,'topic':f'/gp7/{name}/cmd_pos',
                          'initial_position':0.,'use_velocity_commands':'true',
                          'cmd_max':config['velocity_limit_rad_s']}.items():child(controller,key,value)

        # In PID-bypass mode cmd_max is a SPEED in rad/s, not an effort in Nm.
        # Position validity is checked before publishing and in measured feedback.
    preload=grip_force_n if grip_force_n is not None else config['grasp'].get('preload_force_n')
    if preload is not None and not detachable_grasp:set_grip_preload(robot,preload)
    # Standard contact sensors provide physical grasp evidence, not attachment.
    E.SubElement(world, 'plugin', filename='gz-sim-contact-system', name='gz::sim::systems::Contact')
    for joint in robot.findall("joint[@type='prismatic']"):
        link = robot.find(f"link[@name='{joint.findtext('child')}']")
        sensor = E.SubElement(link, 'sensor', name='pick_place_contacts', type='contact')
        child(sensor, 'always_on', 'true')
        child(sensor, 'update_rate', 100)
        child(sensor, 'topic', f"/pick_place/{link.get('name')}/contacts")
        contact = E.SubElement(sensor, 'contact')
        for collision in link.findall('collision'):
            E.SubElement(contact, 'collision').text = collision.get('name')
    cube_link=cube.find('link')
    sensor=E.SubElement(cube_link,'sensor',name='pick_place_support_contacts',type='contact')
    child(sensor,'always_on','true');child(sensor,'update_rate',100)
    contact=E.SubElement(sensor,'contact');child(contact,'collision',cube_link.find('collision').get('name'))
    if detachable_grasp:
        # Pure kinematic scene: no physics/servo can fight the checked q(t).
        # Keep collision geometry for FCL; there is no physical collision response.
        for plugin in list(world.findall('plugin')):
            if plugin.get('name') in ['gz::sim::systems::Physics','gz::sim::systems::Contact']:
                world.remove(plugin)
        for plugin in list(robot.findall('plugin')):
            if plugin.find('joint_name') is not None and plugin.get('name')!='gz::sim::systems::JointStatePublisher':
                robot.remove(plugin)
        for link in robot.findall('link')+cube.findall('link'):
            for sensor in list(link.findall("sensor[@type='contact']")):link.remove(sensor)
        player=E.SubElement(robot,'plugin',filename=str(PKG.parents[1]/'install/gp7_description/lib/libgp7_kinematic_playback.so'),name='gp7::KinematicPlayback')
        names=NAMES+['gripper_joint_positive','gripper_joint_negative']
        for link in robot.findall('link'):
            element=E.SubElement(player,'home_link')
            child(element,'name',link.get('name'));child(element,'pose',link.findtext('pose','0 0 0 0 0 0'))
        for joint in robot.findall('joint'):
            name=joint.get('name');element=E.SubElement(player,'kinematic_joint')
            index=names.index(name) if name in names else -1
            for key,value in {'name':name,'parent':joint.findtext('parent'),'child':joint.findtext('child'),
                              'kind':joint.get('type'),'axis':joint.findtext('axis/xyz','0 0 1'),'index':index}.items():child(element,key,value)
            if index>=0:
                for key in ['lower','upper']:child(element,key,joint.findtext('axis/limit/'+key))
                child(element,'speed',config['velocity_limit_rad_s'] if index<6 else .016)
    if output_path is None:
        sensed=E.Element('sdf',version='1.9');sensed.append(copy.deepcopy(cube));E.indent(sensed)
        E.ElementTree(sensed).write(PKG/'worlds/pick_place_assets/box_b_task_sensed.sdf',encoding='utf-8',xml_declaration=True)
    E.indent(root)
    E.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    manifest = {
        'scope': 'simulation task envelope; URDF/effort/inertia assumptions remain provisional',
        'legacy_and_prepared_assets_unchanged_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'config': config, 'world': str(output),
        'world_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'startup': 'zero link geometry and zero controller targets',
        'contacts': 'not applicable: geometric attachment/support, no contact-force simulation' if detachable_grasp else 'standard finger contact sensors',
        'detachable_grasp_assist': bool(detachable_grasp),
        'physics_max_step_s':float(world.findtext('physics/max_step_size')),
        'arm_actuation': config['execution']['arm_actuation'],
        'arm_controller': 'simulation-tick quintic interpolation into native joint-state/link-pose components; no servo' if detachable_grasp else 'Gazebo JointPositionController PID bypass; cmd_max is rad/s',
        'physics_enabled':not detachable_grasp,
        'actuation_scope': 'pure kinematic playback; no PID, effort servo, gravity, friction, or collision response; FCL geometric checks remain' if detachable_grasp else 'idealized arm position execution; no torque/dynamic feasibility claim',
        'grasp_friction_mu_mu2': GRASP_FRICTION,
        'jaw_actuation':config['execution']['jaw_actuation'],
        'jaw_preload_force_n': None if detachable_grasp else preload,
        'jaw_preload_scope': 'not applicable: idealized jaw positions and geometric attachment' if detachable_grasp else 'nominal effort preload, not measured normal force',
    }
    manifest_path = output.with_suffix('.json') if output_path else PKG/'config/pick_place_simulation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--grip-force',type=float,help='Nominal preload per jaw at 2 mm blocked error; maximum 10 N')
    parser.add_argument('--profile',choices=['fast','faster','sprint'])
    parser.add_argument('--detachable-grasp',action='store_true',help='Opt-in contact-triggered detachable constraint; starts detached')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(build(args.config,args.output,grip_force_n=args.grip_force,profile=args.profile,detachable_grasp=args.detachable_grasp))
