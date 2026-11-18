#!/usr/bin/env python3
"""Prepare task objects only; preserve the bring-up robot and its controllers.

Default: write a separate world, model files and manifest. --apply replaces only
box_a and box_b in the existing gp7_project_lab/gp7_lab simulation. It never
commands the robot, changes joint limits, or starts a second simulator.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as E

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / 'worlds/pick_place_assets'
VERIFY = PKG.parents[1] / 'verification/pick_place_scene'


def child(parent, tag, text=None, **attrs):
    n = E.SubElement(parent, tag, attrs)
    if text is not None:
        n.text = str(text)
    return n


def visual(link, name, size, xyz, colour):
    v = child(link, 'visual', name=name)
    child(v, 'pose', ' '.join(map(str, [*xyz, 0, 0, 0])))
    child(child(child(v, 'geometry'), 'box'), 'size', ' '.join(map(str, size)))
    m = child(v, 'material')
    child(m, 'ambient', colour)
    child(m, 'diffuse', colour)
    return v


def serialize(root):
    E.indent(root)
    return E.tostring(root, encoding='unicode', xml_declaration=True) + '\n'


def service(name, request_type, response_type, request, partition):
    env = dict(os.environ, GZ_PARTITION=partition)
    p = subprocess.run(['gz', 'service', '-s', name, '--reqtype', request_type,
                        '--reptype', response_type, '--timeout', '5000', '--req', request],
                       env=env, capture_output=True, text=True, timeout=10)
    if p.returncode:
        raise RuntimeError(p.stderr + p.stdout)
    return p.stdout


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    source = PKG / 'worlds/gp7_bringup.sdf'
    root = E.parse(source).getroot()
    world = root.find('world')
    robot_before = E.tostring(world.find("model[@name='gp7']"))
    pedestal = world.find("model[@name='box_a']")
    cube = world.find("model[@name='box_b']")
    # Preserve original generated assets for an explicit rollback if needed.
    for name, model in [('box_a', pedestal), ('box_b', cube)]:
        wrapper = E.Element('sdf', version='1.9')
        wrapper.append(copy.deepcopy(model))
        (OUT / f'{name}_bringup_original.sdf').write_text(serialize(wrapper))
    pedestal.find('pose').text = '0.5 0.5 0.1 0 0 0'
    cube.find('pose').text = '0.45 -0.2 0.025 0 0 0'
    link = cube.find('link')
    for v in list(link.findall('visual')):
        link.remove(v)
    # Thin visual-only face plates. Collision/inertia remain one 50 mm cube.
    h, t, width = 0.025, 0.0001, 0.0498
    faces = [
        ('plus_x_red', [t, width, width], [h, 0, 0], '0.95 0.04 0.04 1'),
        ('minus_x_cyan', [t, width, width], [-h, 0, 0], '0.02 0.9 0.95 1'),
        ('plus_y_green', [width, t, width], [0, h, 0], '0.04 0.85 0.08 1'),
        ('minus_y_magenta', [width, t, width], [0, -h, 0], '0.95 0.03 0.8 1'),
        ('plus_z_blue', [width, width, t], [0, 0, h], '0.04 0.12 0.95 1'),
        ('minus_z_yellow', [width, width, t], [0, 0, -h], '0.95 0.85 0.03 1'),
    ]
    for args in faces:
        visual(link, *args)
    # Asymmetric white L and off-centre dot distinguish in-plane rotation.
    visual(link, 'top_L_long', [0.004, 0.029, t], [-0.012, 0, h+t], '1 1 1 1')
    visual(link, 'top_L_short', [0.020, 0.004, t], [-0.004, -0.0125, h+t], '1 1 1 1')
    visual(link, 'top_corner_dot', [0.004, 0.004, t], [0.014, 0.014, h+t], '1 1 1 1')
    # Explicit assumed friction for later physical grasp tests; not a validation.
    collision = link.find('collision')
    surface = child(collision, 'surface')
    ode = child(child(surface, 'friction'), 'ode')
    child(ode, 'mu', 0.8)
    child(ode, 'mu2', 0.8)
    pub = child(cube, 'plugin', filename='gz-sim-pose-publisher-system',
                name='gz::sim::systems::PosePublisher')
    for key, value in [('publish_model_pose', 'true'), ('publish_link_pose', 'false'),
                       ('use_pose_vector_msg', 'true'), ('update_frequency', '50'),
                       ('topic', '/pick_place/box_b/pose')]:
        child(pub, key, value)
    assert E.tostring(world.find("model[@name='gp7']")) == robot_before
    manifest = {
        'frame': 'world; robot base_link fixed at world origin',
        'pedestal': {'model': 'box_a', 'base_xyz_m': [0.5, 0.5, 0],
                     'model_center_xyz_m': [0.5, 0.5, 0.1],
                     'size_m': [0.3, 0.3, 0.2], 'top_z_m': 0.2},
        'cube': {'model': 'box_b', 'center_xyz_m': [0.45, -0.2, 0.025],
                 'size_m': [0.05, 0.05, 0.05], 'mass_kg': 0.1,
                 'initial_quaternion_xyzw': [0, 0, 0, 1],
                 'face_colours': {f[0]: f[3] for f in faces},
                 'top_marker': 'white asymmetric L and off-centre dot',
                 'friction_mu_assumption': 0.8,
                 'pose_topic_gz': '/pick_place/box_b/pose'},
        'placement': {'cube_center_xyz_m': [0.5, 0.5, 0.225],
                      'hover_cube_center_xyz_m': [0.5, 0.5, 0.325],
                      'hover_bottom_clearance_m': 0.1},
        'source_world': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'robot_unchanged': True,
        'limitations': ['Robot retains +/-0.6 rad bring-up limits and self_collision=false.',
                        'Scene preparation only: no grasp or transport implemented.',
                        'Pose IK is not collision or dynamics certification.',
                        'Re-running build_gazebo.py resets the legacy world, not this task overlay.']
    }
    for name, model in [('box_a', pedestal), ('box_b', cube)]:
        wrapper = E.Element('sdf', version='1.9')
        wrapper.append(copy.deepcopy(model))
        (OUT / f'{name}.sdf').write_text(serialize(wrapper))
    task_world = PKG / 'worlds/gp7_pick_place.sdf'
    task_world.write_text(serialize(root))
    (OUT / 'scene_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest, task_world


def apply(partition):
    from google.protobuf import text_format
    from gz.msgs10.scene_pb2 import Scene
    scene = Scene()
    text_format.Parse(service('/world/gp7_lab/scene/info', 'gz.msgs.Empty',
                              'gz.msgs.Scene', '', partition), scene)
    ids = {m.name: m.id for m in scene.model}
    if not all(n in ids for n in ['gp7', 'box_a', 'box_b']):
        raise RuntimeError('Expected gp7, box_a and box_b in existing world; no changes applied')
    (VERIFY / 'scene_before.pbtxt').write_text(text_format.MessageToString(scene))
    responses = []
    for name in ['box_a', 'box_b']:
        request = f'id: {ids[name]} type: MODEL'
        result = service('/world/gp7_lab/remove/blocking', 'gz.msgs.Entity',
                         'gz.msgs.Boolean', request, partition)
        if 'data: true' not in result:
            raise RuntimeError(f'Removal failed for {name}: {result}')
        responses.append({'remove': name, 'response': result.strip()})
        request = f'sdf_filename: {json.dumps(str(OUT / (name+".sdf")))} allow_renaming: false'
        result = service('/world/gp7_lab/create/blocking', 'gz.msgs.EntityFactory',
                         'gz.msgs.Boolean', request, partition)
        if 'data: true' not in result:
            raise RuntimeError(f'Creation failed for {name}: {result}; original asset retained for rollback')
        responses.append({'create': name, 'response': result.strip()})
    (VERIFY / 'apply_responses.json').write_text(json.dumps(responses, indent=2)+'\n')
    print(json.dumps(responses, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--partition', default='gp7_project_lab')
    args = parser.parse_args()
    manifest, path = prepare()
    print('Prepared:', path)
    if args.apply:
        apply(args.partition)
