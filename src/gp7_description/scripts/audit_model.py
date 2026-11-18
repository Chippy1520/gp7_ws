#!/usr/bin/env python3
"""Independent FK and static checks; no claim about physical calibration or collisions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np

PKG = Path(__file__).resolve().parents[1]


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def origin(joint):
    T = np.eye(4)
    o = joint.find('origin')
    if o is not None:
        T[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
        T[:3, :3] = rotation_rpy(np.fromstring(o.get('rpy', '0 0 0'), sep=' '))
    return T


def joint_motion(joint, value):
    T = np.eye(4)
    kind = joint.get('type')
    if kind == 'fixed':
        return T
    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
    assert abs(np.linalg.norm(axis)-1) < 1e-8
    if kind == 'prismatic':
        T[:3, 3] = axis*value
    else:
        x, y, z = axis
        K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        T[:3,:3] = np.eye(3) + math.sin(value)*K + (1-math.cos(value))*(K@K)
    return T


def fk(robot, positions=None):
    positions = positions or {}
    links = {x.get('name') for x in robot.findall('link')}
    children = {j.find('child').get('link') for j in robot.findall('joint')}
    roots = links-children
    assert len(roots) == 1, roots
    transforms = {roots.pop(): np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        count = len(pending)
        for j in pending[:]:
            p, c = j.find('parent').get('link'), j.find('child').get('link')
            if p in transforms:
                assert c not in transforms, f'duplicate child {c}'
                transforms[c] = transforms[p] @ origin(j) @ joint_motion(j, positions.get(j.get('name'), 0))
                pending.remove(j)
        assert len(pending) < count, 'disconnected tree or cycle'
    assert set(transforms) == links
    return transforms


def cad_transform(data):
    T = np.eye(4)
    T[:3, :3] = np.array(data[:9]).reshape(3,3).T
    T[:3, 3] = data[9:12]
    return T


def static_audit():
    robot = ET.parse(PKG/'urdf/gp7_with_gripper.urdf').getroot()
    transforms = fk(robot)
    joints = robot.findall('joint')
    assert len(transforms) == 11 and len(joints) == 10
    assert sum(j.get('type') == 'revolute' for j in joints) == 6
    assert sum(j.get('type') == 'prismatic' for j in joints) == 2
    report = {'status': 'PASS', 'scope': 'geometry and kinematic consistency, not collision or manufacturer calibration',
              'links': len(transforms), 'joints': len(joints), 'mesh_units': 'metres',
              'mesh_checks': {}, 'cad_frame_errors': {}, 'joint_motion_checks': {}}
    bounds=[]
    for link in robot.findall('link'):
        name=link.get('name')
        mesh=link.find('visual/geometry/mesh')
        if mesh is None:
            continue
        assert mesh.get('scale') == '1 1 1'
        prefix='package://gp7_description/'
        assert mesh.get('filename').startswith(prefix)
        path=PKG/mesh.get('filename')[len(prefix):]
        raw=path.read_bytes()
        n=struct.unpack_from('<I',raw,80)[0]
        assert len(raw)==84+50*n and n>0, str(path)
        dt=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        vertices=np.frombuffer(raw,dtype=dt,count=n,offset=84)['vertices'].reshape(-1,3).astype(float)
        assert np.isfinite(vertices).all()
        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
        assert 0.001 < max(hi-lo) < 2.0, f'scale concern {name}: {hi-lo}'
        vworld=vertices@transforms[name][:3,:3].T+transforms[name][:3,3]
        bounds.append([vworld.min(axis=0),vworld.max(axis=0)])
        report['mesh_checks'][name]={'triangles':n,'local_size_m':(hi-lo).tolist(),
                                   'sha256':hashlib.sha256(raw).hexdigest()}
    assert len(report['mesh_checks'])==7
    report['arm_home_aabb_m']={'min':np.min(np.array(bounds)[:,0,:],axis=0).tolist(),
                              'max':np.max(np.array(bounds)[:,1,:],axis=0).tolist()}
    frames=json.loads((PKG/'config/cad_frame_transforms.json').read_text())['frames']
    base_inv=np.linalg.inv(cad_transform(frames['CS_BASE']))
    for frame,link in [('CS_BASE','base_link')]+[(f'CS_{n.upper()}',f'link_{n}') for n in 'slurbt']:
        expected=base_inv@cad_transform(frames[frame])
        error=float(np.max(np.abs(transforms[link]-expected)))
        assert error < 1e-8, (frame,error)
        report['cad_frame_errors'][frame]=error
    for j in joints:
        if j.get('type') == 'fixed':
            continue
        name=j.get('name')
        q=0.006 if j.get('type')=='prismatic' else 0.1
        moved=fk(robot,{name:q})
        child=j.find('child').get('link')
        change=float(np.linalg.norm(moved[child]-transforms[child]))
        assert change > 0.005
        parent=j.find('parent').get('link')
        assert np.allclose(moved[parent],transforms[parent],atol=1e-12)
        report['joint_motion_checks'][name]={'test_value':q,'child_transform_change_norm':change}
    report['gripper_gap_tests_m']={}
    for q,gap in [(0,.070),(.006,.058),(.010,.050),(.012,.046)]:
        moved=fk(robot,{'gripper_joint_positive':q,'gripper_joint_negative':q})
        palm_inv=np.linalg.inv(moved['gripper_palm'])
        pos=palm_inv@moved['gripper_finger_positive']
        neg=palm_inv@moved['gripper_finger_negative']
        measured=float(pos[1,3]-neg[1,3]-.008)
        assert abs(measured-gap)<1e-10
        report['gripper_gap_tests_m'][str(q)]=measured
    t_tcp=np.linalg.inv(transforms['link_t'])@transforms['gripper_tcp']
    assert np.allclose(t_tcp[:3,3],[0,0,.155],atol=1e-10)
    report['home_tcp_position_m']=transforms['gripper_tcp'][:3,3].tolist()
    report['t_frame_to_tcp_m']=t_tcp[:3,3].tolist()
    report['missing_arm_inertials']=[link.get('name') for link in robot.findall('link')
                                     if link.get('name')!='gripper_tcp' and link.find('inertial') is None]
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=static_audit()
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
