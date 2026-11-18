#!/usr/bin/env python3
"""Deterministic task-envelope generation checks; no simulator side effects."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as E
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pick_place_world import build, GRASP_FRICTION
from pick_place_config import PKG, NAMES, arm_limits, load_config
from kinematics import ExportedModel, ik_exported


def canonical(element):
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or '').strip(),
            tuple(canonical(c) for c in element))


class TaskWorldTests(unittest.TestCase):
    def test_generation_preserves_scene_legacy_and_matching_guards(self):
        files = [PKG/'worlds/gp7_bringup.sdf', *[PKG/f'worlds/pick_place_assets/{n}.sdf' for n in ['box_a','box_b']]]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        with tempfile.TemporaryDirectory() as directory:
            output = build(output_path=Path(directory)/'task.sdf')
            world = E.parse(output).getroot().find('world')
            for name in ['box_a','box_b']:
                expected = E.parse(PKG/f'worlds/pick_place_assets/{name}.sdf').getroot().find('model')
                actual=copy.deepcopy(world.find(f"model[@name='{name}']"))
                if name=='box_b':
                    for key in ['mu', 'mu2']:
                        path='link/collision/surface/friction/ode/'+key
                        self.assertEqual(float(actual.findtext(path)),GRASP_FRICTION)
                        expected.find(path).text=str(GRASP_FRICTION)
                    self.assertEqual(len(actual.findall('link/sensor')),1)
                    for link in actual.findall('link'):
                        for sensor in list(link.findall('sensor')):link.remove(sensor)
                self.assertEqual(canonical(actual),canonical(expected))
            robot = world.find("model[@name='gp7']")
            for side in ['positive', 'negative']:
                collision=robot.find(f"link[@name='gripper_finger_{side}']/collision")
                for key in ['mu', 'mu2']:
                    self.assertEqual(float(collision.findtext('surface/friction/ode/'+key)),GRASP_FRICTION)
            config = load_config()
            for name in NAMES:
                interval = [float(robot.findtext(f"joint[@name='{name}']/axis/limit/{key}")) for key in ['lower','upper']]
                self.assertEqual(interval,config['joint_limits'][name])
                plugin = next(p for p in robot.findall('plugin') if p.findtext('joint_name')==name
                              and p.get('name')!='gz::sim::systems::JointStatePublisher')
                self.assertEqual(plugin.get('name'),'gz::sim::systems::JointPositionController')
                self.assertEqual(plugin.findtext('use_velocity_commands'),'true')
                self.assertEqual(float(plugin.findtext('cmd_max')),config['velocity_limit_rad_s'])
                self.assertEqual(float(plugin.findtext('initial_position')),0.)
            self.assertEqual(len(robot.findall('link/sensor')),2)
        self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in files])

    def test_contact_pads_fit_original_envelope_and_are_sensed(self):
        legacy=E.parse(PKG/'worlds/gp7_bringup.sdf').getroot().find("world/model[@name='gp7']")
        with tempfile.TemporaryDirectory() as directory:
            robot=E.parse(build(output_path=Path(directory)/'task.sdf')).getroot().find("world/model[@name='gp7']")
            for side in ['positive','negative']:
                link=robot.find(f"link[@name='gripper_finger_{side}']")
                original=legacy.find(f"link[@name='{link.get('name')}']")
                pads=[c for c in link.findall('collision') if c.get('name').startswith('contact_pad_')]
                self.assertEqual(len(pads),2)
                names=[part.get('name') for part in link if part.get('name')]
                self.assertEqual(len(names),len(set(names)))
                self.assertEqual(canonical(link.find('inertial')),canonical(original.find('inertial')))
                sensed={c.text for c in link.findall('sensor/contact/collision')}
                self.assertEqual(sensed,{c.get('name') for c in link.findall('collision')})
                for kind in ['collision','visual']:
                    bounds=[]
                    for part in link.findall(kind):
                        pose=np.fromstring(part.findtext('pose'),sep=' ')
                        half=np.fromstring(part.findtext('geometry/box/size'),sep=' ')/2
                        np.testing.assert_allclose(pose[3:],0.,atol=1e-12)
                        bounds.append([pose[:3]-half,pose[:3]+half])
                    np.testing.assert_allclose(np.min(np.array(bounds)[:,0],axis=0),[-.01,-.004,0.],atol=1e-12)
                    np.testing.assert_allclose(np.max(np.array(bounds)[:,1],axis=0),[.01,.004,.06],atol=1e-12)
                for pad in pads:
                    np.testing.assert_allclose(np.fromstring(pad.findtext('geometry/box/size'),sep=' '),[.008,.001,.06])
                    y=float(pad.findtext('pose').split()[1])
                    self.assertAlmostEqual(y,-.0035 if side=='positive' else .0035)
                    for key in ['mu','mu2']:
                        self.assertEqual(float(pad.findtext('surface/friction/ode/'+key)),GRASP_FRICTION)

    def test_config_rejects_invalid_envelopes_and_startups(self):
        baseline=load_config()
        for key,value in [('joint_limits',dict(baseline['joint_limits'],joint_l=[-2.,1.])),
                          ('velocity_limit_rad_s',float('nan')),
                          ('acceleration_limit_rad_s2',-1),
                          ('initial_position_rad',[.1,0,0,0,0,0])]:
            config=copy.deepcopy(baseline);config[key]=value
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'bad.yaml';path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):load_config(path)

    def test_all_handoff_hypotheses_have_task_limit_ik(self):
        model=ExportedModel();limits=arm_limits(load_config())
        for xyz in [[.45,-.2,.025],[.45,-.2,.325],[.5,.5,.225],[.5,.5,.325]]:
            target=np.eye(4);target[:3,:3]=np.diag([1.,-1.,-1.]);target[:3,3]=xyz
            solutions=ik_exported(target,model=model,limits=limits)
            self.assertTrue(solutions,xyz)
            self.assertTrue(all(np.all(q>=limits[:,0]) and np.all(q<=limits[:,1]) for q in solutions))


if __name__=='__main__':unittest.main(verbosity=2)
