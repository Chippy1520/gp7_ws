#!/usr/bin/env python3
"""Deterministic numerical tests for derived GP7 DH and inverse kinematics."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

k=load('kinematics')
audit=load('audit_model')
import xml.etree.ElementTree as ET

METRICS={}

class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.model=k.ExportedModel()
        self.rng=np.random.default_rng(3563)

    def test_dh_config_matches_implementation(self):
        data=json.loads((k.URDF.parents[1]/'config/dh_parameters.json').read_text())
        rows=[(r['theta_offset'],r['d'],r['a'],r['alpha']) for r in data['rows']]
        np.testing.assert_allclose(rows,k.dh_rows(np.zeros(6)),atol=1e-14)

    def test_home(self):
        T=k.fk_dh(np.zeros(6))
        np.testing.assert_allclose(T[:3,3],[.635,0,.815],atol=1e-12)

    def test_exact_model_matches_independent_urdf_fk(self):
        robot=ET.parse(k.URDF).getroot()
        for _ in range(50):
            q=self.rng.uniform(-2,2,6)
            expected=audit.fk(robot,dict(zip(k.NAMES,q)))['gripper_tcp']
            np.testing.assert_allclose(self.model.fk(q),expected,atol=1e-12)

    def test_dh_vs_recorded_export(self):
        pmax,rmax=0.,0.
        for _ in range(1000):
            q=self.rng.uniform(-np.pi,np.pi,6)
            ideal,exact=k.fk_dh(q),self.model.fk(q)
            pmax=max(pmax,float(np.linalg.norm(ideal[:3,3]-exact[:3,3])))
            rmax=max(rmax,float(np.linalg.norm(k.so3_log(ideal[:3,:3]@exact[:3,:3].T))))
        METRICS['dh_vs_export_1000_samples']={'position_max_m':pmax,'orientation_max_rad':rmax,'seed':3563}
        self.assertLess(pmax,2e-6)
        self.assertLess(rmax,3e-6)

    def test_analytic_roundtrip(self):
        pmax,rmax=0.,0.
        branches=[]
        for _ in range(500):
            q=self.rng.uniform(-2.7,2.7,6)
            target=k.fk_dh(q)
            solutions=k.ik_analytic(target,seed=q)
            self.assertTrue(solutions)
            branches.append(len(solutions))
            for solution in solutions:
                got=k.fk_dh(solution)
                pmax=max(pmax,float(np.linalg.norm(got[:3,3]-target[:3,3])))
                rmax=max(rmax,float(np.linalg.norm(k.so3_log(got[:3,:3]@target[:3,:3].T))))
        METRICS['analytic_roundtrip_500_samples']={'position_max_m':pmax,'orientation_max_rad':rmax,'branch_counts':sorted(set(branches))}
        self.assertLess(pmax,1e-9)
        self.assertLess(rmax,1e-9)

    def test_exported_roundtrip(self):
        pmax,rmax=0.,0.
        for _ in range(150):
            q=self.rng.uniform(-2.5,2.5,6)
            target=self.model.fk(q)
            solutions=k.ik_exported(target,seed=q,model=self.model)
            self.assertTrue(solutions)
            got=self.model.fk(solutions[0])
            pmax=max(pmax,float(np.linalg.norm(got[:3,3]-target[:3,3])))
            rmax=max(rmax,float(np.linalg.norm(k.so3_log(got[:3,:3]@target[:3,:3].T))))
        METRICS['exact_export_roundtrip_150_samples']={'position_max_m':pmax,'orientation_max_rad':rmax}
        self.assertLess(pmax,2e-9)
        self.assertLess(rmax,2e-9)

    def test_wrist_singularities(self):
        for q5 in [0.,np.pi,-np.pi,1e-10]:
            q=np.array([.3,-.5,.7,.4,q5,-.8])
            target=k.fk_dh(q)
            result=k.ik_analytic(target,seed=q)
            self.assertTrue(result)
            np.testing.assert_allclose(k.fk_dh(result[0]),target,atol=1e-8)

    def test_position_singularities(self):
        # Wrist centre on the base axis: select one seed heading from an
        # infinite family rather than dividing by its horizontal radius.
        target=np.eye(4)
        target[:3,3]=[0,0,.6+k.D6]
        result=k.ik_analytic(target,seed=np.array([.7,0,0,.2,0,0]))
        self.assertTrue(result)
        np.testing.assert_allclose(k.fk_dh(result[0]),target,atol=1e-8)
        # Fully extended and folded effective planar elbow branches merge.
        beta=np.arctan2(k.D4,k.A3)
        for phi in [0,np.pi]:
            q=np.array([.3,-.7,beta+phi,.2,.5,-.4])
            target=k.fk_dh(q)
            result=k.ik_analytic(target,seed=q)
            self.assertTrue(result)
            np.testing.assert_allclose(k.fk_dh(result[0]),target,atol=1e-8)

    def test_geometric_jacobian(self):
        q=np.array([.2,-.4,.6,.3,-.7,.8])
        _,J=self.model.evaluate(q)
        numeric=np.empty((6,6))
        step=1e-6
        for i in range(6):
            dq=np.zeros(6); dq[i]=step
            plus,minus=self.model.fk(q+dq),self.model.fk(q-dq)
            numeric[:3,i]=(plus[:3,3]-minus[:3,3])/(2*step)
            numeric[3:,i]=k.so3_log(plus[:3,:3]@minus[:3,:3].T)/(2*step)
        np.testing.assert_allclose(J,numeric,atol=2e-9)

    def test_unreachable_and_invalid(self):
        target=np.eye(4); target[0,3]=10.
        self.assertEqual(k.ik_analytic(target),[])
        target[0,0]=2.
        with self.assertRaises(ValueError):
            k.ik_analytic(target)

    def test_explicit_exported_limits(self):
        q=np.array([.2,-.4,.6,.3,-.7,.8])
        target=self.model.fk(q)
        limits=np.c_[q-.01,q+.01]
        self.assertTrue(k.ik_exported(target,seed=q,model=self.model,limits=limits))
        self.assertEqual(k.ik_exported(target,seed=q,model=self.model,limits=np.zeros((6,2))),[])
        # Same pose with a 2*pi wrist winding inside an explicit interval.
        winding=q.copy(); winding[5]+=2*np.pi
        limits=np.c_[winding-.01,winding+.01]
        solutions=k.ik_exported(target,seed=q,model=self.model,limits=limits)
        self.assertTrue(solutions)
        np.testing.assert_allclose(solutions[0],winding,atol=1e-7)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(KinematicsTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=HERE.parents[2]/'verification/kinematics_tests.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'metrics':METRICS},indent=2)+'\n')
    print(json.dumps(METRICS,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
