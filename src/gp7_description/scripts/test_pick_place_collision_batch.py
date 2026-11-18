#!/usr/bin/env python3
"""Offline batch equivalence/guards and reproducible scalar-vs-batch timings.

Run directly for tests, or with --benchmark for real wall-clock measurements.
No ROS nodes, services, motion or generated-world writes.
"""
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pick_place_collision as collision
from pick_place_collision import CollisionChecker, _batch_link_fk, link_fk, NAMES
from kinematics import ExportedModel
from pick_place_planner import continuous_extrema, joint_segment, top_down_ik


def scalar_validate(checker, segment, jaws=(0.,0.), holding=False, tcp_cube=None,
                    support=None, period=.005, cube_world=None, contact_allowed=False):
    """Frozen original validate arithmetic/loop, for offline comparisons only."""
    model=ExportedModel() if holding else None
    times=np.linspace(0,segment.duration,int(np.ceil(segment.duration/period))+1)
    peak_speed=np.max(abs(continuous_extrema(segment.spline,1)),axis=1)
    half_interval=float(np.max(np.diff(times)))/2
    joint_motion=peak_speed*half_interval
    padding=checker.point_reach_bound*float(sum(peak_speed))*half_interval
    minimum=np.inf
    for t,q in zip(times,segment.evaluate(times)):
        cube=model.fk(q)@tcp_cube if holding else cube_world
        result=checker.check(q,jaws,cube,holding,support,motion_padding=padding,
                             joint_motion=joint_motion,contact_allowed=contact_allowed)
        minimum=min(minimum,result['minimum_checked_separation_m'])
        if not result['safe']:
            raise ValueError(f'{segment.name} collision at t={t:.6g}: {result["violations"]}')
    return {'collision_samples':len(times),'sample_period_max_s':period,
            'minimum_checked_separation_m':float(minimum),
            'between_sample_point_displacement_bound_m':padding}


class BatchCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cube=CollisionChecker(cube_only=True)
        cls.full=CollisionChecker()
        cls.model=ExportedModel()
        cls.q=top_down_ik([.45,-.2,.325])

    def assert_result_equal(self,first,second):
        self.assertEqual(first['safe'],second['safe'])
        self.assertAlmostEqual(first['minimum_checked_separation_m'],second['minimum_checked_separation_m'],places=11)
        self.assertEqual(len(first['violations']),len(second['violations']))
        for a,b in zip(first['violations'],second['violations']):
            if isinstance(a[-1],float):
                self.assertEqual(a[:-1],b[:-1]);self.assertAlmostEqual(a[-1],b[-1],places=11)
            else:self.assertEqual(a,b)

    def test_all_link_and_tcp_transforms_random_and_winding(self):
        rng=np.random.default_rng(58021)
        qs=np.r_[np.zeros((1,6)),rng.uniform(-4*np.pi,4*np.pi,(129,6))]
        jaws=np.array([.004,.011])
        batch=_batch_link_fk(self.cube.urdf,qs,jaws)
        for i,q in enumerate(qs):
            scalar=link_fk(self.cube.urdf,dict(zip(NAMES+['gripper_joint_positive','gripper_joint_negative'],[*q,*jaws])))
            self.assertEqual(set(batch),set(scalar))
            for name,T in scalar.items():np.testing.assert_allclose(batch[name][i],T,rtol=0,atol=5e-14)
            np.testing.assert_allclose(batch['link_t'][i]@self.model.tool,self.model.fk(q),rtol=0,atol=5e-14)

    def test_real_fcl_verdicts_pairs_padding_and_contacts(self):
        rng=np.random.default_rng(142)
        qs=np.r_[np.zeros((1,6)),self.q[None,:],rng.uniform(-1,1,(10,6))]
        jaws=np.array([.006,.009]);batch=_batch_link_fk(self.cube.urdf,qs,jaws)
        for c in (self.cube,self.full):
            for i,q in enumerate(qs):
                frames={n:v[i] for n,v in batch.items()}
                palm=frames['gripper_palm'].copy()
                resting=np.eye(4);resting[:3,3]=[.45,-.2,.025]
                for options in ({},{'holding':True,'cube_transform':self.model.fk(q)},
                                {'cube_transform':palm,'contact_allowed':True},
                                {'cube_transform':resting,'support':'ground'},
                                {'cube_transform':resting,'support':'box_a','holding':True}):
                    for padding in ({},{'motion_padding':.002,'joint_motion':np.linspace(0,.003,6)}):
                        with patch.object(collision,'separation',wraps=collision.separation) as measured:
                            scalar=c.check(q,jaws,**options,**padding)
                            count=measured.call_count
                        with patch.object(collision,'separation',wraps=collision.separation) as measured:
                            batched=c._check_state(q,jaws,**options,**padding,frames=frames)
                            self.assertEqual(count,measured.call_count)
                        self.assert_result_equal(scalar,batched)

    def test_same_samples_bounds_and_first_rejection(self):
        goal=self.q.copy();goal[0]+=.015
        for c in (self.cube,self.full):
            for holding in (False,True):
                s=joint_segment('batch_edges',self.q,goal,.333)
                options={'holding':holding,'tcp_cube':np.eye(4) if holding else None}
                before=scalar_validate(c,s,**options)
                seen=[];original=c._check_state
                def record(q,*a,**kw):
                    seen.append(np.array(q));return original(q,*a,**kw)
                with patch.object(c,'_check_state',side_effect=record):after=c.validate(s,**options)
                np.testing.assert_array_equal(seen,s.evaluate(np.linspace(0,s.duration,68)))
                for key,value in before.items():
                    if key=='minimum_checked_separation_m':self.assertAlmostEqual(value,after[key],places=12)
                    else:self.assertEqual(value,after[key])
        s=joint_segment('reject',np.zeros(6),np.zeros(6),.1)
        cube=np.eye(4);cube[:3,3]=[0,0,.01]
        messages=[]
        for run in (scalar_validate,self.cube.validate):
            with self.assertRaises(ValueError) as error:
                if run==scalar_validate:run(self.cube,s,cube_world=cube)
                else:run(s,cube_world=cube)
            messages.append(str(error.exception).split(':')[0])
        self.assertEqual(messages[0],messages[1])
        # An actual late collision must not disappear at a batch boundary.
        s=joint_segment('late_reject',self.q,top_down_ik([.45,-.2,.025]),2.)
        messages=[]
        for run in (lambda:scalar_validate(self.cube,s,holding=True,tcp_cube=np.eye(4)),
                    lambda:self.cube.validate(s,holding=True,tcp_cube=np.eye(4))):
            with self.assertRaises(ValueError) as error:run()
            messages.append(str(error.exception))
        self.assertEqual(messages[0],messages[1])
        self.assertIn('t=1.655:',messages[0])

    def test_guard_inside_fk_reentrant_checks_and_nested_validate(self):
        s=joint_segment('guard',self.q,self.q,.7)
        inner=joint_segment('inner',self.q,self.q,.001)
        flags=[];inside=[False];clock=[0.]
        original=collision._batch_link_fk
        def now():clock[0]+=.011;return clock[0]
        def batch(*args,**kw):
            old=inside[0];inside[0]=True
            try:return original(*args,**kw)
            finally:inside[0]=old
        def guard():
            flags.append(inside[0])
            self.assertTrue(self.cube.check(np.zeros(6))['safe'])
            self.cube.validate(inner)
        before=self.cube.validate(s,holding=True,tcp_cube=np.eye(4))
        with patch.object(collision.time,'monotonic',side_effect=now),patch.object(collision,'_batch_link_fk',side_effect=batch):
            after=self.cube.validate(s,holding=True,tcp_cube=np.eye(4),progress_callback=guard)
        self.assertEqual(before,after)
        self.assertGreater(len(flags),3);self.assertTrue(any(flags))
        self.assertFalse(flags[0]);self.assertFalse(flags[-1])

    def test_guard_exception_aborts_during_fk(self):
        s=joint_segment('abort',self.q,self.q,.3);clock=[0.];calls=[]
        def now():clock[0]+=.04;return clock[0]
        def guard():
            calls.append(1)
            if len(calls)==5:raise RuntimeError('feedback/contact guard')
        with patch.object(collision.time,'monotonic',side_effect=now),patch.object(self.cube,'_check_state',wraps=self.cube._check_state) as checks:
            with self.assertRaisesRegex(RuntimeError,'feedback/contact guard'):
                self.cube.validate(s,progress_callback=guard)
            self.assertEqual(checks.call_count,0)


def benchmark():
    import json
    q=top_down_ik([.45,-.2,.325]);goal=q.copy();goal[0]+=.015
    rows=[]
    for full,held,moving in [(False,True,False),(False,True,True),(False,False,True),(True,True,True)]:
        c=CollisionChecker(cube_only=not full)
        s=joint_segment('held_stationary_benchmark' if not moving else 'small_move_benchmark',q,goal if moving else q,5.)
        options={'holding':held,'tcp_cube':np.eye(4) if held else None}
        scalar=[];batched=[];gaps=[]
        for repeat in range(3):
            t=time.perf_counter();before=scalar_validate(c,s,**options);scalar.append(time.perf_counter()-t)
            stamps=[]
            t=time.perf_counter();after=c.validate(s,**options,progress_callback=lambda:stamps.append(time.perf_counter()));batched.append(time.perf_counter()-t)
            gaps.extend(np.diff(stamps).tolist())
        rows.append({'full':full,'holding':held,'moving':moving,'scalar_seconds':scalar,
                     'batch_with_guard_seconds':batched,'speedup_median':float(np.median(scalar)/np.median(batched)),
                     'maximum_guard_gap_seconds':max(gaps),'samples':after['collision_samples'],
                     'minimum_gap_difference_m':after['minimum_checked_separation_m']-before['minimum_checked_separation_m'],
                     'padding_identical':before['between_sample_point_displacement_bound_m']==after['between_sample_point_displacement_bound_m']})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    if '--benchmark' in sys.argv:benchmark()
    else:unittest.main(verbosity=2)
