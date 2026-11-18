#!/usr/bin/env python3
"""Offline ranked approach regressions; mocked planning boundaries, no live nodes."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pick_place_motion as motion
from pick_place_config import NAMES
from pick_place_planner import joint_segment


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.config={'pedestal':{'base_xyz_m':[0.,0.,0.],'size_m':[1.,1.,.1]},
                     'planning':{'collision_clearance_m':.01},
                     'joint_limits':{name:[-2.,2.] for name in NAMES},
                     'velocity_limit_rad_s':.4,'acceleration_limit_rad_s2':.5}
        self.e=SimpleNamespace(q=np.zeros(8),config=self.config,
            checker=SimpleNamespace(point_reach_bound=.1),
            model=SimpleNamespace(evaluate=Mock(return_value=(None,np.eye(6))),fk=Mock(return_value=np.eye(4))))
        self.cube=np.eye(4);self.cube[:3,3]=[-.45,-.2,.025]
        self.timings={'rise':2.,'approach':None,'descent':3.}
        self.ik=self.mock('ik_exported',side_effect=lambda *a,**k:[])
        self.bottom=self.mock('gripper_bottom',return_value=1.)
        self.check=self.mock('checked',side_effect=lambda e,s,**kw:{'phase':s.name})
        self.descent=self.mock('differential_segment',side_effect=self.make_descent)

    def mock(self,name,**kwargs):
        return self.stack.enter_context(patch.object(motion,name,**kwargs))

    @staticmethod
    def goal(value):
        q=np.zeros(6);q[0]=value;return q

    @staticmethod
    def make_descent(name,q,path,config):
        return joint_segment(name,q,q,path.duration),{'phase':name}

    def candidates(self,*goals):
        self.ik.side_effect=[list(goals),[],[],[],[],[]]

    def run_plan(self):
        return motion.approach(self.e,self.cube,self.timings,.25)

    def test_preferred_hover_skips_other_heights_after_a_checked_pair(self):
        self.ik.side_effect=[[self.goal(.1)]]
        motion.approach(self.e,self.cube,self.timings,.25,prefer_hover=True)
        self.ik.assert_called_once()
        self.assertEqual(self.check.call_count,2)
        self.assertTrue(self.e.approach_diagnostics['used_preferred_hover'])

    def test_failed_preferred_descent_uses_checked_fallback_without_repeating_height(self):
        heights=[]
        def inverse(target,**_):
            height=target[2,3]
            heights.append(height)
            return [self.goal(.1 if np.isclose(height,.40) else .3)]
        self.ik.side_effect=inverse
        def descend(name,q,path,config):
            if q[0]<.2:raise ValueError('unsafe descent')
            return self.make_descent(name,q,path,config)
        self.descent.side_effect=descend
        motion.approach(self.e,self.cube,self.timings,.25,prefer_hover=True)
        np.testing.assert_allclose(heights,[.40,.25,.30,.35,.45,.50])
        self.assertEqual(self.check.call_count,3)
        self.assertEqual(self.e.approach_diagnostics['counts']['descent'],1)
        self.assertFalse(self.e.approach_diagnostics['used_preferred_hover'])

    def test_retry_descent_integration_failure_without_committing_rejected_pair(self):
        self.candidates(self.goal(.1),self.goal(.3))
        def descend(name,q,path,config):
            if q[0]<.2:raise ValueError('DESCEND_PICK: wrist singularity at t=.7')
            return self.make_descent(name,q,path,config)
        self.descent.side_effect=descend
        segments,reports=self.run_plan()
        self.assertEqual([s.name for s in segments],['APPROACH_PICKUP','DESCEND_PICK'])
        self.assertEqual(len(reports),2)
        self.assertAlmostEqual(segments[0].evaluate(segments[0].duration)[0],.3)
        self.assertEqual(self.e.approach_diagnostics['counts']['descent'],1)
        self.assertIn('wrist singularity',self.e.approach_diagnostics['rejections'][-1]['error'])
        self.assertIs(reports[0]['branch_selection'],self.e.approach_diagnostics)

    def test_retry_descent_collision_failure(self):
        self.candidates(self.goal(.1),self.goal(.3))
        def check(e,s,**kw):
            if s.name=='DESCEND_PICK' and s.evaluate(0)[0]<.2:
                raise ValueError('finger/cube clearance failed')
            return {'phase':s.name}
        self.check.side_effect=check
        segments,_=self.run_plan()
        self.assertAlmostEqual(segments[-1].evaluate(0)[0],.3)
        self.assertEqual(self.descent.call_count,2)
        self.assertEqual(self.e.approach_diagnostics['counts']['descent'],1)

    def test_retry_approach_check_failure_before_any_descent(self):
        self.candidates(self.goal(.1),self.goal(.3))
        def check(e,s,**kw):
            if s.name=='APPROACH_PICKUP' and s.evaluate(s.duration)[0]<.2:
                raise ValueError('approach cube collision')
            return {'phase':s.name}
        self.check.side_effect=check
        self.run_plan()
        self.assertEqual(self.e.approach_diagnostics['counts']['approach_check'],1)
        self.assertEqual(self.descent.call_count,1)

    def test_ranks_across_heights_and_integrates_only_selected_candidate(self):
        self.ik.side_effect=[[self.goal(1.)],[self.goal(.1)],[],[],[],[]]
        segments,_=self.run_plan()
        self.assertAlmostEqual(self.e.approach_diagnostics['selected']['height_m'],.3)
        self.assertAlmostEqual(segments[0].evaluate(segments[0].duration)[0],.1)
        self.assertEqual(self.descent.call_count,1)
        self.assertEqual(self.check.call_count,2)

    def test_ranking_penalizes_bad_endpoint_conditioning(self):
        self.candidates(self.goal(.1),self.goal(-.1))
        self.e.model.evaluate.side_effect=lambda q:(None,np.eye(6)*(.006 if q[0]>0 else 1.))
        segments,_=self.run_plan()
        self.assertAlmostEqual(segments[-1].evaluate(0)[0],-.1)

    def test_dynamic_asymmetric_limits_forwarded_and_enforced(self):
        self.config['joint_limits'][NAMES[0]]=[-.2,.7]
        self.candidates(self.goal(.8),self.goal(.3))
        self.run_plan()
        np.testing.assert_array_equal(self.ik.call_args_list[0].kwargs['limits'][0],[-.2,.7])
        self.assertEqual(self.e.approach_diagnostics['counts']['joint_limits'],1)
        self.assertEqual(self.e.approach_diagnostics['candidate_count'],2)

    def test_limit_margin_affects_rank_at_equal_travel(self):
        self.config['joint_limits'][NAMES[0]]=[-.11,2.]
        self.candidates(self.goal(-.1),self.goal(.1))
        self.run_plan()
        self.assertEqual(self.e.approach_diagnostics['selected']['branch'],1)

    def test_conditioning_rejection_precedes_expensive_checks(self):
        self.candidates(self.goal(.1))
        self.e.model.evaluate.return_value=(None,np.eye(6)*.004)
        with self.assertRaisesRegex(RuntimeError,'Endpoint Jacobian sigma_min'):
            self.run_plan()
        self.descent.assert_not_called();self.check.assert_not_called()
        self.assertEqual(self.e.approach_diagnostics['counts']['conditioning'],1)

    def test_between_sample_clearance_inflation_still_rejects(self):
        self.candidates(self.goal(.5))
        # Sampled geometry passes .11 m, but the reach/speed inflation does not.
        self.bottom.return_value=.11001
        with self.assertRaisesRegex(RuntimeError,'inflation'):
            self.run_plan()
        self.assertEqual(self.e.approach_diagnostics['counts']['clearance'],1)
        self.check.assert_not_called();self.descent.assert_not_called()

    def test_all_failed_descent_errors_are_actionable_and_no_selection_committed(self):
        self.candidates(self.goal(.1),self.goal(.3))
        self.descent.side_effect=ValueError('DESCEND_PICK: joint limits violated')
        with self.assertRaisesRegex(RuntimeError,'DESCEND_PICK: joint limits violated') as error:
            self.run_plan()
        self.assertIn('"descent": 2',str(error.exception))
        self.assertIsNone(self.e.approach_diagnostics['selected'])
        np.testing.assert_array_equal(self.e.q,np.zeros(8))

    def test_no_ik_counts_heights_and_reports_original_ik_error(self):
        self.ik.side_effect=[[],[],[],[],[],ValueError('IK unreachable rear workspace')]
        with self.assertRaisesRegex(RuntimeError,'IK unreachable rear workspace'):
            self.run_plan()
        self.assertEqual(self.e.approach_diagnostics['counts']['no_ik'],6)
        self.descent.assert_not_called()

    def test_48_candidates_do_not_trigger_48_descent_integrations(self):
        self.ik.side_effect=lambda *a,**kw:[self.goal(.1+i*.05) for i in range(8)]
        self.run_plan()
        self.assertEqual(self.e.approach_diagnostics['candidate_count'],48)
        self.assertEqual(self.e.approach_diagnostics['attempted_count'],1)
        self.descent.assert_called_once()
        self.assertEqual(self.check.call_count,2)

    def test_clearance_rise_is_preserved_once_across_branch_retry(self):
        self.candidates(self.goal(.1),self.goal(.3))
        # Only the initial position needs rising; approach geometry remains safe.
        calls=iter([.05])
        self.bottom.side_effect=lambda *args:next(calls,1.)
        def descend(name,q,path,config):
            if name=='DESCEND_PICK' and q[0]<.2:raise ValueError('descent failed')
            return self.make_descent(name,q,path,config)
        self.descent.side_effect=descend
        segments,reports=self.run_plan()
        self.assertEqual([s.name for s in segments],['CLEARANCE_RISE','APPROACH_PICKUP','DESCEND_PICK'])
        self.assertEqual(len(reports),3)
        self.assertEqual(sum(c.args[0]=='CLEARANCE_RISE' for c in self.descent.call_args_list),1)
        self.assertEqual(self.e.approach_diagnostics['counts']['descent'],1)

    def test_explicit_approach_time_is_a_floor_and_preserved_in_stats(self):
        self.candidates(self.goal(.1));self.timings['approach']=12.
        segments,_=self.run_plan()
        self.assertEqual(segments[0].duration,12.)
        self.assertEqual(self.e.approach_diagnostics['selected']['approach_duration_s'],12.)


if __name__=='__main__':unittest.main()
