#!/usr/bin/env python3
"""Rest-to-rest Cartesian references must stay C2 when joined to guarded holds."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from probe_jacobian_polar_route import CartesianPath,differential_segment
from pick_place_config import load_config
from pick_place_planner import top_down_ik

class CartesianBoundaryTests(unittest.TestCase):
    def test_cartesian_acceleration_is_analytic(self):
        path=CartesianPath([.45,-.2,.15],[.45,-.2,.025],.65)
        for t in [0.,.1,.3,.65]:
            np.testing.assert_allclose(path.acceleration(t),path.curve.evaluate(t).acceleration,atol=1e-12)

    def test_polar_acceleration_matches_velocity_derivative(self):
        path=CartesianPath([.45,-.2,.3],[.5,.5,.3],2.,polar=True)
        for t in [.1,.5,1.,1.7]:
            h=1e-5
            actual=(path.evaluate(t+h)[1]-path.evaluate(t-h)[1])/(2*h)
            np.testing.assert_allclose(path.acceleration(t),actual,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(path.acceleration(0.),0.,atol=1e-12)
        np.testing.assert_allclose(path.acceleration(2.),0.,atol=1e-12)

    def test_differential_ik_endpoints_are_stationary_in_position_velocity_acceleration(self):
        config=load_config(profile='sprint')
        start=top_down_ik([.45,-.2,.15],config=config)
        segment,report=differential_segment('DESCEND_PICK',start,
            CartesianPath([.45,-.2,.15],[.45,-.2,.025],.65),config)
        for t in [0.,segment.duration]:
            np.testing.assert_allclose(segment.evaluate(t,1),0.,atol=1e-7)
            np.testing.assert_allclose(segment.evaluate(t,2),0.,atol=1e-7)
        self.assertLess(report['max_cartesian_error_m'],1e-5)

if __name__=='__main__':unittest.main()
