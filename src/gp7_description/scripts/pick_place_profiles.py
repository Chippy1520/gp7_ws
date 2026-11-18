#!/usr/bin/env python3
"""Bounded simulation trial profiles; never manufacturer calibration or hardware authorization."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as E

PROFILES = {
    'fast': {'velocity': .8, 'acceleration': 2., 'delay_scale': .10,
             'timings': dict(rise=3.,approach=None,descent=2.5,grasp=1.5,lift=2.5,transfer=4.5,lower=1.2,release=1.3,retreat=1.5)},
    'faster': {'velocity': 1.5, 'acceleration': 4., 'delay_scale': .05,
               'timings': dict(rise=2.,approach=None,descent=1.4,grasp=1.5,lift=1.4,transfer=2.5,lower=.7,release=1.8,retreat=.9)},
    'sprint': {'velocity': 2.5, 'acceleration': 8., 'delay_scale': .02,
               'assisted_execution': dict(delay_scale=.005,settle_duration_s=.05,grasp_contact_hold_s=.12),
               'timings': dict(rise=1.5,approach=None,descent=.65,grasp=1.5,lift=.70,transfer=1.20,lower=.40,release=1.2,retreat=.50)},
}


def apply_profile(config,name):
    if name not in PROFILES:raise ValueError('Unknown simulation profile: '+str(name))
    config=deepcopy(config);profile=PROFILES[name]
    root=E.parse(Path(__file__).resolve().parents[1]/'urdf/gp7_with_gripper.urdf').getroot()
    # S/T blocked the rear branch. Do not broaden L on an unverified zero/sign map.
    for key in ['joint_s','joint_t']:
        config['joint_limits'][key]=[float(root.find(f"joint[@name='{key}']/limit").get(k)) for k in ['lower','upper']]
    config['workspace_scope']='Expanded model S/T ranges; other task intervals retained pending manufacturer-zero reconciliation'
    config['velocity_limit_rad_s']=profile['velocity'];config['acceleration_limit_rad_s2']=profile['acceleration']
    config['simulation_profile']=name
    config['planning']['minimum_approach_duration_s']=.5
    config['execution'].update(command_period_s=.005,settle_duration_s=.1,grasp_contact_hold_s=.25,
                               delay_scale=profile['delay_scale'],velocity_feedforward=False)
    return config
