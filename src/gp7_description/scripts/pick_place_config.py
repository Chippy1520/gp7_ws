#!/usr/bin/env python3
"""Validated simulation-only task configuration shared by generation and planning."""
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import yaml

PKG = Path(__file__).resolve().parents[1]
NAMES = ['joint_s', 'joint_l', 'joint_u', 'joint_r', 'joint_b', 'joint_t']


def load_config(path=None,profile=None):
    config = yaml.safe_load(Path(path or PKG/'config/pick_place.yaml').read_text())
    if profile is not None:
        from pick_place_profiles import apply_profile
        config=apply_profile(config,profile)
    if set(config['joint_limits']) != set(NAMES):
        raise ValueError('Task limits must specify exactly the six arm joints')
    limits = np.array([config['joint_limits'][name] for name in NAMES], dtype=float)
    source = E.parse(PKG/'urdf/gp7_with_gripper.urdf').getroot()
    urdf = np.array([[float(source.find(f"joint[@name='{name}']/limit").get(k))
                      for k in ['lower', 'upper']] for name in NAMES])
    if (limits.shape != (6, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(limits[:, 0] < urdf[:, 0]) or np.any(limits[:, 1] > urdf[:, 1])):
        raise ValueError('Task intervals must be finite, ordered and inside provisional URDF intervals')
    initial = np.asarray(config['initial_position_rad'], dtype=float)
    if initial.shape != (6,) or not np.isfinite(initial).all() or np.any(initial != 0):
        raise ValueError('Generated link geometry requires matching zero startup targets')
    if np.any(limits[:, 0] > initial) or np.any(limits[:, 1] < initial):
        raise ValueError('Startup must be inside every task interval')
    for key in ['velocity_limit_rad_s', 'acceleration_limit_rad_s2']:
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    # Official GP7 speeds; keep the lower model cap where its value was rounded down.
    rated=np.deg2rad([375.,315.,410.,550.,550.,1000.])
    model=np.array([float(source.find(f"joint[@name='{name}']/limit").get('velocity')) for name in NAMES])
    speed_cap=float(np.min(np.minimum(model,rated)))
    if config['velocity_limit_rad_s'] > speed_cap:
        raise ValueError('Task speed exceeds a model/manufacturer joint speed cap')
    if config['acceleration_limit_rad_s2']>12.:
        raise ValueError('Task acceleration exceeds the bounded simulation trial envelope')
    if config['execution']['arm_actuation']!='gazebo_position' or config['execution'].get('velocity_feedforward'):
        raise ValueError('Coursework execution requires standard Gazebo PID-bypass position inputs')
    if config['grasp']['mode'] != 'physical' or config['grasp']['ideal_attachment_approved']:
        raise ValueError('This configuration does not authorize ideal attachment')
    return config


def arm_limits(config):
    return np.array([config['joint_limits'][name] for name in NAMES], dtype=float)


def activate_kinematic_playback(config):
    """Explicit runtime opt-in; never grants hardware or physical-grasp authority."""
    config['execution'].update(arm_actuation='kinematic_playback',jaw_actuation='kinematic_playback',
                               kinematic_playback=True,synchronous_position_inputs=False,velocity_feedforward=False)
    config['grasp'].update(mode='geometric_kinematic',ideal_attachment_approved=True,preload_force_n=None)
    return config
