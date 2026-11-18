#!/usr/bin/env python3
"""Rasterize the GP7's fixed-pose floor reach, not the full pick-and-place path."""
import argparse
import json
from math import ceil, hypot, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from build_pick_place_world import build
from kinematics import A1, A2, A3, D1, D4, D6, ExportedModel, ik_exported
from pick_place_collision import CollisionChecker
from pick_place_config import activate_kinematic_playback, arm_limits, load_config
from pick_place_pickup_area import GRASP_MAP_PATH, PROFILE, grasp_signature, passing_cells


def geometric_outer_radius(z):
    """Ideal DH shoulder/elbow triangle, with the TCP pointing straight down."""
    length = A2 + hypot(A3, D4)
    vertical = z + D6 - D1
    return A1 + sqrt(length**2 - vertical**2) if abs(vertical) <= length else 0.


def derive(world, signature, pitch=.025):
    config = activate_kinematic_playback(load_config(profile=PROFILE))
    z = config['cube']['initial_center_xyz_m'][2]
    limits = arm_limits(config)
    model = ExportedModel()
    checker = CollisionChecker(world_path=world, clearance=config['planning']['collision_clearance_m'], cube_only=True)
    radius = geometric_outer_radius(z)
    n = ceil(radius / pitch)
    edges = np.round(np.arange(-n, n + 1) * pitch, 6).tolist()
    mid = [round((a+b)/2, 6) for a, b in zip(edges, edges[1:])]
    target, cube = np.eye(4), np.eye(4)
    target[:3, :3] = np.diag([1., -1., -1.])
    samples = []
    for axis in (edges, mid):
        for x in axis:
            for y in axis:
                # Outside the ideal DH triangle there are no analytic seeds.
                if hypot(x, y) > radius + 1e-9:
                    continue
                target[:3, 3] = cube[:3, 3] = [x, y, z]
                for q in ik_exported(target, model=model, limits=limits):
                    if np.linalg.svd(model.evaluate(q)[1], compute_uv=False)[-1] < .005:
                        continue
                    if checker.check(q, cube_transform=cube, support='ground')['safe']:
                        samples.append({'xy': [x, y], 'pass': True})
                        break
    return {'profile': PROFILE, 'signature': signature, 'scope': 'pre_grasp_endpoint',
            'cube_center_z_m': z, 'cube_yaw_deg': 0, 'geometric_outer_radius_m': radius,
            'x_edges_m': edges, 'y_edges_m': edges, 'samples': samples}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--output', type=Path, default=GRASP_MAP_PATH)
    output = args.parse_args().output
    signature = grasp_signature()
    with TemporaryDirectory() as directory:
        world = build(output_path=Path(directory)/'world.sdf', profile=PROFILE, detachable_grasp=True)
        data = derive(world, signature)
    if signature != grasp_signature():
        raise RuntimeError('Workspace model changed during derivation')
    cells = list(passing_cells(data))
    if not cells:
        raise RuntimeError('No grasp cells passed')
    output.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    radii = [hypot(*row['xy']) for row in data['samples']]
    print(f"z={data['cube_center_z_m']:.3f} m; ideal DH outer bound={data['geometric_outer_radius_m']:.4f} m;")
    print(f'passing sampled endpoint radii={min(radii):.4f}–{max(radii):.4f} m, {len(radii)} samples, {len(cells)} tiles')


if __name__ == '__main__':
    main()
