#!/usr/bin/env python3
"""Shared geometry for the sampled Sprint pre-grasp workspace."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as E

PKG = Path(__file__).resolve().parents[1]
MODEL_NAME = 'gp7_pickup_area'
PROFILE = 'sprint'
GRASP_MAP_PATH = PKG / 'config/grasp_workspace_sprint.json'
ENDPOINT_INPUTS = (
    'config/pick_place.yaml', 'urdf/gp7_with_gripper.urdf', 'worlds/gp7_bringup.sdf',
    'worlds/pick_place_assets/box_a.sdf', 'worlds/pick_place_assets/box_b.sdf',
)


def _signature(inputs, config_path):
    digest = hashlib.sha256()
    for name in inputs:
        source = Path(config_path) if config_path and name == 'config/pick_place.yaml' else PKG / name
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def grasp_signature(config_path=None):
    scripts = ('build_pick_place_world', 'kinematics', 'pick_place_collision',
               'pick_place_config', 'pick_place_profiles', 'derive_grasp_workspace')
    return _signature((*ENDPOINT_INPUTS, *(f'scripts/{name}.py' for name in scripts)), config_path)


def passing_cells(survey):
    """Include a cell only when its four corners and center all passed."""
    passed = {tuple(row['xy']) for row in survey['samples'] if row['pass'] is True}
    xs, ys = survey['x_edges_m'], survey['y_edges_m']
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            points = {(x0, y0), (x0, y1), (x1, y0), (x1, y1),
                      (round((x0+x1)/2, 6), round((y0+y1)/2, 6))}
            if points <= passed:
                yield x0, x1, y0, y1


def load_grasp_workspace(config_path=None):
    """Use the same current workspace for display and quick preflight."""
    try:
        survey = json.loads(GRASP_MAP_PATH.read_text())
    except FileNotFoundError:
        return None
    if (survey['profile'] != PROFILE or survey.get('scope') != 'pre_grasp_endpoint'
            or survey['signature'] != grasp_signature(config_path)):
        return None
    return survey


def workspace_polygons(survey):
    """Round exposed convex corners inward, never bridge unsupported cells.

    At a lattice vertex, round only if exactly one accepted cell touches it.
    Shared edges, concave corners, diagonal contacts and holes stay intact.
    The piecewise-linear outline is used for both rendering and membership.
    """
    cells = list(passing_cells(survey))
    corners = [((x0, y0), (x1, y0), (x1, y1), (x0, y1)) for x0, x1, y0, y1 in cells]
    counts = Counter(vertex for cell in corners for vertex in cell)
    for bounds, cell in zip(cells, corners):
        vertices = []
        for i, corner in enumerate(cell):
            if counts[corner] != 1:
                vertices.append(corner)
                continue
            # Sample a quadratic corner, cutting at most 45% of each edge.
            before, after = cell[i - 1], cell[(i + 1) % 4]
            for t in (0., .25, .5, .75, 1.):
                vertices.append(tuple(c + .45 * ((1-t)**2*(a-c) + t*t*(b-c))
                                      for a, c, b in zip(before, corner, after)))
        yield bounds, vertices


def in_grasp_workspace(x, y, survey):
    for (x0, x1, y0, y1), vertices in workspace_polygons(survey):
        if x0 <= x <= x1 and y0 <= y <= y1:
            # A convex counter-clockwise polygon includes its boundary.
            if all((bx-ax)*(y-ay) - (by-ay)*(x-ax) >= -1e-14
                   for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1])):
                return True
    return False


def write_overlay_mesh(path, polygons):
    """One top-facing visual mesh; no collision geometry or floor z-fighting."""
    with path.open('w') as stream:
        stream.write('o sprint_pre_grasp_workspace\nvn 0 0 1\n')
        first = 1
        for _, polygon in polygons:
            for x, y in polygon:
                stream.write(f'v {x:.12g} {y:.12g} 0.0015\n')
            for index in range(first + 1, first + len(polygon) - 1):
                stream.write(f'f {first}//1 {index}//1 {index+1}//1\n')
            first += len(polygon)


def add_pickup_area(world, profile, detachable_grasp, config_path, mesh_path):
    for previous in world.findall(f"model[@name='{MODEL_NAME}']"):
        world.remove(previous)
    if profile != PROFILE or not detachable_grasp:
        return
    survey = load_grasp_workspace(config_path)
    if survey is None:
        return
    polygons = list(workspace_polygons(survey))
    if not polygons:
        return
    write_overlay_mesh(mesh_path, polygons)
    model = E.SubElement(world, 'model', name=MODEL_NAME)
    E.SubElement(model, 'static').text = 'true'
    link = E.SubElement(model, 'link', name='pre_grasp_endpoint_workspace')
    visual = E.SubElement(link, 'visual', name='blue_workspace')
    E.SubElement(visual, 'cast_shadows').text = 'false'
    E.SubElement(visual, 'transparency').text = '0'
    mesh = E.SubElement(E.SubElement(visual, 'geometry'), 'mesh')
    E.SubElement(mesh, 'uri').text = mesh_path.resolve().as_uri()
    material = E.SubElement(visual, 'material')
    for tag in ('ambient', 'diffuse'):
        E.SubElement(material, tag).text = '0.10 0.38 0.76 1'
