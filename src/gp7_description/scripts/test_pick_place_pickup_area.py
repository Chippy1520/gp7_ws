#!/usr/bin/env python3
"""Sample qualification and world-builder integration; no live Gazebo."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as E

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pick_place_world import build
from pick_place_pickup_area import (GRASP_MAP_PATH, MODEL_NAME, add_pickup_area, grasp_signature,
                                    in_grasp_workspace, passing_cells, workspace_polygons)


class PickupAreaTests(unittest.TestCase):
    def test_failed_or_missing_samples_exclude_cell(self):
        points = [(0, 0), (0, 1), (1, 0), (1, 1), (.5, .5)]
        samples = [{'xy': xy, 'pass': True} for xy in points]
        survey = {'x_edges_m': [0, 1], 'y_edges_m': [0, 1], 'samples': samples}
        self.assertEqual(list(passing_cells(survey)), [(0, 1, 0, 1)])
        samples[-1]['pass'] = False
        self.assertEqual(list(passing_cells(survey)), [])
        samples.pop()
        self.assertEqual(list(passing_cells(survey)), [])

    def test_rounding_only_removes_exposed_corners(self):
        survey = {'x_edges_m': [0, 1], 'y_edges_m': [0, 1],
                  'samples': [{'xy': xy, 'pass': True}
                              for xy in [(0, 0), (0, 1), (1, 0), (1, 1), (.5, .5)]]}
        self.assertFalse(in_grasp_workspace(.01, .01, survey))
        self.assertTrue(in_grasp_workspace(.513, .317, survey))
        self.assertFalse(in_grasp_workspace(1.01, .5, survey))

    def test_grasp_map_is_current_and_keeps_base_gap(self):
        grasp = json.loads(GRASP_MAP_PATH.read_text())
        cells = list(passing_cells(grasp))
        self.assertEqual(grasp['signature'], grasp_signature())
        self.assertEqual(grasp['scope'], 'pre_grasp_endpoint')
        self.assertAlmostEqual(grasp['cube_center_z_m'], .025)
        self.assertGreater(len(cells), 24)
        self.assertTrue(any(x0 <= .45 <= x1 and y0 <= -.20 <= y1 for x0, x1, y0, y1 in cells))
        self.assertFalse(any(x0 <= 0 <= x1 and y0 <= 0 <= y1 for x0, x1, y0, y1 in cells))

    def test_builder_uses_survey_only_for_matching_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = [(True, 'sprint', True), (True, 'faster', False), (False, 'sprint', False)]
            for playback, profile, expected in settings:
                with self.subTest(profile=profile, playback=playback):
                    file = build(output_path=Path(directory)/'world.sdf', profile=profile, detachable_grasp=playback)
                    world = E.parse(file).getroot().find('world')
                    path = f"model[@name='{MODEL_NAME}']"
                    self.assertEqual(len(world.findall(path)), int(expected))
                    if expected:
                        mesh = file.with_suffix('.pickup-area.obj')
                        add_pickup_area(world, profile, playback, None, mesh)
                        self.assertEqual(len(world.findall(path)), 1)
                        model = world.find(path)
                        self.assertIsNone(model.find('.//collision'))
                        self.assertEqual(model.findtext('static'), 'true')
                        self.assertEqual(model.findtext('.//mesh/uri'), mesh.as_uri())
                        polygons = list(workspace_polygons(json.loads(GRASP_MAP_PATH.read_text())))
                        vertices = [point for _, polygon in polygons for point in polygon]
                        rendered = [tuple(map(float, row.split()[1:3])) for row in mesh.read_text().splitlines()
                                    if row.startswith('v ')]
                        self.assertEqual(rendered, vertices)
                        self.assertTrue(all(v.findtext('transparency') == '0'
                                            for v in model.findall('link/visual')))
                        self.assertIsNone(model.find("link[@name='sampled_pickup_to_retreat']"))
                        with patch('pick_place_pickup_area.grasp_signature', return_value='stale'):
                            add_pickup_area(world, profile, playback, None, mesh)
                        self.assertIsNone(world.find(path))


if __name__ == '__main__':
    unittest.main()
