"""The preserved collision reference and CAD must work without the author's checkout."""
from pathlib import Path
import unittest
from unittest.mock import patch

from gp7_demo.runtime import ROOT
import pick_place_collision as collision


class PackageResourceTests(unittest.TestCase):
    def test_collision_geometry_loads_only_this_checkouts_supplied_meshes(self):
        with patch.object(collision, 'load_stl', wraps=collision.load_stl) as loader:
            checker = collision.CollisionChecker(cube_only=False)
        self.assertTrue(checker.meshes)
        self.assertIn('box_a', checker.environment)
        self.assertIn('box_b', checker.environment)
        for call in loader.call_args_list:
            mesh = Path(call.args[0]).resolve()
            self.assertTrue(mesh.is_relative_to(ROOT / 'src/gp7_description/meshes'), mesh)


if __name__ == '__main__':
    unittest.main()
