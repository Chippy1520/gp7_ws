"""Compare the refactor with real calculations captured before backend changes."""
import ast
import hashlib
import json

import unittest

import numpy as np

from gp7_demo.runtime import ROOT, SCRIPTS
from kinematics import ExportedModel
from pick_place_planner import joint_segment
from pick_place_profiles import PROFILES

BASELINE = json.loads((ROOT / "tests/fixtures/numerical_baseline.json").read_text())


class NumericalEquivalenceTests(unittest.TestCase):
    def test_forward_kinematics_and_jacobians(self):
        model = ExportedModel()
        for sample in BASELINE["fk"]:
            pose, jacobian = model.evaluate(sample["q"])
            np.testing.assert_allclose(pose, sample["fk"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(jacobian, sample["jacobian"], atol=1e-12, rtol=0)

    def test_quintic_position_velocity_acceleration(self):
        trajectory = joint_segment("BASELINE", np.zeros(6),
                                   np.array([.01, -.02, .01, .01, -.01, .02]), 2.)
        for order in range(3):
            actual = trajectory.evaluate(BASELINE["trajectory"]["times"], order)
            expected = BASELINE["trajectory"]["values"][str(order)]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_timing_and_execution_profiles_are_exact(self):
        self.assertEqual(PROFILES, BASELINE["profiles"])

    def test_algorithm_syntax_trees_are_unchanged(self):
        for name, expected in BASELINE["algorithm_ast_sha256"].items():
            with self.subTest(module=name):
                syntax = ast.parse((SCRIPTS / name).read_text())
                if name == "verify_pick_place_controllers.py":
                    audit = next(node for node in syntax.body
                                 if isinstance(node, ast.FunctionDef) and node.name == 'live_audit')
                    self.assertEqual(audit.args.args.pop().arg, 'live_scene_response')
                    self.assertIsNone(audit.args.defaults.pop().value)
                    self.assertEqual(audit.args.args.pop().arg, 'live_sdf_response')
                    self.assertIsNone(audit.args.defaults.pop().value)
                    adapter = audit.body[1]
                    self.assertEqual(ast.dump(adapter.test), ast.dump(ast.parse('live_sdf_response is None', mode='eval').body))
                    self.assertEqual(ast.dump(adapter.orelse[0]), ast.dump(ast.parse('message = live_sdf_response').body[0]))
                    audit.body[1:2] = adapter.body
                    scene_adapter = next(node for node in audit.body if isinstance(node, ast.If)
                                         and ast.dump(node.test) == ast.dump(ast.parse('live_scene_response is None', mode='eval').body))
                    self.assertEqual(ast.dump(scene_adapter.orelse[0]), ast.dump(ast.parse('scene = live_scene_response').body[0]))
                    position = audit.body.index(scene_adapter)
                    audit.body[position:position+1] = scene_adapter.body
                if name == "pick_place_collision.py":
                    # Only resource lookup changes. Require that exact adapter,
                    # then compare every original calculation with its old AST.
                    adapter = ast.parse("uri=str(PKG/'meshes'/Path(uri).name) if uri else uri").body[0]
                    removed = []

                    class RemoveAssetAdapter(ast.NodeTransformer):
                        def visit_Assign(self, node):
                            if ast.dump(node) == ast.dump(adapter):
                                removed.append(node)
                                return None
                            return self.generic_visit(node)

                    syntax = RemoveAssetAdapter().visit(syntax)
                    self.assertEqual(len(removed), 1)
                tree = ast.dump(syntax, include_attributes=False)
                self.assertEqual(hashlib.sha256(tree.encode()).hexdigest(), expected)

    def test_geometry_configuration_and_native_playback_are_unchanged(self):
        for name, expected in BASELINE["protected_files"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
