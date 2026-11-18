"""GUI readiness retains a scene identity, never the validation process."""
import unittest
from unittest.mock import patch
from gp7_demo.backend import Backend


class ReadinessTests(unittest.TestCase):
    def test_run_uses_a_new_task_without_validate_flag(self):
        with patch('gp7_demo.backend.read_selection', return_value={'session_id': 4}), \
                patch('gp7_demo.backend.threading.Thread') as worker:
            backend = Backend()
            backend.prepared_profile = 'sprint'
            backend._validated_scene = 4
            backend.start('run', 'sprint')
            self.assertIsNone(backend.process)
            self.assertIsNone(backend.prepared_profile)
            self.assertEqual(worker.call_args.kwargs['args'], ('run', 'sprint', False))
            self.assertNotIn('--validate', backend._arguments('run', 'sprint', 'run', False))

    def test_run_refuses_different_scene(self):
        with patch('gp7_demo.backend.read_selection', return_value={'session_id': 5}):
            backend = Backend()
            backend.prepared_profile = 'sprint'
            backend._validated_scene = 4
            with self.assertRaisesRegex(RuntimeError, 'Validate'):
                backend.start('run', 'sprint')

    def test_completed_run_does_not_require_scene_reset(self):
        with patch('gp7_demo.backend.read_selection', return_value={'profile': 'sprint'}):
            backend = Backend()
            self.assertFalse(backend.requires_reset('sprint'))
            self.assertTrue(backend.requires_reset('fast'))
