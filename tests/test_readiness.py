"""GUI readiness is scoped to the owned scene and Sprint actor."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from gp7_demo.backend import Backend


class ReadinessTests(unittest.TestCase):
    def test_run_dispatches_without_validate_flag(self):
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

    def test_sprint_reuses_the_actor_that_reported_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            validation, run = (Path(directory) / name for name in ('validate', 'run'))
            actor = Mock(poll=Mock(return_value=None), wait=Mock(return_value=0))
            actor.stdout = iter(['GP7_MONITOR {"event":"warm_ready"}\n'])
            def launch(*args, **kwargs):
                (validation / 'result.json').write_text('{"status":"READY_WORKSPACE"}')
                return actor
            def send(line):
                self.assertEqual(json.loads(line), {'action': 'run', 'output': str(run)})
                (run / 'result.json').write_text('{"status":"PASS_ASSISTED_SEQUENCE"}')
                actor.stdout = Mock()
                actor.stdout.__iter__ = Mock(return_value=iter([]))
            actor.stdin.write.side_effect = send
            with patch('gp7_demo.backend.read_selection', return_value={'session_id': 4}), \
                    patch('gp7_demo.backend.output_directory', side_effect=[validation, run]), \
                    patch('gp7_demo.backend.subprocess.Popen', side_effect=launch) as spawn:
                backend = Backend()
                backend._work('validate', 'sprint', False)
                self.assertEqual(backend.prepared_profile, 'sprint')
                backend._work('run', 'sprint', False)
                spawn.assert_called_once()
                self.assertIsNone(backend._warm_process)
                self.assertEqual(list(backend.events.queue)[-1]['returncode'], 0)

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
