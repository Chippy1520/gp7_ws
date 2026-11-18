"""Offline adapter checks. These tests never start a simulator or command motion."""
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from gp7_demo import application
from gp7_demo.backend import Backend
from gp7_demo.runtime import ROOT, environment, ros_command


class ApplicationTests(unittest.TestCase):
    def test_second_task_cannot_acquire_the_shared_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(application, "ROOT", Path(directory)):
                with application.task_lock():
                    with self.assertRaisesRegex(RuntimeError, "Another task"):
                        with application.task_lock():
                            self.fail("Acquired an already owned task lock")

    def test_run_requires_successful_validation(self):
        backend = Backend()
        with self.assertRaisesRegex(RuntimeError, "Validate"):
            backend.start("run", "fast")
        self.assertFalse(backend.busy)

    def test_stop_interrupts_only_the_owned_task(self):
        backend = Backend()
        backend.process = Mock()
        backend.process.poll.return_value = None
        backend.busy = True
        self.assertTrue(backend.stop())
        backend.process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertIsNone(backend.prepared_profile)

    def test_validate_and_open_do_not_request_execution(self):
        backend = Backend()
        output = ROOT / "verification/not-started"
        validate = backend._arguments("validate", "sprint", output, False)
        self.assertIn("--validate", validate)
        self.assertIn("--no-viewer", validate)
        opened = backend._arguments("open", "fast", output, False)
        self.assertIn("--open-only", opened)
        self.assertNotIn("--execute", opened)

    def test_ros_shell_keeps_paths_as_arguments(self):
        arguments = ros_command(["python", "/tmp/a path/task.py", "--validate"])
        self.assertEqual(arguments[-3:], ["python", "/tmp/a path/task.py", "--validate"])
        self.assertIn('exec "$@"', arguments[2])

    def test_child_environment_defaults_to_compatible_protobuf_backend(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(environment()["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], "python")

    def test_missing_success_evidence_is_not_ready(self):
        backend = Backend()
        process = Mock(stdout=iter([]))
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("gp7_demo.backend.output_directory", return_value=output), \
                    patch("gp7_demo.backend.subprocess.Popen", return_value=process):
                backend._work("validate", "fast", False)
        self.assertIsNone(backend.prepared_profile)
        events = list(backend.events.queue)
        self.assertTrue(any(event["event"] == "error" for event in events))
        self.assertEqual(events[-1]["returncode"], 1)


if __name__ == "__main__":
    unittest.main()
