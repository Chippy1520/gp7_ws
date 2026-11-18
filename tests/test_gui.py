"""Offline Qt/control-state tests with an explicit non-executing backend stub."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import queue
import unittest

from PyQt5 import QtWidgets
from gp7_demo.gui import DemoWindow
from gp7_demo.plotting import PlotData


class InertBackend:
    def __init__(self):
        self.events = queue.Queue()
        self.busy = False
        self.completed_cycle = False
        self.commands = []
        self.stops = 0

    def start(self, action, profile, reset=False):
        self.commands.append((action, profile, reset))

    def stop(self):
        self.stops += 1
        return True

    def requires_reset(self, profile):
        return False


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.backend = InertBackend()
        self.window = DemoWindow(self.backend)

    def tearDown(self):
        self.window.active_action = None
        self.window.timer.stop()
        self.window.close()

    def finish_validation(self, ready=True):
        self.window.start_action('validate')
        if ready:
            self.window.consume_event({'event': 'status', 'state': 'ready', 'stage': 'PREPARE'})
        self.window.consume_event({'event': 'finished', 'action': 'validate', 'returncode': 0,
                                   'report': {'status': 'READY_APPROACH_AND_DESCENT'}})

    def test_idle_has_approved_controls_empty_graphs_and_no_motion(self):
        buttons = {button.text() for button in self.window.findChildren(QtWidgets.QPushButton)}
        self.assertEqual(buttons, {'Open Gazebo', 'Validate', 'Run'})
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertFalse(self.window.data.stages)
        self.assertFalse(self.window.data.history)
        self.assertEqual(self.backend.commands, [])

    def test_open_requests_viewer_not_execution(self):
        self.window.open_button.click()
        self.assertEqual(self.backend.commands[0][0], 'open')
        self.assertIsNone(self.window.prepared_profile)

    def test_validation_requires_final_ready_evidence(self):
        self.finish_validation(ready=False)
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertIsNone(self.window.prepared_profile)

    def test_successful_validation_enables_run_for_current_profile_only(self):
        self.finish_validation()
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.profile.setCurrentIndex((self.window.profile.currentIndex() + 1) % 3)
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertIsNone(self.window.prepared_profile)

    def test_run_becomes_guarded_stop(self):
        self.finish_validation()
        self.window.run_button.click()
        self.assertEqual(self.backend.commands[-1][0], 'run')
        self.assertEqual(self.window.run_button.text(), 'Stop')
        self.window.run_button.click()
        self.assertEqual(self.backend.stops, 1)
        self.assertTrue(self.window.stop_requested)
        self.assertFalse(self.window.run_button.isEnabled())

    def test_backend_error_never_enables_run(self):
        self.window.start_action('validate')
        self.window.consume_event({'event': 'error', 'message': 'Preflight failed'})
        self.window.consume_event({'event': 'finished', 'action': 'validate', 'returncode': 1})
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertIn('Preflight failed', self.window.message.text())

    def test_nonfinite_telemetry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            PlotData().accept_sample({'stage': 'TEST', 'sim_time': float('nan')})


if __name__ == '__main__':
    unittest.main()
