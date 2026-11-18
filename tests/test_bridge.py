"""Offline bridge lifecycle checks; no real nodes/processes are launched."""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from gp7_demo import bridge


class BridgeTests(unittest.TestCase):
    def session(self):
        session = MagicMock()
        session.directory = Path('/tmp')
        session._load.return_value = {'version': 1, 'viewers': []}
        return session

    def test_live_owned_bridge_is_reused(self):
        session = self.session()
        with patch.object(bridge, '_members', return_value=[{'pid': 1}]):
            bridge.ensure_bridge(session)
        session._launch.assert_not_called()

    def test_compatible_external_bridge_is_not_adopted_or_duplicated(self):
        session = self.session()
        arguments = ['parameter_bridge', '__node:=gp7_sim_bridge',
                     'config_file:=' + str(bridge.CONFIG)]
        with patch.object(bridge, '_members', return_value=[]), \
                patch.object(bridge, 'matching_bridges', return_value=[arguments]):
            bridge.ensure_bridge(session)
        session._launch.assert_not_called()
        session._save.assert_not_called()

    def test_incompatible_external_bridge_is_not_replaced(self):
        session = self.session()
        with patch.object(bridge, '_members', return_value=[]), \
                patch.object(bridge, 'matching_bridges', return_value=[['parameter_bridge']]):
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                bridge.ensure_bridge(session)
        session._launch.assert_not_called()

    def test_missing_bridge_launches_standard_ros_gz_process_and_records_ownership(self):
        session = self.session()
        with patch.object(bridge, '_members', return_value=[]), \
                patch.object(bridge, 'matching_bridges', return_value=[]):
            bridge.ensure_bridge(session)
        command = session._launch.call_args.args[0]
        self.assertEqual(command[:4], ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge'])
        self.assertIn('__node:=gp7_sim_bridge', command)
        session._save.assert_called_once()
        self.assertIn('bridge', session._save.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
