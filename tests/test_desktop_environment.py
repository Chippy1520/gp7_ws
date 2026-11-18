"""Viewer launch must not inherit the headless Qt backend used by tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gp7_demo import session


class DesktopEnvironmentTests(unittest.TestCase):
    def launch_environment(self, arguments):
        identity = {'pid': 991, 'starttime': 1, 'pgid': 991, 'sid': 991}
        process = Mock(pid=991)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'QT_QPA_PLATFORM': 'offscreen'}), \
                    patch.object(session.subprocess, 'Popen', return_value=process) as spawn, \
                    patch.object(session, '_processes', return_value=[identity]), \
                    patch.object(session.time, 'monotonic', side_effect=[0., .6]):
                session.Session()._launch(arguments, Path(directory) / 'process.log')
        return spawn.call_args.kwargs['env']

    def test_visible_viewer_uses_xcb(self):
        env = self.launch_environment(['gz', 'sim', '-g'])
        self.assertEqual(env['QT_QPA_PLATFORM'], 'xcb')

    def test_headless_server_environment_is_not_rewritten(self):
        env = self.launch_environment(['gz', 'sim', '-s', '-r', 'world.sdf'])
        self.assertEqual(env['QT_QPA_PLATFORM'], 'offscreen')


if __name__ == '__main__':
    unittest.main()
