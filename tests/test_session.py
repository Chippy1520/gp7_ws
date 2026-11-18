"""Mocked session tests: no Gazebo, shell commands, or process signals are run."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gp7_demo import session


SERVER = {'pid': 410, 'starttime': 1234, 'pgid': 410, 'sid': 410}
CHILD = {'pid': 411, 'starttime': 1235, 'pgid': 410, 'sid': 410}
VIEWER = {'pid': 510, 'starttime': 1300, 'pgid': 510, 'sid': 510}


def record(identity):
    return {'sid': identity['sid'], 'members': [identity.copy()], 'log': 'test.log'}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.root_patch = patch.object(session, 'ROOT', root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.client = session.Session()
        self.client.directory.mkdir()
        self.world = self.client.directory / 'world.sdf'
        self.world.write_text('<sdf><world name="gp7_test"/></sdf>')
        self.data = {'version': 1, 'profile': 'fast', 'world': str(self.world),
                     'server': record(SERVER), 'viewers': []}
        self.client._save(self.data)
        # Tests are fail-closed: accidental subprocesses/signals are forbidden.
        for name in ('Popen', 'run'):
            guard = patch.object(session.subprocess, name, side_effect=AssertionError('real subprocess forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(session.os, 'killpg', side_effect=AssertionError('real signal forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_pid_reuse_is_not_ownership_and_never_signalled(self):
        reused = dict(SERVER, starttime=9999)
        with patch.object(session, '_process', return_value=reused), \
                patch.object(session, '_processes', return_value=iter([reused])), \
                patch.object(session.os, 'killpg') as kill:
            self.assertEqual(session._members(record(SERVER)), [])
            self.client._stop(record(SERVER))
            kill.assert_not_called()

    def test_kernel_identity_includes_group_session_and_starttime(self):
        fields = ['S', '1', '410', '410'] + ['0'] * 15 + ['1234']
        with patch.object(Path, 'read_text', return_value='410 (name with ) spaces) ' + ' '.join(fields)):
            self.assertEqual(session._process(410), SERVER)

    def test_live_child_preserves_ownership_after_wrapper_exit(self):
        owned = {'sid': 410, 'members': [SERVER, CHILD]}
        with patch.object(session, '_process', side_effect=lambda pid: CHILD if pid == 411 else None), \
                patch.object(session, '_processes', return_value=iter([CHILD])):
            self.assertEqual(session._members(owned), [CHILD])

    def test_profile_mismatch_requires_explicit_reset(self):
        with patch.object(session, '_members', return_value=[SERVER]), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_launch') as launch, \
                patch.object(self.client, '_stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'profile mismatch'):
                self.client.ensure_scene('sprint')
            launch.assert_not_called()
            stop.assert_not_called()

    def test_external_server_refuses_reset_before_any_stop(self):
        with patch.object(session, '_members', return_value=[SERVER]), \
                patch.object(self.client, '_external_servers', return_value=[CHILD]), \
                patch.object(self.client, '_stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'External same-partition'):
                self.client.ensure_scene('fast', reset=True)
            stop.assert_not_called()

    def test_reused_scene_must_pass_readiness(self):
        with patch.object(session, '_members', return_value=[SERVER]), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_ready') as ready:
            self.assertEqual(self.client.ensure_scene('fast'), self.world)
            ready.assert_called_once_with(self.world)

    def test_visible_viewer_reused_without_launch_or_fake_ownership(self):
        with patch.object(session, '_members', return_value=[SERVER]), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_ready') as ready, \
                patch.object(session, '_gazebo_processes', return_value=[VIEWER]), \
                patch.object(session, '_visible', return_value=True), \
                patch.object(self.client, '_launch') as launch:
            self.client.open_viewer()
            ready.assert_called_once_with(self.world)
            launch.assert_not_called()
            self.assertEqual(self.client._load()['viewers'], [])

    def test_running_unmapped_viewer_blocks_duplicate(self):
        self.data['viewers'] = [record(VIEWER)]
        self.client._save(self.data)
        with patch.object(session, '_members', side_effect=lambda item: item['members'] if item else []), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_ready'), \
                patch.object(session, '_gazebo_processes', return_value=[]), \
                patch.object(session, '_visible', return_value=False), \
                patch.object(self.client, '_launch') as launch:
            with self.assertRaisesRegex(RuntimeError, 'no duplicate launched'):
                self.client.open_viewer()
            launch.assert_not_called()

    def test_failed_service_never_launches_viewer(self):
        with patch.object(session, '_members', return_value=[SERVER]), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_ready', side_effect=RuntimeError('/gazebo/worlds timed out')), \
                patch.object(self.client, '_launch') as launch:
            with self.assertRaisesRegex(RuntimeError, '/gazebo/worlds'):
                self.client.open_viewer()
            launch.assert_not_called()

    def test_readiness_requires_actual_matching_response_not_exit_status(self):
        replies = [subprocess.CompletedProcess([], 0, 'data: "wrong_world"'),
                   subprocess.CompletedProcess([], 0, 'data: "gp7_test"')]
        with patch.object(session, '_capture', side_effect=replies) as capture, \
                patch.object(session.time, 'sleep'):
            self.client._ready(self.world)
            self.assertEqual(capture.call_count, 2)
            self.assertIn('/gazebo/worlds', capture.call_args.args[0])
            self.assertLessEqual(capture.call_args.kwargs['timeout'], 2)

    def test_service_timeout_is_bounded_and_actionable(self):
        with patch.object(session.time, 'monotonic', side_effect=[0, 0, 0, 21, 21]), \
                patch.object(session.time, 'sleep'), \
                patch.object(session, '_capture', side_effect=subprocess.TimeoutExpired('gz', 2)):
            with self.assertRaisesRegex(RuntimeError, 'Viewer not launched.*demo_server.log'):
                self.client._ready(self.world)

    def test_launch_tracks_real_child_and_redirects_log_not_pipe(self):
        proc = MagicMock(pid=410)
        logfile = self.client.directory / 'viewer.log'
        with patch.object(session.subprocess, 'Popen', return_value=proc) as popen, \
                patch.object(session.time, 'monotonic', side_effect=[0, 0, .6]), \
                patch.object(session.time, 'sleep'), \
                patch.object(session, '_processes', side_effect=[iter([SERVER]), iter([CHILD])]):
            launched = self.client._launch(['gz', 'sim', '-g'], logfile)
            self.assertEqual(launched['members'], [CHILD])
            self.assertEqual(launched['sid'], 410)
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs['start_new_session'])
            self.assertNotEqual(kwargs['stdout'], subprocess.PIPE)
            self.assertEqual(kwargs['stdout'].name, str(logfile))
            self.assertEqual(kwargs['env']['ROS_DOMAIN_ID'], '77')
            self.assertEqual(kwargs['env']['GZ_PARTITION'], session.PARTITION)
            self.assertEqual(kwargs['env']['GZ_IP'], '127.0.0.1')
            proc.poll.assert_not_called()

    def test_new_scene_build_arguments_and_registry_saved_before_readiness(self):
        self.data['server'] = None
        self.client._save(self.data)
        def build(*args, **kwargs):
            command = args[0]
            # ros_command may wrap argv in a shell string/list. Verify its input below.
            (self.client.directory / 'demo_world_fast.sdf').write_text('<sdf><world name="gp7_test"/></sdf>')
            return subprocess.CompletedProcess(command, 0)
        def ready(world):
            saved = self.client._load()
            self.assertEqual(saved['server']['members'], [SERVER])
            self.assertEqual(saved['profile'], 'fast')
            self.assertEqual(saved['world'], str(world))
        with patch.object(session, '_members', return_value=[]), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(session, 'ros_command', side_effect=lambda argv: argv), \
                patch.object(session.subprocess, 'run', side_effect=build) as run, \
                patch.object(self.client, '_launch', return_value=record(SERVER)), \
                patch.object(self.client, '_ready', side_effect=ready):
            self.client.ensure_scene('fast')
            argv = run.call_args.args[0]
            self.assertIn('--detachable-grasp', argv)
            self.assertEqual(argv[argv.index('--grip-force') + 1], '8')
            self.assertEqual(argv[argv.index('--profile') + 1], 'fast')
            self.assertTrue(run.call_args.kwargs['start_new_session'])

    def test_reset_stops_only_recorded_viewers_then_server(self):
        self.data['viewers'] = [record(VIEWER)]
        self.client._save(self.data)
        with patch.object(session, '_members', side_effect=lambda item: item['members'] if item else []), \
                patch.object(self.client, '_external_servers', return_value=[]), \
                patch.object(self.client, '_stop') as stop, \
                patch.object(session.subprocess, 'run', side_effect=RuntimeError('mock build stop')):
            with self.assertRaisesRegex(RuntimeError, 'mock build stop'):
                self.client.ensure_scene('fast', reset=True)
            self.assertEqual([call.args[0]['sid'] for call in stop.call_args_list], [510, 410])
            self.assertIsNone(self.client._load()['server'])


if __name__ == '__main__':
    unittest.main()
