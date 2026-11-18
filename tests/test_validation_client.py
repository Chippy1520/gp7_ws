"""Validation startup reuses one read-only query client, never replaces guards."""

import json
from pathlib import Path
import sys
import tempfile

from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gp7_demo import application


class ValidationClientTests(unittest.TestCase):
    def test_query_worker_retries_only_failed_read_only_transport(self):
        from gp7_demo import world_audit
        response = Mock(data='<sdf/>')
        response.SerializeToString.return_value = b'sdf'
        scene = Mock(model=[object()])
        scene.SerializeToString.return_value = b'scene'
        node = Mock()
        node.request.side_effect = [(False, response), (True, scene),
                                    (True, response), (True, scene)]
        connection = Mock()
        connection.recv.side_effect = [True, None]
        modules = {
            'gz.transport13': SimpleNamespace(Node=Mock(return_value=node)),
            'gz.msgs10.sdf_generator_config_pb2': SimpleNamespace(SdfGeneratorConfig=Mock()),
            'gz.msgs10.stringmsg_pb2': SimpleNamespace(StringMsg=Mock()),
            'gz.msgs10.empty_pb2': SimpleNamespace(Empty=Mock()),
            'gz.msgs10.scene_pb2': SimpleNamespace(Scene=Mock()),
        }
        with patch.dict(sys.modules, modules), patch.object(world_audit.time, 'sleep') as pause:
            world_audit._query_worker(connection)
        self.assertEqual(node.request.call_count, 4)
        connection.send.assert_called_once_with((True, b'sdf', b'scene'))
        connection.close.assert_called_once()
        pause.assert_called_once_with(.05)

    def test_rejected_world_audit_is_never_retried(self):
        from gp7_demo import world_audit
        query = Mock(return_value=(True, b'sdf', b'scene'))
        audit = Mock(side_effect=ValueError('Rejected world schema'))
        modules = {
            'verify_pick_place_controllers': SimpleNamespace(live_audit=audit),
            'gz.msgs10.stringmsg_pb2': SimpleNamespace(StringMsg=Mock()),
            'gz.msgs10.scene_pb2': SimpleNamespace(Scene=Mock()),
        }
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, 'Rejected world schema'):
            world_audit.audit_world({}, query=query)
        query.assert_called_once_with()
        self.assertEqual(audit.call_count, 1)

    def test_actor_uses_read_only_world_audit(self):
        from gp7_demo import world_audit
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            query = Mock()
            client = Mock()
            client.__enter__ = Mock(return_value=query)
            client.__exit__ = Mock(return_value=False)
            core = ModuleType('pick_place_motion')

            def execute(sequence, **kwargs):
                self.assertIn('world_audit', kwargs)
                kwargs['world_audit']({}, allow_cube_reposition=True)
                self.assertFalse(kwargs['options'].execute)
                client.__exit__.assert_not_called()
                # A unit fixture, not a simulator result.
                (output / 'result.json').write_text(json.dumps({'status': 'READY_APPROACH_AND_DESCENT'}))

            core.run = execute
            options = SimpleNamespace(validate=True, profile='sprint', output=output)
            with patch.dict(sys.modules, {'pick_place_motion': core}), \
                    patch.object(world_audit, 'world_query_client', return_value=client) as factory, \
                    patch.object(world_audit, 'audit_world') as audit:
                self.assertEqual(application.run_actor(lambda motion: None, options, {}, {}), 0)
                factory.assert_called_once_with()
                audit.assert_called_once_with({}, query=query, allow_cube_reposition=True)
            client.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
