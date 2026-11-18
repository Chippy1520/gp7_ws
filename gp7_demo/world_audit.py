"""Isolate Gazebo world queries from the executor’s pose subscriptions."""
from contextlib import contextmanager
import multiprocessing
import threading
import time


def _query_worker(connection):
    """No subscriptions or commands: isolate synchronous SDK RPC from pose callbacks."""
    from gz.transport13 import Node
    from gz.msgs10.sdf_generator_config_pb2 import SdfGeneratorConfig
    from gz.msgs10.stringmsg_pb2 import StringMsg
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.scene_pb2 import Scene
    node = Node()
    try:
        while connection.recv() is True:
            # A newly discovered service can return an empty/failed transport
            # reply. Retry only the read-only transport, never a rejected audit.
            deadline = time.monotonic() + 10
            def timeout_ms():
                return max(1, min(5000, int((deadline - time.monotonic()) * 1000)))
            for attempt in range(2):
                success, response = node.request('/world/gp7_lab/generate_world_sdf',
                                                SdfGeneratorConfig(), SdfGeneratorConfig, StringMsg, timeout_ms())
                scene_ok, scene = node.request('/world/gp7_lab/scene/info', Empty(), Empty, Scene, timeout_ms())
                complete = success and scene_ok and bool(response.data) and bool(scene.model)
                if complete or time.monotonic() >= deadline:
                    break
                if attempt == 0:
                    time.sleep(.05)
            connection.send((complete, response.SerializeToString(), scene.SerializeToString()))
    except EOFError:
        pass
    finally:
        connection.close()


@contextmanager
def world_query_client():
    context = multiprocessing.get_context('spawn')
    connection, child = context.Pipe()
    process = context.Process(target=_query_worker, args=(child,), daemon=True)
    process.start()
    child.close()

    def query():
        connection.send(True)
        if not connection.poll(11):
            raise RuntimeError('Live scene audit timed out. Validate again.')
        return connection.recv()

    try:
        yield query
    finally:
        try:
            connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
        # Reap the read-only helper without adding its SDK teardown to Run latency.
        threading.Thread(target=process.join, daemon=True).start()


def audit_world(config, *, query, allow_cube_reposition=False):
    """Run the unchanged audit against fresh replies from the isolated client."""
    from verify_pick_place_controllers import live_audit
    from gz.msgs10.stringmsg_pb2 import StringMsg
    from gz.msgs10.scene_pb2 import Scene
    success, payload, scene_payload = query()
    if not success:
        raise RuntimeError('Live scene audit request failed. Validate again.')
    return live_audit(config, allow_cube_reposition=allow_cube_reposition,
                      live_sdf_response=StringMsg.FromString(payload),
                      live_scene_response=Scene.FromString(scene_payload))
