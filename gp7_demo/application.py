"""One execution path for the terminal task and the graphical interface."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
from functools import partial
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import uuid

from . import TIMING_PROFILES
from .bridge import ensure_bridge
from .runtime import PYTHON, ROOT, environment, ros_command
from .session import Session

SELECTION = ROOT / "verification/demo_selection.json"


def read_selection():
    try:
        selection = json.loads(SELECTION.read_text())
        return selection if isinstance(selection, dict) else {}
    except (OSError, ValueError):
        return {}


def write_selection(selection):
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    temporary = SELECTION.with_name("selection-" + uuid.uuid4().hex + ".json")
    try:
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(SELECTION)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def task_lock():
    path = ROOT / "verification/demo_task.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another task owns this simulation. Stop or finish it first.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def output_directory():
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return ROOT / "verification/demo_runs" / name


def argument_parser():
    parser = argparse.ArgumentParser(description="Run the editable kinematic coursework sequence.")
    parser.add_argument("--validate", action="store_true", help="Preflight and plot; no motion commands.")
    parser.add_argument("--profile", choices=TIMING_PROFILES, help="Select an existing timing preset.")
    parser.add_argument("--reset", action="store_true", help="Explicitly restart this application's owned scene.")
    parser.add_argument("--open-only", action="store_true", help="Open Gazebo without starting a task.")
    parser.add_argument("--no-viewer", action="store_true", help="Keep the server headless.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_actor", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_actor(sequence, options, timings, geometry):
    """Adapt presentation options; retain every existing backend execution gate."""
    actor_options = SimpleNamespace(
        execute=not options.validate,
        detachable_grasp=True,
        stop_after_lift=False,
        use_live_cube=True,
        profile=options.profile,
        output=options.output,
    )
    from .world_audit import audit_world, world_query_client
    # Start discovery while the actor imports its ROS/geometry dependencies.
    with world_query_client() as query:
        from pick_place_motion import run as execute
        execute(sequence, timings=timings, options=actor_options,
                use_profile_timings=False,
                world_audit=partial(audit_world, query=query),
                **geometry)
    result = json.loads((options.output / "result.json").read_text())
    expected = 'READY_' if options.validate else 'PASS'
    return 0 if result['status'].startswith(expected) else 1


def wait_for_actor(options):
    """Forward Stop only to the actor, allowing its existing guarded cleanup."""
    actor = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if actor is not None and actor.poll() is None:
            actor.send_signal(signal.SIGINT)

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        if stop_requested:
            return 130
        arguments = [PYTHON, ROOT / "task.py", "--_actor", "--profile", options.profile,
                     "--output", options.output]
        if options.validate:
            arguments.append("--validate")
        actor = subprocess.Popen(ros_command(arguments), cwd=ROOT, env=environment())
        if stop_requested:
            actor.send_signal(signal.SIGINT)
        return actor.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


def run_owned_session(options):
    """Scene creation/restart is separate from validation and robot commands."""
    with task_lock():
        session = Session()
        world = session.ensure_scene(options.profile, reset=options.reset)
        registry = json.loads(session.registry.read_text())
        session_id = registry['server']['sid']
        selection = read_selection()
        if selection.get("session_id") != session_id or options.reset:
            selection = {"world": str(world), "session_id": session_id, "cycle_started": False}
        selection["profile"] = options.profile
        write_selection(selection)
        if options.open_only or not options.no_viewer:
            session.open_viewer()
        if options.open_only:
            return 0
        ensure_bridge(session)
        if not options.validate:
            selection["cycle_started"] = True
            write_selection(selection)
        return wait_for_actor(options)


def run(sequence, *, timings, profile, pickup_height, hover_clearance, retreat_distance):
    """Use the same user-edited task in the GUI and terminal, with no tuning here."""
    options = argument_parser().parse_args()
    options.profile = options.profile or profile
    if options.profile not in TIMING_PROFILES:
        raise ValueError("Unknown timing profile: " + options.profile)
    # Edits in task.py apply to its selected profile in both entry points.
    selected_timings = dict(timings if options.profile == profile
                            else TIMING_PROFILES[options.profile]["timings"])
    options.output = (options.output or output_directory()).resolve()
    geometry = dict(pickup_height=pickup_height, hover_clearance=hover_clearance,
                    retreat_distance=retreat_distance)
    try:
        if options._actor:
            return run_actor(sequence, options, selected_timings, geometry)
        if not PYTHON.is_file():
            raise RuntimeError("The project environment is missing. Run ./setup.sh first.")
        return run_owned_session(options)
    except KeyboardInterrupt:
        print("Stopped before execution or after guarded actor cleanup.", flush=True)
        return 130
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr, flush=True)
        return 1
