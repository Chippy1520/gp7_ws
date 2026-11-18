"""Small public interface; the existing numerical implementation stays private."""
import sys

from .runtime import SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pick_place_profiles import PROFILES as TIMING_PROFILES


def run(sequence, **settings):
    """Execute an editable sequence through the shared, guarded demo application."""
    from .application import run as run_application

    return run_application(sequence, **settings)
