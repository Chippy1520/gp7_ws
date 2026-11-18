"""Checkout-relative paths and the environment for the isolated simulation."""
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src/gp7_description/scripts"
PYTHON = ROOT / ".venv-pick-place/bin/python"
ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")


def environment():
    """Keep all coursework processes in the existing isolated ROS/Gazebo namespace."""
    env = os.environ.copy()
    # Gazebo message decoding can fail with the default native protobuf backend.
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    env.update(
        ROS_DOMAIN_ID="77",
        GZ_PARTITION="gp7_project_lab",
        GZ_IP="127.0.0.1",
        PYTHONUNBUFFERED="1",
    )
    return env


def ros_command(arguments):
    """Source installed dependencies without relying on the caller's terminal."""
    if not ROS_SETUP.is_file():
        raise RuntimeError("ROS Jazzy is missing. Run ./setup.sh first.")
    setup = "source " + shlex.quote(str(ROS_SETUP))
    installed = ROOT / "install/setup.bash"
    if installed.is_file():
        setup += "; source " + shlex.quote(str(installed))
    command = 'set -eo pipefail; ' + setup + '; exec "$@"'
    return ["bash", "-c", command, "gp7-demo", *map(str, arguments)]
