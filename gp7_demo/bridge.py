"""Start/reuse the standard ROS/Gazebo bridge required by existing feedback gates."""
import hashlib
from pathlib import Path

from .runtime import ROOT
from .session import PARTITION, _members, _processes

CONFIG = ROOT / "src/gp7_description/config/ros_gz_bridge.yaml"


def matching_bridges():
    """Inspect only actual bridge executables in this task's isolated namespace."""
    bridges = []
    for identity in _processes():
        proc = Path("/proc") / str(identity["pid"])
        try:
            arguments = proc.joinpath("cmdline").read_bytes().split(b"\0")
            if Path(arguments[0].decode()).name != "parameter_bridge":
                continue
            env = proc.joinpath("environ").read_bytes().split(b"\0")
            if b"ROS_DOMAIN_ID=77" not in env or ("GZ_PARTITION=" + PARTITION).encode() not in env:
                continue
            bridges.append([value.decode() for value in arguments if value])
        except (OSError, UnicodeError):
            continue
    return bridges


def compatible(arguments):
    if "__node:=gp7_sim_bridge" not in arguments:
        return False
    paths = [value.split(":=", 1)[1] for value in arguments if value.startswith("config_file:=")]
    if len(paths) != 1:
        return False
    config = Path(paths[0])
    if config.name != CONFIG.name:
        return False
    try:
        return hashlib.sha256(config.read_bytes()).digest() == hashlib.sha256(CONFIG.read_bytes()).digest()
    except OSError:
        return False


def ensure_bridge(session):
    """Never replace/kill an external bridge or silently add a competing publisher."""
    with session._locked():
        data = session._load()
        if _members(data.get("bridge")):
            return
        existing = matching_bridges()
        if existing:
            if len(existing) != 1 or not compatible(existing[0]):
                raise RuntimeError("An incompatible ROS/Gazebo bridge already uses domain 77. "
                                   "Close it explicitly; no competing bridge was started.")
            return
        arguments = ["ros2", "run", "ros_gz_bridge", "parameter_bridge", "--ros-args",
                     "-r", "__node:=gp7_sim_bridge", "-p", "config_file:=" + str(CONFIG)]
        data["bridge"] = session._launch(arguments, session.directory / "demo_bridge.log")
        session._save(data)
