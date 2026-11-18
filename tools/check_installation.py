#!/usr/bin/env python3
"""Headless installation checks, not a live controller/physical validation.

Run with the checkout venv after sourcing ROS Jazzy and install/setup.bash.
--no-write exercises the same checks but never replaces the local gate.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src/gp7_description"
GATE = ROOT / "verification/pick_place/controller_gate.json"
PINS = {"numpy": "1.26.4", "scipy": "1.11.4", "matplotlib": "3.6.3",
        "python-fcl": "0.7.0.11", "PyYAML": "6.0.1"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(report):
    GATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = GATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(GATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="do not write/invalidate the gate")
    args = parser.parse_args()
    report = {"status": "FAIL", "arm_actuation": "kinematic_playback",
              "scope": "headless installation and compiled kinematic regressions; not live physics/controller validation",
              "physics": "OFF", "root": str(ROOT), "python": sys.executable,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    # A failed/incomplete check must not leave an old machine's PASS usable.
    if not args.no_write:
        write_report(report)

    def check(name, action):
        try:
            result = action()
        except Exception as exc:
            report["checks"].append({"name": name, "status": "FAIL", "error": str(exc)})
            print(f"FAIL {name}: {exc}", flush=True)
            return
        report["checks"].append({"name": name, "status": "PASS", "detail": result})
        print(f"PASS {name}: {result}", flush=True)

    def environment():
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Use Ubuntu Noble /usr/bin/python3 (3.12) through .venv-pick-place")
        if Path(sys.prefix).resolve() != (ROOT / ".venv-pick-place").resolve():
            raise RuntimeError("Run with .venv-pick-place/bin/python")
        if os.environ.get("ROS_DISTRO") != "jazzy":
            raise RuntimeError("Source /opt/ros/jazzy/setup.bash first")
        for command in ("gz", "ros2", "colcon", "check_urdf"):
            if not shutil.which(command):
                raise RuntimeError(f"Missing command: {command}")
        return "CPython 3.12 checkout venv, ROS Jazzy and required commands"

    def assets():
        required = [ROOT / "gui", ROOT / "task.py", ROOT / "gp7_demo/__init__.py",
                    PKG / "config/pick_place.yaml", PKG / "worlds/gp7_bringup.sdf",
                    PKG / "worlds/gp7_pick_place.sdf",
                    PKG / "worlds/pick_place_assets/box_a.sdf", PKG / "worlds/pick_place_assets/box_b.sdf",
                    PKG / "src/PlaybackQuintic.hh", PKG / "src/KinematicState.hh"]
        robot = ET.parse(PKG / "urdf/gp7_with_gripper.urdf").getroot()
        for mesh in robot.findall(".//mesh"):
            uri = mesh.get("filename", "")
            prefix = "package://gp7_description/"
            if uri.startswith(prefix):
                required.append(PKG / uri[len(prefix):])
            elif uri.startswith("file://"):
                required.append(Path(uri[7:]))
            elif "://" not in uri:
                required.append(PKG / "urdf" / uri)
            else:
                raise RuntimeError(f"Unsupported mesh URI: {uri}")
        missing = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                   for p in required if not p.is_file() or not p.stat().st_size]
        if missing:
            raise RuntimeError("Missing/empty checkout assets: " + ", ".join(missing))
        for launcher in (ROOT / "gui", ROOT / "task.py"):
            if not os.access(launcher, os.X_OK):
                raise RuntimeError(f"Launcher is not executable: {launcher}")
        from pick_place_collision import CollisionChecker
        geometry = CollisionChecker(cube_only=False)
        return {"files": len(required), "CAD_meshes_loaded_from_checkout": len(geometry.meshes),
                "environment_models": sorted(geometry.environment)}

    def versions():
        found = {name: importlib.metadata.version(name) for name in PINS}
        if found != PINS:
            raise RuntimeError(f"Dependency pin mismatch: expected {PINS}, found {found}")
        # ROS environment hooks can put system site-packages before the venv.
        # Check loaded code too, not just potentially shadowed dist-info records.
        for distribution, module in (("numpy", "numpy"), ("scipy", "scipy"),
                                     ("matplotlib", "matplotlib"), ("PyYAML", "yaml")):
            loaded = importlib.import_module(module).__version__
            if loaded != PINS[distribution]:
                raise RuntimeError(f"{module} is shadowed: loaded {loaded}, expected {PINS[distribution]}")
        return found

    sys.path[:0] = [str(ROOT), str(PKG / "scripts")]
    check("environment", environment)
    check("checkout_assets", assets)
    check("pinned_python_dependencies", versions)
    modules = ["numpy", "scipy.interpolate", "scipy.spatial.transform", "matplotlib",
               "matplotlib.backends.backend_qt5agg", "yaml", "fcl", "PyQt5.QtWidgets",
               "rclpy", "ament_index_python.packages", "launch_ros", "sensor_msgs.msg",
               "std_msgs.msg", "std_srvs.srv", "geometry_msgs.msg", "trajectory_msgs.msg",
               "rosgraph_msgs.msg", "google.protobuf.text_format", "gz.transport13",
               "gz.msgs10.joint_trajectory_pb2", "gz.msgs10.pose_v_pb2",
               "gz.msgs10.boolean_pb2", "gz.msgs10.scene_pb2", "gz.msgs10.contacts_pb2"]
    for module in modules:
        check("import " + module, lambda name=module: str(importlib.import_module(name).__file__))

    for module in ("gp7_demo", "pick_place_config", "kinematics", "joint_trajectories",
                   "pick_place_planner", "pick_place_collision", "pick_place_playback", "pick_place_motion"):
        check("core import " + module, lambda name=module: str(importlib.import_module(name).__file__))

    def numerical_smoke():
        import numpy as np
        import fcl
        from scipy.interpolate import BPoly
        from pick_place_config import load_config, activate_kinematic_playback
        from kinematics import ExportedModel
        config = activate_kinematic_playback(load_config())
        if config["execution"]["arm_actuation"] != "kinematic_playback":
            raise RuntimeError("Kinematic configuration activation failed")
        ExportedModel()
        value = BPoly.from_derivatives([0., 1.], [[0., 0., 0.], [1., 0., 0.]])(.5)
        if not np.isclose(value, .5):
            raise RuntimeError("Quintic interpolation smoke failed")
        first = fcl.CollisionObject(fcl.Box(1., 1., 1.))
        second = fcl.CollisionObject(fcl.Box(1., 1., 1.), fcl.Transform(np.array([3., 0., 0.])))
        distance = float(fcl.distance(first, second, fcl.DistanceRequest(), fcl.DistanceResult()))
        if not np.isclose(distance, 2.):
            raise RuntimeError(f"FCL native distance smoke failed: {distance}")
        return "URDF model/config loaded; quintic midpoint=0.5; native FCL box distance=2.0"

    check("numerical_and_native_fcl_smoke", numerical_smoke)

    def plugin():
        path = ROOT / "install/gp7_description/lib/libgp7_kinematic_playback.so"
        if not path.is_file():
            raise RuntimeError(f"Missing built playback plugin: {path}")
        ctypes.CDLL(str(path))
        return {"path": str(path), "sha256": digest(path)}

    check("compiled_playback_plugin_load", plugin)
    for name in ("gp7_kinematic_state_test", "gp7_playback_quintic_test"):
        def run_test(name=name):
            executable = ROOT / "build/gp7_description" / name
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise RuntimeError(f"Missing compiled regression: {executable}; run setup.sh")
            result = subprocess.run([str(executable)], cwd=ROOT, text=True,
                                    capture_output=True, timeout=60, check=False)
            output = (result.stdout + result.stderr).strip()
            if result.returncode:
                raise RuntimeError(f"exit={result.returncode}: {output}")
            return {"executable": str(executable), "sha256": digest(executable),
                    "returncode": result.returncode, "output": output}
        check(name, run_test)

    if all(item["status"] == "PASS" for item in report["checks"]):
        report["status"] = "PASS"
    if not args.no_write:
        write_report(report)
    print(f"{report['status']}: {len(report['checks'])} checks; "
          + ("--no-write: gate untouched" if args.no_write else str(GATE)))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
