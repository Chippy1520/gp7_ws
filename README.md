# GP7 kinematic pick-and-place coursework demo

A local Ubuntu 24.04 application using ROS 2 Jazzy, Gazebo Harmonic and a Python/Qt interface. The root `gp7_demo` package and the `./gui` / `./task.py` launchers wrap the existing core in `src/gp7_description/scripts`; they do not replace its kinematic methods, URDF frames or numerical values.

**Execution is `kinematic_playback`, physics OFF.** Checked joint trajectories are reconstructed on simulation ticks and written to joint/link state. Carrying the cube uses an explicit geometric attachment. This is a kinematic animation and collision/trajectory demonstration, **not** validation of motor torque, effort control, friction, contact forces, physical gripping, or hardware safety. Do not connect it to a real robot.

## Quick start

Run from the repository root on Ubuntu 24.04:

| What you want | Command |
| --- | --- |
| Install dependencies, prepare the workspace and build | `./setup.sh` |
| Open Gazebo without moving the arm | `./task.py --profile sprint --open-only` |
| Validate in Gazebo without moving the arm | `./task.py --profile sprint --validate` |
| Run the task with the Gazebo viewer | `./task.py --profile sprint` |
| Open the GP7 control GUI | `./gui` |

No manual ROS sourcing or virtual-environment activation is needed for these launchers. Add `--no-viewer` to the run or Validate command to run headless. The GUI opens without starting Gazebo; use its **Open Gazebo** button. If an owned scene from another profile is already open, add `--reset` to explicitly replace it.

## Local setup details

No public repository URL has been assigned. Use the complete local project folder, including its mesh assets; copying only Python files is insufficient.

Supported installation target: **Ubuntu 24.04 Noble, amd64 or arm64**, with the distribution's **`/usr/bin/python3` 3.12**. A local X11/Wayland desktop and working OpenGL are needed to view Qt and Gazebo. Setup can run headlessly and warns if no display is configured; it does not launch a GUI or scene. Missing displays or inaccessible X11/Wayland sessions must be fixed before `./gui` can run; setting a dummy `DISPLAY` is not a fix.

From the checkout directory, as your normal user:

```bash
./setup.sh
```

If executable bits were lost during an archive/file transfer, use `bash setup.sh` for installation and restore executable permissions on the launchers before using `./gui` / `./task.py`.

Setup requests **sudo** for system changes. It enables Ubuntu Universe, configures the official ROS apt source package and OSRF Gazebo Harmonic repository/key, installs ROS Jazzy desktop/ROS-Gazebo integration, Gazebo development/runtime and Python bindings, Qt, colcon/CMake/compiler tooling, URDF tools, and FCL/OctoMap/Eigen headers. It generates a UTF-8 locale for the installer process; it does not edit your shell startup files. Review `setup.sh` if you manage apt repositories centrally. Existing conflicting Gazebo Classic packages or inconsistent apt sources may require administrator attention; this installer does not silently remove them or perform a full OS upgrade. Keep Noble security/updates repositories enabled and the OS updated as recommended by ROS.

It then creates/reuses **`.venv-pick-place` with `--system-site-packages`**, installs the pinned requirements, generates the base pick-and-place world and Sprint pickup map if missing, builds `gp7_description` with testing enabled, and runs headless installation checks. The build uses one job by default to limit memory use. All workspace paths are resolved relative to the checkout, not a developer's home directory. Re-running setup repeats dependency checks/build/checks without appending duplicate repository entries. Build/install/verification directories are generated locally. The application launchers default to the pure-Python protobuf backend for Gazebo pose compatibility.

For an existing, already provisioned machine:

```bash
./setup.sh --skip-system
```

This skips **all apt/repository/sudo operations**, checks installed system dependencies, and still performs venv/pip/build/check steps. It is **not an offline option** and is not evidence that fresh-machine installation has been tested. Missing system dependencies are reported explicitly. It intentionally rejects unsupported OS versions even with `--skip-system`.

### Network and Python dependency notes

First installation needs access to Ubuntu/ROS/OSRF apt repositories, the official ROS apt-source release service/download, and Python package indexes. A package source build may also fetch build data. Offline installation requires dependencies already installed plus appropriate apt/pip caches or administrator-provided mirrors; no offline bundle is included. A network/repository failure is a setup failure, not a successful install.

`requirements.txt` pins NumPy **1.26.4**, SciPy **1.11.4**, Matplotlib **3.6.3**, python-fcl **0.7.0.11**, and PyYAML **6.0.1**. Noble's scientific Python packages are installed as system fallbacks; pip satisfies these exact requirements in the venv where necessary. Qt/PyQt5, rclpy, ROS messages, protobuf and `gz.msgs10` / `gz.transport13` bindings come from apt/ROS, **not** unrelated pip packages. Do not use Conda or a different Python ABI for this environment. Matplotlib 3.6.3 has no Python 3.12 wheel: if a source build is necessary, setup supplies native headers and NumPy first and disables pip build isolation to avoid an incompatible build-time NumPy. Compiler/Python packaging compatibility remains an external dependency; failures are reported rather than hidden. No `trimesh` dependency is needed: the collision code reads STL triangles itself and uses python-fcl.

Official upstream installation references:

- ROS Jazzy: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- Gazebo Harmonic: https://gazebosim.org/docs/harmonic/install_ubuntu/

## Use the GUI

```bash
./gui
```

The launcher handles the checkout environment and uses the desktop Qt backend (X11/XWayland), rather than inheriting a headless test setting. Opening it starts no simulation or motion.

1. Choose **fast**, **faster** or **sprint**. These are the original bounded timing presets.
2. **Open Gazebo** starts/reuses the owned scene and viewer without executing the sequence.
3. **Validate** checks the current measured cube/robot state without moving the arm. It can start the headless scene if needed.
4. **Run** is enabled after readiness for the selected profile. During motion it becomes **Stop**; Escape also requests a guarded stop.

In Sprint kinematic playback, place the upright, world-aligned cube on the floor with its center inside the opaque blue pre-grasp workspace. Validate checks placement and the gripper's height relative to the pedestal clearance plane. If below the plane, Run attempts a vertical rise first. Run generates new motion from live feedback, with joint and kinematic limits but **no collision sweeps or collision monitoring**. A blue cell does not certify the full path; planning can still fail at Run. This simplified behavior is for Gazebo only. Fast/faster retain the normal path checks.

Sprint holds only the readiness state between Validate and Run, not a trajectory. Moving the cube or arm invalidates readiness. The 25 mm workspace is generated offline from the configured model, limits and endpoint checks; regenerate after model changes with `.venv-pick-place/bin/python src/gp7_description/scripts/derive_grasp_workspace.py`.

Changing the timing profile invalidates readiness; selecting a profile that requires a different scene offers an explicit owned-scene restart confirmation. There is no force panel, PID tuning or physical-grasp mode. The graphs show six joint position/velocity/acceleration traces and the TCP path. Measured acceleration is explicitly an estimate, not a force measurement. No recorded/fabricated traces load on startup.

## Use the editable task script

```bash
./task.py --profile sprint
```

Unlike GUI validation, this entry point **executes** the task. Read and edit the root `task.py` timings and high-level sequence before running it: approach, grasp, carry, release and retreat are expressed at task level. Keep motion generation/kinematics in the shared implementation; do not replace it with new geometry or hard-coded joint samples. Allow the task to report rejected timing, unreachable poses or collision failures instead of bypassing its checks. Neither launcher authorizes physical grasp/torque validation.

## What installation verification means

To repeat the checks manually without reinstalling:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
.venv-pick-place/bin/python tools/check_installation.py
```

The checker verifies checkout assets and mesh references, exact Python versions/imports, ROS/Gazebo/Qt bindings, configuration/model loading, a native FCL distance calculation, plugin loading, and executes the **actual compiled binaries**:

- `build/gp7_description/gp7_kinematic_state_test`
- `build/gp7_description/gp7_playback_quintic_test`

It invalidates the old local gate before checking. Only if all checks succeed does it write a fresh `verification/pick_place/controller_gate.json` with `status: PASS` and `arm_actuation: kinematic_playback`. The report includes this checkout's paths, timestamp and binary SHA-256 hashes. It never copies an old computer's report. A failed check exits nonzero and leaves a FAIL gate; do not hand-edit it to PASS. Add `--no-write` to exercise the same checks without touching the gate (useful for diagnosis; it cannot certify a new installation). The gate is an **installation/compiled-regression prerequisite**, not proof that a Gazebo scene, real controller or physical grasp was validated.

CMake builds only the `gp7_kinematic_playback` plugin and retains `PlaybackQuintic` / `KinematicState` regressions plus relevant numerical and collision tests. It no longer builds the legacy custom PID or physical detachable-grasp libraries/test. Existing legacy files, if present, are not part of those library targets. Broader CTest checks can be run separately after building; default setup deliberately avoids live/UI tests.

## Model and asset attribution

The **university supplied the CAD/mesh assets** for this coursework. The **project author created the URDF and coordinate-frame definitions using SolidWorks**. The required STL files live in `src/gp7_description/meshes/` and are referenced by `urdf/gp7_with_gripper.urdf`; keep these with the project. See `ASSET_NOTICE.md`. No verified vendor restriction is asserted, and no blanket redistribution/open-source permission for supplied assets is implied. The model's frame/joint-zero and physical parameters retain their coursework limitations.
