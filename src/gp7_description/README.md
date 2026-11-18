# GP7 model and numerical implementation

See the workspace-root `README.md` for setup, the approved minimal GUI and the editable `task.py` entry point.

- `urdf/` and `meshes/`: the author-created SolidWorks frame/URDF definitions and university-supplied CAD mesh assets.
- `config/`: preserved task limits, CAD frame transforms, planning references and standard ROS/Gazebo bridge mappings.
- `scripts/`: shared FK/IK, polynomial and differential trajectories, collision/path validation, guarded execution and their regression tests. These are implementation modules, not additional user-facing applications.
- `worlds/`: the preserved scene geometry inputs used to generate a checkout-relative runtime world.
- `src/`: only the kinematic Gazebo playback adapter and its state/quintic helpers are built. No custom PID or physical-grasp library is built.

## Sampled pickup area

The Sprint kinematic world renders cells from `config/grasp_workspace_sprint.json`
whose four corners and center passed the offline pre-grasp endpoint checks at
zero yaw. This sampling orientation is not a live alignment requirement: Sprint
accepts a cube resting flat at any yaw and plans grasp alignment from its measured
pose. The overlay does not snap the cube or certify a continuous or full path.
The builder omits the area if the survey fingerprint no longer matches.

Regenerate the survey from the workspace root (no Gazebo or robot motion):

```sh
OPENBLAS_NUM_THREADS=1 .venv-pick-place/bin/python src/gp7_description/scripts/derive_grasp_workspace.py
```

The selected demo runs with physics OFF and geometric cube attachment/support. It is not torque, friction, grasp-force or hardware validation. No kinematic geometry, motion algorithms, original timing presets or numerical limits were retuned for the GUI refactor. `tests/fixtures/numerical_baseline.json` at the workspace root records the pre-refactor calculations used for regression comparison.
