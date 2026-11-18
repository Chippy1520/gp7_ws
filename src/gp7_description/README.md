# GP7 model and numerical implementation

See the workspace-root `README.md` for setup, the approved minimal GUI and the editable `task.py` entry point.

- `urdf/` and `meshes/`: the author-created SolidWorks frame/URDF definitions and university-supplied CAD mesh assets.
- `config/`: preserved task limits, CAD frame transforms, planning references and standard ROS/Gazebo bridge mappings.
- `scripts/`: shared FK/IK, polynomial and differential trajectories, collision/path validation, guarded execution and their regression tests. These are implementation modules, not additional user-facing applications.
- `worlds/`: the preserved scene geometry inputs used to generate a checkout-relative runtime world.
- `src/`: only the kinematic Gazebo playback adapter and its state/quintic helpers are built. No custom PID or physical-grasp library is built.

## Sampled pickup area

The Sprint kinematic world renders the cells in `config/pickup_area_sprint.json`
whose four corners and center all passed the offline pickup-to-retreat planner.
The survey uses a floor-level, world-aligned cube and checks the home and nominal
post-retreat robot starts. The overlay is visual only: no support surface, no
snapping, and no permission to bypass live validation. Unsurveyed locations are
not necessarily unreachable, and a sampled cell does not certify its interior.
The builder omits the area if the survey fingerprint no longer matches.

Regenerate the survey from the workspace root (no Gazebo or robot motion):

```sh
PYTHONPATH=src/gp7_description/scripts OPENBLAS_NUM_THREADS=1 \
  .venv-pick-place/bin/python -m probe_pickup_area
```

The selected demo runs with physics OFF and geometric cube attachment/support. It is not torque, friction, grasp-force or hardware validation. No kinematic geometry, motion algorithms, original timing presets or numerical limits were retuned for the GUI refactor. `tests/fixtures/numerical_baseline.json` at the workspace root records the pre-refactor calculations used for regression comparison.
