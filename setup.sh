#!/usr/bin/env bash
# Idempotent Ubuntu Noble / ROS Jazzy / Gazebo Harmonic bootstrap.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "Setup failed at line %s. No installation PASS is implied.\n" "$LINENO" >&2' ERR
SKIP_SYSTEM=0
case "${1:-}" in
  --skip-system) SKIP_SYSTEM=1 ;;
  -h|--help) printf 'Usage: %s [--skip-system]\nInstalls system dependencies unless skipped, builds, then checks without launching a scene.\n' "$0"; exit 0 ;;
  '') ;;
  *) fail "Unknown option: $1 (use --help)" ;;
esac
[[ $# -le 1 ]] || fail 'Only one option is accepted.'
[[ -r /etc/os-release ]] || fail 'Requires Ubuntu 24.04 (Noble).'
# shellcheck disable=SC1091
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || fail "Unsupported OS: ${PRETTY_NAME:-unknown}; requires Ubuntu 24.04."
[[ "$(dpkg --print-architecture)" =~ ^(amd64|arm64)$ ]] || fail 'Official binary setup supports amd64 or arm64 only.'
[[ $EUID -ne 0 ]] || fail 'Run as your normal desktop user, not root; sudo is used only for system packages.'
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf 'WARNING: no desktop display detected. Setup checks are headless; ./gui requires a local X11/Wayland desktop.\n' >&2
fi
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
SYSTEM_PACKAGES=(
  build-essential cmake pkg-config python3-dev python3-venv python3-pip
  python3-colcon-common-extensions python3-setuptools python3-wheel
  python3-pyqt5 python3-tk python3-protobuf python3-pybind11
  python3-numpy python3-scipy python3-matplotlib python3-yaml
  libfcl-dev liboctomap-dev libeigen3-dev libfreetype-dev libpng-dev libqhull-dev
  liburdfdom-tools gz-harmonic libgz-sim8-dev libgz-plugin2-dev libgz-math7-dev
  python3-gz-msgs10 python3-gz-transport13
  ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-robot-state-publisher
  ros-jazzy-joint-state-publisher-gui ros-jazzy-interactive-markers
  ros-jazzy-tf2-ros-py ros-jazzy-trajectory-msgs
)
if (( ! SKIP_SYSTEM )); then
  command -v sudo >/dev/null || fail 'sudo is required for system installation.'
  printf 'Installing official ROS/Gazebo apt sources and dependencies (sudo required).\n'
  sudo -v
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg locales software-properties-common
  sudo add-apt-repository -y universe
  sudo locale-gen en_US.UTF-8
  export LANG=en_US.UTF-8
  # Official ros2-apt-source package: discover the current release, never guess a version.
  if ! dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | /usr/bin/python3 -c 'import sys; sys.exit("install ok installed" not in sys.stdin.read())'; then
    TEMP_DIR="$(mktemp -d)"
    trap 'rm -rf -- "${TEMP_DIR:-}"' EXIT
    ROS_APT_SOURCE_VERSION="$(curl --fail --silent --show-error --location --retry 3 https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
    [[ "$ROS_APT_SOURCE_VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || fail 'Invalid ROS apt source release tag.'
    curl --fail --location --retry 3 -o "$TEMP_DIR/ros2-apt-source.deb" "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb"
    sudo dpkg -i "$TEMP_DIR/ros2-apt-source.deb"
  fi
  # Official OSRF Harmonic repository and signing key.
  curl --fail --silent --show-error --location --retry 3 https://packages.osrfoundation.org/gazebo.gpg | sudo tee /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg >/dev/null
  printf 'deb [arch=%s signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main\n' "$(dpkg --print-architecture)" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y "${SYSTEM_PACKAGES[@]}"
else
  printf 'Skipping ALL system changes. Existing apt/ROS/Gazebo dependencies must already be usable.\n'
fi
MISSING=()
for package in "${SYSTEM_PACKAGES[@]}"; do
  status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)"
  [[ "$status" == 'install ok installed' ]] || MISSING+=("$package")
done
if [[ ${#MISSING[@]} -ne 0 ]]; then
  if (( SKIP_SYSTEM )); then
    printf 'Some distro package names are absent (%s); checking actual installed capabilities through the build/import checks below.\n' "${MISSING[*]}"
  else
    fail "Missing system dependencies: ${MISSING[*]}. Re-run installation or install these explicitly."
  fi
fi
[[ -r /opt/ros/jazzy/setup.bash ]] || fail 'ROS Jazzy setup.bash is missing.'
# ROS environment hooks may read unset variables.
set +u
source /opt/ros/jazzy/setup.bash
set -u
[[ "${ROS_DISTRO:-}" == jazzy ]] || fail 'ROS environment is not Jazzy.'
command -v colcon >/dev/null || fail 'colcon is unavailable.'
gz sim --versions
VENV="$ROOT/.venv-pick-place"
/usr/bin/python3 -m venv --system-site-packages "$VENV"
# Matplotlib 3.6.3 has no CPython 3.12 wheel. Its source build needs NumPy
# available at build time; avoid isolation choosing an incompatible NumPy 2.x.
"$VENV/bin/python" -m pip install 'setuptools<70' wheel 'Cython<4' pybind11
"$VENV/bin/python" -m pip install 'numpy==1.26.4'
"$VENV/bin/python" -m pip install --no-build-isolation -r "$ROOT/requirements.txt"
PKG="$ROOT/src/gp7_description"
if [[ ! -s "$PKG/worlds/gp7_pick_place.sdf" ]]; then
  "$VENV/bin/python" "$PKG/scripts/build_pick_place_world.py" --output "$PKG/worlds/gp7_pick_place.sdf"
fi
if [[ ! -s "$PKG/config/grasp_workspace_sprint.json" ]]; then
  printf 'Generating the Sprint pickup map (bounded offline calculation).\n'
  (ulimit -v 1048576; ulimit -c 0; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
   timeout --kill-after=5s 180s "$VENV/bin/python" "$PKG/scripts/derive_grasp_workspace.py")
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
colcon build --base-paths "$ROOT/src" --executor sequential --packages-select gp7_description --symlink-install --cmake-args \
  -DBUILD_TESTING=ON -DPython3_EXECUTABLE="$VENV/bin/python" -DPYTHON_EXECUTABLE=/usr/bin/python3
set +u
source "$ROOT/install/setup.bash"
set -u
"$VENV/bin/python" "$ROOT/tools/check_installation.py"
printf '\nSetup checks passed. No GUI/scene was started.\nOpen Gazebo: ./task.py --profile sprint --open-only\nRun task:    ./task.py --profile sprint\nOpen GP7 UI: ./gui\n'
