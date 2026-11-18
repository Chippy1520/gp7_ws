"""Persistent, conservatively owned Gazebo sessions (no motion/Stop actions)."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from gp7_demo.runtime import ROOT, SCRIPTS, environment, ros_command, PYTHON


PARTITION = 'gp7_project_lab'
SERVICE_TIMEOUT = 20.0
VIEWER_TIMEOUT = 12.0


def _environment():
    env = environment().copy()
    env.update(ROS_DOMAIN_ID='77', GZ_PARTITION=PARTITION, GZ_IP='127.0.0.1')
    return env


def _process(pid):
    """Read kernel identity, not just kill(pid, 0); zombies are not services."""
    try:
        base = Path('/proc') / str(pid)
        fields = base.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'starttime': int(fields[19]),
                'pgid': int(fields[2]), 'sid': int(fields[3])}
    except (OSError, ValueError, IndexError):
        return None


def _alive(identity):
    return _process(identity.get('pid', -1)) == identity


def _processes():
    for path in Path('/proc').glob('[0-9]*'):
        identity = _process(path.name)
        if identity is not None:
            yield identity


def _members(record):
    """A live recorded identity is required before discovering more descendants."""
    if not record or not any(_alive(item) for item in record.get('members', [])):
        return []
    return [item for item in _processes() if item['sid'] == record['sid']]


def _gazebo_processes(kind):
    result = []
    for identity in _processes():
        base = Path('/proc') / str(identity['pid'])
        try:
            env = base.joinpath('environ').read_bytes().split(b'\0')
            if ('GZ_PARTITION=' + PARTITION).encode() not in env:
                continue
            args = base.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            # Unreadable processes cannot be certified compatible or owned.
            continue
        gazebo = bool(re.search(r'\bgz\s+sim\b|gz-sim|gz sim', args))
        viewer = bool(re.search(r'(^|\s)(-g|--gui-only)(\s|$)|gz-sim-gui', args))
        if gazebo and (viewer if kind == 'viewer' else not viewer):
            result.append(identity)
    return result


def _capture(args, timeout=3):
    return subprocess.run(ros_command(args), cwd=str(ROOT), env=_environment(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout, start_new_session=True)


def _visible(pids):
    """Require a mapped client window, not merely a still-running CLI wrapper."""
    if not os.environ.get('DISPLAY'):
        return False
    try:
        clients = _capture(['xprop', '-root', '_NET_CLIENT_LIST'])
        for window in re.findall(r'0x[0-9a-fA-F]+', clients.stdout):
            props = _capture(['xprop', '-id', window, '_NET_WM_PID'])
            match = re.search(r'=\s*(\d+)', props.stdout)
            if match and int(match.group(1)) in pids:
                state = _capture(['xwininfo', '-id', window])
                if state.returncode == 0 and 'Map State: IsViewable' in state.stdout:
                    return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


class Session:
    def __init__(self):
        self.directory = Path(ROOT) / 'verification'
        self.registry = self.directory / 'demo_session.json'

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.registry.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _load(self):
        try:
            data = json.loads(self.registry.read_text())
            if not isinstance(data, dict) or data.get('version') != 1:
                raise ValueError('unsupported registry')
            return data
        except FileNotFoundError:
            return {'version': 1, 'viewers': []}
        except (ValueError, TypeError) as error:
            raise RuntimeError(f'Invalid session registry {self.registry}; inspect it before resetting.') from error

    def _save(self, data):
        temporary = self.registry.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(self.registry)

    def _refresh(self, data):
        for record in [data.get('server'), *data.get('viewers', [])]:
            if record:
                record['members'] = _members(record)
        self._save(data)

    def _external_servers(self, data):
        owned = {item['pid'] for item in _members(data.get('server'))}
        return [item for item in _gazebo_processes('server') if item['pid'] not in owned]

    def _launch(self, args, logfile):
        env = _environment()
        if args[:2] == ['gz', 'sim'] and '-g' in args:
            # A visible Ogre/GLX window cannot use an inherited headless Qt backend.
            env['QT_QPA_PLATFORM'] = 'xcb'
        with logfile.open('ab', buffering=0) as output:
            proc = subprocess.Popen(ros_command(args), cwd=str(ROOT), env=env,
                                    stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        # The wrapper may already have exited while its real GUI/server child lives.
        # During this launch only, the freshly allocated session is our provenance.
        started = time.monotonic()
        deadline = started + 2
        while True:
            members = [item for item in _processes() if item['sid'] == proc.pid]
            now = time.monotonic()
            # Give the CLI time to fork, and persist the child as well as the
            # wrapper. Returning the first shell PID loses ownership when that
            # shell exits before the next API call.
            if members and now - started >= .5:
                return {'sid': proc.pid, 'members': members, 'log': str(logfile)}
            if now >= deadline:
                # Never persist stale samples from a session that disappeared.
                raise RuntimeError(f'Gazebo exited during launch; inspect {logfile}')
            time.sleep(.05)

    def _stop(self, record):
        if not record:
            return
        for sig, duration in [(signal.SIGINT, 3), (signal.SIGTERM, 5), (signal.SIGKILL, 2)]:
            members = _members(record)
            if not members:
                return
            record['members'] = members
            # Groups are signalled only after validating a live recorded identity.
            for pgid in {item['pgid'] for item in members}:
                witnesses = [item for item in members if item['pgid'] == pgid]
                if any(_alive(item) for item in witnesses):
                    try:
                        os.killpg(pgid, sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if not _members(record):
                    return
                time.sleep(.1)
        if _members(record):
            raise RuntimeError('Owned Gazebo process did not exit; inspect session logs before retrying.')

    def _ready(self, world):
        try:
            expected = ET.parse(world).find('world').get('name')
        except (OSError, ET.ParseError, AttributeError) as error:
            raise RuntimeError(f'Cannot read world name from {world}') from error
        deadline = time.monotonic() + SERVICE_TIMEOUT
        last = 'no response'
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = _capture(['gz', 'service', '-s', '/gazebo/worlds',
                                     '--reqtype', 'gz.msgs.Empty', '--reptype', 'gz.msgs.StringMsg_V',
                                     '--timeout', '1000', '--req', ''], timeout=min(2, remaining))
                last = response.stdout.strip()[-500:]
                if response.returncode == 0 and expected in re.findall(r'data:\s*"([^"\n]+)"', response.stdout):
                    return
            except (OSError, subprocess.TimeoutExpired) as error:
                last = str(error)
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        raise RuntimeError(f'/gazebo/worlds did not return world {expected!r} within {SERVICE_TIMEOUT:g}s. '
                           f'Viewer not launched. Inspect {self.directory / "demo_server.log"}; '
                           f'check GZ_PARTITION={PARTITION}, GZ_IP=127.0.0.1 and server health. Last response: {last}')

    def scene_compatible(self, profile):
        with self._locked():
            data = self._load()
            return (data.get('profile') == profile and bool(_members(data.get('server')))
                    and Path(data.get('world', '')).is_file() and not self._external_servers(data))

    def ensure_scene(self, profile, reset=False):
        if profile not in ('fast', 'faster', 'sprint'):
            raise ValueError(f'Unknown scene profile: {profile!r}')
        with self._locked():
            data = self._load()
            self._refresh(data)
            if self._external_servers(data):
                raise RuntimeError('External same-partition Gazebo server detected; compatibility is not provable. '
                                   'No server/viewer was stopped. Close it explicitly or use another partition.')
            if _members(data.get('server')) and not reset:
                if data.get('profile') != profile:
                    raise RuntimeError('Scene profile mismatch; explicitly request reset=True to replace the owned scene.')
                world = Path(data['world'])
                self._ready(world)
                return world
            if reset:
                for viewer in data.get('viewers', []):
                    self._stop(viewer)
                self._stop(data.get('server'))
                data.update(server=None, viewers=[])
                self._save(data)
            world = self.directory / f'demo_world_{profile}.sdf'
            with (self.directory / 'demo_build.log').open('ab') as output:
                subprocess.run(ros_command([str(PYTHON), str(Path(SCRIPTS) / 'build_pick_place_world.py'),
                                            '--profile', profile, '--grip-force', '8', '--detachable-grasp',
                                            '--output', str(world)]), cwd=str(ROOT), env=_environment(),
                               stdout=output, stderr=subprocess.STDOUT, check=True, timeout=60,
                               start_new_session=True)
            if not world.is_file():
                raise RuntimeError(f'Builder did not produce {world}; inspect demo_build.log')
            # Recheck after the build; never knowingly launch alongside an external server.
            if self._external_servers(data):
                raise RuntimeError('External same-partition server appeared during build; launch refused.')
            data.update(profile=profile, world=str(world),
                        server=self._launch(['gz', 'sim', '-s', '-r', str(world)],
                                            self.directory / 'demo_server.log'))
            self._save(data)
            self._ready(world)
            return world

    def open_viewer(self):
        with self._locked():
            data = self._load()
            self._refresh(data)
            if not _members(data.get('server')) or self._external_servers(data):
                raise RuntimeError('No unambiguous owned scene; call ensure_scene(profile) before opening a viewer.')
            self._ready(Path(data['world']))
            existing = _gazebo_processes('viewer')
            owned = [item for viewer in data.get('viewers', []) for item in _members(viewer)]
            candidates = existing + owned
            if _visible({item['pid'] for item in candidates}):
                return
            if candidates:
                raise RuntimeError('A same-partition viewer is running but no mapped window was found. '
                                   'Check DISPLAY/window manager or close that viewer; no duplicate launched.')
            record = self._launch(['gz', 'sim', '-g'], self.directory / 'demo_viewer.log')
            data.setdefault('viewers', []).append(record)
            self._save(data)
            deadline = time.monotonic() + VIEWER_TIMEOUT
            while time.monotonic() < deadline:
                record['members'] = _members(record)
                self._save(data)
                if _visible({item['pid'] for item in record['members']}):
                    return
                if not record['members']:
                    break
                time.sleep(.2)
            raise RuntimeError('Viewer did not map a visible window. Inspect demo_viewer.log and DISPLAY; '
                               'any live owned child is recorded, so retry will not launch a duplicate.')
