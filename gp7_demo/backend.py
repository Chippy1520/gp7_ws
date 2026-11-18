"""Qt adapter: each action invokes the same task.py used by the terminal."""
from collections import deque
import json
import queue
import signal
import subprocess
import threading

from .application import output_directory, read_selection
from .runtime import PYTHON, ROOT, environment, ros_command

PREFIX = 'GP7_MONITOR '


class Backend:
    def __init__(self):
        from task import PROFILE
        self.events = queue.Queue()
        self.selected_profile = read_selection().get('profile', PROFILE)
        self.busy = False
        self.prepared_profile = None
        self.process = None
        self._validated_scene = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def requires_reset(self, profile):
        return self.scene_profile not in (None, profile)

    @property
    def scene_profile(self):
        return read_selection().get('profile')

    def discard_preparation(self):
        with self._lock:
            self.prepared_profile = None
            self._validated_scene = None

    def start(self, action, profile, reset=False):
        if action not in {'open', 'validate', 'run'}:
            raise ValueError('Unknown action: ' + action)
        with self._lock:
            if self.busy:
                raise RuntimeError('A task is already active.')
            if action == 'run' and (self.prepared_profile != profile or reset or
                                    self._validated_scene != read_selection().get('session_id')):
                raise RuntimeError('Validate this timing profile and scene before Run.')
            if action != 'open' or reset:
                self.prepared_profile = None
                self._validated_scene = None
            self.busy = True
            self._stop_requested = False
            threading.Thread(target=self._work, args=(action, profile, reset), daemon=True).start()

    def stop(self):
        with self._lock:
            if not self.busy:
                return False
            self._stop_requested = True
            self.prepared_profile = None
            self._validated_scene = None
            if self.process is not None and self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
        return True

    def _arguments(self, action, profile, output, reset):
        arguments = [PYTHON, ROOT / 'task.py', '--profile', profile, '--output', output]
        arguments.append('--open-only' if action == 'open' else '--no-viewer')
        if action == 'validate':
            arguments.append('--validate')
        if reset:
            arguments.append('--reset')
        return ros_command(arguments)

    def _read_output(self, process, log, tail):
        for line in process.stdout:
            log.write(line)
            tail.append(line.strip())
            if not line.startswith(PREFIX):
                continue
            event = json.loads(line[len(PREFIX):])
            if not isinstance(event, dict) or 'event' not in event:
                raise ValueError('Invalid backend record')
            self.events.put(event)

    def _work(self, action, profile, reset):
        output = output_directory()
        code, error, result = 130, None, {}
        before = read_selection().get('session_id')
        tail = deque(maxlen=8)
        try:
            output.mkdir(parents=True)
            with (output / 'process.log').open('w') as log:
                with self._lock:
                    if self._stop_requested:
                        return
                    process = subprocess.Popen(self._arguments(action, profile, output, reset),
                        cwd=ROOT, env=environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace', start_new_session=True)
                    self.process = process
                self._read_output(process, log, tail)
                code = process.wait()
            report = output / 'result.json'
            if report.is_file():
                result = json.loads(report.read_text())
            if code == 0 and action != 'open':
                prefix = 'PASS' if action == 'run' else 'READY_'
                if not str(result.get('status', '')).startswith(prefix):
                    raise RuntimeError('Task exited without the required result evidence.')
            if code:
                error = result.get('error') or '\n'.join(tail) or 'Task did not complete.'
        except Exception as exception:
            error, code = str(exception), 1
            if 'process' in locals() and process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait()
        finally:
            selection = read_selection()
            scene = selection.get('session_id')
            scene_changed = reset or scene != before
            with self._lock:
                if action == 'validate' and code == 0 and not self._stop_requested:
                    self.prepared_profile = profile
                    self._validated_scene = scene
                elif action == 'open' and scene_changed:
                    self.prepared_profile = None
                    self._validated_scene = None
                self.busy = False
                self.process = None
                self.selected_profile = selection.get('profile', profile)
            if error:
                self.events.put({'event': 'error', 'message': error})
            self.events.put({'event': 'finished', 'action': action, 'returncode': code,
                             'output': str(output), 'result': result, 'scene_changed': scene_changed})
