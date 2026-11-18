"""Minimal GP7 operator window. Opening it never launches simulation or motion."""
import argparse
import json
from pathlib import Path
import queue
import sys
import time

from PyQt5 import QtCore, QtWidgets

from .plotting import JointChart, PlotData, TCPChart

STYLE = '''
QWidget { background: #f3f5f9; color: #16243a;
          font-family: "DejaVu Sans"; font-size: 13px; }
QLabel { background: transparent; }
QLabel#title { font-size: 26px; font-weight: 600; }
QLabel#muted { color: #5b6b80; }
QLabel#heading { font-size: 16px; font-weight: 600; }
QFrame#card { background: white; border: 1px solid #dfe5ed; border-radius: 10px; }
QComboBox { background: white; border: 1px solid #bdc9d8;
            border-radius: 6px; padding: 8px 12px; min-height: 24px; }
QComboBox:focus { border: 2px solid #2563eb; }
QComboBox QAbstractItemView { background: white; selection-background-color: #dbeafe; }
QPushButton { background: white; color: #1d4ed8; border: 1px solid #2563eb;
              border-radius: 6px; padding: 10px 25px; min-height: 22px;
              font-weight: 600; }
QPushButton:hover { background: #eff6ff; }
QPushButton:focus { border: 2px solid #16243a; }
QPushButton#primary { background: #2563eb; color: white; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton:disabled { color: #8794a5; border-color: #cbd5e1; background: #f1f5f9; }
QPushButton#primary:disabled { background: #dbe3ef; color: #718096; border-color: #dbe3ef; }
'''


def label(text, role=None):
    widget = QtWidgets.QLabel(text)
    widget.setTextFormat(QtCore.Qt.PlainText)
    if role:
        widget.setObjectName(role)
    return widget


def card(title, subtitle):
    frame = QtWidgets.QFrame(objectName='card')
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(8)
    layout.addWidget(label(title, 'heading'))
    caption = label(subtitle, 'muted')
    caption.setWordWrap(True)
    layout.addWidget(caption)
    return frame, layout, caption


def read_report(event):
    """Read a backend-supplied report or the actual executor output directory."""
    supplied = event.get('report')
    if isinstance(supplied, dict):
        return supplied
    output = event.get('report_path') or event.get('output')
    if not output:
        return None
    path = Path(output)
    if not path.is_absolute():
        raise ValueError('Backend output path must be absolute')
    if path.is_file():
        paths = [path]
    else:
        paths = [path / name for name in ('result.json', 'prepare_result.json')]
    for candidate in paths:
        if candidate.is_file():
            report = json.loads(candidate.read_text())
            if not isinstance(report, dict):
                raise ValueError('Backend report must be a JSON object')
            return report
    return None


def report_error(report):
    if not report:
        return ''
    error = report.get('error') or report.get('errors')
    if error:
        return str(error)
    status = str(report.get('status', ''))
    if any(word in status.upper() for word in ('FAIL', 'ABORT', 'ERROR', 'STOPPED')):
        return str(report.get('message') or status)
    return ''


class DemoWindow(QtWidgets.QWidget):
    """UI-owned readiness gate; injectable backend for offline verification.

    backend=None is the inert screenshot/test mode. main() alone constructs the
    real Backend, and only a user's button press calls Backend.start().
    """

    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend
        self.data = PlotData()
        self.prepared_profile = None
        self.active_action = None
        self.active_profile = None
        self.ready_seen = False
        self.last_state = None
        self.action_error = ''
        self.stop_requested = False
        self.close_pending = False
        self.dirty = False
        self.last_draw = 0.
        self.setWindowTitle('GP7 Pick & Place')
        self.resize(1460, 900)
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.update_controls()

    @property
    def busy(self):
        return self.active_action is not None or bool(self.backend and self.backend.busy)

    def build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 18)
        layout.setSpacing(20)
        header = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QVBoxLayout()
        heading.setSpacing(6)
        heading.addWidget(label('GP7 Pick & Place', 'title'))
        heading.addWidget(label('Forward / inverse kinematics · Trajectories · Path planning', 'muted'))
        header.addLayout(heading)
        header.addStretch()
        header.addWidget(label('Geometric attachment · physics off', 'muted'))
        layout.addLayout(header)
        controls = QtWidgets.QFrame(objectName='card')
        row = QtWidgets.QHBoxLayout(controls)
        row.setContentsMargins(20, 16, 20, 16)
        row.setSpacing(14)
        row.addWidget(label('Timing profile'))
        self.profile = QtWidgets.QComboBox()
        self.profile.addItems(['fast', 'faster', 'sprint'])
        self.profile.setCurrentText(getattr(self.backend, 'selected_profile', 'fast'))
        self.profile.setMinimumWidth(160)
        self.profile.currentTextChanged.connect(self.profile_changed)
        row.addWidget(self.profile)
        self.open_button = QtWidgets.QPushButton('Open Gazebo')
        self.open_button.setMinimumWidth(160)
        self.open_button.setToolTip('Open or show Gazebo without running robot motion.')
        self.open_button.clicked.connect(lambda: self.start_action('open'))
        row.addWidget(self.open_button)
        row.addStretch()
        self.validate_button = QtWidgets.QPushButton('Validate')
        self.run_button = QtWidgets.QPushButton('Run', objectName='primary')
        self.validate_button.clicked.connect(self.validate)
        self.run_button.clicked.connect(self.run_or_stop)
        for button in (self.validate_button, self.run_button):
            button.setMinimumWidth(140)
            row.addWidget(button)
        layout.addWidget(controls)
        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(20)
        joints, column, self.joint_caption = card(
            'Joint trajectories', 'All six joints · position, velocity and acceleration')
        self.joint_chart = JointChart()
        column.addWidget(self.joint_chart, 1)
        tcp, column, self.tcp_caption = card('TCP path', 'Robot base frame · metres')
        self.tcp_chart = TCPChart()
        column.addWidget(self.tcp_chart, 1)
        plots.addWidget(joints, 3)
        plots.addWidget(tcp, 2)
        layout.addLayout(plots, 1)
        self.message = label('Idle · no plan or telemetry loaded. Validate before Run.', 'muted')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.escape = QtWidgets.QShortcut('Escape', self)
        self.escape.activated.connect(self.request_stop)

    def show_message(self, text, error=False):
        text = str(text)
        summary = text.splitlines()[-1] if text.splitlines() else ''
        self.message.setText(summary[:220] + ('…' if len(summary) > 220 else ''))
        self.message.setToolTip(text)
        self.message.setStyleSheet('color: #b91c1c;' if error else '')

    def profile_changed(self, _profile):
        if self.backend is not None:
            getattr(self.backend, 'discard_preparation', lambda: None)()
        self.prepared_profile = None
        self.data.clear()
        self.dirty = True
        self.show_message('Timing changed · validate this profile before Run.')
        self.update_controls()

    def update_controls(self):
        active = self.busy
        self.profile.setEnabled(not active)
        self.open_button.setEnabled(not active and self.backend is not None)
        self.validate_button.setEnabled(not active and self.backend is not None)
        self.run_button.setText('Stop' if active else 'Run')
        self.run_button.setEnabled(
            (active and not self.stop_requested) or
            (not active and self.backend is not None
             and self.prepared_profile == self.profile.currentText()))
        self.run_button.setToolTip(
            'Guarded stop for the owned actor only; no jaw release, home, or server stop. Escape also requests Stop.'
            if active else 'Requires successful validation for the selected timing profile.')

    def validate(self):
        self.start_action('validate')

    def run_or_stop(self):
        if self.busy:
            self.request_stop()
        else:
            self.start_action('run')

    def start_action(self, action, reset=False):
        if self.backend is None or self.busy:
            return
        profile = self.profile.currentText()
        if action == 'run' and self.prepared_profile != profile:
            self.show_message('Validate the selected timing profile before Run.', error=True)
            return
        if action not in ('open', 'validate', 'run'):
            raise ValueError('Unknown GUI action')
        needs_reset = getattr(self.backend, 'requires_reset', lambda _: self.backend.completed_cycle)
        profile_mismatch = getattr(self.backend, 'scene_profile', None) not in (None, profile)
        if not reset and ((action == 'validate' and needs_reset(profile))
                          or (action == 'open' and profile_mismatch)):
            answer = QtWidgets.QMessageBox.question(
                self, 'Restart owned demo scene?',
                'The selected timing profile requires a different scene configuration. '
                'Restart the owned scene at its original robot/cube poses?\n\n'
                'No leaves the scene unchanged and cancels this action.',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
            reset = True
        if action != 'open' or profile == 'sprint':
            self.prepared_profile = None
        self.active_action, self.active_profile = action, profile
        self.ready_seen = False
        self.last_state = None
        self.action_error = ''
        self.stop_requested = False
        if action == 'validate':
            self.data.clear()
            self.dirty = True
        elif action == 'run':
            self.data.clear_samples()
            self.dirty = True
        self.show_message({'open': 'Opening Gazebo…', 'validate': 'Validating…',
                           'run': 'Running…'}[action])
        self.update_controls()
        try:
            accepted = self.backend.start(action, profile, reset=reset)
            if accepted is False:
                raise RuntimeError('Backend did not accept the action; no readiness claimed.')
        except Exception as error:
            self.active_action = None
            self.prepared_profile = None
            self.show_message(str(error), error=True)
        self.update_controls()

    def request_stop(self):
        if not self.busy or self.backend is None or self.stop_requested:
            return False
        self.prepared_profile = None
        try:
            accepted = self.backend.stop()
            if accepted is False:
                self.show_message('No owned actor accepted Stop; waiting for the active operation.', error=True)
                return False
        except Exception as error:
            self.show_message(str(error), error=True)
            return False
        self.stop_requested = True
        self.show_message('Guarded Stop requested · waiting for actor exit; hold not yet verified.')
        self.update_controls()
        return True

    def consume_event(self, event):
        kind = event['event']
        if kind == 'plan':
            self.data.load_plan(event['path'])
            self.dirty = True
        elif kind == 'sample':
            sample = self.data.accept_sample(event)
            self.dirty = True
            if not self.action_error and not self.stop_requested:
                self.show_message(f'{sample["stage"]} · {sample["elapsed"]:.2f} / {sample["duration"]:.2f} s')
        elif kind == 'status':
            state = str(event.get('state', ''))
            self.last_state = state
            if state == 'ready' and self.active_action == 'validate':
                self.ready_seen = True
            if state in ('aborted', 'error', 'failed'):
                self.action_error = str(event.get('message') or state)
                self.prepared_profile = None
            if not self.stop_requested:
                self.show_message(self.action_error or event.get('message') or state,
                                  error=bool(self.action_error))
        elif kind == 'error':
            self.action_error = str(event.get('message') or 'Backend error')
            self.prepared_profile = None
            self.show_message(self.action_error, error=True)
        elif kind == 'finished':
            self.finish_action(event)
        else:
            raise ValueError('Unknown backend event: ' + str(kind))

    def finish_action(self, event):
        action = event.get('action')
        if action != self.active_action:
            raise ValueError('Finished event does not match the active action')
        profile = self.active_profile
        code = event.get('returncode')
        error = self.action_error
        report = None
        try:
            report = read_report(event)
            error = report_error(report) or error
        except (OSError, ValueError, TypeError) as problem:
            error = f'Report unavailable: {problem}'
        if event.get('scene_changed'):
            self.prepared_profile = None
            if action == 'open':
                self.data.clear()
        success = code == 0 and not error and not self.stop_requested
        if action == 'validate':
            success = success and self.ready_seen and self.last_state == 'ready'
            self.prepared_profile = profile if success and profile == self.profile.currentText() else None
        elif action == 'run':
            self.prepared_profile = None
        if error:
            self.show_message(error, error=True)
        elif self.stop_requested:
            self.show_message(f'Actor exited ({code}) after Stop · no jaw release, home, or server stop requested.')
        elif not success:
            detail = ' · no final ready evidence' if action == 'validate' and code == 0 else ''
            self.show_message(f'{str(action).capitalize()} did not pass · exit {code}{detail}', error=True)
        elif action == 'validate':
            preflight = (report or {}).get('workspace_preflight', {})
            if preflight.get('needs_rise'):
                self.show_message(f'Validated · {profile} · arm will rise first.')
            else:
                self.show_message(f'Validated · {profile} · ready to Run.')
        elif action == 'open':
            self.show_message('Gazebo open request finished · no motion requested.')
        elif report and report.get('status'):
            self.show_message(f'Report: {report["status"]} · last received traces (not live).')
        else:
            self.show_message('Run exited · result report unavailable; cycle success not verified.', error=True)
        self.active_action = None
        self.active_profile = None
        self.dirty = True
        self.update_controls()

    def refresh(self):
        if self.backend is not None:
            for _ in range(300):
                try:
                    event = self.backend.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.consume_event(event)
                except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
                    self.action_error = f'Rejected backend event: {error}'
                    self.prepared_profile = None
                    self.show_message(self.action_error, error=True)
        self.update_controls()
        now = time.monotonic()
        if self.dirty and (not self.busy or now - self.last_draw >= .4):
            parts, recorded = self.data.segments()
            self.joint_chart.update_data(parts, recorded)
            self.tcp_chart.update_data(parts, recorded)
            self.joint_caption.setText(
                'Six joints · faint reference / solid measured · acceleration estimated' if recorded
                else 'All six joints · position, velocity and acceleration · nominal plan')
            self.joint_chart.setToolTip(
                'Simulation timeline preserves repeated stages and blank gaps. Measured acceleration: '
                'finite difference of velocity, dt in (0.00001, 0.5] s, exponential filter τ = 0.12 s; '
                'no derivative across stage changes or timestamp resets. '
                'Nominal timeline excludes execution holds; sample timeline starts at the first sampled stage.')
            self.tcp_caption.setText('Robot base frame · metres' + (' · last received samples' if recorded else ''))
            self.dirty = False
            self.last_draw = now
        if self.close_pending and not self.busy:
            self.close_pending = False
            self.close()

    def closeEvent(self, event):
        if not self.busy:
            if self.backend is not None:
                getattr(self.backend, 'discard_preparation', lambda: None)()
            event.accept()
            return
        event.ignore()
        if self.close_pending:
            return
        answer = QtWidgets.QMessageBox.question(
            self, 'Operation still active',
            'Request guarded Stop and close only after the owned actor exits?\n'
            'No jaw release, home, scene reset, or server shutdown will be sent.',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            self.close_pending = True
            self.request_stop()


# Descriptive alias for embedders; main() remains the root launcher's entry point.
MainWindow = DemoWindow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idle-screenshot', type=Path,
                        help='Save the actual empty GUI and exit without constructing a backend')
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('GP7 Pick & Place')
    app.setStyle('Fusion')
    backend = None
    if not args.idle_screenshot:
        from .backend import Backend
        backend = Backend()
    window = DemoWindow(backend)
    window.show()
    if args.idle_screenshot:
        def capture():
            try:
                args.idle_screenshot.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(args.idle_screenshot)):
                    raise OSError('Could not save idle screenshot')
                print('Saved actual idle GUI:', args.idle_screenshot, flush=True)
            except OSError as error:
                print(str(error), file=sys.stderr)
                app.exit(1)
                return
            app.quit()
        QtCore.QTimer.singleShot(800, capture)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
