"""Read-only GP7 plots; no ROS imports, recordings, or generated telemetry.

The derivative and whole-cycle segmentation rules intentionally match
pick_place_monitor_gui.py. Acceleration from measurements is an estimate.
"""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

COLORS = ['#2563eb', '#d97706', '#059669', '#7c3aed', '#dc2626', '#0891b2']
JOINTS = ['S', 'L', 'U', 'R', 'B', 'T']
LINE_STYLES = ['-', '--', '-.', ':', '-', '--']


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected a numeric value')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite telemetry')
    return value


def vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'Expected {length} values')
    return np.array([number(v) if v is not None else np.nan for v in value])


def cycle_segments(stages, history):
    """Old monitor timeline: sim-clock origin, repeated stages, no gap bridges."""
    samples = sorted((s for values in history.values() for s in values),
                     key=lambda s: s['stamp'])
    parts = []
    if samples:
        origin = samples[0]['stamp'] - samples[0]['elapsed']
        groups = []
        for sample in samples:
            if (not groups or sample['stage'] != groups[-1][-1]['stage']
                    or sample['elapsed'] < groups[-1][-1]['elapsed']):
                groups.append([])
            groups[-1].append(sample)
        for group in groups:
            first = group[0]
            t = np.array([s['stamp'] - origin for s in group])
            start = max(first['stamp'] - first['elapsed'] - origin,
                        parts[-1]['end'] if parts else 0.)
            parts.append(dict(
                name=first['stage'], start=start, end=max(start + 1e-6, t[-1]),
                time=t,
                reference=[np.array([s['refs'][i] for s in group]) for i in range(3)],
                measured=[np.array([s[key] for s in group]) for key in ('q', 'qd', 'qdd')],
                tcp=np.array([s['tcp'] for s in group])))
    else:
        start = 0.
        for name, plan in stages.items():
            end = start + plan['duration']
            parts.append(dict(name=name, start=start, end=end,
                              time=start + plan['time'],
                              reference=[plan.get(key) for key in ('q', 'qd', 'qdd')],
                              measured=None, tcp=plan.get('tcp')))
            start = end
    return parts, bool(samples)


class PlotData:
    """Validated plan/sample storage, independent of the GUI and backend."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.stages = {}
        self.clear_samples()

    def clear_samples(self):
        self.history = {}
        self.previous = None
        self.acceleration = np.full(8, np.nan)
        self.latest = None

    def load_plan(self, path):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('Plan path must be absolute')
        document = json.loads(path.read_text())
        stages = {}
        for item in document['stages']:
            name = str(item['name'])
            t = np.array([number(v) for v in item['time']])
            duration = number(item['duration'])
            if len(t) < 1 or np.any(np.diff(t) < 0) or duration < 0:
                raise ValueError('Invalid plan timeline')
            parsed = {'time': t, 'duration': duration}
            for key, width in [('q', 6), ('qd', 6), ('qdd', 6), ('tcp', 3)]:
                rows = item.get(key)
                if rows is None:
                    parsed[key] = None  # Do not derive nominal qdd from coarse samples.
                else:
                    if len(rows) != len(t):
                        raise ValueError('Plan array length mismatch')
                    parsed[key] = np.array([vector(row, width) for row in rows])
            stages[name] = parsed
        self.stages = stages

    def accept_sample(self, event):
        name = str(event['stage'])
        stamp = number(event['sim_time'])
        elapsed, duration = number(event['elapsed']), number(event['duration'])
        measured = event.get('measured') or {}
        reference = event.get('reference') or {}
        q = vector(measured.get('q', [None] * 8), 8)
        qd = vector(measured.get('qd', [None] * 8), 8)
        refs = [vector(reference.get(key, [None] * 8), 8)
                for key in ('q', 'qd', 'qdd')]
        tcp = vector(event.get('tcp') or [None] * 3, 3)
        previous = self.previous
        reset = previous is not None and stamp < previous[0]
        new_stage = previous is None or name != previous[2] or elapsed < previous[3]
        if reset:
            self.history.clear()
            self.previous = None
            self.acceleration[:] = np.nan
        acceleration = np.full(8, np.nan)
        if self.previous is not None:
            dt = stamp - self.previous[0]
            if 1e-5 < dt <= 0.5 and not new_stage:
                raw = (qd - self.previous[1]) / dt
                alpha = 1.0 - math.exp(-dt / 0.12)
                acceleration = np.where(
                    np.isfinite(self.acceleration),
                    self.acceleration + alpha * (raw - self.acceleration), raw)
        # Exactly as in the old monitor: duplicates have no new derivative and
        # do not replace the previous velocity/filter state.
        if self.previous is None or stamp != self.previous[0]:
            self.previous = (stamp, qd.copy(), name, elapsed)
            self.acceleration = acceleration
        sample = dict(stage=name, stamp=stamp, elapsed=elapsed, duration=duration,
                      q=q, qd=qd, qdd=acceleration, refs=refs, tcp=tcp)
        self.latest = sample
        self.history.setdefault(name, deque(maxlen=6000)).append(sample)
        return sample

    def segments(self):
        return cycle_segments(self.stages, self.history)


class JointChart(FigureCanvasQTAgg):
    def __init__(self):
        figure = Figure(facecolor='white')
        self.axes = figure.subplots(3, 1, sharex=True)
        super().__init__(figure)
        figure.subplots_adjust(left=.11, right=.97, top=.92, bottom=.09, hspace=.29)
        self.update_data([], False)

    def update_data(self, parts, recorded):
        titles = ['Position (rad)', 'Velocity (rad/s)', 'Acceleration (rad/s²)']
        for quantity, (axis, title) in enumerate(zip(self.axes, titles)):
            axis.clear()
            for part in parts:
                ref = part['reference'][quantity]
                for joint, (color, style) in enumerate(zip(COLORS, LINE_STYLES)):
                    if ref is not None and joint < ref.shape[1]:
                        axis.plot(part['time'], ref[:, joint], color=color,
                                  linestyle=style, linewidth=1.5, alpha=.65 if recorded else 1.)
                    if part['measured'] is not None:
                        values = part['measured'][quantity]
                        axis.plot(part['time'], values[:, joint], color=color,
                                  linestyle='-', linewidth=1.1,
                                  marker='.', markersize=2, markevery=8)
            axis.set_ylabel(title, fontsize=9, color='#475569', labelpad=9)
            axis.grid(color='#e6ebf2', linewidth=.7)
            axis.tick_params(labelsize=8, colors='#475569')
            for side in ('top', 'right'):
                axis.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                axis.spines[side].set_color('#cbd5e1')
            axis.margins(x=.015)
            if not parts:
                axis.set_xlim(0, 1)
                axis.set_ylim(-1, 1)
        handles = [Line2D([], [], color=color, linestyle=style, label=joint)
                   for joint, color, style in zip(JOINTS, COLORS, LINE_STYLES)]
        self.axes[0].legend(handles=handles, ncol=6, loc='upper center',
                            bbox_to_anchor=(.5, 1.23), frameon=False,
                            fontsize=9, handlelength=2.3)
        self.axes[-1].set_xlabel(
            'Run simulation time (s) · first sampled stage starts at zero' if recorded
            else 'Nominal motion time (s) · execution holds not included',
            fontsize=9, color='#475569')
        self.draw_idle()


class TCPChart(FigureCanvasQTAgg):
    def __init__(self):
        figure = Figure(facecolor='white')
        self.axis = figure.add_subplot(111, projection='3d')
        self.axis.view_init(elev=25, azim=-54)
        super().__init__(figure)
        figure.subplots_adjust(left=.02, right=.89, bottom=.08, top=.96)
        self.update_data([], False)

    def update_data(self, parts, recorded):
        axis = self.axis
        elevation, azimuth = axis.elev, axis.azim
        axis.clear()
        points = []
        for part in parts:
            xyz = part['tcp']
            if xyz is not None and len(xyz):
                axis.plot(*xyz.T, color='#d97706' if recorded else '#2563eb',
                          linewidth=2, label='Measured TCP' if recorded else 'Nominal TCP')
                points.append(xyz)
        if points:
            values = np.concatenate(points)
            if np.all(np.isfinite(values[0])):
                axis.scatter(*values[0], color='#059669', s=40,
                             label='Start', depthshade=False)
            if np.all(np.isfinite(values[-1])):
                axis.scatter(*values[-1], color='#16243a', s=40, marker='D',
                             label='Latest' if recorded else 'End', depthshade=False)
            finite = values[np.all(np.isfinite(values), axis=1)]
            if len(finite):
                axis.set_box_aspect(np.maximum(np.ptp(finite, axis=0), .15))
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc='upper center',
                        bbox_to_anchor=(.5, .98), ncol=3, frameon=False, fontsize=8)
        else:
            axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel('X (m)', labelpad=8, fontsize=9)
        axis.set_ylabel('Y (m)', labelpad=8, fontsize=9)
        axis.set_zlabel('Z (m)', labelpad=8, fontsize=9)
        axis.tick_params(labelsize=8, colors='#475569')
        axis.view_init(elev=elevation, azim=azimuth)
        for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
            dimension.pane.fill = False
            dimension._axinfo['grid']['color'] = '#e6ebf2'
        self.draw_idle()
