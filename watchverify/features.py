"""Windowed activity descriptor, shared by training and runtime.

The activity model sees a summary of recent movement rather than a single frame. Training
and serving must build that summary identically, so both import from here. A model trained
on whole clips could never be fed by the streaming app; windows are what the app can
actually produce, so windows are what the model is trained on.
"""
from collections import deque

import numpy as np

from .core import FEATURE_NAMES

WINDOW_S = 5.0
STRIDE_S = 1.0
MAX_GAP_S = 0.5
STAT_NAMES = ('mean', 'std', 'p10', 'p50', 'p90')
FRAME_COLUMNS = list(FEATURE_NAMES) + [n + '_valid' for n in FEATURE_NAMES]
MOTION_COLUMNS = ('hip_speed_abs_mean', 'hip_speed_abs_max', 'angular_speed_abs_mean',
                  'angular_speed_abs_max', 'down_fraction', 'upright_fraction', 'quality_mean')
CLIP_COLUMNS = [f'{c}_{s}' for s in STAT_NAMES for c in FRAME_COLUMNS] + list(MOTION_COLUMNS)


def clip_descriptor(x, hip_speed, angular_speed, down, upright, quality):
    """Summarise a run of per-frame observations into one fixed-length vector."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FRAME_COLUMNS) or not len(x):
        raise ValueError(f'Expected (n, {len(FRAME_COLUMNS)}) frame features, got {x.shape}')
    hip_speed, angular_speed = np.asarray(hip_speed, float), np.asarray(angular_speed, float)
    stats = [x.mean(axis=0), x.std(axis=0),
             np.percentile(x, 10, axis=0), np.percentile(x, 50, axis=0), np.percentile(x, 90, axis=0)]
    motion = np.array([np.abs(hip_speed).mean(), np.abs(hip_speed).max(),
                       np.abs(angular_speed).mean(), np.abs(angular_speed).max(),
                       np.asarray(down, float).mean(), np.asarray(upright, float).mean(),
                       np.asarray(quality, float).mean()])
    return np.concatenate(stats + [motion])


def windows(t, x, hip_speed, angular_speed, down, upright, quality,
            window_s=WINDOW_S, stride_s=STRIDE_S):
    """Fixed-length descriptors over a sequence, matching what the app emits live."""
    t = np.asarray(t, dtype=float)
    if not len(t):
        return []
    result = []
    start = float(t[0])
    while start + window_s <= float(t[-1]) + 1e-9:
        inside = (t >= start) & (t < start + window_s)
        if inside.sum() >= 3:
            result.append((start, start + window_s,
                           clip_descriptor(x[inside], hip_speed[inside], angular_speed[inside],
                                           down[inside], upright[inside], quality[inside])))
        start += stride_s
    return result


class WindowAggregator:
    """Per-person rolling window. Emits a descriptor only from a full, ungapped window.

    An incomplete window is not scored: an activity judgement built from two seconds of
    observation is not the thing the model was trained on, and guessing would be worse
    than abstaining.
    """
    def __init__(self, window_s=WINDOW_S, stride_s=STRIDE_S, max_gap_s=MAX_GAP_S):
        self.window_s, self.stride_s, self.max_gap_s = window_s, stride_s, max_gap_s
        self._history = {}
        self._last_emit = {}

    def reset(self, track_id):
        self._history.pop(track_id, None)
        self._last_emit.pop(track_id, None)

    def update(self, track_id, t, vector, mask, hip_speed, angular_speed, down, upright, quality):
        t = float(t)
        history = self._history.setdefault(track_id, deque())
        if history and t - history[-1][0] > self.max_gap_s:
            history.clear()  # A gap breaks the window; do not average across missing time.
            self._last_emit.pop(track_id, None)
        history.append((t, np.concatenate([np.asarray(vector, float), np.asarray(mask, float)]),
                        float(hip_speed), float(angular_speed), bool(down), bool(upright), float(quality)))
        while history and t - history[0][0] > self.window_s:
            history.popleft()
        if len(history) < 3 or t - history[0][0] < self.window_s - self.stride_s:
            return None
        last = self._last_emit.get(track_id)
        if last is not None and t - last < self.stride_s:
            return None
        self._last_emit[track_id] = t
        rows = list(history)
        return clip_descriptor(np.array([r[1] for r in rows]),
                               np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                               np.array([r[4] for r in rows]), np.array([r[5] for r in rows]),
                               np.array([r[6] for r in rows]))
