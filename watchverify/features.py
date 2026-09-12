"""Windowed activity descriptor, shared by training and runtime.

The activity model sees a summary of recent movement rather than a single frame. Training
and serving must build that summary identically, so both import from here. A model trained
on whole clips could never be fed by the streaming app; windows are what the app can
actually produce, so windows are what the model is trained on.

There is one builder. `WindowAggregator` owns the definition of a window and `windows()`
drives it over a finished recording, so the offline and runtime paths cannot drift apart
by editing one and forgetting the other. They differ in exactly one declared parameter:
`max_gap_s=None` offline, because the cached corpora were built before gap handling
existed and tightening them is a training change, not a serving change. Every window the
runtime emits is therefore one the offline builder would also have produced; the offline
builder is the more permissive of the two. That asymmetry is deliberate and is the reason
runtime output can be compared against offline-measured evidence at all.
"""
from collections import deque
from typing import NamedTuple

import numpy as np

from .core import FEATURE_NAMES

WINDOW_S = 5.0
STRIDE_S = 1.0
MAX_GAP_S = 0.5
# A window is a summary of an interval, not of whatever rows happen to fall in it. Three is
# the count below which the percentile columns stop meaning anything; it is not a coverage
# guarantee, and the runtime raises it (see worker.py) because the model was fitted on
# windows holding roughly fifty observations.
MIN_SAMPLES = 3
STAT_NAMES = ('mean', 'std', 'p10', 'p50', 'p90')
FRAME_COLUMNS = list(FEATURE_NAMES) + [n + '_valid' for n in FEATURE_NAMES]
MOTION_COLUMNS = ('hip_speed_abs_mean', 'hip_speed_abs_max', 'angular_speed_abs_mean',
                  'angular_speed_abs_max', 'down_fraction', 'upright_fraction', 'quality_mean')
CLIP_COLUMNS = [f'{c}_{s}' for s in STAT_NAMES for c in FRAME_COLUMNS] + list(MOTION_COLUMNS)


class Window(NamedTuple):
    """One emitted descriptor and the interval it summarises.

    `continuous` says whether the immediately preceding interval of this person also
    produced a descriptor. It is the only thing that distinguishes a run of evidence from
    two observations either side of a hole, and sustained-alert logic depends on it: a
    caller counting consecutive windows must treat `continuous=False` as the start of a
    new run, never as its continuation.
    """
    start: float
    end: float
    descriptor: np.ndarray
    continuous: bool


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


class WindowAggregator:
    """Per-person rolling window. Emits a descriptor only from a full, ungapped interval.

    An incomplete window is not scored: an activity judgement built from two seconds of
    observation is not the thing the model was trained on, and guessing would be worse
    than abstaining.

    Intervals are half-open `[start, start + window_s)`, anchored at the first observation
    after a reset and advanced by `stride_s`. An interval closes when an observation at or
    past its end arrives — one sample past the boundary is enough to know it is complete,
    so this stays causal — which means the first descriptor appears at a full `window_s`
    of observed span, not at `window_s - stride_s`.
    """
    def __init__(self, window_s=WINDOW_S, stride_s=STRIDE_S, max_gap_s=MAX_GAP_S,
                 min_samples=MIN_SAMPLES):
        self.window_s, self.stride_s = float(window_s), float(stride_s)
        self.max_gap_s = None if max_gap_s is None else float(max_gap_s)
        self.min_samples = int(min_samples)
        self._history = {}
        self._next_start = {}
        # Whether the interval before the one about to close produced a descriptor.
        self._run = {}

    def reset(self, track_id):
        self._history.pop(track_id, None)
        self._next_start.pop(track_id, None)
        self._run.pop(track_id, None)

    def update(self, track_id, t, vector, mask, hip_speed, angular_speed, down, upright, quality):
        """Add one observation. Returns the intervals it closed, oldest first.

        Usually empty or a single window. More than one only when observations are sparse
        enough for a single frame to close several intervals at once, which the gap rule
        prevents at any supported sampling rate.
        """
        t = float(t)
        history = self._history.setdefault(track_id, deque())
        if self.max_gap_s is not None and history and t - history[-1][0] > self.max_gap_s:
            # A gap breaks the window; do not average across missing time, and do not let
            # evidence either side of it count as consecutive.
            history.clear()
            self._next_start[track_id] = None
            self._run[track_id] = False
        if self._next_start.get(track_id) is None:
            self._next_start[track_id] = t
            self._run.setdefault(track_id, False)
        history.append((t, np.concatenate([np.asarray(vector, float), np.asarray(mask, float)]),
                        float(hip_speed), float(angular_speed), bool(down), bool(upright), float(quality)))
        emitted = []
        while self._next_start[track_id] + self.window_s <= t + 1e-9:
            start = self._next_start[track_id]
            end = start + self.window_s
            rows = [r for r in history if start <= r[0] < end]
            self._next_start[track_id] = start + self.stride_s
            if len(rows) >= self.min_samples:
                emitted.append(Window(start, end, self._descriptor(rows), self._run[track_id]))
                self._run[track_id] = True
            else:
                # An interval nobody could summarise is a hole in the evidence, so the run
                # of consecutive windows ends here even though the identity survived.
                self._run[track_id] = False
            while history and history[0][0] < self._next_start[track_id]:
                history.popleft()
        return emitted

    @staticmethod
    def _descriptor(rows):
        return clip_descriptor(np.array([r[1] for r in rows]),
                               np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                               np.array([r[4] for r in rows]), np.array([r[5] for r in rows]),
                               np.array([r[6] for r in rows]))


def windows(t, x, hip_speed, angular_speed, down, upright, quality,
            window_s=WINDOW_S, stride_s=STRIDE_S):
    """Fixed-length descriptors over a finished sequence, via the runtime builder.

    `max_gap_s=None` keeps the historical offline behaviour of summarising across missing
    time. Tightening it changes what the models were trained on, so it is a separate,
    measured milestone rather than a side effect of this one.
    """
    t = np.asarray(t, dtype=float)
    if not len(t):
        return []
    x = np.asarray(x, dtype=float)
    split = len(FEATURE_NAMES)
    aggregator = WindowAggregator(window_s, stride_s, max_gap_s=None)
    result = []
    for i in range(len(t)):
        for window in aggregator.update(0, t[i], x[i][:split], x[i][split:], hip_speed[i],
                                        angular_speed[i], down[i], upright[i], quality[i]):
            result.append((window.start, window.end, window.descriptor))
    return result
