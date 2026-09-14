"""Camera preview: the recorded analysis stack, run on a live feed and kept in memory.

This is a preview and says so everywhere. It exists to answer one question that uploaded
footage cannot — *does this camera, in this room, at this angle, produce usable body pose
at all?* — before anyone spends time recording with it. It is not a monitoring service:
nothing is written to disk, no run is created, no incident enters the administrator's
queue, and closing the page ends the session and discards its state.

Everything that decides anything is the same object a recorded analysis uses: the same
`Tracker`, `FeatureBuffer`, `RuleDetector`, `EvidenceRelay`, `IncidentManager` and the same
checksum-verified `Models`. A preview that agreed with the analysis only by coincidence
would be worse than none, because it would be trusted.

Two things genuinely differ from `worker.py`, both forced by the feed being live:

- **Time is the wall clock**, not a container timestamp. A dropped or slow frame is real
  elapsed time here, so the gap rules see it as a gap, exactly as they would in a
  recording that was missing those frames.
- **There is no source duration**, so there is no progress, no end, and no `close_all`.
  An incident is left active when the session ends; it is never resolved by stopping,
  because stopping is not evidence that anything was resolved.
"""
from __future__ import annotations

import time

import cv2

from .core import (EvidenceRelay, FeatureBuffer, IncidentManager, RuleDetector, Tracker,
                   anchor)
from .features import (ACTIVITY_CLEAR_WINDOWS, ACTIVITY_SUSTAINED_WINDOWS, MIN_WINDOW_SAMPLES,
                       STRIDE_S, WINDOW_S, aggregator_for, expected_samples, sampling_supported)
from .models import Models
from .perception import PoseEstimator, draw_skeleton

# A preview that silently stops looking is worse than one that stops: the picture keeps
# moving and nobody can tell the analysis died. Consecutive read failures past this are
# reported to the caller rather than absorbed.
MAX_READ_FAILURES = 30


def available_cameras(limit: int = 4) -> list[int]:
    """Indices that open and return a frame. Opening a camera is the only way to know.

    macOS prompts for camera permission the first time this runs, and refuses silently if
    the prompt is declined, so an empty list here means "none usable", not "none attached".
    """
    found = []
    for index in range(limit):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened() and capture.read()[0]:
                found.append(index)
        finally:
            capture.release()
    return found


class LiveSession:
    """One open camera and the detection state built on top of it.

    Use as a context manager; `read()` returns one annotated frame and whatever the rule
    layer said about it. Frames arrive as fast as the camera delivers them and are always
    displayed, but pose is run no more often than `analysis_fps`, so the preview stays
    smooth on a machine that cannot analyse every frame.
    """

    def __init__(self, camera: int = 0, analysis_fps: float = 10.0, max_people: int = 4,
                 pose_variant: str = 'full', analysis_width: int = 640,
                 mirror: bool = True):
        if not 1 <= analysis_fps <= 30:
            raise ValueError('Analysis rate must be between 1 and 30 frames per second')
        if not 1 <= max_people <= 8:
            raise ValueError('Maximum visible people must be between 1 and 8')
        self.camera = int(camera)
        self.analysis_fps = float(analysis_fps)
        self.max_people = int(max_people)
        self.pose_variant = pose_variant
        self.analysis_width = int(analysis_width)
        self.mirror = bool(mirror)

        self.capture = None
        self.estimator = None
        self.models = Models()
        self.warnings: list[str] = []

        self.tracker = Tracker()
        self.buffers = FeatureBuffer()
        # Dwell constants come from the trained card exactly as the worker takes them, so a
        # preview cannot be tuned differently from the thing it is previewing.
        timing = {}
        if 'fall' in self.models.loaded:
            card = self.models.loaded['fall'][1]
            for key in ('fall_hold', 'down_hold', 'recovery_hold'):
                if f'{key}_s' in card:
                    timing[key] = float(card[f'{key}_s'])
        self.rules = RuleDetector(**timing)
        self.relay = EvidenceRelay.for_detector(self.rules)
        self.manager = IncidentManager('live')

        window_s, stride_s = WINDOW_S, STRIDE_S
        self.sustained_windows = ACTIVITY_SUSTAINED_WINDOWS
        if 'activity' in self.models.loaded:
            card = self.models.loaded['activity'][1]
            window_s = float(card.get('window_s', WINDOW_S))
            stride_s = float(card.get('stride_s', STRIDE_S))
            self.sustained_windows = int(card.get('sustained_windows', self.sustained_windows))
        self.activity_window_s = window_s
        self.windows = aggregator_for(self.analysis_fps, window_s, stride_s)
        # Same refusal as the worker: a rate that cannot fill a window cannot produce the
        # input this model was fitted on, so the branch is disabled out loud.
        if 'activity' in self.models.loaded and not sampling_supported(self.analysis_fps, window_s):
            self.models.loaded.pop('activity')
            self.models.status['activity'] = (
                f'Unavailable: {self.analysis_fps:g} fps puts only '
                f'{expected_samples(self.analysis_fps, window_s)} observations in a '
                f'{window_s:g}s window; at least {MIN_WINDOW_SAMPLES} are needed')
            self.warnings.append('The activity branch is off at this analysis rate.')

        self._activity_run: dict[int, int] = {}
        self._normal_run: dict[int, int] = {}
        self._started = None
        self._last_t = 0.0
        self._next_analysis = 0.0
        self._last_analysis = -999.0
        self._latest: list = []
        self._failures = 0
        self.frames_read = 0
        self.frames_analysed = 0
        self.frames_without_person = 0
        self.events: dict[str, dict] = {}

    def __enter__(self) -> 'LiveSession':
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                f'Camera {self.camera} could not be opened. Another application may be using '
                'it, or this app has not been given camera permission.')
        self.estimator = PoseEstimator(self.pose_variant, self.max_people, self.analysis_width)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self.estimator is not None:
            self.estimator.close()
            self.estimator = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._started is None else time.perf_counter() - self._started

    def active_events(self) -> list[dict]:
        return [event for event in self.events.values() if event['status'] == 'active']

    def read(self) -> dict:
        """Grab, optionally analyse, and annotate one frame.

        Returns a record whose `ok` says whether a frame arrived. Every other field is only
        meaningful when it did.
        """
        if self.capture is None:
            raise RuntimeError('The camera session is not open')
        grabbed, frame = self.capture.read()
        if not grabbed or frame is None:
            self._failures += 1
            return dict(ok=False, frames_lost=self._failures,
                        fatal=self._failures >= MAX_READ_FAILURES,
                        message=('The camera stopped returning frames.'
                                 if self._failures >= MAX_READ_FAILURES
                                 else 'A frame was dropped.'))
        self._failures = 0
        self.frames_read += 1
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # Strictly increasing source time is a precondition of every temporal component.
        # The wall clock is monotonic but can repeat at this resolution, so nudge rather
        # than let a duplicate reach the tracker.
        t = self.elapsed_s
        if t <= self._last_t:
            t = self._last_t + 1e-4
        self._last_t = t

        revisions, candidates = [], []
        analysed = t + 1e-6 >= self._next_analysis
        if analysed:
            candidates, revisions = self._analyse(frame, t)
            self._next_analysis = t + 1.0 / self.analysis_fps
            self._last_analysis = t
            self.frames_analysed += 1

        overlay = frame
        # Do not carry a stale skeleton across a gap; the worker draws under the same rule.
        if t - self._last_analysis <= .25:
            draw_skeleton(overlay, self._latest)
        self._banner(overlay, t)
        return dict(ok=True, frame=overlay, t=t, analysed=analysed,
                    people=len(self._latest), candidates=candidates,
                    revisions=revisions, active=self.active_events())

    def _analyse(self, frame, t: float) -> tuple[list[dict], list[dict]]:
        """One analysed frame, in the same order and with the same rules as `worker.py`."""
        tracks = self.tracker.update(self.estimator.detect(frame, t), t)
        self._latest = tracks
        for retired in self.tracker.retired_ids:
            # An identity that ended mid-fall hands its marker to whoever replaces it.
            self.relay.park(self.rules.release(retired),
                            self.tracker.retired_anchors.get(retired))
            self.buffers.reset(retired)
            self.windows.reset(retired)
            self._activity_run.pop(retired, None)
            self._normal_run.pop(retired, None)
        self.relay.expire(t)

        candidates, visible = [], []
        for track_id, pose in tracks:
            if track_id in self.tracker.new_ids:
                state = self.tracker.tracks[track_id]
                self.relay.adopt_into(self.rules, track_id, t,
                                      anchor(state['centre'], state['scale']))
            features = self.buffers.update(track_id, pose, t)
            fall = None
            decision = features
            if features is not None:
                visible.append(track_id)
                fall = self.models.score('fall', features['vector'], features['feature_mask'])
                candidates.extend(self._activity(track_id, features, t))
                if fall and fall['positive']:
                    decision = dict(features, down=True, upright=False)
            for candidate in self.rules.update(track_id, decision, t):
                candidate['track_id'] = track_id
                if fall and fall['positive'] and candidate['category'] in ('possible_fall', 'person_down'):
                    candidate['observations'] = [v.replace('horizontal_posture', 'down_posture')
                                                 for v in candidate['observations']]
                    candidate['observations'].append('trained_down_posture_support')
                candidates.append(candidate)
        if not visible:
            self.frames_without_person += 1

        revisions = self.manager.step(candidates, t, visible)
        for event in revisions:
            self.events[event['event_id']] = event
        return candidates, revisions

    def _activity(self, track_id: int, features: dict, t: float) -> list[dict]:
        """Windowed activity scoring. Abstains until a full, ungapped window exists."""
        emitted = []
        for window in self.windows.update(track_id, t, features['vector'], features['feature_mask'],
                                          features['hip_speed'], features['angular_speed'],
                                          features['down'], features['upright'], features['quality']):
            activity = self.models.score('activity', window.descriptor)
            if activity is None:
                continue
            # Scores either side of a break in observation are not consecutive evidence.
            if not window.continuous:
                self._activity_run.pop(track_id, None)
                self._normal_run.pop(track_id, None)
            if activity['positive']:
                self._normal_run.pop(track_id, None)
                self._activity_run[track_id] = self._activity_run.get(track_id, 0) + 1
                if self._activity_run[track_id] >= self.sustained_windows:
                    emitted.append(dict(category='unusual_activity', track_id=track_id,
                                        score=activity['score'],
                                        score_type=activity.get('score_type', 'uncalibrated_anomaly_score'),
                                        observations=['unusual_motion_against_training_baseline',
                                                      'human_review_required']))
            else:
                self._activity_run.pop(track_id, None)
                self._normal_run[track_id] = self._normal_run.get(track_id, 0) + 1
                if self._normal_run[track_id] >= ACTIVITY_CLEAR_WINDOWS:
                    emitted.append(dict(category='activity_clear', track_id=track_id,
                                        score=activity['score'],
                                        score_type=activity.get('score_type', 'uncalibrated_anomaly_score'),
                                        observations=['sustained_normal_activity']))
                    self._normal_run.pop(track_id, None)
        return emitted

    def _banner(self, overlay, t: float) -> None:
        labels = [event['category'].replace('_', ' ') for event in self.active_events()]
        if labels:
            state = ' | '.join(dict.fromkeys(labels))
        elif self._latest:
            state = 'Observing - review candidates only'
        else:
            state = 'No usable pose - visibility unknown'
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 52), (21, 27, 36), -1)
        cv2.putText(overlay, f'LIVE {t:07.2f}s  {state[:80]}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(overlay, 'Preview only / nothing is recorded / experimental', (10, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (172, 185, 197), 1, cv2.LINE_AA)
