"""Causal, deterministic geometry and incident logic; thresholds are experimental.

No rules here diagnose injury or establish criminal activity. Coordinates are
original-frame pixels. A new Tracker id represents a new temporal history.
"""
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
REQUIRED = (11, 12, 23, 24)
FEATURE_NAMES = (
    "torso_angle_over_90", "body_aspect", "hip_speed_torso_per_s",
    "joint_motion_torso_per_s", "left_knee_angle_over_180",
    "right_knee_angle_over_180", "left_elbow_angle_over_180",
    "right_elbow_angle_over_180", "left_wrist_hip_torso_distance",
    "right_wrist_hip_torso_distance", "trailing_angle_range_over_90", "joint_coverage",
)
# Bump whenever the meaning, order, units or count of FEATURE_NAMES changes, or when
# FeatureBuffer alters how a vector is computed. Cached features carry this value and
# are recomputed when it moves; a stale cache must never be silently reused.
FEATURE_SCHEMA_VERSION = 1


def _pose(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (33, 4):
        raise ValueError("Pose must have shape (33, 4)")
    return p


def _valid(p, threshold):
    return np.isfinite(p[:, :2]).all(axis=1) & np.isfinite(p[:, 3]) & (p[:, 3] >= threshold)


def _time(t):
    t = float(t)
    if not math.isfinite(t) or t < 0:
        raise ValueError("Source time must be finite and nonnegative")
    return t


class Tracker:
    """Short anonymous tracks; ambiguous assignments start new identities.

    Conservative retirement loses continuity at crossings deliberately. This is
    preferable to sending another person's displacement to a fall detector.
    `retired_ids` lets the worker discard per-person state each update.
    """
    def __init__(self, max_gap=1.0, max_distance=2.0, ambiguity_margin=0.25, confidence=0.5):
        self.max_gap = max_gap
        self.max_distance = max_distance
        self.ambiguity_margin = ambiguity_margin
        self.confidence = confidence
        self.tracks = {}
        self.retired_ids = []
        self._next = 1
        self._last_t = None

    def _geometry(self, p):
        valid = _valid(p, self.confidence)
        chosen = [j for j in JOINTS if valid[j]]
        if len(chosen) < 4:
            return None
        pts = p[chosen, :2]
        if valid[list(REQUIRED)].all():
            centre = p[[23, 24], :2].mean(axis=0)
            scale = np.linalg.norm(p[[11, 12], :2].mean(axis=0) - centre)
        else:
            centre = np.median(pts, axis=0)
            scale = np.linalg.norm(np.ptp(pts, axis=0)) * 0.3
        return centre, max(float(scale), 1.0), valid

    def update(self, poses, t):
        t = _time(t)
        if self._last_t is not None and t <= self._last_t:
            raise ValueError("Tracker timestamps must strictly increase")
        self._last_t = t
        self.retired_ids = [i for i, s in self.tracks.items() if t - s['t'] > self.max_gap]
        for i in self.retired_ids:
            del self.tracks[i]
        observations = []
        for pose in poses:
            p = _pose(pose).copy()
            g = self._geometry(p)
            if g is not None:
                observations.append((p, g))
        ids = sorted(self.tracks)
        costs = np.full((len(ids), len(observations)), 1e6)
        for a, track_id in enumerate(ids):
            s = self.tracks[track_id]
            predicted = s['centre'] + s['velocity'] * (t - s['t'])
            for b, (p, (centre, scale, valid)) in enumerate(observations):
                denom = max(s['scale'], scale)
                dist = np.linalg.norm(centre - predicted) / denom
                common = valid & s['valid']
                if dist <= self.max_distance:
                    shape = 0.0
                    if common.sum() >= 4:
                        shape = np.median(np.linalg.norm(
                            (p[common, :2] - centre) / scale -
                            (s['pose'][common, :2] - s['centre']) / s['scale'], axis=1))
                    costs[a, b] = dist + 0.15 * shape
        ambiguous_rows, ambiguous_cols = set(), set()
        for a in range(len(ids)):
            order = np.argsort(costs[a])
            if len(order) > 1 and costs[a, order[1]] < 1e6 and costs[a, order[1]] - costs[a, order[0]] < self.ambiguity_margin:
                ambiguous_rows.add(a)
                ambiguous_cols.update(order[:2].tolist())
        for b in range(len(observations)):
            order = np.argsort(costs[:, b])
            if len(order) > 1 and costs[order[1], b] < 1e6 and costs[order[1], b] - costs[order[0], b] < self.ambiguity_margin:
                ambiguous_cols.add(b)
                ambiguous_rows.update(order[:2].tolist())
        for a in ambiguous_rows:
            track_id = ids[a]
            self.retired_ids.append(track_id)
            del self.tracks[track_id]
            costs[a, :] = 1e6
        for b in ambiguous_cols:
            costs[:, b] = 1e6
        assigned = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            assigned = {int(b): ids[int(a)] for a, b in zip(rows, cols) if costs[a, b] < 1e6}
        result = []
        for b, (p, (centre, scale, valid)) in enumerate(observations):
            track_id = assigned.get(b)
            if track_id is None:
                track_id = self._next
                self._next += 1
            old = self.tracks.get(track_id)
            velocity = np.zeros(2) if old is None else (centre - old['centre']) / (t - old['t'])
            self.tracks[track_id] = dict(t=t, centre=centre, scale=scale, valid=valid, pose=p, velocity=velocity)
            result.append((track_id, p))
        return result


class FeatureBuffer:
    def __init__(self, window_s=4.0, max_gap=0.5, confidence=0.5):
        self.window_s, self.max_gap, self.confidence = window_s, max_gap, confidence
        self._history = {}
        self._last_t = {}

    def reset(self, track_id):
        self._history.pop(track_id, None)
        self._last_t.pop(track_id, None)

    def update(self, track_id, pose, t):
        t = _time(t)
        last_t = self._last_t.get(track_id)
        if last_t is not None and t <= last_t:
            raise ValueError("Feature timestamps must strictly increase per track")
        self._last_t[track_id] = t
        p = _pose(pose)
        valid = _valid(p, self.confidence)
        if not valid[list(REQUIRED)].all():
            self._history.pop(track_id, None)
            return None
        shoulder = p[[11, 12], :2].mean(axis=0)
        hip = p[[23, 24], :2].mean(axis=0)
        torso = shoulder - hip
        scale = float(np.linalg.norm(torso))
        if scale < 1.0:
            self._history.pop(track_id, None)
            return None
        angle = math.degrees(math.acos(float(np.clip(-torso[1] / scale, -1, 1))))
        points = p[[j for j in JOINTS if valid[j]], :2]
        width, height = np.ptp(points, axis=0)
        aspect = float(width / max(height, 1.0))
        h = self._history.setdefault(track_id, deque())
        if h and t - h[-1]['t'] > self.max_gap:
            h.clear()
        while h and t - h[0]['t'] > self.window_s:
            h.popleft()
        speed, motion, angular_speed = 0.0, 0.0, 0.0
        if h:
            prev = h[-1]
            dt = t - prev['t']
            # Estimate scale using earlier frames only, preserving descent.
            past_scale = max(float(np.median([v['scale'] for v in h])), 1.0)
            speed = float((hip[1] - prev['hip'][1]) / dt / past_scale)
            common = valid & prev['valid']
            common[[i for i in range(33) if i not in JOINTS]] = False
            motion = float(np.median(np.linalg.norm(p[common, :2] - prev['pose'][common, :2], axis=1)) / dt / past_scale)
            angular_speed = (angle - prev['angle']) / dt

        def joint_angle(a, b, c):
            if not valid[[a, b, c]].all():
                return 0.5  # Neutral missing-value code; coverage accompanies it.
            u, v = p[a, :2] - p[b, :2], p[c, :2] - p[b, :2]
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            return 0.5 if denom < 1e-6 else float(math.acos(float(np.clip(u @ v / denom, -1, 1))) / math.pi)

        def wrist_distance(w, hip_index):
            return float(np.linalg.norm(p[w, :2] - p[hip_index, :2]) / scale) if valid[w] else 0.0

        quality = float(valid[list(JOINTS)].mean())
        angle_range = max([angle] + [v['angle'] for v in h]) - min([angle] + [v['angle'] for v in h])
        vector = np.array([angle / 90, aspect, speed, motion,
                           joint_angle(23, 25, 27), joint_angle(24, 26, 28),
                           joint_angle(11, 13, 15), joint_angle(12, 14, 16),
                           wrist_distance(15, 23), wrist_distance(16, 24), angle_range / 90, quality])
        if not np.isfinite(vector).all():
            self._history.pop(track_id, None)
            return None
        h.append(dict(t=t, hip=hip.copy(), scale=scale, pose=p.copy(), valid=valid.copy(), angle=angle))
        feature_mask = np.ones(len(FEATURE_NAMES), dtype=bool)
        for k, group in enumerate(((23, 25, 27), (24, 26, 28), (11, 13, 15), (12, 14, 16)), start=4):
            feature_mask[k] = bool(valid[list(group)].all())
        feature_mask[8], feature_mask[9] = valid[15], valid[16]
        return dict(vector=vector, feature_mask=feature_mask, joint_mask=valid.copy(),
                    quality=quality, angle=angle, hip_y=float(hip[1]),
                    down=bool(angle >= 60 and aspect >= 1.0),
                    upright=bool(angle <= 35 and aspect < 0.85), motion=motion,
                    hip_speed=speed, angular_speed=angular_speed, t=t)


class RuleDetector:
    """Configurable rule prototype. Scores are rule strengths, not probabilities."""
    def __init__(self, down_hold=2.0, fall_hold=0.35, recovery_hold=1.0,
                 transition_window=2.0, descent_speed=0.8, rotation_speed=60.0, max_gap=0.5):
        self.down_hold, self.fall_hold, self.recovery_hold = down_hold, fall_hold, recovery_hold
        self.transition_window, self.descent_speed, self.rotation_speed = transition_window, descent_speed, rotation_speed
        self.max_gap = max_gap
        self._states = {}

    def reset(self, track_id):
        self._states.pop(track_id, None)

    def update(self, track_id, features, t):
        t = _time(t)
        s = self._states.setdefault(track_id, dict(t=None, low=None, upright=None, transition=None,
                                                  non_down=None, fall=False, down=False, recovery_sent=False))
        if s['t'] is not None and t <= s['t']:
            raise ValueError("Rule timestamps must strictly increase per track")
        if features is None or (s['t'] is not None and t - s['t'] > self.max_gap):
            s.update(low=None, upright=None, transition=None, non_down=None, recovery_sent=False)
        s['t'] = t
        if features is None:
            return []
        if float(features['t']) != t:
            raise ValueError("Features must belong to current source timestamp")
        result = []
        if not features['down']:
            s['non_down'] = t
        recent_non_down = s['non_down'] is not None and t - s['non_down'] <= self.transition_window
        if recent_non_down and (features.get('hip_speed', 0) >= self.descent_speed or features.get('angular_speed', 0) >= self.rotation_speed):
            s['transition'] = t
        if features['down']:
            s['upright'] = None
            s['recovery_sent'] = False
            if s['low'] is None:
                s['low'] = t
            duration = t - s['low']
            if not s['fall'] and s['transition'] is not None and t - s['transition'] <= self.transition_window and duration >= self.fall_hold:
                result.append(dict(category='possible_fall', score=1.0, observations=['rapid_posture_change', 'sustained_horizontal_posture']))
                s['fall'] = True
            if not s['down'] and duration >= self.down_hold:
                result.append(dict(category='person_down', score=1.0, observations=['sustained_horizontal_posture']))
                s['down'] = True
        else:
            s['low'] = None
            if features.get('upright', False):
                if s['upright'] is None:
                    s['upright'] = t
                # Recovery evidence also serves alerts from a trained model.
                # The incident manager ignores it if no safety event is active.
                if not s['recovery_sent'] and t - s['upright'] >= self.recovery_hold:
                    result.append(dict(category='recovery', score=1.0, observations=['sustained_upright']))
                    s.update(fall=False, down=False, transition=None, recovery_sent=True)
            else:
                s['upright'] = None
                s['recovery_sent'] = False
        return result


class IncidentManager:
    """Append-only snapshot revisions; loss/EOF never asserts recovery."""
    def __init__(self, run_id, recording_start=None, lost_after=1.0):
        self.run_id = run_id
        self.recording_start = None
        if recording_start is not None:
            start = datetime.fromisoformat(recording_start.replace('Z', '+00:00'))
            if start.tzinfo is None:
                raise ValueError("Recording start needs an explicit UTC offset")
            self.recording_start = start.astimezone(timezone.utc)
        self.lost_after = lost_after
        self.events = {}
        self._active = {}
        self._last_seen = {}
        self._counter = 0
        self._last_t = None

    @staticmethod
    def _family(category):
        return 'person_safety' if category in ('possible_fall', 'person_down') else category

    def _snapshot(self, event, t, status, reason):
        event.update(revision=event['revision'] + 1, source_end_s=t, status=status, reason=reason)
        return deepcopy(event)

    def step(self, candidates, t, visible_ids):
        t = _time(t)
        if self._last_t is not None and t < self._last_t:
            raise ValueError("Incident time cannot move backwards")
        self._last_t = t
        for track_id in visible_ids:
            self._last_seen[track_id] = t
        revisions = []
        for candidate in candidates:
            track_id, category = int(candidate['track_id']), candidate['category']
            if category == 'recovery':
                key = (track_id, 'person_safety')
                event_id = self._active.pop(key, None)
                if event_id and 'sustained_upright' in candidate.get('observations', []):
                    revisions.append(self._snapshot(self.events[event_id], t, 'resolved', 'observed_recovery'))
                elif event_id:
                    self._active[key] = event_id
                continue
            if category == 'activity_clear':
                # Lets an activity incident end on observed evidence instead of waiting
                # for track loss or end of source. It resolves the activity family only:
                # ordinary movement is not evidence that a person got up.
                key = (track_id, 'unusual_activity')
                event_id = self._active.pop(key, None)
                if event_id and 'sustained_normal_activity' in candidate.get('observations', []):
                    revisions.append(self._snapshot(self.events[event_id], t, 'resolved', 'sustained_normal_activity'))
                elif event_id:
                    self._active[key] = event_id
                continue
            key = (track_id, self._family(category))
            event_id = self._active.get(key)
            observations = list(candidate.get('observations', []))
            if event_id is None:
                self._counter += 1
                event_id = f'{self.run_id}-{self._counter:06d}'
                now = datetime.now(timezone.utc).isoformat()
                event = dict(event_id=event_id, revision=0, category=category, track_id=track_id,
                             source_start_s=t, source_end_s=t, emitted_source_s=t,
                             created_at_utc=now, emitted_at_utc=now,
                             occurred_at_utc=None if self.recording_start is None else (self.recording_start + timedelta(seconds=t)).isoformat(),
                             status='active', reason='detected', score=float(candidate.get('score', 0)),
                             observations=observations, score_type=candidate.get('score_type', 'rule_strength'))
                self.events[event_id] = event
                self._active[key] = event_id
                self._last_seen.setdefault(track_id, t)
                revisions.append(deepcopy(event))
            else:
                event = self.events[event_id]
                additions = [v for v in observations if v not in event['observations']]
                if category != event['category'] and category not in event['observations']:
                    additions.append(category)
                if additions:
                    event['observations'].extend(additions)
                    revisions.append(self._snapshot(event, t, 'active', 'additional_evidence'))
        for key, event_id in list(self._active.items()):
            if t - self._last_seen.get(key[0], t) > self.lost_after:
                revisions.append(self._snapshot(self.events[event_id], t, 'incomplete', 'track_lost'))
                del self._active[key]
        return revisions

    def close_all(self, t, reason='source_ended'):
        t = _time(t)
        if self._last_t is not None and t < self._last_t:
            raise ValueError("Incident time cannot move backwards")
        self._last_t = t
        result = [self._snapshot(self.events[event_id], t, 'incomplete', reason)
                  for event_id in self._active.values()]
        self._active.clear()
        return result
