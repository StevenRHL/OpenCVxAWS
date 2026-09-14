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


def anchor(centre, scale):
    """Where a person was, in a form two identities can be compared in.

    Scale travels with the centre because a pixel distance means nothing on its own: two
    metres at the back of a room and two metres at the front are different numbers of
    pixels, and the same number of pixels is a different distance. Body size is the only
    ruler available in a single uncalibrated view, so separation is measured in torsos.
    """
    return (float(centre[0]), float(centre[1]), max(float(scale), 1.0))


def anchor_distance(a, b):
    """Separation between two anchors in torso lengths, or None if either is unknown."""
    if a is None or b is None:
        return None
    return math.dist(a[:2], b[:2]) / max(a[2], b[2])


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
        # Why each identity ended, kept apart so a run can report how often continuity was
        # lost to a crossing rather than to absence. Both are already in `retired_ids`.
        self.gap_retired_ids = []
        self.ambiguous_ids = []
        # Where each identity was standing when it ended, and which identities are new this
        # update. Retirement deletes the track, so without this the last known position is
        # gone by the time a caller sees `retired_ids`. A caller that wants to hand evidence
        # from a retired identity to its replacement needs both: the anchor to check the
        # replacement is in the same place, and `new_ids` to know a replacement appeared.
        self.retired_anchors = {}
        self.new_ids = []
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
        self.gap_retired_ids = list(self.retired_ids)
        self.ambiguous_ids = []
        self.retired_anchors = {}
        self.new_ids = []
        for i in self.retired_ids:
            self.retired_anchors[i] = anchor(self.tracks[i]['centre'], self.tracks[i]['scale'])
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
            self.ambiguous_ids.append(track_id)
            self.retired_anchors[track_id] = anchor(self.tracks[track_id]['centre'],
                                                    self.tracks[track_id]['scale'])
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
                self.new_ids.append(track_id)
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

    @staticmethod
    def _new_state():
        return dict(t=None, low=None, upright=None, transition=None, non_down=None,
                    fall=False, down=False, recovery_sent=False, carried=False)

    def reset(self, track_id):
        self._states.pop(track_id, None)

    def release(self, track_id):
        """Retire a track, handing back fall evidence it gathered but never got to use.

        `reset` discards an identity outright. This is the same thing for everything except
        the rapid-posture-change marker: when an identity ends still holding a live marker
        and no fall was ever raised, the fall it was witnessing is not over — the evidence
        for it merely has nowhere to live. Returning it lets a caller offer it to whoever
        the tracker issues next. Everything else about the identity is dropped as before.

        Returns None whenever there is nothing to carry: no such track, no marker, a marker
        already spent on an alert, or one already too old for `transition_window` to accept.
        """
        state = self._states.pop(track_id, None)
        if state is None or state['fall'] or state['transition'] is None:
            return None
        return dict(transition=state['transition'], released_at=state['t'])

    def adopt(self, track_id, evidence, t):
        """Give a new identity a marker released by the identity it replaced.

        This widens nothing. The marker keeps its original timestamp, so the same
        `transition_window` that governs a fall on one unbroken identity governs this one:
        evidence that would have expired stays expired, and the alert still needs a down
        posture of its own to fire. What changes is only that the evidence is allowed to
        cross the renumbering, which is the difference between reporting a fall and
        reporting nothing at all when pose drops out as the person lands.

        Refuses to overwrite a track that already has its own marker or has already
        alerted. Returns whether the evidence was taken up.
        """
        if evidence is None:
            return False
        t = _time(t)
        if t - evidence['transition'] > self.transition_window:
            return False
        state = self._states.setdefault(track_id, self._new_state())
        if state['fall'] or state['transition'] is not None:
            return False
        state['transition'] = evidence['transition']
        state['carried'] = True
        return True

    def update(self, track_id, features, t):
        t = _time(t)
        s = self._states.setdefault(track_id, self._new_state())
        if s['t'] is not None and t <= s['t']:
            raise ValueError("Rule timestamps must strictly increase per track")
        if features is None or (s['t'] is not None and t - s['t'] > self.max_gap):
            s.update(low=None, upright=None, transition=None, non_down=None, recovery_sent=False,
                     carried=False)
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
                observations = ['rapid_posture_change', 'sustained_horizontal_posture']
                # Say so when the two halves of the evidence were seen under different
                # identities. The alert is the same alert; how it was assembled is not, and
                # a reviewer checking the footage should know to expect a break in it.
                if s['carried']:
                    observations.append('evidence_carried_across_identity_change')
                result.append(dict(category='possible_fall', score=1.0, observations=observations))
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
                    s.update(fall=False, down=False, transition=None, recovery_sent=True, carried=False)
            else:
                s['upright'] = None
                s['recovery_sent'] = False
        return result


class EvidenceRelay:
    """Carries unspent fall evidence across the renumbering caused by a pose dropout.

    The failure this exists for is not hypothetical and not a scoring error. A person falls;
    MediaPipe loses the body for a few tenths of a second exactly while they are landing;
    the tracker, which refuses to guess across a gap, issues a new identity on the far side.
    The rapid-posture-change marker belonged to the old identity and died with it, so the
    new identity sees only somebody already lying still — indistinguishable from somebody
    who lay down deliberately. No fall alert can be raised, and because the person never
    gets up the marker can never be rebuilt. On `fall-01` that silences a fall the model
    scores at 0.97 confidence, and the person-down fallback needs more seconds than the
    recording has left.

    The relay holds released evidence for as long as the rule layer would have accepted it
    anyway and offers it to identities that appear where the old one vanished:

    - **Expiry** is `transition_window`, the detector's own bound, measured from the
      original marker. Nothing lives longer here than it would have lived on one unbroken
      identity, so this buys no extra time — only continuity.
    - **Proximity** is checked in torso lengths when both sides know where they were. A
      replacement standing somewhere else is a different person, not a continuation.
    - **Single use**: evidence is consumed by the first identity that takes it, so one fall
      cannot seed alerts on several people.

    Anchors are optional because not every caller has them. Replay from a feature cache has
    no coordinates, so it matches on time alone and is the more permissive of the two — the
    same declared asymmetry as the offline window builder in `features.py`. Every handover
    the worker makes is therefore one replay would also have made, which is what keeps
    measured evidence comparable to runtime behaviour.
    """
    def __init__(self, window_s, max_distance=2.0):
        self.window_s = float(window_s)
        self.max_distance = float(max_distance)
        self._pending = []

    @classmethod
    def for_detector(cls, detector, max_distance=2.0):
        """A relay bounded by the detector it feeds, so the two cannot disagree."""
        return cls(detector.transition_window, max_distance)

    def park(self, evidence, anchor=None):
        """Hold evidence released by a retired identity. Ignores nothing-to-carry."""
        if evidence is not None:
            self._pending.append(dict(evidence=evidence, anchor=anchor))

    def expire(self, t):
        """Drop evidence the rule layer would no longer accept. Returns how many went."""
        t = _time(t)
        keep = [p for p in self._pending if t - p['evidence']['transition'] <= self.window_s]
        dropped = len(self._pending) - len(keep)
        self._pending = keep
        return dropped

    def adopt_into(self, detector, track_id, t, anchor=None):
        """Offer the nearest live evidence to a new identity. Returns whether it took it.

        Nearest first so that with several falls in flight the replacement is matched to the
        one it most plausibly continues rather than to whichever was released first.
        """
        t = _time(t)
        self.expire(t)
        candidates = []
        for index, pending in enumerate(self._pending):
            separation = anchor_distance(pending['anchor'], anchor)
            if separation is not None and separation > self.max_distance:
                continue
            candidates.append((separation if separation is not None else math.inf, index))
        for _separation, index in sorted(candidates):
            if detector.adopt(track_id, self._pending[index]['evidence'], t):
                del self._pending[index]
                return True
        return False

    @property
    def pending(self):
        """How many pieces of evidence are waiting for an identity to continue them."""
        return len(self._pending)


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
