"""Event-level ground truth, replay and matching for held-out evaluation.

Frame classification is not event detection. Everything here works in source seconds
and counts whole events, including events missed because pose was never available.

Replay feeds cached per-frame features through the same `RuleDetector` the worker uses,
so threshold sweeps cost seconds instead of re-running pose extraction. It reproduces the
worker's call pattern in `worker.py`: a frame with usable features updates the detector,
a frame whose features were rejected updates it with `None`, and a frame with no usable
pose does not update it at all (the detector's own gap logic then resets history).
"""
from pathlib import Path
import csv
import json

import numpy as np

from .core import RuleDetector, EvidenceRelay

ROOT = Path(__file__).resolve().parents[1]
FALL_CATEGORIES = ('possible_fall', 'person_down')
# Publisher posture codes: -1 not lying, 0 falling/transition, 1 lying on the ground.
NOT_LYING, TRANSITION, LYING = -1, 0, 1
# Posture tables per source. Own-camera labels are written in the publisher's three-column
# shape by scripts/prepare_owncam.py precisely so that everything downstream — ground truth,
# replay, matching — works on both corpora without a second code path.
POSTURE_LABEL_FILES = {
    'urfall': ('data/raw/urfall/urfall-cam0-falls.csv', 'data/raw/urfall/urfall-cam0-adls.csv'),
    'owncam': ('data/raw/owncam/owncam-posture.csv',),
}
PREPARED_BY_SOURCE = {'urfall': 'prepare_urfall.py', 'owncam': 'prepare_owncam.py'}


def load_posture_labels(root=ROOT, source='urfall'):
    """{sequence_id: {source_frame: posture_code}} from the label tables for one source."""
    if source not in POSTURE_LABEL_FILES:
        raise ValueError(f'Unknown posture-label source {source!r}')
    labels = {}
    for name in POSTURE_LABEL_FILES[source]:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(
                f'{path} is missing; run {PREPARED_BY_SOURCE.get(source, "the preparation script")} first')
        for row in csv.reader(path.open()):
            if row:
                labels.setdefault(row[0], {})[int(row[1])] = int(row[2])
    return labels


def frame_times(sequence_id, root=ROOT, source='urfall'):
    """{source_frame: output_seconds} from the preparation sidecar."""
    sidecar = root / 'data/processed' / source / f'{sequence_id}.json'
    if not sidecar.exists():
        raise FileNotFoundError(
            f'{sequence_id}: no preparation sidecar; run {PREPARED_BY_SOURCE.get(source, "preparation")}')
    return {int(r['source_frame']): float(r['output_t']) for r in json.loads(sidecar.read_text())['frames']}


def fall_ground_truth(sequence_id, labels, times):
    """Zero or one fall event per sequence, in source seconds.

    Only `fall-*` sequences contain a fall. An ADL sequence may contain transition and
    lying frames — a person deliberately lying down — and is a fall negative. Treating
    those as falls would make the fall/lie distinction untestable.
    """
    posture = labels.get(sequence_id, {})
    if not sequence_id.startswith('fall') or not posture:
        return []
    transition = sorted(f for f, v in posture.items() if v == TRANSITION and f in times)
    lying = sorted(f for f, v in posture.items() if v == LYING and f in times)
    if not transition and not lying:
        return []
    onset_frame = transition[0] if transition else lying[0]
    end_frame = lying[-1] if lying else transition[-1]
    return [dict(sequence_id=sequence_id, category='fall',
                 onset_s=times[onset_frame], end_s=times[end_frame],
                 onset_frame=onset_frame, end_frame=end_frame)]


def sequence_duration(sequence_id, times):
    return max(times.values()) if times else 0.0


def replay(cache_path, detector=None, score=None, threshold=None):
    """Replay one cached sequence through the rule detector. Returns emitted candidates.

    `score` is an optional callable (vector, mask) -> probability. When it is provided and
    the probability reaches `threshold`, the posture decision is overridden exactly as
    `worker.py` does, rather than trusting the hand-written geometric `down` test.
    """
    detector = detector or RuleDetector()
    with np.load(cache_path, allow_pickle=False) as d:
        t = d['t']; track = d['track']
        vectors = d['x']; masks = d['mask']
        down = d['down']; upright = d['upright']
        hip = d['hip_speed']; angular = d['angular_speed']
        attempts = d['attempts']
        reasons = np.array([str(v) for v in d['attempt_reason']])

    by_time = {}
    for i in range(len(t)):
        by_time.setdefault(round(float(t[i]), 6), []).append(i)

    events = []
    last_track = None
    # Mirrors the worker's handover of unspent fall evidence across a renumbering. The cache
    # stores no coordinates, so this matches on time alone and accepts every handover the
    # worker would make and some it would reject on distance: the permissive bound.
    relay = EvidenceRelay.for_detector(detector)
    for (frame_t, _index, _n_poses, _valid), reason in zip(attempts, reasons):
        key = round(float(frame_t), 6)
        rows = by_time.get(key, [])
        if reason in ('no_pose', 'pose_geometry_rejected'):
            continue  # The worker never reaches the detector for these frames.
        if not rows:
            if last_track is not None:
                events.extend(_emit(detector, last_track, None, key, sequence=cache_path))
            continue
        for i in rows:
            track_id = int(track[i])
            if last_track is not None and track_id != last_track:
                # A new id means a new temporal history, but a fall already in progress is
                # not over just because the person was renumbered mid-air.
                relay.park(detector.release(last_track))
                relay.adopt_into(detector, track_id, key)
            last_track = track_id
            features = dict(t=key, down=bool(down[i]), upright=bool(upright[i]),
                            hip_speed=float(hip[i]), angular_speed=float(angular[i]))
            if score is not None:
                probability = float(score(vectors[i], masks[i]))
                features['posture_score'] = probability
                if probability >= threshold:
                    features.update(down=True, upright=False)
            events.extend(_emit(detector, track_id, features, key, sequence=cache_path))
    return events


def _emit(detector, track_id, features, t, sequence):
    emitted = []
    for candidate in detector.update(track_id, features, t):
        candidate.update(track_id=track_id, t=t, sequence_id=Path(sequence).stem)
        emitted.append(candidate)
    return emitted


def match_fall_events(predicted, truth, duration_s, on_time_s=3.0, tolerance_s=1.0,
                      unobserved_s=0.0):
    """Match at most one alert per true event; surplus alerts are false.

    An alert counts as on-time only when it arrives within `on_time_s` of the onset.
    Detecting a fall long after the person is already down is a different product claim
    from detecting it as it happens, so the two are reported separately.

    `unobserved_s` is the source time this sequence could not be observed at all. It is
    carried through so aggregate rates can be read next to how much of the footage the
    system was blind to; it never changes the denominator of recall, which stays every
    true event including the ones nobody could have seen.

    A truth event carrying `track_id` also produces a wrong-person count: an alert that
    lands in the right interval but names a different person is matched here — the event
    was detected — and reported separately, because pointing a reviewer at the wrong
    person is a distinct failure from missing the event.
    """
    alerts = sorted((c for c in predicted if c['category'] in FALL_CATEGORIES), key=lambda c: c['t'])
    unmatched = list(alerts)
    matched, on_time, latencies = 0, 0, []
    identity_labelled, wrong_person = 0, 0
    for event in truth:
        window = [c for c in unmatched
                  if event['onset_s'] - tolerance_s <= c['t'] <= event['end_s'] + tolerance_s]
        if not window:
            continue
        first = min(window, key=lambda c: c['t'])
        unmatched.remove(first)
        matched += 1
        latency = first['t'] - event['onset_s']
        latencies.append(latency)
        if latency <= on_time_s:
            on_time += 1
        if event.get('track_id') is not None:
            identity_labelled += 1
            wrong_person += int(first.get('track_id') != event['track_id'])
    return dict(true_events=len(truth), alerts=len(alerts), matched=matched,
                on_time=on_time, missed=len(truth) - matched, false_alerts=len(unmatched),
                identity_labelled=identity_labelled, wrong_person=wrong_person,
                latencies_s=[round(v, 3) for v in latencies], duration_s=duration_s,
                unobserved_s=float(unobserved_s))


def aggregate(results):
    """Combine per-sequence match results into event-level rates."""
    total = {k: sum(r.get(k, 0) for r in results)
             for k in ('true_events', 'alerts', 'matched', 'on_time', 'missed', 'false_alerts',
                       'identity_labelled', 'wrong_person')}
    seconds = sum(r['duration_s'] for r in results)
    hours = seconds / 3600.0
    unobserved = sum(r.get('unobserved_s', 0.0) for r in results)
    latencies = [v for r in results for v in r['latencies_s']]
    total.update(
        recall=total['matched'] / total['true_events'] if total['true_events'] else None,
        on_time_recall=total['on_time'] / total['true_events'] if total['true_events'] else None,
        precision=total['matched'] / total['alerts'] if total['alerts'] else None,
        false_alerts_per_hour=total['false_alerts'] / hours if hours else None,
        source_hours=round(hours, 4),
        median_latency_s=round(float(np.median(latencies)), 3) if latencies else None,
        p90_latency_s=round(float(np.percentile(latencies, 90)), 3) if latencies else None,
        # Reported only where identity labels exist. Without them this is unknown, not zero.
        wrong_person_rate=(total['wrong_person'] / total['identity_labelled']
                           if total['identity_labelled'] else None),
        unobserved_source_s=round(unobserved, 3),
        unobserved_fraction=round(unobserved / seconds, 4) if seconds else None)
    return total


def failure_gallery(per_sequence, limit=20):
    """Named failures behind the rates: what was missed, and what alerted with nothing there.

    A rate alone cannot be investigated. Each entry points at a sequence so the footage
    and its pose coverage can be inspected directly.
    """
    gallery = []
    for record in per_sequence:
        if record.get('missed'):
            gallery.append(dict(kind='missed_event', sequence=record.get('sequence'),
                                true_events=record.get('true_events'),
                                pose_coverage=record.get('pose_coverage'),
                                alerts=record.get('alerts', [])))
        if record.get('false_alerts'):
            gallery.append(dict(kind='false_alert', sequence=record.get('sequence'),
                                false_alerts=record.get('false_alerts'),
                                pose_coverage=record.get('pose_coverage'),
                                alerts=record.get('alerts', [])))
    gallery.sort(key=lambda entry: (entry['kind'] != 'missed_event',
                                    entry.get('pose_coverage') if entry.get('pose_coverage') is not None else 1.0))
    return gallery[:limit]


def wilson_interval(successes, trials, z=1.96):
    """95% interval for a rate. With ~30 events the interval is the honest headline."""
    if not trials:
        return None
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    margin = z * ((p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) ** 0.5) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
