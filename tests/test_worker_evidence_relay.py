"""The evidence relay through the whole worker, not just the rule layer.

`test_fall_evidence_gap.py` pins the relay against a `RuleDetector` driven by hand, and the
replay path in `evaluation.py` measures it against the cached corpus. Neither shows that the
worker — which owns the real `Tracker`, and so decides when an identity actually ends —
wires the two together. That is what this file is for.

The sequence below is the `fall-01` failure in the shape the worker sees it: a person moves
sharply, body pose is lost for longer than the tracker will hold an identity across, and the
person reappears in the same place already on the ground. The tracker issues a new number,
and before the relay existed the fall evidence died with the old one.

These tests describe wiring. They measure no fall accuracy.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from watchverify import jobs, worker
from watchverify.core import Tracker


def pose_at(cx, cy):
    p = np.zeros((33, 4), dtype=float)
    values = {11: (-10, -40), 12: (10, -40), 13: (-15, -20), 14: (15, -20),
              15: (-18, 10), 16: (18, 10), 23: (-8, 0), 24: (8, 0),
              25: (-8, 30), 26: (8, 30), 27: (-8, 60), 28: (8, 60)}
    for joint, (dx, dy) in values.items():
        p[joint, :2] = (cx + dx, cy + dy)
        p[joint, 3] = 1.0
    return p


# The fall happens in the first half-second; pose is then lost for 1.4s, which is longer
# than Tracker.max_gap (1.0s) and so ends the identity, but leaves the marker inside
# RuleDetector.transition_window (2.0s). The person returns on the ground, in the same
# place, and the recording ends before the two-second person_down fallback could complete.
VISIBLE_UNTIL = 0.45
REAPPEARS_AT = 1.85
DURATION_S = 3.0


def run_worker(monkeypatch, tmp_path, *, relay_enabled, reappear_at=REAPPEARS_AT,
               reappear_x=300.0, duration_s=DURATION_S):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(jobs.shutil, 'disk_usage', lambda _: SimpleNamespace(free=50 * 1024 ** 3))
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'synthetic source')
    monkeypatch.setattr(worker, 'video_info',
                        lambda _: dict(duration_s=duration_s, width=64, height=64, fps=10))

    if not relay_enabled:
        # Reproduce the behaviour before the relay existed without checking out old code:
        # evidence is still released, and nothing is ever allowed to take it up.
        monkeypatch.setattr(worker.EvidenceRelay, 'adopt_into', lambda *a, **k: False)

    class Models:
        """No trained artifacts: the rule layer reads the geometric posture decision."""
        def __init__(self):
            self.loaded = {}
            self.status = {'fall': 'Not trained; rule prototype only'}

        def disclosures(self):
            return {}

        def score(self, *args):
            return None

    class Pose:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, bgr, t):
            if t <= VISIBLE_UNTIL:
                return [pose_at(300.0, 200.0)]
            if t >= reappear_at:
                return [pose_at(reappear_x, 200.0)]
            return []  # the blind span

    class Features:
        """Scripted posture: falling fast while upright, then down and still."""
        def reset(self, track_id):
            pass

        def update(self, track_id, pose, t):
            down = t >= reappear_at
            return dict(vector=np.ones(12), feature_mask=np.ones(12, dtype=bool),
                        joint_mask=np.ones(33, dtype=bool), quality=1.0, angle=90.0 if down else 0.0,
                        hip_y=200.0, down=down, upright=False,
                        motion=0.0, hip_speed=0.0 if down else 3.0,
                        angular_speed=0.0, t=t)

    class Encoder:
        def __init__(self, path, *args):
            self.path = path

        def write(self, *args):
            pass

        def close(self):
            self.path.write_bytes(b'synthetic export')

    monkeypatch.setattr(worker, 'Models', Models)
    monkeypatch.setattr(worker, 'PoseEstimator', Pose)
    monkeypatch.setattr(worker, 'FeatureBuffer', Features)
    monkeypatch.setattr(worker, 'VideoExport', Encoder)
    monkeypatch.setattr(worker, 'Tracker', Tracker)  # the real one: it decides identity
    step = 1 / 10
    monkeypatch.setattr(worker, 'frames',
                        lambda _: iter((i, round(i * step, 6), np.zeros((64, 64, 3), np.uint8))
                                       for i in range(int(duration_s / step))))
    run_id = jobs.create_job(source, {'analysis_fps': 10, 'max_people': 1})
    assert worker.run(run_id) == 'completed'
    events = jobs.load_events(run_id)
    metrics = json.loads((jobs.run_dir(run_id) / 'metrics.json').read_text())
    return events, metrics


def test_the_blind_span_really_does_end_the_identity(monkeypatch, tmp_path):
    """The premise: without a renumbering there would be nothing to carry."""
    _events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert metrics['track_gap_retirements'] == 1


def test_without_the_relay_the_worker_reports_nothing(monkeypatch, tmp_path):
    """The defect, end to end: a person on the ground and no alert of any kind."""
    events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=False)
    assert [e['category'] for e in events] == []
    assert metrics['carried_fall_evidence'] == 0


def test_with_the_relay_the_worker_reports_the_fall(monkeypatch, tmp_path):
    events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert [e['category'] for e in events] == ['possible_fall']
    assert metrics['carried_fall_evidence'] == 1


def test_the_worker_marks_the_alert_as_carried_for_the_reviewer(monkeypatch, tmp_path):
    events, _metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert 'evidence_carried_across_identity_change' in events[0]['observations']


def test_the_carried_alert_is_attributed_to_the_identity_that_landed(monkeypatch, tmp_path):
    """The alert must point the reviewer at the person who is on the ground now."""
    events, _metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert events[0]['track_id'] == 2


def test_a_person_who_reappears_somewhere_else_does_not_inherit_the_fall(monkeypatch, tmp_path):
    """Proximity is enforced with the worker's real coordinates, not just in unit tests."""
    events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True, reappear_x=1400.0)
    assert [e['category'] for e in events] == []
    assert metrics['carried_fall_evidence'] == 0


def test_evidence_older_than_the_window_is_not_carried_by_the_worker(monkeypatch, tmp_path):
    """A blind span longer than transition_window leaves nothing live to hand over."""
    events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True,
                                 reappear_at=2.85, duration_s=4.0)
    assert metrics['carried_fall_evidence'] == 0
    assert 'possible_fall' not in [e['category'] for e in events]


def test_the_run_reports_how_often_evidence_was_carried(monkeypatch, tmp_path):
    """A handover changes how an alert was assembled, so a run must be able to say it happened."""
    _events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert 'carried_fall_evidence' in metrics
