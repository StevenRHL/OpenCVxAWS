"""Activity evidence must never be assembled across a break in observation.

The 2026-09-12 audit reproduced an `unusual_activity` alert built from positive scores at
4.0 s and 8.7 s with the observation window resetting in between. The sustained rule was
expressed in seconds and cleared only on tracker retirement or an opposite score, so a
0.6 s gap — longer than the window's gap limit, shorter than the tracker's 1 s retirement —
kept the evidence timer alive while the window rebuilt from nothing.

These tests assert the required behaviour, not the defect. The audit's own probes in
`runs/project-audit-20260912/` assert the old behaviour and stay there as the record.
"""
import json
from types import SimpleNamespace

import numpy as np

from watchverify import jobs, worker


def _run(monkeypatch, tmp_path, times, positives, duration_s, analysis_fps=10, card=None):
    """Drive the worker over synthetic observations with a scripted activity model.

    `positives` is consumed one entry per activity score, so a test states the sequence of
    window verdicts directly and never has to reverse-engineer them from feature values.
    """
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(jobs.shutil, 'disk_usage', lambda _: SimpleNamespace(free=50 * 1024 ** 3))
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'synthetic source')
    monkeypatch.setattr(worker, 'video_info',
                        lambda _: dict(duration_s=duration_s, width=64, height=64, fps=10))

    verdicts = iter(positives)
    installed = dict(card or {'window_s': 5.0, 'stride_s': 1.0, 'threshold': 0.5})

    class Models:
        """Faithful on the one behaviour these tests turn on: an unloaded branch scores nothing."""
        def __init__(self):
            self.loaded = {'activity': (None, installed)}
            self.status = {'activity': 'Experimental local model'}

        def disclosures(self):
            return {name: {} for name in self.loaded}

        def score(self, name, *args):
            if name not in self.loaded:
                return None
            positive = next(verdicts, False)
            return dict(score=0.9 if positive else 0.1, positive=positive, score_type='probability')

    class Pose:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, *args):
            return [np.zeros((33, 4))]

    class Tracker:
        retired_ids = []
        gap_retired_ids = []
        ambiguous_ids = []
        # One identity that never ends, so no evidence is ever released or carried.
        retired_anchors = {}
        new_ids = []
        tracks = {}

        def update(self, poses, t):
            return [(1, poses[0])]

    class Features:
        def update(self, track, pose, t):
            return dict(vector=np.ones(12), feature_mask=np.ones(12), hip_speed=0., angular_speed=0.,
                        down=False, upright=True, quality=1., angle=0., t=t)

    class Encoder:
        def __init__(self, path, *args):
            self.path = path

        def write(self, *args):
            pass

        def close(self):
            self.path.write_bytes(b'synthetic export')

    monkeypatch.setattr(worker, 'Models', Models)
    monkeypatch.setattr(worker, 'PoseEstimator', Pose)
    monkeypatch.setattr(worker, 'Tracker', Tracker)
    monkeypatch.setattr(worker, 'FeatureBuffer', Features)
    monkeypatch.setattr(worker, 'VideoExport', Encoder)
    monkeypatch.setattr(worker, 'frames',
                        lambda _: iter((i, float(t), np.zeros((64, 64, 3), np.uint8))
                                       for i, t in enumerate(times)))
    run_id = jobs.create_job(source, {'analysis_fps': analysis_fps})
    assert worker.run(run_id) == 'completed'
    scored = [r for r in map(json.loads, (jobs.run_dir(run_id) / 'predictions.jsonl').read_text().splitlines())
              if r['activity_model']]
    return jobs.load_events(run_id), scored, jobs.get_job(run_id)


def test_two_positive_windows_either_side_of_a_gap_do_not_alert(tmp_path, monkeypatch):
    """The audit's reproduction, with the assertion it should always have had.

    0.6 s exceeds the window gap limit (0.5) but is below the tracker's 1 s retirement, so
    the identity survives while the window is rebuilt from scratch.
    """
    times = list(np.arange(0, 5.01, .1)) + list(np.arange(5.6, 10.71, .1))
    events, scored, _ = _run(monkeypatch, tmp_path, times, [True, True], 11.)
    assert len(scored) == 2 and scored[1]['t'] - scored[0]['t'] > 5, 'expected one score either side'
    assert not [e for e in events if e['category'] == 'unusual_activity'], \
        'evidence separated by unobserved time is not a sustained run'


def test_consecutive_positive_windows_still_alert(tmp_path, monkeypatch):
    """The control: without this the test above could pass by never alerting at all."""
    events, scored, _ = _run(monkeypatch, tmp_path, list(np.arange(0, 7.01, .1)), [True] * 3, 8.)
    assert len(scored) == 3
    assert [e for e in events if e['category'] == 'unusual_activity']


def test_ordinary_windows_either_side_of_a_gap_do_not_resolve_an_incident(tmp_path, monkeypatch):
    """The symmetric path the audit identified in code but never reproduced.

    Closing an incident writes `sustained_normal_activity` into an append-only record a
    reviewer reads as observed evidence. Two ordinary windows before a break plus one
    after it is not three consecutive ordinary windows, and must not resolve anything.
    """
    times = list(np.arange(0, 8.06, .1)) + list(np.arange(8.7, 15.81, .1))
    # Windows close at 5, 6, 7, 8 before the gap and at 13.7, 14.7, 15.7 after it.
    events, scored, _ = _run(monkeypatch, tmp_path, times, [True, True] + [False] * 5, 16.)
    assert len(scored) == 7
    activity = [e for e in events if e['category'] == 'unusual_activity']
    assert len(activity) == 1 and activity[0]['status'] == 'resolved'
    assert activity[0]['reason'] == 'sustained_normal_activity'
    # Had the two pre-gap ordinary windows counted, the run would have completed at 13.7.
    assert activity[0]['source_end_s'] >= 15.7 - 1e-6, 'the run must restart after the gap'


def test_an_analysis_rate_too_low_to_fill_a_window_disables_the_branch_out_loud(tmp_path, monkeypatch):
    """Silence is the failure this replaces.

    At two frames per second a fixed half-second gap limit made every single frame look
    like a break, so the branch scored nothing at all and said nothing about why.
    """
    card = {'window_s': 5.0, 'stride_s': 1.0, 'threshold': 0.5}
    events, scored, job = _run(monkeypatch, tmp_path, list(np.arange(0, 12.01, .5)),
                               [True] * 10, 13., analysis_fps=2, card=card)
    assert scored == [], 'a rate that cannot fill a window must not produce scores'
    assert not events
    assert 'Unavailable' in job['model_status']['activity']
    assert '10 observations' in job['model_status']['activity'], 'the reason must be readable'
