"""The camera preview, driven by a fake camera and a fake pose estimator.

What matters about a preview is that it agrees with the thing it previews and that it never
pretends to be that thing. These tests pin both: the same rule timing and the same evidence
relay as `worker.py`, and nothing written to disk or added to anyone's queue.

They do not open a real camera, and they measure no detection accuracy.
"""
import math

import numpy as np
import pytest

from watchverify import live
from watchverify.live import LiveSession


def standing_pose(cx=300.0, cy=200.0, rotation=0.0):
    p = np.zeros((33, 4), dtype=float)
    values = {11: (-10, -40), 12: (10, -40), 13: (-15, -20), 14: (15, -20),
              15: (-18, 10), 16: (18, 10), 23: (-8, 0), 24: (8, 0),
              25: (-8, 30), 26: (8, 30), 27: (-8, 60), 28: (8, 60)}
    theta = math.radians(rotation)
    matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    for joint, point in values.items():
        p[joint, :2] = matrix @ point + (cx, cy)
        p[joint, 3] = 1.0
    return p


class FakeCapture:
    """A camera that yields a fixed number of frames, then fails like a closed device."""
    def __init__(self, frames=10, opened=True, size=(120, 160)):
        self.remaining = frames
        self.opened = opened
        self.size = size
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((*self.size, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def install(monkeypatch, capture, poses=None):
    """Point the session at a fake camera and a pose estimator under our control."""
    monkeypatch.setattr(live.cv2, 'VideoCapture', lambda index: capture)

    class Pose:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def detect(self, bgr, t):
            return poses(t) if poses else [standing_pose()]

        def close(self):
            self.closed = True

    monkeypatch.setattr(live, 'PoseEstimator', Pose)
    return capture


# --- refusing to start on bad settings ---------------------------------------

@pytest.mark.parametrize('kwargs', [dict(analysis_fps=0), dict(analysis_fps=60),
                                    dict(max_people=0), dict(max_people=99)])
def test_an_unusable_configuration_is_refused_before_the_camera_opens(kwargs):
    with pytest.raises(ValueError):
        LiveSession(**kwargs)


def test_a_camera_that_will_not_open_says_so_instead_of_showing_nothing(monkeypatch):
    capture = install(monkeypatch, FakeCapture(opened=False))
    with pytest.raises(RuntimeError, match='could not be opened'):
        with LiveSession():
            pass
    assert capture.released, 'a refused camera must still be released'


def test_reading_before_the_session_opens_is_an_error_not_a_blank_frame():
    with pytest.raises(RuntimeError):
        LiveSession().read()


# --- the loop ----------------------------------------------------------------

def test_a_frame_comes_back_annotated_and_timestamped(monkeypatch):
    install(monkeypatch, FakeCapture(frames=3))
    with LiveSession(analysis_fps=10) as session:
        frame = session.read()
    assert frame['ok'] and frame['frame'].shape == (120, 160, 3) and frame['t'] > 0


def test_source_time_strictly_increases_even_between_back_to_back_frames(monkeypatch):
    """Every temporal component rejects a repeated timestamp; the clock can repeat."""
    install(monkeypatch, FakeCapture(frames=25))
    monkeypatch.setattr(live.time, 'perf_counter', lambda: 1000.0)  # a frozen clock
    with LiveSession(analysis_fps=20) as session:
        times = [session.read()['t'] for _ in range(20)]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_pose_runs_no_more_often_than_the_analysis_rate(monkeypatch):
    """Display keeps up with the camera; analysis keeps to its own budget."""
    install(monkeypatch, FakeCapture(frames=40))
    with LiveSession(analysis_fps=5) as session:
        for _ in range(40):
            session.read()
        assert session.frames_read == 40
        assert session.frames_analysed < session.frames_read


def test_a_dropped_frame_is_reported_but_not_fatal(monkeypatch):
    install(monkeypatch, FakeCapture(frames=1))
    with LiveSession() as session:
        session.read()
        dropped = session.read()
    assert not dropped['ok'] and not dropped['fatal']


def test_a_camera_that_stops_returning_frames_is_eventually_declared_dead(monkeypatch):
    """A preview that silently stops looking is worse than one that stops."""
    install(monkeypatch, FakeCapture(frames=0))
    with LiveSession() as session:
        results = [session.read() for _ in range(live.MAX_READ_FAILURES)]
    assert results[-1]['fatal'] and 'stopped returning frames' in results[-1]['message']


def test_closing_releases_the_camera_and_the_pose_model(monkeypatch):
    capture = install(monkeypatch, FakeCapture(frames=2))
    with LiveSession() as session:
        session.read()
        estimator = session.estimator
    assert capture.released and estimator.closed
    assert session.capture is None and session.estimator is None


def test_mirroring_is_a_display_choice_only(monkeypatch):
    """The analysis must see the same room whichever way the picture is shown."""
    seen = []
    install(monkeypatch, FakeCapture(frames=6), poses=lambda t: (seen.append(t) or [standing_pose()]))
    with LiveSession(mirror=True, analysis_fps=10) as session:
        session.read()
    mirrored = list(seen)
    seen.clear()
    install(monkeypatch, FakeCapture(frames=6), poses=lambda t: (seen.append(t) or [standing_pose()]))
    with LiveSession(mirror=False, analysis_fps=10) as session:
        session.read()
    assert len(mirrored) == len(seen) == 1


# --- agreeing with the recorded analysis -------------------------------------

def test_the_preview_uses_the_rule_timing_from_the_trained_card(monkeypatch):
    """A preview tuned differently from the analysis would be trusted and wrong."""
    install(monkeypatch, FakeCapture(frames=1))
    session = LiveSession()
    if 'fall' in session.models.loaded:
        card = session.models.loaded['fall'][1]
        if 'down_hold_s' in card:
            assert session.rules.down_hold == float(card['down_hold_s'])
        if 'fall_hold_s' in card:
            assert session.rules.fall_hold == float(card['fall_hold_s'])


def test_the_evidence_relay_is_bounded_by_the_detector_it_feeds(monkeypatch):
    install(monkeypatch, FakeCapture(frames=1))
    session = LiveSession()
    assert session.relay.window_s == session.rules.transition_window


def test_an_analysis_rate_too_low_to_fill_a_window_turns_the_branch_off_out_loud(monkeypatch):
    install(monkeypatch, FakeCapture(frames=1))
    session = LiveSession(analysis_fps=1)
    if 'activity' in session.models.status and 'Unavailable' not in session.models.status['activity']:
        pytest.skip('no activity model installed to disable')
    assert 'activity' not in session.models.loaded
    assert session.warnings, 'the branch must not go quiet without saying so'


def test_a_person_who_disappears_is_reported_as_unobserved_not_as_clear(monkeypatch):
    install(monkeypatch, FakeCapture(frames=30), poses=lambda t: [])
    with LiveSession(analysis_fps=20) as session:
        for _ in range(30):
            session.read()
        assert session.frames_without_person == session.frames_analysed


# --- staying a preview -------------------------------------------------------

def test_a_session_writes_nothing_to_the_outputs_directory(monkeypatch, tmp_path):
    """Nothing here is saved, and nothing here reaches the administrator's queue."""
    from watchverify import jobs
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    (tmp_path / 'outputs').mkdir()
    install(monkeypatch, FakeCapture(frames=20))
    with LiveSession(analysis_fps=20) as session:
        for _ in range(20):
            session.read()
    assert list((tmp_path / 'outputs').iterdir()) == []
    assert jobs.list_jobs() == []


def test_stopping_never_resolves_an_open_observation(monkeypatch):
    """Stopping a preview is not evidence that anyone recovered."""
    install(monkeypatch, FakeCapture(frames=5))
    with LiveSession(analysis_fps=20) as session:
        for _ in range(5):
            session.read()
        before = [event['status'] for event in session.events.values()]
    assert [event['status'] for event in session.events.values()] == before


def test_available_cameras_reports_only_indices_that_actually_deliver(monkeypatch):
    captures = {0: FakeCapture(frames=1), 1: FakeCapture(frames=0), 2: FakeCapture(opened=False)}
    monkeypatch.setattr(live.cv2, 'VideoCapture', lambda index: captures[index])
    assert live.available_cameras(limit=3) == [0]
    assert all(capture.released for capture in captures.values())
