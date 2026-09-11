"""How a short blind span destroys fall evidence, and the tool that reports it.

Recorded after the `fall-01` smoke discrepancy: the Extra Trees fall challenger emitted no
alert on a sequence the installed model alerted on. The cause is not posture recognition —
the challenger scores the person on the ground higher than the incumbent does. It is that
the challenger crosses its threshold a fraction later, on the far side of a pose dropout
that ends the tracked identity and takes the rapid-posture-change marker with it.

These tests pin that behaviour so it cannot change silently. They describe the rule layer,
not fall accuracy on any corpus.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from watchverify.core import RuleDetector
from explain_fall_event import diagnose, identity_changes, observation_gaps, trace


def features(t, down=False, upright=False, speed=0.0, rotation=0.0):
    return dict(t=t, down=down, upright=upright, hip_speed=speed, angular_speed=rotation)


def falling(detector, track_id, start=0.0, step=0.1, motion=3.0):
    """Frames that establish a rapid posture change without yet being down."""
    emitted = []
    for i in range(5):
        emitted += detector.update(track_id, features(start + i * step, speed=motion), start + i * step)
    return emitted, start + 4 * step


def test_a_fall_seen_without_interruption_alerts():
    """The control: identical motion and landing on one identity does alert."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    emitted, last = falling(detector, 1)
    emitted += detector.update(1, features(last + 0.1, down=True), last + 0.1)
    assert [e['category'] for e in emitted] == ['possible_fall']


def test_a_blind_span_between_the_motion_and_the_landing_loses_the_fall():
    """The marker lives on the identity, so an interruption discards the fall evidence.

    This is the `fall-01` failure in miniature: the person is on the ground for the rest of
    the clip, so the marker can never be rebuilt, and the person_down fallback needs longer
    than the sequence has left.
    """
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    _emitted, last = falling(detector, 1)
    # Pose is lost; the tracker cannot re-associate across the gap and issues a new identity.
    landing = last + 0.5
    emitted = []
    for i in range(11):
        t = landing + i * 0.1
        emitted += detector.update(2, features(t, down=True), t)
    assert emitted == [], 'a fall split by a dropout currently produces no alert at all'


def test_the_same_landing_still_alerts_once_the_person_down_dwell_is_reached():
    """The fallback path is intact; on `fall-01` the sequence simply ends before it."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    _emitted, last = falling(detector, 1)
    landing = last + 0.5
    emitted = []
    for i in range(25):
        t = landing + i * 0.1
        emitted += detector.update(2, features(t, down=True), t)
    assert [e['category'] for e in emitted] == ['person_down']


def test_observation_gaps_reports_only_spans_the_rule_layer_forgets_across():
    attempts = [(0.0, 0, 1, 1), (0.1, 1, 1, 1), (0.2, 2, 1, 0), (0.9, 3, 1, 1)]
    reasons = ['ok', 'ok', 'no_pose', 'ok']
    assert observation_gaps(attempts, reasons, max_gap=0.5) == [(0.1, 0.9)]
    assert observation_gaps(attempts, reasons, max_gap=1.0) == []


def test_identity_changes_names_the_frame_where_the_person_was_renumbered():
    rows = dict(t=np.array([0.0, 0.1, 0.2]), track=np.array([1, 1, 2]))
    assert identity_changes(rows) == [(0.2, 1, 2)]


# --- the recorded sequence, when the prepared corpus is present ---------------

CACHE = ROOT / 'data/processed/features/urfall/fall-01.npz'
CHALLENGER = ROOT / 'runs/model-comparison-20260911-extended-families/fall_challenger.joblib'
needs_corpus = pytest.mark.skipif(
    not (CACHE.exists() and CHALLENGER.exists()),
    reason='needs the prepared UR Fall cache and the saved comparison artifacts')


@needs_corpus
def test_fall_01_miss_is_attributed_to_the_dropout_and_not_to_posture_scoring():
    import joblib
    card = json.loads((ROOT / 'models/fall.json').read_text())
    incumbent = joblib.load(ROOT / 'models/fall.joblib')
    challenger = joblib.load(CHALLENGER)

    _rows, incumbent_p, _s, incumbent_events, _a, _r, _d = trace(
        CACHE, incumbent, card['threshold'], card)
    rows, challenger_p, steps, challenger_events, attempts, reasons, detector = trace(
        CACHE, challenger, 0.8, card)

    assert [e['category'] for e in incumbent_events if e['category'] != 'recovery'] \
        == ['possible_fall']
    assert [e['category'] for e in challenger_events if e['category'] != 'recovery'] == []

    # Not a recognition failure: the challenger is the more confident of the two once the
    # person is on the ground. It is the operating point that lands on the wrong side.
    assert challenger_p.max() >= incumbent_p.max()

    assert observation_gaps(attempts, reasons, detector.max_gap) == [(3.703, 4.204)]
    assert identity_changes(rows) == [(4.204, 1, 2)]
    assert 'rapid-posture-change marker' in diagnose(steps, challenger_events, detector, 5.305)
