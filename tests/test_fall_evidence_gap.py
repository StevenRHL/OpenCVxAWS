"""How a short blind span destroys fall evidence, and how the relay carries it across.

Recorded after the `fall-01` smoke discrepancy: the Extra Trees fall challenger emitted no
alert on a sequence the installed model alerted on. The cause is not posture recognition —
the challenger scores the person on the ground higher than the incumbent does. It is that
the challenger crosses its threshold a fraction later, on the far side of a pose dropout
that ends the tracked identity and takes the rapid-posture-change marker with it.

`EvidenceRelay` closes that hole. A marker released by a retired identity is offered to the
identity that replaces it, under the detector's own `transition_window` and a proximity
check, so a fall split by a dropout is reported instead of silently lost. The incumbent
model escaped this by luck — its threshold happened to be crossed one frame before the
dropout — and these tests keep that luck from being load-bearing again.

They describe the rule layer, not fall accuracy on any corpus.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from watchverify.core import EvidenceRelay, RuleDetector, anchor
from explain_fall_event import diagnose, identity_changes, observation_gaps, trace


def features(t, down=False, upright=False, speed=0.0, rotation=0.0):
    return dict(t=t, down=down, upright=upright, hip_speed=speed, angular_speed=rotation)


def falling(detector, track_id, start=0.0, step=0.1, motion=3.0):
    """Frames that establish a rapid posture change without yet being down."""
    emitted = []
    for i in range(5):
        emitted += detector.update(track_id, features(start + i * step, speed=motion), start + i * step)
    return emitted, start + 4 * step


def landing(detector, track_id, start, frames=11, step=0.1):
    """Frames of a person already on the ground, under a given identity."""
    emitted = []
    for i in range(frames):
        t = start + i * step
        emitted += detector.update(track_id, features(t, down=True), t)
    return emitted


def test_a_fall_seen_without_interruption_alerts():
    """The control: identical motion and landing on one identity does alert."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    emitted, last = falling(detector, 1)
    emitted += detector.update(1, features(last + 0.1, down=True), last + 0.1)
    assert [e['category'] for e in emitted] == ['possible_fall']


def test_the_detector_alone_cannot_follow_a_fall_across_a_renumbering():
    """Why the relay exists: the marker lives on the identity that recorded it.

    This is the `fall-01` failure in miniature. Held here deliberately — the detector is
    per-identity by design and must stay that way, because carrying state across a
    renumbering is a judgement about people, not about posture. The relay is where that
    judgement is made, under bounds this class does not have.
    """
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    _emitted, last = falling(detector, 1)
    detector.reset(1)  # The tracker gives up on the identity outright.
    assert landing(detector, 2, last + 0.5) == []


def test_the_relay_carries_the_fall_across_the_renumbering():
    """The fix: released evidence reaches the identity that continues the fall."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, last = falling(detector, 1)
    relay.park(detector.release(1), anchor((100, 100), 50), track_id=1)
    start = last + 0.5
    took, handover = relay.adopt_into(detector, 2, start, anchor((110, 130), 50))
    assert took and handover == {'old_track_id': 1, 'released_at': last}
    assert [e['category'] for e in landing(detector, 2, start)] == ['possible_fall']


def test_a_carried_alert_says_the_evidence_crossed_an_identity_change():
    """A reviewer opening this alert will find a break in the footage; tell them first."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, last = falling(detector, 1)
    relay.park(detector.release(1), track_id=1)
    took, handover = relay.adopt_into(detector, 2, last + 0.5)
    assert took and handover['old_track_id'] == 1
    alert = landing(detector, 2, last + 0.5)[0]
    assert 'evidence_carried_across_identity_change' in alert['observations']
    assert 'rapid_posture_change' in alert['observations']


def test_an_uninterrupted_alert_is_not_labelled_as_carried():
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    _emitted, last = falling(detector, 1)
    alert = detector.update(1, features(last + 0.1, down=True), last + 0.1)[0]
    assert 'evidence_carried_across_identity_change' not in alert['observations']


def test_evidence_older_than_the_transition_window_is_not_carried():
    """The relay buys continuity, never time. An expired marker stays expired."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0, transition_window=2.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, last = falling(detector, 1)  # marker at 0.4s
    relay.park(detector.release(1))
    late = last + 2.5
    assert relay.adopt_into(detector, 2, late) == (False, None)
    assert landing(detector, 2, late) == []


def test_a_replacement_standing_somewhere_else_does_not_inherit_the_fall():
    """Proximity is what makes this a continuation rather than a guess about people."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, last = falling(detector, 1)
    relay.park(detector.release(1), anchor((100, 100), 50))
    start = last + 0.5
    # Six torso lengths away: a different person, on the far side of the room.
    assert relay.adopt_into(detector, 2, start, anchor((400, 100), 50)) == (False, None)
    assert landing(detector, 2, start) == []


def test_one_fall_cannot_seed_alerts_on_two_people():
    """Evidence is consumed by the identity that takes it."""
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, last = falling(detector, 1)
    relay.park(detector.release(1))
    start = last + 0.5
    assert relay.adopt_into(detector, 2, start)[0]
    assert relay.adopt_into(detector, 3, start) == (False, None)
    assert relay.pending == 0


def test_a_track_that_already_witnessed_its_own_fall_is_not_overwritten():
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _first, last = falling(detector, 1)
    relay.park(detector.release(1))
    _second, own = falling(detector, 2, start=last + 0.2)
    assert relay.adopt_into(detector, 2, own + 0.1) == (False, None)


def test_a_track_already_alerted_does_not_take_more_evidence():
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    relay = EvidenceRelay.for_detector(detector)
    _first, last = falling(detector, 1)
    detector.update(1, features(last + 0.1, down=True), last + 0.1)  # fires possible_fall
    assert detector.release(1) is None, 'a spent marker is not evidence of a second fall'


def test_a_released_track_without_a_marker_carries_nothing():
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0)
    detector.update(1, features(0.0), 0.0)
    assert detector.release(1) is None
    assert detector.release(99) is None


def test_expire_drops_evidence_the_detector_would_refuse():
    detector = RuleDetector(down_hold=2.0, fall_hold=0.0, transition_window=2.0)
    relay = EvidenceRelay.for_detector(detector)
    _emitted, _last = falling(detector, 1)
    relay.park(detector.release(1))
    assert relay.expire(1.0) == 0 and relay.pending == 1
    assert relay.expire(5.0) == 1 and relay.pending == 0


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
needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason='needs the prepared UR Fall feature cache')


@needs_cache
def test_fall_01_is_reported_at_an_operating_point_that_used_to_go_silent():
    """The regression that started this, on the real sequence.

    Threshold 0.8 is where the rejected challenger sat, and it is the operating point at
    which `fall-01` produced nothing at all: pose is lost at 3.703s, the person is
    renumbered at 4.204s, and the marker died in between. The person is on the ground at
    0.97 confidence for the rest of the clip, so silence here was never a close call.
    """
    import joblib
    card = json.loads((ROOT / 'models/fall.json').read_text())
    model = joblib.load(ROOT / 'models/fall.joblib')

    rows, probabilities, steps, events, attempts, reasons, detector = trace(
        CACHE, model, 0.8, card)

    assert observation_gaps(attempts, reasons, detector.max_gap) == [(3.703, 4.204)]
    assert identity_changes(rows) == [(4.204, 1, 2)]
    assert probabilities.max() >= 0.97, 'the model was never unsure the person was down'

    falls = [e for e in events if e['category'] == 'possible_fall']
    assert [e['t'] for e in falls] == [4.304], 'the fall is reported on the far side of the gap'
    assert 'evidence_carried_across_identity_change' in falls[0]['observations']
    assert 'carried across a pose dropout' in diagnose(steps, events, detector, 5.305)


@needs_cache
def test_the_installed_operating_point_relies_on_the_relay():
    """Whether the installed model needs the relay depends on where its threshold sits,
    and that has changed since this test was first written.

    The model installed when this test was written had threshold 0.3, crossed at 3.603s —
    one frame before the pose dropout — so the relay had nothing to carry for it. Retraining
    on 2026-09-15 (scripts/train_models.py, extra_trees_leaf8) moved the installed threshold
    to 0.8, the same operating point as
    test_fall_01_is_reported_at_an_operating_point_that_used_to_go_silent above: this
    fall-01 sequence now needs the relay under the *installed* card too, not only at the
    hardcoded 0.8 that test exercises. Pinning both keeps this from drifting unnoticed again.
    """
    import joblib
    card = json.loads((ROOT / 'models/fall.json').read_text())
    model = joblib.load(ROOT / 'models/fall.joblib')

    _rows, _p, _steps, events, _a, _r, _d = trace(CACHE, model, card['threshold'], card)
    falls = [e for e in events if e['category'] == 'possible_fall']
    assert [e['t'] for e in falls] == [4.304]
    assert 'evidence_carried_across_identity_change' in falls[0]['observations']
