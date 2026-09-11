"""Software invariants only; these synthetic poses do not measure fall accuracy."""
import copy
import math

import numpy as np
import pytest

from watchverify.core import FEATURE_NAMES, FeatureBuffer, IncidentManager, RuleDetector, Tracker


def pose(cx=100.0, cy=100.0, rotation=0.0):
    p = np.zeros((33, 4), dtype=float)
    values = {11: (-10, -40), 12: (10, -40), 13: (-15, -20), 14: (15, -20),
              15: (-18, 10), 16: (18, 10), 23: (-8, 0), 24: (8, 0),
              25: (-8, 30), 26: (8, 30), 27: (-8, 60), 28: (8, 60)}
    theta = math.radians(rotation)
    matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    for j, point in values.items():
        p[j, :2] = matrix @ point + (cx, cy)
        p[j, 3] = 1
    return p


def features(t, down=False, upright=False, speed=0.0, rotation=0.0):
    return dict(t=t, down=down, upright=upright, hip_speed=speed, angular_speed=rotation)


def candidate(category='possible_fall', track_id=1, observations=None):
    return dict(category=category, track_id=track_id, score=1.0,
                observations=observations or ['rapid_posture_change'])


def test_tracker_preserves_identity_when_output_order_changes():
    tracker = Tracker()
    initial = tracker.update([pose(100), pose(400)], 0)
    later = tracker.update([pose(405), pose(105)], 0.1)
    assert [v[0] for v in later] == [initial[1][0], initial[0][0]]


def test_ambiguous_crossing_does_not_share_person_history():
    tracker = Tracker()
    old = {i for i, _ in tracker.update([pose(100), pose(130)], 0)}
    new = {i for i, _ in tracker.update([pose(114), pose(116)], 0.1)}
    assert not old & new
    assert old <= set(tracker.retired_ids)


def test_tracker_gap_retires_id_and_rejects_reverse_time():
    tracker = Tracker()
    old = tracker.update([pose()], 0)[0][0]
    tracker.update([], 0.5)
    new = tracker.update([pose()], 1.1)[0][0]
    assert new != old and old in tracker.retired_ids
    with pytest.raises(ValueError):
        tracker.update([], 1.0)


def test_unusable_pose_is_not_a_person_at_zero_coordinates():
    assert Tracker().update([np.zeros((33, 4))], 0) == []
    p = pose()
    p[11, 0] = float('nan')
    assert FeatureBuffer().update(1, p, 0) is None


def test_pixel_geometry_upright_and_horizontal():
    f = FeatureBuffer()
    upright = f.update(1, pose(), 0)
    horizontal = f.update(2, pose(rotation=90), 0)
    assert upright['angle'] == pytest.approx(0)
    assert upright['upright'] and not upright['down']
    assert horizontal['angle'] == pytest.approx(90)
    assert horizontal['down'] and not horizontal['upright']
    assert len(upright['vector']) == len(FEATURE_NAMES)
    assert np.isfinite(horizontal['vector']).all()


def test_velocity_uses_elapsed_seconds_and_past_scale():
    f = FeatureBuffer()
    f.update(1, pose(cy=100), 0)
    result = f.update(1, pose(cy=108), 0.2)
    assert result['hip_speed'] == pytest.approx(1.0)  # 8px / .2s / 40px torso
    assert result['motion'] == pytest.approx(1.0)


def test_gap_and_missing_pose_do_not_create_descent():
    f = FeatureBuffer()
    f.update(1, pose(cy=100), 0)
    assert f.update(1, pose(cy=500), 1)['hip_speed'] == 0
    p = pose()
    p[23, 3] = 0
    assert f.update(1, p, 1.1) is None
    assert f.update(1, pose(cy=1000), 1.2)['hip_speed'] == 0


def test_feature_prefixes_are_causal_and_returned_vectors_do_not_mutate():
    prefix = [(0, pose()), (0.1, pose(cy=104)), (0.2, pose(rotation=30))]
    a, b = FeatureBuffer(), FeatureBuffer()
    before = [a.update(1, p, t) for t, p in prefix]
    saved = copy.deepcopy(before)
    a.update(1, pose(cy=1000), 0.3)
    after = [b.update(1, p, t) for t, p in prefix]
    for x, y, original in zip(before, after, saved):
        np.testing.assert_array_equal(x['vector'], y['vector'])
        np.testing.assert_array_equal(x['vector'], original['vector'])


def test_optional_missing_joints_produce_finite_masked_quality():
    p = pose()
    p[15] = (float('nan'), float('nan'), 0, 0)
    result = FeatureBuffer().update(1, p, 0)
    assert result['quality'] == pytest.approx(11 / 12)
    assert np.isfinite(result['vector']).all()
    assert not result['feature_mask'][6] and not result['feature_mask'][8]
    assert not result['joint_mask'][15]


def test_clip_starting_down_is_not_claimed_as_observed_fall():
    r = RuleDetector(down_hold=1)
    events = []
    for t in [0, .25, .5, .75, 1.0, 1.25]:
        events.extend(r.update(1, features(t, down=True), t))
    assert [v['category'] for v in events] == ['person_down']


def test_moving_person_already_down_is_not_an_observed_fall():
    r = RuleDetector(down_hold=1)
    events = []
    for t in [0, .25, .5, .75, 1.0, 1.25]:
        events.extend(r.update(1, features(t, down=True, speed=2), t))
    assert [v['category'] for v in events] == ['person_down']


def test_rule_recovery_can_resolve_alert_from_external_classifier():
    r = RuleDetector(recovery_hold=.4)
    m = IncidentManager('trained-model')
    m.step([candidate()], 0, [1])
    recovery = []
    for t in [.1, .3, .6, .8, 1.0]:
        recovery.extend(r.update(1, features(t, upright=True), t))
    assert len(recovery) == 1 and recovery[0]['category'] == 'recovery'
    recovery[0]['track_id'] = 1
    assert m.step(recovery, 1, [1])[0]['status'] == 'resolved'


def test_fall_requires_transition_plus_causal_hold_and_does_not_repeat():
    r = RuleDetector(down_hold=1, fall_hold=.35)
    assert r.update(1, features(0, upright=True), 0) == []
    assert r.update(1, features(.1, down=True, speed=2), .1) == []
    assert r.update(1, features(.3, down=True), .3) == []
    assert r.update(1, features(.5, down=True), .5)[0]['category'] == 'possible_fall'
    remaining = []
    for t in [.7, .9, 1.1, 1.3, 1.5]:
        remaining.extend(r.update(1, features(t, down=True), t))
    assert [v['category'] for v in remaining] == ['person_down']


def test_missing_interval_resets_hold_and_never_recovers():
    r = RuleDetector(down_hold=.4, recovery_hold=.4)
    r.update(1, features(0, down=True), 0)
    r.update(1, None, .2)
    assert r.update(1, features(.4, down=True), .4) == []
    assert r.update(1, features(.7, down=True), .7) == []
    assert r.update(1, features(.9, down=True), .9)[0]['category'] == 'person_down'
    assert r.update(1, None, 1.1) == []
    assert r.update(1, features(1.2, upright=True), 1.2) == []
    assert r.update(1, features(1.5, upright=True), 1.5) == []
    assert r.update(1, features(1.7, upright=True), 1.7)[0]['category'] == 'recovery'


def test_long_gap_cannot_count_as_down_dwell_time():
    r = RuleDetector(down_hold=1)
    r.update(1, features(0, down=True), 0)
    assert r.update(1, features(10, down=True), 10) == []


def test_initial_incident_snapshot_stays_immutable_and_down_merges():
    m = IncidentManager('test')
    first = m.step([candidate()], .5, [1])[0]
    original = copy.deepcopy(first)
    revision = m.step([candidate('person_down', observations=['sustained_horizontal_posture'])], 1, [1])[0]
    assert first == original
    assert first['revision'] == 0 and revision['revision'] == 1
    assert first['event_id'] == revision['event_id']
    assert revision['source_start_s'] == revision['emitted_source_s'] == .5
    assert first['occurred_at_utc'] is None
    assert 'person_down' in revision['observations']


def test_recording_time_offset_is_known_only_with_timezone():
    m = IncidentManager('test', '2026-09-11T12:00:00+10:00')
    event = m.step([candidate()], 2, [1])[0]
    assert event['occurred_at_utc'] == '2026-09-11T02:00:02+00:00'
    with pytest.raises(ValueError):
        IncidentManager('bad', '2026-09-11T12:00:00')


@pytest.mark.parametrize('end', ['lost', 'eof', 'cancelled'])
def test_loss_end_and_cancel_are_incomplete_never_recovery(end):
    m = IncidentManager('test')
    m.step([candidate()], 0, [1])
    result = m.step([], 2, []) if end == 'lost' else m.close_all(2, end)
    assert result[0]['status'] == 'incomplete'
    assert result[0]['reason'] != 'observed_recovery'


def test_only_explicit_recovery_resolves_and_new_episode_gets_new_id():
    m = IncidentManager('test')
    first = m.step([candidate()], 0, [1])[0]
    assert m.step([candidate('recovery', observations=['standing_guess'])], .1, [1]) == []
    result = m.step([candidate('recovery', observations=['sustained_upright'])], .2, [1])
    assert result[0]['status'] == 'resolved'
    second = m.step([candidate()], .3, [1])[0]
    assert second['event_id'] != first['event_id']


def test_different_people_and_activity_have_separate_incidents():
    m = IncidentManager('test')
    result = m.step([candidate(track_id=1), candidate(track_id=2), candidate('unusual_activity')], 0, [1, 2])
    assert len({v['event_id'] for v in result}) == 3


def test_activity_clear_resolves_an_activity_incident_without_track_loss():
    manager = IncidentManager('run')
    manager.step([dict(category='unusual_activity', track_id=1, score=1.0,
                       observations=['unusual_motion_against_training_baseline'])], 1.0, [1])
    revisions = manager.step([dict(category='activity_clear', track_id=1, score=0.1,
                                   observations=['sustained_normal_activity'])], 4.0, [1])
    assert [r['status'] for r in revisions] == ['resolved']
    assert revisions[0]['reason'] == 'sustained_normal_activity'


def test_activity_clear_never_resolves_a_safety_incident():
    """Ordinary movement is not evidence that a fallen person got up."""
    manager = IncidentManager('run')
    manager.step([candidate(category='person_down')], 1.0, [1])
    revisions = manager.step([dict(category='activity_clear', track_id=1, score=0.1,
                                   observations=['sustained_normal_activity'])], 5.0, [1])
    assert revisions == []
    assert all(e['status'] == 'active' for e in manager.events.values())


def test_activity_clear_without_the_observation_leaves_the_incident_active():
    manager = IncidentManager('run')
    manager.step([dict(category='unusual_activity', track_id=1, score=1.0,
                       observations=['unusual_motion_against_training_baseline'])], 1.0, [1])
    revisions = manager.step([dict(category='activity_clear', track_id=1, score=0.1,
                                   observations=[])], 4.0, [1])
    assert revisions == []
    assert all(e['status'] == 'active' for e in manager.events.values())


def test_activity_clear_for_an_unseen_track_is_ignored():
    manager = IncidentManager('run')
    assert manager.step([dict(category='activity_clear', track_id=9, score=0.1,
                              observations=['sustained_normal_activity'])], 1.0, [9]) == []
