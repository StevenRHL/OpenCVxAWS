"""The administrator's queue: what reaches it, how it is cut, and what it must not claim.

These tests describe the translation from runs to incidents. They say nothing about whether
an observation was correct — only that the right observations are offered for a decision,
that a clip covers the event it claims to, and that a decision recorded in one view is
recorded in the other.
"""
import json

import numpy as np
import pytest

from watchverify import incidents, jobs
from watchverify.perception import VideoExport, video_info


def synthetic_video(path, seconds=12.0, fps=10.0):
    """A short, correctly timestamped recording. Content is irrelevant here."""
    encoder = VideoExport(path, 64, 64, fps)
    for i in range(int(seconds * fps)):
        encoder.write(np.zeros((64, 64, 3), dtype=np.uint8), i / fps)
    encoder.close()
    return path


def event(event_id, category='possible_fall', start=4.0, end=6.0, **extra):
    return dict(event_id=event_id, revision=0, category=category, track_id=1,
                source_start_s=start, source_end_s=end, emitted_source_s=start,
                created_at_utc='2026-09-14T00:00:00+00:00',
                emitted_at_utc='2026-09-14T00:00:00+00:00', occurred_at_utc=None,
                status='active', reason='detected', score=1.0,
                observations=['rapid_posture_change'], score_type='rule_strength', **extra)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """One completed run with a real decodable recording and two events on disk."""
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    monkeypatch.setattr(incidents, 'ROOT', tmp_path)
    source = synthetic_video(tmp_path / 'source.mp4')
    run_id = jobs.create_job(source, {'recording_start': '2026-09-14T14:30:00+08:00'},
                             original_name='shopfloor.mp4')
    folder = jobs.run_dir(run_id)
    events = [event(f'{run_id}-000001'),
              event(f'{run_id}-000002', category='unusual_activity', start=8.0, end=9.0)]
    (folder / 'events.json').write_text(json.dumps(events))
    jobs.update_job(run_id, status='completed',
                    media=video_info(source), started_at_utc='2026-09-14T00:00:00+00:00')
    return run_id, events


# --- what reaches the queue --------------------------------------------------

def test_only_observations_that_ask_for_a_decision_become_incidents(run):
    """Recovery and activity-clear are state changes; a queue of them would be noise."""
    run_id, events = run
    folder = jobs.run_dir(run_id)
    (folder / 'events.json').write_text(json.dumps(
        events + [event(f'{run_id}-000003', category='recovery'),
                  event(f'{run_id}-000004', category='activity_clear')]))
    categories = {row['category'] for row in incidents.collect()}
    assert categories == {'possible_fall', 'unusual_activity'}


def test_each_incident_is_named_for_the_attention_it_asks_for(run):
    rows = {row['category']: row for row in incidents.collect()}
    assert rows['possible_fall']['kind'] == 'Medical'
    assert rows['unusual_activity']['kind'] == 'Security'
    assert 'Not evidence of theft' in rows['unusual_activity']['meaning']


def test_an_undecided_observation_is_waiting_and_a_decided_one_is_not(run):
    run_id, events = run
    assert incidents.counts(incidents.collect())['waiting'] == 2
    jobs.save_escalation(run_id, [events[0]['event_id']], 'ambulance_called')
    summary = incidents.counts(incidents.collect())
    assert summary['waiting'] == 1 and summary['medical'] == 0 and summary['security'] == 1


def test_a_police_decision_does_not_settle_a_medical_observation(run):
    """The branch rule from alerts.py must hold here too, or the queue hides a person down."""
    run_id, events = run
    jobs.save_escalation(run_id, [events[1]['event_id']], 'police_called')
    waiting = [row for row in incidents.collect() if not row['decided']]
    assert [row['category'] for row in waiting] == ['possible_fall']


def test_decided_observations_can_be_hidden_without_being_forgotten(run):
    run_id, events = run
    jobs.save_escalation(run_id, [events[0]['event_id']], 'ambulance_not_called')
    assert len(incidents.collect(include_decided=False)) == 1
    assert len(incidents.collect()) == 2


def test_a_run_whose_events_cannot_be_read_is_skipped_not_fatal(run, tmp_path):
    """One damaged run must not take the whole queue down with it."""
    run_id, _events = run
    broken = jobs.create_job(synthetic_video(tmp_path / 'other.mp4', seconds=2.0),
                             original_name='broken.mp4')
    (jobs.run_dir(broken) / 'events.json').write_text('{ not json')
    assert {row['run_id'] for row in incidents.collect()} == {run_id}


# --- how the clip is cut -----------------------------------------------------

def test_the_clip_opens_before_the_event_so_the_cause_is_visible():
    """A clip starting on the alert shows the consequence and not what caused it."""
    start, end = incidents.clip_bounds(event('x', start=30.0, end=34.0), 120.0,
                                       lead_s=5.0, tail_s=3.0)
    assert (start, end) == (25.0, 37.0)


def test_padding_never_runs_past_the_recording_that_exists():
    start, end = incidents.clip_bounds(event('x', start=1.0, end=9.5), 10.0,
                                       lead_s=5.0, tail_s=3.0)
    assert start == 0.0 and end == 10.0


def test_an_instantaneous_event_still_produces_a_playable_interval():
    start, end = incidents.clip_bounds(event('x', start=0.0, end=0.0), None,
                                       lead_s=0.0, tail_s=0.0)
    assert end > start


def test_an_end_before_its_start_is_not_trusted():
    """A malformed interval must not produce a clip that runs backwards."""
    start, end = incidents.clip_bounds(event('x', start=8.0, end=2.0), 20.0,
                                       lead_s=1.0, tail_s=1.0)
    assert start == 7.0 and end == 9.0


def test_a_clip_is_cut_to_the_incident_and_decodes(run):
    run_id, events = run
    clip = incidents.build_clip(run_id, events[0], lead_s=2.0, tail_s=1.0)
    assert clip is not None and clip.exists()
    # 4.0s to 6.0s, padded to 2.0s-7.0s; the encoder keeps whole frames, so allow one.
    assert video_info(clip)['duration_s'] == pytest.approx(5.0, abs=0.2)


def test_clip_time_restarts_at_zero_so_the_player_opens_on_the_event(run):
    run_id, events = run
    clip = incidents.build_clip(run_id, events[0], lead_s=2.0, tail_s=1.0)
    from watchverify.perception import frames
    assert next(frames(clip))[1] == pytest.approx(0.0, abs=1e-6)


def test_changing_the_padding_changes_the_file_the_player_is_handed(run):
    """The reported bug: moving the clip-length sliders did nothing to the video.

    The cut was rebuilt correctly, but every padding wrote to the same path, so the player
    was handed a URL it had already cached and went on showing the old clip. The padding is
    part of the name now, which is what makes a re-cut visible.
    """
    run_id, events = run
    short = incidents.build_clip(run_id, events[0], lead_s=1.0, tail_s=1.0)
    long = incidents.build_clip(run_id, events[0], lead_s=4.0, tail_s=3.0)
    assert short != long
    assert short.exists() and long.exists()
    assert video_info(long)['duration_s'] > video_info(short)['duration_s'] + 1.0


def test_each_padding_keeps_its_own_cached_clip(run):
    run_id, events = run
    first = incidents.build_clip(run_id, events[0], lead_s=2.0, tail_s=1.0)
    stamp = first.stat().st_mtime_ns
    incidents.build_clip(run_id, events[0], lead_s=6.0, tail_s=2.0)
    assert incidents.build_clip(run_id, events[0], lead_s=2.0, tail_s=1.0).stat().st_mtime_ns == stamp


def test_a_built_clip_is_reused_rather_than_re_encoded(run):
    run_id, events = run
    first = incidents.build_clip(run_id, events[0])
    stamp = first.stat().st_mtime_ns
    assert incidents.build_clip(run_id, events[0]).stat().st_mtime_ns == stamp


def test_force_re_encodes_a_clip_that_already_exists(run):
    """For a recording that changed underneath a clip, such as a late annotated export."""
    run_id, events = run
    first = incidents.build_clip(run_id, events[0])
    stamp = first.stat().st_mtime_ns
    again = incidents.build_clip(run_id, events[0], force=True)
    assert again == first and again.stat().st_mtime_ns != stamp


def test_an_annotated_export_is_preferred_over_the_original(run):
    run_id, _events = run
    assert incidents.source_for_clip(run_id).name.startswith('source')
    annotated = jobs.run_dir(run_id) / 'annotated.mp4'
    synthetic_video(annotated, seconds=12.0)
    assert incidents.source_for_clip(run_id) == annotated


def test_a_missing_recording_yields_no_clip_rather_than_a_broken_one(run):
    run_id, events = run
    jobs.run_dir(run_id).joinpath('source.mp4').unlink()
    assert incidents.build_clip(run_id, events[0]) is None


def test_no_partial_file_is_left_behind_when_a_cut_fails(run):
    run_id, events = run
    jobs.run_dir(run_id).joinpath('source.mp4').unlink()
    incidents.build_clip(run_id, events[0])
    assert not list(jobs.run_dir(run_id).glob('clips/*.partial.mp4'))


def test_an_event_beyond_the_end_of_the_recording_produces_no_clip(run):
    """Nothing was recorded there, so there is nothing to show. Say so by returning None."""
    run_id, _events = run
    assert incidents.build_clip(run_id, event('ghost', start=90.0, end=95.0)) is None
