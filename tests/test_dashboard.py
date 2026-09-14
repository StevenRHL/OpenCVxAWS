"""Dashboard date provenance, persistence and partial-read regression tests."""
from datetime import date
import json
import sqlite3
from zoneinfo import ZoneInfo

import pytest
from watchverify import jobs, review
from watchverify.dashboard import build_snapshot, filter_updates, read_records


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    def create(run_id, category='possible_fall', at='2026-09-14T00:00:00Z', status='completed'):
        folder = jobs.run_dir(run_id)
        folder.mkdir(parents=True)
        source = folder / 'source.mp4'
        source.write_bytes(b'UI fixture; no inference')
        job = dict(run_id=run_id, status=status, source_path=str(source), original_name=run_id+'.mp4',
                   created_at_utc='2026-09-13T23:59:00Z', finished_at_utc='2026-09-14T00:02:00Z', summary={})
        (folder / 'manifest.json').write_text(json.dumps(job))
        event = dict(event_id=run_id+'-event', category=category, source_start_s=3.7,
                     occurred_at_utc=None, emitted_at_utc=at, revision=2, status='incomplete')
        (folder / 'events.json').write_text(json.dumps([event]))
        return job, event
    return create


def test_first_alert_times_do_not_become_recording_dates_or_revision_dates(saved_runs):
    a, event = saved_runs('a', at='2026-09-14T10:00:00+10:00')
    b, _ = saved_runs('b', category='unusual_activity', at='2026-09-14T00:01:00Z')
    result = build_snapshot([a, b])
    observations = [r for r in result['updates'] if r['kind'] == 'Observation']
    assert [r['run_id'] for r in observations] == ['b', 'a']
    assert result['urgent'][0]['event']['occurred_at_utc'] is None
    assert result['urgent'][0]['at'] == event['emitted_at_utc']
    assert len(result['urgent']) == 1  # revision 2 is still one incident
    assert result['other_pending'] == 1
    assert result['legacy_runs'] == 2


def test_filters_use_display_date_and_exclude_unknown_time(saved_runs):
    a, _ = saved_runs('a', at='2026-09-13T15:00:00Z')  # 14 Sep Sydney, 13 Sep UTC
    b, _ = saved_runs('b', at=None)
    rows = build_snapshot([a, b])['updates']
    args = dict(urgency='urgent', kind='Observation', review='unreviewed', dates=(date(2026,9,14),date(2026,9,14)))
    assert [r['run_id'] for r in filter_updates(rows, tz=ZoneInfo('Australia/Sydney'), **args)] == ['a']
    assert not filter_updates(rows, tz=ZoneInfo('UTC'), **args)
    assert filter_updates(rows, kind='Observation')[-1]['at'] is None


def test_review_removes_attention_without_changing_urgency_or_original_event(saved_runs):
    job, event = saved_runs('review')
    original = (jobs.run_dir('review')/'events.json').read_bytes()
    jobs.save_review('review', event['event_id'], 'false_alarm', 'Ordinary movement')
    result = build_snapshot([job])
    assert not result['urgent']
    assert all(r['priority'] == 'urgent' for r in result['updates'] if r['event_id'])
    assert len([r for r in result['updates'] if r['kind'] == 'Review']) == 1
    assert (jobs.run_dir('review')/'events.json').read_bytes() == original
    jobs.save_review('review', event['event_id'], 'unclear')
    assert len(build_snapshot([job])['urgent']) == 1


def test_dashboard_reads_latest_sqlite_revision_without_writing(saved_runs):
    job, event = saved_runs('database')
    db = jobs.run_dir('database')/'events.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE revisions(event_id TEXT, revision INTEGER, payload TEXT)')
        for revision in range(3):
            connection.execute('INSERT INTO revisions VALUES(?,?,?)',
                               (event['event_id'], revision, json.dumps(dict(event,revision=revision))))
    before = db.read_bytes()
    events, reviews, ignored = read_records('database')
    assert events[0]['revision'] == 2 and len(events) == 1 and reviews == {} and ignored == {}
    assert db.read_bytes() == before
    assert len(build_snapshot([job])['urgent']) == 1


def test_ignoring_an_event_drops_it_from_urgent_and_pending_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture bytes for the dashboard ignore test')
    run_id = jobs.create_job(source)
    event = dict(event_id='incident-1', revision=0, category='possible_fall',
                source_start_s=3.7, status='active')
    with sqlite3.connect(jobs.run_dir(run_id) / 'events.db') as connection:
        connection.execute('CREATE TABLE revisions(event_id TEXT, revision INTEGER, payload TEXT, '
                           'PRIMARY KEY(event_id,revision))')
        connection.execute('INSERT INTO revisions VALUES(?,?,?)',
                           (event['event_id'], 0, json.dumps(event)))
    job = jobs.get_job(run_id)
    assert len(build_snapshot([job])['urgent']) == 1
    review.record_action(run_id, 'incident-1', 'ignore')
    snapshot = build_snapshot([job])
    assert len(snapshot['urgent']) == 0
    assert snapshot['other_pending'] == 0
    row = next(r for r in snapshot['updates'] if r['kind'] == 'Observation')
    assert row['ignored'] is True
    # Ignored stays visible when explicitly requested, just excluded from the default feed.
    assert [r for r in filter_updates(snapshot['updates']) if r['kind'] == 'Observation'] == []
    assert len([r for r in filter_updates(snapshot['updates'], show_ignored=True)
               if r['kind'] == 'Observation']) == 1
    ignore_action = review.list_actions(run_id, 'incident-1')[0]
    review.record_action(run_id, 'incident-1', 'undo', previous_action_id=ignore_action['action_id'])
    assert len(build_snapshot([job])['urgent']) == 1


def test_bad_database_is_visible_and_other_runs_remain_available(saved_runs):
    a, _ = saved_runs('a')
    b, _ = saved_runs('b')
    (jobs.run_dir('b')/'events.db').write_bytes(b'broken database')
    snapshot = build_snapshot([a,b])
    assert len(snapshot['errors']) == 1
    assert len(snapshot['urgent']) == 1


def test_status_history_records_transitions_not_progress_ticks(saved_runs, monkeypatch):
    saved_runs('queued', status='queued')
    monkeypatch.setattr(jobs, '_now', lambda: '2026-09-14T01:00:00Z')
    started = jobs.update_job('queued', status='running', started_at_utc='2026-09-14T01:00:00Z')
    assert started['status_history'] == [dict(status='queued',at='2026-09-13T23:59:00Z'),
                                         dict(status='running',at='2026-09-14T01:00:00Z')]
    updated = jobs.update_job('queued', progress=.3)
    assert updated['status_history'] == started['status_history']
    monkeypatch.setattr(jobs, '_now', lambda: '2026-09-14T01:01:00Z')
    ended = jobs.update_job('queued', status='cancelled')
    assert ended['status_history'][-1] == dict(status='cancelled',at='2026-09-14T01:01:00Z')
    assert len(ended['status_history']) == 3


def test_partial_visibility_is_distinct_from_unprocessed_source(saved_runs):
    job, _ = saved_runs('partial',status='cancelled')
    job['summary'] = dict(unobserved_source_s=2.,unprocessed_source_s=75.)
    reasons = build_snapshot([job])['attention'][0]['reasons']
    assert '2.0s unobserved in analysed footage' in reasons
    assert '75.0s of source not analysed' in reasons


def test_unreadable_manifest_is_reported_not_an_empty_safe_dashboard(saved_runs):
    saved_runs('broken')
    (jobs.run_dir('broken')/'manifest.json').write_text('{')
    snapshot = build_snapshot()
    assert snapshot['total_runs'] == 0
    assert 'broken: analysis unavailable' in snapshot['errors'][0]
