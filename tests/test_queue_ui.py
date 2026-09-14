"""Exercise Ignore/Undo and the learning queue through real navigation and script execution."""
import json
from pathlib import Path
import sqlite3

from streamlit.testing.v1 import AppTest
from watchverify import jobs, review

PAGE = Path(__file__).resolve().parents[1] / 'app.py'


def fixture_run(root, run_id, *, source_bytes=b'fixture video bytes'):
    folder = root / 'outputs' / run_id
    folder.mkdir(parents=True)
    source = folder / 'source.mp4'
    source.write_bytes(source_bytes)
    (folder / 'manifest.json').write_text(json.dumps(dict(
        run_id=run_id, status='completed', original_name=run_id + '.mp4', source_path=str(source),
        created_at_utc='2026-09-14T00:00:00Z', finished_at_utc='2026-09-14T00:05:00Z',
        summary={}, config={}, model_disclosures={})))
    event = dict(event_id='incident-1', revision=0, category='possible_fall', track_id=1,
                source_start_s=10.0, source_end_s=11.0, emitted_source_s=10.5, status='active',
                reason='detected', created_at_utc='2026-09-14T00:01:00Z', observations=[])
    with sqlite3.connect(folder / 'events.db') as connection:
        connection.execute("CREATE TABLE revisions(event_id TEXT, revision INTEGER, payload TEXT, "
                           "PRIMARY KEY(event_id,revision))")
        connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                           (event['event_id'], event['revision'], json.dumps(event)))
    return event


def test_ignore_persists_across_a_restart_and_drops_from_urgent_count(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    fixture_run(tmp_path, 'run-a')
    app = AppTest.from_file(str(PAGE)).run()
    assert next(m for m in app.metric if m.label == 'Urgent awaiting review').value == '1'
    app.session_state['page'] = 'Debugger'
    app.session_state['debugger_run'] = 'run-a'
    app.session_state['debugger_event'] = 'incident-1'
    app = app.run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Ignore this observation').click().run()
    assert not app.exception
    assert any('Ignored' in item.value for item in app.caption)
    app.session_state['page'] = 'Dashboard'
    app = app.run()
    assert next(m for m in app.metric if m.label == 'Urgent awaiting review').value == '0'
    reopened = AppTest.from_file(str(PAGE)).run()
    assert next(m for m in reopened.metric if m.label == 'Urgent awaiting review').value == '0'
    reopened.session_state['page'] = 'Debugger'
    reopened.session_state['debugger_run'] = 'run-a'
    reopened.session_state['debugger_event'] = 'incident-1'
    reopened = reopened.run()
    assert any(b.label == 'Undo ignore' for b in reopened.button)


def test_add_to_learning_then_mark_ready_in_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    fixture_run(tmp_path, 'run-a')
    app = AppTest.from_file(str(PAGE)).run()
    app.session_state['page'] = 'Debugger'
    app.session_state['debugger_run'] = 'run-a'
    app.session_state['debugger_event'] = 'incident-1'
    app = app.run()
    next(sb for sb in app.selectbox if sb.label == 'Proposed label').set_value('negative')
    next(ti for ti in app.text_input if ti.label == 'Visible action').set_value('person reaches for a shelf')
    app = next(b for b in app.button if b.label == 'Save to learning queue').click().run()
    assert not app.exception
    assert any('Saved to learning queue' in item.value for item in app.success)
    candidates = review.list_candidates('run-a')
    assert len(candidates) == 1
    assert candidates[0]['proposed_label'] == 'negative'

    queue = AppTest.from_file(str(PAGE)).run()
    queue.session_state['page'] = 'Learning queue'
    queue = queue.run()
    assert not queue.exception
    assert queue.title[0].value == 'Learning queue'
    assert any(b.label == 'Mark ready' for b in queue.button)
    queue = next(b for b in queue.button if b.label == 'Mark ready').click().run()
    assert not queue.exception
    assert review.list_candidates('run-a')[0]['status'] == 'ready_for_dataset_review'


def test_missed_interval_candidate_is_created_from_the_queue_page(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    fixture_run(tmp_path, 'run-a')
    app = AppTest.from_file(str(PAGE)).run()
    app.session_state['page'] = 'Learning queue'
    app = app.run()
    assert not app.exception
    next(sb for sb in app.selectbox if sb.label == 'Proposed label').set_value('uncertain')
    number_inputs = [ni for ni in app.number_input if ni.label in ('Interval start (s)', 'Interval end (s)')]
    start_input = next(ni for ni in number_inputs if ni.label == 'Interval start (s)')
    end_input = next(ni for ni in number_inputs if ni.label == 'Interval end (s)')
    start_input.set_value(40.0)
    end_input.set_value(45.0)
    app = next(b for b in app.button if b.label == 'Save to learning queue').click().run()
    assert not app.exception
    candidates = review.list_candidates('run-a')
    assert len(candidates) == 1
    assert candidates[0]['event_id'] is None
    assert candidates[0]['source_start_s'] == 40.0
