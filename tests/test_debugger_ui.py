"""Exercise the debugger page through real navigation and script execution."""
import json
from pathlib import Path
import sqlite3

from streamlit.testing.v1 import AppTest
from watchverify import jobs

PAGE = Path(__file__).resolve().parents[1] / 'app.py'


def fixture_run(root, run_id, *, revisions, source_bytes=b'fixture video bytes'):
    folder = root / 'outputs' / run_id
    folder.mkdir(parents=True)
    source = folder / 'source.mp4'
    source.write_bytes(source_bytes)
    (folder / 'manifest.json').write_text(json.dumps(dict(
        run_id=run_id, status='completed', original_name=run_id + '.mp4', source_path=str(source),
        created_at_utc='2026-09-14T00:00:00Z', finished_at_utc='2026-09-14T00:05:00Z',
        summary={}, config={}, model_disclosures={})))
    with sqlite3.connect(folder / 'events.db') as connection:
        connection.execute("CREATE TABLE revisions(event_id TEXT, revision INTEGER, payload TEXT, "
                           "PRIMARY KEY(event_id,revision))")
        for revision in revisions:
            connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                               (revision['event_id'], revision['revision'], json.dumps(revision)))
    return source


def _open_debugger(app, run_id, event_id):
    from watchverify.ui.debugger import open_debugger
    app.session_state['page'] = 'Debugger'
    app.session_state['debugger_run'] = run_id
    app.session_state['debugger_event'] = event_id
    return app.run()


def test_first_alert_unaffected_by_later_revision_and_latest_clearly_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    first = dict(event_id='incident-1', revision=0, category='possible_fall', track_id=1,
                source_start_s=10.0, source_end_s=11.0, emitted_source_s=10.5, status='active',
                reason='detected', created_at_utc='2026-09-14T00:01:00Z',
                observations=['rapid_posture_change'])
    later = dict(event_id='incident-1', revision=1, category='person_down', track_id=1,
                source_start_s=10.0, source_end_s=40.0, emitted_source_s=10.5, status='incomplete',
                reason='source_ended', created_at_utc='2026-09-14T00:03:00Z',
                observations=['sustained_horizontal_posture'])
    fixture_run(tmp_path, 'run-a', revisions=[first, later])
    app = AppTest.from_file(str(PAGE)).run()
    app = _open_debugger(app, 'run-a', 'incident-1')
    assert not app.exception
    text = ' '.join(item.value for item in list(app.markdown) + list(app.caption))
    assert 'At first alert' in text
    assert 'Latest evidence' in text
    assert 'Retrospective'.lower() in text.lower() or 'retrospective' in text.lower()
    # The first-alert caption must show revision 0's own observation, not the later one's.
    assert 'Rapid posture change' in text


def test_seek_before_starts_three_seconds_before_the_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    event = dict(event_id='incident-1', revision=0, category='possible_fall', track_id=1,
                source_start_s=10.0, source_end_s=11.0, emitted_source_s=10.5, status='active',
                reason='detected', created_at_utc='2026-09-14T00:01:00Z', observations=[])
    fixture_run(tmp_path, 'run-a', revisions=[event])
    app = AppTest.from_file(str(PAGE)).run()
    app = _open_debugger(app, 'run-a', 'incident-1')
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    assert 'seeking to 00:00:07.0' in captions  # 10.0 - 3


def test_missing_source_media_warns_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    event = dict(event_id='incident-1', revision=0, category='possible_fall', track_id=1,
                source_start_s=10.0, source_end_s=11.0, emitted_source_s=10.5, status='active',
                reason='detected', created_at_utc='2026-09-14T00:01:00Z', observations=[])
    source = fixture_run(tmp_path, 'run-a', revisions=[event])
    source.unlink()
    app = AppTest.from_file(str(PAGE)).run()
    app = _open_debugger(app, 'run-a', 'incident-1')
    assert not app.exception
    assert any('no longer available' in item.value for item in app.warning)


def test_missed_event_mode_shows_no_observation_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    folder = tmp_path / 'outputs' / 'run-empty'
    folder.mkdir(parents=True)
    source = folder / 'source.mp4'
    source.write_bytes(b'fixture')
    (folder / 'manifest.json').write_text(json.dumps(dict(
        run_id='run-empty', status='completed', original_name='run-empty.mp4',
        source_path=str(source), created_at_utc='2026-09-14T00:00:00Z',
        finished_at_utc='2026-09-14T00:05:00Z', summary={}, config={}, model_disclosures={})))
    app = AppTest.from_file(str(PAGE)).run()
    app.session_state['page'] = 'Debugger'
    app.session_state['debugger_run'] = 'run-empty'
    app = app.run()
    assert not app.exception
    assert any('No observation was recorded' in item.value for item in app.info)
