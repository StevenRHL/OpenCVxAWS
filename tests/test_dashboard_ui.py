"""Exercise actual dashboard navigation and review persistence in isolated runs."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest
from watchverify import jobs

PAGE = Path(__file__).resolve().parents[1]/'app.py'


def fixture_run(root, run_id, category, at):
    folder = root/'outputs'/run_id
    folder.mkdir(parents=True)
    source = folder/'source.mp4'
    source.write_bytes(b'UI fixture, no decoding in AppTest')
    (folder/'manifest.json').write_text(json.dumps(dict(run_id=run_id, status='completed',
        original_name=run_id+'.mp4', source_path=str(source), created_at_utc=at,
        finished_at_utc=at, summary={}, config={})))
    (folder/'events.json').write_text(json.dumps([dict(event_id=run_id+'-event', category=category,
        emitted_at_utc=at, source_start_s=3.7, source_end_s=4.5, status='incomplete',
        track_id=1, observations=['rapid_posture_change'], reason='source_ended')]))
    jobs.save_escalation(run_id, [run_id+'-event'],
                         'ambulance_not_called' if category == 'possible_fall' else 'police_not_called')


def test_dashboard_default_filters_and_exact_event_navigation(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    fixture_run(tmp_path, 'urgent', 'possible_fall', '2026-09-14T00:01:00Z')
    fixture_run(tmp_path, 'other', 'unusual_activity', '2026-09-14T00:02:00Z')
    app = AppTest.from_file(str(PAGE)).run()
    assert not app.exception
    assert app.title[0].value == 'Dashboard'
    assert next(m for m in app.metric if m.label=='Urgent awaiting review').value == '1'
    app.selectbox(key='feed-urgency').set_value('review').run()
    assert not app.exception
    assert not any((b.key or '').startswith('feed-urgent:') for b in app.button)
    # Pinned urgent card is independent of feed filters.
    app.button(key='urgent-urgent:event:urgent-event').click().run()
    assert not app.exception
    assert app.radio(key='page').value == 'Debugger'
    assert app.selectbox(key='debugger-event-urgent').value == 'urgent-event'
    next(s for s in app.selectbox if s.label=='Your review').set_value('relevant')
    next(b for b in app.button if b.label=='Save review').click().run()
    app.radio(key='page').set_value('Dashboard').run()
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Urgent awaiting review').value == '0'
    reopened = AppTest.from_file(str(PAGE)).run()
    assert reopened.title[0].value == 'Dashboard'
    assert next(m for m in reopened.metric if m.label=='Urgent awaiting review').value == '0'


def test_empty_dashboard_and_upload_navigation(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    app = AppTest.from_file(str(PAGE)).run()
    assert not app.exception
    assert any('No analyses yet' in item.value for item in app.info)
    next(b for b in app.button if b.label=='Upload video').click().run()
    assert not app.exception
    assert app.radio(key='page').value == 'Analyses'
    assert next(b for b in app.button if b.label=='Start analysis').disabled


def test_fragment_navigation_requests_full_app_rerun(monkeypatch):
    # Reproduce the browser-only path: the fragment reruns after its callback,
    # while the outer page has not yet rendered the new navigation state.
    from watchverify.ui import dashboard
    calls = []
    monkeypatch.setattr(dashboard.st, 'session_state', {'page': 'Analyses'})
    class FullRerun(Exception):
        pass
    def rerun(**kwargs):
        calls.append(kwargs)
        raise FullRerun
    monkeypatch.setattr(dashboard.st, 'rerun', rerun)
    import pytest
    with pytest.raises(FullRerun):
        dashboard.dashboard_view.__wrapped__(str)
    assert calls == [{'scope': 'app'}]
