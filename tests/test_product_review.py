"""Full Streamlit page regression: persisted reviews must refresh counters immediately."""
import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from watchverify import jobs


def test_save_review_refreshes_count_and_survives_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=50*1024**3))
    source=tmp_path/"fixture.mp4"
    source.write_bytes(b"test media; playback tested separately in browser")
    run_id=jobs.create_job(source)
    event={"event_id":"review-fixture", "category":"possible_fall", "track_id":1,
           "source_start_s":1.0, "source_end_s":2.0, "emitted_source_s":1.0,
           "status":"incomplete", "reason":"source_ended"}
    (jobs.run_dir(run_id)/"events.json").write_text(json.dumps([event]))
    jobs.update_job(run_id,status="completed")
    jobs.save_escalation(run_id,[event["event_id"]],"ambulance_not_called")
    page=Path(__file__).resolve().parents[1]/"app.py"
    app=AppTest.from_file(str(page)).run()
    assert not app.exception
    assert next(m for m in app.metric if m.label=="Reviewed").value=="0"
    next(s for s in app.selectbox if s.label=="Your review").set_value("relevant")
    next(t for t in app.text_area if t.label=="Notes (optional)").set_value("Regression review")
    next(b for b in app.button if b.label=="Save review").click().run()
    assert not app.exception
    assert next(m for m in app.metric if m.label=="Reviewed").value=="1"
    assert jobs.get_reviews(run_id)[event["event_id"]]["note"]=="Regression review"
    reopened=AppTest.from_file(str(page)).run()
    assert not reopened.exception
    assert next(m for m in reopened.metric if m.label=="Reviewed").value=="1"
    assert next(s for s in reopened.selectbox if s.label=="Your review").value=="relevant"


def prepared_run(tmp_path, monkeypatch, summary, events, status="completed"):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=50*1024**3))
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"test media; playback tested separately in browser")
    run_id = jobs.create_job(source)
    (jobs.run_dir(run_id) / "events.json").write_text(json.dumps(events))
    jobs.update_job(run_id, status=status, summary=summary)
    page = Path(__file__).resolve().parents[1] / "app.py"
    return run_id, AppTest.from_file(str(page)).run()


def test_a_quiet_timeline_states_the_time_that_could_not_be_observed(tmp_path, monkeypatch):
    """An empty event list must not be readable as a period that was watched and was clear."""
    _run_id, app = prepared_run(
        tmp_path, monkeypatch,
        {"events": 0, "pose_coverage": 0.4, "source_duration_s": 100.0,
         "source_time_s": 100.0, "unobserved_source_s": 60.0, "unobserved_intervals": 3},
        [])
    assert not app.exception
    text = " ".join(str(w.value) for w in app.warning)
    assert "60.0s" in text and "60%" in text and "3 interval" in text
    assert "unknown, not clear" in text


def test_a_fully_observed_run_does_not_invent_a_visibility_warning(tmp_path, monkeypatch):
    _run_id, app = prepared_run(
        tmp_path, monkeypatch,
        {"events": 0, "pose_coverage": 1.0, "source_duration_s": 100.0,
         "source_time_s": 100.0, "unobserved_source_s": 0.0, "unobserved_intervals": 0},
        [])
    assert not app.exception
    assert not any("no usable body position was available" in str(w.value).lower() for w in app.warning)
    assert any("observable throughout" in str(c.value) for c in app.caption)


def test_an_older_run_without_visibility_figures_claims_nothing(tmp_path, monkeypatch):
    """Runs analysed before this was recorded must stay silent, not report zero blind time."""
    _run_id, app = prepared_run(tmp_path, monkeypatch,
                                {"events": 0, "pose_coverage": 0.9, "source_time_s": 100.0}, [])
    assert not app.exception
    assert not any("Visibility:" in str(c.value) for c in app.caption)
    assert not any("Visibility:" in str(w.value) for w in app.warning)


def test_cancelled_run_names_source_time_that_was_not_analysed(tmp_path, monkeypatch):
    """A partially processed source must not sound watched-and-clear."""
    _run_id, app = prepared_run(
        tmp_path, monkeypatch,
        {"events": 0, "pose_coverage": 1.0, "source_duration_s": 100.0,
         "source_time_s": 25.0, "unobserved_source_s": 0.0, "unobserved_intervals": 0,
         "unprocessed_source_s": 75.0},
        [], status="cancelled")
    assert not app.exception
    text = " ".join(str(c.value) for c in app.caption)
    assert "00:25.0" in text
    assert "75.0s of the source was not analysed" in text


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_zero_processed_time_never_falls_back_to_full_duration(tmp_path, monkeypatch, status):
    _, app = prepared_run(tmp_path, monkeypatch,
        {"events": 0, "source_duration_s": 100.0, "source_time_s": 0.0,
         "analysed_frames": 0, "unobserved_source_s": 0.0,
         "unprocessed_source_s": 100.0}, [], status=status)
    assert not app.exception
    assert not any("observable throughout" in str(c.value) for c in app.caption)
    text = " ".join(str(w.value) for w in app.warning)
    assert "no analysed time coverage" in text
    assert "100.0s of the source was not analysed" in text
