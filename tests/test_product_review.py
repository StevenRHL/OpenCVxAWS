"""Full Streamlit page regression: persisted reviews must refresh counters immediately."""
import json
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
